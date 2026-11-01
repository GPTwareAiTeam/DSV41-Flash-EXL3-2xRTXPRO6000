#!/usr/bin/env python3
"""cuda-exl3 as the GX cold-expert path: restack a layer's cold experts per bitrate group.

The production cold path (exllamav3's `BlockSparseMLP.forward` with external routing) plans tiers
per expert and runs one launch per tier per expert group. cuda-exl3 wants the opposite shape: the
experts of one bitrate stacked into a single trellis tensor, so that one launch per bitrate covers
every expert that shares a codebook. That is the whole trade this module makes:

    had_in(x, suh13)  ->  gemm(w13, shards [I, I])  ->  glu_had_in(suh2)  ->  gemm(w2, shards [H])

per bitrate group, with the final gemm fusing the routing weight and the gather back to tokens
(atomics; `CUDA_EXL3_DETERMINISTIC=1` swaps that for `exl3_moe_combine`).

Layout built here for a group of E experts (experts of one layer share a tile width across
gate/up/down, so the group key is the trellis word count):

    trellis13  (E, H/16, 2*I/16, W)   gate tiles then up tiles along N
    suh13      (E, 2, H)
    svh13      (E, 2*I)
    trellis2   (E, I/16, H/16, W)
    suh2       (E, 1, I)
    svh2       (E, H)

`forward` takes the same half-precision token block the server already builds, the clamped
int64 routing ids (misses -> expert 0 with weight 0, exactly like the incumbent path) and the
routing weights, and returns fp32 (M, H): the per-group gemms accumulate in fp32 so the answer
the response slab gets is not rounded twice.
"""

from __future__ import annotations

import math
import os
import sys

import torch

CUDA_EXL3_SRC = os.environ.get("CUDA_EXL3_SRC", "/src/exllamav3/vendor/cuda-exl3/src")
if CUDA_EXL3_SRC not in sys.path:
    sys.path.insert(0, CUDA_EXL3_SRC)
import cuda_exl3.ops as cx_ops  # noqa: E402,F401  (loads the extension)

_C = torch.ops.cuda_exl3_C

TRACE = os.environ.get("CUDA_EXL3_MOE_TRACE", "0") != "0"
CB_3INST, CB_MCG, CB_MUL1 = 0, 1, 2
DETERMINISTIC = os.environ.get("CUDA_EXL3_DETERMINISTIC", "0") != "0"
BLOCK_M_TIERS = (16, 32, 64, 128)
# One extra pass over an expert's weights costs about as many padded rows as this (layer 5,
# GX2, 512..8192 rows): it is what puts the optimum at 128 at 4096 rows but 64 at 8192.
BLOCK_M_ROWS = 24
BLOCK_M_OVERRIDE = int(os.environ.get("CUDA_EXL3_MOE_BLOCK_M", "0"))
# Split-k needs the rows*n_total fp32 accumulator; it is only used when rows is small and the
# upstream knob warns that the accumulator epilogue can read out of bounds when the grid is below
# the SM count (upstream PR #7). The group gemms here are never starved - 268 experts over 3
# groups is thousands of blocks - so the accumulator is capped to zero and every launch is
# unsplit. Set CUDA_EXL3_MOE_ACC_CAP to re-enable it for an experiment.
if int(os.environ.get("CUDA_EXL3_MOE_ACC_CAP", "0")) == 0:
    _C.exl3_set_moe_acc_cap(0)


def _bits_frac(words: int) -> tuple[int, bool]:
    """Trellis words per tile -> (whole bits, fractional-bit tile)."""
    return words // 16, (words % 16) != 0


def _codebook(line) -> int:
    """Codebook id for an exllamav3 EXL3 linear, from the flags the module already carries."""
    inner = getattr(line, "inner", line)
    if getattr(inner, "mul1", False):
        return CB_MUL1
    if getattr(inner, "mcg", False):
        return CB_MCG
    return CB_3INST


def _inner(line):
    inner = getattr(line, "inner", line)
    for name in ("trellis", "suh", "svh"):
        if not hasattr(inner, name):
            raise AttributeError(f"exl3_moe_cuda: {type(line).__name__} has no .{name} "
                                 f"(not an EXL3 linear?)")
    return inner


class Group:
    """One bitrate: the stacked tensors of every expert that shares its tile width."""

    __slots__ = ("bits", "frac", "cb", "trellis13", "suh13", "svh13",
                 "trellis2", "suh2", "svh2", "n_experts", "source", "remap")

    def __init__(self, bits, frac, cb):
        self.bits, self.frac, self.cb = bits, frac, cb
        self.n_experts = 0

    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in
                   (self.trellis13, self.suh13, self.svh13, self.trellis2, self.suh2, self.svh2))


def _stack_group(gates, ups, downs, device) -> Group:
    g0 = _inner(gates[0])
    d0 = _inner(downs[0])
    words = int(g0.trellis.shape[2])
    bits, frac = _bits_frac(words)
    grp = Group(bits, frac, _codebook(gates[0]))
    e_n = len(gates)
    kt, nt = int(g0.trellis.shape[0]), int(g0.trellis.shape[1])
    k2t, n2t = int(d0.trellis.shape[0]), int(d0.trellis.shape[1])
    grp.n_experts = e_n
    grp.trellis13 = torch.empty((e_n, kt, 2 * nt, words), dtype=torch.int16, device=device)
    grp.suh13 = torch.empty((e_n, 2, 16 * kt), dtype=torch.half, device=device)
    grp.svh13 = torch.empty((e_n, 32 * nt), dtype=torch.half, device=device)
    grp.trellis2 = torch.empty((e_n, k2t, n2t, int(d0.trellis.shape[2])), dtype=torch.int16,
                               device=device)
    grp.suh2 = torch.empty((e_n, 1, 16 * k2t), dtype=torch.half, device=device)
    grp.svh2 = torch.empty((e_n, 16 * n2t), dtype=torch.half, device=device)
    for e, (g, u, d) in enumerate(zip(gates, ups, downs)):
        gi, ui, di = _inner(g), _inner(u), _inner(d)
        grp.trellis13[e, :, :nt] = gi.trellis
        grp.trellis13[e, :, nt:] = ui.trellis
        grp.suh13[e, 0] = gi.suh
        grp.suh13[e, 1] = ui.suh
        grp.svh13[e, :32 * nt // 2] = gi.svh
        grp.svh13[e, 32 * nt // 2:] = ui.svh
        grp.trellis2[e] = di.trellis
        grp.suh2[e, 0] = di.suh
        grp.svh2[e] = di.svh
    return grp


def align_block(local_ids: torch.Tensor, block_m: int, num_experts: int, m_valid: int):
    """vLLM's `moe_align_block_size`, in torch: bucket the routed rows by expert and pad every
    bucket to whole `block_m` blocks.

    `local_ids` is (M, T) of this group's expert index per pick, -1 where the pick belongs to
    another bitrate group (that row is that group's to compute, and is left out here). The
    returned `sorted_ids` maps a padded row position to a flat token-major pick `idx < m_valid`,
    which is what the kernels gather from; the pad sentinel is `m_valid` and not -1 because the
    kernels test `sorted_ids[row] < m_valid` for a live row, so a negative sentinel would read as
    a live row pointing before the token block.
    """
    flat = local_ids.reshape(-1)
    live = flat >= 0
    rows_idx = torch.nonzero(live, as_tuple=False).squeeze(1)
    n = int(rows_idx.numel())
    if n == 0:
        return None, None, None, block_m
    ids_live = flat[rows_idx]
    order = torch.argsort(ids_live, stable=True)
    counts = torch.bincount(ids_live, minlength=num_experts)
    nz = counts > 0
    bpe = torch.zeros_like(counts)
    bpe[nz] = (counts[nz] + block_m - 1) // block_m
    blk_off = torch.cumsum(bpe, 0) - bpe
    nblk = int(blk_off[-1] + bpe[-1])
    total = nblk * block_m
    ent = ids_live[order]
    off = torch.cumsum(counts, 0) - counts
    pos = blk_off[ent] * block_m + (torch.arange(n, device=flat.device) - off[ent])
    sorted_ids = torch.full((total,), m_valid, dtype=torch.int32, device=flat.device)
    sorted_ids[pos] = rows_idx[order].to(torch.int32)
    expert_ids = torch.repeat_interleave(
        torch.arange(num_experts, dtype=torch.int32, device=flat.device), bpe)
    return sorted_ids, expert_ids, int(pos.max()) + 1, block_m


def pick_block_m(rows: int, n_experts: int) -> int:
    """Per-expert row-block size, from the two costs a block size trades against.

    Both projections read an expert's whole trellis slice once per block, so the traffic is
    `ceil(per/bm)` weight passes, and each expert's run is padded up to `ceil(per/bm)*bm` rows of
    compute. `BLOCK_M_ROWS` weighs one weight pass against that many padded rows; it was fitted
    to layer 5 on GX2 (measured minima: per=11.5 -> 16, 46 -> 64, 91 -> 128, 182 -> 64) and is
    what makes the answer 128 at 4096 rows and 64 at both 2048 and 8192.
    """
    per = rows / max(n_experts, 1)
    bm = min(BLOCK_M_TIERS,
             key = lambda b: (math.ceil(per / b) * b + BLOCK_M_ROWS * math.ceil(per / b)))
    if BLOCK_M_OVERRIDE:
        bm = BLOCK_M_OVERRIDE
    if bm not in BLOCK_M_TIERS:
        raise ValueError(f"exl3_moe_cuda: block_m {bm} is not a compiled tile "
                         f"({BLOCK_M_TIERS}; 256 only with the h_acc path)")
    return bm


class CudaExl3Moe:
    """A cold layer's experts, restacked per bitrate, forwarded by one launch per projection."""

    def __init__(self, ffn, device=None, drop_source: bool = True):
        self.device = device or next(iter(ffn.gates)).device
        self.hidden = None
        self.inter = None
        self.top_k = int(getattr(ffn, "num_experts_per_tok", 0) or 0)
        self.num_experts = int(len(ffn.gates))
        by_key: dict[tuple[int, int, int], list[int]] = {}
        for e, g in enumerate(ffn.gates):
            gi, ui, di = _inner(g), _inner(ffn.ups[e]), _inner(ffn.downs[e])
            key = (int(gi.trellis.shape[2]), int(di.trellis.shape[2]), _codebook(g))
            if (int(ui.trellis.shape[2]), _codebook(ffn.ups[e])) != (key[0], key[2]):
                raise ValueError(f"expert {e}: gate/up tile widths differ "
                                 f"({key[0]} vs {ui.trellis.shape[2]})")
            by_key.setdefault(key, []).append(e)
        self.groups: list[Group] = []
        for key, idx in sorted(by_key.items()):
            grp = _stack_group([ffn.gates[e] for e in idx], [ffn.ups[e] for e in idx],
                               [ffn.downs[e] for e in idx], self.device)
            grp.source = idx
            self.groups.append(grp)
        g0 = _inner(ffn.gates[0])
        d0 = _inner(ffn.downs[0])
        self.hidden = 16 * int(g0.trellis.shape[0])
        self.inter = 16 * int(g0.trellis.shape[1])
        if self.top_k == 0:
            self.top_k = int(getattr(ffn, "num_experts_per_tok", 6) or 6)
        del g0, d0
        # ids arrive as this layer's cold-expert indices; each group needs its own 0..E-1 labels,
        # with -1 marking the picks another group owns.
        for grp in self.groups:
            remap = torch.full((self.num_experts,), -1, dtype=torch.int64, device=self.device)
            remap[torch.tensor(grp.source, dtype=torch.int64, device=self.device)] = torch.arange(
                len(grp.source), dtype=torch.int64, device=self.device)
            grp.remap = remap
        if drop_source:
            # The per-expert copies are dead once stacked; the layer must not pay for both. Three
            # things hold C++ references to those tensors and every one of them has to go or the
            # drop frees nothing: the module-wide BC_BlockSparseMLP keeps a shared_ptr per expert
            # BC (block_sparse_mlp.py:771-773), each BC_LinearEXL3 keeps its own three tensors,
            # and the multilinear bundles keep the fusions built over the same experts.
            ffn.bc = None
            for name in ("multi_gate", "multi_up", "multi_down"):
                if getattr(ffn, name, None) is not None:
                    setattr(ffn, name, None)
            for e in range(self.num_experts):
                for line in (ffn.gates[e], ffn.ups[e], ffn.downs[e]):
                    inner = _inner(line)
                    inner.bc = None
                    inner.trellis = inner.suh = inner.svh = None

    # -- geometry -------------------------------------------------------------------------------
    def nbytes(self) -> int:
        return sum(g.nbytes() for g in self.groups)

    def describe(self) -> str:
        return " ".join(f"bits{g.bits}{'+frac' if g.frac else ''}:{g.n_experts}"
                        for g in self.groups)

    # -- forward --------------------------------------------------------------------------------
    def forward(self, x: torch.Tensor, ids: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """x half (M, H), ids int64 (M, T) this layer's cold experts, w half (M, T) -> fp32 (M, H).

        Each group does its own alignment over the picks it owns and its own gemm/combine, so the
        outputs are summed in fp32 here; that keeps the caller's slab fp32 and leaves the mixed-K
        module's dtype contract untouched.
        """
        ids = ids.to(self.device)
        M, T = ids.shape
        m_valid = M * T
        wf = w.float().contiguous()
        total = None
        # Optional per-phase attribution (CUDA_EXL3_MOE_TRACE=1, read by qmesh/prof_cx_phases.py):
        # each mark is an event on the caller's stream, so consecutive marks bound one op.
        trace = [] if TRACE else None

        def mark(label, blk=0, nrows=0):
            if trace is not None:
                ev = torch.cuda.Event(enable_timing=True)
                ev.record()
                trace.append((label, blk, nrows, ev))

        for grp in self.groups:
            if grp.n_experts == 0:
                continue
            mark(f"align:{grp.bits}{'f' if grp.frac else ''}")
            loc = grp.remap[ids]
            live = int((loc >= 0).sum())
            if live == 0:
                continue
            block_m = pick_block_m(live, grp.n_experts)
            sid, eids, n_rows, block_m = align_block(loc, block_m, grp.n_experts, m_valid)
            if eids is None:
                continue
            rows = int(eids.numel()) * block_m
            n_rows = torch.tensor([n_rows], dtype=torch.int32, device=self.device)
            tag = f"{grp.bits}{'f' if grp.frac else ''}"
            mark(f"had_in:{tag}", block_m, rows)
            # skip_padding leaves the padded rows of a13 unwritten, so both gemms have to be
            # told about sorted_ids or they would fetch them; with it, a padded row is never
            # staged and never written back, which is where a decode-shaped call saves most.
            a13 = torch.empty((2, rows, self.hidden), dtype=torch.half, device=x.device)
            _C.exl3_moe_had_in(x, a13, grp.suh13, sid, eids, n_rows, block_m, T, m_valid, True)
            mark(f"gemm13:{tag}", block_m, rows)
            inter = _C.exl3_moe_gemm(a13, grp.trellis13, grp.suh13, grp.svh13, eids, n_rows,
                                     [self.inter, self.inter], grp.cb, block_m, torch.half,
                                     sid, None, M, T)
            del a13
            mark(f"glu:{tag}", block_m, rows)
            a2 = torch.empty((1, rows, self.inter), dtype=torch.half, device=x.device)
            _C.exl3_moe_glu_had_in(inter, a2, grp.suh2, eids, n_rows, block_m)
            del inter
            mark(f"gemm2:{tag}", block_m, rows)
            if DETERMINISTIC:
                routed = _C.exl3_moe_gemm(a2, grp.trellis2, grp.suh2, grp.svh2, eids, n_rows,
                                          [self.hidden], grp.cb, block_m, torch.half,
                                          sid, None, M, T)
                mark(f"combine:{tag}", block_m, rows)
                out = _C.exl3_moe_combine(routed, sid, wf, M, eids, block_m)
                del routed
            else:
                out = _C.exl3_moe_gemm(a2, grp.trellis2, grp.suh2, grp.svh2, eids, n_rows,
                                       [self.hidden], grp.cb, block_m, torch.half,
                                       sid, wf, M, T)
            del a2
            mark(f"sum:{tag}", block_m, rows)
            total = out.float() if total is None else total + out.float()
        mark("end")
        if trace is not None:
            self.trace = trace
        if total is None:
            return torch.zeros((M, self.hidden), dtype=torch.float32, device=x.device)
        return total
