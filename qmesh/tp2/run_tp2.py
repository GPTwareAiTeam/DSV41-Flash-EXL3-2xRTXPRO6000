"""TP=2 per-rank runner for the pre-sharded DeepSeek-V4.1 checkpoint (qmesh/tp2).

One process per GPU (RANK / WORLD_SIZE / INIT_METHOD env, launched by qmesh/tp2/launch_tp2.sh). Each rank
loads ONLY its rank dir from the overlay (rank shards + engram-only tables) as a single-device model on
cuda:RANK, built with the local TP dims of the architecture. Decoding uses the standard Generator.

Collectives (torch.distributed NCCL, see TPBackendNCCL):
- attention and MoE partial sums: all_reduce(SUM) via the modules' tp_reduce hook (params["backend"]).
- vocab-split embedding: rank r owns rows [r*V, (r+1)*V); out-of-range ids -> zero, then all_reduce(SUM).
- vocab-split head: local logits [.., V] all_gather'd in rank order -> full [.., V*WORLD] on both ranks, so
  greedy sampling is identical on both ranks.

Env: OVERLAY (dir holding rank0/ rank1/), MAX_NEW (64), PROMPT_IDS (0,1,2), IGNORE_EOS (0|1),
INIT_METHOD (file://...), TOKENS_OUT (optional json path), PROFILE (0|1), COMM (p2p|hybrid|nccl), P2P_RDV (fresh dir, p2p/hybrid),
P2P_STATS_OUT (optional prefix, p2p: enables device spin/timing stats, dumped to <prefix>.rank<r>.json at run end).
"""
from __future__ import annotations

import json
import os
import sys
import time

os.environ.setdefault("EXL3_MOE_SPLIT_FUSED", "0")
os.environ.setdefault("EXL3_MOE_CPU_SWAP", "0")
os.environ.setdefault("TORCH_EXTENSIONS_DIR", "/root/.cache/torch_extensions")

import torch
import torch.distributed as dist

sys.path.insert(0, "/models/dsv41/DeepSeek-V4.1-Flash-1017-orig/encoding")
from encoding import encode_messages  # noqa: E402

from exllamav3 import Cache, Config, Generator, GreedySampler, Job, Model, Tokenizer  # noqa: E402
from exllamav3 import CacheLayer_quant  # noqa: E402
from exllamav3.constants import PAGE_SIZE  # noqa: E402
from exllamav3.generator.generator import _DRAFT_HIST  # noqa: E402
from exllamav3.modules.transformer import TransformerBlock  # noqa: E402
from exllamav3.tp2.p2p import ARENA_MAXB  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import content_seed  # noqa: E402

OVERLAY = os.environ.get("OVERLAY", "/artifacts/tp2-overlay")
DSPARK = os.environ.get("DSPARK", "0") == "1"
DSPARK_HEAD = os.environ.get("DSPARK_HEAD", "ours")
if DSPARK_HEAD not in ("ours", "diffbot"):
    raise ValueError("DSPARK_HEAD must be ours or diffbot")
if DSPARK and DSPARK_HEAD == "diffbot":
    OVERLAY = os.environ.get("DSPARK_DIFFBOT_OVERLAY", "/work/k7db_overlay")
DSPARK_DYNAMIC = os.environ.get("DSPARK_DYNAMIC", "0") == "1"
DSPARK_TARGET_ACC = float(os.environ.get("DSPARK_TARGET_ACC", "0.7"))
DSPARK_POLICY = os.environ.get("DSPARK_POLICY", "always")
DSPARK_COUNT_FORWARDS = os.environ.get("DSPARK_COUNT_FORWARDS", "0") == "1"
DRAFT_TIMING = os.environ.get("DRAFT_TIMING", "0") == "1"
NUM_DRAFT = int(os.environ.get("NUM_DRAFT", "5"))
# Recurrent-cache history must cover the speculative pass width (generator requires max_history >= K); 5 is the
# previous fixed value, so K<=5 runs are unchanged.
MAX_HISTORY = max(5, NUM_DRAFT)
MAX_NEW = int(os.environ.get("MAX_NEW", "64"))
MAX_CHUNK_RAW = os.environ.get("MAX_CHUNK", "auto")  # "auto" -> largest chunk that fits; else a fixed row count
WARM_TOKENS = int(os.environ.get("WARM_TOKENS", "8"))  # untimed warmup length at the configured batch width
BATCH = int(os.environ.get("BATCH", "1"))  # concurrent streams: Cache slots, Model.load + Generator max_batch_size
CACHE_TOKENS_PER_SEQ = int(os.environ.get("CACHE_TOKENS_PER_SEQ", "4096"))  # KV pool tokens per concurrent stream
# CTX_TOKENS: total KV budget in tokens across the batch (0 = derive from CACHE_TOKENS_PER_SEQ). Set this to the context
# you want to support, e.g. 1048576 for 1M; the chunk auto-sizer then gives whatever VRAM is left to the prefill chunk.
CTX_TOKENS = int(os.environ.get("CTX_TOKENS", "0"))
CACHE_TOKENS = int(os.environ.get("CACHE_TOKENS", str(CTX_TOKENS if CTX_TOKENS else BATCH * CACHE_TOKENS_PER_SEQ)))
KV_BITS = int(os.environ.get("KV_BITS", "16"))
PREFIX_CACHE = os.environ.get("EXL3_PREFIX_CACHE", "1") == "1"
PREFIX_CACHE_CAPACITY = int(os.environ.get("EXL3_PREFIX_CACHE_CAPACITY", str(4 * 1024**3)))
PREFIX_CACHE_MIN_MATCH = int(os.environ.get("EXL3_PREFIX_CACHE_MIN_MATCH", "256"))
PREFIX_CACHE_INTERVAL = int(os.environ.get("EXL3_PREFIX_CACHE_INTERVAL", "1024"))
if PREFIX_CACHE_INTERVAL <= 0 or PREFIX_CACHE_INTERVAL % 256:
    raise ValueError("EXL3_PREFIX_CACHE_INTERVAL must be a positive multiple of 256")
if PREFIX_CACHE_CAPACITY < 0 or PREFIX_CACHE_MIN_MATCH < 0:
    raise ValueError("prefix cache capacity and minimum match must be nonnegative")
if PREFIX_CACHE:
    os.environ["EXL3_PREFILL_RC_SPLIT"] = "1"
if KV_BITS not in (4, 8, 16):
    raise ValueError("KV_BITS must be 4, 8 or 16")
# KV_BITS=4: DeepSeek's trained FP4 layout for the main compressed pools (latent + indexer keys)
KV_CACHE_KWARGS = ({"layer_type": CacheLayer_quant, "k_bits": 8, "v_bits": 8} if KV_BITS == 8 else
                   {"kv_format": "fp4"} if KV_BITS == 4 else {})
IGNORE_EOS = os.environ.get("IGNORE_EOS", "0") == "1"
MEM_BREAKDOWN = os.environ.get("MEM_BREAKDOWN", "0") == "1"  # per-GPU resident GiB by owner after load + peak (MEMB lines)
COMM = os.environ.get("COMM", "p2p")  # nccl | p2p | hybrid (hybrid: P2P <64KB, NCCL >=64KB)
P2P_STATS_OUT = os.environ.get("P2P_STATS_OUT")  # p2p only: prefix; enables device spin/timing stats, dumped at run end to <prefix>.rank<r>.json
PROFILE = os.environ.get("PROFILE", "0") == "1"
PROFILE_STACK = os.environ.get("PROFILE_STACK", "0") == "1"   # attach python stacks (names the .to()/copy_ call sites)
PROFILE_SHAPES = os.environ.get("PROFILE_SHAPES", "0") == "1"  # record tensor shapes (names the copied buffers)
TOKENS_OUT = os.environ.get("TOKENS_OUT")
LOGITS_OUT = os.environ.get("LOGITS_OUT")
PROBE = [int(n) for n in os.environ.get("PROBE", "").split(",") if n.strip()]
PROMPT_IDS = [int(i) for i in os.environ.get("PROMPT_IDS", "0,1,2").split(",")]
RANK = int(os.environ["RANK"])
WORLD = int(os.environ["WORLD_SIZE"])
TAG = f"TP2R{RANK}"
# --- prefill chunk auto-sizing -------------------------------------------------------------------------------------
# The prefill chunk and the KV budget compete for VRAM, so rather than fixing the chunk we give the prefill chunk
# whatever VRAM the weights and the requested context leave. Driven by MEASURED device peaks, not a synthetic model:
# the disk size of a shard (89.81 GiB) is NOT the resident footprint (peak was 87.6 GiB at chunk 2048), so a
# weights-plus-overhead calculation would be wrong in the unsafe direction.
# CHUNK_PEAK_TABLE: "rows=peak_MiB" pairs sampled from `nvidia-smi` during real prefill runs, with the KV budget at
# CHUNK_PEAK_REF_TOKENS and KV_BYTES_PER_TOKEN. Sizes outside the table are linearly extrapolated between neighbours
# (and beyond the largest entry by the last slope), and VRAM_RESERVE_MIB is kept free for graphs, workspaces and
# allocator slack.
CHUNK_CANDIDATES = [int(x) for x in os.environ.get("PREFILL_CHUNK_STEPS", "16384,8192,4096,2048").split(",") if x.strip()]
CHUNK_MIN = int(os.environ.get("PREFILL_CHUNK_MIN", "512"))
CHUNK_PEAK_TABLE_RAW = os.environ.get("CHUNK_PEAK_TABLE", "2048=87600,4096=90600,8192=94000,16384=97100")
CHUNK_PEAK_REF_TOKENS = int(os.environ.get("CHUNK_PEAK_REF_TOKENS", "33536"))
VRAM_RESERVE_MIB = float(os.environ.get("VRAM_RESERVE_MIB", "2048"))  # graphs, workspaces, allocator slack
# Measured long-context DSpark with an 8192-row chunk exceeded the device by
# about 640 MiB; reserve additional space for drafter weights and graphs.
DSPARK_RESERVE_MIB = float(os.environ.get("DSPARK_RESERVE_MIB", "2048"))
KV_BYTES_PER_TOKEN = float(os.environ.get("KV_BYTES_PER_TOKEN", {4: "890", 8: "2150"}.get(KV_BITS, "3200")))  # per rank, pool only
CHUNK_PEAK_TABLE = sorted((int(rows), float(mib)) for rows, mib in
                          (pair.split("=") for pair in CHUNK_PEAK_TABLE_RAW.split(",") if pair.strip()))


def _peak_mib(rows):
    """Interpolated/extrapolated device peak (MiB) for a given chunk row count."""
    pts = CHUNK_PEAK_TABLE
    if rows <= pts[0][0]:
        (r0, m0), (r1, m1) = pts[0], pts[1] if len(pts) > 1 else (pts[0][0] + 1, pts[0][1])
        return m0 - (m0 - m1) * (r0 - rows) / max(1, (r1 - r0))
    for (r0, m0), (r1, m1) in zip(pts, pts[1:]):
        if rows <= r1:
            return m0 + (m1 - m0) * (rows - r0) / (r1 - r0)
    (r0, m0), (r1, m1) = pts[-2], pts[-1]
    return m1 + (m1 - m0) * (rows - r1) / (r1 - r0)


def resolve_max_chunk(device):
    """MAX_CHUNK=auto -> largest candidate chunk whose measured peak, re-based to our KV budget, fits with reserve."""
    if MAX_CHUNK_RAW != "auto":
        return int(MAX_CHUNK_RAW), None
    _free, total = torch.cuda.mem_get_info(device)
    total_mib = total / (1024 * 1024)
    kv_mib = CACHE_TOKENS * KV_BYTES_PER_TOKEN / (1024 * 1024)
    kv_ref_mib = CHUNK_PEAK_REF_TOKENS * KV_BYTES_PER_TOKEN / (1024 * 1024)
    limit = total_mib - VRAM_RESERVE_MIB - (DSPARK_RESERVE_MIB if DSPARK else 0) - (kv_mib - kv_ref_mib)
    fits = [c for c in sorted(CHUNK_CANDIDATES, reverse = True) if _peak_mib(c) <= limit]
    chunk = fits[0] if fits else CHUNK_MIN
    return chunk, dict(total_mib = total_mib, kv_mib = kv_mib, kv_ref_mib = kv_ref_mib, limit_mib = limit,
                       peak_mib = _peak_mib(chunk), free_mib = _free / (1024 * 1024))

# --- adaptive prefill chunk (depth-aware OOM guard) ---------------------------------------------------------------
# A prefill chunk's transient (peak over the steady allocation) is the larger of two terms, fitted to 598 measured
# chunks of a 876k-token prefill (FP4 KV, 1M cache): activations, ~382 KiB per row, flat in depth; and a depth term
# that takes over past ~300k tokens (the uncompressed pool layer's fp16 staging window and the indexer score
# slab): ~461 MiB + 1280 B per context token at 2048 rows, plus the row term for rows beyond 2048. A fixed chunk
# that fits a shallow prompt can therefore run out of memory deep into a long one. Before every prefill forward
# the limiter reads free memory once and solves both terms for the largest page-aligned chunk that fits under a
# small safety margin, capped at the planned chunk; both ranks take the minimum.
PREFILL_ADAPTIVE = os.environ.get("PREFILL_ADAPTIVE", "1") == "1"
PREFILL_ROW_BYTES = float(os.environ.get("PREFILL_ROW_KIB", "382")) * 2**10        # activations per prefill row
PREFILL_DEPTH_BASE = float(os.environ.get("PREFILL_DEPTH_BASE_MIB", "461")) * 2**20
PREFILL_DEPTH_BYTES = float(os.environ.get("PREFILL_DEPTH_BYTES", "1280"))          # per context token
PREFILL_DEPTH_ROWS = 2048                                                            # rows the depth term was fit at
# Safety cushion left free on the card at the predicted peak. Torch's held-but-free memory counts as usable
# (expandable segments let any freed block serve any later shape); PREFILL_FRAG > 0 discounts that share of it
PREFILL_MARGIN_MIB = float(os.environ.get("PREFILL_MARGIN_MIB", "1024"))
PREFILL_FRAG = float(os.environ.get("PREFILL_FRAG", "0"))
PREFILL_ADAPTIVE_LOG = os.environ.get("PREFILL_ADAPTIVE_LOG", "0") == "1"


def prefill_rise(start, rows):
    """Predicted transient bytes over the steady allocation for a chunk of `rows` starting at `start`."""
    deep = PREFILL_DEPTH_BASE + PREFILL_DEPTH_BYTES * (start + rows) + PREFILL_ROW_BYTES * max(0, rows - PREFILL_DEPTH_ROWS)
    return max(PREFILL_ROW_BYTES * rows, deep)


def prefill_fit(start, room):
    """Largest row count whose prefill_rise at `start` is <= room: the minimum of each term solved for rows."""
    by_rows = room / PREFILL_ROW_BYTES
    by_depth_lo = (room - PREFILL_DEPTH_BASE - PREFILL_DEPTH_BYTES * start) / PREFILL_DEPTH_BYTES   # rows <= 2048
    by_depth = by_depth_lo if by_depth_lo <= PREFILL_DEPTH_ROWS else \
        (room - PREFILL_DEPTH_BASE - PREFILL_DEPTH_BYTES * start + PREFILL_ROW_BYTES * PREFILL_DEPTH_ROWS) \
        / (PREFILL_DEPTH_BYTES + PREFILL_ROW_BYTES)
    return min(by_rows, by_depth)


class PrefillChunkLimiter:
    """Generator.prefill_chunk_limit: one free-memory read and one solve per prefill chunk for the largest
    page-aligned chunk that fits (at most the planned chunk, at least one page), agreed (MIN) across ranks. Keeps
    the highest per-chunk allocator peak it has seen (it resets torch's peak counter per chunk to measure each),
    and the time each decision takes."""

    def __init__(self, device):
        self.device = device
        self.agree = torch.zeros(1, dtype = torch.int32, device = device)
        self.prev = None
        self.peak = 0
        self.shrunk = 0
        self.decide_us = []

    def __call__(self, start: int, chunk: int) -> int:
        t0 = time.perf_counter()
        dev = self.device
        alloc = torch.cuda.memory_allocated(dev)
        peak = torch.cuda.max_memory_allocated(dev)
        self.peak = max(self.peak, peak)
        if PREFILL_ADAPTIVE_LOG and self.prev is not None:
            p_start, p_rows, p_alloc, p_t = self.prev
            dt = t0 - p_t
            log(f"PREFILL_CHUNK_PEAK start={p_start} rows={p_rows} rise_MiB={(peak - p_alloc) / 2**20:.0f} "
                f"predicted_MiB={prefill_rise(p_start, p_rows) / 2**20:.0f} ms={dt * 1e3:.0f} "
                f"rows_per_s={p_rows / dt:.0f}")
        torch.cuda.reset_peak_memory_stats(dev)
        # The cushion is kept on the card, not inside torch's pool: torch's caching allocator holds freed blocks
        # (reserved - allocated) that a larger chunk cannot always reuse, so count only memory the card still
        # has free plus what torch can hand back, less the cushion
        free, _ = torch.cuda.mem_get_info(dev)
        reserved = torch.cuda.memory_reserved(dev)
        room = free + reserved - alloc - PREFILL_MARGIN_MIB * 2**20 - (reserved - alloc) * PREFILL_FRAG
        rows = max(PAGE_SIZE, min(chunk, int(prefill_fit(start, room)) // PAGE_SIZE * PAGE_SIZE))
        self.agree.fill_(rows)
        dist.all_reduce(self.agree, op = dist.ReduceOp.MIN)
        rows = int(self.agree.item())
        self.decide_us.append((time.perf_counter() - t0) * 1e6)
        if rows < chunk or PREFILL_ADAPTIVE_LOG:
            self.shrunk += rows < chunk
            if PREFILL_ADAPTIVE_LOG:
                total = torch.cuda.mem_get_info(dev)[1]
                log(f"PREFILL_CHUNK_ADAPT start={start} planned={chunk} rows={rows} room_MiB={room / 2**20:.0f} "
                    f"free_MiB={free / 2**20:.0f} reserved_MiB={reserved / 2**20:.0f} alloc_MiB={alloc / 2**20:.0f} "
                    f"non_torch_MiB={(total - free - reserved) / 2**20:.0f}")
        self.prev = (start, rows, alloc, time.perf_counter())
        return rows

    def prefill_done(self):
        """A prompt's prefill finished: record its last chunk's peak, then hand torch's held-but-free blocks back
        to the card. Prefill chunks free their transients into torch's general pool, while decode captures its
        CUDA graphs from a separate pool that can only grow into memory the card still has free; without this the
        first decode capture after a prefill that used big chunks finds the card full. Called once per prompt."""
        dev = self.device
        self.peak = max(self.peak, torch.cuda.max_memory_allocated(dev))
        if PREFILL_ADAPTIVE_LOG and self.prev is not None:
            p_start, p_rows, p_alloc, p_t = self.prev
            dt = time.perf_counter() - p_t
            log(f"PREFILL_CHUNK_PEAK start={p_start} rows={p_rows} "
                f"rise_MiB={(torch.cuda.max_memory_allocated(dev) - p_alloc) / 2**20:.0f} "
                f"predicted_MiB={prefill_rise(p_start, p_rows) / 2**20:.0f} ms={dt * 1e3:.0f} rows_per_s={p_rows / dt:.0f}")
        self.prev = None
        torch.cuda.empty_cache()

PROMPTS = [
    "The capital of France is",
    "Write a Python function that returns the nth Fibonacci number.",
    "List three primary colors and explain why they matter in painting.",
    "Summarize the following passage in one sentence: " + "".join(
        "Paragraph %d: the river wound through the valley past terraced fields of barley, "
        "where farmers repaired stone walls each spring and traded seed in the market town. " % i
        for i in range(1, 36)),
    # index 4: code-heavy (fenced block, braces, def/return)
    "Write a Python class `LRUCache` with get and put methods, then show a ```python fenced example that uses it.",
    # index 5: structured JSON (keys, braces, schema-shaped output)
    "Return only a JSON object describing a user with keys name, age, email, and a list field roles of strings. "
    "Use valid JSON with no commentary.",
    "Explain why the sky appears blue.",
    "Give a short recipe for tomato soup.",
    "Name three planets in our solar system.",
]
if os.environ.get("SHORT_PROMPTS") == "1":
    PROMPTS[3] = "Describe a peaceful morning in one sentence."
# LONG_PROMPT_TOKENS: append synthetic filler to every prompt so the runner can exercise long-context behaviour
# (20k-100k prompts) without a real document. The log prints the true prompt_tokens, so quote that rather than
# assuming the requested size was hit exactly.
# LONG_PROMPT_STYLE picks the filler content, because draft acceptance depends on the text: 'prose' is narrative,
# 'code' is a fenced Python listing with identifiers and braces, 'json' is a repeated structured record, and 'mixed'
# interleaves all three (closest to an OMP/Hermes transcript of tool output plus code plus commentary).
LONG_PROMPT_TOKENS = int(os.environ.get("LONG_PROMPT_TOKENS", "0"))
LONG_PROMPT_STYLE = os.environ.get("LONG_PROMPT_STYLE", "prose")
_FILLERS = {
    "prose": ("The agent reviewed the archive, reconciled the ledger entries, and recorded each result "
              "in the running log before continuing to the next section of the report. ", 24),
    "code": ("def reconcile(ledger, entries):\n    totals = {}\n    for key, value in entries.items():\n"
             "        totals[key] = totals.get(key, 0) + value\n    return {k: v for k, v in totals.items() if v}\n", 62),
    "json": ('{"id": 4417, "status": "ok", "fields": {"name": "archive", "tags": ["ledger", "review"], '
             '"count": 12}, "ok": true}\n', 30),
}
if LONG_PROMPT_STYLE == "mixed":
    _unit = "".join([_FILLERS[k][0] for k in ("prose", "code", "json")])
    _tokens_per_unit = sum(_FILLERS[k][1] for k in ("prose", "code", "json"))
else:
    _unit, _tokens_per_unit = _FILLERS[LONG_PROMPT_STYLE]
if os.environ.get("LONG_PROMPT_FILE"):
    # Real-content long prompt: a captured transcript/fixture appended to every prompt, so the round is
    # measured on agentic text (acceptance and indexer behaviour differ from synthetic filler).
    with open(os.environ["LONG_PROMPT_FILE"], encoding = "utf-8") as f:
        _transcript = f.read()
    PROMPTS = [p + "\n\n" + _transcript for p in PROMPTS]
elif LONG_PROMPT_TOKENS:
    _reps = max(1, LONG_PROMPT_TOKENS // _tokens_per_unit)
    PROMPTS = [p + (_unit * _reps) for p in PROMPTS]


def log(msg: str):
    print(f"[{TAG}] {msg}", flush=True)



class RankAgreedGreedySampler(GreedySampler):
    """Keep verification acceptance and policy state identical on both ranks."""

    # Set by the generator's verify path while it launches a whole window: the per-call broadcast is
    # then replaced by one agree_window() collective, so a W-position window pays one host-blocking
    # collective instead of W. Values are unchanged (rank 0 still wins per position).
    defer_agree = False

    def forward(self, *args, **kwargs):
        token = super().forward(*args, **kwargs)
        if not self.defer_agree:
            dist.broadcast(token, src = 0)
        return token

    def agree_window(self, tokens: list) -> list:
        """Rank-0 agreement for one verify window's sampled tokens, in one collective."""
        dev = [t for t in tokens if t.is_cuda and t.numel() == 1]
        if not dev or len(dev) != len(tokens) or len({t.dtype for t in dev}) != 1:
            for t in tokens:
                dist.broadcast(t, src = 0)
            return list(tokens)
        batch = torch.cat([t.reshape(1) for t in dev])
        dist.broadcast(batch, src = 0)
        return [v.view(t.shape) for v, t in zip(batch.unbind(), tokens)]


class TPBackendNCCL:
    """Collective backend for the modules' params["backend"] hook (torch.distributed NCCL)."""

    def fwd_barrier(self):
        dist.barrier()

    def all_reduce(self, tensor: torch.Tensor, contribution: bool = True):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("NCCL all_reduce during CUDA graph capture is unsafe; use COMM=p2p")
        # A non-contributing rank reduces a zero stand-in so the collective stays aligned
        if not contribution:
            tensor.zero_()
        dist.all_reduce(tensor, op = dist.ReduceOp.SUM)

    def broadcast(self, tensor: torch.Tensor, src_device: int):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("NCCL broadcast during CUDA graph capture is unsafe; use COMM=p2p")
        dist.broadcast(tensor, src = src_device)

    def all_gather(self, t: torch.Tensor, world: int) -> torch.Tensor:
        """Concatenate the per-rank t along the last dim, in rank order, on both ranks (== cat(parts, -1))."""
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("NCCL all_gather during CUDA graph capture is unsafe; use COMM=p2p")
        assert world == 2
        parts = [torch.empty_like(t) for _ in range(world)]
        dist.all_gather(parts, t.contiguous())
        return torch.cat(parts, dim = -1)
    def run_cpu_reduce_jobs(self):
        pass

    def end_cpu_reduce_jobs(self):
        pass
P2P_CHUNK = int(os.environ.get("P2P_CHUNK", str(ARENA_MAXB // 4)))  # fp32 elements
if P2P_CHUNK <= 0 or P2P_CHUNK % 4 or P2P_CHUNK > ARENA_MAXB // 4:
    raise ValueError(f"P2P_CHUNK must be a positive multiple of 4 fp32 elements, at most {ARENA_MAXB // 4}")


class TPBackendP2P:
    """Collective backend on P2PComm (one-shot cudaIpc / pinned-host one-shot all-reduce, fp32 only).
    Same hook surface as TPBackendNCCL. Non-fp32 tensors go through an fp32 staging buffer; payloads are
    chunked to fit the P2P arena (ARENA_MAXB, 4 MiB by default), and every rank must issue the same sequence
    of collectives."""
    piecewise_p2p_capture = True

    def __init__(self, comm):
        self.comm = comm
        self._stage = {}

    def _buf(self, t: torch.Tensor) -> torch.Tensor:
        key = (t.numel(), t.device)
        if key not in self._stage:
            self._stage[key] = torch.empty(t.numel(), dtype = torch.float32, device = t.device)
        return self._stage[key]

    def fwd_barrier(self):
        self.comm.barrier()

    def all_reduce(self, tensor: torch.Tensor, contribution: bool = True):
        if not contribution:
            tensor.zero_()
        direct = tensor.dtype == torch.float32 and tensor.is_contiguous()
        s = tensor.view(-1) if direct else self._buf(tensor)
        if not direct:
            s.copy_(tensor.reshape(-1))
        assert s.numel() % 4 == 0, f"P2P all_reduce payload {s.numel()} fp32 not 16-byte divisible"
        # chunk to the arena size; sizes depend only on numel, so both ranks issue the same sequence
        for off in range(0, s.numel(), P2P_CHUNK):
            self.comm.all_reduce_(s[off:off + P2P_CHUNK])
        if not direct:
            tensor.copy_(s.view(tensor.shape))

    def broadcast(self, tensor: torch.Tensor, src_device: int):
        raise NotImplementedError("tp_owner broadcast not supported by P2PComm backend; TP2 splits every module")

    def all_gather(self, t: torch.Tensor, world: int) -> torch.Tensor:
        """Concatenate the per-rank t along the last dim, in rank order, on both ranks (== cat(parts, -1))."""
        assert world == 2
        t = t.contiguous()
        n = t.numel()
        src = t.float().reshape(-1)
        flat = torch.empty(2 * n, dtype = torch.float32, device = t.device)
        # chunk to the arena size; both ranks issue the same sequence (sizes depend only on numel)
        for off in range(0, n, P2P_CHUNK):
            m = min(P2P_CHUNK, n - off)
            part = torch.empty(2 * m, dtype = torch.float32, device = t.device)
            self.comm.all_gather(src[off:off + m].contiguous(), part)
            flat[off:off + m].copy_(part[:m])
            flat[n + off:n + off + m].copy_(part[m:])
        shards = flat.view(2, *t.shape)
        return torch.cat([shards[0], shards[1]], dim = -1).to(t.dtype)

    def run_cpu_reduce_jobs(self):
        pass

    def end_cpu_reduce_jobs(self):
        pass

HYBRID_NCCL_MIN_BYTES = int(os.environ.get("HYBRID_NCCL_MIN_BYTES", 64 * 1024))


class TPBackendHybrid:
    """Use NCCL for large prefill reductions and P2P for every decode reduction.

    Decode transport must not depend on local graph-capture state: ranks can be
    capturing and replaying at different times, but must issue the same collective.
    """

    def __init__(self, nccl: "TPBackendNCCL", p2p: "TPBackendP2P"):
        self.nccl = nccl
        self.p2p = p2p
        self.prefill = False

    @property
    def piecewise_p2p_capture(self):
        return not self.prefill

    @property
    def disable_piecewise_capture(self):
        return self.prefill

    def sync_draft_ids(self, ids):
        shared = ids.contiguous()
        dist.broadcast(shared, src = 0)
        ids.copy_(shared)


    def fwd_barrier(self):
        self.p2p.fwd_barrier()

    def _pick(self, tensor: torch.Tensor):
        if not self.prefill:
            return self.p2p
        return self.nccl if tensor.numel() * tensor.element_size() >= HYBRID_NCCL_MIN_BYTES else self.p2p

    def all_reduce(self, tensor: torch.Tensor, contribution: bool = True):
        self._pick(tensor).all_reduce(tensor, contribution)

    def broadcast(self, tensor: torch.Tensor, src_device: int):
        self.nccl.broadcast(tensor, src_device)

    def all_gather(self, t: torch.Tensor, world: int) -> torch.Tensor:
        return self.p2p.all_gather(t, world)

    def run_cpu_reduce_jobs(self):
        pass

    def end_cpu_reduce_jobs(self):
        pass


class TPBackendTrace:
    """Opt-in host-side collective sequence trace; graph replays are not host calls."""
    trace_draft_window = True

    def trace_window(self, length):
        print(f"DRAFT_WINDOW rank={RANK} step={self.step} seq={self.sequence} length={length}", flush=True)

    def __init__(self, backend):
        self.backend = backend
        self.step = 0
        self.sequence = 0

    def __getattr__(self, name):
        return getattr(self.backend, name)

    def _trace(self, op, tensor, transport):
        self.sequence += 1
        print(f"COLLECTIVE rank={RANK} step={self.step} seq={self.sequence} op={op} "
              f"shape={tuple(tensor.shape)} dtype={tensor.dtype} bytes={tensor.numel() * tensor.element_size()} "
              f"transport={transport} capture={torch.cuda.is_current_stream_capturing()}", flush=True)

    def all_reduce(self, tensor, contribution=True):
        backend = self.backend
        transport = ("nccl" if backend._pick(tensor) is backend.nccl else "p2p") if isinstance(backend, TPBackendHybrid) else COMM
        self._trace("all_reduce", tensor, transport)
        return backend.all_reduce(tensor, contribution)

    def all_gather(self, tensor, world):
        self._trace("all_gather", tensor, "p2p" if COMM == "hybrid" else COMM)
        return self.backend.all_gather(tensor, world)

    def broadcast(self, tensor, src_device):
        self._trace("broadcast", tensor, "nccl" if COMM == "hybrid" else COMM)
        return self.backend.broadcast(tensor, src_device)

def set_tp_reduce(model: Model, backend: TPBackendNCCL) -> int:
    """Enable the attention/MoE partial-sum collectives on every TP-local block, and hand the backend to
    every forward through params (Generator builds its params internally, so the entrypoints inject it)."""
    n = 0
    for m in model.modules:
        if isinstance(m, TransformerBlock):
            m.attn.tp_reduce = True
            m.mlp.tp_reduce = True
            n += 1

    def with_backend(fn, is_prefill):
        def wrapped(input_ids, params = None):
            if params is None:
                params = {}
            params["backend"] = backend
            hybrid = backend.backend if isinstance(backend, TPBackendTrace) else backend
            from qmesh.tp2.prefill_ar import PrefillAR
            if isinstance(hybrid, PrefillAR):
                hybrid = hybrid.inner
            if not isinstance(hybrid, TPBackendHybrid):
                return fn(input_ids, params)
            previous = hybrid.prefill
            # MTP prompt processing uses forward to export final-layer states.
            hybrid.prefill = is_prefill or bool(params.get("last_tokens_only"))
            try:
                return fn(input_ids, params)
            finally:
                hybrid.prefill = previous
        return wrapped

    model.prefill = with_backend(model.prefill, True)
    model.forward = with_backend(model.forward, False)
    return n



def wrap_vocab_split(model: Model, rank: int, world: int, backend: TPBackendNCCL):
    """Vocab-split embedding (masked local lookup + all_reduce) and head (all_gather of local logits)."""
    emb = next(m for m in model.modules if m.key == "embed")
    head = model.modules[model.logit_layer_idx]
    V = emb.vocab_size
    lo = rank * V
    orig_emb = emb.forward
    orig_head = head.forward

    gpu = torch.device("cuda", rank)
    def embed_fwd(x, params, out_dtype = None):
        params.setdefault("input_ids", x)  # hash-MoE routing needs the global ids
        loc = x - lo
        mask = (loc >= 0) & (loc < V)
        loc = torch.where(mask, loc, torch.zeros_like(loc))
        y = orig_emb(loc, params, out_dtype)
        y = y.to(gpu) if y.device != gpu else y
        y = y * mask.to(gpu).unsqueeze(-1).to(y.dtype)
        backend.all_reduce(y)
        return y

    def head_fwd(x, params, out_dtype = None):
        y = orig_head(x, params, out_dtype)
        return backend.all_gather(y.contiguous(), world)

    emb.forward = embed_fwd
    head.forward = head_fwd
    return V, head.out_features * world


def main():
    if DSPARK_POLICY not in ("always", "auto"):
        raise ValueError("DSPARK_POLICY must be always or auto")
    if DSPARK_POLICY == "auto" and (not DSPARK or BATCH != 1 or DSPARK_DYNAMIC):
        raise ValueError("DSPARK_POLICY=auto requires DSPARK=1, BATCH=1 and DSPARK_DYNAMIC=0")
    torch.cuda.set_device(RANK)
    dist.init_process_group(backend = "nccl", rank = RANK, world_size = WORLD,
                            init_method = os.environ["INIT_METHOD"])
    device = torch.device("cuda", RANK)
    rank_dir = os.path.join(OVERLAY, f"rank{RANK}")

    t0 = time.time()
    config = Config.from_directory(rank_dir)
    assert config.tp_size == WORLD and config.tp_rank == RANK, \
        f"checkpoint tp_size/tp_rank {config.tp_size}/{config.tp_rank} != launch {WORLD}/{RANK}"
    model = Model.from_config(config)
    max_chunk, chunk_info = resolve_max_chunk(device)
    # Both ranks must process identical shapes even if their local budgets differ.
    agreed_chunk = torch.tensor(max_chunk, dtype = torch.int32, device = device)
    dist.all_reduce(agreed_chunk, op = dist.ReduceOp.MIN)
    max_chunk = int(agreed_chunk.item())
    if chunk_info is None:
        log(f"PREFILL_CHUNK fixed at {max_chunk} rows (MAX_CHUNK={MAX_CHUNK_RAW}), CTX_TOKENS={CACHE_TOKENS}")
    else:
        log(f"PREFILL_CHUNK auto -> {max_chunk} rows | total={chunk_info['total_mib']:.0f} MiB "
            f"kv={chunk_info['kv_mib']:.0f} MiB (ref {chunk_info['kv_ref_mib']:.0f}) "
            f"reserve={VRAM_RESERVE_MIB + (DSPARK_RESERVE_MIB if DSPARK else 0):.0f} MiB limit={chunk_info['limit_mib']:.0f} MiB "
            f"predicted_peak={chunk_info['peak_mib']:.0f} MiB free_now={chunk_info['free_mib']:.0f} MiB "
            f"CTX_TOKENS={CACHE_TOKENS}")
    cache = Cache(model, max_num_tokens = CACHE_TOKENS, max_history = MAX_HISTORY, max_batch_size = BATCH, **KV_CACHE_KWARGS)
    model.load(device = device, max_chunk_size = max_chunk, max_output_size = MAX_NEW,
               max_batch_size = BATCH, progressbar = False)
    if COMM in ("p2p", "hybrid"):
        from exllamav3.tp2.p2p import P2PComm
        rdv = os.environ["P2P_RDV"]
        p2p = TPBackendP2P(P2PComm(RANK, world = WORLD, rendezvous_path = rdv, device = RANK,
                                   stats = P2P_STATS_OUT is not None))
        backend = p2p if COMM == "p2p" else TPBackendHybrid(TPBackendNCCL(), p2p)
        vocab_backend = backend
        from qmesh.tp2.prefill_ar import wrap_prefill_ar
        backend = wrap_prefill_ar(backend, p2p)
        comm = p2p.comm if hasattr(p2p, "comm") else None
    else:
        comm = None
        backend = TPBackendNCCL()
        vocab_backend = backend
        if os.environ.get("PREFILL_AR", "0") != "0":
            raise ValueError("PREFILL_AR requires COMM=p2p or hybrid")
    if os.environ.get("TP_COLLECTIVE_TRACE") == "1":
        backend = TPBackendTrace(backend)
    n_blocks = set_tp_reduce(model, backend)
    local_v, full_v = wrap_vocab_split(model, RANK, WORLD, vocab_backend)
    load_s = time.time() - t0
    log(f"LOADED {load_s:.1f}s blocks={n_blocks} vocab_local={local_v} vocab_full={full_v} "
        f"max_alloc_GiB={torch.cuda.max_memory_allocated(device) / 2**30:.2f}")

    tokenizer = Tokenizer.from_config(config)
    draft_model = draft_cache = None
    if DSPARK:
        draft_model = Model.from_config(config, component = "mtp")
        draft_cache = Cache(draft_model, max_num_tokens = CACHE_TOKENS, max_history = MAX_HISTORY,
                            max_batch_size = BATCH, **KV_CACHE_KWARGS)
        draft_model.load(device = device, progressbar = False)
        n_draft = set_tp_reduce(draft_model, backend)
        log(f"DRAFT_LOADED blocks={n_draft} free_GiB={torch.cuda.mem_get_info(device)[0] / 2**30:.2f}")
        # Replicated confidence heads can differ numerically between ranks.
        # Agree on draft IDs and the confidence gate before verification.
        sample_from_state = draft_model.sample_from_state
        def synchronized_draft_sample(state, params):
            ids = sample_from_state(state, params)
            dist.broadcast(ids, src = 0)
            if "draft_confidence_len" in params:
                length = torch.tensor(params["draft_confidence_len"], dtype = torch.int32, device = device)
                dist.broadcast(length, src = 0)
                params["draft_confidence_len"] = int(length.item())
            if "draft_conf" in params:
                dist.broadcast(params["draft_conf"], src = 0)
            return ids
        draft_model.sample_from_state = synchronized_draft_sample
    # DSPARK_DYNAMIC=1: the generator's confidence calibrator sizes each round's window against DSPARK_TARGET_ACC
    # (drafter must export draft_conf; run with EXL3_DSPARK_CONF=0 so the static gate does not also cut)
    generator = Generator(model, cache, tokenizer, max_batch_size = BATCH,
                          max_chunk_size = max_chunk,
                          draft_model = draft_model, draft_cache = draft_cache,
                          recurrent_cache_size = PREFIX_CACHE_CAPACITY if PREFIX_CACHE else 0,
                          recurrent_checkpoint_interval = PREFIX_CACHE_INTERVAL if PREFIX_CACHE else None,
                          recurrent_checkpoint_interval_pp = PREFIX_CACHE_INTERVAL if PREFIX_CACHE else 32768,
                          num_draft_tokens = NUM_DRAFT if DSPARK else None,
                          dynamic_draft_tokens = DSPARK and DSPARK_DYNAMIC,
                          draft_confidence = DSPARK_TARGET_ACC,
                          record_draft_stats = DSPARK and (DSPARK_DYNAMIC or DSPARK_POLICY == "auto" or os.environ.get("EXL3_DRAFT_POSITION_STATS", "0") == "1"))
    if DSPARK:
        # State the effective window explicitly: NUM_DRAFT beyond the drafter's block is clamped, not honoured
        log(f"DRAFT_K requested={NUM_DRAFT} effective={generator.num_draft_tokens} "
            f"block={generator.max_draft_size} "
            f"clamped={'yes' if generator.num_draft_tokens != NUM_DRAFT else 'no'}")
    chunk_limiter = PrefillChunkLimiter(device) if PREFILL_ADAPTIVE else None
    generator.prefill_chunk_limit = chunk_limiter
    if MEM_BREAKDOWN:
        # Resident memory by owner after load (target + drafter + caches + generator statics), before any job
        from exllamav3.util.memory import vram_accounting
        g = 2 ** 30
        for name, m, c in (("target", model, cache), ("drafter", draft_model, draft_cache)):
            if m is None:
                continue
            r = vram_accounting(m, c, generator if name == "target" else None, [device])[0]
            log(f"MEMB {name} weights={r.weights / g:.3f} kv_cache={r.cache / g:.3f} "
                f"recurrent={r.recurrent / g:.3f} statics={r.statics / g:.3f} generator={r.generator / g:.3f} "
                f"other={r.other / g:.3f}")
        free, total = torch.cuda.mem_get_info(device)
        log(f"MEMB device total={total / g:.3f} used={(total - free) / g:.3f} "
            f"torch_reserved={torch.cuda.memory_reserved(device) / g:.3f} "
            f"torch_allocated={torch.cuda.memory_allocated(device) / g:.3f} "
            f"non_torch={(total - free - torch.cuda.memory_reserved(device)) / g:.3f}")
        torch.cuda.reset_peak_memory_stats(device)
    if isinstance(backend, TPBackendTrace):
        original_trace_iterate = generator.iterate
        def trace_iterate(*args, **kwargs):
            backend.step += 1
            log(f"COLLECTIVE_STEP rank={RANK} step={backend.step} sequence={backend.sequence}")
            return original_trace_iterate(*args, **kwargs)
        generator.iterate = trace_iterate
    generator.prefix_cache_min_match = PREFIX_CACHE_MIN_MATCH if PREFIX_CACHE else None
    generator.observe_draft_confidence = DSPARK_POLICY == "auto"
    generator.count_draft_forwards = DSPARK_COUNT_FORWARDS
    generator.draft_forward_count = 0
    draft_timing_rounds = []
    DIAG_SPLIT = os.environ.get("EXL3_DIAG_SPLIT", "0") == "1"
    DIAG_SPLIT_EVERY = int(os.environ.get("EXL3_DIAG_SPLIT_EVERY", "4"))
    if DRAFT_TIMING and DSPARK:
        # Events measure stream elapsed time, including any idle gaps, not kernel-only time.
        current_events = None
        sample_now = False
        def wait_total():
            if not DIAG_SPLIT or comm is None or not getattr(comm, "stats_enabled", False):
                return 0
            return sum(b["wait_ns"] for b in comm.stats()["blocks"])
        def timed_forward(original, component):
            def call(*args, **kwargs):
                if current_events is None:
                    return original(*args, **kwargs)
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                w0 = wait_total() if sample_now else 0
                h0 = time.perf_counter()
                start.record()
                try:
                    return original(*args, **kwargs)
                finally:
                    end.record()
                    w1 = wait_total() if sample_now else 0
                    current_events.append((component, start, end, w1 - w0, time.perf_counter() - h0))
            return call
        model.forward = timed_forward(model.forward, "target")
        draft_model.forward = timed_forward(draft_model.forward, "draft")
        original_iterate = generator.iterate
        def timed_iterate(*args, **kwargs):
            nonlocal current_events, sample_now
            events = []
            current_events = events
            sample_now = DIAG_SPLIT and (len(draft_timing_rounds) % DIAG_SPLIT_EVERY == 0)
            start = time.perf_counter()
            try:
                return original_iterate(*args, **kwargs)
            finally:
                wall = time.perf_counter() - start
                current_events = None
                sample_now = False
                if any(component == "draft" for component, *_ in events):
                    draft_timing_rounds.append((wall, events))
    FORCE_EXPORT = os.environ.get("FORCE_EXPORT", "0") == "1"
    if FORCE_EXPORT and not DSPARK:
        _export_layers = set(config.dspark_target_layer_ids)
        _orig_forward = model.forward
        def _forward_export(*a, params = None, **kw):
            params["export_state_layers"] = _export_layers
            return _orig_forward(*a, params = params, **kw)
        model.forward = _forward_export
        log(f"FORCE_EXPORT layers={sorted(_export_layers)}")
    dist.barrier()
    # Warm pass: one short job so graph capture and allocator warmup stay out of the timed prompts
    _warm_text = encode_messages([{"role": "user", "content": PROMPTS[PROMPT_IDS[0]]}], thinking_mode = "chat")
    _warm_ids = tokenizer.encode(_warm_text, add_bos = False, encode_special_tokens = True)
    if WARM_TOKENS:
        for _ in range(BATCH):
            _warm = Job(input_ids = _warm_ids, max_new_tokens = WARM_TOKENS, min_new_tokens = WARM_TOKENS,
                        sampler = RankAgreedGreedySampler(), stop_conditions = [])
            generator.enqueue(_warm)
    while generator.num_remaining_jobs():
        for _ in generator.iterate():
            pass
    out = {}
    logit_out = {}
    WARM = os.environ.get("WARM", "0") == "1"
    run_ids = (PROMPT_IDS + PROMPT_IDS) if WARM else PROMPT_IDS
    if BATCH > 1:
        # Concurrent streams: one job per slot, prompts cycled from PROMPT_IDS. All jobs are enqueued before the
        # first iterate so the generator admits them together. Results are keyed "<prompt idx>.<slot>".
        jobs, streams = [], []
        for slot in range(BATCH):
            idx = PROMPT_IDS[slot % len(PROMPT_IDS)]
            text_in = encode_messages([{"role": "user", "content": PROMPTS[idx]}], thinking_mode = "chat")
            ids = tokenizer.encode(text_in, add_bos = False, encode_special_tokens = True)
            job = Job(input_ids = ids, max_new_tokens = MAX_NEW,
                      min_new_tokens = MAX_NEW if IGNORE_EOS else 0, sampler = RankAgreedGreedySampler(),
                      stop_conditions = [] if IGNORE_EOS else [tokenizer.eos_token_id],
                      return_logits = bool(LOGITS_OUT))
            jobs.append(job)
            streams.append({"key": f"{idx}.{slot}", "idx": idx, "serial": None, "tokens": [], "marks": [],
                            "logits": [], "eos": None, "prompt_tokens": ids.shape[-1]})
            generator.enqueue(job)
        by_serial = {j.serial_number: s for j, s in zip(jobs, streams)}
        t1 = time.time()
        while generator.num_remaining_jobs():
            for result in generator.iterate():
                if "error" in result:
                    raise result["error"]
                s = by_serial.get(result.get("serial"))
                if s is None:
                    continue
                if "token_ids" in result:
                    s["tokens"] += result["token_ids"][0].tolist()
                    s["marks"].append(time.time() - t1)
                if "logits" in result and LOGITS_OUT:
                    s["logits"].append(result["logits"].float().cpu())
                if result.get("eos"):
                    s["eos"] = result.get("eos_reason")
        wall = time.time() - t1
        agg = 0
        for s in streams:
            n = len(s["tokens"])
            agg += n
            first = s["marks"][0] if s["marks"] else float("nan")
            decode = (s["marks"][-1] - first) / max(n - 1, 1) if n > 1 else float("nan")
            out[s["key"]] = s["tokens"]
            if LOGITS_OUT and s["logits"]:
                rows = torch.cat([row.reshape(-1, row.shape[-1]) for row in s["logits"]])
                if os.environ.get("LOGITS_FULL") == "1":
                    torch.save(rows, f"{LOGITS_OUT}.{s['key']}.rank{RANK}.pt")
                values, indices = torch.topk(rows, 5, dim = -1)
                logit_out[s["key"]] = {"values": values.tolist(), "indices": indices.tolist()}
            log(f"STREAM {s['key']} prompt_tokens={s['prompt_tokens']} new_tokens={n} first_token_ms={first * 1e3:.1f} "
                f"decode_ms_per_tok={decode * 1e3:.2f} eos={s['eos']}")
            log(f"TOKENS {s['key']} {s['tokens']}")
        log(f"BATCH {BATCH} wall={wall:.3f}s aggregate_tok={agg} aggregate_tok_s={agg / wall:.2f} "
            f"per_stream_tok_s={agg / wall / BATCH:.2f}")
        run_ids = []
    for pass_i, idx in enumerate(run_ids):
        timed = not WARM or pass_i >= len(PROMPT_IDS)
        prompt = PROMPTS[idx]
        text_in = encode_messages([{"role": "user", "content": prompt}], thinking_mode = "chat")
        ids = tokenizer.encode(text_in, add_bos = False, encode_special_tokens = True)
        last_ids = ids
        job = Job(
            input_ids = ids,
            max_new_tokens = MAX_NEW,
            min_new_tokens = MAX_NEW if IGNORE_EOS else 0,
            sampler = RankAgreedGreedySampler(),
            stop_conditions = [] if IGNORE_EOS else [tokenizer.eos_token_id],
            return_logits = bool(LOGITS_OUT),
        )
        policy_class, policy_ema = content_seed.seed_acceptance(prompt)
        policy_off = DSPARK_POLICY == "auto" and policy_ema < content_seed.POLICY_THRESHOLD
        if DSPARK_POLICY == "auto":
            generator.num_draft_tokens = 0 if policy_off else min(NUM_DRAFT, generator.max_draft_size or NUM_DRAFT)
        draft_forward_start = generator.draft_forward_count
        if DSPARK_POLICY == "auto":
            log(f"POLICY {idx} class={policy_class} seed_al={policy_ema:.3f} speculative={not policy_off}")
        generator.enqueue(job)
        forced_path = os.environ.get("EXL3_DIAG_FORCE_TOKENS")
        if forced_path:
            with open(forced_path) as f:
                reference_tokens = json.load(f)[str(idx)]
            job.constrain_output_now(torch.tensor([reference_tokens], dtype = torch.long))
        t1 = time.time()
        marks, text, tokens, eos_info, logit_rows, spec = [], "", [], None, [], None
        while generator.num_remaining_jobs():
            stats_before = len(job.draft_stats)
            results = generator.iterate()
            if DSPARK_POLICY == "auto" and not policy_off and len(job.draft_stats) > stats_before:
                for _, window, accepted in job.draft_stats[stats_before:]:
                    predicted = 1 + window * generator.last_draft_confidence
                    observed = (1 - content_seed.POLICY_CONF_WEIGHT) * (1 + accepted) + content_seed.POLICY_CONF_WEIGHT * predicted
                    policy_ema += (observed - policy_ema) / content_seed.POLICY_EMA_WINDOW
                if policy_ema < content_seed.POLICY_THRESHOLD:
                    policy_off = True
                    generator.num_draft_tokens = 0
                    log(f"POLICY_OFF {idx} generated={job.new_tokens} ema_al={policy_ema:.3f}")
            for result in results:
                if "error" in result:
                    raise result["error"]
                chunk = result.get("text", "")
                if chunk:
                    text += chunk
                    marks.append(time.time() - t1)
                if "token_ids" in result:
                    tokens += result["token_ids"][0].tolist()
                if "logits" in result and LOGITS_OUT:
                    logit_rows.append(result["logits"].float().cpu())
                if "accepted_draft_tokens" in result:
                    spec = result
                if result.get("eos"):
                    eos_info = result.get("eos_reason")
        total = time.time() - t1
        if not timed:
            continue
        if DSPARK_COUNT_FORWARDS:
            log(f"DRAFT_FORWARDS {idx} count={generator.draft_forward_count - draft_forward_start}")
        first = marks[0] if marks else float("nan")
        decode = (total - first) / max(len(marks) - 1, 1) if len(marks) > 1 else float("nan")
        out[str(idx)] = tokens
        if LOGITS_OUT and logit_rows:
            if os.environ.get("LOGITS_FULL") == "1":
                torch.save(torch.cat([row.reshape(-1, row.shape[-1]) for row in logit_rows]),
                           f"{LOGITS_OUT}.{idx}.rank{RANK}.pt")
            top = []
            for row in logit_rows:
                v, i = torch.topk(row.reshape(-1, row.shape[-1]), 2, dim = -1)
                top.append({"top1": int(i[-1, 0]), "top2": int(i[-1, 1]), "gap": float(v[-1, 0] - v[-1, 1])})
            logit_out[str(idx)] = top
        log(f"PROMPT {prompt!r} prompt_tokens={ids.shape[-1]} new_tokens={len(tokens)} total={total:.3f}s "
            f"first_token_ms={first * 1e3:.1f} decode_ms_per_tok={decode * 1e3:.2f} eos={eos_info} text={text!r}")
        log(f"TOKENS {idx} {tokens}")
        seg_replays = 0
        for blk in model.modules:
            for segs in getattr(blk, "_segments", {}).values():
                for seg in (segs if isinstance(segs, tuple) else (segs,)):
                    seg_replays += getattr(seg, "replays", 0)
        log(f"SEG_REPLAYS {idx} {seg_replays}")
        if spec is not None:
            acc, rej, rnd = spec["accepted_draft_tokens"], spec["rejected_draft_tokens"], spec.get("verify_rounds", 0)
            acc_rate = acc / max(acc + rej, 1)
            tpr = 1 + acc / max(rnd, 1)
            log(f"SPEC {idx} accepted={acc} rejected={rej} acc_rate={acc_rate:.4f} verify_rounds={rnd} "
                f"tokens_per_round={tpr:.3f} eff_tok_s={len(tokens) / total:.2f}")
            if os.environ.get("EXL3_DRAFT_POSITION_STATS", "0") == "1" and job.draft_stats:
                positions = []
                for pos in range(1, NUM_DRAFT + 1):
                    attempted = sum(window >= pos and accepted >= pos - 1 for _, window, accepted in job.draft_stats)
                    accepted = sum(accepted >= pos for _, _, accepted in job.draft_stats)
                    positions.append((pos, attempted, accepted))
                log(f"DRAFT_POSITION {idx} (position,attempted,accepted)={positions}")
            if DSPARK_DYNAMIC and job.draft_stats:
                # draft_stats rows: (position, window, accepted-drafts) per verify round
                wins = [w for _, w, _ in job.draft_stats]
                acc_drafts = [a for _, _, a in job.draft_stats]
                hist = {w: wins.count(w) for w in sorted(set(wins))}
                log(f"DYN {idx} rounds={len(wins)} mean_window={sum(wins) / len(wins):.3f} "
                    f"window_hist={hist} mean_accepted={sum(acc_drafts) / len(acc_drafts):.3f}")
            if _DRAFT_HIST is not None:
                log(f"DRAFT_HIST {idx} accepted_drafts_hist={_DRAFT_HIST[:NUM_DRAFT + 1]}")
        log(f"TIMING {idx} total={total:.3f}s eff_tok_s={len(tokens) / total:.2f}")

    if PROBE and "0" in out:
        # Seq>1 probe: prompt + the first N generated tokens as ONE prefill forward, then the next-token logits.
        # Compared against the 1-token-decode reference (logit_out["0"][N]) to isolate the multi-row target path.
        ids0 = tokenizer.encode(encode_messages([{"role": "user", "content": PROMPTS[0]}], thinking_mode = "chat"),
                                add_bos = False, encode_special_tokens = True)
        for n in PROBE:
            pj = Job(input_ids = torch.cat([ids0, torch.tensor([out["0"][:n]], dtype = ids0.dtype)], dim = -1),
                     max_new_tokens = 1, min_new_tokens = 1, sampler = RankAgreedGreedySampler(), stop_conditions = [],
                     return_logits = True)
            generator.enqueue(pj)
            got = None
            while generator.num_remaining_jobs():
                for r in generator.iterate():
                    if "error" in r: raise r["error"]
                    if "logits" in r: got = r["logits"].float().cpu()
                    if "token_ids" in r: probe_tok = r["token_ids"][0].tolist()
            v, i = torch.topk(got.reshape(-1, got.shape[-1])[-1], 2)
            log(f"PROBE n={n} next_tok={probe_tok} top1={int(i[0])} top2={int(i[1])} gap={float(v[0]-v[1]):.5f}")
    if draft_timing_rounds:
        torch.cuda.synchronize(device)
        split_rows = []
        for i, (wall, events) in enumerate(draft_timing_rounds):
            draft = [(start.elapsed_time(end), w, h) for component, start, end, w, h in events if component == "draft"]
            target = [(start.elapsed_time(end), w, h) for component, start, end, w, h in events if component == "target"]
            if DIAG_SPLIT:
                split_rows.append((wall, draft, target))
            log(f"DRAFT_TIMING iteration={i} wall_ms={wall * 1000:.3f} "
                f"draft_n={len(draft)} draft_ms={sum(d for d, _, _ in draft):.3f} "
                f"target_n={len(target)} target_ms={sum(d for d, _, _ in target):.3f}")
        if split_rows:
            # Rounds with sampled p2p-wait snapshots only; the snapshot syncs the device, so these rows
            # attribute GPU-side collective wait, they are not timing arms.
            sampled = [(w, d, t) for w, d, t in split_rows if any(x[1] or x[2] for x in d + t)]
            if sampled:
                def mean(xs):
                    return sum(xs) / len(xs) if xs else float("nan")
                wall_ms = mean([w * 1000 for w, _, _ in sampled])
                draft_ms = mean([sum(x[0] for x in d) for _, d, _ in sampled])
                target_ms = mean([sum(x[0] for x in t) for _, _, t in sampled])
                draft_wait = mean([sum(x[1] for x in d) / 1e6 for _, d, _ in sampled])
                target_wait = mean([sum(x[1] for x in t) / 1e6 for _, _, t in sampled])
                draft_host = mean([sum(x[2] for x in d) * 1000 for _, d, _ in sampled])
                target_host = mean([sum(x[2] for x in t) * 1000 for _, _, t in sampled])
                log(f"ROUND_SPLIT rounds={len(sampled)} wall_ms={wall_ms:.3f} draft_ms={draft_ms:.3f} "
                    f"target_ms={target_ms:.3f} host_ms={wall_ms - draft_ms - target_ms:.3f} "
                    f"draft_wait_ms={draft_wait:.3f} target_wait_ms={target_wait:.3f} "
                    f"draft_py_ms={draft_host:.3f} target_py_ms={target_host:.3f}")
    run_peak = max(torch.cuda.max_memory_allocated(device), chunk_limiter.peak if chunk_limiter else 0)
    log(f"MEM max_alloc_GiB={run_peak / 2**30:.2f}")
    if chunk_limiter is not None and chunk_limiter.decide_us:
        us = sorted(chunk_limiter.decide_us)
        log(f"PREFILL_ADAPTIVE chunks={len(us)} shrunk_chunks={chunk_limiter.shrunk} "
            f"decide_us_median={us[len(us) // 2]:.0f} decide_us_max={us[-1]:.0f}")
    if MEM_BREAKDOWN:
        g = 2 ** 30
        log(f"MEMB peak_since_load allocated={run_peak / g:.3f} "
            f"reserved={torch.cuda.max_memory_reserved(device) / g:.3f} "
            f"device_used_now={(torch.cuda.mem_get_info(device)[1] - torch.cuda.mem_get_info(device)[0]) / g:.3f}")
    if LOGITS_OUT:
        with open(f"{LOGITS_OUT}.rank{RANK}.json", "w") as f:
            json.dump(logit_out, f)
    if TOKENS_OUT:
        with open(f"{TOKENS_OUT}.rank{RANK}.json", "w") as f:
            json.dump(out, f)

    PROFILE_SEQ = [int(n) for n in os.environ.get("PROFILE_SEQ", "").split(",") if n.strip()]
    if PROFILE and PROFILE_SEQ:
        # Width-filtered profile: profiles exactly one model.forward call per width in PROFILE_SEQ
        # (verify = seq 6, decode = seq 1), because torch.profiler clears events per cycle.
        from torch.profiler import profile, ProfilerActivity
        _orig_fwd = model.forward
        widths, done_w = {}, set()
        profs = {}
        def _filtered_forward(input_ids, params = None, **kw):
            w = int(input_ids.shape[-1])
            widths[w] = widths.get(w, 0) + 1
            if w not in PROFILE_SEQ or w in done_w or widths[w] < 3:
                return _orig_fwd(input_ids, params = params, **kw)
            done_w.add(w)
            p = profile(activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA],
                        with_stack = PROFILE_STACK, record_shapes = PROFILE_SHAPES)
            profs[w] = p
            torch.cuda.synchronize()
            p.start()
            try:
                return _orig_fwd(input_ids, params = params, **kw)
            finally:
                torch.cuda.synchronize()
                p.stop()
        model.forward = _filtered_forward
        _orig_draft_fwd = draft_model.forward if draft_model is not None else None
        _draft_prof = []
        _draft_calls = [0]
        if _orig_draft_fwd is not None and os.environ.get("PROFILE_DRAFT") == "1":
            def _profile_draft(*args, **kwargs):
                _draft_calls[0] += 1
                if _draft_calls[0] != 3:
                    return _orig_draft_fwd(*args, **kwargs)
                p = profile(activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA],
                            with_stack = PROFILE_STACK, record_shapes = PROFILE_SHAPES)
                _draft_prof.append(p)
                torch.cuda.synchronize()
                p.start()
                try:
                    return _orig_draft_fwd(*args, **kwargs)
                finally:
                    torch.cuda.synchronize()
                    p.stop()
            draft_model.forward = _profile_draft
        job = Job(input_ids = last_ids, max_new_tokens = 64, min_new_tokens = 64,
                  sampler = RankAgreedGreedySampler(), stop_conditions = [])
        generator.enqueue(job)
        while generator.num_remaining_jobs():
            for result in generator.iterate():
                if "error" in result: raise result["error"]
        model.forward = _orig_fwd
        if _orig_draft_fwd is not None:
            draft_model.forward = _orig_draft_fwd
        if RANK == 0 and _draft_prof:
            _draft_prof[0].export_chrome_trace(f"{os.environ.get('PROFILE_TRACE', '/artifacts/tp2logs/profile_trace.json')}.draft.json")
        if RANK == 0:
            log(f"PROFILE_SEQ widths_seen={sorted(widths.items())} recorded={sorted(profs)}")
            for w in sorted(profs):
                print(f"=== PROFILE seq={w} (one forward) ===", flush = True)
                print(profs[w].key_averages().table(sort_by = "cuda_time_total", row_limit = 30), flush = True)
                profs[w].export_chrome_trace(f"{os.environ.get('PROFILE_TRACE', '/artifacts/tp2logs/profile_trace.json')}.seq{w}.json")
    if PROFILE and not PROFILE_SEQ:
        from torch.profiler import profile, ProfilerActivity
        # both ranks must run the same collective sequence; only rank 0 prints the table.
        # Profiling starts after the first emitted token, so prefill is excluded: decode steps only.
        prof = profile(activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA],
                        with_stack = PROFILE_STACK, record_shapes = PROFILE_SHAPES)
        job = Job(input_ids = last_ids, max_new_tokens = 16, min_new_tokens = 16,
                  sampler = RankAgreedGreedySampler(), stop_conditions = [])
        generator.enqueue(job)
        started, steps, t_dec, wall_ms = False, 0, None, float("nan")
        while generator.num_remaining_jobs():
            for result in generator.iterate():
                if "token_ids" in result and not started:
                    torch.cuda.synchronize()
                    prof.start()
                    started, t_dec = True, time.time()
                elif "token_ids" in result:
                    steps += 1
        torch.cuda.synchronize()
        if started:
            wall_ms = (time.time() - t_dec) * 1e3 / max(steps, 1)
            prof.stop()
        if RANK == 0:
            log(f"PROFILE decode_steps={steps} wall_ms_per_step={wall_ms:.2f}")
            prof.export_chrome_trace(os.environ.get("PROFILE_TRACE", "/artifacts/tp2logs/profile_trace.json"))
            print(prof.key_averages().table(sort_by = "self_cpu_time_total", row_limit = 45), flush = True)

    if P2P_STATS_OUT and COMM == "p2p":
        # Rank-tagged dump after all collectives drained (stats() syncs the device). Written on every rank.
        path = comm.dump_stats(P2P_STATS_OUT)
        log(f"P2P_STATS dumped {path}")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
