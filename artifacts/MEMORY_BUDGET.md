# Memory budget — DeepSeek-V4.1-Flash-1017-EXL3-mixed-2.8535, Recipe A (P7 + GX1)

All figures are measured on the production pair unless labelled DEV. Unit = GiB unless stated.
Load config: `EXL3_WHOLE_GRAPH=0 DSPARK=1 NUM_DRAFT=5 EXL3_DSPARK_GATE=auto KV_BITS=4
CACHE_TOKENS=1048576 EXL3_LOAD_ARENA=0 PREFILL_ADAPTIVE=1 MAX_CHUNK=8192`,
checkpoint `/models/dsv41/exl3-2.8535-p7hot-hc1`.

## Owner summary (production pair, measured unless marked)

- **Free after load: 13.11 GiB** (`EXL3_LOAD_ARENA=0`, `MEMB after_load used=81.861` of 94.970).
- **Free at the prefill peak: 8.86 GiB** (`MEM_PEAK prefill peak_reserved=86.107` at 131k).
- **Per-chunk rise: 3.0 GiB** for an 8192-row chunk (3056 MiB, 15 of 16 chunks); the **first chunk rose
  3.5 GiB** (3573 MiB) because the uncompressed-pool staging window and indexer slab are allocated on
  first use.
- **Overhead to keep free = 6.25 GiB** = 13.11 after load − 6.86 room. Breakdown: 3.5 GiB first-chunk
  transient (8192 rows, measured 3573 MiB) + 0.56 GiB statics at 131k (measured, 0.000 at load) + 2.0 GiB
  margin for allocator slack and the limiter's own prediction error. The margin is the owner's choice.
- **Room for extra experts = 6.86 GiB** [INFERENCE] (= 8.86 free at prefill peak − the same 2.0 margin).
  Expert = 35.4 M params ⇒ 4.22 MiB per bpw per expert; the pack's hot set measures 16.92 MiB/expert,
  so at 4 bpw one expert is 16.88 MiB:

| bpw | MiB/expert | experts in 6.86 GiB |
|---|---|---|
| 2 | 8.44 | 832 |
| 2.5 | 10.55 | 666 |
| 3 | 12.66 | 555 |
| 4 | 16.88 | 416 |

- **Margin sensitivity**: each 1 GiB taken off the margin buys **+60 experts at 4 bpw / +121 at 2 bpw**
  (1024 / 16.88 and 1024 / 8.44).

Details, per-device tables and caveats below. Sections that report the long arm carry the run log name;
rows marked *planning* come from the placement ledger and are not measurements.

## P7 (RTX PRO 6000, 94.970 GiB usable) — target + HC1 drafter

Source: `MEMB after_load` / `MEMB drafter_after_load` / `MEM_WITEM` of the `final_prod` load
(log `artifacts/memb/final_prod.log`), identical on the DEV pair.

| piece | GiB | note |
|---|---|---|
| target weights (`MEMB weights`) | 73.979 | engine-owned tensors |
| target tensor bytes (`MEM_WITEM TOTAL`, 40256 views) | 75.348 | sums aliased views; see arena note |
| KV cache pool | 0.869 | 1 048 576 tokens, 890.0 B/token (0.054 @64k, 0.109 @131k) |
| recurrent state | 0.500 | linear-attention state, fixed |
| statics | 0.000 | — |
| generator / drafter (HC1, `n_views=0`) | 5.611 | head 0.123, shared.trellis 0.049, ring 0.036 |
| other | 0.199 | fused-mode temp buffers 0.056+0.056, temp_intermediate 0.025+0.025 |
| **torch_allocated** | **81.177** | reserved 81.189 |
| non_torch (CUDA ctx, NCCL, cuBLAS workspaces) | 0.671 | |
| **free** | **13.109** | of 94.970 total |

Largest single items in the target: `inner.trellis` (experts) 72.751, KV `inner.weight` 0.735,
`pool_fp4` 0.625, `ring` 0.469, `pool_idx_fp4` 0.156, `pool_fp4_scale` 0.078,
hyperconnection functions 0.073 + 0.073.

### The 1.0 GiB "config-allocated buffer" — resolved
It was never a model buffer. The loader's slab arena (default `EXL3_ARENA_BLOCK_MIB=128`,
8 retained blocks = 1.024 GiB) keeps its block storages alive, and `vram_accounting` counts
them inside `weights`. `MEM_WITEM … engine=73.979 arena_tail=-1.369` under
`EXL3_LOAD_ARENA=0` shows the engine now counts *less* than the tensor byte total, i.e. no
hidden allocation is left. Measured arms (all with the same 40256 tensors unless noted):

| arm | allocated | free | verdict |
|---|---|---|---|
| base (arena on, 128 MiB blocks) | 82.830 | 11.453 | — |
| `ARENA_COMPACT` | 74.557 (−143 tensors) | — | unsafe, rejected |
| `EXL3_ARENA_BLOCK_MIB=32` | 90.049 | 4.246 | −7.2 GiB, rejected |
| `EXL3_ARENA_MAX_OPEN=2` | 83.830 | 10.457 | worse |
| **`EXL3_LOAD_ARENA=0`** | **81.177** | **13.109** | **shipped** (also fastest load: 5.09 s) |

## Long-context arm: 131 222-token prompt (production pair)

Source: `final_prod7` (log `artifacts/memb/final_prod7.log`), one process, load once, arms then the long
prompt. Prompt `/artifacts/2x-rtx-tp2/fp4kv/needle/prose_128k_needle.txt` encodes to 131 222 tokens.

| stage | figure | note |
|---|---|---|
| steady after load | alloc 81.177 / free 13.109 | identical to the short-context load |
| prefill, 16 chunks x 8192 rows | 48.3-49.9 s each, 164-173 rows/s | first chunk 48.4 s |
| prefill total | 777.44 s (wall 777.79), **168.7 tok/s** | chunks=16, shrunk=0 |
| transient per 8192-row chunk | rise 3056 MiB = predicted | row term 382 KiB/row dominates |
| first chunk (start=0) | rise **3573** MiB vs predicted 3056 | +517 MiB: depth/base term has no floor |
| peak, prefill | alloc **84.986**, reserved **86.107** | rise_alloc 3.497, **8.863 free at peak** |
| decode, 80 greedy tokens | 3.79 s = **47.4 ms/token** | vs 23.687 ms/token short-context (DEV): 2.0x |
| peak, decode | alloc 82.278, reserved 82.318 | rise 0.252 |
| `MEMB after_long` | weights 74.199, kv 0.869, rec 0.500, statics **0.559**, gen 5.621, other 0.202 | |
| device after long | alloc **82.042**, reserved 82.318, non_torch 0.793, **free 11.859** | total 94.970 |

Two things this arm changed in the accounting: `statics` is 0.000 at load and **0.559 GiB at 131k** (the
uncompressed-pool staging window and indexer score slab are allocated on first use at depth, which is what
the depth term in `qmesh/prefill_limit.py` models), and `weights` grows 73.979 -> 74.199 on first decode.
Worst case on the card is therefore the prefill peak (86.107 reserved + 8.863 free), not the after-load
figure. [INFERENCE] A 1 M-token prompt would hold ~9 GiB of headroom at this chunk size only if the
depth term stays near its 131k slope (0.559 GiB of statics there); it is not measured, and `statics`
grows with context, so treat 131k as the last verified point.

### Prefill is seam-bound, not GPU-bound

One 8192-row chunk is 128 bucket-64 seam calls per layer x 40 layers = 5120 calls; at the measured
8871.2 us for a 64-row round trip that is **45.4 s of the 48.4 s chunk (94 %)**. The card side has 8192
rows to push in the remaining 3 s. Halving prefill time needs fewer/larger seam trips or overlap, not a
faster attention kernel; see the "do not retry" list in `HISTORY_REVIEW.md` before proposing a transport.

### Needle check: not answered — cause under investigation

The fixture ends with `--- Final request ---` and five numbered questions. The arm generated 80 greedy
tokens that close the document ("--- End of request ---") and continue the narrative, so the request was
seen but not followed, and no answer was produced.

The likely cause is prompt format, not memory and not the tail: the arm fed the file as raw text with
`add_bos=True` and no chat template, while the v1.2 reference run (`RESULTS.md:530`, 5/5 on this exact
file, `fp4kv/needle.sh`, `PROMPT_IDS=8`) builds the prompt as
`"Name three planets in our solar system.\n\n" + <transcript>` and wraps the whole thing in the
checkpoint's own `encoding.encode_messages([{"role": "user", ...}], thinking_mode = "chat")`, encoding
with `add_bos = False`. The engine's chunking is not implicated: `job.py:1421-1423` runs the final chunk
at `min(kv_position + chunk, len(ids) - 1)` = 149 rows with the limiter skipped (it prints no
`PREFILL_CHUNK` line, which is why the log shows `chunks=16`), and `prefill_done()` then fires at
`kv_position = 131 221`, i.e. the request block was prefilled.

Next test, in order: reproduce the reference formatting in `qmesh/mem_budget.py` (`LONG_LEAD` +
`LONG_CHAT=1`), verify on a short needle fixture (facts + request, ~8k tokens, ~1 min of GPU), then one
131k run. Until that run exists, this section records "not answered" only.

## GX1 (GB10, 121.7 GiB host, 117.0 GiB budget)

Placement produced by `qmesh/pack_hot.py`: 4100 hot experts on P7, the rest cold on GX1. Rows marked
**planning** are ledger entries from that placement, not measurements of the production run; only the
server RSS row is measured here.

| piece | GiB | note |
|---|---|---|
| hot experts (P7) | 67.736 | planning |
| P7 total (weights + hot) | 91.561 | planning, budget 94.970 — the shipped engine actually held 81.177 measured, see the table above |
| GX1 cold experts | 113.475 | planning, budget 117.0 |
| dense layer | 5.461 | planning |
| non-layer | 9.499 | planning |
| server RSS (idle, listening) | 1.14 | cgroup 1.70 GB |

## Seam cost (GX1, warm)

Per-layer round trip, 40 layers, measured (`dev_us`):

| rows | 1 | 2 | 3 | 4 | 5 | 6 | 64 |
|---|---|---|---|---|---|---|---|
| µs | 591.1 | 919.1 | 1209.6 | 1536.1 | 1746.4 | 2015.5 | 8871.2 |

Fit: 343.3 µs fixed + 283.7 µs per row. Transport-only (no compute) is 49.4–74.7 µs, i.e.
**2.5–3.7 % of the seam cost is wire**; the rest is GX1 compute. The 2 517 606 µs `p50` in the
first profiling run was a cold first call and is not a steady-state figure.

## Caveats

- Any long-context number recorded before commit `a69950e8` is invalid: `mem_budget.py` had an
  unbound `n` that aborted every long-context arm before the prompt was fed.
- `qmesh/arena_probe.py` prints bogus `MEMB drafter` / `MEM_WITEM drafter engine=` values (it
  reports the target); use its itemized drafter total (5.611) and its `end allocated/free`.
- `MEMB weights` (engine accounting) and `MEM_WITEM TOTAL` (per-view sum) are not the same
  quantity; use `MEM_WITEM` only for itemization and `torch_allocated` for the budget.
- The tree this was measured on needed two repairs before it could build at all: the standalone rebase had
  dropped `exllamav3/exllamav3_ext/{dsa_topk.cu,dsa_topk.cuh}` (`df856721`) and
  `parallel/{p2p_oneshot.cu,p2p_oneshot.cuh,prefill_ar.cu,prefill_ar.cuh}` (`3d1fde98`), the latter while
  `bindings.cpp` still referenced them, so every earlier run was on a stale prebuilt `.so`. Kernel tests now
  pass on the restored sources: `tests/kernels/dsa/test_dsa_topk_tiles.py` + `test_dsa_kernels.py` = 115
  passed in 115.5 s, and the built extension exports `dsa_compact_ent`, `dsa_compact_ent_rows`,
  `dsa_compact_ent_rows_map`, `prefill_ar_alloc`. Test the exported extension as `exllamav3.ext`'s
  attribute `exllamav3_ext`, not as `exllamav3.ext.<name>` (`ext.py` re-exports the module; the
  `exllamav3_ext` directory has no `__init__.py`).
- A run killed with SIGKILL leaves the GX seam server resident (it does not exit on client disconnect and
  holds the ~113 GiB pack); `mem_budget_run.sh`'s exit trap is bypassed, so pkill it in the container.
  A stale `~/.cache/torch_extensions/py312_cu130/exllamav3_ext/{lock,.ninja_lock}` makes the next server
  block forever in `torch/utils/file_baton.py`.
- `DEPTH_PROBES=""` does not disable the depth probes in this wrapper: the empty value falls back to the
  default `524288,1048570`. Stop the run after the long arm if the probes are not wanted.
