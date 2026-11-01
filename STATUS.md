# STATUS — Recipe A (GPU0 + GX1)

Last update: 2026-11-01. Work paused by owner. Details and commands in `qmesh/notes/`.

## Caveat on hardware used so far
Several measurements below were taken on **P7 GPU1** by mistake (same card model as GPU0, so numbers
transfer, but the recipe itself is GPU0 + GX1 only). Marked [GPU1] where so.

## Measured
| Item | Result | Source |
|---|---|---|
| Handoff baseline (TCP/pickle transport, group worker) | ~100 ms/token (10 tok/s) | EXL3_HANDOFF.md |
| P7-local decode, null remote, eager | 18.6–19.6 ms/token, ~1,290 launches/token | notes/Profiler.md (GPU0) |
| P7-local decode, piecewise graphs `EXL3_WHOLE_GRAPH=1` | 18.92 → 14.59 ms/token, bit-exact (32/32 short, 64/64 long, diff 0.0), 1225 → 445 kernels + 87 graph launches | notes/GraphCapture.md (GPU0) |
| Mixed-K fused MoE | 1 launch for mixed 2/2.5/3/3.5/4 bpw; 259 MoE tests green; real-layer rel err ≤1.3e-3 | notes/MixedK.md [GPU1] |
| MoE coop kernel | ~780 GB/s (ceiling 1.65 TB/s), latency-bound; WIDE fix 108 → 93 µs/layer bsz 1 | notes/CoopPerf.md [GPU1] |
| Native RDMA seam (stub compute) | bsz1 16–23 µs/layer, graph-capturable | notes/Transport.md (GPU0 small) |
| GX1 fused mixed-K compute (layer 0, 259 cold) | rows=1 358 µs/layer, rows=6 2.0 ms/layer | notes/Transport.md |
| Round trip with real compute | rows=1 ~525 µs/layer pipelined (~21 ms/token for 40 layers) | notes/Transport.md |
| End-to-end hybrid over seam, coherent | 28–31 ms/token; DSpark K=5 best 17.6 ms/token (acceptance rate 0.74–0.82 per drafted token) | notes/E2ESeam.md [GPU1] |
| Dense defaults | HC fold on (bit-exact, −40 launches/token); x-fan opt-in (0.5–0.8 logit drift) | notes/DenseFuse.md [GPU1] |

## Main bottleneck for this recipe
GX1 cold-expert compute: ~358 µs/layer at rows=1 vs a bandwidth floor of roughly ~100 µs/layer
(≈1.2 GB/token cold reads at 273 GB/s, spread over 40 layers) — [INFERENCE] floor estimate.

## Not done
- Re-run E2E with the P7 side on **GPU0** (was run on GPU1).
- GX1 compute speed-up; seam issue/collect inside the piecewise graphs; real Engram device path.

## Artifacts outside git
- `/home/treadall/recipe-artifacts/gpu0-gx1/densefuse_trace.json`
- Seam binaries: rebuild with `qmesh/native/build.sh` (P7 x86_64, GX1 aarch64).

## Known incompatibility (2026-11-01)
`EXL3_WHOLE_GRAPH=1` and the MTP drafter (`DSPARK=1`) cannot be combined. The piecewise decode
graphs replay the blocks, so the module-level export taps (`exllamav3/modules/transformer.py:253`,
`rmsnorm.py:169`) never run and `params["export_states"]` is never filled. The drafter then hits
`deepseek_v4_mtp.py:490 T = target_hidden[0].shape[1]` with `target_hidden = None` and raises
`TypeError: 'NoneType' object is not subscriptable` on the first drafted step (observed as the
warm-up failure of `final_prod3`). Until the graph path exports the collapsed hidden state of the
last trunk block (or drafting is skipped while the export is missing), run either
`WHOLE_GRAPH=0 DSPARK=1` or `WHOLE_GRAPH=1 DSPARK=0`. Memory is identical either way; eager vs
graph decode measured neutral on this pair (25.38 vs 25.97 ms/token) while the drafter is worth
~1.20x. `scripts/mem_budget_run.sh` takes `WHOLE_GRAPH` (default 1); the production arms run
`WHOLE_GRAPH=0 DSPARK=1`.

## Client transport switch (cost one night of failed arms)
The qes seam is only used when the client sets `REMOTE_TRANSPORT=seam` (`block_sparse_mlp_cpu.py:891`);
the default `tcp` sends the legacy pickle protocol to the seam server, which reads it as a hello,
prints `qes: peer layout mismatch` and exits while the client reports `ConnectionResetError`.
`mem_budget_run.sh` now passes `REMOTE_TRANSPORT=seam` and `SEAM_BUCKETS` explicitly.

## Needle prompt format: the long-context arm was scoring a different prompt (2026-11-01)
The 131k arm of `qmesh/mem_budget.py` read the needle fixture as raw text with `add_bos=True`. That is
not the prompt v1.2 scored 5/5 on. The reference harness (`artifacts/2x-rtx-tp2/fp4kv/needle.sh`,
`PROMPT_IDS=8`) builds `PROMPTS[8] + "\n\n" + transcript` (`tp2/run_tp2.py:189`) and wraps it in the
checkpoint's own chat template, `encoding.encode_messages([{"role": "user", "content": prompt}],
thinking_mode="chat")`, then encodes with `add_bos=False, encode_special_tokens=True`
(`qmesh/tp2_bench.py:66-71`). Without the template the base model continues the prose instead of
answering the request block, which is what `final_prod7` produced.

`LONG_CHAT=1` now takes the reference path (`ENC_DIR` defaults to
`/models/dsv41/DeepSeek-V4.1-Flash-1017-orig/encoding`, pure python, no torch), prints
`LONG_CHAT lead=... chars=...`, `LONG_PROMPT tokens=...` and `LONG_PROMPT_TAIL`, and `IGNORE_EOS=1`
mirrors `tp2_bench.run_prompt` (`min_new_tokens=max_new`, no stop conditions). Generated ids are read
from `token_ids` (the key the generator reports, and the one `tp2_bench` reads) with `ids` as fallback;
the earlier arm printed `LONG_TEXT_IDS []` because only `ids` was read. Token counts with the template:
short fixture 3122, 131k fixture 131 233 (raw 131 222). `mem_budget_run.sh` now defaults `DEPTH_PROBES`
with `${VAR-default}`, so `DEPTH_PROBES=""` really disables the 512k/1M probes.

## Prefill chunk limiter: first-chunk term (2026-11-01)
`qmesh/prefill_limit.py` predicted 3056 MiB for an 8192-row chunk and measured it on 15 of 16 chunks;
the first chunk rose 3573 MiB. The extra ~517 MiB is one-time (uncompressed-pool staging window and
indexer score slab: `statics` 0.000 GiB after load, 0.559 GiB at 131k) and stays allocated, so it is
charged to the chunk that allocates it: `PREFILL_FIRST_MIB=560` is added to `prefill_rise` and removed
from the room in `prefill_fit` only when `start == 0`. Later chunks keep the old prediction (3056
measured / 3056 predicted) and an 8192-row first chunk still fits the measured 11403 MiB of room.
