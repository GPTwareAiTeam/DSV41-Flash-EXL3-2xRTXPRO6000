"""
MixedK per-layer decode timing (item 4). Runs on one GPU (CUDA_VISIBLE_DEVICES=0 inside exl3-worker).

Measures BlockSparseMLP.forward at bsz 1 and 6, eager and CUDA graph, for:
  - mixed: real DeepSeek-V4.1-Flash-1017 mixed-2.565 layer 3 (per-expert K in {2..4})
  - uniform: the same layer geometry with every expert at K=3.0 (synthetic, random trellis)
Routing is pinned (external_selected / external_weights) and identical across runs.
Effective GB/s = bytes of the routed experts' packed weights touched per call / time.
"""
import json
import os
import statistics
import sys
import time

import torch

from exllamav3 import Config
from exllamav3.modules.block_sparse_mlp import BlockSparseMLP

CKPT = "/models/dsv41/exl3-out/DeepSeek-V4.1-Flash-1017-EXL3-mixed-2.565-tp2"
LAYER = 3
ITERS = int(os.environ.get("BENCH_ITERS", "200"))
WARM = 20


def build(cfg, device, key):
    mlp = BlockSparseMLP(
        config = cfg,
        key = key,
        hidden_size = cfg.hidden_size,
        intermediate_size = cfg.moe_intermediate_size,
        num_experts = cfg.num_experts,
        num_experts_per_tok = cfg.num_experts_per_tok,
        key_up = "experts.{expert_idx}.w3",
        key_gate = "experts.{expert_idx}.w1",
        key_down = "experts.{expert_idx}.w2",
        key_routing_gate = "gate",
        key_e_score_bias = "gate.bias",
        qmap = "block.mlp",
        interm_dtype = torch.half,
        out_dtype = torch.float,
        activation_fn = "silu",
        act_limit = cfg.swiglu_limit,
        router_type = "sqrtsp",
    )
    mlp.load(device)
    return mlp


def expert_bytes(mlp):
    """Packed EXL3 bytes of one routed expert (gate+up+down trellis), from the loaded tensors."""
    b = 0
    for lin in list(mlp.gates) + list(mlp.ups) + list(mlp.downs):
        b += lin.inner.trellis.numel() * lin.inner.trellis.element_size()
    return b / mlp.num_experts


def time_eager(fn):
    for _ in range(WARM):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(ITERS):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1e6)
    return statistics.median(ts)


def time_graph(fn):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(WARM):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    torch.cuda.synchronize()
    for _ in range(WARM):
        g.replay()
    torch.cuda.synchronize()
    ts = []
    for _ in range(ITERS):
        t0 = time.perf_counter()
        g.replay()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1e6)
    return statistics.median(ts)


def run_case(mlp, bsz, device, mode):
    gen = torch.Generator(device = "cpu").manual_seed(bsz)
    E, k = mlp.num_experts, mlp.num_experts_per_tok
    x = (torch.randn((bsz, mlp.hidden_size), generator = gen) * 0.05).half().to(device)
    sel = torch.stack([torch.randperm(E, generator = gen)[:k] for _ in range(bsz)]).to(device)
    w = (torch.rand((bsz, k), generator = gen) * 0.5 + 0.1).half().to(device)
    params = {"external_selected": sel, "external_weights": w}
    x3 = x.view(1, bsz, mlp.hidden_size)

    def fn():
        mlp.forward(x3, params)

    us = time_eager(fn) if mode == "eager" else time_graph(fn)
    # routed experts touched: distinct experts over the bsz*k slots
    touched = len(torch.unique(sel))
    gbps = touched * expert_bytes(mlp) / (us * 1e-6) / 1e9
    return us, touched, gbps


def main():
    device = torch.device("cuda:0")
    cfg = Config.from_directory(CKPT)
    mixed = build(cfg, device, f"layers.{LAYER}.ffn")
    assert mixed.mixed_k, "expected per-expert mixed rates on layer 3"
    rows = []
    for bsz in (1, 6):
        for mode in ("eager", "graph"):
            us, touched, gbps = run_case(mixed, bsz, device, mode)
            rows.append({"variant": "mixed", "bsz": bsz, "mode": mode, "us": round(us, 2),
                         "experts_touched": touched, "GBps": round(gbps, 2)})
            print(json.dumps(rows[-1]), flush = True)
    json.dump(rows, sys.stdout, indent = 1)


if __name__ == "__main__":
    main()
