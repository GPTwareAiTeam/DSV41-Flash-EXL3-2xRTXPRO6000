import json
import torch

from exllamav3 import Config, Model

CKPT = "/models/dsv41/exl3-serve"
PLAN = __file__.rsplit("/", 1)[0] + "/placement.exl3.json"
LAYER = 0


def ffn_forward(ffn, x, ids, weights):
    """Reference per-expert path for the cold subset (one row per assignment)."""
    out = torch.zeros((x.shape[0], ffn.hidden_size), dtype=torch.float32, device=x.device)
    for row in range(x.shape[0]):
        for local, weight in zip(ids[row].tolist(), weights[row].tolist()):
            if local < 0:
                continue
            xi = x[row:row + 1]
            g = ffn.gates[local].forward(xi, {})
            u = ffn.ups[local].forward(xi, {})
            h = torch.nn.functional.silu(g) * u
            d = ffn.downs[local].forward(h, {})
            out[row] += d.float().reshape(-1) * weight
    return out


def main():
    plan = json.load(open(PLAN))
    layer = plan["layers"][str(LAYER)]
    cold = layer["warm"]
    config = Config.from_directory(CKPT)
    config.infer_params.no_reconstruct = True
    model = Model.from_config(config)
    block = next(m for m in model.modules if m.key == f"layers.{LAYER}")
    ffn = block.mlp
    device = torch.device("cuda:0")

    # Full module reference, then prune/reload the cold subset.
    ffn.load(device)
    torch.cuda.synchronize()
    x = torch.randn(4, ffn.hidden_size, dtype=torch.half, device=device) * 0.02
    local_ids = torch.tensor([[0, 1, -1, -1], [2, 3, -1, -1], [0, 2, -1, -1], [1, 3, -1, -1]], dtype=torch.long, device=device)
    global_ids = torch.tensor([[cold[0], cold[1], -1, -1], [cold[2], cold[3], -1, -1], [cold[0], cold[2], -1, -1], [cold[1], cold[3], -1, -1]], dtype=torch.long, device=device)
    weights = torch.tensor([[0.5, 0.5, 0.0, 0.0]] * 4, dtype=torch.float32, device=device)
    ref = ffn_forward(ffn, x, global_ids, weights)
    print("FULL_OK", torch.isfinite(ref).all().item(), ref.float().abs().max().item(), flush=True)
    ffn.unload()
    torch.cuda.empty_cache()

    # Prune to the cold list in the same order the placement uses.
    ffn.gates = [ffn.gates[e] for e in cold]
    ffn.ups = [ffn.ups[e] for e in cold]
    ffn.downs = [ffn.downs[e] for e in cold]
    keep = set(ffn.gates + ffn.ups + ffn.downs)
    ffn.modules = [m for m in ffn.modules if m in keep or m.key.endswith(".gate")]
    ffn.num_local_experts = len(cold)
    ffn.routing_first = 0
    ffn.routing_last = len(cold)
    ffn.cpu_split_first = None
    ffn.load(device)
    torch.cuda.synchronize()
    got = ffn_forward(ffn, x, local_ids, weights)
    err = (got - ref).abs().max().item()
    print("COLD_OK", torch.isfinite(got).all().item(), err, flush=True)
    print("VRAM_GiB", round(torch.cuda.memory_allocated(0) / 2**30, 2), flush=True)


if __name__ == "__main__":
    main()
