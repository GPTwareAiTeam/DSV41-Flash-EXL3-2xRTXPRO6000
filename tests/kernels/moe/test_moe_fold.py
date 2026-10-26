"""
Shared-expert fold check: one DeepSeek-V4.1-Flash-1017 EXL3 mixed-2.565 MoE layer (layer 3) built twice through
BlockSparseMLP with identical arguments, once with the shared expert folded into the routed fused launch and once
with EXL3_MOE_FOLD=0 (separate shared launch). Routing is pinned via external_selected / external_weights, so both
layers consume identical inputs. The folded output must match the separate-launch output at bsz 1/2/6/8.
Checkpoint is read-only. Run on a single GPU within the TP2 container only.
"""
import os

import pytest
import torch

from exllamav3 import Config
from exllamav3.modules import GatedMLP, BlockSparseMLP

CKPT = "/models/dsv41/exl3-out/DeepSeek-V4.1-Flash-1017-EXL3-mixed-2.565-tp2"
LAYER = 3


def build_layer(cfg, device, fold: bool):
    prev = os.environ.get("EXL3_MOE_FOLD")
    os.environ["EXL3_MOE_FOLD"] = "1" if fold else "0"
    try:
        mlp = BlockSparseMLP(
            config = cfg,
            key = f"layers.{LAYER}.ffn",
            hidden_size = cfg.hidden_size,
            intermediate_size = cfg.moe_intermediate_size,
            num_experts = cfg.num_experts,
            num_experts_per_tok = cfg.num_experts_per_tok,
            key_up = "experts.{expert_idx}.w3",
            key_gate = "experts.{expert_idx}.w1",
            key_down = "experts.{expert_idx}.w2",
            key_routing_gate = "gate",
            key_e_score_bias = "gate.bias",
            router_type = "sqrtsp",
            routed_scaling_factor = cfg.routed_scaling_factor,
            shared_experts = GatedMLP(
                config = cfg,
                key = f"layers.{LAYER}.ffn.shared_experts",
                hidden_size = cfg.hidden_size,
                intermediate_size = cfg.moe_intermediate_size * cfg.num_shared_experts,
                key_up = "w3",
                key_gate = "w1",
                key_down = "w2",
                qmap = "block.mlp",
                out_dtype = torch.float,
                activation_fn = "silu",
                act_limit = cfg.swiglu_limit,
                select_hq_bits = 2,
            ),
            out_dtype = torch.float,
            activation_fn = "silu",
            act_limit = cfg.swiglu_limit,
            interm_dtype = torch.half,
        )
        mlp.load(device)
    finally:
        if prev is None:
            os.environ.pop("EXL3_MOE_FOLD", None)
        else:
            os.environ["EXL3_MOE_FOLD"] = prev
    return mlp


@pytest.fixture(scope = "module")
def layers(device):
    if not os.path.isdir(CKPT):
        pytest.skip("checkpoint not mounted")
    cfg = Config.from_directory(CKPT)
    folded = build_layer(cfg, device, fold = True)
    separate = build_layer(cfg, device, fold = False)
    assert folded.fold_recorded, "production DSv4 layer should be fold-eligible"
    assert not separate.fold_recorded, "EXL3_MOE_FOLD=0 must disable the fold"
    return folded, separate


@pytest.mark.parametrize("bsz", [1, 2, 6, 8])
def test_fold_matches_separate_shared_launch(device, layers, bsz):
    folded, separate = layers
    gen = torch.Generator(device = "cpu").manual_seed(100 + bsz)
    E, k = folded.num_experts, folded.num_experts_per_tok
    x = (torch.randn((bsz, folded.hidden_size), generator = gen) * 0.05).half().to(device)
    sel = torch.stack([torch.randperm(E, generator = gen)[:k] for _ in range(bsz)]).to(device)
    w = (torch.rand((bsz, k), generator = gen) * 0.5 + 0.1).half().to(device)

    params = {"external_selected": sel, "external_weights": w}
    xin = x.view(1, bsz, folded.hidden_size)
    got = folded.forward(xin, params).view(bsz, -1).float()
    ref = separate.forward(xin, params).view(bsz, -1).float()
    scale = ref.abs().max().item()
    err = (got - ref).abs().max().item()
    print(f"bsz {bsz}: fold vs separate err {err:.3e} scale {scale:.3e} rel {err / max(scale, 1e-12):.3e}")
    assert err <= 3e-3 * scale, f"bsz {bsz}: err {err:.3e} vs scale {scale:.3e}"
