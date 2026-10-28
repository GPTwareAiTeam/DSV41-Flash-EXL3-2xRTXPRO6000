"""
ext.dsa_compact_ent (the long-context candidate pre-filter) against the eager entry-list construction it
replaces: `((~drop).any(0).nonzero() * bs + arange(bs)).flatten()[< ec]`, ascending, read up to the first
-1. Also checks the empty case and that capacity truncation keeps the ascending prefix.
"""
import pytest
import torch

from exllamav3.ext import exllamav3_ext as ext


def ref_entries(keep, bs, ec):
    ent = (keep.nonzero() * bs + torch.arange(bs, device = keep.device)).flatten()
    return ent[ent < ec].to(torch.int32)


def compact(keep, bs, ec, cap):
    out = torch.full((cap,), 12345, dtype = torch.int32, device = keep.device)
    ext.dsa_compact_ent(keep, out, bs, ec)
    return out


def check(keep, bs, ec, cap):
    out = compact(keep, bs, ec, cap)
    exp = ref_entries(keep, bs, ec)[:cap]
    neg = (out < 0).nonzero()
    n = int(neg[0]) if neg.numel() else cap
    assert out[:n].cpu().tolist() == exp.cpu().tolist(), \
        f"entry list differs: nb={keep.numel()} bs={bs} ec={ec} cap={cap} n={n} exp_n={exp.numel()}"
    assert bool((out[n:] == -1).all()), f"padding after the count must be -1 (nb={keep.numel()}, cap={cap})"


CASES = [
    # nb, bs, ec, cap, keep_fraction
    (10, 8, 37, 80, 1.0),           # every block kept, tail block clipped by ec
    (10, 8, 37, 80, 0.0),           # no block kept
    (1, 8, 5, 8, 0.0),              # nb=1, nothing kept
    (3022, 8, 24166, 16384, 0.68),  # 24k long-context shape
    (2049, 8, 16385, 16384, 1.0),   # full entry count == cap
    (12500, 8, 100000, 16384, 0.17),
    (300, 8, 2400, 16384, 0.5),
    (5000, 8, 40000, 16384, 1.0),
]


@pytest.mark.parametrize("nb,bs,ec,cap,frac", CASES, ids = [f"nb{c[0]}-bs{c[1]}-ec{c[2]}-cap{c[3]}-f{c[4]}"
                                                            for c in CASES])
@torch.inference_mode()
def test_compact_entries(device, nb, bs, ec, cap, frac):
    torch.manual_seed(0)
    keep = (torch.rand(nb, device = device) < frac) if frac not in (0.0, 1.0) \
        else torch.full((nb,), frac > 0, dtype = torch.bool, device = device)
    check(keep, bs, ec, cap)


@torch.inference_mode()
def test_compact_capacity_truncation(device):
    # cap below the entry count: the kept prefix must still be the ascending eager prefix
    keep = torch.ones(3022, dtype = torch.bool, device = device)
    check(keep, 8, 24166, 2048)


@torch.inference_mode()
def test_compact_byte_mask(device):
    # the eager path passes (~drop).any(0) as bool; uint8 must behave identically
    keep = torch.randint(0, 2, (777,), dtype = torch.uint8, device = device)
    check(keep, 8, 5000, 4096)
