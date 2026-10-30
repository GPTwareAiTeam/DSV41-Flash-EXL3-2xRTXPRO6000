import pytest
import torch
from exllamav3.ext import exllamav3_ext as ext


def reference(x, idx):
    group = 32 if idx else 16
    z = x.float().reshape(x.shape[0], -1, group)
    a = z.abs().amax(-1)
    if idx:
        v = a.clamp_min(6 * 2**-126) * (1 / 6)
        bits = v.contiguous().view(torch.int32)
        exponent = ((bits >> 23) & 255) + ((bits & 0x7fffff) != 0).int()
        scales = exponent.to(torch.uint8)
        s = (exponent << 23).contiguous().view(torch.float32)
    else:
        # CUDA E4M3 conversion uses satfinite, unlike torch's overflow-to-NaN cast.
        s8 = (a.clamp_min(6 * 2**-9) / 6).clamp_max(448).to(torch.float8_e4m3fn)
        scales = s8.view(torch.uint8)
        s = s8.float()
    y = z / s.unsqueeze(-1)
    mag = y.abs()
    code = torch.zeros_like(mag, dtype=torch.uint8)
    for i, midpoint in enumerate((.25, .75, 1.25, 1.75, 2.5, 3.5, 5.)):
        code += ((mag > midpoint) | ((mag == midpoint) & bool(i & 1))).to(torch.uint8)
    code |= torch.signbit(y).to(torch.uint8) << 3
    code = code.reshape(x.shape)
    packed = code[:, ::2] | (code[:, 1::2] << 4)
    vals = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6], device=x.device)
    value = vals[(code & 7).long()] * torch.where((code & 8) != 0, -1., 1.)
    decoded = (value.reshape_as(z) * s.unsqueeze(-1)).reshape_as(x).half()
    return packed, scales, decoded


@pytest.mark.parametrize('idx', [False, True])
def test_exact(idx):
    width = 128 if idx else 512
    torch.manual_seed(12)
    x = (torch.randn(64, width, device='cuda') * 8).half()
    x[0] = 0
    x[1] = torch.finfo(torch.float16).tiny / 1024
    x[2] = 60000
    x[3, ::2] *= -1
    # Groups with a scale of one exercise every midpoint and signed zero.
    x[4] = torch.tensor([6, .25, .75, 1.25, 1.75, 2.5, 3.5, 5, -0., -.25, -.75, -1.25, -1.75, -2.5, -3.5, -5], device='cuda').repeat(width // 16)
    p, s = ext.dsv4_fp4_quant(x, idx)
    rp, rs, rd = reference(x, idx)
    assert torch.equal(p, rp), [(int(i), int(j), int(p[i,j]), int(rp[i,j]), x[i,2*j:2*j+2].tolist(), int(s[i,j//8]), int(rs[i,j//8])) for i,j in (p != rp).nonzero()[:10].tolist()]
    assert torch.equal(s, rs)
    assert torch.equal(ext.dsv4_fp4_dequant(p, s, idx), rd)


@pytest.mark.parametrize('idx', [False, True])
def test_empty(idx):
    width = 128 if idx else 512
    x = torch.empty((0, width), device='cuda', dtype=torch.float16)
    p, s = ext.dsv4_fp4_quant(x, idx)
    assert ext.dsv4_fp4_dequant(p, s, idx).shape == x.shape


@pytest.mark.parametrize('idx', [False, True])
def test_invalid(idx):
    width = 128 if idx else 512
    x = torch.zeros((2, width), device='cuda', dtype=torch.float16)
    with pytest.raises(RuntimeError): ext.dsv4_fp4_quant(x.float(), idx)
    with pytest.raises(RuntimeError): ext.dsv4_fp4_quant(x[:, :-1], idx)
    p, s = ext.dsv4_fp4_quant(x, idx)
    with pytest.raises(RuntimeError): ext.dsv4_fp4_dequant(p, s.float(), idx)
    with pytest.raises(RuntimeError): ext.dsv4_fp4_dequant(p, s[:, :-1], idx)
