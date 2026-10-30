import pytest
import torch
from exllamav3.ext import exllamav3_ext as ext
from exllamav3.modules.attention_fn.dsa_triton import dsa_attn


@pytest.mark.parametrize('splits', [1, 8])
def test_fp4_attention(splits):
    torch.manual_seed(42)
    rows = torch.randn((32, 512), device='cuda', dtype=torch.float16)
    packed, scales = ext.dsv4_fp4_quant(rows, False)
    decoded = ext.dsv4_fp4_dequant(packed, scales, False)
    q = torch.randn((4, 8, 512), device='cuda', dtype=torch.float16)
    bt = torch.tensor([[0, 1]], device='cuda', dtype=torch.int32)
    rope = decoded[:, 448:].contiguous()
    kwargs = dict(pool_len=32, q_pos0=128, compress_rate=4, page_size=16, n_splits=splits)
    ref = dsa_attn(q, decoded[:, :448].contiguous(), rope, bt, **kwargs).clone()
    got = dsa_attn(q, packed, rope, bt, qc=(scales, -1), **kwargs)
    torch.testing.assert_close(got, ref, atol=0.002, rtol=0.002)


@pytest.mark.parametrize('compress_rate', [1, 2])
@pytest.mark.parametrize('rows', [1, 6, 512])
def test_fp4_gathered_direct_matches_staged(monkeypatch, compress_rate, rows):
    """Top-k (gathered) attention over an FP4 pool: in-kernel decode of the selected entries (the deep-context
    path) and staging of the whole window must both match the fp16 kernel on the dequantized pool."""
    import exllamav3.modules.attention_fn.dsa_triton as dt
    torch.manual_seed(7)
    epp = 256 // compress_rate
    entries = 4096
    pool = torch.randn((entries, 512), device='cuda', dtype=torch.float16)
    packed, scales = ext.dsv4_fp4_quant(pool, False)
    decoded = ext.dsv4_fp4_dequant(packed, scales, False)
    bt = torch.arange(entries // epp, device='cuda', dtype=torch.int32).unsqueeze(0)
    q = torch.randn((rows, 32, 512), device='cuda', dtype=torch.float16) * 0.1
    q0 = entries * compress_rate - rows
    bound = (q0 + torch.arange(rows, device='cuda') + 1) // compress_rate
    idx = torch.stack([torch.randperm(int(b), device='cuda')[:512].sort().values for b in bound.tolist()]).int()
    kw = dict(indices=idx, k_len=512, pool_len=entries, q_pos0=q0, compress_rate=compress_rate, page_size=epp)
    ref = dsa_attn(q, decoded[:, :448].contiguous(), decoded[:, 448:].contiguous(), bt, **kw).clone()
    rope0 = decoded[:1, 448:].contiguous()
    monkeypatch.setattr(dt, "_fp4_gather_direct_min", 0)
    direct = dsa_attn(q, packed, rope0, bt, qc=(scales, -1), **kw).clone()
    monkeypatch.setattr(dt, "_fp4_gather_direct_min", 1 << 40)
    staged = dsa_attn(q, packed, rope0, bt, qc=(scales, -1), **kw).clone()
    torch.testing.assert_close(direct, ref, atol=0.002, rtol=0.002)
    torch.testing.assert_close(staged, ref, atol=0.002, rtol=0.002)


@pytest.mark.parametrize('splits', [1, 8])
def test_fp4_attention_cuda_graph_matches_eager(splits):
    rows = torch.randn((32, 512), device='cuda', dtype=torch.float16)
    packed, scales = ext.dsv4_fp4_quant(rows, False)
    q = torch.randn((4, 8, 512), device='cuda', dtype=torch.float16)
    bt = torch.tensor([[0, 1]], device='cuda', dtype=torch.int32)
    rope = torch.empty((32, 64), device='cuda', dtype=torch.float16)
    kwargs = dict(pool_len=32, q_pos0=128, compress_rate=4, page_size=16, n_splits=splits,
                  qc=(scales, -1))
    dsa_attn(q, packed, rope, bt, **kwargs)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = dsa_attn(q, packed, rope, bt, **kwargs)
    for _ in range(2):
        q.normal_()
        expected = dsa_attn(q, packed, rope, bt, **kwargs).clone()
        graph.replay()
        torch.testing.assert_close(captured, expected, atol=0, rtol=0)


@pytest.mark.parametrize('count', [1, 8])
def test_fp4_indexer(count):
    from exllamav3.modules.attention_fn.dsa_triton import dsa_indexer_scores
    torch.manual_seed(123)
    keys = torch.randn((32, 128), device='cuda', dtype=torch.float16)
    p, s = ext.dsv4_fp4_quant(keys, True)
    keys = ext.dsv4_fp4_dequant(p, s, True)
    q = torch.randn((count, 8, 128), device='cuda', dtype=torch.float16)
    w = torch.randn((count, 8), device='cuda', dtype=torch.float16)
    bt = torch.tensor([[1, 0]], device='cuda', dtype=torch.int32)
    args = dict(q_pos0=128, compress_rate=4, bound_max=32, block_table=bt, epp=16)
    ref = dsa_indexer_scores(q, w, keys, **args)
    got = dsa_indexer_scores(q, w, p, idx_scales=s, **args)
    torch.testing.assert_close(got, ref, atol=0.001, rtol=0.001)
    assert torch.equal(got.topk(8).indices, ref.topk(8).indices)


@pytest.mark.parametrize('count', [1, 8])
def test_fp4_indexer_cuda_graph_matches_eager(count):
    from exllamav3.modules.attention_fn.dsa_triton import dsa_indexer_scores
    keys = torch.randn((32, 128), device='cuda', dtype=torch.float16)
    packed, scales = ext.dsv4_fp4_quant(keys, True)
    q = torch.randn((count, 8, 128), device='cuda', dtype=torch.float16)
    w = torch.randn((count, 8), device='cuda', dtype=torch.float16)
    bt = torch.tensor([[1, 0]], device='cuda', dtype=torch.int32)
    args = dict(q_pos0=128, compress_rate=4, bound_max=32, block_table=bt, epp=16,
                idx_scales=scales)
    dsa_indexer_scores(q, w, packed, **args)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = dsa_indexer_scores(q, w, packed, **args)
    for _ in range(2):
        q.normal_()
        expected = dsa_indexer_scores(q, w, packed, **args).clone()
        graph.replay()
        torch.testing.assert_close(captured, expected, atol=0, rtol=0)
