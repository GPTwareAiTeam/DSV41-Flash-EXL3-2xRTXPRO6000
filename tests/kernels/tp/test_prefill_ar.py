"""Two-GPU regression for prefill all-reduce tile boundaries and buffer guards."""
import os
import tempfile

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

pytestmark = pytest.mark.multi_gpu(2)

# Keep in sync with PAR_TILE in prefill_ar.cu.
TILE = 262144
GUARD = 512
SENTINEL = 1234567.0


def _rank(rank, root):
    os.environ['RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = '2'
    from exllamav3.tp2.p2p import P2PComm
    from qmesh.tp2.run_tp2 import TPBackendHybrid, TPBackendNCCL, TPBackendP2P
    from qmesh.tp2.prefill_ar import wrap_prefill_ar

    os.environ['PREFILL_AR'] = 'exact'
    torch.cuda.set_device(rank)
    dist.init_process_group('nccl', init_method='file://' + root + '/store', rank=rank, world_size=2)
    p2p = TPBackendP2P(P2PComm(rank, rendezvous_path=root + '/p2p', device=rank))
    os.environ['P2P_RDV'] = root + '/p2p'
    hybrid = TPBackendHybrid(TPBackendNCCL(), p2p)
    hybrid.prefill = True
    ar = wrap_prefill_ar(hybrid, p2p)
    try:
        # Literal ragged sizes exercise fallback; aligned neighbors exercise the kernel.
        sizes = (TILE - 1, TILE + 1, TILE * 3 // 2, 3 * TILE + 7,
                 TILE - 512, TILE, TILE + 512, 3 * TILE + 512)
        for n in sizes:
            torch.manual_seed(1234 + rank)
            original = torch.randn(n, device='cuda')
            expected = original.clone()
            dist.all_reduce(expected)
            storage = torch.full((n + 2 * GUARD,), SENTINEL, device='cuda')
            for _ in range(50):
                value = storage[GUARD:GUARD + n]
                value.copy_(original)
                # Two dimensions are required for prefill eligibility.
                if n % 512 == 0:
                    ar.all_reduce(value.view(256, -1))
                else:
                    # Ragged requests must delegate, rather than launch the vectorized kernel.
                    ar.all_reduce(value.view(1, n))
                assert torch.equal(value, expected), (rank, n)
                assert torch.all(storage[:GUARD] == SENTINEL).item(), (rank, n, 'before')
                assert torch.all(storage[GUARD + n:] == SENTINEL).item(), (rank, n, 'after')
        # Isolated buffers ending at each tile boundary expose writes across the boundary.
        for n in (TILE, 2 * TILE, 3 * TILE):
            storage = torch.full((n + GUARD,), SENTINEL, device='cuda')
            value = storage[:n].view(256, -1)
            for _ in range(50):
                value.fill_(rank + 1)
                ar.all_reduce(value)
                torch.cuda.synchronize()
                assert torch.all(value == 3).item(), (rank, n)
                assert torch.all(storage[n:] == SENTINEL).item(), (rank, n, 'tile guard')
        dist.barrier()
    finally:
        dist.destroy_process_group()


def test_prefill_ar_tile_guards():
    if torch.cuda.device_count() < 2:
        pytest.skip('requires two CUDA devices')
    with tempfile.TemporaryDirectory() as root:
        os.mkdir(root + '/p2p')
        mp.spawn(_rank, args=(root,), nprocs=2, join=True)
