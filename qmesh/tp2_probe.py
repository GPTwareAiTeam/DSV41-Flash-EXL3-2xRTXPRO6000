"""Worker-side probes dispatched to TP ranks via Model.tp_dispatch_all (see tp2_bench.py)."""
import torch


def peak_mem(local_context):
    dev = torch.cuda.current_device()
    return dev, torch.cuda.max_memory_allocated(dev) / 2**30


def reset_peak(local_context):
    torch.cuda.reset_peak_memory_stats(torch.cuda.current_device())
    return None
