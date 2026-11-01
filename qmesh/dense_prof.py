"""DenseFuse: profile one warm decode step on GPU (P7-local, null remote). Writes per-layer kernel list.
Usage (exl3-worker): EXL3_MOE_SPLIT_FUSED=0 python3 -u qmesh/dense_prof.py [trace.json]"""
import sys, os, json, collections, time
os.environ.setdefault("EXL3_MOE_SPLIT_FUSED", "0")
os.environ.setdefault("EXL3_MOE_CPU_SWAP", "0")
sys.path.insert(0, "/src/exllamav3")
import torch
from torch.profiler import profile, ProfilerActivity
from exllamav3 import Cache, Config, Generator, GreedySampler, Job, Model, Tokenizer
from qmesh.null_remote import install

CHECKPOINT = os.environ.get("CHECKPOINT", "/models/dsv41/exl3-2.8535-p7hot-hc1")
PLACEMENT = os.environ.get("PLACEMENT", "/src/exllamav3/qmesh/placement.final.json")

def load(max_history=5):
    install()
    placement = json.load(open(PLACEMENT))
    hot = {f"layers.{layer}.ffn": v["hot"] for layer, v in placement["layers"].items()}
    config = Config.from_directory(CHECKPOINT)
    config.infer_params.moe_cpu_split = 1
    config.infer_params.moe_remote_hot = hot
    config.infer_params.remote_expert_host = {"host": "null", "port": 0}
    model = Model.from_config(config)
    cache = Cache(model, max_num_tokens=4096, max_history=max_history)
    t0 = time.time()
    model.load(device="cuda:0", progressbar=False)
    print(f"LOAD {time.time()-t0:.1f}s vram={torch.cuda.memory_allocated(0)/2**30:.2f}GiB", flush=True)
    tok = Tokenizer.from_config(config)
    gen = Generator(model, cache, tok, max_batch_size=1, max_chunk_size=2048)
    return model, tok, gen

def job(tok, n, prompt="Hi"):
    ids = tok.encode(prompt, add_bos=True, encode_special_tokens=True)
    return Job(input_ids=ids, max_new_tokens=n, min_new_tokens=n, sampler=GreedySampler())

if __name__ == "__main__":
    model, tok, gen = load()
    for _ in range(2):
        gen.enqueue(job(tok, 8))
        while gen.num_remaining_jobs(): gen.iterate()
    torch.cuda.synchronize()
    gen.enqueue(job(tok, 6)); gen.iterate(); torch.cuda.synchronize()  # prefill
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=False) as prof:
        for _ in range(4):
            gen.iterate(); torch.cuda.synchronize()
    out = sys.argv[1] if len(sys.argv) > 1 else "/tmp/densefuse_trace.json"
    prof.export_chrome_trace(out)
    print("TRACE", out)
