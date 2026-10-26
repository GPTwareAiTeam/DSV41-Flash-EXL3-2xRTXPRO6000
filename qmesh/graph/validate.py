"""Validation: decode logits of the eager path vs the piecewise decode graphs (EXL3_WHOLE_GRAPH=1).

MODE=eager  : plain model.forward per decode step (reference)
MODE=graph  : model.forward per decode step with EXL3_WHOLE_GRAPH=1; each TransformerBlock replays its own
              graph A / graph B segments (exllamav3.modules.transformer._forward_piecewise). The first
              decode step of each block is an eager warmup, the second captures.
Writes /tmp/val_<MODE>.pt with per-step logits (fp32, last row) and greedy tokens.
Engram: PROBE_STUB_ENGRAM=1 replaces the host-table gather with device zeros in BOTH modes, so the
comparison isolates the graph."""
import sys, os; sys.path.insert(0, "/src/exllamav3")
import torch
from qmesh.graph.load import load, new_job

MODE = os.environ.get("MODE", "eager")
NSTEP = int(os.environ.get("NSTEP", 32))
config, model, cache, tok, gen = load(max_history=5)

if os.environ.get("PROBE_STUB_ENGRAM", "0") == "1":
    from exllamav3.modules.arch_specific.deepseek_v41 import EngramLayer
    def _zero_engram(self, x, params, out_dtype = None):
        # Device-only stub: the Engram contribution is zero, so the layer returns its input unchanged
        return x
    EngramLayer.forward = _zero_engram

rec = {"logits": []}

orig_forward = model.forward

def fwd(input_ids = None, params = None):
    out = orig_forward(input_ids = input_ids, params = params)
    if input_ids is not None and input_ids.shape[-1] == 1:
        rec["logits"].append(out[:, -1, :].float().cpu().clone())
    return out

model.forward = fwd

# PROMPT_TOKENS=N: synthetic long prompt of about N tokens (crosses the DSA top-k regime boundary during decode)
_ptoks = int(os.environ.get("PROMPT_TOKENS", 0))
_prompt = " ".join(f"item {i}: the quick brown fox jumps over the lazy dog." for i in range(_ptoks // 12)) \
    + " Summarize the list." if _ptoks else "The capital of France is"
gen.enqueue(new_job(tok, NSTEP, prompt = _prompt))
toks = []
while gen.num_remaining_jobs():
    for r in gen.iterate():
        if "error" in r:
            raise RuntimeError(f"generation job failed: {r['error']!r}")
        if "token_ids" in r:
            toks += r["token_ids"].view(-1).tolist()
torch.cuda.synchronize()

L = rec["logits"]
torch.save({"logits": L, "tokens": toks}, f"/tmp/val_{MODE}.pt")
print(f"[val] MODE={MODE} decode_logits={len(L)} tokens={len(toks)}", flush=True)
print("[val] tokens", toks[:32], flush=True)
