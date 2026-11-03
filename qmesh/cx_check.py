#!/usr/bin/env python3
"""Does the vendored cuda-exl3 extension in this tree export the ops the GX MoE path calls?

`import cuda_exl3.ops` registers torch.ops.cuda_exl3_C from vendor/cuda-exl3/src/cuda_exl3/_C*.so.
That .so is gitignored, so a tree copied from a worktree has none and the op namespace stays empty;
qmesh/exl3_moe_cuda.py then raises AttributeError on its first `_C.exl3_set_moe_acc_cap(0)` and the
worker prints SEAM_MOE_NOEXT and serves every layer from the mixed-K module - which looks like a
working server with a slower path, not like a missing build.

The five names checked here are the ones qmesh/exl3_moe_cuda.py calls; anything absent is a .so that
predates the tree it is being served from. Prints one line, exits 1 when anything is missing:

    CX_ATTRS ok=1 missing=none file=/src/.../_C.cpython-312-aarch64-linux-gnu.so
"""

import os
import sys

SRC = os.environ.get("CUDA_EXL3_SRC", "/src/exllamav3/vendor/cuda-exl3/src")
sys.path.insert(0, SRC)

WANT = ("exl3_moe_gemm", "exl3_moe_combine", "exl3_moe_had_in", "exl3_moe_glu_had_in",
        "exl3_set_moe_acc_cap")


def main() -> int:
    try:
        import cuda_exl3.ops  # noqa: F401  (loads the extension)
        import torch
    except Exception as e:  # noqa: BLE001 - a tree without the package is the same answer
        print(f"CX_ATTRS ok=0 missing=import err={type(e).__name__}: {str(e).splitlines()[0][:160]}",
              flush=True)
        return 1
    c = torch.ops.cuda_exl3_C
    missing = [n for n in WANT if not hasattr(c, n)]
    f = ""
    for mod in ("cuda_exl3._C", "cuda_exl3_C"):
        m = sys.modules.get(mod)
        if getattr(m, "__file__", None):
            f = f" file={m.__file__}"
            break
    print(f"CX_ATTRS ok={int(not missing)} missing={','.join(missing) or 'none'}{f}", flush=True)
    return 0 if not missing else 1


if __name__ == "__main__":
    sys.exit(main())
