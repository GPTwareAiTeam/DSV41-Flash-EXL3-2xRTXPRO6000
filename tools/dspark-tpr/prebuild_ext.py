"""Prebuild the exllamav3 torch extension for the --src tree, into its own cache, with no GPU in sight.

Why this file exists: a --src arm runs a *different source path*, so torch's extension hash differs and
`torch.utils.cpp_extension.load` writes a fresh build. If TORCH_EXTENSIONS_DIR is left at the pair's
/pair/extcache that rebuild lands on top of the RELEASE extension and puts the pair into DRIFT (this
happened at 20:59:31). So every --src arm passes
    --env TORCH_EXTENSIONS_DIR=/artifacts/dspark-tpr/extcache
and this script builds that cache once, out of band, before any arm needs it.

CPU only: CUDA_VISIBLE_DEVICES is empty and the arch list is explicit (sm 12.0 = RTX PRO 6000), so nvcc
compiles without a device being opened and no GPU is touched. Run as a file:

  docker exec -e CUDA_VISIBLE_DEVICES= -e PYTHONPATH=/artifacts/dspark-tpr/src \
    -e TORCH_EXTENSIONS_DIR=/artifacts/dspark-tpr/extcache -e TORCH_CUDA_ARCH_LIST=12.0 \
    -e MAX_JOBS=8 seam-p7-dev bash -lc 'cd /tmp && python3 -u /artifacts/dspark-tpr/prebuild_ext.py'
"""
import os
import sys
import time

want = "/artifacts/dspark-tpr/extcache"
have = os.environ.get("TORCH_EXTENSIONS_DIR", "")
src = os.environ.get("PYTHONPATH", "")

# Refuse to build into anything but the dedicated cache: a mistake here is what put dev into DRIFT.
if have.rstrip("/") != want:
    sys.exit(f"PREBUILD_FAIL TORCH_EXTENSIONS_DIR={have!r} must be {want!r}")
if "/artifacts/dspark-tpr/src" not in src:
    sys.exit(f"PREBUILD_FAIL PYTHONPATH={src!r} must point at the --src tree")
if os.environ.get("CUDA_VISIBLE_DEVICES", None) not in ("", "-1"):
    sys.exit("PREBUILD_FAIL CUDA_VISIBLE_DEVICES must be empty: this build is CPU only")

import torch

t0 = time.time()
import exllamav3.ext as ext

print("PREBUILD_OK torch=" + torch.__version__)
print("PREBUILD_OK ext_file=" + str(os.path.dirname(getattr(ext, "__file__", ""))))
for name in ("exllamav3_ext", "cuda_ext"):
    mod = getattr(ext, name, None)
    if mod is not None:
        print(f"PREBUILD_OK {name}={getattr(mod, '__file__', '?')}")
print(f"PREBUILD_OK cache={have} secs={time.time() - t0:.1f}")
print("PREBUILD_OK torch_files=" + str(sorted(os.listdir(have)) if os.path.isdir(have) else "missing"))
