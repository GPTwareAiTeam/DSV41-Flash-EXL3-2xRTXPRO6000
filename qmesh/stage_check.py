#!/usr/bin/env python3
"""Preflight for a staged tree: does the worker import, and does the MoE module carry what it needs?

A staging that cannot start should cost a torch import (~15 s), not a four-minute pack load and two
arms. Written after 2026-11-02 07:57, when the integration tree's `worker_seam_server.py` had lost
`CX_MARGIN_MIB` in a refactor: `py_compile` passed, the pack loaded, and the worker died with
`NameError: name 'CX_MARGIN_MIB' is not defined` at layer 0 during the restack. The names are read off
the module object, which is the same resolution the worker does at runtime.

    cd /src/gxport2-cpp && python3 -m qmesh.stage_check

Exit 0 when every name resolves, 3 otherwise; each line starts with `STAGE_CHECK` so a log grep is
enough.
"""

import sys

import qmesh.worker_seam_server as srv
from qmesh import exl3_moe_cuda as cxm

ok = True
for name in ("CX_MARGIN_MIB", "SWAP_WARN_MIB", "WS_KIB_PER_ROW"):
    value = getattr(srv, name, None)
    print(f"STAGE_CHECK const={name} value={value}", flush=True)
    ok = ok and value is not None

for name in ("PROTO", "CTL_PORT"):
    if hasattr(srv, name):
        print(f"STAGE_CHECK const={name} value={getattr(srv, name)}", flush=True)

print(f"STAGE_CHECK exl3_moe_cuda from_pack={hasattr(cxm.CudaExl3Moe, 'from_pack')} "
      f"trace={hasattr(cxm, 'TRACE')} file={getattr(cxm, '__file__', '?')}", flush=True)
print(f"STAGE_CHECK {'OK' if ok else 'FAIL'} python={sys.version.split()[0]}", flush=True)
sys.exit(0 if ok else 3)
