#!/usr/bin/env python3
"""Turn a SEAM_ID_TRACE capture into the .npz that moe_cpp_bench.py --layers wants.

The client prints, one request late and only when that request's event has signalled,

    SEAM_ID_PY layer=<L> rows=<R> pre=[...] caller=[...] staged=[...]

(qmesh/remote_seam_host.py:_trace_print). `staged` is the ids buffer as the wire saw it, i.e. the
routing that request actually carried into the layer's cold experts, already in the layer's own cold
index space and TOPK wide - exactly what the bench's slab takes, so nothing is remapped here. A run
can also end on the traced request; atexit dumps what is left (_trace_final), which is why the last
request of a bucket is still in the log.

Writes one key per (layer, rows): L<layer>_r<rows>, int32 (rows, TOPK). For a row count that the
capture never produced - a harness that only issues 6-row verify steps has no 1-row request - the
first row of the 6-row capture is used instead and reported, because one token's routing *is* a real
1-row routing; the layer's traffic is then that token's experts, not a hypothetical one's.
"""
from __future__ import annotations

import argparse
import ast
import re
import sys

import numpy as np

TOPK = 6
LINE = re.compile(r"^SEAM_ID_PY layer=(\d+) rows=(\d+) pre=(\[.*?\]) caller=(\[.*?\]) "
                  r"staged=(\[.*?\])\s*$")


def parse(path):
    """-> {(layer, rows): ids} keeping the last capture per (layer, rows)."""
    seen, bad = {}, 0
    with open(path, "r", errors="replace") as fh:
        for ln in fh:
            m = LINE.match(ln.strip())
            if not m:
                continue
            layer, rows = int(m.group(1)), int(m.group(2))
            try:
                staged = ast.literal_eval(m.group(5))
            except (ValueError, SyntaxError):
                bad += 1
                continue
            arr = np.asarray(staged, dtype=np.int32).reshape(-1)
            if rows <= 0 or arr.size != rows * TOPK:
                bad += 1
                continue
            seen[(layer, rows)] = arr.reshape(rows, TOPK)
    return seen, bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log", help="client stderr capture, e.g. idcap.out.cpp")
    ap.add_argument("-o", "--out", default="/home/treadall/gxport2-p7/ids_real.npz")
    ap.add_argument("--rows", default="1,6", help="row counts the bench will ask for")
    ap.add_argument("--fallback-rows", default="6",
                    help="captured row count to take a single row from when a count is missing")
    a = ap.parse_args()

    seen, bad = parse(a.log)
    if not seen:
        print(f"IDS_FROM_TRACE empty no SEAM_ID_PY staged lines in {a.log} (unparsed={bad})")
        return 2
    have_rows = sorted({r for (_, r) in seen})
    layers = sorted({l for (l, _) in seen})
    print(f"IDS_FROM_TRACE parsed={len(seen)} unparsed={bad} layers={len(layers)} "
          f"layer_span={layers[0]}..{layers[-1]} row_counts={have_rows}")

    out, notes = {}, []
    for rows in (int(x) for x in a.rows.split(",") if x.strip()):
        for layer in layers:
            ids = seen.get((layer, rows))
            if ids is None:
                src = seen.get((layer, int(a.fallback_rows)))
                if src is None or src.shape[0] < 1:
                    continue
                ids = src[0:1]
                notes.append((layer, rows))
            out[f"L{layer}_r{rows}"] = ids.astype(np.int32)

    missing = [l for l in layers if f"L{l}_r{int(a.rows.split(',')[0])}" not in out]
    np.savez(a.out, **out)
    print(f"IDS_FROM_TRACE wrote {a.out} keys={len(out)} from_row0={len(notes)} "
          f"layers_without={len(missing)}")
    if missing:
        print(f"IDS_FROM_TRACE missing_layers={missing}")
    for rows in (int(x) for x in a.rows.split(",") if x.strip()):
        keys = [k for k in out if k.endswith(f"_r{rows}")]
        if not keys:
            continue
        distinct = [len(np.unique(out[k])) for k in keys]
        top = max(int(out[k].max()) for k in keys)
        lo = min(int(out[k].min()) for k in keys)
        print(f"IDS_SUM rows={rows} layers={len(keys)} distinct_experts_mean={np.mean(distinct):.1f} "
              f"distinct_min={min(distinct)} distinct_max={max(distinct)} id_max={top} id_min={lo}")
        n_from0 = sum(1 for (_, r) in notes if r == rows)
        if n_from0:
            print(f"IDS_NOTE rows={rows} from_row0_of_{a.fallback_rows}={n_from0} layers")
    return 0


if __name__ == "__main__":
    sys.exit(main())
