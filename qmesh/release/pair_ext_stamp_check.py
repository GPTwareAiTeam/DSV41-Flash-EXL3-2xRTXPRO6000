#!/usr/bin/env python3
"""pair_ext_stamp_check.py <side> <build_id> <sha> -- check a BUILD_STAMP.json arrives on stdin.

Exit 0 with `EXT_OK <side> ...` when the stamp names this build_id and this source sha; exit 1 with
EXT_STAMP_MISSING / EXT_STAMP_MISMATCH otherwise. Used by `pair_build_ext.sh verify` (before the
flip in a flip-only ticket) and by the builders themselves, so the check has one definition.
"""
import json
import sys


def main():
    if len(sys.argv) != 4:
        print(__doc__, file=sys.stderr)
        return 2
    side, bid, sha = sys.argv[1:4]
    try:
        d = json.loads(sys.stdin.read())
    except Exception:
        print(f"EXT_STAMP_MISSING {side}")
        return 1
    bad = [k for k, v in (("build_id", bid), ("source_sha", sha)) if d.get(k) != v]
    if bad:
        print(f"EXT_STAMP_MISMATCH {side} " + " ".join(f"{k}={d.get(k)}" for k in bad))
        return 1
    print(f"EXT_OK {side} sha256={d.get('sha256')} build={d.get('build_id')} (stamp)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
