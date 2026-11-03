#!/usr/bin/env python3
"""ids_compare.py <ctl.log> <ids.log> -- is the position-1 capture free, and why does position 1 miss?

Two verdicts, both from the two arms' row JSON:
  1. token identity: every row's `text` must be byte-identical between control and capture, and the
     capture arm must carry the same tokens_per_round. The capture only reads ids that the accept loop
     had already compared, so any text difference means it is not free and must not ship.
  2. position-1 failure classes: each [draft_text, target_text, match] pair is classified as
     match / whitespace-boundary (same stripped text) / digit / punctuation / other.
"""
import json
import sys


def rows(path):
    out = []
    for line in open(path, errors="replace"):
        line = line.strip()
        for pre in ("[k3] ", "[k4] ", "[k5] ", "[auto] ", "[thr] ", "[ctl] ", "[ids] "):
            if line.startswith(pre):
                line = line[len(pre):]
        if not line.startswith("{"):
            continue
        try:
            o = json.loads(line)
        except Exception:
            continue
        if "i" in o and "text" in o:
            out.append(o)
    return out


def cls(d, t, match):
    if match:
        return "match"
    if d.strip() == t.strip() and d.strip():
        return "whitespace"
    if d.strip().isdigit() and t.strip().isdigit():
        return "digit"
    if d.strip() and t.strip() and d.strip() in ("(", ")", "[", "]", "{", "}", ",", ".", ";", ":", "-", "=", "+", "*", "/", "|", ">", "<", '"', "'", "`"):
        return "punct"
    if d.strip() and t.strip() and d.strip().isdigit() != t.strip().isdigit():
        return "digit-vs-text"
    return "other"


def main(a, b):
    ra, rb = rows(a), rows(b)
    print(f"control rows={len(ra)} capture rows={len(rb)}")
    same = ident = 0
    for x, y in zip(ra, rb):
        same += 1
        if x.get("text") == y.get("text") and x.get("new_tokens") == y.get("new_tokens"):
            ident += 1
        else:
            print(f"  MISMATCH prompt {x.get('label')}: new_tokens {x.get('new_tokens')} vs {y.get('new_tokens')}")
    print(f"token identity: {ident}/{same} rows byte-identical"
          f" -> {'FREE (capture does not change the decode)' if ident == same else 'NOT FREE: do not ship'}")

    def p1(rs):
        t = sum((r.get("pos") or {}).get("tested", [0])[0] for r in rs if r.get("pos"))
        a_ = sum((r.get("pos") or {}).get("accepted", [0])[0] for r in rs if r.get("pos"))
        return a_, t

    for name, rs in (("control", ra), ("capture", rb)):
        a_, t = p1(rs)
        if t:
            print(f"  p1 {name}: {a_}/{t} = {a_ / t:.3f}")

    pairs = [p for r in rb for p in (r.get("ids") or [])]
    if not pairs:
        print("no id pairs in the capture arm (EXL3_DSPARK_IDSTATS=1 missing, or pos_summary off)")
        return
    # pairs are [draft_text, alt_text, target_text, match, target_is_alt]
    counts = {}
    for p in pairs:
        counts[cls(p[0], p[2], p[3])] = counts.get(cls(p[0], p[2], p[3]), 0) + 1
    tot = len(pairs) or 1
    print(f"\nposition-1 classes over {len(pairs)} rounds")
    for k, v in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {k}: {v} ({100.0 * v / tot:.1f}%)")
    misses = [p for p in pairs if not p[3]]
    alt = [p for p in misses if p[4]]
    if misses:
        print(f"\nrunner-up mass (position 1): target was the drafter's SECOND candidate in "
              f"{len(alt)}/{len(misses)} misses = {len(alt) / len(misses):.3f}")
        print("  that is the number a 2-candidate tree at position 1 is worth: it converts a miss into a "
              "hit only for this fraction, and it costs one extra verify row per round")
        print(f"  upper bound on tpr gain if every converted miss also unlocked the rest of the window: "
              f"+{len(alt) / len(misses):.3f} x (tokens per converted round)")
    else:
        print("\nno position-1 misses: nothing to convert, tree premise untested")
    print("\nsample misses (first 12)")
    shown = 0
    for p in misses:
        if shown >= 12:
            continue
        shown += 1
        print(f"  {cls(p[0], p[2], False):16s} draft={p[0]!r} alt={p[1]!r} target={p[2]!r} "
              f"{'TARGET-WAS-ALT' if p[4] else ''}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
