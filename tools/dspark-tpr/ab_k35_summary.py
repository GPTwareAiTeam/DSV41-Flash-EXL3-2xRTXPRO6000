#!/usr/bin/env python3
"""ab_k35_summary.py -- the K A/B table from ab_k35.sh's rows.

Reads every <arm>.rows.jsonl in the directory, honours the void/nojudge markers the swap window wrote,
and prints, for the three policies (fixed5 / auto / k35):

  * per-arm and per-pass medians: decode ms/token, tok/s, tokens/round, ms/round (ms/token x
    tokens/round -- the cost of one verify round), accept rate, mean verify width;
  * the same medians split by the five content buckets, which is the only way to see that a policy
    helps prose and hurts code instead of averaging the two;
  * a paired comparison on the *same prompt* (each prompt's median over passes), because prompt-to-prompt
    variance is far larger than the policy effect;
  * token identity: the per-arm ARM_IDS sha256 from pair_arm.sh --ids, and per-prompt text equality
    against fixed5 (the local tripwire when the hashes differ).

Content buckets are by prompt index: the driver labels the held list p0..p39 in file order, so the index
is stable. It is a hand classification of the 40 prompts (code 17, prose 8, factual 7, instruction 6,
math 2) and is the classification the ticket asked for; `kind()` by keyword cannot see it (a Kotlin
pagination snippet is code, a release-notes changelog is factual prose).
"""
import json
import os
import re
import statistics as st
import sys

BUCKETS = {
    "code": {0, 5, 7, 9, 11, 12, 14, 15, 18, 19, 24, 31, 33, 35, 36, 37, 39},
    "math": {23, 26},
    "prose": {1, 3, 13, 22, 27, 29, 32, 38},
    "factual": {8, 10, 16, 17, 21, 25, 34},
    "instruction": {2, 4, 6, 20, 28, 30},
}
POL = (("fixed5", "fixed5"), ("auto", "auto"), ("k35", "k35"))


def bucket_of(label):
    if not label.startswith("p") or not label[1:].isdigit():
        return "32k" if label == "long" else "other"
    i = int(label[1:])
    for b, idx in BUCKETS.items():
        if i in idx:
            return b
    return "other"


def med(vals):
    vals = [v for v in vals if v is not None]
    return st.median(vals) if vals else None


def mean(vals):
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


def fmt(v, w=7, p=2):
    if v is None:
        return " " * (w - 2) + "--"
    return f"{v:{w}.{p}f}"


_POLRE = re.compile(r"_(fixed5|auto|k35)(?=_p[AB]$|_32k$|_pos$|$)")


def policy_of(arm):
    """The policy of one arm name, tag-independent: k35_fixed5_pA, k35_k35_32k and k35_auto_pos all
    resolve; the tag `k35` in front must not be mistaken for the policy."""
    m = _POLRE.search(arm)
    return m.group(1) if m else None


def load(outdir):
    arms, ids = {}, {}
    for fn in sorted(os.listdir(outdir)):
        if fn.endswith(".rows.jsonl"):
            arm = fn[: -len(".rows.jsonl")]
            void = os.path.exists(os.path.join(outdir, arm + ".void"))
            nojudge = os.path.exists(os.path.join(outdir, arm + ".nojudge"))
            rows = []
            with open(os.path.join(outdir, fn)) as fh:
                for line in fh:
                    line = line.strip()
                    if line.startswith("{") and '"label"' in line:
                        rows.append(json.loads(line))
            arms[arm] = {"rows": rows, "void": void, "nojudge": nojudge}
        elif fn.endswith(".ids.txt"):
            arm = fn[: -len(".ids.txt")]
            try:
                txt = open(os.path.join(outdir, fn)).read()
            except OSError:
                txt = ""
            for line in txt.splitlines():
                if "sha256=" in line:
                    ids[arm] = line.split("sha256=")[1].split()[0][:16]
                    break
    return arms, ids


def arm_stats(rows):
    d = {
        "n": len(rows),
        "ms_tok": med([r.get("decode_ms_per_tok") for r in rows]),
        "tok_s": med([1000.0 / r["decode_ms_per_tok"] for r in rows
                      if r.get("decode_ms_per_tok")]),
        "tpr": med([r.get("tokens_per_round") for r in rows]),
        "accept": mean([r.get("accept_rate") for r in rows]),
        "rounds": mean([r.get("verify_rounds") for r in rows]),
        "wall": sum(r.get("wall_s") or 0 for r in rows),
    }
    d["ms_round"] = d["ms_tok"] * d["tpr"] if d["ms_tok"] and d["tpr"] else None
    hist = {}
    for r in rows:
        for k, v in (r.get("verify_widths") or {}).items():
            hist[int(k)] = hist.get(int(k), 0) + v
    n = sum(hist.values())
    d["width"] = sum(k * v for k, v in hist.items()) / n if n else None
    d["hist"] = {k: hist[k] for k in sorted(hist)}
    return d


def per_prompt(arms, pol):
    """label -> median decode_ms_per_tok over the usable passes of one policy."""
    by = {}
    for arm, a in arms.items():
        if policy_of(arm) != pol or a["void"] or a["nojudge"]:
            continue
        if "_32k" in arm or "_pos" in arm or "warm" in arm:
            continue
        for r in a["rows"]:
            if r.get("decode_ms_per_tok"):
                by.setdefault(r["label"], []).append(r["decode_ms_per_tok"])
    return {k: st.median(v) for k, v in by.items()}


def pos_table(arms, pol):
    """Sum the per-position tables of one policy's rows, per bucket."""
    out = {}
    for arm, a in arms.items():
        if policy_of(arm) != pol or a["void"]:
            continue
        for r in a["rows"]:
            p = r.get("pos")
            if not p:
                continue
            b = bucket_of(r["label"])
            cur = out.setdefault(b, {"tested": [], "accepted": [], "rounds": 0, "calibr": {}})
            cur["rounds"] += p.get("rounds") or 0
            while len(cur["tested"]) < len(p["tested"]):
                cur["tested"].append(0)
                cur["accepted"].append(0)
            for k, v in enumerate(p["tested"]):
                cur["tested"][k] += v
                cur["accepted"][k] += p["accepted"][k]
            for c, (hit, tot) in (p.get("calibr") or {}).items():
                t = cur["calibr"].setdefault(c, [0, 0])
                t[0] += hit
                t[1] += tot
    for b, cur in out.items():
        cur["rate"] = [(a / t if t else None) for a, t in zip(cur["accepted"], cur["tested"])]
    return out


def main(outdir):
    arms, ids = load(outdir)
    if not arms:
        print(f"SUMMARY_FAIL no rows in {outdir}")
        return 1
    print(f"# rows dir {outdir}  arms={len(arms)}")
    hdr = (f"{'arm':<18}{'n':>4}{'ms/tok':>9}{'tok/s':>8}{'tpr':>7}{'acc':>7}"
           f"{'width':>7}{'ms/rnd':>8}{'wall_s':>8}  {'ids':<17}flags")
    print(hdr)
    print("-" * len(hdr))
    for arm in sorted(arms):
        a = arms[arm]
        s = arm_stats(a["rows"])
        flags = ("VOID " if a["void"] else "") + ("NOJUDGE" if a["nojudge"] else "")
        print(f"{arm:<18}{s['n']:>4}{fmt(s['ms_tok'],9)}{fmt(s['tok_s'],8)}{fmt(s['tpr'],7,3)}"
              f"{fmt(s['accept'],7,3)}{fmt(s['width'],7,2)}{fmt(s['ms_round'],8,1)}"
              f"{fmt(s['wall'],8,0)}  {ids.get(arm, '--'):<17}{flags}")

    # Content split, per policy, over both passes and all 40 prompts.
    print("\n## per content (median over prompts; tok/s = 1000/ms per token)")
    print(f"{'bucket':<12}{'n':>3} " + "".join(f"{p:>26}" for p, _ in POL))
    print(f"{'':<12}{'':>3} " + "".join(f"{'tok/s':>9}{'tpr':>8}{'ms/rnd':>9}" for _ in POL))
    pooled = {}
    for pol, _ in POL:
        rows = [r for arm, a in arms.items() if policy_of(arm) == pol and not a["void"]
                and not a["nojudge"] and "_32k" not in arm and "_pos" not in arm and "warm" not in arm
                for r in a["rows"]]
        pooled[pol] = rows
    order = ["code", "math", "prose", "factual", "instruction", "other"]
    for b in order:
        cells, labels, any_row = [], set(), False
        for pol, _ in POL:
            rs = [r for r in pooled[pol] if bucket_of(r["label"]) == b]
            labels |= {r["label"] for r in rs}
            any_row = any_row or bool(rs)
            s = arm_stats(rs)
            cells.append(f"{fmt(s['tok_s'],9)}{fmt(s['tpr'],8,3)}{fmt(s['ms_round'],9,1)}")
        if not any_row:
            continue
        print(f"{b:<12}{len(labels):>3} " + "".join(cells))
    cells = []
    for pol, _ in POL:
        s = arm_stats(pooled[pol])
        cells.append(f"{fmt(s['tok_s'],9)}{fmt(s['tpr'],8,3)}{fmt(s['ms_round'],9,1)}")
    nlab = len({r["label"] for rows in pooled.values() for r in rows})
    print(f"{'ALL':<12}{nlab:>3} " + "".join(cells))

    print("\n## per content detail (the explanation's numbers, n = prompt-rows over the passes)")
    print(f"{'bucket':<12}{'policy':<9}{'n':>3}{'acc':>7}{'tpr':>7}{'W':>6}{'ms/rnd':>8}"
          f"{'rounds/tok':>11}  widths 1..5 (%)")
    for b in order:
        for pol, _ in POL:
            rs = [r for r in pooled[pol] if bucket_of(r["label"]) == b]
            if not rs:
                continue
            s = arm_stats(rs)
            tot = sum(s["hist"].values()) or 1
            wh = " ".join(f"{100.0 * s['hist'].get(k, 0) / tot:5.1f}" for k in range(1, 6))
            print(f"{b:<12}{pol:<9}{s['n']:>3}{fmt(s['accept'],7,3)}{fmt(s['tpr'],7,3)}"
                  f"{fmt(s['width'],6,2)}{fmt(s['ms_round'],8,1)}{fmt(1.0 / s['tpr'] if s['tpr'] else None,11,3)}"
                  f"  {wh}")

    # Paired: same prompt, each policy's median over passes.
    pp = {pol: per_prompt(arms, pol) for pol, _ in POL}
    print("\n## paired on the same prompt (median over passes, decode ms/token)")
    for pol in ("auto", "k35"):
        common = sorted(set(pp["fixed5"]) & set(pp[pol]))
        if not common:
            continue
        faster = [l for l in common if pp[pol][l] < pp["fixed5"][l]]
        rel = [(pp["fixed5"][l] - pp[pol][l]) / pp["fixed5"][l] * 100 for l in common]
        print(f"{pol:>8} vs fixed5: n={len(common)} faster={len(faster)}/{len(common)} "
              f"median_delta={st.median(rel):+.2f}%  mean_delta={mean(rel):+.2f}%")
        for b in order:
            ls = [l for l in common if bucket_of(l) == b]
            if not ls:
                continue
            rs = [(pp["fixed5"][l] - pp[pol][l]) / pp["fixed5"][l] * 100 for l in ls]
            print(f"         {b:<12} n={len(ls):<3} median_delta={st.median(rs):+.2f}%")

    # Identity.
    print("\n## identity (pair_arm.sh --ids sha256, first 16 hex)")
    ref = next((ids[a] for a in sorted(ids) if policy_of(a) == "fixed5" and "_32k" not in a), None)
    for arm in sorted(ids):
        tag = "ref" if ref and ids[arm] == ref else ("SAME" if ref else "?")
        print(f"{arm:<18}{ids[arm]}  {tag}")
    rows_f5 = {(r["label"]): r.get("text") for arm, a in arms.items() if arm.startswith("fixed5_p")
               for r in a["rows"]}
    for pol in ("auto", "k35"):
        tot = same = 0
        for arm, a in arms.items():
            if policy_of(arm) != pol or a["void"] or "_32k" in arm:
                continue
            for r in a["rows"]:
                if r["label"] in rows_f5:
                    tot += 1
                    same += int(r.get("text") == rows_f5[r["label"]])
        if tot:
            print(f"per-prompt text equality {pol} vs fixed5: {same}/{tot}")

    # Per-position acceptance (POSSTATS=1 only).
    pt = {pol: pos_table(arms, pol) for pol, _ in POL}
    if any(pt[p] for p in pt):
        print("\n## per-position acceptance (POSSTATS=1: tested/accepted per draft position)")
        for pol in ("fixed5", "k35", "auto"):
            if not pt[pol]:
                continue
            print(f"[{pol}]")
            for b in order:
                cur = pt[pol].get(b)
                if not cur:
                    continue
                rates = " ".join(fmt(x, 6, 3) for x in cur["rate"][:5])
                print(f"  {b:<12} rounds={cur['rounds']:<5} p1..p5={rates}  "
                      f"tested={cur['tested'][:5]}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "."))
