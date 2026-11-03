#!/usr/bin/env python3
"""Tables for the DSpark tpr ticket from pair_arm JSONL output.

  analyze.py <file.jsonl> [more.jsonl...] [--pos] [--calib]

One JSON object per line: prompt rows and SUMMARY lines (see qmesh/e2e_prompts.py). Groups rows by
arm, and per arm prints decode cost, tokens per round, acceptance and the verify-width histogram.
--pos adds the per-position verify table (EXL3_DSPARK_POSSTATS) and the confidence calibration,
--calib adds the "what does truncating at k cost/gain" table using a round cost model c(W)=a+b*W
(a and b default to the pair's measured width fit; override with --fit a,b).
"""
import json
import sys

GIB = 2 ** 30


def load(path):
    rows = []
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        # Rows appear bare in the .jsonl and as "SUMMARY {...}" in the arm logs (and behind an
        # "  [tag] " prefix in SUMMARY.<tag>), so take anything that carries a JSON object.
        if "SUMMARY " in line:
            line = line[line.index("SUMMARY ") + len("SUMMARY "):]
        if not line.startswith("{"):
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def arms(rows):
    out = {}
    for r in rows:
        out.setdefault(r.get("arm") or ("SUMMARY:" + str(r.get("num_draft"))), []).append(r)
    return out


def fmt(x, n=2):
    return "-" if x is None else (f"{x:.{n}f}" if isinstance(x, float) else str(x))


def summarize(objs, path):
    rows = [o for o in objs if "label" in o]
    sums = [o for o in objs if "tokens_per_round_mean" in o]
    print(f"\n## {path}")
    print("\n| arm | K | ms/tok | tok/s | tokens/round | accept | width mean | widths |")
    print("|---|---|---|---|---|---|---|---|")
    for arm, group in arms(rows).items():
        g = [o for o in group if "label" in o]
        if not g:
            continue
        ms = sum(o["decode_ms_per_tok"] for o in g) / len(g)
        tpr = sum(o["tokens_per_round"] or 0 for o in g) / len(g)
        acc = [o["accept_rate"] for o in g if o.get("accept_rate") is not None]
        hist = {}
        for o in g:
            for k, v in o["verify_widths"].items():
                hist[int(k)] = hist.get(int(k), 0) + v
        n = sum(hist.values())
        wm = sum(k * v for k, v in hist.items()) / n if n else None
        K = next((o.get("num_draft") for o in sums if o.get("arm") == arm), None) or "-"
        print(f"| {arm} | {K} | {ms:.2f} | {1000 / ms:.1f} | {tpr:.3f} | "
              f"{fmt(sum(acc) / len(acc) if acc else None)} | {fmt(wm)} | "
              f"{' '.join(f'{k}:{v}' for k, v in sorted(hist.items()))} |")
    for s in sums:
        print(f"\nSUMMARY {s.get('arm')}: tpr={s.get('tokens_per_round_mean')} "
              f"accept={s.get('accept_rate_mean')} ms/tok={s.get('decode_ms_per_tok_mean')} "
              f"width_mean={s.get('verify_width_mean')} gate={json.dumps(s.get('gate'))}")


def pos_table(objs, path):
    for o in objs:
        p = o.get("pos")
        if not p or "tested" not in p:
            continue
        who = f"{o.get('arm')} {o.get('label', 'SUMMARY')}"
        tested = p["tested"]
        acc = p["accepted"]
        rate = [a / t if t else None for a, t in zip(acc, tested)]
        print(f"\n{who}: rounds={p['rounds']}  pos  tested accepted rate")
        print("  " + "  ".join(f"{i + 1}:{t}/{a}/{r:.2f}" if r is not None else f"{i + 1}:{t}/{a}/-"
                               for i, (t, a, r) in enumerate(zip(tested, acc, rate))))
        if p.get("accepted_rate") or p.get("accept_rate"):
            print("  rates " + " ".join(f"{i + 1}:{r}" for i, r in
                                        enumerate(p.get("accept_rate") or [])))


def p1_table(objs, path):
    """Position-1 acceptance per arm and prompt: the only position whose failure makes the whole
    window unreachable, so the per-prompt split says whether p1 is a drafter-wide number or a
    prompt-class one (code/prose/structured)."""
    rows = [o for o in objs if o.get("pos") and o["pos"].get("tested")]
    if not rows:
        return
    print("\nposition-1 acceptance by prompt")
    print("  arm | prompt | tested | p1")
    for o in rows:
        t, a = o["pos"]["tested"], o["pos"]["accepted"]
        if not t or not t[0]:
            continue
        print(f"  {o.get('arm')} | {o.get('label', o.get('i', '?'))} | {t[0]} | {a[0] / t[0]:.3f}")
    tot = {}
    for o in rows:
        t, a = o["pos"]["tested"], o["pos"]["accepted"]
        if t and t[0]:
            k = o.get("arm")
            tt, aa = tot.get(k, (0, 0))
            tot[k] = (tt + t[0], aa + a[0])
    for k, (t, a) in sorted(tot.items(), key=lambda kv: str(kv[0])):
        print(f"  pooled {k}: {a}/{t} = {a / t:.3f}")


def calib_table(objs, path):
    for o in objs:
        p = o.get("pos")
        if not p or not p.get("calibr"):
            continue
        print(f"\ncalibration {o.get('arm')} {o.get('label', 'SUMMARY')} "
              f"(score decile: accepted/tested)")
        tot = sum(t for _, t in p["calibr"].values()) or 1
        for b, (hit, t) in sorted(p["calibr"].items(), key=lambda kv: int(kv[0])):
            print(f"  {float(b) / 10:.1f}-{(float(b) + 1) / 10:.1f}: {hit}/{t} "
                  f"({100.0 * hit / t:.0f}%, {100.0 * t / tot:.0f}% of rows)")
        c1 = p.get("calibr_p1")
        if c1:
            print("  position-1 score (0.05 bins): accepted/tested")
            tot1 = sum(t for _, t in c1.values()) or 1
            for b, (hit, t) in sorted(c1.items(), key=lambda kv: int(kv[0])):
                print(f"    {float(b) / 20:.2f}-{(float(b) + 1) / 20:.2f}: {hit}/{t} "
                      f"({100.0 * hit / t:.0f}%, {100.0 * t / tot1:.0f}% of rows)")


def truncation(objs, path, fit):
    a, b = fit
    for o in objs:
        p = o.get("pos")
        if not p or "tested" not in p:
            continue
        tested = p["tested"]
        acc = p["accepted"]
        conf = p.get("conf_mean") or []
        print(f"\ntruncation {o.get('arm')} {o.get('label', 'SUMMARY')} "
              f"(round cost c(W)={a:.1f}+{b:.2f}*W ms; E[tokens|k] from measured prefix survival)")
        print("  k  W  P(prefix>=k)  E[tok]  c(W) ms  tok/ms")
        for k in range(max(len(tested), 1) + 1):
            w = k + 1
            surv = (tested[k - 1] / tested[0]) if k and tested[0] else 1.0
            etok = 1.0 + sum((acc[i] / tested[0]) for i in range(len(acc)))
            # tokens emitted if the window is cut at k: seed + accepted drafts among the first k
            etok_k = 1.0 + sum(acc[i] / tested[0] for i in range(min(k, len(acc))))
            c = a + b * w
            print(f"  {k}  {w}  {surv:.2f}  {etok_k:.2f}  {c:.1f}  {etok_k / c * 1000:.1f}")
        print(f"  measured full-window E[tok]={etok:.2f}")


def main(argv):
    paths = [a for a in argv if not a.startswith("--") and not a.replace(".", "").replace(",", "").isdigit()]
    pos = "--pos" in argv
    calib = "--calib" in argv
    fit = (58.0, 3.0)
    for i, a in enumerate(argv):
        if a == "--fit":
            fit = tuple(float(x) for x in argv[i + 1].split(","))
    for path in paths:
        objs = load(path)
        summarize(objs, path)
        if pos:
            pos_table(objs, path)
            p1_table(objs, path)
        if calib:
            calib_table(objs, path)
            truncation(objs, path, fit)


if __name__ == "__main__":
    main(sys.argv[1:])
