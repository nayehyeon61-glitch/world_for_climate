#!/usr/bin/env python
"""Stage 15.5 report: does the memory infer a hidden SST anomaly? (rules T4-T6, README_KO section 15)

Inputs: `scripts/eval_hidden_forcing.py` outputs, hidden.json (rows) and probe.pt (filled memory states), and
optionally signal.json (`--signal-only`: the anomaly's footprint in the truth, anomaly branch minus nature run).

Per branch point, history h and free lead, the memory's gain over a reference variant is
gain = 1 - e(memory) / e(reference), with reference = frozen (the nature-filled memory of the start date kept
fixed) or markov (m = 0). The hidden-forcing gain is a difference in differences:
DiD = gain on the anomaly branch - gain on the nature run from the same date (the memory's generic gain, e.g.
correcting the column's drift, cancels). + = the memory helps more when there is a hidden forcing to infer.
The shown arm uses the same model's nature rows (on the nature run nothing is shown).

Ceiling: a model that knows nothing of the anomaly has e_a^2 ~ e_n^2 + s^2 (s = the forced signal), a model
that knows it perfectly removes s^2, so the anomaly-branch gain is at most 1 - sqrt(e_frozen^2 - s^2) / e_frozen
(RMSE metrics) or s / e_frozen (the all-variable MSE). s is the h = 0 signal at the same lead (both runs leave
the same state, so all of it is forced): for h > 0 the anomaly branch minus nature also holds the weather the
two runs have drifted into, which the fill shows the model (it starts from the anomaly branch's own state).

T4  hidden DiD (memory vs frozen) >= +5% at h = 5 d; with the anomaly shown < 2%
T5  probe R^2 >= 0.3 (ridge map memory column -> SST anomaly under it, leave one branch point out)
T6  DiD grows with h (best of 5/10/30 d >= DiD(0) + 2 pts) and saturates (30 d adds <= half of the 0 -> 10 d rise)

usage: python scripts/report_hidden_forcing.py --hidden .../hidden.json --probe .../probe.pt \
           --signal .../signal.json --out docs/results/stage15_teacher/hidden_sst.md
"""
from __future__ import annotations

import argparse
import json
import os

import _path  # noqa: F401
import numpy as np

METRICS = (("t850", "t850 near the patches (K)"), ("t2m", "t2m near the patches (K)"),
           ("all", "all variables (normalised MSE)"))
LEADS = (24, 48, 72, 120)


def k_of(lead):
    return lead // 6 - 1


def val(row, metric, k):
    if row is None or metric not in row:
        return np.nan
    v = row[metric][k]
    return np.nan if v is None else float(v)


class Rows:
    def __init__(self, rows):
        self.idx = {(r["init"], r["truth"], r["model"], bool(r["shown"]), r["variant"], int(r["h"])): r for r in rows}
        self.inits = sorted({r["init"] for r in rows})
        self.hs = sorted({int(r["h"]) for r in rows})
        self.arms = []
        for r in rows:
            arm = (r["model"], bool(r["shown"]))
            if arm not in self.arms:
                self.arms.append(arm)

    def get(self, init, truth, model, shown, variant, h):
        return self.idx.get((init, truth, model, shown if truth == "anomaly" else False, variant, h))


def boot(x, rng, n=4000):
    x = np.asarray([v for v in x if np.isfinite(v)])
    if len(x) == 0:
        return np.nan, np.nan, np.nan, 0
    if len(x) == 1:
        return float(x[0]), np.nan, np.nan, 1
    m = rng.choice(x, (n, len(x))).mean(1)
    return float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5)), len(x)


def did_points(R, model, shown, h, metric, k, ref="frozen"):
    """Per branch point: (DiD, anomaly gain, nature gain, e_ref on the anomaly branch, e_ref on the nature run)."""
    out = {}
    for init in R.inits:
        e = {(t, v): val(R.get(init, t, model, shown, v, h), metric, k)
             for t in ("anomaly", "nature") for v in ("memory", ref)}
        if not all(np.isfinite(x) and x > 0 for x in e.values()):
            continue
        ga = 1 - e["anomaly", "memory"] / e["anomaly", ref]
        gn = 1 - e["nature", "memory"] / e["nature", ref]
        out[init] = (ga - gn, ga, gn, e["anomaly", ref], e["nature", ref])
    return out


def ceiling(R, sig, model, shown, h, metric, k):
    if sig is None:
        return np.nan
    c = []
    for init in R.inits:
        e = val(R.get(init, "anomaly", model, shown, "frozen", h), metric, k)
        s = val(sig.get((init, 0)), metric, k)              # the forced part (see the module docstring)
        if not (np.isfinite(e) and np.isfinite(s) and e > 0):
            continue
        c.append(min(s / e, 1.0) if metric == "all" else 1 - np.sqrt(max(e ** 2 - s ** 2, 0.0)) / e)
    return float(np.mean(c)) if c else np.nan


def fmt(m, lo, hi, n, pct=True):
    if not np.isfinite(m):
        return "-"
    f = (lambda v: f"{100 * v:+.1f}") if pct else (lambda v: f"{v:.3f}")
    ci = f" [{f(lo)}, {f(hi)}]" if np.isfinite(lo) else ""
    return f"{f(m)}{ci} (n={n})"


def arm_name(model, shown):
    return f"{model} shown" if shown else model


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hidden", required=True)
    ap.add_argument("--probe", default=None)
    ap.add_argument("--signal", default=None)
    ap.add_argument("--lead", type=int, default=72, help="primary lead (h); the last one before any blow-up")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    with open(args.hidden) as f:
        R = Rows(json.load(f)["rows"])
    sig = None
    if args.signal and os.path.exists(args.signal):
        with open(args.signal) as f:
            sig = {(r["init"], int(r["h"])): r for r in json.load(f)["rows"]}
    kp = k_of(args.lead)
    L = [f"# Stage 15.5: hidden SST anomaly and the memory (T4-T6)\n",
         f"Branch points: {len(R.inits)} ({', '.join(i[:10] for i in R.inits)}); histories h = "
         f"{', '.join(str(h) for h in R.hs)} d; arms {', '.join(arm_name(*a) for a in R.arms)}. "
         f"Primary lead {args.lead} h. DiD = memory gain on the anomaly branch minus on the nature run "
         f"(+ = the memory helps more when there is a hidden forcing to infer); 95% bootstrap over branch points.\n"]

    # blow-ups
    L.append("## Blow-ups (runs that crossed the error cap during the 5-day free run)\n")
    L.append("| arm | variant | anomaly blown | nature blown | first blown lead (h) |")
    L.append("|---|---|---|---|---|")
    first = []
    for model, shown in R.arms:
        for v in ("memory", "frozen", "markov"):
            rs = [r for key, r in R.idx.items() if key[2] == model and key[3] == shown and key[4] == v]
            if not rs:
                continue
            ra = [r for r in rs if r["truth"] == "anomaly"]
            rn = [r for r in rs if r["truth"] == "nature"]
            fl = [6 * (1 + next(i for i, x in enumerate(r["all"]) if x is None or not np.isfinite(x)))
                  for r in rs if r["blown"]]
            first += fl
            L.append(f"| {arm_name(model, shown)} | {v} | {sum(r['blown'] for r in ra)}/{len(ra)} | "
                     f"{sum(r['blown'] for r in rn)}/{len(rn)} | {min(fl) if fl else '-'} |")
    L.append(f"\nEarliest blow-up at {min(first) if first else '-'} h, so the primary lead is {args.lead} h "
             f"(every run finite); 120 h uses the finite points only.\n")

    # T4 / T6 tables
    verdict = {}
    for metric, mname in METRICS:
        L.append(f"## DiD memory vs frozen, {mname}\n")
        L.append("| arm | h (d) | " + " | ".join(f"{ld} h" for ld in LEADS) + f" | ceiling at {args.lead} h |")
        L.append("|---|---|" + "---|" * len(LEADS) + "---|")
        for model, shown in R.arms:
            if not any(key[2] == model and key[4] == "frozen" for key in R.idx):
                continue
            for h in R.hs:
                cells = []
                for ld in LEADS:
                    d = did_points(R, model, shown, h, metric, k_of(ld))
                    m, lo, hi, n = boot([x[0] for x in d.values()], rng)
                    cells.append(fmt(m, lo, hi, n))
                    if ld == args.lead:
                        verdict[(model, shown, metric, h)] = (m, lo, hi, n)
                c = ceiling(R, sig, model, shown, h, metric, kp)
                L.append(f"| {arm_name(model, shown)} | {h} | " + " | ".join(cells) +
                         f" | {'-' if not np.isfinite(c) else f'{100 * c:+.1f}'} |")
        L.append("")

    L.append(f"## DiD memory vs markov (m = 0), at {args.lead} h\n")
    L.append("| arm | h (d) | " + " | ".join(n for _, n in METRICS) + " |")
    L.append("|---|---|" + "---|" * len(METRICS))
    for model, shown in R.arms:
        if shown or not any(key[2] == model and key[4] == "markov" for key in R.idx):
            continue
        for h in R.hs:
            cells = []
            for metric, _ in METRICS:
                d = did_points(R, model, shown, h, metric, kp, ref="markov")
                cells.append(fmt(*boot([x[0] for x in d.values()], rng)))
            L.append(f"| {arm_name(model, shown)} | {h} | " + " | ".join(cells) + " |")
    L.append("")

    # what the anomaly does to the errors, and the signal
    L.append(f"## Size of the hidden forcing at {args.lead} h\n")
    L.append("Frozen-memory error on the anomaly branch and on the nature run, the anomaly branch minus the nature "
             "run in the same units (signal at h; for h > 0 it includes the weather the runs drifted into), and "
             "the forced part (the h = 0 signal).\n")
    L.append("| arm | h (d) | metric | e frozen anomaly | e frozen nature | signal at h | forced signal |")
    L.append("|---|---|---|---|---|---|---|")
    for model, shown in R.arms:
        if shown:
            continue
        for h in R.hs:
            for metric, _ in METRICS:
                d = did_points(R, model, shown, h, metric, kp)
                ea = np.mean([x[3] for x in d.values()]) if d else np.nan
                en = np.mean([x[4] for x in d.values()]) if d else np.nan
                s = np.mean([val(sig.get((i, h)), metric, kp) for i in d]) if (sig and d) else np.nan
                s0 = np.mean([val(sig.get((i, 0)), metric, kp) for i in d]) if (sig and d) else np.nan
                L.append(f"| {model} | {h} | {metric} | {ea:.3f} | {en:.3f} | {s:.3f} | {s0:.3f} |")
    L.append("")

    # shown vs hidden
    shown_models = [m for m, s in R.arms if s]
    if shown_models:
        L.append(f"## Showing the anomaly: gain of the shown run over the hidden run (memory on, anomaly branch)\n")
        L.append("| model | h (d) | " + " | ".join(f"{n}, {ld} h" for _, n in METRICS[:1] + METRICS[2:]
                                                  for ld in (24, args.lead)) + " |")
        L.append("|---|---|" + "---|" * 4)
        for model in shown_models:
            for h in R.hs:
                cells = []
                for metric, _ in METRICS[:1] + METRICS[2:]:
                    for ld in (24, args.lead):
                        g = []
                        for init in R.inits:
                            es = val(R.get(init, "anomaly", model, True, "memory", h), metric, k_of(ld))
                            eh = val(R.get(init, "anomaly", model, False, "memory", h), metric, k_of(ld))
                            if np.isfinite(es) and np.isfinite(eh) and eh > 0:
                                g.append(1 - es / eh)
                        cells.append(fmt(*boot(g, rng)))
                L.append(f"| {model} | {h} | " + " | ".join(cells) + " |")
        L.append("")

    # T5 probe
    r2 = {}
    if args.probe and os.path.exists(args.probe):
        import torch
        from climate_wm.eval.hidden_forcing import probe_r2
        recs = torch.load(args.probe, weights_only=False)
        r2 = probe_r2(recs)
        L.append("## T5 probe: R^2 of the SST anomaly from the filled memory (held-out branch points)\n")
        L.append("| model | " + " | ".join(f"h = {h} d" for h in R.hs) + " |")
        L.append("|---|" + "---|" * len(R.hs))
        for model in sorted({m for m, _ in r2}):
            L.append(f"| {model} | " + " | ".join(f"{r2.get((model, h), np.nan):+.3f}" for h in R.hs) + " |")
        L.append("\nh = 0 is the null (the memory has not seen the anomaly yet).\n")

    # verdicts
    L.append("## Verdicts\n")
    for model, shown in R.arms:
        if shown:
            continue
        for metric in ("t850", "all"):
            m5 = verdict.get((model, False, metric, 5), (np.nan,) * 4)
            ms = verdict.get((model, True, metric, 5), (np.nan,) * 4)
            ok = np.isfinite(m5[0]) and m5[0] >= 0.05 and (not np.isfinite(ms[0]) or ms[0] < 0.02)
            L.append(f"- T4 {model}, {metric}: hidden DiD at h = 5 d {fmt(*m5)}"
                     + (f", shown {fmt(*ms)}" if np.isfinite(ms[0]) else "")
                     + f" -> {'supported' if ok else 'not supported'}")
            curve = [verdict.get((model, False, metric, h), (np.nan,))[0] for h in R.hs]
            if 0 in R.hs and len(R.hs) > 2:
                d0 = curve[R.hs.index(0)]
                best = np.nanmax([c for h, c in zip(R.hs, curve) if h >= 5]) if any(h >= 5 for h in R.hs) else np.nan
                grows = np.isfinite(best) and best >= d0 + 0.02
                sat = None
                if 10 in R.hs and 30 in R.hs:
                    d10, d30 = curve[R.hs.index(10)], curve[R.hs.index(30)]
                    sat = np.isfinite(d30) and (d30 - d10) <= 0.5 * max(d10 - d0, 0.0)
                L.append(f"- T6 {model}, {metric}: DiD by h " + ", ".join(
                    f"{h} d {100 * c:+.1f}" for h, c in zip(R.hs, curve)) +
                    f" -> grows {'yes' if grows else 'no'}" + ("" if sat is None else f", saturates {'yes' if sat else 'no'}"))
        if r2:
            best = max((r2.get((model, h), -np.inf), h) for h in R.hs if h > 0)
            L.append(f"- T5 {model}: best probe R^2 {best[0]:+.3f} (h = {best[1]} d) -> "
                     f"{'supported' if best[0] >= 0.3 else 'not supported'}")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
