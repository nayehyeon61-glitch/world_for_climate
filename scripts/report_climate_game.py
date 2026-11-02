#!/usr/bin/env python
"""Stage 16 board 1 report: the N1 climate-bias game on 2020 (README_KO section 16).

    python scripts/report_climate_game.py --runs runs/stage16 --out docs/results/stage16_rain_game/board1.md \
        [--compare N1=runs/physics_n1/compare_graphcast_2020.json td3=runs/stage16/compare_td3_2020.json]

Bias = RMS over latitude and variables of the 30-day mean zonal-mean difference to ERA5 (normalised units for
"all", physical units per variable). Gains are paired over the initialisations where both runs stayed finite,
with a bootstrap 90% interval; + = the policy is better. Rules R1-R5 as registered in the design doc.
"""
from __future__ import annotations

import argparse
import json
import os

import _path  # noqa: F401
import numpy as np

from climate_wm.models.knobs import KNOBS

POLICIES = [("K0", "default knobs"), ("K0_E1", "default knobs + stage 13 E1 energy bound"), ("Krand", "random knobs"), ("Kconst", "best constant (CEM)"),
            ("td3", "TD3"), ("ppo", "PPO"), ("Kgrad", "gradient through N1"),
            ("td3_nostate", "TD3 without the state (Knostate)"), ("td3_sparse5", "TD3, sparse reward lag 5")]
GROUPS = ("heating", "moistening", "momentum", "diffusion")
UNITS = {"temperature": ("T (K)", 1.0), "u_component_of_wind": ("u (m/s)", 1.0),
         "specific_humidity": ("q (g/kg)", 1000.0), "2m_temperature": ("t2m (K)", 1.0),
         "mean_sea_level_pressure": ("msl (hPa)", 0.01)}
RL = ("td3", "ppo")


def load(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def paired_gain(a, b, key="all", n_boot=2000, seed=0):
    """Gain of run a over run b on the bias [%]: 1 - mean(a) / mean(b) on the inits finite in both."""
    if a is None or b is None:
        return None
    ra = {r["init"]: r for r in a["rows"] if r["blown_day"] < 0}
    rb = {r["init"]: r for r in b["rows"] if r["blown_day"] < 0}
    common = sorted(set(ra) & set(rb))
    if not common:
        return None
    x = np.array([ra[i]["bias"][key] for i in common])
    y = np.array([rb[i]["bias"][key] for i in common])
    g = 100 * (1 - x.mean() / y.mean())
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(n_boot):
        j = rng.integers(0, len(common), len(common))
        boot.append(100 * (1 - x[j].mean() / y[j].mean()))
    lo, hi = np.percentile(boot, [5, 95])
    return {"gain": float(g), "lo": float(lo), "hi": float(hi), "n": len(common)}


def paired_daily(a, b, n_boot=2000, seed=0):
    """Gain of a over b on the daily 5-day smoothed zonal distance, over the days both runs were alive [%]."""
    if a is None or b is None:
        return None
    ra, rb = {r["init"]: r["err"] for r in a["rows"]}, {r["init"]: r["err"] for r in b["rows"]}
    pairs = []
    for i in sorted(set(ra) & set(rb)):
        d = [(x, y) for x, y in zip(ra[i], rb[i]) if x is not None and y is not None]
        if d:
            pairs.append((np.mean([x for x, _ in d]), np.mean([y for _, y in d])))
    if not pairs:
        return None
    p = np.array(pairs)
    g = 100 * (1 - p[:, 0].mean() / p[:, 1].mean())
    rng = np.random.default_rng(seed)
    boot = [100 * (1 - p[j, 0].mean() / p[j, 1].mean()) for j in (rng.integers(0, len(p), len(p)) for _ in range(n_boot))]
    lo, hi = np.percentile(boot, [5, 95])
    return {"gain": float(g), "lo": float(lo), "hi": float(hi), "n": len(p)}


def alive_days(e):
    days = e["cfg"]["days"]
    return np.array([r["blown_day"] if r["blown_day"] >= 0 else days for r in e["rows"]], float)


def fmt(g):
    return "-" if g is None else f"{g['gain']:+.1f} [{g['lo']:+.1f}, {g['hi']:+.1f}] (n={g['n']})"


def blown(e):
    return sum(r["blown_day"] >= 0 for r in e["rows"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--compare", nargs="*", default=[], help="NAME=compare_graphcast json (R4); N1 first")
    args = ap.parse_args()
    ev = {name: load(os.path.join(args.runs, "eval", f"{name}.json")) for name, _ in POLICIES}
    ko = {g: load(os.path.join(args.runs, "eval", f"td3_no_{g}.json")) for g in GROUPS}
    k0 = ev["K0"]
    if k0 is None:
        raise SystemExit("K0 evaluation missing")
    L = ["# Stage 16 board 1: the N1 climate-bias game (2020 test)", ""]
    L += [f"2020 test: {k0['summary']['inits']} initialisations, {k0['cfg']['days']}-day free runs, knobs decided "
          "every 24 h. Bias = RMS of the episode-mean zonal-mean difference to ERA5. Gain = 1 - bias / bias of the "
          "reference, paired over the initialisations finite in both runs, bootstrap 90% interval; + = better.", ""]

    L += ["## Survival and the daily distance while alive", "",
          "Daily distance = the 5-day smoothed zonal-mean distance to ERA5 (the game's daily cost), averaged over "
          "the days both runs were alive.", "",
          "| policy | blown | alive days mean / median | daily distance gain vs K0 (%) | vs Kconst (%) |",
          "|---|---|---|---|---|"]
    for name, desc in POLICIES:
        e = ev[name]
        if e is None:
            continue
        a = alive_days(e)
        L.append(f"| {name} ({desc}) | {blown(e)}/{len(e['rows'])} | {a.mean():.1f} / {np.median(a):.0f} | "
                 f"{fmt(paired_daily(e, k0)) if name != 'K0' else '0'} | "
                 f"{fmt(paired_daily(e, ev['Kconst'])) if name != 'Kconst' else '0'} |")
    L.append("")
    L += ["## Bias of the runs that stayed finite, and blow-ups", "",
          "| policy | blown | return | bias (normalised) | gain vs K0 (%) | gain vs Kconst (%) | "
          + " | ".join(UNITS[v][0] for v in UNITS) + " |",
          "|---|---|---|---|---|---|" + "---|" * len(UNITS)]
    for name, desc in POLICIES:
        e = ev[name]
        if e is None:
            L.append(f"| {name} ({desc}) | pending |" + " |" * (5 + len(UNITS)))
            continue
        s = e["summary"]
        phys = " | ".join(f"{s['bias'].get(v, float('nan')) * UNITS[v][1]:.3g}" for v in UNITS)
        L.append(f"| {name} ({desc}) | {s['blown']}/{s['inits']} | {s['return']:.2f} | {s['bias']['all']:.4f} | "
                 f"{fmt(paired_gain(e, k0)) if name != 'K0' else '0'} | "
                 f"{fmt(paired_gain(e, ev['Kconst'])) if name != 'Kconst' else '0'} | {phys} |")
    L.append("")

    L += ["## Gain vs K0 per variable (%)", "", "| policy | " + " | ".join(UNITS[v][0] for v in UNITS) + " |",
          "|---|" + "---|" * len(UNITS)]
    for name, _ in POLICIES[1:]:
        if ev[name] is not None:
            L.append(f"| {name} | " + " | ".join(fmt(paired_gain(ev[name], k0, v)) for v in UNITS) + " |")
    L.append("")

    full = paired_gain(ev["td3"], k0)
    if any(ko.values()) and full is not None:
        L += ["## Knob-group knock-outs (TD3 with one group held at the default)", "",
              "| held group | blown | gain vs K0 (%) | share of the TD3 gain lost |", "|---|---|---|---|",
              f"| none | {blown(ev['td3'])} | {fmt(full)} | 0 |"]
        for g in GROUPS:
            if ko[g] is None:
                L.append(f"| {g} | pending | | |")
                continue
            gg = paired_gain(ko[g], k0)
            lost = "-" if gg is None or abs(full["gain"]) < 1e-9 else f"{1 - gg['gain'] / full['gain']:+.2f}"
            L.append(f"| {g} | {blown(ko[g])} | {fmt(gg)} | {lost} |")
        L.append("")

    L += ["## Knob settings (mean over initialisations and days; multiplier 1 + 0.5 a, diffusion time x 2^a)", "",
          "| policy | " + " | ".join(KNOBS) + " | spread across inits | spread across days |",
          "|---|" + "---|" * (len(KNOBS) + 2)]
    for name, _ in POLICIES[1:]:
        e = ev[name]
        if e is None:
            continue
        am = np.array([r["act_mean"] for r in e["rows"]])
        sd = np.array([r["act_std"] for r in e["rows"]])
        L.append(f"| {name} | " + " | ".join(f"{x:+.2f}" for x in am.mean(0)) +
                 f" | {am.std(0).mean():.3f} | {sd.mean():.3f} |")
    L += ["", "Spread across inits = state (season) dependence of the daily-mean setting; spread across days = "
          "within-episode change. A constant policy has both 0.", ""]

    L += ["## Training", "", "| agent | iterations | first 5: return / blown | last 5: return / blown |",
          "|---|---|---|---|"]
    for name in ("td3", "ppo", "cem", "grad", "td3_nostate", "td3_sparse5"):
        h = load(os.path.join(args.runs, name, "train_history.json"))
        if not h:
            continue
        if "return" in h[0]:
            f5 = f"{np.mean([r['return'] for r in h[:5]]):.2f} / {sum(r['blown'] for r in h[:5])}"
            l5 = f"{np.mean([r['return'] for r in h[-5:]]):.2f} / {sum(r['blown'] for r in h[-5:])}"
        else:
            f5 = f"loss {np.nanmean([r['loss'] for r in h[:5]]):.3f}"
            l5 = f"loss {np.nanmean([r['loss'] for r in h[-5:]]):.3f}"
        L.append(f"| {name} | {len(h)} | {f5} | {l5} |")
    L.append("")

    if args.compare:
        L += ["## R4: WB2 2020 skill with the policy on (lat-weighted RMSE)", ""]
        cmp = {}
        for item in args.compare:
            name, path = item.split("=", 1)
            c = load(path)
            if c is not None:
                cmp[name] = c
        L.append(json.dumps({k: list(v)[:3] for k, v in cmp.items()}))   # filled by hand into README if needed
        L.append("")

    L += ["## Rules", ""]
    best = max((n for n in RL if ev[n] is not None), key=lambda n: paired_daily(ev[n], k0)["gain"], default=None)
    if best:
        g0, gc, gg = paired_daily(ev[best], k0), paired_daily(ev[best], ev["Kconst"]), paired_daily(ev["Kgrad"], k0)
        L.append(f"- best RL policy: {best}")
        L.append(f"- R1 daily distance gain vs K0 >= +20% (days both alive): {fmt(g0)} -> {'supported' if g0['gain'] >= 20 else 'not supported'}")
        if gc:
            L.append(f"- R2 gain vs Kconst >= +5% (the state matters): {fmt(gc)} -> "
                     f"{'supported' if gc['gain'] >= 5 else 'not supported'}")
        b_rl, b_0 = blown(ev[best]), blown(k0)
        L.append(f"- R3 blow-ups at most half of K0's: {b_rl} vs {b_0} -> "
                 f"{'supported' if b_rl <= b_0 / 2 else 'not supported'}")
        L.append("- R4 WB2 2020 within 3% of N1: see the comparison table")
        if gg:
            L.append(f"- R5 RL reaches >= 90% of Kgrad's gain: {g0['gain']:+.1f} vs {gg['gain']:+.1f} -> "
                     f"{'supported' if g0['gain'] >= 0.9 * gg['gain'] else 'not supported'}")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L[-8:]))


if __name__ == "__main__":
    main()
