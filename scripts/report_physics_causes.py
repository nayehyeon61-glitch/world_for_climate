"""Stage 9 cause-analysis report: the coupled physics of N1 under interventions.

    python scripts/report_physics_causes.py --n1 runs/physics_n1/causes_2018.json runs/physics_n1/causes_2020.json \
        --n0 runs/physics_n0/causes_2018.json runs/physics_n0/causes_2020.json \
        --out docs/results/stage9_physics/causes.md

Inputs: scripts/eval_physics_causes.py, one file per model and year (same
initialisation dates in both models). Tables:
  - the arrows (skill change vs the intact model, blow-ups, N0's sign);
  - propagation (response per variable group and lead);
  - the sun;
  - the pairs (interaction, eta, rho);
  - invariance across the years.
Each table ends with the verdicts on the pre-registered expectations E1-E6
(EXPECT; docs/results/README_KO.md §9).
"""

import argparse
import json
import os

import numpy as np

EXPECT = {
    "E1": "do(TISR = 0) cools day points more than night points at 6 h, and the 2 m temperature response "
          "correlates with the withheld energy (r < -0.3)",
    "E2": "the column heating carries the sun's effect on temperature by 72 h: rho(sun_off given no_heating) < 0.5",
    "E3": "the latent-heat loop is non-additive: eta(no_moisture + q_to_heating) >= 0.1 at 72 h on "
          "temperature or water, larger than at 6 h",
    "E4": "effects propagate: the response grows from 6 h to 72 h for >= 80% of (intervention, group)",
    "E5": "the transports that N0 is better without (no_vertical, no_moisture at 72 h) hurt N1",
    "E6": "invariance: >= 80% sign agreement 2018 vs 2020 on the significant per-variable changes at 24 and "
          "72 h, and rank correlation of the 72 h effect sizes >= 0.8",
}
GROUPS = ("thermodynamics", "momentum", "hydrostatic", "water")
SHORT_G = {"thermodynamics": "T", "momentum": "u", "hydrostatic": "Φ", "water": "q"}
SHORT = {"geopotential@500": "z500", "temperature@850": "t850", "specific_humidity@700": "q700",
         "u_component_of_wind@850": "u850", "2m_temperature": "t2m", "mean_sea_level_pressure": "msl",
         "10m_u_component_of_wind": "u10", "total_precipitation_6hr": "tp"}


def _load(path):
    if path and os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def _li(res, hours):
    return res["leads_h"].index(hours)


def _pct(b, l, ci=True):
    m = b["mean"][l]
    if m is None or not np.isfinite(m):
        return "blown"
    s = f"{100 * m:+.0f}%"
    if ci and np.isfinite(b["lo"][l]) and np.isfinite(b["hi"][l]):
        s += f" [{100 * b['lo'][l]:+.0f}, {100 * b['hi'][l]:+.0f}]"
    return s


def _num(b, l, fmt="{:.2f}"):
    m = b["mean"][l]
    return fmt.format(m) if m is not None and np.isfinite(m) else "–"


def _sig(b, l):
    lo, hi = b["lo"][l], b["hi"][l]
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return 0
    return 1 if lo > 0 else (-1 if hi < 0 else 0)


def _last_finite(b):
    """(lead index, value) of the last finite mean, None if all blew up."""
    vals = [(l, v) for l, v in enumerate(b["mean"]) if v is not None and np.isfinite(v)]
    return vals[-1] if vals else None


def arrow_table(n1s, n0s, years):
    head = ["intervention", "arrow"] + [f"{y} {h} h" for y in years for h in (6, 24, 72)] + ["blow-ups"]
    if n0s:
        head.append("N0 72 h")
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    ref = n1s[0]
    for name, s in ref["interventions"].items():
        if name == "full" or "+" in name:
            continue
        row = [f"`{name}`", s["edge"]]
        blow = []
        for r in n1s:
            x = r["interventions"].get(name)
            for h in (6, 24, 72):
                row.append(_pct(x["change_all"], _li(r, h), ci=False) if x else "–")
            blow.append(str(x["blowups"][-1]) if x else "–")
        row.append("/".join(blow))
        if n0s:
            vals = []
            for r in n0s:
                x = r["interventions"].get(name)
                vals.append(_pct(x["change_all"], _li(r, 72), ci=False) if x else "n/a")
            row.append(" / ".join(vals))
        lines.append("| " + " | ".join(row) + " |")
    return lines


def variable_table(res, hours=72):
    keys = res["keys"]
    l = _li(res, hours)
    lines = ["| intervention | " + " | ".join(SHORT.get(k, k) for k in keys) + " |", "|---|" + "---|" * len(keys)]
    for name, s in res["interventions"].items():
        if name == "full":
            continue
        lines.append(f"| `{name}` | " + " | ".join(_pct(s["change"][k], l, ci=False) for k in keys) + " |")
    return lines


def arrival(res, b, frac=0.5):
    """First lead (h) at which a response reaches `frac` of its 72 h value."""
    m = b["mean"]
    end = m[_li(res, 72)]
    if end is None or not np.isfinite(end) or end <= 0:
        return None
    for l, v in enumerate(m):
        if v is not None and np.isfinite(v) and v >= frac * end:
            return res["leads_h"][l]
    return None


def propagation_table(res):
    lines = ["| intervention | " + " | ".join(f"{g} 6/24/72 h" for g in GROUPS)
             + " | arrival (h, half of 72 h): " + " / ".join(SHORT_G[g] for g in GROUPS) + " | order |",
             "|---|" + "---|" * (len(GROUPS) + 2)]
    grow, total = 0, 0
    for name, s in res["interventions"].items():
        if name == "full" or "+" in name:
            continue
        cells, arr = [], {}
        for g in GROUPS:
            b = s["response"][g]
            vals = [b["mean"][_li(res, h)] for h in (6, 24, 72)]
            cells.append(" / ".join(f"{v:.3f}" if v is not None and np.isfinite(v) else "–" for v in vals))
            if all(v is not None and np.isfinite(v) for v in (vals[0], vals[2])):
                total += 1
                grow += vals[2] > vals[0]
            arr[g] = arrival(res, b)
        known = sorted((a, SHORT_G[g]) for g, a in arr.items() if a is not None)
        order = " → ".join(" = ".join(x for t, x in known if t == a) for a in sorted({t for t, _ in known}))
        lines.append(f"| `{name}` | " + " | ".join(cells) + " | "
                     + " / ".join("–" if arr[g] is None else str(arr[g]) for g in GROUPS) + f" | {order or '–'} |")
    return lines, (grow / total if total else float("nan"))


def sun_table(res):
    lines = ["| intervention | Δt2m day 6/24/72 h (K) | Δt2m night | r(Δt2m, sunlight) |", "|---|---|---|---|"]
    for name in ("sun_off", "sun_to_heating", "ocean_column", "q_to_heating", "no_heating", "sun_off+no_heating"):
        s = res["interventions"].get(name)
        if not s:
            continue
        r = s["response"]
        cell = lambda b: " / ".join(_num(b, _li(res, h), "{:+.2f}") for h in (6, 24, 72))
        lines.append(f"| `{name}` | {cell(r['t2m_day_K'])} | {cell(r['t2m_night_K'])} | {cell(r['t2m_corr_energy'])} |")
    return lines


def pair_table(res):
    pl = res["pair_leads_h"]
    lines = ["| pair (A + B) | interaction, all-MSE 6/24/72 h | eta T / u / Φ / q at 72 h | "
             "rho(A given B) T / u / Φ / q at 72 h | rho(B given A) at 72 h |", "|---|---|---|---|---|"]
    for name, s in res["interventions"].items():
        p = s.get("pair")
        if not p:
            continue
        it = p.get("interaction_all")
        inter = " / ".join(_pct(it, _li(res, h), ci=False) for h in (6, 24, 72)) if it else "–"
        last = len(pl) - 1
        tri = lambda key: " / ".join(_num(p[key][g], last) for g in GROUPS)
        lines.append(f"| `{p['a']}` + `{p['b']}` | {inter} | {tri('eta')} | {tri('rho_a_given_b')} | "
                     f"{tri('rho_b_given_a')} |")
    return lines


def invariance(a, b):
    """Sign agreement on significant per-variable changes (24, 72 h) and rank correlation of |all| at 72 h."""
    agree = n = 0
    sizes_a, sizes_b = [], []
    for name, sa in a["interventions"].items():
        sb = b["interventions"].get(name)
        if name == "full" or sb is None:
            continue
        for k in a["keys"]:
            for h in (24, 72):
                ga, gb = _sig(sa["change"][k], _li(a, h)), _sig(sb["change"][k], _li(b, h))
                if ga and gb:
                    n += 1
                    agree += ga == gb
        xa, xb = sa["change_all"]["mean"][_li(a, 72)], sb["change_all"]["mean"][_li(b, 72)]
        if all(x is not None and np.isfinite(x) for x in (xa, xb)):
            sizes_a.append(abs(xa))
            sizes_b.append(abs(xb))
    rank = lambda x: np.argsort(np.argsort(x))
    rho = float(np.corrcoef(rank(sizes_a), rank(sizes_b))[0, 1]) if len(sizes_a) > 2 else float("nan")
    return (agree / n if n else float("nan")), n, rho, len(sizes_a)


def verdicts(n1s, n0s, growth):
    r = n1s[0]
    out = {}
    s = r["interventions"].get("sun_off")
    if s:
        day, night = s["response"]["t2m_day_K"]["mean"][0], s["response"]["t2m_night_K"]["mean"][0]
        corr = s["response"]["t2m_corr_energy"]["mean"][0]
        ok = day < night and abs(day) > abs(night) and corr < -0.3
        out["E1"] = (ok, f"6 h: day {day:+.2f} K, night {night:+.2f} K, r = {corr:+.2f}")
    p = r["interventions"].get("sun_off+no_heating", {}).get("pair")
    if p:
        rho = p["rho_a_given_b"]["thermodynamics"]["mean"]
        out["E2"] = (rho[-1] < 0.5, "rho(sun given no heating) on T at " + ", ".join(
            f"{h} h {v:.2f}" for h, v in zip(r["pair_leads_h"], rho)))
    p = r["interventions"].get("no_moisture+q_to_heating", {}).get("pair")
    if p:
        e = {g: p["eta"][g]["mean"] for g in ("thermodynamics", "water")}
        ok = any(v[-1] >= 0.1 and v[-1] > v[0] for v in e.values())
        out["E3"] = (ok, "eta 6 → 72 h: " + ", ".join(f"{g} {v[0]:.2f} → {v[-1]:.2f}" for g, v in e.items()))
    out["E4"] = (growth >= 0.8, f"{100 * growth:.0f}% of (intervention, group) grow from 6 to 72 h")
    if n0s:
        flips = []
        for name in ("no_vertical", "no_moisture"):
            a, b = n0s[0]["interventions"].get(name), r["interventions"].get(name)
            if a and b:
                fa, fb = _last_finite(a["change_all"]), _last_finite(b["change_all"])
                if fa and fb:
                    flips.append((name, fa[1], fb[1]))
        ok = bool(flips) and all(x0 < 0 < x1 for _, x0, x1 in flips)
        out["E5"] = (ok, "; ".join(f"{n}: N0 {100 * x0:+.0f}%, N1 {100 * x1:+.0f}%" for n, x0, x1 in flips))
    if len(n1s) > 1:
        frac, n, rho, m = invariance(n1s[0], n1s[1])
        out["E6"] = (frac >= 0.8 and rho >= 0.8,
                     f"sign agreement {100 * frac:.0f}% of {n} significant changes; rank r = {rho:.2f} over {m}")
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n1", nargs="+", required=True, help="N1 causes json per year (first = main)")
    parser.add_argument("--n0", nargs="*", default=[], help="N0 causes json per year (same order)")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    n1s = [r for r in map(_load, args.n1) if r]
    n0s = [r for r in map(_load, args.n0) if r]
    years = [r["year"] for r in n1s]
    md = ["# Stage 9: the coupled physics of N1 under interventions", "",
          f"Model: `{'/'.join(n1s[0]['checkpoint'].split('/')[-3:])}`; {n1s[0]['inits']} initialisations per year ({', '.join(map(str, years))}),"
          " 72 h, the same dates in N0. Changes: channel-mean normalised MSE vs the intact model"
          " (+ = worse without the arrow). Blow-ups: runs that diverged by 72 h (per year).", "",
          "## The arrows", ""]
    md += arrow_table(n1s, n0s, years)
    md += ["", "Arrows without a path in N1:", ""]
    md += [f"- {k}: {v}" for k, v in n1s[0]["missing_edges"].items()]
    for r in n1s:
        md += ["", f"## Per variable at 72 h, {r['year']}", ""] + variable_table(r)
    prop, growth = propagation_table(n1s[0])
    md += ["", f"## Propagation ({years[0]}): RMS response vs the intact model, normalised units", ""] + prop
    for r in n1s:
        md += ["", f"## The sun ({r['year']})", ""] + sun_table(r)
    for r in n1s:
        md += ["", f"## Pairs: loops of the diagram ({r['year']})", "",
               "interaction = change(A+B) - change(A) - change(B); eta = |D_AB - D_A - D_B| / (|D_A| + |D_B|) "
               "(0: independent paths); rho(A given B) = share of A's effect left once B is cut "
               "(0: B mediates A).", ""] + pair_table(r)
    if len(n1s) > 1:
        frac, n, rho, m = invariance(n1s[0], n1s[1])
        md += ["", f"## Invariance {years[0]} vs {years[1]}", "",
               f"Sign agreement on the {n} per-variable changes significant in both years (24 and 72 h): "
               f"{100 * frac:.0f}%. Rank correlation of the 72 h effect sizes over {m} interventions: {rho:.2f}."]
    md += ["", "## Expectations (pre-registered)", "", "| | expectation | verdict | numbers |", "|---|---|---|---|"]
    for k, (ok, text) in verdicts(n1s, n0s, growth).items():
        md.append(f"| {k} | {EXPECT[k]} | {'supported' if ok else 'not supported'} | {text} |")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
