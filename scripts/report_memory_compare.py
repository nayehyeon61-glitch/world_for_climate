"""The stage 10 memory model against GraphCast, N1 and ClimODE (README_KO §10).

    python scripts/report_memory_compare.py \
        --cold2020 memory=runs/stage10_Dnoactor/compare_graphcast_2020.json n1=runs/physics_n1/compare_graphcast_2020.json \
        --warm2020 runs/stage10_Dnoactor/compare_memory_2020.json \
        --cold2018 memory=runs/stage10_Dnoactor/compare_graphcast_2018.json n1=runs/physics_n1/compare_graphcast.json \
                   n0=runs/physics_n0/compare_graphcast.json \
        --out docs/results/stage10_memory/compare.md

Inputs: the JSON files of scripts/compare_graphcast.py (name=path; "memory" is
the memory model, cold start) and the OUT of scripts/compare_memory.py (its
OUT_warm.json, OUT_warm_no_memory.json and OUT_summary.json). Scores are means
over the common initialisations of the GraphCast protocol (31 per year); the
protocol keeps no per-initialisation record, so there are no intervals here.
ClimODE's numbers are its Table 6 (Verma, Heinonen & Garg, ICLR 2024):
WeatherBench 5.625 deg, trained 2006-2015, test 2017-2018.
"""

import argparse
import json
import os

import numpy as np

VARS = ["geopotential@500", "temperature@850", "specific_humidity@700", "u_component_of_wind@850",
        "2m_temperature", "mean_sea_level_pressure", "10m_u_component_of_wind", "10m_v_component_of_wind",
        "total_precipitation_6hr"]
SHORT = {"geopotential@500": "z500", "temperature@850": "t850", "specific_humidity@700": "q700",
         "u_component_of_wind@850": "u850", "2m_temperature": "t2m", "mean_sea_level_pressure": "msl",
         "10m_u_component_of_wind": "u10", "10m_v_component_of_wind": "v10", "total_precipitation_6hr": "tp6"}
HEADLINE = ["geopotential@500", "temperature@850", "2m_temperature", "10m_u_component_of_wind"]  # ClimODE's set
LEADS = (6, 24, 72, 120)
CLIMODE_LEADS = (6, 12, 18, 24, 36)
# Verma, Heinonen & Garg (ICLR 2024), Table 6: latitude-weighted RMSE at 6/12/18/24/36 h (None = not reported)
CLIMODE = {
    "z500": {"ClimODE": [102.9, 134.8, 162.7, 193.4, 259.6], "FourCastNet": [149.4, 217.8, 275.0, 333.0, 449.0],
             "ClimaX": [247.5, 265.3, 319.8, 364.9, 455.0], "IFS": [26.9, None, None, 51.0, None]},
    "t850": {"ClimODE": [1.16, 1.32, 1.47, 1.55, 1.75], "FourCastNet": [1.18, 1.47, 1.65, 1.83, 2.21],
             "ClimaX": [1.64, 1.77, 1.93, 2.17, 2.49], "IFS": [0.69, None, None, 0.87, None]},
    "t2m": {"ClimODE": [1.21, 1.45, 1.43, 1.40, 1.70], "FourCastNet": [1.28, 1.48, 1.61, 1.68, 1.90],
            "ClimaX": [2.02, 2.26, 2.45, 2.37, 2.87], "IFS": [0.97, None, None, 1.02, None]},
    "u10": {"ClimODE": [1.41, 1.81, 1.97, 2.01, 2.25], "FourCastNet": [1.47, 1.89, 2.05, 2.33, 2.87],
            "ClimaX": [1.58, 1.96, 2.24, 2.49, 2.98], "IFS": [0.80, None, None, 1.11, None]},
    "v10": {"ClimODE": [1.53, 1.81, 1.96, 2.04, 2.29], "FourCastNet": [1.54, 1.81, 2.11, 2.39, 2.95],
            "ClimaX": [1.60, 1.97, 2.26, 2.48, 2.98], "IFS": [0.94, None, None, 1.33, None]},
}
RULES = {
    "C1": "memory helps the forecast itself: the warm start lowers the cold start's RMSE by >= 3% at 24 h "
          "(mean over z500, t850, t2m, u10) while warm_no_memory stays within 1% of the cold start",
    "C2": "lead: the lead at which the warm-start gain falls below 1% (where the filled memory stops mattering)",
    "C3": "GraphCast: RMSE ratio to GraphCast (WB2 forecasts coarsened to 5.625 deg) at 24 h and 72 h; "
          "<= 1.5 would be GraphCast-class, which no model of this stage is expected to reach",
    "C4": "ClimODE (indirect): on 2018 at 24 h the cold memory model's RMSE is below ClimODE's published RMSE "
          "on >= 3 of z500, t850, t2m, u10 (2018 is a fine-tuning year of the memory model; ClimODE's test "
          "years are 2017-2018 after ten training years)",
}


def load(path):
    """name -> {(variable, lead_h): row} for one compare_graphcast / compare_memory JSON."""
    if not path or not os.path.exists(path):
        return None
    with open(path) as f:
        payload = json.load(f)
    return {name: {(r["variable"], r["lead_h"]): r for r in rows}
            for name, rows in payload.items() if isinstance(rows, list)}


def rmse(tab, var, lead, key="rmse"):
    row = tab.get((var, lead)) if tab else None
    return float(row[key]) if row and key in row else float("nan")


def _f(x, digits=None):
    if x is None or not np.isfinite(x):
        return "-"
    if digits is not None:
        return f"{x:.{digits}f}"
    return f"{x:.4g}" if abs(x) < 100 else f"{x:.0f}"


def _pct(x):
    return "-" if not np.isfinite(x) else f"{100 * x:+.1f}%"


def same_inits(a, b, var="geopotential@500"):
    """Two files scored the same initialisations if their GraphCast rows agree."""
    if not (a and b and "graphcast" in a and "graphcast" in b):
        return True
    x = [rmse(a["graphcast"], var, h) for h in (6, 24)]
    y = [rmse(b["graphcast"], var, h) for h in (6, 24)]
    return np.allclose(x, y, rtol=1e-4)


def score_table(columns, variables, leads, key="rmse"):
    """columns: [(label, table)]; rows per variable and lead."""
    lines = ["| variable | lead | " + " | ".join(c for c, _ in columns) + " |",
             "|---|---|" + "---|" * len(columns)]
    for var in variables:
        for h in leads:
            cells = [_f(rmse(t, var, h, key)) for _, t in columns]
            if all(c == "-" for c in cells):
                continue
            lines.append(f"| {SHORT.get(var, var)} | {h} h | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def mean_change(num, den, variables, lead):
    r = [rmse(num, v, lead) / rmse(den, v, lead) - 1 for v in variables]
    r = [x for x in r if np.isfinite(x)]
    return float(np.mean(r)) if r else float("nan")


def ablation_section(cold, warm, knock, leads_all):
    lines = ["Mean relative RMSE change over z500, t850, t2m, u10 (negative = better):", "",
             "| lead | warm vs cold (filled memory) | warm_no_memory vs cold (spin-up alone) | warm vs warm_no_memory (the kernel) |",
             "|---|---|---|---|"]
    gains = {}
    for h in leads_all:
        g = mean_change(warm, cold, HEADLINE, h)
        gains[h] = g
        lines.append(f"| {h} h | {_pct(g)} | {_pct(mean_change(knock, cold, HEADLINE, h))} | "
                     f"{_pct(mean_change(warm, knock, HEADLINE, h))} |")
    lines += ["", "Per variable, warm vs cold:", "",
              "| variable | " + " | ".join(f"{h} h" for h in LEADS) + " |", "|---|" + "---|" * len(LEADS)]
    for var in VARS:
        cells = [_pct(rmse(warm, var, h) / rmse(cold, var, h) - 1) for h in LEADS]
        if any(c != "-" for c in cells):
            lines.append(f"| {SHORT[var]} | " + " | ".join(cells) + " |")
    return "\n".join(lines), gains


def graphcast_section(gc, named, leads=(24, 72, 120)):
    lines = ["| variable | " + " | ".join(f"{n} / GraphCast {h} h" for n in named for h in leads) + " |",
             "|---|" + "---|" * (len(named) * len(leads))]
    ratios = {n: {} for n in named}
    for var in HEADLINE:
        cells = []
        for n, t in named.items():
            for h in leads:
                r = rmse(t, var, h) / rmse(gc, var, h)
                ratios[n].setdefault(h, []).append(r)
                cells.append(_f(r, 2) + "x" if np.isfinite(r) else "-")
        lines.append(f"| {SHORT[var]} | " + " | ".join(cells) + " |")
    return "\n".join(lines), {n: {h: float(np.nanmean(v)) for h, v in d.items()} for n, d in ratios.items()}


def climode_section(ours):
    """ours: [(label, table)] on 2018; ClimODE's baselines from its Table 6."""
    theirs = ["ClimODE", "FourCastNet", "ClimaX", "IFS"]
    lines = ["| variable | lead | " + " | ".join(theirs) + " | " + " | ".join(c for c, _ in ours) + " |",
             "|---|---|" + "---|" * (len(theirs) + len(ours))]
    below = {}
    for var in VARS:
        s = SHORT[var]
        if s not in CLIMODE:
            continue
        for i, h in enumerate(CLIMODE_LEADS):
            cells = [_f(CLIMODE[s][n][i]) for n in theirs]
            cells += [_f(rmse(t, var, h)) for _, t in ours]
            lines.append(f"| {s} | {h} h | " + " | ".join(cells) + " |")
            if h == 24:
                for label, t in ours:
                    x = rmse(t, var, h)
                    if np.isfinite(x):
                        below.setdefault(label, []).append(bool(x < CLIMODE[s]["ClimODE"][i]))
    return "\n".join(lines), below


def verdicts(gains, knock_gain_24, ratios, below):
    out = {}
    if 24 in gains and np.isfinite(gains[24]):
        ok = gains[24] <= -0.03 and abs(knock_gain_24) <= 0.01
        out["C1"] = (f"{'supported' if ok else 'not supported'} (warm vs cold {_pct(gains[24])}, spin-up alone "
                     f"{_pct(knock_gain_24)})")
        gone = [h for h in sorted(gains) if np.isfinite(gains[h]) and gains[h] > -0.01]
        out["C2"] = (f"the gain is below 1% from {gone[0]} h on" if gone else
                     f"the gain stays above 1% to {max(gains)} h")
    for n, d in ratios.items():
        out.setdefault("C3", "")
        out["C3"] += f"{n}: " + ", ".join(f"{h} h {_f(r, 2)}x" for h, r in d.items()) + "; "
    label = "memory cold" if "memory cold" in below else next(iter(below), None)
    if label:
        flags = below[label]
        out["C4"] = (f"{'supported' if sum(flags) >= 3 else 'not supported'} ({label} below ClimODE at 24 h on "
                     f"{sum(flags)} of {len(flags)} variables)")
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cold2020", nargs="*", default=[], help="memory=... n1=... (compare_graphcast JSON)")
    parser.add_argument("--warm2020", default=None, help="OUT of compare_memory.py")
    parser.add_argument("--cold2018", nargs="*", default=[], help="memory=... n1=... n0=...")
    parser.add_argument("--warm2018", default=None)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    cold20 = {k: load(v) for k, v in (s.split("=", 1) for s in args.cold2020)}
    cold18 = {k: load(v) for k, v in (s.split("=", 1) for s in args.cold2018)}
    warm = lambda base, name: load(base.replace(".json", f"_{name}.json")) if base else None
    w20, k20 = warm(args.warm2020, "warm"), warm(args.warm2020, "warm_no_memory")
    w18 = warm(args.warm2018, "warm")
    summ = None
    if args.warm2020 and os.path.exists(args.warm2020.replace(".json", "_summary.json")):
        with open(args.warm2020.replace(".json", "_summary.json")) as f:
            summ = json.load(f)
    md = ["# Stage 10: the memory model against GraphCast, N1 and ClimODE", "",
          "Latitude-weighted RMSE in physical units, means over the 31 initialisations of the GraphCast "
          "protocol (every 12 days, 00 UTC), 6 h steps. GraphCast is the WeatherBench-2 forecast coarsened to "
          "the 5.625 deg grid. The memory model is D-noactor (stage 10; N1 + memory kernel, fine-tuned on "
          "2018-2019); cold = empty memory, warm = 10 days of fully re-observed one-step forecasts first.", ""]
    mem, n1 = cold20.get("memory"), cold20.get("n1")
    base = mem or n1
    notes = []
    for a, b, what in ((mem, n1, "memory cold / N1 2020"), (mem, w20, "memory cold / warm 2020")):
        if a and b and not same_inits(a, b):
            notes.append(f"the initialisations of {what} differ")
    md += ["## 2020 (held out for every model)", ""]
    if base:
        cols = [("GraphCast", base["graphcast"]), ("N1 cold", n1 and n1["model"]), ("memory cold", mem and mem["model"]),
                ("memory warm", w20 and w20["model"]), ("warm, no memory", k20 and k20["model"]),
                ("persistence", base["persistence"]), ("climatology", base["climatology"])]
        cols = [(c, t) for c, t in cols if t]
        md += [score_table(cols, VARS, LEADS), "", "ACC:", "",
               score_table([(c, t) for c, t in cols if c != "climatology"], HEADLINE, (24, 72, 120), key="acc"), ""]
        if summ:
            md += ["Blow-ups (samples that kept their last finite forecast): " +
                   ", ".join(f"{k} {v['blowups']}" for k, v in summ["variants"].items()) + f" of {summ['inits']}", ""]
    else:
        md.append("No 2020 comparison yet.")
    gains, knock24, ratios, below = {}, float("nan"), {}, {}
    md += ["", "## Cause ablations (2020)", ""]
    if mem and w20 and k20:
        leads_all = sorted({h for _, h in mem["model"]})
        text, gains = ablation_section(mem["model"], w20["model"], k20["model"], leads_all)
        knock24 = mean_change(k20["model"], mem["model"], HEADLINE, 24)
        md += [text, ""]
    else:
        md.append("The warm start has not finished.")
    md += ["", "## Against GraphCast (2020)", ""]
    if base:
        named = {k: v for k, v in (("N1", n1 and n1["model"]), ("memory cold", mem and mem["model"]),
                                   ("memory warm", w20 and w20["model"])) if v}
        text, ratios = graphcast_section(base["graphcast"], named)
        md += [text, ""]
    md += ["", "## Against ClimODE (2018; ClimODE's Table 6 is its 2017-2018 test)", "",
           "ClimODE, FourCastNet, ClimaX and IFS as printed in Verma et al. (2024); ours on the 31 initialisations "
           "of 2018. 2018 is a fine-tuning year of the memory model (its N1 base was trained on 2015-2017), so "
           "its 2018 scores are a reference, not a held-out test.", ""]
    ours = [(c, t) for c, t in (("GraphCast (WB2)", (cold18.get("memory") or cold18.get("n1") or {}).get("graphcast")),
                                ("N0", (cold18.get("n0") or {}).get("model")), ("N1", (cold18.get("n1") or {}).get("model")),
                                ("memory cold", (cold18.get("memory") or {}).get("model")),
                                ("memory warm", w18 and w18["model"])) if t]
    if ours:
        text, below = climode_section(ours)
        md += [text, ""]
    else:
        md.append("No 2018 scores yet.")
    v = verdicts(gains, knock24, ratios, below)
    md += ["", "## Decision rules", ""]
    md += [f"- **{k}** {RULES[k]}  \n  → {v.get(k, 'pending')}" for k in RULES]
    if notes:
        md += ["", "Notes: " + "; ".join(notes)]
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
