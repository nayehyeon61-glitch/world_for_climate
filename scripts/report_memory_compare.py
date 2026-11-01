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
    "C5": "observations (감자 09:49): with the points outside a hidden cap re-observed every 6 h, the memory "
          "model's RMSE on the cap is >= 10% below N1 + nudging at 24 h (mean over z500, t850, t2m, u10; the "
          "stage 10 P1 in physical units)",
    "C6": "crossing: the first lead at which the observation-fed memory model beats GraphCast's free forecast on "
          "the cap (mean ratio < 1), if any within 120 h",
    "L1": "saturation (10.2 cheap test): the two-year continuous fill and the 30-day spin-up differ by <= 1% at "
          "24 h and 120 h (mean over z500, t850, t2m, u10) with the stage 10 kernel (tau <= 10 days)",
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


def observed_section(gc, nudging, memory, summ):
    leads = (6, 24, 72, 120)
    lines = [f"Hidden cap of {100 * summ.get('region_fraction', 0.25):.0f}% of the sphere per initialisation "
             f"(mean hidden weight {summ.get('hidden_fraction', float('nan')):.2f}); RMSE over the cap only. "
             f"Blow-ups (of {summ.get('inits', '-')}): " + ", ".join(f"{k} {v.get('blowups', '-')}" for k, v in summ.get("variants", {}).items()) + ".", "",
             "| variable | lead | GraphCast (free) | N1 + nudging | memory (posterior) | memory / GraphCast | memory / nudging |",
             "|---|---|---|---|---|---|---|"]
    vs_gc, vs_nudge = {}, {}
    # a blown-up sample keeps its last finite forecast, which can be astronomically large and
    # dominate the mean: a lead whose memory / nudging ratio exceeds 100 is reported as blown
    blown_lead = lambda m, n: not np.isfinite(m) or (np.isfinite(n) and n > 0 and m / n > 100)
    for var in VARS:
        for h in leads:
            g, n, m = rmse(gc, var, h), rmse(nudging, var, h), rmse(memory, var, h)
            if not np.isfinite(m):
                continue
            if blown_lead(m, n):
                lines.append(f"| {SHORT[var]} | {h} h | {_f(g)} | {_f(n)} | blown | - | - |")
                continue
            lines.append(f"| {SHORT[var]} | {h} h | {_f(g)} | {_f(n)} | {_f(m)} | {_f(m / g, 2)}x | {_f(m / n, 2)}x |")
            if var in HEADLINE:
                vs_gc.setdefault(h, []).append(m / g)
                vs_nudge.setdefault(h, []).append(m / n)
    all_leads = sorted({h for _, h in memory})
    cross = [h for h in all_leads
             if not any(blown_lead(rmse(memory, v, h), rmse(nudging, v, h)) for v in HEADLINE)
             and np.nanmean([rmse(memory, v, h) / rmse(gc, v, h) for v in HEADLINE]) < 1]
    return "\n".join(lines), {h: float(np.mean(v)) for h, v in vs_gc.items()}, {h: float(np.mean(v)) for h, v in vs_nudge.items()}, cross


def fill_section(warm10, warm30, fill, blowups):
    """Fill length: 10-day spin-up, 30-day spin-up, continuous two-year fill (gain + = better)."""
    lines = ["Per variable: RMSE of the free run after a 10-day spin-up, a 30-day spin-up and a continuous fill from "
             "2018-01-01 (memory carried through two years); gains against the 10-day spin-up, + = better.", "",
             "| variable | lead | 10 d | 30 d | 2 y | gain 30 d | gain 2 y | 2 y vs 30 d |", "|---|---|---|---|---|---|---|---|"]
    diff = {}
    for var in VARS:
        for h in (24, 72, 120):
            a, b, c = rmse(warm10, var, h), rmse(warm30, var, h), rmse(fill, var, h)
            if not all(np.isfinite(x) for x in (a, b, c)):
                continue
            lines.append(f"| {SHORT[var]} | {h} h | {_f(a)} | {_f(b)} | {_f(c)} | {_pct((a - b) / a)} | {_pct((a - c) / a)} | "
                         f"{_pct((b - c) / b)} |")
            if var in HEADLINE:
                diff.setdefault(h, []).append((b - c) / b)
    gain = {h: float(np.mean(v)) for h, v in diff.items()}
    lines += ["", "Mean gain of the two-year fill over the 30-day spin-up (z500, t850, t2m, u10): " +
              ", ".join(f"{h} h {_pct(g)}" for h, g in gain.items()) +
              (f"; blow-ups {blowups}" if blowups else "")]
    return "\n".join(lines), gain


def observed_extra_section(nudging, extras):
    """Further variants of the observed protocol against N1 + nudging on the cap
    (extras: [(label, table, blowups)]); a lead whose ratio exceeds 100 is blown."""
    leads = (6, 24, 72, 120)
    lines = ["Further variants of the same protocol (RMSE on the cap as a ratio to N1 + nudging; blown = the "
             "mean is dominated by blown-up samples):", "",
             "| variant | blow-ups | " + " | ".join(f"{SHORT[v]} {h} h" for v in HEADLINE for h in leads) + " |",
             "|---|---|" + "---|" * (len(HEADLINE) * len(leads))]
    for label, table, blow in extras:
        cells = []
        for v in HEADLINE:
            for h in leads:
                m, n = rmse(table, v, h), rmse(nudging, v, h)
                cells.append("blown" if not np.isfinite(m) or (n > 0 and m / n > 100) else f"{m / n:.2f}x")
        lines.append(f"| {label} | {blow if blow is not None else '-'} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def verdicts(gains, knock_gain_24, ratios, below, obs=None, obs_blowups=None, fill_gain=None):
    out = {}
    if fill_gain and 24 in fill_gain and 120 in fill_gain:
        ok = abs(fill_gain[24]) <= 0.01 and abs(fill_gain[120]) <= 0.01
        out["L1"] = (f"{'supported' if ok else 'not supported'} (two-year fill vs 30 days: 24 h {_pct(fill_gain[24])}, "
                     f"120 h {_pct(fill_gain[120])}, + = the longer fill is better" +
                     ("" if ok else "; the memory state after two years is not the state after 30 days, so the kernel's "
                                    "effective time scale is longer than its nominal tau") + ")")
    if obs:
        vs_gc, vs_nudge, cross = obs
        blow = f"; blow-ups {obs_blowups}" if obs_blowups else ""
        if 24 in vs_nudge:
            out["C5"] = (f"{'supported' if vs_nudge[24] <= 0.9 else 'not supported'} (memory / nudging on the cap: " +
                         ", ".join(f"{h} h {r:.2f}x" for h, r in vs_nudge.items()) + blow + ")")
        else:
            out["C5"] = ("not supported (the memory model blew up before 24 h on most initialisations; 6 h ratio "
                         + ", ".join(f"{r:.2f}x" for h, r in vs_nudge.items() if h == 6) + blow + ")")
        out["C6"] = ((f"crosses at {cross[0]} h" if cross else "no crossing within the forecast") +
                     " (memory / GraphCast on the cap: " + ", ".join(f"{h} h {r:.2f}x" for h, r in vs_gc.items()) + blow + ")")
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
    label = "memory cold" if "memory cold" in below else None  # the rule names the memory model
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
    parser.add_argument("--obs2020", default=None, help="OUT of compare_memory.py --obs-kind region (warm + nudging)")
    parser.add_argument("--fill2020", nargs="*", default=[], help="warm30=OUT(--spinup 120) fill=OUT(--fill continuous)")
    parser.add_argument("--obs-extra", nargs="*", default=[], metavar="LABEL=OUT:VARIANT",
                        help="further observed-protocol runs (e.g. E1=...compare_obs_energy_2020.json:warm "
                             "no_readout=...compare_obs_knock_2020.json:warm_no_readout)")
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
    any18 = next((v for v in cold18.values() if v), {})
    ours = [(c, t) for c, t in (("GraphCast (WB2)", any18.get("graphcast")),
                                ("N0", (cold18.get("n0") or {}).get("model")), ("N1", (cold18.get("n1") or {}).get("model")),
                                ("memory cold", (cold18.get("memory") or {}).get("model")),
                                ("memory warm", w18 and w18["model"])) if t]
    if ours:
        text, below = climode_section(ours)
        md += [text, ""]
    else:
        md.append("No 2018 scores yet.")
    obs = None
    if args.obs2020:
        ow, on = warm(args.obs2020, "warm"), warm(args.obs2020, "nudging")
        osumm = {}
        if os.path.exists(args.obs2020.replace(".json", "_summary.json")):
            with open(args.obs2020.replace(".json", "_summary.json")) as f:
                osumm = json.load(f)
        if ow and on:
            md += ["", "## Forecasts fed by observations outside a hidden cap (2020)", ""]
            entry = dict(osumm, **osumm.get("variants", {}).get("warm", {}))
            text, vs_gc, vs_nudge, cross = observed_section(ow["graphcast"], on["model"], ow["model"], entry)
            md += [text, ""]
            obs = (vs_gc, vs_nudge, cross)
            obs_blowups = {k: v.get("blowups") for k, v in osumm.get("variants", {}).items() if "blowups" in v}
            extras = []
            for item in args.obs_extra:
                label, rest = item.split("=", 1)
                base_out, variant = rest.rsplit(":", 1)
                table = warm(base_out, variant)
                blow = None
                sp = base_out.replace(".json", "_summary.json")
                if os.path.exists(sp):
                    with open(sp) as f:
                        blow = json.load(f).get("variants", {}).get(variant, {}).get("blowups")
                if table:
                    extras.append((label, table["model"], blow))
            if extras:
                md += [observed_extra_section(on["model"], extras), ""]
    fill_gain = None
    if args.fill2020 and w20:
        paths = dict(s.split("=", 1) for s in args.fill2020)
        w30, fl = warm(paths.get("warm30"), "warm"), warm(paths.get("fill"), "fill")
        if w30 and fl:
            blow = {}
            for name, path in (("30 d", paths["warm30"]), ("2 y", paths["fill"])):
                sp = path.replace(".json", "_summary.json")
                if os.path.exists(sp):
                    with open(sp) as f:
                        blow[name] = next(iter(json.load(f)["variants"].values())).get("blowups")
            md += ["", "## Fill length: 10 days, 30 days, two years (2020)", ""]
            text, fill_gain = fill_section(w20["model"], w30["model"], fl["model"], blow)
            md += [text, ""]
    v = verdicts(gains, knock24, ratios, below, obs, obs_blowups if obs else None, fill_gain)
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
