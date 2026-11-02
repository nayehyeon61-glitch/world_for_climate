"""Scalability report for the stage 9 physics model: is N1 limited by its size?

    python scripts/report_physics_scaling.py --base runs/physics_n1/skill_2018.json \
        --arms w05=runs/physics_scale_w05/skill_2018.json w2=... l05=... t21=... \
        --bench runs/bench.jsonl --games osel=runs/stage11_osel/game_eval.json osel8=... \
        --out docs/results/stage9_physics/scaling.md

Inputs:
  - skill files from scripts/eval_physics_causes.py with no interventions (the
    intact rollout on the same 2018 initialisations), one per arm; changes are
    paired with the base over initialisations (bootstrap 95%);
  - the JSON lines of scripts/benchmark_physics.py (time and memory per truncation);
  - optionally the masked-game evaluations of the K = 4 and K = 8 operator banks.
The verdicts S1-S5 are the decision rules fixed before the runs (README_KO §9.3).
"""

import argparse
import json
import os

import _path  # noqa: F401  (adds src/ to sys.path)
import numpy as np

from climate_wm.eval.physics_causes import bootstrap
from report_physics_causes import SHORT, _load, _pct

RULES = {
    "S1": "width: w2 lowers the all-variable error at 24 h or 72 h by >= 5% (interval below 0) -> width-limited; "
          "w2 within 3% and w05 not worse by >= 5% -> width is not the bottleneck",
    "S2": "length: l05 (half the updates) within 3% of N1 at 24 and 72 h -> not limited by training length; "
          "worse by >= 5% -> more updates or data should help",
    "S3": "resolution: T21 worse than N1 (T31) by >= 10% at 24 h or 72 h -> resolution matters, so the cluster "
          "T119 version is the next lever",
    "S4": "compute: the training step grows no faster than N^1.5 in grid points N, and the extrapolated T119 "
          "training (4-step rollout) fits in 96 GB",
    "S5": "controller: K = 8 operators beat K = 4 by >= 3% on the 2020 hidden 24 h error",
}
ARM_TEXT = {"w05": "column width x0.5 (128)", "w2": "column width x2 (512)", "l05": "half the updates (500)",
            "t21": "core T21 instead of T31"}
LEADS = (6, 24, 72)
T119 = 180  # quadratic Gaussian grid latitudes at T119 (1 degree)


def _records(res):
    return res["interventions"]["full"]["records"]


def paired(base, arm):
    """Relative change of the arm vs the base per lead: all variables and each headline key."""
    b, a = _records(base), _records(arm)
    ref = np.array(b["all"], np.float64)
    out = {"all": bootstrap((np.array(a["all"], np.float64) - ref) / np.nanmean(ref, 0)),
           "blowups": int(np.array(a["blown"])[:, -1].sum()), "base_blowups": int(np.array(b["blown"])[:, -1].sum())}
    for k in base["keys"]:
        r = np.array(b["rmse"][k], np.float64)
        out[k] = bootstrap((np.array(a["rmse"][k], np.float64) - r) / np.nanmean(r, 0))
    return out


def _leads(base, wanted):
    """The wanted leads the evaluation reached (the last lead if none)."""
    return [h for h in wanted if h in base["leads_h"]] or base["leads_h"][-1:]


def capacity_table(base, changes):
    leads, shown, last = base["leads_h"], _leads(base, LEADS), _leads(base, LEADS)[-1]
    rows = ["| arm | change | " + " | ".join(f"all {h} h" for h in shown) + " | " +
            " | ".join(f"{SHORT.get(k, k)} {last} h" for k in base["keys"]) + " | blow-ups |",
            "|---|---|" + "---|" * (len(shown) + len(base["keys"]) + 1)]
    for name, c in changes.items():
        cells = [_pct(c["all"], leads.index(h)) for h in shown]
        cells += [_pct(c[k], leads.index(last), ci=False) for k in base["keys"]]
        rows.append(f"| {name} | {ARM_TEXT.get(name, name)} | " + " | ".join(cells) +
                    f" | {c['blowups']} (N1 {c['base_blowups']}) |")
    return "\n".join(rows)


def fit(x, y):
    """Exponent and prefactor of y = c x^p (least squares in logs)."""
    p, logc = np.polyfit(np.log(x), np.log(y), 1)
    return float(p), float(np.exp(logc))


def compute_section(bench):
    rows = [b for b in bench if "train_step_s" in b]
    res = [b for b in rows if b.get("operators") == 4]  # the resolution sweep (N1's width, K = 4)
    if len(res) < 3:
        return "Too few truncations finished for a fit.", None
    lines = ["| truncation | grid | dt (min) | 6 h forecast (s) | training step (s) | operator bank (s) | peak memory (GB) |",
             "|---|---|---|---|---|---|---|"]
    for b in res:
        lines.append(f"| T{b['trunc']} | {b['grid'][0]}x{b['grid'][1]} | {b['dt_min']:.0f} | {b['forecast_6h_s']:.2f} | "
                     f"{b['train_step_s']:.2f} | {b.get('operator_bank_s', float('nan')):.3f} | {b['peak_rss_gb']:.2f} |")
    n = np.array([b["grid"][0] * b["grid"][1] for b in res], np.float64)
    fits = {key: fit(n, [b[key] for b in res]) for key in ("forecast_6h_s", "train_step_s")}
    n119 = T119 * 2 * T119
    ext = {key: c * n119 ** p for key, (p, c) in fits.items()}
    # memory = fixed overhead (interpreter, tables) + a part per grid point; only the latter
    # grows with the rollout length (4 steps in training, before gradient checkpointing saves)
    per_point, overhead = np.polyfit(n, [b["peak_rss_gb"] for b in res], 1)
    ext["memory_1step_gb"] = overhead + per_point * n119
    ext["memory_4step_gb"] = overhead + 4 * per_point * n119
    lines += ["", "Fits y = c N^p over grid points N (CPU, 1 thread, two other jobs running, so times are "
              "upper bounds); memory as overhead + slope x N:", ""]
    for key, (p, c) in fits.items():
        lines.append(f"- {key}: p = {p:.2f}; extrapolated T119 ({T119}x{2 * T119}): {ext[key]:.3g} s")
    lines.append(f"- memory: {overhead:.2f} GB + {1e6 * per_point:.1f} GB per million points; T119 one-step training "
                 f"{ext['memory_1step_gb']:.1f} GB, 4-step rollout ~{ext['memory_4step_gb']:.1f} GB")
    widths = [b for b in rows if b["trunc"] == 31 and b.get("operators") == 8]
    if widths:
        lines += ["", "Column width at T31 (K = 8 operator bank):", "",
                  "| column width | parameters | training step (s) | operator bank K = 8 (s) | peak memory (GB) |",
                  "|---|---|---|---|---|"]
        for b in sorted(widths, key=lambda b: b["column_hidden"]):
            lines.append(f"| {b['column_hidden']} | {b['params']:,} | {b['train_step_s']:.2f} | "
                         f"{b['operator_bank_s']:.3f} | {b['peak_rss_gb']:.2f} |")
    return "\n".join(lines), {"fits": fits, "t119": ext}


def game_section(games):
    if not all(games.get(k) for k in ("osel", "osel8")):
        return "The K = 8 bank has not been played yet.", None
    lines = ["| kind | K = 4 hidden 24 h vs no control | K = 8 hidden 24 h vs no control | K = 8 / K = 4 error |",
             "|---|---|---|---|"]
    ratios = []
    a4, a8 = (next(iter(games[k]["arms"].values())) for k in ("osel", "osel8"))
    for kind in a4["kinds"]:
        if kind not in a8["kinds"]:
            continue
        c4, c8 = a4["kinds"][kind]["change"]["control_hidden"], a8["kinds"][kind]["change"]["control_hidden"]
        m4, m8 = a4["kinds"][kind]["mse"]["control_hidden"][-1], a8["kinds"][kind]["mse"]["control_hidden"][-1]
        ratios.append(m8 / m4 - 1)
        lines.append(f"| {kind} | {_pct(c4, -1)} | {_pct(c8, -1)} | {100 * (m8 / m4 - 1):+.1f}% |")
    return "\n".join(lines), float(np.mean(ratios)) if ratios else None


def verdicts(changes, base, comp, game_ratio):
    li = {h: base["leads_h"].index(h) for h in _leads(base, (24, 72))}
    m = lambda arm: {h: changes[arm]["all"]["mean"][l] for h, l in li.items()} if arm in changes else None
    hi = lambda arm: {h: changes[arm]["all"]["hi"][l] for h, l in li.items()} if arm in changes else None
    out = {}
    w2, w05 = m("w2"), m("w05")
    if w2:
        if any(w2[h] <= -0.05 and hi("w2")[h] < 0 for h in w2):
            out["S1"] = "supported: width-limited"
        elif all(abs(v) <= 0.03 for v in w2.values()) and (w05 is None or all(v < 0.05 for v in w05.values())):
            out["S1"] = "width is not the bottleneck"
        else:
            out["S1"] = "in between"
    l05 = m("l05")
    if l05:
        out["S2"] = ("not limited by training length" if all(abs(v) <= 0.03 for v in l05.values()) else
                     "length-limited" if any(v >= 0.05 for v in l05.values()) else "in between")
    t21 = m("t21")
    if t21:
        out["S3"] = "resolution matters" if any(v >= 0.10 for v in t21.values()) else "resolution is not the lever here"
    if comp:
        p = comp["fits"]["train_step_s"][0]
        mem = comp["t119"]["memory_4step_gb"]
        out["S4"] = f"{'ok' if p <= 1.5 and mem <= 96 else 'not ok'} (p = {p:.2f}, T119 4-step memory ~{mem:.0f} GB)"
    if game_ratio is not None:
        out["S5"] = f"{'supported' if game_ratio <= -0.03 else 'not supported'} (K = 8 / K = 4 error {100 * game_ratio:+.1f}%)"
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--arms", nargs="*", default=[], help="name=skill.json")
    parser.add_argument("--bench", default=None)
    parser.add_argument("--games", nargs="*", default=[], help="osel=game_eval.json osel8=game_eval.json")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    base = _load(args.base)
    changes = {}
    for item in args.arms:
        name, path = item.split("=", 1)
        res = _load(path)
        if res:
            changes[name] = paired(base, res)
    bench = []
    if args.bench and os.path.exists(args.bench):
        with open(args.bench) as f:
            bench = [json.loads(line) for line in f if line.strip()]
    comp_text, comp = compute_section(bench)
    games = {k: _load(v) for k, v in (g.split("=", 1) for g in args.games)}
    game_text, game_ratio = game_section(games)
    v = verdicts(changes, base, comp, game_ratio)
    md = ["# Stage 9: scalability of the physics model (N1)", "",
          f"Base: N1 intact rollout, {base['inits']} initialisations of 2018. Changes are paired with N1 "
          "(negative = better), mean [95% bootstrap].", "",
          "## Capacity", "", capacity_table(base, changes) if changes else "No arm finished yet.", "",
          "## Compute", "", comp_text, "", "## Controller size (stage 11)", "", game_text, "",
          "## Decision rules", ""]
    md += [f"- **{k}** {RULES[k]}  \n  → {v.get(k, 'pending')}" for k in RULES]
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
