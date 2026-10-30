"""Side-by-side report of a base model and its refinements on one test year.

Each input is compare_graphcast.json from the same initialisations; the first
is the reference (the base model). Writes refinement.md (RMSE, ACC, CRPS and
spread/skill at selected leads, change vs the reference, GraphCast and
persistence for context) and rmse_change_vs_base.png.

    python scripts/report_refinement.py base=runs/test2020/base.json \
        supervised=runs/test2020/supervised.json rl=runs/test2020/rl.json \
        --out-dir runs/test2020/report
"""

import argparse
import json
import os

import _path  # noqa: F401  (adds src/ to sys.path)
import numpy as np

from climate_wm.eval.graphcast_metrics import HEADLINE_KEYS
from report_comparison import UNITS

COLORS = ["#8a8984", "#2a78d6", "#eb6834", "#1baf7a"]  # reference gray, then categorical slots


def load(path):
    with open(path) as f:
        payload = json.load(f)
    table = {}
    for name, rows in payload.items():
        if name == "ensemble":
            continue
        for r in rows:
            table[(name, r["variable"], r["lead_h"])] = r
    leads = sorted({k[2] for k in table})
    return table, payload.get("ensemble", {}), leads


def ens_at(ens, metric, key, leads, lead):
    vals = ens.get(metric, {}).get(key)
    return vals[leads.index(lead)] if vals is not None else np.nan


def pct(a, b):
    return f"{100 * (a - b) / b:+.1f}%"


def markdown(runs, leads, keys):
    names = list(runs)
    ref = names[0]
    lines = [f"Reference: **{ref}**. Change columns are (run − {ref}) / {ref}; negative is better "
             "for RMSE and CRPS.", ""]
    for key in keys:
        if (("model", key, leads[0]) not in runs[ref][0]):
            continue
        unit, scale = UNITS.get(key, ("", 1.0))
        head = ["lead"] + [f"RMSE {n}" for n in names] + [f"Δ {n}" for n in names[1:]] \
            + ["RMSE GraphCast", "RMSE persistence"] + [f"ACC {n}" for n in names] \
            + [f"CRPS {n}" for n in names] + [f"spread/skill {n}" for n in names]
        lines += [f"### {key}  (RMSE and CRPS in {unit})", "",
                  "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
        for lead in leads:
            r = {n: runs[n][0].get(("model", key, lead)) for n in names}
            if any(v is None for v in r.values()):
                continue
            table, ens, all_leads = runs[ref]
            gc = table.get(("graphcast", key, lead), {}).get("rmse", np.nan)
            pe = table.get(("persistence", key, lead), {}).get("rmse", np.nan)
            row = [f"{lead}h"] + [f"{r[n]['rmse'] * scale:.4g}" for n in names] \
                + [pct(r[n]["rmse"], r[ref]["rmse"]) for n in names[1:]] \
                + [f"{gc * scale:.4g}", f"{pe * scale:.4g}"] \
                + [f"{r[n]['acc']:.3f}" for n in names] \
                + [f"{ens_at(runs[n][1], 'crps', key, runs[n][2], lead) * scale:.4g}" for n in names] \
                + [f"{ens_at(runs[n][1], 'spread_skill', key, runs[n][2], lead):.2f}" for n in names]
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")
    # one line per run: mean change over every variable and lead
    lines += ["### All variables", "", "| run | mean RMSE change | share of (variable, lead) improved |"
              " mean CRPS change | median spread/skill |", "|---|---|---|---|---|"]
    table_ref, ens_ref, leads_ref = runs[ref]
    for n in names[1:]:
        table, ens, _ = runs[n]
        d = [(table[k]["rmse"] - v["rmse"]) / v["rmse"] for k, v in table_ref.items()
             if k[0] == "model" and k in table]
        c = [(np.asarray(ens["crps"][v]) - np.asarray(ens_ref["crps"][v])) / np.asarray(ens_ref["crps"][v])
             for v in ens_ref.get("crps", {}) if v in ens.get("crps", {})]
        ss = [x for v in ens.get("spread_skill", {}).values() for x in v]
        lines.append(f"| {n} | {100 * np.mean(d):+.2f}% | {np.mean(np.asarray(d) < 0):.0%} | "
                     f"{100 * np.mean(np.concatenate(c)) if c else np.nan:+.2f}% | "
                     f"{np.median(ss) if ss else np.nan:.2f} |")
    ss_ref = [x for v in ens_ref.get("spread_skill", {}).values() for x in v]
    lines.append(f"| {ref} | 0 | – | 0 | {np.median(ss_ref) if ss_ref else np.nan:.2f} |")
    return "\n".join(lines) + "\n"


def plot(runs, keys, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(runs)
    ref = names[0]
    keys = [k for k in keys if ("model", k, runs[ref][2][0]) in runs[ref][0]]
    cols = min(4, len(keys))
    rows = int(np.ceil(len(keys) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.6 * cols, 2.8 * rows), squeeze=False)
    for ax, key in zip(axes.flat, keys):
        leads = runs[ref][2]
        base = np.array([runs[ref][0][("model", key, l)]["rmse"] for l in leads])
        ax.axhline(0, color=COLORS[0], lw=1)
        for i, n in enumerate(names[1:], start=1):
            cur = np.array([runs[n][0][("model", key, l)]["rmse"] for l in leads])
            ax.plot(leads, 100 * (cur - base) / base, color=COLORS[i % len(COLORS)], lw=2, label=n)
        ax.set_title(key, fontsize=10)
        ax.set_xlabel("lead (h)", fontsize=9)
        ax.set_ylabel(f"RMSE change vs {ref} (%)", fontsize=9)
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=8)
    for ax in list(axes.flat)[len(keys):]:
        ax.axis("off")
    axes.flat[0].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=130)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+", metavar="NAME=compare_graphcast.json",
                        help="first is the reference")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--leads", type=int, nargs="*", default=[6, 24, 72, 120])
    args = parser.parse_args()
    runs = {}
    for item in args.runs:
        name, path = item.split("=", 1)
        runs[name] = load(path)
    out_dir = args.out_dir or os.path.join(os.path.dirname(args.runs[0].split("=", 1)[1]), "report")
    os.makedirs(out_dir, exist_ok=True)
    leads = [l for l in args.leads if l in runs[next(iter(runs))][2]]
    with open(os.path.join(out_dir, "refinement.md"), "w") as f:
        f.write(markdown(runs, leads, HEADLINE_KEYS))
    plot(runs, HEADLINE_KEYS, os.path.join(out_dir, "rmse_change_vs_base.png"))
    print(f"wrote {out_dir}/refinement.md and rmse_change_vs_base.png")


if __name__ == "__main__":
    main()
