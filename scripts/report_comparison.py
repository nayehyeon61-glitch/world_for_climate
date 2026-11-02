"""Markdown scorecard and RMSE / ACC plots from compare_graphcast.json.

    python scripts/report_comparison.py runs/wb2_64x32/compare_graphcast.json \
        [--out-dir runs/wb2_64x32/report] [--leads 6 24 72 120]
"""

import argparse
import json
import os

import _path  # noqa: F401  (adds src/ to sys.path)
import numpy as np

from climate_wm.eval.graphcast_metrics import HEADLINE_KEYS

LABELS = {
    "model": "ours (ensemble mean)",
    "model_member": "ours (single member)",
    "graphcast": "GraphCast (WB2)",
    "persistence": "persistence",
    "climatology": "climatology",
}
STYLE = {  # categorical slots 1-3 + a gray reference line
    "model": dict(color="#2a78d6", ls="-"),
    "model_member": dict(color="#2a78d6", ls="--"),
    "graphcast": dict(color="#eb6834", ls="-"),
    "persistence": dict(color="#1baf7a", ls="-"),
    "climatology": dict(color="#8a8984", ls=":"),
}
UNITS = {
    "geopotential@500": ("m²/s²", 1.0),
    "temperature@850": ("K", 1.0),
    "specific_humidity@700": ("g/kg", 1e3),
    "u_component_of_wind@850": ("m/s", 1.0),
    "2m_temperature": ("K", 1.0),
    "mean_sea_level_pressure": ("hPa", 1e-2),
    "10m_u_component_of_wind": ("m/s", 1.0),
    "total_precipitation_6hr": ("mm", 1e3),
}


def load(path):
    with open(path) as f:
        payload = json.load(f)
    curves = {}
    for name, rows in payload.items():
        if name == "ensemble":
            continue
        for r in rows:
            c = curves.setdefault(name, {}).setdefault(r["variable"], {"lead": [], "rmse": [], "acc": []})
            c["lead"].append(r["lead_h"])
            c["rmse"].append(r["rmse"])
            c["acc"].append(r.get("acc", np.nan))
    return curves, payload.get("ensemble")


def skill(a, b):
    return (a - b) / b


def markdown(curves, leads, keys):
    names = [n for n in LABELS if n in curves]
    lines = []
    for key in keys:
        if key not in curves["model"]:
            continue
        unit, scale = UNITS.get(key, ("", 1.0))
        lines += [f"### {key}  (RMSE in {unit})", "",
                  "| lead | " + " | ".join(f"RMSE {LABELS[n]}" for n in names)
                  + " | " + " | ".join(f"ACC {LABELS[n]}" for n in names if n != "climatology")
                  + " | RMSE skill vs GraphCast |",
                  "|" + "---|" * (2 * len(names) + 1)]
        base = curves["model"][key]
        for lead in leads:
            if lead not in base["lead"]:
                continue
            i = base["lead"].index(lead)
            rm = {n: curves[n][key]["rmse"][i] for n in names}
            ac = {n: curves[n][key]["acc"][i] for n in names}
            row = [f"{lead}h"] + [f"{rm[n] * scale:.4g}" for n in names]
            row += [f"{ac[n]:.3f}" for n in names if n != "climatology"]
            row.append(f"{skill(rm['model'], rm['graphcast']):+.2f}" if "graphcast" in rm else "")
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")
    return "\n".join(lines)


def plot(curves, metric, keys, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, muted, grid = "#0b0b0b", "#52514e", "#e4e3df"
    keys = [k for k in keys if k in curves["model"]]
    ncol = 4
    nrow = int(np.ceil(len(keys) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.2 * nrow), squeeze=False)
    fig.patch.set_facecolor("#fcfcfb")
    names = [n for n in LABELS if n in curves and not (metric == "acc" and n == "climatology")]
    for ax, key in zip(axes.flat, keys):
        unit, scale = UNITS.get(key, ("", 1.0))
        for n in names:
            c = curves[n][key]
            y = np.asarray(c[metric], float) * (scale if metric == "rmse" else 1.0)
            ax.plot(np.asarray(c["lead"]) / 24, y, lw=2, solid_capstyle="round",
                    label=LABELS[n], **STYLE[n])
        ax.set_facecolor("#fcfcfb")
        ax.set_title(key, fontsize=10, color=ink, loc="left")
        ax.set_xlabel("lead (days)", fontsize=8, color=muted)
        ax.set_ylabel(f"RMSE ({unit})" if metric == "rmse" else "ACC", fontsize=8, color=muted)
        ax.grid(color=grid, lw=1)
        ax.tick_params(colors=muted, labelsize=8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(grid)
    for ax in list(axes.flat)[len(keys):]:
        ax.axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), frameon=False,
               fontsize=9, labelcolor=ink)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=130, facecolor=fig.get_facecolor())
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--leads", type=int, nargs="*", default=[6, 24, 72, 120])
    args = parser.parse_args()
    out = args.out_dir or os.path.join(os.path.dirname(args.results), "report")
    os.makedirs(out, exist_ok=True)
    curves, ensemble = load(args.results)
    text = markdown(curves, args.leads, HEADLINE_KEYS)
    if ensemble:
        text += "\n### ensemble (ours)\n\n| variable | lead | CRPS | spread/skill |\n|---|---|---|---|\n"
        lead = curves["model"][HEADLINE_KEYS[0]]["lead"]
        for key in HEADLINE_KEYS:
            if key in ensemble.get("crps", {}):
                for lh in args.leads:
                    if lh in lead:
                        i = lead.index(lh)
                        text += (f"| {key} | {lh}h | {ensemble['crps'][key][i]:.4g} | "
                                 f"{ensemble['spread_skill'][key][i]:.3f} |\n")
    with open(os.path.join(out, "scorecard.md"), "w") as f:
        f.write(text)
    plot(curves, "rmse", HEADLINE_KEYS, os.path.join(out, "rmse_vs_lead.png"))
    plot(curves, "acc", HEADLINE_KEYS, os.path.join(out, "acc_vs_lead.png"))
    print(text)
    print(f"wrote {out}/scorecard.md, rmse_vs_lead.png, acc_vs_lead.png")


if __name__ == "__main__":
    main()
