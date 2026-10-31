"""Cause-analysis report: error decomposition and ablations across arms.

    python scripts/report_causes.py S=runs/independent/causes.json P=runs/wb2_pyramid/causes.json \
        G1=runs/wb2_g1/causes.json --out-dir docs/results/stage5_causal_lie/causes [--leads 6 24 72]

Inputs are scripts/eval_causes.py outputs. Writes causes.md and causes.png.
"""

import argparse
import os

import _path  # noqa: F401  (adds src/ to sys.path)
import numpy as np

from climate_wm.eval.graphcast_metrics import HEADLINE_KEYS
from report_pyramid import ARM_COLORS, UNITS, parse_named

VARIANT_LABEL = {"no_denoiser": "no denoiser (group action only)", "composite": "composite group off",
                 "shift": "vertical shift off", "radiation": "radiation off", "mixing": "mixing off",
                 "surface": "surface off", "condensation": "condensation off"}


def _value(res, name, lead, key):
    li, j = res["lead_h"].index(lead), res["variables"].index(key)
    return res["rmse"][name][li][j]


def decomposition_table(arms, leads, keys):
    names = list(arms)
    text = ["## 1. Error decomposition: total = autoencoder floor (rec) + latent dynamics (dyn)", "",
            "RMSE in physical units (scaled as in the stage 4 report). `rec share` = rec² / total², "
            "the part of the squared error that no transition could remove with this autoencoder.", "",
            "| variable | lead | " + " | ".join(f"{n} total / rec / dyn (rec share)" for n in names) + " |",
            "|---|---|" + "---|" * len(names)]
    for key in keys:
        unit, scale = UNITS.get(key, ("", 1.0))
        for lead in leads:
            cells = []
            for n in names:
                r = arms[n]
                if key not in r["variables"] or lead not in r["lead_h"]:
                    cells.append("–")
                    continue
                t, rc, d = (_value(r, k, lead, key) * scale for k in ("full", "rec", "dyn"))
                share = r["rec_share"][r["lead_h"].index(lead)][r["variables"].index(key)]
                cells.append(f"{t:.4g} / {rc:.4g} / {d:.4g} ({share:.2f})")
            label = f"{key} ({unit})" if unit else key
            text.append(f"| {label} | {lead}h | " + " | ".join(cells) + " |")
    text += ["", "Mean rec share over all variables:", "",
             "| arm | " + " | ".join(f"{l}h" for l in leads) + " |", "|---|" + "---|" * len(leads)]
    for n, r in arms.items():
        shares = [np.mean(r["rec_share"][r["lead_h"].index(l)]) if l in r["lead_h"] else np.nan for l in leads]
        text.append(f"| {n} | " + " | ".join(f"{s:.2f}" for s in shares) + " |")
    return text


def ablation_table(arms, leads, keys):
    text = ["", "## 2. Eval-time ablations (same states and noise, paired over initialisations)", "",
            "RMSE change of the ablated rollout against the full model, mean and 90% bootstrap interval. "
            "Positive = the part helps (removing it hurts).", "",
            "| arm | ablation | variable | " + " | ".join(f"{l}h" for l in leads) + " |",
            "|---|---|---|" + "---|" * len(leads)]
    for n, r in arms.items():
        for name, ch in r.get("variant_change", {}).items():
            for key in keys:
                if key not in r["variables"]:
                    continue
                j = r["variables"].index(key)
                cells = []
                for l in leads:
                    if l not in r["lead_h"]:
                        cells.append("–")
                        continue
                    li = r["lead_h"].index(l)
                    m, lo, hi = (100 * ch[k][li][j] for k in ("mean", "lo", "hi"))
                    sig = "**" if lo > 0 or hi < 0 else ""
                    cells.append(f"{sig}{m:+.1f}%{sig} [{lo:+.1f}, {hi:+.1f}]")
                text.append(f"| {n} | {VARIANT_LABEL.get(name, name)} | {key} | " + " | ".join(cells) + " |")
    return text


def figure(arms, path, keys=("geopotential@500", "2m_temperature", "specific_humidity@700")):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(keys), figsize=(5 * len(keys), 4))
    for ax, key in zip(np.atleast_1d(axes), keys):
        unit, scale = UNITS.get(key, ("", 1.0))
        for c, (n, r) in zip(ARM_COLORS * 3, arms.items()):
            if key not in r["variables"]:
                continue
            j = r["variables"].index(key)
            lead = r["lead_h"]
            ax.plot(lead, [x[j] * scale for x in r["rmse"]["full"]], color=c, label=f"{n} total")
            ax.plot(lead, [x[j] * scale for x in r["rmse"]["rec"]], color=c, ls="--", label=f"{n} rec floor")
        ax.set_title(key)
        ax.set_xlabel("lead (h)")
        ax.set_ylabel(f"RMSE ({unit})" if unit else "RMSE")
        ax.grid(alpha=0.3)
    np.atleast_1d(axes)[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("arms", nargs="+", help="NAME=causes.json")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--leads", type=int, nargs="*", default=[6, 24, 72])
    args = parser.parse_args()
    arms = parse_named(args.arms)
    keys = [k for k in HEADLINE_KEYS if any(k in r["variables"] for r in arms.values())]
    os.makedirs(args.out_dir, exist_ok=True)
    n_init = {n: r["inits"] for n, r in arms.items()}
    text = ["# Cause analysis: where the forecast error comes from", "",
            f"Arms {', '.join(arms)}; initialisations {n_init}; leads {args.leads} h.", ""]
    text += decomposition_table(arms, args.leads, keys)
    text += ablation_table(arms, args.leads, keys)
    md = "\n".join(text) + "\n"
    with open(os.path.join(args.out_dir, "causes.md"), "w") as f:
        f.write(md)
    figure(arms, os.path.join(args.out_dir, "causes.png"))
    print(md)
    print(f"saved {args.out_dir}/causes.md and causes.png")


if __name__ == "__main__":
    main()
