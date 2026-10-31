"""Stage 8 report: transition-cost topology tests T1-T3 (scripts/eval_topology.py analyse).

    python scripts/report_topology.py --topology runs/topology/topology.json \
        --out-dir docs/results/stage8_topology

Writes topology.md and topology.png. Representations: raw_z (frozen latent),
energy (dry total energy norm) and the commute-time heads psi_*.
"""

import argparse
import json
import os

import _path  # noqa: F401  (adds src/ to sys.path)
import numpy as np

from report_pyramid import ARM_COLORS


SHORT = {"geopotential@500": "z500", "temperature@850": "t850", "2m_temperature": "t2m"}


def _fmt(ci, scale=1.0, pct=False):
    m, lo, hi = (scale * ci[k] for k in ("mean", "lo", "hi"))
    sig = "**" if lo > 0 or hi < 0 else ""
    if pct:
        return f"{sig}{m:+.1f}%{sig} [{lo:+.1f}, {hi:+.1f}]"
    return f"{sig}{m:+.2f}{sig} [{lo:+.2f}, {hi:+.2f}]"


def tables(res):
    arms = res["arms"]
    rows = res["rows"]
    text = [f"Heads trained on {rows['train']} states ({rows['train_pairs']} 6 h transitions); tests on "
            f"{rows['test']} held-out states. Markov state model: {res['msm']['latent']['clusters']} k-means "
            "clusters of all years' transitions (two partitions: latent and energy principal components). "
            "Intervals: 90% bootstrap (clusters for T1, 7-day blocks for T3). Bold: excludes 0.", "",
            "## T1. Does distance reflect transition difficulty?", "",
            "Spearman between the distance of cluster centroids and the commute time; differences to raw_z "
            "and energy (paired). `lag rho`: Spearman of pair distance against time lag (1-40 steps).", "",
            "| representation | dim | rho, latent partition | − raw_z | − energy | rho, energy partition | − raw_z | − energy | lag rho |",
            "|---|---|---|---|---|---|---|---|---|"]
    for name, r in arms.items():
        t1 = r["t1"]
        cells = []
        for part in ("latent", "energy"):
            cells.append(_fmt(t1[part]))
            cells += [_fmt(t1[f"{part}_minus_{ref}"]) if f"{part}_minus_{ref}" in t1 else "–" for ref in ("raw_z", "energy")]
        text.append(f"| {name} | {r['dim']} | " + " | ".join(cells) + f" | {r['lag']['spearman_lag']:+.2f} |")
    text += ["", "## T2. New geometry or new topology?", "",
             "Geometry: slope of log variance of the principal directions against log Laplacian eigenvalue of the "
             "Markov state model (CTWM predicts −1). Topology, in the top principal directions of the held-out "
             "year: H0 components that outlive a Gaussian cloud of the same covariance (`excess`), the longest "
             "H0 deaths relative to the median, and the mean residence (steps) in 8 k-means regimes.", "",
             "| representation | spectrum slope | H0 excess | longest H0 / median (null) | regime residence | H1 longest |",
             "|---|---|---|---|---|---|"]
    for name, r in arms.items():
        t2 = r["t2"]
        h0 = t2["h0"]
        h1 = t2.get("h1")
        h1s = f"{h1['top'][0]:.2f}" if h1 and h1["top"] else "–"
        text.append(f"| {name} | {t2['spectrum']['slope']:+.2f} | {h0['excess']} | {h0['top'][0]:.1f} "
                    f"({h0['null_top'][0]:.1f}) | {t2['residence']['mean_run']:.1f} | {h1s} |")
    if res.get("regional"):
        text += ["", "Regional patches (shell features of one patch; psi_patch = the regional head):", "",
                 "| region | representation | H0 excess | regime residence |", "|---|---|---|---|"]
        for region, reg in res["regional"].items():
            for name, v in reg.items():
                if name == "patch":
                    continue
                text.append(f"| {region} (patch {reg['patch']}) | {name} | {v['h0']['excess']} | "
                            f"{v['residence']['mean_run']:.1f} |")
    leads, keys = res["leads_h"], res["variables"]
    text += ["", "## T3. Do cost neighbourhoods predict better?", "",
             f"Analog forecasts: mean of the successors of the {res['config']['topology']['analogs']} nearest "
             "library states (2015-17). RMSE change against raw_z neighbours; regimes from the z500 field at the "
             f"initial time ({', '.join(f'{k} {v}' for k, v in res['regime_counts'].items())} states).", "",
             "| representation | subset | " + " | ".join(f"{SHORT.get(k, k)} {l}h" for l in leads for k in keys) + " |",
             "|---|---|" + "---|" * (len(leads) * len(keys))]
    for name, r in arms.items():
        for sub, ch in r["t3"].get("vs_raw_z", {}).items():
            ci = [{k: ch[k][i] for k in ("mean", "lo", "hi")} for i in range(len(leads) * len(keys))]
            text.append(f"| {name} | {sub} | " + " | ".join(_fmt(c, 100, pct=True) for c in ci) + " |")
    text += ["", "Absolute RMSE (all test states):", "",
             "| representation | " + " | ".join(f"{SHORT.get(k, k)} {l}h" for l in leads for k in keys) + " |",
             "|---|" + "---|" * (len(leads) * len(keys))]
    for name, r in arms.items():
        vals = np.asarray(r["t3"]["rmse"]).reshape(-1)
        text.append(f"| {name} | " + " | ".join(f"{v:.3g}" for v in vals) + " |")
    pred = [(n, r["predictability"]) for n, r in arms.items() if "predictability" in r]
    if pred:
        text += ["", "World-model predictability: Spearman of local speed and of rarity (distance to the nearest "
                 f"library states) with the frozen model's own error ({pred[0][1]['inits']} initial times).", "",
                 "| representation | speed vs error | rarity vs error |", "|---|---|---|"]
        for n, p in pred:
            text.append(f"| {n} | {p['speed_vs_error']:+.2f} | {p['rarity_vs_error']:+.2f} |")
    return text


def figure(res, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    names = list(res["arms"])
    for i, name in enumerate(names):
        r = res["arms"][name]
        c = ARM_COLORS[i % len(ARM_COLORS)]
        ls = "--" if name in ("raw_z", "energy") else "-"
        axes[0].plot(r["lag"]["lags"], r["lag"]["relative_distance"], ls, color=c, marker="o", ms=3, label=name)
        sp = r["t2"]["spectrum"]
        axes[1].loglog(sp["lambda"], np.asarray(sp["variance"]) / sp["variance"][0], ls, color=c, marker="o", ms=3,
                       label=f"{name} ({sp['slope']:+.2f})")
    lam = np.asarray(res["arms"][names[0]]["t2"]["spectrum"]["lambda"])
    axes[1].loglog(lam, lam[0] / lam, "k:", lw=1, label="1/λ")
    ref = [n for n in names if "vs_raw_z" in res["arms"][n]["t3"]]
    if ref:
        li = res["leads_h"].index(72) if 72 in res["leads_h"] else len(res["leads_h"]) // 2
        j = li * len(res["variables"])  # z500 at that lead
        x = np.arange(len(ref))
        ch = [res["arms"][n]["t3"]["vs_raw_z"]["all"] for n in ref]
        m = np.array([100 * c["mean"][j] for c in ch])
        err = np.array([[m_ - 100 * c["lo"][j], 100 * c["hi"][j] - m_] for m_, c in zip(m, ch)]).T
        axes[2].bar(x, m, yerr=err, color=[ARM_COLORS[names.index(n) % len(ARM_COLORS)] for n in ref], capsize=3)
        axes[2].set_xticks(x, ref, rotation=45, ha="right", fontsize=8)
        axes[2].axhline(0, color="k", lw=0.8)
        axes[2].set_ylabel(f"analog z500 {res['leads_h'][li]} h RMSE change vs raw_z (%)")
    axes[0].set_xlabel("lag (6 h steps)")
    axes[0].set_ylabel("mean distance / random-pair distance")
    axes[0].set_title("T1: distance grows with lag")
    axes[1].set_xlabel("Laplacian eigenvalue λk")
    axes[1].set_ylabel("variance of direction k (relative)")
    axes[1].set_title("T2: CTWM predicts variance ∝ 1/λ")
    axes[2].set_title("T3: analog forecasts (negative = better)")
    for ax in axes[:2]:
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topology", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    with open(args.topology) as f:
        res = json.load(f)
    os.makedirs(args.out_dir, exist_ok=True)
    text = ["# Stage 8: transition-cost topology (phase 1, frozen latents)", ""] + tables(res)
    md = "\n".join(text) + "\n"
    with open(os.path.join(args.out_dir, "topology.md"), "w") as f:
        f.write(md)
    figure(res, os.path.join(args.out_dir, "topology.png"))
    print(md)
    print(f"saved {args.out_dir}/topology.md and topology.png")


if __name__ == "__main__":
    main()
