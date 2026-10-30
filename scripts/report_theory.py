"""Stage 3 report: arms of the theory experiment side by side.

    python scripts/report_theory.py C=runs/theory_lie/transfer.json A=runs/theory_cont/transfer.json \
        [E=... D=...] [--invariance C=runs/theory_lie/transfer_2020.json ...] [--out-dir DIR]

Writes theory.md (self-explainability, analog transfer, length generalisation,
physics usage and intervention tests, cross-year invariance) and theory.png.
"""

import argparse
import json
import os

import _path  # noqa: F401  (adds src/ to sys.path)
import numpy as np

COLORS = ["#2a78d6", "#8a8a8a", "#d6702a", "#3aa655", "#8e5cc4"]


def load(items):
    runs = {}
    for item in items:
        name, path = item.split("=", 1)
        with open(path) as f:
            runs[name] = json.load(f)
    return runs


def pct(x):
    return f"{100 * x:.0f}%"


def tv_distance(p, q):
    p, q = np.asarray(p, float), np.asarray(q, float)
    n = max(len(p), len(q))
    p, q = np.pad(p, (0, n - len(p))), np.pad(q, (0, n - len(q)))
    return 0.5 * float(np.abs(p / max(p.sum(), 1e-12) - q / max(q.sum(), 1e-12)).sum())


def markdown(runs, inv):
    names = list(runs)
    L = ["## Self-explainability (6 h pairs)", "",
         "| arm | Lie codes | physics codes | loss: persistence | MDL-selected | full K | floor | "
         "explained fraction | mean k* | codes used / perplexity | physics share |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for n in names:
        r, se = runs[n], runs[n]["self_explainability"]
        codes = f"{se['codes_used']} / {se['perplexity']:.1f}" if "codes_used" in se else "continuous"
        phys = pct(se["phys_share"]) if "phys_share" in se else "–"
        L.append(f"| {n} ({r['experiment']}) | {r['codebook_size'] or 'cont.'} | {r.get('physics_codes', 0)} | "
                 f"{se['loss_persistence']:.4f} | {se['loss_selected']:.4f} | {se['loss_by_length'][-1]:.4f} | "
                 f"{se['loss_floor']:.4f} | {se['explained_fraction']:.3f} | {se['mean_length']:.2f} | {codes} | {phys} |")
    L += ["", "## Analog transfer (program of pair i executed on its nearest state j ≥ 7 days away)", "",
          "| arm | persistence | delta copy | random program | **transferred** | oracle | "
          "transfer gain | delta gain | transfer beats delta |",
          "|---|---|---|---|---|---|---|---|---|"]
    for n in names:
        t = runs[n].get("analog_transfer", {})
        if not t.get("pairs"):
            continue
        L.append(f"| {n} | {t['loss_persistence']:.4f} | {t['loss_delta']:.4f} | {t['loss_random']:.4f} | "
                 f"**{t['loss_transfer']:.4f}** | {t['loss_oracle']:.4f} | {t['transfer_gain']:.3f} | "
                 f"{t['delta_gain']:.3f} | {pct(t['win_vs_delta'])} |")
    L += ["", "## Length generalisation (12 h pairs, trained on 6 h)", "",
          "| arm | max steps | persistence | selected | full | floor | mean k* | explained fraction |",
          "|---|---|---|---|---|---|---|---|"]
    for n in names:
        ood = runs[n].get("length_ood", {})
        for k, v in ood.get("by_steps", {}).items():
            L.append(f"| {n} | {k} | {v['loss_persistence']:.4f} | {v['loss_selected']:.4f} | {v['loss_full']:.4f} | "
                     f"{v['loss_floor']:.4f} | {v['mean_length']:.2f} | {v['explained_fraction']:.3f} |")
    phys_arms = [n for n in names if "phys_frac" in runs[n]["self_explainability"]]
    if phys_arms:
        L += ["", "## Physics codes: usage and intervention tests", "",
              "| arm | physics share of steps / of error reduction | usage per shell (surface → top) | "
              "usage tropics / extratropics | Lie usage tropics / extratropics |", "|---|---|---|---|---|"]
        for n in phys_arms:
            se = runs[n]["self_explainability"]
            pu, lu = se["phys_usage"], se.get("lie_usage", {})
            L.append(f"| {n} | {pct(se['phys_frac'])} / {pct(se['phys_share'])} | "
                     + ", ".join(pct(v) for v in pu["per_shell"]) + " | "
                     + (f"{pct(pu['tropics'])} / {pct(pu['extratropics'])}" if "tropics" in pu else "–") + " | "
                     + (f"{pct(lu['tropics'])} / {pct(lu['extratropics'])}" if "tropics" in lu else "–") + " |")
        L += ["", "| arm | TISR := night: latent effect where radiation / mixing / surface / no code was used | "
              "decoded 2 m temperature change | knock-out loss increase: radiation / mixing / surface |",
              "|---|---|---|---|"]
        for n in phys_arms:
            ca = runs[n].get("causal")
            if not ca:
                continue
            cf, ko = ca["counterfactual"], ca["knockout"]
            w = cf["effect_where_family_used"]
            eff = " / ".join(f"{w.get(k, float('nan')):.1e}" for k in ("radiation", "mixing", "surface", "none"))
            t2m = cf["decoded_channel_change"].get("2m_temperature", float("nan"))
            kos = " / ".join(f"{ko[k]['loss_increase_pct']:+.2f}%" for k in ("radiation", "mixing", "surface"))
            L.append(f"| {n} | {eff} | {t2m:+.4f} | {kos} |")
    if inv:
        L += ["", "## Cross-year invariance (2015–17 vocabulary applied to 2020 pairs)", "",
              "| arm | TV distance of Lie code usage (2018 vs 2020) | TV of physics usage per shell | "
              "mean k* 2018 → 2020 | explained fraction 2018 → 2020 | transfer gain 2018 → 2020 |",
              "|---|---|---|---|---|---|"]
        for n, r20 in inv.items():
            if n not in runs:
                continue
            a, b = runs[n]["self_explainability"], r20["self_explainability"]
            tv = f"{tv_distance(a['code_usage'], b['code_usage']):.3f}" if "code_usage" in a and "code_usage" in b else "–"
            tvp = "–"
            if "phys_usage" in a and "phys_usage" in b:
                tvp = f"{tv_distance(a['phys_usage']['per_shell'], b['phys_usage']['per_shell']):.3f}"
            ta, tb = runs[n].get("analog_transfer", {}), r20.get("analog_transfer", {})
            tg = (f"{ta.get('transfer_gain', float('nan')):.3f} → {tb.get('transfer_gain', float('nan')):.3f}")
            L.append(f"| {n} | {tv} | {tvp} | {a['mean_length']:.2f} → {b['mean_length']:.2f} | "
                     f"{a['explained_fraction']:.3f} → {b['explained_fraction']:.3f} | {tg} |")
    return "\n".join(L)


def plot(runs, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(runs)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    x = np.arange(len(names))
    ef = [runs[n]["self_explainability"]["explained_fraction"] for n in names]
    axes[0].bar(x, ef, color=COLORS[:len(names)])
    axes[0].set_title("explained fraction of the 6 h change", fontsize=10)
    tg = [runs[n].get("analog_transfer", {}).get("transfer_gain", np.nan) for n in names]
    dg = [runs[n].get("analog_transfer", {}).get("delta_gain", np.nan) for n in names]
    axes[1].bar(x - 0.2, tg, 0.4, color=COLORS[0], label="transferred program")
    axes[1].bar(x + 0.2, dg, 0.4, color=COLORS[1], label="delta copy")
    axes[1].axhline(0, color="k", lw=0.5)
    axes[1].set_title("analog transfer gain (1 = oracle)", fontsize=10)
    axes[1].legend(fontsize=8, frameon=False)
    ml = [runs[n]["self_explainability"]["mean_length"] for n in names]
    axes[2].bar(x, ml, color=COLORS[:len(names)])
    axes[2].set_title("mean program length k*", fontsize=10)
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(names)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+", metavar="NAME=transfer.json")
    parser.add_argument("--invariance", nargs="*", default=[], metavar="NAME=transfer_2020.json",
                        help="the same arms evaluated on another year")
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()
    runs = load(args.runs)
    inv = load(args.invariance)
    out_dir = args.out_dir or os.path.dirname(args.runs[0].split("=", 1)[1])
    os.makedirs(out_dir, exist_ok=True)
    md = markdown(runs, inv)
    with open(os.path.join(out_dir, "theory.md"), "w") as f:
        f.write(md + "\n")
    plot(runs, os.path.join(out_dir, "theory.png"))
    print(md)
    print(f"\nwrote {out_dir}/theory.md and theory.png")


if __name__ == "__main__":
    main()
