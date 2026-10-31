"""Stage 7 report: the masked prediction game and the unmasked forecast of controlled models.

    python scripts/report_game.py --game runs/game_eval.json \
        --forecast C1=runs/game_c1/causes_2020.json C1-grad=runs/game_c1_grad/causes_2020.json \
        --out-dir docs/results/stage7_prediction_game

Inputs: scripts/eval_game.py (masked game, all arms in one file) and
scripts/eval_causes.py --controller (unmasked 6-72 h forecast, one file per
arm, with the no_control / random_control / no_diag variants). Writes game.md
and game.png. Changes are relative to the frozen model (no control);
negative = the controller lowers the error.
"""

import argparse
import json
import os

import _path  # noqa: F401  (adds src/ to sys.path)
import numpy as np

from climate_wm.eval.graphcast_metrics import HEADLINE_KEYS
from report_pyramid import ARM_COLORS, parse_named

VARIANT_LABEL = {"control": "controller", "random_control": "same actions, shuffled places",
                 "no_diag": "controller without diagnostics"}


def _cell(ch, li):
    m, lo, hi = (100 * ch[k][li] for k in ("mean", "lo", "hi"))
    sig = "**" if lo > 0 or hi < 0 else ""
    return f"{sig}{m:+.1f}%{sig} [{lo:+.1f}, {hi:+.1f}]"


def game_table(game):
    text = ["## 1. Masked prediction game (held-out year)", "",
            f"{game['inits']} initialisations {game['period'][0][:10]} .. {game['period'][1][:10]}, one fixed mask "
            "per initialisation and kind; GraphCast-weighted MSE change against the frozen model (no control), "
            "mean and 90% bootstrap interval. `hidden` = the masked points, `all` = every point. "
            "Bold: the interval excludes 0.", ""]
    for name, arm in game["arms"].items():
        lead = arm["lead_h"]
        text += [f"### {name}", "",
                 "| mask | variant | " + " | ".join(f"hidden {l}h" for l in lead) + f" | all {lead[-1]}h |",
                 "|---|---|" + "---|" * (len(lead) + 1)]
        for kind, r in arm["kinds"].items():
            for v in ("control", "random_control", "no_diag"):
                h, a = r["change"].get(f"{v}_hidden"), r["change"].get(f"{v}_all")
                if h is None:
                    continue
                cells = [_cell(h, li) for li in range(len(lead))] + [_cell(a, len(lead) - 1)]
                text.append(f"| {kind} | {VARIANT_LABEL[v]} | " + " | ".join(cells) + " |")
        text.append("")
    return text


def forecast_table(arms, leads):
    if not arms:
        return []
    keys = [k for k in HEADLINE_KEYS if any(k in r["variables"] for r in arms.values())]
    text = ["## 2. Unmasked forecast (held-out year)", "",
            "RMSE change of the controlled model against the frozen model, i.e. -(no_control change), "
            "and of the two knock-outs against the controller. Negative in the first row = the controller helps.", "",
            "| arm | comparison | variable | " + " | ".join(f"{l}h" for l in leads) + " |",
            "|---|---|---|" + "---|" * len(leads)]
    for n, r in arms.items():
        ch = r.get("variant_change", {})
        for name, label, sign in (("no_control", "controller vs frozen", -1),
                                  ("random_control", "shuffled places vs controller", 1),
                                  ("no_diag", "no diagnostics vs controller", 1)):
            if name not in ch:
                continue
            for key in keys:
                j = r["variables"].index(key)
                cells = []
                for l in leads:
                    if l not in r["lead_h"]:
                        cells.append("–")
                        continue
                    li = r["lead_h"].index(l)
                    vals = [sign * 100 * ch[name][k][li][j] for k in ("mean", "lo", "hi")]
                    m, lo, hi = vals[0], min(vals[1:]), max(vals[1:])
                    sig = "**" if lo > 0 or hi < 0 else ""
                    cells.append(f"{sig}{m:+.1f}%{sig} [{lo:+.1f}, {hi:+.1f}]")
                text.append(f"| {n} | {label} | {key} | " + " | ".join(cells) + " |")
    return text


def figure(game, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    kinds = list(next(iter(game["arms"].values()))["kinds"])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    width = 0.8 / max(len(game["arms"]), 1)
    for i, (name, arm) in enumerate(game["arms"].items()):
        c = ARM_COLORS[(i + 1) % len(ARM_COLORS)]
        ch = [arm["kinds"][k]["change"]["control_hidden"] for k in kinds]
        x = np.arange(len(kinds)) + (i - (len(game["arms"]) - 1) / 2) * width
        m = np.array([100 * h["mean"][-1] for h in ch])
        err = np.array([[m_ - 100 * h["lo"][-1], 100 * h["hi"][-1] - m_] for m_, h in zip(m, ch)]).T
        axes[0].bar(x, m, width, yerr=err, color=c, label=name, capsize=3)
        hist = arm.get("history", [])
        pts = [(h["step"], 100 * h["gain_mean_policy"]) for h in hist if "gain_mean_policy" in h]
        if pts:
            axes[1].plot(*zip(*pts), color=c, marker="o", ms=3, label=name)
    axes[0].axhline(0, color="k", lw=0.8)
    axes[0].set_xticks(np.arange(len(kinds)), kinds)
    axes[0].set_ylabel("24 h hidden-point MSE change vs frozen (%)")
    axes[0].set_title("Held-out game score (negative = better)")
    axes[1].axhline(0, color="k", lw=0.8)
    axes[1].set_xlabel("controller update")
    axes[1].set_ylabel("training-episode return gain (%)")
    axes[1].set_title("Mean policy during training")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", nargs="+", required=True, help="scripts/eval_game.py outputs (arms merged)")
    parser.add_argument("--forecast", nargs="*", default=[], help="NAME=causes.json (eval_causes --controller)")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--leads", type=int, nargs="*", default=[6, 24, 72])
    args = parser.parse_args()
    game = None
    for path in args.game:
        with open(path) as f:
            part = json.load(f)
        if game is None:
            game = part
        else:
            game["arms"].update(part["arms"])
    forecast = parse_named(args.forecast)
    os.makedirs(args.out_dir, exist_ok=True)
    text = ["# Stage 7: prediction game and GNN feature control", "",
            f"Frozen world model `{game['checkpoint']}`; arms {', '.join(game['arms'])}.", ""]
    text += game_table(game)
    text += forecast_table(forecast, args.leads)
    md = "\n".join(text) + "\n"
    with open(os.path.join(args.out_dir, "game.md"), "w") as f:
        f.write(md)
    figure(game, os.path.join(args.out_dir, "game.png"))
    print(md)
    print(f"saved {args.out_dir}/game.md and game.png")


if __name__ == "__main__":
    main()
