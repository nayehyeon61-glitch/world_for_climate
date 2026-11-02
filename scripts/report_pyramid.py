"""Stage 4 report: single-scale S vs pyramid P vs P-skip.

    python scripts/report_pyramid.py S=runs/independent/compare_graphcast.json \
        P=runs/wb2_pyramid/compare_graphcast.json "P-skip=runs/wb2_pyramid_noskip/compare_graphcast.json" \
        --pyramid P=runs/wb2_pyramid/pyramid_eval.json "P-skip=runs/wb2_pyramid_noskip/pyramid_eval.json" \
        --out-dir docs/results/stage4_pyramid [--leads 6 24 72 120]

Joins the stage 1 protocol (compare_graphcast.json: RMSE / ACC vs GraphCast,
CRPS, spread/skill) with the pyramid evaluation (energy spectrum, per-level
transfer) and states the verdict on the three predictions of the design.
"""

import argparse
import json
import os

import _path  # noqa: F401  (adds src/ to sys.path)
import numpy as np

from climate_wm.eval.graphcast_metrics import HEADLINE_KEYS

ARM_COLORS = ["#5b5b5b", "#2a78d6", "#9b59b6", "#1baf7a"]
REF_STYLE = {"graphcast": dict(color="#eb6834", ls="-", label="GraphCast (WB2)"),
             "persistence": dict(color="#8a8984", ls=":", label="persistence")}
UNITS = {"geopotential@500": ("m²/s²", 1.0), "temperature@850": ("K", 1.0),
         "specific_humidity@700": ("g/kg", 1e3), "u_component_of_wind@850": ("m/s", 1.0),
         "2m_temperature": ("K", 1.0), "mean_sea_level_pressure": ("hPa", 1e-2),
         "10m_u_component_of_wind": ("m/s", 1.0), "total_precipitation_6hr": ("mm", 1e3)}


def parse_named(items):
    out = {}
    for item in items or []:
        name, path = item.split("=", 1)
        with open(path) as f:
            out[name] = json.load(f)
    return out


def curves(payload):
    out = {}
    for name, rows in payload.items():
        if name == "ensemble":
            continue
        for r in rows:
            c = out.setdefault(name, {}).setdefault(r["variable"], {"lead": [], "rmse": [], "acc": []})
            c["lead"].append(r["lead_h"])
            c["rmse"].append(r["rmse"])
            c["acc"].append(r.get("acc", np.nan))
    return out


def at_lead(curve, lead, key):
    if curve is None or lead not in curve["lead"]:
        return np.nan
    return curve[key][curve["lead"].index(lead)]


def mean_over(curve, leads, key):
    vals = [at_lead(curve, l, key) for l in leads if l in curve["lead"]]
    return float(np.nanmean(vals)) if vals else np.nan


def scorecard(arms, leads):
    names = list(arms)
    base = names[0]
    text = ["## 1. GraphCast protocol: RMSE and ACC per arm (2018)", "",
            f"Skill is the RMSE change against **{base}** (negative = better). "
            "GraphCast (WB2 forecasts at 5.625°) is the reference column.", ""]
    head = "| variable | lead | GraphCast | " + " | ".join(f"{n} RMSE" for n in names) + " | " + \
        " | ".join(f"{n} vs {base}" for n in names[1:]) + " | " + " | ".join(f"{n} ACC" for n in names) + " |"
    text += [head, "|" + "---|" * (head.count("|") - 1)]
    for var in HEADLINE_KEYS:
        unit, scale = UNITS.get(var, ("", 1.0))
        for lead in leads:
            row = [f"{var} ({unit})", f"{lead}h"]
            gc = at_lead(arms[base].get("graphcast", {}).get(var), lead, "rmse") * scale
            row.append(f"{gc:.4g}" if gc == gc else "-")
            rm = {n: at_lead(arms[n]["model"].get(var), lead, "rmse") for n in names}
            row += [f"{rm[n] * scale:.4g}" if rm[n] == rm[n] else "-" for n in names]
            for n in names[1:]:
                s = (rm[n] - rm[base]) / rm[base] * 100 if rm[base] == rm[base] and rm[n] == rm[n] else np.nan
                row.append(f"{s:+.1f}%" if s == s else "-")
            row += [f"{at_lead(arms[n]['model'].get(var), lead, 'acc'):.3f}" for n in names]
            text.append("| " + " | ".join(row) + " |")
    return text


def long_lead_verdict(arms, var="geopotential@500", leads=(72, 96, 120)):
    names = list(arms)
    base = names[0]
    text = ["", f"### 72–120 h verdict (prediction 3, {var})", "",
            "| arm | mean RMSE 72–120h | vs " + base + " | mean ACC 72–120h | ACC change |", "|---|---|---|---|---|"]
    r0 = mean_over(arms[base]["model"][var], leads, "rmse")
    a0 = mean_over(arms[base]["model"][var], leads, "acc")
    verdict = {}
    for n in names:
        c = arms[n]["model"].get(var)
        r, a = mean_over(c, leads, "rmse"), mean_over(c, leads, "acc")
        skill = (r - r0) / r0 * 100
        verdict[n] = (skill, a - a0)
        text.append(f"| {n} | {r:.4g} | {skill:+.1f}% | {a:.3f} | {a - a0:+.3f} |")
    return text, verdict


def ensemble_table(arms, leads):
    names = [n for n in arms if arms[n].get("ensemble")]
    if not names:
        return []
    text = ["", "## 2. Ensemble: CRPS and spread/skill", "",
            "| variable | lead | " + " | ".join(f"{n} CRPS" for n in names) + " | " +
            " | ".join(f"{n} spread/skill" for n in names) + " |"]
    text.append("|" + "---|" * (text[-1].count("|") - 1))
    first = arms[names[0]]["ensemble"]
    for var in first.get("crps", {}):
        lead_list = arms[names[0]]["model"].get(var, {}).get("lead", [])
        for lead in leads:
            if lead not in lead_list:
                continue
            i = lead_list.index(lead)
            row = [var, f"{lead}h"]
            row += [f"{arms[n]['ensemble']['crps'][var][i]:.4g}" if var in arms[n]["ensemble"].get("crps", {}) else "-"
                    for n in names]
            row += [f"{arms[n]['ensemble']['spread_skill'][var][i]:.3f}"
                    if var in arms[n]["ensemble"].get("spread_skill", {}) else "-" for n in names]
            text.append("| " + " | ".join(row) + " |")
    return text


def pyramid_tables(pyr):
    text = []
    for n, res in pyr.items():
        en, tr = res.get("energy"), res.get("transfer", {})
        text += ["", f"## 3. so(3) pyramid: {n} (state levels {res.get('levels')}, {res.get('pairs')} pairs)", ""]
        if en:
            text += ["| level | patches | energy (rad²/step) | share | rms (°) | tropics | midlat | polar |",
                     "|---|---|---|---|---|---|---|---|"]
            for r in en["levels"]:
                text.append(f"| {r['level']} | {r['patches']} | {r['energy']:.3e} | {r['share']:.2f} | "
                            f"{r['rms_deg']:.2f} | {r['tropics']:.3e} | {r['midlat']:.3e} | {r['polar']:.3e} |")
            text += ["", f"Prediction 2 — energy falls with the level: **{en['monotone']}**; finest level "
                         f"midlat / tropics = **{en['fine_midlat_over_tropics']:.2f}** (expected > 1).", ""]
        if tr and tr.get("pairs"):
            text += ["| variant | latent MSE | decoded loss | gain (persistence→own = 1) | win vs persistence |",
                     "|---|---|---|---|---|"]
            for name in ("persistence", "own", "all", "random_all", "coarse", "random_coarse", "fine", "random_fine"):
                g = tr.get("gain_" + name, np.nan)
                w = tr.get("win_" + name, np.nan)
                text.append(f"| {name} | {tr['latent_' + name]:.4f} | {tr['loss_' + name]:.4f} | "
                            f"{g:.3f} | {w:.2f} |")
            text += ["", f"Explained fraction of the model's own rotations: {tr['explained_fraction_own']:.3f}. "
                         f"Matched-minus-random gain: all **{tr['all_gap']:+.3f}**, coarse **{tr['coarse_gap']:+.3f}** "
                         f"(expected ≥ 0.20), fine **{tr['fine_gap']:+.3f}** (expected < 0.10).", ""]
    return text


def figure(arms, pyr, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = list(arms)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))
    var = "geopotential@500"
    for key in ("rmse", "acc"):
        ax = axes[0, 0 if key == "rmse" else 1]
        for ref, st in REF_STYLE.items():
            c = arms[names[0]].get(ref, {}).get(var)
            if c and key == "rmse" or (c and ref == "graphcast"):
                ax.plot(c["lead"], c[key], color=st["color"], ls=st["ls"], label=st["label"], lw=1.6)
        for i, n in enumerate(names):
            c = arms[n]["model"].get(var)
            if c:
                ax.plot(c["lead"], c[key], color=ARM_COLORS[i % len(ARM_COLORS)], lw=2, label=n)
        ax.set_xlabel("lead (h)")
        ax.set_ylabel("z500 RMSE (m²/s²)" if key == "rmse" else "z500 ACC")
        ax.set_title("z500 RMSE vs lead" if key == "rmse" else "z500 ACC vs lead")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    ax = axes[1, 0]
    width = 0.8 / max(len(pyr), 1)
    for i, (n, res) in enumerate(pyr.items()):
        levels = res.get("energy", {}).get("levels", [])
        x = np.arange(len(levels)) + i * width
        ax.bar(x, [r["energy"] for r in levels], width, color=ARM_COLORS[(names.index(n) if n in names else i + 1) % 4], label=n)
    ax.set_yscale("log")
    ax.set_xticks(np.arange(3) + width * (len(pyr) - 1) / 2)
    ax.set_xticklabels(["ω(0) 12", "δ(1) 42", "δ(2) 162"][:3])
    ax.set_ylabel("mean |rotation|² per 6 h (rad²)")
    ax.set_title("rotation energy per pyramid level")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=8)
    ax = axes[1, 1]
    variants = ["all", "coarse", "fine"]
    x = np.arange(len(variants))
    for i, (n, res) in enumerate(pyr.items()):
        tr = res.get("transfer", {})
        if not tr.get("pairs"):
            continue
        col = ARM_COLORS[(names.index(n) if n in names else i + 1) % 4]
        ax.bar(x + i * width, [tr["gain_" + v] for v in variants], width * 0.5, color=col, label=f"{n} matched")
        ax.bar(x + i * width + width * 0.5, [tr["gain_random_" + v] for v in variants], width * 0.5,
               color=col, alpha=0.4, label=f"{n} random")
    ax.axhline(1.0, color="#8a8984", ls=":", lw=1)
    ax.set_xticks(x + width * (len(pyr) - 1) / 2)
    ax.set_xticklabels(["all levels", "coarse (ω0, δ1)", "fine (δ2)"])
    ax.set_ylabel("gain (persistence → own = 1)")
    ax.set_title("analog transfer per level: matched vs random")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("arms", nargs="+", help="NAME=compare_graphcast.json (first = baseline S)")
    parser.add_argument("--pyramid", nargs="*", default=[], help="NAME=pyramid_eval.json")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--leads", type=int, nargs="*", default=[6, 24, 72, 120])
    args = parser.parse_args()
    arms_raw = parse_named(args.arms)
    arms = {n: dict(curves(p), ensemble=p.get("ensemble")) for n, p in arms_raw.items()}
    pyr = parse_named(args.pyramid)
    os.makedirs(args.out_dir, exist_ok=True)

    text = ["# Stage 4: graph V-Net so(3) pyramid vs single-scale model", "",
            f"Arms: {', '.join(arms)} (baseline {list(arms)[0]}); leads {args.leads} h.", ""]
    text += scorecard(arms, args.leads)
    verdict_text, verdict = long_lead_verdict(arms)
    text += verdict_text
    text += ensemble_table(arms, args.leads)
    text += pyramid_tables(pyr)
    text += ["", "## 4. Verdict", ""]
    base = list(arms)[0]
    for n in list(arms)[1:]:
        skill, dacc = verdict[n]
        p3 = skill < 0 and dacc > 0
        line = f"- **{n}**: prediction 3 (better 72–120 h z500 than {base}): {'supported' if p3 else 'not supported'} " \
               f"(RMSE {skill:+.1f}%, ACC {dacc:+.3f})."
        if n in pyr:
            tr, en = pyr[n].get("transfer", {}), pyr[n].get("energy", {})
            if tr.get("pairs"):
                p1 = tr["coarse_gap"] >= 0.2 and tr["fine_gap"] < 0.1
                line += f" Prediction 1 (coarse transfers, fine does not): {'supported' if p1 else 'not supported'} " \
                        f"(coarse gap {tr['coarse_gap']:+.3f}, fine gap {tr['fine_gap']:+.3f})."
            if en:
                p2 = en["monotone"] and en["fine_midlat_over_tropics"] > 1
                line += f" Prediction 2 (spectrum): {'supported' if p2 else 'not supported'}."
        text.append(line)
    md = "\n".join(text) + "\n"
    with open(os.path.join(args.out_dir, "pyramid.md"), "w") as f:
        f.write(md)
    figure(arms, pyr, os.path.join(args.out_dir, "pyramid.png"))
    print(md)
    print(f"saved {args.out_dir}/pyramid.md and pyramid.png")


if __name__ == "__main__":
    main()
