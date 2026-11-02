"""Stage 5 report: arms G0 (stage 4 P) / G1 (+Vert·Fib) / G2 (+diagnostics) / G3 (+critic).

    python scripts/report_stage5.py G0=runs/pyramid_P/compare_graphcast.json \
        G1=runs/wb2_g1/compare_graphcast.json G2=... G3=... \
        --eval G0=runs/pyramid_P/stage5_eval.json G1=runs/wb2_g1/stage5_eval.json ... \
        [--eval-2020 G1=runs/wb2_g1/stage5_eval_2020.json ...] \
        [--compare-2020 G0=... G1=...] --out-dir docs/results/stage5_causal_lie

Joins the stage 1 protocol (compare_graphcast.json) with scripts/eval_stage5.py
and states the verdict on predictions P1-P5 of the design (doc section 7):

  P1  G1 lowers the 72 h RMSE of t2m, q700 and t850 by >= 3% against G0; z500 within 3%
  P2  G2's mixing is >= 2x stronger in unstable (N^2 < 0) columns, and the fibre
      activity spreads over >= 3 shells (no shell above 60%; stage 3 E2: 95% in one)
  P3  G3's critic-weighted ensemble has >= 5% lower 72 h CRPS than G2's equal
      weights, with spread/skill closer to 1
  P4  the TISR := night effect where the radiation generator is strong is >= 10x
      stage 3 E2's (3.7e-3) and concentrated there (>= 2x elsewhere)
  P5  on 2020 the coarse-rotation Karcher mean moves < 10% of its norm, and G1's
      72 h RMSE gain over G0 keeps >= half of its 2018 size
"""

import argparse
import json
import os

import _path  # noqa: F401  (adds src/ to sys.path)
import numpy as np

from report_pyramid import UNITS, at_lead, curves, parse_named

P1_VARS = ("2m_temperature", "specific_humidity@700", "temperature@850")
E2_CF_EFFECT = 3.7e-3  # stage 3 phase 2, latent effect where radiation codes were used
COLORS = ["#5b5b5b", "#2a78d6", "#1baf7a", "#d6702a"]


def pct(a, b):
    return 100.0 * (a - b) / b if b == b and b != 0 else float("nan")


def rmse72(arms, lead):
    return {n: {v: at_lead(curves(p).get("model", {}).get(v), lead, "rmse") for v in UNITS}
            for n, p in arms.items()}


def section_p1(arms, lead, base="G0"):
    r = rmse72(arms, lead)
    names = list(arms)
    lines = [f"## P1 — {lead} h RMSE against {base}", "",
             "| variable | " + " | ".join(names) + " | " + " | ".join(f"{n} vs {base}" for n in names if n != base) + " |",
             "|---|" + "---|" * (2 * len(names) - 1)]
    for v, (unit, scale) in UNITS.items():
        if base not in r or r[base].get(v) != r[base].get(v):
            continue
        row = [f"{v} ({unit})"] + [f"{r[n][v] * scale:.4g}" for n in names]
        row += [f"{pct(r[n][v], r[base][v]):+.2f}%" for n in names if n != base]
        lines.append("| " + " | ".join(row) + " |")
    verdict = None
    if "G1" in r and base in r:
        gains = [-pct(r["G1"][v], r[base][v]) for v in P1_VARS if r[base].get(v) == r[base].get(v)]
        z = pct(r["G1"].get("geopotential@500", np.nan), r[base].get("geopotential@500", np.nan))
        verdict = bool(gains and np.mean(gains) >= 3.0 and abs(z) <= 3.0)
        lines += ["", f"G1 mean gain on {', '.join(P1_VARS)}: **{np.mean(gains):.2f}%**, z500 change {z:+.2f}% "
                  f"→ P1 {'passes' if verdict else 'fails'}."]
    return lines, verdict, r


def section_usage(evals):
    lines = ["## Generator usage, knock-outs and the TISR counterfactual (one step, 2018 pairs)", ""]
    have = {n: e["usage"] for n, e in evals.items() if e.get("usage")}
    if not have:
        return lines + ["No composite arm evaluated."], None, None
    shells = next(iter(have.values()))["shells"]
    lines += ["| arm | fibre share per shell (" + " / ".join(shells) + ") | top shell | "
              "mixing unstable / stable | |shift| | knock-out loss +% (rad / mix / sfc / cond / shift) |",
              "|---|---|---|---|---|---|"]
    p2 = None
    for n, u in have.items():
        share = u.get("fibre_share_per_shell", [])
        mix = u["families"].get("mixing", {})
        ratio = mix.get("unstable_over_stable", float("nan"))
        shift = u["families"].get("shift", {}).get("mean", float("nan"))
        ko = u.get("knockout", {})
        kos = " / ".join(f"{ko[f]['loss_increase_pct']:+.2f}" if f in ko else "–"
                         for f in ("radiation", "mixing", "surface", "condensation", "shift"))
        lines.append(f"| {n} | {', '.join(f'{100 * s:.0f}%' for s in share)} | "
                     f"{100 * u.get('fibre_top_shell_share', float('nan')):.0f}% | {ratio:.2f} | {shift:.3g} | {kos} |")
        if n == "G2":
            spread = sum(s >= 0.1 for s in share) >= 3 and max(share, default=1) <= 0.6
            p2 = bool(ratio >= 2.0 and spread)
    lines += ["", f"Unstable columns: {100 * next(iter(have.values()))['unstable_fraction']:.1f}% of latent nodes "
              "(N² < 0 in one of the lowest three layers of the observed state).", ""]
    lines += ["| arm | TISR := night: effect where radiation strong / elsewhere | concentration | "
              "vs stage 3 E2 | decoded t2m change | Cartan spin / transport energy (coarse, all shells) |",
              "|---|---|---|---|---|---|"]
    p4 = None
    for n, u in have.items():
        cf = u.get("counterfactual", {})
        strong, other = cf.get("effect_where_radiation_strong", np.nan), cf.get("effect_elsewhere", np.nan)
        conc = cf.get("concentration", np.nan)
        cart = u.get("cartan", {})
        spin = float(np.sum(cart.get("spin_energy_per_shell", [np.nan])))
        trans = float(np.sum(cart.get("transport_energy_per_shell", [np.nan])))
        lines.append(f"| {n} | {strong:.3g} / {other:.3g} | {conc:.2f} | {strong / E2_CF_EFFECT:.1f}× | "
                     f"{cf.get('decoded_t2m_change', np.nan):+.2g} | {spin:.3g} / {trans:.3g} |")
        if strong == strong:
            ok = strong >= 10 * E2_CF_EFFECT and conc >= 2.0
            p4 = ok if p4 is None else (p4 or ok)
    lines += ["", "The latent effect is the node-mean squared change of the latent state; stage 3 measured "
              "the patch error of the replayed program, so the factor against E2 is indicative."]
    return lines, p2, p4


def section_p3(evals, lead_index=-1):
    lines = ["## P3 — critic-weighted ensemble", ""]
    rows = {n: e["ensemble"] for n, e in evals.items() if e.get("ensemble")}
    if not rows:
        return lines + ["No ensemble evaluated."], None
    any_e = next(iter(rows.values()))
    lead = any_e["lead_h"][lead_index]
    lines += [f"{any_e['members']} members, {any_e['inits']} initialisations, scores standardised over "
              f"members, tau {any_e['tau']}. Arms without a trained critic use one calibrated on training "
              "frames at evaluation time.", "",
              f"| arm | variable | CRPS {lead} h equal | weighted | change | spread/skill equal | weighted | "
              "critic–error rank corr. |", "|---|---|---|---|---|---|---|---|"]
    crps = {}
    for n, e in rows.items():
        for v, d in e["variables"].items():
            eq, wt = d["equal"]["crps"][lead_index], d["weighted"]["crps"][lead_index]
            crps[(n, v)] = (eq, wt, d["equal"]["spread_skill"][lead_index], d["weighted"]["spread_skill"][lead_index])
            lines.append(f"| {n} | {v} | {eq:.4g} | {wt:.4g} | {pct(wt, eq):+.2f}% | "
                         f"{crps[(n, v)][2]:.2f} | {crps[(n, v)][3]:.2f} | {e['critic_error_rank_corr']:.2f} |")
    verdict = None
    if "G3" in rows and "G2" in rows:
        vs = [v for (n, v) in crps if n == "G3" and ("G2", v) in crps]
        gains = [-pct(crps[("G3", v)][1], crps[("G2", v)][0]) for v in vs]
        closer = np.mean([abs(1 - crps[("G3", v)][3]) < abs(1 - crps[("G2", v)][2]) for v in vs])
        verdict = bool(gains and np.mean(gains) >= 5.0 and closer >= 0.5)
        lines += ["", f"G3 weighted vs G2 equal: mean CRPS change {-np.mean(gains):+.2f}%, spread/skill closer "
                  f"to 1 for {100 * closer:.0f}% of the variables → P3 {'passes' if verdict else 'fails'}."]
    return lines, verdict


def _geodesic(a, b):
    import torch
    from climate_wm.geometry.composite_group import geodesic_distance
    return geodesic_distance(torch.tensor(a), torch.tensor(b)).numpy()


def section_p5(evals, evals20, r18, r20, lead):
    lines = ["## P5 — 2020 invariance", ""]
    if not evals20:
        return lines + ["No 2020 evaluation."], None
    lines += ["| arm | Karcher mean shift / norm (coarse rotations) | log-cov change | coefficient RMS 2018 → 2020 |",
              "|---|---|---|---|"]
    ok_inv = []
    for n, e20 in evals20.items():
        u18, u20 = evals.get(n, {}).get("usage", {}), e20.get("usage", {})
        if not u18 or not u20:
            continue
        m18 = np.array(u18["omega0_group_stats"]["karcher_mean"])
        m20 = np.array(u20["omega0_group_stats"]["karcher_mean"])
        shift = float(np.mean(_geodesic(m18, m20)) / max(np.linalg.norm(m18, axis=-1).mean(), 1e-12))
        c18 = np.array(u18["omega0_group_stats"]["log_cov"])
        c20 = np.array(u20["omega0_group_stats"]["log_cov"])
        cov = float(np.linalg.norm(c20 - c18) / max(np.linalg.norm(c18), 1e-12))
        coefs = ", ".join(f"{k} {u18['coefficients'][k]['rms']:.3g} → {u20['coefficients'][k]['rms']:.3g}"
                          for k in u18.get("coefficients", {}) if k in u20.get("coefficients", {}))
        lines.append(f"| {n} | {100 * shift:.1f}% | {100 * cov:.1f}% | {coefs} |")
        ok_inv.append(shift < 0.1)
    verdict = None
    if r20 and "G1" in r20 and "G0" in r20 and "G1" in r18:
        g18 = np.mean([-pct(r18["G1"][v], r18["G0"][v]) for v in P1_VARS])
        g20 = np.mean([-pct(r20["G1"][v], r20["G0"][v]) for v in P1_VARS])
        keep = g20 >= 0.5 * g18 if g18 > 0 else False
        verdict = bool(keep and all(ok_inv))
        lines += ["", f"G1 gain over G0 on {', '.join(P1_VARS)} at {lead} h: 2018 {g18:.2f}%, 2020 {g20:.2f}% "
                  f"→ P5 {'passes' if verdict else 'fails'}."]
    return lines, verdict


def plot(r, evals, path, lead):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    names = list(r)
    base = names[0]
    vars_ = list(P1_VARS) + ["geopotential@500"]
    x = np.arange(len(vars_))
    width = 0.8 / max(len(names) - 1, 1)
    for i, n in enumerate(names[1:]):
        axes[0].bar(x + i * width, [pct(r[n][v], r[base][v]) for v in vars_], width,
                    color=COLORS[(i + 1) % len(COLORS)], label=n)
    axes[0].axhline(0, color="k", lw=0.8)
    axes[0].axhline(-3, color="k", lw=0.6, ls=":")
    axes[0].set_xticks(x + width * (len(names) - 2) / 2)
    axes[0].set_xticklabels([v.replace("specific_humidity", "q").replace("temperature", "t")
                             .replace("geopotential", "z").replace("2m_t", "t2m") for v in vars_])
    axes[0].set_ylabel(f"{lead} h RMSE change vs {base} (%)")
    axes[0].legend(frameon=False)
    have = {n: e["usage"] for n, e in evals.items() if e.get("usage")}
    for i, (n, u) in enumerate(have.items()):
        share = u.get("fibre_share_per_shell", [])
        axes[1].plot(range(len(share)), share, "o-", color=COLORS[(names.index(n) if n in names else i) % len(COLORS)],
                     label=n)
    if have:
        axes[1].set_xticks(range(len(next(iter(have.values()))["shells"])))
        axes[1].set_xticklabels(next(iter(have.values()))["shells"], rotation=20, fontsize=8)
    axes[1].axhline(0.95, color="#d6702a", lw=0.8, ls=":", label="stage 3 E2 (one shell)")
    axes[1].set_ylabel("share of fibre activity")
    axes[1].legend(frameon=False, fontsize=8)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=120)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("arms", nargs="+", metavar="NAME=compare_graphcast.json")
    parser.add_argument("--eval", nargs="*", default=[], metavar="NAME=stage5_eval.json")
    parser.add_argument("--eval-2020", nargs="*", default=[], metavar="NAME=stage5_eval_2020.json")
    parser.add_argument("--compare-2020", nargs="*", default=[], metavar="NAME=compare_graphcast_2020.json")
    parser.add_argument("--lead", type=int, default=72)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()
    arms = parse_named(args.arms)
    evals, evals20 = parse_named(args.eval), parse_named(args.eval_2020)
    arms20 = parse_named(args.compare_2020)
    p1_lines, p1, r18 = section_p1(arms, args.lead)
    use_lines, p2, p4 = section_usage(evals)
    p3_lines, p3 = section_p3(evals)
    r20 = rmse72(arms20, args.lead) if arms20 else None
    p5_lines, p5 = section_p5(evals, evals20, r18, r20, args.lead)
    mark = lambda v: "–" if v is None else ("pass" if v else "fail")
    if p1 and p2 and p4:
        overall = "success: the group extension works causally (P1, P2, P4)"
    elif p1:
        overall = "partial: skill improved but the generators are not shown to act physically"
    elif p1 is False and all(
            abs(u.get("knockout", {}).get(f, {}).get("loss_increase_pct", 0.0)) < 0.1
            for e in evals.values() for u in [e.get("usage") or {}] for f in ("radiation", "mixing", "surface")):
        overall = "failure: no skill gain and no knock-out response at this resolution / budget"
    else:
        overall = "mixed: see the predictions"
    text = ["# Stage 5 — reality-consistent causal Lie world model", "",
            "| prediction | P1 | P2 | P3 | P4 | P5 |", "|---|---|---|---|---|---|",
            f"| verdict | {mark(p1)} | {mark(p2)} | {mark(p3)} | {mark(p4)} | {mark(p5)} |", "",
            f"**Overall:** {overall}.", ""]
    text += p1_lines + [""] + use_lines + [""] + p3_lines + [""] + p5_lines
    md = "\n".join(text) + "\n"
    print(md)
    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)
        with open(os.path.join(args.out_dir, "stage5.md"), "w") as f:
            f.write(md)
        plot(r18, evals, os.path.join(args.out_dir, "stage5.png"), args.lead)
        print(f"wrote {args.out_dir}/stage5.md and stage5.png")


if __name__ == "__main__":
    main()
