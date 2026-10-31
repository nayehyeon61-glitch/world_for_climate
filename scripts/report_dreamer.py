"""Stage 10 report: the adaptive memory kernel and DreamerV3-style fluid training in the masked game.

    python scripts/report_dreamer.py --arm D=runs/stage10_D Dmarkov=runs/stage10_Dmarkov ... \
        --base runs/stage10_base/eval_2020.json --out-dir docs/results/stage10_memory

Each arm directory holds eval_2020.json (scripts/eval_dreamer.py), optionally
eval_2018_19.json (--split train) and train_history.json. All evaluations use
the same initialisations and seed, so the arms pair per initialisation.
Changes are relative to nudging on the stage 9 model (N1); negative = lower
error. Writes memory.md, memory.png and tau.png, with the verdicts on the
pre-registered predictions P1-P6 (docs/results/README_KO.md).
"""

import argparse
import json
import os

import numpy as np

LEAD_24 = 3  # lead index of 24 h (6 h steps)
KNOCK_LABEL = {"no_actor": "actor off", "no_memory": "memory off", "no_innovation": "innovation gain off",
               "no_readout": "readout off", "no_spread": "spreading off", "tau_fixed": "adaptive τ off"}
BASE_LABEL = {"nudging_half": "nudging, half strength", "game_rule": "game rule (restart, hidden at the mean)"}


def _load(path):
    if path and os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def _boot(rel, seed=0):
    """Mean and 95% bootstrap interval over initialisations of rel [N, K]."""
    idx = np.random.default_rng(seed).integers(0, len(rel), (2000, len(rel)))
    means = rel[idx].mean(1)
    return rel.mean(0), np.percentile(means, 2.5, 0), np.percentile(means, 97.5, 0)


def paired(a, b, ref):
    """(a - b) / mean(ref) per initialisation and lead, as mean / lo / hi."""
    a, b, ref = (np.asarray(x, dtype=float) for x in (a, b, ref))
    return _boot((a - b) / ref.mean(0))


def _cell(m, lo, hi, li):
    sig = "**" if lo[li] > 0 or hi[li] < 0 else ""
    return f"{sig}{100 * m[li]:+.1f}%{sig} [{100 * lo[li]:+.1f}, {100 * hi[li]:+.1f}]"


def _per(ev, kind, name, part="hidden"):
    return ev["kinds"][kind]["per_init"].get(f"{name}_{part}")


def training_table(arms):
    text = ["## 1. Training (2018–19 game years)", "",
            "Mean over the last 10 iterations. `hidden` = masked-point error of the world-model rollouts, "
            "`blow-ups` = rollout steps cut before a non-finite forecast (all iterations), `skipped` = model "
            "updates with a non-finite gradient, `imagined blow-ups` = actor episodes that blew up.", "",
            "| arm | iterations | hidden | all | blow-ups | skipped | imagined blow-ups | return scale | action rms |",
            "|---|---|---|---|---|---|---|---|---|"]
    for name, arm in arms.items():
        hist = arm["history"]
        if not hist:
            continue
        last = hist[-10:]

        def mean(key):
            v = [h[key] for h in last if key in h and np.isfinite(h[key])]
            return f"{np.mean(v):.4g}" if v else "–"

        total = lambda key: f"{sum(h.get(key, 0) for h in hist):.0f}"
        text.append(f"| {name} | {len(hist)} | {mean('hidden')} | {mean('all')} | {total('blowups')} | "
                    f"{total('skipped')} | {total('imagined_blowups')} | {mean('return_scale')} | "
                    f"{mean('action_rms')} |")
    return text + [""]


def headline_table(arms, base):
    ev0 = next(a["eval"] for a in arms.values() if a["eval"])
    lead = ev0["lead_h"]
    kinds = list(ev0["kinds"])
    text = ["## 2. Held-out game (2020)", "",
            f"{ev0['inits']} initialisations {ev0['period'][0][:10]} .. {ev0['period'][1][:10]}, one fixed mask per "
            f"initialisation and kind, re-observation chance {ev0['obs_prob']}. Hidden-point error change "
            "against nudging on N1 (direct insertion, the reference), mean and 95% paired bootstrap interval; "
            "bold: the interval excludes 0. Blow-ups: episodes with a non-finite forecast (scored at the cap "
            "from then on), out of the initialisations.", "",
            "| arm | " + " | ".join(f"{k} {lead[LEAD_24]} h | {k} {lead[-1]} h" for k in kinds)
            + " | blow-ups (arm / nudging) |",
            "|---|" + "---|---|" * len(kinds) + "---|"]
    for name, arm in arms.items():
        ev = arm["eval"]
        if ev is None:
            continue
        cells, blow = [], []
        for k in kinds:
            ch = ev["kinds"][k]["change"].get("full_hidden")
            if ch is None:
                cells += ["–", "–"]
            else:
                m, lo, hi = (np.asarray(ch[x]) for x in ("mean", "lo", "hi"))
                cells += [_cell(m, lo, hi, LEAD_24), _cell(m, lo, hi, -1)]
            b = ev["kinds"][k]["blowups"]
            blow.append(f"{b.get('full', 0)}/{b.get('nudging', 0)}")
        text.append(f"| {name} | " + " | ".join(cells) + f" | {', '.join(blow)} |")
    if base is not None:
        for v, label in BASE_LABEL.items():
            cells = []
            for k in kinds:
                ch = base["kinds"].get(k, {}).get("change", {}).get(f"{v}_hidden")
                if ch is None:
                    cells += ["–", "–"]
                    continue
                m, lo, hi = (np.asarray(ch[x]) for x in ("mean", "lo", "hi"))
                cells += [_cell(m, lo, hi, LEAD_24), _cell(m, lo, hi, -1)]
            blow = [f"{base['kinds'][k]['blowups'].get(v, 0)}" for k in kinds if k in base["kinds"]]
            text.append(f"| N1, {label} | " + " | ".join(cells) + f" | {', '.join(blow)} |")
    return text + [""]


def knockout_table(arms):
    text = ["## 3. Knock-outs (2020, change against the same arm in full)", "",
            "Hidden-point error of the knock-out minus the full arm, relative to nudging; positive = the "
            "part helps. 24 h / 36 h.", ""]
    for name, arm in arms.items():
        ev = arm["eval"]
        if ev is None:
            continue
        knocks = [v for v in KNOCK_LABEL if v in ev["variants"]]
        if not knocks:
            continue
        kinds = list(ev["kinds"])
        text += [f"### {name}", "", "| knock-out | " + " | ".join(kinds) + " |", "|---|" + "---|" * len(kinds)]
        for v in knocks:
            cells = []
            for k in kinds:
                a, b, ref = _per(ev, k, v), _per(ev, k, "full"), _per(ev, k, "nudging")
                m, lo, hi = paired(a, b, ref)
                cells.append(f"{_cell(m, lo, hi, LEAD_24)} / {100 * m[-1]:+.1f}%")
            text.append(f"| {KNOCK_LABEL[v]} | " + " | ".join(cells) + " |")
        text.append("")
    return text


def _mean_kinds(ev, fn):
    vals = [fn(k) for k in ev["kinds"]]
    return float(np.mean(vals)) if vals else float("nan")


def predictions(arms):
    """Verdicts on P1-P6 at 24 h (hidden points, mean over the mask kinds)."""
    get = lambda n: arms.get(n, {}).get("eval")
    out = []

    def change(ev, name, k):
        return ev["kinds"][k]["change"][f"{name}_hidden"]["mean"][LEAD_24]

    d = get("D")
    if d:
        p1 = _mean_kinds(d, lambda k: change(d, "full", k))
        out.append(("P1", "D ≥10% below nudging at 24 h", f"{100 * p1:+.1f}%", p1 <= -0.10))
    fx = get("Dfixed")
    if d and fx:
        rel = _mean_kinds(d, lambda k: paired(_per(d, k, "full"), _per(fx, k, "full"),
                                              _per(fx, k, "full"))[0][LEAD_24])
        out.append(("P2", "adaptive ≥3% below the fixed kernel", f"{100 * rel:+.1f}%", rel <= -0.03))
    mk = get("Dmarkov")
    if mk:
        p3 = _mean_kinds(mk, lambda k: change(mk, "full", k))
        tr = arms["Dmarkov"].get("eval_train")
        gap = (_mean_kinds(tr, lambda k: change(tr, "full", k)) if tr else float("nan"))
        ok = p3 <= -0.05 and (abs(p3 - gap) < 0.05 if np.isfinite(gap) else False)
        out.append(("P3", "D-markov ≥5% below nudging; 2018–19 vs 2020 gap <5 points",
                    f"{100 * p3:+.1f}% (2018–19 {100 * gap:+.1f}%)", ok))
    if d and "no_actor" in d["variants"]:
        gain = _mean_kinds(d, lambda k: change(d, "full", k) - change(d, "no_actor", k))
        out.append(("P4", "the actor adds <2%", f"{100 * gain:+.1f}%", gain > -0.02))
    if d and "no_memory" in d["variants"]:
        mem = {k: change(d, "no_memory", k) - change(d, "full", k) for k in d["kinds"]}
        best = max(mem, key=mem.get)
        text = ", ".join(f"{k} {100 * v:+.1f}" for k, v in mem.items())
        ok = best == "region"
        if "no_spread" in d["variants"] and "region" in d["kinds"]:
            lost = change(d, "no_spread", "region") - change(d, "full", "region")
            text += f"; spreading off loses {100 * lost:+.1f} of {100 * mem['region']:+.1f} (region)"
            ok = ok and mem["region"] > 0 and lost >= 0.5 * mem["region"]
        out.append(("P5", "memory gain largest under region masks; spreading carries ≥ half of it", text, ok))
    if d and "log10_tau_hours_channel_p10_p50_p90" in d:
        # state dependence: the spread of each channel's τ(x) over columns (the channels start log-spaced)
        spread = [10 ** (q[2] - q[0]) for q in d["log10_tau_hours_channel_p10_p50_p90"]]
        ratio = float(np.median(spread))
        out.append(("P6", "τ(x) p90/p10 per channel ≥ 2 (median over channels)",
                    f"{ratio:.2f} (max {max(spread):.2f})", ratio >= 2))
    text = ["## 4. Pre-registered predictions", "", "| | prediction | result | verdict |", "|---|---|---|---|"]
    for key, what, res, ok in out:
        text.append(f"| {key} | {what} | {res} | {'supported' if ok else 'not supported'} |")
    return text + [""], out


def overfit_table(arms):
    rows = [(n, a) for n, a in arms.items() if a.get("eval_train") and a.get("eval")]
    if not rows:
        return []
    text = ["## 5. Game years vs held-out year", "",
            "Hidden 24 h change against nudging on 2018–19 (15 initialisations of the training game) and on "
            "2020. A large gap means the memory or the actor fits the training years.", "",
            "| arm | mask | 2018–19 | 2020 |", "|---|---|---|---|"]
    for name, arm in rows:
        for k in arm["eval"]["kinds"]:
            a = arm["eval_train"]["kinds"].get(k, {}).get("change", {}).get("full_hidden")
            b = arm["eval"]["kinds"][k]["change"].get("full_hidden")
            if a and b:
                text.append(f"| {name} | {k} | {100 * a['mean'][LEAD_24]:+.1f}% | {100 * b['mean'][LEAD_24]:+.1f}% |")
    return text + [""]


def figures(arms, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    evs = {n: a["eval"] for n, a in arms.items() if a["eval"]}
    if not evs:
        return []
    kinds = list(next(iter(evs.values()))["kinds"])
    fig, axes = plt.subplots(1, len(kinds), figsize=(4 * len(kinds), 3.6), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, k in zip(axes, kinds):
        for n, ev in evs.items():
            ch = ev["kinds"][k]["change"].get("full_hidden")
            if ch:
                ax.plot(ev["lead_h"], 100 * np.asarray(ch["mean"]), marker="o", label=n)
        ax.axhline(0, color="0.5", lw=0.8)
        ax.set_title(f"{k} mask")
        ax.set_xlabel("lead (h)")
    axes[0].set_ylabel("hidden error vs nudging (%)")
    axes[-1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "memory.png"), dpi=130)
    plt.close(fig)
    made = ["memory.png"]
    taus = {n: ev["log10_tau_hours_zonal"] for n, ev in evs.items() if "log10_tau_hours_zonal" in ev}
    if taus:
        fig, ax = plt.subplots(figsize=(5, 3.6))
        for n, z in taus.items():
            lat = np.degrees(np.arcsin(np.polynomial.legendre.leggauss(len(z))[0]))[::-1]  # north to south
            ax.plot(lat, 10 ** np.asarray(z), label=n)
        ax.set_yscale("log")
        ax.set_xlabel("latitude")
        ax.set_ylabel("mean memory timescale τ (h)")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "tau.png"), dpi=130)
        plt.close(fig)
        made.append("tau.png")
    return made


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", nargs="+", required=True, metavar="NAME=DIR")
    parser.add_argument("--base", default=None, help="eval_dreamer.py on N1 (nudging_half, game_rule)")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    arms = {}
    for item in args.arm:
        name, d = item.split("=", 1)
        arms[name] = {"eval": _load(os.path.join(d, "eval_2020.json")),
                      "eval_train": _load(os.path.join(d, "eval_2018_19.json")),
                      "history": _load(os.path.join(d, "train_history.json")) or []}
    base = _load(args.base)
    os.makedirs(args.out_dir, exist_ok=True)
    text = ["# Stage 10 — adaptive memory kernel and DreamerV3-style fluid training", ""]
    text += training_table(arms) + headline_table(arms, base) + knockout_table(arms)
    verdict_text, _ = predictions(arms)
    text += verdict_text + overfit_table(arms)
    d = arms.get("D", {}).get("eval")
    if d and "log10_tau_hours_p10_p50_p90" in d:
        p = [10 ** x for x in d["log10_tau_hours_p10_p50_p90"]]
        text += ["## 6. Learned timescales (D)", "",
                 f"τ(x) over all channels, columns and updates of the 2020 game: p10 {p[0]:.1f} h, median "
                 f"{p[1]:.1f} h, p90 {p[2]:.1f} h (initial channels 3 h – 240 h).", "",
                 "| channel | τ p10 (h) | median | p90 | p90/p10 |", "|---|---|---|---|---|"]
        for c, q in enumerate(d.get("log10_tau_hours_channel_p10_p50_p90", [])):
            text.append(f"| {c} | {10 ** q[0]:.1f} | {10 ** q[1]:.1f} | {10 ** q[2]:.1f} | {10 ** (q[2] - q[0]):.2f} |")
        text.append("")
    for fig in figures(arms, args.out_dir):
        text += [f"![{fig}]({fig})", ""]
    with open(os.path.join(args.out_dir, "memory.md"), "w") as f:
        f.write("\n".join(text))
    print("\n".join(text))


if __name__ == "__main__":
    main()
