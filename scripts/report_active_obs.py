"""Stage 12 report: the observation game, the climate frame, the selection maps and the rules A1-A6.

    python scripts/report_active_obs.py --game runs/stage12_P/game_2020.json \
        [--game-train runs/stage12_P/game_2018_19.json] [--climate runs/stage12_P/climate_2020.json] \
        [--nomem runs/stage12_Pnomem/game_2020.json] [--history runs/stage12_P/train_history.json] \
        --out docs/results/stage12_active_obs/active.md
"""

import argparse
import json
import os
from typing import Dict, List, Optional

import numpy as np

SHOW_LEADS = (6, 24, 48, 72, 120)
HEAD = {"geopotential@500": "z500", "temperature@850": "t850", "2m_temperature": "t2m",
        "10m_u_component_of_wind": "u10"}


def load(path: Optional[str]):
    if not path or not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def _pct(x: float) -> str:
    return "-" if x is None or not np.isfinite(x) else f"{100 * x:+.1f}%"


def _ci(ch: dict) -> str:
    """Gain (+ = better) with its interval: the negative of the relative error change."""
    return f"{_pct(-ch['mean'][0])} [{_pct(-ch['hi'][0])}, {_pct(-ch['lo'][0])}]"


def free_mean(run: dict, obs_steps: int) -> float:
    return float(np.mean(run["mse"][obs_steps:]))


def spearman(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    ra, rb = a.argsort().argsort(), b.argsort().argsort()
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def grid_table(values: List[float], patches: dict, fmt="{:.2f}") -> List[str]:
    """A bands x sectors table (north at the top)."""
    bands, sectors = patches["bands"], patches["sectors"]
    lat = np.array(patches["lat_centre"]).reshape(bands, sectors)[:, 0]
    lon = np.array(patches["lon_centre"]).reshape(bands, sectors)[0]
    v = np.array(values, float).reshape(bands, sectors)
    lines = ["| band \\ lon | " + " | ".join(f"{x:.0f}E" for x in lon) + " |", "|---" * (sectors + 1) + "|"]
    for i in reversed(range(bands)):
        lines.append(f"| {lat[i]:+.0f} | " + " | ".join(fmt.format(x) for x in v[i]) + " |")
    return lines


def game_section(g: dict, title: str) -> List[str]:
    obs, leads = g["obs_steps"], g["lead_h"]
    cols = [h for h in SHOW_LEADS if h in leads]
    idx = [leads.index(h) for h in cols]
    lines = [f"## {title}", "",
             f"{len(g['inits'])} initialisations ({g.get('period', ['', ''])[0][:10]} .. {g.get('period', ['', ''])[1][:10]}), "
             f"{leads[-1]} h episodes: observations chosen after the first {obs} forecasts ({6 * obs} h), then free. "
             f"GraphCast-weighted all-point MSE (normalised units) of the forecast before each re-observation; "
             f"'free' = mean over the free leads ({6 * (obs + 1)}-{leads[-1]} h); gains (+ = better) against {g['reference']} "
             f"(paired, bootstrap 95% over initialisations). Blow-ups are counted and scored 10 from then on.", "",
             "| run | budget | " + " | ".join(f"{h} h" for h in cols) + " | free | gain vs random (+ = better) | blow-ups | selection entropy (nat) | rho(freq, log tau) |",
             "|---" * (len(cols) + 7) + "|"]
    for name, r in g["runs"].items():
        ch = g["change"].get(name)
        rho = g["spearman_tau"].get(name, float("nan"))
        lines.append(f"| {name} | {r['budget']:g} | " + " | ".join(f"{r['mse'][i]:.3f}" for i in idx)
                     + f" | {free_mean(r, obs):.3f} | {_ci(ch['free']) if ch else 'reference'} | {r['blowups']} | "
                     + (f"{r['freq_entropy']:.2f}" if np.isfinite(r['freq_entropy']) else "-")
                     + f" | {rho:+.2f} |" if np.isfinite(rho) else f" | - |")
    lines += ["", "Headline RMSE (physical units) at 24 / 72 / 120 h:", "",
              "| run | " + " | ".join(HEAD[k] for k in g["keys"] if k in HEAD) + " |",
              "|---" * (1 + sum(k in HEAD for k in g["keys"])) + "|"]
    at = [leads.index(h) for h in (24, 72, 120) if h in leads]
    for name, r in g["runs"].items():
        cells = []
        for k in g["keys"]:
            if k in HEAD:
                cells.append(" / ".join(f"{r['rmse'][k][i]:.3g}" for i in at))
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return lines


def budget_section(g: dict) -> List[str]:
    obs = g["obs_steps"]
    lines = ["## Cause ablation: the observation budget", "",
             "Free-lead MSE of the policy against the random rule at the same budget (memory model; gain + = better):", "",
             "| budget | random | policy | policy gain vs random | age | readout | tendency |", "|---|---|---|---|---|---|---|"]
    for b in g["budgets"]:
        suf = "" if b == g["budgets"][0] else f"@{b:g}"
        get = lambda rule: g["runs"].get(f"memory:{rule}{suf}")
        rnd, pol = get("random"), get("policy")
        if rnd is None or pol is None:
            continue
        fr, fp = free_mean(rnd, obs), free_mean(pol, obs)
        others = " | ".join(f"{free_mean(get(r), obs):.3f}" if get(r) else "-" for r in ("age", "readout", "tendency"))
        lines.append(f"| {b:g} | {fr:.3f} | {fp:.3f} | {_pct((fr - fp) / fr)} | {others} |")
    return lines


def maps_section(g: dict) -> List[str]:
    pt = g["patches"]
    lines = ["## Where the rules look", "",
             "Selection frequency per patch (fraction of observation times; north at the top, longitude of the sector's west edge + 30):", ""]
    for name in ("memory:policy", "memory:age", "memory:readout", "memory:tendency"):
        r = g["runs"].get(name)
        if r is None or not any(r["freq"]):
            continue
        lines += [f"**{name}**", ""] + grid_table(r["freq"], pt) + [""]
    if "tau_patch" in g:
        lines += ["**The memory's timescale, log10 tau (hours), patch mean under full observation**", ""] \
            + grid_table(g["tau_patch"], pt) + ["", "**The memory's readout |R(m)| (normalised units) under full observation**", ""] \
            + grid_table(g["readout_patch"], pt, "{:.3f}") + [""]
    return lines


def climate_section(c: dict) -> List[str]:
    weeks = c["weeks"]
    lines = ["## The climate frame (2020)", "",
             f"{len(c['inits'])} initialisations; {c['spinup_days']:g} days of budgeted observations (budget {c['budget']:g}) "
             f"under each rule, then {len(weeks)} weeks free; weekly means scored against ERA5 (GraphCast-weighted "
             f"all-point MSE) and the climatology; gains (+ = better) against {c['reference']} (paired bootstrap). A run that "
             f"blows up is scored with the climatology from then on.", "",
             "| run | " + " | ".join(f"week {w}" for w in weeks) + " | gain vs random (+ = better), weeks " + ", ".join(map(str, weeks)) + " | blow-ups |",
             "|---" * (len(weeks) + 3) + "|"]
    for name, r in c["runs"].items():
        ch = c["change"].get(name)
        lines.append(f"| {name} | " + " | ".join(f"{x:.3f}" for x in r["all"]) + " | "
                     + (" ".join(_pct(-x) for x in ch["mean"]) if ch else "reference") + f" | {r['blowups']} |")
    keys = [k for k in c["keys"] if k in HEAD]
    lines += ["", "Weekly RMSE (physical units) and ACC, z500 and t2m:", "",
              "| run | " + " | ".join(f"{HEAD[k]} RMSE w1..w{len(weeks)}" for k in keys[:3]) + " | z500 ACC |",
              "|---" * (len(keys[:3]) + 2) + "|"]
    for name, r in c["runs"].items():
        cells = [" / ".join(f"{x:.3g}" for x in r["rmse"][k]) for k in keys[:3]]
        acc = r["acc"].get("geopotential@500", [])
        lines.append(f"| {name} | " + " | ".join(cells) + " | " + " / ".join(f"{x:.2f}" for x in acc) + " |")
    return lines


def history_section(h: list) -> List[str]:
    if not h:
        return []
    q = max(1, len(h) // 4)
    lines = ["## Learning curve", "",
             "| quarter | error (obs part) | error (free part) | entropy (first pick, nat) | blow-ups | grad |",
             "|---|---|---|---|---|---|"]
    for i in range(0, len(h), q):
        part = h[i:i + q]
        lines.append(f"| {i}-{i + len(part) - 1} | {np.mean([x['error_obs'] for x in part]):.3f} | "
                     f"{np.mean([x['error_free'] for x in part]):.3f} | {np.mean([x.get('entropy', np.nan) for x in part]):.2f} | "
                     f"{sum(x['blowups'] for x in part)} | {np.nanmean([x.get('grad', np.nan) for x in part]):.3g} |")
    lines += ["", f"{len(h)} iterations, {h[-1]['time_s'] / 60:.0f} min."]
    return lines


def verdicts(g: dict, c: Optional[dict], gt: Optional[dict], nomem: Optional[dict]) -> List[str]:
    obs = g["obs_steps"]
    runs, ch = g["runs"], g["change"]
    out = ["## Decision rules", ""]
    pol = ch.get("memory:policy")
    a1 = pol is not None and pol["free"]["mean"][0] <= -0.05 and pol["free"]["hi"][0] < 0
    out.append(f"- **A1** the policy's free-lead error is >= 5% below the random rule (paired CI excludes 0)  \n"
               f"  → {'supported' if a1 else 'not supported'} ({_ci(pol['free']) if pol else 'no policy run'})")
    fixed = {r: free_mean(runs[f"memory:{r}"], obs) for r in ("age", "readout", "tendency") if f"memory:{r}" in runs}
    if fixed and "memory:policy" in runs:
        best = min(fixed, key=fixed.get)
        rel = (free_mean(runs["memory:policy"], obs) - fixed[best]) / fixed[best]
        a2 = rel <= -0.03
        out.append(f"- **A2** the policy is >= 3% below the best fixed rule  \n  → {'supported' if a2 else 'not supported'} "
                   f"(best fixed rule {best}: policy gain {_pct(-rel)}; " + ", ".join(f"{r} {v:.3f}" for r, v in fixed.items()) + ")")
    rho = g["spearman_tau"].get("memory:policy", float("nan"))
    a3 = np.isfinite(rho) and rho >= 0.3
    out.append(f"- **A3** Spearman(selection frequency, patch log tau) >= +0.3: the policy looks where the memory lasts  \n"
               f"  → {'supported' if a3 else ('not supported: the policy looks where the memory is short' if np.isfinite(rho) and rho <= -0.3 else 'not supported')} (rho {rho:+.2f})")
    if "memory:policy" in runs and "n1:policy" in runs:
        fm, fn = free_mean(runs["memory:policy"], obs), free_mean(runs["n1:policy"], obs)
        rel = (fm - fn) / fn
        out.append(f"- **A4** the memory model is >= 5% below N1 + nudging under the same policy (memory maps zero on N1)  \n"
                   f"  → {'supported' if rel <= -0.05 else 'not supported'} (memory {fm:.3f} vs N1 {fn:.3f}, gain {_pct(-rel)})")
    if c is not None and "memory:policy" in c["change"]:
        m = c["change"]["memory:policy"]["mean"]
        w12 = float(np.mean(m[:2]))
        out.append(f"- **A5** climate frame: the policy's weeks 1-2 error is >= 3% below random  \n"
                   f"  → {'supported' if w12 <= -0.03 else 'not supported'} (weeks 1-2 gain {_pct(-w12)}; weeks 3-4 gain "
                   + ", ".join(_pct(-x) for x in m[2:]) + ")")
    if gt is not None and "memory:policy" in gt["runs"] and "memory:policy" in runs:
        r = spearman(gt["runs"]["memory:policy"]["freq"], runs["memory:policy"]["freq"])
        out.append(f"- **A6** invariance: rank correlation of the policy's patch frequencies, 2018-19 vs 2020, >= 0.7  \n"
                   f"  → {'supported' if r >= 0.7 else 'not supported'} (r {r:+.2f})")
    if nomem is not None and "memory:policy" in nomem["change"]:
        out.append(f"- ablation P-nomem (the policy without the memory's maps) vs random: {_ci(nomem['change']['memory:policy']['free'])}"
                   + (f"; with the maps: {_ci(pol['free'])}" if pol else ""))
    return out


def direction(g: dict, c: Optional[dict]) -> List[str]:
    obs = g["obs_steps"]
    runs, ch = g["runs"], g["change"]
    lines = ["## Direction", ""]
    pol = ch.get("memory:policy")
    full, none = runs.get("memory:full"), runs.get("memory:none")
    if full and none and "memory:random" in runs:
        rnd = free_mean(runs["memory:random"], obs)
        span = free_mean(none, obs) - free_mean(full, obs)
        lines.append(f"- The budget's room: full observation {free_mean(full, obs):.3f}, none {free_mean(none, obs):.3f}, "
                     f"random at the budget {rnd:.3f}; the random rule already closes "
                     f"{100 * (free_mean(none, obs) - rnd) / span:.0f}% of the gap, so the choice can only move the rest.")
    if pol is not None:
        if pol["free"]["mean"][0] <= -0.05:
            lines.append("- The choice carries information for this model: next, widen the action (which variables to observe, "
                         "and the memory's timescale as a second action), and train the policy jointly with the memory (the "
                         "DreamerV3 loop of stage 10) so the memory learns to keep what the policy will not look at again.")
        else:
            lines.append("- The learned choice does not beat the random rule: either the error is dominated by the model's "
                         "own drift (the free-run collapse of §10.1), so no observation placement helps, or the 60 h training "
                         "episodes are too short for the memory to matter. Next: the stage 13 energy constraint first "
                         "(a stable column), then longer episodes (the 10-day spin-up of the climate frame as the training game).")
    if "n1:policy" in runs and "memory:policy" in runs:
        if free_mean(runs["memory:policy"], obs) < free_mean(runs["n1:policy"], obs):
            lines.append("- The memory model keeps the observations longer than N1 + nudging: the memory is the right place "
                         "for the observing system's action.")
        else:
            lines.append("- N1 + nudging is at least as good under the same observations: the fine-tuned column (§10.1, the "
                         "causes of stage 10) still costs more than the memory gains; retrain the memory with the stage 10.2 "
                         "continuous filter before the next observation-policy round.")
    if c is not None and "memory:policy" in c["change"] and len(c["change"]["memory:policy"]["mean"]) > 2:
        m = c["change"]["memory:policy"]["mean"]
        lines.append(f"- Climate frame: weeks 3-4 gain vs random {', '.join(_pct(-x) for x in m[2:])}; "
                     + ("observation placement reaches the subseasonal range." if min(m[2:]) <= -0.03
                        else "beyond two weeks the placement no longer matters, as expected without slow memory channels (10.2)."))
    return lines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", required=True)
    parser.add_argument("--game-train", default=None)
    parser.add_argument("--climate", default=None)
    parser.add_argument("--nomem", default=None)
    parser.add_argument("--history", default=None)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    g, gt, c, nomem, h = load(args.game), load(args.game_train), load(args.climate), load(args.nomem), load(args.history)
    lines = ["# Stage 12: active observation on the memory model", "",
             "The frozen stage 10 memory model (D-noactor) plays the observation game (train/active_obs.py): after each "
             "6 h forecast a rule picks which equal-area patches to observe (budget 25% = 6 of 24), the posterior takes "
             "them in, the rest keeps the forecast plus the memory's readout. Rules: the learned policy, random (control), "
             "age (round-robin), readout (where the memory expects the largest correction), tendency (largest 6 h change), "
             "full (upper bound) and none (cold free run). Design and rules A1-A6: README_KO §12.", ""]
    lines += history_section(h or []) + [""]
    lines += game_section(g, "The observation game (2020)") + [""]
    lines += budget_section(g) + [""]
    lines += maps_section(g) + [""]
    if gt is not None:
        lines += game_section(gt, "The game on the training years (2018-19)") + [""]
    if nomem is not None:
        lines += game_section(nomem, "Ablation: the policy without the memory's maps (P-nomem)") + [""]
    if c is not None:
        lines += climate_section(c) + [""]
    lines += verdicts(g, c, gt, nomem) + [""]
    lines += direction(g, c)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines[-14:]))
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
