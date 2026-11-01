"""Stage 13 report: blow-ups and the column's energy input under the E1 bounds (rules B1, B4).

    python scripts/report_energy.py --energy memory=runs/stage10_Dnoactor/energy_2020.json n1=runs/physics_n1/energy_2020.json \
        [--game runs/stage10_Dnoactor/eval_2020.json runs/stage10_Dnoactor/eval_2020_energy.json] \
        [--causes runs/physics_n1/causes_2020.json runs/physics_n1/causes_2020_energy.json] \
        --out docs/results/stage13_energy/energy.md
"""

import argparse
import json
import os
from typing import Dict, List, Optional

import numpy as np

TERM_NAMES = {"heat_mean": "heating c_p∫Q (global mean)", "moist_mean": "moistening L_v∫S (global mean)",
              "work_mean": "work ∫F·u (global mean)", "flux_max": "largest |flux| of a column",
              "work_max": "largest work of a column", "clamped": "clamped fraction"}


def load(path: Optional[str]):
    if not path or not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def _pct(x) -> str:
    return "-" if x is None or not np.isfinite(x) else f"{100 * x:+.1f}%"


def energy_section(label: str, e: dict) -> List[str]:
    leads = e["lead_h"]
    n = len(e["inits"])
    at = [leads.index(h) for h in (24, 72, 120) if h in leads]
    lines = [f"## {label}: free runs under the bounds (2020, {n} initialisations, {leads[-1]} h)", "",
             "Blow-ups per start kind and setting; the error ratio against the unbounded model over the "
             "initialisations finite in both (GraphCast-weighted all-point MSE, normalised units); the energy "
             "terms are means over the run (W/m^2).", "",
             "| start | setting | blow-ups | error ratio vs none (24 / 72 / 120 h) | heating | moistening | work | largest flux | clamped |",
             "|---|---|---|---|---|---|---|---|---|"]
    for name, r in e["runs"].items():
        kind, setting = name.split(":")
        ratio = r.get("err_ratio_finite")
        rat = " / ".join(f"{ratio[i]:.3f}" for i in at) + f" (n={r['finite_both']})" if ratio else ("reference" if setting == "none" else "-")
        en = np.array(r["energy"]).mean(0)
        lines.append(f"| {kind} | {setting} | {r['blowups']}/{n} | {rat} | {en[0]:.1f} | {en[1]:.1f} | {en[2]:.2f} | {en[3]:.0f} | {100 * en[5]:.1f}% |")
    before = [(name, r) for name, r in e["runs"].items() if "before_blowup" in r]
    if before:
        lines += ["", "The energy terms over the four steps before a blow-up, against the same leads of the runs that stayed finite:", "",
                  "| run | term | before blow-up | finite runs, same leads | ratio |", "|---|---|---|---|---|"]
        for name, r in before:
            same = r.get("same_leads_finite")
            for j, t in enumerate(e["terms"]):
                b = r["before_blowup"][j]
                s = same[j] if same else float("nan")
                ratio = abs(b) / max(abs(s), 1e-9) if same else float("nan")
                lines.append(f"| {name} | {TERM_NAMES.get(t, t)} | {b:.3g} | {s:.3g} | {ratio:.1f} |")
    return lines


def game_section(before: dict, after: dict) -> List[str]:
    lines = ["## The stage 10 game on region masks, D-noactor, without and with the bound (31 initialisations, 2020)", "",
             "| variant | blow-ups without | blow-ups with | 24 h hidden error without | with |", "|---|---|---|---|---|"]
    kb, ka = before["kinds"].get("region", {}), after["kinds"].get("region", {})
    for v in ("full", "nudging"):
        if f"{v}_hidden" in kb.get("mse", {}) and f"{v}_hidden" in ka.get("mse", {}):
            lines.append(f"| {v} | {kb['blowups'].get(v, '-')} | {ka['blowups'].get(v, '-')} | "
                         f"{kb['mse'][f'{v}_hidden'][3]:.3f} | {ka['mse'][f'{v}_hidden'][3]:.3f} |")
    return lines


def causes_section(before: dict, after: dict) -> List[str]:
    lines = ["## The stage 9 cuts on N1 (2020, 8 initialisations, 72 h), without and with the bound", "",
             "| cut | blow-ups without | with | first blow-up (h) without | with | 72 h all-MSE gain of the cut (+ = better) without | with |",
             "|---|---|---|---|---|---|---|"]
    for name in ("no_moistening", "no_heating"):
        b, a = before["interventions"].get(name), after["interventions"].get(name)
        if not b or not a:
            continue
        fb = lambda d: (f"{max(d['blowups'])}", f"{min(h for h in d['first_blowup_h'] if h is not None) if any(h is not None for h in d['first_blowup_h']) else '-'}",
                        _pct(-d["change_all"]["mean"][-1]) if d.get("change_all") else "-")
        (bb, bh, bc), (ab, ah, ac) = fb(b), fb(a)
        lines.append(f"| {name} | {bb} | {ab} | {bh} | {ah} | {bc} | {ac} |")
    return lines


def verdicts(energies: Dict[str, dict], game, causes) -> List[str]:
    out = ["## Decision rules", ""]
    mem, n1 = energies.get("memory"), energies.get("n1")
    removed, total = 0, 0
    parts = []
    if mem:
        for kind in mem["kinds"]:
            b0 = mem["runs"].get(f"{kind}:none", {}).get("blowups", 0)
            b1 = mem["runs"].get(f"{kind}:flux500_nowork", {}).get("blowups", 0)
            removed += max(b0 - b1, 0)
            total += b0
            parts.append(f"memory {kind} {b0}→{b1}")
    if game:
        b0 = game[0]["kinds"].get("region", {}).get("blowups", {}).get("full", 0)
        b1 = game[1]["kinds"].get("region", {}).get("blowups", {}).get("full", 0)
        removed += max(b0 - b1, 0)
        total += b0
        parts.append(f"game region {b0}→{b1}")
    if causes:
        c0 = causes[0]["interventions"].get("no_moistening", {}).get("blowups")
        c1 = causes[1]["interventions"].get("no_moistening", {}).get("blowups")
        if c0 and c1:
            b0, b1 = max(c0), max(c1)
            removed += max(b0 - b1, 0)
            total += b0
            parts.append(f"no_moistening {b0}→{b1}")
    cost = None
    if n1:
        r = n1["runs"].get("none:flux500_nowork", {}).get("err_ratio_finite")
        if r:
            cost = r[3] - 1
    frac = removed / total if total else float("nan")
    b1_ok = total > 0 and frac >= 0.8 and (cost is None or cost <= 0.03)
    out.append(f"- **B1** the bound (500 W/m^2 + no work) removes >= 80% of the blow-ups (region starts, cold memory, "
               f"the moisture cut) and costs N1 <= 3% at 24 h  \n  → {'supported' if b1_ok else 'not supported'} "
               f"(removed {removed} of {total} = {_pct(frac)}; " + ", ".join(parts) + f"; N1 24 h cost {_pct(cost)})")
    if mem:
        found = []
        for name, r in mem["runs"].items():
            if "before_blowup" in r and r.get("same_leads_finite"):
                for j, t in enumerate(mem["terms"][:5]):
                    b, s = abs(r["before_blowup"][j]), abs(r["same_leads_finite"][j])
                    if b >= 2 * max(s, 1e-9):
                        found.append(f"{name}: {t} x{b / max(s, 1e-9):.1f}")
        out.append(f"- **B4** the injecting term: a term's mean over the four steps before a blow-up is >= 2x the same "
                   f"leads of the finite runs  \n  → {'supported' if found else 'not supported'}" + (f" ({'; '.join(found)})" if found else " (no term stands out, or no blow-up recorded)"))
    out += ["- **B2** (E2 against E1) and **B3** (soft against hard) wait for the E2 / E3 arms."]
    return out


def direction(energies: Dict[str, dict]) -> List[str]:
    mem = energies.get("memory")
    lines = ["## Direction", ""]
    if not mem:
        return lines
    b0 = sum(mem["runs"].get(f"{k}:none", {}).get("blowups", 0) for k in mem["kinds"])
    b1 = sum(mem["runs"].get(f"{k}:flux500_nowork", {}).get("blowups", 0) for k in mem["kinds"])
    bf = sum(mem["runs"].get(f"{k}:flux500", {}).get("blowups", 0) for k in mem["kinds"])
    bw = sum(mem["runs"].get(f"{k}:nowork", {}).get("blowups", 0) for k in mem["kinds"])
    lines.append(f"- Blow-ups of the memory model: none {b0}, flux bound {bf}, no-work {bw}, both {b1}. "
                 + ("No blow-up in this sample: the bound's cost and the energy terms are the result." if b0 == 0 else
                    ("The energy input is the lever: next, E2 learns where the excess is taken from, and the bound goes into "
                    "the stage 10.2 training so the column never learns to rely on it." if b1 < b0 else
                    "The bounds alone do not stop the blow-ups: the instability is not the column's net energy input "
                    "but its vertical structure or the dynamics' response; next, bound the static stability (no heating that "
                    "makes a layer superadiabatic) and look at the momentum forcing's vertical shear.")))
    r = mem["runs"].get("none:flux500_nowork", {}).get("err_ratio_finite")
    if r:
        lines.append(f"- On the finite runs the bound gains the memory model {_pct(1 - r[3])} at 24 h and "
                     f"{_pct(1 - r[min(11, len(r) - 1)])} at 72 h (+ = better): " + ("the unbounded column was injecting energy the forecast did "
                     "not need." if r[3] < 1 else "the column's energy input carries real signal; a tighter bound costs skill."))
    return lines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--energy", nargs="+", required=True, metavar="LABEL=PATH")
    parser.add_argument("--game", nargs=2, default=None, metavar=("WITHOUT", "WITH"))
    parser.add_argument("--causes", nargs=2, default=None, metavar=("WITHOUT", "WITH"))
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    energies = {}
    for item in args.energy:
        label, path = item.split("=", 1)
        e = load(path)
        if e is not None:
            energies[label] = e
    game = [load(p) for p in args.game] if args.game else None
    game = game if game and all(game) else None
    causes = [load(p) for p in args.causes] if args.causes else None
    causes = causes if causes and all(causes) else None
    lines = ["# Stage 13: the column's energy input and the blow-ups", "",
             "E1 bounds the learned column's energy input at run time (physics_world_model.energy_bound): the moist "
             "static energy flux of every column is clamped to +-500 W/m^2 (the excess heating removed uniformly per "
             "unit mass) and the momentum forcing may not do work on the wind. No retraining. Design and rules B1-B4: "
             "README_KO §13.", ""]
    for label, e in energies.items():
        lines += energy_section(label, e) + [""]
    if game:
        lines += game_section(*game) + [""]
    if causes:
        lines += causes_section(*causes) + [""]
    lines += verdicts(energies, game, causes) + [""] + direction(energies)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines[-10:]))
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
