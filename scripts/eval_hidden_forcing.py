#!/usr/bin/env python
"""Stage 15.5: hidden SST memory test in the SPEEDY world (rules T4-T6; README_KO section 15).

    python scripts/eval_hidden_forcing.py --init runs/stage14/M_adapt/checkpoint.pt --label M_adapt \
        --models M_lp=runs/stage14/M_lp/checkpoint.pt --shown M_lp --snapshots runs/stage14/M_snaps.pt \
        --speedy $S/sim/speedy --data $S/speedy_wb2 --work $S/hiddenwork --out runs/stage15/hidden

For each 1984 branch point a SPEEDY branch with a persistent hidden SST anomaly (3 patches of +-2 K) and
the nature run from the same date; every model re-observes h = 0/1/5/10/30 days from the filled memory
(the 1982-84 nature pass), then runs 5 days free, with the memory kernel on and knocked out. `--shown`
models also run with the anomaly in the action channel. Writes hidden.json (rows, resumable per point)
and probe.pt (the filled memory states with the anomaly on the core grid, for the T5 probe).
`--signal-only` writes signal.json instead: the anomaly's footprint in the truth (anomaly branch minus
nature run) per h and lead, the ceiling of any memory gain (scripts/report_hidden_forcing.py).
"""
from __future__ import annotations

import argparse
import copy
import json
import os

import _path  # noqa: F401
import numpy as np
import torch

from climate_wm.eval.hidden_forcing import HiddenConfig, forced_signal, hidden_forcing_eval, probe_r2
from train_speedy_game import build


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--init", required=True, help="the first model (it also sets up the SPEEDY world)")
    ap.add_argument("--label", default="M_adapt")
    ap.add_argument("--models", nargs="*", default=[], help="more LABEL=checkpoint models (same architecture)")
    ap.add_argument("--shown", nargs="*", default=[], help="labels that also run with the anomaly shown")
    ap.add_argument("--snapshots", default=None, help="filled-memory snapshots (train_speedy_game.py fill)")
    ap.add_argument("--speedy", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--work", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--points", type=int, default=6)
    ap.add_argument("--histories", type=int, nargs="*", default=[0, 1, 5, 10, 30])
    ap.add_argument("--free-steps", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--signal-only", action="store_true",
                    help="only the forced signal (anomaly branch minus nature per h and lead) -> signal.json")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    args.work = args.work or os.path.join(args.data, "work")
    os.makedirs(args.out, exist_ok=True)
    _, model, _, world = build(args, args.device)
    model.eval()
    models = {args.label: model}
    for spec in args.models:
        label, path = spec.split("=", 1)
        other = copy.deepcopy(model)
        other.load_state_dict(torch.load(path, map_location="cpu", weights_only=False)["model"])
        other.eval()
        models[label] = other
    snaps = None
    if args.snapshots:
        snaps = torch.load(args.snapshots, map_location="cpu", weights_only=False)["snaps"]
        print(f"memory snapshots: {len(snaps)} from {args.snapshots}")
    cfg = HiddenConfig(histories_days=tuple(args.histories), free_steps=args.free_steps, seed=args.seed)
    days = max(cfg.histories_days) + int(np.ceil(cfg.free_steps / 4)) + 1
    points = world.branch_points("1984-01-01", "1984-12-31", days)
    idx = np.linspace(0, len(points) - 1, args.points).astype(int)
    points = [points[i] for i in idx]
    print(f"{len(points)} branch points: {[p.date().isoformat() for p in points]}")
    if args.signal_only:
        sig = forced_signal(model, world, points, cfg, device=args.device)
        with open(os.path.join(args.out, "signal.json"), "w") as f:
            json.dump({"rows": sig}, f)
        for r in sig:
            print(f"[signal] {r['init'][:10]} h={r['h']}d: all 24/72/120 h {r['all'][3]:.4f}/{r['all'][11]:.4f}/"
                  f"{r['all'][-1]:.4f}, t850 near patch {r['t850'][3]:.2f}/{r['t850'][11]:.2f}/{r['t850'][-1]:.2f} K")
        return
    js, pt = os.path.join(args.out, "hidden.json"), os.path.join(args.out, "probe.pt")
    done = None
    if os.path.exists(js) and os.path.exists(pt):
        with open(js) as f:
            done = json.load(f)
        done["probe"] = torch.load(pt, weights_only=False)
        print(f"resuming after {len(done['points'])} points")

    def save(res):
        torch.save(res["probe"], pt + ".tmp")
        with open(js + ".tmp", "w") as f:
            json.dump({k: v for k, v in res.items() if k != "probe"}, f)
        os.replace(pt + ".tmp", pt)
        os.replace(js + ".tmp", js)

    res = hidden_forcing_eval(models, world, points, cfg, snaps=snaps, shown=args.shown, device=args.device,
                              done=done, save=save)
    save(res)
    r2 = probe_r2(res["probe"])
    with open(os.path.join(args.out, "probe_r2.json"), "w") as f:
        json.dump({f"{m} h={h}": v for (m, h), v in r2.items()}, f, indent=1)
    print("probe R2:", {f"{m} h={h}": round(v, 3) for (m, h), v in r2.items()})


if __name__ == "__main__":
    main()
