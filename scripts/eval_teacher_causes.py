"""Stage 15.4: ground-truth causal scoring of a world model against SPEEDY (eval/teacher_causes.py).

    python scripts/eval_teacher_causes.py --init runs/stage15/N1T/checkpoint.pt --speedy sim/speedy15 \
        --data DATA/speedy_wb2 --work WORK --out runs/stage15/N1T/causes15.json [--inits 12 --days 5]

Initialisations: the nature-run restarts of the test year 1984, evenly spaced. The
cuts and the perturbed ensemble (--members, default 4) need the speedy15 binary.
Resumable (OUT.progress).
"""
import argparse
import datetime as dt
import json
import os

import _path  # noqa: F401
import numpy as np
import torch

from climate_wm.eval.teacher_causes import CUTS, evaluate_teacher_causes
from train_speedy_game import build


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--init", required=True)
    ap.add_argument("--speedy", required=True, help="the speedy15 SPEEDY copy (cut switches)")
    ap.add_argument("--data", required=True, help="nature-run root (nature.zarr, restarts/, static.nc)")
    ap.add_argument("--work", required=True)
    ap.add_argument("--out", required=True, help="output json")
    ap.add_argument("--inits", type=int, default=12)
    ap.add_argument("--days", type=int, default=5)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--start", default="1984-01-01")
    ap.add_argument("--end", default="1984-12-31")
    ap.add_argument("--cuts", nargs="+", default=list(CUTS))
    ap.add_argument("--members", type=int, default=4,
                    help="perturbed SPEEDY restarts whose mean response is the target (0 = the single branch)")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)
    args.out_dir = out_dir
    bargs = argparse.Namespace(init=args.init, speedy=args.speedy, data=args.data, work=args.work,
                               out=os.path.join(out_dir, "causes15_work"))
    cfg, model, stats, world = build(bargs, args.device)
    points = world.branch_points(args.start, args.end, args.days)
    idx = np.unique(np.linspace(0, len(points) - 1, args.inits).round().astype(int))
    points = [points[i] for i in idx]
    prog = args.out + ".progress"
    done = json.load(open(prog)) if os.path.exists(prog) else []
    print(f"{len(points)} initialisations {points[0]:%Y-%m-%d} .. {points[-1]:%Y-%m-%d}; resumed {len(done)}")

    def save(rows):
        with open(prog + ".tmp", "w") as f:
            json.dump(rows, f)
        os.replace(prog + ".tmp", prog)

    res = evaluate_teacher_causes(cfg["model"], model.state_dict(), model.layout, model.lat, model.lon, world,
                                  points, args.steps, args.days, args.device,
                                  cuts={c: CUTS[c] for c in args.cuts}, members=args.members, done=done, save=save)
    res["members"] = args.members
    res["checkpoint"] = args.init
    with open(args.out, "w") as f:
        json.dump(res, f)
    if os.path.exists(prog):
        os.remove(prog)
    print(f"T2 sign agreement at 72 h {res['T2_sign_72h']:.3f}; saved {args.out}")


if __name__ == "__main__":
    main()
