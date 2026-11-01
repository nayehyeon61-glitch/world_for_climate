#!/usr/bin/env python
"""Stage 14.2 board 4: the sequential intervention game with the forced-response reward.

    # one arm (policy | random | bandit | single | noctx) from the N1-T teacher student
    python scripts/train_sequential_game.py train --init $S/runs/stage15/N1T/checkpoint.pt --arm policy \
        --speedy $S/sim/speedy15 --data $S/speedy_wb2 --out runs/stage14_2/policy --episodes 30
    # G1: the student's error on the ensemble forced response, held-out sites + global forcings, 1984
    python scripts/train_sequential_game.py eval --init runs/stage14_2/policy/checkpoint.pt ... \
        --out runs/stage14_2/policy

On the cluster (GPU, 96 GB): the student runs on the GPU with --device cuda; SPEEDY stays on the
CPU (~1 s per 2-day member), so the per-decision cost is the student's 40 steps. Suggested scale:
--device cuda --threads 8 --members 8 --episodes 200 (800 decisions; the container runs 20 episodes
with 4 members).

Resumable at episode boundaries (checkpoint.pt carries the progress). `--preflight` runs one
decision and exits (chains run it before the real job: a broken pipeline stops in minutes).
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os

import _path  # noqa: F401
import numpy as np
import torch

from climate_wm import pipeline
from climate_wm.sim.intervention_game import test_actions, train_actions
from climate_wm.sim.sequential_game import SeqConfig, SequentialWorld, forced_response_eval, train_sequential
from train_speedy_game import build


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=["train", "eval"])
    ap.add_argument("--init", required=True)
    ap.add_argument("--speedy", required=True, help="the speedy15 build (perturbed restarts)")
    ap.add_argument("--data", required=True, help="nature-run root (nature.zarr, restarts/, static.nc)")
    ap.add_argument("--work", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--arm", default="policy", choices=["policy", "random", "bandit", "single", "noctx"])
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--members", type=int, default=4)
    ap.add_argument("--response-weight", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--no-randomize", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--preflight", action="store_true", help="train: one decision, no checkpoint")
    ap.add_argument("--models", nargs="*", default=[], help="eval: more LABEL=checkpoint students")
    ap.add_argument("--members-eval", type=int, default=5)
    ap.add_argument("--eval-points", type=int, default=6)
    ap.add_argument("--eval-steps", type=int, default=8)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    args.work = args.work or os.path.join(args.data, "work")
    os.makedirs(args.out, exist_ok=True)
    cfg, model, stats, world = build(args, args.device)
    world = SequentialWorld(world)

    if args.mode == "train":
        sc = SeqConfig(arm=args.arm, episodes=args.episodes, members=args.members, lr=args.lr,
                       response_weight=args.response_weight, randomize=not args.no_randomize, seed=args.seed)
        if args.preflight:
            sc = dataclasses.replace(sc, episodes=1, decisions=1)
            hist = train_sequential(model, world, sc, args.device)
            bad = [k for k in ("state_loss", "resp_before", "resp_after") if not np.isfinite(hist[k][0])]
            if bad or hist["state_loss"][0] > 1e3:
                raise SystemExit(f"preflight failed: {bad or hist['state_loss'][0]}")
            print(f"preflight ok: {hist['seconds'][0]:.0f} s per decision")
            return
        ckpt = os.path.join(args.out, "checkpoint.pt")
        resume = None
        if os.path.exists(ckpt):
            saved = torch.load(ckpt, map_location="cpu", weights_only=False)
            model.load_state_dict(saved["model"])
            resume = saved.get("progress")
            if resume is None:
                print(f"{ckpt} is complete")
                return
            print(f"resuming {ckpt} after {resume['episodes_done']} episodes")

        def save(progress):
            pipeline.save_checkpoint(ckpt, cfg, model, stats, {"seq_game": progress["history"]}, progress)

        hist = train_sequential(model, world, sc, args.device, resume, save)
        pipeline.save_checkpoint(ckpt, cfg, model, stats, {"seq_game": hist})
        with open(os.path.join(args.out, "seq_history.json"), "w") as f:
            json.dump({"config": dataclasses.asdict(sc), "history": hist,
                       "train_actions": [a.name for a in train_actions()]}, f)
        print(f"saved {ckpt}")
    else:
        models = {"init": model}
        for spec in args.models:   # more students on the same SPEEDY truth
            label, path = spec.split("=", 1)
            saved = torch.load(path, map_location="cpu", weights_only=False)
            other = _clone(model)
            other.load_state_dict(saved["model"])
            models[label] = other
        days = int(np.ceil(args.eval_steps / 4))
        points = world.branch_points("1984-01-01", "1984-12-31", days + 1)
        idx = np.linspace(0, len(points) - 1, args.eval_points).astype(int)
        path = os.path.join(args.out, "forced_response.json")
        done = None
        if os.path.exists(path + ".progress"):
            with open(path + ".progress") as f:
                done = json.load(f)
            print(f"resuming after {len(done['points_done'])} points")

        def save(res):
            with open(path + ".progress", "w") as f:
                json.dump(res, f)

        res = forced_response_eval(models, world, [points[i] for i in idx], test_actions(), args.eval_steps,
                                   args.members_eval, args.device, done=done, save=save)
        res["checkpoints"] = {"init": args.init, **dict(s.split("=", 1) for s in args.models)}
        with open(path, "w") as f:
            json.dump(res, f)
        os.remove(path + ".progress")
        for lab in models:
            rel = [r[lab]["rel_error"] for r in res["rows"] if np.isfinite(r[lab]["rel_error"])]
            print(f"{lab}: mean relative forced-response error {np.mean(rel):.3f} over {len(rel)} pairs")


def _clone(model):
    import copy
    return copy.deepcopy(model)

if __name__ == "__main__":
    main()
