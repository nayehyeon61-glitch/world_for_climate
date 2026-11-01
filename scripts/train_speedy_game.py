#!/usr/bin/env python
"""Stage 14 board 2: the intervention game in the SPEEDY world.

    # domain adaptation of N1 to the SPEEDY world (nature windows, action channels at zero)
    python scripts/train_speedy_game.py train --init runs/physics_n1/checkpoint.pt --arm adapt \
        --speedy $S/sim/speedy --data $S/speedy_wb2 --out runs/stage14/adapt --updates 300
    # the three arms from the adapted model
    python scripts/train_speedy_game.py train --init runs/stage14/adapt/checkpoint.pt --arm lp ...
    # S1/S4 response evaluation on the 1984 branch points
    python scripts/train_speedy_game.py eval --init runs/stage14/lp/checkpoint.pt ... --out runs/stage14/lp/response.json
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
from climate_wm.sim.intervention_game import (GameConfig, SpeedyWorld, response_eval, test_actions,
                                              train_actions, train_game)
from climate_wm.train.memory_filter import expand_state_dict


def build(args, device):
    cfg, state, stats = pipeline.load_checkpoint(args.init, device)
    cfg = dict(cfg)
    model_cfg = dict(cfg["model"])
    model_cfg["action_channels"] = 3
    model_cfg["solar"] = "daily_mean"
    cfg["model"] = model_cfg
    data = dict(cfg["data"])
    data.update(source="zarr", path=os.path.join(args.data, "nature.zarr"), load_into_memory=False,
                time_range=["1982-01-01", "1985-01-01"],
                train={"start": "1982-01-01T06:00", "end": "1983-12-31T18:00"},
                val={"start": "1984-01-01", "end": "1984-12-31T12:00"})
    data.pop("stats_path", None)
    cfg["data"] = data
    cfg["eval"] = dict(cfg.get("eval", {}), climatology_zarr=None)
    cfg["output_dir"] = args.out
    layout = pipeline.build_layout(cfg)
    nature = pipeline.open_dataset(cfg)
    train_ds, _, _ = pipeline.build_datasets(cfg, layout, nature, stats)
    model = pipeline.build_model(cfg, layout, train_ds).to(device)
    grown = expand_state_dict(model, state)
    print(f"loaded {args.init}; widened: {grown}")
    world = SpeedyWorld(args.speedy, args.data, os.path.join(args.work, os.path.basename(args.out.rstrip('/'))),
                        nature, layout, stats, layout.task.pressure_levels)
    world.check_sites()
    return cfg, model, stats, world


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=["train", "eval"])
    ap.add_argument("--init", required=True)
    ap.add_argument("--speedy", required=True)
    ap.add_argument("--data", required=True, help="nature-run root (nature.zarr, restarts/, static.nc)")
    ap.add_argument("--work", default=None, help="SPEEDY branch run directories (default DATA/work)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--arm", default="lp", choices=["lp", "random", "nature", "adapt"])
    ap.add_argument("--updates", type=int, default=150)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-points", type=int, default=4)
    ap.add_argument("--eval-steps", type=int, default=20)
    ap.add_argument("--eval-subset", choices=["warm", "all"], default="warm")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    args.work = args.work or os.path.join(args.data, "work")
    os.makedirs(args.out, exist_ok=True)
    cfg, model, stats, world = build(args, args.device)

    if args.mode == "train":
        gc = GameConfig(arm=args.arm, updates=args.updates, lr=args.lr, seed=args.seed)
        ckpt = os.path.join(args.out, "checkpoint.pt")
        resume = None
        if os.path.exists(ckpt):
            saved = torch.load(ckpt, map_location="cpu", weights_only=False)
            model.load_state_dict(saved["model"])
            resume = saved.get("progress")
            print(f"resuming {ckpt} after {len(resume['history']['update']) if resume else 'end'} updates")
            if resume is None:
                return

        def save(progress):
            pipeline.save_checkpoint(ckpt, cfg, model, stats, {"game": progress["history"]}, progress)

        hist = train_game(model, world, gc, args.device, resume, save)
        pipeline.save_checkpoint(ckpt, cfg, model, stats, {"game": hist})
        with open(os.path.join(args.out, "game_history.json"), "w") as f:
            json.dump({"config": dataclasses.asdict(gc), "history": hist,
                       "train_actions": [a.name for a in train_actions()]}, f)
        print(f"saved {ckpt}")
    else:
        points = world.branch_points("1984-01-01", "1984-12-31", int(np.ceil(args.eval_steps / 4)) + 1)
        idx = np.linspace(0, len(points) - 1, args.eval_points).astype(int)
        acts = test_actions()
        if args.eval_subset == "warm":   # held-out sites +2 K, solar +2 %, CO2 x1.25
            acts = [a for a in acts if a.name.endswith("+") or a.name in ("solar1.02", "co2x1.25")]
        res = response_eval(model, world, [points[i] for i in idx], acts, args.eval_steps, args.device)
        res["checkpoint"] = args.init
        with open(os.path.join(args.out, "response.json"), "w") as f:
            json.dump(res, f)
        print(f"saved {os.path.join(args.out, 'response.json')}")


if __name__ == "__main__":
    main()
