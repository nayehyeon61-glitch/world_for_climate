#!/usr/bin/env python
"""Stage 16 board 1: the N1 climate-bias game (README_KO section 16; train/climate_game.py).

    # K0: the default model's error curve (the reward scale), blow-ups, observation statistics
    python scripts/play_climate_game.py calibrate --checkpoint runs/physics_n1/checkpoint.pt \
        --data wb2_2020/era5.zarr --out runs/stage16/calib.json
    # agents: td3 | ppo | cem (the best constant, Kconst) | grad (Kgrad, back-propagation through N1)
    python scripts/play_climate_game.py train --agent td3 --iters 40 --calib runs/stage16/calib.json \
        --checkpoint ... --data ... --out runs/stage16/td3
    # 2020 test: 31 initialisations, 30-day free runs, deterministic policy
    python scripts/play_climate_game.py eval --policy td3 --agent-dir runs/stage16/td3 --days 30 \
        --calib runs/stage16/calib.json --checkpoint ... --data ... --out runs/stage16/eval_td3.json

Policies for eval: zero (K0), rand, const (CEM mean), td3, ppo, grad. Cause ablations:
--fixed-groups (knob groups held at the default), --no-state (Knostate), --sparse-lag 5 (RAIN v2 reward);
--set column.energy_max_flux=500 = the stage 13 E1 energy bound as a non-learning blow-up control.
Train windows: 00 UTC initialisations of 2018-19 (N1 was trained on 2015-17); test: 2020.
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
from climate_wm.data.era5_lazy_loader import GraphCastWindowDataset
from climate_wm.train.climate_game import (GameConfig, Scorer, Windows, calibrate, evaluate, load_policy, obs_dim,
                                           train_agent)

SPLITS = {"train": ("2018-01-01", "2019-12-31"), "test": ("2020-01-01", "2020-12-31")}


def build(args, steps: int, split: str):
    cfg, state, stats = pipeline.load_checkpoint(args.checkpoint, "cpu")
    pipeline.apply_overrides(cfg, [f"data.path={args.data}", "data.time_range=null", "data.load_into_memory=false",
                                   *args.set])
    layout = pipeline.build_layout(cfg)
    ds = pipeline.open_dataset(cfg)
    lo, hi = SPLITS[split]
    wds = GraphCastWindowDataset(ds, layout, stats, steps, start=lo, end=hi, init_hours=[0])
    model = pipeline.build_model(cfg, layout, wds)
    model.load_state_dict(state)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, Windows(wds, args.device)


def game_config(args, days: int) -> GameConfig:
    return GameConfig(days=days, sparse_lag=args.sparse_lag, state=not args.no_state,
                      fixed_groups=tuple(args.fixed_groups), cap=args.cap, penalty=args.penalty)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["calibrate", "train", "eval"])
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data", required=True, help="WB2 zarr with 2018-2020 (e.g. wb2_2020/era5.zarr)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--calib", default=None)
    ap.add_argument("--agent", choices=["td3", "ppo", "cem", "grad"], default="td3")
    ap.add_argument("--policy", choices=["zero", "rand", "const", "td3", "ppo", "grad"], default="zero")
    ap.add_argument("--agent-dir", default=None)
    ap.add_argument("--days", type=int, default=None, help="episode days (train 15, grad 5, eval 30)")
    ap.add_argument("--iters", type=int, default=40)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--windows", type=int, default=16, help="calibrate: training windows")
    ap.add_argument("--inits", type=int, default=31, help="eval: initialisations")
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sparse-lag", type=int, default=0)
    ap.add_argument("--no-state", action="store_true")
    ap.add_argument("--fixed-groups", nargs="*", default=[], choices=["heating", "moistening", "momentum", "diffusion"])
    ap.add_argument("--cap", type=float, default=10.0)
    ap.add_argument("--penalty", type=float, default=10.0)
    ap.add_argument("--threads", type=int, default=1, help="1: more threads stall under the other lanes (45 s vs 0.6 s per 6 h)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                    help="model config overrides, e.g. column.energy_max_flux=500 (the stage 13 E1 bound control)")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    log = lambda s: print(s, flush=True)
    if args.command == "calibrate":
        days = args.days or 15
        model, win = build(args, 4 * days, "train")
        cfg = game_config(args, days)
        sc = Scorer(model)
        log(f"calibrate: {len(win)} training windows, {days} days, {sc.n} reward channels")
        calib = calibrate(model, win, cfg, sc, n=args.windows, batch=args.batch, log=log)
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(calib, f, indent=1)
        log(f"K0 daily error {np.round(calib['day'], 4).tolist()}, episode {calib['term']:.4f}, "
            f"blown {calib['blown']}/{calib['windows']}")
        return
    with open(args.calib) as f:
        calib = json.load(f)
    if args.command == "train":
        days = args.days or (5 if args.agent == "grad" else 15)
        model, win = build(args, 4 * days, "train")
        cfg = game_config(args, days)
        if days < len(calib["day"]):
            calib = dict(calib, day=calib["day"][:days])
        log(f"train {args.agent}: {len(win)} windows, {days}-day episodes, batch {args.batch}, {args.iters} iterations")
        train_agent(args.agent, model, win, cfg, Scorer(model), calib, args.out, args.iters, args.batch,
                    args.seed, args.lr, log=log)
        return
    days = args.days or 30
    model, win = build(args, 4 * days, "test")
    cfg = game_config(args, days)
    sc = Scorer(model)
    path = os.path.join(args.agent_dir, "agent.pt") if args.agent_dir else None
    policy = load_policy(args.policy, path, obs_dim(sc), args.seed)
    log(f"eval {args.policy}: {len(win)} test windows, {args.inits} initialisations, {days} days")
    res = evaluate(model, win, policy, cfg, sc, calib, args.out, n=args.inits, batch=args.batch, log=log)
    log(json.dumps(res["summary"], indent=1))


if __name__ == "__main__":
    main()
