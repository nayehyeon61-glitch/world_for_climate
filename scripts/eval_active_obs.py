"""Stage 12: scores the observation rules on 2020 in the game (--mode game) or the climate frame.

    python scripts/eval_active_obs.py --checkpoint runs/stage10_Dnoactor/checkpoint.pt \
        --base runs/physics_n1/checkpoint.pt --policy runs/stage12_P/policy.pt \
        --config configs/active_obs_wb2.yaml --data-dir DATA --mode game \
        --inits 31 --horizon 20 --obs-steps 8 --budgets 0.25 0.125 0.5 --out runs/stage12_P/game_2020.json
    ... --mode climate --inits 8 --spinup-days 10 --weeks 4 --out runs/stage12_P/climate_2020.json

Every initialisation gets the same draws under every rule, so the rules pair.
Without --policy the untrained policy (= the random rule) stands in, for the
fixed rules alone; --random-policy evaluates the trained policy's network with
random choices (an ablation of the choice itself). Resumable (OUT.progress).
"""

import argparse
import json
import os

import _path  # noqa: F401  (adds src/ to sys.path)
import numpy as np
import torch
from train_active_obs import build_policy, load_world_model

from climate_wm import pipeline
from climate_wm.train.active_obs import RULES, evaluate_active_climate, evaluate_active_obs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="the memory model")
    parser.add_argument("--base", default=None, help="the stage 9 model (no memory: nudging); none to skip")
    parser.add_argument("--policy", default=None, help="OUT/policy.pt of train_active_obs.py")
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--mode", choices=["game", "climate"], default="game")
    parser.add_argument("--split", choices=["val", "train"], default="val")
    parser.add_argument("--inits", type=int, default=31)
    parser.add_argument("--rules", nargs="*", default=list(RULES))
    parser.add_argument("--base-rules", nargs="*", default=["policy", "random", "full", "none"])
    parser.add_argument("--budgets", nargs="*", type=float, default=[0.25])
    parser.add_argument("--horizon", type=int, default=20, help="game: 6 h steps (120 h)")
    parser.add_argument("--obs-steps", type=int, default=8, help="game: choices after the first forecasts (48 h)")
    parser.add_argument("--spinup-days", type=float, default=10.0, help="climate: days of budgeted observations")
    parser.add_argument("--weeks", type=int, default=4)
    parser.add_argument("--random-policy", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--out", required=True)
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    update = pipeline.load_config(args.config, args.data_dir)
    spinup = int(round(4 * args.spinup_days))
    steps = args.horizon if args.mode == "game" else spinup + 28 * args.weeks
    cfg, model, data, stats = load_world_model(args.checkpoint, update, args.split, steps, args.device, args.set)
    base = None
    if args.base and args.base != "none":
        _, base, _, _ = load_world_model(args.base, update, args.split, steps, args.device, args.set)
    if args.policy and os.path.exists(args.policy):
        saved = torch.load(args.policy, map_location=args.device, weights_only=False)
        policy = build_policy(model, data, saved["policy_cfg"]).to(args.device)
        policy.load_state_dict(saved["state"])
        print(f"policy {args.policy} ({len(saved['history'])} iterations)")
    else:
        policy = build_policy(model, data, cfg["policy"]).to(args.device)
        print("untrained policy (= the random rule)")
    policy.random_actions = args.random_policy
    policy.eval()
    indices = np.linspace(0, len(data) - 1, min(args.inits, len(data))).astype(int)
    first, last = data.times[data.indices[indices[0]]], data.times[data.indices[indices[-1]]]
    print(f"{len(indices)} initialisations ({first} .. {last}), mode {args.mode}, rules {args.rules}, "
          f"base {'yes' if base is not None else 'no'} {args.base_rules}, budgets {args.budgets}")
    if args.mode == "game":
        res = evaluate_active_obs(model, policy, base, data, indices, stats, rules=args.rules, budgets=args.budgets,
                                  horizon=args.horizon, obs_steps=args.obs_steps, base_rules=args.base_rules,
                                  device=args.device, log=print, progress_path=args.out + ".progress")
        for name, ch in res["change"].items():
            r = res["runs"][name]
            print(f"{name:22s} vs {res['reference']}: free {100 * ch['free']['mean'][0]:+.1f}% "
                  f"[{100 * ch['free']['lo'][0]:+.1f}, {100 * ch['free']['hi'][0]:+.1f}] all {100 * ch['all']['mean'][0]:+.1f}% "
                  f"| blow-ups {r['blowups']} | rho(freq, tau) {res['spearman_tau'].get(name, float('nan')):+.2f}")
    else:
        clim = pipeline.build_climatology(cfg, data)
        res = evaluate_active_climate(model, policy, base, data, indices, clim, stats, rules=args.rules,
                                      budget=args.budgets[0], spinup=spinup, weeks=args.weeks,
                                      base_rules=args.base_rules, device=args.device, log=print,
                                      progress_path=args.out + ".progress")
        for name, ch in res["change"].items():
            print(f"{name:22s} vs {res['reference']} weekly: " + " ".join(f"{100 * x:+.1f}%" for x in ch["mean"])
                  + f" | blow-ups {res['runs'][name]['blowups']}")
    res.update(checkpoint=args.checkpoint, base=args.base, policy=args.policy, split=args.split,
               period=[str(first), str(last)], random_policy=args.random_policy)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=1)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
