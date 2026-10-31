"""Stage 10: scores a trained arm in the masked prediction game (2020, or the 2018-19 game years).

    python scripts/eval_dreamer.py --checkpoint runs/stage10_D/checkpoint.pt \
        --base runs/physics_n1/checkpoint.pt --config configs/dreamer_wb2.yaml --data-dir DATA \
        --variants full no_actor no_memory ... --out runs/stage10_D/eval_2020.json [--split train --inits 15]

Every initialisation gets one fixed mask per kind and one fixed sequence of
re-observations, so the arm, its knock-outs (train/dreamer_game.VARIANTS) and
the baselines on the stage 9 model (nudging = the reference, nudging_half,
game_rule) are paired; runs with the same --inits and --split pair across arms.
--checkpoint may be the stage 9 model itself (baselines only).
"""

import argparse
import json
import os

import _path  # noqa: F401  (adds src/ to sys.path)
import numpy as np
import torch

from climate_wm import pipeline
from climate_wm.data.era5_lazy_loader import GraphCastWindowDataset
from climate_wm.models.tendency_control import TendencyControlConfig
from climate_wm.train.dreamer_game import VARIANTS, DreamerConfig, MemoryController, evaluate_memory_game


def _load(path, update, device, overrides):
    cfg, state, stats = pipeline.load_checkpoint(path, device)
    cfg = pipeline.merge_config(cfg, {"data": update["data"]})
    pipeline.apply_overrides(cfg, overrides)
    return cfg, state, stats


def load_arm(checkpoint, base_path, update, split, target_steps, device="cpu", overrides=()):
    """The arm (model with its memory), its actor (agent.pt beside the
    checkpoint, None without one), the stage 9 base model and the split's
    windows of `target_steps` 6 h targets."""
    cfg, state, stats = _load(checkpoint, update, device, overrides)
    layout = pipeline.build_layout(cfg)
    ds = pipeline.open_dataset(cfg)
    data = GraphCastWindowDataset(ds, layout, stats, target_steps, **cfg["data"][split])
    model = pipeline.build_model(cfg, layout, data).to(device)
    model.load_state_dict(state)
    bcfg, bstate, _ = _load(base_path, update, device, overrides)
    base = pipeline.build_model(bcfg, layout, data).to(device)
    base.load_state_dict(bstate)
    ctrl = None
    agent_path = os.path.join(os.path.dirname(checkpoint), "agent.pt")
    if os.path.exists(agent_path):
        agent = torch.load(agent_path, map_location=device, weights_only=False)
        if agent.get("controller"):
            ctrl = MemoryController(model, TendencyControlConfig(**agent["controller"]["cfg"]))
            ctrl.load_state_dict(agent["controller"]["state"])
            ctrl = ctrl.to(device)
            ctrl.cost_weight = 0.0
    return cfg, model, base, ctrl, data, stats


def usable(variants, model, ctrl):
    """Drops the variants an arm cannot run (memory knock-outs without a memory, no actor to switch off)."""
    return [v for v in variants
            if not (VARIANTS[v][2] == "own" and ((model.memory is None and VARIANTS[v][0])
                                                 or (ctrl is None and v == "no_actor")))]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="stage 10 arm (OUT/checkpoint.pt, agent.pt beside it)")
    parser.add_argument("--base", required=True, help="the stage 9 model (baselines)")
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--split", choices=["val", "train"], default="val")
    parser.add_argument("--inits", type=int, default=31)
    parser.add_argument("--kinds", nargs="*", default=["variables", "shell", "region"])
    parser.add_argument("--variants", nargs="*", default=list(VARIANTS))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--out", required=True)
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    update = pipeline.load_config(args.config, args.data_dir)
    dc = DreamerConfig(**update.get("dreamer", {}))
    cfg, model, base, ctrl, data, _ = load_arm(args.checkpoint, args.base, update, args.split, dc.horizon,
                                               args.device, args.set)
    variants = usable(args.variants, model, ctrl)
    indices = np.linspace(0, len(data) - 1, min(args.inits, len(data))).astype(int)
    first, last = data.times[data.indices[indices[0]]], data.times[data.indices[indices[-1]]]
    print(f"{len(indices)} initialisations ({first} .. {last}), kinds {args.kinds}, variants {variants}, "
          f"actor {'yes' if ctrl is not None else 'no'}, memory {cfg['model'].get('memory')}")
    res = evaluate_memory_game(model, ctrl, base, data, indices, args.kinds, dc.horizon, dc.obs_prob,
                               args.device, variants=variants, region_fraction=dc.region_fraction, log=print)
    res.update(checkpoint=args.checkpoint, base=args.base, split=args.split, period=[str(first), str(last)],
               inits=len(indices))
    for kind, r in res["kinds"].items():
        line = f"{kind:10s} 24 h / 36 h hidden vs nudging:"
        for name, ch in r["change"].items():
            if name.endswith("_hidden"):
                line += f" | {name[:-7]} {100 * ch['mean'][3]:+.1f}% / {100 * ch['mean'][-1]:+.1f}%"
        print(line)
    if "log10_tau_hours_p10_p50_p90" in res:
        print("timescales (h) p10/p50/p90:", [f"{10 ** x:.1f}" for x in res["log10_tau_hours_p10_p50_p90"]])
        spread = [10 ** (q[2] - q[0]) for q in res["log10_tau_hours_channel_p10_p50_p90"]]
        print(f"state dependence, p90/p10 of tau per channel: median {np.median(spread):.2f}, "
              f"max {max(spread):.2f}")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=1)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
