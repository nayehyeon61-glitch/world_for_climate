"""Stage 7: scores feature controllers in the masked prediction game on the held-out year.

    python scripts/eval_game.py --checkpoint runs/wb2_pyramid_nodrop/checkpoint.pt \
        --config configs/game_wb2.yaml --data-dir DATA \
        C1=runs/game_c1/controller.pt C1-grad=runs/game_c1_grad/controller.pt [--inits 31] [--out game_eval.json]

Every initialisation gets one fixed mask per kind (and, with the game's
obs_prob > 0, one fixed sequence of re-observations); the controller, the frozen
model (no_control), the controller's actions at shuffled places
(random_control) and the controller without its diagnostics (no_diag) play
the same episodes, so changes are paired over initialisations. The unmasked
6-72 h forecast of the same arms comes from scripts/eval_causes.py --controller.
"""

import argparse
import json
import os

import _path  # noqa: F401  (adds src/ to sys.path)
import numpy as np
import torch

from climate_wm import pipeline
from climate_wm.data.era5_lazy_loader import GraphCastWindowDataset
from climate_wm.models.feature_control import load_controller
from climate_wm.train.prediction_game import CONTROL_VARIANTS, GameConfig, evaluate_game


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("controllers", nargs="+", help="NAME=controller.pt")
    parser.add_argument("--checkpoint", required=True, help="the frozen world model")
    parser.add_argument("--config", required=True, help="data splits + game section (configs/game_wb2.yaml)")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--inits", type=int, default=31)
    parser.add_argument("--kinds", nargs="*", default=["none", "variables", "shell", "region"])
    parser.add_argument("--variants", nargs="*", default=list(CONTROL_VARIANTS),
                        help="no_control is always played (the reference)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", required=True)
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()
    base_cfg, state, stats = pipeline.load_checkpoint(args.checkpoint, args.device)
    update = pipeline.load_config(args.config, args.data_dir)
    cfg = pipeline.merge_config(base_cfg, {"data": update["data"]})
    pipeline.apply_overrides(cfg, args.set)
    gc = GameConfig(**update.get("game", {}))
    layout = pipeline.build_layout(cfg)
    ds = pipeline.open_dataset(cfg)
    test = GraphCastWindowDataset(ds, layout, stats, gc.horizon, **cfg["data"]["val"])
    model = pipeline.build_model(cfg, layout, test).to(args.device)
    model.load_state_dict(state)
    indices = np.linspace(0, len(test) - 1, min(args.inits, len(test))).astype(int)
    first, last = test.times[test.indices[indices[0]]], test.times[test.indices[indices[-1]]]
    print(f"{len(indices)} initialisations ({first} .. {last}), kinds {args.kinds}")
    out = {"checkpoint": args.checkpoint, "period": [str(first), str(last)], "inits": len(indices),
           "region_fraction": gc.region_fraction, "arms": {}}
    for item in args.controllers:
        name, path = item.split("=", 1)
        saved = torch.load(path, map_location=args.device, weights_only=False)
        ctrl = load_controller(model, test.lat, test.lon, saved["controller"], stats)
        res = evaluate_game(model, ctrl, test, indices, args.kinds, gc.horizon, args.device,
                            region_fraction=gc.region_fraction,
                            variants=["no_control", *[v for v in args.variants if v != "no_control"]],
                            obs_prob=gc.obs_prob, log=print)
        res.update(controller=path, history=saved.get("history", []))
        out["arms"][name] = res
        for kind, r in res["kinds"].items():
            ch = r["change"]
            line = (f"{name:10s} {kind:10s} 24h hidden {100 * ch['control_hidden']['mean'][-1]:+.2f}% "
                    f"[{100 * ch['control_hidden']['lo'][-1]:+.2f}, {100 * ch['control_hidden']['hi'][-1]:+.2f}] "
                    f"all {100 * ch['control_all']['mean'][-1]:+.2f}%")
            for v in ("random_control", "no_diag"):
                if f"{v}_hidden" in ch:
                    line += f" | {v} {100 * ch[f'{v}_hidden']['mean'][-1]:+.2f}%"
            print(line)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
