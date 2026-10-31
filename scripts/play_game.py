"""Stage 7: trains a feature controller on a frozen world model by playing the prediction game.

    python scripts/play_game.py --checkpoint runs/wb2_pyramid_nodrop/checkpoint.pt \
        --config configs/game_wb2.yaml --data-dir DATA --mode rl --out runs/game_c1
    python scripts/play_game.py ... --mode grad --out runs/game_c1_grad   # the gradient control

Stage 9 (N3): a physics checkpoint (model.type: physics) gets the tendency
controller of models/tendency_control.py and the same game on the physics model
(configs/physics_game_wb2.yaml).

Only the controller learns (climate_wm.train.prediction_game); the world model
checkpoint is not changed. Writes controller.pt and game_history.json.
"""

import argparse
import os

import _path  # noqa: F401  (adds src/ to sys.path)
import torch
from torch.utils.data import DataLoader

from climate_wm import pipeline
from climate_wm.data.era5_lazy_loader import GraphCastWindowDataset
from climate_wm.models import tendency_control
from climate_wm.models.feature_control import ControlConfig, FeatureController, controller_state
from climate_wm.train.physics_game import play_physics
from climate_wm.train.prediction_game import GameConfig, train_controller


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="frozen world model")
    parser.add_argument("--config", required=True, help="data splits + game + control sections")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--mode", choices=["rl", "grad"], default=None)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resume", action="store_true", help="continue from OUT/progress.pt")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()
    base_cfg, state, stats = pipeline.load_checkpoint(args.checkpoint, args.device)
    update = pipeline.load_config(args.config, args.data_dir)
    cfg = pipeline.merge_config(base_cfg, {k: v for k, v in update.items() if k not in ("game", "control")})
    cfg["game"] = dict(update.get("game", {}), **({"mode": args.mode} if args.mode else {}))
    cfg["control"] = dict(update.get("control", {}))
    cfg["output_dir"] = args.out
    pipeline.apply_overrides(cfg, args.set)
    physics = cfg["model"].get("type") == "physics"
    gc = GameConfig(**cfg["game"])
    cc = (tendency_control.TendencyControlConfig if physics else ControlConfig)(**cfg["control"])
    torch.manual_seed(gc.seed)

    layout = pipeline.build_layout(cfg)
    ds = pipeline.open_dataset(cfg)
    data = GraphCastWindowDataset(ds, layout, stats, gc.horizon, **cfg["data"]["train"])
    print(f"game windows: {len(data)} ({data.times[data.indices[0]]} .. "
          f"{data.times[data.indices[-1]]}), mode={gc.mode}")
    model = pipeline.build_model(cfg, layout, data).to(args.device)
    model.load_state_dict(state)
    if physics:
        for p in model.parameters():
            p.requires_grad_(False)
        ctrl = tendency_control.TendencyController(model, cc, stats.diff_std).to(args.device)
        ctrl.cost_weight = cc.control_cost
        print(f"controller: {sum(p.numel() for p in ctrl.parameters()):,} parameters, "
              f"{ctrl.n_action} actions (n <= {cc.max_degree}, {len(ctrl.groups)} layers); action unit "
              + ", ".join(f"{f} {list(map('{:.2g}'.format, u.tolist()))}" for f, u in zip(ctrl.layer_fields, ctrl.unit))
              + f", phib {float(ctrl.unit_phib):.3g}")
    else:
        ctrl = FeatureController(model, data.lat, data.lon, cc, stats).to(args.device)
        print(f"controller: {sum(p.numel() for p in ctrl.parameters()):,} parameters, "
              f"{ctrl.num_shells} shells x {ctrl.num_patches} patches x {2 * cc.rank} actions")
    loader = DataLoader(data, batch_size=gc.batch_size, shuffle=True, drop_last=True,
                        generator=torch.Generator().manual_seed(gc.seed))  # same windows in every arm
    os.makedirs(args.out, exist_ok=True)
    progress_path = os.path.join(args.out, "progress.pt")
    progress = None
    if args.resume and os.path.exists(progress_path):
        progress = torch.load(progress_path, map_location=args.device, weights_only=False)

    def save(state):
        torch.save(state, progress_path + ".tmp")
        os.replace(progress_path + ".tmp", progress_path)

    history = train_controller(model, ctrl, loader, gc, args.device, progress=progress, save=save,
                               play_fn=play_physics if physics else None)
    path = os.path.join(args.out, "controller.pt")
    state_fn = tendency_control.controller_state if physics else controller_state
    torch.save({"config": cfg, "base_checkpoint": args.checkpoint,
                "controller": state_fn(ctrl), "history": history}, path)
    pipeline.save_json(os.path.join(args.out, "game_history.json"), history)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
