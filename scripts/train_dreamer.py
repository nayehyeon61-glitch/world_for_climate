"""Stage 10: trains the physics world model with an adaptive memory kernel and an
actor-critic, DreamerV3-style, on the masked prediction game (2018-19).

    python scripts/train_dreamer.py --checkpoint runs/physics_n1/checkpoint.pt \
        --config configs/dreamer_wb2.yaml --data-dir DATA --out runs/stage10_D [--resume] [--set ...]

The base checkpoint is the stage 9 model (its weights load; the memory starts
at zero effect) or, for the frozen-world-model arm, a stage 10 checkpoint.
Writes OUT/checkpoint.pt (the world model with memory, loadable by
pipeline.load_checkpoint), OUT/agent.pt (actor and heads) and train_history.json.
"""

import argparse
import os

import _path  # noqa: F401  (adds src/ to sys.path)
import torch
from torch.utils.data import DataLoader

from climate_wm import pipeline
from climate_wm.data.era5_lazy_loader import GraphCastWindowDataset
from climate_wm.models.tendency_control import TendencyControlConfig, controller_state
from climate_wm.train.dreamer_game import DreamerConfig, Heads, MemoryController, train_dreamer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True, help="data splits + memory, dreamer and control sections")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--resume", action="store_true", help="continue from OUT/progress.pt")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    base_cfg, state, stats = pipeline.load_checkpoint(args.checkpoint, args.device)
    update = pipeline.load_config(args.config, args.data_dir)
    cfg = pipeline.merge_config(base_cfg, {k: v for k, v in update.items() if k not in ("memory", "dreamer", "control")})
    for key in ("memory", "dreamer", "control"):
        cfg[key] = dict(update.get(key, {}))
    if base_cfg["model"].get("memory"):  # a stage 10 checkpoint keeps its memory
        cfg["memory"] = dict(base_cfg["model"]["memory"])
    cfg["output_dir"] = args.out
    pipeline.apply_overrides(cfg, args.set)
    cfg["model"]["memory"] = cfg["memory"] if cfg["memory"].get("channels", 0) > 0 else None
    dc = DreamerConfig(**cfg["dreamer"])
    torch.manual_seed(dc.seed)
    layout = pipeline.build_layout(cfg)
    ds = pipeline.open_dataset(cfg)
    data = GraphCastWindowDataset(ds, layout, stats, dc.horizon, **cfg["data"]["train"])
    print(f"windows: {len(data)} ({data.times[data.indices[0]]} .. {data.times[data.indices[-1]]}), "
          f"memory {cfg['model']['memory']}, train_model={dc.train_model}, train_actor={dc.train_actor}")
    model = pipeline.build_model(cfg, layout, data).to(args.device)
    missing = model.load_state_dict(state, strict=False).missing_keys
    if any(not k.startswith("memory.") for k in missing):
        raise ValueError(f"checkpoint lacks {[k for k in missing if not k.startswith('memory.')]}")
    print(f"loaded {args.checkpoint}; new memory parameters: {len(missing)}")
    ctrl = heads = None
    if dc.train_actor:
        ctrl = MemoryController(model, TendencyControlConfig(**cfg["control"]), stats.diff_std).to(args.device)
        ctrl.cost_weight = ctrl.cfg.control_cost
        heads = Heads(ctrl.in_dim, ctrl.n_action, dc.head_hidden).to(args.device)
        print(f"actor: {sum(p.numel() for p in ctrl.parameters()):,} parameters, {ctrl.n_action} actions")
    if model.memory is not None:
        print(f"memory: {sum(p.numel() for p in model.memory.parameters()):,} parameters")
    loader = DataLoader(data, batch_size=dc.batch_size, shuffle=True, drop_last=True,
                        generator=torch.Generator().manual_seed(dc.seed))  # the same windows in every arm
    os.makedirs(args.out, exist_ok=True)
    progress_path = os.path.join(args.out, "progress.pt")
    progress = None
    if args.resume and os.path.exists(progress_path):
        progress = torch.load(progress_path, map_location=args.device, weights_only=False)

    def save(prog):
        torch.save(prog, progress_path + ".tmp")
        os.replace(progress_path + ".tmp", progress_path)

    history = train_dreamer(model, ctrl, heads, loader, dc, args.device, progress=progress, save=save)
    pipeline.save_checkpoint(os.path.join(args.out, "checkpoint.pt"), cfg, model, stats, {"dreamer": history})
    torch.save({"config": cfg, "base_checkpoint": args.checkpoint, "history": history,
                "controller": controller_state(ctrl) if ctrl is not None else None,
                "heads": heads.state_dict() if heads is not None else None},
               os.path.join(args.out, "agent.pt"))
    pipeline.save_json(os.path.join(args.out, "train_history.json"), history)
    print(f"saved {args.out}/checkpoint.pt and agent.pt")


if __name__ == "__main__":
    main()
