"""Stage 10.2: continuous-filter training of the memory model (train/memory_filter.py).

    python scripts/train_memory_filter.py --checkpoint runs/stage10_Dnoactor/checkpoint.pt \
        --config configs/memory_filter_wb2.yaml --data-dir DATA2020 --out runs/stage10_long \
        --set memory.long_channels=4            # L0: memory.long_channels=0 (the training change alone)

The checkpoint's memory is kept (its config under `memory` in the YAML and
--set adds or changes fields such as long_channels); the model then starts as
the checkpoint with the new readers of the slow channels at zero. Windows tile
data.train.start .. data.train.end back to back. Resumable (--resume).
"""

import argparse
import os

import _path  # noqa: F401  (adds src/ to sys.path)
import torch

from climate_wm import pipeline
from climate_wm.train.memory_filter import FilterConfig, consecutive_windows, expand_state_dict, train_filter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="a stage 10 checkpoint (memory kernel)")
    parser.add_argument("--config", required=True, help="data splits + memory and filter sections")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--resume", action="store_true", help="continue from OUT/progress.pt")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    parser.add_argument("--windows", type=int, default=None, help="only the first N windows (smoke test)")
    args = parser.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    base_cfg, state, stats = pipeline.load_checkpoint(args.checkpoint, args.device)
    update = pipeline.load_config(args.config, args.data_dir)
    cfg = pipeline.merge_config(base_cfg, {k: v for k, v in update.items() if k not in ("memory", "filter")})
    cfg["memory"] = dict(base_cfg["model"].get("memory") or {})
    cfg["memory"].update(update.get("memory", {}))
    cfg["filter"] = dict(update.get("filter", {}))
    cfg["output_dir"] = args.out
    pipeline.apply_overrides(cfg, args.set)
    if cfg["memory"].get("channels", 0) <= 0:
        parser.error("the checkpoint has no memory kernel")
    cfg["model"]["memory"] = cfg["memory"]
    fc = FilterConfig(**cfg["filter"])
    torch.manual_seed(fc.seed)
    layout = pipeline.build_layout(cfg)
    ds = pipeline.open_dataset(cfg)
    train = cfg["data"]["train"]
    data, order = consecutive_windows(ds, layout, stats, train["start"], train["end"], fc.steps)
    if args.windows:
        order = order[:args.windows]
    print(f"{len(order)} windows of {fc.steps} steps ({train['start']} .. {train['end']}), "
          f"memory {cfg['memory']}, {fc}")
    model = pipeline.build_model(cfg, layout, data).to(args.device)
    expanded = expand_state_dict(model, state)
    print(f"loaded {args.checkpoint}; expanded: {expanded or 'nothing'}; memory parameters "
          f"{sum(p.numel() for p in model.memory.parameters()):,}")
    os.makedirs(args.out, exist_ok=True)
    progress_path = os.path.join(args.out, "progress.pt")
    progress = None
    if args.resume and os.path.exists(progress_path):
        progress = torch.load(progress_path, map_location=args.device, weights_only=False)

    def save(prog):
        torch.save(prog, progress_path + ".tmp")
        os.replace(progress_path + ".tmp", progress_path)

    history = train_filter(model, data, order, fc, args.device, progress=progress, save=save, log=print)
    pipeline.save_checkpoint(os.path.join(args.out, "checkpoint.pt"), cfg, model, stats, {"filter": history})
    pipeline.save_json(os.path.join(args.out, "train_history.json"), history)
    if os.path.exists(progress_path):
        os.remove(progress_path)
    print(f"saved {args.out}/checkpoint.pt")


if __name__ == "__main__":
    main()
