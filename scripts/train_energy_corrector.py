"""Stage 13 E2 / E3: train an energy corrector of the column tendencies on a frozen checkpoint.

    python scripts/train_energy_corrector.py --checkpoint runs/stage10_Dnoactor/checkpoint.pt \
        --config configs/energy_corrector_wb2.yaml --data-dir DATA2020 --out runs/stage13_E2 \
        --set model.column.energy_max_flux=500 model.column.energy_no_work=true

The checkpoint's model (column, memory) is kept and frozen; the corrector
(`corrector` section of the YAML, --set corrector.*) is added at zero and
learns by the free-run loss of train/memory_filter.train_filter (fill_steps
re-observed, free_steps free, 2018-2019 in sequence). The E1 bound goes in by
--set model.column.energy_max_flux=... and is saved with the checkpoint.
Resumable (--resume).
"""

import argparse
import os

import _path  # noqa: F401  (adds src/ to sys.path)
import torch

from climate_wm import pipeline
from climate_wm.train.memory_filter import FilterConfig, consecutive_windows, expand_state_dict, train_filter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="a stage 9 or 10 physics checkpoint")
    parser.add_argument("--config", required=True, help="data splits + corrector and filter sections")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--resume", action="store_true", help="continue from OUT/progress.pt")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    parser.add_argument("--windows", type=int, default=None, help="only the first N windows (smoke test)")
    parser.add_argument("--stride", type=int, default=1, help="every k-th window (a cheaper pass)")
    args = parser.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    base_cfg, state, stats = pipeline.load_checkpoint(args.checkpoint, args.device)
    update = pipeline.load_config(args.config, args.data_dir)
    cfg = pipeline.merge_config(base_cfg, {k: v for k, v in update.items() if k not in ("corrector", "filter")})
    cfg["corrector"] = dict(update.get("corrector", {}))
    cfg["filter"] = dict(update.get("filter", {}))
    cfg["filter"].setdefault("new_prefix", "corrector.")
    cfg["output_dir"] = args.out
    pipeline.apply_overrides(cfg, args.set)
    cfg["model"]["corrector"] = cfg["corrector"]
    fc = FilterConfig(**cfg["filter"])
    torch.manual_seed(fc.seed)
    layout = pipeline.build_layout(cfg)
    ds = pipeline.open_dataset(cfg)
    train = cfg["data"]["train"]
    data, order = consecutive_windows(ds, layout, stats, train["start"], train["end"], fc.steps)
    order = order[::max(1, args.stride)]
    if args.windows:
        order = order[:args.windows]
    print(f"{len(order)} windows of {fc.steps} steps ({train['start']} .. {train['end']}), "
          f"corrector {cfg['corrector']}, column energy "
          f"{ {k: v for k, v in cfg['model']['column'].items() if k.startswith('energy')} }, {fc}")
    model = pipeline.build_model(cfg, layout, data).to(args.device)
    expanded = expand_state_dict(model, state)
    print(f"loaded {args.checkpoint}; expanded: {expanded or 'nothing'}; corrector parameters "
          f"{sum(p.numel() for p in model.corrector.parameters()):,}")
    os.makedirs(args.out, exist_ok=True)
    progress_path = os.path.join(args.out, "progress.pt")
    progress = None
    if args.resume and os.path.exists(progress_path):
        progress = torch.load(progress_path, map_location=args.device, weights_only=False)

    def save(prog):
        torch.save(prog, progress_path + ".tmp")
        os.replace(progress_path + ".tmp", progress_path)

    history = train_filter(model, data, order, fc, args.device, progress=progress, save=save, log=print)
    pipeline.save_checkpoint(os.path.join(args.out, "checkpoint.pt"), cfg, model, stats, {"corrector": history})
    pipeline.save_json(os.path.join(args.out, "train_history.json"), history)
    if os.path.exists(progress_path):
        os.remove(progress_path)
    print(f"saved {args.out}/checkpoint.pt")


if __name__ == "__main__":
    main()
