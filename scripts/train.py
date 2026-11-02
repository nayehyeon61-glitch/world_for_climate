"""Train the climate world model stage by stage.

    python scripts/train.py --config configs/smoke.yaml [--resume]
"""

import argparse
import os

import _path  # noqa: F401  (adds src/ to sys.path)
import torch

from climate_wm import pipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data-dir", default=None, help='replaces "{data_dir}" in the config')
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                        help="override config entries, e.g. model.lie.shell_coupling=independent")
    parser.add_argument("--resume", action="store_true",
                        help="continue from output_dir/checkpoint.pt: skips finished stages and "
                             "resumes an interrupted one from its last saved step")
    parser.add_argument("--init-from", default=None,
                        help="initialise weights from a checkpoint on another grid (1 deg -> 0.25 deg)")
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    cfg = pipeline.load_config(args.config, args.data_dir)
    pipeline.apply_overrides(cfg, args.set)
    path = os.path.join(cfg["output_dir"], "checkpoint.pt")
    state, stats, histories, progress = None, None, {}, None
    if args.resume and os.path.exists(path):
        _, state, stats = pipeline.load_checkpoint(path, args.device)
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        histories, progress = ckpt.get("histories", {}), ckpt.get("progress")
        at = f", {progress['stage']} at step {progress['step']}" if progress else ""
        print(f"resuming from {path}: finished stages {list(histories)}{at}")
    layout = pipeline.build_layout(cfg)
    print(layout.summary())
    ds = pipeline.open_dataset(cfg)
    train_ds, _, stats = pipeline.build_datasets(cfg, layout, ds, stats)
    model = pipeline.build_model(cfg, layout, train_ds).to(args.device)
    if state is not None:
        model.load_state_dict(state)
    elif args.init_from:
        pipeline.load_matching_state(model, args.init_from)
    print(model.graphs.summary())
    print(f"parameters: {sum(p.numel() for p in model.parameters()):,}")

    def checkpoint(done):  # after every stage, so an interrupted run can --resume
        pipeline.save_checkpoint(path, cfg, model, stats, done)
        pipeline.save_json(os.path.join(cfg["output_dir"], "train_history.json"), done)
        print(f"saved {path} (stages {list(done)})")

    def mid_stage(prog):  # every train.engine.save_every steps
        pipeline.save_checkpoint(path, cfg, model, stats, histories_so_far, prog)
        print(f"saved {path} ({prog['stage']} step {prog['step']})")

    histories_so_far = dict(histories)

    def stage_done(done):
        histories_so_far.clear()
        histories_so_far.update(done)
        checkpoint(done)

    histories = pipeline.train(cfg, model, train_ds, args.device, histories=histories,
                               on_stage_end=stage_done, progress=progress,
                               on_progress=mid_stage)
    if set(histories) != set(histories_so_far):  # e.g. a skipped stochastic stage
        checkpoint(histories)


if __name__ == "__main__":
    main()
