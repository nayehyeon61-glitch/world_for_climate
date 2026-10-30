"""Train the climate world model stage by stage.

    python scripts/train.py --config configs/smoke.yaml
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
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    cfg = pipeline.load_config(args.config)
    layout = pipeline.build_layout(cfg)
    print(layout.summary())
    ds = pipeline.open_dataset(cfg)
    train_ds, _, stats = pipeline.build_datasets(cfg, layout, ds)
    model = pipeline.build_model(cfg, layout, train_ds).to(args.device)
    print(model.graphs.summary())
    print(f"parameters: {sum(p.numel() for p in model.parameters()):,}")
    pipeline.train(cfg, model, train_ds, args.device)
    path = os.path.join(cfg["output_dir"], "checkpoint.pt")
    pipeline.save_checkpoint(path, cfg, model, stats)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
