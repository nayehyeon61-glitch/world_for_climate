"""Reinforces a trained world model on new years (rl) or fine-tunes it (supervised control).

    python scripts/refine.py --checkpoint runs/wb2_64x32/checkpoint.pt \
        --config configs/refine_wb2.yaml --data-dir data/wb2_2018_2020 --mode rl --out runs/refine_rl
    python scripts/refine.py ... --mode supervised --out runs/refine_sup

Both modes see the same windows in the same order and train only the denoiser;
`rl` adds the advantage-weighted regression term (climate_wm.train.rl_refine).
The saved checkpoint carries the new data splits, so compare_graphcast.py
evaluates it on the held-out year directly.
"""

import argparse
import os

import _path  # noqa: F401  (adds src/ to sys.path)
import torch
from torch.utils.data import DataLoader

from climate_wm import pipeline
from climate_wm.data.era5_lazy_loader import GraphCastWindowDataset
from climate_wm.train.rl_refine import RefineConfig, refine


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True, help="data splits + refine section")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--mode", choices=["rl", "supervised"], default=None)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()
    base_cfg, state, stats = pipeline.load_checkpoint(args.checkpoint, args.device)
    update = pipeline.load_config(args.config, args.data_dir)
    cfg = pipeline.merge_config(base_cfg, {k: v for k, v in update.items() if k != "refine"})
    cfg["refine"] = dict(update.get("refine", {}), **({"mode": args.mode} if args.mode else {}))
    cfg["output_dir"] = args.out
    pipeline.apply_overrides(cfg, args.set)
    rc = RefineConfig(**cfg["refine"])
    torch.manual_seed(rc.seed)

    layout = pipeline.build_layout(cfg)
    ds = pipeline.open_dataset(cfg)
    data = GraphCastWindowDataset(ds, layout, stats, rc.horizon, **cfg["data"]["train"])
    print(f"refinement windows: {len(data)} ({data.times[data.indices[0]]} .. "
          f"{data.times[data.indices[-1]]}), mode={rc.mode}")
    model = pipeline.build_model(cfg, layout, data).to(args.device)
    model.load_state_dict(state)
    loader = DataLoader(data, batch_size=rc.batch_size, shuffle=True, drop_last=True,
                        generator=torch.Generator().manual_seed(rc.seed))  # same windows in both modes
    history = refine(model, loader, rc, args.device)
    path = os.path.join(args.out, "checkpoint.pt")
    histories = dict(torch.load(args.checkpoint, map_location="cpu",
                                weights_only=False).get("histories", {}))
    histories[f"refine_{rc.mode}"] = history
    pipeline.save_checkpoint(path, cfg, model, stats, histories)
    pipeline.save_json(os.path.join(args.out, "refine_history.json"), history)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
