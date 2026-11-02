"""Stage 12: trains the observation policy on the frozen memory model (2018-19 windows).

    python scripts/train_active_obs.py --checkpoint runs/stage10_Dnoactor/checkpoint.pt \
        --config configs/active_obs_wb2.yaml --data-dir DATA --out runs/stage12_P [--resume] [--set ...]

Writes OUT/policy.pt (the policy, its config, the base checkpoint path and the
training history) and OUT/train_history.json. The world model does not change.
"""

import argparse
import dataclasses
import os

import _path  # noqa: F401  (adds src/ to sys.path)
import torch
from torch.utils.data import DataLoader

from climate_wm import pipeline
from climate_wm.data.era5_lazy_loader import GraphCastWindowDataset
from climate_wm.train.active_obs import ActiveConfig, ObservationPolicy, ObsPolicyConfig, Patches, train_policy


def build_policy(model, data, pcfg: dict) -> ObservationPolicy:
    cfg = ObsPolicyConfig(**pcfg)
    patches = Patches(data.lat, data.lon, model.lat_g, model.lon_g, cfg.bands, cfg.sectors)
    return ObservationPolicy(model.layout, patches, cfg)


def load_world_model(checkpoint, update, split, target_steps, device="cpu", overrides=()):
    """The world model of a checkpoint and the split's windows of `target_steps` 6 h targets."""
    cfg, state, stats = pipeline.load_checkpoint(checkpoint, device)
    cfg = pipeline.merge_config(cfg, {k: update[k] for k in ("data", "eval") if k in update})
    for key in ("policy", "active"):
        cfg[key] = dict(update.get(key, {}))
    pipeline.apply_overrides(cfg, overrides)
    layout = pipeline.build_layout(cfg)
    ds = pipeline.open_dataset(cfg)
    data = GraphCastWindowDataset(ds, layout, stats, target_steps, **cfg["data"][split])
    model = pipeline.build_model(cfg, layout, data).to(device)
    model.load_state_dict(state)
    model.eval()
    return cfg, model, data, stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="the world model (stage 10 memory model)")
    parser.add_argument("--config", required=True, help="data splits + policy and active sections")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--resume", action="store_true", help="continue from OUT/progress.pt")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    update = pipeline.load_config(args.config, args.data_dir)
    horizon = int(pipeline.merge_config(update.get("active", {}), _overrides(args.set, "active")).get("horizon", 10))
    cfg, model, data, stats = load_world_model(args.checkpoint, update, "train", horizon, args.device, args.set)
    ac = ActiveConfig(**cfg["active"])
    torch.manual_seed(ac.seed)
    policy = build_policy(model, data, cfg["policy"]).to(args.device)
    print(f"windows: {len(data)} ({data.times[data.indices[0]]} .. {data.times[data.indices[-1]]}), "
          f"memory {cfg['model'].get('memory')}, patches {policy.patches.n}, budget {policy.n_obs()} patches, "
          f"policy {sum(p.numel() for p in policy.parameters()):,} parameters, {ac}")
    loader = DataLoader(data, batch_size=1, shuffle=True, generator=torch.Generator().manual_seed(ac.seed))
    os.makedirs(args.out, exist_ok=True)
    progress_path = os.path.join(args.out, "progress.pt")
    progress = None
    if args.resume and os.path.exists(progress_path):
        progress = torch.load(progress_path, map_location=args.device, weights_only=False)

    def save(prog):
        torch.save(prog, progress_path + ".tmp")
        os.replace(progress_path + ".tmp", progress_path)

    history = train_policy(model, policy, loader, ac, args.device, progress=progress, save=save)
    counts = torch.load(progress_path, weights_only=False).get("counts") if os.path.exists(progress_path) else None
    torch.save({"config": cfg, "policy_cfg": dataclasses.asdict(policy.cfg), "state": policy.state_dict(),
                "base_checkpoint": args.checkpoint, "history": history, "counts": counts},
               os.path.join(args.out, "policy.pt"))
    pipeline.save_json(os.path.join(args.out, "train_history.json"), history)
    print(f"saved {args.out}/policy.pt")


def _overrides(sets, section):
    out = {}
    for item in sets:
        key, _, value = item.partition("=")
        if key.startswith(section + "."):
            try:
                out[key[len(section) + 1:]] = int(value)
            except ValueError:
                pass
    return out


if __name__ == "__main__":
    main()
