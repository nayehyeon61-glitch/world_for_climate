"""Stage 8 phase 1: transition-cost topology of the frozen world model's states.

    python scripts/eval_topology.py encode --checkpoint runs/wb2_pyramid_nodrop/checkpoint.pt \
        --config configs/topology_wb2.yaml --data-dir DATA --out runs/topology
    python scripts/eval_topology.py analyse --config configs/topology_wb2.yaml --out runs/topology

encode   every 6 h state of the data range through the frozen encoder:
         latent invariants, the forcing of each transition, analog fields
         (z500, t850, t2m) and dry-energy features -> OUT/states.npz; plus the
         world model's own deterministic z500 error at `wm_steps` for every
         `wm_stride`-th test state (for the predictability check)
analyse  the commute-time heads of the config's arms and tests T1-T3
         (climate_wm.eval.topology.analyse) -> OUT/topology.json

The world model checkpoint is not changed.
"""

import argparse
import os

import _path  # noqa: F401  (adds src/ to sys.path)
import numpy as np
import torch

from climate_wm import pipeline
from climate_wm.data.era5_lazy_loader import GraphCastWindowDataset
from climate_wm.eval import topology
from climate_wm.geometry.shells import SpherePatches
from climate_wm.models.commute_time import CommuteConfig


def encode(args, cfg):
    tc = cfg["topology"]
    base_cfg, state, stats = pipeline.load_checkpoint(args.checkpoint, args.device)
    cfg = pipeline.merge_config(base_cfg, {"data": cfg["data"]})
    pipeline.apply_overrides(cfg, args.set)
    layout = pipeline.build_layout(cfg)
    ds = pipeline.open_dataset(cfg)
    data = GraphCastWindowDataset(ds, layout, stats, 1)
    print(f"states: {len(data)} ({data.times[data.indices[0]]} .. {data.times[data.indices[-1]]})")
    model = pipeline.build_model(cfg, layout, data).to(args.device)
    model.load_state_dict(state)
    model.eval()
    patches = SpherePatches(model.graphs.latent_pos.cpu().numpy(), tc.get("patch_splits", 0)).to(args.device)
    store = topology.encode_states(model, data, stats, patches, tc.get("batch_size", 4), log=print)
    wm = GraphCastWindowDataset(ds, layout, stats, tc["wm_steps"], start=tc["train_end"], end=tc["test_end"])
    ks = list(range(0, len(wm), tc["wm_stride"]))
    print(f"world model {6 * tc['wm_steps']} h errors at {len(ks)} test states")
    store["wm_err"] = topology.world_model_errors(model, wm, stats, ks, tc["wm_steps"])
    t = np.asarray(store["times"], "datetime64[s]")
    store["wm_rows"] = np.searchsorted(t, np.asarray(wm.times[wm.indices[ks]], "datetime64[s]"))
    path = os.path.join(args.out, "states.npz")
    np.savez(path, **store)
    print(f"saved {path}: " + ", ".join(f"{k} {tuple(v.shape)}" for k, v in store.items()))


def analyse(args, cfg):
    tc = cfg["topology"]
    raw = np.load(os.path.join(args.out, "states.npz"))
    store = {k: raw[k] for k in raw.files}
    arms = {name: CommuteConfig(**{**cfg.get("head", {}), **(over or {})}) for name, over in cfg["arms"].items()}
    if args.arms:
        arms = {k: v for k, v in arms.items() if k in args.arms}
    res = topology.analyse(store, arms, train_end=tc["train_end"], test_end=tc["test_end"],
                           n_clusters=tc["n_clusters"], leads=tuple(tc["leads"]), pca_dim=tc["pca_dim"],
                           topo_dim=tc["topo_dim"], analogs=tc["analogs"], n_boot=tc["n_boot"], log=print)
    res["config"] = {"topology": tc, "head": cfg.get("head", {}), "arms": cfg["arms"]}
    path = os.path.join(args.out, "topology.json")
    pipeline.save_json(path, res)
    print(f"saved {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["encode", "analyse"])
    parser.add_argument("--checkpoint", help="frozen world model (encode)")
    parser.add_argument("--config", required=True, help="data + topology + head + arms sections")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--out", required=True)
    parser.add_argument("--arms", nargs="*", default=None, help="analyse only these arms")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="overrides of the model config")
    args = parser.parse_args()
    cfg = pipeline.load_config(args.config, args.data_dir)
    torch.set_num_threads(cfg["topology"].get("threads", 1))
    os.makedirs(args.out, exist_ok=True)
    (encode if args.phase == "encode" else analyse)(args, cfg)


if __name__ == "__main__":
    main()
