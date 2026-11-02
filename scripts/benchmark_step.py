"""Measures step time and peak memory of every training stage without data.

    python scripts/benchmark_step.py --config configs/p2_025deg_96gb.yaml --device cuda
    python scripts/benchmark_step.py --config configs/p2_1deg_96gb.yaml --device cuda \
        --set model.gnn.hidden_dim=384 train.batch_size=2

Builds the model on the configured grid (`benchmark.resolution`, default the
archive grid) and trains a few steps of each stage on random batches of the
real shapes, then times one ensemble rollout step. Run it on the GPU before a
long job to pick mesh / width / batch size that fit in memory.
"""

import argparse
import resource
import time

import _path  # noqa: F401  (adds src/ to sys.path)
import numpy as np
import torch

from climate_wm import pipeline
from climate_wm.train.trainer import EngineConfig, StageConfig, train_stage


class _Grid:
    def __init__(self, resolution: float):
        n_lat = int(round(180 / resolution)) + 1
        self.lat = np.linspace(-90, 90, n_lat).astype(np.float32)
        self.lon = np.arange(0, 360, resolution).astype(np.float32)


def fake_batch(layout, grid, batch, steps, device):
    h, w = len(grid.lat), len(grid.lon)
    n_in = layout.task.input_steps
    r = lambda *s: torch.randn(*s, device=device)
    return {
        "inputs": r(batch, n_in, len(layout.dynamic), h, w),
        "forcings_in": r(batch, n_in, len(layout.forcings), h, w),
        "targets": r(batch, steps, len(layout.targets), h, w),
        "forcings_tgt": r(batch, steps, len(layout.forcings), h, w),
        "static": r(batch, len(layout.static), h, w),
        "init_time": torch.zeros(batch, dtype=torch.long, device=device),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resolution", type=float, default=None, help="grid spacing in degrees")
    parser.add_argument("--steps", type=int, default=3, help="optimizer steps per stage")
    parser.add_argument("--stages", nargs="*", default=None)
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()
    cfg = pipeline.load_config(args.config)
    pipeline.apply_overrides(cfg, args.set)
    res = args.resolution or cfg.get("benchmark", {}).get("resolution", 1.0)
    grid = _Grid(res)
    layout = pipeline.build_layout(cfg)
    dev = torch.device(args.device)
    t = time.time()
    model = pipeline.build_model(cfg, layout, grid).to(dev)
    print(f"grid {len(grid.lat)} x {len(grid.lon)} ({res} deg), graphs built in {time.time() - t:.0f}s")
    print(model.graphs.summary())
    print(f"parameters: {sum(p.numel() for p in model.parameters()):,}")
    engine = EngineConfig(**cfg["train"].get("engine", {}))
    model.set_grad_checkpointing(engine.grad_checkpointing)
    engine.save_every = 0
    bsz = cfg["train"]["batch_size"]
    rows = []
    for stage in cfg["train"]["stages"]:
        if args.stages and stage["name"] not in args.stages:
            continue
        if stage["name"] == "stochastic" and model.denoiser is None:
            continue
        sc = StageConfig(**{**stage, "steps": args.steps + 1, "warmup_steps": 0})
        batch = fake_batch(layout, grid, bsz, sc.rollout_steps, dev)
        if dev.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats(dev)
        times = []

        def log(msg, _t=[time.time()]):
            now = time.time()
            times.append(now - _t[0])
            _t[0] = now
            print("  " + msg)

        train_stage(model, [batch], sc, dev, log_every=1, log=log, engine=engine)
        step_s = float(np.median(times[1:])) if len(times) > 1 else float("nan")
        peak = (torch.cuda.max_memory_allocated(dev) / 2 ** 30 if dev.type == "cuda"
                else resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20)
        rows.append((stage["name"], sc.rollout_steps, step_s, peak))
    members = cfg["eval"].get("members", 1)
    batch = fake_batch(layout, grid, 1, 2, dev)
    with torch.no_grad(), torch.autocast(dev.type, dtype=torch.bfloat16, enabled=engine.amp == "bf16"):
        model.rollout(batch, 1, members=members)  # warm-up
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t = time.time()
        model.rollout(batch, 2, members=members)
        if dev.type == "cuda":
            torch.cuda.synchronize()
    lead_s = (time.time() - t) / 2
    unit = "GB (CUDA peak)" if dev.type == "cuda" else "GB (process peak RSS)"
    print(f"\n{'stage':<12}{'rollout':>8}{'s/step':>10}   peak {unit}")
    for name, roll, s, m in rows:
        print(f"{name:<12}{roll:>8}{s:>10.2f}   {m:.1f}")
    print(f"eval: {lead_s:.2f} s per 6h lead with {members} members "
          f"({lead_s * cfg['eval']['lead_steps']:.1f} s per initialisation)")


if __name__ == "__main__":
    main()
