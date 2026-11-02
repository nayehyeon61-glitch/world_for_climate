"""Compute scaling of the stage 9 physics model (and the stage 11 operator bank) with resolution.

    python scripts/benchmark_physics.py --checkpoint runs/physics_n1/checkpoint.pt --trunc 42 [--hidden 512]

Builds N1's model at spectral truncation `--trunc` (quadratic Gaussian grid,
data grid of the same size) with the checkpoint's channels and normalisation,
on random states near the mean. It times:
  - one 6 h forecast step without gradients;
  - one training step (a one-step rollout, forward and backward with gradient
    checkpointing);
  - one call of a K-operator SFNO bank (models/operator_control.py) on the core grid.
It prints one JSON line with the timings and the peak resident memory. Run
each size in its own process so the peak memory is per size; the time step
follows the stage 9 configs (30 min at T31, 20 at T42, 10 at T119).
"""

import argparse
import json
import resource
import time

import _path  # noqa: F401  (adds src/ to sys.path)
import numpy as np
import torch

from climate_wm import pipeline
from climate_wm.models.physics_world_model import PhysicsModelConfig, PhysicsWorldModel

DT = {21: 40.0, 31: 30.0, 42: 20.0, 63: 15.0, 85: 12.0, 119: 10.0}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--trunc", type=int, default=31)
    parser.add_argument("--hidden", type=int, default=None, help="column width (default: the checkpoint's)")
    parser.add_argument("--operators", type=int, default=4, help="SFNO bank size (0: skip)")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    cfg, state, stats = pipeline.load_checkpoint(args.checkpoint, "cpu")
    t = args.trunc
    nlat = 2 * ((3 * (t + 1) // 2 + 1) // 2)  # quadratic Gaussian grid, even
    mcfg = dict(cfg["model"])
    mcfg["core"] = dict(mcfg.get("core", {}), trunc=t, nlat=nlat, nlon=2 * nlat, dt_minutes=DT.get(t, 30.0))
    if args.hidden:
        mcfg["column"] = dict(mcfg.get("column", {}), hidden=args.hidden)
    layout = pipeline.build_layout(cfg)
    d = 180.0 / nlat
    lat = np.linspace(-90 + d / 2, 90 - d / 2, nlat)
    lon = np.arange(2 * nlat) * d
    model = PhysicsWorldModel(PhysicsModelConfig.from_dict(mcfg), layout, lat, lon)
    model.set_normalization(stats)
    params = sum(p.numel() for p in model.parameters())
    h, w = len(lat), len(lon)
    gen = torch.Generator().manual_seed(0)
    r = lambda *s: 0.1 * torch.randn(*s, generator=gen)
    n_in = layout.task.input_steps
    batch = {"inputs": r(1, n_in, len(layout.dynamic), h, w), "forcings_in": r(1, n_in, len(layout.forcings), h, w),
             "targets": r(1, 1, len(layout.targets), h, w), "forcings_tgt": r(1, 1, len(layout.forcings), h, w),
             "static": r(1, len(layout.static), h, w),
             "init_time": torch.tensor([int(np.datetime64("2018-01-01T00", "s").astype(np.int64))])}

    def timed(fn):
        fn()  # warm-up (tables, caches)
        out = []
        for _ in range(args.repeats):
            t0 = time.time()
            fn()
            out.append(time.time() - t0)
        return float(np.median(out))

    with torch.no_grad():
        fwd = timed(lambda: model.forward_rollout(batch, 1))
    model.train()

    def train_step():
        model.zero_grad(set_to_none=True)
        pred = model.forward_rollout(batch, 1)
        (pred - batch["targets"]).square().mean().backward()

    trn = timed(train_step)
    model.eval()
    res = {"trunc": t, "grid": [nlat, 2 * nlat], "data_grid": [h, w], "dt_min": DT.get(t, 30.0),
           "column_hidden": model.cfg.column.hidden, "params": params, "forecast_6h_s": fwd, "train_step_s": trn}
    if args.operators:
        from climate_wm.models.operator_control import _sfno
        from climate_wm.models.tendency_control import MASK_GROUPS
        n_in = 4 * 4 + 1 + len(MASK_GROUPS) + 2
        bank = [_sfno(n_in, 13, 16, 2, min(16, t + 1)) for _ in range(args.operators)]
        x = torch.randn(1, n_in, nlat, 2 * nlat)

        def call():
            u = sum(op(x) for op in bank)
            u.square().mean().backward()

        res["operator_bank_s"] = timed(call)
        res["operators"] = args.operators
    res["peak_rss_gb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20
    print(json.dumps(res))


if __name__ == "__main__":
    main()
