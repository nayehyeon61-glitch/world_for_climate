"""Stage 5 evaluation of a checkpoint: generator usage, interventions, critic-weighted ensemble.

    python scripts/eval_stage5.py --checkpoint runs/wb2_g1/checkpoint.pt --data-dir DATA \
        [--stride 4] [--ensemble-stride 12] [--members 4] [--lead-steps 12] [--out stage5_eval.json]

See climate_wm/eval/stage5.py. Works on every pyramid checkpoint: arms without
the composite group (G0 = stage 4 P) get only the ensemble part, and arms
without a trained critic get a diagnostic critic calibrated on training frames
(the post-hoc weighting control). The 72 h RMSE (prediction P1) comes from
scripts/compare_graphcast.py; scripts/report_stage5.py joins both.
"""

import argparse
import json
import os

import _path  # noqa: F401  (adds src/ to sys.path)
import numpy as np
import torch

from climate_wm import pipeline
from climate_wm.eval import stage5


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", default=None, help='replaces "{data_dir}" in the config')
    parser.add_argument("--stride", type=int, default=4, help="every n-th validation pair for the usage tests")
    parser.add_argument("--ensemble-stride", type=int, default=12, help="every n-th init for the ensemble")
    parser.add_argument("--members", type=int, default=4)
    parser.add_argument("--lead-steps", type=int, default=12, help="6 h steps (12 = 72 h)")
    parser.add_argument("--tau", type=float, default=1.0, help="member weight temperature")
    parser.add_argument("--calibration-windows", type=int, default=64,
                        help="training windows for a critic calibrated at evaluation time")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", default=None)
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()

    torch.manual_seed(0)
    cfg, state, stats = pipeline.load_checkpoint(args.checkpoint, args.device)
    if args.data_dir:
        cfg = pipeline.substitute_data_dir(cfg, args.data_dir)
    pipeline.apply_overrides(cfg, args.set)
    cfg["eval"]["lead_steps"] = max(args.lead_steps, int(cfg["eval"].get("lead_steps", 1)))
    layout = pipeline.build_layout(cfg)
    ds = pipeline.open_dataset(cfg)
    train_ds, val_ds, _ = pipeline.build_datasets(cfg, layout, ds, stats)
    model = pipeline.build_model(cfg, layout, train_ds).to(args.device)
    model.load_state_dict(state)
    if model.graphs.pyramid is None:
        parser.error("stage 5 evaluation needs a pyramid checkpoint")
    indices = np.arange(len(val_ds))[::args.stride]
    first, last = val_ds.times[val_ds.indices[indices[0]]], val_ds.times[val_ds.indices[indices[-1]]]
    print(f"{len(indices)} pairs ({first} .. {last})")

    key = "toa_incident_solar_radiation"
    night = (0.0 - stats.mean[key]) / stats.std[key] if key in stats.mean else None
    results = {
        "experiment": cfg.get("experiment", ""),
        "checkpoint": args.checkpoint,
        "period": [str(first), str(last)],
        "composite": bool(getattr(model.lie, "composite", None) is not None),
        "diagnostics": model.diagnostics is not None,
        "trained_critic": model.critic is not None,
    }
    results["usage"] = stage5.composite_usage(model, val_ds, indices, args.device, args.batch_size,
                                              tisr_night=night, log=print)
    cal = np.linspace(0, len(train_ds) - 1, min(len(train_ds), args.calibration_windows)).astype(int)
    critic = stage5.calibrate_critic(model, train_ds, cal, args.device, args.batch_size)
    if model.critic is not None:
        results["critic"] = model.critic.summary()
    ens_idx = np.arange(len(val_ds))[::args.ensemble_stride]
    results["ensemble"] = stage5.weighted_ensemble(
        model, val_ds, ens_idx, critic, members=args.members, lead_steps=args.lead_steps,
        device=args.device, batch_size=max(1, args.batch_size // 2), tau=args.tau, log=print)
    out = args.out or os.path.join(cfg["output_dir"], "stage5_eval.json")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=1)
    usage = results["usage"]
    if usage:
        print("fibre share per shell", [round(x, 3) for x in usage.get("fibre_share_per_shell", [])])
        print("knock-outs %", {k: round(v["loss_increase_pct"], 3) for k, v in usage["knockout"].items()})
        if "counterfactual" in usage:
            print("counterfactual", {k: v for k, v in usage["counterfactual"].items()})
    ens = results["ensemble"]
    for v, e in ens["variables"].items():
        print(f"{v}: CRPS@{ens['lead_h'][-1]}h equal {e['equal']['crps'][-1]:.4g} "
              f"weighted {e['weighted']['crps'][-1]:.4g}")
    print(f"critic/error rank correlation {ens['critic_error_rank_corr']:.3f}")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
