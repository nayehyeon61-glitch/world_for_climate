"""Learning-to-Theorize evaluation of a lie_program checkpoint.

    python scripts/eval_transfer.py --checkpoint runs/wb2_theory_lie/checkpoint.pt \
        --data-dir DATA [--stride 4] [--min-gap-days 7] [--out results.json]

Scores self-explainability of 6 h pairs, analog transfer of inferred programs
and length generalisation on 12 h pairs (see climate_wm/eval/transfer.py).
Run it on the VQ arm and on the continuous control with the same arguments.
"""

import argparse
import json
import os

import _path  # noqa: F401  (adds src/ to sys.path)
import numpy as np
import torch

from climate_wm import pipeline
from climate_wm.eval import transfer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", default=None, help='replaces "{data_dir}" in the config')
    parser.add_argument("--stride", type=int, default=4, help="every n-th validation initialisation")
    parser.add_argument("--min-gap-days", type=float, default=7.0,
                        help="an analog must be at least this far away in time")
    parser.add_argument("--ood-steps", type=int, nargs="*", default=None,
                        help="maximal program lengths for the 12 h pairs (default K and 2K)")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", default=None)
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                        help="override the checkpoint config, e.g. data.val.start=2018-06-01")
    args = parser.parse_args()

    cfg, state, stats = pipeline.load_checkpoint(args.checkpoint, args.device)
    if args.data_dir:
        cfg = pipeline.substitute_data_dir(cfg, args.data_dir)
    pipeline.apply_overrides(cfg, args.set)
    cfg["eval"]["lead_steps"] = max(2, int(cfg["eval"].get("lead_steps", 2)))
    layout = pipeline.build_layout(cfg)
    ds = pipeline.open_dataset(cfg)
    train_ds, val_ds, _ = pipeline.build_datasets(cfg, layout, ds, stats)
    model = pipeline.build_model(cfg, layout, train_ds).to(args.device)
    model.load_state_dict(state)
    if model.theory is None:
        parser.error("the checkpoint's transition is not lie_program")
    th = model.theory
    indices = np.arange(len(val_ds))[::args.stride]
    print(f"{len(indices)} pairs ({val_ds.times[val_ds.indices[indices[0]]]} .. "
          f"{val_ds.times[val_ds.indices[indices[-1]]]})")

    pairs = transfer.collect_pairs(model, val_ds, indices, 0, args.device, args.batch_size, print)
    exp = transfer.explain_all(model, pairs, None, args.device, args.batch_size, print)
    centres = model.graphs.patches.centres.cpu()
    patch_lat = torch.rad2deg(torch.asin(centres[:, 2].clamp(-1, 1)))
    results = {
        "experiment": cfg.get("experiment", ""),
        "checkpoint": args.checkpoint,
        "lead_hours": pairs.lead_hours,
        "codebook_size": th.cfg.codebook_size,
        "physics_codes": th.cfg.physics_codes,
        "max_steps": th.cfg.max_steps,
        "self_explainability": transfer.self_explainability(exp, th.cfg.codebook_size, patch_lat),
    }
    analogs = transfer.find_analogs(pairs, 24.0 * args.min_gap_days)
    results["analog_transfer"] = transfer.analog_transfer(model, pairs, exp, analogs,
                                                          args.device, args.batch_size)
    if th.physics is not None:  # intervention tests: TOA solar radiation := 0 (night), family knock-outs
        mean, std = stats.arrays(layout.forcings)
        tisr = [i for i, c in enumerate(layout.forcings) if c.name == "toa_incident_solar_radiation"]
        report = {c.name: i for i, c in enumerate(layout.targets)
                  if c.name in ("2m_temperature", "mean_sea_level_pressure")}
        if tisr:
            night = float(-mean[tisr[0]] / std[tisr[0]])
            print(f"[causal] counterfactual TISR = 0 (normalised {night:.2f}); knock-outs")
            results["causal"] = transfer.causal_tests(model, pairs, exp, patch_lat, tisr[0], night,
                                                      report, args.device, args.batch_size)
    pairs12 = transfer.collect_pairs(model, val_ds, indices, 1, args.device, args.batch_size, print)
    steps = args.ood_steps or [th.cfg.max_steps, 2 * th.cfg.max_steps]
    results["length_ood"] = transfer.length_ood(model, pairs12, steps, args.device,
                                                args.batch_size, print)
    print(transfer.markdown(results))
    out = args.out or os.path.join(cfg["output_dir"], "transfer.json")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=1)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
