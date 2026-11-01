"""Warm-started forecasts of the stage 10 memory model on the GraphCast protocol.

    python scripts/compare_memory.py --checkpoint runs/stage10_Dnoactor/checkpoint.pt \
        --data-dir wb2_2020 --init-stride 6 --lead-steps 20 --spinup 40 \
        --variants warm warm_no_memory --resume --out runs/stage10_Dnoactor/compare_memory_2020.json

scripts/compare_graphcast.py starts every forecast of the memory model with an
empty memory (a cold start). Here the memory is filled first: for `spinup` 6 h
steps before the initialisation the model makes a one-step forecast and
re-observes the full truth through its posterior (the innovation feeds the
memory kernel, as in the stage 10 game with nothing hidden), then it runs free
for `lead_steps` steps from the same initialisations as the cold start. The free
run is scored by pipeline.evaluate against GraphCast, persistence and climatology
exactly as the cold start, so the two are paired over initialisations.

Variants (the cause ablations): `warm` (the memory as trained) and
`warm_no_memory` (the kernel knocked out during the spin-up and the free run:
the free run then starts from the re-observed truth, so this is the column
model's cold start under the warm-start code path, the control for the spin-up
itself). A sample whose state blows up keeps its last finite forecast from then
on (counted in the summary); the cold start lets non-finite values through.
"""

import argparse
import json
import os
from typing import Dict, Sequence

import _path  # noqa: F401  (adds src/ to sys.path)
import numpy as np
import torch
from torch.utils.data import Dataset

from climate_wm import pipeline
from climate_wm.data.era5_lazy_loader import GraphCastWindowDataset
from climate_wm.eval.graphcast_metrics import HEADLINE_KEYS, acc_skill_score, rmse_skill_score
from climate_wm.train.dreamer_game import play_memory
from compare_graphcast import GraphCastForecasts, format_scorecard

VARIANTS = {"warm": (), "warm_no_memory": ("memory",)}
STEP_S = 6 * 3600


def _seconds(ds, i) -> int:
    return int(ds.times[i].astype("datetime64[s]").astype(np.int64))


class SpinUpWindows(Dataset):
    """The forecast windows of `short` (the evaluation's val dataset) joined with
    the window of `long` that starts `spinup` steps earlier and reaches the same
    targets (its tensors under spin_*). Windows without a partner are left out."""

    def __init__(self, short: GraphCastWindowDataset, long: GraphCastWindowDataset, spinup: int):
        if long.target_steps != short.target_steps + spinup:
            raise ValueError("the long windows must hold spinup + lead target steps")
        self.short, self.long, self.spinup = short, long, spinup
        self.layout, self.lat, self.lon = short.layout, short.lat, short.lon
        starts = {_seconds(long, i): j for j, i in enumerate(long.indices)}
        self.pairs = [(k, starts[t]) for k, i in enumerate(short.indices)
                      if (t := _seconds(short, i) - spinup * STEP_S) in starts]

    def __len__(self) -> int:
        return len(self.pairs)

    def init_seconds(self) -> np.ndarray:
        return np.array([_seconds(self.short, self.short.indices[k]) for k, _ in self.pairs])

    def __getitem__(self, n: int) -> Dict[str, torch.Tensor]:
        k, j = self.pairs[n]
        item = self.short[k]
        item.update({"spin_" + key: v for key, v in self.long[j].items()})
        return item


class WarmStart:
    """`rollout(batch, steps, members)` for pipeline.evaluate: the spin-up on the
    spin_* window, then the free run; normalised [M, B, L, C, H, W] (copies)."""

    def __init__(self, model, spinup: int, knock: Sequence[str] = (), cap: float = 10.0,
                 blown_path: str = None):
        self.model, self.spinup, self.knock, self.cap = model, spinup, set(knock), cap
        self.layout = model.layout
        self.blown_path = blown_path

    def eval(self):
        self.model.eval()
        return self

    @torch.no_grad()
    def rollout(self, batch: Dict[str, torch.Tensor], steps: int, members: int = 1, **_) -> torch.Tensor:
        long = {k[5:]: v for k, v in batch.items() if k.startswith("spin_")}
        x = long["inputs"]
        b = x.shape[0]
        if not torch.allclose(long["targets"][:, self.spinup:], batch["targets"]):
            raise ValueError("the spin-up window does not end on the forecast window")
        mask = torch.zeros(b, len(self.layout.dynamic), *x.shape[-2:], device=x.device)
        preds, last, blown = [], [None], torch.zeros(b, dtype=torch.bool)

        def on_step(k, pred, dead):
            if k < self.spinup:
                return
            pred, dead = pred.float().cpu(), dead.cpu()
            if last[0] is not None:  # a blown-up sample keeps its last finite forecast
                pred = torch.where(dead[:, None, None, None], last[0], pred)
            last[0] = pred
            blown.logical_or_(dead)
            preds.append(pred)

        mem = self.model.memory
        if mem is not None:
            mem.knock = set(self.knock)
        try:
            play_memory(self.model, None, long, mask, self.spinup + steps, obs_prob=1.0, rule="posterior",
                        gen=torch.Generator().manual_seed(0), cap=self.cap, obs_steps=self.spinup,
                        on_step=on_step)
        finally:
            if mem is not None:
                mem.knock = set()
        if self.blown_path:
            with open(self.blown_path, "a") as f:
                f.write(json.dumps({"init": batch["init_time"].cpu().tolist(), "blown": blown.tolist()}) + "\n")
        pred = torch.stack(preds, 1).to(x.device)  # [B, L, C, H, W]
        return pred[None].expand(members, *pred.shape)


def _blowups(path: str) -> int:
    if not os.path.exists(path):
        return 0
    seen = {}
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            seen.update(zip(rec["init"], rec["blown"]))  # a resumed batch overwrites its first try
    return int(sum(seen.values()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", default=None,
                        help='replaces "{data_dir}" in the config; holds graphcast.zarr by default')
    parser.add_argument("--lead-steps", type=int, default=None)
    parser.add_argument("--init-stride", type=int, default=1)
    parser.add_argument("--spinup", type=int, default=40,
                        help="re-observed 6 h steps before each initialisation (40 = 10 days, as the climate "
                             "evaluation of stage 10)")
    parser.add_argument("--variants", nargs="*", default=list(VARIANTS), choices=list(VARIANTS))
    parser.add_argument("--cap", type=float, default=10.0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", required=True, help="OUT_<variant>.json per variant, OUT_summary.json")
    parser.add_argument("--graphcast", default=None, help="GraphCast forecast zarr (default DATA/graphcast.zarr)")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--coarsen", type=int, default=1)
    args = parser.parse_args()

    cfg, state, stats = pipeline.load_checkpoint(args.checkpoint, args.device)
    if args.data_dir:
        cfg = pipeline.substitute_data_dir(cfg, args.data_dir)
    if not (args.graphcast or args.data_dir):
        parser.error("give --graphcast or --data-dir")
    pipeline.apply_overrides(cfg, args.set)
    if args.lead_steps:
        cfg["eval"]["lead_steps"] = args.lead_steps
    cfg["eval"]["members"] = 1
    layout = pipeline.build_layout(cfg)
    ds = pipeline.open_dataset(cfg)
    train_ds, val_ds, _ = pipeline.build_datasets(cfg, layout, ds, stats)
    model = pipeline.build_model(cfg, layout, train_ds).to(args.device)
    model.load_state_dict(state)
    if getattr(model, "memory", None) is None:
        parser.error("the checkpoint has no memory kernel (stage 10)")
    clim = pipeline.build_climatology(cfg, train_ds)
    gc = GraphCastForecasts(args.graphcast or os.path.join(args.data_dir, "graphcast.zarr"),
                            layout.targets, coarsen=args.coarsen)
    lead = cfg["eval"]["lead_steps"]
    val = cfg["data"].get("val", {})
    shift = lambda s: str(np.datetime64(s) - np.timedelta64(6 * args.spinup, "h")) if s else None
    long_ds = GraphCastWindowDataset(ds, layout, stats, lead + args.spinup, start=shift(val.get("start")),
                                     end=shift(val.get("end")), init_hours=cfg["eval"].get("init_hours"))
    data = SpinUpWindows(val_ds, long_ds, args.spinup)
    if len(data) < len(val_ds):
        print(f"{len(val_ds) - len(data)} forecast windows have no {args.spinup}-step spin-up (data.time_range)")
    usable = np.where(gc.available(data.init_seconds()))[0][::args.init_stride]
    t = lambda n: np.datetime64(int(data.init_seconds()[n]), "s")
    print(f"{len(usable)} initialisations ({t(usable[0])} .. {t(usable[-1])}), spin-up {args.spinup} steps")
    index = {c.key: i for i, c in enumerate(layout.targets)}
    summary = {"spinup_steps": args.spinup, "inits": int(len(usable)), "variants": {}}
    for name in args.variants:
        out = args.out.replace(".json", f"_{name}.json")
        blown = out.replace(".json", "_blown.jsonl")
        wrapper = WarmStart(model, args.spinup, VARIANTS[name], args.cap, blown)
        progress = out + ".progress.pt" if args.resume else None
        if not args.resume and os.path.exists(blown):
            os.remove(blown)
        print(f"== {name}: memory knock-outs {sorted(wrapper.knock) or 'none'}")
        results = pipeline.evaluate(cfg, wrapper, data, stats, clim, args.device, indices=usable,
                                    extra_forecasts={"graphcast": gc}, log=print, progress_path=progress)
        metrics = results["metrics"]
        print(format_scorecard(metrics, "model", "graphcast"))
        pipeline.results_to_json(results, out)
        ours, theirs = metrics["model"].compute(), metrics["graphcast"].compute()
        entry = {"blowups": _blowups(blown)}
        for key in HEADLINE_KEYS:
            if key in index:
                j = index[key]
                entry[key] = {"lead_h": metrics["model"].lead_hours,
                              "rmse_skill_vs_graphcast": rmse_skill_score(ours["rmse"][:, j], theirs["rmse"][:, j]).tolist(),
                              "acc_skill_vs_graphcast": acc_skill_score(ours["acc"][:, j], theirs["acc"][:, j]).tolist()}
        summary["variants"][name] = entry
        if progress is not None and os.path.exists(progress):
            os.remove(progress)
        print(f"saved {out} ({entry['blowups']} blow-ups)")
    with open(args.out.replace(".json", "_summary.json"), "w") as f:
        json.dump(summary, f, indent=1)


if __name__ == "__main__":
    main()
