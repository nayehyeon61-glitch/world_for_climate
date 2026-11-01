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

`--fill continuous --fill-start 2018-01-01` fills the memory with one
continuous pass instead: from `fill-start` to the last initialisation the model
makes one-step forecasts and re-observes the full truth, the memory carried
across consecutive windows (the pass is resumable), and m is snapshotted at
each initialisation; the free run then starts from the snapshot. With a kernel
whose time scales are bounded (tau <= 10 days in stage 10) this should equal a
spin-up of a month or more (saturation); it is the fill of a long-memory kernel.

`--obs-kind region` (감자 09:49 KST 11/01: "관측을 통해서 예보가 되면"): observations
keep arriving during the forecast. The spin-up observes everything, so at the
initialisation the model holds the same analysis as GraphCast; from then on a
cap of the sphere (`--region-fraction` of its area, one cap per
initialisation, the same for every forecast) is hidden and the points outside
it are re-observed every 6 h through the forecast. Every forecast, GraphCast's
free run included, is scored on the cap only (the residual outside is zeroed
and inside scaled by 1 / sqrt(hidden weight fraction), so the RMSE is the
latitude-weighted RMSE over the cap; ACC and the persistence and climatology
rows are not valid in this mode). The question is at which lead a forecast fed
by observations outside the cap beats the best free forecast inside it.
Variants: `warm` (the memory model, posterior re-observation) and `nudging`
(--base: N1 with the truth inserted on the observed points, the stage 10
reference).
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
from climate_wm.eval.graphcast_metrics import (HEADLINE_KEYS, acc_skill_score, normalized_latitude_weights,
                                               rmse_skill_score)
from climate_wm.train.dreamer_game import play_memory
from climate_wm.train.prediction_game import reobserved_fields, sample_mask, target_mask
from compare_graphcast import GraphCastForecasts, format_scorecard

VARIANTS = {"warm": (), "warm_no_memory": ("memory",), "nudging": ()}
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


def _finite_rows(state) -> torch.Tensor:
    """[B] whether every state tensor of the sample is finite."""
    return torch.stack([torch.isfinite(v).flatten(1).all(1) for v in state.values()]).all(0).cpu()


class ContinuousFill:
    """Memory states at the wanted initialisation times from one continuous pass
    of fully re-observed one-step forecasts from `start`: the memory is carried
    across consecutive windows of `chunk` steps (the dynamic state restarts
    from the truth at every step anyway, so a window boundary is not a restart
    of the memory), and m is snapshotted after the re-observation at each
    wanted time. `path`: the pass is saved there after every window and
    continued from it. A non-finite memory restarts from zero (counted)."""

    def __init__(self, model, ds, layout, stats, start: str, wanted, chunk: int = 120, knock=(),
                 device="cpu", path: str = None, log=None):
        self.model, self.device, self.knock = model, device, set(knock)
        self.snaps: Dict[int, torch.Tensor] = {}
        self.resets = 0
        self.wanted = sorted(int(t) for t in wanted)
        t0 = int(np.datetime64(start).astype("datetime64[s]").astype(np.int64))
        last = self.wanted[-1]
        if t0 >= last:
            raise ValueError("the fill must start before the first initialisation")
        self.segments = []  # (start seconds, steps)
        t = t0
        while t < last:
            n = min(chunk, (last - t) // STEP_S)
            self.segments.append((t, n))
            t += n * STEP_S
        self.datasets = {n: GraphCastWindowDataset(ds, layout, stats, n, start=str(np.datetime64(t0, "s")))
                         for n in {n for _, n in self.segments}}
        self.starts = {n: {_seconds(d, i): j for j, i in enumerate(d.indices)} for n, d in self.datasets.items()}
        self.path = path
        self.run(log)

    def run(self, log=None):
        model, mask = self.model, None
        t2d = model.layout.target_to_dynamic
        mem, done = None, 0
        if self.path and os.path.exists(self.path):
            saved = torch.load(self.path, weights_only=False)
            mem, done, self.snaps, self.resets = saved["mem"], saved["done"], saved["snaps"], saved["resets"]
            if log is not None:
                log(f"[fill] resumed after {done}/{len(self.segments)} windows")
        kernel = model.memory
        kernel.knock = set(self.knock)
        try:
            with torch.no_grad():
                for s, (t, n) in enumerate(self.segments):
                    if s < done:
                        continue
                    j = self.starts[n].get(t)
                    if j is None:
                        raise ValueError(f"no window of {n} steps starts at {np.datetime64(t, 's')}")
                    item = self.datasets[n][j]
                    batch = {k: v[None].to(self.device) for k, v in item.items()}
                    state, ctx = model.begin(batch)
                    if mask is None:
                        mask = torch.zeros_like(batch["inputs"][:, -1])
                    if mem is None:  # the start of the pass: as play_memory
                        snap = batch["inputs"][:, -1]
                        state = model.posterior(state, ctx, snap, snap, mask)
                    else:
                        state["mem"] = mem
                    for k in range(n):
                        state, precip = model.advance(state, ctx, k)
                        pred = model.predict(state, ctx, k, precip)
                        truth = batch["targets"][:, k][:, t2d]
                        state = model.posterior(state, ctx, torch.nan_to_num(pred[:, t2d]), truth, mask)
                        if not torch.isfinite(state["mem"]).all():
                            state["mem"] = torch.zeros_like(state["mem"])
                            self.resets += 1
                        tk = t + (k + 1) * STEP_S
                        if tk in self.wanted:
                            self.snaps[tk] = state["mem"].detach().cpu().clone()
                    mem = state["mem"]
                    if self.path:
                        torch.save({"mem": mem, "done": s + 1, "snaps": self.snaps, "resets": self.resets},
                                   self.path + ".tmp")
                        os.replace(self.path + ".tmp", self.path)
                    if log is not None:
                        log(f"[fill] {s + 1}/{len(self.segments)} windows ({np.datetime64(t + n * STEP_S, 's')}), "
                            f"{len(self.snaps)} snapshots, {self.resets} memory resets")
        finally:
            kernel.knock = set()
        missing = [t for t in self.wanted if t not in self.snaps]
        if missing:
            raise RuntimeError(f"no snapshot for {len(missing)} initialisations, e.g. {np.datetime64(missing[0], 's')}")


class FilledStart(WarmStart):
    """The free run from a ContinuousFill snapshot of the memory (no spin-up)."""

    def __init__(self, model, snaps: Dict[int, torch.Tensor], knock=(), cap: float = 10.0, blown_path: str = None):
        super().__init__(model, 0, knock, cap, blown_path)
        self.snaps = snaps

    @torch.no_grad()
    def rollout(self, batch: Dict[str, torch.Tensor], steps: int, members: int = 1, **_) -> torch.Tensor:
        model, x = self.model, batch["inputs"]
        inits = batch["init_time"].cpu().tolist()
        mem_k = model.memory
        mem_k.knock = set(self.knock)
        try:
            state, ctx = model.begin(batch)
            state["mem"] = torch.cat([self.snaps[t] for t in inits]).to(x.device)
            snap = x[:, -1]
            state = model.posterior(state, ctx, snap, snap, torch.zeros_like(snap))  # as play_memory's start
            preds, last, blown = [], None, torch.zeros(len(inits), dtype=torch.bool)
            for k in range(steps):
                state, precip = model.advance(state, ctx, k)
                pred = model.predict(state, ctx, k, precip).float().cpu()
                blown |= ~(torch.isfinite(pred).flatten(1).all(1) & _finite_rows(state))
                pred = torch.nan_to_num(pred) if last is None else torch.where(blown[:, None, None, None], last, pred)
                last = pred
                preds.append(pred)
        finally:
            mem_k.knock = set()
        if self.blown_path:
            with open(self.blown_path, "a") as f:
                f.write(json.dumps({"init": inits, "blown": blown.tolist()}) + "\n")
        pred = torch.stack(preds, 1).to(x.device)
        return pred[None].expand(members, *pred.shape)


class ObservedStart(WarmStart):
    """Observations keep arriving during the forecast (module docstring): the
    spin-up observes everything (play_memory with nothing hidden, the state
    after its last re-observation kept), then the forecast re-observes the
    points outside a cap every step; the cap is sampled per initialisation
    (seeded by its time, so every forecast gets the same cap) and `hide`
    scores it alone. `rule`: "posterior" (the memory model) or a nudging
    strength (N1, 1.0 = insertion)."""

    def __init__(self, model, spinup: int, lat, lon, kind: str = "region", region_fraction: float = 0.25,
                 seed: int = 0, rule="posterior", knock=(), cap: float = 10.0, blown_path: str = None):
        super().__init__(model, spinup, knock, cap, blown_path)
        self.lat, self.lon, self.kind, self.frac, self.seed, self.rule = lat, lon, kind, region_fraction, seed, rule
        self.lat_w = torch.tensor(normalized_latitude_weights(lat), dtype=torch.float32)
        self.current: Dict[str, torch.Tensor] = {}  # of the latest batch: target mask, physical target
        self.hidden_fraction: list = []

    def masks(self, inits) -> torch.Tensor:
        out = []
        for t in inits:
            gen = torch.Generator().manual_seed(self.seed + int(t) // STEP_S)
            out.append(sample_mask(self.layout, self.kind, self.lat, self.lon, gen, self.frac)[0])
        return torch.stack(out)

    def hide(self, pred: torch.Tensor, target: torch.Tensor, tmask: torch.Tensor) -> torch.Tensor:
        """pred, target [B, L, C, H, W]; tmask [B, C, H, W] (1 = hidden): the residual on the cap scaled
        so that its latitude-weighted mean square over the grid is the mean square over the cap."""
        w = self.lat_w.to(pred.device)[None, None, :, None]
        frac = (tmask * w).sum((-2, -1)) / w.expand_as(tmask).sum((-2, -1))  # [B, C]
        scale = (tmask / frac.clamp_min(1e-6).sqrt()[..., None, None])[:, None]
        return target + (pred - target) * scale

    @torch.no_grad()
    def rollout(self, batch: Dict[str, torch.Tensor], steps: int, members: int = 1, **_) -> torch.Tensor:
        long = {k[5:]: v for k, v in batch.items() if k.startswith("spin_")}
        x = long["inputs"]
        if not torch.allclose(long["targets"][:, self.spinup:], batch["targets"]):
            raise ValueError("the spin-up window does not end on the forecast window")
        inits = batch["init_time"].cpu().tolist()
        mask = self.masks(inits).to(x.device)
        tmask = target_mask(self.model, mask)
        model, t2d = self.model, self.model.layout.target_to_dynamic
        preds, last, blown = [], None, torch.zeros(len(inits), dtype=torch.bool)
        mem = model.memory
        if mem is not None:
            mem.knock = set(self.knock)
        try:
            # the spin-up with everything observed: the state before step `spinup` is the
            # analysis at the initialisation (after the re-observation of its truth)
            keep: list = []
            play_memory(model, None, long, torch.zeros_like(mask), self.spinup + 1, obs_prob=1.0, rule=self.rule,
                        gen=torch.Generator().manual_seed(0), cap=self.cap, keep=keep)
            state, resid, _, ctx, _ = keep[self.spinup]
            state, ctx = dict(state), dict(ctx, resid=resid)
            for k in range(steps):
                state, precip = model.advance(state, ctx, self.spinup + k)
                pred = model.predict(state, ctx, self.spinup + k, precip)
                truth = long["targets"][:, self.spinup + k][:, t2d]
                ok = torch.isfinite(pred).flatten(1).all(1).cpu() & _finite_rows(state)
                blown |= ~ok
                scored = pred.float().cpu()
                scored = torch.nan_to_num(scored) if last is None else torch.where(blown[:, None, None, None], last, scored)
                last = scored
                preds.append(scored)
                fc = torch.nan_to_num(pred[:, t2d])
                if self.rule == "posterior":  # the observed points re-observed, the cap carried
                    state = model.posterior(state, ctx, fc, truth, mask)
                else:
                    fresh = model.reobserve(ctx, reobserved_fields(truth, fc, mask, None if self.rule == "reset" else self.rule))
                    if "mem" in state:
                        fresh["mem"] = state["mem"]
                    state = fresh
        finally:
            if mem is not None:
                mem.knock = set()
        if self.blown_path:
            with open(self.blown_path, "a") as f:
                f.write(json.dumps({"init": inits, "blown": blown.tolist()}) + "\n")
        pred = self.hide(torch.stack(preds, 1).to(x.device), batch["targets"], tmask)
        w = self.lat_w[None, None, :, None]
        self.hidden_fraction += ((tmask.cpu() * w).sum((-2, -1)) / w.expand_as(tmask.cpu()).sum((-2, -1))).mean(1).tolist()
        self.current = {"tmask": tmask.cpu(), "init": inits}
        return pred[None].expand(members, *pred.shape)


class HiddenGraphCast:
    """GraphCast's free forecast scored on the hidden cap of the batch the wrapper just ran."""

    def __init__(self, gc, wrapper: ObservedStart, stats, targets, val_items):
        self.gc, self.wrapper, self.stats, self.targets, self.items = gc, wrapper, stats, targets, val_items

    def available(self, inits):
        return self.gc.available(inits)

    def __call__(self, init, lead):
        pred = self.gc(init, lead)  # physical [B, L, C, H, W]
        cur = self.wrapper.current
        if [int(t) for t in init] != [int(t) for t in cur["init"]]:
            raise RuntimeError("GraphCast batch does not match the wrapper's")
        target = self.stats.unnormalize(torch.stack([self.items(int(t))["targets"] for t in init]), self.targets)
        return self.wrapper.hide(pred.float(), target.float(), cur["tmask"])


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
    parser.add_argument("--variants", nargs="*", default=["warm", "warm_no_memory"], choices=list(VARIANTS),
                        help="nudging needs --base; ignored with --fill continuous")
    parser.add_argument("--fill", choices=["spinup", "continuous"], default="spinup",
                        help="continuous: one pass from --fill-start with the memory carried (variant 'fill')")
    parser.add_argument("--fill-start", default=None, help="e.g. 2018-01-01 (the fill of 2020 then holds two years)")
    parser.add_argument("--fill-chunk", type=int, default=120, help="window length of the pass (steps)")
    parser.add_argument("--obs-kind", choices=["none", "region"], default="none",
                        help="region: observations keep arriving outside a hidden cap, scored on the cap")
    parser.add_argument("--region-fraction", type=float, default=0.25)
    parser.add_argument("--obs-seed", type=int, default=0)
    parser.add_argument("--base", default=None, help="N1 checkpoint for the nudging reference (variant nudging)")
    parser.add_argument("--cap", type=float, default=10.0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", required=True, help="OUT_<variant>.json per variant, OUT_summary.json")
    parser.add_argument("--graphcast", default=None,
                        help="GraphCast forecast zarr (default DATA/graphcast.zarr); 'none' scores without it, on "
                             "every initialisation of the split (a 28-day free run, say, outruns its forecasts)")
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
    gc = None if args.graphcast == "none" else GraphCastForecasts(
        args.graphcast or os.path.join(args.data_dir, "graphcast.zarr"), layout.targets, coarsen=args.coarsen)
    lead = cfg["eval"]["lead_steps"]
    val = cfg["data"].get("val", {})
    shift = lambda s: str(np.datetime64(s) - np.timedelta64(6 * args.spinup, "h")) if s else None
    long_ds = GraphCastWindowDataset(ds, layout, stats, lead + args.spinup, start=shift(val.get("start")),
                                     end=shift(val.get("end")), init_hours=cfg["eval"].get("init_hours"))
    data = SpinUpWindows(val_ds, long_ds, args.spinup)
    if len(data) < len(val_ds):
        print(f"{len(val_ds) - len(data)} forecast windows have no {args.spinup}-step spin-up (data.time_range)")
    usable = (np.where(gc.available(data.init_seconds()))[0] if gc is not None
              else np.arange(len(data)))[::args.init_stride]
    t = lambda n: np.datetime64(int(data.init_seconds()[n]), "s")
    print(f"{len(usable)} initialisations ({t(usable[0])} .. {t(usable[-1])}), spin-up {args.spinup} steps")
    index = {c.key: i for i, c in enumerate(layout.targets)}
    summary = {"spinup_steps": args.spinup, "inits": int(len(usable)), "variants": {}}
    variants = list(args.variants)
    base = None
    if "nudging" in variants and args.fill != "continuous":
        if not args.base:
            parser.error("the nudging variant needs --base")
        bcfg, bstate, _ = pipeline.load_checkpoint(args.base, args.device)
        base = pipeline.build_model(dict(cfg, model=bcfg["model"]), layout, train_ds).to(args.device)
        base.load_state_dict(bstate)
        base.eval()
    if args.obs_kind != "none":
        summary.update({"obs_kind": args.obs_kind, "region_fraction": args.region_fraction, "obs_seed": args.obs_seed,
                        "note": "RMSE over the hidden cap; ACC, persistence and climatology are not valid here"})
    items = lambda t: data[int(np.where(data.init_seconds() == t)[0][0])]
    if args.fill == "continuous":
        if not args.fill_start:
            parser.error("--fill continuous needs --fill-start")
        wanted = data.init_seconds()[usable]
        fill = ContinuousFill(model, ds, layout, stats, args.fill_start, wanted, args.fill_chunk, device=args.device,
                              path=args.out.replace(".json", "_fill_pass.pt"), log=print)
        summary.update({"fill_start": args.fill_start, "fill_chunk": args.fill_chunk, "memory_resets": fill.resets,
                        "spinup_steps": None})
        variants = ["fill"]
    for name in variants:
        out = args.out.replace(".json", f"_{name}.json")
        blown = out.replace(".json", "_blown.jsonl")
        extra = {"graphcast": gc} if gc is not None else None
        if name == "fill":
            wrapper = FilledStart(model, fill.snaps, (), args.cap, blown)
        elif args.obs_kind != "none":
            m, rule = (base, 1.0) if name == "nudging" else (model, "posterior")
            wrapper = ObservedStart(m, args.spinup, data.lat, data.lon, args.obs_kind, args.region_fraction,
                                    args.obs_seed, rule, VARIANTS[name], args.cap, blown)
            if gc is not None:
                extra = {"graphcast": HiddenGraphCast(gc, wrapper, stats, layout.targets, items)}
        elif name == "nudging":
            parser.error("the nudging variant is for --obs-kind region")
        else:
            wrapper = WarmStart(model, args.spinup, VARIANTS[name], args.cap, blown)
        progress = out + ".progress.pt" if args.resume else None
        if not args.resume and os.path.exists(blown):
            os.remove(blown)
        print(f"== {name}: memory knock-outs {sorted(wrapper.knock) or 'none'}")
        results = pipeline.evaluate(cfg, wrapper, data, stats, clim, args.device, indices=usable,
                                    extra_forecasts=extra, log=print, progress_path=progress)
        metrics = results["metrics"]
        print(format_scorecard(metrics, "model", "graphcast" if gc is not None else "persistence"))
        pipeline.results_to_json(results, out)
        ours = metrics["model"].compute()
        theirs = metrics["graphcast"].compute() if gc is not None else None
        entry = {"blowups": _blowups(blown)}
        if isinstance(wrapper, ObservedStart) and wrapper.hidden_fraction:
            entry["hidden_fraction"] = float(np.mean(wrapper.hidden_fraction))
        for key in HEADLINE_KEYS:
            if key in index and theirs is not None:
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
