"""Stage 15.3: train the physics world model on SPEEDY episodes with the teacher's tendencies.

    # N1-T (state + tendency loss) and N1-S (state only) from N1
    python scripts/train_teacher.py train --init runs/physics_n1/checkpoint.pt --episodes DATA/speedy_episodes \
        --nature DATA/speedy_wb2 --out runs/stage15/N1T --lambda-tend 1.0
    python scripts/train_teacher.py train ... --out runs/stage15/N1S --lambda-tend 0
    # T1 on the held-out episodes: state error per lead, tendency pattern correlation
    python scripts/train_teacher.py eval --init runs/stage15/N1T/checkpoint.pt ... --out runs/stage15/N1T

The model gets the stage 14 action channels (CO2 / solar of group C) and
SPEEDY's daily-mean insolation. Resumable (OUT/progress.pt).
"""
import argparse
import dataclasses
import json
import os

import _path  # noqa: F401
import torch

from climate_wm import pipeline
from climate_wm.train.memory_filter import expand_state_dict
from climate_wm.train.teacher import EpisodeSet, TeacherConfig, evaluate_teacher, train_teacher


def build(args, device):
    cfg, state, stats = pipeline.load_checkpoint(args.init, device)
    cfg = dict(cfg)
    model_cfg = dict(cfg["model"])
    model_cfg["action_channels"] = 3
    model_cfg["solar"] = "daily_mean"
    cfg["model"] = model_cfg
    data = dict(cfg["data"])
    data.update(source="zarr", path=os.path.join(args.nature, "nature.zarr"), load_into_memory=False,
                time_range=["1982-01-01", "1985-01-01"],
                train={"start": "1982-01-01T06:00", "end": "1983-12-31T18:00"},
                val={"start": "1984-01-01", "end": "1984-12-31T12:00"})
    data.pop("stats_path", None)
    cfg["data"] = data
    cfg["eval"] = dict(cfg.get("eval", {}), climatology_zarr=None)
    cfg["output_dir"] = args.out
    layout = pipeline.build_layout(cfg)
    nature = pipeline.open_dataset(cfg)
    train_ds, _, _ = pipeline.build_datasets(cfg, layout, nature, stats)
    model = pipeline.build_model(cfg, layout, train_ds).to(device)
    grown = expand_state_dict(model, state)
    print(f"loaded {args.init}; widened: {grown}")
    return cfg, model, stats, layout


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=["train", "eval"])
    ap.add_argument("--init", required=True)
    ap.add_argument("--episodes", required=True)
    ap.add_argument("--nature", required=True, help="nature-run root (nature.zarr: layout and grid)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--lambda-tend", type=float, default=1.0)
    ap.add_argument("--tend-keys", nargs="+", default=None)
    ap.add_argument("--groups", nargs="+", default=None)
    ap.add_argument("--curriculum", nargs="+", default=None, help="steps:updates ... e.g. 1:150 4:150 8:100")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-steps", type=int, default=20)
    ap.add_argument("--eval-stride", type=int, default=16)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=1)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    os.makedirs(args.out, exist_ok=True)
    cfg, model, stats, layout = build(args, args.device)
    tc = dict(lambda_tend=args.lambda_tend, seed=args.seed)
    if args.tend_keys:
        tc["tend_keys"] = args.tend_keys
    if args.groups:
        tc["groups"] = args.groups
    if args.curriculum:
        tc["curriculum"] = [tuple(int(v) for v in c.split(":")) for c in args.curriculum]
    tcfg = TeacherConfig.from_dict(tc)
    if args.mode == "train":
        data = EpisodeSet(args.episodes, layout, stats, "train", tcfg.groups)
        print(f"{len(data)} training episodes, {tcfg}")
        ckpt = os.path.join(args.out, "checkpoint.pt")
        prog_path = os.path.join(args.out, "progress.pt")
        progress = None
        if os.path.exists(prog_path):
            progress = torch.load(prog_path, map_location=args.device, weights_only=False)
            model.load_state_dict(progress["model"])
            print(f"resumed at update {progress['update']}")

        def save(p):
            p = dict(p, model=model.state_dict())
            torch.save(p, prog_path + ".tmp")
            os.replace(prog_path + ".tmp", prog_path)

        hist = train_teacher(model, data, tcfg, args.device, progress, save)
        cfg["teacher"] = dataclasses.asdict(tcfg)
        pipeline.save_checkpoint(ckpt, cfg, model, stats, {"teacher": hist})
        with open(os.path.join(args.out, "train_history.json"), "w") as f:
            json.dump({"config": dataclasses.asdict(tcfg), "history": hist}, f)
        if os.path.exists(prog_path):
            os.remove(prog_path)
        print(f"saved {ckpt}")
    else:
        data = EpisodeSet(args.episodes, layout, stats, "test", ("A", "B", "C"))
        std = torch.from_numpy(EpisodeSet(args.episodes, layout, stats, "train", ("A", "B", "C"))
                               .tendency_std(seed=0)).to(args.device)
        res = evaluate_teacher(model, data, args.eval_steps, args.eval_stride, std, args.device)
        res["checkpoint"] = args.init
        with open(os.path.join(args.out, "teacher_eval.json"), "w") as f:
            json.dump(res, f)
        print(f"saved {args.out}/teacher_eval.json")


if __name__ == "__main__":
    main()
