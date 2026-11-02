"""Stage 9: train the column physics of the physics-centred world model.

    python scripts/train_physics.py --config configs/physics_wb2.yaml --data-dir DATA [--resume]

The primitive-equation core has no parameters; the column network and the
surface head learn from multi-step rollouts through the differentiable core
(GraphCast-weighted MSE on every lead). `train.steps: 0` saves the untrained
model, i.e. arm N0 when `model.column.enabled` is false.
"""

import argparse
import json
import math
import os
import time

import _path  # noqa: F401  (adds src/ to sys.path)
import torch

from climate_wm import pipeline
from climate_wm.train.losses import GraphCastLoss


def rollout_length(schedule, step: int) -> int:
    length = 1
    for start, n in schedule:
        if step >= start:
            length = n
    return length


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data-dir", default=None, help='replaces "{data_dir}" in the config')
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--threads", type=int, default=None)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    if args.threads:
        torch.set_num_threads(args.threads)
    cfg = pipeline.load_config(args.config, args.data_dir)
    pipeline.apply_overrides(cfg, args.set)
    tc = cfg["train"]
    path = os.path.join(cfg["output_dir"], "checkpoint.pt")
    state, stats, progress = None, None, None
    if args.resume and os.path.exists(path):
        _, state, stats = pipeline.load_checkpoint(path, args.device)
        progress = torch.load(path, map_location="cpu", weights_only=False).get("progress")
        print(f"resuming from {path} at step {progress['step'] if progress else 'end'}")
    layout = pipeline.build_layout(cfg)
    ds = pipeline.open_dataset(cfg)
    train_ds, _, stats = pipeline.build_datasets(cfg, layout, ds, stats)
    model = pipeline.build_model(cfg, layout, train_ds).to(args.device)
    if state is not None:
        model.load_state_dict(state)
    params = [p for p in model.parameters() if p.requires_grad]
    print(f"trainable parameters: {sum(p.numel() for p in params):,}")
    steps = tc.get("steps", 0)
    history = list(progress.get("history", [])) if progress else []
    start = progress["step"] if progress else 0
    if steps == 0 or not params or (state is not None and progress is None):
        pipeline.save_checkpoint(path, cfg, model, stats, {"physics": history})
        print(f"saved {path} (no training steps)")
        return
    opt = torch.optim.AdamW(params, lr=tc["lr"], weight_decay=tc.get("weight_decay", 0.0))
    warm = tc.get("warmup_steps", 0)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / max(warm, 1)) * 0.5 * (1 + math.cos(math.pi * min(s / steps, 1.0))))
    if progress:
        opt.load_state_dict(progress["optimizer"])
        sched.load_state_dict(progress["scheduler"])
    loss_fn = GraphCastLoss(layout, train_ds.lat).to(args.device)
    loader = pipeline.make_loader(cfg, train_ds, args.device)
    schedule = tc.get("rollout_schedule", [[0, 1]])
    model.train()
    step, t0 = start, time.time()

    def save(done=False):
        prog = None if done else {"step": step, "optimizer": opt.state_dict(),
                                  "scheduler": sched.state_dict(), "history": history}
        pipeline.save_checkpoint(path, cfg, model, stats, {"physics": history}, prog)
        pipeline.save_json(os.path.join(cfg["output_dir"], "train_history.json"), {"physics": history})

    while step < steps:
        for batch in loader:
            if step >= steps:
                break
            batch = {k: v.to(args.device) for k, v in batch.items()}
            n = min(rollout_length(schedule, step), batch["targets"].shape[1])
            pred = model.forward_rollout(batch, n)
            loss = loss_fn(pred, batch["targets"][:, :n])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(params, tc.get("grad_clip", 1.0))
            opt.step()
            sched.step()
            step += 1
            history.append({"step": step, "loss": float(loss), "rollout": n, "grad_norm": float(gnorm)})
            if step % tc.get("log_every", 10) == 0:
                rate = (time.time() - t0) / (step - start)
                print(f"[physics] step {step}/{steps} rollout {n} loss {float(loss):.4f} "
                      f"grad {float(gnorm):.2f} lr {sched.get_last_lr()[0]:.2e} ({rate:.1f} s/step)",
                      flush=True)
            if step % tc.get("save_every", 25) == 0:
                save()
    save(done=True)
    print(f"saved {path} after {step} steps")
    with open(os.path.join(cfg["output_dir"], "train_summary.json"), "w") as f:
        json.dump({"steps": step, "final_loss": history[-1]["loss"] if history else None}, f)


if __name__ == "__main__":
    main()
