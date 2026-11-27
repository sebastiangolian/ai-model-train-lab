"""Train a model on Tiny Shakespeare with periodic validation, metrics CSV and checkpoints.

Every model uses the same recipe (data, AdamW, learning rate, budget), so runs are comparable.
Usage: python -m lab.train --model bdh --preset small [--iters N] [--device mps]
"""

import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from .data import ByteDataset, decode, encode
from .device import describe_hardware, pick_device, synchronize
from .spec import RUNS_DIR, available_models, load_spec


def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())  # tied weights are counted once


@torch.no_grad()
def estimate_loss(model, dataset, config, device) -> dict[str, float]:
    model.eval()
    losses = {}
    for split in ("train", "val"):
        values = []
        for _ in range(config.eval_iters):
            x, y = dataset.batch(split, config.batch_size, config.block_size, device)
            _, loss = model(x, y)
            values.append(loss.item())
        losses[split] = float(np.mean(values))
    model.train()
    return losses


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, choices=available_models())
    parser.add_argument("--preset", help="preset name (default: the model's DEFAULT_PRESET)")
    parser.add_argument("--iters", type=int, help="override max_iters from the preset")
    parser.add_argument("--batch-size", type=int, help="override batch_size from the preset")
    parser.add_argument("--block-size", type=int, help="override block_size from the preset")
    parser.add_argument("--device", default="auto", help="auto | cpu | mps | cuda")
    parser.add_argument("--compile", action="store_true", help="wrap the model in torch.compile")
    parser.add_argument("--run-dir", type=Path, help="output directory (default: runs/<model>/<preset>)")
    parser.add_argument("--prompt", default="To be or ", help="prompt for the post-training sample")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    spec = load_spec(args.model)
    preset = args.preset or spec.DEFAULT_PRESET
    if preset not in spec.PRESETS:
        raise SystemExit(f"Unknown preset {preset!r} for {args.model}. Available: {', '.join(spec.PRESETS)}")
    config = spec.PRESETS[preset].copy()
    for name in ("iters", "batch_size", "block_size"):
        value = getattr(args, name)
        if value is not None:
            setattr(config, "max_iters" if name == "iters" else name, value)

    run_dir = args.run_dir or RUNS_DIR / args.model / preset
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(config.to_dict(), indent=2))

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    device = pick_device(args.device)
    dataset = ByteDataset()
    (run_dir / "run.json").write_text(json.dumps({"device": str(device), "hardware": describe_hardware(device)}))

    model = spec.build_model(config, device)
    print(
        f"{args.model}/{preset}: {count_params(model) / 1e6:.2f}M params, device {device}, "
        f"{config.max_iters} iters x batch {config.batch_size} x block {config.block_size}"
    )
    train_model = torch.compile(model) if args.compile else model
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    checkpoint = run_dir / "checkpoint.pt"
    started = window_started = time.time()

    def save(step: int, val_loss: float | None) -> None:
        torch.save(
            {
                "model_name": args.model,
                "preset": preset,
                "model": model.state_dict(),
                "config": config.to_dict(),
                "step": step,
                "val_loss": val_loss,
                "device": str(device),
                "train_seconds": round(time.time() - started, 1),
            },
            checkpoint,
        )

    metrics_file = (run_dir / "metrics.csv").open("w", newline="")
    metrics = csv.writer(metrics_file)
    metrics.writerow(["step", "train_loss", "val_loss", "val_bits_per_byte", "elapsed_s"])

    tokens_per_step = config.batch_size * config.block_size
    running_loss, running_steps = 0.0, 0
    val_loss = best_val = None
    step = 0
    model.train()
    try:
        for step in range(1, config.max_iters + 1):
            x, y = dataset.batch("train", config.batch_size, config.block_size, device)
            _, loss = train_model(x, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            running_steps += 1

            if step % config.log_interval == 0 or step == config.max_iters:
                synchronize(device)
                now = time.time()
                speed = running_steps * tokens_per_step / (now - window_started)
                print(
                    f"step {step:5d}/{config.max_iters}  loss {running_loss / running_steps:.4f}  "
                    f"{speed:,.0f} tok/s  elapsed {now - started:,.0f}s"
                )
                running_loss, running_steps = 0.0, 0
                window_started = now

            if step % config.eval_interval == 0 or step == config.max_iters:
                losses = estimate_loss(model, dataset, config, device)
                val_loss = losses["val"]
                bpb = val_loss / math.log(2)
                print(f"  eval: train {losses['train']:.4f}  val {val_loss:.4f}  ({bpb:.3f} bits/byte)")
                metrics.writerow(
                    [step, f"{losses['train']:.4f}", f"{val_loss:.4f}", f"{bpb:.4f}", f"{time.time() - started:.1f}"]
                )
                metrics_file.flush()
                if best_val is None or val_loss < best_val:  # keep the best model, like nanoGPT
                    best_val = val_loss
                    save(step, val_loss)
                window_started = time.time()
    except KeyboardInterrupt:
        print(f"\nInterrupted at step {step}.")
        if not checkpoint.exists():
            save(step, val_loss)
    finally:
        metrics_file.close()

    print(f"Done in {time.time() - started:,.0f}s. Best val {best_val}. Checkpoint: {checkpoint}")
    model.load_state_dict(torch.load(checkpoint, map_location=device)["model"])

    model.eval()
    prompt = torch.tensor([encode(args.prompt)], dtype=torch.long, device=device)
    out = model.generate(prompt, max_new_tokens=200, top_k=3)
    sample = decode(out[0].tolist())
    (run_dir / "sample.txt").write_text(sample)
    print("\n--- sample ---\n" + sample + "\n--------------")


if __name__ == "__main__":
    main()
