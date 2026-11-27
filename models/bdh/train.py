"""Train BDH on Tiny Shakespeare with periodic validation, metrics CSV and checkpoints.

Usage: python -m models.bdh.train --preset small [--iters N] [--device mps]
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
from .device import pick_device, synchronize
from .presets import PRESETS, TrainConfig
from .upstream import BDH, BDHConfig

RUNS_DIR = Path(__file__).resolve().parents[2] / "runs" / "bdh"


def build_model(config: TrainConfig, device: torch.device) -> BDH:
    return BDH(BDHConfig(**config.model.__dict__)).to(device)


def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


@torch.no_grad()
def estimate_loss(model, dataset, config: TrainConfig, device) -> dict[str, float]:
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


def save_checkpoint(path: Path, model, config: TrainConfig, step: int, val_loss: float | None):
    torch.save(
        {
            "model": model.state_dict(),
            "config": config.to_dict(),
            "step": step,
            "val_loss": val_loss,
        },
        path,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--preset", choices=PRESETS, default="small")
    parser.add_argument("--iters", type=int, help="override max_iters from the preset")
    parser.add_argument("--batch-size", type=int, help="override batch_size from the preset")
    parser.add_argument("--block-size", type=int, help="override block_size from the preset")
    parser.add_argument("--device", default="auto", help="auto | cpu | mps | cuda")
    parser.add_argument("--compile", action="store_true", help="wrap the model in torch.compile")
    parser.add_argument("--run-dir", type=Path, help="output directory (default: runs/bdh/<preset>)")
    parser.add_argument("--prompt", default="To be or ", help="prompt for the post-training sample")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = TrainConfig.from_dict(PRESETS[args.preset].to_dict())
    if args.iters is not None:
        config.max_iters = args.iters
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.block_size is not None:
        config.block_size = args.block_size

    run_dir = args.run_dir or RUNS_DIR / args.preset
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(config.to_dict(), indent=2))

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    device = pick_device(args.device)
    dataset = ByteDataset()

    model = build_model(config, device)
    print(
        f"Preset {args.preset}: {count_params(model) / 1e6:.2f}M params, device {device}, "
        f"{config.max_iters} iters x batch {config.batch_size} x block {config.block_size}"
    )
    train_model = torch.compile(model) if args.compile else model
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )

    metrics_file = (run_dir / "metrics.csv").open("w", newline="")
    metrics = csv.writer(metrics_file)
    metrics.writerow(["step", "train_loss", "val_loss", "val_bits_per_byte", "elapsed_s"])

    checkpoint = run_dir / "checkpoint.pt"
    tokens_per_step = config.batch_size * config.block_size
    started = window_started = time.time()
    running_loss, running_steps = 0.0, 0
    val_loss = None
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
                save_checkpoint(checkpoint, model, config, step, val_loss)
                window_started = time.time()
    except KeyboardInterrupt:
        print(f"\nInterrupted at step {step}, saving checkpoint ...")
        save_checkpoint(checkpoint, model, config, step, val_loss)
    finally:
        metrics_file.close()

    print(f"Done in {time.time() - started:,.0f}s. Checkpoint: {checkpoint}")

    model.eval()
    prompt = torch.tensor([encode(args.prompt)], dtype=torch.long, device=device)
    out = model.generate(prompt, max_new_tokens=200, top_k=3)
    sample = decode(out[0].tolist())
    (run_dir / "sample.txt").write_text(sample)
    print("\n--- sample ---\n" + sample + "\n--------------")


if __name__ == "__main__":
    main()
