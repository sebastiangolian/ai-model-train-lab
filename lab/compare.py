"""Compare all training runs (params, final/best validation loss, training time) and write web/results.json.

Usage: python -m lab.compare
"""

import csv
import json

import torch

from .config import TrainConfig
from .spec import RUNS_DIR, WEB_DIR, available_models, load_spec

RESULTS_FILE = WEB_DIR / "results.json"


def collect() -> list[dict]:
    runs = []
    for metrics_path in sorted(RUNS_DIR.glob("*/*/metrics.csv")):
        run_dir = metrics_path.parent
        model_name = run_dir.parent.name
        history = list(csv.DictReader(metrics_path.open()))
        if not history or model_name not in available_models():
            continue
        config = TrainConfig.from_dict(json.loads((run_dir / "config.json").read_text()))
        model = load_spec(model_name).build_model(config, torch.device("cpu"))
        info_path = run_dir / "run.json"
        info = json.loads(info_path.read_text()) if info_path.exists() else {}
        best = min(history, key=lambda r: float(r["val_loss"]))
        last = history[-1]
        runs.append(
            {
                "model": model_name,
                "preset": run_dir.name,
                "params": sum(p.numel() for p in model.parameters()),
                "steps": int(last["step"]),
                "batch_size": config.batch_size,
                "block_size": config.block_size,
                "tokens": int(last["step"]) * config.batch_size * config.block_size,
                "best_val_bits_per_byte": float(best["val_bits_per_byte"]),
                "best_step": int(best["step"]),
                "train_minutes": round(float(last["elapsed_s"]) / 60, 1),
                "hardware": info.get("hardware", "unknown"),
            }
        )
    return runs


def main() -> None:
    runs = collect()
    header = ["model", "preset", "params", "steps", "tokens", "best val bits/byte", "train time", "hardware"]
    rows = [
        [
            r["model"],
            r["preset"],
            f"{r['params'] / 1e6:.1f}M",
            str(r["steps"]),
            f"{r['tokens'] / 1e6:.1f}M",
            f"{r['best_val_bits_per_byte']:.3f} @{r['best_step']}",
            f"{r['train_minutes']} min",
            r["hardware"],
        ]
        for r in runs
    ]
    widths = [max(len(row[i]) for row in [header, *rows]) for i in range(len(header))]
    for row in [header, ["-" * w for w in widths], *rows]:
        print("  ".join(cell.ljust(w) for cell, w in zip(row, widths)))
    RESULTS_FILE.write_text(json.dumps({"runs": runs}, indent=2))
    print(f"\nWrote {RESULTS_FILE}")


if __name__ == "__main__":
    main()
