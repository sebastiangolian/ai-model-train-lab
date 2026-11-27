"""Export a trained checkpoint for the browser demo: web/<model>/model/model.json + weights.bin.

Weights are stored as float16 (half the download), except tensors the model lists in FLOAT32_TENSORS.
Tensors that share storage (tied weights) are written once and referenced by "alias".

Usage: python -m lab.export --model bdh --preset small [--out web/bdh/model]
"""

import argparse
import csv
import json
from pathlib import Path

import torch

from .spec import RUNS_DIR, WEB_DIR, available_models, load_spec, upstream_ref


def export_state(state: dict, out_dir: Path, metadata: dict, float32_tensors: set[str] = frozenset()) -> Path:
    """Write manifest + packed little-endian tensors; returns the manifest path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    tensors, chunks, offset, seen = {}, [], 0, {}
    for name, tensor in state["model"].items():
        key = (tensor.untyped_storage().data_ptr(), tensor.storage_offset(), tuple(tensor.shape))
        if key in seen:
            tensors[name] = {"alias": seen[key]}
            continue
        seen[key] = name
        dtype = "float32" if name in float32_tensors else "float16"
        array = tensor.detach().cpu().numpy().astype("<f4" if dtype == "float32" else "<f2")
        data = array.tobytes()
        tensors[name] = {"dtype": dtype, "shape": list(array.shape), "offset": offset}
        padding = -len(data) % 4  # keep every tensor 4-byte aligned for typed-array views
        chunks.append(data + b"\0" * padding)
        offset += len(data) + padding

    (out_dir / "weights.bin").write_bytes(b"".join(chunks))
    unique = [t for n, t in state["model"].items() if "alias" not in tensors[n]]
    manifest = {
        "config": state["config"]["model"],
        "training": {k: v for k, v in state["config"].items() if k != "model"},
        "step": state["step"],
        "val_loss": state["val_loss"],
        "params": int(sum(t.numel() for t in unique)),  # includes small buffers (e.g. BDH RoPE freqs)
        "train_seconds": state.get("train_seconds"),
        "device": state.get("device"),
        "tensors": tensors,
        **metadata,
    }
    path = out_dir / "model.json"
    path.write_text(json.dumps(manifest, indent=2))
    return path


def round_to_export_precision(model: torch.nn.Module, float32_tensors: set[str] = frozenset()) -> None:
    """Apply the same float16 rounding as export_state, so references match the browser exactly."""
    with torch.no_grad():
        for name, tensor in model.state_dict().items():
            if name not in float32_tensors:
                tensor.copy_(tensor.half().float())


def train_seconds_from_metrics(run_dir: Path) -> float | None:
    """Fallback for checkpoints saved before train_seconds was recorded."""
    path = run_dir / "metrics.csv"
    if not path.exists():
        return None
    rows = list(csv.DictReader(path.open()))
    return float(rows[-1]["elapsed_s"]) if rows else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, choices=available_models())
    parser.add_argument("--preset")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--out", type=Path, help="default: web/<model>/model")
    args = parser.parse_args()

    spec = load_spec(args.model)
    preset = args.preset or spec.DEFAULT_PRESET
    path = args.checkpoint or RUNS_DIR / args.model / preset / "checkpoint.pt"
    if not path.exists():
        raise SystemExit(f"No checkpoint at {path}. Train first: make train MODEL={args.model} PRESET={preset}")
    state = torch.load(path, map_location="cpu")
    if state.get("train_seconds") is None:
        state["train_seconds"] = train_seconds_from_metrics(path.parent)
    out = args.out or WEB_DIR / args.model / "model"
    metadata = {"model_name": args.model, "preset": preset, "upstream_ref": upstream_ref(args.model)}
    manifest = export_state(state, out, metadata, spec.FLOAT32_TENSORS)
    size = (out / "weights.bin").stat().st_size / 1e6
    print(f"Exported {path} -> {manifest} (+ weights.bin, {size:.1f} MB)")


if __name__ == "__main__":
    main()
