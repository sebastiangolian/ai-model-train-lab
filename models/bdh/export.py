"""Export a trained checkpoint for the browser demo (web/bdh/model/): model.json (manifest) + weights.bin.

Weights are stored as float16 (half the download); the RoPE frequency buffer stays float32
because positions multiply it and half precision would visibly shift the phases.

Usage: python -m models.bdh.export --preset small [--out web/bdh/model]
"""

import argparse
import json
from pathlib import Path

import torch

from .train import RUNS_DIR
from .upstream import VENDOR_DIR

WEB_MODEL_DIR = Path(__file__).resolve().parents[2] / "web" / "bdh" / "model"
FLOAT32_TENSORS = {"attn.freqs"}


def export_state(state: dict, out_dir: Path, metadata: dict) -> Path:
    """Write manifest + packed little-endian tensors; returns the manifest path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    tensors, chunks, offset = {}, [], 0
    for name, tensor in state["model"].items():
        dtype = "float32" if name in FLOAT32_TENSORS else "float16"
        array = tensor.detach().cpu().numpy().astype("<f4" if dtype == "float32" else "<f2")
        data = array.tobytes()
        tensors[name] = {"dtype": dtype, "shape": list(array.shape), "offset": offset}
        padding = -len(data) % 4  # keep every tensor 4-byte aligned for typed-array views
        chunks.append(data + b"\0" * padding)
        offset += len(data) + padding

    (out_dir / "weights.bin").write_bytes(b"".join(chunks))
    manifest = {
        "config": state["config"]["model"],
        "training": {k: v for k, v in state["config"].items() if k != "model"},
        "step": state["step"],
        "val_loss": state["val_loss"],
        "params": int(sum(t.numel() for t in state["model"].values() if t.ndim > 0 and t.numel() > 1)),
        "tensors": tensors,
        **metadata,
    }
    path = out_dir / "model.json"
    path.write_text(json.dumps(manifest, indent=2))
    return path


def round_to_export_precision(model: torch.nn.Module) -> None:
    """Apply the same float16 rounding as export_state, so references match the browser exactly."""
    with torch.no_grad():
        for name, tensor in model.state_dict().items():
            if name not in FLOAT32_TENSORS:
                tensor.copy_(tensor.half().float())


def upstream_ref() -> str:
    head = VENDOR_DIR / ".git" / "HEAD"
    return head.read_text().strip() if head.exists() else "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--preset", default="small")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--out", type=Path, default=WEB_MODEL_DIR)
    args = parser.parse_args()

    path = args.checkpoint or RUNS_DIR / args.preset / "checkpoint.pt"
    if not path.exists():
        raise SystemExit(f"No checkpoint at {path}. Train first: make train MODEL=bdh PRESET={args.preset}")
    state = torch.load(path, map_location="cpu")
    manifest = export_state(state, args.out, {"preset": args.preset, "upstream_ref": upstream_ref()})
    size = (args.out / "weights.bin").stat().st_size / 1e6
    print(f"Exported {path} -> {manifest} (+ weights.bin, {size:.1f} MB)")


if __name__ == "__main__":
    main()
