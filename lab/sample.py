"""Generate text from a trained checkpoint.

Usage: python -m lab.sample --model bdh --preset small --prompt "ROMEO:" --tokens 400
"""

import argparse
from pathlib import Path

import torch

from .config import TrainConfig
from .data import decode, encode
from .device import pick_device
from .spec import RUNS_DIR, available_models, load_spec


def load_checkpoint(model_name: str, preset: str | None, checkpoint: Path | None, device):
    spec = load_spec(model_name)
    path = checkpoint or RUNS_DIR / model_name / (preset or spec.DEFAULT_PRESET) / "checkpoint.pt"
    if not path.exists():
        raise SystemExit(f"No checkpoint at {path}. Train first: make train MODEL={model_name}")
    state = torch.load(path, map_location=device)
    config = TrainConfig.from_dict(state["config"])
    model = spec.build_model(config, device)
    model.load_state_dict(state["model"])
    return model.eval(), state, path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, choices=available_models())
    parser.add_argument("--preset", help="reads runs/<model>/<preset>/checkpoint.pt")
    parser.add_argument("--checkpoint", type=Path, help="explicit checkpoint path (overrides --preset)")
    parser.add_argument("--prompt", default="ROMEO:\n")
    parser.add_argument("--tokens", type=int, default=400, help="number of bytes to generate")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)
    device = pick_device(args.device)
    model, state, path = load_checkpoint(args.model, args.preset, args.checkpoint, device)
    print(f"Loaded {path} (step {state['step']}, val loss {state['val_loss']})\n")

    prompt = torch.tensor([encode(args.prompt)], dtype=torch.long, device=device)
    for i in range(args.num_samples):
        out = model.generate(prompt, max_new_tokens=args.tokens, temperature=args.temperature, top_k=args.top_k)
        if args.num_samples > 1:
            print(f"=== sample {i + 1} ===")
        print(decode(out[0].tolist()))
        print()


if __name__ == "__main__":
    main()
