"""Generate text from a trained BDH checkpoint.

Usage: python -m models.bdh.sample --preset small --prompt "ROMEO:" --tokens 400
"""

import argparse
from pathlib import Path

import torch

from .data import decode, encode
from .device import pick_device
from .presets import TrainConfig
from .train import RUNS_DIR, build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--preset", default="small", help="reads runs/bdh/<preset>/checkpoint.pt")
    parser.add_argument("--checkpoint", type=Path, help="explicit checkpoint path (overrides --preset)")
    parser.add_argument("--prompt", default="ROMEO:\n")
    parser.add_argument("--tokens", type=int, default=400, help="number of bytes to generate")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = args.checkpoint or RUNS_DIR / args.preset / "checkpoint.pt"
    if not path.exists():
        raise SystemExit(f"No checkpoint at {path}. Train first: make train MODEL=bdh PRESET={args.preset}")
    if args.seed is not None:
        torch.manual_seed(args.seed)

    device = pick_device(args.device)
    state = torch.load(path, map_location=device)
    config = TrainConfig.from_dict(state["config"])
    model = build_model(config, device)
    model.load_state_dict(state["model"])
    model.eval()
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
