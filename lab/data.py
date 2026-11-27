"""Tiny Shakespeare as a byte-level dataset (vocab = 256), same as upstream train.py."""

import urllib.request
from pathlib import Path

import numpy as np
import torch

DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DATA_FILE = DATA_DIR / "tinyshakespeare.txt"
TRAIN_FRACTION = 0.9


def fetch(path: Path = DATA_FILE) -> Path:
    """Download the dataset once; later calls reuse the local file."""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading Tiny Shakespeare to {path} ...")
        with urllib.request.urlopen(DATA_URL, timeout=60) as response:
            path.write_bytes(response.read())
    return path


class ByteDataset:
    def __init__(self, path: Path = DATA_FILE):
        data = np.frombuffer(fetch(path).read_bytes(), dtype=np.uint8)
        split = int(TRAIN_FRACTION * len(data))
        self.splits = {"train": data[:split], "val": data[split:]}

    def batch(self, split: str, batch_size: int, block_size: int, device: torch.device):
        """Random contiguous windows; targets are inputs shifted by one byte."""
        data = self.splits[split]
        starts = np.random.randint(0, len(data) - block_size - 1, size=batch_size)
        windows = np.stack([data[s : s + block_size + 1] for s in starts]).astype(np.int64)
        windows = torch.from_numpy(windows)
        x, y = windows[:, :-1], windows[:, 1:]
        return x.to(device), y.to(device)


def encode(text: str) -> list[int]:
    return list(text.encode("utf-8"))


def decode(tokens: list[int]) -> str:
    return bytes(tokens).decode("utf-8", errors="replace")


if __name__ == "__main__":
    print(fetch())
