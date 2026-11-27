import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch

from lab.config import TrainConfig
from lab.export import export_state, round_to_export_precision

ROOT = Path(__file__).resolve().parent.parent
SAMPLE_TOKENS = list(b"To be, or not to be: that is the question.")


def js_logits(core: str, model_dir: Path, tokens: list[int]) -> np.ndarray:
    result = subprocess.run(
        ["node", str(ROOT / "tests" / "parity.mjs"), str(ROOT / core), str(model_dir), ",".join(map(str, tokens))],
        capture_output=True,
        text=True,
        check=True,
    )
    return np.array(json.loads(result.stdout))


def assert_js_matches_pytorch(spec, config: TrainConfig, core: str, tmp_path: Path, tokens=SAMPLE_TOKENS):
    """Export a random model, then compare the JS core's step-by-step logits with the PyTorch forward pass."""
    if shutil.which("node") is None:
        pytest.skip("node is not installed")
    torch.manual_seed(0)
    model = spec.build_model(config, torch.device("cpu")).eval()
    round_to_export_precision(model, spec.FLOAT32_TENSORS)
    state = {"model": model.state_dict(), "config": config.to_dict(), "step": 0, "val_loss": None}
    export_state(state, tmp_path, {}, spec.FLOAT32_TENSORS)

    idx = torch.tensor([tokens])
    with torch.no_grad():
        expected, _ = model(idx, idx)  # passing targets makes every model return logits for all positions
    actual = js_logits(core, tmp_path, tokens)
    np.testing.assert_allclose(actual, expected[0].numpy(), atol=1e-4, rtol=1e-4)


def assert_trains_and_generates(spec, config: TrainConfig):
    """Forward shapes, loss decreases when overfitting one batch, generate() appends tokens."""
    torch.manual_seed(0)
    model = spec.build_model(config, torch.device("cpu"))
    x = torch.randint(0, 256, (4, config.block_size))
    y = torch.roll(x, -1, dims=1)
    logits, first = model(x, y)
    assert logits.shape == (4, config.block_size, 256)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    for _ in range(30):
        _, loss = model(x, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    assert loss.item() < first.item() * 0.8
    out = model.eval().generate(torch.tensor([[72, 105]]), max_new_tokens=5, top_k=3)
    assert out.shape == (1, 7)
