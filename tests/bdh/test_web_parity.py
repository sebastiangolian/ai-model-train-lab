"""The browser implementation (web/bdh/bdh.js, recurrent form) must match upstream PyTorch (parallel form)."""

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch

from models.bdh.export import export_state, round_to_export_precision
from models.bdh.presets import ModelConfig, TrainConfig
from models.bdh.train import build_model

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_js_recurrent_inference_matches_pytorch(tmp_path):
    torch.manual_seed(0)
    config = TrainConfig(model=ModelConfig(n_layer=2, n_embd=32, n_head=2, mlp_internal_dim_multiplier=8))
    model = build_model(config, torch.device("cpu")).eval()
    round_to_export_precision(model)
    export_state({"model": model.state_dict(), "config": config.to_dict(), "step": 0, "val_loss": None}, tmp_path, {})

    tokens = list(b"To be, or not to be: that is the question.")
    with torch.no_grad():
        expected, _ = model(torch.tensor([tokens]))

    result = subprocess.run(
        ["node", str(ROOT / "tests" / "bdh" / "parity.mjs"), str(tmp_path), ",".join(map(str, tokens))],
        capture_output=True,
        text=True,
        check=True,
    )
    actual = np.array(json.loads(result.stdout))
    np.testing.assert_allclose(actual, expected[0].numpy(), atol=1e-4, rtol=1e-4)
