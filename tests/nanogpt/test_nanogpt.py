import shutil

import numpy as np
import pytest
import torch

from lab.config import TrainConfig
from lab.export import export_state, round_to_export_precision
from models.nanogpt import spec
from tests.conftest import assert_js_matches_pytorch, assert_trains_and_generates, js_logits

CORE = "web/nanogpt/gpt.js"
TINY = TrainConfig(
    model={"n_layer": 2, "n_embd": 32, "n_head": 4, "dropout": 0.0, "bias": True},
    block_size=48,
    batch_size=4,
)


def test_trains_and_generates():
    assert_trains_and_generates(spec, TINY)


def test_js_kv_cache_inference_matches_pytorch(tmp_path):
    # web/nanogpt/gpt.js decodes one token at a time with a KV cache; PyTorch runs the full sequence.
    assert_js_matches_pytorch(spec, TINY, CORE, tmp_path)


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_js_window_refresh_matches_pytorch_on_kept_window(tmp_path):
    # Past block_size the JS core keeps the last half of the window and re-encodes it from position 0.
    torch.manual_seed(0)
    model = spec.build_model(TINY, torch.device("cpu")).eval()
    round_to_export_precision(model)
    export_state({"model": model.state_dict(), "config": TINY.to_dict(), "step": 0, "val_loss": None}, tmp_path, {})

    block = TINY.block_size
    tokens = list((b"Now is the winter of our discontent made glorious summer " * 2)[: block + 5])
    kept = tokens[block // 2 :]  # window after the refresh triggered by token #block, plus later tokens
    with torch.no_grad():
        expected, _ = model(torch.tensor([kept]), torch.tensor([kept]))
    actual = js_logits(CORE, tmp_path, tokens)
    np.testing.assert_allclose(actual[-1], expected[0, -1].numpy(), atol=1e-4, rtol=1e-4)
