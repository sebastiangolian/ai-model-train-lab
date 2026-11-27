"""nanoGPT (karpathy/nanoGPT): a plain GPT-2 style Transformer, the baseline to compare new architectures with.

Presets are parameter-matched to the BDH presets and use the same training recipe (bytes, AdamW,
constant lr 1e-3, dropout 0.1, same batch/block/iterations), so the two are directly comparable.
"""

import contextlib
import io

from lab.config import TrainConfig
from lab.spec import load_vendor_module

NAME = "nanogpt"
DEFAULT_PRESET = "small-long"  # the model shipped in the web demo
FLOAT32_TENSORS: set[str] = set()


def _model(n_layer, n_embd, n_head):
    return {"n_layer": n_layer, "n_embd": n_embd, "n_head": n_head, "dropout": 0.1, "bias": True}


PRESETS = {
    "tiny": TrainConfig(  # 1.9M params (BDH tiny: 1.6M)
        model=_model(4, 192, 4), block_size=128, max_iters=300, log_interval=25, eval_interval=100, eval_iters=10
    ),
    "small": TrainConfig(model=_model(7, 384, 6)),  # 12.6M params (BDH small: 12.7M), same data budget
    # Same model, ~same wall-clock as BDH small on Apple M4 Pro (~20 min): GPT steps are ~7x cheaper.
    "small-long": TrainConfig(model=_model(7, 384, 6), max_iters=10000, log_interval=250, eval_interval=500),
    "full": TrainConfig(  # 25.6M params (BDH full: 25.3M)
        model=_model(8, 512, 8), block_size=512, batch_size=32, max_iters=3000, log_interval=100, eval_interval=500
    ),
}


def build_model(config: TrainConfig, device):
    upstream = load_vendor_module("nanogpt", "model.py")
    gpt_config = upstream.GPTConfig(block_size=config.block_size, vocab_size=256, **config.model)
    with contextlib.redirect_stdout(io.StringIO()):  # upstream prints the parameter count
        model = upstream.GPT(gpt_config)
    return model.to(device)
