"""BDH (Dragon Hatchling) from pathwaycom/bdh. The "full" preset mirrors upstream train.py exactly."""

from lab.config import TrainConfig
from lab.spec import load_vendor_module

NAME = "bdh"
DEFAULT_PRESET = "small"
# RoPE frequencies get multiplied by the position; float16 would visibly shift the phases.
FLOAT32_TENSORS = {"attn.freqs"}


def _model(n_layer, n_embd, mlp_internal_dim_multiplier, n_head=4):
    return {
        "n_layer": n_layer,
        "n_embd": n_embd,
        "n_head": n_head,
        "mlp_internal_dim_multiplier": mlp_internal_dim_multiplier,
        "dropout": 0.1,
        "vocab_size": 256,
    }


PRESETS = {
    # ~15 s on Apple M4 Pro; output is Shakespeare-shaped noise.
    "tiny": TrainConfig(
        model=_model(2, 128, 32), block_size=128, max_iters=300, log_interval=25, eval_interval=100, eval_iters=10
    ),
    # ~20 min on Apple M4 Pro; the model shipped in the web demo (12.7M params).
    "small": TrainConfig(model=_model(4, 256, 64)),
    # Upstream defaults (25.3M params): meant for a GPU, hours on a laptop.
    "full": TrainConfig(
        model=_model(6, 256, 128), block_size=512, batch_size=32, max_iters=3000, log_interval=100, eval_interval=500
    ),
}


def build_model(config: TrainConfig, device):
    bdh = load_vendor_module("bdh", "bdh.py")
    return bdh.BDH(bdh.BDHConfig(**config.model)).to(device)
