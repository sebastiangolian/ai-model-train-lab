from tests.conftest import assert_js_matches_pytorch, assert_trains_and_generates
from lab.config import TrainConfig
from models.bdh import spec

TINY = TrainConfig(
    model={"n_layer": 2, "n_embd": 32, "n_head": 2, "mlp_internal_dim_multiplier": 8, "dropout": 0.0, "vocab_size": 256},
    block_size=16,
    batch_size=4,
)


def test_trains_and_generates():
    assert_trains_and_generates(spec, TINY)


def test_js_recurrent_inference_matches_pytorch(tmp_path):
    # web/bdh/bdh.js runs attention as a fixed-size recurrent state; PyTorch runs it in parallel.
    assert_js_matches_pytorch(spec, TINY, "web/bdh/bdh.js", tmp_path)
