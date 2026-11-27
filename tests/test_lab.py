import json

import numpy as np
import torch

from lab.config import TrainConfig
from lab.data import decode, encode
from lab.export import export_state
from lab.spec import available_models, load_spec


def test_every_model_has_a_complete_spec():
    assert {"bdh", "nanogpt"} <= set(available_models())
    for name in available_models():
        spec = load_spec(name)
        assert spec.NAME == name
        assert spec.DEFAULT_PRESET in spec.PRESETS
        assert callable(spec.build_model)


def test_config_round_trip_keeps_model_fields():
    config = TrainConfig(model={"n_layer": 2}, max_iters=7)
    assert TrainConfig.from_dict(config.to_dict()) == config


def test_byte_codec_round_trip():
    assert decode(encode("ROMEO: ó")) == "ROMEO: ó"


def test_export_writes_tied_weights_once(tmp_path):
    shared = torch.randn(4, 3)
    state = {
        "model": {"wte": shared, "lm_head": shared, "freqs": torch.randn(5)},
        "config": {"model": {}, "block_size": 8},
        "step": 1,
        "val_loss": 1.0,
    }
    export_state(state, tmp_path, {}, {"freqs"})
    manifest = json.loads((tmp_path / "model.json").read_text())
    assert manifest["tensors"]["lm_head"] == {"alias": "wte"}
    assert manifest["tensors"]["freqs"]["dtype"] == "float32"
    assert (tmp_path / "weights.bin").stat().st_size == 4 * 3 * 2 + 4 * 5
    weights = np.frombuffer((tmp_path / "weights.bin").read_bytes()[:24], dtype="<f2").reshape(4, 3)
    np.testing.assert_allclose(weights, shared.half().numpy())
