import torch

from models.bdh.presets import ModelConfig, TrainConfig
from models.bdh.train import build_model


def tiny_config() -> TrainConfig:
    return TrainConfig(
        model=ModelConfig(n_layer=1, n_embd=32, n_head=2, mlp_internal_dim_multiplier=4, dropout=0.0),
        block_size=16,
        batch_size=4,
    )


def test_forward_returns_logits_and_loss():
    model = build_model(tiny_config(), torch.device("cpu"))
    x = torch.randint(0, 256, (4, 16))
    logits, loss = model(x, x)
    assert logits.shape == (4, 16, 256)
    assert loss.ndim == 0 and torch.isfinite(loss)


def test_overfits_single_batch():
    torch.manual_seed(0)
    model = build_model(tiny_config(), torch.device("cpu"))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    x = torch.randint(0, 256, (4, 16))
    y = torch.roll(x, -1, dims=1)
    _, first = model(x, y)
    for _ in range(30):
        _, loss = model(x, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    assert loss.item() < first.item() * 0.8


def test_generate_appends_tokens():
    model = build_model(tiny_config(), torch.device("cpu")).eval()
    out = model.generate(torch.tensor([[72, 105]]), max_new_tokens=5, top_k=3)
    assert out.shape == (1, 7)
