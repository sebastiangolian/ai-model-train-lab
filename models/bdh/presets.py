"""Training presets. "full" mirrors the upstream train.py defaults exactly."""

from dataclasses import asdict, dataclass, field


@dataclass
class ModelConfig:
    n_layer: int = 6
    n_embd: int = 256
    n_head: int = 4
    mlp_internal_dim_multiplier: int = 128
    dropout: float = 0.1
    vocab_size: int = 256


@dataclass
class TrainConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    block_size: int = 512
    batch_size: int = 32
    max_iters: int = 3000
    learning_rate: float = 1e-3
    weight_decay: float = 0.1
    log_interval: int = 100
    eval_interval: int = 500
    eval_iters: int = 20
    seed: int = 1337

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "TrainConfig":
        data = dict(data)
        data["model"] = ModelConfig(**data["model"])
        return cls(**data)


PRESETS: dict[str, TrainConfig] = {
    # Smoke run: a couple of minutes, output is still mostly noise.
    "tiny": TrainConfig(
        model=ModelConfig(n_layer=2, n_embd=128, mlp_internal_dim_multiplier=32),
        block_size=128,
        batch_size=16,
        max_iters=300,
        log_interval=25,
        eval_interval=100,
        eval_iters=10,
    ),
    # Laptop run: Shakespeare-looking text (names, line breaks, archaic words).
    "small": TrainConfig(
        model=ModelConfig(n_layer=4, n_embd=256, mlp_internal_dim_multiplier=64),
        block_size=256,
        batch_size=16,
        max_iters=1500,
        log_interval=50,
        eval_interval=250,
    ),
    # Upstream defaults (~25M params). Meant for a GPU; hours on a laptop.
    "full": TrainConfig(),
}
