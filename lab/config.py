from dataclasses import asdict, dataclass, field


@dataclass
class TrainConfig:
    """Training recipe shared by every model; `model` holds the architecture's own config fields."""

    model: dict = field(default_factory=dict)
    block_size: int = 256
    batch_size: int = 16
    max_iters: int = 1500
    learning_rate: float = 1e-3
    weight_decay: float = 0.1
    log_interval: int = 50
    eval_interval: int = 250
    eval_iters: int = 20
    seed: int = 1337

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "TrainConfig":
        return cls(**{**data, "model": dict(data["model"])})

    def copy(self) -> "TrainConfig":
        return TrainConfig.from_dict(self.to_dict())
