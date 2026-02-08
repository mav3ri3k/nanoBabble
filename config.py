from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import tomllib


@dataclass
class Config:
    train_data_path: str = "train.parquet"
    data_column: str = "token_id"
    checkpoint_dir: str = "checkpoints"
    metrics_dir: str = "metrics"
    experiments_db: str = "experiments.db"
    experiment_id: str = "default"

    seed: int = 42
    batch_size: int = 16
    seq_len: int = 128
    epochs: int = 5
    learning_rate: float = 3e-4
    checkpoint_every_steps: int = 500
    log_every_steps: int = 50
    resume: bool = False

    vocab_size: int = 32000
    dim: int = 256
    num_heads: int = 8
    num_layers: int = 6
    ffn_dim: int = 1024

    attn: str = "mha"
    attn_local: str = "swa"
    attn_global_every: int = 1
    swa_window: int = 128
    mla_latent_dim: int = 128

    @classmethod
    def from_toml(cls, path: str | Path) -> "Config":
        with open(path, "rb") as f:
            raw = tomllib.load(f)
        return cls(**raw)

    def to_dict(self) -> dict:
        return asdict(self)
