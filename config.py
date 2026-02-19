from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
import tomllib

from synth import get_synth_config_class


SynthConfig = get_synth_config_class()
HASH_EXCLUDE_KEYS = {
    "checkpoint_dir",
    "checkpoint_every_steps",
    "enable_metrics",
    "experiment_name",
    "metrics_db_path",
    "metrics_flush_every",
    "resume",
    "run_description",
    "save_checkpoint",
    "test",
    "train_steps",
}


def _to_plain(value: Any) -> Any:
    if hasattr(value, "__dataclass_fields__"):
        return {k: _to_plain(v) for k, v in asdict(value).items()}
    if isinstance(value, dict):
        return {str(k): _to_plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_plain(v) for v in value]
    return value


@dataclass
class Config:
    data_source: str = "file"
    train_data_path: str = "train.parquet"
    data_column: str = "token_id"
    synth: SynthConfig = field(default_factory=SynthConfig)
    checkpoint_dir: str = "checkpoints"

    seed: int = 42
    batch_size: int = 16
    ctx_len: int = 128
    train_steps: int = 1000
    learning_rate: float = 3e-4
    checkpoint_every_steps: int = 500
    save_checkpoint: bool = True
    resume: bool = False
    test: bool = False
    enable_metrics: bool = True
    metrics_db_path: str = "experiments.db"
    metrics_flush_every: int = 100
    experiment_name: str = "default"
    run_description: str = ""

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
        ignore_raw = raw.pop("ignore", {})
        raw.update(ignore_raw)
        synth_raw = raw.pop("synth", {})
        synth = SynthConfig(**synth_raw)
        return cls(synth=synth, **raw)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_plain_dict(self) -> dict[str, Any]:
        return _to_plain(self)

    def hash_config_dict(self) -> dict[str, Any]:
        full = self.to_plain_dict()
        return {k: v for k, v in full.items() if k not in HASH_EXCLUDE_KEYS}

    def config_hash(self) -> str:
        hash_cfg = self.hash_config_dict()
        blob = json.dumps(hash_cfg, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()


def hash_config_dict(cfg: Config) -> dict[str, Any]:
    return cfg.hash_config_dict()


def config_hash(cfg: Config) -> str:
    return cfg.config_hash()
