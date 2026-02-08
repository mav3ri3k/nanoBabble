from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import jax.numpy as jnp
import optax
import polars as pl


def cross_entropy_loss(logits_blv: jnp.ndarray, labels_bl: jnp.ndarray) -> jnp.ndarray:
    loss_bl = optax.losses.softmax_cross_entropy_with_integer_labels(
        logits=logits_blv,
        labels=labels_bl,
    )
    return jnp.mean(loss_bl)


def token_accuracy(logits_blv: jnp.ndarray, labels_bl: jnp.ndarray) -> jnp.ndarray:
    preds_bl = jnp.argmax(logits_blv, axis=-1)
    return jnp.mean((preds_bl == labels_bl).astype(jnp.float32))


@dataclass
class MetricsLogger:
    base_dir: Path
    run_id: str

    def __init__(self, metrics_dir: str | Path, run_id: str):
        self.base_dir = Path(metrics_dir) / run_id
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id
        self._buffer: list[dict] = []

    def log_step(
        self,
        *,
        step: int,
        epoch: int,
        loss: float,
        accuracy: float,
        learning_rate: float,
        tokens_seen: int,
    ) -> Path:
        row = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "run_id": self.run_id,
            "step": int(step),
            "epoch": int(epoch),
            "loss": float(loss),
            "accuracy": float(accuracy),
            "learning_rate": float(learning_rate),
            "tokens_seen": int(tokens_seen),
        }
        self._buffer.append(row)
        return self.flush(step=step)

    def flush(self, *, step: int) -> Path:
        if not self._buffer:
            return self.base_dir / f"step_{int(step):08d}.parquet"
        out_path = self.base_dir / f"step_{int(step):08d}.parquet"
        pl.DataFrame(self._buffer).write_parquet(out_path)
        self._buffer.clear()
        return out_path
