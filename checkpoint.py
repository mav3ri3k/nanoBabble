from __future__ import annotations

import re
from pathlib import Path

from flax import nnx
import orbax.checkpoint as ocp

from config import Config, config_hash, hash_config_dict


_CKPT_PATTERN = re.compile(r"step_(\d+)$")


class CheckpointManager:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.enabled = bool(cfg.save_checkpoint)
        self._checkpointer = ocp.PyTreeCheckpointer()
        self.run_hash = config_hash(cfg)
        self.root = (Path(cfg.checkpoint_dir).resolve() / self.run_hash)

        if self.enabled:
            self.root.mkdir(parents=True, exist_ok=True)

    def _step_path(self, step: int) -> Path:
        return self.root / f"step_{step}"

    def latest_step(self) -> int | None:
        if not self.enabled or not self.root.exists():
            return None
        latest = None
        for child in self.root.iterdir():
            match = _CKPT_PATTERN.match(child.name)
            if match:
                value = int(match.group(1))
                latest = value if latest is None else max(latest, value)
        return latest

    def save(self, step: int, model: nnx.Module, optimizer: nnx.Optimizer) -> Path | None:
        if not self.enabled:
            return None
        ckpt_path = self._step_path(step)
        payload = {
            "step": step,
            "model": nnx.state(model),
            "optimizer": nnx.state(optimizer),
        }
        custom_metadata = {
            "config_hash": self.run_hash,
            "hash_config": hash_config_dict(self.cfg),
            "config": self.cfg.to_plain_dict(),
        }
        self._checkpointer.save(
            str(ckpt_path),
            payload,
            force=True,
            custom_metadata=custom_metadata,
        )
        return ckpt_path

    def restore(
        self,
        model: nnx.Module,
        optimizer: nnx.Optimizer,
        step: int | None = None,
    ) -> int | None:
        if not self.enabled:
            return None
        restore_step = self.latest_step() if step is None else step
        if restore_step is None:
            return None

        ckpt_path = self._step_path(restore_step)
        try:
            payload = self._checkpointer.restore(str(ckpt_path))
            nnx.update(model, payload["model"])
            nnx.update(optimizer, payload["optimizer"])
            return int(payload["step"])
        except Exception as exc:
            print(f"warning: failed to restore checkpoint `{ckpt_path}`: {exc}")
            print("warning: continuing without checkpoint restore")
            return None


def create_checkpoint_manager(cfg: Config) -> CheckpointManager:
    return CheckpointManager(cfg)
