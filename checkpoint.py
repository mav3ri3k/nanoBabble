from __future__ import annotations

from pathlib import Path
import re

from flax import nnx
import orbax.checkpoint as ocp

from config import Config


_CKPT_PATTERN = re.compile(r"step_(\d+)$")


def _checkpoint_root(config: Config) -> Path:
    root = Path(config.checkpoint_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


def _checkpoint_path(config: Config, step: int) -> Path:
    return _checkpoint_root(config) / f"step_{step}"


def latest_checkpoint_step(config: Config) -> int | None:
    root = _checkpoint_root(config)
    latest = None
    for child in root.iterdir():
        match = _CKPT_PATTERN.match(child.name)
        if match:
            step = int(match.group(1))
            latest = step if latest is None else max(latest, step)
    return latest


def save_checkpoint(config: Config, step: int, model: nnx.Module, optimizer: nnx.Optimizer) -> Path:
    ckpt_path = _checkpoint_path(config, step)
    payload = {
        "step": step,
        "model": nnx.state(model),
        "optimizer": nnx.state(optimizer),
    }
    checkpointer = ocp.PyTreeCheckpointer()
    checkpointer.save(str(ckpt_path), payload, force=True)
    return ckpt_path


def restore_checkpoint(
    config: Config,
    model: nnx.Module,
    optimizer: nnx.Optimizer,
    step: int | None = None,
) -> int | None:
    restore_step = latest_checkpoint_step(config) if step is None else step
    if restore_step is None:
        return None

    ckpt_path = _checkpoint_path(config, restore_step)
    checkpointer = ocp.PyTreeCheckpointer()
    try:
        payload = checkpointer.restore(str(ckpt_path))
        nnx.update(model, payload["model"])
        nnx.update(optimizer, payload["optimizer"])
        return int(payload["step"])
    except Exception as exc:
        print(f"warning: failed to restore checkpoint `{ckpt_path}`: {exc}")
        print("warning: continuing without checkpoint restore")
        return None
