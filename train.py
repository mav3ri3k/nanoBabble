from __future__ import annotations

import argparse
import numpy as np

from flax import nnx
import jax
import jax.numpy as jnp
import optax
from tqdm.auto import tqdm
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from checkpoint import restore_checkpoint, save_checkpoint
from config import Config
from model import Transformer
from synth import get_synth_batch_iterator

jnp.set_printoptions(threshold=jnp.inf)

def _loss(model: Transformer, x_bl: jnp.ndarray, y_bl: jnp.ndarray, l_bl: jnp.ndarray):
    # jax.debug.visualize_array_sharding(x_bl)
    # print(jax.typeof(x_bl))
    logits_blv = model(x_bl)
    # print(logits_blv.shape)
    # print(jax.typeof(logits_blv))
    # jax.debug.print(logits_blv.sharding)
    loss_bl = optax.losses.softmax_cross_entropy_with_integer_labels(
        logits=logits_blv,
        labels=y_bl,
    )
    mask_bl = (l_bl == 1).astype(loss_bl.dtype)
    masked_loss = loss_bl * mask_bl
    denom = jnp.maximum(jnp.sum(mask_bl), 1.0)
    loss = jnp.sum(masked_loss) / denom
    return loss


@nnx.jit
def train_step(
    model: Transformer,
    optimizer: nnx.Optimizer,
    x_bl: jnp.ndarray,
    y_bl: jnp.ndarray,
    l_bl: jnp.ndarray,
):
    grad_fn = nnx.value_and_grad(_loss)
    loss, grads = grad_fn(model, x_bl, y_bl, l_bl)
    optimizer.update(model, grads)
    return loss


def train(cfg: Config):
    if cfg.data_source.lower() != "synth":
        raise ValueError("Only `data_source = \"synth\"` is supported.")

    if not jax._src.xla_bridge.backends_are_initialized():
      jax.config.update('jax_num_cpu_devices', 2)
    print(jax.devices())

    mesh = jax.make_mesh((2,), ('batch',))
    jax.set_mesh(mesh)

    model = Transformer(cfg=cfg, mesh=mesh, rngs=nnx.Rngs(cfg.seed))
    optimizer = nnx.Optimizer(model, optax.adamw(cfg.learning_rate), wrt=nnx.Param)

    restored_step = None
    if cfg.resume:
        restored_step = restore_checkpoint(cfg, model, optimizer)
    step = 0 if restored_step is None else restored_step + 1
    if restored_step is not None:
        print(f"Restored checkpoint from step {restored_step}")

    synth_batch_iterator = get_synth_batch_iterator()

    tokens_per_step = cfg.batch_size * cfg.ctx_len
    target_steps = max(1, int(cfg.train_steps))
    log_every = max(1, int(cfg.log_every_steps))
    last_loss = 0.0
    progress = tqdm(range(step, target_steps), total=target_steps, initial=step, desc="train")
    for current_step in progress:
        x_bl, l_bl, y_bl = synth_batch_iterator(
            global_seed=cfg.seed,
            step=current_step,
            batch_size=cfg.batch_size,
            ctx_len=cfg.ctx_len,
            synth_cfg=cfg.synth,
        )
        x_bl = jnp.asarray(x_bl, dtype=jnp.int32)
        l_bl = jnp.asarray(l_bl, dtype=jnp.int32)
        y_bl = jnp.asarray(y_bl, dtype=jnp.int32)

        x_bl = jax.device_put(x_bl, NamedSharding(mesh, P("batch")))
        y_bl = jax.device_put(y_bl, NamedSharding(mesh, P("batch")))
        l_bl = jax.device_put(l_bl, NamedSharding(mesh, P("batch")))

        loss = train_step(model, optimizer, x_bl, y_bl, l_bl)
        last_loss = float(loss)

        if current_step % log_every == 0:
            progress.set_postfix(
                step=current_step,
                loss=f"{last_loss:.4f}",
                tokens_seen=current_step * tokens_per_step,
            )

        if current_step > 0 and current_step % cfg.checkpoint_every_steps == 0:
            ckpt = save_checkpoint(cfg, current_step, model, optimizer)
            progress.write(f"saved checkpoint: {ckpt}")

    final_ckpt = save_checkpoint(cfg, target_steps, model, optimizer)
    print(f"training complete. final checkpoint: {final_ckpt} last_loss={last_loss:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Train the nanoBabble transformer model.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.toml",
        help="Path to TOML config file. Falls back to Config() defaults if file is missing.",
    )
    args = parser.parse_args()

    try:
        cfg = Config.from_toml(args.config)
    except FileNotFoundError:
        print(f"config file `{args.config}` not found; using Config() defaults")
        cfg = Config()
    train(cfg)


if __name__ == "__main__":
    main()
