from __future__ import annotations

import argparse

from flax import nnx
import jax.numpy as jnp
import optax

from checkpoint import restore_checkpoint, save_checkpoint
from config import Config
from data import batch_iterator, load_token_ids, make_sequences
from experiments import ExperimentTracker
from metrics import MetricsLogger, cross_entropy_loss, token_accuracy
from model import Transformer


def _loss_and_metrics(model: Transformer, x_bl: jnp.ndarray, y_bl: jnp.ndarray):
    logits_blv = model(x_bl)
    loss = cross_entropy_loss(logits_blv, y_bl)
    acc = token_accuracy(logits_blv, y_bl)
    return loss, acc


@nnx.jit
def train_step(model: Transformer, optimizer: nnx.Optimizer, x_bl: jnp.ndarray, y_bl: jnp.ndarray):
    grad_fn = nnx.value_and_grad(_loss_and_metrics, has_aux=True)
    (loss, acc), grads = grad_fn(model, x_bl, y_bl)
    optimizer.update(model, grads)
    return loss, acc


def train(cfg: Config):
    tracker = ExperimentTracker(cfg)
    run_id = tracker.start_run(cfg.experiment_id, cfg)
    metrics_logger = MetricsLogger(cfg.metrics_dir, run_id)
    print(f"run_id={run_id}")

    model = Transformer(cfg=cfg, rngs=nnx.Rngs(cfg.seed))
    optimizer = nnx.Optimizer(model, optax.adamw(cfg.learning_rate), wrt=nnx.Param)

    restored_step = None
    if cfg.resume:
        restored_step = restore_checkpoint(cfg, model, optimizer)
    step = 0 if restored_step is None else restored_step + 1
    if restored_step is not None:
        print(f"Restored checkpoint from step {restored_step}")

    token_ids = load_token_ids(cfg.train_data_path, cfg.data_column)
    sequences = make_sequences(token_ids, cfg.seq_len)

    tokens_per_step = cfg.batch_size * cfg.seq_len
    log_every = max(1, int(cfg.log_every_steps))
    last_loss = 0.0
    last_acc = 0.0

    try:
        for epoch in range(cfg.epochs):
            epoch_seed = cfg.seed + epoch
            for x_bl, y_bl in batch_iterator(
                sequences=sequences,
                batch_size=cfg.batch_size,
                seed=epoch_seed,
                shuffle=True,
            ):
                loss, acc = train_step(model, optimizer, x_bl, y_bl)
                last_loss = float(loss)
                last_acc = float(acc)

                if step % log_every == 0:
                    metrics_path = metrics_logger.log_step(
                        step=step,
                        epoch=epoch,
                        loss=last_loss,
                        accuracy=last_acc,
                        learning_rate=cfg.learning_rate,
                        tokens_seen=step * tokens_per_step,
                    )
                    print(
                        f"step={step} epoch={epoch} loss={last_loss:.4f} "
                        f"acc={last_acc:.4f} metrics={metrics_path}"
                    )

                if step > 0 and step % cfg.checkpoint_every_steps == 0:
                    ckpt = save_checkpoint(cfg, step, model, optimizer)
                    print(f"saved checkpoint: {ckpt}")
                step += 1

        final_ckpt = save_checkpoint(cfg, step, model, optimizer)
        tracker.end_run(
            run_id,
            "completed",
            {
                "final_checkpoint": str(final_ckpt),
                "final_step": step,
                "loss": last_loss,
                "accuracy": last_acc,
            },
        )
        print(f"training complete. final checkpoint: {final_ckpt}")
    except Exception as exc:
        tracker.end_run(run_id, "failed", {"error": str(exc), "step": step})
        raise
    finally:
        tracker.close()


def main():
    parser = argparse.ArgumentParser(description="Train the nanoBabble transformer model.")
    parser.add_argument(
        "--config",
        type=str,
        default="config.toml",
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
