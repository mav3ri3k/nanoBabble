from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import polars as pl


def _read_table(path: str | Path) -> pl.DataFrame:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pl.read_parquet(path)
    if suffix in {".csv", ".tsv"}:
        return pl.read_csv(path, separator="\t" if suffix == ".tsv" else ",")
    if suffix in {".jsonl", ".ndjson"}:
        return pl.read_ndjson(path)
    raise ValueError(f"Unsupported data extension: {suffix}")


def load_token_ids(path: str | Path, column: str) -> jnp.ndarray:
    df = _read_table(path)
    if column not in df.columns:
        raise ValueError(f"Column `{column}` not found in {path}.")
    token_series = df[column]

    if token_series.dtype == pl.List(pl.Int64) or token_series.dtype == pl.List(pl.Int32):
        token_series = token_series.explode()

    if token_series.dtype not in (pl.Int32, pl.Int64):
        token_series = token_series.cast(pl.Int32, strict=False)
    if token_series.null_count() > 0:
        token_series = token_series.drop_nulls()

    tokens = token_series.to_list()
    if len(tokens) == 0:
        raise ValueError(f"No tokens found in `{path}` column `{column}`.")
    return jnp.asarray(tokens, dtype=jnp.int32)


def make_sequences(token_ids: jnp.ndarray, seq_len: int) -> jnp.ndarray:
    block_size = seq_len + 1
    n_blocks = token_ids.shape[0] // block_size
    if n_blocks == 0:
        raise ValueError(
            f"Need at least {block_size} tokens to build one sample, got {token_ids.shape[0]}."
        )
    usable = token_ids[: n_blocks * block_size]
    return usable.reshape((n_blocks, block_size))


def batch_iterator(sequences: jnp.ndarray, batch_size: int, seed: int, shuffle: bool = True):
    if shuffle:
        key = jax.random.PRNGKey(seed)
        perm = jax.random.permutation(key, sequences.shape[0])
        sequences = sequences[perm]

    x = sequences[:, :-1]
    y = sequences[:, 1:]
    n_batches = x.shape[0] // batch_size
    for i in range(n_batches):
        start = i * batch_size
        end = start + batch_size
        yield x[start:end], y[start:end]
