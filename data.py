from __future__ import annotations

from pathlib import Path

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


def make_sequences(token_ids: jnp.ndarray, ctx_len: int) -> jnp.ndarray:
    block_size = ctx_len + 1
    n_blocks = token_ids.shape[0] // block_size
    if n_blocks == 0:
        raise ValueError(
            f"Need at least {block_size} tokens to build one sample, got {token_ids.shape[0]}."
        )
    usable = token_ids[: n_blocks * block_size]
    return usable.reshape((n_blocks, block_size))
