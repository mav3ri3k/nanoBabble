# nanoBabble

Minimal JAX/NNX trainer scaffold with:

- Transformer model (`mha`, `mla`, `swa` attention backends)
- Orbax checkpointing
- Polars-based data loading/batching
- Parquet metrics logging
- SQLite experiment tracking

## Train

```bash
uv run train.py
```

You can optionally provide TOML config:

```bash
uv run train.py --config config.toml
```

## Data Contract

Training expects a table file (`.parquet`, `.csv`, `.tsv`, `.jsonl`, `.ndjson`) at `train_data_path`
with a token column (default: `token_id`) containing integer token IDs or list-of-int token IDs.

## Experiment Tracking

For this trainer, the recommended workflow is:

1. Store all runs.
2. Promote only key runs for reporting/comparison.

### Why keep all runs

- Failed/aborted runs are useful for debugging and reproducibility.
- Small tweak runs help explain regressions and improvements.
- Full history makes it easier to recover from mistakes and reproduce results later.

### How to keep comparisons clean

- Keep a small curated set of promoted runs (for example: baseline, best, final, key ablations).
- Mark all other runs as raw history.
- Do not delete raw runs unless storage constraints require archival.

### Suggested run metadata

- `experiment_id`
- `run_id`
- run `status` (`completed`, `failed`, `killed`, `invalid`)
- config (or config hash)
- git commit
- dataset/version
- seed
- optional tags (`baseline`, `retry`, `hotfix`, `smoke`, `ablation`)

Rule of thumb: store everything, curate what you compare.
