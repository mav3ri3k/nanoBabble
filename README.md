# nanoBabble

A minimal JAX/NNX trainer scaffold for transformer experiments.

`nanoBabble` is the more mature continuation of earlier model experiments, focused on training pipeline structure and reproducibility.

## What Works Today

- Config-driven training (`.toml`)
- Transformer model with selectable attention backends:
  - `mha`
  - `mla`
  - `swa`
- Orbax checkpoint save/restore
- Synthetic data integration (`synth_data`-based flow)
- Metric logging to SQLite (`experiments.db`)

## Train

```bash
uv run train.py
```

You can optionally provide TOML config:

```bash
uv run train.py --config configs/config.toml
```

For quick testing:

```bash
uv run main.py --config ./configs/test.toml
```

## Current Limitations

- Training currently supports only `data_source = "synth"`.
- Synthetic data path expects external `../synth_data` modules/files.
- Current training mesh is configured for a 2-device setup in `train.py`.

## Data Contract

Training expects a table file (`.parquet`, `.csv`, `.tsv`, `.jsonl`, `.ndjson`) at `train_data_path`
with a token column (default: `token_id`) containing integer token IDs or list-of-int token IDs.

For local `../synth_data` generation, set config fields directly:

- `data_source = "synth"`
- `[synth]` section in TOML:
- `dataset = "brevo"`, `length = 5000`, `batch_size = 16`, `seed = 42`
- dataset-specific knobs in the same section:
- `brevo_N`, `brevo_multi`
- `depo_N`, `depo_K`, `depo_M`, `depo_qa`, `depo_separator`, `depo_mini_vocab`, `depo_min_tlen`, `depo_max_tlen`, `depo_emit_token_type`
- `mano_L`, `mano_ttype`, `mano_value_mod`, `mano_knowledge_augment`
- `lano_config`, `lano_bos_token`, `lano_eos_token`
- `capo_capo_file`, `capo_fields_dir`, `capo_order`

## Roadmap

- Add non-synthetic dataset path for training
- Expand evaluation/inference utilities
- Improve multi-device sharding ergonomics
- Harden experiment tracking and run metadata
