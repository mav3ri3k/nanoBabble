# nanoBabble

Minimal JAX/NNX trainer scaffold with:

- Transformer model (`mha`, `mla`, `swa` attention backends)
- Orbax checkpointing
- Polars-based data loading/batching

## Train

```bash
uv run train.py
```

You can optionally provide TOML config:

```bash
uv run train.py --config configs/config.toml
```

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
