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
- `depo_N`, `depo_K`, `depo_M`, `depo_qa`, `depo_qa_mode`, `depo_separator`, `depo_mini_vocab`, `depo_min_tlen`, `depo_max_tlen`, `depo_emit_token_type`
- `mano_L`, `mano_ttype`, `mano_value_mod`, `mano_knowledge_augment`
- `lano_config`, `lano_bos_token`, `lano_eos_token`
- `capo_capo_file`, `capo_fields_dir`, `capo_order`, `capo_N`, `capo_exposures_per_person`

Capo-specific note:
- `synth_data` emits raw biography text for Capo (paper-facing generation).
- `nanoBabble` performs GPT-2 tokenization during ingestion/evaluation.

## Capo Capacity Probe (Scaffold)

Run the approximate bits-per-parameter evaluator:

```bash
uv run capo_bits_eval.py --config configs/capo_gpt2_template.toml --samples 64
```

Important:
- This scaffold requires `vocab_size >= 50257` for GPT-2 token IDs.
- It is not a full replacement for the paper’s complete Pareto frontier protocol.

## Capo Pareto Runner

Run the end-to-end sweep across model sizes and Capo dataset sizes:

```bash
uv run capo_pareto_runner.py --base-config configs/capo_gpt2_template.toml
```

Custom grid example:

```bash
uv run capo_pareto_runner.py \
  --model-spec tiny,64,2,4,256 \
  --model-spec small,128,4,4,512 \
  --capo-n-values 50000,100000,200000 \
  --train-steps 2000 \
  --eval-samples 64
```

Outputs:
- `sessions/capo_pareto/<timestamp>/pareto_report.json`
- `sessions/capo_pareto/<timestamp>/pareto_runs.csv`

## Roadmap

- Add non-synthetic dataset path for training
- Expand evaluation/inference utilities
- Improve multi-device sharding ergonomics
- Harden experiment tracking and run metadata
