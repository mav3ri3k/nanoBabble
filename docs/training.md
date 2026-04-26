# Training

This page covers day-to-day training runs in `nanoBabble`.

## Basic Run

```bash
uv run train.py --config configs/config.toml
```

Quick smoke test:

```bash
uv run main.py --config configs/test.toml
```

## RQ1 Attention Layout Runs

```bash
uv run train.py --config configs/rq1_ffff.toml
uv run train.py --config configs/rq1_s1f1.toml
uv run train.py --config configs/rq1_s3f1.toml
```

## Capo/GPT-2 Training Baseline

Use:

```bash
uv run train.py --config configs/capo_gpt2_template.toml
```

Important:
- Capo ingestion/tokenization happens in `nanoBabble/synth.py`.
- `vocab_size` must be large enough for GPT-2 token IDs (`>= 50257`).

## Config Notes

Core model knobs:
- `dim`, `num_layers`, `num_heads`, `ffn_dim`

Attention layout knobs:
- `attn`, `attn_local`, `attn_global_every`, `swa_window`

Synthetic dataset selection:
- `[synth].dataset` in TOML (`brevo`, `depo`, `mano`, `lano`, `capo`)
