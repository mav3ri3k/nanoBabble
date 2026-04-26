# Quickstart

Run these three commands from the `nanoBabble` repo root.

## 1) Train (smoke)

```bash
uv run main.py --config configs/test.toml
```

## 2) Conformance Check

```bash
uv run polm_conformance_check.py --synth-root ../synth_data
```

## 3) Capo Pareto Sweep

```bash
uv run capo_pareto_runner.py --base-config configs/capo_gpt2_template.toml
```
