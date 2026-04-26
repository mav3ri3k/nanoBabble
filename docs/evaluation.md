# Evaluation

This page covers available evaluation/scoring utilities in `nanoBabble`.

## Capo Capacity Probe (Scaffold)

Run:

```bash
uv run capo_bits_eval.py --config configs/capo_gpt2_template.toml --samples 64
```

This script:
- loads model/checkpoint,
- runs QA-style probes over Capo biographies,
- reports approximate stored bits and `bits_per_param`,
- includes per-attribute diagnostics.

Notes:
- This is an approximation scaffold, not a strict reproduction of the full original paper metric stack.
- Requires config with `vocab_size >= 50257`.

## Capo Pareto Sweep Runner

Run full grid:

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
- `sessions/capo_pareto/<timestamp>/runs/<run_name>/eval_summary.json`
