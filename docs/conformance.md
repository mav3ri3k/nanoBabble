# Conformance

This page documents the PoLM conformance checks for generation + workflow wiring.

## Run Conformance Checker

```bash
uv run polm_conformance_check.py --synth-root ../synth_data
```

## What It Checks

- Depo training format and answer-mask labels
- Depo appendix QA mode (`k in {K/2, K}`)
- Brevo framing and labels
- Mano answer consistency
- Lano CFG validity (DP-style membership check against `cfg3f`)
- Capo raw output and attribute inclusion
- Capo bioS-style indexing (`N_person_id`, `exposure_index`)
- Capo deterministic employer->city mapping
- Capo Pareto runner dry-run artifact generation

## Interpretation

- `PASS`: check succeeded
- `FAIL`: actionable mismatch/bug in generator or integration path
- `warn=0`: no outstanding coverage warnings in the checker

## Responsibility Split

- `synth_data`: generation-only, paper-facing dataset behavior
- `nanoBabble`: tokenization, ingestion, training, eval orchestration
