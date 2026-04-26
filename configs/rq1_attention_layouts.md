# RQ1 Attention Layout Configs

Supported ablation layouts:
- `configs/rq1_ffff.toml`
- `configs/rq1_s1f1.toml`
- `configs/rq1_s3f1.toml`

Run:

```bash
uv run train.py --config configs/rq1_ffff.toml
uv run train.py --config configs/rq1_s1f1.toml
uv run train.py --config configs/rq1_s3f1.toml
```

`S3F1+Head` is intentionally omitted for now because SWA head-expansion/gating is not implemented yet.
