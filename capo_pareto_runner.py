from __future__ import annotations

import argparse
import copy
import csv
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import subprocess
from typing import Any

from config import Config


@dataclass
class ModelSpec:
    name: str
    dim: int
    num_layers: int
    num_heads: int
    ffn_dim: int


def _parse_model_spec(raw: str) -> ModelSpec:
    # format: name,dim,num_layers,num_heads,ffn_dim
    parts = [p.strip() for p in raw.split(",")]
    if len(parts) != 5:
        raise ValueError(
            f"Invalid model spec `{raw}`. Expected format: name,dim,num_layers,num_heads,ffn_dim"
        )
    return ModelSpec(
        name=parts[0],
        dim=int(parts[1]),
        num_layers=int(parts[2]),
        num_heads=int(parts[3]),
        ffn_dim=int(parts[4]),
    )


def _fmt_toml(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        return repr(v)
    if isinstance(v, str):
        escaped = v.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    if isinstance(v, list):
        return "[" + ", ".join(_fmt_toml(x) for x in v) + "]"
    raise TypeError(f"Unsupported TOML value type: {type(v)}")


def _write_cfg(path: Path, cfg: Config) -> None:
    data = cfg.to_plain_dict()
    synth = data.pop("synth")

    lines: list[str] = []
    for key, value in data.items():
        if value is None:
            continue
        lines.append(f"{key} = {_fmt_toml(value)}")
    lines.append("")
    lines.append("[synth]")
    for key, value in synth.items():
        if value is None:
            continue
        lines.append(f"{key} = {_fmt_toml(value)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _run_cmd(cmd: list[str], cwd: Path) -> None:
    subprocess.run(cmd, cwd=str(cwd), check=True)


def _pareto_frontier(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    # Maximize total_stored_bits with increasing num_params.
    filtered = [r for r in rows if r.get("status") == "ok"]
    filtered.sort(key=lambda r: (int(r["num_params"]), -float(r["total_stored_bits"])))
    frontier: list[dict[str, Any]] = []
    best_bits = float("-inf")
    for r in filtered:
        bits = float(r["total_stored_bits"])
        if bits > best_bits:
            frontier.append(r)
            best_bits = bits
    return frontier


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Capo capacity sweep and build Pareto frontier.")
    parser.add_argument("--base-config", type=str, default="configs/capo_gpt2_template.toml")
    parser.add_argument(
        "--model-spec",
        action="append",
        default=[],
        help="Repeated. Format: name,dim,num_layers,num_heads,ffn_dim",
    )
    parser.add_argument("--capo-n-values", type=str, default="50000,100000,200000")
    parser.add_argument("--train-steps", type=int, default=2000)
    parser.add_argument("--checkpoint-every-steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--ctx-len", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-samples", type=int, default=64)
    parser.add_argument("--eval-max-new-tokens", type=int, default=24)
    parser.add_argument("--synth-root", type=str, default="../synth_data")
    parser.add_argument("--output-dir", type=str, default="sessions/capo_pareto")
    parser.add_argument("--force-rerun", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent
    base_cfg = Config.from_toml(Path(args.base_config).resolve())

    if not args.model_spec:
        args.model_spec = [
            "mini,64,2,4,256",
            "small,128,4,4,512",
            "base,256,6,8,1024",
        ]
    model_specs = [_parse_model_spec(s) for s in args.model_spec]
    capo_ns = [int(x.strip()) for x in args.capo_n_values.split(",") if x.strip()]

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = Path(args.output_dir).resolve() / stamp
    cfg_dir = out_root / "configs"
    run_dir = out_root / "runs"
    out_root.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, Any]] = []

    for spec in model_specs:
        for capo_n in capo_ns:
            run_name = f"{spec.name}_N{capo_n}"
            cfg = copy.deepcopy(base_cfg)
            cfg.seed = int(args.seed)
            cfg.batch_size = int(args.batch_size)
            cfg.ctx_len = int(args.ctx_len)
            cfg.learning_rate = float(args.learning_rate)
            cfg.train_steps = int(args.train_steps)
            cfg.checkpoint_every_steps = int(args.checkpoint_every_steps)
            cfg.save_checkpoint = True
            cfg.resume = False
            cfg.enable_metrics = False
            cfg.test = True
            cfg.experiment_name = f"capo_{run_name}"
            cfg.run_description = f"capo_pareto_runner {run_name}"

            cfg.vocab_size = 50257
            cfg.dim = int(spec.dim)
            cfg.num_layers = int(spec.num_layers)
            cfg.num_heads = int(spec.num_heads)
            cfg.ffn_dim = int(spec.ffn_dim)

            cfg.synth.dataset = "capo"
            cfg.synth.capo_N = int(capo_n)
            cfg.synth.capo_exposures_per_person = 100

            cfg_path = cfg_dir / f"{run_name}.toml"
            eval_json_path = run_dir / run_name / "eval_summary.json"
            _write_cfg(cfg_path, cfg)

            row: dict[str, Any] = {
                "run_name": run_name,
                "model_name": spec.name,
                "capo_N": capo_n,
                "config_path": str(cfg_path),
                "status": "pending",
            }
            all_rows.append(row)

            if eval_json_path.exists() and not args.force_rerun:
                summary = json.loads(eval_json_path.read_text(encoding="utf-8"))
                row.update(
                    {
                        "status": "ok",
                        "num_params": int(summary["num_params"]),
                        "total_stored_bits": float(summary["total_stored_bits"]),
                        "total_max_bits": float(summary["total_max_bits"]),
                        "normalized_capacity": float(summary["normalized_capacity"]),
                        "bits_per_param": float(summary["bits_per_param"]),
                        "checkpoint_step": summary.get("checkpoint_step"),
                        "eval_json": str(eval_json_path),
                    }
                )
                continue

            train_cmd = ["uv", "run", "train.py", "--config", str(cfg_path)]
            eval_cmd = [
                "uv",
                "run",
                "capo_bits_eval.py",
                "--config",
                str(cfg_path),
                "--samples",
                str(args.eval_samples),
                "--max-new-tokens",
                str(args.eval_max_new_tokens),
                "--seed",
                str(args.seed),
                "--synth-root",
                str(Path(args.synth_root).resolve()),
                "--output-json",
                str(eval_json_path),
            ]
            row["train_cmd"] = " ".join(train_cmd)
            row["eval_cmd"] = " ".join(eval_cmd)

            if args.dry_run:
                row["status"] = "dry_run"
                continue

            try:
                eval_json_path.parent.mkdir(parents=True, exist_ok=True)
                _run_cmd(train_cmd, cwd=repo_root)
                _run_cmd(eval_cmd, cwd=repo_root)
                summary = json.loads(eval_json_path.read_text(encoding="utf-8"))
                row.update(
                    {
                        "status": "ok",
                        "num_params": int(summary["num_params"]),
                        "total_stored_bits": float(summary["total_stored_bits"]),
                        "total_max_bits": float(summary["total_max_bits"]),
                        "normalized_capacity": float(summary["normalized_capacity"]),
                        "bits_per_param": float(summary["bits_per_param"]),
                        "checkpoint_step": summary.get("checkpoint_step"),
                        "eval_json": str(eval_json_path),
                    }
                )
            except Exception as exc:
                row["status"] = "failed"
                row["error"] = str(exc)

    frontier = _pareto_frontier(all_rows)
    report = {
        "created_at": datetime.now().isoformat(),
        "output_root": str(out_root),
        "base_config": str(Path(args.base_config).resolve()),
        "capo_n_values": capo_ns,
        "model_specs": [spec.__dict__ for spec in model_specs],
        "runs": all_rows,
        "pareto_frontier": frontier,
    }

    report_json = out_root / "pareto_report.json"
    report_csv = out_root / "pareto_runs.csv"
    report_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with report_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "run_name",
                "status",
                "model_name",
                "capo_N",
                "num_params",
                "total_stored_bits",
                "total_max_bits",
                "normalized_capacity",
                "bits_per_param",
                "checkpoint_step",
                "eval_json",
                "error",
            ],
        )
        writer.writeheader()
        for row in all_rows:
            writer.writerow({k: row.get(k, "") for k in writer.fieldnames})

    print(f"wrote report: {report_json}")
    print(f"wrote runs csv: {report_csv}")
    print(f"total_runs={len(all_rows)} frontier_points={len(frontier)}")
    if frontier:
        print("frontier:")
        for p in frontier:
            print(
                f"  {p['run_name']} params={p.get('num_params')} bits={p.get('total_stored_bits'):.4f} "
                f"bpp={p.get('bits_per_param'):.8f}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
