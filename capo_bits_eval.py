from __future__ import annotations

import argparse
from dataclasses import dataclass
import importlib.util
import json
import math
from pathlib import Path
import re
from typing import Any

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import tiktoken

from checkpoint import create_checkpoint_manager
from config import Config
from model import Transformer


MONTH_NAMES = {
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
}


@dataclass
class Probe:
    name: str
    question: str
    expected: str
    max_bits: float


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", " ", text.lower())).strip()


def _month_day_year(text: str) -> tuple[str | None, str | None, str | None]:
    lowered = text.lower()
    month = None
    for m in MONTH_NAMES:
        if m in lowered:
            month = m
            break
    year_match = re.search(r"\b(19|20)\d{2}\b", text)
    day_match = re.search(r"\b([1-9]|[12]\d|3[01])\b", text)
    year = year_match.group(0) if year_match else None
    day = day_match.group(0) if day_match else None
    return month, day, year


def _load_synth_main(synth_root: Path):
    main_path = (synth_root / "main.py").resolve()
    if not main_path.exists():
        raise FileNotFoundError(f"synth_data main.py not found at `{main_path}`.")
    spec = importlib.util.spec_from_file_location("synth_data_eval_main", main_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load synth_data main module from `{main_path}`.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _make_mesh(cfg: Config):
    if cfg.test:
        jax.config.update("jax_num_cpu_devices", 2)
    devs = jax.devices()
    mesh_n = 2 if len(devs) >= 2 else 1
    mesh = jax.make_mesh((mesh_n,), ("batch",))
    jax.set_mesh(mesh)
    return mesh


def _restore_model_only(cfg: Config, model: Transformer, checkpoint_step: int | None) -> int | None:
    manager = create_checkpoint_manager(cfg)
    if not manager.enabled:
        return None
    step = manager.latest_step() if checkpoint_step is None else checkpoint_step
    if step is None:
        return None
    ckpt_path = manager._step_path(step)  # noqa: SLF001
    try:
        payload_template = {"step": int(step), "model": nnx.state(model)}
        payload = manager._checkpointer.restore(  # noqa: SLF001
            str(ckpt_path),
            item=payload_template,
            partial_restore=True,
        )
        model_state = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
        nnx.update(model, model_state)
        return int(step)
    except Exception as exc:
        print(f"warning: failed model-only restore from `{ckpt_path}`: {exc}")
        print("warning: continuing with current model weights")
        return None


def _num_params(model: Transformer) -> int:
    state = nnx.state(model, nnx.Param)
    leaves = jax.tree_util.tree_leaves(state)
    count = 0
    for leaf in leaves:
        arr = np.asarray(leaf.value if hasattr(leaf, "value") else leaf)
        count += int(arr.size)
    return count


def _next_token_logits(model: Transformer, context_tokens: list[int], ctx_len: int) -> jnp.ndarray:
    if not context_tokens:
        raise ValueError("Context tokens must be non-empty.")
    ctx = context_tokens[-ctx_len:]
    mesh_batch = int(model.mesh.shape["batch"])
    x = jnp.asarray([ctx for _ in range(mesh_batch)], dtype=jnp.int32)
    logits = model(x)
    return jnp.asarray(np.asarray(logits)[0, -1])


def _greedy_generate_answer(
    model: Transformer,
    encoder,
    prompt_tokens: list[int],
    *,
    ctx_len: int,
    max_new_tokens: int,
) -> str:
    generated: list[int] = []
    context = list(prompt_tokens)
    for _ in range(max_new_tokens):
        logits = _next_token_logits(model, context, ctx_len=ctx_len)
        next_tok = int(jnp.argmax(logits))
        generated.append(next_tok)
        context.append(next_tok)
        if next_tok in {198, 13, 50256}:  # newline / period / GPT2 EOT
            break
    return encoder.decode(generated).strip()


def _mean_answer_nll_bits(
    model: Transformer,
    prompt_tokens: list[int],
    answer_tokens: list[int],
    *,
    ctx_len: int,
) -> float:
    if not answer_tokens:
        return 0.0
    bits: list[float] = []
    context = list(prompt_tokens)
    for tok in answer_tokens:
        logits = _next_token_logits(model, context, ctx_len=ctx_len)
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        bits.append(float((-log_probs[int(tok)]) / math.log(2.0)))
        context.append(int(tok))
    return float(sum(bits) / len(bits))


def _build_probes(person: dict[str, Any], domain_bits: dict[str, float]) -> list[Probe]:
    birth_date = f"{person['birthmonth']} {person['birthday']}, {person['birthyear']}"
    return [
        Probe(
            name="birth_date",
            question="What is this person's birth date?",
            expected=birth_date,
            max_bits=domain_bits["birth_date_bits"],
        ),
        Probe(
            name="birth_city",
            question="Which city was this person born in?",
            expected=str(person["birthcity"]),
            max_bits=domain_bits["birth_city_bits"],
        ),
        Probe(
            name="university",
            question="Which university did this person attend?",
            expected=str(person["university"]),
            max_bits=domain_bits["university_bits"],
        ),
        Probe(
            name="major",
            question="What major did this person study?",
            expected=str(person["field"]),
            max_bits=domain_bits["major_bits"],
        ),
        Probe(
            name="employer",
            question="Which company did this person work for?",
            expected=str(person["company1name"]),
            max_bits=domain_bits["employer_bits"],
        ),
        Probe(
            name="working_city",
            question="In which city did this person work?",
            expected=str(person["company1city"]),
            max_bits=domain_bits["working_city_bits"],
        ),
    ]


def _partial_credit_bits(probe_name: str, expected: str, predicted: str, max_bits: float) -> float:
    expected_norm = _normalize_text(expected)
    pred_norm = _normalize_text(predicted)
    if expected_norm and pred_norm.startswith(expected_norm):
        return max_bits
    if expected_norm == pred_norm:
        return max_bits
    if probe_name != "birth_date":
        return 0.0

    exp_month, exp_day, exp_year = _month_day_year(expected)
    pred_month, pred_day, pred_year = _month_day_year(predicted)
    credited = 0.0
    if exp_year is not None and pred_year == exp_year:
        credited += math.log2(200)
    if exp_month is not None and exp_day is not None and pred_month == exp_month and pred_day == exp_day:
        credited += math.log2(12 * 28)
    return min(max_bits, credited)


def _domain_bits_from_fields(fields_dir: Path) -> dict[str, float]:
    def count_values(name: str) -> int:
        p = fields_dir / f"{name}.txt"
        vals = [line.strip() for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
        return max(1, len(vals))

    city_count = count_values("city")
    university_count = count_values("university")
    field_count = count_values("field")
    company_count = count_values("company")
    return {
        "birth_date_bits": math.log2(12 * 28 * 200),
        "birth_city_bits": math.log2(city_count),
        "university_bits": math.log2(university_count),
        "major_bits": math.log2(field_count),
        # Paper describes employer count 263 and working city derived from employer HQ.
        "employer_bits": math.log2(company_count),
        "working_city_bits": math.log2(company_count),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Capo capacity probe scaffold (approximate bits/parameter).")
    parser.add_argument("--config", type=str, default="configs/config.toml")
    parser.add_argument("--checkpoint-step", type=int, default=None)
    parser.add_argument("--samples", type=int, default=64, help="Number of Capo records to probe.")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--synth-root", type=str, default="../synth_data")
    parser.add_argument("--output-json", type=str, default="")
    args = parser.parse_args()

    cfg = Config.from_toml(args.config)
    if int(cfg.vocab_size) <= 50256:
        raise ValueError(
            "Capo GPT-2 probing requires `vocab_size >= 50257`. "
            f"Current config has vocab_size={cfg.vocab_size}."
        )
    mesh = _make_mesh(cfg)
    model = Transformer(cfg=cfg, mesh=mesh, rngs=nnx.Rngs(cfg.seed))
    restored_step = _restore_model_only(cfg, model, checkpoint_step=args.checkpoint_step)
    if restored_step is None:
        print("warning: no checkpoint restored; evaluating random-init model.")
    else:
        print(f"restored checkpoint step={restored_step}")

    encoder = tiktoken.get_encoding("gpt2")
    synth_root = Path(args.synth_root).resolve()
    synth_main = _load_synth_main(synth_root)
    capo_file = Path(cfg.synth.capo_capo_file)
    if not capo_file.is_absolute():
        capo_file = (synth_root / capo_file).resolve()
    fields_dir = Path(cfg.synth.capo_fields_dir)
    if not fields_dir.is_absolute():
        fields_dir = (synth_root / fields_dir).resolve()

    records = synth_main.generate_records(
        "capo",
        samples=int(args.samples),
        seed=int(args.seed),
        capo_file=str(capo_file),
        fields_dir=str(fields_dir),
        order=str(cfg.synth.capo_order),
        N=int(getattr(cfg.synth, "capo_N", 50000)),
        exposures_per_person=int(getattr(cfg.synth, "capo_exposures_per_person", 100)),
        offset=int(args.offset),
    )

    domain_bits = _domain_bits_from_fields(fields_dir)
    max_bits_per_record = sum(domain_bits.values())

    total_stored_bits = 0.0
    total_max_bits = 0.0
    per_attr = {
        key: {"exact": 0, "count": 0, "stored_bits": 0.0, "nll_bits": 0.0}
        for key in ["birth_date", "birth_city", "university", "major", "employer", "working_city"]
    }

    for rec in records:
        bio_text = str(rec["text"])
        person = rec["person"]
        probes = _build_probes(person, domain_bits=domain_bits)
        total_max_bits += max_bits_per_record

        for probe in probes:
            prompt = f"{bio_text}\n\nQuestion: {probe.question}\nAnswer:"
            prompt_tokens = encoder.encode_ordinary(prompt)
            expected_tokens = encoder.encode_ordinary(probe.expected)

            pred_text = _greedy_generate_answer(
                model,
                encoder,
                prompt_tokens,
                ctx_len=int(cfg.ctx_len),
                max_new_tokens=int(args.max_new_tokens),
            )
            mean_nll_bits = _mean_answer_nll_bits(
                model,
                prompt_tokens,
                expected_tokens,
                ctx_len=int(cfg.ctx_len),
            )
            credited_bits = _partial_credit_bits(probe.name, probe.expected, pred_text, probe.max_bits)
            exact = credited_bits >= (probe.max_bits - 1e-9)

            total_stored_bits += credited_bits
            per_attr[probe.name]["count"] += 1
            per_attr[probe.name]["exact"] += int(exact)
            per_attr[probe.name]["stored_bits"] += credited_bits
            per_attr[probe.name]["nll_bits"] += mean_nll_bits

    n_params = _num_params(model)
    bits_per_param = total_stored_bits / max(1, n_params)
    normalized_capacity = total_stored_bits / max(1e-9, total_max_bits)

    summary = {
        "config": str(args.config),
        "checkpoint_step": restored_step,
        "num_params": n_params,
        "records": len(records),
        "total_stored_bits": total_stored_bits,
        "total_max_bits": total_max_bits,
        "normalized_capacity": normalized_capacity,
        "bits_per_param": bits_per_param,
        "per_attribute": {},
        "notes": [
            "Scaffold metric: approximate bits score using exact/partial QA recall.",
            "Not equivalent to the full PoLM Capo Pareto frontier protocol.",
        ],
    }
    for name, stats in per_attr.items():
        count = max(1, int(stats["count"]))
        summary["per_attribute"][name] = {
            "exact_match_rate": float(stats["exact"]) / count,
            "avg_stored_bits": float(stats["stored_bits"]) / count,
            "avg_answer_nll_bits": float(stats["nll_bits"]) / count,
        }

    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.output_json:
        out = Path(args.output_json).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"wrote summary to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
