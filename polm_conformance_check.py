from __future__ import annotations

import argparse
import importlib.util
from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
from typing import Callable


@dataclass
class CheckResult:
    name: str
    status: str
    detail: str


def _load_synth_main(synth_root: Path):
    main_path = (synth_root / "main.py").resolve()
    if not main_path.exists():
        raise FileNotFoundError(f"synth_data main.py not found at `{main_path}`.")
    spec = importlib.util.spec_from_file_location("synth_data_audit_main", main_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load synth_data module from `{main_path}`.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run_check(name: str, fn: Callable[[], str]) -> CheckResult:
    try:
        detail = fn()
        return CheckResult(name=name, status="PASS", detail=detail)
    except Exception as exc:
        return CheckResult(name=name, status="FAIL", detail=f"{type(exc).__name__}: {exc}")


def _lano_parse_positions(
    tokens: list[int],
    all_layers: list[list[dict]],
    depth: int,
    node_idx: int,
    start: int,
    memo: dict[tuple[int, int, int], set[int]],
) -> set[int]:
    key = (depth, node_idx, start)
    if key in memo:
        return memo[key]

    node = all_layers[depth][node_idx]
    children = node.get("children")
    if children is None:
        if start < len(tokens) and int(tokens[start]) == int(node["id"]):
            memo[key] = {start + 1}
        else:
            memo[key] = set()
        return memo[key]

    out: set[int] = set()
    for rule in children:
        ends = {start}
        for child_idx in rule:
            nxt: set[int] = set()
            for pos in ends:
                nxt |= _lano_parse_positions(tokens, all_layers, depth + 1, int(child_idx), pos, memo)
            ends = nxt
            if not ends:
                break
        out |= ends
    memo[key] = out
    return out


def _lano_is_valid(tokens: list[int], cfg_obj: dict) -> bool:
    all_layers = cfg_obj["all"]
    memo: dict[tuple[int, int, int], set[int]] = {}
    ends = _lano_parse_positions(tokens, all_layers, depth=0, node_idx=0, start=0, memo=memo)
    return len(tokens) in ends


def main() -> int:
    parser = argparse.ArgumentParser(description="PoLM synthetic dataset conformance checks")
    parser.add_argument(
        "--synth-root",
        type=Path,
        default=Path("../synth_data"),
        help="Path to synth_data repository",
    )
    args = parser.parse_args()

    synth_root = args.synth_root.resolve()
    repo_root = Path(__file__).resolve().parent
    synth_main = _load_synth_main(synth_root)
    cfg3f = (synth_root / "data-synthetic-pretrain/Lano-cfg/configs/cfg3f.json").resolve()
    capo_file = (synth_root / "data-synthetic-pretrain/Capo-bioS-bioR/Capo-bioS-bioR.py").resolve()
    capo_fields = (synth_root / "data-synthetic-pretrain/Capo-bioS-bioR/fields").resolve()

    results: list[CheckResult] = []

    def check_depo_train() -> str:
        r = synth_main.generate_records(
            "depo",
            samples=1,
            seed=11,
            N=100,
            K=16,
            M=10,
            qa=False,
            qa_mode="appendix",
        )[0]
        assert "tokens" in r and "label" in r
        assert len(r["tokens"]) == len(r["label"])
        assert sum(int(x) for x in r["label"]) > 0
        assert any(int(t) == 9500 for t in r["tokens"])
        return "Depo training format and answer-token masking are valid."

    def check_depo_eval_mode() -> str:
        k_values: set[int] = set()
        for i in range(16):
            r = synth_main.generate_records(
                "depo",
                samples=1,
                seed=100 + i,
                N=100,
                K=16,
                M=10,
                qa=True,
                qa_mode="appendix",
            )[0]
            for token in r["tokens"]:
                if 9001 <= int(token) <= 9200:
                    k_values.add(int(token) - 9000)
        assert k_values.issubset({8, 16}), f"Observed k values: {sorted(k_values)}"
        return f"Depo QA appendix mode uses expected k values: {sorted(k_values)}"

    def check_brevo() -> str:
        r = synth_main.generate_records("brevo", samples=1, seed=21, N=90, multi=False)[0]
        assert r["tokens"][0] == 0
        assert r["tokens"][-1] == 3
        assert 1 in r["tokens"] and 2 in r["tokens"]
        assert len(r["tokens"]) == len(r["label"])
        assert sum(int(x) for x in r["label"]) > 0
        return "Brevo token framing and labels are valid."

    def check_mano() -> str:
        r = synth_main.generate_records(
            "mano",
            samples=1,
            seed=31,
            L=10,
            ttype="asm",
            value_mod=23,
            knowledge_augment=True,
        )[0]
        assert "tokens" in r and "answer" in r
        assert int(r["tokens"][-1]) == int(r["answer"])
        return "Mano sequence/answer generation is valid."

    def check_lano() -> str:
        cfg_obj = json.loads(cfg3f.read_text(encoding="utf-8"))
        r = synth_main.generate_records(
            "lano",
            samples=1,
            seed=41,
            config=str(cfg3f),
            bos_token=None,
            eos_token=None,
        )[0]
        assert "tokens" in r and len(r["tokens"]) > 0
        assert all(isinstance(t, int) and t > 0 for t in r["tokens"])
        tokens = [int(t) for t in r["tokens"]]
        assert _lano_is_valid(tokens, cfg_obj), "Generated Lano sample failed CFG membership check."
        bad = tokens.copy()
        bad[0] = 0
        assert not _lano_is_valid(bad, cfg_obj), "Corrupted Lano sample unexpectedly passed CFG check."
        return "Lano cfg3f generation passes DP-style CFG validity checks."

    def check_capo_tokenized() -> str:
        r = synth_main.generate_records(
            "capo",
            samples=1,
            seed=51,
            capo_file=str(capo_file),
            fields_dir=str(capo_fields),
            order="random",
            N=100,
            exposures_per_person=100,
        )[0]
        assert "text" in r and isinstance(r["text"], str)
        assert "tokens" not in r
        person = r["person"]
        text = r["text"]
        assert person["birthmonth"] in text and str(person["birthyear"]) in text
        assert person["birthcity"] in text
        assert person["university"] in text
        assert person["field"] in text
        assert person["company1name"] in text
        assert person["company1city"] in text
        return "Capo emits raw bio text with six attribute mentions (tokenization deferred to nanoBabble)."

    def check_capo_protocol_indexing() -> str:
        records = synth_main.generate_records(
            "capo",
            samples=205,
            seed=55,
            capo_file=str(capo_file),
            fields_dir=str(capo_fields),
            N=100,
            exposures_per_person=100,
            offset=0,
        )
        assert int(records[0]["N_person_id"]) == 0 and int(records[0]["exposure_index"]) == 0
        assert int(records[99]["N_person_id"]) == 99 and int(records[99]["exposure_index"]) == 0
        assert int(records[100]["N_person_id"]) == 0 and int(records[100]["exposure_index"]) == 1
        assert int(records[204]["N_person_id"]) == 4 and int(records[204]["exposure_index"]) == 2
        # Working city should be derived from employer HQ (same employer -> same city).
        company_to_city: dict[str, str] = {}
        for rec in records:
            person = rec["person"]
            company = str(person["company1name"])
            city = str(person["company1city"])
            prev = company_to_city.get(company)
            if prev is None:
                company_to_city[company] = city
            else:
                assert prev == city, f"Company `{company}` mapped to multiple cities: `{prev}` vs `{city}`"
        return "Capo supports bioS-style person/exposure indexing and deterministic employer->city mapping."

    def check_capo_frontier_runner() -> str:
        out_dir = repo_root / "sessions" / "capo_pareto_conformance"
        cmd = [
            "uv",
            "run",
            str(repo_root / "capo_pareto_runner.py"),
            "--base-config",
            str(repo_root / "configs/capo_gpt2_template.toml"),
            "--model-spec",
            "conformance,64,2,4,256",
            "--capo-n-values",
            "50000",
            "--train-steps",
            "1",
            "--eval-samples",
            "1",
            "--output-dir",
            str(out_dir),
            "--dry-run",
            "--force-rerun",
        ]
        subprocess.run(cmd, cwd=str(repo_root), check=True)
        reports = sorted(out_dir.glob("*/pareto_report.json"))
        assert reports, "No pareto_report.json produced by dry-run runner."
        latest = reports[-1]
        report = json.loads(latest.read_text(encoding="utf-8"))
        runs = report.get("runs", [])
        assert len(runs) == 1, f"Expected 1 run in dry-run report, got {len(runs)}."
        assert runs[0].get("status") == "dry_run", f"Unexpected run status: {runs[0].get('status')}"
        return "Capo frontier runner dry-run generated expected report artifacts."

    results.append(_run_check("Depo Train Format", check_depo_train))
    results.append(_run_check("Depo QA Appendix Mode", check_depo_eval_mode))
    results.append(_run_check("Brevo Format", check_brevo))
    results.append(_run_check("Mano Format", check_mano))
    results.append(_run_check("Lano Format", check_lano))
    results.append(_run_check("Capo Raw Output", check_capo_tokenized))
    results.append(_run_check("Capo Protocol Indexing", check_capo_protocol_indexing))
    results.append(_run_check("Capo Frontier Runner", check_capo_frontier_runner))

    warnings: list[str] = []

    print("PoLM Synthetic Conformance Report")
    print(f"synth_root={synth_root}")
    print("")
    for r in results:
        print(f"[{r.status}] {r.name}: {r.detail}")
    print("")
    for w in warnings:
        print(f"[WARN] {w}")

    fail_count = sum(1 for r in results if r.status == "FAIL")
    print("")
    print(f"Summary: pass={len(results) - fail_count} fail={fail_count} warn={len(warnings)}")
    return 1 if fail_count > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
