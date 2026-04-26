from __future__ import annotations

from dataclasses import asdict
import hashlib
from functools import lru_cache
import importlib.util
from pathlib import Path
import sys

GPT2_EOS_TOKEN_ID = 50256


def _load_module_from_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module `{name}` from `{path}`.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@lru_cache(maxsize=1)
def _load_synth_modules():
    synth_root = Path(__file__).resolve().parent.parent / "synth_data"
    config_path = synth_root / "config.py"
    main_path = synth_root / "main.py"
    for p in (config_path, main_path):
        if not p.exists():
            raise FileNotFoundError(f"synth_data module file not found: `{p}`.")

    config_mod = _load_module_from_path("synth_data_external_config", config_path)
    main_mod = _load_module_from_path("synth_data_external_main", main_path)
    return config_mod, main_mod


@lru_cache(maxsize=1)
def _get_gpt2_encoder():
    try:
        import tiktoken  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "Capo dataset ingestion requires `tiktoken` in nanoBabble. "
            "Install with: `uv add tiktoken`."
        ) from exc
    return tiktoken.get_encoding("gpt2")


def _stable_seed(*parts: object) -> int:
    h = hashlib.blake2b(digest_size=8)
    for part in parts:
        h.update(str(part).encode("utf-8"))
        h.update(b"|")
    return int.from_bytes(h.digest(), "big")


def _resolve_synth_path(raw_path: str, *, synth_root: Path) -> str:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return str(path)

    cwd_candidate = (Path.cwd() / path).resolve()
    if cwd_candidate.exists():
        return str(cwd_candidate)

    synth_candidate = (synth_root / path).resolve()
    return str(synth_candidate)


def _normalize_synth_cfg_paths(synth_cfg):
    dataset = str(getattr(synth_cfg, "dataset", "")).lower()
    if dataset not in {"lano", "capo"}:
        return synth_cfg

    synth_root = Path(__file__).resolve().parent.parent / "synth_data"
    patch: dict[str, str] = {}
    if dataset == "lano":
        lano_config = getattr(synth_cfg, "lano_config", None)
        if isinstance(lano_config, str) and lano_config:
            patch["lano_config"] = _resolve_synth_path(lano_config, synth_root=synth_root)
    elif dataset == "capo":
        capo_file = getattr(synth_cfg, "capo_capo_file", None)
        capo_fields_dir = getattr(synth_cfg, "capo_fields_dir", None)
        if isinstance(capo_file, str) and capo_file:
            patch["capo_capo_file"] = _resolve_synth_path(capo_file, synth_root=synth_root)
        if isinstance(capo_fields_dir, str) and capo_fields_dir:
            patch["capo_fields_dir"] = _resolve_synth_path(capo_fields_dir, synth_root=synth_root)

    if not patch:
        return synth_cfg

    cfg_dict = asdict(synth_cfg)
    cfg_dict.update(patch)
    return type(synth_cfg)(**cfg_dict)


def get_synth_config_class():
    config_mod, _ = _load_synth_modules()
    return config_mod.SynthConfig


def _extract_tokens_and_labels(dataset: str, record: dict) -> tuple[list[int], list[int]]:
    tokens = record.get("tokens")
    if isinstance(tokens, list) and tokens:
        token_ids = [int(t) for t in tokens]
        labels = record.get("label")
        if isinstance(labels, list) and len(labels) == len(token_ids):
            return token_ids, [int(x) for x in labels]
        return token_ids, [1] * len(token_ids)

    if dataset == "capo":
        text = record.get("text")
        if not isinstance(text, str) or not text:
            raise ValueError("Capo record must contain non-empty `text`.")
        encoder = _get_gpt2_encoder()
        token_ids = [int(t) for t in encoder.encode_ordinary(text)]
        token_ids.append(GPT2_EOS_TOKEN_ID)
        return token_ids, [1] * len(token_ids)

    raise ValueError(f"Synthetic record for dataset `{dataset}` lacks usable tokens.")


def get_synth_batch_iterator():
    _, main_mod = _load_synth_modules()
    generate_records = main_mod.generate_records

    def _batch_iterator(
        global_seed: int,
        step: int,
        batch_size: int,
        ctx_len: int,
        synth_cfg,
    ):
        normalized_cfg = _normalize_synth_cfg_paths(synth_cfg)
        dataset = str(getattr(normalized_cfg, "dataset", "")).lower()
        kwargs = normalized_cfg.synth_kwargs_for_dataset()

        if batch_size <= 0:
            raise ValueError("`batch_size` must be > 0.")
        if ctx_len <= 0:
            raise ValueError("`ctx_len` must be > 0.")

        target_len = int(ctx_len) + 1
        x_bl: list[list[int]] = []
        l_bl: list[list[int]] = []
        y_bl: list[list[int]] = []

        for row_idx in range(int(batch_size)):
            row_tokens: list[int] = []
            row_labels: list[int] = []
            chunk_idx = 0
            while len(row_tokens) < target_len:
                record_seed = _stable_seed(
                    "nanobabble-synth-row",
                    int(global_seed),
                    dataset,
                    int(step),
                    int(row_idx),
                    int(chunk_idx),
                )
                records = generate_records(dataset, samples=1, seed=record_seed, **kwargs)
                if not records:
                    raise ValueError(f"No records generated for dataset `{dataset}`.")
                rec_tokens, rec_labels = _extract_tokens_and_labels(dataset, records[0])
                row_tokens.extend(rec_tokens)
                row_labels.extend(rec_labels)
                chunk_idx += 1

            row_tokens = row_tokens[:target_len]
            row_labels = row_labels[:target_len]
            x_bl.append(row_tokens[:-1])
            y_bl.append(row_tokens[1:])
            l_bl.append(row_labels[1:])

        return x_bl, l_bl, y_bl

    return _batch_iterator
