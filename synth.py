from __future__ import annotations

from functools import lru_cache
import importlib.util
from pathlib import Path
import sys


def _load_module_from_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module `{name}` from `{path}`.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@lru_cache(maxsize=1)
def _load_synth_batch_modules():
    synth_root = Path(__file__).resolve().parent.parent / "synth_data"
    config_path = synth_root / "config.py"
    main_path = synth_root / "main.py"
    batch_path = synth_root / "batch_iterator.py"
    for p in (config_path, main_path, batch_path):
        if not p.exists():
            raise FileNotFoundError(f"synth_data module file not found: `{p}`.")

    config_mod = _load_module_from_path("synth_data_external_config", config_path)
    main_mod = _load_module_from_path("synth_data_external_main", main_path)

    old_config = sys.modules.get("config")
    old_main = sys.modules.get("main")
    try:
        sys.modules["config"] = config_mod
        sys.modules["main"] = main_mod
        batch_mod = _load_module_from_path("synth_data_external_batch_iterator", batch_path)
    finally:
        if old_config is None:
            sys.modules.pop("config", None)
        else:
            sys.modules["config"] = old_config
        if old_main is None:
            sys.modules.pop("main", None)
        else:
            sys.modules["main"] = old_main

    return config_mod, batch_mod


def get_synth_config_class():
    config_mod, _ = _load_synth_batch_modules()
    return config_mod.SynthConfig


def get_synth_batch_iterator():
    _, batch_mod = _load_synth_batch_modules()
    return batch_mod.batch_iterator
