from __future__ import annotations

import importlib
import os
from pathlib import Path
from typing import Any


def default_ms_asset_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "data" / "maniskill_assets"


def resolve_ms_asset_dir(explicit: str | None) -> Path:
    if explicit:
        return Path(explicit).expanduser().resolve()
    env_value = os.environ.get("MS_ASSET_DIR")
    if env_value:
        return Path(env_value).expanduser().resolve()
    return default_ms_asset_dir()


def prepare_runtime_env(explicit: str | None) -> Path:
    ms_asset_dir = resolve_ms_asset_dir(explicit)
    os.environ["MS_ASSET_DIR"] = str(ms_asset_dir)
    return ms_asset_dir


def load_env_factory(explicit: str | None) -> tuple[Path, Any, Any]:
    ms_asset_dir = prepare_runtime_env(explicit)
    importlib.import_module("mshab.envs")
    make_module = importlib.import_module("mshab.envs.make")
    return ms_asset_dir, make_module.EnvConfig, make_module.make_env
