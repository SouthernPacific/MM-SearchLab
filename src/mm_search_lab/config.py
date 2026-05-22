"""Configuration helpers for MM-SearchLab."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Union

import yaml


def load_yaml(path: Union[str, Path]) -> Dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"config must be a mapping: {config_path}")
    return data


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]
