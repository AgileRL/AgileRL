"""Load YAML from ``configs/debugging/`` relative to the repo root."""

from __future__ import annotations

from pathlib import Path

import yaml

_REPO_ROOT = Path(__file__).resolve().parent.parent


def load_debug_config(filename: str) -> dict:
    path = _REPO_ROOT / "configs" / "debugging" / filename
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)
