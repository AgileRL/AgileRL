"""Load YAML from ``debugging/configs/`` relative to the repo root."""

from __future__ import annotations

from pathlib import Path

import yaml

_REPO_ROOT = Path(__file__).resolve().parent.parent


def load_debug_config(filename: str) -> dict:
    """Load a YAML config from ``debugging/configs/`` relative to the repo root.

    :param filename: The name of the config file to load.
    :type filename: str
    :return: The config.
    :rtype: dict
    """
    path = _REPO_ROOT / "debugging" / "configs" / filename
    with Path.open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)
