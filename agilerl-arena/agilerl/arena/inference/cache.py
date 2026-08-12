# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Persist deployment bindings and active agent selection in ``~/.arena/inference.json``."""

from __future__ import annotations

import json
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any

INFERENCE_FILE = Path.home() / ".arena" / "inference.json"
DEPLOYMENTS_KEY = "deployments"
ACTIVE_AGENT_KEY = "active_agent"


def normalized_deployment_name(name: str) -> str:
    """CLI/cache key for a deployment name.

    :param name: The name of the deployment.
    :type name: str
    :return: The normalized deployment name.
    :rtype: str
    """
    return name.strip()


def _load_store() -> dict[str, Any]:
    """Load the inference store from ``~/.arena/inference.json``."""
    if not INFERENCE_FILE.is_file():
        return {}
    try:
        data = json.loads(INFERENCE_FILE.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError):
        return {}


def _write_store(data: dict[str, Any]) -> None:
    """Write the inference store to ``~/.arena/inference.json``."""
    INFERENCE_FILE.parent.mkdir(parents=True, exist_ok=True)
    INFERENCE_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")
    os.chmod(INFERENCE_FILE, stat.S_IRUSR | stat.S_IWUSR)


def load_binding(name: str) -> tuple[str, str] | None:
    """Return ``(url, api_key)`` for *name* if cached, else ``None``.

    :param name: The name of the deployment.
    :type name: str
    :return: The URL and API key of the deployment.
    :rtype: tuple[str, str] | None
    """
    # Load data from ~/.arena/inference.json
    data = _load_store()
    raw = data.get(DEPLOYMENTS_KEY)
    if not isinstance(raw, dict):
        return None
    entry = raw.get(normalized_deployment_name(name))
    if not isinstance(entry, dict):
        return None
    url = entry.get("url")
    api_key = entry.get("api_key")
    if not isinstance(url, str) or not isinstance(api_key, str):
        return None
    if not url.strip() or not api_key.strip():
        return None
    return url.strip(), api_key.strip()


def save_binding(name: str, url: str, api_key: str) -> None:
    """Merge a deployment binding into ``~/.arena/inference.json``.

    :param name: The name of the deployment.
    :type name: str
    :param url: The URL of the deployment.
    :type url: str
    :param api_key: The API key of the deployment.
    :type api_key: str
    """
    data = _load_store()
    deployments = data.get(DEPLOYMENTS_KEY)
    if not isinstance(deployments, dict):
        deployments = {}
    key = normalized_deployment_name(name)
    deployments[key] = {"url": url.strip(), "api_key": api_key.strip()}
    data[DEPLOYMENTS_KEY] = deployments
    _write_store(data)


@dataclass(frozen=True)
class ActiveAgentSelection:
    """CLI-selected deployment used by ``arena agent generate`` without a name argument.

    :param deployment_name: The name of the deployment.
    :type deployment_name: str
    :param experiment_name: The name of the experiment.
    :type experiment_name: str | None
    :param project_name: The name of the project.
    :type project_name: str | None
    """

    deployment_name: str
    experiment_name: str | None = None
    project_name: str | None = None


def save_active_agent(
    name: str,
    *,
    experiment_name: str | None = None,
    project_name: str | None = None,
) -> None:
    """Persist the default deployment for ``arena agent generate``.

    :param name: The name of the deployment.
    :type name: str
    :param experiment_name: The name of the experiment.
    :type experiment_name: str | None
    :param project_name: The name of the project.
    :type project_name: str | None
    """
    data = _load_store()
    entry: dict[str, str] = {"deployment": normalized_deployment_name(name)}
    if experiment_name and experiment_name.strip():
        entry["experiment_name"] = experiment_name.strip()
    if project_name and project_name.strip():
        entry["project_name"] = project_name.strip()
    data[ACTIVE_AGENT_KEY] = entry
    _write_store(data)


def load_active_agent() -> ActiveAgentSelection | None:
    """Return the active deployment selection, or ``None`` if unset.

    :return: The active deployment selection.
    :rtype: ActiveAgentSelection | None
    """
    data = _load_store()
    raw = data.get(ACTIVE_AGENT_KEY)
    if not isinstance(raw, dict):
        return None
    deployment = raw.get("deployment")
    if not isinstance(deployment, str) or not deployment.strip():
        return None
    experiment_name = raw.get("experiment_name")
    project_name = raw.get("project_name")
    return ActiveAgentSelection(
        deployment_name=normalized_deployment_name(deployment),
        experiment_name=(
            experiment_name.strip()
            if isinstance(experiment_name, str) and experiment_name.strip()
            else None
        ),
        project_name=(
            project_name.strip()
            if isinstance(project_name, str) and project_name.strip()
            else None
        ),
    )
