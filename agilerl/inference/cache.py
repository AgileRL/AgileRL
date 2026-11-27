# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Persist deployment bindings and active agent selection in ``~/.arena/inference.json``.

Bindings hold the deployment URL only. Credentials are never written here.
"""

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
ACTIVE_SESSIONS_KEY = "active_sessions"


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


def load_binding(name: str) -> str | None:
    """Return the deployment URL for *name* if cached, else ``None``.

    Entries that still carry an ``api_key`` are stale and rejected.

    :param name: The name of the deployment.
    :type name: str
    :return: The URL of the deployment.
    :rtype: str | None
    """
    data = _load_store()
    raw = data.get(DEPLOYMENTS_KEY)
    if not isinstance(raw, dict):
        return None
    entry = raw.get(normalized_deployment_name(name))
    if not isinstance(entry, dict):
        return None
    if "api_key" in entry:
        return None
    url = entry.get("url")
    if not isinstance(url, str) or not url.strip():
        return None
    return url.strip()


def save_binding(name: str, url: str) -> None:
    """Merge a deployment binding into ``~/.arena/inference.json``.

    Any credential left in the store by an older release is dropped here, so one
    write purges every stale entry rather than only the one being saved.

    :param name: The name of the deployment.
    :type name: str
    :param url: The URL of the deployment.
    :type url: str
    """
    data = _load_store()
    deployments = data.get(DEPLOYMENTS_KEY)
    if not isinstance(deployments, dict):
        deployments = {}
    scrubbed = {
        existing: {k: v for k, v in entry.items() if k != "api_key"}
        for existing, entry in deployments.items()
        if isinstance(entry, dict)
    }
    scrubbed[normalized_deployment_name(name)] = {"url": url.strip()}
    data[DEPLOYMENTS_KEY] = scrubbed
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


def save_active_session(deployment_name: str, session_id: str) -> None:
    """Resume *session_id* on later calls to this deployment.

    Sessions are stored per deployment, since an id from one deployment means
    nothing to another.

    :param deployment_name: The name of the deployment.
    :type deployment_name: str
    :param session_id: The chat session to resume.
    :type session_id: str
    """
    data = _load_store()
    sessions = data.get(ACTIVE_SESSIONS_KEY)
    if not isinstance(sessions, dict):
        sessions = {}
    sessions[normalized_deployment_name(deployment_name)] = session_id.strip()
    data[ACTIVE_SESSIONS_KEY] = sessions
    _write_store(data)


def load_active_session(deployment_name: str) -> str | None:
    """Return the session being resumed on *deployment_name*, else ``None``.

    :param deployment_name: The name of the deployment.
    :type deployment_name: str
    :return: The chat session to resume.
    :rtype: str | None
    """
    data = _load_store()
    sessions = data.get(ACTIVE_SESSIONS_KEY)
    if not isinstance(sessions, dict):
        return None
    session_id = sessions.get(normalized_deployment_name(deployment_name))
    if not isinstance(session_id, str) or not session_id.strip():
        return None
    return session_id.strip()


def clear_active_session(deployment_name: str) -> bool:
    """Stop resuming a session on *deployment_name*.

    :param deployment_name: The name of the deployment.
    :type deployment_name: str
    :return: ``True`` if a session was being resumed, ``False`` if there was
        nothing to clear.
    :rtype: bool
    """
    data = _load_store()
    sessions = data.get(ACTIVE_SESSIONS_KEY)
    if not isinstance(sessions, dict):
        return False
    if sessions.pop(normalized_deployment_name(deployment_name), None) is None:
        return False
    data[ACTIVE_SESSIONS_KEY] = sessions
    _write_store(data)
    return True


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
