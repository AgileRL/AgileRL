"""Hardcoded Arena on-prem API call descriptors (:class:`ManifestInvoke`).

These mirror the dynamic on-prem manifest the server publishes, but are baked
into the CLI so ``arena on-prem install`` / ``teardown`` work without first
fetching the capabilities document. They are consumed by :class:`OnPremApi`.
"""

from __future__ import annotations

from typing import Any, Literal

from agilerl.arena.client import ManifestInvoke

# Supported deployment bundle flavors. ``dockerSwarm`` installs over SSH on a
# manager + workers; ``helm`` runs a local ``helm upgrade --install``.
SetupKind = Literal["dockerSwarm", "helm"]

ENABLE: ManifestInvoke = {
    "method": "POST",
    "path": "/api/cli/v1/on-prem/enable",
    "responseKind": "json",
    "params": [],
}

LIST_CLASSES: ManifestInvoke = {
    "method": "GET",
    "path": "/api/cli/v1/on-prem/classes/list",
    "responseKind": "json",
    "params": [],
}

CREATE_CLASS: ManifestInvoke = {
    "method": "POST",
    "path": "/api/cli/v1/on-prem/classes/create",
    "responseKind": "json",
    "params": [],
}

BUNDLE: ManifestInvoke = {
    "method": "GET",
    "path": "/api/cli/v1/on-prem/classes/deployment-setup",
    "responseKind": "binary",
    "params": [],
}

DELETE_CLASS: ManifestInvoke = {
    "method": "DELETE",
    "path": "/api/cli/v1/on-prem/classes/delete",
    "responseKind": "json",
    # The endpoint reads ``name`` from the query string, so declare it explicitly
    # rather than relying on the method-based fallback (which routes DELETE
    # payloads to the JSON body).
    "params": [{"name": "name", "in": "query", "type": "string", "required": True}],
}

DISABLE: ManifestInvoke = {
    "method": "POST",
    "path": "/api/cli/v1/on-prem/disable",
    "responseKind": "json",
    "params": [],
}

DEFAULT_METADATA: dict[str, Any] = {
    "computeResource": {
        "numCpus": 8,
        "numGpus": 0,
        "memoryBytes": "64 GiB",
    },
}
