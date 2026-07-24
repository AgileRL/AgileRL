# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

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

REGISTER_CLUSTER: ManifestInvoke = {
    "method": "POST",
    "path": "/api/cli/v1/on-prem/clusters/register",
    "responseKind": "json",
    "params": [
        {"name": "name", "in": "body", "type": "string", "required": True},
        {
            "name": "installProfile",
            "in": "body",
            "type": "string",
            "required": False,
        },
        {
            "name": "storageEndpoint",
            "in": "body",
            "type": "string",
            "required": False,
        },
        {
            "name": "storageBucket",
            "in": "body",
            "type": "string",
            "required": False,
        },
        {
            "name": "storagePrefix",
            "in": "body",
            "type": "string",
            "required": False,
        },
        {
            "name": "storageSecretName",
            "in": "body",
            "type": "string",
            "required": False,
        },
        {
            "name": "ingressClassName",
            "in": "body",
            "type": "string",
            "required": False,
        },
        {
            "name": "hostnameTemplate",
            "in": "body",
            "type": "string",
            "required": False,
        },
        {
            "name": "gatewayApiParentRefs",
            "in": "body",
            "type": "json",
            "required": False,
        },
        {
            "name": "tlsSecretName",
            "in": "body",
            "type": "string",
            "required": False,
        },
        {
            "name": "preprocessingResourceClassId",
            "in": "body",
            "type": "int",
            "required": False,
        },
        {
            "name": "rayDataStorageClassName",
            "in": "body",
            "type": "string",
            "required": False,
        },
        {
            "name": "rayDataPvcSize",
            "in": "body",
            "type": "string",
            "required": False,
        },
    ],
}

DEFAULT_METADATA: dict[str, Any] = {
    "computeResource": {
        "numCpus": 8,
        "numGpus": 0,
        "memoryBytes": "64 GiB",
    },
}
