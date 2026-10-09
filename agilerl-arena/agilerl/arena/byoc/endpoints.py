# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Hardcoded Arena BYOC API call descriptors (:class:`ManifestInvoke`).

These mirror the dynamic BYOC manifest the server publishes, but are baked
into the CLI so :class:`ByocApi` can call them without first fetching the
capabilities document.
"""

from __future__ import annotations

from typing import Any, Literal

from agilerl.arena.client import ManifestInvoke

# Deployment bundle flavor. ``helm`` runs a local ``helm upgrade --install``.
SetupKind = Literal["helm"]

ENABLE: ManifestInvoke = {
    "method": "POST",
    "path": "/api/cli/v1/byoc/enable",
    "responseKind": "json",
    "params": [],
}

LIST_CLASSES: ManifestInvoke = {
    "method": "GET",
    "path": "/api/cli/v1/byoc/classes/list",
    "responseKind": "json",
    "params": [],
}

CREATE_CLASS: ManifestInvoke = {
    "method": "POST",
    "path": "/api/cli/v1/byoc/classes/create",
    "responseKind": "json",
    "params": [],
}

BUNDLE: ManifestInvoke = {
    "method": "GET",
    "path": "/api/cli/v1/byoc/classes/deployment-setup",
    "responseKind": "binary",
    "params": [],
}

DELETE_CLASS: ManifestInvoke = {
    "method": "DELETE",
    "path": "/api/cli/v1/byoc/classes/delete",
    "responseKind": "json",
    # The endpoint reads ``name`` from the query string, so declare it explicitly
    # rather than relying on the method-based fallback (which routes DELETE
    # payloads to the JSON body).
    "params": [{"name": "name", "in": "query", "type": "string", "required": True}],
}

DISABLE: ManifestInvoke = {
    "method": "POST",
    "path": "/api/cli/v1/byoc/disable",
    "responseKind": "json",
    "params": [],
}

REGISTER_CLUSTER: ManifestInvoke = {
    "method": "POST",
    "path": "/api/cli/v1/byoc/clusters/register",
    "responseKind": "json",
    "params": [
        {"name": "name", "in": "body", "type": "string", "required": True},
        {
            "name": "installStorage",
            "in": "body",
            "type": "bool",
            "required": False,
        },
        {
            "name": "narrowAllowedIps",
            "in": "body",
            "type": "bool",
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
            "name": "domain",
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
            "name": "preprocessingResourceClass",
            "in": "body",
            "type": "string",
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
        {
            "name": "byocProvider",
            "in": "body",
            "type": "json",
            "required": False,
        },
    ],
}

LIST_CLUSTERS: ManifestInvoke = {
    "method": "GET",
    "path": "/api/cli/v1/byoc/clusters/list",
    "responseKind": "json",
    "params": [],
}

UNREGISTER_CLUSTER: ManifestInvoke = {
    "method": "DELETE",
    "path": "/api/cli/v1/byoc/clusters/unregister",
    "responseKind": "json",
    "params": [
        {"name": "name", "in": "body", "type": "string", "required": True},
    ],
}

ROTATE_CLUSTER_TOKEN: ManifestInvoke = {
    "method": "POST",
    "path": "/api/cli/v1/byoc/clusters/rotate-token",
    "responseKind": "json",
    "params": [
        {"name": "name", "in": "body", "type": "string", "required": True},
    ],
}

INSTALL_CLUSTER_PACKAGE: ManifestInvoke = {
    "method": "POST",
    "path": "/api/cli/v1/byoc/clusters/install-package",
    "responseKind": "binary",
    "params": [
        {"name": "cluster", "in": "body", "type": "string", "required": True},
        {
            "name": "agentHelmValuesYaml",
            "in": "body",
            "type": "string",
            "required": True,
        },
        {
            "name": "storageHelmValuesYaml",
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
