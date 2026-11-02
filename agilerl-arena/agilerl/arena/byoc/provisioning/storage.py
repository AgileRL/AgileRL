# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Object storage credentials for Arena workloads on a provisioned cluster."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from agilerl.arena.byoc.provisioning.kubectl import (
    KubectlRun,
    apply_manifests,
    ensure_namespace,
    kubeconfig_env,
    require_kubectl,
)

# Host of a regional AWS S3 API endpoint, including virtual-hosted bucket names.
AWS_S3_ENDPOINT_HOST = re.compile(
    r"^(?:.+\.)?s3(?:\.dualstack)?\.(?P<region>[a-z0-9-]+)\.amazonaws\.com(?:\.cn)?$"
)


def aws_region_from_s3_endpoint(endpoint: str) -> str | None:
    """Return the AWS region encoded in an S3 API endpoint host."""
    host = urlparse(endpoint).hostname or ""
    match = AWS_S3_ENDPOINT_HOST.fullmatch(host)
    if match is None:
        return None
    return match["region"]


def storage_secret_endpoint_data(endpoint: str) -> dict[str, str]:
    """Return Secret keys the agent reads for this object-storage endpoint."""
    data = {
        "endpoint": endpoint,
        "AWS_ENDPOINT_URL": endpoint,
        "APP_AWS_CLIENT_ENDPOINT": endpoint,
    }
    region = aws_region_from_s3_endpoint(endpoint)
    if region is not None:
        # SigV4 uses AWS_REGION; unset, the SDK signs as us-east-1.
        data["AWS_REGION"] = region
        data["AWS_DEFAULT_REGION"] = region
    return data


def build_storage_secret_manifest(
    secret_name: str,
    namespace: str,
    access_key_id: str,
    secret_access_key: str,
    endpoint: str,
) -> dict[str, Any]:
    """Return the Secret the agent, Ray, and inference charts read with ``envFrom``."""
    return {
        "apiVersion": "v1",
        "kind": "Secret",
        "metadata": {"name": secret_name, "namespace": namespace},
        "type": "Opaque",
        "stringData": {
            "AWS_ACCESS_KEY_ID": access_key_id,
            "AWS_SECRET_ACCESS_KEY": secret_access_key,
            **storage_secret_endpoint_data(endpoint),
        },
    }


def ensure_storage_secret(
    kubeconfig_path: Path,
    secret_name: str,
    access_key_id: str,
    secret_access_key: str,
    endpoint: str,
    namespace: str = "arena",
    run: KubectlRun = subprocess.run,
) -> None:
    """Create or update the object storage credentials Secret in the cluster."""
    require_kubectl()
    env = kubeconfig_env(kubeconfig_path)
    ensure_namespace(namespace, env=env, run=run)
    apply_manifests(
        [
            build_storage_secret_manifest(
                secret_name=secret_name,
                namespace=namespace,
                access_key_id=access_key_id,
                secret_access_key=secret_access_key,
                endpoint=endpoint,
            )
        ],
        env=env,
        run=run,
    )
