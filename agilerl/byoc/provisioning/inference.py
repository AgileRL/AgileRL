# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Inference hostname helpers for BYOC cluster provisioning."""

from __future__ import annotations

DEFAULT_INFERENCE_HOSTNAME_TEMPLATE = "inference-{deploymentId}"
DEFAULT_INFERENCE_TLS_SECRET_NAME = "arena-inference-tls"


def normalize_inference_domain(domain: str) -> str:
    """Return a bare DNS domain suitable for Gateway listeners and agent values."""
    value = domain.strip().lower().rstrip(".")
    if not value or "://" in value or "/" in value or " " in value:
        msg = (
            "inference domain must be a bare DNS name "
            "(for example inference.example.com)."
        )
        raise ValueError(msg)
    return value


def resolve_inference_hostname_template(template: str | None) -> str:
    """Return the host-part template used with a separate inference domain."""
    value = (template or DEFAULT_INFERENCE_HOSTNAME_TEMPLATE).strip()
    if not value:
        msg = "inference hostname template must be a non-empty host pattern."
        raise ValueError(msg)
    return value


def build_inference_wildcard_hostname(domain: str) -> str:
    """Return the Gateway listener hostname for inference routes."""
    return f"*.{normalize_inference_domain(domain)}"
