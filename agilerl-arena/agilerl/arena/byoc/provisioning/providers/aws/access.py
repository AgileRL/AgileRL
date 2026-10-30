# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Grant the provisioning caller access to an EKS cluster."""

from __future__ import annotations

import json
import subprocess

import click

from agilerl.arena.byoc.provisioning.kubectl import KubectlRun
from agilerl.arena.byoc.provisioning.providers.aws.kubeconfig import (
    resolve_aws_executable,
)

CLUSTER_ADMIN_POLICY_ARN = (
    "arn:aws:eks::aws:cluster-access-policy/AmazonEKSClusterAdminPolicy"
)


def ensure_caller_cluster_admin(
    cluster_name: str,
    region: str,
    run: KubectlRun = subprocess.run,
) -> None:
    """Give the current IAM principal cluster admin on *cluster_name*.

    Setting ``authentication_mode`` without
    ``bootstrap_cluster_creator_admin_permissions`` leaves the creator with
    no access entry, so kubectl is rejected.
    """
    identity = _run(
        ["sts", "get-caller-identity", "--output", "json"],
        run=run,
    )
    principal_arn = _principal_arn(identity.stdout)
    if principal_arn.endswith(":root"):
        msg = (
            "EKS cannot authenticate the account root user. "
            "Provision with an IAM user or role."
        )
        raise click.ClickException(msg)
    created = _run(
        [
            "eks",
            "create-access-entry",
            "--cluster-name",
            cluster_name,
            "--principal-arn",
            principal_arn,
            "--region",
            region,
        ],
        run=run,
        check=False,
    )
    if created.returncode != 0 and "ResourceInUseException" not in _detail(created):
        msg = f"aws eks create-access-entry failed. {_detail(created)}".strip()
        raise click.ClickException(msg)
    _run(
        [
            "eks",
            "associate-access-policy",
            "--cluster-name",
            cluster_name,
            "--principal-arn",
            principal_arn,
            "--policy-arn",
            CLUSTER_ADMIN_POLICY_ARN,
            "--access-scope",
            "type=cluster",
            "--region",
            region,
        ],
        run=run,
    )


def _principal_arn(payload: str) -> str:
    try:
        identity = json.loads(payload)
    except json.JSONDecodeError as exc:
        msg = "aws sts get-caller-identity did not return JSON."
        raise click.ClickException(msg) from exc
    arn = identity.get("Arn") if isinstance(identity, dict) else None
    if not isinstance(arn, str) or not arn:
        msg = "aws sts get-caller-identity did not return an ARN."
        raise click.ClickException(msg)
    return arn


def _detail(result: subprocess.CompletedProcess[str]) -> str:
    return result.stderr.strip() or result.stdout.strip()


def _run(
    args: list[str],
    run: KubectlRun,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    executable = resolve_aws_executable()
    if executable is None:
        msg = "AWS CLI not found. Install it and add aws to PATH."
        raise click.ClickException(msg)
    result = run(
        [executable, *args],
        check=False,
        capture_output=True,
        text=True,
    )
    if check and result.returncode != 0:
        rendered = " ".join([executable, *args])
        detail = _detail(result)
        msg = f"{rendered} failed.{f' {detail}' if detail else ''}"
        raise click.ClickException(msg)
    return result
