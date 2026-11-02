# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Dedicated IAM user that runs AWS cluster provisioning."""

from __future__ import annotations

import json
import os
import subprocess
import time
from collections.abc import Callable

import click

from agilerl.arena.byoc.provisioning.providers.aws.terraform import (
    AwsRun,
    run_aws,
    run_aws_json,
    run_aws_ok,
)

PROVISIONER_USER = "byoc-provisioner"
PROVISIONER_POLICY = "byoc-provision"
PROVISIONER_PROFILE = "byoc"
STATIC_CREDENTIAL_VARIABLES = (
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
)
PROVISIONER_IAM_ACTIONS = (
    "iam:AttachRolePolicy",
    "iam:CreateAccessKey",
    "iam:CreatePolicy",
    "iam:CreatePolicyVersion",
    "iam:CreateRole",
    "iam:CreateServiceLinkedRole",
    "iam:CreateUser",
    "iam:DeleteAccessKey",
    "iam:DeletePolicy",
    "iam:DeletePolicyVersion",
    "iam:DeleteRole",
    "iam:DeleteRolePolicy",
    "iam:DeleteUser",
    "iam:DeleteUserPolicy",
    "iam:DetachRolePolicy",
    "iam:GetPolicy",
    "iam:GetPolicyVersion",
    "iam:GetRole",
    "iam:GetRolePolicy",
    "iam:GetServiceLinkedRoleDeletionStatus",
    "iam:GetUser",
    "iam:GetUserPolicy",
    "iam:ListAccessKeys",
    "iam:ListAttachedRolePolicies",
    "iam:ListPolicies",
    "iam:ListPolicyVersions",
    "iam:ListRolePolicies",
    "iam:ListRoleTags",
    "iam:ListRoles",
    "iam:ListUserPolicies",
    "iam:ListUserTags",
    "iam:PassRole",
    "iam:PutRolePolicy",
    "iam:PutUserPolicy",
    "iam:TagPolicy",
    "iam:TagRole",
    "iam:TagUser",
    "iam:UntagRole",
    "iam:UntagUser",
    "iam:UpdateAssumeRolePolicy",
    "iam:UpdateRole",
)


def provisioner_policy(user_arn: str) -> dict[str, object]:
    """Return the inline policy for the provisioner user.

    It creates cluster roles, the storage user, and the state-bucket user, but
    cannot change its own policies or keys.

    :param user_arn: ARN of the provisioner user.
    :return: IAM policy document.
    """
    return {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Sid": "ClusterResources",
                "Effect": "Allow",
                "Action": [
                    "autoscaling:*",
                    "ec2:*",
                    "eks:*",
                    "elasticfilesystem:*",
                    "elasticloadbalancing:*",
                    "s3:*",
                ],
                "Resource": "*",
            },
            {
                "Sid": "IamForProvisionedResources",
                "Effect": "Allow",
                "Action": list(PROVISIONER_IAM_ACTIONS),
                "Resource": "*",
            },
            {
                "Sid": "CallerIdentity",
                "Effect": "Allow",
                "Action": "sts:GetCallerIdentity",
                "Resource": "*",
            },
            {
                "Sid": "DenyChangingThisUser",
                "Effect": "Deny",
                "Action": [
                    "iam:AddUserToGroup",
                    "iam:AttachUserPolicy",
                    "iam:CreateAccessKey",
                    "iam:DeleteUser",
                    "iam:DeleteUserPolicy",
                    "iam:DetachUserPolicy",
                    "iam:PutUserPolicy",
                ],
                "Resource": user_arn,
            },
        ],
    }


def use_provisioner_profile(
    region: str, run: AwsRun = subprocess.run
) -> Callable[[], None]:
    """Run the rest of this process as the provisioner user.

    The current AWS identity creates or adopts the user, its policy, and an
    access key in the ``byoc`` profile. Every later AWS CLI and Terraform call
    in this process then uses that profile. The returned function restores the
    identity from before this switch. The byoc user cannot delete IAM roles.

    :param region: Region written to the profile when a new key is created.
    :param run: Subprocess runner.
    :return: Restores the calling AWS profile and static keys.
    """
    profile = os.environ.get("AWS_PROFILE")
    static = {name: os.environ.get(name) for name in STATIC_CREDENTIAL_VARIABLES}

    def restore_caller_profile() -> None:
        if profile in (None, PROVISIONER_PROFILE):
            os.environ.pop("AWS_PROFILE", None)
        else:
            os.environ["AWS_PROFILE"] = profile
        for name, value in static.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value

    if os.environ.get("AWS_PROFILE") == PROVISIONER_PROFILE:
        return restore_caller_profile
    identity = run_aws_json(["sts", "get-caller-identity", "--output", "json"], run)
    if str(identity.get("Arn", "")).endswith(f":user/{PROVISIONER_USER}"):
        return restore_caller_profile
    user_arn = _ensure_user(run)
    run_aws_ok(
        [
            "iam",
            "put-user-policy",
            "--user-name",
            PROVISIONER_USER,
            "--policy-name",
            PROVISIONER_POLICY,
            "--policy-document",
            json.dumps(provisioner_policy(user_arn)),
        ],
        run=run,
    )
    _ensure_profile_key(region, run)
    # Static keys in the environment take precedence over AWS_PROFILE.
    for name in STATIC_CREDENTIAL_VARIABLES:
        os.environ.pop(name, None)
    # Subprocesses and generated EKS kubeconfigs read the profile from here.
    os.environ["AWS_PROFILE"] = PROVISIONER_PROFILE
    click.echo(
        f"Provisioning as IAM user {PROVISIONER_USER!r} "
        f"(AWS profile {PROVISIONER_PROFILE!r})."
    )
    return restore_caller_profile


def _ensure_user(run: AwsRun) -> str:
    """Return the provisioner user ARN, creating the user when it is missing."""
    found = run_aws(["iam", "get-user", "--user-name", PROVISIONER_USER], run=run)
    if found.returncode != 0:
        detail = (found.stderr or found.stdout or "").strip()
        if "NoSuchEntity" not in detail:
            msg = f"Could not look up IAM user {PROVISIONER_USER!r}. {detail}".strip()
            raise click.ClickException(msg)
        payload = run_aws_json(
            [
                "iam",
                "create-user",
                "--user-name",
                PROVISIONER_USER,
                "--tags",
                "Key=Purpose,Value=byoc-provisioning",
            ],
            run=run,
        )
    else:
        payload = json.loads(found.stdout)
    user = payload.get("User")
    arn = user.get("Arn") if isinstance(user, dict) else None
    if not isinstance(arn, str) or not arn:
        msg = f"IAM did not return an ARN for user {PROVISIONER_USER!r}."
        raise click.ClickException(msg)
    return arn


def _ensure_profile_key(region: str, run: AwsRun) -> None:
    """Keep the profile key when it belongs to the user, otherwise create one."""
    saved = run_aws(
        ["configure", "get", "aws_access_key_id", "--profile", PROVISIONER_PROFILE],
        run=run,
    )
    saved_key_id = saved.stdout.strip() if saved.returncode == 0 else ""
    listed = run_aws_json(
        ["iam", "list-access-keys", "--user-name", PROVISIONER_USER], run=run
    )
    metadata = listed.get("AccessKeyMetadata", [])
    if not isinstance(metadata, list):
        msg = f"IAM user {PROVISIONER_USER!r} returned an unexpected access key list."
        raise click.ClickException(msg)
    active = {
        item.get("AccessKeyId")
        for item in metadata
        if isinstance(item, dict) and item.get("Status") == "Active"
    }
    if saved_key_id and saved_key_id in active:
        return
    if len(metadata) >= 2:
        msg = (
            f"IAM user {PROVISIONER_USER!r} already has two access keys and "
            f"neither is in AWS profile {PROVISIONER_PROFILE!r}. Delete one."
        )
        raise click.ClickException(msg)
    created = run_aws_json(
        ["iam", "create-access-key", "--user-name", PROVISIONER_USER], run=run
    )
    access_key = created.get("AccessKey")
    if not isinstance(access_key, dict):
        msg = f"IAM did not return an access key for user {PROVISIONER_USER!r}."
        raise click.ClickException(msg)
    for name, value in (
        ("aws_access_key_id", access_key.get("AccessKeyId")),
        ("aws_secret_access_key", access_key.get("SecretAccessKey")),
        ("region", region),
    ):
        run_aws_ok(
            ["configure", "set", name, str(value), "--profile", PROVISIONER_PROFILE],
            run=run,
            failure=f"Could not write AWS profile {PROVISIONER_PROFILE!r}.",
        )
    _wait_for_profile(run)


def _wait_for_profile(run: AwsRun) -> None:
    # IAM rejects a new access key for several seconds after it is created.
    for _ in range(30):
        result = run_aws(
            ["sts", "get-caller-identity", "--profile", PROVISIONER_PROFILE], run=run
        )
        if result.returncode == 0:
            return
        time.sleep(2)
    detail = (result.stderr or result.stdout or "").strip()
    msg = f"New provisioner access key was not accepted by AWS. {detail}".strip()
    raise click.ClickException(msg)
