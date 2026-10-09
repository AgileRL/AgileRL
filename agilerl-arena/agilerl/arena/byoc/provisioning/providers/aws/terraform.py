# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""AWS CLI helpers for the Terraform state bucket."""

from __future__ import annotations

import json
import os
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import click

from agilerl.arena.byoc.provisioning.providers.aws.kubeconfig import (
    resolve_aws_executable,
)

if TYPE_CHECKING:
    from agilerl.arena.byoc.provisioning.spec import ClusterSpec

TFSTATE_CREDENTIALS_FILE = "tfstate-credentials.json"
TFSTATE_BACKEND_CREDENTIALS_FILE = "tfstate-backend-credentials"
TFSTATE_BACKEND_PROFILE = "arena-tfstate"
AwsRun = Callable[..., subprocess.CompletedProcess[str]]
ConfirmFn = Callable[[str], bool]


@dataclass(frozen=True)
class TerraformStateCredentials:
    """Keys for the IAM user that can read and write this cluster's state."""

    access_key_id: str
    secret_access_key: str

    def environ(self) -> dict[str, str]:
        """Return AWS CLI environment updates for this key."""
        return {
            "AWS_ACCESS_KEY_ID": self.access_key_id,
            "AWS_SECRET_ACCESS_KEY": self.secret_access_key,
        }


def credentials_path(state_dir: Path) -> Path:
    """Return the local file that caches Terraform state-bucket keys."""
    return state_dir.expanduser().resolve().parent / TFSTATE_CREDENTIALS_FILE


def _run_aws(
    args: list[str],
    run: AwsRun,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    executable = resolve_aws_executable()
    if executable is None:
        msg = "AWS CLI not found. Install it and add aws to PATH."
        raise click.ClickException(msg)
    subprocess_env = {**os.environ, **env} if env else None
    return run(
        [executable, *args],
        check=False,
        capture_output=True,
        text=True,
        env=subprocess_env,
    )


def _bucket_missing(stdout: str, stderr: str) -> bool:
    combined = f"{stdout} {stderr}".lower()
    return any(
        marker in combined
        for marker in ("404", "not found", "nosuchbucket", "notfound")
    )


def terraform_state_bucket_exists(
    spec: ClusterSpec,
    run: AwsRun = subprocess.run,
) -> bool:
    """Return whether the spec's Terraform state bucket exists in AWS."""
    result = _run_aws(
        [
            "s3api",
            "head-bucket",
            "--bucket",
            spec.terraform_state.bucket,
            "--region",
            spec.aws_cloud().region,
        ],
        run=run,
    )
    if result.returncode == 0:
        return True
    if _bucket_missing(result.stdout or "", result.stderr or ""):
        return False
    detail = (result.stderr or result.stdout or "").strip()
    msg = (
        "Could not look up Terraform state bucket "
        f"{spec.terraform_state.bucket!r}."
        f"{f' {detail}' if detail else ''}"
    )
    raise click.ClickException(msg)


def ensure_terraform_state_bucket(
    spec: ClusterSpec,
    state_dir: Path,
    create: bool = False,
    prompt: ConfirmFn | None = None,
    run: AwsRun = subprocess.run,
) -> TerraformStateCredentials:
    """Require the state bucket and the IAM user Terraform uses to reach it."""
    if not terraform_state_bucket_exists(spec, run=run):
        _create_terraform_state_bucket(spec, create=create, prompt=prompt, run=run)
    return ensure_terraform_state_credentials(spec, state_dir, run=run)


def ensure_terraform_state_credentials(
    spec: ClusterSpec,
    state_dir: Path,
    run: AwsRun = subprocess.run,
) -> TerraformStateCredentials:
    """Return keys for the state IAM user, creating the user and a key if needed."""
    credentials = read_terraform_state_credentials(state_dir)
    if credentials is None:
        credentials = provision_tfstate_user(spec, run=run)
        write_terraform_state_credentials(state_dir, credentials)
    return credentials


def read_terraform_state_credentials(
    state_dir: Path,
) -> TerraformStateCredentials | None:
    """Return state-bucket keys from the local credentials file."""
    path = credentials_path(state_dir)
    if not path.is_file():
        return None
    raw = json.loads(path.read_text(encoding="utf-8"))
    access_key_id = str(raw.get("aws_access_key_id", "")).strip()
    secret_access_key = str(raw.get("aws_secret_access_key", "")).strip()
    if access_key_id and secret_access_key:
        return TerraformStateCredentials(access_key_id, secret_access_key)
    return None


def write_terraform_state_credentials(
    state_dir: Path, credentials: TerraformStateCredentials
) -> Path:
    """Write Terraform state-bucket keys next to the Terraform work directory."""
    path = credentials_path(state_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "aws_access_key_id": credentials.access_key_id,
                "aws_secret_access_key": credentials.secret_access_key,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    path.chmod(0o600)
    return path


def write_terraform_state_profile(
    work_dir: Path, credentials: TerraformStateCredentials
) -> None:
    """Write the shared-credentials file the S3 backend reads its keys from."""
    path = work_dir / TFSTATE_BACKEND_CREDENTIALS_FILE
    path.write_text(
        f"[{TFSTATE_BACKEND_PROFILE}]\n"
        f"aws_access_key_id = {credentials.access_key_id}\n"
        f"aws_secret_access_key = {credentials.secret_access_key}\n",
        encoding="utf-8",
    )
    path.chmod(0o600)


def provision_tfstate_user(
    spec: ClusterSpec,
    run: AwsRun = subprocess.run,
) -> TerraformStateCredentials:
    """Create the state IAM user and a new access key."""
    user_name = f"{spec.name}-tfstate"
    if not _iam_user_exists(user_name, run=run):
        _run_aws_json(["iam", "create-user", "--user-name", user_name], run=run)
    _run_aws_ok(
        [
            "iam",
            "put-user-policy",
            "--user-name",
            user_name,
            "--policy-name",
            user_name,
            "--policy-document",
            json.dumps(_state_bucket_policy(spec.terraform_state.bucket)),
        ],
        run=run,
    )
    credentials = _create_access_key(user_name, run=run)
    _wait_for_access_key(credentials, run=run)
    return credentials


def delete_cluster_terraform_state(
    spec: ClusterSpec,
    state_dir: Path,
    run: AwsRun = subprocess.run,
) -> None:
    """Delete this cluster's Terraform state object. The state bucket is kept."""
    credentials = read_terraform_state_credentials(state_dir)
    if credentials is None:
        msg = (
            "Terraform state credentials are missing. Restore "
            f"{credentials_path(state_dir)}."
        )
        raise click.ClickException(msg)
    key = spec.terraform_state.object_key(spec.name)
    for object_key in (key, f"{key}.tflock"):
        result = _run_aws(
            [
                "s3api",
                "delete-object",
                "--bucket",
                spec.terraform_state.bucket,
                "--key",
                object_key,
                "--region",
                spec.aws_cloud().region,
            ],
            run=run,
            env=credentials.environ(),
        )
        if result.returncode != 0:
            detail = (result.stderr or result.stdout or "").strip()
            msg = (
                "Could not delete Terraform state for cluster "
                f"{spec.name!r}. {detail}".strip()
            )
            raise click.ClickException(msg)


def _create_terraform_state_bucket(
    spec: ClusterSpec,
    create: bool,
    prompt: ConfirmFn | None,
    run: AwsRun,
) -> None:
    bucket = spec.terraform_state.bucket
    if not create:
        if prompt is None or not prompt(
            f"Terraform state bucket {bucket!r} does not exist. Create it in AWS?"
        ):
            msg = (
                f"Terraform state bucket {bucket!r} is required. "
                "Set terraform_state.create_bucket, pass --create-state-bucket, "
                "or set terraform_state.bucket to an existing bucket."
            )
            raise click.ClickException(msg)
    region = spec.aws_cloud().region
    args = ["s3api", "create-bucket", "--bucket", bucket, "--region", region]
    if region != "us-east-1":
        args.extend(
            [
                "--create-bucket-configuration",
                f"LocationConstraint={region}",
            ]
        )
    _run_aws_ok(
        args, run=run, failure=f"Could not create Terraform state bucket {bucket!r}."
    )


def _state_bucket_policy(bucket: str) -> dict[str, object]:
    arn = f"arn:aws:s3:::{bucket}"
    return {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Action": "s3:*",
                "Resource": [arn, f"{arn}/*"],
            }
        ],
    }


def _iam_user_exists(user_name: str, run: AwsRun) -> bool:
    result = _run_aws(["iam", "get-user", "--user-name", user_name], run=run)
    if result.returncode == 0:
        return True
    if "NoSuchEntity" in f"{result.stderr or ''} {result.stdout or ''}":
        return False
    detail = (result.stderr or result.stdout or "").strip()
    msg = f"Could not look up IAM user {user_name!r}. {detail}".strip()
    raise click.ClickException(msg)


def _create_access_key(user_name: str, run: AwsRun) -> TerraformStateCredentials:
    listed = _run_aws_json(
        ["iam", "list-access-keys", "--user-name", user_name], run=run
    )
    existing = listed.get("AccessKeyMetadata", [])
    if not isinstance(existing, list):
        msg = f"IAM user {user_name!r} returned an unexpected access key list."
        raise click.ClickException(msg)
    if len(existing) >= 2:
        msg = (
            f"IAM user {user_name!r} already has two access keys. "
            "Delete one, or restore the Terraform state credentials file."
        )
        raise click.ClickException(msg)
    created = _run_aws_json(
        ["iam", "create-access-key", "--user-name", user_name], run=run
    )
    access_key = created.get("AccessKey")
    if not isinstance(access_key, dict):
        msg = f"IAM did not return an access key for user {user_name!r}."
        raise click.ClickException(msg)
    access_key_id = access_key.get("AccessKeyId")
    secret_access_key = access_key.get("SecretAccessKey")
    if not isinstance(access_key_id, str) or not isinstance(secret_access_key, str):
        msg = f"IAM did not return an access key for user {user_name!r}."
        raise click.ClickException(msg)
    return TerraformStateCredentials(access_key_id, secret_access_key)


def _wait_for_access_key(credentials: TerraformStateCredentials, run: AwsRun) -> None:
    # IAM rejects a new access key for several seconds after it is created.
    for _ in range(30):
        result = _run_aws(
            ["sts", "get-caller-identity"], run=run, env=credentials.environ()
        )
        if result.returncode == 0:
            return
        time.sleep(2)
    detail = (result.stderr or result.stdout or "").strip()
    msg = f"New Terraform state access key was not accepted by AWS. {detail}".strip()
    raise click.ClickException(msg)


def _run_aws_ok(args: list[str], run: AwsRun, failure: str | None = None) -> None:
    result = _run_aws(args, run=run)
    if result.returncode == 0:
        return
    detail = (result.stderr or result.stdout or "").strip()
    prefix = failure or f"aws {' '.join(args)} failed."
    msg = f"{prefix} {detail}".strip()
    raise click.ClickException(msg)


def _run_aws_json(args: list[str], run: AwsRun) -> dict[str, object]:
    result = _run_aws(args, run=run)
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        msg = f"aws {' '.join(args)} failed. {detail}".strip()
        raise click.ClickException(msg)
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        msg = f"aws {' '.join(args)} returned invalid JSON."
        raise click.ClickException(msg) from exc
    if not isinstance(payload, dict):
        msg = f"aws {' '.join(args)} returned invalid JSON."
        raise click.ClickException(msg)
    return payload
