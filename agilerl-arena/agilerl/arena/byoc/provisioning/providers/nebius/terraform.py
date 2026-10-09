# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Nebius object-storage bucket used as the Terraform S3 backend."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.parse import quote, urlparse
from urllib.request import Request, urlopen

import click

from agilerl.arena.byoc.provisioning.providers.nebius.kubeconfig import (
    resolve_kubeconfig_executable,
)
from agilerl.arena.byoc.provisioning.spec import ClusterSpec

NebiusRun = Callable[..., subprocess.CompletedProcess[str]]
ConfirmFn = Callable[[str], bool]

TFSTATE_CREDENTIALS_FILE = "tfstate-credentials.json"
NEBIUS_STORAGE_HOST = re.compile(r"^storage\.(?P<region>[a-z0-9-]+)\.nebius\.cloud$")
DEFAULT_NEBIUS_REGION = "eu-north1"
EMPTY_PAYLOAD_SHA256 = hashlib.sha256(b"").hexdigest()
# Nebius list commands default to 10 items and page with next_page_token.
NEBIUS_LIST_PAGE_SIZE = "100"
# Most list responses carry `items`; group memberships carry `memberships`.
NEBIUS_LIST_ITEM_KEYS = ("items", "memberships")
MISSING_BUCKET_MARKERS = (
    "nosuchbucket",
    "not_found",
    "not found",
    "code = notfound",
    "bucket doesn't exist",
    "bucket does not exist",
)


@dataclass(frozen=True)
class TerraformStateCredentials:
    """AWS-compatible keys for the Terraform S3 backend."""

    access_key_id: str
    secret_access_key: str

    def environ(self) -> dict[str, str]:
        """Return subprocess environment updates for Terraform."""
        return {
            "AWS_ACCESS_KEY_ID": self.access_key_id,
            "AWS_SECRET_ACCESS_KEY": self.secret_access_key,
        }


def nebius_region_from_endpoint(endpoint: str) -> str:
    """Return the Nebius region encoded in an object-storage endpoint."""
    host = urlparse(endpoint).hostname or ""
    match = NEBIUS_STORAGE_HOST.fullmatch(host)
    return match["region"] if match else DEFAULT_NEBIUS_REGION


def storage_project_id(spec: ClusterSpec) -> str:
    """Return the Nebius project that owns the state and experiment buckets."""
    project_id = spec.nebius_cloud().storage_project_id
    if project_id is None:
        msg = (
            "The Nebius storage project is unknown. Provisioning creates it before "
            "the Terraform state bucket; set nebius.storage_project_id to use an "
            "existing project."
        )
        raise click.ClickException(msg)
    return project_id


def credentials_path(state_dir: Path) -> Path:
    """Return the local file that caches Terraform state-bucket keys."""
    return state_dir.expanduser().resolve().parent / TFSTATE_CREDENTIALS_FILE


def is_missing_bucket_error(stdout: str, stderr: str) -> bool:
    """Return whether Nebius CLI output means the bucket does not exist."""
    combined = f"{stdout} {stderr}".lower()
    return any(marker in combined for marker in MISSING_BUCKET_MARKERS)


def terraform_state_bucket_exists(
    spec: ClusterSpec,
    run: NebiusRun = subprocess.run,
) -> bool:
    """Return whether the spec's Terraform state bucket exists in Nebius."""
    result = _run_nebius(
        [
            "storage",
            "bucket",
            "get-by-name",
            "--name",
            spec.terraform_state.bucket,
            "--parent-id",
            storage_project_id(spec),
        ],
        run=run,
        check=False,
    )
    if result.returncode == 0:
        return True
    if is_missing_bucket_error(result.stdout, result.stderr):
        return False
    detail = (result.stderr or result.stdout).strip()
    msg = (
        "Could not look up Terraform state bucket "
        f"{spec.terraform_state.bucket!r}."
        f"{f' {detail}' if detail else ''}"
    )
    raise click.ClickException(msg)


def experiment_storage_import_ids(
    spec: ClusterSpec,
    run: NebiusRun = subprocess.run,
) -> dict[str, str]:
    """Return Terraform import ids for existing experiment storage, or empty."""
    bucket_name = spec.arena.storage.bucket
    sa_name = f"{spec.name}-storage"
    group_name = f"{spec.name}-storage-editors"
    project_id = storage_project_id(spec)
    bucket = _find_named_resource(
        ["storage", "bucket"],
        name=bucket_name,
        parent_id=project_id,
        run=run,
    )
    sa = _find_named_resource(
        ["iam", "service-account"],
        name=sa_name,
        parent_id=project_id,
        run=run,
    )
    group = _find_named_resource(
        ["iam", "group"],
        name=group_name,
        parent_id=spec.nebius_cloud().tenant_id,
        run=run,
    )
    if bucket is None and sa is None and group is None:
        return {}
    if bucket is None:
        msg = (
            f"Storage IAM for cluster {spec.name!r} still exists, but experiment "
            f"bucket {bucket_name!r} was not found. Restore the bucket, or pass "
            "--delete-storage and provision again."
        )
        raise click.ClickException(msg)
    if sa is None:
        msg = (
            f"Experiment bucket {bucket_name!r} exists, but service account "
            f"{sa_name!r} was not found. Restore it, or pass --delete-storage "
            "to wipe the bucket and provision again."
        )
        raise click.ClickException(msg)
    if group is None:
        msg = (
            f"Experiment bucket {bucket_name!r} exists, but group "
            f"{group_name!r} was not found. Restore it, or pass --delete-storage "
            "to wipe the bucket and provision again."
        )
        raise click.ClickException(msg)
    sa_id = _resource_id(sa)
    group_id = _resource_id(group)
    membership_id = _membership_id(group_id, sa_id, run=run)
    access_key_id = _storage_access_key_id(sa_id, sa_name, run=run)
    return {
        "nebius_iam_v1_service_account.storage": sa_id,
        "nebius_iam_v1_group.storage_editors": group_id,
        "nebius_iam_v1_group_membership.storage_editor": membership_id,
        "nebius_iam_v2_access_key.storage": access_key_id,
        "nebius_storage_v1_bucket.arena_data": _resource_id(bucket),
    }


def ensure_terraform_state_bucket(
    spec: ClusterSpec,
    state_dir: Path,
    create: bool = False,
    prompt: ConfirmFn | None = None,
    run: NebiusRun = subprocess.run,
) -> dict[str, str]:
    """Require a Terraform state bucket, creating it with the Nebius CLI if asked."""
    if terraform_state_bucket_exists(spec, run=run):
        credentials = read_terraform_state_credentials(state_dir)
        if credentials is None:
            credentials = fetch_terraform_state_credentials(spec, run=run)
            write_terraform_state_credentials(state_dir, credentials)
        return credentials.environ()
    bucket = spec.terraform_state.bucket
    if not create:
        if prompt is None or not prompt(
            f"Terraform state bucket {bucket!r} does not exist. Create it in Nebius?"
        ):
            msg = (
                f"Terraform state bucket {bucket!r} is required. "
                "Set terraform_state.create_bucket, pass --create-state-bucket, "
                "or set terraform_state.bucket to an existing bucket and export "
                "AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY."
            )
            raise click.ClickException(msg)
    credentials = provision_terraform_state_bucket(spec, run=run)
    write_terraform_state_credentials(state_dir, credentials)
    return credentials.environ()


def read_terraform_state_credentials(
    state_dir: Path,
) -> TerraformStateCredentials | None:
    """Return S3 backend keys from the environment or the local credentials file."""
    access_key_id = os.environ.get("AWS_ACCESS_KEY_ID", "").strip()
    secret_access_key = os.environ.get("AWS_SECRET_ACCESS_KEY", "").strip()
    if access_key_id and secret_access_key:
        return TerraformStateCredentials(access_key_id, secret_access_key)
    path = credentials_path(state_dir)
    if path.is_file():
        raw = json.loads(path.read_text(encoding="utf-8"))
        file_id = str(raw.get("aws_access_key_id", "")).strip()
        file_secret = str(raw.get("aws_secret_access_key", "")).strip()
        if file_id and file_secret:
            return TerraformStateCredentials(file_id, file_secret)
    return None


def load_terraform_state_credentials(state_dir: Path) -> TerraformStateCredentials:
    """Load S3 backend keys from the environment or the local credentials file."""
    credentials = read_terraform_state_credentials(state_dir)
    if credentials is not None:
        return credentials
    msg = (
        "AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY are required to use the "
        "Terraform state bucket. Export them, or keep "
        f"{credentials_path(state_dir)} from a previous --create-state-bucket run."
    )
    raise click.ClickException(msg)


def fetch_terraform_state_credentials(
    spec: ClusterSpec,
    run: NebiusRun = subprocess.run,
) -> TerraformStateCredentials:
    """Read the cluster's Terraform state access key back from Nebius."""
    name = f"{spec.name}-tfstate"
    sa = _find_named_resource(
        ["iam", "service-account"],
        name=name,
        parent_id=storage_project_id(spec),
        run=run,
    )
    access_key_id = (
        _find_access_key_id(_resource_id(sa), name, run=run) if sa is not None else None
    )
    if access_key_id is None:
        msg = (
            f"Terraform state bucket {spec.terraform_state.bucket!r} exists, but no "
            f"access key named {name!r} was found in Nebius. Export "
            "AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY for that bucket."
        )
        raise click.ClickException(msg)
    payload = _run_nebius_json(
        ["iam", "v2", "access-key", "get-secret", "--id", access_key_id], run=run
    )
    return _credentials_from_fields(payload)


def delete_terraform_state_bucket(
    spec: ClusterSpec,
    run: NebiusRun = subprocess.run,
) -> None:
    """Delete the Terraform state bucket. Nebius keeps it out of Terraform state."""
    bucket = _find_named_resource(
        ["storage", "bucket"],
        name=spec.terraform_state.bucket,
        parent_id=storage_project_id(spec),
        run=run,
    )
    if bucket is None:
        return
    # A zero TTL deletes an active bucket instead of scheduling it.
    _run_nebius(
        ["storage", "bucket", "delete", "--id", _resource_id(bucket), "--ttl", "0s"],
        run=run,
        check=True,
    )


def delete_nebius_project(
    project_id: str,
    run: NebiusRun = subprocess.run,
) -> None:
    """Delete a Nebius project that Terraform created for a cluster."""
    _run_nebius(
        ["iam", "v2", "project", "delete", "--id", project_id], run=run, check=True
    )


def delete_cluster_terraform_state(
    spec: ClusterSpec,
    state_dir: Path,
    opener: Callable[..., object] = urlopen,
) -> None:
    """Delete this cluster's Terraform state object. The state bucket is kept."""
    credentials = load_terraform_state_credentials(state_dir)
    endpoint = spec.terraform_state.s3_endpoint(spec.arena.storage.endpoint)
    region = nebius_region_from_endpoint(endpoint)
    key = spec.terraform_state.object_key(spec.name)
    for object_key in (key, f"{key}.tflock"):
        _delete_s3_object(
            bucket=spec.terraform_state.bucket,
            key=object_key,
            endpoint=endpoint,
            region=region,
            credentials=credentials,
            opener=opener,
        )


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


def provision_terraform_state_bucket(
    spec: ClusterSpec,
    run: NebiusRun = subprocess.run,
) -> TerraformStateCredentials:
    """Create the Terraform state bucket, service account, and access key.

    Each step reuses what an earlier attempt left behind. The group lives at
    tenant scope, so its name collides across attempts even for a new project.
    """
    cluster_name = spec.name
    project_id = storage_project_id(spec)
    sa_name = f"{cluster_name}-tfstate"
    sa_id = _ensure_named_resource(
        ["iam", "service-account"],
        name=sa_name,
        parent_id=project_id,
        run=run,
    )
    group_id = _ensure_named_resource(
        ["iam", "group"],
        name=f"{cluster_name}-tfstate-editors",
        parent_id=spec.nebius_cloud().tenant_id,
        run=run,
    )
    if _find_membership_id(group_id, sa_id, run=run) is None:
        _run_nebius_json(
            [
                "iam",
                "group-membership",
                "create",
                "--parent-id",
                group_id,
                "--member-id",
                sa_id,
            ],
            run=run,
        )
    policy = json.dumps(
        [{"paths": ["*"], "roles": ["storage.editor"], "group_id": group_id}]
    )
    _run_nebius_json(
        [
            "storage",
            "bucket",
            "create",
            "--name",
            spec.terraform_state.bucket,
            "--parent-id",
            project_id,
            "--bucket-policy-rules",
            policy,
        ],
        run=run,
    )
    return _ensure_state_access_key(sa_id, name=sa_name, project_id=project_id, run=run)


def _ensure_named_resource(
    service_args: list[str],
    name: str,
    parent_id: str,
    run: NebiusRun,
) -> str:
    """Return the id of a named Nebius resource, creating it when absent."""
    existing = _find_named_resource(
        service_args, name=name, parent_id=parent_id, run=run
    )
    if existing is not None:
        return _resource_id(existing)
    created = _run_nebius_json(
        [*service_args, "create", "--name", name, "--parent-id", parent_id],
        run=run,
    )
    return _resource_id(created)


def _ensure_state_access_key(
    sa_id: str,
    name: str,
    project_id: str,
    run: NebiusRun,
) -> TerraformStateCredentials:
    """Return the service account's state access key, creating it when absent."""
    access_key_id = _find_access_key_id(sa_id, name, run=run)
    if access_key_id is not None:
        return _credentials_from_fields(
            _run_nebius_json(
                ["iam", "v2", "access-key", "get-secret", "--id", access_key_id],
                run=run,
            )
        )
    created = _run_nebius_json(
        [
            "iam",
            "v2",
            "access-key",
            "create",
            "--name",
            name,
            "--parent-id",
            project_id,
            "--account-service-account-id",
            sa_id,
            "--secret-delivery-mode",
            "inline",
        ],
        run=run,
    )
    return _access_key_credentials(created)


def _delete_s3_object(
    bucket: str,
    key: str,
    endpoint: str,
    region: str,
    credentials: TerraformStateCredentials,
    opener: Callable[..., object],
) -> None:
    url = _s3_object_url(endpoint, bucket, key)
    amz_date = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    headers = _s3_sigv4_headers(
        method="DELETE",
        url=url,
        region=region,
        amz_date=amz_date,
        credentials=credentials,
    )
    request = Request(url, method="DELETE", headers=headers)
    try:
        with opener(request, timeout=30):
            return
    except HTTPError as exc:
        if exc.code == 404:
            return
        detail = exc.reason if isinstance(exc.reason, str) else str(exc.reason)
        msg = f"Could not delete Terraform state object {key!r}: HTTP {exc.code} {detail}."
        raise click.ClickException(msg) from exc


def _s3_object_url(endpoint: str, bucket: str, key: str) -> str:
    encoded_key = "/".join(quote(part, safe="") for part in key.split("/"))
    return f"{endpoint.rstrip('/')}/{quote(bucket, safe='')}/{encoded_key}"


def _s3_sigv4_headers(
    method: str,
    url: str,
    region: str,
    amz_date: str,
    credentials: TerraformStateCredentials,
) -> dict[str, str]:
    parsed = urlparse(url)
    host = parsed.netloc
    canonical_uri = parsed.path or "/"
    date_stamp = amz_date[:8]
    signed_headers = "host;x-amz-content-sha256;x-amz-date"
    canonical_headers = (
        f"host:{host}\n"
        f"x-amz-content-sha256:{EMPTY_PAYLOAD_SHA256}\n"
        f"x-amz-date:{amz_date}\n"
    )
    canonical_request = (
        f"{method}\n{canonical_uri}\n\n{canonical_headers}\n"
        f"{signed_headers}\n{EMPTY_PAYLOAD_SHA256}"
    )
    credential_scope = f"{date_stamp}/{region}/s3/aws4_request"
    payload_hash = hashlib.sha256(canonical_request.encode("utf-8")).hexdigest()
    string_to_sign = f"AWS4-HMAC-SHA256\n{amz_date}\n{credential_scope}\n{payload_hash}"
    signing_key = _aws_signing_key(credentials.secret_access_key, date_stamp, region)
    signature = hmac.new(
        signing_key, string_to_sign.encode("utf-8"), hashlib.sha256
    ).hexdigest()
    authorization = (
        "AWS4-HMAC-SHA256 "
        f"Credential={credentials.access_key_id}/{credential_scope}, "
        f"SignedHeaders={signed_headers}, "
        f"Signature={signature}"
    )
    return {
        "Host": host,
        "x-amz-content-sha256": EMPTY_PAYLOAD_SHA256,
        "x-amz-date": amz_date,
        "Authorization": authorization,
    }


def _aws_signing_key(secret: str, date_stamp: str, region: str) -> bytes:
    date_key = hmac.new(
        f"AWS4{secret}".encode(), date_stamp.encode("utf-8"), hashlib.sha256
    ).digest()
    region_key = hmac.new(date_key, region.encode("utf-8"), hashlib.sha256).digest()
    service_key = hmac.new(region_key, b"s3", hashlib.sha256).digest()
    return hmac.new(service_key, b"aws4_request", hashlib.sha256).digest()


def _nebius_cmd(*args: str) -> list[str]:
    executable = resolve_kubeconfig_executable(["nebius"])
    if executable is None:
        msg = (
            "Nebius CLI not found. Install it, set NEBIUS_CLI to its path, "
            "or add ~/.nebius/bin to PATH."
        )
        raise click.ClickException(msg)
    return [str(executable), *args]


def _run_nebius(
    args: list[str],
    run: NebiusRun,
    check: bool,
) -> subprocess.CompletedProcess[str]:
    result = run(
        [*_nebius_cmd(*args), "--format", "json"],
        check=False,
        capture_output=True,
        text=True,
    )
    if check and result.returncode:
        detail = (result.stderr or result.stdout).strip()
        msg = f"nebius {' '.join(args)} failed.{f' {detail}' if detail else ''}"
        raise click.ClickException(msg)
    return result


def _run_nebius_json(args: list[str], run: NebiusRun) -> dict[str, Any]:
    payload = _parse_nebius_json(_run_nebius(args, run=run, check=True).stdout)
    if not isinstance(payload, dict):
        msg = "Nebius CLI returned invalid JSON."
        raise click.ClickException(msg)
    return payload


def _get_by_name(args: list[str], run: NebiusRun) -> dict[str, Any] | None:
    result = _run_nebius(args, run=run, check=False)
    if result.returncode == 0:
        payload = _parse_nebius_json(result.stdout)
        if not isinstance(payload, dict):
            msg = "Nebius CLI returned invalid JSON."
            raise click.ClickException(msg)
        return payload
    if is_missing_bucket_error(result.stdout, result.stderr):
        return None
    detail = (result.stderr or result.stdout).strip()
    msg = f"nebius {' '.join(args)} failed.{f' {detail}' if detail else ''}"
    raise click.ClickException(msg)


def _find_named_resource(
    service_args: list[str],
    name: str,
    parent_id: str,
    run: NebiusRun,
) -> dict[str, Any] | None:
    found = _get_by_name(
        [*service_args, "get-by-name", "--name", name, "--parent-id", parent_id],
        run=run,
    )
    if found is not None:
        return found
    for item in _nebius_list_items(
        [*service_args, "list", "--parent-id", parent_id], run=run
    ):
        if _resource_name(item) == name:
            return item
    return None


def _resource_name(payload: dict[str, Any]) -> str:
    metadata = payload.get("metadata")
    if isinstance(metadata, dict):
        name = metadata.get("name")
        if isinstance(name, str) and name.strip():
            return name.strip()
    name = payload.get("name")
    if isinstance(name, str) and name.strip():
        return name.strip()
    return ""


def _parse_nebius_json(stdout: str) -> object:
    try:
        return json.loads(stdout)
    except json.JSONDecodeError as exc:
        msg = "Nebius CLI returned invalid JSON."
        raise click.ClickException(msg) from exc


def _nebius_list_items(args: list[str], run: NebiusRun) -> list[dict[str, Any]]:
    """Return every item of a Nebius list command, following page tokens."""
    items: list[dict[str, Any]] = []
    page_token = ""
    while True:
        page_args = [*args, "--page-size", NEBIUS_LIST_PAGE_SIZE]
        if page_token:
            page_args.extend(["--page-token", page_token])
        result = _run_nebius(page_args, run=run, check=False)
        if result.returncode != 0:
            if is_missing_bucket_error(result.stdout, result.stderr):
                return items
            detail = (result.stderr or result.stdout).strip()
            msg = f"nebius {' '.join(args)} failed.{f' {detail}' if detail else ''}"
            raise click.ClickException(msg)
        payload = _parse_nebius_json(result.stdout)
        items.extend(_payload_items(payload))
        page_token = _next_page_token(payload)
        if not page_token:
            return items


def _payload_items(payload: object) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if not isinstance(payload, dict):
        msg = "Nebius CLI returned invalid JSON."
        raise click.ClickException(msg)
    for key in NEBIUS_LIST_ITEM_KEYS:
        value = payload.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
    # A list with no results is `{}`.
    return []


def _next_page_token(payload: object) -> str:
    if not isinstance(payload, dict):
        return ""
    token = payload.get("next_page_token")
    return token.strip() if isinstance(token, str) else ""


def _find_membership_id(group_id: str, sa_id: str, run: NebiusRun) -> str | None:
    for item in _nebius_list_items(
        ["iam", "group-membership", "list-members", "--parent-id", group_id],
        run=run,
    ):
        if _member_id(item) == sa_id:
            return _resource_id(item)
    return None


def _membership_id(group_id: str, sa_id: str, run: NebiusRun) -> str:
    membership_id = _find_membership_id(group_id, sa_id, run=run)
    if membership_id is not None:
        return membership_id
    msg = (
        f"Group {group_id!r} has no membership for service account {sa_id!r}. "
        "Restore it, or pass --delete-storage to wipe the bucket and provision again."
    )
    raise click.ClickException(msg)


def _member_id(item: dict[str, Any]) -> str:
    value = item.get("member_id")
    if isinstance(value, str) and value.strip():
        return value.strip()
    spec = item.get("spec")
    if isinstance(spec, dict):
        nested = spec.get("member_id")
        if isinstance(nested, str) and nested.strip():
            return nested.strip()
    return ""


def _find_access_key_id(sa_id: str, name: str, run: NebiusRun) -> str | None:
    for item in _nebius_list_items(
        [
            "iam",
            "v2",
            "access-key",
            "list-by-account",
            "--account-service-account-id",
            sa_id,
        ],
        run=run,
    ):
        if _resource_name(item) == name:
            return _resource_id(item)
    return None


def _storage_access_key_id(sa_id: str, name: str, run: NebiusRun) -> str:
    access_key_id = _find_access_key_id(sa_id, name, run=run)
    if access_key_id is not None:
        return access_key_id
    msg = (
        f"No access key named {name!r} for service account {sa_id!r}. "
        "Restore it, or pass --delete-storage to wipe the bucket and provision again."
    )
    raise click.ClickException(msg)


def _resource_id(payload: dict[str, Any]) -> str:
    metadata = payload.get("metadata")
    if isinstance(metadata, dict):
        resource_id = metadata.get("id")
        if isinstance(resource_id, str) and resource_id.strip():
            return resource_id.strip()
    msg = "Nebius CLI response missing metadata.id."
    raise click.ClickException(msg)


def _access_key_credentials(payload: dict[str, Any]) -> TerraformStateCredentials:
    status = payload.get("status")
    return _credentials_from_fields(status if isinstance(status, dict) else {})


def _credentials_from_fields(payload: dict[str, Any]) -> TerraformStateCredentials:
    access_key_id = payload.get("aws_access_key_id")
    secret_access_key = payload.get("secret")
    if (
        isinstance(access_key_id, str)
        and access_key_id.strip()
        and isinstance(secret_access_key, str)
        and secret_access_key.strip()
    ):
        return TerraformStateCredentials(
            access_key_id.strip(), secret_access_key.strip()
        )
    msg = "Nebius CLI access-key response missing AWS credentials."
    raise click.ClickException(msg)
