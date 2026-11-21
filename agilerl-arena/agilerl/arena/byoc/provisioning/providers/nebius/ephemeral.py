# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Ensure per-region Nebius IAM for ephemeral GPU jobs."""

from __future__ import annotations

import subprocess

import click

from agilerl.arena.byoc.provisioning.providers.nebius.terraform import (
    NebiusRun,
    _delete_nebius_resource,
    _find_membership_id,
    _find_named_resource,
    _nebius_list_items,
    _resource_id,
    _resource_name,
    _run_nebius,
    _run_nebius_json,
    delete_nebius_project,
)
from agilerl.arena.byoc.provisioning.spec import ClusterSpec

TUNNEL_GROUP = "arena-tunnel-agents"
TUNNEL_SERVICE_ACCOUNT = "arena-tunnel-agent"
TUNNEL_ROLE = "applicationtunnel.agent"
PROVISIONERS_GROUP = "arena-byoc-provisioners"
PROVISIONER_SERVICE_ACCOUNT = "arena-byoc-provisioner"
PROVISIONER_ROLE = "admin"


def region_project_name(cluster_name: str, region: str) -> str:
    """Return the ephemeral IAM project name for one cluster in one region."""
    return f"{cluster_name}-{region}"


def ensure_ephemeral_regions(
    spec: ClusterSpec, run: NebiusRun = subprocess.run
) -> None:
    """Ensure a ``{cluster}-{region}`` project for every region this tenant uses.

    :param spec: Cluster spec. No-op unless ``values.placement`` is ephemeral.
    :param run: Subprocess runner for the Nebius CLI.
    """
    if spec.values.placement != "ephemeral":
        return
    # Shared by every cluster in the tenant. Not in Terraform state.
    cloud = spec.nebius_cloud()
    projects = _nebius_list_items(
        ["iam", "project", "list", "--parent-id", cloud.tenant_id],
        run=run,
    )
    regions: list[str] = []
    default_project_ids: list[str] = []
    for project in projects:
        if not _resource_name(project).startswith("default"):
            continue
        region = _project_region(project)
        if not region or region in regions:
            continue
        regions.append(region)
        default_project_ids.append(_resource_id(project))
    if cloud.region and cloud.region not in regions:
        regions.append(cloud.region)
    if not regions:
        msg = (
            f"Tenant {cloud.tenant_id!r} has no default projects and "
            "nebius.region is unset, so there is no ephemeral region to ensure."
        )
        raise click.ClickException(msg)
    click.echo(f"Ensuring ephemeral Nebius projects for {', '.join(regions)}.")
    provisioner_id = _provisioner_service_account(spec, default_project_ids, run)
    provisioners_group_id = _ensure_named(
        ["iam", "group"],
        PROVISIONERS_GROUP,
        cloud.tenant_id,
        run,
    )
    _ensure_membership(provisioners_group_id, provisioner_id, run)
    for region in regions:
        project_id = _ensure_region_project(spec.name, cloud.tenant_id, region, run)
        _ensure_tunnel_agent(project_id, run)
        _ensure_permit(provisioners_group_id, PROVISIONER_ROLE, project_id, run)


def cluster_region_project_ids(
    spec: ClusterSpec, run: NebiusRun = subprocess.run
) -> list[str]:
    """Return this cluster's ``{cluster}-{region}`` project ids."""
    return [
        _resource_id(project)
        for project in _nebius_list_items(
            ["iam", "project", "list", "--parent-id", spec.nebius_cloud().tenant_id],
            run=run,
        )
        if _is_region_project(project, spec.name)
    ]


def delete_ephemeral_regions(
    spec: ClusterSpec, run: NebiusRun = subprocess.run
) -> None:
    """Delete this cluster's region projects, leaving the shared provisioners group while another cluster has one.

    :param spec: Cluster spec. The provisioner service account is kept.
    :param run: Subprocess runner for the Nebius CLI.
    """
    cloud = spec.nebius_cloud()
    projects = _nebius_list_items(
        ["iam", "project", "list", "--parent-id", cloud.tenant_id],
        run=run,
    )
    project_ids = [
        _resource_id(project)
        for project in projects
        if _is_region_project(project, spec.name)
    ]
    group = _find_named_resource(
        ["iam", "group"],
        PROVISIONERS_GROUP,
        cloud.tenant_id,
        run=run,
    )
    if group is not None:
        group_id = _resource_id(group)
        if any(_is_other_region_project(project, spec.name) for project in projects):
            _revoke_project_permits(group_id, project_ids, run)
        else:
            _delete_group(group_id, run)
    for project_id in project_ids:
        delete_nebius_project(project_id, run=run)
    if project_ids:
        click.echo(f"Deleted {len(project_ids)} Nebius region projects.")


def _is_region_project(project: dict[str, object], cluster_name: str) -> bool:
    """Return whether this project is this cluster's ephemeral region project."""
    region = _project_region(project)
    return bool(region) and _resource_name(project) == region_project_name(
        cluster_name, region
    )


def _is_other_region_project(project: dict[str, object], cluster_name: str) -> bool:
    """Return whether this is another cluster's ``{cluster}-{region}`` project."""
    if _is_region_project(project, cluster_name):
        return False
    region = _project_region(project)
    if not region:
        return False
    name = _resource_name(project)
    suffix = f"-{region}"
    if name.startswith("default") or not name.endswith(suffix):
        return False
    return len(name) > len(suffix)


def _revoke_project_permits(
    group_id: str, project_ids: list[str], run: NebiusRun
) -> None:
    """Remove this cluster's admin permits from the shared provisioners group."""
    targets = set(project_ids)
    for item in _nebius_list_items(
        ["iam", "access-permit", "list", "--parent-id", group_id],
        run=run,
    ):
        permit = item.get("spec")
        if not isinstance(permit, dict) or permit.get("resource_id") not in targets:
            continue
        _delete_nebius_resource(
            ["iam", "access-permit", "delete", "--id", _resource_id(item)],
            run,
        )


def _delete_group(group_id: str, run: NebiusRun) -> None:
    """Delete a group's memberships and permits, then the group."""
    for item in _nebius_list_items(
        ["iam", "group-membership", "list-members", "--parent-id", group_id],
        run=run,
    ):
        _delete_nebius_resource(
            ["iam", "group-membership", "delete", "--id", _resource_id(item)],
            run,
        )
    for item in _nebius_list_items(
        ["iam", "access-permit", "list", "--parent-id", group_id],
        run=run,
    ):
        _delete_nebius_resource(
            ["iam", "access-permit", "delete", "--id", _resource_id(item)],
            run,
        )
    _delete_nebius_resource(["iam", "group", "delete", "--id", group_id], run)


def _provisioner_service_account(
    spec: ClusterSpec, default_project_ids: list[str], run: NebiusRun
) -> str:
    """Return the provisioner service account, creating it in the cluster project."""
    parents = [_cli_parent_id(run), *default_project_ids]
    seen: set[str] = set()
    for parent_id in parents:
        if not parent_id or parent_id in seen:
            continue
        seen.add(parent_id)
        found = _find_named_resource(
            ["iam", "service-account"],
            PROVISIONER_SERVICE_ACCOUNT,
            parent_id,
            run=run,
        )
        if found is not None:
            return _resource_id(found)
    return _ensure_named(
        ["iam", "service-account"],
        PROVISIONER_SERVICE_ACCOUNT,
        _cluster_project_id(spec, run),
        run,
    )


def _cluster_project_id(spec: ClusterSpec, run: NebiusRun) -> str:
    """Return the managed-Kubernetes project, looked up by cluster name if needed."""
    project_id = spec.nebius_cloud().project_id
    if project_id is not None:
        return project_id
    found = _find_named_resource(
        ["iam", "project"],
        spec.name,
        spec.nebius_cloud().tenant_id,
        run=run,
    )
    if found is None:
        msg = (
            f"Nebius project {spec.name!r} was not found. "
            f"Service account {PROVISIONER_SERVICE_ACCOUNT!r} is created there."
        )
        raise click.ClickException(msg)
    return _resource_id(found)


def _cli_parent_id(run: NebiusRun) -> str:
    """Return the Nebius CLI profile parent project id."""
    result = _run_nebius(["config", "get", "parent-id"], run=run, check=True)
    parent_id = result.stdout.strip().strip('"')
    if not parent_id:
        msg = "Nebius CLI has no parent-id. Set it with nebius config set parent-id."
        raise click.ClickException(msg)
    return parent_id


def _ensure_region_project(
    cluster_name: str, tenant_id: str, region: str, run: NebiusRun
) -> str:
    """Return this cluster's region project id, creating the project if needed."""
    name = region_project_name(cluster_name, region)
    found = _find_named_resource(["iam", "project"], name, tenant_id, run=run)
    if found is not None:
        return _resource_id(found)
    created = _run_nebius_json(
        [
            "iam",
            "v2",
            "project",
            "create",
            "--parent-id",
            tenant_id,
            "--name",
            name,
            "--region",
            region,
        ],
        run=run,
    )
    return _resource_id(created)


def _ensure_tunnel_agent(project_id: str, run: NebiusRun) -> None:
    """Ensure the tunnel group, service account, membership, and permit."""
    group_id = _ensure_named(["iam", "group"], TUNNEL_GROUP, project_id, run)
    service_account_id = _ensure_named(
        ["iam", "service-account"],
        TUNNEL_SERVICE_ACCOUNT,
        project_id,
        run,
    )
    _ensure_membership(group_id, service_account_id, run)
    _ensure_permit(group_id, TUNNEL_ROLE, project_id, run)


def _ensure_named(
    service_args: list[str], name: str, parent_id: str, run: NebiusRun
) -> str:
    """Return the id of a named IAM resource, creating it under *parent_id*."""
    found = _find_named_resource(service_args, name, parent_id, run=run)
    if found is not None:
        return _resource_id(found)
    created = _run_nebius_json(
        [*service_args, "create", "--parent-id", parent_id, "--name", name],
        run=run,
    )
    return _resource_id(created)


def _ensure_membership(group_id: str, member_id: str, run: NebiusRun) -> None:
    """Add *member_id* to *group_id* when the membership is missing."""
    if _find_membership_id(group_id, member_id, run=run) is not None:
        return
    _run_nebius_json(
        [
            "iam",
            "group-membership",
            "create",
            "--parent-id",
            group_id,
            "--member-id",
            member_id,
        ],
        run=run,
    )


def _ensure_permit(group_id: str, role: str, resource_id: str, run: NebiusRun) -> None:
    """Grant *role* on *resource_id* to *group_id* when that permit is missing."""
    for item in _nebius_list_items(
        ["iam", "access-permit", "list", "--parent-id", group_id],
        run=run,
    ):
        spec = item.get("spec")
        if not isinstance(spec, dict):
            continue
        if spec.get("role") == role and spec.get("resource_id") == resource_id:
            return
    _run_nebius_json(
        [
            "iam",
            "access-permit",
            "create",
            "--parent-id",
            group_id,
            "--role",
            role,
            "--resource-id",
            resource_id,
        ],
        run=run,
    )


def _project_region(project: dict[str, object]) -> str:
    """Return the region stored on a project, or an empty string."""
    for key in ("status", "spec"):
        section = project.get(key)
        if not isinstance(section, dict):
            continue
        region = section.get("region")
        if isinstance(region, str) and region.strip():
            return region.strip()
    return ""
