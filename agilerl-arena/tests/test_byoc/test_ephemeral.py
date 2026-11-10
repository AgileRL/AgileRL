# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Tests for ephemeral Nebius region IAM."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import click
import pytest

from agilerl.arena.byoc.commands import _ensure_registered_ephemeral_regions
from agilerl.arena.byoc.provisioning.providers.nebius.ephemeral import (
    PROVISIONER_ROLE,
    PROVISIONER_SERVICE_ACCOUNT,
    PROVISIONERS_GROUP,
    TUNNEL_GROUP,
    TUNNEL_ROLE,
    TUNNEL_SERVICE_ACCOUNT,
    delete_ephemeral_regions,
    ensure_ephemeral_regions,
)
from agilerl.arena.byoc.provisioning.providers.nebius.provider import NEBIUS
from agilerl.arena.byoc.provisioning.spec import ClusterSpec
from agilerl.arena.byoc.provisioning.terraform import ClusterOutputs

TENANT_ID = "tenant-e00t69c28j8pxr9tp4"
NORTH_PROJECT = "project-e00gb1w7pr00mtp6dhj9q5"
NORTH_TUNNEL_GROUP = "group-e00pv90qc9py2wf0e0"
NORTH_TUNNEL_SA = "serviceaccount-e00qe3er31tn5wezeg"
PROVISIONER_SA = "serviceaccount-e00yanfg5tmyvp2jtt"
PROVISIONERS = "group-e00fmrn3xka3ekmraa"
DEFAULT_NORTH = "project-e00avfkhpr00p76qmrfkvb"


def _spec(placement: str = "ephemeral", **nebius: object) -> ClusterSpec:
    """Return a Nebius cluster spec."""
    cloud: dict[str, object] = {
        "tenant_id": TENANT_ID,
        "project_id": "project",
        "storage_project_id": "storage-project",
        "subnet_id": "subnet",
        "object_storage": {"size_gib": 1024},
    }
    cloud.update(nebius)
    return ClusterSpec.model_validate(
        {
            "provider": "nebius",
            "name": "test",
            "terraform_state": {"bucket": "test-tfstate", "endpoint": "https://s3"},
            "arena": {
                "storage": {
                    "bucket": "bucket",
                    "endpoint": "https://storage.example.com",
                },
                "inference": {"domain": "inference.example.com"},
            },
            "values": {"placement": placement},
            "nebius": cloud,
        }
    )


def _outputs(tmp_path: Path) -> ClusterOutputs:
    """Return cluster outputs that post-apply can stamp."""
    return ClusterOutputs(
        cluster_name="test",
        context="test",
        kubeconfig_path=tmp_path / "kubeconfig",
        storage_access_key_id="key",
        storage_secret_access_key="secret",
        worker_node_group_ids={},
    )


def _resource(resource_id: str, name: str, **spec: object) -> dict[str, object]:
    """Return a Nebius resource payload."""
    payload: dict[str, object] = {"metadata": {"id": resource_id, "name": name}}
    if spec:
        payload["spec"] = spec
        payload["status"] = spec
    return payload


def _metadata_id(payload: dict[str, object]) -> str:
    """Return ``metadata.id``, or an empty string."""
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        return ""
    resource_id = metadata.get("id")
    return resource_id if isinstance(resource_id, str) else ""


def _flag(argv: list[str], name: str) -> str:
    """Return the value of a CLI flag."""
    return argv[argv.index(name) + 1]


def _ok(argv: list[str], payload: object) -> subprocess.CompletedProcess[str]:
    """Return a successful CLI result."""
    text = payload if isinstance(payload, str) else json.dumps(payload)
    return subprocess.CompletedProcess(argv, 0, text, "")


def _missing(argv: list[str]) -> subprocess.CompletedProcess[str]:
    """Return a Nebius not-found result."""
    return subprocess.CompletedProcess(argv, 1, "", "rpc error: code = NotFound")


class _NebiusIamCli:
    """Nebius CLI fake for project, group, membership, and permit calls."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []
        self.parent_id = DEFAULT_NORTH
        self.project_pages: list[list[dict[str, object]]] = [[]]
        self.named: dict[tuple[str, str, str], dict[str, object]] = {}
        self.memberships: dict[str, list[dict[str, object]]] = {}
        self.permits: dict[str, list[dict[str, object]]] = {}
        self._seq = 0

    def add(
        self,
        kind: str,
        parent_id: str,
        name: str,
        resource_id: str,
        **spec: object,
    ) -> None:
        """Remember a resource ``get-by-name`` can return."""
        self.named[(kind, parent_id, name)] = _resource(resource_id, name, **spec)

    def add_membership(self, group_id: str, member_id: str, membership_id: str) -> None:
        """Remember a group member."""
        self.memberships.setdefault(group_id, []).append(
            {
                "metadata": {"id": membership_id},
                "spec": {"member_id": member_id},
            }
        )

    def add_permit(
        self, group_id: str, role: str, resource_id: str, permit_id: str
    ) -> None:
        """Remember an access permit."""
        self.permits.setdefault(group_id, []).append(
            {
                "metadata": {"id": permit_id},
                "spec": {"role": role, "resource_id": resource_id},
            }
        )

    def __call__(
        self, argv: list[str], **_kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        self.calls.append(argv)
        args = argv[1:]
        if args[:3] == ["config", "get", "parent-id"]:
            return _ok(argv, self.parent_id)
        if args[:4] == ["iam", "project", "list", "--parent-id"]:
            return self._project_page(argv)
        if args[:3] == ["iam", "group-membership", "list-members"]:
            rows = self.memberships.get(_flag(argv, "--parent-id"), [])
            return _ok(argv, {"memberships": rows} if rows else {})
        if args[:3] == ["iam", "group-membership", "create"]:
            return self._create_membership(argv)
        if args[:3] == ["iam", "access-permit", "list"]:
            rows = self.permits.get(_flag(argv, "--parent-id"), [])
            return _ok(argv, {"items": rows} if rows else {})
        if args[:3] == ["iam", "access-permit", "create"]:
            return self._create_permit(argv)
        if len(args) >= 3 and args[2] == "get-by-name":
            found = self.named.get(
                (args[1], _flag(argv, "--parent-id"), _flag(argv, "--name"))
            )
            return _ok(argv, found) if found is not None else _missing(argv)
        if len(args) >= 3 and args[2] == "list":
            return self._named_list(argv, args[1])
        if args[:4] == ["iam", "v2", "project", "create"]:
            return self._create_named(argv, "project", region=_flag(argv, "--region"))
        if args[:4] == ["iam", "v2", "project", "delete"]:
            self._drop_id(_flag(argv, "--id"))
            return _ok(argv, {})
        if len(args) >= 3 and args[2] == "create":
            return self._create_named(argv, args[1])
        if len(args) >= 3 and args[2] == "delete":
            resource_id = _flag(argv, "--id")
            self._drop_id(resource_id)
            self._drop_membership(resource_id)
            self._drop_permit(resource_id)
            return _ok(argv, {})
        return subprocess.CompletedProcess(argv, 1, "", f"unexpected {' '.join(args)}")

    def _project_page(self, argv: list[str]) -> subprocess.CompletedProcess[str]:
        index = 0
        if "--page-token" in argv:
            index = int(_flag(argv, "--page-token").removeprefix("page-")) - 1
        page = self.project_pages[index]
        body: dict[str, object] = {"items": page}
        if index + 1 < len(self.project_pages):
            body["next_page_token"] = f"page-{index + 2}"
        return _ok(argv, body if page or "next_page_token" in body else {})

    def _named_list(
        self, argv: list[str], kind: str
    ) -> subprocess.CompletedProcess[str]:
        parent_id = _flag(argv, "--parent-id")
        items = [
            payload
            for (item_kind, item_parent, _name), payload in self.named.items()
            if item_kind == kind and item_parent == parent_id
        ]
        return _ok(argv, {"items": items} if items else {})

    def _create_named(
        self, argv: list[str], kind: str, **spec: object
    ) -> subprocess.CompletedProcess[str]:
        self._seq += 1
        name = _flag(argv, "--name")
        parent_id = _flag(argv, "--parent-id")
        resource_id = f"{kind}-{self._seq}"
        payload = _resource(resource_id, name, **spec)
        self.named[(kind, parent_id, name)] = payload
        return _ok(argv, payload)

    def _drop_id(self, resource_id: str) -> None:
        """Forget a resource id in named lookups and project pages."""
        self.named = {
            key: payload
            for key, payload in self.named.items()
            if _metadata_id(payload) != resource_id
        }
        self.project_pages = [
            [item for item in page if _metadata_id(item) != resource_id]
            for page in self.project_pages
        ]

    def _drop_membership(self, membership_id: str) -> None:
        """Forget one group membership."""
        self.memberships = {
            group_id: [item for item in rows if _metadata_id(item) != membership_id]
            for group_id, rows in self.memberships.items()
        }

    def _drop_permit(self, permit_id: str) -> None:
        """Forget one access permit."""
        self.permits = {
            group_id: [item for item in rows if _metadata_id(item) != permit_id]
            for group_id, rows in self.permits.items()
        }

    def _create_membership(self, argv: list[str]) -> subprocess.CompletedProcess[str]:
        self._seq += 1
        group_id = _flag(argv, "--parent-id")
        member_id = _flag(argv, "--member-id")
        payload = {
            "metadata": {"id": f"membership-{self._seq}"},
            "spec": {"member_id": member_id},
        }
        self.memberships.setdefault(group_id, []).append(payload)
        return _ok(argv, payload)

    def _create_permit(self, argv: list[str]) -> subprocess.CompletedProcess[str]:
        self._seq += 1
        group_id = _flag(argv, "--parent-id")
        role = _flag(argv, "--role")
        resource_id = _flag(argv, "--resource-id")
        payload: dict[str, object] = {
            "metadata": {"id": f"permit-{self._seq}"},
            "spec": {"role": role, "resource_id": resource_id},
        }
        self.permits.setdefault(group_id, []).append(payload)
        return _ok(argv, payload)


def _creates(cli: _NebiusIamCli) -> list[list[str]]:
    """Return CLI invocations that create a resource."""
    return [argv for argv in cli.calls if "create" in argv]


def _adopted_cli() -> _NebiusIamCli:
    """Return a tenant that already has the eu-north1 and eu-west1 IAM."""
    cli = _NebiusIamCli()
    cli.project_pages = [
        [_resource(DEFAULT_NORTH, "default-project-eu-north1", region="eu-north1")],
        [
            _resource(
                "project-default-west",
                "default-project-eu-west1",
                region="eu-west1",
            )
        ],
    ]
    cli.add(
        "service-account",
        DEFAULT_NORTH,
        PROVISIONER_SERVICE_ACCOUNT,
        PROVISIONER_SA,
    )
    cli.add("group", TENANT_ID, PROVISIONERS_GROUP, PROVISIONERS)
    cli.add_membership(PROVISIONERS, PROVISIONER_SA, "groupmembership-provisioner")
    cli.add("project", TENANT_ID, "arena-eu-north1", NORTH_PROJECT, region="eu-north1")
    cli.add("group", NORTH_PROJECT, TUNNEL_GROUP, NORTH_TUNNEL_GROUP)
    cli.add("service-account", NORTH_PROJECT, TUNNEL_SERVICE_ACCOUNT, NORTH_TUNNEL_SA)
    cli.add_membership(NORTH_TUNNEL_GROUP, NORTH_TUNNEL_SA, "groupmembership-north")
    cli.add_permit(
        NORTH_TUNNEL_GROUP, TUNNEL_ROLE, NORTH_PROJECT, "permit-north-tunnel"
    )
    cli.add_permit(PROVISIONERS, PROVISIONER_ROLE, NORTH_PROJECT, "permit-north-admin")
    west = "project-e01mjqekpa009q4bqfat7c"
    cli.add("project", TENANT_ID, "arena-eu-west1", west, region="eu-west1")
    west_group = "group-west-tunnel"
    west_sa = "serviceaccount-west-tunnel"
    cli.add("group", west, TUNNEL_GROUP, west_group)
    cli.add("service-account", west, TUNNEL_SERVICE_ACCOUNT, west_sa)
    cli.add_membership(west_group, west_sa, "groupmembership-west")
    cli.add_permit(west_group, TUNNEL_ROLE, west, "permit-west-tunnel")
    cli.add_permit(PROVISIONERS, PROVISIONER_ROLE, west, "permit-west-admin")
    return cli


@pytest.fixture
def nebius_cli(monkeypatch: pytest.MonkeyPatch) -> None:
    """Point the Nebius runner at the test executable name."""
    monkeypatch.setattr(
        "agilerl.arena.byoc.provisioning.providers.nebius.terraform.resolve_kubeconfig_executable",
        lambda _names: "nebius",
    )


class TestEnsureEphemeralRegions:
    def test_adopts_an_existing_region_without_creating(self, nebius_cli: None) -> None:
        # Arrange
        cli = _adopted_cli()

        # Act
        ensure_ephemeral_regions(_spec(region="eu-north1"), run=cli)

        # Assert
        assert _creates(cli) == []
        assert not any("delete" in argv or "destroy" in argv for argv in cli.calls)
        looked_up = [
            _flag(argv, "--name")
            for argv in cli.calls
            if "get-by-name" in argv and "project" in argv
        ]
        assert "arena-eu-north1" in looked_up
        assert "arena-eu-west1" in looked_up

    def test_creates_a_region_that_only_has_a_default_project(
        self, nebius_cli: None
    ) -> None:
        # Arrange
        cli = _NebiusIamCli()
        cli.parent_id = "project-cli"
        cli.project_pages = [
            [
                _resource(
                    "project-default-west",
                    "default-project-eu-west1",
                    region="eu-west1",
                )
            ]
        ]
        cli.add(
            "service-account",
            "project-default-west",
            PROVISIONER_SERVICE_ACCOUNT,
            PROVISIONER_SA,
        )

        # Act
        ensure_ephemeral_regions(_spec(), run=cli)
        created = _creates(cli)
        ensure_ephemeral_regions(_spec(), run=cli)

        # Assert
        assert _creates(cli) == created
        assert not any("delete" in argv or "destroy" in argv for argv in cli.calls)
        assert [
            _flag(argv, "--name")
            for argv in created
            if argv[1:4] == ["iam", "v2", "project"]
        ] == ["arena-eu-west1"]
        project_create = next(argv for argv in created if "v2" in argv)
        assert _flag(project_create, "--region") == "eu-west1"
        assert _flag(project_create, "--parent-id") == TENANT_ID
        created_names = [_flag(argv, "--name") for argv in created if "--name" in argv]
        assert PROVISIONERS_GROUP in created_names
        assert TUNNEL_GROUP in created_names
        assert TUNNEL_SERVICE_ACCOUNT in created_names
        roles = [
            _flag(argv, "--role")
            for argv in created
            if argv[1:3] == ["iam", "access-permit"]
        ]
        assert roles == [TUNNEL_ROLE, PROVISIONER_ROLE]

    def test_includes_the_spec_region_when_no_default_project_has_it(
        self, nebius_cli: None
    ) -> None:
        # Arrange
        cli = _NebiusIamCli()
        cli.add(
            "service-account",
            cli.parent_id,
            PROVISIONER_SERVICE_ACCOUNT,
            PROVISIONER_SA,
        )

        # Act
        ensure_ephemeral_regions(_spec(region="eu-south1"), run=cli)

        # Assert
        assert [
            _flag(argv, "--name")
            for argv in _creates(cli)
            if argv[1:4] == ["iam", "v2", "project"]
        ] == ["arena-eu-south1"]

    def test_delete_all_removes_arena_region_projects(self, nebius_cli: None) -> None:
        # Arrange
        cli = _adopted_cli()
        cli.project_pages[0].append(
            _resource(NORTH_PROJECT, "arena-eu-north1", region="eu-north1")
        )
        cli.project_pages[1].append(
            _resource(
                "project-e01mjqekpa009q4bqfat7c",
                "arena-eu-west1",
                region="eu-west1",
            )
        )
        cli.project_pages[0].append(
            _resource(
                "project-cluster",
                "arena-dev-nebius-global",
                region="eu-north1",
            )
        )

        # Act
        delete_ephemeral_regions(_spec(region="eu-north1"), run=cli)

        # Assert
        deleted = [
            _flag(argv, "--id")
            for argv in cli.calls
            if argv[1:5] == ["iam", "v2", "project", "delete"]
        ]
        assert deleted == [NORTH_PROJECT, "project-e01mjqekpa009q4bqfat7c"]
        assert any(
            argv[1:4] == ["iam", "group", "delete"]
            and _flag(argv, "--id") == PROVISIONERS
            for argv in cli.calls
        )
        assert not any(
            argv[1:3] == ["iam", "service-account"] and "delete" in argv
            for argv in cli.calls
        )
        group_delete = next(
            index
            for index, argv in enumerate(cli.calls)
            if argv[1:4] == ["iam", "group", "delete"]
        )
        project_delete = next(
            index
            for index, argv in enumerate(cli.calls)
            if argv[1:5] == ["iam", "v2", "project", "delete"]
        )
        assert group_delete < project_delete

    def test_skips_in_cluster_placement(self, nebius_cli: None) -> None:
        cli = _NebiusIamCli()

        ensure_ephemeral_regions(_spec(placement="in-cluster"), run=cli)

        assert cli.calls == []

    def test_requires_the_provisioner_service_account(self, nebius_cli: None) -> None:
        cli = _NebiusIamCli()
        cli.project_pages = [
            [_resource(DEFAULT_NORTH, "default-project-eu-north1", region="eu-north1")]
        ]

        with pytest.raises(click.ClickException, match=PROVISIONER_SERVICE_ACCOUNT):
            ensure_ephemeral_regions(_spec(region="eu-north1"), run=cli)

        assert _creates(cli) == []

    def test_requires_a_region(self, nebius_cli: None) -> None:
        cli = _NebiusIamCli()

        with pytest.raises(click.ClickException, match="no ephemeral region"):
            ensure_ephemeral_regions(_spec(), run=cli)

    def test_skips_non_default_and_duplicate_regions(self, nebius_cli: None) -> None:
        cli = _NebiusIamCli()
        cli.project_pages = [
            [
                _resource("project-other", "cluster-project", region="eu-north1"),
                _resource(
                    DEFAULT_NORTH, "default-project-eu-north1", region="eu-north1"
                ),
                _resource(
                    "project-default-dup",
                    "default-project-eu-north1-copy",
                    region="eu-north1",
                ),
                _resource("project-no-region", "default-project-blank"),
            ]
        ]
        cli.add(
            "service-account",
            DEFAULT_NORTH,
            PROVISIONER_SERVICE_ACCOUNT,
            PROVISIONER_SA,
        )

        ensure_ephemeral_regions(_spec(region="eu-north1"), run=cli)

        created_projects = [
            _flag(argv, "--name")
            for argv in _creates(cli)
            if argv[1:4] == ["iam", "v2", "project"]
        ]
        assert created_projects == ["arena-eu-north1"]

    def test_requires_a_cli_parent_id(self, nebius_cli: None) -> None:
        cli = _adopted_cli()
        cli.parent_id = ""

        with pytest.raises(click.ClickException, match="no parent-id"):
            ensure_ephemeral_regions(_spec(region="eu-north1"), run=cli)

    def test_skips_permits_without_a_spec(self, nebius_cli: None) -> None:
        cli = _adopted_cli()
        cli.permits.setdefault(PROVISIONERS, []).insert(
            0, {"metadata": {"id": "permit-broken"}}
        )

        ensure_ephemeral_regions(_spec(region="eu-north1"), run=cli)

        assert not any(
            argv[1:4] == ["iam", "access-permit"] and "create" in argv
            for argv in cli.calls
        )


class TestEphemeralHooks:
    def test_post_apply_ensures_regions(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        seen: list[str] = []
        monkeypatch.setattr(
            "agilerl.arena.byoc.provisioning.providers.nebius.provider.ensure_ephemeral_regions",
            lambda spec: seen.append(spec.values.placement),
        )
        monkeypatch.setattr(
            "agilerl.arena.byoc.provisioning.providers.nebius.provider.ensure_storage_secret",
            lambda **_kwargs: None,
        )
        monkeypatch.setattr(
            "agilerl.arena.byoc.provisioning.providers.nebius.provider.ensure_gpu_runtime_class",
            lambda **_kwargs: None,
        )
        monkeypatch.setattr(
            "agilerl.arena.byoc.provisioning.providers.nebius.provider.configure_gateway_api",
            lambda **_kwargs: [{"name": "arena", "namespace": "arena"}],
        )

        NEBIUS.post_apply(_spec(), _outputs(tmp_path))

        assert seen == ["ephemeral"]

    def test_register_uses_the_registration_placement(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen: list[str] = []
        monkeypatch.setattr(
            "agilerl.arena.byoc.commands.ensure_ephemeral_regions",
            lambda spec: seen.append(spec.values.placement),
        )

        _ensure_registered_ephemeral_regions(
            _spec(placement="in-cluster"),
            {"placement": "ephemeral"},
        )
        _ensure_registered_ephemeral_regions(
            _spec(placement="ephemeral"),
            {"placement": "in-cluster"},
        )

        assert seen == ["ephemeral"]
