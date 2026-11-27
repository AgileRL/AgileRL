# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Tests for ByocApi and its pure helpers."""

from __future__ import annotations

from collections.abc import Callable
from unittest.mock import MagicMock

import httpx
import pytest

from agilerl.byoc import ByocApi
from agilerl.byoc.api import class_by_name, cluster_by_name
from agilerl.client import ArenaClient
from agilerl.exceptions import ArenaAPIError

CLUSTER_AND_CLASS_CALLS: list[tuple[Callable[[ByocApi], object], str, str]] = [
    (lambda api: api.list_clusters(), "/clusters/list", "query"),
    (lambda api: api.unregister_cluster("arena"), "/clusters/unregister", "body"),
    (
        lambda api: api.renew_cluster_credential("arena"),
        "/clusters/credential/renew",
        "body",
    ),
    (
        lambda api: api.revoke_cluster_credentials("arena"),
        "/clusters/credential/revoke",
        "body",
    ),
    (
        lambda api: api.fetch_cluster_install_artifacts("arena"),
        "/clusters/install-artifacts",
        "body",
    ),
    (
        lambda api: api.download_cluster_install_package("arena", "image: {}\n"),
        "/clusters/install-package",
        "body",
    ),
    (lambda api: api.list_classes(), "/classes/list", "query"),
    (
        lambda api: api.create_class("pool", 1, "arena", {}, {}),
        "/classes/create",
        "body",
    ),
    (lambda api: api.update_class("pool", 1, {}, {}), "/classes/update", "query"),
    (lambda api: api.delete_class("pool"), "/classes/delete", "query"),
    (
        lambda api: api.fetch_bundle("pool", "helm"),
        "/classes/deployment-setup",
        "query",
    ),
]


class TestClassByName:
    def test_returns_single_match(self) -> None:
        classes = [{"name": "a", "id": 1}, {"name": "b", "id": 2}]
        assert class_by_name(classes, "b") == {"name": "b", "id": 2}

    @pytest.mark.parametrize("classes", [[], [{"name": "other"}], "not-a-list", None])
    def test_returns_none_when_absent(self, classes: object) -> None:
        assert class_by_name(classes, "missing") is None

    def test_rejects_duplicates(self) -> None:
        classes = [{"name": "dup"}, {"name": "dup"}]
        with pytest.raises(ArenaAPIError, match="Multiple BYOC classes"):
            class_by_name(classes, "dup")


class TestClusterByName:
    def test_returns_single_match(self) -> None:
        clusters = [{"name": "a"}, {"name": "prod"}]

        assert cluster_by_name(clusters, "prod") == {"name": "prod"}

    def test_rejects_duplicates(self) -> None:
        clusters = [{"name": "dup"}, {"name": "dup"}]

        with pytest.raises(ArenaAPIError, match="Multiple BYOC clusters"):
            cluster_by_name(clusters, "dup")


class TestByocApi:
    def test_enable_disable_invoke_endpoints(
        self, byoc_api: ByocApi, mock_client: MagicMock
    ) -> None:
        byoc_api.enable()
        byoc_api.disable()
        paths = [
            c.args[0]["path"]
            for c in mock_client._invoke_manifest_command.call_args_list
        ]
        assert paths == [
            "/api/cli/v1/byoc/enable",
            "/api/cli/v1/byoc/disable",
        ]

    def test_find_class_uses_list(
        self, byoc_api: ByocApi, mock_client: MagicMock
    ) -> None:
        mock_client._invoke_manifest_command.return_value = [{"name": "pool", "id": 7}]
        assert byoc_api.find_class("pool") == {"name": "pool", "id": 7}

    def test_load_classes_lists_once(
        self, byoc_api: ByocApi, mock_client: MagicMock
    ) -> None:
        mock_client._invoke_manifest_command.return_value = [{"name": "pool", "id": 7}]

        byoc_api.load_classes()

        assert byoc_api.find_class("pool") == {"name": "pool", "id": 7}
        assert byoc_api.find_class("other") is None
        mock_client._invoke_manifest_command.assert_called_once()

    def test_delete_class_skips_when_absent(
        self, byoc_api: ByocApi, mock_client: MagicMock
    ) -> None:
        mock_client._invoke_manifest_command.return_value = []
        byoc_api.delete_class("pool")
        mock_client._invoke_manifest_command.assert_called_once()  # only the list

    def test_delete_class_deletes_when_present(
        self, byoc_api: ByocApi, mock_client: MagicMock
    ) -> None:
        mock_client._invoke_manifest_command.side_effect = [
            [{"name": "pool", "id": 1}],
            {},
        ]
        byoc_api.delete_class("pool")
        delete = mock_client._invoke_manifest_command.call_args_list[1]
        assert delete.args[0]["path"].endswith("/classes/delete")
        assert delete.args[1] == {"name": "pool"}

    def test_delete_without_cluster_rejects_duplicate_names(
        self, byoc_api: ByocApi, mock_client: MagicMock
    ) -> None:
        mock_client._invoke_manifest_command.return_value = [
            {"name": "pool"},
            {"name": "pool"},
        ]

        with pytest.raises(ArenaAPIError, match="Multiple BYOC classes"):
            byoc_api.delete_class("pool")

        mock_client._invoke_manifest_command.assert_called_once()

    def test_fetch_bundle_returns_bytes_and_passes_query(
        self, byoc_api: ByocApi, mock_client: MagicMock
    ) -> None:
        mock_client._invoke_manifest_command.return_value = (
            b"zip-bytes",
            "application/zip",
            None,
        )
        data = byoc_api.fetch_bundle("pool", "helm")
        assert data == b"zip-bytes"
        _invoke, query = mock_client._invoke_manifest_command.call_args.args
        assert query == {"name": "pool", "setupType": "helm", "archivedType": "zip"}

    def test_register_cluster_enterprise_body(
        self, byoc_api: ByocApi, mock_client: MagicMock
    ) -> None:
        mock_client._invoke_manifest_command.return_value = {
            "cluster": {"name": "prod-cluster"}
        }
        byoc_api.register_cluster(
            name="prod-cluster",
            storage_endpoint="http://s3.example.com",
            storage_bucket="arena-prod",
            storage_secret_name="corp-s3",
            preprocessing_resource_class="cpu-pool",
        )
        _invoke, body = mock_client._invoke_manifest_command.call_args.args
        assert body == {
            "name": "prod-cluster",
            "storageEndpoint": "http://s3.example.com",
            "storageBucket": "arena-prod",
            "storageSecretName": "corp-s3",
            "preprocessingResourceClass": "cpu-pool",
        }

    def test_register_cluster_sends_byoc_provider(
        self, byoc_api: ByocApi, mock_client: MagicMock
    ) -> None:
        mock_client._invoke_manifest_command.return_value = {
            "cluster": {"name": "prod-cluster"}
        }
        byoc_api.register_cluster(
            name="prod-cluster",
            byoc_provider={
                "provider": "nebius",
                "config": {"tenant_id": "tenant-1", "region": "eu-north1"},
            },
        )
        _invoke, body = mock_client._invoke_manifest_command.call_args.args
        assert body["byocProvider"] == {
            "provider": "nebius",
            "config": {"tenant_id": "tenant-1", "region": "eu-north1"},
        }

    def test_register_cluster_sends_placement(
        self, byoc_api: ByocApi, mock_client: MagicMock
    ) -> None:
        mock_client._invoke_manifest_command.return_value = {
            "cluster": {"name": "prod-cluster"}
        }
        byoc_api.register_cluster(name="prod-cluster", placement="ephemeral")
        _invoke, body = mock_client._invoke_manifest_command.call_args.args
        assert body["placement"] == "byoc_ephemeral"

    def test_register_cluster_sends_byoc_scope(
        self, byoc_api: ByocApi, mock_client: MagicMock
    ) -> None:
        mock_client._invoke_manifest_command.return_value = {
            "cluster": {"name": "prod-cluster"}
        }

        byoc_api.register_cluster(name="prod-cluster", byoc_scope="platform")

        _invoke, body = mock_client._invoke_manifest_command.call_args.args
        assert body["byocScope"] == "platform"

    def test_register_cluster_rejects_unknown_scope(
        self, byoc_api: ByocApi, mock_client: MagicMock
    ) -> None:
        with pytest.raises(ValueError, match="customer or platform"):
            byoc_api.register_cluster(name="prod-cluster", byoc_scope="shared")

    def test_register_cluster_omits_byoc_provider_when_unset(
        self, byoc_api: ByocApi, mock_client: MagicMock
    ) -> None:
        mock_client._invoke_manifest_command.return_value = {
            "cluster": {"name": "prod-cluster"}
        }
        byoc_api.register_cluster(name="prod-cluster")
        _invoke, body = mock_client._invoke_manifest_command.call_args.args
        assert "byocProvider" not in body

    def test_create_class_sends_cluster_and_node_selector(
        self, byoc_api: ByocApi, mock_client: MagicMock
    ) -> None:
        mock_client._invoke_manifest_command.return_value = {"name": "arena-nebius-gpu"}

        created = byoc_api.create_class(
            name="arena-nebius-gpu",
            num_nodes=1,
            cluster_name="prod-cluster",
            node_selector={
                "nebius.com/node-group-id": "mk8snodegroup-e00bwd6prsz2dynk6s"
            },
            metadata={
                "computeResource": {
                    "numCpus": 16,
                    "numGpus": 1,
                    "memoryBytes": "200 GiB",
                }
            },
            description="1 GPU worker",
        )

        invoke, body = mock_client._invoke_manifest_command.call_args.args
        assert invoke["path"] == "/api/cli/v1/byoc/classes/create"
        assert body["num_nodes"] == 1
        assert body["clusterName"] == "prod-cluster"
        assert "onPremCluster" not in body
        assert body["nodeSelector"] == {
            "nebius.com/node-group-id": "mk8snodegroup-e00bwd6prsz2dynk6s"
        }
        assert body["description"] == "1 GPU worker"
        assert created["name"] == "arena-nebius-gpu"

    def test_update_class_sends_the_class_fields(
        self, byoc_api: ByocApi, mock_client: MagicMock
    ) -> None:
        mock_client._invoke_manifest_command.return_value = {"name": "arena-nebius-gpu"}

        updated = byoc_api.update_class(
            name="arena-nebius-gpu",
            num_nodes=4,
            node_selector={"nebius.com/node-group-id": "ng-1"},
            metadata={"computeResource": {"numCpus": 16}},
            description="NVIDIA® L40S Intel 1 GPU - 16 CPUs - 64 GB RAM.",
        )

        invoke, body = mock_client._invoke_manifest_command.call_args.args
        assert invoke["method"] == "PATCH"
        assert invoke["path"] == "/api/cli/v1/byoc/classes/update"
        assert body == {
            "name": "arena-nebius-gpu",
            "num_nodes": 4,
            "node_selector": {"nebius.com/node-group-id": "ng-1"},
            "metadata": {"computeResource": {"numCpus": 16}},
            "description": "NVIDIA® L40S Intel 1 GPU - 16 CPUs - 64 GB RAM.",
        }
        assert updated["name"] == "arena-nebius-gpu"

    def test_create_class_sends_region_availability(
        self, byoc_api: ByocApi, mock_client: MagicMock
    ) -> None:
        mock_client._invoke_manifest_command.return_value = {"name": "arena-nebius-gpu"}
        availabilities = [
            {"region": "eu-north1", "zones": ["fabric-2"], "enabled": True}
        ]

        byoc_api.create_class(
            name="arena-nebius-gpu",
            num_nodes=1,
            cluster_name="prod-cluster",
            node_selector={"nebius.com/platform": "gpu-l40s-a"},
            metadata={"computeResource": {"numCpus": 16}},
            availabilities=availabilities,
        )

        _invoke, body = mock_client._invoke_manifest_command.call_args.args
        assert body["availabilities"] == availabilities

    def test_create_class_disables_when_no_region_offers_the_preset(
        self, byoc_api: ByocApi, mock_client: MagicMock
    ) -> None:
        mock_client._invoke_manifest_command.return_value = {"name": "gpu"}

        byoc_api.create_class(
            name="gpu",
            num_nodes=1,
            cluster_name="prod-cluster",
            node_selector={"nebius.com/platform": "gpu-l40s-d"},
            metadata={"computeResource": {"numCpus": 32}},
            availabilities=[],
            enabled=False,
        )

        _invoke, body = mock_client._invoke_manifest_command.call_args.args
        assert body["enabled"] is False
        assert body["availabilities"] == []

    def test_update_class_sends_region_availability(
        self, byoc_api: ByocApi, mock_client: MagicMock
    ) -> None:
        mock_client._invoke_manifest_command.return_value = {"name": "arena-nebius-gpu"}
        availabilities = [
            {"region": "eu-north1", "zones": ["fabric-2"], "enabled": True}
        ]

        byoc_api.update_class(
            name="arena-nebius-gpu",
            num_nodes=1,
            node_selector={"nebius.com/platform": "gpu-l40s-a"},
            metadata={"computeResource": {"numCpus": 16}},
            availabilities=availabilities,
        )

        _invoke, body = mock_client._invoke_manifest_command.call_args.args
        assert body["availabilities"] == availabilities

    def test_update_class_sends_enabled(
        self, byoc_api: ByocApi, mock_client: MagicMock
    ) -> None:
        mock_client._invoke_manifest_command.return_value = {"name": "gpu"}

        byoc_api.update_class(
            name="gpu",
            num_nodes=1,
            node_selector={"nebius.com/platform": "gpu-l40s-d"},
            metadata={"computeResource": {"numCpus": 32}},
            enabled=False,
        )

        _invoke, body = mock_client._invoke_manifest_command.call_args.args
        assert body["enabled"] is False

    def test_replace_class_availabilities_sends_only_regions(
        self, byoc_api: ByocApi, mock_client: MagicMock
    ) -> None:
        mock_client._invoke_manifest_command.return_value = {"name": "arena-nebius-gpu"}
        availabilities = [
            {"region": "eu-north1", "zones": ["fabric-2"], "enabled": True}
        ]

        byoc_api.replace_class_availabilities("arena-nebius-gpu", availabilities)

        invoke, body = mock_client._invoke_manifest_command.call_args.args
        assert invoke["path"] == "/api/cli/v1/byoc/classes/update"
        assert body == {
            "name": "arena-nebius-gpu",
            "availabilities": availabilities,
        }

    def test_replace_class_availabilities_sends_enabled(
        self, byoc_api: ByocApi, mock_client: MagicMock
    ) -> None:
        mock_client._invoke_manifest_command.return_value = {"name": "gpu"}

        byoc_api.replace_class_availabilities("gpu", [], enabled=False)

        _invoke, body = mock_client._invoke_manifest_command.call_args.args
        assert body == {
            "name": "gpu",
            "availabilities": [],
            "enabled": False,
        }

    def test_replace_class_availabilities_rejects_non_dict_response(
        self, byoc_api: ByocApi, mock_client: MagicMock
    ) -> None:
        mock_client._invoke_manifest_command.return_value = []

        with pytest.raises(ArenaAPIError, match="Unexpected resource class update"):
            byoc_api.replace_class_availabilities("gpu", [])

    def test_download_install_package_sends_cluster_name(
        self, byoc_api: ByocApi, mock_client: MagicMock
    ) -> None:
        mock_client._invoke_manifest_command.return_value = (
            b"pkg",
            "application/gzip",
            None,
        )

        data = byoc_api.download_cluster_install_package(
            cluster="prod-cluster",
            agent_helm_values_yaml="cluster: {}\n",
        )

        assert data == b"pkg"
        invoke, body = mock_client._invoke_manifest_command.call_args.args
        assert invoke["path"] == "/api/cli/v1/byoc/clusters/install-package"
        assert body == {
            "cluster": "prod-cluster",
            "agentHelmValuesYaml": "cluster: {}\n",
        }

    def test_list_clusters_invokes_list_endpoint(
        self, byoc_api: ByocApi, mock_client: MagicMock
    ) -> None:
        mock_client._invoke_manifest_command.return_value = [{"name": "prod-cluster"}]

        clusters = byoc_api.list_clusters()

        invoke, body = mock_client._invoke_manifest_command.call_args.args
        assert clusters == [{"name": "prod-cluster"}]
        assert invoke["method"] == "GET"
        assert invoke["path"] == "/api/cli/v1/byoc/clusters/list"
        assert body == {}

    def test_find_cluster_uses_list(
        self, byoc_api: ByocApi, mock_client: MagicMock
    ) -> None:
        mock_client._invoke_manifest_command.return_value = [
            {"name": "other"},
            {"name": "prod-cluster"},
        ]

        found = byoc_api.find_cluster("prod-cluster")

        assert found == {"name": "prod-cluster"}

    def test_unregister_cluster_sends_json_name_body(
        self, byoc_api: ByocApi, mock_client: MagicMock
    ) -> None:
        mock_client._invoke_manifest_command.return_value = {"name": "prod-cluster"}

        byoc_api.unregister_cluster("prod-cluster")

        invoke, body = mock_client._invoke_manifest_command.call_args.args
        assert invoke["method"] == "DELETE"
        assert invoke["path"] == "/api/cli/v1/byoc/clusters/unregister"
        assert body == {"name": "prod-cluster"}

    def test_unregister_cluster_sends_stop_runs_and_force(
        self, byoc_api: ByocApi, mock_client: MagicMock
    ) -> None:
        mock_client._invoke_manifest_command.return_value = {"status": "Archived"}

        result = byoc_api.unregister_cluster("prod-cluster", stop_runs=True, force=True)

        _invoke, body = mock_client._invoke_manifest_command.call_args.args
        assert body == {"name": "prod-cluster", "stopRuns": True, "force": True}
        assert result == {"status": "Archived"}

    def test_unregister_cluster_rejects_a_non_object_response(
        self, byoc_api: ByocApi, mock_client: MagicMock
    ) -> None:
        mock_client._invoke_manifest_command.return_value = ["unexpected"]

        with pytest.raises(ArenaAPIError, match="Unexpected cluster unregister"):
            byoc_api.unregister_cluster("prod-cluster")

    @pytest.mark.parametrize(
        ("call", "path"),
        [
            (
                lambda api: api.renew_cluster_credential("prod-cluster"),
                "/api/cli/v1/byoc/clusters/credential/renew",
            ),
            (
                lambda api: api.revoke_cluster_credentials("prod-cluster"),
                "/api/cli/v1/byoc/clusters/credential/revoke",
            ),
        ],
    )
    def test_credential_calls_send_the_name(
        self,
        byoc_api: ByocApi,
        mock_client: MagicMock,
        call: Callable[[ByocApi], dict[str, object]],
        path: str,
    ) -> None:
        mock_client._invoke_manifest_command.return_value = {"kind": "agent"}

        result = call(byoc_api)

        invoke, body = mock_client._invoke_manifest_command.call_args.args
        assert result == {"kind": "agent"}
        assert invoke["method"] == "POST"
        assert invoke["path"] == path
        assert body == {"name": "prod-cluster"}

    def test_credential_call_rejects_non_dict(
        self, byoc_api: ByocApi, mock_client: MagicMock
    ) -> None:
        mock_client._invoke_manifest_command.return_value = ["token"]

        with pytest.raises(ArenaAPIError, match="Unexpected cluster credential"):
            byoc_api.renew_cluster_credential("prod-cluster")

    def test_request_cluster_smoke_sends_the_name_and_cluster_uid(
        self, byoc_api: ByocApi, mock_client: MagicMock
    ) -> None:
        mock_client._invoke_manifest_command.return_value = {"id": 3}

        result = byoc_api.request_cluster_smoke("prod-cluster", "uid-1")

        invoke, body = mock_client._invoke_manifest_command.call_args.args
        assert result == {"id": 3}
        assert (invoke["method"], invoke["path"]) == (
            "POST",
            "/api/cli/v1/byoc/clusters/smoke",
        )
        assert body == {"name": "prod-cluster", "clusterUid": "uid-1"}

    def test_cluster_smoke_reads_the_smoke_by_id(
        self, byoc_api: ByocApi, mock_client: MagicMock
    ) -> None:
        mock_client._invoke_manifest_command.return_value = {"status": "Passed"}

        result = byoc_api.cluster_smoke("prod-cluster", 3)

        invoke, args = mock_client._invoke_manifest_command.call_args.args
        assert result == {"status": "Passed"}
        assert (invoke["method"], invoke["path"]) == (
            "GET",
            "/api/cli/v1/byoc/clusters/smoke",
        )
        assert args == {"name": "prod-cluster", "id": 3}

    def test_smoke_calls_reject_a_non_object_response(
        self, byoc_api: ByocApi, mock_client: MagicMock
    ) -> None:
        mock_client._invoke_manifest_command.return_value = ["unexpected"]

        with pytest.raises(ArenaAPIError, match="Unexpected cluster smoke response"):
            byoc_api.request_cluster_smoke("prod-cluster", "uid-1")
        with pytest.raises(ArenaAPIError, match="Unexpected cluster smoke response"):
            byoc_api.cluster_smoke("prod-cluster", 3)


class TestByocApiStoredNebiusConfig:
    def test_returns_config_for_the_named_cluster(
        self, byoc_api: ByocApi, mock_client: MagicMock
    ) -> None:
        mock_client._request.return_value = [
            {
                "name": "other",
                "byoc_provider": {"provider": "nebius", "config": {"project_id": "no"}},
            },
            {
                "name": "spec-cluster",
                "byoc_provider": {
                    "provider": "nebius",
                    "config": {
                        "storage_project_id": "storage-1",
                        "terraform_state": {"bucket": "state"},
                    },
                },
            },
        ]

        config = byoc_api.stored_provider_config("spec-cluster", "nebius")

        assert config == {
            "storage_project_id": "storage-1",
            "terraform_state": {"bucket": "state"},
        }
        mock_client._request.assert_called_once_with("GET", "/api/byoc-clusters")

    def test_reads_a_data_wrapper(
        self, byoc_api: ByocApi, mock_client: MagicMock
    ) -> None:
        mock_client._request.return_value = {
            "data": [
                {
                    "name": "spec-cluster",
                    "byocProvider": {
                        "provider": "nebius",
                        "config": {"project_id": "project-1"},
                    },
                }
            ]
        }

        assert byoc_api.stored_provider_config("spec-cluster", "nebius") == {
            "project_id": "project-1"
        }

    @pytest.mark.parametrize(
        "payload",
        [
            [],
            [{"name": "other", "byoc_provider": None}],
            [
                {
                    "name": "spec-cluster",
                    "byoc_provider": {
                        "provider": "custom",
                        "config": {"project_id": "p"},
                    },
                }
            ],
            {"data": "nope"},
            None,
        ],
    )
    def test_returns_none_when_the_cluster_has_no_nebius_config(
        self, byoc_api: ByocApi, mock_client: MagicMock, payload: object
    ) -> None:
        mock_client._request.return_value = payload

        assert byoc_api.stored_provider_config("spec-cluster", "nebius") is None

    def test_rejects_duplicate_names(
        self, byoc_api: ByocApi, mock_client: MagicMock
    ) -> None:
        row = {
            "name": "dup",
            "byoc_provider": {"provider": "nebius", "config": {"project_id": "p"}},
        }
        mock_client._request.return_value = [row, dict(row)]

        with pytest.raises(ArenaAPIError, match="Multiple BYOC clusters named 'dup'"):
            byoc_api.stored_provider_config("dup", "nebius")

    def test_register_cluster_rejects_non_dict_response(
        self, byoc_api: ByocApi, mock_client: MagicMock
    ) -> None:
        mock_client._invoke_manifest_command.return_value = []

        with pytest.raises(ArenaAPIError, match="Unexpected cluster registration"):
            byoc_api.register_cluster(name="prod-cluster")

    def test_create_class_rejects_non_dict_response(
        self, byoc_api: ByocApi, mock_client: MagicMock
    ) -> None:
        mock_client._invoke_manifest_command.return_value = "created"

        with pytest.raises(ArenaAPIError, match="Unexpected resource class create"):
            byoc_api.create_class(
                name="gpu",
                num_nodes=1,
                cluster_name="prod-cluster",
                node_selector={},
                metadata={},
            )

    def test_update_class_rejects_non_dict_response(
        self, byoc_api: ByocApi, mock_client: MagicMock
    ) -> None:
        mock_client._invoke_manifest_command.return_value = []

        with pytest.raises(ArenaAPIError, match="Unexpected resource class update"):
            byoc_api.update_class(
                name="gpu",
                num_nodes=1,
                node_selector={},
                metadata={},
            )

    def test_stored_nebius_config_requires_config_mapping(
        self, byoc_api: ByocApi, mock_client: MagicMock
    ) -> None:
        mock_client._request.return_value = [
            {
                "name": "spec-cluster",
                "byocProvider": {"provider": "nebius", "config": "not-a-map"},
            }
        ]

        assert byoc_api.stored_provider_config("spec-cluster", "nebius") is None

    def test_download_cluster_install_package_sends_storage_yaml(
        self, byoc_api: ByocApi, mock_client: MagicMock
    ) -> None:
        mock_client._invoke_manifest_command.return_value = (
            b"pkg",
            "application/gzip",
            None,
        )

        data = byoc_api.download_cluster_install_package(
            cluster="prod-cluster",
            agent_helm_values_yaml="clusterName: prod\n",
            storage_helm_values_yaml="bucket: arena-data\n",
        )

        assert data == b"pkg"
        _invoke, body = mock_client._invoke_manifest_command.call_args.args
        assert body["storageHelmValuesYaml"] == "bucket: arena-data\n"


class TestByocApiPlatformScope:
    @pytest.mark.parametrize(("call", "route", "location"), CLUSTER_AND_CLASS_CALLS)
    def test_platform_sends_the_platform_scope(
        self,
        http_client: ArenaClient,
        sent_scope: Callable[[ArenaClient, str], dict[str, object]],
        call: Callable[[ByocApi], object],
        route: str,
        location: str,
    ) -> None:
        # Act
        call(ByocApi(http_client, platform=True))

        # Assert
        sent = sent_scope(http_client, route)
        assert sent[location] == "platform"
        assert list(sent.values()).count("platform") == 1

    @pytest.mark.parametrize(("call", "route", "location"), CLUSTER_AND_CLASS_CALLS)
    def test_customer_calls_send_no_scope(
        self,
        http_client: ArenaClient,
        sent_scope: Callable[[ArenaClient, str], dict[str, object]],
        call: Callable[[ByocApi], object],
        route: str,
        location: str,
    ) -> None:
        call(ByocApi(http_client))

        assert sent_scope(http_client, route) == {"query": None, "body": None}

    def test_platform_keeps_provider_routes_unscoped(
        self, http_client: ArenaClient
    ) -> None:
        ByocApi(http_client, platform=True).enable()

        assert http_client._http.request.call_args.kwargs.get("json") is None

    def test_platform_403_names_the_staff_requirement(
        self, http_client: ArenaClient
    ) -> None:
        http_client._http.request.side_effect = None
        http_client._http.request.return_value = httpx.Response(
            403, json={"error": "forbidden"}
        )

        with pytest.raises(
            ArenaAPIError, match="Only AgileRL staff can manage platform clusters"
        ) as raised:
            ByocApi(http_client, platform=True).fetch_cluster_install_artifacts("arena")

        assert raised.value.status_code == 403

    def test_customer_403_keeps_the_arena_message(
        self, http_client: ArenaClient
    ) -> None:
        http_client._http.request.side_effect = None
        http_client._http.request.return_value = httpx.Response(
            403, json={"error": "Not an org admin"}
        )

        with pytest.raises(ArenaAPIError, match="Not an org admin"):
            ByocApi(http_client).unregister_cluster("arena")

    def test_platform_keeps_other_errors(self, http_client: ArenaClient) -> None:
        http_client._http.request.side_effect = None
        http_client._http.request.return_value = httpx.Response(
            404, json={"error": "Cluster not found"}
        )

        with pytest.raises(ArenaAPIError, match="Cluster not found"):
            ByocApi(http_client, platform=True).renew_cluster_credential("arena")


class TestByocApiClusterUid:
    @staticmethod
    def _sent_body(client: ArenaClient, route: str) -> dict[str, object]:
        calls = [
            c for c in client._http.request.call_args_list if c.args[1].endswith(route)
        ]
        assert calls, f"no request to {route}"
        return calls[-1].kwargs["json"]

    def test_register_sends_the_cluster_uid(self, http_client: ArenaClient) -> None:
        ByocApi(http_client).register_cluster(name="arena", cluster_uid="uid-1")

        body = self._sent_body(http_client, "/clusters/register")
        assert body["clusterUid"] == "uid-1"


CLASS_WRITE_CALLS: list[tuple[Callable[..., object], str]] = [
    (
        lambda api, **cluster: api.update_class("pool", 1, {}, {}, **cluster),
        "/classes/update",
    ),
    (
        lambda api, **cluster: api.replace_class_availabilities("pool", [], **cluster),
        "/classes/update",
    ),
    (lambda api, **cluster: api.delete_class("pool", **cluster), "/classes/delete"),
]


class TestByocApiClassCluster:
    @pytest.mark.parametrize(("call", "route"), CLASS_WRITE_CALLS)
    def test_sends_the_cluster_name_and_id_in_the_query(
        self, http_client: ArenaClient, call: Callable[..., object], route: str
    ) -> None:
        # Act
        call(ByocApi(http_client), cluster="arena", cluster_id=7)

        # Assert
        sent = http_client._http.request.call_args.kwargs
        assert http_client._http.request.call_args.args[1].endswith(route)
        assert sent["params"] == {"name": "pool", "cluster": "arena", "cluster_id": 7}
        assert "cluster" not in (sent.get("json") or {})

    @pytest.mark.parametrize(("call", "route"), CLASS_WRITE_CALLS)
    def test_omits_the_cluster_when_not_given(
        self, http_client: ArenaClient, call: Callable[..., object], route: str
    ) -> None:
        call(ByocApi(http_client))

        assert http_client._http.request.call_args.args[1].endswith(route)
        assert http_client._http.request.call_args.kwargs["params"] == {"name": "pool"}

    def test_sends_the_platform_scope_next_to_the_cluster(
        self, http_client: ArenaClient
    ) -> None:
        ByocApi(http_client, platform=True).delete_class("pool", cluster="arena")

        assert http_client._http.request.call_args.kwargs["params"] == {
            "name": "pool",
            "cluster": "arena",
            "byocScope": "platform",
        }

    def test_delete_picks_one_of_two_same_named_classes_by_cluster(
        self, http_client: ArenaClient
    ) -> None:
        # Arrange
        rows = [{"name": "pool", "id": 1}, {"name": "pool", "id": 2}]
        http_client._http.request.side_effect = lambda _method, path, **_kw: (
            httpx.Response(200, json=rows if path.endswith("/list") else {})
        )

        # Act
        ByocApi(http_client).delete_class("pool", cluster="arena")

        # Assert
        delete = http_client._http.request.call_args
        assert delete.args[1].endswith("/classes/delete")
        assert delete.kwargs["params"] == {"name": "pool", "cluster": "arena"}

    def test_delete_with_cluster_skips_the_name_lookup(
        self, byoc_api: ByocApi, mock_client: MagicMock
    ) -> None:
        # Arrange
        mock_client._invoke_manifest_command.return_value = {}

        # Act
        byoc_api.delete_class("pool", cluster="arena")

        # Assert
        mock_client._invoke_manifest_command.assert_called_once()
        invoke, query = mock_client._invoke_manifest_command.call_args.args
        assert invoke["path"].endswith("/classes/delete")
        assert query == {"name": "pool", "cluster": "arena"}

    def test_delete_with_cluster_id_skips_the_name_lookup(
        self, byoc_api: ByocApi, mock_client: MagicMock
    ) -> None:
        mock_client._invoke_manifest_command.return_value = {}

        byoc_api.delete_class("pool", cluster_id=4)

        invoke, query = mock_client._invoke_manifest_command.call_args.args
        assert invoke["path"].endswith("/classes/delete")
        assert query == {"name": "pool", "cluster_id": 4}
        mock_client._invoke_manifest_command.assert_called_once()


class TestByocApiSuggestedClasses:
    def test_gets_the_suggestions_for_the_named_cluster(
        self, byoc_api: ByocApi, mock_client: MagicMock
    ) -> None:
        found = {"suggestions": [], "classes_without_nodes": [], "reported_at": None}
        mock_client._invoke_manifest_command.return_value = found

        result = byoc_api.suggested_classes("corp-k8s")

        invoke, query = mock_client._invoke_manifest_command.call_args.args
        assert result == found
        assert invoke["method"] == "GET"
        assert invoke["path"] == "/api/cli/v1/byoc/clusters/suggested-classes"
        assert query == {"name": "corp-k8s"}

    def test_rejects_a_non_object_response(
        self, byoc_api: ByocApi, mock_client: MagicMock
    ) -> None:
        mock_client._invoke_manifest_command.return_value = []

        with pytest.raises(ArenaAPIError, match="Unexpected class suggestions"):
            byoc_api.suggested_classes("corp-k8s")


class TestByocApiCreateSuggestedClass:
    def test_posts_the_suggestion_without_unset_fields(
        self, byoc_api: ByocApi, mock_client: MagicMock
    ) -> None:
        suggestion = {
            "name": "lab-h100-80-x8",
            "num_nodes": 2,
            "on_prem_cluster_id": 7,
            "node_selector": {"nvidia.com/gpu.product": "NVIDIA-H100-80GB-HBM3"},
            "placement": None,
            "availabilities": None,
        }

        byoc_api.create_suggested_class(suggestion)

        invoke, body = mock_client._invoke_manifest_command.call_args.args
        assert invoke["path"] == "/api/cli/v1/byoc/classes/create"
        assert body == {
            "name": "lab-h100-80-x8",
            "num_nodes": 2,
            "on_prem_cluster_id": 7,
            "node_selector": {"nvidia.com/gpu.product": "NVIDIA-H100-80GB-HBM3"},
        }
