# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Tests for ByocApi and its pure helpers."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from agilerl.arena.byoc import ByocApi
from agilerl.arena.byoc.api import class_by_name, cluster_by_name
from agilerl.arena.exceptions import ArenaAPIError


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
        )

        invoke, body = mock_client._invoke_manifest_command.call_args.args
        assert invoke["path"] == "/api/cli/v1/byoc/classes/create"
        assert body["num_nodes"] == 1
        assert body["clusterName"] == "prod-cluster"
        assert "onPremCluster" not in body
        assert body["nodeSelector"] == {
            "nebius.com/node-group-id": "mk8snodegroup-e00bwd6prsz2dynk6s"
        }
        assert created["name"] == "arena-nebius-gpu"

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

    def test_rotate_cluster_token_sends_json_name_body(
        self, byoc_api: ByocApi, mock_client: MagicMock
    ) -> None:
        mock_client._invoke_manifest_command.return_value = {"token": "rotated.tok"}

        result = byoc_api.rotate_cluster_token("prod-cluster")

        invoke, body = mock_client._invoke_manifest_command.call_args.args
        assert result == {"token": "rotated.tok"}
        assert invoke["method"] == "POST"
        assert invoke["path"] == "/api/cli/v1/byoc/clusters/rotate-token"
        assert body == {"name": "prod-cluster"}

    def test_rotate_cluster_token_rejects_non_dict(
        self, byoc_api: ByocApi, mock_client: MagicMock
    ) -> None:
        mock_client._invoke_manifest_command.return_value = ["token"]

        with pytest.raises(ArenaAPIError, match="Unexpected cluster token rotation"):
            byoc_api.rotate_cluster_token("prod-cluster")


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
        mock_client._request.assert_called_once_with("GET", "/api/on-prem-clusters")

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
