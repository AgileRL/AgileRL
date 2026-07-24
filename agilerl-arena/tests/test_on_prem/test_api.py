# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Tests for OnPremApi and its pure helpers."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from agilerl.arena.exceptions import ArenaAPIError
from agilerl.arena.on_prem import OnPremApi
from agilerl.arena.on_prem.api import class_by_name


class TestClassByName:
    def test_returns_single_match(self) -> None:
        classes = [{"name": "a", "id": 1}, {"name": "b", "id": 2}]
        assert class_by_name(classes, "b") == {"name": "b", "id": 2}

    @pytest.mark.parametrize("classes", [[], [{"name": "other"}], "not-a-list", None])
    def test_returns_none_when_absent(self, classes: object) -> None:
        assert class_by_name(classes, "missing") is None

    def test_rejects_duplicates(self) -> None:
        classes = [{"name": "dup"}, {"name": "dup"}]
        with pytest.raises(ArenaAPIError, match="Multiple on-prem classes"):
            class_by_name(classes, "dup")


class TestOnPremApi:
    def test_enable_disable_invoke_endpoints(
        self, on_prem_api: OnPremApi, mock_client: MagicMock
    ) -> None:
        on_prem_api.enable()
        on_prem_api.disable()
        paths = [
            c.args[0]["path"]
            for c in mock_client._invoke_manifest_command.call_args_list
        ]
        assert paths == [
            "/api/cli/v1/on-prem/enable",
            "/api/cli/v1/on-prem/disable",
        ]

    def test_find_class_uses_list(
        self, on_prem_api: OnPremApi, mock_client: MagicMock
    ) -> None:
        mock_client._invoke_manifest_command.return_value = [{"name": "pool", "id": 7}]
        assert on_prem_api.find_class("pool") == {"name": "pool", "id": 7}

    def test_delete_class_skips_when_absent(
        self, on_prem_api: OnPremApi, mock_client: MagicMock
    ) -> None:
        mock_client._invoke_manifest_command.return_value = []
        on_prem_api.delete_class("pool")
        mock_client._invoke_manifest_command.assert_called_once()  # only the list

    def test_delete_class_deletes_when_present(
        self, on_prem_api: OnPremApi, mock_client: MagicMock
    ) -> None:
        mock_client._invoke_manifest_command.side_effect = [
            [{"name": "pool", "id": 1}],
            {},
        ]
        on_prem_api.delete_class("pool")
        delete = mock_client._invoke_manifest_command.call_args_list[1]
        assert delete.args[0]["path"].endswith("/classes/delete")
        assert delete.args[1] == {"name": "pool"}

    def test_fetch_bundle_returns_bytes_and_passes_query(
        self, on_prem_api: OnPremApi, mock_client: MagicMock
    ) -> None:
        mock_client._invoke_manifest_command.return_value = (
            b"zip-bytes",
            "application/zip",
            None,
        )
        data = on_prem_api.fetch_bundle("pool", "helm")
        assert data == b"zip-bytes"
        _invoke, query = mock_client._invoke_manifest_command.call_args.args
        assert query == {"name": "pool", "setupType": "helm", "archivedType": "zip"}

    def test_register_cluster_enterprise_body(
        self, on_prem_api: OnPremApi, mock_client: MagicMock
    ) -> None:
        mock_client._invoke_manifest_command.return_value = {"clusterId": 1}
        on_prem_api.register_cluster(
            name="prod-cluster",
            install_profile="enterprise",
            storage_endpoint="http://s3.example.com",
            storage_bucket="arena-prod",
            storage_secret_name="corp-s3",
            preprocessing_resource_class_id=9,
        )
        _invoke, body = mock_client._invoke_manifest_command.call_args.args
        assert body == {
            "name": "prod-cluster",
            "installProfile": "enterprise",
            "storageEndpoint": "http://s3.example.com",
            "storageBucket": "arena-prod",
            "storageSecretName": "corp-s3",
            "preprocessingResourceClassId": 9,
        }
