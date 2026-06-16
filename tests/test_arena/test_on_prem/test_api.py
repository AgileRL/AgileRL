"""Tests for OnPremApi and its pure helpers."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from agilerl.arena.exceptions import ArenaAPIError
from agilerl.arena.on_prem import OnPremApi
from agilerl.arena.on_prem.api import class_by_name, resolve_num_nodes


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


class TestResolveNumNodes:
    @pytest.mark.parametrize(
        ("existing", "explicit", "default", "expected"),
        [
            ({"num_nodes": 5}, None, 9, 5),  # existing wins
            (None, 3, 9, 3),  # explicit next
            (None, None, 9, 9),  # falls back to default
            ({"num_nodes": 0}, 4, 9, 4),  # invalid existing ignored
            ({"num_nodes": "x"}, None, 9, 9),  # non-int existing ignored
        ],
    )
    def test_precedence(
        self,
        existing: dict[str, object] | None,
        explicit: int | None,
        default: int,
        expected: int,
    ) -> None:
        assert (
            resolve_num_nodes(existing, explicit=explicit, default=default) == expected
        )


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

    def test_ensure_class_reuses_existing(
        self, on_prem_api: OnPremApi, mock_client: MagicMock
    ) -> None:
        mock_client._invoke_manifest_command.return_value = [{"name": "pool", "id": 9}]
        row = on_prem_api.ensure_class("pool", num_nodes=2)
        assert row["id"] == 9
        # Only the list call; no create.
        mock_client._invoke_manifest_command.assert_called_once()

    def test_ensure_class_creates_when_absent(
        self, on_prem_api: OnPremApi, mock_client: MagicMock
    ) -> None:
        mock_client._invoke_manifest_command.side_effect = [[], {"name": "p", "id": 7}]
        row = on_prem_api.ensure_class("p", num_nodes=2)
        assert row["id"] == 7
        create = mock_client._invoke_manifest_command.call_args_list[1]
        assert create.args[0]["path"].endswith("/classes/create")
        assert create.args[1]["num_nodes"] == 2

    def test_ensure_class_rejects_non_object_create_response(
        self, on_prem_api: OnPremApi, mock_client: MagicMock
    ) -> None:
        mock_client._invoke_manifest_command.side_effect = [[], "oops"]
        with pytest.raises(ArenaAPIError, match="not an object"):
            on_prem_api.ensure_class("p", num_nodes=1)

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
