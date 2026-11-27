# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Tests for rotating and revoking a cluster's agent credential."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from agilerl.byoc.api import ByocApi
from agilerl.byoc.cluster_token import (
    run_cluster_revoke_token,
    run_cluster_rotate_token,
)
from agilerl.client import ArenaClient

CREDENTIAL = {"kind": "agent", "renewal_requested_at": "2026-11-27T03:00:00Z"}


class TestRunClusterRotateToken:
    def test_asks_the_agent_to_renew_and_writes_nothing_to_the_cluster(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        client = MagicMock(spec=ArenaClient)
        with patch.object(
            ByocApi, "renew_cluster_credential", return_value=CREDENTIAL
        ) as renew:
            result = run_cluster_rotate_token(client, name="prod")

        assert result == CREDENTIAL
        renew.assert_called_once_with("prod")
        out = capsys.readouterr().out
        assert "renew its credential" in out
        assert "keeps working" in out

    def test_platform_scope(self) -> None:
        client = MagicMock(spec=ArenaClient)
        client._invoke_manifest_command.return_value = CREDENTIAL

        run_cluster_rotate_token(client, name="nebius-eu", platform=True)

        _invoke, body = client._invoke_manifest_command.call_args.args
        assert body == {"name": "nebius-eu", "byocScope": "platform"}


class TestRunClusterRevokeToken:
    def test_revokes_and_names_the_reconnect_command(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        client = MagicMock(spec=ArenaClient)
        with patch.object(
            ByocApi, "revoke_cluster_credentials", return_value={"kind": None}
        ) as revoke:
            result = run_cluster_revoke_token(client, name="prod")

        assert result == {"kind": None}
        revoke.assert_called_once_with("prod")
        assert "arena cluster connect --name prod" in capsys.readouterr().out

    def test_platform_revoke_points_at_the_staff_page(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        client = MagicMock(spec=ArenaClient)
        with patch.object(
            ByocApi, "revoke_cluster_credentials", return_value={"kind": None}
        ):
            run_cluster_revoke_token(client, name="nebius-eu", platform=True)

        out = capsys.readouterr().out
        assert "Platform clusters staff page" in out
        assert "arena cluster connect" not in out
