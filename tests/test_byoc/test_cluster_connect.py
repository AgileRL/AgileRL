# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Tests for ``arena cluster connect``: preflight, register, install, wait."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import click
import pytest

from agilerl.byoc.api import ByocApi
from agilerl.byoc.cluster_connect import (
    ConnectOptions,
    context_flag,
    health_checks,
    registered_cluster,
    run_cluster_connect,
    wait_for_agent,
)
from agilerl.byoc.kube_target import KubeTarget
from agilerl.byoc.preflight import CheckStatus, PreflightCheck
from agilerl.config import CommandConfig
from agilerl.exceptions import ArenaAPIError

from .conftest import FAKE_CLUSTER_UID

MODULE = "agilerl.byoc.cluster_connect"
TARGET = KubeTarget(
    kubeconfig=Path("pinned-kubeconfig"),
    context="corp",
    server="https://k8s.example:6443",
    cluster_uid=FAKE_CLUSTER_UID,
)
GREEN = {"name": "tunnel", "status": "green", "detail": "handshake 20s ago"}
RED = {
    "name": "storage",
    "status": "red",
    "detail": "HeadBucket refused",
    "fix": "Check the storage Secret",
}
SUGGESTION = {
    "name": "corp-k8s-h100-80-x8",
    "num_nodes": 2,
    "on_prem_cluster_id": 7,
    "node_selector": {"nvidia.com/gpu.product": "NVIDIA-H100-80GB-HBM3"},
    "metadata": {
        "computeResource": {
            "numCpus": 128,
            "numGpus": 8,
            "memoryBytes": "1.9 TiB",
            "gpu": {"type": "nvidia-h100-80gb", "count": 8, "driverVersion": "default"},
        }
    },
}
FOUND = {
    "reported_at": "2026-11-25T12:00:00Z",
    "suggestions": [SUGGESTION],
    "classes_without_nodes": [],
    "needs_agent_upgrade": False,
}


def row(
    last_poll_at: str | None = None,
    reported_at: str | None = None,
    checks: list[dict[str, str]] | None = None,
    last_error: str | None = None,
) -> dict[str, Any]:
    """Arena's cluster row with a heartbeat at *last_poll_at* and a report at *reported_at*."""
    health = None
    if reported_at is not None:
        health = {"reported_at": reported_at, "checks": checks or [GREEN]}
    return {
        "name": "corp-k8s",
        "connectivity_status": None if last_poll_at is None else "Connected",
        "last_poll_at": last_poll_at,
        "last_error": last_error,
        "health": health,
    }


@pytest.fixture
def api() -> MagicMock:
    return MagicMock(spec=ByocApi)


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    sleep = MagicMock()
    monkeypatch.setattr(f"{MODULE}.time.sleep", sleep)
    return sleep


class TestRegisteredCluster:
    def test_returns_arenas_cluster_in_this_kubernetes_cluster(
        self, api: MagicMock
    ) -> None:
        found = {"name": "corp-k8s", "clusterUid": FAKE_CLUSTER_UID}
        api.find_cluster.return_value = found

        assert registered_cluster(api, "corp-k8s", TARGET) == found

    def test_unregistered_name_is_none(self, api: MagicMock) -> None:
        api.find_cluster.return_value = None

        assert registered_cluster(api, "corp-k8s", TARGET) is None

    def test_org_without_an_enabled_provider_has_no_cluster(
        self, api: MagicMock
    ) -> None:
        api.find_cluster.side_effect = ArenaAPIError("Not Found", status_code=404)

        assert registered_cluster(api, "corp-k8s", TARGET) is None

    def test_other_errors_propagate(self, api: MagicMock) -> None:
        api.find_cluster.side_effect = ArenaAPIError("boom", status_code=500)

        with pytest.raises(ArenaAPIError, match="boom"):
            registered_cluster(api, "corp-k8s", TARGET)

    def test_registered_in_another_cluster_is_refused(self, api: MagicMock) -> None:
        api.find_cluster.return_value = {"name": "corp-k8s", "clusterUid": "other-uid"}

        with pytest.raises(click.ClickException, match="cluster UID other-uid"):
            registered_cluster(api, "corp-k8s", TARGET)


class TestWaitForAgent:
    def test_new_cluster_waits_for_a_heartbeat_then_a_report(
        self, api: MagicMock, capsys: pytest.CaptureFixture[str]
    ) -> None:
        reported = row("t1", "r1")
        api.byoc_cluster.side_effect = [row(), row("t1"), reported]

        cluster = wait_for_agent(api, "corp-k8s", None, 60.0)

        assert cluster == reported
        assert "Agent status: Connected." in capsys.readouterr().out

    def test_reconnect_ignores_the_heartbeat_and_report_from_before(
        self, api: MagicMock
    ) -> None:
        before = row("t0", "r0")
        fresh = row("t1", "r1")
        api.byoc_cluster.side_effect = [row("t0", "r0"), row("t1", "r0"), fresh]

        cluster = wait_for_agent(api, "corp-k8s", before, 60.0)

        assert cluster == fresh

    def test_no_heartbeat_in_time_names_the_last_agent_error(
        self, api: MagicMock
    ) -> None:
        api.byoc_cluster.return_value = row(last_error="dial udp: i/o timeout")

        with pytest.raises(
            click.ClickException,
            match=r"has not reached Arena after 0s.*dial udp: i/o timeout",
        ):
            wait_for_agent(api, "corp-k8s", None, 0.0)

    def test_no_report_in_time_fails(self, api: MagicMock) -> None:
        api.byoc_cluster.return_value = row("t1")

        with pytest.raises(click.ClickException, match="sent no health report"):
            wait_for_agent(api, "corp-k8s", None, 0.0)

    def test_cluster_that_disappears_fails(self, api: MagicMock) -> None:
        api.byoc_cluster.return_value = None

        with pytest.raises(click.ClickException, match="no longer lists"):
            wait_for_agent(api, "corp-k8s", None, 60.0)


class TestHealthChecks:
    def test_maps_report_colours_to_check_results(self) -> None:
        amber = {
            "name": "storage_class",
            "status": "amber",
            "detail": "none",
            "fix": "x",
        }

        checks = health_checks(row("t1", "r1", [GREEN, amber, RED]))

        assert checks == [
            PreflightCheck("tunnel", CheckStatus.PASS, "handshake 20s ago"),
            PreflightCheck("storage_class", CheckStatus.WARN, "none", "x"),
            PreflightCheck(
                "storage",
                CheckStatus.FAIL,
                "HeadBucket refused",
                "Check the storage Secret",
            ),
        ]


def connect_options(**overrides: object) -> ConnectOptions:
    fields: dict[str, Any] = {
        "name": "corp-k8s",
        "kubeconfig": None,
        "kube_context": "corp",
        "agent_namespace": "arena",
        "storage_endpoint": "https://s3.example",
        "storage_bucket": "arena-data",
        "storage_prefix": None,
        "storage_secret_name": "arena-storage",
        "ray_data_storage_class_name": None,
        "rotate_token": False,
        "wait_timeout": 60.0,
        "yes": False,
    }
    fields.update(overrides)
    return ConnectOptions(**fields)


class ConnectRun:
    """Mocks for one ``run_cluster_connect`` call."""

    def __init__(
        self,
        monkeypatch: pytest.MonkeyPatch,
        fake_kube_target: Callable[..., MagicMock],
        client_context: Callable[[MagicMock], MagicMock],
        preflight: list[PreflightCheck],
        rows: list[dict[str, Any] | None],
        helm: bool = True,
        registered: dict[str, Any] | None = None,
    ) -> None:
        self.kube_target = fake_kube_target(MODULE)
        self.preflight = MagicMock(return_value=preflight)
        self.register = MagicMock()
        self.byoc_cluster = MagicMock(side_effect=rows)
        self.suggested = MagicMock(return_value=FOUND)
        self.create_suggested = MagicMock()
        monkeypatch.setattr(f"{MODULE}.helm_available", lambda: helm)
        monkeypatch.setattr(
            f"{MODULE}.arena_client", lambda _config: client_context(MagicMock())
        )
        monkeypatch.setattr(f"{MODULE}.run_preflight", self.preflight)
        monkeypatch.setattr(f"{MODULE}.run_cluster_register", self.register)
        monkeypatch.setattr(ByocApi, "find_cluster", MagicMock(return_value=registered))
        monkeypatch.setattr(
            f"{MODULE}.resolve_install_release", lambda name, _namespace: name
        )
        monkeypatch.setattr(ByocApi, "byoc_cluster", self.byoc_cluster)
        monkeypatch.setattr(ByocApi, "suggested_classes", self.suggested)
        monkeypatch.setattr(ByocApi, "create_suggested_class", self.create_suggested)


PASSING = [PreflightCheck("one_agent", CheckStatus.PASS, "No other Arena agent")]


class TestContextFlag:
    def test_quotes_the_context_for_copy_and_paste(self) -> None:
        assert context_flag("arn:aws:eks:us-east-1:1:cluster/corp") == (
            "--context 'arn:aws:eks:us-east-1:1:cluster/corp'"
        )

    def test_escapes_a_single_quote(self) -> None:
        assert context_flag("it's") == "--context 'it'\"'\"'s'"


class TestRunClusterConnect:
    def test_preflights_registers_installs_and_waits(
        self,
        monkeypatch: pytest.MonkeyPatch,
        fake_kube_target: Callable[..., MagicMock],
        client_context: Callable[[MagicMock], MagicMock],
        command_config: CommandConfig,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        # Arrange
        run = ConnectRun(
            monkeypatch,
            fake_kube_target,
            client_context,
            PASSING,
            [row(), row("t1"), row("t1", "r1")],
        )

        # Act
        run_cluster_connect(command_config, connect_options())

        # Assert
        run.kube_target.assert_called_once_with("corp-k8s", None, "corp")
        kubeconfig, namespace, own_release, storage_class = run.preflight.call_args.args
        assert (namespace, own_release, storage_class) == ("arena", None, None)
        assert kubeconfig.name == "pinned-kubeconfig"
        register = run.register.call_args.kwargs
        assert register["install"] is True
        assert register["no_write"] is True
        assert register["cluster_uid"] == FAKE_CLUSTER_UID
        assert register["rotate_token"] is False
        assert register["storage_bucket"] == "arena-data"
        assert register["placement"] == "in-cluster"
        out = capsys.readouterr().out
        assert "Preflight on --context 'corp':" in out
        assert "corp-k8s-h100-80-x8: 2 node(s)" in out
        assert out.rstrip().endswith(
            "Cluster 'corp-k8s' is connected (--context 'corp')."
        )

    def test_reconnect_waits_for_a_report_after_helm(
        self,
        monkeypatch: pytest.MonkeyPatch,
        fake_kube_target: Callable[..., MagicMock],
        client_context: Callable[[MagicMock], MagicMock],
        command_config: CommandConfig,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        # Arrange
        during_install = row("t-during", "r-during", [RED])
        run = ConnectRun(
            monkeypatch,
            fake_kube_target,
            client_context,
            PASSING,
            [],
            registered={"name": "corp-k8s", "clusterUid": FAKE_CLUSTER_UID},
        )
        installed = False
        reads = 0

        def install(*_args: object, **_kwargs: object) -> None:
            nonlocal installed
            installed = True

        def read(_name: str) -> dict[str, Any]:
            nonlocal reads
            assert installed
            reads += 1
            if reads <= 2:
                return during_install
            if reads == 3:
                return row("t-new", "r-during", [RED])
            return row("t-new", "r-new")

        run.register.side_effect = install
        run.byoc_cluster.side_effect = read

        # Act
        run_cluster_connect(command_config, connect_options())

        # Assert: the report the old pod sent during Helm is the baseline.
        assert reads == 4
        assert run.preflight.call_args.args[2] == "corp-k8s"
        out = capsys.readouterr().out
        assert "handshake 20s ago" in out
        assert "HeadBucket refused" not in out

    def test_failed_preflight_changes_nothing(
        self,
        monkeypatch: pytest.MonkeyPatch,
        fake_kube_target: Callable[..., MagicMock],
        client_context: Callable[[MagicMock], MagicMock],
        command_config: CommandConfig,
    ) -> None:
        failed = [PreflightCheck("one_agent", CheckStatus.FAIL, "Another agent")]
        run = ConnectRun(monkeypatch, fake_kube_target, client_context, failed, [])

        with pytest.raises(click.ClickException, match="Preflight failed: one_agent"):
            run_cluster_connect(command_config, connect_options())

        run.register.assert_not_called()
        run.byoc_cluster.assert_not_called()

    def test_needs_helm_before_touching_anything(
        self,
        monkeypatch: pytest.MonkeyPatch,
        fake_kube_target: Callable[..., MagicMock],
        client_context: Callable[[MagicMock], MagicMock],
        command_config: CommandConfig,
    ) -> None:
        run = ConnectRun(
            monkeypatch, fake_kube_target, client_context, PASSING, [], helm=False
        )

        with pytest.raises(click.ClickException, match="helm is not on PATH"):
            run_cluster_connect(command_config, connect_options())

        run.kube_target.assert_not_called()

    def test_red_health_check_fails_after_printing_the_report(
        self,
        monkeypatch: pytest.MonkeyPatch,
        fake_kube_target: Callable[..., MagicMock],
        client_context: Callable[[MagicMock], MagicMock],
        command_config: CommandConfig,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        run = ConnectRun(
            monkeypatch,
            fake_kube_target,
            client_context,
            PASSING,
            [row(), row("t1"), row("t1", "r1", [GREEN, RED])],
        )

        with pytest.raises(
            click.ClickException,
            match=r"Cluster 'corp-k8s' \(--context 'corp'\) is connected, but",
        ):
            run_cluster_connect(command_config, connect_options(rotate_token=True))

        assert run.register.call_args.kwargs["rotate_token"] is True
        out = capsys.readouterr().out
        assert "HeadBucket refused" in out
        assert "corp-k8s-h100-80-x8: 2 node(s)" in out


class TestRunClusterConnectSuggestedClasses:
    def test_yes_creates_the_suggestions_once_the_agent_has_reported(
        self,
        monkeypatch: pytest.MonkeyPatch,
        fake_kube_target: Callable[..., MagicMock],
        client_context: Callable[[MagicMock], MagicMock],
        command_config: CommandConfig,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        # Arrange
        run = ConnectRun(
            monkeypatch,
            fake_kube_target,
            client_context,
            PASSING,
            [row(), row("t1"), row("t1", "r1")],
        )

        # Act
        run_cluster_connect(command_config, connect_options(yes=True))

        # Assert
        assert run.register.call_args.kwargs["suggest_classes"] is False
        run.suggested.assert_called_once_with("corp-k8s")
        run.create_suggested.assert_called_once_with(SUGGESTION)
        assert (
            "Created resource class 'corp-k8s-h100-80-x8'." in capsys.readouterr().out
        )

    def test_without_yes_or_a_terminal_lists_them_and_creates_nothing(
        self,
        monkeypatch: pytest.MonkeyPatch,
        fake_kube_target: Callable[..., MagicMock],
        client_context: Callable[[MagicMock], MagicMock],
        command_config: CommandConfig,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        run = ConnectRun(
            monkeypatch,
            fake_kube_target,
            client_context,
            PASSING,
            [row(), row("t1"), row("t1", "r1")],
        )
        monkeypatch.setattr("sys.stdin.isatty", lambda: False)

        run_cluster_connect(command_config, connect_options())

        run.create_suggested.assert_not_called()
        out = capsys.readouterr().out
        assert "corp-k8s-h100-80-x8: 2 node(s)" in out
        assert "Re-run with --yes to create them" in out
