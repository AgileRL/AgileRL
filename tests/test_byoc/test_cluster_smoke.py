# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Tests for ``arena cluster smoke``."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import click
import httpx
import pytest
from click.testing import CliRunner

from agilerl.byoc import build_cluster_smoke_command
from agilerl.byoc.cluster_smoke import run_cluster_smoke
from agilerl.client import ArenaClient, ManifestInvoke
from agilerl.config import CommandConfig

from .conftest import FAKE_CLUSTER_UID

SMOKE_MODULE = "agilerl.byoc.cluster_smoke"
NAMESPACE = "arena-ent-train-0a1b2c3d-smoke-3"


def _smoke(status: str, error: str | None = None) -> dict[str, Any]:
    return {
        "id": 3,
        "cluster": "corp-k8s",
        "status": status,
        "namespace": NAMESPACE,
        "error": error,
    }


class FakeArena:
    """Answers the cluster list and smoke routes; reads return *statuses* in turn."""

    def __init__(
        self, statuses: list[dict[str, Any]], cluster_uid: str = FAKE_CLUSTER_UID
    ) -> None:
        self.statuses = statuses
        self.cluster_uid = cluster_uid
        self.calls: list[tuple[str, str, dict[str, object]]] = []

    def __call__(self, invoke: ManifestInvoke, args: Mapping[str, object]) -> object:
        self.calls.append((invoke["method"], invoke["path"], dict(args)))
        if invoke["path"].endswith("/clusters/list"):
            return [{"name": "corp-k8s", "clusterUid": self.cluster_uid}]
        if invoke["method"] == "POST":
            return _smoke("Requested")
        return self.statuses.pop(0)


@pytest.fixture(autouse=True)
def pinned_target(fake_kube_target: Callable[..., MagicMock]) -> MagicMock:
    """Point every smoke at a fixed context."""
    return fake_kube_target(SMOKE_MODULE)


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    """Poll without waiting."""
    monkeypatch.setattr(f"{SMOKE_MODULE}.time.sleep", lambda _seconds: None)


def _run(arena: FakeArena, mock_client: MagicMock, timeout: float = 60.0) -> None:
    mock_client._invoke_manifest_command.side_effect = arena
    run_cluster_smoke(
        mock_client,
        name="corp-k8s",
        kubeconfig=None,
        context="corp",
        platform=False,
        timeout=timeout,
    )


def test_a_passing_smoke_says_the_release_came_up_and_was_removed(
    mock_client: MagicMock, capsys: pytest.CaptureFixture[str]
) -> None:
    arena = FakeArena([_smoke("Deploying"), _smoke("Passed")])

    _run(arena, mock_client)

    out = capsys.readouterr().out
    assert f"deploys Helm release {NAMESPACE}" in out
    assert "Waiting for the agent to pick up the smoke" in out
    assert "Waiting for its Ray head to be Ready" in out
    assert "Smoke passed on cluster 'corp-k8s'" in out
    requested = [call for call in arena.calls if call[0] == "POST"]
    assert requested == [
        (
            "POST",
            "/api/cli/v1/byoc/clusters/smoke",
            {"name": "corp-k8s", "clusterUid": FAKE_CLUSTER_UID},
        )
    ]
    reads = [call for call in arena.calls if call[0] == "GET" and "smoke" in call[1]]
    assert reads[-1][2] == {"name": "corp-k8s", "id": 3}


def test_a_failed_smoke_fails_the_command_with_the_agents_reason(
    mock_client: MagicMock,
) -> None:
    reason = "install the smoke release: roles is forbidden"
    arena = FakeArena([_smoke("Failed", reason)])

    with pytest.raises(click.ClickException) as raised:
        _run(arena, mock_client)

    assert raised.value.message == f"Smoke failed on cluster 'corp-k8s': {reason}"


def test_a_failed_smoke_without_a_reason_says_so(mock_client: MagicMock) -> None:
    arena = FakeArena([_smoke("Failed")])

    with pytest.raises(click.ClickException, match="no reason given"):
        _run(arena, mock_client)


def test_a_smoke_with_no_result_in_time_says_how_to_keep_waiting(
    mock_client: MagicMock,
) -> None:
    arena = FakeArena([_smoke("Deploying")])

    with pytest.raises(click.ClickException) as raised:
        _run(arena, mock_client, timeout=0.0)

    assert raised.value.message == (
        "Smoke 3 on cluster 'corp-k8s' is still Requested after 0s. "
        "Re-run this command to keep waiting for it."
    )


def test_an_unknown_status_is_shown_as_it_is(
    mock_client: MagicMock, capsys: pytest.CaptureFixture[str]
) -> None:
    arena = FakeArena([_smoke("Queued"), _smoke("Passed")])

    _run(arena, mock_client)

    assert "Smoke is Queued…" in capsys.readouterr().out


def test_another_kubernetes_cluster_starts_no_smoke(mock_client: MagicMock) -> None:
    arena = FakeArena([], cluster_uid="11111111-2222-3333-4444-555555555555")

    with pytest.raises(click.ClickException, match="is not Arena cluster 'corp-k8s'"):
        _run(arena, mock_client)

    assert all(call[0] == "GET" for call in arena.calls)


def test_command_passes_its_flags(
    command_config: CommandConfig, tmp_path: Path
) -> None:
    kubeconfig = tmp_path / "kubeconfig"
    kubeconfig.write_text("apiVersion: v1\nkind: Config\n", encoding="utf-8")

    with patch("agilerl.byoc.commands.run_cluster_smoke") as smoke:
        result = CliRunner().invoke(
            build_cluster_smoke_command(),
            [
                "--name",
                " corp-k8s ",
                "--kubeconfig",
                str(kubeconfig),
                "--context",
                "corp",
                "--timeout",
                "90",
                "--platform",
            ],
            obj=command_config,
        )

    assert result.exit_code == 0, result.output
    assert smoke.call_args.kwargs == {
        "name": "corp-k8s",
        "kubeconfig": kubeconfig,
        "context": "corp",
        "platform": True,
        "timeout": 90.0,
    }


def test_command_refuses_a_blank_name(command_config: CommandConfig) -> None:
    result = CliRunner().invoke(
        build_cluster_smoke_command(), ["--name", " "], obj=command_config
    )

    assert result.exit_code == 1
    assert "--name is required." in result.output


def test_command_shows_arenas_refusal(
    command_config: CommandConfig,
    http_client: ArenaClient,
    capsys: pytest.CaptureFixture[str],
) -> None:
    refusal = 'The agent of cluster "corp-k8s" has not reported that it runs smokes.'

    def answer(method: str, path: str, **_kwargs: object) -> httpx.Response:
        if path.endswith("/clusters/list"):
            return httpx.Response(200, json=[{"name": "corp-k8s"}])
        return httpx.Response(
            400, json={"error": "bad_request", "description": refusal}
        )

    http_client._http.request.side_effect = answer

    with patch("agilerl.config.build_client", return_value=http_client):
        result = CliRunner().invoke(
            build_cluster_smoke_command(),
            ["--name", "corp-k8s", "--context", "corp"],
            obj=command_config,
        )

    assert result.exit_code == 1
    printed = " ".join((result.output + capsys.readouterr().err).split())
    assert f"API error (400): {refusal}" in printed
