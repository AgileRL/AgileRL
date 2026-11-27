# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Shared fixtures for the BYOC subpackage tests."""

from __future__ import annotations

import io
import zipfile
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import pytest

from agilerl.byoc import ByocApi
from agilerl.byoc.kube_target import KubeTarget
from agilerl.client import ArenaClient
from agilerl.config import CommandConfig

FAKE_CLUSTER_UID = "0f6b2c1e-8d4a-4b7e-9c3f-5a2d1e0b7c64"


@pytest.fixture
def fake_kube_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Callable[..., MagicMock]:
    """Return an installer that swaps ``kube_target`` in a module for a fixed target.

    The installer takes the module path and an optional cluster UID and returns
    the mock, called as ``kube_target(name, kubeconfig, context)``.
    """

    def install(module: str, cluster_uid: str = FAKE_CLUSTER_UID) -> MagicMock:
        target = KubeTarget(
            kubeconfig=tmp_path / "pinned-kubeconfig",
            context="corp",
            server="https://k8s.example:6443",
            cluster_uid=cluster_uid,
        )

        @contextmanager
        def resolve(
            _name: str, _kubeconfig: Path | None, _context: str | None
        ) -> Iterator[KubeTarget]:
            yield target

        mock = MagicMock(side_effect=resolve)
        monkeypatch.setattr(f"{module}.kube_target", mock)
        return mock

    return install


@pytest.fixture
def command_config() -> CommandConfig:
    """A minimal :class:`CommandConfig` for CliRunner invocations."""
    return CommandConfig(
        api_key="test-key",
        base_url=None,
        keycloak_url=None,
        realm=None,
        client_id=None,
        request_timeout=30,
        upload_timeout=300,
    )


@pytest.fixture
def mock_client() -> MagicMock:
    """A spec'd :class:`ArenaClient` mock (only real methods are callable)."""
    return MagicMock(spec=ArenaClient)


@pytest.fixture
def byoc_api(mock_client: MagicMock) -> ByocApi:
    """A :class:`ByocApi` wrapping the ``mock_client`` fixture."""
    return ByocApi(mock_client)


def _arena_response(method: str, path: str, **_kwargs: object) -> httpx.Response:
    """Answer a BYOC CLI route the way Arena does for a cluster named ``arena``."""
    if path.endswith(("/install-package", "/deployment-setup")):
        return httpx.Response(200, content=b"pkg")
    if path.endswith("/classes/list"):
        return httpx.Response(200, json=[{"name": "pool", "id": 1}])
    if path.endswith("/clusters/list"):
        return httpx.Response(200, json=[{"name": "arena"}])
    if path.endswith("/install-artifacts"):
        return httpx.Response(200, json={"agentHelmValuesYaml": "image: {}\n"})
    return httpx.Response(200, json={"name": "arena", "token": "rotated.tok"})


@pytest.fixture
def http_client() -> ArenaClient:
    """A real :class:`ArenaClient` whose HTTP transport answers like Arena."""
    with patch("agilerl.auth.KeycloakOpenID"):
        client = ArenaClient(api_key="test-key")
    client._http.request = MagicMock(side_effect=_arena_response)
    return client


@pytest.fixture
def sent_scope() -> Callable[[ArenaClient, str], dict[str, object]]:
    """Return a lookup of where the last request to a route put ``byocScope``."""

    def _sent(client: ArenaClient, route: str) -> dict[str, object]:
        calls = [
            c for c in client._http.request.call_args_list if c.args[1].endswith(route)
        ]
        assert calls, f"no request to {route}"
        kwargs = calls[-1].kwargs
        return {
            "query": (kwargs.get("params") or {}).get("byocScope"),
            "body": (kwargs.get("json") or {}).get("byocScope"),
        }

    return _sent


@pytest.fixture
def client_context() -> Callable[[MagicMock], MagicMock]:
    """Return a factory building a context manager that yields a client mock."""

    def _ctx(client: MagicMock) -> MagicMock:
        cm = MagicMock()
        cm.__enter__.return_value = client
        cm.__exit__.return_value = False
        return cm

    return _ctx


@pytest.fixture
def make_zip() -> Callable[[dict[str, str]], bytes]:
    """Return a factory that zips ``{path: text}`` into in-memory bytes."""

    def _make(files: dict[str, str]) -> bytes:
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as zf:
            for name, content in files.items():
                zf.writestr(name, content)
        return buffer.getvalue()

    return _make


_VALID_HELM_VALUES = """
clusterName: "pool"
wireguard:
  gatewayHost: "gw"
  gatewayPublicKey: "pub"
  peerPrivateKey: "priv"
  peerIp: "172.24.0.1"
  preSharedKey: "psk"
  allowedIps: []
"""

_VALID_ENTERPRISE_HELM_VALUES = """
clusterName: "arena-train-42"
wireguard:
  enabled: false
nfs:
  enabled: false
"""


@pytest.fixture
def helm_bundle(tmp_path: Path) -> Path:
    """A directory shaped like a valid extracted helm bundle."""
    root = tmp_path / "bundle"
    (root / "chart").mkdir(parents=True)
    (root / "chart" / "values.yaml").write_text(_VALID_HELM_VALUES, encoding="utf-8")
    (root / "setup.sh").write_text("#!/bin/sh\n", encoding="utf-8")
    return root


@pytest.fixture
def enterprise_helm_bundle(tmp_path: Path) -> Path:
    """Enterprise Ray Helm bundle: WireGuard disabled on the head pod."""
    root = tmp_path / "bundle"
    (root / "chart").mkdir(parents=True)
    (root / "chart" / "values.yaml").write_text(
        _VALID_ENTERPRISE_HELM_VALUES, encoding="utf-8"
    )
    (root / "setup.sh").write_text("#!/bin/sh\n", encoding="utf-8")
    return root
