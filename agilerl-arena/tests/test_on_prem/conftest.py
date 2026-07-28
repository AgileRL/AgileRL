# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Shared fixtures for the on-prem subpackage tests."""

from __future__ import annotations

import io
import zipfile
from collections.abc import Callable
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from agilerl.arena.client import ArenaClient
from agilerl.arena.config import CommandConfig
from agilerl.arena.on_prem import OnPremApi


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
def on_prem_api(mock_client: MagicMock) -> OnPremApi:
    """An :class:`OnPremApi` wrapping the ``mock_client`` fixture."""
    return OnPremApi(mock_client)


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


_VALID_TUN0 = (
    "[Interface]\n[Peer]\nPrivateKey = x\nPublicKey = y\n"
    "PresharedKey = z\nEndpoint = h:1\nAllowedIPs = 0.0.0.0/0\n"
)
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


@pytest.fixture
def swarm_bundle(tmp_path: Path) -> Path:
    """A directory shaped like a valid extracted dockerSwarm bundle."""
    root = tmp_path / "bundle"
    (root / "config.d").mkdir(parents=True)
    (root / "config.d" / "tun0.conf").write_text(_VALID_TUN0, encoding="utf-8")
    (root / "arena-stack.yaml").write_text(
        "volumes:\n  - ./config.d:/etc/wireguard/\n", encoding="utf-8"
    )
    for script in (
        "install-docker.sh",
        "install-nvidia-driver.sh",
        "install-nvidia-container-toolkit.sh",
        "init-docker-swarm.sh",
        "join-docker-swarm.sh",
        "label-docker-swarm-gpus.sh",
        "deploy-arena-stack.sh",
    ):
        (root / script).write_text("#!/bin/sh\n", encoding="utf-8")
    return root


@pytest.fixture
def helm_bundle(tmp_path: Path) -> Path:
    """A directory shaped like a valid extracted helm bundle."""
    root = tmp_path / "bundle"
    (root / "chart").mkdir(parents=True)
    (root / "chart" / "values.yaml").write_text(_VALID_HELM_VALUES, encoding="utf-8")
    (root / "setup.sh").write_text("#!/bin/sh\n", encoding="utf-8")
    return root
