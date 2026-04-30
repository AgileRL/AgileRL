"""Tests for on-prem install helpers (bundle validation, env)."""

from __future__ import annotations

from pathlib import Path

import pytest
from click import ClickException

from agilerl.arena.cli_on_prem_install import (
    _swarm_script_env,
    _validate_tun0_conf,
    _validate_wireguard_bundle,
)


def test_swarm_script_env_sets_docker_reboot_assume_yes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("DOCKER_REBOOT_ASSUME_YES", raising=False)
    env = _swarm_script_env({"FOO": "bar"})
    assert env["DOCKER_REBOOT_ASSUME_YES"] == "1"
    assert env["FOO"] == "bar"


def test_validate_tun0_conf_accepts_complete_file(tmp_path: Path) -> None:
    conf = tmp_path / "tun0.conf"
    conf.write_text(
        """
[Interface]
Address = 172.24.0.50/32
PrivateKey = abc

[Peer]
PublicKey = def
PresharedKey = ghi
Endpoint = gw.example.com:51820
AllowedIPs = 10.0.0.0/8
""",
        encoding="utf-8",
    )
    _validate_tun0_conf(conf)


def test_validate_tun0_conf_rejects_incomplete_file(tmp_path: Path) -> None:
    conf = tmp_path / "tun0.conf"
    conf.write_text("[Interface]\n", encoding="utf-8")
    with pytest.raises(ClickException, match="Invalid tun0.conf"):
        _validate_tun0_conf(conf)


def test_validate_wireguard_bundle_swarm(tmp_path: Path) -> None:
    root = tmp_path / "bundle"
    (root / "config.d").mkdir(parents=True)
    (root / "config.d" / "tun0.conf").write_text(
        "[Interface]\n[Peer]\nPrivateKey = x\nPublicKey = y\n"
        "PresharedKey = z\nEndpoint = h:1\nAllowedIPs = 0.0.0.0/0\n",
        encoding="utf-8",
    )
    (root / "arena-stack.yaml").write_text(
        "volumes:\n  - ./config.d:/etc/wireguard/\n",
        encoding="utf-8",
    )
    _validate_wireguard_bundle(root, "dockerSwarm")


def test_validate_wireguard_bundle_helm(tmp_path: Path) -> None:
    root = tmp_path / "bundle"
    chart = root / "chart"
    chart.mkdir(parents=True)
    (chart / "values.yaml").write_text(
        """
wireguard:
  gatewayHost: "gw"
  gatewayPublicKey: "pub"
  peerPrivateKey: "priv"
  peerIp: "172.24.0.1"
  preSharedKey: "psk"
  allowedIps: []
""",
        encoding="utf-8",
    )
    _validate_wireguard_bundle(root, "helm")
