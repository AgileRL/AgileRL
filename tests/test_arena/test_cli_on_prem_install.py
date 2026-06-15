"""Tests for on-prem install helpers (bundle validation, env)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click import ClickException

from agilerl.arena.cli_on_prem_install import (
    _run_docker_swarm_teardown,
    _ssh_connection_target,
    _swarm_script_env,
    _validate_tun0_conf,
    _validate_wireguard_bundle,
    _verify_swarm_stack,
)


def test_ssh_connection_target_uses_host_alias_without_ssh_user(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SSH_USER", raising=False)
    assert _ssh_connection_target("op-ray-head", None) == "op-ray-head"
    assert _ssh_connection_target("ubuntu@op-ray-head", None) == "ubuntu@op-ray-head"
    assert _ssh_connection_target("op-ray-head", "deploy") == "deploy@op-ray-head"
    assert _ssh_connection_target("op-ray-head:2222", None) == "op-ray-head"


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


def test_verify_swarm_stack_quotes_stack_name() -> None:
    """A malicious stack name must not break out of the remote shell command."""
    with patch("agilerl.arena.cli_on_prem_install._ssh_remote_command") as ssh_mock:
        _verify_swarm_stack(
            "manager-host",
            "arena; rm -rf /",
            ssh_user=None,
            ssh_extra_opts=None,
        )
    remote_cmd = ssh_mock.call_args.args[1]
    # The injected command is neutralized by shlex.quote (single-quoted, no bare ;).
    assert "'arena; rm -rf /'" in remote_cmd
    assert "; rm -rf / --format" not in remote_cmd


def test_docker_swarm_teardown_quotes_stack_name() -> None:
    """``docker stack rm`` must escape a caller-supplied stack name."""
    with patch("agilerl.arena.cli_on_prem_install._ssh_remote_command") as ssh_mock:
        _run_docker_swarm_teardown(
            manager="manager-host",
            workers=(),
            stack_name="arena$(whoami)",
            ssh_user=None,
            ssh_extra_opts=None,
            leave_swarm=False,
        )
    remote_cmd = ssh_mock.call_args.args[1]
    assert remote_cmd == "sudo docker stack rm 'arena$(whoami)'"
