"""Tests for SshTarget parsing and SshExecutor dispatch."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from click import ClickException

from agilerl.arena.on_prem.ssh import SshExecutor, SshTarget


class TestSshTarget:
    @pytest.mark.parametrize(
        ("host", "expected_host", "expected_port"),
        [
            ("host", "host", None),
            ("user@host", "host", None),
            ("host:2222", "host", 2222),
            ("user@host:2222", "host", 2222),
            ("[::1]:22", "::1", 22),
            ("[::1]", "::1", None),
        ],
    )
    def test_hostname_and_port(
        self, host: str, expected_host: str, expected_port: int | None
    ) -> None:
        target = SshTarget.parse(host)
        assert target.hostname == expected_host
        assert target.port == expected_port

    @pytest.mark.parametrize("host", ["localhost", "127.0.0.1", "[::1]"])
    def test_loopback_is_local(self, host: str) -> None:
        assert SshTarget.parse(host).is_local is True

    def test_remote_is_not_local(self) -> None:
        assert SshTarget.parse("some-remote-box.example.com").is_local is False

    def test_is_local_false_when_hostname_lookup_fails(self) -> None:
        # A non-loopback host whose local hostname lookup raises OSError must
        # safely resolve to "not local" rather than propagating the error.
        with patch(
            "agilerl.arena.on_prem.ssh.socket.gethostname",
            side_effect=OSError("no host"),
        ):
            assert SshTarget.parse("some-remote-box.example.com").is_local is False

    def test_connection_target_uses_host_alias_without_user(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("SSH_USER", raising=False)
        assert SshTarget.parse("op-ray-head").connection_target(None) == "op-ray-head"
        assert (
            SshTarget.parse("ubuntu@op-ray-head").connection_target(None)
            == "ubuntu@op-ray-head"
        )
        assert (
            SshTarget.parse("op-ray-head").connection_target("deploy")
            == "deploy@op-ray-head"
        )
        assert (
            SshTarget.parse("op-ray-head:2222").connection_target(None) == "op-ray-head"
        )


class TestSshExecutor:
    def test_runs_locally_without_ssh(self) -> None:
        completed = MagicMock(returncode=0)
        with patch(
            "agilerl.arena.on_prem.ssh.subprocess.run", return_value=completed
        ) as run_mock:
            SshExecutor().run("localhost", "echo hi")
        assert run_mock.call_args.args[0] == ["bash", "-lc", "echo hi"]

    def test_builds_ssh_invocation(self) -> None:
        completed = MagicMock(returncode=0)
        with (
            patch(
                "agilerl.arena.on_prem.ssh.shutil.which", return_value="/usr/bin/ssh"
            ),
            patch(
                "agilerl.arena.on_prem.ssh.subprocess.run", return_value=completed
            ) as run_mock,
        ):
            SshExecutor(ssh_user="ubuntu", ssh_extra_opts="-v").run(
                "host:2222", "echo hi"
            )
        cmd = run_mock.call_args.args[0]
        assert cmd[0] == "ssh"
        assert "-p" in cmd and "2222" in cmd
        assert "-v" in cmd  # extra opts split in
        assert "ubuntu@host" in cmd
        assert cmd[-1] == "echo hi"  # remote command is a single argv entry

    def test_requires_ssh_on_path_for_remote(self) -> None:
        with (
            patch("agilerl.arena.on_prem.ssh.shutil.which", return_value=None),
            pytest.raises(ClickException, match="ssh not found"),
        ):
            SshExecutor().run("remote-box", "echo hi")

    def test_capture_returns_stdout(self) -> None:
        completed = MagicMock(returncode=0, stdout="captured")
        with patch(
            "agilerl.arena.on_prem.ssh.subprocess.run", return_value=completed
        ) as run_mock:
            out = SshExecutor().run("localhost", "echo hi", capture=True)
        assert out == "captured"
        assert run_mock.call_args.kwargs["stdout"] is not None

    def test_nonzero_warns_by_default(self) -> None:
        completed = MagicMock(returncode=3)
        with (
            patch("agilerl.arena.on_prem.ssh.subprocess.run", return_value=completed),
            patch("agilerl.arena.on_prem.ssh.logger") as log,
        ):
            SshExecutor().run("localhost", "false")  # must not raise
        log.warning.assert_called_once()

    def test_nonzero_raises_when_check(self) -> None:
        completed = MagicMock(returncode=3)
        with (
            patch("agilerl.arena.on_prem.ssh.subprocess.run", return_value=completed),
            pytest.raises(ClickException, match="exited 3"),
        ):
            SshExecutor().run("localhost", "false", check=True)
