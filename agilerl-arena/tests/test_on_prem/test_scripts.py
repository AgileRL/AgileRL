"""Tests for BundleScriptRunner, shell resolution, and stage helpers."""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from agilerl.arena.on_prem.scripts import (
    BundleScriptRunner,
    StageFailed,
    _shell_runner,
    stage_failure,
    swarm_script_env,
)
from click import ClickException


class TestSwarmScriptEnv:
    def test_sets_docker_reboot_assume_yes(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("DOCKER_REBOOT_ASSUME_YES", raising=False)
        env = swarm_script_env({"FOO": "bar"})
        assert env["DOCKER_REBOOT_ASSUME_YES"] == "1"
        assert env["FOO"] == "bar"


class TestShellRunner:
    @pytest.mark.parametrize(
        ("which_map", "expected"),
        [
            ({"bash": "/usr/bin/bash", "sh": "/bin/sh"}, "/usr/bin/bash"),
            ({"sh": "/bin/sh"}, "/bin/sh"),  # falls back when bash absent
        ],
    )
    def test_resolves_interpreter(
        self, which_map: dict[str, str], expected: str
    ) -> None:
        with patch(
            "agilerl.arena.on_prem.scripts.shutil.which",
            side_effect=lambda name: which_map.get(name),
        ):
            assert _shell_runner() == expected

    def test_raises_when_no_shell(self) -> None:
        with (
            patch("agilerl.arena.on_prem.scripts.shutil.which", return_value=None),
            pytest.raises(ClickException, match="bash or sh not found"),
        ):
            _shell_runner()


class TestStageFailure:
    def test_formats_stage_named_message(self) -> None:
        exc = StageFailed(
            "install-docker.sh", 2, "install-docker.sh: daemon not running"
        )
        err = stage_failure("Installing Docker", "manager", exc, index=1, total=7)
        message = str(err)
        assert "Stage 1/7" in message
        assert "Installing Docker" in message
        assert "install-docker.sh: daemon not running" in message


class TestBundleScriptRunner:
    def test_missing_file_raises(self, tmp_path: Path) -> None:
        runner = BundleScriptRunner(tmp_path, env={})
        with pytest.raises(ClickException, match="missing script"):
            runner.run("absent.sh", [])

    def test_invokes_runner_with_args(self, tmp_path: Path) -> None:
        script = tmp_path / "go.sh"
        script.write_text("#!/bin/sh\n", encoding="utf-8")
        completed = MagicMock(returncode=0, stdout="")
        with (
            patch(
                "agilerl.arena.on_prem.scripts._shell_runner",
                return_value="/bin/bash",
            ),
            patch(
                "agilerl.arena.on_prem.scripts.subprocess.run",
                return_value=completed,
            ) as run_mock,
        ):
            BundleScriptRunner(tmp_path, env={"K": "V"}).run("go.sh", ["a1", "a2"])
        assert run_mock.call_args.args[0] == [
            "/bin/bash",
            str(script.resolve()),
            "a1",
            "a2",
        ]
        # Quiet (default): capture stdout, leave stderr attached, don't raise.
        assert run_mock.call_args.kwargs["check"] is False
        assert run_mock.call_args.kwargs["stdout"] == subprocess.PIPE
        assert run_mock.call_args.kwargs["cwd"] == tmp_path
        assert run_mock.call_args.kwargs["env"] == {"K": "V"}

    def test_raises_stage_failed_with_captured_output(self, tmp_path: Path) -> None:
        script = tmp_path / "go.sh"
        script.write_text("#!/bin/sh\n", encoding="utf-8")
        completed = MagicMock(returncode=2, stdout="script failed: command not found")
        with (
            patch(
                "agilerl.arena.on_prem.scripts._shell_runner",
                return_value="/bin/bash",
            ),
            patch(
                "agilerl.arena.on_prem.scripts.subprocess.run",
                return_value=completed,
            ),
            pytest.raises(StageFailed) as excinfo,
        ):
            BundleScriptRunner(tmp_path, env={}).run("go.sh", [])
        assert excinfo.value.returncode == 2
        assert excinfo.value.output == "script failed: command not found"

    def test_streams_live_when_verbose(self, tmp_path: Path) -> None:
        script = tmp_path / "go.sh"
        script.write_text("#!/bin/sh\n", encoding="utf-8")
        completed = MagicMock(returncode=0)
        with (
            patch(
                "agilerl.arena.on_prem.scripts._shell_runner",
                return_value="/bin/bash",
            ),
            patch(
                "agilerl.arena.on_prem.scripts.subprocess.run",
                return_value=completed,
            ) as run_mock,
            patch("agilerl.arena.on_prem.scripts.logger") as log,
        ):
            log.isEnabledFor.return_value = True  # DEBUG / --verbose
            BundleScriptRunner(tmp_path, env={}).run("go.sh", [])
        assert "stdout" not in run_mock.call_args.kwargs  # no capture when streaming
