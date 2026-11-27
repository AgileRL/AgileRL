# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Upgrade-and-re-run when Arena refuses this agilerl release."""

from __future__ import annotations

import io
import os
import subprocess
import sys
from collections.abc import Callable, Iterator
from unittest.mock import patch

import click
import httpx
import pytest

from agilerl import upgrade
from agilerl.client import ArenaClient
from agilerl.exceptions import ArenaAPIError, ArenaClientUpgradeRequiredError
from agilerl.output import handle_error
from agilerl.upgrade import (
    AUTO_UPGRADE_ENV,
    UPGRADE_RETRIED_ENV,
    installed_extras,
    offer_upgrade,
    upgrade_requirement,
)

DESCRIPTION = (
    "agilerl 0.9.0 is no longer supported; the minimum is 1.0.0. "
    "Run: pip install -U agilerl"
)
SNOWFLAKE_DESCRIPTION = (
    "agilerl 0.9.0 is no longer supported; the minimum is 1.0.0. "
    'Run: pip install -U "agilerl[snowflake]"'
)


class Execed(Exception):
    """Stands in for ``os.execv``, which never returns."""


class TtyInput(io.StringIO):
    def isatty(self) -> bool:
        return True


class Host:
    """Records the pip run and the re-exec instead of performing them."""

    def __init__(self, pip_exit: int) -> None:
        self.pip_exit = pip_exit
        self.reexec_exit = 0
        self.pip_args: list[str] | None = None
        self.reexec_args: list[str] | None = None
        self.execv_args: tuple[str, list[str]] | None = None

    def run(self, args: list[str], check: bool) -> subprocess.CompletedProcess[str]:
        if len(args) > 2 and args[2] == "pip":
            self.pip_args = args
            return subprocess.CompletedProcess(args, self.pip_exit)
        self.reexec_args = args
        return subprocess.CompletedProcess(args, self.reexec_exit)

    def execv(self, path: str, argv: list[str]) -> None:
        self.execv_args = (path, argv)
        raise Execed


@pytest.fixture
def host(monkeypatch: pytest.MonkeyPatch) -> Iterator[Host]:
    recorded = Host(pip_exit=0)
    monkeypatch.setattr(upgrade.subprocess, "run", recorded.run)
    monkeypatch.setattr(upgrade.os, "execv", recorded.execv)
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(upgrade, "installed_extras", list)
    monkeypatch.setattr(sys, "argv", ["arena", "experiments", "list"])
    monkeypatch.delenv(AUTO_UPGRADE_ENV, raising=False)
    monkeypatch.delenv(UPGRADE_RETRIED_ENV, raising=False)
    yield recorded
    # offer_upgrade sets this for the re-run; monkeypatch does not track it.
    os.environ.pop(UPGRADE_RETRIED_ENV, None)


@pytest.fixture
def tty(monkeypatch: pytest.MonkeyPatch) -> Callable[[str], None]:
    def answer(text: str) -> None:
        monkeypatch.setattr(sys, "stdin", TtyInput(text))
        monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
        monkeypatch.setattr(sys.stderr, "isatty", lambda: True)

    return answer


class TestOfferUpgrade:
    def test_yes_upgrades_and_re_runs_the_command_once(
        self,
        host: Host,
        tty: Callable[[str], None],
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        # Arrange
        tty("y\n")

        # Act
        with pytest.raises(Execed):
            offer_upgrade(DESCRIPTION)

        # Assert
        assert host.pip_args == [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-U",
            "agilerl",
        ]
        assert host.reexec_args is None
        assert host.execv_args == (
            sys.executable,
            [sys.executable, "-m", "agilerl.cli", "experiments", "list"],
        )
        assert upgrade.os.environ[UPGRADE_RETRIED_ENV] == "1"
        err = capsys.readouterr().err
        assert DESCRIPTION in err
        assert "Upgrade agilerl now? [Y/n]" in err

    def test_an_empty_answer_means_yes(
        self, host: Host, tty: Callable[[str], None], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        tty("\n")

        with pytest.raises(Execed):
            offer_upgrade(DESCRIPTION)

        assert host.execv_args is not None

    def test_no_exits_without_running_pip(
        self, host: Host, tty: Callable[[str], None]
    ) -> None:
        # Arrange
        tty("n\n")

        # Act
        with pytest.raises(click.exceptions.Exit) as raised:
            offer_upgrade(DESCRIPTION)

        # Assert
        assert raised.value.exit_code == 1
        assert host.pip_args is None
        assert host.execv_args is None

    def test_a_pip_failure_prints_the_manual_command_and_exits(
        self,
        host: Host,
        tty: Callable[[str], None],
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        # Arrange
        host.pip_exit = 1
        tty("y\n")

        # Act
        with pytest.raises(click.exceptions.Exit) as raised:
            offer_upgrade(DESCRIPTION)

        # Assert
        assert raised.value.exit_code == 1
        assert host.execv_args is None
        assert UPGRADE_RETRIED_ENV not in upgrade.os.environ
        assert capsys.readouterr().err.endswith(
            'The upgrade failed. Run: pip install -U "agilerl"\n'
        )

    def test_without_a_tty_nothing_is_asked_or_run(
        self, host: Host, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # Act
        offer_upgrade(DESCRIPTION)

        # Assert
        assert host.pip_args is None
        assert capsys.readouterr().err == ""

    def test_redirected_stderr_does_not_prompt(
        self, host: Host, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange
        monkeypatch.setattr(sys, "stdin", TtyInput("y\n"))
        monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
        monkeypatch.setattr(sys.stderr, "isatty", lambda: False)

        # Act
        offer_upgrade(DESCRIPTION)

        # Assert
        assert host.pip_args is None
        assert host.execv_args is None
        assert host.reexec_args is None

    def test_windows_waits_for_the_re_run_and_exits_with_its_status(
        self,
        host: Host,
        tty: Callable[[str], None],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Arrange
        monkeypatch.setattr(sys, "platform", "win32")
        host.reexec_exit = 4
        tty("y\n")

        # Act
        with pytest.raises(SystemExit) as raised:
            offer_upgrade(DESCRIPTION)

        # Assert
        assert raised.value.code == 4
        assert host.execv_args is None
        assert host.reexec_args == [
            sys.executable,
            "-m",
            "agilerl.cli",
            "experiments",
            "list",
        ]
        assert upgrade.os.environ[UPGRADE_RETRIED_ENV] == "1"

    def test_auto_upgrade_runs_without_a_tty_or_a_prompt(
        self,
        host: Host,
        capsys: pytest.CaptureFixture[str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Arrange
        monkeypatch.setenv(AUTO_UPGRADE_ENV, "1")

        # Act
        with pytest.raises(Execed):
            offer_upgrade(DESCRIPTION)

        # Assert
        assert host.pip_args is not None
        assert "[Y/n]" not in capsys.readouterr().err

    def test_a_re_run_does_not_offer_again(
        self,
        host: Host,
        tty: Callable[[str], None],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Arrange
        monkeypatch.setenv(UPGRADE_RETRIED_ENV, "1")
        monkeypatch.setenv(AUTO_UPGRADE_ENV, "1")
        tty("y\n")

        # Act
        offer_upgrade(DESCRIPTION)

        # Assert
        assert host.pip_args is None
        assert host.execv_args is None


class TestUpgradeRequirement:
    def test_keeps_installed_extras(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(upgrade, "installed_extras", lambda: ["hub"])

        assert upgrade_requirement(DESCRIPTION) == "agilerl[hub]"

    def test_adds_extras_named_in_the_description(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(upgrade, "installed_extras", lambda: ["hub"])

        assert upgrade_requirement(SNOWFLAKE_DESCRIPTION) == ("agilerl[hub,snowflake]")

    def test_no_extras_is_the_bare_name(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(upgrade, "installed_extras", list)

        assert upgrade_requirement(DESCRIPTION) == "agilerl"


class FakeDistribution:
    def __init__(self, requires: list[str], extras: list[str]) -> None:
        self.requires = requires
        self.metadata = FakeMetadata({"Provides-Extra": extras})


class FakeMetadata(dict):
    def get_all(self, key: str) -> list[str] | None:
        return self.get(key)


class TestInstalledExtras:
    def test_an_extra_counts_when_all_its_requirements_are_installed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange
        arena = FakeDistribution(
            requires=[
                "click>=8.3.0",
                'huggingface-hub>=0.30; extra == "hub"',
                'snowflake-connector-python>=4.8; extra == "snowflake"',
                'snowflake-snowpark-python; extra == "snowflake"',
            ],
            extras=["hub", "snowflake"],
        )
        installed = {"agilerl", "click", "huggingface-hub"}

        def distribution(name: str) -> object:
            if name == "agilerl":
                return arena
            if name in installed:
                return object()
            raise upgrade.metadata.PackageNotFoundError(name)

        monkeypatch.setattr(upgrade.metadata, "distribution", distribution)

        # Act
        extras = installed_extras()

        # Assert
        assert extras == ["hub"]


class TestUpgradeAtTheCliErrorBoundary:
    @staticmethod
    def _client(monkeypatch: pytest.MonkeyPatch, body: dict[str, str]) -> ArenaClient:
        real_client = httpx.Client
        monkeypatch.setattr(
            "agilerl.client.httpx.Client",
            lambda **kwargs: real_client(
                transport=httpx.MockTransport(
                    lambda request: httpx.Response(426, json=body)
                ),
                **kwargs,
            ),
        )
        with patch("agilerl.auth.KeycloakOpenID"):
            return ArenaClient(api_key="test-key")

    def test_a_426_from_arena_offers_the_upgrade(
        self,
        host: Host,
        tty: Callable[[str], None],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Arrange
        client = self._client(
            monkeypatch,
            {"error": "client_upgrade_required", "description": DESCRIPTION},
        )
        tty("y\n")
        with pytest.raises(ArenaClientUpgradeRequiredError) as raised:
            client.list_models()

        # Act
        with pytest.raises(Execed):
            handle_error(raised.value)

        # Assert
        assert host.pip_args is not None
        assert host.pip_args[-1] == "agilerl"

    def test_another_426_error_is_not_an_upgrade(
        self, host: Host, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        client = self._client(
            monkeypatch, {"error": "something_else", "description": "No."}
        )

        with pytest.raises(ArenaAPIError) as raised:
            client.list_models()

        assert not isinstance(raised.value, ArenaClientUpgradeRequiredError)
        assert raised.value.status_code == 426

    def test_without_a_tty_the_error_prints_as_before(
        self, host: Host, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange
        client = self._client(
            monkeypatch,
            {"error": "client_upgrade_required", "description": DESCRIPTION},
        )
        with pytest.raises(ArenaClientUpgradeRequiredError) as raised:
            client.list_models()

        # Act
        with pytest.raises(click.exceptions.Exit) as exited:
            handle_error(raised.value)

        # Assert
        assert exited.value.exit_code == 1
        assert host.pip_args is None
