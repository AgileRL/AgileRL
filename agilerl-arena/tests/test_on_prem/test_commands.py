"""Tests for the install/teardown Click commands and verbosity helper."""

from __future__ import annotations

import logging
from collections.abc import Callable
from unittest.mock import MagicMock, patch

import click
from agilerl.arena.config import CommandConfig
from agilerl.arena.on_prem.commands import (
    _apply_verbosity,
    build_down_command,
    build_teardown_command,
)
from click.testing import CliRunner

from agilerl.arena.on_prem import build_install_command, register_on_prem_install


def test_install_command_parses_workers_and_delegates(
    command_config: CommandConfig, client_context: Callable[[MagicMock], MagicMock]
) -> None:
    client = MagicMock()
    with (
        patch(
            "agilerl.arena.on_prem.commands.arena_client",
            return_value=client_context(client),
        ),
        patch("agilerl.arena.on_prem.commands.run_on_prem_install") as run_mock,
    ):
        result = CliRunner().invoke(
            build_install_command(),
            ["pool", "--manager", "10.0.0.1", "--workers", "w1, w2"],
            obj=command_config,
        )
    assert result.exit_code == 0, result.output
    kwargs = run_mock.call_args.kwargs
    assert kwargs["name"] == "pool"
    assert kwargs["manager"] == "10.0.0.1"
    assert kwargs["workers"] == ("w1", "w2")  # comma-split + stripped
    assert kwargs["setup_type"] == "dockerSwarm"  # default


def test_teardown_command_maps_flags(
    command_config: CommandConfig, client_context: Callable[[MagicMock], MagicMock]
) -> None:
    client = MagicMock()
    with (
        patch(
            "agilerl.arena.on_prem.commands.arena_client",
            return_value=client_context(client),
        ),
        patch("agilerl.arena.on_prem.commands.run_on_prem_teardown") as run_mock,
    ):
        result = CliRunner().invoke(
            build_teardown_command(),
            [
                "pool",
                "--manager",
                "m",
                "--disable-provider",
                "--leave-swarm",
            ],
            obj=command_config,
        )
    assert result.exit_code == 0, result.output
    kwargs = run_mock.call_args.kwargs
    assert kwargs["name"] == "pool"
    assert kwargs["disable_provider"] is True
    assert kwargs["leave_swarm"] is True


def test_down_command_maps_flags(
    command_config: CommandConfig, client_context: Callable[[MagicMock], MagicMock]
) -> None:
    client = MagicMock()
    with (
        patch(
            "agilerl.arena.on_prem.commands.arena_client",
            return_value=client_context(client),
        ),
        patch("agilerl.arena.on_prem.commands.run_on_prem_down") as run_mock,
    ):
        result = CliRunner().invoke(
            build_down_command(),
            [
                "pool",
                "--manager",
                "m",
                "--stack-name",
                "my-stack",
            ],
            obj=command_config,
        )
    assert result.exit_code == 0, result.output
    kwargs = run_mock.call_args.kwargs
    assert kwargs["name"] == "pool"
    assert kwargs["manager"] == "m"
    assert kwargs["stack_name"] == "my-stack"


def test_register_on_prem_install_replaces_install_with_commands() -> None:
    group = click.Group(name="on-prem")
    group.add_command(click.Command(name="install"))  # placeholder manifest install
    register_on_prem_install(group)
    assert set(group.commands) >= {"install", "down", "teardown"}
    # The replacement install is the hardcoded command (has a NAME argument).
    install = group.commands["install"]
    assert any(p.name == "name" for p in install.params)


def test_apply_verbosity_toggles_debug_level() -> None:
    arena_logger = logging.getLogger("agilerl.arena")
    original = arena_logger.level
    try:
        _apply_verbosity(verbose=False)
        assert arena_logger.level == original  # unchanged
        _apply_verbosity(verbose=True)
        assert arena_logger.level == logging.DEBUG
    finally:
        arena_logger.setLevel(original)
