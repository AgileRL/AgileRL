# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Every visible ``arena`` option has help text."""

from __future__ import annotations

from typing import TYPE_CHECKING

import click

from agilerl.cli import main

if TYPE_CHECKING:
    from collections.abc import Iterator


def visible_options(
    command: click.Command, path: str
) -> Iterator[tuple[str, click.Option]]:
    """Yield ``(command path, option)`` for every non-hidden option under ``command``.

    :param command: Click command or group to walk.
    :param path: Command path so far, e.g. ``arena env``.
    :return: Iterator of command path and option pairs.
    """
    for param in command.params:
        if isinstance(param, click.Option) and not param.hidden:
            yield path, param
    if isinstance(command, click.Group):
        for name, sub in command.commands.items():
            yield from visible_options(sub, f"{path} {name}")


def test_every_visible_option_has_help() -> None:
    missing = [
        f"{path} {option.opts[0]}"
        for path, option in visible_options(main, "arena")
        if not option.help
    ]

    assert missing == []
