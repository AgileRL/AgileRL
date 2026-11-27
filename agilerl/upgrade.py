# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Upgrade agilerl and re-run the command when Arena refuses this release."""

from __future__ import annotations

import os
import re
import subprocess
import sys
from importlib import metadata

import click
from packaging.requirements import Requirement

DISTRIBUTION = "agilerl"
CLI_MODULE = "agilerl.cli"
AUTO_UPGRADE_ENV = "ARENA_AUTO_UPGRADE"
UPGRADE_RETRIED_ENV = "ARENA_UPGRADE_RETRIED"


def offer_upgrade(description: str) -> None:
    """Offer to upgrade agilerl, then re-run the command on the new release.

    Prompts on a TTY, or upgrades without asking when ``ARENA_AUTO_UPGRADE=1``.
    Returns without upgrading otherwise, or after one re-run already happened.
    On Windows the upgraded command runs in a child process and this process
    exits with that command's status.

    :param description: Arena's 426 description, which names the pip command.
    :raises click.exceptions.Exit: The user declined, or pip failed.
    :raises SystemExit: Windows, with the status of the re-run command.
    """
    if os.environ.get(UPGRADE_RETRIED_ENV):
        return
    auto = os.environ.get(AUTO_UPGRADE_ENV) == "1"
    interactive = sys.stdin.isatty() and sys.stdout.isatty() and sys.stderr.isatty()
    if not auto and not interactive:
        return
    click.echo(description, err=True)
    if not auto and not click.confirm("Upgrade agilerl now?", default=True, err=True):
        raise click.exceptions.Exit(1)
    requirement = upgrade_requirement(description)
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-U", requirement], check=False
    )
    if result.returncode != 0:
        click.echo(f'The upgrade failed. Run: pip install -U "{requirement}"', err=True)
        raise click.exceptions.Exit(1)
    os.environ[UPGRADE_RETRIED_ENV] = "1"
    _reexec_cli()


def _reexec_cli() -> None:
    """Run this command again. On Windows, wait and exit with its status."""
    argv = [sys.executable, "-m", CLI_MODULE, *sys.argv[1:]]
    if sys.platform == "win32":
        # os.execv on Windows exits 0 without waiting for the command.
        completed = subprocess.run(argv, check=False)
        raise SystemExit(completed.returncode)
    os.execv(sys.executable, argv)


def upgrade_requirement(description: str) -> str:
    """The pip requirement that upgrades agilerl with its current extras.

    :param description: Arena's 426 description; extras it names are kept.
    :return: ``agilerl`` or ``agilerl[extra,...]``.
    """
    description_extras = re.compile(r"agilerl\[([A-Za-z0-9_.,\- ]+)\]")
    extras = set(installed_extras())
    for match in description_extras.findall(description):
        extras.update(extra.strip() for extra in match.split(",") if extra.strip())
    if not extras:
        return DISTRIBUTION
    return f"{DISTRIBUTION}[{','.join(sorted(extras))}]"


def installed_extras() -> list[str]:
    """Extras of the installed agilerl whose requirements are all installed.

    :return: Extra names, in metadata order.
    """
    dist = metadata.distribution(DISTRIBUTION)
    requirements = [Requirement(line) for line in dist.requires or []]
    found = []
    for extra in dist.metadata.get_all("Provides-Extra") or []:
        needed = [
            req
            for req in requirements
            if req.marker is not None
            and req.marker.evaluate({"extra": extra})
            and not req.marker.evaluate({"extra": ""})
        ]
        if needed and all(_is_installed(req.name) for req in needed):
            found.append(extra)
    return found


def _is_installed(name: str) -> bool:
    try:
        metadata.distribution(name)
    except metadata.PackageNotFoundError:
        return False
    return True
