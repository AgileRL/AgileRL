# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Client version header Arena reads, and the upgrade signals it sends back."""

from __future__ import annotations

import sys
from functools import cache

import httpx

from agilerl.arena.exceptions import ArenaAPIError
from agilerl.arena.models import __version__

CLIENT_VERSION_HEADER = "X-Arena-Client-Version"
CLIENT_UPDATE_HEADER = "X-Arena-Client-Update"
UPGRADE_COMMAND = "pip install -U agilerl-arena"
SNOWFLAKE_UPGRADE_COMMAND = 'pip install -U "agilerl-arena[snowflake]"'


def upgrade_required_message(raw: str) -> str:
    """Arena's message from a 426 body. It already names the pip command.

    :param raw: Response body, ``{"error": ..., "description": ...}``.
    :return: The description, or the body text when it is not an error envelope.
    """
    return ArenaAPIError.from_response_body(raw, status_code=426).detail


def print_update_notice(
    response: httpx.Response, command: str = UPGRADE_COMMAND
) -> None:
    """Print one stderr line when Arena names a newer release. Fits an httpx response hook.

    :param response: Arena response.
    :param command: Upgrade command shown to the user.
    """
    latest = response.headers.get(CLIENT_UPDATE_HEADER, "").strip()
    if latest:
        announce_update(latest, command)


@cache
def announce_update(latest: str, command: str) -> None:
    """Print the newer-release line. Cached, so it prints once per process.

    :param latest: Newest agilerl-arena release.
    :param command: Upgrade command shown to the user.
    """
    print(
        f"agilerl-arena {latest} is available (you have {__version__}). Run: {command}",
        file=sys.stderr,
    )
