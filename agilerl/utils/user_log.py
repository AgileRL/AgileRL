# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Messages for the person running training, on the ``agilerl.user`` logger."""

from __future__ import annotations

import logging
from typing import TypeAlias

USER_LOGGER_NAME = "agilerl.user"

UserField: TypeAlias = int | float | str


def log_for_user(
    event: str,
    message: str,
    level: int = logging.INFO,
    stacklevel: int = 1,
    **fields: UserField,
) -> None:
    """Log a message meant for the person running training, not for debugging.

    :param event: Event type, e.g. ``"step_progress"``.
    :type event: str
    :param message: The full text the person reads: no hosts, paths, ranks or
        internal service names.
    :type message: str
    :param level: ``logging.INFO`` for progress, ``logging.WARNING`` for something
        they can act on, ``logging.ERROR`` for a failure.
    :type level: int
    :param stacklevel: Which caller the record's file and line name: 1 is the
        caller of this function, 2 its caller, as for :meth:`logging.Logger.log`.
    :type stacklevel: int
    :param fields: Structured values behind the message, e.g. ``step=12``.
    :type fields: UserField
    """
    logging.getLogger(USER_LOGGER_NAME).log(
        level,
        message,
        extra={"user_event": event, "user_fields": fields},
        stacklevel=stacklevel + 1,
    )


def format_duration(seconds: float) -> str:
    """Render a duration for a person: ``"1h 48m"``, ``"3m 40s"`` or ``"12s"``.

    :param seconds: Duration in seconds.
    :type seconds: float
    :return: The duration in its two largest units.
    :rtype: str
    """
    whole = round(seconds)
    hours, rest = divmod(whole, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{hours}h {minutes}m"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"
