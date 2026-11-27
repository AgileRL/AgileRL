# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Free trial of the Arena Snowflake Native App: when an unpurchased listing install stops.

Every other app module imports this one, which imports none of them, so it
also sets up logging and tracebacks for whichever module a procedure's handler
is in. The setup script lists every module in each procedure's ``IMPORTS``,
which puts them on the path side by side.
"""

from __future__ import annotations

import importlib
import logging
import sys
from datetime import timedelta
from typing import TYPE_CHECKING

from agilerl.snowflake.catalog import collect
from agilerl.snowflake.job import drop_service_sql, lowered

if TYPE_CHECKING:
    from agilerl.snowflake.source import SnowflakeSession

APP_SCHEMA = "app_state"
# How long an unpurchased listing install may train and serve after install: the
# listing's 7-day trial less one reconcile_usage_task interval (SCHEDULE in
# setup_script.sql), so a task run drops the services before Snowflake ends access.
TRIAL = timedelta(days=7) - timedelta(minutes=60)

logger = logging.getLogger(__name__)

# Snowflake writes procedure logs to the event table from the root logger; the
# CLI's console handler would keep agilerl records on stderr instead.
# Loading the console first installs that handler now, so it is removed here
# rather than added by a later import.
importlib.import_module("agilerl._console")
ARENA_LOGGER = logging.getLogger("agilerl")
ARENA_LOGGER.handlers.clear()
ARENA_LOGGER.propagate = True

# Snowflake returns a procedure's formatted exception to the caller; without this
# the consumer sees the app's stack and file paths around the message.
sys.tracebacklimit = 0


def trial_ended(session: SnowflakeSession) -> bool:
    """Whether the app is a listing install, not purchased, installed over :data:`TRIAL` ago.

    :param session: Procedure session.
    :return: True once the trial has ended unpurchased.
    """
    [row] = collect(
        session,
        "SELECT NOT app_code.listing_purchased() AND MIN(started_at) <= "
        f"DATEADD('second', -{int(TRIAL.total_seconds())}, CURRENT_TIMESTAMP()) "
        "AS ENDED FROM app_state.trial",
    )
    return lowered(row)["ended"] is True


def refuse_ended_trial(session: SnowflakeSession) -> None:
    """Refuse new work once the trial has ended unpurchased.

    :param session: Procedure session.
    :raises PermissionError: The trial has ended and the listing is not purchased.
    """
    if trial_ended(session):
        msg = (
            "The free trial of AgileRL Arena has ended.\n"
            "Purchase the AgileRL Arena listing in Snowflake Marketplace "
            "to train and serve again.\nNothing was started."
        )
        raise PermissionError(msg)


def end_trial(session: SnowflakeSession, services: list[str]) -> str:
    """Drop the given services and forget the served models.

    Their open deployments close when metering next finds the service gone.

    :param session: Procedure session.
    :param services: Every live service and every served model's service in ``app_state``.
    :return: How many services were dropped.
    """
    for name in services:
        collect(session, drop_service_sql(f"{APP_SCHEMA}.{name}", if_exists=True))
        logger.info("Trial ended unpurchased: dropped service %s", name)
    collect(session, f"DELETE FROM {APP_SCHEMA}.served_models")
    return f"trial ended, dropped services: {len(services)}"
