# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Free trial of the Arena Snowflake Native App: when an unpurchased listing install stops.

``arena_app`` imports this module. The setup script lists both files in each
procedure's ``IMPORTS``, which puts them on the path side by side.
"""

from __future__ import annotations

import logging
from datetime import timedelta
from typing import TYPE_CHECKING

from agilerl.arena.snowflake.catalog import collect
from agilerl.arena.snowflake.job import drop_service_sql, lowered

if TYPE_CHECKING:
    from agilerl.arena.snowflake.source import SnowflakeSession

APP_SCHEMA = "app_state"
# How long an unpurchased listing install may train and serve after install: the
# listing's 7-day trial less one reconcile_usage_task interval (SCHEDULE in
# setup_script.sql), so a task run drops the services before Snowflake ends access.
TRIAL = timedelta(days=7) - timedelta(minutes=60)

logger = logging.getLogger(__name__)


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
