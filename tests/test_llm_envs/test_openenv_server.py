# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Tests for ``agilerl.llm_envs.openenv_server`` logging."""

from __future__ import annotations

import logging
import sys
from itertools import pairwise

import pytest
from starlette.websockets import WebSocketDisconnect
from uvicorn.protocols.utils import ClientDisconnected

from agilerl.llm_envs.openenv_server import ClientDisconnectFilter


def _asgi_error(*chain: BaseException) -> logging.LogRecord:
    """Build the record uvicorn logs when the app raises ``chain[-1]`` while handling the earlier ones."""
    for earlier, later in pairwise(chain):
        later.__context__ = earlier
    try:
        raise chain[-1]
    except BaseException:
        exc_info = sys.exc_info()
    return logging.LogRecord(
        "uvicorn.error",
        logging.ERROR,
        __file__,
        1,
        "Exception in ASGI application\n",
        (),
        exc_info,
    )


class TestClientDisconnectFilter:
    def test_drops_client_disconnect_and_logs_debug(self, caplog):
        # Arrange: the chain openenv raises when it answers a client that is gone
        record = _asgi_error(
            ClientDisconnected(),
            WebSocketDisconnect(code=1006),
            RuntimeError('Cannot call "send" once a close message has been sent.'),
        )

        # Act
        with caplog.at_level(logging.DEBUG, logger="agilerl.llm_envs.openenv_server"):
            keep = ClientDisconnectFilter().filter(record)

        # Assert
        assert keep is False
        assert [(r.levelno, r.getMessage()) for r in caplog.records] == [
            (logging.DEBUG, "Env client closed its session: ClientDisconnected"),
        ]

    @pytest.mark.parametrize(
        "record",
        [
            _asgi_error(RuntimeError("env step failed")),
            _asgi_error(
                RuntimeError("env step failed"), WebSocketDisconnect(code=1006)
            ),
            logging.LogRecord(
                "uvicorn.error", logging.ERROR, __file__, 1, "no exception", (), None
            ),
        ],
    )
    def test_keeps_other_errors(self, record):
        assert ClientDisconnectFilter().filter(record) is True
