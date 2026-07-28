# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Shared fixtures for Arena auth/client tests."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from agilerl.arena.auth import ArenaOAuth2


@pytest.fixture(autouse=True)
def _detach_arena_rich_handler() -> None:
    """Detach the package ``RichHandler`` from ``agilerl.arena`` during tests.

    The package attaches a ``RichHandler`` to the ``agilerl.arena`` logger at
    import time. Under coverage instrumentation that handler's highlighter can
    raise, turning any test that merely emits a log record into a spurious
    failure. Tests don't assert on rendered log output (they patch the module
    logger when they care), so drop the handler and let records propagate to
    pytest's capture instead.
    """
    arena_logger = logging.getLogger("agilerl.arena")
    saved_handlers = arena_logger.handlers[:]
    saved_propagate = arena_logger.propagate
    arena_logger.handlers = []
    arena_logger.propagate = True
    try:
        yield
    finally:
        arena_logger.handlers = saved_handlers
        arena_logger.propagate = saved_propagate


@pytest.fixture(autouse=True)
def _isolate_arena_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Prevent a developer's real ``ARENA_API_KEY`` from leaking into tests.

    Without this, a key exported in the shell makes the CLI attempt real
    authentication (e.g. ``arena --help`` building a client), producing
    spurious 401 failures. Tests that need a key set it explicitly.
    """
    monkeypatch.delenv("ARENA_API_KEY", raising=False)


@pytest.fixture
def arena_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Temporary HOME with ``~/.arena`` created under it."""
    monkeypatch.setenv("HOME", str(tmp_path))
    arena_dir = tmp_path / ".arena"
    arena_dir.mkdir(parents=True, exist_ok=True)
    return tmp_path


@pytest.fixture
def credentials_path(arena_home: Path) -> Path:
    """Path to ``credentials.json`` in the isolated arena home."""
    return arena_home / ".arena" / "credentials.json"


@pytest.fixture
def credentials_file(credentials_path: Path) -> Path:
    """Write a minimal valid credentials file on the tmp arena home."""
    payload: dict[str, Any] = {"access_token": "at", "refresh_token": "rt"}
    credentials_path.write_text(json.dumps(payload), encoding="utf-8")
    return credentials_path


@pytest.fixture(autouse=True)
def _isolate_arena_credentials(
    arena_home: Path,
) -> None:
    """Point ArenaOAuth2 at the tmp ``~/.arena/credentials.json`` for every test."""
    arena_dir = arena_home / ".arena"
    cred_file = arena_dir / "credentials.json"
    orig_dir = ArenaOAuth2.CREDENTIALS_DIR
    orig_file = ArenaOAuth2.CREDENTIALS_FILE
    ArenaOAuth2.CREDENTIALS_DIR = arena_dir
    ArenaOAuth2.CREDENTIALS_FILE = cred_file
    yield
    ArenaOAuth2.CREDENTIALS_DIR = orig_dir
    ArenaOAuth2.CREDENTIALS_FILE = orig_file
