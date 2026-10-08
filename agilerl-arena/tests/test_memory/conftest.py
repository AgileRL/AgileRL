# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0
"""Fixtures for the memory estimator tests."""

import pytest


@pytest.fixture(autouse=True)
def uncoloured_report(monkeypatch):
    """Rich forces colour on captured output when FORCE_COLOR is set; tests read plain text."""
    monkeypatch.delenv("FORCE_COLOR", raising=False)
