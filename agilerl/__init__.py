# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

import lazy_loader as lazy

from agilerl import _framework_guard
from agilerl.models import AgentType

if TYPE_CHECKING:
    from rich.console import Console

    from agilerl.client import ArenaClient
    from agilerl.inference import Agent

    console: Console
    error_console: Console

_framework_guard.install()

lazy_getattr, __dir__, _ = lazy.attach(
    __name__,
    submod_attrs={
        "_console": ["console", "error_console"],
        "client": ["ArenaClient"],
        "inference": ["Agent"],
    },
)


def __getattr__(name: str) -> object:
    if name in _framework_guard.FRAMEWORK_PACKAGES:
        raise _framework_guard.missing_attribute_error(sys.modules[__name__], name)
    return lazy_getattr(name)


__all__ = ["Agent", "AgentType", "ArenaClient", "console", "error_console"]
