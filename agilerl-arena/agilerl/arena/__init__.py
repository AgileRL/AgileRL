# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

import lazy_loader as lazy

from agilerl.arena.models import AgentType

if TYPE_CHECKING:
    from rich.console import Console

    from agilerl.arena.client import ArenaClient
    from agilerl.arena.inference import Agent

    console: Console
    error_console: Console

__getattr__, __dir__, _ = lazy.attach(
    __name__,
    submod_attrs={
        "_console": ["console", "error_console"],
        "client": ["ArenaClient"],
        "inference": ["Agent"],
    },
)

__all__ = ["Agent", "AgentType", "ArenaClient", "console", "error_console"]
