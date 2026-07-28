# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from enum import Enum
from typing import TYPE_CHECKING

import lazy_loader as lazy
from rich.console import Console
from rich.logging import RichHandler

if TYPE_CHECKING:
    from agilerl.arena.client import ArenaClient
    from agilerl.arena.inference import Agent

console = Console()
error_console = Console(stderr=True)

_logger = logging.getLogger("agilerl.arena")
if not _logger.handlers:
    _logger.setLevel(logging.INFO)
    _logger.addHandler(
        RichHandler(show_time=False, show_path=False, markup=True, console=console)
    )
    _logger.propagate = False

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

__getattr__, __dir__, _ = lazy.attach(
    __name__,
    submod_attrs={
        "client": ["ArenaClient"],
        "inference": ["Agent"],
    },
)

__all__ = ["Agent", "ArenaClient"]


class AgentType(Enum):
    """Enumeration of supported agent types."""

    SingleAgent = "single_agent"
    MultiAgent = "multi_agent"
    LLMAgent = "llm_agent"
