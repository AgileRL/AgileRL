from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import lazy_loader as lazy
from rich.console import Console
from rich.logging import RichHandler

from agilerl import HAS_ARENA_DEPENDENCIES

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

if not HAS_ARENA_DEPENDENCIES:
    msg = "Arena dependencies are not installed. Please install them using: pip install agilerl[arena]"
    raise ImportError(msg)

__getattr__, __dir__, _ = lazy.attach(
    __name__,
    submod_attrs={
        "client": ["ArenaClient"],
        "inference": ["Agent"],
    },
)

__all__ = ["Agent", "ArenaClient"]
