from __future__ import annotations

import logging

from rich.console import Console
from rich.logging import RichHandler

from agilerl import HAS_ARENA_DEPENDENCIES

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

if HAS_ARENA_DEPENDENCIES:
    from agilerl.arena.client import ArenaClient
    from agilerl.arena.inference import Agent

    __all__ = ["Agent", "ArenaClient"]
else:
    msg = "Arena dependencies are not installed. Please install them using: pip install agilerl[arena]"
    raise ImportError(msg)
