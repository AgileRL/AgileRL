# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Rich consoles and logging setup for the CLI.

Kept out of ``agilerl.arena.__init__`` so importing the manifest contract does
not pull in rich or install a log handler as a side effect: the platform's
submit preflight imports ``agilerl.arena.models`` and nothing else.
"""

from __future__ import annotations

import logging

from rich.console import Console
from rich.logging import RichHandler

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
