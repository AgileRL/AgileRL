from __future__ import annotations

from agilerl import HAS_ARENA_DEPENDENCIES

if HAS_ARENA_DEPENDENCIES:
    from agilerl.arena.client import ArenaClient

    __all__ = ["ArenaClient"]
else:
    msg = "Arena dependencies are not installed. Please install them using: pip install agilerl[arena]"
    raise ImportError(msg)
