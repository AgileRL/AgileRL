from __future__ import annotations

from agilerl import HAS_ARENA_DEPENDENCIES

_INSTALL_MSG = (
    "Arena dependencies are not installed. "
    "Please install them using: pip install agilerl[arena]"
)

if HAS_ARENA_DEPENDENCIES:
    from agilerl.arena.agent import Agent
    from agilerl.arena.client import ArenaClient
    from agilerl.arena.exceptions import (
        ArenaAPIError,
        ArenaAuthError,
        ArenaError,
        ArenaValidationError,
    )
    from agilerl.arena.logs import EventStream, LogDisplay, LogEvent
    from agilerl.arena.models import (
        AgentConfig,
        AlgorithmSpec,
        EnvironmentRef,
        MutationSpec,
        ResourceSpec,
        TournamentSpec,
        TrainingJobConfig,
    )

    __all__ = [
        "Agent",
        "AgentConfig",
        "AlgorithmSpec",
        "ArenaAPIError",
        "ArenaAuthError",
        "ArenaClient",
        "ArenaError",
        "ArenaValidationError",
        "EnvironmentRef",
        "EventStream",
        "LogDisplay",
        "LogEvent",
        "MutationSpec",
        "ResourceSpec",
        "TournamentSpec",
        "TrainingJobConfig",
    ]

else:

    def __getattr__(name: str):
        raise ImportError(_INSTALL_MSG)
