"""Gymnasium-style environments for LLM training."""

from typing import TYPE_CHECKING, Any

from agilerl.llm_envs.dataset_env import DatasetEnv
from agilerl.llm_envs.rollout_env import (
    BatchPointer,
    BatchRolloutEnv,
    RolloutEnv,
)
from agilerl.utils.llm_utils import apply_chat_template

if TYPE_CHECKING:
    from agilerl.llm_envs.openenv import (
        LocalEnvClient,
        OpenEnvClient,
        OpenEnvServer,
        OpenEnvWrapper,
        ServedEnvClient,
        TextAction,
        TextObservation,
        load_env,
        resolve_env,
    )

_OPENENV_EXPORTS = frozenset(
    {
        "LocalEnvClient",
        "OpenEnvClient",
        "OpenEnvServer",
        "OpenEnvWrapper",
        "ServedEnvClient",
        "TextAction",
        "TextObservation",
        "load_env",
        "resolve_env",
    }
)


def __getattr__(name: str) -> Any:
    """Import the OpenEnv-backed exports lazily.

    The OpenEnv backend needs the optional ``openenv`` package (llm extra);
    deferring its import keeps ``RolloutEnv`` / ``DatasetEnv`` importable on a
    base install.
    """
    if name in _OPENENV_EXPORTS:
        from agilerl.llm_envs import openenv

        return getattr(openenv, name)
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)


__all__ = [
    "BatchPointer",
    "BatchRolloutEnv",
    "DatasetEnv",
    "LocalEnvClient",
    "OpenEnvClient",
    "OpenEnvServer",
    "OpenEnvWrapper",
    "RolloutEnv",
    "ServedEnvClient",
    "TextAction",
    "TextObservation",
    "apply_chat_template",
    "load_env",
    "resolve_env",
]
