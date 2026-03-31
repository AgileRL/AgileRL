from __future__ import annotations

from pydantic import Field
from pydantic.dataclasses import dataclass


@dataclass
class EnvSpec:
    """Environment specification from an Arena manifest.

    Provides information that allows us to construct both gymnasium as well as
    pettingzoo environments, and also custom environments from an entrypoint.

    :param name: Name of the environment
    :type name: str
    :param num_envs: Number of environments to run in parallel
    :type num_envs: int
    :param entrypoint: Entrypoint for the environment, if custom. Defaults to None.
    :type entrypoint: str or None
    :param env_path: Path to the environment, if custom. Defaults to None.
    :type env_path: str or None
    :param env_config: Environment configuration, if custom. Defaults to None.
    :type env_config: dict[str, Any] or None
    :param env_wrappers: Environment wrappers, if custom. Defaults to None.
    :type env_wrappers: list[tuple[Any, dict[str, Any]] | str] or None
    :param sync: Use synchronous vectorization instead of async.
    :type sync: bool
    """

    name: str
    num_envs: int


@dataclass
class ArenaEnvSpec(EnvSpec):
    """Environment specification for Arena environments.

    :param version: Version of the environment
    :type version: str
    """

    version: str = Field(default="latest")
