from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from agilerl.utils.cache import Cache


class Language_Observation(ABC):
    @abstractmethod
    def to_sequence(self) -> tuple[list[str, float | None], bool]:
        # returns a List of Tuples and a bool indicating terminal
        # each state Tuple should be: (str, None)
        # each action Tuple should be: (str, reward)
        pass

    @abstractmethod
    def __str__(self) -> str:
        pass

    def metadata(self) -> dict[str, Any] | None:
        return None


class Language_Environment(ABC):
    @abstractmethod
    def step(self, action: str) -> tuple[Language_Observation, float, bool]:
        pass

    @abstractmethod
    def reset(self) -> Language_Observation:
        pass

    @abstractmethod
    def is_terminal(self) -> bool:
        pass


class Policy(ABC):
    def __init__(self) -> None:
        super().__init__()
        self.cache = Cache()

    @abstractmethod
    def act(self, obs: Language_Observation) -> str:
        pass

    @abstractmethod
    def train(self) -> None:
        """Set policy to training mode; override in subclasses if needed."""
        ...

    @abstractmethod
    def eval(self) -> None:
        """Set policy to eval mode; override in subclasses if needed."""
        ...


def interact_environment(
    env: Language_Environment,
    policy: Policy,
    obs: Language_Observation | None = None,
) -> tuple[
    Language_Observation, list[tuple[Language_Observation, str | None, float, bool]]
]:
    obs_sequence: list[tuple[Language_Observation, str | None, float, bool]] = []
    if obs is None:
        obs = env.reset()
    while not env.is_terminal():
        action = policy.act(obs)
        new_obs, r, t = env.step(action)
        obs_sequence.append((obs, action, r, t))
        obs = new_obs
    obs_sequence.append((obs, None, 0, True))
    return obs, obs_sequence
