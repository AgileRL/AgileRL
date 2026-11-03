# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""AgileRL rubric helpers on top of OpenEnv's ``Rubric`` API.

A rubric scores every turn of any rollout env: the harness calls it with a
:class:`TurnAction` and :class:`TurnObservation`, and its score replaces
the env's reward for that turn. The simple path is one reward function wrapped
with :func:`reward_fn_to_rubric`; :class:`EnvReward` brings the env's own reward
in as a component. Component-level metrics come from composing OpenEnv
``Rubric``s (child attributes auto-register); AgileRL does not add its own
aggregation containers.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Protocol

from openenv.core.rubrics.base import Rubric

from agilerl.llm_envs.chat_formats import ParsedToolCall


@dataclass
class TurnAction:
    """One policy turn, as the rubric sees it.

    :param message: Text the env received after ``action_processor``:
        post-reasoning, or the whole turn under ``include_reasoning``. A
        tool-call turn keeps the decoded text; the env received the parsed
        calls.
    :param completion: The whole turn, reasoning and its markers included.
    :param tool_calls: Tool calls parsed from the turn; empty on a text env.
    """

    message: str
    completion: str
    tool_calls: list[ParsedToolCall] = field(default_factory=list)


@dataclass
class TurnObservation:
    """What the env did with one turn.

    :param env_reward: The env's summed reward for the turn, or ``min_reward``
        when the turn never reached the env.
    :param env_ran: Whether the env stepped; ``False`` for a turn rejected for
        a malformed or missing tool call.
    :param done: Whether this is the episode's last turn: terminated, truncated,
        or the turn budget spent.
    :param turn_index: Zero-based turn number in the episode.
    :param prompt: The episode's first prompt text.
    :param question: The env observation's ``question`` field, else ``None``.
    :param answer: The env observation's ``answer`` field, else ``None``.
    :param payload: The env's last observation fields, or ``None`` when the
        env did not step.
    :param env_info: The env's merged step info.
    :param metadata: Written by rubric hooks (``rubric_scores``).
    """

    env_reward: float
    env_ran: bool
    done: bool
    turn_index: int
    prompt: str
    question: Any = None
    answer: Any = None
    payload: Any = None
    env_info: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


class RewardFn(Protocol):
    def __call__(self, completion: str, answer: object, question: object) -> float: ...


COMPONENT_NAME_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_.]*")

# Attribute stamped on a rubric node holding the component names it already has a
# hook for. register_component_hooks is idempotent so a second call on the same
# node does not add a duplicate hook.
HOOKED_COMPONENTS_ATTR = "_agilerl_hooked_components"


def _snake_case(name: str) -> str:
    """Convert ``PascalCase`` / ``camelCase`` to ``snake_case``."""
    name = name.lstrip("_")
    s1 = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s1).lower()


def _identifier_or(name: str, fallback: str) -> str:
    """Return ``name`` when it is a valid component identifier, else ``fallback``."""
    return name if COMPONENT_NAME_RE.fullmatch(name) else fallback


def _require_rubric(rubric: Rubric) -> Rubric:
    """Require an OpenEnv ``Rubric``; point callers at ``reward_fn_to_rubric`` otherwise."""
    if not isinstance(rubric, Rubric):
        msg = (
            f"rubric must be an openenv Rubric, got {type(rubric).__name__}. "
            "Wrap a callable of completion, answer, and question with "
            "reward_fn_to_rubric(...)."
        )
        raise TypeError(msg)
    return rubric


def reward_fn_to_rubric(
    fn: RewardFn,
    name: str | None = None,
) -> Rubric:
    """Wrap a reward function called as ``completion``, ``answer``, and ``question``.

    One scorer, one scalar reward. Parameter order does not matter. For
    per-criterion metrics, write OpenEnv ``Rubric`` subclasses and compose
    them (assign children as attributes so they auto-register).

    :param fn: Scorer called with completion, answer, and question by name.
    :param name: Component name for metrics; defaults to ``fn.__name__`` when valid.
    :returns: An OpenEnv ``Rubric`` instance.
    """
    return RewardFnRubric(fn, name)


class RewardFnRubric(Rubric):
    """Adapter for :func:`reward_fn_to_rubric`."""

    def __init__(
        self,
        fn: RewardFn,
        name: str | None,
    ) -> None:
        super().__init__()
        self._fn = fn
        self.component_name = name or _identifier_or(getattr(fn, "__name__", ""), "fn")

    def forward(self, action: TurnAction, observation: TurnObservation) -> float:
        return float(
            self._fn(
                completion=action.message,
                answer=observation.answer,
                question=observation.question,
            )
        )


class EnvReward(Rubric):
    """The env's own reward for the turn, reported as the ``env`` component."""

    component_name = "env"

    def forward(self, action: TurnAction, observation: TurnObservation) -> float:
        return float(observation.env_reward)


def register_component_hooks(rubric: Rubric | None) -> tuple[str, ...]:
    """Register leaf forward hooks that write into ``observation.metadata``.

    OpenEnv's observability pattern: each leaf's post-``forward`` hook stamps
    ``observation.metadata["rubric_scores"][name]`` for that call. Returns the
    ordered leaf names (empty when ``rubric`` is ``None``). Registration is
    idempotent per node: the hook writes into whichever observation it is handed
    and holds no per-env state.
    """
    if rubric is None:
        return ()

    names: list[str] = []

    def walk(node: Rubric, path: str) -> None:
        children = list(node.named_children())
        if children:
            for child_name, child in children:
                walk(child, f"{path}.{child_name}" if path else child_name)
            return

        name = (
            getattr(node, "component_name", None)
            or path
            or _snake_case(type(node).__name__)
        )
        if not isinstance(name, str) or not name:
            name = _snake_case(type(node).__name__)

        def hook(
            _rubric: Rubric,
            _action: object,
            observation: object,
            result: float,
            *,
            _name: str = name,
        ) -> None:
            metadata = getattr(observation, "metadata", None)
            if metadata is not None:
                metadata.setdefault("rubric_scores", {})[_name] = float(result)

        hooked = getattr(node, HOOKED_COMPONENTS_ATTR, None)
        if hooked is None:
            hooked = set()
            setattr(node, HOOKED_COMPONENTS_ATTR, hooked)
        if name not in hooked:
            node.register_forward_hook(hook)
            hooked.add(name)
        names.append(name)

    walk(rubric, "")
    return tuple(names)
