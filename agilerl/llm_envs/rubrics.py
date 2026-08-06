# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""AgileRL rubric helpers on top of OpenEnv's ``Rubric`` API.

The simple path is one reward function wrapped with :func:`reward_fn_to_rubric`.
Component-level metrics come from composing OpenEnv ``Rubric``s (child attributes
auto-register); AgileRL does not add its own aggregation containers.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

from openenv.core.rubrics.base import Rubric

_COMPONENT_NAME_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_.]*")


def _snake_case(name: str) -> str:
    """Convert ``PascalCase`` / ``camelCase`` to ``snake_case``."""
    name = name.lstrip("_")
    s1 = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s1).lower()


def _identifier_or(name: str, fallback: str) -> str:
    """Return ``name`` when it is a valid component identifier, else ``fallback``."""
    return name if _COMPONENT_NAME_RE.fullmatch(name) else fallback


def _require_rubric(rubric: Rubric) -> Rubric:
    """Require an OpenEnv ``Rubric``; point callers at ``reward_fn_to_rubric`` otherwise."""
    if not isinstance(rubric, Rubric):
        msg = (
            f"rubric must be an openenv Rubric, got {type(rubric).__name__}. "
            "Wrap a (completion, answer, question) -> float callable with "
            "reward_fn_to_rubric(...)."
        )
        raise TypeError(msg)
    return rubric


def reward_fn_to_rubric(
    fn: Callable[[str, Any, Any], float],
    name: str | None = None,
) -> Rubric:
    """Wrap a ``(completion, answer, question) -> float`` callable as a ``Rubric``.

    One scorer, one scalar reward. For per-criterion metrics, write OpenEnv
    ``Rubric`` subclasses and compose them (assign children as attributes so
    they auto-register).

    :param fn: Scorer taking completion text, answer label, and question.
    :param name: Component name for metrics; defaults to ``fn.__name__`` when valid.
    :returns: An OpenEnv ``Rubric`` instance.
    """
    return RewardFnRubric(fn, name)


class RewardFnRubric(Rubric):
    """Adapter for :func:`reward_fn_to_rubric`."""

    def __init__(
        self,
        fn: Callable[[str, Any, Any], float],
        name: str | None,
    ) -> None:
        super().__init__()
        self._fn = fn
        self.component_name = name or _identifier_or(getattr(fn, "__name__", ""), "fn")

    def forward(self, action: Any, observation: Any) -> float:  # noqa: ANN401
        return float(self._fn(action.message, observation.answer, observation.question))


def register_component_hooks(rubric: Rubric | None) -> tuple[str, ...]:
    """Register leaf forward hooks that write into ``observation.metadata``.

    OpenEnv's observability pattern: each leaf's post-``forward`` hook stamps
    ``observation.metadata["rubric_scores"][name]`` for that call. Returns the
    ordered leaf names (empty when ``rubric`` is ``None``). Give each env its
    own rubric tree — sharing registers duplicate hooks on the same nodes.
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

        node.register_forward_hook(hook)
        names.append(name)

    walk(rubric, "")
    return tuple(names)
