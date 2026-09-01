# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Chat roles on env observations.

An env turn is not automatically "the user said X". A tool result is a ``tool``
turn and an injected directive is a ``system`` turn, and the chat template has to
be told which, or the model reads a tool's output as a human message. The env
labels the turn; the harness renders a per-role boundary frame from that label.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import pytest
import torch

from agilerl.llm_envs import RolloutHarness
from agilerl.llm_envs.observation import (
    DEFAULT_OBSERVATION_ROLE,
    OBSERVATION_ROLES,
    observation_role,
)
from tests.helpers.rollout_doubles import MiniTokenizer, bare_rollout_env


class _RoleRecordingTokenizer:
    """Renders each message as ``<role>:<content>|`` and records what it saw."""

    pad_token_id = 0

    def __init__(self) -> None:
        self.messages: list[list[dict[str, str]]] = []

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        tokenize: bool = True,
        add_generation_prompt: bool = True,
        **_: Any,
    ) -> Any:
        self.messages.append([dict(m) for m in messages])
        rendered = "".join(f"{m['role']}:{m['content']}|" for m in messages)
        if add_generation_prompt:
            rendered += "assistant:"
        if not tokenize:
            return rendered
        return {"input_ids": [[len(rendered)]]}

    def encode(self, text: str, **_: Any) -> list[int]:
        return [ord(c) % 251 for c in text]


class _RoleEnvClient:
    """``EnvClientProtocol`` double whose step observation carries a role label."""

    def __init__(self, role: str | None, *, on_payload: bool = False) -> None:
        self._role = role
        self._on_payload = on_payload

    def reset(
        self, seed: int | None = None, *, row_index: int | None = None
    ) -> tuple[Any, dict[str, Any]]:
        del seed, row_index
        return {"prompt": "start"}, {}

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        del action
        payload: dict[str, Any] = {"prompt": "tool said 42"}
        info: dict[str, Any] = {}
        if self._role is not None:
            if self._on_payload:
                payload["role"] = self._role
            else:
                info["role"] = self._role
        return payload, 1.0, False, False, info

    def close(self) -> None:
        return

    @property
    def dataset_size(self) -> int:
        return 0

    @property
    def tools(self) -> list[Any]:
        return []

    @property
    def rubric_components(self) -> tuple[str, ...]:
        return ()

    @contextmanager
    def eval_mode(self) -> Iterator[None]:
        yield


# --- observation_role ------------------------------------------------------


def test_unlabelled_observation_is_a_user_turn() -> None:
    assert observation_role({"message": "hi"}, {}) == DEFAULT_OBSERVATION_ROLE
    assert DEFAULT_OBSERVATION_ROLE == "user"


@pytest.mark.parametrize("role", sorted(OBSERVATION_ROLES))
def test_info_role_is_read(role: str) -> None:
    assert observation_role({"message": "hi"}, {"role": role}) == role


def test_payload_role_is_read_when_info_is_silent() -> None:
    assert observation_role({"message": "hi", "role": "tool"}, {}) == "tool"


def test_info_role_wins_over_the_payload() -> None:
    assert observation_role({"message": "hi", "role": "tool"}, {"role": "system"}) == (
        "system"
    )


def test_assistant_is_refused() -> None:
    """An env cannot inject an assistant turn: those spans are trained."""
    with pytest.raises(ValueError, match="assistant"):
        observation_role({"message": "hi"}, {"role": "assistant"})


def test_payload_assistant_is_refused() -> None:
    with pytest.raises(ValueError, match="assistant"):
        observation_role({"role": "assistant"}, {})


def test_unknown_role_is_refused_rather_than_treated_as_user() -> None:
    with pytest.raises(ValueError, match="banana"):
        observation_role({"message": "hi"}, {"role": "banana"})


# --- the role reaches the transcript ---------------------------------------


def test_step_env_carries_the_role_alongside_the_text() -> None:
    harness = RolloutHarness(
        _RoleEnvClient("tool"),
        MiniTokenizer(),
        max_turns=3,
        apply_chat_template=False,
    )
    harness.reset()
    text, role, reward, terminated, truncated, _info = harness._step_env("go")
    assert (text, role, reward, terminated, truncated) == (
        "tool said 42",
        "tool",
        1.0,
        False,
        False,
    )


def test_payload_role_does_not_eat_the_observation_text() -> None:
    """``role`` labels the turn; it is not a second string field to guess against."""
    harness = RolloutHarness(
        _RoleEnvClient("tool", on_payload=True),
        MiniTokenizer(),
        max_turns=3,
        apply_chat_template=False,
    )
    harness.reset()
    text, role, *_rest = harness._step_env("go")
    assert (text, role) == ("tool said 42", "tool")


def test_tool_feedback_is_framed_as_a_tool_turn() -> None:
    """The rendered boundary names the env's role, not ``user``."""
    w = bare_rollout_env()
    w.apply_chat_template = True
    w.tokenizer = _RoleRecordingTokenizer()
    w._tokenize_feedback("42", "tool")
    probe = w.tokenizer.messages[-1]
    assert [m["role"] for m in probe] == ["user", "assistant", "tool"]


def test_each_role_caches_its_own_frame() -> None:
    w = bare_rollout_env()
    w.apply_chat_template = True
    w.tokenizer = _RoleRecordingTokenizer()
    w._tokenize_feedback("a", "user")
    w._tokenize_feedback("b", "tool")
    w._tokenize_feedback("c", "user")
    assert set(w._boundary_parts) == {"user", "tool"}
    # Three feedback turns, two template renders: the repeat came from cache.
    assert len(w.tokenizer.messages) == 2


def test_user_and_tool_frames_differ() -> None:
    w = bare_rollout_env()
    w.apply_chat_template = True
    w.tokenizer = _RoleRecordingTokenizer()
    as_user = w._tokenize_feedback("42", "user")
    as_tool = w._tokenize_feedback("42", "tool")
    assert not torch.equal(as_user, as_tool)


def test_chatml_fallback_names_the_role() -> None:
    """Without a usable template frame the ChatML markers still carry the role."""
    w = bare_rollout_env()
    w.apply_chat_template = True
    w.tokenizer = MiniTokenizer()  # no apply_chat_template -> fallback
    w._boundary_parts = {"tool": None}
    with pytest.warns(UserWarning, match="'tool'"):
        w._tokenize_feedback("42", "tool")


# --- system prompt ---------------------------------------------------------


def test_system_prompt_leads_the_initial_prompt() -> None:
    tokenizer = _RoleRecordingTokenizer()
    harness = RolloutHarness(
        _RoleEnvClient(None),
        tokenizer,
        max_turns=2,
        system_prompt="be terse",
    )
    harness.reset()
    assert tokenizer.messages[0] == [
        {"role": "system", "content": "be terse"},
        {"role": "user", "content": "start"},
    ]


def test_no_system_prompt_leaves_the_prompt_untouched() -> None:
    tokenizer = _RoleRecordingTokenizer()
    harness = RolloutHarness(_RoleEnvClient(None), tokenizer, max_turns=2)
    harness.reset()
    assert tokenizer.messages[0] == [{"role": "user", "content": "start"}]


def test_system_prompt_needs_the_chat_template() -> None:
    """Raw encoding has no system slot, so asking for one is a setup error."""
    with pytest.raises(ValueError, match="system_prompt"):
        RolloutHarness(
            _RoleEnvClient(None),
            MiniTokenizer(),
            apply_chat_template=False,
            system_prompt="be terse",
        )


class _PlainTextEnv:
    """In-process text env; ``system_prompt`` is an attribute, not a constructor kwarg."""

    def reset(self, seed: int | None = None) -> tuple[str, dict[str, Any]]:
        del seed
        return "start", {}

    def step(self, action: Any) -> tuple[str, float, bool, bool, dict[str, Any]]:
        del action
        return "next", 0.0, True, False, {}

    def close(self) -> None:
        return


class _EnvWithOwnSystemPrompt(_PlainTextEnv):
    def __init__(self) -> None:
        self.system_prompt = "from env"


def test_from_spec_passes_env_config_system_prompt_to_the_harness() -> None:
    """``gem.make`` rejects ``system_prompt``; the harness still has to render it."""
    tokenizer = _RoleRecordingTokenizer()
    harness = RolloutHarness.from_spec(
        _PlainTextEnv,
        {"system_prompt": "be terse"},
        tokenizer,
        max_turns=2,
    )
    harness.reset()
    assert tokenizer.messages[0] == [
        {"role": "system", "content": "be terse"},
        {"role": "user", "content": "start"},
    ]


def test_local_reads_system_prompt_from_the_env() -> None:
    tokenizer = _RoleRecordingTokenizer()
    harness = RolloutHarness.local(_EnvWithOwnSystemPrompt(), tokenizer, max_turns=2)
    harness.reset()
    assert tokenizer.messages[0][0] == {"role": "system", "content": "from env"}


def test_reset_info_system_prompt_is_adopted_when_the_harness_has_none() -> None:
    class _InfoClient(_RoleEnvClient):
        def reset(
            self, seed: int | None = None, *, row_index: int | None = None
        ) -> tuple[Any, dict[str, Any]]:
            del seed, row_index
            return {"prompt": "start"}, {"system_prompt": "from info"}

    tokenizer = _RoleRecordingTokenizer()
    harness = RolloutHarness(_InfoClient(None), tokenizer, max_turns=1)
    harness.reset()
    assert tokenizer.messages[0] == [
        {"role": "system", "content": "from info"},
        {"role": "user", "content": "start"},
    ]
