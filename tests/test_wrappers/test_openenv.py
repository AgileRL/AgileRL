"""The OpenEnv env interface: client + server, resolver, reasoning env, from_dataset.

Every LLM-training env is reached the same way — text in, text out, over the small
OpenEnv HTTP protocol. These cover both halves (``OpenEnvServer`` host +
``OpenEnvClient`` client), the socket-free ``local_transport``, the spec resolver, the
``ReasoningEnv`` dataset env, and ``RolloutEnv.from_dataset``.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch

from agilerl.llm_envs import (
    OpenEnvClient,
    OpenEnvServer,
    ReasoningEnv,
    RolloutEnv,
    local_transport,
    resolve_env,
    serve,
)
from agilerl.llm_envs.openenv import _normalize_reset, _normalize_step


class _CountingEnv:
    """Tiny multi-turn local env (the Gym/gem text contract) used across the suite."""

    def __init__(self, target: int = 2, tools: list[Any] | None = None) -> None:
        self.target = target
        self.n = 0
        self.tools = tools or []
        self.closed = False

    def reset(self, seed: int | None = None) -> tuple[str, dict[str, Any]]:
        del seed
        self.n = 0
        return "Start.", {"suffix": "Reply 'go'."}

    def step(self, action: str) -> tuple[str, float, bool, bool, dict[str, Any]]:
        self.n += 1
        reward = 1.0 if "go" in str(action).lower() else 0.0
        return f"turn {self.n}", reward, self.n >= self.target, False, {}

    def close(self) -> None:
        self.closed = True


class _MiniTok:
    """Minimal tokenizer for the non-chat-template token path."""

    pad_token_id = 0

    def __call__(self, texts: list[str], **_: Any) -> dict[str, torch.Tensor]:
        del texts
        ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}

    def decode(self, ids: Any, **_: Any) -> str:
        del ids
        return "go"

    def encode(self, text: str, **_: Any) -> list[int]:
        del text
        return [7, 7]


# --- server + client over real localhost HTTP ------------------------------
def test_server_and_client_round_trip_over_http() -> None:
    """``OpenEnvServer`` serves a local env; ``OpenEnvClient`` drives it over HTTP."""
    env = _CountingEnv(target=2)
    with OpenEnvServer(env) as server:
        assert server.base_url.startswith("http://127.0.0.1:")
        client = OpenEnvClient(base_url=server.base_url)
        prompt, info = client.reset()
        # The env's info['suffix'] is folded into the prompt (OpenEnv drops obs metadata).
        assert prompt == "Start.\nReply 'go'." and info == {}
        obs1, r1, term1, trunc1, _ = client.step("go now")
        obs2, r2, term2, _, _ = client.step("stop")
        assert (obs1, r1, term1, trunc1) == ("turn 1", 1.0, False, False)
        assert (obs2, r2, term2) == ("turn 2", 0.0, True)
        client.close()
        assert env.closed is True


def test_serve_helper_starts_immediately() -> None:
    """``serve`` returns an already-running server."""
    server = serve(_CountingEnv())
    try:
        assert (
            OpenEnvClient(base_url=server.base_url).reset()[0] == "Start.\nReply 'go'."
        )
    finally:
        server.stop()


def test_info_reports_dataset_size_and_tools() -> None:
    """``/state`` carries dataset size and advertised tool schemas to the client."""
    tools = [{"type": "function", "function": {"name": "calc", "parameters": {}}}]
    server = serve(_CountingEnv(tools=tools))
    try:
        client = OpenEnvClient(base_url=server.base_url)
        assert client.dataset_size == 0  # _CountingEnv has no dataset_size
        assert client.tools == tools
    finally:
        server.stop()


def test_base_url_or_transport_required() -> None:
    """The client needs a URL or an injected transport."""
    with pytest.raises(ValueError, match="base_url"):
        OpenEnvClient()


# --- local_transport: socket-free parity -----------------------------------
def test_local_transport_matches_http() -> None:
    """``local_transport`` drives the same ``GymEnvironment`` as the server, no socket."""
    client = OpenEnvClient(transport=local_transport(_CountingEnv(target=1)))
    prompt, _ = client.reset()
    obs, reward, terminated, _, _ = client.step("go")
    assert prompt == "Start.\nReply 'go'." and (obs, reward, terminated) == (
        "turn 1",
        1.0,
        True,
    )


def test_local_transport_unknown_route_raises() -> None:
    """An unknown route is an error (surfaced as a 500 over HTTP)."""
    transport = local_transport(_CountingEnv())
    with pytest.raises(ValueError, match="unknown OpenEnv route"):
        transport("/nope", {})


# --- gym-tuple normalisation -----------------------------------------------
def test_normalize_step_accepts_four_tuple() -> None:
    """A legacy 4-tuple ``(obs, reward, done, info)`` fills ``truncated=False``."""
    assert _normalize_step(("o", 1.0, True, {"k": 1})) == (
        "o",
        1.0,
        True,
        False,
        {"k": 1},
    )


def test_normalize_step_rejects_non_tuple() -> None:
    with pytest.raises(TypeError, match="must return a tuple"):
        _normalize_step("not a tuple")


def test_normalize_reset_accepts_bare_observation() -> None:
    """An env returning just an observation gets an empty info dict."""
    assert _normalize_reset("obs") == ("obs", {})


# --- ReasoningEnv ----------------------------------------------------------
def _reasoning() -> ReasoningEnv:
    return ReasoningEnv(
        questions=["q0", "q1", "q2"],
        answers=["a0", "a1", "a2"],
        reward_fn=lambda c, a, q: 1.0 if c == a else 0.0,
        prompt_builder=lambda q: f"P:{q}",
        test_questions=["t0"],
        test_answers=["ta0"],
    )


def test_reasoning_env_row_index_and_scoring() -> None:
    env = _reasoning()
    assert env.dataset_size == 3
    prompt, info = env.reset(row_index=1)
    assert prompt == "P:q1" and info == {}
    _, reward, terminated, truncated, _ = env.step("a1")
    assert reward == 1.0 and terminated is True and truncated is False
    assert env.step("a1")  # callable again after re-bind
    env.reset(row_index=2)
    assert env.step("wrong")[1] == 0.0


def test_reasoning_env_eval_split_and_cursor() -> None:
    env = _reasoning()
    # Standalone cursor walks the active split sequentially.
    assert env.reset()[0] == "P:q0"
    assert env.reset()[0] == "P:q1"
    # evaluation=True serves the held-out split (and resets the cursor).
    assert env.reset(evaluation=True)[0] == "P:t0"
    # The evaluation_mode attribute is honoured when reset omits the flag.
    env.evaluation_mode = True
    assert env.reset(row_index=0)[0] == "P:t0"


def test_reasoning_env_defaults_are_lenient() -> None:
    """Missing reward_fn/prompt_builder default to zero-reward / identity."""
    env = ReasoningEnv(questions=["x"], answers=["y"])
    assert env.reset(row_index=0)[0] == "x"
    assert env.step("anything")[1] == 0.0


# --- RolloutEnv.from_dataset (reasoning over local_transport) ---------------
def test_from_dataset_dataset_size_and_row_pinning() -> None:
    a = RolloutEnv.from_dataset(
        ["q0", "q1", "q2"],
        ["a0", "a1", "a2"],
        lambda c, a_, q: 0.0,
        _MiniTok(),
        prompt_builder=lambda q: f"P:{q}",
        apply_chat_template=False,
    )
    b = RolloutEnv.from_dataset(
        ["q0", "q1", "q2"],
        ["a0", "a1", "a2"],
        lambda c, a_, q: 0.0,
        _MiniTok(),
        prompt_builder=lambda q: f"P:{q}",
        apply_chat_template=False,
    )
    assert a.dataset_size == 3
    # Same row_index -> same prompt (a GRPO group sees one prompt).
    a.reset(row_index=2)
    b.reset(row_index=2)
    assert a._prompt_text == b._prompt_text == "P:q2"


def test_from_dataset_eval_mode_routes_split() -> None:
    env = RolloutEnv.from_dataset(
        ["q0"],
        ["a0"],
        lambda c, a, q: 0.0,
        _MiniTok(),
        prompt_builder=lambda q: f"P:{q}",
        test_questions=["t0"],
        test_answers=["ta0"],
        apply_chat_template=False,
    )
    env.reset(row_index=0)
    assert env._prompt_text == "P:q0"
    with env.eval_mode():
        env.reset(row_index=0)
        assert env._prompt_text == "P:t0"
    env.reset(row_index=0)
    assert env._prompt_text == "P:q0"


def test_from_dataset_full_step_loop_scores_and_terminates() -> None:
    env = RolloutEnv.from_dataset(
        ["q0"],
        ["a0"],
        lambda c, a, q: 1.0 if c == "go" else 0.0,
        _MiniTok(),
        prompt_builder=lambda q: q,
        max_turns=1,
        apply_chat_template=False,
    )
    env.reset(row_index=0)
    _, reward, terminated, _, _ = env.step(
        torch.tensor([[1, 2, 3, 7, 7]], dtype=torch.long)
    )
    assert reward == 1.0 and terminated is True  # decode() -> "go"
    full_ids, action_mask, turn_ids, turn_rewards, _ = env.get_episode_data()
    assert turn_rewards.tolist() == [1.0]


# --- RolloutEnv(url) over a real server ------------------------------------
def test_rollout_env_drives_url_and_applies_suffix() -> None:
    server = serve(_CountingEnv())
    try:
        env = RolloutEnv(
            server.base_url, _MiniTok(), max_turns=2, apply_chat_template=False
        )
        env.reset()
        # reset prompt + info["suffix"] are stitched by _format_obs.
        assert env._prompt_text == "Start.\nReply 'go'."
    finally:
        server.stop()


# --- resolve_env -----------------------------------------------------------
def test_resolve_env_url_is_used_raw() -> None:
    url, server = resolve_env("https://example.invalid/openenv")
    assert url == "https://example.invalid/openenv" and server is None


def test_resolve_env_entrypoint_hosts_locally() -> None:
    url, server = resolve_env(
        "agilerl.llm_envs.rollout_env:ReasoningEnv",
        {"questions": ["q"], "answers": ["a"], "prompt_builder": (lambda q: f"P:{q}")},
    )
    try:
        client = OpenEnvClient(base_url=url)
        assert client.reset(row_index=0)[0] == "P:q"
        assert client.dataset_size == 1
    finally:
        server.stop()


def test_resolve_env_disk_path_entrypoint(tmp_path: Any) -> None:
    module = tmp_path / "disk_env.py"
    module.write_text(
        "class DiskEnv:\n"
        "    def reset(self, seed=None):\n"
        "        return 'from disk', {}\n"
        "    def step(self, action):\n"
        "        return '', 0.0, True, False, {}\n"
    )
    url, server = resolve_env(f"{module}:DiskEnv")
    try:
        assert OpenEnvClient(base_url=url).reset()[0] == "from disk"
    finally:
        server.stop()


def test_resolve_env_unknown_spec_raises() -> None:
    # Not a URL and not a 'module:Class' entrypoint (no colon) -> a clear error.
    with pytest.raises(ValueError, match="neither a URL nor"):
        resolve_env("not-a-url-or-entrypoint")
