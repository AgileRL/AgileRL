"""The OpenEnv env interface: client + server, resolver, from_dataset.

Every LLM-training env is reached the same way — text in, text out, over the small
OpenEnv HTTP protocol. These cover both halves (``OpenEnvServer`` host +
``OpenEnvClient`` client over httpx), the spec resolver, and
``RolloutEnv.from_dataset`` (which hosts the prompt dataset on its own server).
"""

from __future__ import annotations

from typing import Any

import pytest
import torch

from agilerl.llm_envs import (
    BatchRolloutEnv,
    OpenEnvClient,
    OpenEnvServer,
    OpenEnvWrapper,
    RolloutEnv,
    resolve_env,
    serve,
)
from agilerl.llm_envs.openenv import _name_from_spec, _normalize_reset, _normalize_step


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
        assert prompt == "Start.\nReply 'go'."
        assert info == {}
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


def test_base_url_required() -> None:
    """The client needs a non-empty base URL."""
    with pytest.raises(ValueError, match="base_url"):
        OpenEnvClient("")


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


# --- env identity: the served env's real name in OpenEnv metadata ----------
def test_openenv_wrapper_metadata_reports_real_env_name() -> None:
    """The wrapped env is identified by its own name, not ``OpenEnvWrapper``."""
    assert OpenEnvWrapper(_CountingEnv()).get_metadata().name == "_CountingEnv"
    assert (
        OpenEnvWrapper(_CountingEnv(), env_name="counting").get_metadata().name
        == "counting"
    )


def test_name_from_spec_takes_trailing_identifier() -> None:
    """An env spec's label is the trailing identifier of the entrypoint / path."""
    assert _name_from_spec("game:GuessTheNumber-v0") == "GuessTheNumber-v0"
    assert _name_from_spec("/pkg/file.py:Env") == "Env"
    assert _name_from_spec("plainname") == "plainname"


# --- RolloutEnv.from_dataset (reasoning over a hosted server) ---------------
def test_from_dataset_cursor_and_eval_split() -> None:
    """reset() with no row walks the split; evaluation routes to the held-out split."""
    env = RolloutEnv.from_dataset(
        ["q0", "q1", "q2"],
        ["a0", "a1", "a2"],
        lambda c, a, q: 0.0,
        _MiniTok(),
        prompt_builder=lambda q: f"P:{q}",
        test_questions=["t0"],
        test_answers=["ta0"],
        apply_chat_template=False,
    )
    try:
        # Standalone cursor walks the active split sequentially.
        env.reset()
        assert env._prompt_text == "P:q0"
        env.reset()
        assert env._prompt_text == "P:q1"
        # evaluation routes to the held-out split (and resets the cursor).
        with env.eval_mode():
            env.reset()
            assert env._prompt_text == "P:t0"
        # The evaluation_mode flag is honoured when reset omits the row.
        env.evaluation_mode = True
        env.reset(row_index=0)
        assert env._prompt_text == "P:t0"
    finally:
        env.close()


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
    try:
        assert a.dataset_size == 3
        # Same row_index -> same prompt (a GRPO group sees one prompt).
        a.reset(row_index=2)
        b.reset(row_index=2)
        assert a._prompt_text == b._prompt_text == "P:q2"
    finally:
        a.close()
        b.close()


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
    try:
        env.reset(row_index=0)
        assert env._prompt_text == "P:q0"
        with env.eval_mode():
            env.reset(row_index=0)
            assert env._prompt_text == "P:t0"
        env.reset(row_index=0)
        assert env._prompt_text == "P:q0"
    finally:
        env.close()


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
    try:
        env.reset(row_index=0)
        _, reward, terminated, _, _ = env.step(
            torch.tensor([[1, 2, 3, 7, 7]], dtype=torch.long)
        )
        assert reward == 1.0  # decode() -> "go"
        assert terminated is True
        _full_ids, _action_mask, _turn_ids, turn_rewards, _ = env.get_episode_data()
        assert turn_rewards.tolist() == [1.0]
    finally:
        env.close()


# --- serving: one OpenEnv server instance per rollout ----------------------
def test_serving_hosts_owns_and_stops_its_server() -> None:
    """``serving`` hosts a fresh env on its own server and stops it on close."""
    env = RolloutEnv.serving(_CountingEnv, _MiniTok(), apply_chat_template=False)
    assert env._owned_server is not None
    env.reset()  # reachable over real HTTP
    assert env._prompt_text == "Start.\nReply 'go'."
    env.close()
    assert env._owned_server is None  # server stopped + released


def test_from_dataset_serve_hosts_a_per_instance_server() -> None:
    """``from_dataset`` hosts the prompt dataset on its own per-instance server."""
    env = RolloutEnv.from_dataset(
        ["q0"],
        ["a0"],
        lambda c, a, q: 0.0,
        _MiniTok(),
        prompt_builder=lambda q: f"P:{q}",
        apply_chat_template=False,
    )
    assert env._owned_server is not None
    assert env.dataset_size == 1
    env.reset(row_index=0)
    assert env._prompt_text == "P:q0"
    env.close()
    assert env._owned_server is None


def test_batch_serving_factory_gives_one_server_per_env() -> None:
    """A serving ``env_factory`` gives BatchRolloutEnv one isolated server per env.

    The state race (one stateful server can't serve concurrent rollouts) is avoided:
    each of the ``batch_size * group_size`` envs binds a distinct server URL, and
    ``close`` tears them all down.
    """
    batch = BatchRolloutEnv(
        lambda **_: RolloutEnv.serving(
            _CountingEnv, _MiniTok(), apply_chat_template=False
        ),
        batch_size=2,
        group_size=1,
    )
    batch.reset(seed=0)
    urls = [env._owned_server.base_url for env in batch.envs]
    assert len(batch.envs) == 2
    assert len(set(urls)) == 2  # distinct server instances
    batch.close()
    assert all(env._owned_server is None for env in batch.envs)


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
    assert url == "https://example.invalid/openenv"
    assert server is None


def test_resolve_env_entrypoint_hosts_locally() -> None:
    url, server = resolve_env(
        "agilerl.llm_envs.rollout_env:_PromptDatasetEnv",
        {
            "questions": ["q"],
            "answers": ["a"],
            "reward_fn": (lambda c, a, q: 0.0),
            "prompt_builder": (lambda q: f"P:{q}"),
        },
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
