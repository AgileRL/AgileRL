# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""The OpenEnv env interface: client + server, resolver.

Every LLM-training env is reached the same way — text in, text out, over the
OpenEnv protocol. These cover both halves (``OpenEnvServer`` host +
``OpenEnvSessionClient`` over a WebSocket session) and the spec resolver.
"""

from __future__ import annotations

import logging
import socket
import threading
import time
from types import SimpleNamespace
from typing import Any

import pytest
import torch
import websockets.exceptions

from agilerl.llm_envs import (
    BatchRolloutEnv,
    RolloutEnv,
    TaskAssigner,
)
from agilerl.llm_envs import openenv as openenv_module
from agilerl.llm_envs import openenv_server as openenv_server_module
from agilerl.llm_envs.openenv import (
    LocalEnvClient,
    OpenEnvServer,
    OpenEnvSessionClient,
    OpenEnvWrapper,
    _observation_text,
    resolve_env,
)
from agilerl.llm_envs.openenv_server import (
    _name_from_spec,
    _normalize_reset,
    _normalize_step,
)
from agilerl.llm_envs.prompt_dataset import PromptDatasetEnv
from agilerl.llm_envs.rubrics import reward_fn_to_rubric
from tests.helpers.rollout_doubles import MiniTokenizer


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
        return "Start.\nReply 'go'.", {}

    def step(self, action: str) -> tuple[str, float, bool, bool, dict[str, Any]]:
        self.n += 1
        reward = 1.0 if "go" in str(action).lower() else 0.0
        return f"turn {self.n}", reward, self.n >= self.target, False, {}

    def close(self) -> None:
        self.closed = True


class _RowDatasetEnv:
    """A tiny dataset-backed local env (the text contract), used in-process and by spec.

    ``reset`` returns a row prompt plus a ``suffix`` info the client folds in;
    ``step`` rewards an action containing ``"x"``.
    """

    def __init__(self, n: int = 3, prefix: str = "row") -> None:
        self._n = n
        self._prefix = prefix
        self._q = ""
        self.tools: list[Any] = []

    @property
    def dataset_size(self) -> int:
        return self._n

    def reset(
        self,
        seed: int | None = None,
        *,
        row_index: int | None = None,
        evaluation: bool | None = None,
    ) -> tuple[str, dict[str, Any]]:
        del seed
        idx = (row_index or 0) % self._n
        self._q = f"{'eval' if evaluation else self._prefix}{idx}"
        return f"{self._q}\nanswer:", {}

    def step(self, action: str) -> tuple[str, float, bool, bool, dict[str, Any]]:
        return "", (1.0 if "x" in str(action) else 0.0), True, False, {}

    def close(self) -> None:
        pass


# --- server + client over real localhost HTTP ------------------------------
def test_server_and_client_round_trip_over_http() -> None:
    """``OpenEnvServer`` serves a local env; ``OpenEnvSessionClient`` drives it over HTTP."""
    env = _CountingEnv(target=2)
    with OpenEnvServer(env) as server:
        assert server.base_url.startswith("http://127.0.0.1:")
        client = OpenEnvSessionClient(base_url=server.base_url)
        prompt, info = client.reset()
        assert prompt == "Start.\nReply 'go'."
        assert info == {}
        obs1, r1, term1, trunc1, _ = client.step("go now")
        obs2, r2, term2, _, _ = client.step("stop")
        assert (obs1, r1, term1, trunc1) == ("turn 1", 1.0, False, False)
        assert (obs2, r2, term2) == ("turn 2", 0.0, True)
        client.close()
        # not closed per request or on client close — only once when the server stops
        assert env.closed is False
    assert env.closed is True


def test_serve_helper_starts_immediately() -> None:
    """``serve`` returns an already-running server."""
    server = OpenEnvServer(_CountingEnv()).start()
    try:
        assert (
            OpenEnvSessionClient(base_url=server.base_url).reset()[0]
            == "Start.\nReply 'go'."
        )
    finally:
        server.stop()


def test_info_reports_dataset_size_and_tools() -> None:
    """``/state`` carries dataset size and advertised tool schemas to the client."""
    tools = [{"type": "function", "function": {"name": "calc", "parameters": {}}}]
    server = OpenEnvServer(_CountingEnv(tools=tools)).start()
    try:
        client = OpenEnvSessionClient(base_url=server.base_url)
        assert client.dataset_size == 0  # _CountingEnv has no dataset_size
        assert client.tools == tools
    finally:
        server.stop()


def test_base_url_required() -> None:
    """The client needs a non-empty base URL."""
    with pytest.raises(ValueError, match="base_url"):
        OpenEnvSessionClient("")


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


def test_normalize_reset_accepts_singleton_tuple() -> None:
    """A 1-tuple ``(obs,)`` reset return normalizes to empty info."""
    assert _normalize_reset(("obs",)) == ("obs", {})


def test_normalize_step_rejects_unsupported_tuple_length() -> None:
    """Only 4/5-tuples are accepted from ``env.step``."""
    with pytest.raises(ValueError, match="expected 4 or 5"):
        _normalize_step(("o", 1.0, True))


# --- observation rendering ---------------------------------------------------
@pytest.mark.parametrize(
    ("obs", "expected"),
    [
        ("plain text", "plain text"),
        ({"result": {"content": [{"text": "a"}, {"text": "b"}]}}, "a\nb"),
        ({"result": {"data": "tool data"}}, "tool data"),
        ({"prompt": "p"}, "p"),
        ({"text": "t"}, "t"),
        ({"message": "m"}, "m"),
        ({"observation": "o"}, "o"),
        ({"error": "boom"}, "Error: boom"),
        ({"unknown": 1}, ""),
        (123, ""),
    ],
)
def test_observation_text_supports_openenv_and_mcp_shapes(
    obs: Any, expected: str
) -> None:
    """``_observation_text`` handles plain text, MCP blocks, and fallback keys."""
    assert _observation_text(obs) == expected


# --- env identity: the hosted env's real name in OpenEnv metadata ----------
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
    assert _name_from_spec(r"C:\pkg\file.py:Env") == "Env"
    assert _name_from_spec("plainname") == "plainname"


# --- RolloutEnv(url) over a real server ------------------------------------
def test_rollout_env_drives_url_and_applies_suffix() -> None:
    server = OpenEnvServer(_CountingEnv()).start()
    try:
        env = RolloutEnv(
            server.base_url, MiniTokenizer(), max_turns=2, apply_chat_template=False
        )
        env.reset()
        # The env returns the full observation text; no info folding.
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
        "tests.helpers.rollout_doubles:TinyDatasetEnv",
        {
            "questions": ["q"],
            "answers": ["a"],
            "reward_fn": (lambda c, a, q: 0.0),
            "prompt_builder": (lambda q: f"P:{q}"),
        },
    )
    try:
        client = OpenEnvSessionClient(base_url=url)
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
        assert OpenEnvSessionClient(base_url=url).reset()[0] == "from disk"
    finally:
        server.stop()


def test_resolve_env_unknown_spec_raises() -> None:
    # Not a URL and not a 'module:Class' entrypoint (no colon) -> a clear error.
    with pytest.raises(ValueError, match="neither a URL"):
        resolve_env("not-a-url-or-entrypoint")


def test_resolve_env_capacity_hosts_a_fresh_env_per_session() -> None:
    """``max_concurrent_envs`` serves concurrent sessions, each with its own env."""
    url, server = resolve_env(
        "tests.helpers.rollout_doubles:TinyDatasetEnv",
        {
            "questions": ["q0", "q1"],
            "answers": ["a0", "a1"],
            "reward_fn": (lambda c, a, q: 0.0),
            "prompt_builder": (lambda q: f"P:{q}"),
        },
        max_concurrent_envs=2,
    )
    try:
        first = OpenEnvSessionClient(base_url=url)
        second = OpenEnvSessionClient(base_url=url)
        # Both sessions live at once, and each holds its own row rather than
        # sharing one env's cursor.
        assert first.reset(row_index=0)[0] == "P:q0"
        assert second.reset(row_index=1)[0] == "P:q1"
        assert first.step("")[1] == 0.0
    finally:
        server.stop()


def test_resolve_env_advertises_a_routable_host() -> None:
    """A server bound to all interfaces advertises the address clients should dial."""
    url, server = resolve_env(
        "tests.helpers.rollout_doubles:TinyDatasetEnv",
        {
            "questions": ["q"],
            "answers": ["a"],
            "reward_fn": (lambda c, a, q: 0.0),
            "prompt_builder": (lambda q: f"P:{q}"),
        },
        host="0.0.0.0",
        advertise_host="127.0.0.1",
    )
    try:
        assert url == f"http://127.0.0.1:{server.port}"
        assert OpenEnvSessionClient(base_url=url).reset(row_index=0)[0] == "P:q"
    finally:
        server.stop()


def test_port_and_base_url_require_a_running_server() -> None:
    """Both accessors fail loudly before ``start`` rather than reporting port 0."""
    server = OpenEnvServer(_CountingEnv())
    with pytest.raises(RuntimeError, match="not running"):
        _ = server.port
    with pytest.raises(RuntimeError, match="not running"):
        _ = server.base_url


# --- LocalEnvClient: drive a local env directly, no HTTP -------------------
class TestLocalEnvClient:
    """In-process env client (no HTTP): folds suffix, routes eval split, forwards close."""

    def test_drives_env_directly(self) -> None:
        """``LocalEnvClient`` calls the env in-process and folds the suffix like the server."""
        client = LocalEnvClient(_RowDatasetEnv())
        assert client.dataset_size == 3
        prompt, info = client.reset(row_index=1)
        assert prompt == "row1\nanswer:"  # the env returns the full text
        assert info == {}
        assert client.step("x marks it") == ("", 1.0, True, False, {})

    def test_eval_mode_routes_split(self) -> None:
        """``eval_mode`` flags reset for the held-out split, then restores."""
        client = LocalEnvClient(_RowDatasetEnv())
        with client.eval_mode():
            assert client.reset(row_index=0)[0] == "eval0\nanswer:"
        assert client.reset(row_index=0)[0] == "row0\nanswer:"

    def test_close_is_forwarded(self) -> None:
        """``close`` reaches the wrapped env when it has one."""
        closed = {"n": 0}

        class _Env:
            def reset(self, seed: int | None = None) -> tuple[str, dict]:
                del seed
                return "p", {}

            def step(self, action: str) -> tuple[str, float, bool, bool, dict]:
                del action
                return "", 0.0, True, False, {}

            def close(self) -> None:
                closed["n"] += 1

        LocalEnvClient(_Env()).close()
        assert closed["n"] == 1


# --- RolloutEnv.local / from_spec: picking the env client backend ----------
class TestRolloutEnvLocal:
    """``RolloutEnv.local`` runs the env in-process via a ``LocalEnvClient``."""

    def test_uses_in_process_env_client(self) -> None:
        env = RolloutEnv.local(
            _RowDatasetEnv(), MiniTokenizer(), max_turns=1, apply_chat_template=False
        )
        assert isinstance(env._env_client, LocalEnvClient)
        obs, _ = env.reset(row_index=2)
        assert "input_ids" in obs
        assert env.dataset_size == 3
        env.close()


class TestRolloutEnvFromSpec:
    """``RolloutEnv.from_spec`` routes a URL to HTTP and an entrypoint to in-process."""

    def test_url_builds_http_client(self) -> None:
        """A URL spec routes to the HTTP ``OpenEnvSessionClient`` env client."""
        env = RolloutEnv.from_spec("http://localhost:0", None, MiniTokenizer())
        assert isinstance(env._env_client, OpenEnvSessionClient)
        env.close()

    def test_entrypoint_runs_in_process(self) -> None:
        """A ``path.py:Class`` entrypoint loads + drives the env in-process (no HTTP)."""
        spec = f"{__file__}:_RowDatasetEnv"
        env = RolloutEnv.from_spec(
            spec,
            {"prefix": "Q"},
            MiniTokenizer(),
            max_turns=1,
            apply_chat_template=False,
        )
        assert isinstance(env._env_client, LocalEnvClient)
        obs, _ = env.reset(row_index=0)
        assert "input_ids" in obs
        env.close()


# --- TaskAssigner: one task per GRPO group ----------------------------------
class TestTaskAssigner:
    """Epoch-reshuffled dataset rows, one task shared across each group."""

    def test_reshuffles_each_epoch(self) -> None:
        """Each epoch is a fresh full permutation of the dataset rows."""
        assigner = TaskAssigner(4, seed=0)
        rows = [assigner.next_row() for _ in range(8)]
        assert sorted(rows[:4]) == [0, 1, 2, 3]
        assert sorted(rows[4:]) == [0, 1, 2, 3]

    def test_assign_is_group_contiguous(self) -> None:
        """``assign`` gives one ``(seed, row)`` per slot, shared within each group."""
        assigned = TaskAssigner(4, seed=0).assign(2, 3, base_seed=100)
        assert len(assigned) == 6
        assert assigned[0] == assigned[1] == assigned[2]  # group 0 shares its task
        assert assigned[3] == assigned[4] == assigned[5]  # group 1 shares
        assert assigned[0][0] == 100  # seed = base_seed + batch item 0
        assert assigned[3][0] == 101  # seed = base_seed + batch item 1

    def test_procedural_env_yields_no_rows(self) -> None:
        """A procedural env (``dataset_size=0``) gets ``row_index=None``; only seeds pin tasks."""
        assert TaskAssigner(0).assign(2, 2) == [(None, None)] * 4

    def test_ranks_shard_disjointly_and_cover_the_dataset(self) -> None:
        """Two ranks emit disjoint contiguous row blocks covering every row — no process group."""
        rank0 = TaskAssigner(10, seed=0, rank=0, world_size=2)
        rank1 = TaskAssigner(10, seed=0, rank=1, world_size=2)
        rows0 = {rank0.next_row() for _ in range(5)}
        rows1 = {rank1.next_row() for _ in range(5)}
        assert rows0 == set(range(5))
        assert rows1 == set(range(5, 10))
        assert rank0.num_epochs == 0
        rank0.next_row()
        assert rank0.num_epochs == 1  # epochs count passes over the rank's shard

    def test_uneven_shards_split_without_overlap(self) -> None:
        """A remainder row lands in exactly one rank's block."""
        shards = [TaskAssigner(7, seed=0, rank=r, world_size=3) for r in range(3)]
        seen: list[int] = []
        for assigner, size in zip(shards, (2, 2, 3), strict=True):
            rows = [assigner.next_row() for _ in range(size)]
            assert sorted(rows) == sorted(set(rows))
            seen.extend(rows)
        assert sorted(seen) == list(range(7))

    def test_empty_shard_is_rejected(self) -> None:
        """A rank with no rows fails at construction, not in an infinite draw loop."""
        with pytest.raises(ValueError, match="empty shard"):
            TaskAssigner(1, seed=0, rank=0, world_size=2)

    def test_procedural_env_ignores_sharding(self) -> None:
        """``dataset_size=0`` never draws rows, so any rank/world_size is fine."""
        assert TaskAssigner(0, rank=3, world_size=8).assign(1, 2) == [(None, None)] * 2

    def test_invalid_rank_or_world_size_raises(self) -> None:
        with pytest.raises(ValueError, match="world_size"):
            TaskAssigner(4, seed=0, rank=0, world_size=0)
        with pytest.raises(ValueError, match="rank"):
            TaskAssigner(4, seed=0, rank=4, world_size=4)



# --- /state strictness + MCP tools/list fallback ----------------------------
class _StubSync:
    """Stand-in for OpenEnv's ``SyncEnvClient`` — records reset/step, scripts errors."""

    def __init__(
        self,
        state: Any = None,
        state_error: Exception | None = None,
        step_errors: list[Exception] | None = None,
        reset_errors: list[Exception] | None = None,
    ) -> None:
        self.reset_calls: list[dict[str, Any]] = []
        self.step_calls: list[Any] = []
        self._state = state if state is not None else {}
        self._state_error = state_error
        self._step_errors = list(step_errors or [])
        self._reset_errors = list(reset_errors or [])

    def connect(self) -> None:
        return None

    def reset(self, **kwargs: Any) -> Any:
        if self._reset_errors:
            raise self._reset_errors.pop(0)
        self.reset_calls.append(kwargs)
        return SimpleNamespace(observation="prompt", reward=None, done=False)

    def step(self, action: Any) -> Any:
        self.step_calls.append(action)
        if self._step_errors:
            raise self._step_errors.pop(0)
        return SimpleNamespace(observation="hi", reward=1.0, done=True)

    def state(self) -> Any:
        if self._state_error is not None:
            raise self._state_error
        return self._state


def _session_client(**stub_kwargs: Any) -> OpenEnvSessionClient:
    """A session client whose transport is a :class:`_StubSync` (lazy connect: no I/O)."""
    client = OpenEnvSessionClient("http://stub.invalid")
    client._sync = _StubSync(**stub_kwargs)
    return client


def test_session_reset_forwards_eval_and_positive_seed_only() -> None:
    """``reset`` forwards row/eval flags to the session and drops negative seeds."""
    client = _session_client()
    with client.eval_mode():
        prompt, info = client.reset(seed=-2, row_index=3)
    assert prompt == "prompt"
    assert info == {}
    assert client._sync.reset_calls[-1] == {"row_index": 3, "evaluation": True}

    client.reset(seed=5)
    assert client._sync.reset_calls[-1] == {"seed": 5}


def test_session_state_is_best_effort() -> None:
    """A session that can't read state advertises no dataset and no tools."""
    client = _session_client(state_error=RuntimeError("no state channel"))
    assert client.dataset_size == 0
    assert client.tools == []


def test_session_state_reports_dataset_size_and_tools() -> None:
    """``dataset_size`` / ``tools`` come from the env's OpenEnv state, cached."""
    client = _session_client(state={"dataset_size": 5, "tools": [{"name": "t"}]})
    assert client.dataset_size == 5
    assert client.tools == [{"name": "t"}]


def test_session_state_reports_rubric_components() -> None:
    client = _session_client(state={"rubric_components": ["fmt", "correct"]})
    assert client.rubric_components == ("fmt", "correct")


def test_session_step_forwards_metadata_and_obs_rubric_scores() -> None:
    """Result metadata merges into info; obs-level rubric_scores fill gaps."""
    client = _session_client()

    def _step(action: Any) -> Any:
        del action
        return SimpleNamespace(
            observation={
                "prompt": "hi",
                "rubric_scores": {"fmt": 1.0},
                "metadata": {"extra": 2},
            },
            reward=1.0,
            done=True,
            metadata={"from_result": True},
        )

    client._sync.step = _step  # type: ignore[method-assign]
    _obs, _reward, _term, _trunc, info = client.step("go")
    assert info["from_result"] is True
    assert info["extra"] == 2
    assert info["rubric_scores"] == {"fmt": 1.0}


# --- broken mid-episode, re-dialled at the next reset ------------------------
def test_websocket_error_breaks_the_episode_and_reset_redials() -> None:
    """A dropped WebSocket propagates, later steps fail fast, the next reset re-dials."""
    client = _session_client(
        step_errors=[websockets.exceptions.WebSocketException("dropped")]
    )
    with pytest.raises(websockets.exceptions.WebSocketException, match="dropped"):
        client.step("go")
    with pytest.raises(RuntimeError, match="broken"):
        client.step("go")
    fresh = _StubSync()
    client._build_session = lambda: fresh  # type: ignore[method-assign]
    prompt, _info = client.reset()
    assert prompt == "prompt"
    assert client._sync is fresh
    assert client._redials == 1


def test_timeout_breaks_the_episode_and_reset_redials() -> None:
    """A per-message timeout is a transport error: the episode dies, reset re-dials."""
    client = _session_client(step_errors=[TimeoutError("too slow")])
    with pytest.raises(TimeoutError, match="too slow"):
        client.step("go")
    with pytest.raises(RuntimeError, match="broken"):
        client.step("go")
    client._build_session = lambda: _StubSync()  # type: ignore[method-assign]
    prompt, _info = client.reset()
    assert prompt == "prompt"


def test_redial_resolves_the_url_provider_again(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A callable ``base_url`` is re-resolved per dial, finding a moved host."""
    import openenv.core

    urls = iter(["http://host-a:1", "http://host-b:2"])
    dialled: list[str] = []
    stubs = [
        _StubSync(step_errors=[websockets.exceptions.WebSocketException("dropped")]),
        _StubSync(),
    ]

    class _CapturingClient:
        def __init__(self, **kwargs: Any) -> None:
            dialled.append(kwargs["base_url"])

        def sync(self) -> Any:
            return stubs[len(dialled) - 1]

    monkeypatch.setattr(openenv.core, "GenericEnvClient", _CapturingClient)
    client = OpenEnvSessionClient(lambda: next(urls))
    client.reset()
    with pytest.raises(websockets.exceptions.WebSocketException):
        client.step("go")
    prompt, _info = client.reset()
    assert prompt == "prompt"
    assert dialled == ["http://host-a:1", "http://host-b:2"]


def test_server_error_frame_leaves_the_session_usable() -> None:
    """A plain server error frame propagates without breaking the session."""
    client = _session_client(
        step_errors=[RuntimeError("Server error: unsupported (code: UNKNOWN)")]
    )
    with pytest.raises(RuntimeError, match="UNKNOWN"):
        client.step("go")
    obs, reward, terminated, _truncated, _info = client.step("go")
    assert (obs, reward, terminated) == ("hi", 1.0, True)


# --- _fetch_state: transport errors propagate, server errors degrade ---------
def test_state_transport_error_redials_once_then_propagates() -> None:
    """A transport error on ``state`` re-dials once (pre-episode); twice propagates."""
    client = _session_client(
        state_error=RuntimeError("Server error: capacity (code: CAPACITY_REACHED)")
    )
    fresh = _StubSync(state={"dataset_size": 3})
    client._build_session = lambda: fresh  # type: ignore[method-assign]
    assert client.dataset_size == 3
    assert client._redials == 1

    stubborn = _session_client(
        state_error=RuntimeError("Server error: capacity (code: CAPACITY_REACHED)")
    )
    stubborn._build_session = lambda: _StubSync(  # type: ignore[method-assign]
        state_error=RuntimeError("Server error: capacity (code: CAPACITY_REACHED)")
    )
    with pytest.raises(RuntimeError, match="CAPACITY_REACHED"):
        _ = stubborn.tools


def test_state_server_error_degrades_with_a_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """An unsupported ``state`` message degrades to no dataset / no tools, warns,
    and leaves the session usable.
    """
    client = _session_client(
        state_error=RuntimeError("Server error: unsupported (code: UNKNOWN)")
    )
    with caplog.at_level(logging.WARNING, logger="agilerl.llm_envs.openenv"):
        assert client.tools == []
        assert client.dataset_size == 0
    assert "could not read state" in caplog.text
    assert client.reset()[0] == "prompt"


# --- timeout pass-through to OpenEnv's client --------------------------------
def test_timeout_passes_through_to_the_openenv_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``timeout_s`` reaches ``GenericEnvClient`` verbatim; ``None`` stays unbounded."""
    import openenv.core

    captured: list[dict[str, Any]] = []

    class _CapturingClient:
        def __init__(self, **kwargs: Any) -> None:
            captured.append(kwargs)

        def sync(self) -> Any:
            return _StubSync()

    monkeypatch.setattr(openenv.core, "GenericEnvClient", _CapturingClient)
    # The session dials lazily, so the client is only built on first use.
    OpenEnvSessionClient("http://stub.invalid", timeout_s=None).reset()
    OpenEnvSessionClient("http://stub.invalid", timeout_s=12.5).reset()
    assert captured[0]["message_timeout_s"] is None
    assert captured[1]["message_timeout_s"] == 12.5


# --- truncated round-trips over HTTP ----------------------------------------
class _TruncatingEnv:
    """Env whose first step ends the episode by truncation, not termination."""

    def reset(self, seed: int | None = None) -> tuple[str, dict[str, Any]]:
        del seed
        return "start", {}

    def step(self, action: str) -> tuple[str, float, bool, bool, dict[str, Any]]:
        del action
        return "", 0.5, False, True, {}


def test_truncated_round_trips_over_http() -> None:
    """The HTTP backend reports the same terminated/truncated split as local."""
    with OpenEnvServer(_TruncatingEnv()) as server:
        client = OpenEnvSessionClient(server.base_url)
        client.reset()
        _obs, reward, terminated, truncated, _info = client.step("a")
        client.close()
    assert reward == 0.5
    assert truncated is True
    assert terminated is False


class _KwargsOnlyResetEnv:
    """An env whose ``reset`` takes only ``**kwargs`` (a delegating proxy)."""

    def __init__(self) -> None:
        self.reset_kwargs: dict[str, Any] = {}

    def reset(self, **kwargs: Any) -> tuple[str, dict[str, Any]]:
        self.reset_kwargs = kwargs
        return "prompt", {}

    def step(self, action: str) -> tuple[str, float, bool, bool, dict[str, Any]]:
        del action
        return "", 1.0, True, False, {}


def test_kwargs_only_reset_still_receives_seed_and_row() -> None:
    """A ``**kwargs`` reset counts every forwardable name as supported.

    Otherwise a proxy env would silently lose the seed/row pinning that keeps a
    rollout group on one prompt.
    """
    inner = _KwargsOnlyResetEnv()
    client = LocalEnvClient(inner)
    with client.eval_mode():
        client.reset(seed=7, row_index=2)
    assert inner.reset_kwargs == {"seed": 7, "row_index": 2, "evaluation": True}


@pytest.mark.filterwarnings("ignore::pytest.PytestUnhandledThreadExceptionWarning")
def test_server_start_fails_fast_when_port_is_taken() -> None:
    """A dead uvicorn thread is detected immediately (no 30s spin), env released."""
    inner = _CountingEnv()
    blocker = socket.socket()
    blocker.bind(("127.0.0.1", 0))
    blocker.listen(1)
    port = blocker.getsockname()[1]
    try:
        started = time.monotonic()
        with pytest.raises(RuntimeError, match="during startup"):
            OpenEnvServer(inner, port=port).start()
        assert time.monotonic() - started < 10.0
        assert inner.closed  # the failed start did not leak the hosted env
    finally:
        blocker.close()


def test_server_requires_exactly_one_env_source() -> None:
    """A server hosts either one shared env or a per-session factory, never both."""
    with pytest.raises(ValueError, match="exactly one of env or make_env"):
        OpenEnvServer()
    with pytest.raises(ValueError, match="exactly one of env or make_env"):
        OpenEnvServer(_CountingEnv(), make_env=_CountingEnv)


# --- max_turns is enforced by the env itself --------------------------------
class _EndlessEnv:
    """Env that never terminates on its own."""

    def reset(self, seed: int | None = None) -> tuple[str, dict[str, Any]]:
        del seed
        return "start", {}

    def step(self, action: str) -> tuple[str, float, bool, bool, dict[str, Any]]:
        del action
        return "more", 0.0, False, False, {}


def test_rollout_env_truncates_at_max_turns() -> None:
    """The episode self-truncates at ``max_turns`` — no external cap needed, and
    no dangling feedback turn is appended after the final generation.
    """
    env = RolloutEnv.local(
        _EndlessEnv(), MiniTokenizer(), max_turns=2, apply_chat_template=False
    )
    env.reset()
    _obs, _r, terminated, truncated, _info = env.step(
        torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    )
    assert not terminated
    assert not truncated
    seq_len = env.full_ids.shape[1]
    _obs, _r, terminated, truncated, info = env.step(
        torch.ones((1, seq_len + 1), dtype=torch.long)
    )
    assert truncated
    assert not terminated
    assert env.done
    assert info == {}
    assert env.full_ids.shape[1] == seq_len + 1  # generation only, no feedback
    env.close()


# --- context budget: over-budget rows truncate at turn 0 --------------------
class _LenTok(MiniTokenizer):
    """Tokenizer whose token count tracks the text length (for budget tests)."""

    def __call__(self, texts: list[str], **_: Any) -> dict[str, torch.Tensor]:
        ids = torch.ones((1, max(1, len(texts[0]))), dtype=torch.long)
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}


class _VariedRowEnv:
    """Dataset env with one row that tokenizes far past any small budget."""

    def __init__(self) -> None:
        self.prompts = ["aa", "b" * 300, "cc"]

    @property
    def dataset_size(self) -> int:
        return len(self.prompts)

    def reset(
        self,
        seed: int | None = None,
        *,
        row_index: int | None = None,
        evaluation: bool | None = None,
    ) -> tuple[str, dict[str, Any]]:
        del seed, evaluation
        return self.prompts[(row_index or 0) % len(self.prompts)], {}

    def step(self, action: str) -> tuple[str, float, bool, bool, dict[str, Any]]:
        del action
        return "", 1.0, True, False, {}


def test_reset_truncates_over_budget_row() -> None:
    """An over-budget initial prompt truncates the episode at turn 0 (``done``
    with an empty prompt) rather than raising — it just never becomes active.
    """
    env = RolloutEnv.local(
        _VariedRowEnv(), _LenTok(), apply_chat_template=False, max_model_len=64
    )
    obs, _ = env.reset(row_index=1)  # over-budget row
    assert obs == {}
    assert env.done is True
    obs, _ = env.reset(row_index=0)  # in-budget rows reset fine
    assert env.done is False
    assert obs["input_ids"].shape[1] <= 63
    env.close()


def test_batch_reset_truncates_over_budget_rows() -> None:
    """Over-budget rows self-truncate: those envs are ``done`` and excluded from
    the returned active prompts — no row-skipping, no cursor advance.
    """
    batch = BatchRolloutEnv(
        lambda **_: RolloutEnv.local(
            _VariedRowEnv(), _LenTok(), apply_chat_template=False, max_model_len=64
        ),
        batch_size=3,
        group_size=2,
    )
    # 3 rows, batch_size=3 -> one full permutation, so exactly the single
    # over-budget row's group (group_size envs) truncates.
    prompts = batch.reset(seed=0)
    assert prompts is not None
    assert all(p["input_ids"].shape[1] <= 63 for p in prompts)
    n_done = sum(1 for e in batch.envs if e.done)
    assert n_done == batch.group_size
    assert len(prompts) + n_done == len(batch.envs)
    batch.close()


# --- epoch counter + partial-init hardening ----------------------------------
def test_task_assigner_counts_completed_epochs() -> None:
    """``num_epochs`` counts completed passes, not the first epoch's start."""
    assigner = TaskAssigner(2, seed=0)
    assigner.next_row()
    assigner.next_row()
    assert assigner.num_epochs == 0
    assigner.next_row()  # crosses into the second epoch
    assert assigner.num_epochs == 1


def test_batch_rollout_env_exposes_num_epochs() -> None:
    """The collector surfaces the assigner's epoch count (e.g. for KL refresh)."""
    batch = BatchRolloutEnv(
        lambda **_: RolloutEnv.local(
            _RowDatasetEnv(n=2), MiniTokenizer(), apply_chat_template=False
        ),
        batch_size=2,
        group_size=1,
    )
    assert batch.num_epochs == 0
    batch.reset(seed=0)  # consumes the full 2-row epoch
    batch.reset(seed=0)  # draws again -> the first pass is complete
    assert batch.num_epochs == 1
    batch.close()


def test_batch_reset_cleans_up_after_partial_init_failure() -> None:
    """A factory failure mid-build closes the built envs and empties the batch,
    so a retried reset starts clean instead of misaligning slots.
    """
    inners: list[_CountingEnv] = []
    calls = {"n": 0}

    def factory(**_: Any) -> RolloutEnv:
        calls["n"] += 1
        if calls["n"] == 2:
            msg = "boom"
            raise RuntimeError(msg)
        inner = _CountingEnv()
        inners.append(inner)
        return RolloutEnv.local(inner, MiniTokenizer(), apply_chat_template=False)

    batch = BatchRolloutEnv(factory, batch_size=2, group_size=1)
    with pytest.raises(RuntimeError, match="boom"):
        batch.reset(seed=0)
    assert batch.envs == []
    assert inners[0].closed
    prompts = batch.reset(seed=0)  # retry rebuilds from scratch
    assert prompts is not None
    assert len(batch.envs) == 2
    batch.close()


# --- gym-tuple normalisation: remaining single-element / bad-length branches --
def test_normalize_reset_accepts_single_element_tuple() -> None:
    """A 1-tuple ``(obs,)`` normalises to ``(obs, {})``."""
    assert _normalize_reset(("obs",)) == ("obs", {})


def test_normalize_step_rejects_wrong_length_tuple() -> None:
    """A tuple that is neither a 4- nor 5-tuple is a clear error."""
    with pytest.raises(ValueError, match="3-tuple; expected 4 or 5"):
        _normalize_step(("o", 1.0, True))


# --- OpenEnvServer: the 30s start deadline ----------------------------------
@pytest.mark.filterwarnings("ignore::pytest.PytestUnhandledThreadExceptionWarning")
def test_server_start_times_out_when_never_ready(monkeypatch: Any) -> None:
    """A server thread that stays alive but never binds fails at the deadline."""
    import uvicorn

    class _StuckServer:
        def __init__(self, config: Any) -> None:
            del config
            self.started = False
            self.should_exit = False

        def run(self) -> None:
            while not self.should_exit:
                time.sleep(0.001)

    monkeypatch.setattr(uvicorn, "Server", _StuckServer)
    # First call sets the deadline; the loop's next reading is already past it.
    clock = {"n": 0}

    def _fake_monotonic() -> float:
        clock["n"] += 1
        return 0.0 if clock["n"] == 1 else 1000.0

    monkeypatch.setattr(openenv_server_module.time, "monotonic", _fake_monotonic)
    inner = _CountingEnv()
    with pytest.raises(RuntimeError, match="within 30s"):
        OpenEnvServer(inner).start()
    assert inner.closed  # the failed start released the hosted env


# --- OpenEnvSessionClient: MCP-tool action encoding --------------------------
def test_step_encodes_mcp_call_tool_action() -> None:
    """With ``mcp_tool`` set, the text is sent as OpenEnv's typed ``CallToolAction``."""
    client = OpenEnvSessionClient("http://stub.invalid", mcp_tool="echo", arg="text")
    client._sync = _StubSync()
    obs, reward, terminated, _truncated, _info = client.step("hello")
    assert obs == "hi"
    assert reward == 1.0
    assert terminated is True
    assert client._sync.step_calls[-1] == {
        "type": "call_tool",
        "tool_name": "echo",
        "arguments": {"text": "hello"},
        "metadata": {},
    }


# --- LocalEnvClient: tool schemas --------------------------------------------
def test_local_env_client_exposes_tools() -> None:
    """``LocalEnvClient`` surfaces the wrapped env's advertised tool schemas."""
    env = _CountingEnv(tools=[{"name": "t"}])
    assert LocalEnvClient(env).tools == [{"name": "t"}]


# --- _observation_text: rendering our own + third-party observations ---------
def test_observation_text_renders_all_shapes() -> None:
    """Bare strings, MCP content blocks, keyed fields, and errors all render."""
    assert _observation_text("plain") == "plain"
    assert _observation_text(12345) == ""
    # MCP tool-result content blocks: only well-formed text blocks are kept.
    assert (
        _observation_text(
            {"result": {"content": [{"text": "a"}, {"nope": 1}, {"text": "b"}]}}
        )
        == "a\nb"
    )
    # No text blocks -> the string ``data`` field is the fallback.
    assert _observation_text({"result": {"content": [], "data": "d"}}) == "d"
    # Error field renders as a message; an empty observation is empty text.
    assert _observation_text({"error": "bad"}) == "Error: bad"
    assert _observation_text({}) == ""


# --- entrypoint resolution: malformed specs ---------------------------------
def test_load_env_requires_module_and_class() -> None:
    """An entrypoint missing the ``:`` or the class name is rejected."""
    with pytest.raises(ValueError, match="Invalid entrypoint format"):
        openenv_module.load_env("no_colon_here")
    with pytest.raises(ValueError, match="must include both module and target"):
        openenv_module.load_env("module:")


class TestBatchRolloutEnvConcurrentStep:
    """``BatchRolloutEnv.step`` overlaps the per-env backend round-trips.

    The tokenizer stays single-threaded (used only in the sequential
    prepare/apply phases); only the pure-I/O ``env_client.step`` round-trips run
    concurrently, so these tests cover overlap, index-correct routing, error
    propagation, and pool teardown.
    """

    @staticmethod
    def _batch(env_cls, batch_size):
        batch = BatchRolloutEnv(
            lambda **_: RolloutEnv.local(
                env_cls(), MiniTokenizer(), max_turns=1, apply_chat_template=False
            ),
            batch_size=batch_size,
            group_size=1,
        )
        batch.reset(seed=0)
        return batch

    @staticmethod
    def _completions(n):
        return [torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long) for _ in range(n)]

    def test_step_round_trips_run_concurrently(self):
        """A barrier needing all N envs releases only if the round-trips overlap;
        sequential stepping would block the first env until the barrier times out.
        """
        n = 4
        barrier = threading.Barrier(n, timeout=15)

        class _BarrierEnv:
            def reset(self, seed=None):
                del seed
                return "start", {}

            def step(self, action):
                del action
                barrier.wait()  # only releases when all n envs arrive together
                return "done", 1.0, True, False, {}

        batch = self._batch(_BarrierEnv, n)
        try:
            # Raises BrokenBarrierError (via a worker) if the steps ran serially.
            batch.step(self._completions(n))
            assert not barrier.broken
        finally:
            batch.close()

    def test_step_routes_results_to_correct_env(self):
        """Each env applies its own backend result (index-correct routing)."""
        counter = {"i": 0}

        class _IdEnv:
            def __init__(self):
                self.ident = counter["i"]
                counter["i"] += 1

            def reset(self, seed=None):
                del seed
                return "s", {}

            def step(self, action):
                del action
                return "d", float(self.ident), True, False, {}

        batch = self._batch(_IdEnv, 4)
        try:
            batch.step(self._completions(4))
            assert [env.turn_rewards[-1] for env in batch.envs] == [0.0, 1.0, 2.0, 3.0]
        finally:
            batch.close()

    def test_step_propagates_env_exception(self):
        """A single env's backend error propagates out of the batched step."""

        class _BoomEnv:
            def reset(self, seed=None):
                del seed
                return "s", {}

            def step(self, action):
                del action
                msg = "boom"
                raise RuntimeError(msg)

        batch = self._batch(_BoomEnv, 3)
        try:
            with pytest.raises(RuntimeError, match="boom"):
                batch.step(self._completions(3))
        finally:
            batch.close()

    def test_reset_round_trips_run_concurrently(self):
        """Group-lead reset fetches overlap; a barrier needing all N releases only
        if they run concurrently (sequential reset would time out).
        """
        n = 4
        barrier = threading.Barrier(n, timeout=15)

        class _BarrierResetEnv:
            def reset(self, seed=None):
                del seed
                barrier.wait()  # only releases when all n leads fetch together
                return "start", {}

            def step(self, action):
                del action
                return "done", 1.0, True, False, {}

        batch = BatchRolloutEnv(
            lambda **_: RolloutEnv.local(
                _BarrierResetEnv(),
                MiniTokenizer(),
                max_turns=1,
                apply_chat_template=False,
            ),
            batch_size=n,
            group_size=1,
        )
        try:
            batch.reset(seed=0)  # BrokenBarrierError if leads fetched serially
            assert not barrier.broken
        finally:
            batch.close()

    def test_close_shuts_down_io_pool(self):
        """The I/O pool is lazily built by concurrent reset/step and released."""
        batch = BatchRolloutEnv(
            lambda **_: RolloutEnv.local(
                _CountingEnv(), MiniTokenizer(), max_turns=1, apply_chat_template=False
            ),
            batch_size=2,
            group_size=1,
        )
        assert batch._io_pool is None  # not built at construction
        batch.reset(seed=0)  # concurrent lead fetches build the pool
        batch.step(self._completions(2))
        assert batch._io_pool is not None
        batch.close()
        assert batch._io_pool is None


def test_wrapper_owns_inner_closes_it_on_close() -> None:
    """``owns_inner=True`` closes the wrapped env on ``close``; ``False`` leaves it."""
    owned = _CountingEnv()
    OpenEnvWrapper(owned, owns_inner=True).close()
    assert owned.closed is True

    shared = _CountingEnv()
    OpenEnvWrapper(shared, owns_inner=False).close()
    assert shared.closed is False


def test_server_base_url_before_start_raises() -> None:
    """``base_url`` is only available once the server has started."""
    server = OpenEnvServer(_CountingEnv())
    with pytest.raises(RuntimeError, match="not running"):
        _ = server.base_url


def test_server_make_env_serves_a_fresh_env_per_session() -> None:
    """``make_env`` + ``max_concurrent_envs`` hosts one server (one URL) with a fresh
    env per WebSocket session — the container / Ray-actor deployment shape.
    """
    built: list[_CountingEnv] = []

    def make_env() -> _CountingEnv:
        env = _CountingEnv()
        built.append(env)
        return env

    server = OpenEnvServer(make_env=make_env, max_concurrent_envs=4).start()
    try:
        c1 = OpenEnvSessionClient(server.base_url)
        c2 = OpenEnvSessionClient(server.base_url)
        c1.reset()
        c2.reset()
        # Each session drives its own fresh env instance (plus one the server
        # builds up front to read the env's schema).
        assert len(built) >= 2
        assert len({id(env) for env in built}) == len(built)
        c1.close()
        c2.close()
    finally:
        server.stop()


# --- from_spec: factory callables and registered resolvers -------------------
def test_from_spec_accepts_env_factory_callable() -> None:
    """A callable spec is an env factory: built with env_config, driven in-process."""
    env = RolloutEnv.from_spec(
        lambda target: _CountingEnv(target=target),
        {"target": 3},
        MiniTokenizer(),
        max_turns=5,
        apply_chat_template=False,
    )
    assert isinstance(env._env_client, LocalEnvClient)
    assert env._env_client._env.target == 3


def test_from_spec_consults_registered_resolvers() -> None:
    """A registered resolver claims its spec ids before entrypoint loading."""
    from agilerl.llm_envs import register_env_spec_resolver
    from agilerl.llm_envs.spec_resolvers import _ENV_SPEC_RESOLVERS

    def fake_resolver(spec: str):
        if spec == "fake:Env-v0":
            return lambda **cfg: _CountingEnv(**cfg)
        return None

    register_env_spec_resolver("fake", fake_resolver)
    try:
        env = RolloutEnv.from_spec(
            "fake:Env-v0",
            {"target": 2},
            MiniTokenizer(),
            apply_chat_template=False,
        )
        assert isinstance(env._env_client, LocalEnvClient)
        assert env._env_client._env.target == 2
    finally:
        _ENV_SPEC_RESOLVERS.pop("fake", None)


def test_from_spec_unclaimed_spec_still_loads_entrypoints() -> None:
    """Specs no resolver claims fall through to module:Class loading."""
    env = RolloutEnv.from_spec(
        f"{__name__}:_CountingEnv",
        {"target": 4},
        MiniTokenizer(),
        apply_chat_template=False,
    )
    assert isinstance(env._env_client, LocalEnvClient)
    assert env._env_client._env.target == 4


def test_reset_retries_once_when_the_server_closed_an_idle_session() -> None:
    """A server-side idle close surfaces on the boundary reset; one re-dial heals it."""
    client = _session_client(
        reset_errors=[websockets.exceptions.ConnectionClosedOK(None, None)]
    )
    fresh = _StubSync()
    client._build_session = lambda: fresh  # type: ignore[method-assign]
    prompt, _info = client.reset()
    assert prompt == "prompt"
    assert client._sync is fresh
    assert client._redials == 1


def test_server_hosts_prompt_dataset_environment_as_is() -> None:
    """OpenEnv ``Environment`` subclasses are hosted without ``OpenEnvWrapper``."""
    world = PromptDatasetEnv(
        [{"question": "q", "answer": "a"}],
        rubric=reward_fn_to_rubric(lambda c, a, q: 1.0),
    )
    with OpenEnvServer(world) as server:
        client = OpenEnvSessionClient(server.base_url)
        assert client.reset(row_index=0)[0] == "q"
        _obs, reward, terminated, _truncated, _info = client.step("done")
        assert reward == 1.0
        assert terminated is True
        assert client.rubric_components == ("fn",)
