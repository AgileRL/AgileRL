"""The OpenEnv env interface: client + server, resolver.

Every LLM-training env is reached the same way — text in, text out, over the small
OpenEnv HTTP protocol. These cover both halves (``OpenEnvServer`` host +
``OpenEnvClient`` client over httpx) and the spec resolver.
"""

from __future__ import annotations

import socket
import threading
import time
from typing import Any

import httpx
import pytest
import torch

from agilerl.llm_envs import (
    BatchPointer,
    BatchRolloutEnv,
    LocalEnvClient,
    OpenEnvClient,
    OpenEnvServer,
    OpenEnvWrapper,
    RolloutEnv,
    ServedEnvClient,
    resolve_env,
)
from agilerl.llm_envs import openenv as openenv_module
from agilerl.llm_envs.openenv import (
    _load_entrypoint,
    _module_from_path,
    _name_from_spec,
    _normalize_reset,
    _normalize_step,
    _observation_text,
)


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


class _DatasetEnv:
    """A tiny dataset-backed env, imported by its ``module:Class`` path to exercise
    ``resolve_env``'s entrypoint loading: ``reset`` serves a row's prompt, ``step``
    scores the completion.
    """

    def __init__(
        self,
        questions: list[str],
        answers: list[str],
        reward_fn: Any,
        *,
        prompt_builder: Any = None,
    ) -> None:
        self.questions = questions
        self.answers = answers
        self.reward_fn = reward_fn
        self.prompt_builder = prompt_builder or (lambda q: q)
        self._answer = ""

    @property
    def dataset_size(self) -> int:
        return len(self.questions)

    def reset(
        self,
        seed: int | None = None,
        *,
        row_index: int = 0,
        evaluation: bool | None = None,
    ) -> tuple[str, dict[str, Any]]:
        del seed, evaluation
        row = (row_index or 0) % len(self.questions)
        self._answer = self.answers[row]
        return self.prompt_builder(self.questions[row]), {}

    def step(self, action: str) -> tuple[str, float, bool, bool, dict[str, Any]]:
        reward = float(self.reward_fn(action, self._answer, ""))
        return "", reward, True, False, {}


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
        return self._q, {"suffix": "answer:"}

    def step(self, action: str) -> tuple[str, float, bool, bool, dict[str, Any]]:
        return "", (1.0 if "x" in str(action) else 0.0), True, False, {}

    def close(self) -> None:
        pass


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
        # not closed per request or on client close — only once when the server stops
        assert env.closed is False
    assert env.closed is True


def test_serve_helper_starts_immediately() -> None:
    """``serve`` returns an already-running server."""
    server = OpenEnvServer(_CountingEnv()).start()
    try:
        assert (
            OpenEnvClient(base_url=server.base_url).reset()[0] == "Start.\nReply 'go'."
        )
    finally:
        server.stop()


def test_info_reports_dataset_size_and_tools() -> None:
    """``/state`` carries dataset size and advertised tool schemas to the client."""
    tools = [{"type": "function", "function": {"name": "calc", "parameters": {}}}]
    server = OpenEnvServer(_CountingEnv(tools=tools)).start()
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
    assert _name_from_spec(r"C:\pkg\file.py:Env") == "Env"
    assert _name_from_spec("plainname") == "plainname"


def test_load_entrypoint_windows_style_path_uses_last_colon(monkeypatch: Any) -> None:
    r"""``_load_entrypoint`` treats ``C:\...\file.py:Class`` as a path entrypoint."""

    class _FakeModule:
        DiskEnv = _DatasetEnv

    captured: dict[str, str] = {}

    def _fake_module_from_path(path: str) -> _FakeModule:
        captured["path"] = path
        return _FakeModule()

    monkeypatch.setattr(
        "agilerl.llm_envs.openenv._module_from_path",
        _fake_module_from_path,
    )
    cls = _load_entrypoint(r"C:\tmp\disk_env.py:DiskEnv")
    assert cls is _DatasetEnv
    assert captured["path"] == r"C:\tmp\disk_env.py"


# --- serving: one OpenEnv server instance per rollout ----------------------
def test_serving_hosts_owns_and_stops_its_server() -> None:
    """``serving`` binds a ``ServedEnvClient`` whose server ``close`` stops."""
    env = RolloutEnv.serving(_CountingEnv, _MiniTok(), apply_chat_template=False)
    assert isinstance(env._env_client, ServedEnvClient)
    env.reset()  # reachable over real HTTP
    assert env._prompt_text == "Start.\nReply 'go'."
    env.close()
    with pytest.raises(RuntimeError):
        _ = env._env_client.base_url  # server stopped + released


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
    urls = [env._env_client.base_url for env in batch.envs]
    assert len(batch.envs) == 2
    assert len(set(urls)) == 2  # distinct server instances
    batch.close()
    for env in batch.envs:
        with pytest.raises(RuntimeError):
            _ = env._env_client.base_url


# --- RolloutEnv(url) over a real server ------------------------------------
def test_rollout_env_drives_url_and_applies_suffix() -> None:
    server = OpenEnvServer(_CountingEnv()).start()
    try:
        env = RolloutEnv(
            server.base_url, _MiniTok(), max_turns=2, apply_chat_template=False
        )
        env.reset()
        # reset prompt + info["suffix"] are folded server-side by OpenEnvWrapper.
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
        "tests.test_wrappers.test_openenv:_DatasetEnv",
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


# --- LocalEnvClient: drive a local env directly, no HTTP -------------------
class TestLocalEnvClient:
    """In-process env client (no HTTP): folds suffix, routes eval split, forwards close."""

    def test_drives_env_directly(self) -> None:
        """``LocalEnvClient`` calls the env in-process and folds the suffix like the server."""
        client = LocalEnvClient(_RowDatasetEnv())
        assert client.dataset_size == 3
        prompt, info = client.reset(row_index=1)
        assert prompt == "row1\nanswer:"  # suffix folded in-process (no OpenEnvWrapper)
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
            _RowDatasetEnv(), _MiniTok(), max_turns=1, apply_chat_template=False
        )
        assert isinstance(env._env_client, LocalEnvClient)
        obs, _ = env.reset(row_index=2)
        assert "input_ids" in obs
        assert env.dataset_size == 3
        env.close()


class TestRolloutEnvFromSpec:
    """``RolloutEnv.from_spec`` routes a URL to HTTP and an entrypoint to in-process."""

    def test_url_builds_http_client(self) -> None:
        """A URL spec routes to the HTTP ``OpenEnvClient`` env client."""
        env = RolloutEnv.from_spec("http://localhost:0", None, _MiniTok())
        assert isinstance(env._env_client, OpenEnvClient)
        env.close()

    def test_entrypoint_runs_in_process(self) -> None:
        """A ``path.py:Class`` entrypoint loads + drives the env in-process (no HTTP)."""
        spec = f"{__file__}:_RowDatasetEnv"
        env = RolloutEnv.from_spec(
            spec, {"prefix": "Q"}, _MiniTok(), max_turns=1, apply_chat_template=False
        )
        assert isinstance(env._env_client, LocalEnvClient)
        obs, _ = env.reset(row_index=0)
        assert "input_ids" in obs
        env.close()

    def test_entrypoint_with_serve_hosts_on_a_server(self) -> None:
        """``serve=True`` hosts the entrypoint env on its own server (HTTP loopback)."""
        spec = f"{__file__}:_RowDatasetEnv"
        env = RolloutEnv.from_spec(
            spec,
            {"prefix": "Q"},
            _MiniTok(),
            max_turns=1,
            serve=True,
            apply_chat_template=False,
        )
        assert isinstance(env._env_client, ServedEnvClient)
        obs, _ = env.reset(row_index=0)
        assert "input_ids" in obs
        env.close()


# --- BatchPointer: group-consistent dataset cursor -------------------------
class TestBatchPointer:
    """Reshuffled row stream: full permutation per epoch, group-contiguous assignment."""

    def test_reshuffles_each_epoch(self) -> None:
        """Each epoch is a fresh full permutation of the rows."""
        pointer = BatchPointer(4, seed=0)
        rows = [pointer.next_row() for _ in range(8)]
        assert sorted(rows[:4]) == [0, 1, 2, 3]
        assert sorted(rows[4:]) == [0, 1, 2, 3]

    def test_assign_is_group_contiguous(self) -> None:
        """``assign`` gives one ``(seed, row)`` per slot, shared within each group."""
        assigned = BatchPointer(4, seed=0).assign(2, 3, base_seed=100)
        assert len(assigned) == 6
        assert assigned[0] == assigned[1] == assigned[2]  # group 0 shares prompt + seed
        assert assigned[3] == assigned[4] == assigned[5]  # group 1 shares
        assert assigned[0][0] == 100  # seed = base_seed + batch item 0
        assert assigned[3][0] == 101  # seed = base_seed + batch item 1

    def test_without_dataset_yields_no_rows(self) -> None:
        """A non-dataset env (size 0) gets ``row_index=None`` for every slot."""
        assert BatchPointer(0).assign(2, 2) == [(None, None)] * 4


# --- /state strictness + MCP tools/list fallback ----------------------------
class _StubResponse:
    """Minimal httpx-response stand-in for driving ``_fetch_state`` branches."""

    def __init__(self, status_code: int = 200, body: Any = None) -> None:
        self.status_code = status_code
        self._body = body if body is not None else {}

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            msg = "boom"
            raise httpx.HTTPStatusError(msg, request=None, response=None)

    def json(self) -> Any:
        return self._body


class _StubHTTP:
    """Scripted httpx.Client stand-in: one queued result per GET, a table for POSTs."""

    def __init__(
        self,
        get_results: list[Any],
        post_results: dict[str, Any] | None = None,
    ) -> None:
        self._get_results = list(get_results)
        self._post_results = post_results or {}
        self.base_url = "http://stub.invalid"
        self.is_closed = False

    def get(self, path: str) -> Any:
        del path
        result = self._get_results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result

    def post(self, path: str, json: Any = None) -> Any:
        del json
        result = self._post_results[path]
        if isinstance(result, Exception):
            raise result
        return result

    def close(self) -> None:
        self.is_closed = True


def _stubbed_client(
    get_results: list[Any],
    post_results: dict[str, Any] | None = None,
) -> OpenEnvClient:
    client = OpenEnvClient("http://stub.invalid")
    client._http = _StubHTTP(get_results, post_results)
    return client


class TestStateFetchStrictness:
    """``/state`` drives row pinning + tool schemas, so it is fetched strictly."""

    def test_transient_failure_then_success(self, monkeypatch: Any) -> None:
        """A cold-start hiccup is retried; the eventual state is served + cached."""
        monkeypatch.setattr(openenv_module.time, "sleep", lambda _s: None)
        client = _stubbed_client(
            [
                httpx.ConnectError("cold start"),
                _StubResponse(200, {"dataset_size": 5, "tools": [{"name": "t"}]}),
            ]
        )
        assert client.dataset_size == 5
        assert client.tools == [{"name": "t"}]  # cached: no further GETs queued

    def test_persistent_failure_raises(self, monkeypatch: Any) -> None:
        """A server that never answers /state fails the run loudly, not silently."""
        monkeypatch.setattr(openenv_module.time, "sleep", lambda _s: None)
        client = _stubbed_client([httpx.ConnectError("down")] * 4)
        with pytest.raises(RuntimeError, match="/state"):
            _ = client.dataset_size

    def test_missing_route_is_empty_not_fatal(self) -> None:
        """404/405 = no /state route (e.g. production mode): empty, no retries."""
        client = _stubbed_client([_StubResponse(404)])
        assert client.dataset_size == 0

    def test_mcp_tools_list_fallback(self) -> None:
        """Tools absent from /state are discovered via MCP ``tools/list``."""
        mcp_tools = [{"name": "echo", "inputSchema": {}}]
        client = _stubbed_client(
            [_StubResponse(200, {"dataset_size": 0})],
            {"/mcp": _StubResponse(200, {"result": {"tools": mcp_tools}})},
        )
        assert client.tools == mcp_tools

    def test_no_mcp_channel_means_no_tools(self) -> None:
        """A server with neither /state tools nor an MCP channel yields []."""
        client = _stubbed_client(
            [_StubResponse(200, {})],
            {"/mcp": httpx.ConnectError("no mcp")},
        )
        assert client.tools == []

    def test_non_object_state_payload_is_treated_as_empty(self) -> None:
        """A non-dict ``/state`` body normalizes to ``{}``."""
        client = _stubbed_client([_StubResponse(200, ["not", "a", "dict"])])
        assert client.dataset_size == 0
        assert client.tools == []


def test_openenv_client_reset_forwards_eval_and_positive_seed_only() -> None:
    """``reset`` sends row/eval flags and drops negative seeds."""
    payloads: list[dict[str, Any]] = []
    client = OpenEnvClient("http://stub.invalid")

    def _capture(path: str, payload: dict[str, Any]) -> dict[str, Any]:
        assert path == "/reset"
        payloads.append(payload)
        return {"observation": "prompt"}

    client._post = _capture  # type: ignore[method-assign]

    with client.eval_mode():
        prompt, info = client.reset(seed=-2, row_index=3)
    assert prompt == "prompt"
    assert info == {}
    assert payloads[-1] == {"row_index": 3, "evaluation": True}

    prompt2, _ = client.reset(seed=5)
    assert prompt2 == "prompt"
    assert payloads[-1] == {"seed": 5}


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
        client = OpenEnvClient(server.base_url)
        client.reset()
        _obs, reward, terminated, truncated, _info = client.step("a")
        client.close()
    assert reward == 0.5
    assert truncated is True
    assert terminated is False


# --- ServedEnvClient: one owner for the server + client pair ----------------
def test_served_env_client_owns_server_and_pool() -> None:
    """``close`` stops the server, closes the hosted env, and releases the pool."""
    inner = _CountingEnv()
    client = ServedEnvClient(inner)
    prompt, _ = client.reset()
    assert prompt == "Start.\nReply 'go'."
    http = client._client._http
    client.close()
    assert inner.closed  # server teardown closed the hosted env
    assert http.is_closed  # connection pool released without a /close round-trip
    with pytest.raises(RuntimeError):
        _ = client.base_url
    client.close()  # idempotent


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
        _EndlessEnv(), _MiniTok(), max_turns=2, apply_chat_template=False
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


# --- context budget: over-budget rows fail fast / are skipped ---------------
class _LenTok(_MiniTok):
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


def test_reset_raises_prompt_overflow_for_over_budget_row() -> None:
    """An over-budget initial prompt fails at reset with a clear error, not
    hours later inside the generation engine.
    """
    env = RolloutEnv.local(
        _VariedRowEnv(), _LenTok(), apply_chat_template=False, max_model_len=64
    )
    with pytest.raises(RuntimeError, match="row_index=1"):
        env.reset(row_index=1)
    obs, _ = env.reset(row_index=0)  # in-budget rows reset fine
    assert obs["input_ids"].shape[1] <= 63
    env.close()


def test_batch_reset_skips_over_budget_rows_with_warning() -> None:
    """The batch collector warns and advances the cursor past over-budget rows."""
    batch = BatchRolloutEnv(
        lambda **_: RolloutEnv.local(
            _VariedRowEnv(), _LenTok(), apply_chat_template=False, max_model_len=64
        ),
        batch_size=3,
        group_size=2,
    )
    with pytest.warns(UserWarning, match="Skipping dataset row"):
        prompts = batch.reset(seed=0)
    assert prompts is not None
    assert all(p["input_ids"].shape[1] <= 63 for p in prompts)
    batch.close()


# --- epoch counter + partial-init hardening ----------------------------------
def test_batch_pointer_counts_completed_epochs() -> None:
    """``num_epochs`` counts completed passes, not the first epoch's start."""
    pointer = BatchPointer(2, seed=0)
    pointer.next_row()
    pointer.next_row()
    assert pointer.num_epochs == 0
    pointer.next_row()  # crosses into the second epoch
    assert pointer.num_epochs == 1


def test_batch_rollout_env_exposes_num_epochs() -> None:
    """The collector surfaces the cursor's epoch count (e.g. for KL refresh)."""
    batch = BatchRolloutEnv(
        lambda **_: RolloutEnv.local(
            _RowDatasetEnv(n=2), _MiniTok(), apply_chat_template=False
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
        return RolloutEnv.local(inner, _MiniTok(), apply_chat_template=False)

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

    monkeypatch.setattr(openenv_module.time, "monotonic", _fake_monotonic)
    inner = _CountingEnv()
    with pytest.raises(RuntimeError, match="within 30s"):
        OpenEnvServer(inner).start()
    assert inner.closed  # the failed start released the hosted env


# --- OpenEnvClient: MCP-tool action encoding + strict payloads ---------------
def test_step_encodes_mcp_call_tool_action() -> None:
    """With ``mcp_tool`` set, the model's text is sent as a ``call_tool`` action."""
    captured: dict[str, Any] = {}

    class _CapturingHTTP(_StubHTTP):
        def post(self, path: str, json: Any = None) -> Any:
            captured["path"] = path
            captured["json"] = json
            return _StubResponse(
                200, {"observation": "hi", "reward": 1.0, "done": True}
            )

    client = OpenEnvClient("http://stub.invalid", mcp_tool="echo", arg="text")
    client._http = _CapturingHTTP([])
    obs, reward, terminated, _truncated, _info = client.step("hello")
    assert obs == "hi"
    assert reward == 1.0
    assert terminated is True
    assert captured["json"]["action"] == {
        "type": "call_tool",
        "tool_name": "echo",
        "arguments": {"text": "hello"},
    }


def test_fetch_state_retries_after_bad_status(monkeypatch: Any) -> None:
    """A transient error status is retried, then the good state is served."""
    monkeypatch.setattr(openenv_module.time, "sleep", lambda _s: None)
    client = _stubbed_client(
        [_StubResponse(500), _StubResponse(200, {"dataset_size": 3})]
    )
    assert client.dataset_size == 3


def test_fetch_mcp_tools_is_cached(monkeypatch: Any) -> None:
    """The MCP ``tools/list`` result is fetched once and cached."""
    calls = {"n": 0}

    class _CountingHTTP(_StubHTTP):
        def post(self, path: str, json: Any = None) -> Any:
            calls["n"] += 1
            return _StubResponse(200, {"result": {"tools": [{"name": "e"}]}})

    client = OpenEnvClient("http://stub.invalid")
    client._http = _CountingHTTP([_StubResponse(200, {"dataset_size": 0})])
    assert client.tools == [{"name": "e"}]
    assert client.tools == [{"name": "e"}]  # served from the cache, no second POST
    assert calls["n"] == 1


def test_post_rejects_non_object_payload() -> None:
    """A non-object JSON body from the server is a clear protocol error."""
    client = _stubbed_client([], {"/step": _StubResponse(200, ["not", "a", "dict"])})
    with pytest.raises(TypeError, match="non-object payload"):
        client.step("x")


# --- LocalEnvClient / ServedEnvClient: tool schemas + delegation -------------
def test_local_env_client_exposes_tools() -> None:
    """``LocalEnvClient`` surfaces the wrapped env's advertised tool schemas."""
    env = _CountingEnv(tools=[{"name": "t"}])
    assert LocalEnvClient(env).tools == [{"name": "t"}]


def test_served_env_client_steps_and_reports_tools() -> None:
    """The served backend forwards ``step`` and ``tools`` to its owned client."""
    client = ServedEnvClient(_CountingEnv(tools=[{"name": "t"}]))
    try:
        client.reset()
        _obs, reward, _term, _trunc, _info = client.step("go")
        assert reward == 1.0
        assert client.tools == [{"name": "t"}]
    finally:
        client.close()


def test_served_env_client_stops_server_if_client_build_fails(
    monkeypatch: Any,
) -> None:
    """If the client fails to build, the owned server is stopped (no leak)."""
    inner = _CountingEnv()

    def _boom(*_a: Any, **_k: Any) -> Any:
        msg = "client build failed"
        raise RuntimeError(msg)

    monkeypatch.setattr(openenv_module, "OpenEnvClient", _boom)
    with pytest.raises(RuntimeError, match="client build failed"):
        ServedEnvClient(inner)
    assert inner.closed  # the started server was torn down, closing the env


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


# --- entrypoint resolution: malformed specs + unloadable paths ---------------
def test_load_entrypoint_requires_module_and_class() -> None:
    """An entrypoint missing the ``:`` or the class name is rejected."""
    with pytest.raises(ValueError, match="must be 'module:Class'"):
        _load_entrypoint("no_colon_here")
    with pytest.raises(ValueError, match="must be 'module:Class'"):
        _load_entrypoint("module:")


def test_module_from_path_raises_when_unloadable() -> None:
    """A path Python can't build a module spec for is a clear import error."""
    with pytest.raises(ImportError, match="cannot load env module from path"):
        _module_from_path("/tmp/not_a_python_module.bogus")


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
                env_cls(), _MiniTok(), max_turns=1, apply_chat_template=False
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

    def test_close_shuts_down_io_pool(self):
        """The lazily-built I/O pool is created on a multi-env step and released."""
        batch = self._batch(_CountingEnv, 2)
        assert batch._io_pool is None
        batch.step(self._completions(2))
        assert batch._io_pool is not None
        batch.close()
        assert batch._io_pool is None
