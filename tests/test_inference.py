# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Tests for arena.inference (agent and cache)."""

from __future__ import annotations

import json
import uuid
from collections.abc import Callable
from typing import Any
from unittest.mock import MagicMock, patch

import httpx
import pytest

from agilerl.auth import EXTERNAL_USER_ID_HEADER
from agilerl.exceptions import ArenaAuthError, ArenaInferenceError
from agilerl.inference import (
    Agent,
    DeploymentAssertion,
    LLMParams,
    LLMResults,
    SessionDetail,
    SessionInfo,
    StatusResponse,
)
from agilerl.inference.cache import (
    ActiveAgentSelection,
    clear_active_session,
    load_active_agent,
    load_active_session,
    load_binding,
    normalized_deployment_name,
    save_active_agent,
    save_active_session,
    save_binding,
)

STATUS_BODY = {
    "success": True,
    "deployment_id": "dep-1",
    "instance_id": "inst-1",
    "agent": {
        "algo": "DQN",
        "multi_agent": False,
        "llm": False,
        "recurrent": False,
    },
}


def _default_metadata(**agent_overrides: object) -> StatusResponse:
    body = {**STATUS_BODY, "agent": {**STATUS_BODY["agent"], **agent_overrides}}
    return StatusResponse.model_validate(body)


def _json_response(status_code: int = 200, body: Any = None) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = body if body is not None else {}
    resp.text = ""
    return resp


def _llm_response() -> MagicMock:
    return _json_response(
        200,
        {
            "results": [{"prompt": "hi", "completion": "hello"}],
            "batch_size": 1,
            "inference_time_ms": 1.0,
            "tokens_per_second": 10.0,
            "success": True,
        },
    )


def _mock_agent(**agent_overrides: object) -> Agent:
    """An Agent wired to a mock transport, without going through ``__init__``."""
    agent = Agent.__new__(Agent)
    agent._base_url = "http://test"
    agent.metadata = _default_metadata(**agent_overrides)
    agent.generate_params = LLMParams()
    agent._http = MagicMock(spec=httpx.Client)
    return agent


def _llm_agent() -> Agent:
    return _mock_agent(llm=True, algo="GRPO")


def _stream_response(chunks: list[str], status_code: int = 200) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = {}
    resp.text = ""
    resp.iter_text.return_value = iter(chunks)
    ctx = MagicMock()
    ctx.__enter__.return_value = resp
    ctx.__exit__.return_value = False
    return ctx


class TestAgentRequestErrors:
    def test_401_raises_auth_error(self):
        agent = _llm_agent()
        agent._http.request.return_value = _json_response(401, {})

        with pytest.raises(ArenaAuthError, match="401 Unauthorized"):
            agent.generate("hi")

    def test_non_200_raises_api_error(self):
        agent = _llm_agent()
        agent._http.request.return_value = _json_response(500, {"error": "internal"})

        with pytest.raises(ArenaInferenceError) as exc_info:
            agent.generate("hi")
        assert exc_info.value.status_code == 500

    def test_network_error_raises_api_error(self):
        agent = _llm_agent()
        agent._http.request.side_effect = httpx.ConnectError("refused")

        with pytest.raises(ArenaInferenceError) as exc_info:
            agent.generate("hi")
        assert exc_info.value.status_code == 0

    def test_non_200_with_text_fallback(self):
        agent = _llm_agent()
        mock_resp = MagicMock()
        mock_resp.status_code = 502
        mock_resp.json.side_effect = ValueError("not json")
        mock_resp.text = "Bad Gateway"
        agent._http.request.return_value = mock_resp

        with pytest.raises(ArenaInferenceError) as exc_info:
            agent.generate("hi")
        assert "Bad Gateway" in exc_info.value.detail

    def test_400_from_server(self):
        agent = _llm_agent()
        agent._http.request.return_value = _json_response(
            400, {"error": "wrong agent type"}
        )

        with pytest.raises(ArenaInferenceError) as exc_info:
            agent.generate("hi")
        assert exc_info.value.status_code == 400


class TestAgentInit:
    @patch("agilerl.inference.agent.httpx.Client")
    def test_no_token(self, mock_http_cls):
        mock_client = MagicMock()
        mock_http_cls.return_value = mock_client
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = STATUS_BODY
        mock_client.request.return_value = mock_resp

        agent = Agent("http://endpoint/")
        call_kwargs = mock_http_cls.call_args[1]
        assert call_kwargs["headers"] == {}
        assert agent._base_url == "http://endpoint"
        assert agent.metadata.deployment_id == "dep-1"

    @patch("agilerl.inference.agent.httpx.Client")
    def test_with_api_key(self, mock_http_cls):
        mock_client = MagicMock()
        mock_http_cls.return_value = mock_client
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = STATUS_BODY
        mock_client.request.return_value = mock_resp

        Agent("http://endpoint", api_key="abc")
        call_kwargs = mock_http_cls.call_args[1]
        assert call_kwargs["headers"]["Authorization"] == "Bearer abc"

    @patch("agilerl.inference.agent.httpx.Client")
    def test_probe_on_init_false_defers_metadata(self, mock_http_cls):
        mock_client = MagicMock()
        mock_http_cls.return_value = mock_client

        agent = Agent("http://endpoint", probe_on_init=False)
        assert agent.metadata is None
        mock_client.request.assert_not_called()

    @patch("agilerl.inference.agent.httpx.Client")
    def test_generate_params_defaults_when_none(self, mock_http_cls):
        mock_client = MagicMock()
        mock_http_cls.return_value = mock_client
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = STATUS_BODY
        mock_client.request.return_value = mock_resp

        agent = Agent("http://endpoint")
        assert agent.generate_params == LLMParams()


class TestStatus:
    def _make_agent_with_mock_http(self):
        return _mock_agent()

    def test_status_without_refresh_returns_cached(self):
        agent = self._make_agent_with_mock_http()
        cached = agent.metadata
        result = agent.status()
        assert result is cached
        agent._http.request.assert_not_called()

    def test_status_refresh_refetches(self):
        agent = self._make_agent_with_mock_http()
        updated = {
            **STATUS_BODY,
            "agent": {**STATUS_BODY["agent"], "algo": "PPO"},
        }
        agent._http.request.return_value = _json_response(200, updated)

        result = agent.status(refresh=True)
        assert result.agent.algo == "PPO"
        assert agent.metadata.agent.algo == "PPO"
        agent._http.request.assert_called_once()


class TestGenerate:
    def _make_agent_with_mock_http(self):
        return _llm_agent()

    def test_generate_string_prompt(self):
        agent = self._make_agent_with_mock_http()
        body = {
            "results": [{"prompt": "hi", "completion": "hello"}],
            "batch_size": 1,
            "inference_time_ms": 100.0,
            "tokens_per_second": 50.0,
            "success": True,
        }
        agent._http.request.return_value = _json_response(200, body)

        result = agent.generate("hi", params={"temperature": 0.5})
        assert isinstance(result, LLMResults)
        assert result.results[0].completion == "hello"
        sent = agent._http.request.call_args[1]["json"]
        assert sent["prompts"] == ["hi"]
        assert sent["params"]["temperature"] == 0.5
        assert sent["params"]["max_new_tokens"] == agent.generate_params.max_new_tokens

    def test_generate_uses_custom_agent_generate_params(self):
        agent = self._make_agent_with_mock_http()
        agent.generate_params = LLMParams(max_new_tokens=256, temperature=0.2)
        body = {
            "results": [{"prompt": "hi", "completion": "hello"}],
            "batch_size": 1,
            "inference_time_ms": 1.0,
            "tokens_per_second": 10.0,
            "success": True,
        }
        agent._http.request.return_value = _json_response(200, body)

        agent.generate("hi")
        sent = agent._http.request.call_args[1]["json"]
        assert sent["params"]["max_new_tokens"] == 256
        assert sent["params"]["temperature"] == 0.2

    def test_generate_rejects_empty_prompt(self):
        agent = self._make_agent_with_mock_http()
        with pytest.raises(ArenaInferenceError, match="empty"):
            agent.generate("")

    def test_generate_includes_session_id(self):
        agent = self._make_agent_with_mock_http()
        agent._http.request.return_value = _llm_response()

        agent.generate("hi", session_id=" sess-1 ")
        sent = agent._http.request.call_args[1]["json"]
        assert sent["session_id"] == "sess-1"

    def test_generate_sends_a_new_session_id_when_none(self):
        agent = self._make_agent_with_mock_http()
        agent._http.request.return_value = _llm_response()

        agent.generate("hi")
        first = agent._http.request.call_args[1]["json"]["session_id"]
        agent.generate("hi")
        second = agent._http.request.call_args[1]["json"]["session_id"]

        assert uuid.UUID(first).version == 4
        assert first != second

    def test_generate_rejects_blank_session_id(self):
        agent = self._make_agent_with_mock_http()
        with pytest.raises(ArenaInferenceError, match="session_id"):
            agent.generate("hi", session_id="  ")


class TestGenerateStream:
    def test_generate_stream_yields_chunks(self):
        agent = _llm_agent()
        agent._http.stream.return_value = _stream_response(["foo", "bar"])

        chunks = list(agent.generate_stream("Say hi"))
        assert chunks == ["foo", "bar"]
        sent = agent._http.stream.call_args[1]["json"]
        assert sent["prompt"] == "Say hi"

    def test_generate_stream_raises_on_in_band_error(self):
        agent = _llm_agent()
        agent._http.stream.return_value = _stream_response(["\n[ERROR: model failed]"])

        with pytest.raises(ArenaInferenceError, match="ERROR"):
            list(agent.generate_stream("hi"))

    def test_stream_includes_session_id(self):
        agent = _llm_agent()
        agent._http.stream.return_value = _stream_response(["ok"])

        list(agent.generate_stream("hi", session_id="sess-1"))
        sent = agent._http.stream.call_args[1]["json"]
        assert sent["session_id"] == "sess-1"

    def test_stream_sends_a_new_session_id_when_none(self):
        agent = _llm_agent()
        agent._http.stream.return_value = _stream_response(["ok"])

        list(agent.generate_stream("hi"))
        sent = agent._http.stream.call_args[1]["json"]

        assert uuid.UUID(sent["session_id"]).version == 4

    def test_stream_error_reports_the_status_and_detail(self):
        agent = _llm_agent()
        agent._http = httpx.Client(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(422, json={"detail": "bad prompt"})
            )
        )

        with pytest.raises(ArenaInferenceError, match="bad prompt") as exc_info:
            list(agent.generate_stream("hi"))

        assert exc_info.value.status_code == 422

    def test_stream_rejects_blank_session_id(self):
        agent = _llm_agent()
        with pytest.raises(ArenaInferenceError, match="session_id"):
            list(agent.generate_stream("hi", session_id=""))


class TestAgentCredentialResolution:
    @patch("agilerl.inference.agent.httpx.Client")
    def test_explicit_api_key_wins(self, mock_http_cls, monkeypatch):
        monkeypatch.setenv("ARENA_API_KEY", "env-key")
        Agent("http://endpoint", api_key="explicit-key", probe_on_init=False)
        headers = mock_http_cls.call_args[1]["headers"]
        assert headers == {"Authorization": "Bearer explicit-key"}

    @patch("agilerl.inference.agent.httpx.Client")
    def test_falls_back_to_env_var(self, mock_http_cls, monkeypatch):
        monkeypatch.setenv("ARENA_API_KEY", "arena_pat_env")
        Agent("http://endpoint", probe_on_init=False)
        headers = mock_http_cls.call_args[1]["headers"]
        assert headers == {"Authorization": "Bearer arena_pat_env"}

    @patch("agilerl.inference.agent.httpx.Client")
    def test_no_credential_sends_no_auth_header(self, mock_http_cls):
        Agent("http://endpoint", probe_on_init=False)
        assert mock_http_cls.call_args[1]["headers"] == {}

    @patch("agilerl.inference.agent.httpx.Client")
    def test_org_key_headers(self, mock_http_cls):
        Agent(
            "http://endpoint",
            org_key="arena_org_secret",
            external_user_id="partner-user-1",
            probe_on_init=False,
        )
        headers = mock_http_cls.call_args[1]["headers"]
        assert headers == {
            "Authorization": "Bearer arena_org_secret",
            EXTERNAL_USER_ID_HEADER: "partner-user-1",
        }

    @patch("agilerl.inference.agent.httpx.Client")
    def test_org_key_wins_over_api_key(self, mock_http_cls):
        Agent(
            "http://endpoint",
            api_key="arena_pat_abc",
            org_key="arena_org_secret",
            external_user_id="partner-user-1",
            probe_on_init=False,
        )
        headers = mock_http_cls.call_args[1]["headers"]
        assert headers == {
            "Authorization": "Bearer arena_org_secret",
            EXTERNAL_USER_ID_HEADER: "partner-user-1",
        }

    def test_org_key_without_external_user_id_raises(self):
        with pytest.raises(ArenaAuthError, match="must be set together"):
            Agent(
                "http://endpoint",
                org_key="arena_org_secret",
                probe_on_init=False,
            )


def _recording_transport(seen: list[str | None]) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers.get("authorization"))
        return httpx.Response(200, json=_body_for_path(request.url.path))

    return httpx.MockTransport(handler)


class TestAgentCallableCredential:
    def test_reads_the_credential_for_every_request(self):
        # Arrange
        seen: list[str | None] = []
        tokens = iter(["tok-1", "tok-2", "tok-3"])
        real_client = httpx.Client
        transport = _recording_transport(seen)

        # Act
        with patch(
            "agilerl.inference.agent.httpx.Client",
            lambda **kwargs: real_client(transport=transport, **kwargs),
        ):
            agent = Agent("http://endpoint", api_key=lambda: next(tokens))
        agent.generate("hi")
        agent.generate("hi again")

        # Assert
        assert seen == ["Bearer tok-1", "Bearer tok-2", "Bearer tok-3"]

    def test_sends_no_header_while_the_callable_returns_none(self):
        seen: list[str | None] = []
        real_client = httpx.Client
        transport = _recording_transport(seen)

        with patch(
            "agilerl.inference.agent.httpx.Client",
            lambda **kwargs: real_client(transport=transport, **kwargs),
        ):
            Agent("http://endpoint", api_key=lambda: None)

        assert seen == [None]

    @patch("agilerl.inference.agent.httpx.Client")
    def test_sets_no_static_header(self, mock_http_cls):
        Agent("http://endpoint", api_key=lambda: SECRET, probe_on_init=False)

        assert mock_http_cls.call_args[1]["headers"] == {}


SECRET = "arena_pat_supersecret"


def _body_for_path(path: str) -> dict[str, Any]:
    if path == "/status":
        return {"success": True, "agent": {"algo": "GRPO", "llm": True}}
    if path == "/generate":
        return {
            "results": [{"prompt": "hi", "completion": "yo"}],
            "batch_size": 1,
            "inference_time_ms": 1.0,
            "tokens_per_second": 1.0,
            "success": True,
        }
    if path == "/sessions":
        return {"success": True, "sessions": []}
    return {}


def _agent_over_mock_transport(
    seen: list[tuple[str, str | None]],
    *,
    status_code: int = 200,
) -> Agent:
    """An LLM agent whose transport records the path and Authorization it saw."""

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((request.url.path, request.headers.get("authorization")))
        return httpx.Response(status_code, json=_body_for_path(request.url.path))

    agent = _llm_agent()
    agent._http = httpx.Client(
        headers={"Authorization": f"Bearer {SECRET}"},
        transport=httpx.MockTransport(handler),
    )
    return agent


class TestSingleCredentialOnEveryRoute:
    def test_same_header_on_status_generate_and_sessions(self):
        seen: list[tuple[str, str | None]] = []
        agent = _agent_over_mock_transport(seen)

        agent.status(refresh=True)
        agent.generate("hi")
        agent.list_sessions()

        assert [path for path, _ in seen] == ["/status", "/generate", "/sessions"]
        assert {auth for _, auth in seen} == {f"Bearer {SECRET}"}


class TestCredentialNeverLeaks:
    @patch("agilerl.inference.agent.httpx.Client")
    def test_absent_from_repr(self, mock_http_cls):
        agent = Agent("http://endpoint", api_key=SECRET, probe_on_init=False)
        assert repr(agent) == "<Agent endpoint='http://endpoint'>"

    @patch("agilerl.inference.agent.httpx.Client")
    def test_not_stored_as_instance_attribute(self, mock_http_cls):
        agent = Agent("http://endpoint", api_key=SECRET, probe_on_init=False)
        assert SECRET not in json.dumps(agent.__dict__, default=str)

    @patch("agilerl.inference.cache._write_store")
    @patch(
        "agilerl.inference.cache._load_store",
        return_value={"deployments": {"other": {"url": "http://o", "api_key": SECRET}}},
    )
    def test_save_binding_purges_credentials_left_by_older_releases(
        self, mock_load, mock_write
    ):
        save_binding("dep1", "http://url")
        assert SECRET not in json.dumps(mock_write.call_args[0][0])

    def test_absent_from_raised_error_detail(self):
        seen: list[tuple[str, str | None]] = []
        agent = _agent_over_mock_transport(seen, status_code=500)

        with pytest.raises(ArenaInferenceError) as exc_info:
            agent.generate("hi")
        assert seen[0][1] == f"Bearer {SECRET}"
        assert SECRET not in str(exc_info.value)


class TestForbiddenErrors:
    def test_403_asks_for_login(self):
        agent = _llm_agent()
        agent._http.request.return_value = _json_response(403, {"error": "forbidden"})

        with pytest.raises(ArenaAuthError, match="not allowed"):
            agent.generate("hi")

    def test_403_on_stream_asks_for_login(self):
        agent = _llm_agent()
        agent._http.stream.return_value = _stream_response([], status_code=403)

        with pytest.raises(ArenaAuthError, match="not allowed"):
            list(agent.generate_stream("hi"))


class TestSessions:
    def test_list_sessions_parses_the_top_level_array(self):
        agent = _llm_agent()
        agent._http.request.return_value = _json_response(
            200,
            [
                {
                    "session_id": "s1",
                    "created_at": "2026-07-30T00:00:00Z",
                    "last_updated": "2026-08-01T00:00:00Z",
                    "created_by": "user-1",
                    "title": "chat",
                },
                {"session_id": "s2", "created_at": None, "last_updated": None},
            ],
        )

        sessions = agent.list_sessions()
        assert [s.session_id for s in sessions] == ["s1", "s2"]
        assert sessions[0].created_at == "2026-07-30T00:00:00Z"
        assert sessions[0].last_updated == "2026-08-01T00:00:00Z"
        assert sessions[1].last_updated is None
        assert isinstance(sessions[0], SessionInfo)
        assert not hasattr(sessions[0], "created_by")
        assert not hasattr(sessions[0], "title")
        assert agent._http.request.call_args[0][1] == "http://test/sessions"

    def test_list_sessions_preserves_deployment_ordering(self):
        agent = _llm_agent()
        agent._http.request.return_value = _json_response(
            200,
            [
                {"session_id": "newest", "last_updated": "2026-08-02T00:00:00Z"},
                {"session_id": "oldest", "last_updated": "2026-07-01T00:00:00Z"},
            ],
        )
        assert [s.session_id for s in agent.list_sessions()] == ["newest", "oldest"]

    def test_list_sessions_accepts_a_wrapped_object(self):
        agent = _llm_agent()
        agent._http.request.return_value = _json_response(
            200, {"success": True, "sessions": [{"session_id": "s1"}]}
        )
        assert [s.session_id for s in agent.list_sessions()] == ["s1"]

    def test_list_sessions_ignores_unknown_fields(self):
        agent = _llm_agent()
        agent._http.request.return_value = _json_response(
            200,
            [{"session_id": "s1", "session_name": "chat", "turns": 4}],
        )

        sessions = agent.list_sessions()
        assert sessions[0].session_id == "s1"
        assert not hasattr(sessions[0], "session_name")

    def test_list_sessions_with_no_conversations(self):
        agent = _llm_agent()
        agent._http.request.return_value = _json_response(200, [])
        assert agent.list_sessions() == []

    def test_list_sessions_without_sessions_key(self):
        agent = _llm_agent()
        agent._http.request.return_value = _json_response(200, {})
        assert agent.list_sessions() == []

    def test_list_sessions_with_non_list_payload(self):
        agent = _llm_agent()
        agent._http.request.return_value = _json_response(200, {"sessions": "nope"})
        assert agent.list_sessions() == []

    def test_a_route_expecting_an_object_still_rejects_an_array(self):
        agent = _llm_agent()
        agent._http.request.return_value = _json_response(200, [{"prompt": "hi"}])

        with pytest.raises(ArenaInferenceError, match="Expected JSON object"):
            agent.generate("hi")

    def test_list_sessions_requires_llm_deployment(self):
        agent = _llm_agent()
        agent.metadata = _default_metadata()
        with pytest.raises(ArenaInferenceError, match="not an LLM"):
            agent.list_sessions()

    def test_get_session_parses_messages(self):
        agent = _llm_agent()
        agent._http.request.return_value = _json_response(
            200,
            {
                "session_id": "s1",
                "last_updated": "2026-08-01T00:00:00Z",
                "success": True,
                "messages": [
                    {"role": "user", "content": "hi"},
                    {"role": "assistant", "content": "hello"},
                ],
            },
        )

        detail = agent.get_session("s1")
        assert isinstance(detail, SessionDetail)
        assert detail.session_id == "s1"
        assert [m.role for m in detail.messages] == ["user", "assistant"]
        assert not hasattr(detail, "success")
        assert agent._http.request.call_args[0][1] == "http://test/sessions/s1"

    def test_delete_session_on_204(self):
        agent = _llm_agent()
        agent._http.request.return_value = _json_response(204, None)

        assert agent.delete_session("s1") is True
        method, url = agent._http.request.call_args[0]
        assert (method, url) == ("DELETE", "http://test/sessions/s1")

    def test_delete_session_sends_no_body(self):
        agent = _llm_agent()
        agent._http.request.return_value = _json_response(204, None)
        agent.delete_session("s1")
        assert agent._http.request.call_args.kwargs["json"] is None

    def test_delete_session_never_reads_the_body(self):
        """A 204 carries no body, so decoding one would raise."""
        agent = _llm_agent()
        resp = _json_response(204, None)
        resp.json.side_effect = ValueError("no body to decode")
        agent._http.request.return_value = resp

        assert agent.delete_session("s1") is True

    def test_delete_session_accepts_200_as_well(self):
        agent = _llm_agent()
        agent._http.request.return_value = _json_response(200, {"success": True})
        assert agent.delete_session("s1") is True

    def test_delete_session_404_reports_nothing_deleted(self):
        agent = _llm_agent()
        agent._http.request.return_value = _json_response(
            404, {"detail": "unknown session_id"}
        )
        assert agent.delete_session("s1") is False

    def test_delete_session_strips_the_id(self):
        agent = _llm_agent()
        agent._http.request.return_value = _json_response(204, None)
        agent.delete_session("  s1  ")
        assert agent._http.request.call_args[0][1] == "http://test/sessions/s1"

    def test_delete_session_rejects_blank_id(self):
        agent = _llm_agent()
        with pytest.raises(ArenaInferenceError, match="session_id"):
            agent.delete_session("")

    def test_delete_session_requires_llm_deployment(self):
        agent = _llm_agent()
        agent.metadata = _default_metadata()
        with pytest.raises(ArenaInferenceError, match="not an LLM"):
            agent.delete_session("s1")

    @pytest.mark.parametrize("status", [401, 403])
    def test_delete_session_auth_failures_still_raise(self, status):
        agent = _llm_agent()
        agent._http.request.return_value = _json_response(status, {"error": "nope"})
        with pytest.raises(ArenaAuthError):
            agent.delete_session("s1")

    def test_delete_session_server_error_still_raises(self):
        agent = _llm_agent()
        agent._http.request.return_value = _json_response(500, {"error": "boom"})
        with pytest.raises(ArenaInferenceError):
            agent.delete_session("s1")

    def test_get_session_rejects_blank_id(self):
        agent = _llm_agent()
        with pytest.raises(ArenaInferenceError, match="session_id"):
            agent.get_session("")

    def test_get_session_404_names_the_session(self):
        agent = _llm_agent()
        agent._http.request.return_value = _json_response(404, {"error": "missing"})

        with pytest.raises(ArenaInferenceError) as exc_info:
            agent.get_session("s1")
        assert exc_info.value.status_code == 404
        assert "s1" in exc_info.value.detail

    def test_get_session_other_error_passes_through(self):
        agent = _llm_agent()
        agent._http.request.return_value = _json_response(500, {"error": "boom"})

        with pytest.raises(ArenaInferenceError) as exc_info:
            agent.get_session("s1")
        assert exc_info.value.status_code == 500


class TestAgentContextManager:
    def test_enter_exit(self):
        agent = Agent.__new__(Agent)
        agent._http = MagicMock()

        assert agent.__enter__() is agent
        agent.__exit__(None, None, None)
        agent._http.close.assert_called_once()

    def test_close(self):
        agent = Agent.__new__(Agent)
        agent._http = MagicMock()
        agent.close()
        agent._http.close.assert_called_once()


class TestAgentRepr:
    def test_repr(self):
        agent = Agent.__new__(Agent)
        agent._base_url = "http://test"
        assert repr(agent) == "<Agent endpoint='http://test'>"


class TestLoadBinding:
    @patch("agilerl.inference.cache._load_store", return_value={})
    def test_no_deployment_key(self, mock_load_store):
        assert load_binding("my-dep") is None

    @patch(
        "agilerl.inference.cache._load_store",
        return_value={"deployments": "not-a-dict"},
    )
    def test_deployment_key_not_dict(self, mock_load_store):
        assert load_binding("my-dep") is None

    @patch(
        "agilerl.inference.cache._load_store",
        return_value={"deployments": {}},
    )
    def test_missing_entry(self, mock_load_store):
        assert load_binding("missing") is None

    @patch(
        "agilerl.inference.cache._load_store",
        return_value={"deployments": {"my-dep": "not-a-dict"}},
    )
    def test_entry_not_dict(self, mock_load_store):
        assert load_binding("my-dep") is None

    @patch(
        "agilerl.inference.cache._load_store",
        return_value={"deployments": {"my-dep": {"url": "https://x.com"}}},
    )
    def test_valid_entry(self, mock_load_store):
        assert load_binding("my-dep") == "https://x.com"

    @patch(
        "agilerl.inference.cache._load_store",
        return_value={"deployments": {"my-dep": {"url": 123}}},
    )
    def test_url_not_string(self, mock_load_store):
        assert load_binding("my-dep") is None

    @patch(
        "agilerl.inference.cache._load_store",
        return_value={"deployments": {"my-dep": {"url": ""}}},
    )
    def test_empty_url(self, mock_load_store):
        assert load_binding("my-dep") is None

    @patch(
        "agilerl.inference.cache._load_store",
        return_value={"deployments": {"my-dep": {"url": "  http://x  "}}},
    )
    def test_strips_whitespace(self, mock_load_store):
        assert load_binding("my-dep") == "http://x"

    @pytest.mark.parametrize("stale_key", ["key123", None, ""])
    def test_entry_carrying_an_api_key_is_stale(self, stale_key):
        store = {"deployments": {"my-dep": {"url": "http://x", "api_key": stale_key}}}
        with patch("agilerl.inference.cache._load_store", return_value=store):
            assert load_binding("my-dep") is None


class TestSaveBinding:
    @patch("agilerl.inference.cache._write_store")
    @patch(
        "agilerl.inference.cache._load_store",
        return_value={},
    )
    def test_creates_new_section(self, mock_load, mock_write):
        save_binding("dep1", "http://url")
        written = mock_write.call_args[0][0]
        assert written["deployments"]["dep1"] == {"url": "http://url"}

    @patch("agilerl.inference.cache._write_store")
    @patch(
        "agilerl.inference.cache._load_store",
        return_value={"deployments": {"existing": {"url": "http://old"}}},
    )
    def test_merges_with_existing(self, mock_load, mock_write):
        save_binding("new-dep", "http://new")
        written = mock_write.call_args[0][0]
        assert "existing" in written["deployments"]
        assert written["deployments"]["new-dep"] == {"url": "http://new"}

    @patch("agilerl.inference.cache._write_store")
    @patch(
        "agilerl.inference.cache._load_store",
        return_value={"deployments": "corrupt"},
    )
    def test_replaces_non_dict_section(self, mock_load, mock_write):
        save_binding("dep1", "http://url")
        written = mock_write.call_args[0][0]
        assert isinstance(written["deployments"], dict)
        assert "dep1" in written["deployments"]

    @patch("agilerl.inference.cache._write_store")
    @patch(
        "agilerl.inference.cache._load_store",
        return_value={},
    )
    def test_strips_whitespace(self, mock_load, mock_write):
        save_binding("  dep1  ", "  http://url  ")
        written = mock_write.call_args[0][0]
        assert "dep1" in written["deployments"]
        assert written["deployments"]["dep1"] == {"url": "http://url"}

    @patch("agilerl.inference.cache._write_store")
    @patch(
        "agilerl.inference.cache._load_store",
        return_value={
            "deployments": {"untouched": {"url": "http://o", "api_key": "stale"}}
        },
    )
    def test_scrubs_credentials_from_entries_it_is_not_writing(
        self, mock_load, mock_write
    ):
        save_binding("dep1", "http://new")
        written = mock_write.call_args[0][0]
        assert written["deployments"]["untouched"] == {"url": "http://o"}
        assert "stale" not in json.dumps(written)


class TestAgentHttpHelpers:
    def test_url_returns_absolute_path_unchanged(self):
        agent = Agent.__new__(Agent)
        agent._base_url = "http://test"
        assert agent._url("https://other.example/v1") == "https://other.example/v1"

    def test_request_json_rejects_non_dict_body(self):
        agent = Agent.__new__(Agent)
        agent._base_url = "http://test"
        agent._http = MagicMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = ["not", "a", "dict"]
        agent._http.request.return_value = mock_resp
        with pytest.raises(ArenaInferenceError, match="JSON object"):
            agent._request_json("GET", "/status")

    def test_request_json_rejects_success_false(self):
        agent = Agent.__new__(Agent)
        agent._base_url = "http://test"
        agent._http = MagicMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"success": False, "detail": "nope"}
        mock_resp.text = "nope"
        agent._http.request.return_value = mock_resp
        with pytest.raises(ArenaInferenceError):
            agent._request_json("POST", "/predict", json_body={})

    def test_request_stream_wraps_httpx_error(self):
        agent = Agent.__new__(Agent)
        agent._base_url = "http://test"
        agent._http = MagicMock()
        agent._http.stream.side_effect = httpx.ConnectError("refused")
        with pytest.raises(ArenaInferenceError, match="Network error"):
            list(agent._request_stream("/generate_stream", json_body={"prompt": "hi"}))

    def test_agent_info_requires_metadata(self):
        agent = Agent.__new__(Agent)
        agent.metadata = None
        with pytest.raises(ArenaInferenceError, match="Metadata not loaded"):
            agent._agent_info()

    def test_ensure_llm_rejects_a_non_llm_deployment(self):
        agent = Agent.__new__(Agent)
        agent.metadata = _default_metadata(llm=False, algo="DQN")
        with pytest.raises(ArenaInferenceError, match="supports only LLM deployments"):
            agent._ensure_llm()

    def test_validate_prompts_rejects_empty_list(self):
        with pytest.raises(ArenaInferenceError, match="At least one prompt"):
            Agent._validate_prompts([])


class TestInferenceCacheOnDisk:
    def test_load_store_corrupt_json_returns_empty(self, monkeypatch, tmp_path):
        inference_file = tmp_path / "inference.json"
        inference_file.write_text("{not json", encoding="utf-8")
        monkeypatch.setattr("agilerl.inference.cache.INFERENCE_FILE", inference_file)
        assert load_binding("dep") is None

    def test_load_active_agent_empty_deployment_name(self, monkeypatch, tmp_path):
        inference_file = tmp_path / "inference.json"
        inference_file.write_text(
            json.dumps({"active_agent": {"deployment": "  "}}),
            encoding="utf-8",
        )
        monkeypatch.setattr("agilerl.inference.cache.INFERENCE_FILE", inference_file)
        assert load_active_agent() is None

    def test_save_binding_round_trip(self, monkeypatch, tmp_path):
        inference_file = tmp_path / ".arena" / "inference.json"
        monkeypatch.setattr("agilerl.inference.cache.INFERENCE_FILE", inference_file)
        save_binding("dep-a", "http://x")
        assert load_binding("dep-a") == "http://x"
        assert "api_key" not in inference_file.read_text(encoding="utf-8")

    def test_stale_entry_is_rewritten_without_the_credential(
        self, monkeypatch, tmp_path
    ):
        inference_file = tmp_path / ".arena" / "inference.json"
        inference_file.parent.mkdir(parents=True, exist_ok=True)
        inference_file.write_text(
            json.dumps(
                {
                    "deployments": {
                        "dep-a": {"url": "http://old", "api_key": "stale-secret"}
                    }
                }
            ),
            encoding="utf-8",
        )
        monkeypatch.setattr("agilerl.inference.cache.INFERENCE_FILE", inference_file)

        assert load_binding("dep-a") is None

        save_binding("dep-a", "http://new")
        assert load_binding("dep-a") == "http://new"
        assert "stale-secret" not in inference_file.read_text(encoding="utf-8")


class TestNormalizedDeploymentName:
    def test_strips_whitespace(self):
        assert normalized_deployment_name("  my-dep  ") == "my-dep"

    def test_no_change_needed(self):
        assert normalized_deployment_name("my-dep") == "my-dep"


class TestActiveAgent:
    @patch("agilerl.inference.cache._load_store", return_value={})
    def test_load_missing(self, mock_load_store):
        assert load_active_agent() is None

    @patch(
        "agilerl.inference.cache._load_store",
        return_value={"active_agent": "not-a-dict"},
    )
    def test_load_invalid_section(self, mock_load_store):
        assert load_active_agent() is None

    @patch("agilerl.inference.cache._write_store")
    @patch(
        "agilerl.inference.cache._load_store",
        return_value={},
    )
    def test_save_and_load(self, mock_load_store_fn, mock_write):
        save_active_agent("my-dep", experiment_name="exp1", project_name="proj1")
        written = mock_write.call_args[0][0]
        assert written["active_agent"] == {
            "deployment": "my-dep",
            "experiment_name": "exp1",
            "project_name": "proj1",
        }

    @patch(
        "agilerl.inference.cache._load_store",
        return_value={
            "active_agent": {
                "deployment": "  my-dep  ",
                "experiment_name": " exp ",
                "project_name": " proj ",
            }
        },
    )
    def test_load_strips_fields(self, mock_load_store):
        result = load_active_agent()
        assert result == ActiveAgentSelection(
            deployment_name="my-dep",
            experiment_name="exp",
            project_name="proj",
        )


class TestActiveSession:
    """These round-trip through the tmp inference.json from ``conftest``."""

    def test_round_trip(self):
        save_active_session("my-dep", "sess-1")
        assert load_active_session("my-dep") == "sess-1"

    def test_missing_returns_none(self):
        assert load_active_session("my-dep") is None

    def test_sessions_are_per_deployment(self):
        save_active_session("dep-a", "sess-a")
        save_active_session("dep-b", "sess-b")
        assert load_active_session("dep-a") == "sess-a"
        assert load_active_session("dep-b") == "sess-b"

    def test_resuming_again_replaces_the_previous_session(self):
        save_active_session("my-dep", "sess-1")
        save_active_session("my-dep", "sess-2")
        assert load_active_session("my-dep") == "sess-2"

    def test_names_and_ids_are_stripped(self):
        save_active_session("  my-dep  ", "  sess-1  ")
        assert load_active_session("my-dep") == "sess-1"

    def test_clear_reports_that_it_removed_one(self):
        save_active_session("my-dep", "sess-1")
        assert clear_active_session("my-dep") is True
        assert load_active_session("my-dep") is None

    def test_clear_reports_nothing_to_remove(self):
        assert clear_active_session("my-dep") is False

    def test_clear_reports_nothing_when_other_deployments_have_sessions(self):
        save_active_session("other-dep", "sess-b")
        assert clear_active_session("my-dep") is False
        assert load_active_session("other-dep") == "sess-b"

    def test_clear_leaves_other_deployments_alone(self):
        save_active_session("dep-a", "sess-a")
        save_active_session("dep-b", "sess-b")
        clear_active_session("dep-a")
        assert load_active_session("dep-b") == "sess-b"

    def test_sessions_survive_a_binding_write(self):
        save_active_session("my-dep", "sess-1")
        save_binding("my-dep", "http://pod")
        assert load_active_session("my-dep") == "sess-1"

    def test_a_binding_survives_resuming_a_session(self):
        save_binding("my-dep", "http://pod")
        save_active_session("my-dep", "sess-1")
        assert load_binding("my-dep") == "http://pod"

    @patch(
        "agilerl.inference.cache._load_store",
        return_value={"active_sessions": "not-a-dict"},
    )
    def test_load_invalid_section(self, mock_load_store):
        assert load_active_session("my-dep") is None

    @patch(
        "agilerl.inference.cache._load_store",
        return_value={"active_sessions": {"my-dep": "   "}},
    )
    def test_load_blank_session_id(self, mock_load_store):
        assert load_active_session("my-dep") is None


MINIMAL_STATUS = {"deployment_id": "560"}


def _agent_with_status(status_body: dict) -> Agent:
    """A real Agent, built through __init__, whose /status returns *status_body*."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/status":
            return httpx.Response(200, json=status_body)
        return httpx.Response(200, json={})

    real = httpx.Client
    with patch(
        "agilerl.inference.agent.httpx.Client",
        lambda **kw: real(transport=httpx.MockTransport(handler), **kw),
    ):
        return Agent("http://pod", api_key="k")


class TestStatusWithoutAgentInfo:
    """A public /status may report only the deployment id."""

    def test_metadata_parses_with_no_agent_block(self):
        agent = _agent_with_status(MINIMAL_STATUS)
        assert agent.metadata is not None
        assert agent.metadata.deployment_id == "560"
        assert agent.metadata.agent is None

    def test_an_absent_agent_block_is_not_a_reported_shape(self):
        """Absent must not read as 'every flag is False'."""
        assert _agent_with_status(MINIMAL_STATUS).metadata.agent is None
        reported = _agent_with_status(
            {"deployment_id": "1", "agent": {}}
        ).metadata.agent
        assert reported is not None
        assert reported.llm is False

    @pytest.mark.parametrize(
        "call",
        [
            lambda a: a.generate("hi"),
            lambda a: list(a.generate_stream("hi")),
            lambda a: a.list_sessions(),
            lambda a: a.get_session("s1"),
            lambda a: a.delete_session("s1"),
        ],
    )
    def test_type_guards_defer_to_the_deployment(self, call):
        """Unknown type must not be refused locally; the route answers 400."""
        agent = _agent_with_status(MINIMAL_STATUS)
        agent._http = MagicMock()
        agent._http.request.return_value = _json_response(
            400, {"error": "wrong endpoint"}
        )
        agent._http.stream.return_value = _stream_response([], status_code=400)

        with pytest.raises(ArenaInferenceError) as exc_info:
            call(agent)
        assert exc_info.value.status_code == 400
        assert "Deployment is not" not in str(exc_info.value)

    def test_non_llm_status_fields_are_ignored(self):
        agent = _agent_with_status(
            {
                "deployment_id": "1",
                "agent": {"multi_agent": True, "recurrent": True, "supervised": True},
            }
        )

        assert agent.metadata.agent is not None
        assert agent.metadata.agent.model_dump() == {"algo": "", "llm": False}

    def test_reported_llm_still_guards(self):
        agent = _agent_with_status({"deployment_id": "1", "agent": {"llm": False}})
        with pytest.raises(ArenaInferenceError, match="not an LLM"):
            agent.generate("hi")

    def test_metadata_not_loaded_still_raises(self):
        agent = _agent_with_status(MINIMAL_STATUS)
        agent.metadata = None
        with pytest.raises(ArenaInferenceError, match="Metadata not loaded"):
            agent.generate("hi")


ASSERTION_HEADER = "x-arena-deployment-assertion"


class OnPremPod:
    """A deployment behind the on-prem proxy, which drops ``Authorization``."""

    def __init__(self) -> None:
        self.valid: set[str] = set()
        self.seen: list[tuple[str, str | None]] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        assertion = request.headers.get(ASSERTION_HEADER)
        self.seen.append((path, assertion))
        if path == "/status":
            return httpx.Response(
                200, json={"deployment_id": "658", "agent": {"llm": True}}
            )
        if assertion not in self.valid:
            return httpx.Response(401, json={"detail": "unauthorized"})
        if path == "/generate_stream":
            return httpx.Response(200, text="42")
        return httpx.Response(200, json=_body_for_path(path))


class AssertionMinter:
    """Stands in for Arena's ``POST /api/deployments/{id}/inference-assertion``."""

    def __init__(self, pod: OnPremPod | None) -> None:
        self.pod = pod
        self.minted_for: list[str] = []

    def __call__(self, deployment_id: str) -> DeploymentAssertion:
        self.minted_for.append(deployment_id)
        assertion = f"arena_dpa_{len(self.minted_for)}"
        if self.pod is not None:
            self.pod.valid = {assertion}
        return DeploymentAssertion(
            assertion=assertion, header=ASSERTION_HEADER, expires_in=60
        )


def _agent_over(
    handler: Callable[[httpx.Request], httpx.Response], minter: AssertionMinter
) -> Agent:
    real = httpx.Client
    with patch(
        "agilerl.inference.agent.httpx.Client",
        lambda **kw: real(transport=httpx.MockTransport(handler), **kw),
    ):
        return Agent("http://pod", api_key=SECRET, assertion_source=minter)


class TestDeploymentAssertion:
    def test_generate_mints_for_the_status_deployment_id_and_retries(self):
        pod = OnPremPod()
        minter = AssertionMinter(pod)
        agent = _agent_over(pod.handler, minter)

        result = agent.generate("hi")

        assert result.results[0].completion == "yo"
        assert minter.minted_for == ["658"]
        assert pod.seen == [
            ("/status", None),
            ("/generate", None),
            ("/generate", "arena_dpa_1"),
        ]

    def test_generate_stream_mints_and_retries(self):
        pod = OnPremPod()
        agent = _agent_over(pod.handler, AssertionMinter(pod))

        assert "".join(agent.generate_stream("hi")) == "42"

    def test_later_requests_reuse_the_assertion_until_it_nears_expiry(self):
        pod = OnPremPod()
        minter = AssertionMinter(pod)
        agent = _agent_over(pod.handler, minter)
        clock = [1000.0]

        with patch("agilerl.inference.agent.time.monotonic", lambda: clock[0]):
            agent.generate("hi")
            clock[0] += 45
            agent.list_sessions()
            clock[0] += 10
            agent.list_sessions()

        assert minter.minted_for == ["658", "658"]
        assert pod.seen[-2:] == [
            ("/sessions", "arena_dpa_1"),
            ("/sessions", "arena_dpa_2"),
        ]

    def test_callable_credential_is_sent_with_the_assertion_on_retry(self):
        # Arrange
        pod = OnPremPod()
        minter = AssertionMinter(pod)
        tokens = iter(["tok-1", "tok-2", "tok-3"])
        authorizations: list[str | None] = []

        def handler(request: httpx.Request) -> httpx.Response:
            authorizations.append(request.headers.get("authorization"))
            return pod.handler(request)

        real = httpx.Client
        with patch(
            "agilerl.inference.agent.httpx.Client",
            lambda **kw: real(transport=httpx.MockTransport(handler), **kw),
        ):
            agent = Agent(
                "http://pod", api_key=lambda: next(tokens), assertion_source=minter
            )

        # Act
        agent.generate("hi")

        # Assert
        assert authorizations == ["Bearer tok-1", "Bearer tok-2", "Bearer tok-2"]
        assert pod.seen[-1] == ("/generate", "arena_dpa_1")

    def test_cloud_deployment_never_mints(self):
        seen: list[str | None] = []

        def cloud(request: httpx.Request) -> httpx.Response:
            seen.append(request.headers.get(ASSERTION_HEADER))
            return httpx.Response(200, json=_body_for_path(request.url.path))

        minter = AssertionMinter(None)
        agent = _agent_over(cloud, minter)
        agent.generate("hi")
        agent.list_sessions()

        assert minter.minted_for == []
        assert seen == [None, None, None]

    def test_still_unauthorized_after_one_retry_asks_for_login(self):
        def always_401(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/status":
                return httpx.Response(200, json={"deployment_id": "7"})
            return httpx.Response(401, json={})

        minter = AssertionMinter(None)
        agent = _agent_over(always_401, minter)

        with pytest.raises(ArenaAuthError, match="401 Unauthorized"):
            agent.generate("hi")
        assert minter.minted_for == ["7"]

    def test_a_401_before_metadata_loads_raises_instead_of_minting(self):
        minter = AssertionMinter(None)
        real = httpx.Client
        with patch(
            "agilerl.inference.agent.httpx.Client",
            lambda **kw: real(
                transport=httpx.MockTransport(lambda _: httpx.Response(401)), **kw
            ),
        ):
            agent = Agent("http://pod", probe_on_init=False, assertion_source=minter)

        with pytest.raises(ArenaInferenceError, match="Metadata not loaded"):
            agent.status()
        assert minter.minted_for == []

    def test_assertion_absent_from_repr(self):
        assertion = DeploymentAssertion(
            assertion="arena_dpa_secret", header=ASSERTION_HEADER, expires_in=60
        )
        assert "arena_dpa_secret" not in repr(assertion)
