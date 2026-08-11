# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Tests for arena.inference (agent, cache, serde)."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch

import httpx
import numpy as np
import pytest

from agilerl.arena.exceptions import ArenaAuthError, ArenaInferenceError
from agilerl.arena.inference import (
    Agent,
    LLMParams,
    LLMResults,
    PredictResult,
    SessionDetail,
    SessionInfo,
    StatusResponse,
)
from agilerl.arena.inference.cache import (
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
        "supervised": False,
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


class TestSerializeDeserialize:
    def test_1d_array_unbatched_round_trip(self):
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        encoded = Agent.serialize(arr, batched=False)
        decoded = Agent.deserialize(encoded, batched=False)
        np.testing.assert_array_equal(decoded, arr)

    def test_2d_array_batched_round_trip(self):
        arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
        encoded = Agent.serialize(arr, batched=True)
        decoded = Agent.deserialize(encoded, batched=True)
        np.testing.assert_array_equal(decoded, arr)

    def test_unbatched_adds_and_removes_batch_dim(self):
        arr = np.array([10, 20, 30])
        encoded = Agent.serialize(arr, batched=False)
        # The encoded version has a batch dim
        decoded_batched = Agent.deserialize(encoded, batched=True)
        assert decoded_batched.shape[0] == 1
        # Unbatched decode removes it
        decoded = Agent.deserialize(encoded, batched=False)
        np.testing.assert_array_equal(decoded, arr)

    def test_dict_round_trip(self):
        data = {
            "obs": np.array([1.0, 2.0]),
            "vel": np.array([3.0, 4.0]),
        }
        encoded = Agent.serialize(data, batched=False)
        assert isinstance(encoded, dict)
        decoded = Agent.deserialize(encoded, batched=False)
        for k in data:
            np.testing.assert_array_equal(decoded[k], data[k])

    def test_tuple_round_trip(self):
        data = (np.array([1.0]), np.array([2.0]))
        encoded = Agent.serialize(data, batched=False)
        assert isinstance(encoded, tuple)
        decoded = Agent.deserialize(encoded, batched=False)
        assert isinstance(decoded, tuple)
        for orig, dec in zip(data, decoded, strict=False):
            np.testing.assert_array_equal(dec, orig)

    def test_none_passthrough(self):
        assert Agent.serialize(None) is None
        assert Agent.deserialize(None) is None

    def test_nested_dict_with_tuple_values(self):
        data = {"sensor": (np.array([1.0, 2.0]), np.array([3.0]))}
        encoded = Agent.serialize(data, batched=False)
        decoded = Agent.deserialize(encoded, batched=False)
        np.testing.assert_array_equal(decoded["sensor"][0], data["sensor"][0])
        np.testing.assert_array_equal(decoded["sensor"][1], data["sensor"][1])

    def test_list_treated_as_tuple(self):
        data = [np.array([5.0]), np.array([6.0])]
        encoded = Agent.serialize(data, batched=False)
        assert isinstance(encoded, tuple)


class TestGetBatchSize:
    def test_flat_array(self):
        obs = np.zeros((4, 8))
        assert Agent.get_batch_size(obs) == 4

    def test_dict_observation(self):
        obs = {"image": np.zeros((3, 64, 64)), "vector": np.zeros((3, 10))}
        assert Agent.get_batch_size(obs) == 3

    def test_tuple_observation(self):
        obs = (np.zeros((2, 4)),)
        assert Agent.get_batch_size(obs) == 2

    def test_nested_dict_tuple(self):
        obs = {"a": (np.zeros((5, 3)),)}
        assert Agent.get_batch_size(obs) == 5

    def test_none_leaf_raises(self):
        with pytest.raises(ValueError, match="None observation"):
            Agent.get_batch_size({"a": None})


class TestBuildPayload:
    def _make_agent(self):
        with patch("agilerl.arena.inference.agent.httpx.Client") as mock_http_cls:
            mock_client = MagicMock()
            mock_http_cls.return_value = mock_client
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = STATUS_BODY
            mock_client.request.return_value = mock_resp
            return Agent("http://test")

    def test_basic_payload(self):
        agent = self._make_agent()
        obs = np.array([1.0, 2.0])
        payload = agent._build_payload(
            obs,
            batched=False,
            hidden_state=None,
            info=None,
            env_defined_actions=None,
        )
        assert "obs" in payload
        assert payload["batch_size"] == 1
        assert "hidden_state" not in payload
        assert "action_mask" not in payload

    def test_batched_payload_batch_size(self):
        agent = self._make_agent()
        obs = np.zeros((4, 8))
        payload = agent._build_payload(
            obs,
            batched=True,
            hidden_state=None,
            info=None,
            env_defined_actions=None,
        )
        assert payload["batch_size"] == 4

    def test_action_mask_from_info(self):
        agent = self._make_agent()
        obs = np.array([1.0])
        mask = np.array([1, 0, 1])
        info = {"action_mask": mask}
        payload = agent._build_payload(
            obs,
            batched=False,
            hidden_state=None,
            info=info,
            env_defined_actions=None,
        )
        assert payload["action_mask"] is not None

    def test_hidden_state_serialized(self):
        agent = self._make_agent()
        obs = np.array([1.0])
        hs = {"h": np.array([0.5, 0.5])}
        agent.metadata = _default_metadata(recurrent=True)
        payload = agent._build_payload(
            obs,
            batched=False,
            hidden_state=hs,
            info=None,
            env_defined_actions=None,
        )
        assert payload["hidden_state"] is not None

    def test_non_recurrent_omits_hidden_state(self):
        agent = self._make_agent()
        obs = np.array([1.0])
        payload = agent._build_payload(
            obs,
            batched=False,
            hidden_state={"h": np.array([0.5])},
            info=None,
            env_defined_actions=None,
        )
        assert "hidden_state" not in payload

    def test_multi_agent_payload(self):
        agent = self._make_agent()
        agent.metadata = _default_metadata(multi_agent=True)
        obs = {"agent_0": np.array([1.0, 2.0]), "agent_1": np.array([3.0, 4.0])}
        masks = {
            "agent_0": np.array([1, 0]),
            "agent_1": np.array([0, 1]),
        }
        info = {"action_mask": masks}
        eda = {"agent_0": np.array([0.0])}
        payload = agent._build_payload(
            obs,
            batched=False,
            hidden_state=None,
            info=info,
            env_defined_actions=eda,
        )
        assert isinstance(payload["obs"], dict)
        assert "agent_0" in payload["obs"]
        assert "hidden_state" not in payload
        assert "action_mask" in payload
        assert "env_defined_actions" in payload

    def test_multi_agent_recurrent_includes_hidden_state(self):
        agent = self._make_agent()
        agent.metadata = _default_metadata(multi_agent=True, recurrent=True)
        obs = {"agent_0": np.array([1.0, 2.0]), "agent_1": np.array([3.0, 4.0])}
        hs = {"agent_0": np.array([0.5, 0.5]), "agent_1": np.array([0.1, 0.2])}
        payload = agent._build_payload(
            obs,
            batched=False,
            hidden_state=hs,
            info=None,
            env_defined_actions=None,
        )
        assert payload["hidden_state"] is not None

    def test_multi_agent_rejects_array_action_mask(self):
        agent = self._make_agent()
        agent.metadata = _default_metadata(multi_agent=True)
        obs = {"agent_0": np.array([1.0])}
        info = {"action_mask": np.array([1, 0])}
        with pytest.raises(ArenaInferenceError, match="dict\\[agent_id"):
            agent._build_payload(
                obs,
                batched=False,
                hidden_state=None,
                info=info,
                env_defined_actions=None,
            )


class TestParseGetActionResponse:
    def test_extracts_action_and_hidden_state_when_recurrent(self):
        action = np.array([2])
        hs = {"h": np.array([0.1, 0.2])}
        resp = {
            "action": Agent.serialize(action, batched=False),
            "hidden_state": Agent.serialize(hs, batched=False),
        }
        parsed_action, parsed_hs = Agent._parse_get_action_response(
            resp, batched=False, multi_agent=False, recurrent=True
        )
        np.testing.assert_array_equal(parsed_action, action)
        np.testing.assert_array_almost_equal(parsed_hs["h"], hs["h"])

    def test_ignores_hidden_state_when_not_recurrent(self):
        action = np.array([0])
        resp = {
            "action": Agent.serialize(action, batched=False),
            "hidden_state": Agent.serialize({"h": np.array([0.1])}, batched=False),
        }
        parsed_action, parsed_hs = Agent._parse_get_action_response(
            resp, batched=False, multi_agent=False, recurrent=False
        )
        np.testing.assert_array_equal(parsed_action, action)
        assert parsed_hs is None

    def test_multi_agent_action_dict(self):
        actions = {
            "agent_0": np.array([1]),
            "agent_1": np.array([2]),
        }
        resp = {
            "action": {
                k: Agent.serialize(v, batched=False) for k, v in actions.items()
            },
        }
        parsed_action, parsed_hs = Agent._parse_get_action_response(
            resp, batched=False, multi_agent=True, recurrent=False
        )
        assert isinstance(parsed_action, dict)
        np.testing.assert_array_equal(parsed_action["agent_0"], actions["agent_0"])
        np.testing.assert_array_equal(parsed_action["agent_1"], actions["agent_1"])
        assert parsed_hs is None


class TestGetAction:
    def _make_agent_with_mock_http(self):
        return _mock_agent()

    def _mock_json_response(self, status_code: int, body: dict):
        return _json_response(status_code, body)

    def test_successful_get_action(self):
        agent = self._make_agent_with_mock_http()
        action_arr = np.array([1])
        resp_json = {
            "action": Agent.serialize(action_arr, batched=False),
            "hidden_state": None,
        }
        agent._http.request.return_value = self._mock_json_response(200, resp_json)

        obs = np.array([1.0, 2.0, 3.0])
        action, hs = agent.get_action(obs)

        np.testing.assert_array_equal(action, action_arr)
        assert hs is None
        agent._http.request.assert_called_once()
        call_args = agent._http.request.call_args
        assert call_args[0][0] == "POST"
        assert call_args[0][1] == "http://test/get_action"

    def test_401_raises_auth_error(self):
        agent = self._make_agent_with_mock_http()
        agent._http.request.return_value = self._mock_json_response(401, {})

        with pytest.raises(ArenaAuthError, match="401 Unauthorized"):
            agent.get_action(np.array([1.0]))

    def test_non_200_raises_api_error(self):
        agent = self._make_agent_with_mock_http()
        agent._http.request.return_value = self._mock_json_response(
            500, {"error": "internal"}
        )

        with pytest.raises(ArenaInferenceError) as exc_info:
            agent.get_action(np.array([1.0]))
        assert exc_info.value.status_code == 500

    def test_network_error_raises_api_error(self):
        agent = self._make_agent_with_mock_http()
        agent._http.request.side_effect = httpx.ConnectError("refused")

        with pytest.raises(ArenaInferenceError) as exc_info:
            agent.get_action(np.array([1.0]))
        assert exc_info.value.status_code == 0

    def test_non_200_with_text_fallback(self):
        agent = self._make_agent_with_mock_http()
        mock_resp = MagicMock()
        mock_resp.status_code = 502
        mock_resp.json.side_effect = ValueError("not json")
        mock_resp.text = "Bad Gateway"
        agent._http.request.return_value = mock_resp

        with pytest.raises(ArenaInferenceError) as exc_info:
            agent.get_action(np.array([1.0]))
        assert "Bad Gateway" in exc_info.value.detail

    def test_400_from_server(self):
        agent = self._make_agent_with_mock_http()
        agent._http.request.return_value = self._mock_json_response(
            400, {"error": "wrong agent type"}
        )

        with pytest.raises(ArenaInferenceError) as exc_info:
            agent.get_action(np.array([1.0]))
        assert exc_info.value.status_code == 400


class TestAgentInit:
    @patch("agilerl.arena.inference.agent.httpx.Client")
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

    @patch("agilerl.arena.inference.agent.httpx.Client")
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

    @patch("agilerl.arena.inference.agent.httpx.Client")
    def test_probe_on_init_false_defers_metadata(self, mock_http_cls):
        mock_client = MagicMock()
        mock_http_cls.return_value = mock_client

        agent = Agent("http://endpoint", probe_on_init=False)
        assert agent.metadata is None
        mock_client.request.assert_not_called()

    @patch("agilerl.arena.inference.agent.httpx.Client")
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


class TestPredict:
    def _make_agent_with_mock_http(self):
        return _mock_agent(supervised=True, algo="SFT")

    def test_predict_round_trip(self):
        agent = self._make_agent_with_mock_http()
        inputs = np.array([1.0, 2.0, 3.0])
        results_arr = np.array([0.9, 0.1, 0.0])
        body = {
            "results": Agent.serialize(results_arr, batched=False),
            "batch_size": 1,
            "inference_time_ms": 12.3,
            "success": True,
        }
        agent._http.request.return_value = _json_response(200, body)

        results, meta = agent.predict(inputs)
        np.testing.assert_array_almost_equal(results, results_arr)
        assert isinstance(meta, PredictResult)
        assert meta.inference_time_ms == 12.3
        sent = agent._http.request.call_args[1]["json"]
        assert sent == {"inputs": Agent.serialize(inputs, batched=False)}

    def test_predict_rejects_null_results(self):
        agent = self._make_agent_with_mock_http()
        body = {
            "results": None,
            "batch_size": 1,
            "inference_time_ms": 1.0,
            "success": True,
        }
        agent._http.request.return_value = _json_response(200, body)

        with pytest.raises(ArenaInferenceError, match="no results"):
            agent.predict(np.array([1.0]))


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

    def test_generate_omits_session_id_when_none(self):
        agent = self._make_agent_with_mock_http()
        agent._http.request.return_value = _llm_response()

        agent.generate("hi")
        sent = agent._http.request.call_args[1]["json"]
        assert "session_id" not in sent

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

    def test_stream_omits_session_id_when_none(self):
        agent = _llm_agent()
        agent._http.stream.return_value = _stream_response(["ok"])

        list(agent.generate_stream("hi"))
        sent = agent._http.stream.call_args[1]["json"]
        assert "session_id" not in sent

    def test_stream_rejects_blank_session_id(self):
        agent = _llm_agent()
        with pytest.raises(ArenaInferenceError, match="session_id"):
            list(agent.generate_stream("hi", session_id=""))


class TestAgentCredentialResolution:
    @patch("agilerl.arena.inference.agent.httpx.Client")
    def test_explicit_api_key_wins(self, mock_http_cls, monkeypatch):
        monkeypatch.setenv("ARENA_API_KEY", "env-key")
        Agent("http://endpoint", api_key="explicit-key", probe_on_init=False)
        headers = mock_http_cls.call_args[1]["headers"]
        assert headers == {"Authorization": "Bearer explicit-key"}

    @patch("agilerl.arena.inference.agent.httpx.Client")
    def test_falls_back_to_env_var(self, mock_http_cls, monkeypatch):
        monkeypatch.setenv("ARENA_API_KEY", "arena_pat_env")
        Agent("http://endpoint", probe_on_init=False)
        headers = mock_http_cls.call_args[1]["headers"]
        assert headers == {"Authorization": "Bearer arena_pat_env"}

    @patch("agilerl.arena.inference.agent.httpx.Client")
    def test_no_credential_sends_no_auth_header(self, mock_http_cls):
        Agent("http://endpoint", probe_on_init=False)
        assert mock_http_cls.call_args[1]["headers"] == {}


SECRET = "arena_pat_supersecret"

# Satisfies StatusResponse, LLMResults and the sessions body at once.
_ANY_ROUTE_BODY = {
    "success": True,
    "agent": {"algo": "GRPO", "llm": True},
    "results": [{"prompt": "hi", "completion": "yo"}],
    "batch_size": 1,
    "inference_time_ms": 1.0,
    "tokens_per_second": 1.0,
    "sessions": [],
}


def _agent_over_mock_transport(
    seen: list[tuple[str, str | None]],
    *,
    status_code: int = 200,
) -> Agent:
    """An LLM agent whose transport records the path and Authorization it saw."""

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((request.url.path, request.headers.get("authorization")))
        return httpx.Response(status_code, json=_ANY_ROUTE_BODY)

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
    @patch("agilerl.arena.inference.agent.httpx.Client")
    def test_absent_from_repr(self, mock_http_cls):
        agent = Agent("http://endpoint", api_key=SECRET, probe_on_init=False)
        assert repr(agent) == "<Agent endpoint='http://endpoint'>"

    @patch("agilerl.arena.inference.agent.httpx.Client")
    def test_not_stored_as_instance_attribute(self, mock_http_cls):
        agent = Agent("http://endpoint", api_key=SECRET, probe_on_init=False)
        assert SECRET not in json.dumps(agent.__dict__, default=str)

    @patch("agilerl.arena.inference.cache._write_store")
    @patch(
        "agilerl.arena.inference.cache._load_store",
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

    def test_403_on_rl_route_also_maps_to_auth_error(self):
        agent = _llm_agent()
        agent.metadata = _default_metadata()
        agent._http.request.return_value = _json_response(403, {"error": "forbidden"})

        with pytest.raises(ArenaAuthError, match="not allowed"):
            agent.get_action(np.array([1.0]))

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


class TestSerializeDeserializeComplex:
    def test_dict_with_none_values(self):
        data = {"obs": np.array([1.0, 2.0]), "mask": None}
        encoded = Agent.serialize(data, batched=False)
        assert isinstance(encoded, dict)
        assert encoded["mask"] is None
        decoded = Agent.deserialize(encoded, batched=False)
        np.testing.assert_array_equal(decoded["obs"], data["obs"])
        assert decoded["mask"] is None

    def test_nested_dict_with_arrays_and_none(self):
        data = {
            "sensor_a": np.array([1.0, 2.0, 3.0]),
            "sensor_b": None,
            "group": {
                "vel": np.array([4.0, 5.0]),
                "empty": None,
            },
        }
        encoded = Agent.serialize(data, batched=False)
        decoded = Agent.deserialize(encoded, batched=False)
        np.testing.assert_array_equal(decoded["sensor_a"], data["sensor_a"])
        assert decoded["sensor_b"] is None
        np.testing.assert_array_equal(decoded["group"]["vel"], data["group"]["vel"])
        assert decoded["group"]["empty"] is None

    def test_tuple_with_none_element(self):
        data = (np.array([1.0]), None, np.array([2.0]))
        encoded = Agent.serialize(data, batched=False)
        decoded = Agent.deserialize(encoded, batched=False)
        assert isinstance(decoded, tuple)
        np.testing.assert_array_equal(decoded[0], data[0])
        assert decoded[1] is None
        np.testing.assert_array_equal(decoded[2], data[2])

    def test_deeply_nested_dict_tuple_mix(self):
        data = {
            "level1": (
                np.array([10.0, 20.0]),
                {"inner": np.array([30.0])},
            )
        }
        encoded = Agent.serialize(data, batched=False)
        decoded = Agent.deserialize(encoded, batched=False)
        np.testing.assert_array_equal(decoded["level1"][0], data["level1"][0])
        np.testing.assert_array_equal(
            decoded["level1"][1]["inner"], data["level1"][1]["inner"]
        )


class TestLoadBinding:
    @patch("agilerl.arena.inference.cache._load_store", return_value={})
    def test_no_deployment_key(self, mock_load_store):
        assert load_binding("my-dep") is None

    @patch(
        "agilerl.arena.inference.cache._load_store",
        return_value={"deployments": "not-a-dict"},
    )
    def test_deployment_key_not_dict(self, mock_load_store):
        assert load_binding("my-dep") is None

    @patch(
        "agilerl.arena.inference.cache._load_store",
        return_value={"deployments": {}},
    )
    def test_missing_entry(self, mock_load_store):
        assert load_binding("missing") is None

    @patch(
        "agilerl.arena.inference.cache._load_store",
        return_value={"deployments": {"my-dep": "not-a-dict"}},
    )
    def test_entry_not_dict(self, mock_load_store):
        assert load_binding("my-dep") is None

    @patch(
        "agilerl.arena.inference.cache._load_store",
        return_value={"deployments": {"my-dep": {"url": "https://x.com"}}},
    )
    def test_valid_entry(self, mock_load_store):
        assert load_binding("my-dep") == "https://x.com"

    @patch(
        "agilerl.arena.inference.cache._load_store",
        return_value={"deployments": {"my-dep": {"url": 123}}},
    )
    def test_url_not_string(self, mock_load_store):
        assert load_binding("my-dep") is None

    @patch(
        "agilerl.arena.inference.cache._load_store",
        return_value={"deployments": {"my-dep": {"url": ""}}},
    )
    def test_empty_url(self, mock_load_store):
        assert load_binding("my-dep") is None

    @patch(
        "agilerl.arena.inference.cache._load_store",
        return_value={"deployments": {"my-dep": {"url": "  http://x  "}}},
    )
    def test_strips_whitespace(self, mock_load_store):
        assert load_binding("my-dep") == "http://x"

    @pytest.mark.parametrize("stale_key", ["key123", None, ""])
    def test_entry_carrying_an_api_key_is_stale(self, stale_key):
        store = {"deployments": {"my-dep": {"url": "http://x", "api_key": stale_key}}}
        with patch("agilerl.arena.inference.cache._load_store", return_value=store):
            assert load_binding("my-dep") is None


class TestSaveBinding:
    @patch("agilerl.arena.inference.cache._write_store")
    @patch(
        "agilerl.arena.inference.cache._load_store",
        return_value={},
    )
    def test_creates_new_section(self, mock_load, mock_write):
        save_binding("dep1", "http://url")
        written = mock_write.call_args[0][0]
        assert written["deployments"]["dep1"] == {"url": "http://url"}

    @patch("agilerl.arena.inference.cache._write_store")
    @patch(
        "agilerl.arena.inference.cache._load_store",
        return_value={"deployments": {"existing": {"url": "http://old"}}},
    )
    def test_merges_with_existing(self, mock_load, mock_write):
        save_binding("new-dep", "http://new")
        written = mock_write.call_args[0][0]
        assert "existing" in written["deployments"]
        assert written["deployments"]["new-dep"] == {"url": "http://new"}

    @patch("agilerl.arena.inference.cache._write_store")
    @patch(
        "agilerl.arena.inference.cache._load_store",
        return_value={"deployments": "corrupt"},
    )
    def test_replaces_non_dict_section(self, mock_load, mock_write):
        save_binding("dep1", "http://url")
        written = mock_write.call_args[0][0]
        assert isinstance(written["deployments"], dict)
        assert "dep1" in written["deployments"]

    @patch("agilerl.arena.inference.cache._write_store")
    @patch(
        "agilerl.arena.inference.cache._load_store",
        return_value={},
    )
    def test_strips_whitespace(self, mock_load, mock_write):
        save_binding("  dep1  ", "  http://url  ")
        written = mock_write.call_args[0][0]
        assert "dep1" in written["deployments"]
        assert written["deployments"]["dep1"] == {"url": "http://url"}

    @patch("agilerl.arena.inference.cache._write_store")
    @patch(
        "agilerl.arena.inference.cache._load_store",
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

    def test_ensure_rejects_wrong_modality(self):
        agent = Agent.__new__(Agent)
        agent.metadata = _default_metadata(llm=True, algo="GRPO")
        with pytest.raises(ArenaInferenceError, match="not RL"):
            agent._ensure(rl=True)

    def test_agent_info_requires_metadata(self):
        agent = Agent.__new__(Agent)
        agent.metadata = None
        with pytest.raises(ArenaInferenceError, match="Metadata not loaded"):
            agent._agent_info()

    def test_ensure_rejects_supervised_on_llm(self):
        agent = Agent.__new__(Agent)
        agent.metadata = _default_metadata(llm=True, supervised=False, algo="GRPO")
        with pytest.raises(ArenaInferenceError, match="not supervised"):
            agent._ensure(supervised=True)

    def test_ensure_rejects_llm_on_rl(self):
        agent = Agent.__new__(Agent)
        agent.metadata = _default_metadata(llm=False, supervised=False, algo="DQN")
        with pytest.raises(ArenaInferenceError, match="not an LLM"):
            agent._ensure(llm=True)

    def test_multi_agent_mask_rejects_non_dict(self):
        with pytest.raises(ArenaInferenceError, match="action_mask must be dict"):
            Agent._multi_agent_mask({"action_mask": np.array([1, 0])})

    def test_build_payload_includes_env_defined_actions_from_info(self):
        agent = Agent.__new__(Agent)
        agent.metadata = _default_metadata(multi_agent=True)
        obs = {"agent_0": np.array([1.0, 2.0], dtype=np.float32)}
        eda = {"agent_0": np.array([0.5], dtype=np.float32)}
        payload = agent._build_payload(
            obs,
            batched=False,
            hidden_state=None,
            info={"env_defined_actions": eda},
            env_defined_actions=None,
        )
        assert "env_defined_actions" in payload

    def test_validate_prompts_rejects_empty_list(self):
        with pytest.raises(ArenaInferenceError, match="At least one prompt"):
            Agent._validate_prompts([])

    def test_multi_agent_get_action_rejects_non_dict_observation(self):
        agent = Agent.__new__(Agent)
        agent.metadata = _default_metadata(multi_agent=True)
        with pytest.raises(ArenaInferenceError, match="dict\\[agent_id"):
            agent._build_payload(
                np.array([1.0]),
                batched=False,
                hidden_state=None,
                info=None,
                env_defined_actions=None,
            )

    def test_parse_get_action_rejects_non_dict_multi_agent_action(self):
        with pytest.raises(ArenaInferenceError, match="multi-agent action dict"):
            Agent._parse_get_action_response(
                {"action": Agent.serialize(np.array([1.0]), batched=False)},
                batched=False,
                multi_agent=True,
                recurrent=False,
            )

    def test_parse_get_action_rejects_null_single_agent_action(self):
        with pytest.raises(ArenaInferenceError, match="no action"):
            Agent._parse_get_action_response(
                {"action": None},
                batched=False,
                multi_agent=False,
                recurrent=False,
            )


class TestInferenceCacheOnDisk:
    def test_load_store_corrupt_json_returns_empty(self, monkeypatch, tmp_path):
        inference_file = tmp_path / "inference.json"
        inference_file.write_text("{not json", encoding="utf-8")
        monkeypatch.setattr(
            "agilerl.arena.inference.cache.INFERENCE_FILE", inference_file
        )
        assert load_binding("dep") is None

    def test_load_active_agent_empty_deployment_name(self, monkeypatch, tmp_path):
        inference_file = tmp_path / "inference.json"
        inference_file.write_text(
            json.dumps({"active_agent": {"deployment": "  "}}),
            encoding="utf-8",
        )
        monkeypatch.setattr(
            "agilerl.arena.inference.cache.INFERENCE_FILE", inference_file
        )
        assert load_active_agent() is None

    def test_save_binding_round_trip(self, monkeypatch, tmp_path):
        inference_file = tmp_path / ".arena" / "inference.json"
        monkeypatch.setattr(
            "agilerl.arena.inference.cache.INFERENCE_FILE", inference_file
        )
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
        monkeypatch.setattr(
            "agilerl.arena.inference.cache.INFERENCE_FILE", inference_file
        )

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
    @patch("agilerl.arena.inference.cache._load_store", return_value={})
    def test_load_missing(self, mock_load_store):
        assert load_active_agent() is None

    @patch(
        "agilerl.arena.inference.cache._load_store",
        return_value={"active_agent": "not-a-dict"},
    )
    def test_load_invalid_section(self, mock_load_store):
        assert load_active_agent() is None

    @patch("agilerl.arena.inference.cache._write_store")
    @patch(
        "agilerl.arena.inference.cache._load_store",
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
        "agilerl.arena.inference.cache._load_store",
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
        "agilerl.arena.inference.cache._load_store",
        return_value={"active_sessions": "not-a-dict"},
    )
    def test_load_invalid_section(self, mock_load_store):
        assert load_active_session("my-dep") is None

    @patch(
        "agilerl.arena.inference.cache._load_store",
        return_value={"active_sessions": {"my-dep": "   "}},
    )
    def test_load_blank_session_id(self, mock_load_store):
        assert load_active_session("my-dep") is None


MINIMAL_STATUS = {"deployment_id": "560"}


def _agent_over(status_body, *, route_body=None):
    """A real Agent whose /status returns *status_body*."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/status":
            return httpx.Response(200, json=status_body)
        return httpx.Response(200, json=route_body if route_body is not None else {})

    real = httpx.Client
    with patch(
        "agilerl.arena.inference.agent.httpx.Client",
        lambda **kw: real(transport=httpx.MockTransport(handler), **kw),
    ):
        return Agent("http://pod", api_key="k")


class TestStatusWithoutAgentInfo:
    """A public /status may report only the deployment id."""

    def test_metadata_parses_with_no_agent_block(self):
        agent = _agent_over(MINIMAL_STATUS)
        assert agent.metadata is not None
        assert agent.metadata.deployment_id == "560"
        assert agent.metadata.agent is None

    def test_an_absent_agent_block_is_not_a_reported_shape(self):
        """Absent must not read as 'every flag is False'."""
        assert _agent_over(MINIMAL_STATUS).metadata.agent is None
        reported = _agent_over({"deployment_id": "1", "agent": {}}).metadata.agent
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
            lambda a: a.predict(np.zeros(4)),
            lambda a: a.get_action(np.zeros(8)),
        ],
    )
    def test_type_guards_defer_to_the_deployment(self, call):
        """Unknown type must not be refused locally; the route answers 400."""
        agent = _agent_over(MINIMAL_STATUS)
        agent._http = MagicMock()
        agent._http.request.return_value = _json_response(
            400, {"error": "wrong endpoint"}
        )
        agent._http.stream.return_value = _stream_response([], status_code=400)

        with pytest.raises(ArenaInferenceError) as exc_info:
            call(agent)
        assert exc_info.value.status_code == 400
        assert "Deployment is not" not in str(exc_info.value)

    def test_get_action_falls_back_to_single_agent_non_recurrent(self):
        """No reported shape means the common case: one agent, no hidden state."""
        agent = _agent_over(MINIMAL_STATUS)
        payload = agent._build_payload(np.zeros(4), False, None, None, None)

        assert "obs" in payload
        assert payload["batch_size"] == 1
        assert "hidden_state" not in payload

    def test_a_reported_shape_still_drives_the_payload(self):
        agent = _agent_over(
            {"deployment_id": "1", "agent": {"multi_agent": False, "recurrent": True}}
        )
        payload = agent._build_payload(
            np.zeros(4), False, {"h": np.zeros(2)}, None, None
        )
        assert "hidden_state" in payload

    def test_reported_llm_still_guards(self):
        agent = _agent_over({"deployment_id": "1", "agent": {"llm": False}})
        with pytest.raises(ArenaInferenceError, match="not an LLM"):
            agent.generate("hi")

    def test_metadata_not_loaded_still_raises(self):
        agent = _agent_over(MINIMAL_STATUS)
        agent.metadata = None
        with pytest.raises(ArenaInferenceError, match="Metadata not loaded"):
            agent.generate("hi")
