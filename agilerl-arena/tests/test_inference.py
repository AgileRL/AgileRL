# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Tests for arena.inference (agent, cache, serde)."""

from __future__ import annotations

import json
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
    StatusResponse,
)
from agilerl.arena.inference.cache import (
    ActiveAgentSelection,
    load_active_agent,
    load_binding,
    normalized_deployment_name,
    save_active_agent,
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
        agent = Agent.__new__(Agent)
        agent._base_url = "http://test"
        agent._endpoint = "http://test"
        agent.metadata = _default_metadata()
        agent.generate_params = LLMParams()
        agent._http = MagicMock(spec=httpx.Client)
        return agent

    def _mock_json_response(self, status_code: int, body: dict):
        mock_resp = MagicMock()
        mock_resp.status_code = status_code
        mock_resp.json.return_value = body
        mock_resp.text = ""
        return mock_resp

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
        agent = Agent.__new__(Agent)
        agent._base_url = "http://test"
        agent._endpoint = "http://test"
        agent.metadata = _default_metadata()
        agent.generate_params = LLMParams()
        agent._http = MagicMock(spec=httpx.Client)
        return agent

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
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = updated
        agent._http.request.return_value = mock_resp

        result = agent.status(refresh=True)
        assert result.agent.algo == "PPO"
        assert agent.metadata.agent.algo == "PPO"
        agent._http.request.assert_called_once()


class TestPredict:
    def _make_agent_with_mock_http(self):
        agent = Agent.__new__(Agent)
        agent._base_url = "http://test"
        agent._endpoint = "http://test"
        agent.metadata = _default_metadata(supervised=True, algo="SFT")
        agent.generate_params = LLMParams()
        agent._http = MagicMock(spec=httpx.Client)
        return agent

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
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = body
        agent._http.request.return_value = mock_resp

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
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = body
        agent._http.request.return_value = mock_resp

        with pytest.raises(ArenaInferenceError, match="no results"):
            agent.predict(np.array([1.0]))


class TestGenerate:
    def _make_agent_with_mock_http(self):
        agent = Agent.__new__(Agent)
        agent._base_url = "http://test"
        agent._endpoint = "http://test"
        agent.metadata = _default_metadata(llm=True, algo="GRPO")
        agent.generate_params = LLMParams()
        agent._http = MagicMock(spec=httpx.Client)
        return agent

    def test_generate_string_prompt(self):
        agent = self._make_agent_with_mock_http()
        body = {
            "results": [{"prompt": "hi", "completion": "hello"}],
            "batch_size": 1,
            "inference_time_ms": 100.0,
            "tokens_per_second": 50.0,
            "success": True,
        }
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = body
        agent._http.request.return_value = mock_resp

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
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = body
        agent._http.request.return_value = mock_resp

        agent.generate("hi")
        sent = agent._http.request.call_args[1]["json"]
        assert sent["params"]["max_new_tokens"] == 256
        assert sent["params"]["temperature"] == 0.2

    def test_generate_rejects_empty_prompt(self):
        agent = self._make_agent_with_mock_http()
        with pytest.raises(ArenaInferenceError, match="empty"):
            agent.generate("")


class TestGenerateStream:
    def test_generate_stream_yields_chunks(self):
        agent = Agent.__new__(Agent)
        agent._base_url = "http://test"
        agent._endpoint = "http://test"
        agent.metadata = _default_metadata(llm=True, algo="GRPO")
        agent.generate_params = LLMParams()
        agent._http = MagicMock(spec=httpx.Client)

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.iter_text.return_value = iter(["foo", "bar"])

        mock_ctx = MagicMock()
        mock_ctx.__enter__.return_value = mock_resp
        mock_ctx.__exit__.return_value = False
        agent._http.stream.return_value = mock_ctx

        chunks = list(agent.generate_stream("Say hi"))
        assert chunks == ["foo", "bar"]
        sent = agent._http.stream.call_args[1]["json"]
        assert sent["prompt"] == "Say hi"

    def test_generate_stream_raises_on_in_band_error(self):
        agent = Agent.__new__(Agent)
        agent._base_url = "http://test"
        agent.metadata = _default_metadata(llm=True, algo="GRPO")
        agent.generate_params = LLMParams()
        agent._http = MagicMock(spec=httpx.Client)

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.iter_text.return_value = iter(["\n[ERROR: model failed]"])
        mock_ctx = MagicMock()
        mock_ctx.__enter__.return_value = mock_resp
        mock_ctx.__exit__.return_value = False
        agent._http.stream.return_value = mock_ctx

        with pytest.raises(ArenaInferenceError, match="ERROR"):
            list(agent.generate_stream("hi"))


class TestAgentContextManager:
    def test_enter_exit(self):
        agent = Agent.__new__(Agent)
        agent._http = MagicMock()
        agent._endpoint = "http://test"

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
        return_value={
            "deployments": {"my-dep": {"url": "https://x.com", "api_key": "key123"}}
        },
    )
    def test_valid_entry(self, mock_load_store):
        result = load_binding("my-dep")
        assert result == ("https://x.com", "key123")

    @patch(
        "agilerl.arena.inference.cache._load_store",
        return_value={"deployments": {"my-dep": {"url": 123, "api_key": "key"}}},
    )
    def test_url_not_string(self, mock_load_store):
        assert load_binding("my-dep") is None

    @patch(
        "agilerl.arena.inference.cache._load_store",
        return_value={"deployments": {"my-dep": {"url": "http://x", "api_key": None}}},
    )
    def test_api_key_not_string(self, mock_load_store):
        assert load_binding("my-dep") is None

    @patch(
        "agilerl.arena.inference.cache._load_store",
        return_value={"deployments": {"my-dep": {"url": "", "api_key": "key"}}},
    )
    def test_empty_url(self, mock_load_store):
        assert load_binding("my-dep") is None

    @patch(
        "agilerl.arena.inference.cache._load_store",
        return_value={"deployments": {"my-dep": {"url": "http://x", "api_key": "  "}}},
    )
    def test_whitespace_only_api_key(self, mock_load_store):
        assert load_binding("my-dep") is None

    @patch(
        "agilerl.arena.inference.cache._load_store",
        return_value={
            "deployments": {"my-dep": {"url": "  http://x  ", "api_key": " key "}}
        },
    )
    def test_strips_whitespace(self, mock_load_store):
        result = load_binding("my-dep")
        assert result == ("http://x", "key")

    @patch(
        "agilerl.arena.inference.cache._load_store",
        return_value={"deployments": {"my-dep": {"url": "http://x"}}},
    )
    def test_missing_api_key_field(self, mock_load_store):
        assert load_binding("my-dep") is None


class TestSaveBinding:
    @patch("agilerl.arena.inference.cache._write_store")
    @patch(
        "agilerl.arena.inference.cache._load_store",
        return_value={},
    )
    def test_creates_new_section(self, mock_load, mock_write):
        save_binding("dep1", "http://url", "key1")
        written = mock_write.call_args[0][0]
        assert written["deployments"]["dep1"] == {
            "url": "http://url",
            "api_key": "key1",
        }

    @patch("agilerl.arena.inference.cache._write_store")
    @patch(
        "agilerl.arena.inference.cache._load_store",
        return_value={
            "deployments": {"existing": {"url": "http://old", "api_key": "old_key"}}
        },
    )
    def test_merges_with_existing(self, mock_load, mock_write):
        save_binding("new-dep", "http://new", "new_key")
        written = mock_write.call_args[0][0]
        assert "existing" in written["deployments"]
        assert written["deployments"]["new-dep"] == {
            "url": "http://new",
            "api_key": "new_key",
        }

    @patch("agilerl.arena.inference.cache._write_store")
    @patch(
        "agilerl.arena.inference.cache._load_store",
        return_value={"deployments": "corrupt"},
    )
    def test_replaces_non_dict_section(self, mock_load, mock_write):
        save_binding("dep1", "http://url", "key1")
        written = mock_write.call_args[0][0]
        assert isinstance(written["deployments"], dict)
        assert "dep1" in written["deployments"]

    @patch("agilerl.arena.inference.cache._write_store")
    @patch(
        "agilerl.arena.inference.cache._load_store",
        return_value={},
    )
    def test_strips_whitespace(self, mock_load, mock_write):
        save_binding("  dep1  ", "  http://url  ", "  key1  ")
        written = mock_write.call_args[0][0]
        assert "dep1" in written["deployments"]
        entry = written["deployments"]["dep1"]
        assert entry["url"] == "http://url"
        assert entry["api_key"] == "key1"


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
        save_binding("dep-a", "http://x", "key")
        assert load_binding("dep-a") == ("http://x", "key")


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
