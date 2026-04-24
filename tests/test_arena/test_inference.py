"""Tests for agilerl.arena.inference — Agent (serialize/deserialize, get_action)."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import httpx
import numpy as np
import pytest

from agilerl.arena.exceptions import ArenaAPIError, ArenaAuthError
from agilerl.arena.inference import Agent


# ---------------------------------------------------------------------------
# serialize / deserialize round-trips
# ---------------------------------------------------------------------------
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
        for orig, dec in zip(data, decoded):
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


# ---------------------------------------------------------------------------
# get_batch_size
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# _build_payload
# ---------------------------------------------------------------------------
class TestBuildPayload:
    def _make_agent(self):
        with patch("agilerl.arena.inference.httpx.Client"):
            return Agent("http://test/get_action")

    def test_basic_payload(self):
        agent = self._make_agent()
        obs = np.array([1.0, 2.0])
        payload = agent._build_payload(obs, batched=False, hidden_state=None, info=None)
        assert "obs" in payload
        assert "hidden_state" in payload
        assert "action_mask" in payload
        assert payload["batch_size"] == 1

    def test_batched_payload_batch_size(self):
        agent = self._make_agent()
        obs = np.zeros((4, 8))
        payload = agent._build_payload(obs, batched=True, hidden_state=None, info=None)
        assert payload["batch_size"] == 4

    def test_action_mask_from_info(self):
        agent = self._make_agent()
        obs = np.array([1.0])
        mask = np.array([1, 0, 1])
        info = {"action_mask": mask}
        payload = agent._build_payload(obs, batched=False, hidden_state=None, info=info)
        assert payload["action_mask"] is not None

    def test_hidden_state_serialized(self):
        agent = self._make_agent()
        obs = np.array([1.0])
        hs = {"h": np.array([0.5, 0.5])}
        payload = agent._build_payload(obs, batched=False, hidden_state=hs, info=None)
        assert payload["hidden_state"] is not None
        assert isinstance(payload["hidden_state"], dict)


# ---------------------------------------------------------------------------
# _parse_response
# ---------------------------------------------------------------------------
class TestParseResponse:
    def test_extracts_action_and_hidden_state(self):
        action = np.array([2])
        hs = {"h": np.array([0.1, 0.2])}
        resp = {
            "action": Agent.serialize(action, batched=False),
            "hidden_state": Agent.serialize(hs, batched=False),
        }
        parsed_action, parsed_hs = Agent._parse_response(resp, batched=False)
        np.testing.assert_array_equal(parsed_action, action)
        np.testing.assert_array_almost_equal(parsed_hs["h"], hs["h"])

    def test_hidden_state_none(self):
        action = np.array([0])
        resp = {
            "action": Agent.serialize(action, batched=False),
        }
        parsed_action, parsed_hs = Agent._parse_response(resp, batched=False)
        np.testing.assert_array_equal(parsed_action, action)
        assert parsed_hs is None


# ---------------------------------------------------------------------------
# get_action
# ---------------------------------------------------------------------------
class TestGetAction:
    def _make_agent_with_mock_http(self):
        agent = Agent.__new__(Agent)
        agent._endpoint = "http://test"
        agent._get_action_endpoint = "http://test/get_action"
        agent._http = MagicMock(spec=httpx.Client)
        return agent

    def test_successful_get_action(self):
        agent = self._make_agent_with_mock_http()
        action_arr = np.array([1])
        resp_json = {
            "action": Agent.serialize(action_arr, batched=False),
            "hidden_state": None,
        }

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = resp_json
        agent._http.post.return_value = mock_resp

        obs = np.array([1.0, 2.0, 3.0])
        status, action, hs = agent.get_action(obs)

        assert status == 200
        np.testing.assert_array_equal(action, action_arr)
        assert hs is None

    def test_401_raises_auth_error(self):
        agent = self._make_agent_with_mock_http()
        mock_resp = MagicMock()
        mock_resp.status_code = 401
        agent._http.post.return_value = mock_resp

        with pytest.raises(ArenaAuthError, match="401 Unauthorized"):
            agent.get_action(np.array([1.0]))

    def test_non_200_raises_api_error(self):
        agent = self._make_agent_with_mock_http()
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_resp.json.return_value = {"error": "internal"}
        agent._http.post.return_value = mock_resp

        with pytest.raises(ArenaAPIError) as exc_info:
            agent.get_action(np.array([1.0]))
        assert exc_info.value.status_code == 500

    def test_network_error_raises_api_error(self):
        agent = self._make_agent_with_mock_http()
        agent._http.post.side_effect = httpx.ConnectError("refused")

        with pytest.raises(ArenaAPIError) as exc_info:
            agent.get_action(np.array([1.0]))
        assert exc_info.value.status_code == 0

    def test_non_200_with_text_fallback(self):
        agent = self._make_agent_with_mock_http()
        mock_resp = MagicMock()
        mock_resp.status_code = 502
        mock_resp.json.side_effect = ValueError("not json")
        mock_resp.text = "Bad Gateway"
        agent._http.post.return_value = mock_resp

        with pytest.raises(ArenaAPIError) as exc_info:
            agent.get_action(np.array([1.0]))
        assert "Bad Gateway" in exc_info.value.detail


# ---------------------------------------------------------------------------
# __init__, context manager, repr
# ---------------------------------------------------------------------------
class TestAgentInit:
    @patch("agilerl.arena.inference.httpx.Client")
    def test_no_token(self, mock_http_cls):
        agent = Agent("http://endpoint/get_action")
        call_kwargs = mock_http_cls.call_args[1]
        assert call_kwargs["headers"] == {}

    @patch("agilerl.arena.inference.httpx.Client")
    def test_with_api_key(self, mock_http_cls):
        agent = Agent("http://endpoint/get_action", api_key="abc")
        call_kwargs = mock_http_cls.call_args[1]
        assert call_kwargs["headers"]["authorization"] == "Bearer abc"


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
        agent._endpoint = "http://test/get_action"
        assert repr(agent) == "<Agent endpoint='http://test/get_action'>"


# ---------------------------------------------------------------------------
# Serialize / deserialize — complex nested structures
# ---------------------------------------------------------------------------


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
