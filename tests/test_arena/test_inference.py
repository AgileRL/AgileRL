"""Tests for agilerl.arena.inference and agilerl.arena.inference_cache."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import httpx
import numpy as np
import pytest

from agilerl.arena.exceptions import ArenaAPIError, ArenaAuthError, ArenaValidationError
from agilerl.arena.inference import Agent
from agilerl.arena.inference_cache import (
    load_inference_binding,
    save_inference_binding,
    normalized_deployment_name,
)


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


class TestIsJsonNumberScalar:
    def test_int_returns_true(self):
        assert Agent._is_json_number_scalar(42) is True

    def test_float_returns_true(self):
        assert Agent._is_json_number_scalar(3.14) is True

    def test_bool_returns_false(self):
        assert Agent._is_json_number_scalar(True) is False
        assert Agent._is_json_number_scalar(False) is False

    def test_string_returns_false(self):
        assert Agent._is_json_number_scalar("123") is False

    def test_none_returns_false(self):
        assert Agent._is_json_number_scalar(None) is False

    def test_list_returns_false(self):
        assert Agent._is_json_number_scalar([1]) is False


class TestIsNumericJsonList:
    def test_all_numbers(self):
        assert Agent._is_numeric_json_list([1, 2.0, 3]) is True

    def test_empty_list(self):
        assert Agent._is_numeric_json_list([]) is True

    def test_contains_string(self):
        assert Agent._is_numeric_json_list([1, "two", 3]) is False

    def test_contains_bool(self):
        assert Agent._is_numeric_json_list([1, True]) is False


class TestParseFloatToken:
    def test_valid_float(self):
        assert Agent._parse_float_token("3.14") == 3.14

    def test_valid_int_string(self):
        assert Agent._parse_float_token("42") == 42.0

    def test_negative(self):
        assert Agent._parse_float_token("-1.5") == -1.5

    def test_invalid_returns_none(self):
        assert Agent._parse_float_token("abc") is None

    def test_empty_string_returns_none(self):
        assert Agent._parse_float_token("") is None


class TestParseBracketFloatVector:
    def test_valid_vector(self):
        result = Agent._parse_bracket_float_vector("[ 1.0 2.0 3.0 ]")
        np.testing.assert_array_equal(result, [1.0, 2.0, 3.0])

    def test_with_commas(self):
        result = Agent._parse_bracket_float_vector("[1.0, 2.0, 3.0]")
        np.testing.assert_array_equal(result, [1.0, 2.0, 3.0])

    def test_single_element(self):
        result = Agent._parse_bracket_float_vector("[5.0]")
        np.testing.assert_array_equal(result, [5.0])

    def test_empty_brackets(self):
        result = Agent._parse_bracket_float_vector("[ ]")
        assert result is not None
        assert len(result) == 0
        assert result.dtype == np.float64

    def test_non_bracket_returns_none(self):
        assert Agent._parse_bracket_float_vector("hello") is None

    def test_too_short_returns_none(self):
        assert Agent._parse_bracket_float_vector("x") is None

    def test_invalid_content_returns_none(self):
        assert Agent._parse_bracket_float_vector("[abc def]") is None

    def test_mixed_valid_invalid_returns_none(self):
        assert Agent._parse_bracket_float_vector("[1.0 abc 3.0]") is None

    def test_whitespace_padded(self):
        result = Agent._parse_bracket_float_vector("  [ 1.0 2.0 ]  ")
        np.testing.assert_array_equal(result, [1.0, 2.0])


class TestObservationFromString:
    def test_empty_string_raises(self):
        with pytest.raises(ArenaValidationError, match="empty"):
            Agent.observation_from_string("")

    def test_whitespace_only_raises(self):
        with pytest.raises(ArenaValidationError, match="empty"):
            Agent.observation_from_string("   ")

    def test_json_array_of_numbers(self):
        result = Agent.observation_from_string("[1.0, 2.0, 3.0]")
        np.testing.assert_array_equal(result, [1.0, 2.0, 3.0])
        assert result.dtype == np.float64

    def test_json_scalar(self):
        result = Agent.observation_from_string("42")
        np.testing.assert_array_equal(result, 42.0)

    def test_json_float_scalar(self):
        result = Agent.observation_from_string("3.14")
        np.testing.assert_array_equal(result, 3.14)

    def test_bracket_float_vector(self):
        result = Agent.observation_from_string("[ 1.0 2.0 3.0 ]")
        np.testing.assert_array_equal(result, [1.0, 2.0, 3.0])

    def test_base64_blob(self):
        arr = np.array([10.0, 20.0, 30.0])
        encoded = Agent.serialize(arr, batched=False)
        result = Agent.observation_from_string(encoded)
        np.testing.assert_array_almost_equal(result, arr)

    def test_nested_json_dict(self):
        data = {"obs": np.array([1.0, 2.0]), "vel": np.array([3.0])}
        serialized = Agent.serialize(data, batched=False)
        text = json.dumps(serialized)
        result = Agent.observation_from_string(text)
        assert isinstance(result, dict)
        np.testing.assert_array_almost_equal(result["obs"], data["obs"])

    def test_null_json_raises(self):
        with pytest.raises(ArenaValidationError, match="null"):
            Agent.observation_from_string("null")

    def test_json_boolean_decoded_as_serialized(self):
        # `true` is valid JSON but not a scalar number, so goes to deserialize path
        # which should raise because bool can't be base64-decoded
        with pytest.raises(Exception):
            Agent.observation_from_string("true")


class TestLoadInferenceBinding:
    @patch("agilerl.arena.inference_cache.load_credentials_payload", return_value={})
    def test_no_deployment_key(self, _mock):
        assert load_inference_binding("my-dep") is None

    @patch(
        "agilerl.arena.inference_cache.load_credentials_payload",
        return_value={"deployment_inference": "not-a-dict"},
    )
    def test_deployment_key_not_dict(self, _mock):
        assert load_inference_binding("my-dep") is None

    @patch(
        "agilerl.arena.inference_cache.load_credentials_payload",
        return_value={"deployment_inference": {}},
    )
    def test_missing_entry(self, _mock):
        assert load_inference_binding("missing") is None

    @patch(
        "agilerl.arena.inference_cache.load_credentials_payload",
        return_value={"deployment_inference": {"my-dep": "not-a-dict"}},
    )
    def test_entry_not_dict(self, _mock):
        assert load_inference_binding("my-dep") is None

    @patch(
        "agilerl.arena.inference_cache.load_credentials_payload",
        return_value={
            "deployment_inference": {
                "my-dep": {"url": "https://x.com", "api_key": "key123"}
            }
        },
    )
    def test_valid_entry(self, _mock):
        result = load_inference_binding("my-dep")
        assert result == ("https://x.com", "key123")

    @patch(
        "agilerl.arena.inference_cache.load_credentials_payload",
        return_value={
            "deployment_inference": {"my-dep": {"url": 123, "api_key": "key"}}
        },
    )
    def test_url_not_string(self, _mock):
        assert load_inference_binding("my-dep") is None

    @patch(
        "agilerl.arena.inference_cache.load_credentials_payload",
        return_value={
            "deployment_inference": {"my-dep": {"url": "http://x", "api_key": None}}
        },
    )
    def test_api_key_not_string(self, _mock):
        assert load_inference_binding("my-dep") is None

    @patch(
        "agilerl.arena.inference_cache.load_credentials_payload",
        return_value={
            "deployment_inference": {"my-dep": {"url": "", "api_key": "key"}}
        },
    )
    def test_empty_url(self, _mock):
        assert load_inference_binding("my-dep") is None

    @patch(
        "agilerl.arena.inference_cache.load_credentials_payload",
        return_value={
            "deployment_inference": {"my-dep": {"url": "http://x", "api_key": "  "}}
        },
    )
    def test_whitespace_only_api_key(self, _mock):
        assert load_inference_binding("my-dep") is None

    @patch(
        "agilerl.arena.inference_cache.load_credentials_payload",
        return_value={
            "deployment_inference": {
                "my-dep": {"url": "  http://x  ", "api_key": " key "}
            }
        },
    )
    def test_strips_whitespace(self, _mock):
        result = load_inference_binding("my-dep")
        assert result == ("http://x", "key")

    @patch(
        "agilerl.arena.inference_cache.load_credentials_payload",
        return_value={"deployment_inference": {"my-dep": {"url": "http://x"}}},
    )
    def test_missing_api_key_field(self, _mock):
        assert load_inference_binding("my-dep") is None


class TestSaveInferenceBinding:
    @patch("agilerl.arena.inference_cache.ArenaOAuth2._write_credentials")
    @patch(
        "agilerl.arena.inference_cache.load_credentials_payload",
        return_value={},
    )
    def test_creates_new_section(self, mock_load, mock_write):
        save_inference_binding("dep1", "http://url", "key1")
        written = mock_write.call_args[0][0]
        assert written["deployment_inference"]["dep1"] == {
            "url": "http://url",
            "api_key": "key1",
        }

    @patch("agilerl.arena.inference_cache.ArenaOAuth2._write_credentials")
    @patch(
        "agilerl.arena.inference_cache.load_credentials_payload",
        return_value={
            "deployment_inference": {
                "existing": {"url": "http://old", "api_key": "old_key"}
            }
        },
    )
    def test_merges_with_existing(self, mock_load, mock_write):
        save_inference_binding("new-dep", "http://new", "new_key")
        written = mock_write.call_args[0][0]
        assert "existing" in written["deployment_inference"]
        assert written["deployment_inference"]["new-dep"] == {
            "url": "http://new",
            "api_key": "new_key",
        }

    @patch("agilerl.arena.inference_cache.ArenaOAuth2._write_credentials")
    @patch(
        "agilerl.arena.inference_cache.load_credentials_payload",
        return_value={"deployment_inference": "corrupt"},
    )
    def test_replaces_non_dict_section(self, mock_load, mock_write):
        save_inference_binding("dep1", "http://url", "key1")
        written = mock_write.call_args[0][0]
        assert isinstance(written["deployment_inference"], dict)
        assert "dep1" in written["deployment_inference"]

    @patch("agilerl.arena.inference_cache.ArenaOAuth2._write_credentials")
    @patch(
        "agilerl.arena.inference_cache.load_credentials_payload",
        return_value={},
    )
    def test_strips_whitespace(self, mock_load, mock_write):
        save_inference_binding("  dep1  ", "  http://url  ", "  key1  ")
        written = mock_write.call_args[0][0]
        assert "dep1" in written["deployment_inference"]
        entry = written["deployment_inference"]["dep1"]
        assert entry["url"] == "http://url"
        assert entry["api_key"] == "key1"


class TestNormalizedDeploymentName:
    def test_strips_whitespace(self):
        assert normalized_deployment_name("  my-dep  ") == "my-dep"

    def test_no_change_needed(self):
        assert normalized_deployment_name("my-dep") == "my-dep"
