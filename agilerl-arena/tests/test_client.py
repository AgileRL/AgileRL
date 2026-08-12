# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Tests for arena.client — _TokenStore and ArenaClient."""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import pytest

from agilerl.arena.auth import ArenaOAuth2
from agilerl.arena.client import ArenaClient, _TokenStore
from agilerl.arena.exceptions import (
    ArenaAPIError,
    ArenaAuthError,
    ArenaConfigError,
    ArenaFileNotFoundError,
    ArenaInferenceError,
    ArenaTrainingError,
    ArenaValidationError,
)
from agilerl.arena.output import StreamRichRenderer
from agilerl.arena.stream import NDJsonStream


def _jwt_with_exp(exp: int) -> str:
    import base64

    def seg(obj: dict) -> str:
        raw = json.dumps(obj, separators=(",", ":")).encode()
        return base64.urlsafe_b64encode(raw).decode().rstrip("=")

    return f"{seg({'alg': 'none'})}.{seg({'exp': exp})}.x"


def _mock_ndjson_stream(result: dict | None = None) -> MagicMock:
    """Create a mock NDJsonStream with a preset collect() result."""
    mock = MagicMock(spec=NDJsonStream)
    mock.collect.return_value = result or {}
    mock.result = result
    mock.__enter__ = MagicMock(return_value=mock)
    mock.__exit__ = MagicMock(return_value=False)
    return mock


class TestTokenStore:
    def test_defaults_are_none(self):
        store = _TokenStore()
        assert store.access_token is None
        assert store.refresh_token is None

    def test_repr_no_tokens(self):
        store = _TokenStore()
        assert repr(store) == "_TokenStore(access=False, refresh=False)"

    def test_repr_with_tokens(self):
        store = _TokenStore(access_token="a", refresh_token="r")
        assert repr(store) == "_TokenStore(access=True, refresh=True)"

    def test_repr_partial(self):
        store = _TokenStore(access_token="a")
        assert repr(store) == "_TokenStore(access=True, refresh=False)"

    def test_clear(self):
        store = _TokenStore(access_token="a", refresh_token="r")
        store.clear()
        assert store.access_token is None
        assert store.refresh_token is None


@pytest.fixture
def api_key_client():
    """ArenaClient with a static API key (no OAuth, no session restore)."""
    with patch("agilerl.arena.auth.KeycloakOpenID"):
        return ArenaClient(api_key="test-key")


@pytest.fixture
def token_client():
    """ArenaClient with OAuth tokens pre-loaded (no API key)."""
    env = {k: v for k, v in os.environ.items() if k != "ARENA_API_KEY"}
    with patch.dict(os.environ, env, clear=True):
        with patch("agilerl.arena.auth.KeycloakOpenID"):
            with patch.object(ArenaClient, "_try_restore_session"):
                client = ArenaClient()
    client._tokens.access_token = "tok_access"
    client._tokens.refresh_token = "tok_refresh"
    return client


@pytest.fixture
def unauthenticated_client():
    """ArenaClient with no credentials at all."""
    env = {k: v for k, v in os.environ.items() if k != "ARENA_API_KEY"}
    with patch.dict(os.environ, env, clear=True):
        with patch("agilerl.arena.auth.KeycloakOpenID"):
            with patch.object(ArenaClient, "_try_restore_session"):
                client = ArenaClient()
    return client


class TestArenaClientInit:
    @patch("agilerl.arena.auth.KeycloakOpenID")
    def test_with_explicit_api_key(self, mock_keycloak):
        client = ArenaClient(api_key="my-key")
        assert client._api_key == "my-key"
        assert client.is_authenticated

    @patch("agilerl.arena.auth.KeycloakOpenID")
    def test_with_env_var(self, mock_keycloak):
        with patch.dict(os.environ, {"ARENA_API_KEY": "env-key"}):
            client = ArenaClient()
        assert client._api_key == "env-key"

    @patch("agilerl.arena.auth.KeycloakOpenID")
    def test_without_key_calls_restore_session(self, mock_keycloak):
        with patch.dict(os.environ, {}, clear=False):
            env = os.environ.copy()
            env.pop("ARENA_API_KEY", None)
            with patch.dict(os.environ, env, clear=True):
                with patch.object(ArenaClient, "_try_restore_session") as mock_restore:
                    ArenaClient()
                mock_restore.assert_called_once()

    @patch("agilerl.arena.auth.KeycloakOpenID")
    def test_http_client_config(self, mock_keycloak):
        client = ArenaClient(api_key="k", request_timeout=60)
        assert client._request_timeout == 60
        assert client._upload_timeout == 300  # default

    @patch("agilerl.arena.auth.KeycloakOpenID")
    def test_base_url_env_override(self, mock_keycloak):
        with patch.dict(os.environ, {"ARENA_BASE_URL": "https://arena.example.com/"}):
            client = ArenaClient(api_key="k")
        assert client._base_url == "https://arena.example.com"


class TestArenaClientConfigure:
    def setup_method(self):
        self._orig_url = ArenaClient.BASE_URL

    def teardown_method(self):
        ArenaClient.BASE_URL = self._orig_url

    @patch("agilerl.arena.auth.ArenaOAuth2.configure")
    def test_overrides_base_url(self, mock_oauth_cfg):
        result = ArenaClient.configure(base_url="http://localhost:3001")
        assert ArenaClient.BASE_URL == "http://localhost:3001"
        assert result is ArenaClient

    @patch("agilerl.arena.auth.ArenaOAuth2.configure")
    def test_delegates_to_oauth_configure(self, mock_oauth_cfg):
        ArenaClient.configure(
            keycloak_url="http://kc:8080",
            realm="test",
            client_id="cli",
        )
        mock_oauth_cfg.assert_called_once_with(
            keycloak_url="http://kc:8080",
            realm="test",
            client_id="cli",
        )

    @patch("agilerl.arena.auth.ArenaOAuth2.configure")
    def test_none_base_url_keeps_original(self, mock_oauth_cfg):
        orig = ArenaClient.BASE_URL
        ArenaClient.configure(base_url=None)
        assert orig == ArenaClient.BASE_URL


class TestArenaClientLogin:
    def test_login_stores_tokens(self, unauthenticated_client):
        client = unauthenticated_client
        tokens = {"access_token": "at", "refresh_token": "rt"}
        client._auth.device_login = MagicMock(return_value=tokens)

        client.login(timeout=60)
        assert client._tokens.access_token == "at"
        assert client._tokens.refresh_token == "rt"
        client._auth.device_login.assert_called_once_with(timeout=60)

    def test_login_skips_device_when_jwt_still_valid(self, unauthenticated_client):
        client = unauthenticated_client
        client._tokens.access_token = _jwt_with_exp(int(time.time()) + 3600)
        client._tokens.refresh_token = "rt"
        client._auth.device_login = MagicMock()
        client.login()
        client._auth.device_login.assert_not_called()

    def test_login_refreshes_when_jwt_expired(self, unauthenticated_client):
        client = unauthenticated_client
        client._tokens.access_token = _jwt_with_exp(int(time.time()) - 120)
        client._tokens.refresh_token = "rt"
        client._auth.device_login = MagicMock()
        client._auth.refresh_access_token = MagicMock(
            return_value={"access_token": "new_at", "refresh_token": "new_rt"}
        )
        client.login()
        client._auth.device_login.assert_not_called()
        client._auth.refresh_access_token.assert_called_once_with("rt")
        assert client._tokens.access_token == "new_at"
        assert client._tokens.refresh_token == "new_rt"

    def test_login_falls_back_to_device_when_refresh_fails(
        self, unauthenticated_client
    ):
        client = unauthenticated_client
        client._tokens.access_token = _jwt_with_exp(int(time.time()) - 120)
        client._tokens.refresh_token = "rt"
        client._auth.refresh_access_token = MagicMock(side_effect=ArenaAuthError("bad"))
        client._auth.device_login = MagicMock(
            return_value={"access_token": "dev_at", "refresh_token": "dev_rt"}
        )
        client.login()
        client._auth.device_login.assert_called_once()

    def test_login_force_runs_device_even_when_valid(self, unauthenticated_client):
        client = unauthenticated_client
        client._tokens.access_token = _jwt_with_exp(int(time.time()) + 3600)
        tokens = {"access_token": "from_dev", "refresh_token": "rt2"}
        client._auth.device_login = MagicMock(return_value=tokens)
        client.login(force=True)
        client._auth.device_login.assert_called_once()

    def test_login_noop_with_api_key(self, api_key_client):
        client = api_key_client
        client._auth.device_login = MagicMock()
        client.login()
        client._auth.device_login.assert_not_called()

    @patch("agilerl.arena.client.load_credentials")
    @patch("agilerl.arena.auth.KeycloakOpenID")
    def test_restore_session_proactively_refreshes_expired_jwt(
        self, mock_keycloak, mock_load_credentials
    ):
        past = int(time.time()) - 120
        mock_load_credentials.return_value = {
            "access_token": _jwt_with_exp(past),
            "refresh_token": "stored_rt",
        }
        with patch.object(ArenaOAuth2, "refresh_access_token") as mock_refresh:
            mock_refresh.return_value = {
                "access_token": "refreshed_at",
                "refresh_token": "refreshed_rt",
            }
            env = {k: v for k, v in os.environ.items() if k != "ARENA_API_KEY"}
            with patch.dict(os.environ, env, clear=True):
                client = ArenaClient()
        mock_refresh.assert_called_once_with("stored_rt")
        assert client._tokens.access_token == "refreshed_at"
        assert client._tokens.refresh_token == "refreshed_rt"

    def test_proactively_refresh_updates_tokens(self, token_client):
        client = token_client
        client._tokens.access_token = _jwt_with_exp(int(time.time()) - 120)
        client._auth.refresh_access_token = MagicMock(
            return_value={"access_token": "new_at", "refresh_token": "new_rt"}
        )
        client._proactively_refresh_oauth()
        assert client._tokens.access_token == "new_at"
        assert client._tokens.refresh_token == "new_rt"

    def test_proactively_refresh_skips_valid_token(self, token_client):
        client = token_client
        client._tokens.access_token = _jwt_with_exp(int(time.time()) + 3600)
        client._auth.refresh_access_token = MagicMock()
        client._proactively_refresh_oauth()
        client._auth.refresh_access_token.assert_not_called()

    def test_proactively_refresh_ignores_auth_error(self, token_client):
        client = token_client
        expired = _jwt_with_exp(int(time.time()) - 120)
        client._tokens.access_token = expired
        client._auth.refresh_access_token = MagicMock(side_effect=ArenaAuthError("bad"))
        client._proactively_refresh_oauth()
        assert client._tokens.access_token == expired

    def test_proactively_refresh_skips_when_api_key_set(self):
        with patch("agilerl.arena.auth.KeycloakOpenID"):
            client = ArenaClient(api_key="test-key")
        client._auth.refresh_access_token = MagicMock()
        client._proactively_refresh_oauth()
        client._auth.refresh_access_token.assert_not_called()


class TestArenaClientLogout:
    def test_logout_revokes_and_clears(self, token_client):
        client = token_client
        client._auth.revoke = MagicMock()

        client.logout()
        client._auth.revoke.assert_called_once_with("tok_refresh")
        assert client._tokens.access_token is None
        assert client._tokens.refresh_token is None

    def test_logout_without_refresh_token(self, unauthenticated_client):
        client = unauthenticated_client
        client._tokens.access_token = "at"
        client._auth.revoke = MagicMock()

        client.logout()
        client._auth.revoke.assert_not_called()
        assert client._tokens.access_token is None


class TestIsAuthenticated:
    def test_true_with_api_key(self, api_key_client):
        assert api_key_client.is_authenticated is True

    def test_true_with_access_token(self, token_client):
        assert token_client.is_authenticated is True

    def test_false_with_nothing(self, unauthenticated_client):
        assert unauthenticated_client.is_authenticated is False


class TestUserMethods:
    def test_get_current_user(self, api_key_client):
        api_key_client._request = MagicMock(return_value={"email": "a@b.com"})
        assert api_key_client.get_current_user() == {"email": "a@b.com"}

    def test_get_user_credits(self, api_key_client):
        api_key_client._request = MagicMock(return_value=100)
        assert api_key_client.get_user_credits() == 100


class TestSetStreamHandler:
    def test_default_handler_is_none(self, api_key_client):
        assert api_key_client._stream_handler is None

    def test_set_handler(self, api_key_client):
        handler = MagicMock()
        api_key_client.set_stream_handler(handler)
        assert api_key_client._stream_handler is handler

    def test_clear_handler(self, api_key_client):
        api_key_client.set_stream_handler(lambda event: None)
        api_key_client.set_stream_handler(None)
        assert api_key_client._stream_handler is None


class TestAuthHeaders:
    def test_api_key_header(self, api_key_client):
        headers = api_key_client._auth_headers()
        assert headers == {"Authorization": "Bearer test-key"}

    def test_token_header(self, token_client):
        headers = token_client._auth_headers()
        assert headers == {"Authorization": "Bearer tok_access"}

    def test_no_auth_raises(self, unauthenticated_client):
        with pytest.raises(ArenaAuthError, match="not been authenticated"):
            unauthenticated_client._auth_headers()


class TestRequest:
    def test_successful_json_response(self, api_key_client):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.is_success = True
        mock_resp.headers = {"content-type": "application/json"}
        mock_resp.json.return_value = {"result": "ok"}

        api_key_client._http.request = MagicMock(return_value=mock_resp)
        result = api_key_client._request("GET", "/api/test")
        assert result == {"result": "ok"}

    def test_unwraps_cli_envelope(self, api_key_client):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.is_success = True
        mock_resp.headers = {"content-type": "application/json"}
        mock_resp.json.return_value = {
            "ok": True,
            "data": {"MyEnv": {"v1": {"validated": True}}},
        }

        api_key_client._http.request = MagicMock(return_value=mock_resp)
        result = api_key_client._request("GET", "/api/cli/v1/environments")
        assert result == {"MyEnv": {"v1": {"validated": True}}}

    def test_does_not_unwrap_without_ok_true(self, api_key_client):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.is_success = True
        mock_resp.headers = {"content-type": "application/json"}
        mock_resp.json.return_value = {"ok": False, "data": "should stay"}

        api_key_client._http.request = MagicMock(return_value=mock_resp)
        result = api_key_client._request("GET", "/api/test")
        assert result == {"ok": False, "data": "should stay"}

    def test_successful_text_response(self, api_key_client):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.is_success = True
        mock_resp.headers = {"content-type": "text/plain"}
        mock_resp.text = "plain text"

        api_key_client._http.request = MagicMock(return_value=mock_resp)
        result = api_key_client._request("GET", "/api/test")
        assert result == "plain text"

    def test_network_error_raises_api_error(self, api_key_client):
        api_key_client._http.request = MagicMock(
            side_effect=httpx.ConnectError("refused")
        )
        with pytest.raises(ArenaAPIError) as exc_info:
            api_key_client._request("GET", "/api/test")
        assert exc_info.value.status_code == 0
        assert "Network error" in exc_info.value.detail

    def test_401_with_refresh_retries(self, token_client):
        first_resp = MagicMock()
        first_resp.status_code = 401
        first_resp.is_success = False

        second_resp = MagicMock()
        second_resp.status_code = 200
        second_resp.is_success = True
        second_resp.headers = {"content-type": "application/json"}
        second_resp.json.return_value = {"ok": True}

        token_client._http.request = MagicMock(side_effect=[first_resp, second_resp])
        token_client._auth.refresh_access_token = MagicMock(
            return_value={"access_token": "new_at", "refresh_token": "new_rt"}
        )

        result = token_client._request("GET", "/api/test")
        assert result == {"ok": True}
        assert token_client._tokens.access_token == "new_at"
        assert token_client._http.request.call_count == 2

    def test_default_timeout_defers_to_client_default(self, api_key_client):
        """timeout=None must not disable timeouts (httpx semantics)."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.is_success = True
        mock_resp.headers = {"content-type": "application/json"}
        mock_resp.json.return_value = {}

        api_key_client._http.request = MagicMock(return_value=mock_resp)
        api_key_client._request("GET", "/api/test")
        timeout = api_key_client._http.request.call_args.kwargs["timeout"]
        assert timeout is httpx.USE_CLIENT_DEFAULT

    def test_explicit_timeout_is_forwarded(self, api_key_client):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.is_success = True

        api_key_client._http.request = MagicMock(return_value=mock_resp)
        api_key_client._send("GET", "/api/test", timeout=77)
        assert api_key_client._http.request.call_args.kwargs["timeout"] == 77

    def test_stream_disables_read_timeout(self, api_key_client):
        """Streaming requests must not bound the idle gap between NDJSON chunks.

        Slow server-side work (installing requirements, building images) goes
        silent without emitting an event; a fixed read timeout aborts it with
        httpx.ReadTimeout. Connect/write/pool stay bounded to the given value.
        """
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.is_success = True

        built_request = MagicMock()
        api_key_client._http.build_request = MagicMock(return_value=built_request)
        api_key_client._http.send = MagicMock(return_value=mock_resp)

        api_key_client._send("POST", "/api/stream", stream=True, timeout=300)

        timeout = api_key_client._http.build_request.call_args.kwargs["timeout"]
        assert isinstance(timeout, httpx.Timeout)
        assert timeout.read is None
        assert timeout.connect == 300
        assert timeout.write == 300

    def test_401_after_retry_raises_auth_error(self, token_client):
        mock_resp = MagicMock()
        mock_resp.status_code = 401
        mock_resp.is_success = False
        mock_resp.text = "Unauthorized"

        token_client._http.request = MagicMock(return_value=mock_resp)
        token_client._auth.refresh_access_token = MagicMock(
            return_value={"access_token": "new_at"}
        )

        with pytest.raises(ArenaAuthError, match="Session expired"):
            token_client._request("GET", "/api/test")

    def test_401_without_refresh_token_raises(self, api_key_client):
        api_key_client._api_key = None
        api_key_client._tokens.access_token = "at"
        api_key_client._tokens.refresh_token = None

        mock_resp = MagicMock()
        mock_resp.status_code = 401
        mock_resp.is_success = False
        mock_resp.text = "Unauthorized"

        api_key_client._http.request = MagicMock(return_value=mock_resp)
        with pytest.raises(ArenaAuthError, match="Session expired"):
            api_key_client._request("GET", "/api/test")

    def test_401_invalid_api_key_raises(self, api_key_client):
        api_key_client._tokens.access_token = None
        api_key_client._tokens.refresh_token = None

        mock_resp = MagicMock()
        mock_resp.status_code = 401
        mock_resp.is_success = False
        mock_resp.text = "Unauthorized"

        api_key_client._http.request = MagicMock(return_value=mock_resp)
        with pytest.raises(ArenaAuthError, match="Invalid API key"):
            api_key_client._request("GET", "/api/test")

    def test_401_api_key_falls_back_to_oauth_credentials(self, api_key_client):
        api_key_client._tokens.access_token = "oauth_at"
        api_key_client._tokens.refresh_token = "oauth_rt"

        first_resp = MagicMock()
        first_resp.status_code = 401
        first_resp.is_success = False
        first_resp.text = "Unauthorized"

        second_resp = MagicMock()
        second_resp.status_code = 200
        second_resp.is_success = True
        second_resp.headers = {"content-type": "application/json"}
        second_resp.json.return_value = {"ok": True}

        api_key_client._http.request = MagicMock(side_effect=[first_resp, second_resp])
        result = api_key_client._request("GET", "/api/test")

        assert result == {"ok": True}
        assert api_key_client._api_key is None
        assert api_key_client._http.request.call_count == 2

    def test_request_raw_returns_bytes_and_headers(self, api_key_client):
        mock_resp = MagicMock()
        mock_resp.content = b"csv,data"
        mock_resp.headers = {
            "content-type": "text/csv",
            "content-disposition": 'attachment; filename="m.csv"',
        }
        api_key_client._send = MagicMock(return_value=mock_resp)

        payload, content_type, disposition = api_key_client._request_raw(
            "GET", "/api/metrics"
        )
        assert payload == b"csv,data"
        assert content_type == "text/csv"
        assert disposition == 'attachment; filename="m.csv"'

    def test_non_success_raises_api_error(self, api_key_client):
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_resp.is_success = False
        mock_resp.text = "Internal Server Error"

        api_key_client._http.request = MagicMock(return_value=mock_resp)
        with pytest.raises(ArenaAPIError) as exc_info:
            api_key_client._request("GET", "/api/test")
        assert exc_info.value.status_code == 500

    def test_includes_auth_headers(self, api_key_client):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.is_success = True
        mock_resp.headers = {"content-type": "application/json"}
        mock_resp.json.return_value = {}

        api_key_client._http.request = MagicMock(return_value=mock_resp)
        api_key_client._request("POST", "/api/test", json={"data": 1})

        call_kwargs = api_key_client._http.request.call_args
        headers = call_kwargs.kwargs.get("headers") or call_kwargs[1].get("headers", {})
        assert "Authorization" in headers


class TestEnvironmentListMethods:
    def test_list_environments(self, api_key_client):
        api_key_client._request = MagicMock(
            return_value={"MyEnv": {"v1": {"validated": True}}}
        )
        result = api_key_client.list_environments()
        api_key_client._request.assert_called_once_with(
            "GET",
            "/api/cli/v1/environments",
            params={"name": None, "include_arena": False},
        )
        assert "MyEnv" in result

    def test_environment_exists(self, api_key_client):
        api_key_client._request = MagicMock(return_value={"exists": True})
        assert api_key_client.environment_exists("CartPole-v1", "v1") is True

    def test_list_environment_entrypoints(self, api_key_client):
        payload = {
            "entrypoints": ["main:MyEnv", "alt:AltEnv"],
            "version": "v2",
        }
        api_key_client._request = MagicMock(return_value=payload)
        result = api_key_client.list_environment_entrypoints("MyEnv", version="v2")
        api_key_client._request.assert_called_once_with(
            "GET",
            "/api/cli/v1/environments/entrypoints",
            params={"name": "MyEnv", "version": "v2"},
        )
        assert result == payload

    def test_environment_exists_is_registered_key(self, api_key_client):
        api_key_client._request = MagicMock(return_value={"is_registered": False})
        assert api_key_client.environment_exists("MyEnv", "v1") is False

    def test_environment_exists_non_dict_response(self, api_key_client):
        api_key_client._request = MagicMock(return_value=True)
        with pytest.raises(AttributeError):
            api_key_client.environment_exists("MyEnv", "v1")


class TestValidateEnvironment:
    def test_name_is_required(self, api_key_client):
        # ``name`` is mandatory and is never inferred from ``source``.
        with pytest.raises(TypeError):
            api_key_client.validate_environment()

    def test_no_source_collects_by_default(self, api_key_client):
        mock_stream = _mock_ndjson_stream({"valid": True})
        api_key_client._open_stream = MagicMock(return_value=mock_stream)
        result = api_key_client.validate_environment(name="MyEnv", version="v1")
        api_key_client._open_stream.assert_called_once_with(
            "POST",
            "/api/cli/v1/environments/validate",
            json={"name": "MyEnv", "version": "v1", "do_rollouts": False},
            timeout=api_key_client._upload_timeout,
        )
        mock_stream.collect.assert_called_once()
        assert result == {"valid": True}

    def test_source_file_calls_create_and_validate(self, api_key_client, tmp_path):
        archive = tmp_path / "env.tar.gz"
        archive.write_bytes(b"fake")
        cfg = tmp_path / "env_config.yaml"
        cfg.write_text("key: val")
        reqs = tmp_path / "requirements.txt"
        reqs.write_text("numpy")

        mock_stream = _mock_ndjson_stream({"status": "ok"})
        api_key_client._open_stream = MagicMock(return_value=mock_stream)
        result = api_key_client.validate_environment(
            name="MyEnv",
            version="v1",
            source=archive,
            env_config=cfg,
            requirements=reqs,
        )
        call_args = api_key_client._open_stream.call_args
        assert call_args[0] == ("POST", "/api/cli/v1/environments/create-and-validate")
        assert "files" in call_args[1]
        assert "data" in call_args[1]
        assert call_args[1]["data"]["name"] == "MyEnv"
        mock_stream.collect.assert_called_once()
        assert result == {"status": "ok"}

    def test_source_directory_calls_create_and_validate(self, api_key_client, tmp_path):
        env_dir = tmp_path / "my_env"
        env_dir.mkdir()
        (env_dir / "env.py").write_text("class MyEnv: pass")
        cfg = tmp_path / "env_config.yaml"
        cfg.write_text("key: val")
        reqs = tmp_path / "requirements.txt"
        reqs.write_text("numpy")

        mock_stream = _mock_ndjson_stream({"status": "ok"})
        api_key_client._open_stream = MagicMock(return_value=mock_stream)
        result = api_key_client.validate_environment(
            name="MyEnv",
            version="v1",
            source=env_dir,
            env_config=cfg,
            requirements=reqs,
        )
        call_args = api_key_client._open_stream.call_args
        assert call_args[0] == ("POST", "/api/cli/v1/environments/create-and-validate")
        file_tuple = call_args[1]["files"]["file"]
        assert file_tuple[0] == "my_env.tar.gz"
        # Path sources are streamed from disk (open handle, closed after send)
        assert not isinstance(file_tuple[1], bytes)
        assert file_tuple[1].closed
        mock_stream.collect.assert_called_once()
        assert result == {"status": "ok"}

    def test_source_bytes_calls_create_and_validate(self, api_key_client, tmp_path):
        cfg = tmp_path / "env_config.yaml"
        cfg.write_text("key: val")
        reqs = tmp_path / "requirements.txt"
        reqs.write_text("numpy")

        mock_stream = _mock_ndjson_stream({"status": "ok"})
        api_key_client._open_stream = MagicMock(return_value=mock_stream)
        result = api_key_client.validate_environment(
            name="MyEnv",
            version="v1",
            source=b"raw-archive-bytes",
            env_config=cfg,
            requirements=reqs,
        )
        call_args = api_key_client._open_stream.call_args
        file_tuple = call_args[1]["files"]["file"]
        assert file_tuple[0] == "environment.tar.gz"
        assert file_tuple[1] == b"raw-archive-bytes"
        mock_stream.collect.assert_called_once()
        assert result == {"status": "ok"}

    def test_source_without_config_sends_empty_defaults(self, api_key_client, tmp_path):
        archive = tmp_path / "env.tar.gz"
        archive.write_bytes(b"fake")

        mock_stream = _mock_ndjson_stream({"status": "ok"})
        api_key_client._open_stream = MagicMock(return_value=mock_stream)
        api_key_client.validate_environment(name="MyEnv", source=archive)
        call_args = api_key_client._open_stream.call_args
        files = call_args[1]["files"]
        assert files["env_config"] == ("env_config.yaml", b"", "application/x-yaml")
        assert files["requirements"] == ("requirements.txt", b"", "text/plain")

    def test_source_missing_path_raises(self, api_key_client):
        with pytest.raises(ArenaFileNotFoundError, match="not found"):
            api_key_client.validate_environment(
                name="MyEnv",
                source="/nonexistent/path.tar.gz",
                env_config="/nonexistent/config.yaml",
                requirements="/nonexistent/reqs.txt",
            )

    def test_create_and_validate_missing_env_config(self, api_key_client, tmp_path):
        archive = tmp_path / "env.tar.gz"
        archive.write_bytes(b"fake")
        with pytest.raises(ArenaFileNotFoundError, match="Upload file not found"):
            api_key_client.validate_environment(
                name="MyEnv",
                version="v1",
                source=archive,
                env_config=tmp_path / "missing.yaml",
            )

    def test_create_and_validate_missing_requirements(self, api_key_client, tmp_path):
        archive = tmp_path / "env.tar.gz"
        archive.write_bytes(b"fake")
        with pytest.raises(ArenaFileNotFoundError, match="Upload file not found"):
            api_key_client.validate_environment(
                name="MyEnv",
                version="v1",
                source=archive,
                requirements=tmp_path / "missing.txt",
            )

    def test_directory_autodetects_sidecar_files(self, api_key_client, tmp_path):
        env_dir = tmp_path / "my_env"
        env_dir.mkdir()
        (env_dir / "env.py").write_text("class MyEnv: pass")
        (env_dir / "requirements.txt").write_text("numpy")
        (env_dir / "env_config.yaml").write_text("key: val")

        mock_stream = _mock_ndjson_stream({"status": "ok"})
        api_key_client._open_stream = MagicMock(return_value=mock_stream)
        # Neither requirements nor env_config passed explicitly.
        api_key_client.validate_environment(name="MyEnv", source=env_dir)
        files = api_key_client._open_stream.call_args[1]["files"]
        # Real files were uploaded, not the empty placeholders (which are bytes).
        assert files["requirements"][0] == "requirements.txt"
        assert not isinstance(files["requirements"][1], bytes)
        assert files["env_config"][0] == "env_config.yaml"
        assert files["env_config"][2] == "application/x-yaml"
        assert not isinstance(files["env_config"][1], bytes)

    def test_directory_autodetects_json_env_config(self, api_key_client, tmp_path):
        env_dir = tmp_path / "my_env"
        env_dir.mkdir()
        (env_dir / "env.py").write_text("class MyEnv: pass")
        (env_dir / "env_config.json").write_text("{}")

        mock_stream = _mock_ndjson_stream({"status": "ok"})
        api_key_client._open_stream = MagicMock(return_value=mock_stream)
        api_key_client.validate_environment(name="MyEnv", source=env_dir)
        files = api_key_client._open_stream.call_args[1]["files"]
        assert files["env_config"][0] == "env_config.json"
        assert files["env_config"][2] == "application/json"

    def test_explicit_args_override_directory_sidecars(self, api_key_client, tmp_path):
        env_dir = tmp_path / "my_env"
        env_dir.mkdir()
        (env_dir / "env.py").write_text("x = 1")
        (env_dir / "requirements.txt").write_text("numpy")
        explicit = tmp_path / "explicit-reqs.txt"
        explicit.write_text("torch")

        mock_stream = _mock_ndjson_stream({"status": "ok"})
        api_key_client._open_stream = MagicMock(return_value=mock_stream)
        api_key_client.validate_environment(
            name="MyEnv", source=env_dir, requirements=explicit
        )
        files = api_key_client._open_stream.call_args[1]["files"]
        assert files["requirements"][0] == "explicit-reqs.txt"


class TestDefaultProjectConfig:
    def test_read_config_missing_file(self):
        with patch.object(ArenaClient, "CONFIG_FILE", Path("/nonexistent/config.json")):
            assert ArenaClient._read_config() == {}

    def test_read_config_invalid_json(self, tmp_path):
        config_file = tmp_path / "config.json"
        config_file.write_text("{not json", encoding="utf-8")
        with patch.object(ArenaClient, "CONFIG_FILE", config_file):
            assert ArenaClient._read_config() == {}

    def test_get_default_project(self, api_key_client, tmp_path):
        config_file = tmp_path / "config.json"
        config_file.write_text(
            json.dumps({"default_project": "CoreRL"}), encoding="utf-8"
        )
        with patch.object(ArenaClient, "CONFIG_FILE", config_file):
            assert api_key_client.get_default_project() == "CoreRL"

    def test_set_default_project_persists(self, api_key_client, tmp_path):
        config_file = tmp_path / "config.json"
        with patch.object(ArenaClient, "CONFIG_FILE", config_file):
            with patch.object(ArenaClient, "CONFIG_DIR", tmp_path):
                api_key_client.list_projects = MagicMock(
                    return_value=[
                        {"name": "CoreRL", "type": "Classic RL", "description": ""}
                    ]
                )
                api_key_client.set_default_project("CoreRL")
        assert json.loads(config_file.read_text())["default_project"] == "CoreRL"

    def test_set_default_project_unknown_raises(self, api_key_client):
        api_key_client.list_projects = MagicMock(return_value=[])
        with pytest.raises(ArenaConfigError, match="not found"):
            api_key_client.set_default_project("Missing")


class TestResolveProject:
    def test_explicit_project_returned(self, api_key_client):
        assert api_key_client._resolve_project("explicit") == "explicit"

    def test_falls_back_to_default(self, api_key_client):
        with patch.object(
            api_key_client, "get_default_project", return_value="my-default"
        ):
            assert api_key_client._resolve_project(None) == "my-default"

    def test_list_experiments_raises_config_error_when_no_project(self, api_key_client):
        with patch.object(api_key_client, "get_default_project", return_value=None):
            with pytest.raises(ArenaConfigError, match="No project specified"):
                api_key_client.list_experiments(project=None)


class TestStopExperiment:
    def test_posts_cli_stop_by_name(self, api_key_client):
        api_key_client._request = MagicMock(return_value="Ok")
        api_key_client.stop_experiment("my-exp")
        api_key_client._request.assert_called_once_with(
            "POST",
            "/api/cli/v1/experiments/jobs/stop",
            json={"experiment_name": "my-exp"},
        )


# ---------------------------------------------------------------------------


class TestContextManager:
    def test_enter_returns_self(self, api_key_client):
        assert api_key_client.__enter__() is api_key_client

    def test_exit_closes(self, api_key_client):
        api_key_client._http = MagicMock()
        api_key_client.__exit__(None, None, None)
        api_key_client._http.close.assert_called_once()

    def test_close(self, api_key_client):
        api_key_client._http = MagicMock()
        api_key_client.close()
        api_key_client._http.close.assert_called_once()


class TestRepr:
    def test_authenticated_repr(self, api_key_client):
        r = repr(api_key_client)
        assert "authenticated" in r
        assert "unauthenticated" not in r

    def test_unauthenticated_repr(self, unauthenticated_client):
        r = repr(unauthenticated_client)
        assert "unauthenticated" in r


class TestTryRestoreSession:
    @patch("agilerl.arena.auth.KeycloakOpenID")
    def test_restores_from_credentials(self, mock_keycloak, tmp_path):
        cred_file = tmp_path / "creds.json"
        cred_file.write_text(
            json.dumps({"access_token": "saved_at", "refresh_token": "saved_rt"}),
            encoding="utf-8",
        )

        from agilerl.arena.auth import ArenaOAuth2

        orig = ArenaOAuth2.CREDENTIALS_FILE
        try:
            ArenaOAuth2.CREDENTIALS_FILE = cred_file
            with patch.dict(os.environ, {}, clear=False):
                env = os.environ.copy()
                env.pop("ARENA_API_KEY", None)
                with patch.dict(os.environ, env, clear=True):
                    client = ArenaClient()
            assert client._tokens.access_token == "saved_at"
            assert client._tokens.refresh_token == "saved_rt"
        finally:
            ArenaOAuth2.CREDENTIALS_FILE = orig

    @patch("agilerl.arena.auth.KeycloakOpenID")
    def test_restores_access_token_only(self, mock_keycloak, tmp_path):
        """Credentials file with access_token but no refresh_token."""
        cred_file = tmp_path / "creds.json"
        cred_file.write_text(
            json.dumps({"access_token": "at_only"}),
            encoding="utf-8",
        )

        from agilerl.arena.auth import ArenaOAuth2

        orig = ArenaOAuth2.CREDENTIALS_FILE
        try:
            ArenaOAuth2.CREDENTIALS_FILE = cred_file
            with patch.dict(os.environ, {}, clear=False):
                env = os.environ.copy()
                env.pop("ARENA_API_KEY", None)
                with patch.dict(os.environ, env, clear=True):
                    client = ArenaClient()
            assert client._tokens.access_token == "at_only"
            assert client._tokens.refresh_token is None
        finally:
            ArenaOAuth2.CREDENTIALS_FILE = orig


class TestValidateEnvironmentParams:
    def test_forwards_version_and_entrypoint(self, api_key_client):
        mock_stream = _mock_ndjson_stream()
        api_key_client._open_stream = MagicMock(return_value=mock_stream)
        api_key_client.validate_environment(
            name="MyEnv",
            version="v2",
            entrypoint="my_env:make",
        )
        call_kwargs = api_key_client._open_stream.call_args[1]
        payload = call_kwargs["json"]
        assert payload["version"] == "v2"
        assert payload["entrypoint"] == "my_env:make"

    def test_create_forwards_multi_agent(self, api_key_client, tmp_path):
        archive = tmp_path / "env.tar.gz"
        archive.write_bytes(b"fake")
        cfg = tmp_path / "env_config.yaml"
        cfg.write_text("key: val")
        reqs = tmp_path / "requirements.txt"
        reqs.write_text("numpy")

        mock_stream = _mock_ndjson_stream()
        api_key_client._open_stream = MagicMock(return_value=mock_stream)
        api_key_client.validate_environment(
            name="MyEnv",
            source=archive,
            env_config=cfg,
            requirements=reqs,
            multi_agent=True,
        )
        call_kwargs = api_key_client._open_stream.call_args[1]
        assert call_kwargs["data"]["multi_agent"] == "true"

    def test_create_forwards_language_based(self, api_key_client, tmp_path):
        archive = tmp_path / "env.tar.gz"
        archive.write_bytes(b"fake")
        cfg = tmp_path / "env_config.yaml"
        cfg.write_text("key: val")
        reqs = tmp_path / "requirements.txt"
        reqs.write_text("numpy")

        mock_stream = _mock_ndjson_stream()
        api_key_client._open_stream = MagicMock(return_value=mock_stream)
        api_key_client.validate_environment(
            name="MyEnv",
            source=archive,
            env_config=cfg,
            requirements=reqs,
            language_based=True,
        )
        call_kwargs = api_key_client._open_stream.call_args[1]
        assert call_kwargs["data"]["language_based"] == "true"

    def test_create_forwards_entrypoint(self, api_key_client, tmp_path):
        archive = tmp_path / "env.tar.gz"
        archive.write_bytes(b"fake")

        mock_stream = _mock_ndjson_stream()
        api_key_client._open_stream = MagicMock(return_value=mock_stream)
        api_key_client.validate_environment(
            name="MyEnv",
            source=archive,
            entrypoint="my_env:make",
        )
        call_kwargs = api_key_client._open_stream.call_args[1]
        assert call_kwargs["data"]["entrypoint"] == "my_env:make"

    def test_create_forwards_description(self, api_key_client, tmp_path):
        archive = tmp_path / "env.tar.gz"
        archive.write_bytes(b"fake")
        cfg = tmp_path / "env_config.yaml"
        cfg.write_text("key: val")
        reqs = tmp_path / "requirements.txt"
        reqs.write_text("numpy")

        mock_stream = _mock_ndjson_stream()
        api_key_client._open_stream = MagicMock(return_value=mock_stream)
        api_key_client.validate_environment(
            name="MyEnv",
            source=archive,
            env_config=cfg,
            requirements=reqs,
            description="A test environment",
        )
        call_kwargs = api_key_client._open_stream.call_args[1]
        assert call_kwargs["data"]["description"] == "A test environment"

    def test_create_omits_description_when_none(self, api_key_client, tmp_path):
        archive = tmp_path / "env.tar.gz"
        archive.write_bytes(b"fake")
        cfg = tmp_path / "env_config.yaml"
        cfg.write_text("key: val")
        reqs = tmp_path / "requirements.txt"
        reqs.write_text("numpy")

        mock_stream = _mock_ndjson_stream()
        api_key_client._open_stream = MagicMock(return_value=mock_stream)
        api_key_client.validate_environment(
            name="MyEnv",
            source=archive,
            env_config=cfg,
            requirements=reqs,
        )
        call_kwargs = api_key_client._open_stream.call_args[1]
        assert "description" not in call_kwargs["data"]


class TestCliV1EndpointPaths:
    def test_profile_environment_uses_cli_v1_path(self, api_key_client):
        mock_stream = _mock_ndjson_stream()
        api_key_client._open_stream = MagicMock(return_value=mock_stream)
        api_key_client.profile_environment(name="MyEnv", version="v1")
        api_key_client._open_stream.assert_called_once_with(
            "POST",
            "/api/cli/v1/environments/profile",
            json={"name": "MyEnv", "version": "v1"},
            timeout=api_key_client._upload_timeout,
        )

    @patch("builtins.input", return_value="y")
    def test_delete_environment_uses_cli_v1_path(self, mock_input, api_key_client):
        api_key_client.list_environments = MagicMock(
            return_value={"MyEnv": {"v1": {"validated": True}}}
        )
        api_key_client._request = MagicMock(return_value={"deleted": True})
        result = api_key_client.delete_environment(name="MyEnv", version="v1")
        assert result == {"deleted": True}
        api_key_client._request.assert_called_once_with(
            "DELETE",
            "/api/cli/v1/environments/delete",
            json={"name": "MyEnv", "version": "v1"},
        )


class TestOpenStream:
    def test_verbose_creates_renderer_with_error_map(self, api_key_client):
        """Verbose client creates a StreamRichRenderer with the correct error_cls."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.is_success = True
        api_key_client._http.build_request = MagicMock(return_value=MagicMock())
        api_key_client._http.send = MagicMock(return_value=mock_resp)

        stream = api_key_client._open_stream(
            "POST", "/api/cli/v1/environments/validate"
        )

        assert isinstance(stream, NDJsonStream)
        assert isinstance(stream._renderer, StreamRichRenderer)
        assert stream._renderer._error_cls is ArenaValidationError
        assert stream._handler is not None

    def test_verbose_training_path_uses_training_error(self, api_key_client):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.is_success = True
        api_key_client._http.build_request = MagicMock(return_value=MagicMock())
        api_key_client._http.send = MagicMock(return_value=mock_resp)

        stream = api_key_client._open_stream(
            "POST", "/api/cli/v1/experiments/jobs/submit"
        )

        assert stream._renderer._error_cls is ArenaTrainingError

    def test_verbose_unmapped_path_uses_generic_error(self, api_key_client):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.is_success = True
        api_key_client._http.build_request = MagicMock(return_value=MagicMock())
        api_key_client._http.send = MagicMock(return_value=mock_resp)

        stream = api_key_client._open_stream("GET", "/api/some/other/path")

        assert stream._renderer._error_cls is ArenaAPIError

    def test_custom_stream_handler_bypasses_renderer(self, api_key_client):
        """When set_stream_handler is used, _open_stream uses that handler."""
        custom = MagicMock()
        api_key_client.set_stream_handler(custom)

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.is_success = True
        api_key_client._http.build_request = MagicMock(return_value=MagicMock())
        api_key_client._http.send = MagicMock(return_value=mock_resp)

        stream = api_key_client._open_stream(
            "POST", "/api/cli/v1/environments/validate"
        )

        assert stream._handler is custom
        assert stream._renderer is None

    def test_non_verbose_no_handler_no_renderer(self):
        """verbose=False means no renderer and no handler (unless custom set)."""
        with patch("agilerl.arena.auth.KeycloakOpenID"):
            client = ArenaClient(api_key="test-key", verbose=False)

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.is_success = True
        client._http.build_request = MagicMock(return_value=MagicMock())
        client._http.send = MagicMock(return_value=mock_resp)

        stream = client._open_stream("POST", "/api/cli/v1/environments/validate")

        assert stream._handler is None
        assert stream._renderer is None


class TestSendStreaming:
    def test_success_returns_raw_response(self, api_key_client):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.is_success = True
        api_key_client._http.build_request = MagicMock(return_value=MagicMock())
        api_key_client._http.send = MagicMock(return_value=mock_resp)

        resp = api_key_client._send("POST", "/api/test", stream=True)
        assert resp is mock_resp
        mock_resp.read.assert_not_called()

    def test_401_closes_stream_then_retries(self, token_client):
        first_resp = MagicMock()
        first_resp.status_code = 401
        first_resp.is_success = False

        second_resp = MagicMock()
        second_resp.status_code = 200
        second_resp.is_success = True

        request_mock = MagicMock()
        token_client._http.build_request = MagicMock(return_value=request_mock)
        token_client._http.send = MagicMock(side_effect=[first_resp, second_resp])
        token_client._auth.refresh_access_token = MagicMock(
            return_value={"access_token": "new_at", "refresh_token": "new_rt"}
        )

        resp = token_client._send("POST", "/api/test", stream=True)
        first_resp.close.assert_called_once()
        assert resp is second_resp
        assert token_client._tokens.access_token == "new_at"

    def test_401_after_retry_reads_body_raises(self, token_client):
        mock_resp = MagicMock()
        mock_resp.status_code = 401
        mock_resp.is_success = False
        mock_resp.read.return_value = b"Session expired"

        request_mock = MagicMock()
        token_client._http.build_request = MagicMock(return_value=request_mock)
        token_client._http.send = MagicMock(return_value=mock_resp)
        token_client._auth.refresh_access_token = MagicMock(
            return_value={"access_token": "new_at"}
        )

        with pytest.raises(ArenaAuthError, match="Session expired"):
            token_client._send("POST", "/api/test", stream=True)

    def test_non_success_on_validation_path_raises_validation_error(
        self, api_key_client
    ):
        mock_resp = MagicMock()
        mock_resp.status_code = 422
        mock_resp.is_success = False
        mock_resp.read.return_value = b'{"detail": "Bad environment"}'

        api_key_client._http.build_request = MagicMock(return_value=MagicMock())
        api_key_client._http.send = MagicMock(return_value=mock_resp)

        with pytest.raises(ArenaValidationError):
            api_key_client._send(
                "POST",
                "/api/cli/v1/environments/validate",
                stream=True,
            )

    def test_non_success_on_unmapped_path_raises_generic(self, api_key_client):
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_resp.is_success = False
        mock_resp.read.return_value = b'{"message": "Server error"}'

        api_key_client._http.build_request = MagicMock(return_value=MagicMock())
        api_key_client._http.send = MagicMock(return_value=mock_resp)

        with pytest.raises(ArenaAPIError) as exc_info:
            api_key_client._send("POST", "/api/other", stream=True)
        assert exc_info.value.status_code == 500

    def test_network_error_raises_api_error(self, api_key_client):
        api_key_client._http.build_request = MagicMock(return_value=MagicMock())
        api_key_client._http.send = MagicMock(side_effect=httpx.ConnectError("refused"))

        with pytest.raises(ArenaAPIError) as exc_info:
            api_key_client._send("POST", "/api/test", stream=True)
        assert exc_info.value.status_code == 0
        assert "Network error" in exc_info.value.detail


class TestLoginForceWithApiKey:
    def test_force_clears_api_key_and_runs_device_login(self, api_key_client):
        client = api_key_client
        assert client._api_key == "test-key"
        tokens = {"access_token": "new_at", "refresh_token": "new_rt"}
        client._auth.device_login = MagicMock(return_value=tokens)

        client.login(force=True)

        assert client._api_key is None
        client._auth.device_login.assert_called_once()
        assert client._tokens.access_token == "new_at"
        assert client._tokens.refresh_token == "new_rt"


class TestDeleteEnvironmentMultiVersion:
    def test_delete_no_versions_found(self, api_key_client):
        api_key_client.list_environments = MagicMock(return_value={"MyEnv": {}})
        result = api_key_client.delete_environment(name="MyEnv")
        assert result is None

    def test_delete_unknown_version_returns_none(self, api_key_client):
        api_key_client.list_environments = MagicMock(
            return_value={"MyEnv": {"v1": {"validated": True}}}
        )
        result = api_key_client.delete_environment(name="MyEnv", version="v9")
        assert result is None

    @patch("builtins.input", return_value="n")
    def test_delete_aborted_by_user(self, mock_input, api_key_client):
        api_key_client.list_environments = MagicMock(
            return_value={"MyEnv": {"v1": {"validated": True}}}
        )
        api_key_client._request = MagicMock()
        result = api_key_client.delete_environment(name="MyEnv", version="v1")
        assert result is None
        api_key_client._request.assert_not_called()

    @patch("builtins.input", return_value="n")
    def test_delete_all_versions_aborted(self, mock_input, api_key_client):
        api_key_client.list_environments = MagicMock(
            return_value={"MyEnv": {"v1": {}, "v2": {}}}
        )
        api_key_client._request = MagicMock()
        result = api_key_client.delete_environment(name="MyEnv")
        assert result is None
        api_key_client._request.assert_not_called()

    @patch("builtins.input", return_value="y")
    def test_delete_specific_version(self, mock_input, api_key_client):
        api_key_client.list_environments = MagicMock(
            return_value={"MyEnv": {"v1": {"validated": True}}}
        )
        api_key_client._request = MagicMock(return_value={"deleted": True})
        result = api_key_client.delete_environment(name="MyEnv", version="v1")
        api_key_client._request.assert_called_once_with(
            "DELETE",
            "/api/cli/v1/environments/delete",
            json={"name": "MyEnv", "version": "v1"},
        )
        assert result == {"deleted": True}


class TestDuplicateEnvironmentVersion:
    def test_calls_request_with_correct_payload(self, api_key_client):
        api_key_client._request = MagicMock(
            return_value={"source_version": "v1", "duplicated": True}
        )
        result = api_key_client.duplicate_environment_version(
            name="MyEnv", new_version="v2", version="v1"
        )
        api_key_client._request.assert_called_once_with(
            "POST",
            "/api/cli/v1/environments/duplicate",
            json={"name": "MyEnv", "new_version_name": "v2", "version": "v1"},
        )
        assert result == {"source_version": "v1", "duplicated": True}

    def test_version_none_uses_latest(self, api_key_client):
        api_key_client._request = MagicMock(return_value={"source_version": "v1"})
        api_key_client.duplicate_environment_version(name="Env", new_version="v3")
        payload = api_key_client._request.call_args[1]["json"]
        assert payload["version"] is None


class TestExperimentMethods:
    def test_list_experiments(self, api_key_client):
        api_key_client._request = MagicMock(return_value=[{"name": "exp1"}])
        result = api_key_client.list_experiments("proj1")
        api_key_client._request.assert_called_once_with(
            "GET", "/api/cli/v1/experiments/list", params={"project": "proj1"}
        )
        assert result == [{"name": "exp1"}]

    @patch("agilerl.arena.client.TrainingManifest.get_validated")
    def test_submit_experiment(self, mock_validated, api_key_client):
        mock_validated.return_value = {"algorithm": "PPO"}
        mock_stream = _mock_ndjson_stream({"job_id": 1})
        api_key_client._open_stream = MagicMock(return_value=mock_stream)

        result = api_key_client.submit_experiment(
            manifest={"algorithm": "PPO"},
            resource_id=2,
            num_nodes=1,
            project="proj",
            experiment_name="exp1",
        )

        assert result == {"job_id": 1}
        mock_validated.assert_called_once()
        call_kwargs = api_key_client._open_stream.call_args[1]
        assert call_kwargs["json"]["manifest"] == {"algorithm": "PPO"}
        assert call_kwargs["json"]["project"] == "proj"
        assert "files" not in call_kwargs

    @patch("agilerl.arena.client.TrainingManifest.get_validated")
    def test_submit_experiment_with_reward_file(
        self, mock_validated, api_key_client, tmp_path
    ):
        mock_validated.return_value = {
            "algorithm": {"name": "GRPO"},
            "environment": {"name": "ds", "num_envs": 16},
        }
        reward_path = tmp_path / "reward.py"
        reward_path.write_text(
            "def reward(question, answer, completion):\n    return 1.0\n",
        )
        mock_stream = _mock_ndjson_stream({"job_id": 2})
        api_key_client._open_stream = MagicMock(return_value=mock_stream)

        result = api_key_client.submit_experiment(
            manifest={"algorithm": {"name": "GRPO"}},
            resource_id="arena-medium",
            num_nodes=2,
            project="proj",
            experiment_name="exp-reasoning",
            reward_file=reward_path,
            completion="wrong answer",
        )

        assert result == {"job_id": 2}
        call_kwargs = api_key_client._open_stream.call_args[1]
        assert "json" not in call_kwargs
        files = call_kwargs["files"]
        assert json.loads(files["manifest"][1]) == mock_validated.return_value
        assert files["project"] == (None, "proj")
        assert files["resource_id"] == (None, "arena-medium")
        assert files["num_nodes"] == (None, "2")
        assert files["experiment_name"] == (None, "exp-reasoning")
        assert files["completion"] == (None, "wrong answer")
        assert files["reward_file"][0] == "reward.py"

    def test_resume_experiment(self, api_key_client):
        api_key_client._request = MagicMock(return_value={"resumed": True})
        result = api_key_client.resume_experiment("exp1", max_steps=1000)
        api_key_client._request.assert_called_once_with(
            "POST",
            "/api/cli/v1/experiments/jobs/resume",
            json={"experiment_name": "exp1", "max_steps": 1000},
        )
        assert result == {"resumed": True}

    def test_list_checkpoints(self, api_key_client):
        api_key_client._request = MagicMock(return_value=[{"step": 100}])
        result = api_key_client.list_checkpoints("exp1")
        api_key_client._request.assert_called_once_with(
            "GET",
            "/api/cli/v1/experiments/jobs/checkpoints",
            params={"experiment_name": "exp1"},
        )
        assert result == [{"step": 100}]


class TestPreviewExperimentMetricsCsv:
    def test_basic_call(self, api_key_client):
        api_key_client._request_raw = MagicMock(
            return_value=(b"col1,col2\n1,2\n", "text/csv", None)
        )
        payload, _ct, _disp = api_key_client.preview_experiment_metrics_csv(
            "exp1", preview_rows=10
        )
        call_kwargs = api_key_client._request_raw.call_args[1]
        params = call_kwargs["params"]
        assert ("experiment_name", "exp1") in params
        assert ("preview_rows", 10) in params
        assert payload == b"col1,col2\n1,2\n"

    def test_with_metrics_and_project(self, api_key_client):
        api_key_client._request_raw = MagicMock(
            return_value=(b"data", "text/csv", None)
        )
        api_key_client.preview_experiment_metrics_csv(
            "exp1", preview_rows=5, metrics=["loss", "reward"], project="proj1"
        )
        call_kwargs = api_key_client._request_raw.call_args[1]
        params = call_kwargs["params"]
        assert ("project", "proj1") in params
        # Each metric must be a ("metric", name) query pair, not flattened strings
        assert ("metric", "loss") in params
        assert ("metric", "reward") in params
        assert all(isinstance(p, tuple) and len(p) == 2 for p in params)


class TestListExperimentMetricNames:
    def test_basic_call(self, api_key_client):
        api_key_client._request = MagicMock(return_value=["loss", "reward"])
        with patch.object(api_key_client, "get_default_project", return_value=None):
            result = api_key_client.list_experiment_metric_names("exp1")
        api_key_client._request.assert_called_once_with(
            "GET",
            "/api/cli/v1/experiments/metrics",
            params={"experiment_name": "exp1"},
        )
        assert result == ["loss", "reward"]

    def test_with_project_and_details(self, api_key_client):
        api_key_client._request = MagicMock(
            return_value={"experiment_id": "123", "metrics": ["a"]}
        )
        result = api_key_client.list_experiment_metric_names(
            "exp1", project="proj1", details=True
        )
        call_kwargs = api_key_client._request.call_args[1]
        assert call_kwargs["params"] == {
            "experiment_name": "exp1",
            "project": "proj1",
            "details": True,
        }
        assert result == {"experiment_id": "123", "metrics": ["a"]}


class TestListResources:
    def test_calls_correct_endpoint(self, api_key_client):
        api_key_client._request = MagicMock(return_value={"tiers": []})
        result = api_key_client.list_resources()
        api_key_client._request.assert_called_once_with(
            "GET", "/api/cli/v1/resources/list"
        )
        assert result == {"tiers": []}


class TestDownloadExperimentMetrics:
    def test_default_output_path(self, api_key_client, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        api_key_client.preview_experiment_metrics_csv = MagicMock(
            return_value=(b"csv-data", "text/csv", None)
        )
        result = api_key_client.download_experiment_metrics("exp1")
        assert result == Path("exp1_metrics.csv")
        assert result.read_bytes() == b"csv-data"
        api_key_client.preview_experiment_metrics_csv.assert_called_once_with(
            "exp1", preview_rows=50_000, metrics=None
        )

    def test_output_path_as_directory(self, api_key_client, tmp_path):
        api_key_client.preview_experiment_metrics_csv = MagicMock(
            return_value=(b"data", "text/csv", 'attachment; filename="custom.csv"')
        )
        result = api_key_client.download_experiment_metrics(
            "exp1", output_path=tmp_path
        )
        assert result == tmp_path / "custom.csv"
        assert result.read_bytes() == b"data"

    def test_output_path_as_directory_no_disposition(self, api_key_client, tmp_path):
        api_key_client.preview_experiment_metrics_csv = MagicMock(
            return_value=(b"data", "text/csv", None)
        )
        result = api_key_client.download_experiment_metrics(
            "exp1", output_path=tmp_path
        )
        assert result == tmp_path / "exp1_metrics.csv"

    def test_raises_if_file_exists_before_downloading(
        self, api_key_client, tmp_path, monkeypatch
    ):
        monkeypatch.chdir(tmp_path)
        existing = tmp_path / "exp1_metrics.csv"
        existing.write_text("old")
        api_key_client.preview_experiment_metrics_csv = MagicMock(
            return_value=(b"new", "text/csv", None)
        )
        with pytest.raises(FileExistsError, match="a file of that name already exists"):
            api_key_client.download_experiment_metrics("exp1")
        api_key_client.preview_experiment_metrics_csv.assert_not_called()
        assert existing.read_text() == "old"

    def test_creates_parent_directory_if_missing(self, api_key_client, tmp_path):
        api_key_client.preview_experiment_metrics_csv = MagicMock(
            return_value=(b"new", "text/csv", None)
        )
        target = tmp_path / "nested" / "dirs" / "metrics.csv"
        result = api_key_client.download_experiment_metrics("exp1", output_path=target)
        assert result == target
        assert target.read_bytes() == b"new"

    def test_raises_if_parent_is_not_a_directory(self, api_key_client, tmp_path):
        not_a_dir = tmp_path / "blocker"
        not_a_dir.write_text("x")
        api_key_client.preview_experiment_metrics_csv = MagicMock(
            return_value=(b"new", "text/csv", None)
        )
        target = not_a_dir / "metrics.csv"
        with pytest.raises(NotADirectoryError, match="is not a directory"):
            api_key_client.download_experiment_metrics("exp1", output_path=target)
        api_key_client.preview_experiment_metrics_csv.assert_not_called()

    def test_raises_if_resolved_directory_target_exists(self, api_key_client, tmp_path):
        (tmp_path / "custom.csv").write_text("old")
        api_key_client.preview_experiment_metrics_csv = MagicMock(
            return_value=(b"new", "text/csv", 'attachment; filename="custom.csv"')
        )
        with pytest.raises(FileExistsError, match="a file of that name already exists"):
            api_key_client.download_experiment_metrics("exp1", output_path=tmp_path)
        assert (tmp_path / "custom.csv").read_text() == "old"

    def test_metrics_param_forwarded(self, api_key_client, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        api_key_client.preview_experiment_metrics_csv = MagicMock(
            return_value=(b"data", "text/csv", None)
        )
        api_key_client.download_experiment_metrics("exp1", metrics=["loss"])
        api_key_client.preview_experiment_metrics_csv.assert_called_once_with(
            "exp1", preview_rows=50_000, metrics=["loss"]
        )

    def test_rejects_non_csv_response(self, api_key_client, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        api_key_client.preview_experiment_metrics_csv = MagicMock(
            return_value=(b"<!DOCTYPE html>", "text/html", None)
        )
        with pytest.raises(ArenaAPIError):
            api_key_client.download_experiment_metrics("exp1")


class TestProjectMethods:
    def test_list_projects(self, api_key_client):
        api_key_client._request = MagicMock(
            return_value=[
                {"name": "p1", "type": "Classic RL", "description": "d1"},
            ]
        )
        result = api_key_client.list_projects()
        api_key_client._request.assert_called_once_with("GET", "/api/cli/v1/projects")
        assert result == [
            {"name": "p1", "type": "Classic RL", "description": "d1"},
        ]

    def test_list_projects_empty_response(self, api_key_client):
        api_key_client._request = MagicMock(return_value=[])
        assert api_key_client.list_projects() == []

    def test_create_project(self, api_key_client):
        api_key_client._request = MagicMock(return_value={"id": 1, "name": "p1"})
        result = api_key_client.create_project("p1", "desc", llm_based=True)
        api_key_client._request.assert_called_once_with(
            "POST",
            "/api/cli/v1/projects/create",
            json={"name": "p1", "description": "desc", "llm_based": True},
        )
        assert result == {"id": 1, "name": "p1"}

    def test_delete_project(self, api_key_client):
        api_key_client._request = MagicMock(return_value=None)
        result = api_key_client.delete_project("p1")
        api_key_client._request.assert_called_once_with(
            "DELETE", "/api/cli/v1/projects/delete", json={"name": "p1"}
        )
        assert result is None


class TestInferenceDeployments:
    def test_deploy_agent(self, api_key_client, caplog):
        api_key_client._request = MagicMock(return_value={"deployed": True})
        with caplog.at_level(logging.INFO, logger="agilerl.arena.client"):
            result = api_key_client.deploy_agent("exp1", checkpoint="step_100")
        api_key_client._request.assert_called_once_with(
            "POST",
            "/api/cli/v1/inference/deploy",
            json={"experiment_name": "exp1", "checkpoint": "step_100"},
        )
        assert result == {"deployed": True}
        assert "submitted for deployment" in caplog.text
        assert "Check its status in Arena" in caplog.text
        assert "step_100" in caplog.text

    def test_deploy_agent_no_checkpoint(self, api_key_client):
        api_key_client._request = MagicMock(return_value={"deployed": True})
        api_key_client.deploy_agent("exp1")
        payload = api_key_client._request.call_args[1]["json"]
        assert payload["checkpoint"] is None

    @pytest.mark.parametrize("scope", ["user", "organization"])
    def test_deploy_agent_sends_memory_scope(self, api_key_client, scope):
        api_key_client._request = MagicMock(return_value={"deployed": True})
        api_key_client.deploy_agent("exp1", memory_scope=scope)
        assert api_key_client._request.call_args[1]["json"]["memoryScope"] == scope

    def test_deploy_agent_omits_memory_scope_to_keep_the_stored_one(
        self, api_key_client
    ):
        api_key_client._request = MagicMock(return_value={"deployed": True})
        api_key_client.deploy_agent("exp1")
        assert "memoryScope" not in api_key_client._request.call_args[1]["json"]

    def test_deploy_agent_rejects_unknown_memory_scope(self, api_key_client):
        api_key_client._request = MagicMock(return_value={"deployed": True})
        with pytest.raises(ArenaValidationError, match="Unknown memory scope"):
            api_key_client.deploy_agent("exp1", memory_scope="team")
        api_key_client._request.assert_not_called()

    def test_list_inference_deployments(self, api_key_client):
        api_key_client._request = MagicMock(
            return_value=[{"name": "dep1"}, {"name": "dep2"}]
        )
        result = api_key_client.list_inference_deployments(experiment_name="exp1")
        assert result == [{"name": "dep1"}, {"name": "dep2"}]

    def test_list_inference_deployments_non_list_response(self, api_key_client):
        api_key_client._request = MagicMock(return_value="unexpected")
        result = api_key_client.list_inference_deployments()
        assert result == []

    def test_list_inference_deployments_filters_non_dicts(self, api_key_client):
        api_key_client._request = MagicMock(
            return_value=[{"name": "dep1"}, "garbage", 42]
        )
        result = api_key_client.list_inference_deployments()
        assert result == [{"name": "dep1"}]

    def test_fetch_deployment_for_inference_single_match(self, api_key_client):
        api_key_client._request = MagicMock(
            return_value={
                "name": "my-dep",
                "spec": {"url": "http://x"},
                "api_key": "key",
            }
        )
        result = api_key_client._fetch_deployment_for_inference("my-dep")
        assert result == {
            "name": "my-dep",
            "spec": {"url": "http://x"},
            "api_key": "key",
        }
        api_key_client._request.assert_called_once_with(
            "GET",
            "/api/cli/v1/inference/deployments/one",
            params={"name": "my-dep"},
        )

    def test_fetch_deployment_for_inference_no_match_raises(self, api_key_client):
        api_key_client._request = MagicMock(
            side_effect=ArenaAPIError("not found", status_code=404)
        )
        with pytest.raises(ArenaAPIError, match="not found"):
            api_key_client._fetch_deployment_for_inference("missing")

    def test_fetch_deployment_for_inference_multiple_raises(self, api_key_client):
        api_key_client._request = MagicMock(
            side_effect=ArenaAPIError("ambiguous", status_code=400)
        )
        with pytest.raises(ArenaAPIError, match="ambiguous"):
            api_key_client._fetch_deployment_for_inference("d")

    def test_fetch_deployment_for_inference_preserves_existing_hint(
        self, api_key_client
    ):
        original = ArenaAPIError(
            "deployment query failed",
            status_code=400,
            cli_hint="Run 'arena login' first.",
        )
        api_key_client._request = MagicMock(side_effect=original)
        with pytest.raises(ArenaAPIError) as exc_info:
            api_key_client._fetch_deployment_for_inference("dep")
        # An error that already carries a hint propagates unchanged, rather than
        # being replaced with the disambiguation hint.
        assert exc_info.value is original
        assert exc_info.value.cli_hint == "Run 'arena login' first."

    def test_fetch_deployment_for_inference_non_dict_raises(self, api_key_client):
        api_key_client._request = MagicMock(return_value="not a dict")
        with pytest.raises(ArenaAPIError, match="Unexpected deployment detail"):
            api_key_client._fetch_deployment_for_inference("dep")

    def test_fetch_deployment_for_inference_empty_name_raises(self, api_key_client):
        with pytest.raises(ArenaAPIError, match="deployment name is required"):
            api_key_client._fetch_deployment_for_inference("   ")

    def test_deployment_lookup_params(self):
        params = ArenaClient._deployment_lookup_params(
            name=" dep ",
            experiment_name=" exp ",
            project_name=" proj ",
        )
        assert params == {
            "name": "dep",
            "experimentName": "exp",
            "projectName": "proj",
        }


class TestDeploymentUrl:
    def test_happy_path(self):
        row = {"url": "http://inference.example.com"}
        assert ArenaClient._deployment_url(row) == "http://inference.example.com"

    def test_missing_url_raises(self):
        with pytest.raises(ArenaAPIError, match="no inference URL"):
            ArenaClient._deployment_url({})

    def test_empty_url_raises(self):
        with pytest.raises(ArenaAPIError, match="no inference URL"):
            ArenaClient._deployment_url({"url": "  "})

    def test_strips_whitespace(self):
        assert ArenaClient._deployment_url({"url": " http://x "}) == "http://x"


class TestEnsureInferenceBinding:
    @patch("agilerl.arena.client.save_binding")
    @patch("agilerl.arena.client.load_binding")
    def test_returns_cached_when_not_refresh(
        self, mock_load, mock_save, api_key_client
    ):
        mock_load.return_value = "http://cached"
        result = api_key_client._ensure_inference_binding("my-dep")
        assert result == "http://cached"
        mock_save.assert_not_called()

    @patch("agilerl.arena.client.save_binding")
    @patch("agilerl.arena.client.load_binding")
    def test_fetches_and_caches_on_refresh(self, mock_load, mock_save, api_key_client):
        mock_load.return_value = "http://cached"
        api_key_client._fetch_deployment_for_inference = MagicMock(
            return_value={"url": "http://new"}
        )
        result = api_key_client._ensure_inference_binding("my-dep", refresh=True)
        assert result == "http://new"
        mock_save.assert_called_once_with("my-dep", "http://new")

    @patch("agilerl.arena.client.save_binding")
    @patch("agilerl.arena.client.load_binding")
    def test_fetches_when_no_cache(self, mock_load, mock_save, api_key_client):
        mock_load.return_value = None
        api_key_client._fetch_deployment_for_inference = MagicMock(
            return_value={"url": "http://new"}
        )
        result = api_key_client._ensure_inference_binding("dep")
        assert result == "http://new"
        mock_save.assert_called_once_with("dep", "http://new")

    @patch("agilerl.arena.client.save_binding")
    @patch("agilerl.arena.client.load_binding")
    def test_ignores_any_api_key_the_row_still_carries(
        self, mock_load, mock_save, api_key_client
    ):
        mock_load.return_value = None
        api_key_client._fetch_deployment_for_inference = MagicMock(
            return_value={"url": "http://new", "api_key": "legacy-secret"}
        )
        api_key_client._ensure_inference_binding("dep")
        mock_save.assert_called_once_with("dep", "http://new")


class TestOpenInferenceAgent:
    @patch("agilerl.arena.client.Agent")
    def test_returns_agent_instance(self, mock_agent_cls, api_key_client):
        api_key_client._ensure_inference_binding = MagicMock(return_value="http://url")
        mock_agent = MagicMock()
        mock_agent_cls.return_value = mock_agent

        result = api_key_client.open_inference_agent("dep1")

        mock_agent_cls.assert_called_once_with(
            "http://url",
            api_key="test-key",
            timeout=api_key_client._request_timeout,
        )
        assert result is mock_agent

    @patch("agilerl.arena.client.Agent")
    def test_forwards_the_pat_when_the_client_has_one(
        self, mock_agent_cls, api_key_client
    ):
        api_key_client._ensure_inference_binding = MagicMock(return_value="http://url")
        api_key_client._api_key = "arena_pat_abc"
        api_key_client._tokens.access_token = "jwt-123"

        api_key_client.open_inference_agent("dep1")

        assert mock_agent_cls.call_args[1]["api_key"] == "arena_pat_abc"

    @patch("agilerl.arena.client.Agent")
    def test_falls_back_to_the_oauth_token(self, mock_agent_cls, token_client):
        token_client._ensure_inference_binding = MagicMock(return_value="http://url")

        token_client.open_inference_agent("dep1")

        assert mock_agent_cls.call_args[1]["api_key"] == "tok_access"

    @patch("agilerl.arena.client.Agent")
    def test_no_credential_passes_none(self, mock_agent_cls, api_key_client):
        api_key_client._ensure_inference_binding = MagicMock(return_value="http://url")
        api_key_client._api_key = None
        api_key_client._tokens.access_token = None

        api_key_client.open_inference_agent("dep1")

        assert mock_agent_cls.call_args[1]["api_key"] is None

    @patch("agilerl.arena.client.Agent")
    def test_custom_timeout(self, mock_agent_cls, api_key_client):
        api_key_client._ensure_inference_binding = MagicMock(return_value="http://url")
        api_key_client.open_inference_agent("dep", timeout=120)
        call_kwargs = mock_agent_cls.call_args[1]
        assert call_kwargs["timeout"] == 120

    @patch("agilerl.arena.client.Agent")
    def test_forwards_refresh_and_filters(self, mock_agent_cls, api_key_client):
        api_key_client._ensure_inference_binding = MagicMock(return_value="http://url")
        api_key_client.open_inference_agent(
            "dep", refresh=True, experiment_name="e", project_name="p"
        )
        api_key_client._ensure_inference_binding.assert_called_once_with(
            "dep", refresh=True, experiment_name="e", project_name="p"
        )


def _capabilities_response(
    *, status_code=200, is_success=True, json_value=None, raise_json=False
):
    """Build a fake httpx response for ``_get_cli_capabilities`` tests."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.is_success = is_success
    resp.text = "error body"
    if raise_json:
        resp.json.side_effect = ValueError("not json")
    else:
        resp.json.return_value = json_value
    return resp


class TestRewindUploadFiles:
    def test_rewinds_seekable_payload(self):
        import io

        buf = io.BytesIO(b"payload")
        buf.seek(7)  # simulate a consumed stream after a failed attempt
        files = {"manifest": ("manifest.json", buf, "application/json")}
        ArenaClient._rewind_upload_files(files)
        assert buf.tell() == 0

    def test_handles_none_and_non_seekable_entries(self):
        # No tuple payload and a non-seekable payload must both be ignored.
        ArenaClient._rewind_upload_files(None)
        ArenaClient._rewind_upload_files({"a": ("name", object(), "text/plain")})


class TestGetCliCapabilities:
    def test_returns_cached_without_refetch(self, api_key_client):
        api_key_client._cli_capabilities_cache = {"schemaVersion": 1}
        with patch.object(api_key_client._http, "request") as request:
            result = api_key_client._get_cli_capabilities()
        assert result == {"schemaVersion": 1}
        request.assert_not_called()

    def test_returns_none_on_http_error(self, api_key_client):
        api_key_client._cli_capabilities_cache = {"stale": True}
        with patch.object(api_key_client, "_auth_headers", return_value={}):
            with patch.object(
                api_key_client._http,
                "request",
                side_effect=httpx.ConnectError("connection refused"),
            ):
                result = api_key_client._get_cli_capabilities(force_refresh=True)
        assert result is None
        assert api_key_client._cli_capabilities_cache is None

    def test_returns_none_on_404(self, api_key_client):
        resp = _capabilities_response(status_code=404, is_success=False)
        with patch.object(api_key_client, "_auth_headers", return_value={}):
            with patch.object(api_key_client._http, "request", return_value=resp):
                assert api_key_client._get_cli_capabilities(force_refresh=True) is None

    def test_raises_on_other_error_status(self, api_key_client):
        resp = _capabilities_response(status_code=500, is_success=False)
        with patch.object(api_key_client, "_auth_headers", return_value={}):
            with patch.object(api_key_client._http, "request", return_value=resp):
                with pytest.raises(ArenaAPIError):
                    api_key_client._get_cli_capabilities(force_refresh=True)

    def test_returns_none_when_envelope_not_ok(self, api_key_client):
        resp = _capabilities_response(json_value={"ok": False})
        with patch.object(api_key_client, "_auth_headers", return_value={}):
            with patch.object(api_key_client._http, "request", return_value=resp):
                assert api_key_client._get_cli_capabilities(force_refresh=True) is None

    def test_returns_none_when_data_not_dict(self, api_key_client):
        resp = _capabilities_response(json_value={"ok": True, "data": "nope"})
        with patch.object(api_key_client, "_auth_headers", return_value={}):
            with patch.object(api_key_client._http, "request", return_value=resp):
                assert api_key_client._get_cli_capabilities(force_refresh=True) is None


class TestValidateManifestInvoke:
    def test_rejects_non_string_path(self, api_key_client):
        with pytest.raises(ArenaValidationError, match="invalid on-prem command"):
            api_key_client._validate_manifest_invoke(
                {"path": 123, "method": "GET", "responseKind": "json"}
            )


class TestPartitionManifestArgs:
    def test_skips_absent_optional_and_defaults_empty_body(self, api_key_client):
        params = [{"in": "body", "name": "x", "required": False, "type": "string"}]
        query, body = api_key_client._partition_manifest_args(
            method="POST", params_list=params, parsed_args={}
        )
        # Optional + absent -> skipped; body still defaults to {} when expected.
        assert query == {}
        assert body == {}

    def test_raises_when_required_value_is_none(self, api_key_client):
        params = [{"in": "body", "name": "x", "required": True, "type": "string"}]
        with pytest.raises(ArenaValidationError, match="Missing required argument"):
            api_key_client._partition_manifest_args(
                method="POST", params_list=params, parsed_args={"x": None}
            )

    def test_skips_optional_none_value(self, api_key_client):
        params = [{"in": "query", "name": "x", "required": False, "type": "string"}]
        query, body = api_key_client._partition_manifest_args(
            method="GET", params_list=params, parsed_args={"x": None}
        )
        assert query == {}
        assert body is None


class TestOpenInferenceAgentAfterARedeploy:
    """A redeploy moves the URL, leaving the cached one answering 404."""

    @staticmethod
    def _binding(client, *urls):
        """Return each url in turn, so refresh=True yields the next one."""
        client._ensure_inference_binding = MagicMock(side_effect=list(urls))

    @patch("agilerl.arena.client.Agent")
    def test_a_404_refetches_the_binding_and_retries(
        self, mock_agent_cls, api_key_client
    ):
        self._binding(api_key_client, "http://stale", "http://fresh")
        agent = MagicMock()
        mock_agent_cls.side_effect = [
            ArenaInferenceError(status_code=404, detail="No details"),
            agent,
        ]

        result = api_key_client.open_inference_agent("dep1")

        assert result is agent
        assert [c.args[0] for c in mock_agent_cls.call_args_list] == [
            "http://stale",
            "http://fresh",
        ]
        assert api_key_client._ensure_inference_binding.call_args_list[1].kwargs[
            "refresh"
        ]

    @patch("agilerl.arena.client.Agent")
    def test_the_refetch_carries_the_disambiguating_names(
        self, mock_agent_cls, api_key_client
    ):
        self._binding(api_key_client, "http://stale", "http://fresh")
        mock_agent_cls.side_effect = [
            ArenaInferenceError(status_code=404, detail=""),
            MagicMock(),
        ]

        api_key_client.open_inference_agent(
            "dep1", experiment_name="exp1", project_name="proj1"
        )

        retry = api_key_client._ensure_inference_binding.call_args_list[1].kwargs
        assert retry["experiment_name"] == "exp1"
        assert retry["project_name"] == "proj1"

    @patch("agilerl.arena.client.Agent")
    def test_an_unchanged_url_raises_rather_than_retrying(
        self, mock_agent_cls, api_key_client
    ):
        self._binding(api_key_client, "http://same", "http://same")
        original = ArenaInferenceError(status_code=404, detail="No details")
        mock_agent_cls.side_effect = original

        with pytest.raises(ArenaInferenceError) as exc_info:
            api_key_client.open_inference_agent("dep1")

        assert exc_info.value is original
        assert mock_agent_cls.call_count == 1

    @patch("agilerl.arena.client.Agent")
    def test_refresh_already_requested_does_not_retry(
        self, mock_agent_cls, api_key_client
    ):
        self._binding(api_key_client, "http://url")
        mock_agent_cls.side_effect = ArenaInferenceError(status_code=404, detail="")

        with pytest.raises(ArenaInferenceError):
            api_key_client.open_inference_agent("dep1", refresh=True)

        assert mock_agent_cls.call_count == 1
        assert api_key_client._ensure_inference_binding.call_count == 1

    @pytest.mark.parametrize("status", [500, 503, 0])
    @patch("agilerl.arena.client.Agent")
    def test_other_failures_are_not_a_moved_deployment(
        self, mock_agent_cls, api_key_client, status
    ):
        """A pod that is down answers on the right URL; refetching would not help."""
        self._binding(api_key_client, "http://url")
        mock_agent_cls.side_effect = ArenaInferenceError(
            status_code=status, detail="no healthy upstream"
        )

        with pytest.raises(ArenaInferenceError):
            api_key_client.open_inference_agent("dep1")

        assert mock_agent_cls.call_count == 1
        assert api_key_client._ensure_inference_binding.call_count == 1

    @patch("agilerl.arena.client.Agent")
    def test_an_auth_error_is_not_retried(self, mock_agent_cls, api_key_client):
        self._binding(api_key_client, "http://url")
        mock_agent_cls.side_effect = ArenaAuthError("nope")

        with pytest.raises(ArenaAuthError):
            api_key_client.open_inference_agent("dep1")

        assert mock_agent_cls.call_count == 1

    @patch("agilerl.arena.client.Agent")
    def test_the_move_is_logged(self, mock_agent_cls, api_key_client, caplog):
        self._binding(api_key_client, "http://stale", "http://fresh")
        mock_agent_cls.side_effect = [
            ArenaInferenceError(status_code=404, detail=""),
            MagicMock(),
        ]

        with caplog.at_level(logging.INFO, logger="agilerl.arena.client"):
            api_key_client.open_inference_agent("dep1")

        assert "moved to http://fresh" in caplog.text

    def test_a_real_agent_recovers_from_the_stale_url(self, api_key_client):
        """End to end through a real Agent: the stale URL 404s, the fresh one serves."""
        seen: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request.url.host)
            if request.url.host == "stale":
                return httpx.Response(404, text="")
            return httpx.Response(
                200, json={"success": True, "agent": {"algo": "GRPO", "llm": True}}
            )

        real_client = httpx.Client

        def fake_client(**kwargs):
            return real_client(transport=httpx.MockTransport(handler), **kwargs)

        self._binding(api_key_client, "http://stale", "http://fresh")
        with patch("agilerl.arena.inference.agent.httpx.Client", fake_client):
            agent = api_key_client.open_inference_agent("dep1")

        assert seen == ["stale", "fresh"]
        assert agent.metadata is not None
        assert agent.metadata.agent.algo == "GRPO"
        assert repr(agent) == "<Agent endpoint='http://fresh'>"
