"""Tests for agilerl.arena.client — _TokenStore and ArenaClient."""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import httpx
import pytest

from agilerl.arena.client import ArenaClient, _TokenStore, prepare_env_upload
from agilerl.arena.exceptions import (
    ArenaAPIError,
    ArenaAuthError,
    ArenaTrainingError,
    ArenaValidationError,
)
from agilerl.arena.output import StreamRichRenderer
from agilerl.arena.stream import NDJsonStream, StreamEvent


def _mock_ndjson_stream(result: dict | None = None) -> MagicMock:
    """Create a mock NDJsonStream with a preset collect() result."""
    mock = MagicMock(spec=NDJsonStream)
    mock.collect.return_value = result or {}
    mock.result = result
    mock.__enter__ = MagicMock(return_value=mock)
    mock.__exit__ = MagicMock(return_value=False)
    return mock


# ---------------------------------------------------------------------------
# _TokenStore
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Helpers — Create an ArenaClient bypassing real auth / HTTP
# ---------------------------------------------------------------------------
@pytest.fixture
def api_key_client():
    """ArenaClient with a static API key (no OAuth, no session restore)."""
    with patch("agilerl.arena.auth.KeycloakOpenID"):
        client = ArenaClient(api_key="test-key")
    return client


@pytest.fixture
def token_client():
    """ArenaClient with OAuth tokens pre-loaded (no API key)."""
    with patch("agilerl.arena.auth.KeycloakOpenID"):
        with patch.object(ArenaClient, "_try_restore_session"):
            client = ArenaClient()
    client._tokens.access_token = "tok_access"
    client._tokens.refresh_token = "tok_refresh"
    return client


@pytest.fixture
def unauthenticated_client():
    """ArenaClient with no credentials at all."""
    with patch("agilerl.arena.auth.KeycloakOpenID"):
        with patch.object(ArenaClient, "_try_restore_session"):
            client = ArenaClient()
    return client


# ---------------------------------------------------------------------------
# ArenaClient.__init__
# ---------------------------------------------------------------------------
class TestArenaClientInit:
    @patch("agilerl.arena.auth.KeycloakOpenID")
    def test_with_explicit_api_key(self, _kc):
        client = ArenaClient(api_key="my-key")
        assert client._api_key == "my-key"
        assert client.is_authenticated

    @patch("agilerl.arena.auth.KeycloakOpenID")
    def test_with_env_var(self, _kc):
        with patch.dict(os.environ, {"ARENA_API_KEY": "env-key"}):
            client = ArenaClient()
        assert client._api_key == "env-key"

    @patch("agilerl.arena.auth.KeycloakOpenID")
    def test_without_key_calls_restore_session(self, _kc):
        with patch.dict(os.environ, {}, clear=False):
            env = os.environ.copy()
            env.pop("ARENA_API_KEY", None)
            with patch.dict(os.environ, env, clear=True):
                with patch.object(ArenaClient, "_try_restore_session") as mock_restore:
                    ArenaClient()
                mock_restore.assert_called_once()

    @patch("agilerl.arena.auth.KeycloakOpenID")
    def test_http_client_config(self, _kc):
        client = ArenaClient(api_key="k", request_timeout=60)
        assert client._request_timeout == 60
        assert client._upload_timeout == 300  # default


# ---------------------------------------------------------------------------
# ArenaClient.configure
# ---------------------------------------------------------------------------
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
        assert ArenaClient.BASE_URL == orig


# ---------------------------------------------------------------------------
# login / logout
# ---------------------------------------------------------------------------
class TestArenaClientLogin:
    def test_login_stores_tokens(self, unauthenticated_client):
        client = unauthenticated_client
        tokens = {"access_token": "at", "refresh_token": "rt"}
        client._auth.device_login = MagicMock(return_value=tokens)

        client.login(timeout=60)
        assert client._tokens.access_token == "at"
        assert client._tokens.refresh_token == "rt"
        client._auth.device_login.assert_called_once_with(timeout=60)


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


# ---------------------------------------------------------------------------
# is_authenticated
# ---------------------------------------------------------------------------
class TestIsAuthenticated:
    def test_true_with_api_key(self, api_key_client):
        assert api_key_client.is_authenticated is True

    def test_true_with_access_token(self, token_client):
        assert token_client.is_authenticated is True

    def test_false_with_nothing(self, unauthenticated_client):
        assert unauthenticated_client.is_authenticated is False


# ---------------------------------------------------------------------------
# set_stream_handler
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# _auth_headers
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# _request
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Environment methods
# ---------------------------------------------------------------------------
class TestEnvironmentListMethods:
    def test_list_environments(self, api_key_client):
        api_key_client._request = MagicMock(
            return_value={"MyEnv": {"v1": {"validated": True}}}
        )
        result = api_key_client.list_environments()
        api_key_client._request.assert_called_once_with(
            "GET", "/api/cli/v1/environments", params={"name": None}
        )
        assert "MyEnv" in result

    def test_environment_exists(self, api_key_client):
        api_key_client._request = MagicMock(return_value={"exists": True})
        assert api_key_client.environment_exists("CartPole-v1", "v1") is True

    def test_list_environment_entrypoints(self, api_key_client):
        api_key_client._request = MagicMock(return_value=["main:MyEnv", "alt:AltEnv"])
        result = api_key_client.list_environment_entrypoints("MyEnv", version="v2")
        assert result == ["main:MyEnv", "alt:AltEnv"]


class TestValidateEnvironment:
    def test_no_source_collects_by_default(self, api_key_client):
        mock_stream = _mock_ndjson_stream({"valid": True})
        api_key_client._open_stream = MagicMock(return_value=mock_stream)
        result = api_key_client.validate_environment(name="MyEnv", version="v1")
        api_key_client._open_stream.assert_called_once_with(
            "POST",
            "/api/cli/v1/environments/validate",
            json={"name": "MyEnv", "version": "v1", "do_rollouts": True},
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
        assert isinstance(file_tuple[1], bytes)
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
        with pytest.raises(FileNotFoundError, match="not found"):
            api_key_client.validate_environment(
                name="MyEnv",
                source="/nonexistent/path.tar.gz",
                env_config="/nonexistent/config.yaml",
                requirements="/nonexistent/reqs.txt",
            )


# ---------------------------------------------------------------------------
# prepare_env_upload
# ---------------------------------------------------------------------------


class TestPrepareEnvUpload:
    def test_directory_is_compressed(self, tmp_path):
        env_dir = tmp_path / "my_env"
        env_dir.mkdir()
        (env_dir / "main.py").write_text("print('hello')")
        (env_dir / "utils.py").write_text("x = 1")

        name, data = prepare_env_upload(env_dir)
        assert name == "my_env.tar.gz"
        assert isinstance(data, bytes) and len(data) > 0

        import tarfile, io

        with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
            members = sorted(tar.getnames())
        assert members == ["main.py", "utils.py"]

    def test_directory_recurses_subdirs(self, tmp_path):
        env_dir = tmp_path / "env"
        (env_dir / "sub").mkdir(parents=True)
        (env_dir / "a.py").write_text("a")
        (env_dir / "sub" / "b.py").write_text("b")

        name, data = prepare_env_upload(env_dir)
        import tarfile, io

        with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
            members = sorted(tar.getnames())
        assert members == ["a.py", "sub/b.py"]

    def test_file_is_read_as_is(self, tmp_path):
        archive = tmp_path / "env.tar.gz"
        archive.write_bytes(b"fake-archive-content")

        name, data = prepare_env_upload(archive)
        assert name == "env.tar.gz"
        assert data == b"fake-archive-content"

    def test_bytes_passthrough(self):
        raw = b"raw-bytes"
        name, data = prepare_env_upload(raw)
        assert name == "environment.tar.gz"
        assert data is raw

    def test_missing_path_raises(self):
        with pytest.raises(FileNotFoundError, match="not found"):
            prepare_env_upload("/does/not/exist.tar.gz")


# ---------------------------------------------------------------------------
# Job methods
# ---------------------------------------------------------------------------


class TestStopJob:
    def test_posts_stop(self, api_key_client):
        api_key_client._request = MagicMock(return_value=None)
        api_key_client.stop_job("j1")
        api_key_client._request.assert_called_once_with(
            "POST", "api/v1/jobs/stop", params={"job_id": "j1"}
        )


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Context manager and repr
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
    def test_restores_from_credentials(self, _kc, tmp_path):
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
    def test_restores_access_token_only(self, _kc, tmp_path):
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


# ---------------------------------------------------------------------------
# ValidateEnvironment — parameter forwarding
# ---------------------------------------------------------------------------


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

    def test_delete_environment_uses_cli_v1_path(self, api_key_client):
        api_key_client._request = MagicMock(return_value=None)
        result = api_key_client.delete_environment(name="MyEnv", version="v1")
        assert result is None
        api_key_client._request.assert_called_once_with(
            "DELETE",
            "/api/cli/v1/environments/delete",
            json={"name": "MyEnv", "version": "v1"},
        )


# ---------------------------------------------------------------------------
# _open_stream wiring
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# _send with stream=True
# ---------------------------------------------------------------------------
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
