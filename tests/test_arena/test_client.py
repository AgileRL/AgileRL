"""Tests for agilerl.arena.client — _TokenStore and ArenaClient."""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import httpx
import pytest

from agilerl.arena.client import ArenaClient, _TokenStore
from agilerl.arena.exceptions import (
    ArenaAPIError,
    ArenaAuthError,
    ArenaValidationError,
)
from agilerl.models import ArenaCluster, ArenaResource, JobStatus, TrainingManifest


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
# _check_validation_result
# ---------------------------------------------------------------------------
class TestCheckValidationResult:
    def test_no_errors_passes(self):
        ArenaClient._check_validation_result({"errors": None})
        ArenaClient._check_validation_result({"errors": []})
        ArenaClient._check_validation_result({})

    def test_errors_raises_validation_error(self):
        with pytest.raises(ArenaValidationError):
            ArenaClient._check_validation_result({"errors": [{"msg": "bad field"}]})


# ---------------------------------------------------------------------------
# Environment methods
# ---------------------------------------------------------------------------
class TestRegisterEnvironment:
    def test_posts_with_archive(self, api_key_client, tmp_path):
        src = tmp_path / "env_src"
        src.mkdir()
        (src / "main.py").write_text("print('hello')")

        api_key_client._request = MagicMock(return_value={"status": "ok"})
        result = api_key_client._register_environment("TestEnv", source=src)

        assert result == {"status": "ok"}
        call_args = api_key_client._request.call_args
        assert call_args[0] == ("POST", "api/custom-gym-env-impls/create")
        assert "files" in call_args[1]
        assert "data" in call_args[1]
        assert call_args[1]["data"]["name"] == "TestEnv"

    def test_missing_source_raises(self, api_key_client):
        with pytest.raises(FileNotFoundError, match="not found"):
            api_key_client._register_environment("TestEnv", source="/nonexistent/path")


class TestValidate:
    def test_returns_resp_without_operation_id(self, api_key_client):
        api_key_client._request = MagicMock(
            return_value={"valid": True, "errors": None}
        )
        result = api_key_client._validate("CartPole-v1")
        assert result["valid"] is True

    def test_streams_logs_when_operation_id_present(self, api_key_client):
        api_key_client._request = MagicMock(
            return_value={"operation_id": "op123", "errors": None}
        )
        api_key_client.stream_logs = MagicMock(return_value={"result": "done"})

        result = api_key_client._validate("CartPole-v1")
        api_key_client.stream_logs.assert_called_once_with("op123")
        assert result == {"result": "done"}


class TestEnvironmentListMethods:
    def test_list_environments(self, api_key_client):
        api_key_client._request = MagicMock(return_value=None)
        api_key_client.list_environments()
        api_key_client._request.assert_called_once_with(
            "GET", "api/v1/custom-gym-env-impls/list"
        )

    def test_list_entrypoints(self, api_key_client):
        api_key_client._request = MagicMock(
            return_value={"entrypoints": ["main:MyEnv", "alt:AltEnv"]}
        )
        result = api_key_client.list_entrypoints("MyEnv", version="v2")
        assert result == ["main:MyEnv", "alt:AltEnv"]

    def test_is_registered_environment(self, api_key_client):
        api_key_client._request = MagicMock(return_value={"is_registered": True})
        assert api_key_client.is_registered_environment("CartPole-v1") is True


class TestValidateEnvironment:
    def test_source_not_registered_calls_register(self, api_key_client, tmp_path):
        src = tmp_path / "env"
        src.mkdir()
        (src / "env.py").write_text("pass")

        api_key_client.is_registered_environment = MagicMock(return_value=False)
        api_key_client._register_environment = MagicMock(
            return_value={"registered": True}
        )
        result = api_key_client.validate_environment("MyEnv", source=src)
        api_key_client._register_environment.assert_called_once()
        assert result == {"registered": True}

    def test_source_already_registered_validates(self, api_key_client, tmp_path):
        src = tmp_path / "env"
        src.mkdir()
        (src / "env.py").write_text("pass")

        api_key_client.is_registered_environment = MagicMock(return_value=True)
        api_key_client._validate = MagicMock(return_value={"valid": True})
        result = api_key_client.validate_environment("MyEnv", source=src)
        api_key_client._validate.assert_called_once()

    def test_no_source_validates_directly(self, api_key_client):
        api_key_client._validate = MagicMock(return_value={"valid": True})
        result = api_key_client.validate_environment("MyEnv")
        api_key_client._validate.assert_called_once()


# ---------------------------------------------------------------------------
# Job methods
# ---------------------------------------------------------------------------
class TestSubmitJob:
    def test_submits_manifest(self, api_key_client):
        manifest = MagicMock(spec=TrainingManifest)
        manifest.model_dump.return_value = {"algorithm": {}, "training": {}}

        api_key_client._request = MagicMock(
            return_value={"job_id": "j1", "status": "pending"}
        )
        result = api_key_client.submit_job(manifest)

        api_key_client._request.assert_called_once_with(
            "POST",
            "api/v1/jobs/submit",
            json={"algorithm": {}, "training": {}},
        )
        assert result["job_id"] == "j1"

    def test_submits_with_cluster(self, api_key_client):
        manifest = MagicMock(spec=TrainingManifest)
        manifest.model_dump.return_value = {"algorithm": {}}
        cluster = ArenaCluster(resource=ArenaResource.MEDIUM_ACCELERATED, num_nodes=1)

        api_key_client._request = MagicMock(return_value={"job_id": "j2"})
        api_key_client.submit_job(manifest, cluster=cluster)

        call_kwargs = api_key_client._request.call_args[1]
        assert "cluster" in call_kwargs["json"]


class TestGetJobStatus:
    def test_returns_job_status_enum(self, api_key_client):
        api_key_client._request = MagicMock(return_value="running")
        status = api_key_client.get_job_status("j1")
        assert status == JobStatus.RUNNING
        assert isinstance(status, JobStatus)


class TestStopJob:
    def test_posts_stop(self, api_key_client):
        api_key_client._request = MagicMock(return_value=None)
        api_key_client.stop_job("j1")
        api_key_client._request.assert_called_once_with(
            "POST", "api/v1/jobs/stop", params={"job_id": "j1"}
        )


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------
class TestIterEvents:
    def test_returns_event_stream(self, api_key_client):
        stream = api_key_client.iter_events("op123", max_retries=3)
        from agilerl.arena.logs import EventStream

        assert isinstance(stream, EventStream)


class TestStreamLogs:
    def test_renders_events_and_returns_result(self, api_key_client):
        from agilerl.arena.logs import LogEvent

        events = [
            LogEvent(
                id="1",
                type="log",
                level="INFO",
                message="Starting",
                timestamp="2024-01-01T00:00:00",
            ),
            LogEvent(
                id="2",
                type="complete",
                level="INFO",
                message="Done",
                timestamp="2024-01-01T00:01:00",
                metadata={"result": {"fitness": 0.95}},
            ),
        ]

        with patch.object(api_key_client, "iter_events") as mock_iter:
            mock_stream = MagicMock()
            mock_stream.__enter__ = MagicMock(return_value=iter(events))
            mock_stream.__exit__ = MagicMock(return_value=False)
            mock_iter.return_value = mock_stream

            result = api_key_client.stream_logs("op123")

        assert result == {"fitness": 0.95}


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
    def test_forwards_rollouts_and_max_steps(self, api_key_client):
        api_key_client._validate = MagicMock(return_value={"valid": True})
        api_key_client.validate_environment(
            "MyEnv",
            rollouts=True,
            max_steps=500,
        )
        call_kwargs = api_key_client._validate.call_args[1]
        assert call_kwargs["rollouts"] == "true"
        assert call_kwargs["max_steps"] == "500"

    def test_forwards_version_and_entrypoint(self, api_key_client):
        api_key_client._validate = MagicMock(return_value={"valid": True})
        api_key_client.validate_environment(
            "MyEnv",
            version="v2",
            entrypoint="my_env:make",
        )
        call_kwargs = api_key_client._validate.call_args[1]
        assert call_kwargs["version"] == "v2"
        assert call_kwargs["entrypoint"] == "my_env:make"

    def test_register_forwards_multi_agent_and_description(
        self, api_key_client, tmp_path
    ):
        src = tmp_path / "env"
        src.mkdir()
        (src / "env.py").write_text("pass")

        api_key_client.is_registered_environment = MagicMock(return_value=False)
        api_key_client._register_environment = MagicMock(
            return_value={"registered": True}
        )
        api_key_client.validate_environment(
            "MyEnv",
            source=src,
            multi_agent=True,
            description="A test env",
        )
        call_kwargs = api_key_client._register_environment.call_args[1]
        assert call_kwargs["multi_agent"] is True
        assert call_kwargs["description"] == "A test env"


class TestCliV1EnvironmentAndExperimentEndpoints:
    def test_list_custom_environments_uses_cli_v1(self, api_key_client):
        api_key_client._request = MagicMock(
            return_value={"ok": True, "data": {"MyEnv": {"v1": {"validated": True}}}}
        )
        result = api_key_client.list_custom_environments()
        api_key_client._request.assert_called_once_with(
            "GET", "/api/cli/v1/environments"
        )
        assert "MyEnv" in result

    def test_custom_environment_exists_unwraps_standard_payload(self, api_key_client):
        api_key_client._request = MagicMock(
            return_value={"ok": True, "data": {"exists": True}}
        )
        exists = api_key_client.custom_environment_exists("MyEnv", "v1")
        assert exists is True
        api_key_client._request.assert_called_once_with(
            "GET",
            "/api/cli/v1/environments/exists",
            params={"name": "MyEnv", "version": "v1"},
        )

    def test_list_custom_environment_entrypoints_unwraps_standard_payload(
        self, api_key_client
    ):
        api_key_client._request = MagicMock(
            return_value={"ok": True, "data": ["main:Env", "alt:Env"]}
        )
        entrypoints = api_key_client.list_custom_environment_entrypoints("MyEnv", "v1")
        assert entrypoints == ["main:Env", "alt:Env"]
        api_key_client._request.assert_called_once_with(
            "GET",
            "/api/cli/v1/environments/entrypoints",
            params={"name": "MyEnv", "version": "v1"},
        )

    def test_validate_custom_environment_uses_cli_v1_path(self, api_key_client):
        api_key_client._request = MagicMock(return_value={"ok": True})
        api_key_client.validate_custom_environment(
            name="MyEnv", version="v1", stream=False
        )
        api_key_client._request.assert_called_once_with(
            "POST",
            "/api/cli/v1/environments/validate",
            json={"name": "MyEnv", "version": "v1", "do_rollouts": True},
        )

    def test_profile_custom_environment_uses_cli_v1_path(self, api_key_client):
        api_key_client._request = MagicMock(return_value={"ok": True})
        api_key_client.profile_custom_environment(
            name="MyEnv", version="v1", multi_agent=False, stream=False
        )
        api_key_client._request.assert_called_once_with(
            "POST",
            "/api/cli/v1/environments/profile",
            json={"name": "MyEnv", "version": "v1", "multi_agent": False},
        )

    def test_delete_custom_environment_uses_cli_v1_path_and_unwraps(
        self, api_key_client
    ):
        api_key_client._request = MagicMock(return_value={"ok": True, "data": None})
        result = api_key_client.delete_custom_environment(name="MyEnv", version="v1")
        assert result is None
        api_key_client._request.assert_called_once_with(
            "DELETE",
            "/api/cli/v1/environments/delete",
            json={"name": "MyEnv", "version": "v1"},
        )

    def test_submit_experiment_job_uses_cli_v1_path(self, api_key_client):
        api_key_client._request = MagicMock(return_value={"ok": True})
        api_key_client.submit_experiment_job(
            manifest={"algorithm": {}},
            custom_gym_env_impl_id=1,
            stream=False,
        )
        api_key_client._request.assert_called_once_with(
            "POST",
            "/api/cli/v1/experiments/jobs/submit",
            json={"manifest": {"algorithm": {}}, "custom_gym_env_impl_id": 1},
            timeout=api_key_client._upload_timeout,
        )

    def test_validate_job_run_spec_uses_cli_v1_and_handles_null_data(
        self, api_key_client
    ):
        api_key_client._request = MagicMock(return_value={"ok": True, "data": None})
        result = api_key_client.validate_job_run_spec({"algorithm": {"name": "DQN"}})
        assert result == {"valid": True}
        api_key_client._request.assert_called_once_with(
            "POST",
            "/api/cli/v1/experiments/validate-run-spec",
            json={"algorithm": {"name": "DQN"}},
        )
