# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Tests for arena.auth — load_credentials and ArenaOAuth2."""

from __future__ import annotations

import json
import logging
import os
import stat
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from keycloak.exceptions import KeycloakError

from agilerl.auth import (
    ArenaOAuth2,
    load_credentials,
    load_credentials_payload,
    oauth_access_token_expires_at,
    oauth_token_claims,
    stored_issuer,
)
from agilerl.exceptions import ArenaAuthError, ArenaTimeoutError


def _chmod_mode_bits_visible(base_dir: Path) -> bool:
    """True when chmod(0o700) is reflected in st_mode (not true on Windows / DrvFs)."""
    if os.name == "nt" or sys.platform == "win32":
        return False
    probe_dir = base_dir / ".chmod_mode_probe"
    probe_dir.mkdir(exist_ok=True)
    try:
        os.chmod(probe_dir, stat.S_IRWXU)
        return (probe_dir.stat().st_mode & 0o777) == stat.S_IRWXU
    except OSError:
        return False
    finally:
        probe_dir.rmdir()


# ---------------------------------------------------------------------------
# load_credentials
# ---------------------------------------------------------------------------
class TestLoadCredentials:
    def test_returns_none_when_file_missing(self, tmp_path):
        result = load_credentials(tmp_path / "nonexistent.json")
        assert result is None

    def test_returns_none_for_invalid_json(self, tmp_path):
        cred_file = tmp_path / "bad.json"
        cred_file.write_text("{invalid json", encoding="utf-8")
        assert load_credentials(cred_file) is None

    def test_returns_none_when_not_a_dict(self, tmp_path):
        cred_file = tmp_path / "list.json"
        cred_file.write_text('["not", "a", "dict"]', encoding="utf-8")
        assert load_credentials(cred_file) is None

    def test_returns_none_when_access_token_missing(self, tmp_path):
        cred_file = tmp_path / "no_token.json"
        cred_file.write_text('{"refresh_token": "rt"}', encoding="utf-8")
        assert load_credentials(cred_file) is None

    def test_returns_dict_on_valid_file(self, credentials_file):
        result = load_credentials(credentials_file)
        assert result == {"access_token": "at", "refresh_token": "rt"}

    def test_handles_os_error(self, credentials_file):
        with patch.object(Path, "read_text", side_effect=OSError("read failed")):
            result = load_credentials(credentials_file)
        assert result is None


# ---------------------------------------------------------------------------
# JWT helpers (OAuth access tokens from Keycloak)
# ---------------------------------------------------------------------------
def _jwt_with_exp(exp: int) -> str:
    return _jwt_payload({"exp": exp})


def _jwt_payload(payload: dict) -> str:
    import base64

    def seg(obj: dict) -> str:
        raw = json.dumps(obj, separators=(",", ":")).encode()
        return base64.urlsafe_b64encode(raw).decode().rstrip("=")

    return f"{seg({'alg': 'none'})}.{seg(payload)}.x"


class TestOAuthAccessTokenJwt:
    def test_expires_at_none_for_opaque(self):
        assert oauth_access_token_expires_at("opaque-string") is None

    def test_expires_at_decodes_exp(self):
        assert oauth_access_token_expires_at(_jwt_with_exp(1700000000)) == 1700000000.0

    def test_expires_at_none_for_jwt_without_exp(self):
        assert oauth_access_token_expires_at(_jwt_payload({})) is None

    def test_expires_at_none_for_malformed_jwt(self):
        assert oauth_access_token_expires_at("a.b") is None

    @pytest.mark.parametrize("exp", ["soon", [1700000000]])
    def test_expires_at_none_for_non_numeric_exp(self, exp):
        assert oauth_access_token_expires_at(_jwt_payload({"exp": exp})) is None


class TestLoadCredentialsPayload:
    def test_returns_empty_when_missing(self, tmp_path):
        assert load_credentials_payload(tmp_path / "missing.json") == {}

    def test_returns_empty_for_invalid_json(self, tmp_path):
        path = tmp_path / "creds.json"
        path.write_text("{bad", encoding="utf-8")
        assert load_credentials_payload(path) == {}

    def test_returns_dict_when_valid(self, credentials_file):
        assert load_credentials_payload(credentials_file) == {
            "access_token": "at",
            "refresh_token": "rt",
        }


# ---------------------------------------------------------------------------
# ArenaOAuth2
# ---------------------------------------------------------------------------
class TestArenaOAuth2Configure:
    def setup_method(self):
        self._orig = {
            "KEYCLOAK_URL": ArenaOAuth2.KEYCLOAK_URL,
            "REALM": ArenaOAuth2.REALM,
            "CLIENT_ID": ArenaOAuth2.CLIENT_ID,
            "CREDENTIALS_DIR": ArenaOAuth2.CREDENTIALS_DIR,
            "CREDENTIALS_FILE": ArenaOAuth2.CREDENTIALS_FILE,
        }

    def teardown_method(self):
        ArenaOAuth2.KEYCLOAK_URL = self._orig["KEYCLOAK_URL"]
        ArenaOAuth2.REALM = self._orig["REALM"]
        ArenaOAuth2.CLIENT_ID = self._orig["CLIENT_ID"]
        ArenaOAuth2.CREDENTIALS_DIR = self._orig["CREDENTIALS_DIR"]
        ArenaOAuth2.CREDENTIALS_FILE = self._orig["CREDENTIALS_FILE"]

    def test_configure_overrides_attrs(self):
        result = ArenaOAuth2.configure(
            keycloak_url="http://localhost:8080",
            realm="test-realm",
            client_id="test-client",
            credentials_dir=Path("/tmp/test-arena"),
            credentials_file=Path("/tmp/test-arena/creds.json"),
        )
        assert result is ArenaOAuth2
        assert ArenaOAuth2.KEYCLOAK_URL == "http://localhost:8080"
        assert ArenaOAuth2.REALM == "test-realm"
        assert ArenaOAuth2.CLIENT_ID == "test-client"
        assert Path("/tmp/test-arena") == ArenaOAuth2.CREDENTIALS_DIR
        assert Path("/tmp/test-arena/creds.json") == ArenaOAuth2.CREDENTIALS_FILE

    def test_configure_skips_none_values(self):
        orig_url = ArenaOAuth2.KEYCLOAK_URL
        ArenaOAuth2.configure(keycloak_url=None)
        assert orig_url == ArenaOAuth2.KEYCLOAK_URL

    @patch("agilerl.auth.KeycloakOpenID")
    def test_configured_keycloak_url_beats_env(self, mock_kc_cls, monkeypatch):
        """``arena --keycloak-url`` must win over an exported ARENA_KEYCLOAK_URL."""
        monkeypatch.setenv("ARENA_KEYCLOAK_URL", "https://auth.example.com")
        ArenaOAuth2.configure(keycloak_url="http://localhost:8023")
        ArenaOAuth2()
        assert mock_kc_cls.call_args.kwargs["server_url"] == "http://localhost:8023"

    @patch("agilerl.auth.KeycloakOpenID")
    def test_env_keycloak_url_beats_default(self, mock_kc_cls, monkeypatch):
        monkeypatch.setenv("ARENA_KEYCLOAK_URL", "https://auth.example.com")
        ArenaOAuth2()
        assert mock_kc_cls.call_args.kwargs["server_url"] == "https://auth.example.com"


class TestArenaOAuth2Init:
    @patch("agilerl.auth.KeycloakOpenID")
    def test_creates_keycloak_client(self, mock_kc_cls):
        auth = ArenaOAuth2()
        mock_kc_cls.assert_called_once_with(
            server_url=ArenaOAuth2.KEYCLOAK_URL,
            realm_name=ArenaOAuth2.REALM,
            client_id=ArenaOAuth2.CLIENT_ID,
        )
        assert auth.kc is mock_kc_cls.return_value


class TestArenaOAuth2IssuerAttribute:
    def test_built_from_keycloak_url_and_realm(self):
        with (
            patch("agilerl.auth.KeycloakOpenID"),
            patch.dict(os.environ, {"ARENA_KEYCLOAK_URL": "https://kc.example.com/"}),
        ):
            auth = ArenaOAuth2()

        assert auth.issuer == f"https://kc.example.com/realms/{ArenaOAuth2.REALM}"


class TestOauthTokenClaims:
    def test_decodes_payload(self):
        assert oauth_token_claims(_jwt_payload({"iss": "https://kc/realms/a"})) == {
            "iss": "https://kc/realms/a"
        }

    def test_none_for_opaque_token(self):
        assert oauth_token_claims("opaque") is None

    def test_none_for_missing_token(self):
        assert oauth_token_claims(None) is None


class TestStoredIssuer:
    def test_prefers_recorded_issuer(self):
        creds = {
            "issuer": "https://dev/realms/arena",
            "access_token": _jwt_payload({"iss": "https://prod/realms/arena"}),
        }

        assert stored_issuer(creds) == "https://dev/realms/arena"

    def test_falls_back_to_token_iss(self):
        creds = {"access_token": _jwt_payload({"iss": "https://prod/realms/arena"})}

        assert stored_issuer(creds) == "https://prod/realms/arena"

    def test_none_when_unknown(self):
        assert stored_issuer({"access_token": "opaque"}) is None


class TestArenaOAuth2EndSession:
    @patch("agilerl.auth.KeycloakOpenID")
    def test_same_issuer_logs_out_on_own_keycloak(self, mock_kc_cls):
        auth = ArenaOAuth2()

        auth.end_session("rt", auth.issuer)

        auth.kc.logout.assert_called_once_with("rt")
        assert mock_kc_cls.call_count == 1

    @patch("agilerl.auth.KeycloakOpenID")
    def test_other_issuer_logs_out_on_that_keycloak(self, mock_kc_cls):
        auth = ArenaOAuth2()
        other_kc = MagicMock()
        mock_kc_cls.side_effect = [other_kc]

        auth.end_session("rt", "https://dev-auth.example.com/realms/arena-dev")

        mock_kc_cls.assert_called_with(
            server_url="https://dev-auth.example.com",
            realm_name="arena-dev",
            client_id=ArenaOAuth2.CLIENT_ID,
        )
        other_kc.logout.assert_called_once_with("rt")
        auth.kc.logout.assert_not_called()

    @patch("agilerl.auth.KeycloakOpenID")
    def test_failure_warns_and_does_not_raise(self, mock_kc_cls, caplog):
        auth = ArenaOAuth2()
        auth.kc.logout.side_effect = KeycloakError("unreachable")

        with caplog.at_level(logging.WARNING, logger="agilerl.auth"):
            auth.end_session("rt")

        assert "Could not end the previous Arena session" in caplog.text


class TestWriteCredentials:
    def test_creates_directory_and_writes(self, tmp_path):
        cred_dir = tmp_path / "arena"
        cred_file = cred_dir / "credentials.json"

        orig_dir = ArenaOAuth2.CREDENTIALS_DIR
        orig_file = ArenaOAuth2.CREDENTIALS_FILE
        try:
            ArenaOAuth2.CREDENTIALS_DIR = cred_dir
            ArenaOAuth2.CREDENTIALS_FILE = cred_file

            data = {"access_token": "at123", "refresh_token": "rt456"}
            if _chmod_mode_bits_visible(tmp_path):
                ArenaOAuth2._write_credentials(data)
                assert (cred_dir.stat().st_mode & 0o777) == stat.S_IRWXU
                assert (cred_file.stat().st_mode & 0o777) == (
                    stat.S_IRUSR | stat.S_IWUSR
                )
            else:
                # Native Windows, WSL on /mnt/c, etc.: verify chmod calls only.
                with patch("agilerl.auth.os.chmod") as mock_chmod:
                    ArenaOAuth2._write_credentials(data)
                mock_chmod.assert_any_call(cred_dir, stat.S_IRWXU)
                mock_chmod.assert_any_call(cred_file, stat.S_IRUSR | stat.S_IWUSR)

            assert cred_dir.is_dir()
            assert cred_file.is_file()
            written = json.loads(cred_file.read_text(encoding="utf-8"))
            assert written == data
        finally:
            ArenaOAuth2.CREDENTIALS_DIR = orig_dir
            ArenaOAuth2.CREDENTIALS_FILE = orig_file


class TestExtractError:
    def test_parses_json_response_body_bytes(self):
        exc = KeycloakError()
        exc.response_body = json.dumps({"error": "authorization_pending"}).encode()
        assert ArenaOAuth2._extract_error(exc) == "authorization_pending"

    def test_parses_json_response_body_string(self):
        exc = KeycloakError()
        exc.response_body = json.dumps({"error": "slow_down"})
        assert ArenaOAuth2._extract_error(exc) == "slow_down"

    def test_falls_back_to_string_matching(self):
        exc = KeycloakError("something authorization_pending something")
        assert ArenaOAuth2._extract_error(exc) == "authorization_pending"

    def test_falls_back_to_expired_token(self):
        exc = KeycloakError("expired_token in message")
        assert ArenaOAuth2._extract_error(exc) == "expired_token"

    def test_returns_empty_on_unknown_error(self):
        exc = KeycloakError("totally unknown error")
        assert ArenaOAuth2._extract_error(exc) == ""

    def test_handles_invalid_json_body(self):
        exc = KeycloakError()
        exc.response_body = b"not json"
        result = ArenaOAuth2._extract_error(exc)
        assert isinstance(result, str)


class TestDeviceLogin:
    @patch("agilerl.auth.webbrowser")
    @patch("agilerl.auth.time")
    @patch("agilerl.auth.KeycloakOpenID")
    def test_happy_path(self, mock_kc_cls, mock_time, mock_wb):
        mock_kc = mock_kc_cls.return_value
        mock_kc.device.return_value = {
            "device_code": "dc123",
            "verification_uri_complete": "https://auth.example.com/verify?code=abc",
            "interval": 5,
        }
        tokens = {"access_token": "at", "refresh_token": "rt"}
        mock_kc.token.return_value = tokens
        mock_time.monotonic.side_effect = [0, 1]
        mock_wb.open.return_value = True

        with patch.object(ArenaOAuth2, "_write_credentials") as mock_write:
            auth = ArenaOAuth2()
            result = auth.device_login(timeout=300)

        assert result == tokens
        mock_write.assert_called_once()
        written = mock_write.call_args[0][0]
        assert written["access_token"] == tokens["access_token"]
        assert written["refresh_token"] == tokens["refresh_token"]
        assert written["issuer"] == auth.issuer
        mock_wb.open.assert_called_once()
        mock_kc.token.assert_called_once()
        call_kw = mock_kc.token.call_args.kwargs
        assert call_kw["grant_type"] == ("urn:ietf:params:oauth:grant-type:device_code")
        assert call_kw["device_code"] == "dc123"

    @patch("agilerl.auth.webbrowser")
    @patch("agilerl.auth.time")
    @patch("agilerl.auth.KeycloakOpenID")
    def test_webbrowser_open_failure_logs_warning(
        self, mock_kc_cls, mock_time, mock_wb
    ):
        mock_kc = mock_kc_cls.return_value
        mock_kc.device.return_value = {
            "device_code": "dc",
            "verification_uri_complete": "https://example.com",
            "interval": 1,
        }
        mock_kc.token.return_value = {"access_token": "at"}
        mock_time.monotonic.side_effect = [0, 1]
        mock_wb.open.return_value = False

        with (
            patch.object(ArenaOAuth2, "_write_credentials"),
            patch("agilerl.auth.logger") as mock_logger,
        ):
            ArenaOAuth2().device_login(timeout=30)

        mock_logger.warning.assert_called_once()
        assert "Could not open browser" in mock_logger.warning.call_args[0][0]

    @patch("agilerl.auth.webbrowser")
    @patch("agilerl.auth.time")
    @patch("agilerl.auth.KeycloakOpenID")
    def test_authorization_pending_keeps_polling(self, mock_kc_cls, mock_time, mock_wb):
        mock_kc = mock_kc_cls.return_value
        mock_kc.device.return_value = {
            "device_code": "dc",
            "verification_uri": "https://example.com",
            "interval": 1,
        }
        pending_exc = KeycloakError("authorization_pending")
        tokens = {"access_token": "at"}
        mock_kc.token.side_effect = [pending_exc, pending_exc, tokens]
        mock_time.monotonic.side_effect = [0, 1, 2, 3]
        mock_wb.open.return_value = True

        with patch.object(ArenaOAuth2, "_write_credentials"):
            auth = ArenaOAuth2()
            result = auth.device_login(timeout=300)

        assert result == tokens
        assert mock_kc.token.call_count == 3

    @patch("agilerl.auth.webbrowser")
    @patch("agilerl.auth.time")
    @patch("agilerl.auth.KeycloakOpenID")
    def test_slow_down_increases_interval(self, mock_kc_cls, mock_time, mock_wb):
        mock_kc = mock_kc_cls.return_value
        mock_kc.device.return_value = {
            "device_code": "dc",
            "verification_uri": "https://example.com",
            "interval": 2,
        }
        slow_exc = KeycloakError()
        slow_exc.response_body = json.dumps({"error": "slow_down"}).encode()
        tokens = {"access_token": "at"}
        mock_kc.token.side_effect = [slow_exc, tokens]
        mock_time.monotonic.side_effect = [0, 1, 2]
        mock_wb.open.return_value = True

        with patch.object(ArenaOAuth2, "_write_credentials"):
            auth = ArenaOAuth2()
            result = auth.device_login(timeout=300)

        assert result == tokens
        # After slow_down, sleep should have been called with increased interval (2 + 5 = 7)
        sleep_calls = [c.args[0] for c in mock_time.sleep.call_args_list]
        assert sleep_calls[-1] == 7

    @patch("agilerl.auth.webbrowser")
    @patch("agilerl.auth.time")
    @patch("agilerl.auth.KeycloakOpenID")
    def test_expired_token_raises_timeout(self, mock_kc_cls, mock_time, mock_wb):
        mock_kc = mock_kc_cls.return_value
        mock_kc.device.return_value = {
            "device_code": "dc",
            "verification_uri": "https://example.com",
            "interval": 1,
        }
        exc = KeycloakError()
        exc.response_body = json.dumps({"error": "expired_token"}).encode()
        mock_kc.token.side_effect = exc
        mock_time.monotonic.side_effect = [0, 1]
        mock_wb.open.return_value = True

        auth = ArenaOAuth2()
        with pytest.raises(ArenaTimeoutError, match="expired"):
            auth.device_login(timeout=300)

    @patch("agilerl.auth.KeycloakOpenID")
    def test_keycloak_error_on_device_raises_auth_error(self, mock_kc_cls):
        mock_kc = mock_kc_cls.return_value
        mock_kc.device.side_effect = KeycloakError("connection refused")

        auth = ArenaOAuth2()
        with pytest.raises(ArenaAuthError, match="Failed to initiate"):
            auth.device_login()

    @patch("agilerl.auth.webbrowser")
    @patch("agilerl.auth.time")
    @patch("agilerl.auth.KeycloakOpenID")
    def test_timeout_raises_timeout_error(self, mock_kc_cls, mock_time, mock_wb):
        mock_kc = mock_kc_cls.return_value
        mock_kc.device.return_value = {
            "device_code": "dc",
            "verification_uri": "https://example.com",
            "interval": 1,
        }
        pending_exc = KeycloakError("authorization_pending")
        mock_kc.token.side_effect = pending_exc
        # First call: 0, then always past deadline
        mock_time.monotonic.side_effect = [0, 999]
        mock_wb.open.return_value = True

        auth = ArenaOAuth2()
        with pytest.raises(ArenaTimeoutError, match="timed out"):
            auth.device_login(timeout=10)

    @patch("agilerl.auth.webbrowser")
    @patch("agilerl.auth.time")
    @patch("agilerl.auth.KeycloakOpenID")
    def test_unknown_keycloak_error_during_poll_raises_auth_error(
        self, mock_kc_cls, mock_time, mock_wb
    ):
        mock_kc = mock_kc_cls.return_value
        mock_kc.device.return_value = {
            "device_code": "dc",
            "verification_uri": "https://example.com",
            "interval": 1,
        }
        mock_kc.token.side_effect = KeycloakError("unknown keycloak failure")
        mock_time.monotonic.side_effect = [0, 1]
        mock_wb.open.return_value = True

        auth = ArenaOAuth2()
        with pytest.raises(ArenaAuthError, match="Device authorization failed"):
            auth.device_login(timeout=300)


class TestRefreshAccessToken:
    @patch("agilerl.auth.KeycloakOpenID")
    def test_happy_path(self, mock_kc_cls, tmp_path):
        mock_kc = mock_kc_cls.return_value
        new_tokens = {"access_token": "new_at", "refresh_token": "new_rt"}
        mock_kc.refresh_token.return_value = new_tokens

        cred_file = tmp_path / "creds.json"
        cred_file.write_text(json.dumps({"access_token": "old_at"}), encoding="utf-8")

        orig_file = ArenaOAuth2.CREDENTIALS_FILE
        orig_dir = ArenaOAuth2.CREDENTIALS_DIR
        try:
            ArenaOAuth2.CREDENTIALS_FILE = cred_file
            ArenaOAuth2.CREDENTIALS_DIR = tmp_path

            with patch.object(ArenaOAuth2, "_write_credentials") as mock_write:
                auth = ArenaOAuth2()
                result = auth.refresh_access_token("old_rt")

            assert result == new_tokens
            written_data = mock_write.call_args[0][0]
            assert written_data["access_token"] == "new_at"
        finally:
            ArenaOAuth2.CREDENTIALS_FILE = orig_file
            ArenaOAuth2.CREDENTIALS_DIR = orig_dir

    @patch("agilerl.auth.KeycloakOpenID")
    def test_keycloak_error_raises_auth_error(self, mock_kc_cls):
        mock_kc = mock_kc_cls.return_value
        mock_kc.refresh_token.side_effect = KeycloakError("invalid refresh token")

        auth = ArenaOAuth2()
        with pytest.raises(ArenaAuthError, match="Token refresh failed"):
            auth.refresh_access_token("bad_rt")


class TestRevoke:
    @patch("agilerl.auth.KeycloakOpenID")
    def test_calls_logout_and_removes_file(self, mock_kc_cls, tmp_path):
        mock_kc = mock_kc_cls.return_value
        cred_file = tmp_path / "creds.json"
        cred_file.write_text("{}", encoding="utf-8")

        orig_file = ArenaOAuth2.CREDENTIALS_FILE
        try:
            ArenaOAuth2.CREDENTIALS_FILE = cred_file
            auth = ArenaOAuth2()
            auth.revoke("rt123")

            mock_kc.logout.assert_called_once_with("rt123")
            assert not cred_file.exists()
        finally:
            ArenaOAuth2.CREDENTIALS_FILE = orig_file

    @patch("agilerl.auth.KeycloakOpenID")
    def test_unlink_oserror_is_silent(self, mock_kc_cls, tmp_path):
        mock_kc = mock_kc_cls.return_value
        cred_file = tmp_path / "creds.json"
        cred_file.write_text("{}", encoding="utf-8")

        orig_file = ArenaOAuth2.CREDENTIALS_FILE
        try:
            ArenaOAuth2.CREDENTIALS_FILE = cred_file
            with patch.object(Path, "unlink", side_effect=OSError("permission denied")):
                ArenaOAuth2().revoke("rt")
            mock_kc.logout.assert_called_once()
        finally:
            ArenaOAuth2.CREDENTIALS_FILE = orig_file

    @patch("agilerl.auth.KeycloakOpenID")
    def test_keycloak_logout_failure_is_silent(self, mock_kc_cls, tmp_path):
        mock_kc = mock_kc_cls.return_value
        mock_kc.logout.side_effect = KeycloakError("already expired")
        cred_file = tmp_path / "creds.json"
        cred_file.write_text("{}", encoding="utf-8")

        orig_file = ArenaOAuth2.CREDENTIALS_FILE
        try:
            ArenaOAuth2.CREDENTIALS_FILE = cred_file
            auth = ArenaOAuth2()
            auth.revoke("rt123")  # Should not raise
            assert not cred_file.exists()
        finally:
            ArenaOAuth2.CREDENTIALS_FILE = orig_file
