"""Tests for agilerl.arena.auth — load_credentials and ArenaOAuth2."""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from keycloak.exceptions import KeycloakError

from agilerl.arena.auth import ArenaOAuth2, load_credentials
from agilerl.arena.exceptions import ArenaAuthError, ArenaTimeoutError


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

    def test_returns_dict_on_valid_file(self, tmp_path):
        cred_file = tmp_path / "creds.json"
        data = {"access_token": "at", "refresh_token": "rt"}
        cred_file.write_text(json.dumps(data), encoding="utf-8")
        result = load_credentials(cred_file)
        assert result == data

    def test_handles_os_error(self, tmp_path):
        cred_file = tmp_path / "unreadable.json"
        cred_file.write_text('{"access_token": "at"}', encoding="utf-8")
        cred_file.chmod(0o000)
        try:
            result = load_credentials(cred_file)
            assert result is None
        finally:
            cred_file.chmod(stat.S_IRUSR | stat.S_IWUSR)


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
        assert ArenaOAuth2.CREDENTIALS_DIR == Path("/tmp/test-arena")
        assert ArenaOAuth2.CREDENTIALS_FILE == Path("/tmp/test-arena/creds.json")

    def test_configure_skips_none_values(self):
        orig_url = ArenaOAuth2.KEYCLOAK_URL
        ArenaOAuth2.configure(keycloak_url=None)
        assert ArenaOAuth2.KEYCLOAK_URL == orig_url


class TestArenaOAuth2Init:
    @patch("agilerl.arena.auth.KeycloakOpenID")
    def test_creates_keycloak_client(self, mock_kc_cls):
        auth = ArenaOAuth2()
        mock_kc_cls.assert_called_once_with(
            server_url=ArenaOAuth2.KEYCLOAK_URL,
            realm_name=ArenaOAuth2.REALM,
            client_id=ArenaOAuth2.CLIENT_ID,
        )
        assert auth.kc is mock_kc_cls.return_value


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
            ArenaOAuth2._write_credentials(data)

            assert cred_dir.is_dir()
            assert cred_file.is_file()
            written = json.loads(cred_file.read_text(encoding="utf-8"))
            assert written == data

            dir_mode = cred_dir.stat().st_mode & 0o777
            assert dir_mode == stat.S_IRWXU

            file_mode = cred_file.stat().st_mode & 0o777
            assert file_mode == (stat.S_IRUSR | stat.S_IWUSR)
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
    @patch("agilerl.arena.auth.webbrowser")
    @patch("agilerl.arena.auth.time")
    @patch("agilerl.arena.auth.KeycloakOpenID")
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
        mock_write.assert_called_once_with(tokens)
        mock_wb.open.assert_called_once()
        mock_kc.token.assert_called_once_with(
            grant_type="urn:ietf:params:oauth:grant-type:device_code",
            device_code="dc123",
        )

    @patch("agilerl.arena.auth.webbrowser")
    @patch("agilerl.arena.auth.time")
    @patch("agilerl.arena.auth.KeycloakOpenID")
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

    @patch("agilerl.arena.auth.webbrowser")
    @patch("agilerl.arena.auth.time")
    @patch("agilerl.arena.auth.KeycloakOpenID")
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

    @patch("agilerl.arena.auth.webbrowser")
    @patch("agilerl.arena.auth.time")
    @patch("agilerl.arena.auth.KeycloakOpenID")
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

    @patch("agilerl.arena.auth.KeycloakOpenID")
    def test_keycloak_error_on_device_raises_auth_error(self, mock_kc_cls):
        mock_kc = mock_kc_cls.return_value
        mock_kc.device.side_effect = KeycloakError("connection refused")

        auth = ArenaOAuth2()
        with pytest.raises(ArenaAuthError, match="Failed to initiate"):
            auth.device_login()

    @patch("agilerl.arena.auth.webbrowser")
    @patch("agilerl.arena.auth.time")
    @patch("agilerl.arena.auth.KeycloakOpenID")
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

    @patch("agilerl.arena.auth.webbrowser")
    @patch("agilerl.arena.auth.time")
    @patch("agilerl.arena.auth.KeycloakOpenID")
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
    @patch("agilerl.arena.auth.KeycloakOpenID")
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

    @patch("agilerl.arena.auth.KeycloakOpenID")
    def test_keycloak_error_raises_auth_error(self, mock_kc_cls):
        mock_kc = mock_kc_cls.return_value
        mock_kc.refresh_token.side_effect = KeycloakError("invalid refresh token")

        auth = ArenaOAuth2()
        with pytest.raises(ArenaAuthError, match="Token refresh failed"):
            auth.refresh_access_token("bad_rt")


class TestRevoke:
    @patch("agilerl.arena.auth.KeycloakOpenID")
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

    @patch("agilerl.arena.auth.KeycloakOpenID")
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
