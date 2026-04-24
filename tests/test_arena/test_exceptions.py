"""Tests for agilerl.arena.exceptions."""

from __future__ import annotations

import json

import pytest

from agilerl.arena.exceptions import (
    ArenaAPIError,
    ArenaAuthError,
    ArenaError,
    ArenaTimeoutError,
    ArenaTrainingError,
    ArenaValidationError,
    _sanitize_detail,
)


class TestArenaError:
    def test_is_base_exception(self):
        assert issubclass(ArenaError, Exception)

    def test_can_be_raised_and_caught(self):
        with pytest.raises(ArenaError, match="boom"):
            raise ArenaError("boom")


class TestArenaAuthError:
    def test_is_subclass_of_arena_error(self):
        assert issubclass(ArenaAuthError, ArenaError)

    def test_caught_by_arena_error_handler(self):
        with pytest.raises(ArenaError):
            raise ArenaAuthError("auth failed")


class TestArenaAPIError:
    def test_stores_status_code_and_detail(self):
        err = ArenaAPIError(status_code=404, detail="not found")
        assert err.status_code == 404
        assert err.detail == "not found"

    def test_message_format(self):
        err = ArenaAPIError(status_code=500, detail="server error")
        assert str(err) == "API error (500): server error"

    def test_is_subclass_of_arena_error(self):
        assert issubclass(ArenaAPIError, ArenaError)

    def test_caught_by_arena_error_handler(self):
        with pytest.raises(ArenaError):
            raise ArenaAPIError(status_code=400, detail="bad request")


class TestArenaTimeoutError:
    def test_is_subclass_of_arena_auth_error(self):
        assert issubclass(ArenaTimeoutError, ArenaAuthError)

    def test_is_subclass_of_arena_error(self):
        assert issubclass(ArenaTimeoutError, ArenaError)

    def test_caught_by_auth_error_handler(self):
        with pytest.raises(ArenaAuthError):
            raise ArenaTimeoutError("timed out")


class TestArenaValidationError:
    def test_is_subclass_of_arena_api_error(self):
        assert issubclass(ArenaValidationError, ArenaAPIError)

    def test_is_subclass_of_arena_error(self):
        assert issubclass(ArenaValidationError, ArenaError)

    def test_label_is_validation_error(self):
        err = ArenaValidationError(detail="bad input", status_code=400)
        assert "ValidationError" in str(err)

    def test_extras_rendered(self):
        err = ArenaValidationError(
            detail="Ambiguous entrypoint",
            status_code=400,
            extras={"available_entrypoints": ["mod:Env1", "mod:Env2"]},
        )
        msg = str(err)
        assert "mod:Env1" in msg
        assert "mod:Env2" in msg

    def test_from_response_body(self):
        raw = '{"error": "Ambiguous entrypoint", "available_entrypoints": ["a:A", "b:B"], "error_code": "AMBIGUOUS_ENTRYPOINT"}'
        err = ArenaValidationError.from_response_body(raw, status_code=400)
        assert err.status_code == 400
        assert "Ambiguous entrypoint" in str(err)


# ---------------------------------------------------------------------------
# ArenaError._parse_body
# ---------------------------------------------------------------------------
class TestParseBody:
    def test_valid_json_dict(self):
        raw = '{"error": "something bad", "status": 500}'
        result = ArenaError._parse_body(raw)
        assert result == {"error": "something bad", "status": 500}

    def test_newline_in_string_repair(self):
        raw = '{"message": "line1\\nline2", "code": 1}'
        result = ArenaError._parse_body(raw)
        assert result is not None
        assert result["code"] == 1

    def test_ndjson_body_picks_error_line(self):
        raw = (
            '{"kind": "status", "message": "ok"}\n'
            '{"error": "Something failed", "code": 42}\n'
            '{"kind": "status", "message": "done"}\n'
        )
        result = ArenaError._parse_body(raw)
        assert result is not None
        assert result["error"] == "Something failed"

    def test_nested_json_in_string(self):
        inner = '{"message": "real error", "code": 99}'
        raw = f'{{"detail": {json.dumps(inner)}}}'
        result = ArenaError._parse_body(raw)
        assert result is not None
        assert result["message"] == "real error"

    def test_totally_unparseable(self):
        raw = "This is not JSON at all!!! {{{"
        result = ArenaError._parse_body(raw)
        assert result is None


# ---------------------------------------------------------------------------
# ArenaAPIError.from_response_body
# ---------------------------------------------------------------------------
class TestFromResponseBody:
    def test_empty_raw_string(self):
        err = ArenaAPIError.from_response_body("", status_code=500)
        assert err.detail == "No error details"
        assert err.status_code == 500

    def test_non_json_text(self):
        raw = "x" * 600
        err = ArenaAPIError.from_response_body(raw, status_code=502)
        assert len(err.detail) == 500
        assert err.status_code == 502

    def test_extras_with_list_values(self):
        raw = '{"error": "Ambiguous", "available_entrypoints": ["a:A", "b:B"]}'
        err = ArenaAPIError.from_response_body(raw, status_code=400)
        msg = str(err)
        assert "a:A" in msg
        assert "b:B" in msg

    def test_sdk_hint_shown_by_default(self):
        raw = '{"error": "Ambiguous entrypoint", "error_code": "AMBIGUOUS_ENTRYPOINT", "available_entrypoints": ["mod:Env"]}'
        err = ArenaAPIError.from_response_body(raw, status_code=400)
        msg = str(err)
        assert "entrypoint=" in msg

    def test_description_key_used_as_primary(self):
        raw = '{"description": "Detailed error"}'
        err = ArenaAPIError.from_response_body(raw, status_code=400)
        assert err.detail == "Detailed error"

    def test_internal_url_is_sanitized(self):
        raw = json.dumps(
            {
                "detail": (
                    "Environment creation failed: Failed to call list-entrypoints: "
                    "error sending request for url "
                    "(http://env-validator:8080/api/v1/validations/custom-envs/list-entrypoints)"
                )
            }
        )
        err = ArenaValidationError.from_response_body(raw, status_code=500)
        assert "env-validator" not in str(err)
        assert "Something went wrong" in err.detail

    def test_public_url_is_not_sanitized(self):
        raw = json.dumps(
            {"detail": "See https://docs.agilerl.com/errors/123 for details"}
        )
        err = ArenaAPIError.from_response_body(raw, status_code=400)
        assert "docs.agilerl.com" in err.detail


# ---------------------------------------------------------------------------
# _sanitize_detail
# ---------------------------------------------------------------------------
class TestSanitizeDetail:
    def test_strips_internal_http_url(self):
        msg = "Failed to call: error sending request for url (http://env-validator:8080/api/v1/validations/foo)"
        assert _sanitize_detail(msg) == "Something went wrong. Please try again later."

    def test_strips_internal_https_url(self):
        msg = "Error: request to https://internal-svc:443/api/v2/thing failed"
        assert _sanitize_detail(msg) == "Something went wrong. Please try again later."

    def test_preserves_normal_message(self):
        msg = "Ambiguous entrypoint detected"
        assert _sanitize_detail(msg) == msg

    def test_preserves_empty_message(self):
        assert _sanitize_detail("") == ""

    def test_public_url_is_preserved(self):
        msg = "See https://arena.agilerl.com/docs for info"
        assert _sanitize_detail(msg) == msg


# ---------------------------------------------------------------------------
# enable_cli_mode
# ---------------------------------------------------------------------------
class TestEnableCliMode:
    def setup_method(self):
        self._orig = ArenaError._cli_mode

    def teardown_method(self):
        ArenaError._cli_mode = self._orig

    def test_cli_mode_affects_auth_error(self):
        ArenaError.enable_cli_mode()
        err = ArenaAuthError(
            "auth failed",
            sdk_hint="Use client.login()",
            cli_hint="Run 'arena login'",
        )
        assert "arena login" in str(err)
        assert "client.login" not in str(err)

    def test_sdk_hint_when_cli_mode_off(self):
        ArenaError._cli_mode = False
        err = ArenaAuthError(
            "auth failed",
            sdk_hint="Use client.login()",
            cli_hint="Run 'arena login'",
        )
        assert "client.login" in str(err)

    def test_cli_mode_affects_api_error(self):
        ArenaError.enable_cli_mode()
        err = ArenaAPIError(
            detail="error",
            status_code=400,
            sdk_hint="SDK hint",
            cli_hint="CLI hint",
        )
        msg = str(err)
        assert "CLI hint" in msg
        assert "SDK hint" not in msg

    def test_cli_mode_affects_subclasses(self):
        ArenaError.enable_cli_mode()
        err = ArenaValidationError(
            detail="bad",
            status_code=422,
            sdk_hint="sdk",
            cli_hint="cli",
        )
        assert "cli" in str(err)


# ---------------------------------------------------------------------------
# ArenaTrainingError
# ---------------------------------------------------------------------------
class TestArenaTrainingError:
    def test_label_is_training_error(self):
        err = ArenaTrainingError(detail="job failed", status_code=500)
        assert "TrainingError" in str(err)

    def test_is_subclass_of_api_error(self):
        assert issubclass(ArenaTrainingError, ArenaAPIError)


# ---------------------------------------------------------------------------
# _generate_hints
# ---------------------------------------------------------------------------
class TestGenerateHints:
    def test_ambiguous_entrypoint_with_list(self):
        body = {"error_code": "AMBIGUOUS_ENTRYPOINT"}
        extras = {"available_entrypoints": ["mod:Env1", "mod:Env2"]}
        sdk_hint, cli_hint = ArenaError._generate_hints(body, extras)
        assert "mod:Env1" in sdk_hint
        assert "mod:Env1" in cli_hint

    def test_ambiguous_entrypoint_empty_list(self):
        body = {"error_code": "AMBIGUOUS_ENTRYPOINT"}
        extras = {"available_entrypoints": []}
        sdk_hint, cli_hint = ArenaError._generate_hints(body, extras)
        assert "<entrypoint>" in sdk_hint
        assert "<entrypoint>" in cli_hint

    def test_unknown_error_code_empty_hints(self):
        body = {"error_code": "SOME_OTHER_CODE"}
        extras = {}
        sdk_hint, cli_hint = ArenaError._generate_hints(body, extras)
        assert sdk_hint == ""
        assert cli_hint == ""

    def test_no_error_code_empty_hints(self):
        body = {}
        extras = {}
        sdk_hint, cli_hint = ArenaError._generate_hints(body, extras)
        assert sdk_hint == ""
        assert cli_hint == ""


# ---------------------------------------------------------------------------
# arena/__init__.py import guard
# ---------------------------------------------------------------------------


class TestArenaInitImportGuard:
    def test_import_error_when_deps_missing(self):
        """Importing agilerl.arena raises ImportError when arena extras are
        missing (HAS_ARENA_DEPENDENCIES is False)."""
        import importlib
        import sys

        import agilerl.arena as arena_mod

        saved_module = sys.modules.pop("agilerl.arena", None)
        try:
            with pytest.MonkeyPatch.context() as mp:
                mp.setattr("agilerl.HAS_ARENA_DEPENDENCIES", False)
                with pytest.raises(ImportError, match="Arena dependencies"):
                    importlib.import_module("agilerl.arena")
        finally:
            if saved_module is not None:
                sys.modules["agilerl.arena"] = saved_module
