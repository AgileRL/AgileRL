# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Tests for arena.exceptions."""

from __future__ import annotations

import json
from urllib.parse import urlparse

import pytest

from agilerl.arena.exceptions import (
    ArenaAPIError,
    ArenaAuthError,
    ArenaConfigError,
    ArenaError,
    ArenaInferenceError,
    ArenaTimeoutError,
    ArenaTrainingError,
    ArenaValidationError,
    _sanitize_detail,
)


class TestArenaError:
    def test_is_base_exception(self):
        assert issubclass(ArenaError, Exception)

    def test_can_be_raised_and_caught(self):
        msg = "arena operation failed"

        def _raise():
            raise ArenaError(msg)

        with pytest.raises(ArenaError, match="arena operation failed"):
            _raise()


class TestArenaAuthError:
    def test_is_subclass_of_arena_error(self):
        assert issubclass(ArenaAuthError, ArenaError)

    def test_caught_by_arena_error_handler(self):
        msg = "auth failed"

        def _raise():
            raise ArenaAuthError(msg)

        with pytest.raises(ArenaError, match="auth failed"):
            _raise()


class TestArenaConfigError:
    def setup_method(self):
        self._orig = ArenaError._cli_mode

    def teardown_method(self):
        ArenaError._cli_mode = self._orig

    def test_is_subclass_of_arena_error(self):
        assert issubclass(ArenaConfigError, ArenaError)

    def test_sdk_mode_shows_sdk_hint(self):
        ArenaError._cli_mode = False
        err = ArenaConfigError(
            "No project specified.",
            sdk_hint="Pass project= to the method.",
            cli_hint="Use --project flag.",
        )
        assert "Pass project=" in str(err)
        assert "--project" not in str(err)

    def test_cli_mode_shows_cli_hint(self):
        ArenaError._cli_mode = True
        err = ArenaConfigError(
            "No project specified.",
            sdk_hint="Pass project= to the method.",
            cli_hint="Use --project flag.",
        )
        assert "--project" in str(err)
        assert "Pass project=" not in str(err)

    def test_str_without_hint_returns_message_only(self):
        err = ArenaConfigError("No project specified.")
        assert str(err) == "No project specified."


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

    def test_scalar_extra_rendered(self):
        err = ArenaAPIError(detail="failed", extras={"retry_after": 30})
        assert "Retry after: 30" in str(err)

    def test_format_without_status_code(self):
        err = ArenaAPIError(detail="failed")
        assert str(err) == "API error: failed"


class TestArenaTimeoutError:
    def test_is_subclass_of_arena_auth_error(self):
        assert issubclass(ArenaTimeoutError, ArenaAuthError)

    def test_is_subclass_of_arena_error(self):
        assert issubclass(ArenaTimeoutError, ArenaError)

    def test_caught_by_auth_error_handler(self):
        msg = "timed out"

        def _raise():
            raise ArenaTimeoutError(msg)

        with pytest.raises(ArenaAuthError, match="timed out"):
            _raise()


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

    def test_ndjson_skips_blank_lines(self):
        raw = '\n\n{"error": "failed"}\n'
        result = ArenaError._parse_body(raw)
        assert result == {"error": "failed"}

    def test_ndjson_skips_status_line_and_blank_line_before_error(self):
        raw = '{"kind": "status", "message": "ok"}\n\n{"error": "failed"}'
        result = ArenaError._parse_body(raw)
        assert result == {"error": "failed"}

    def test_ndjson_only_blank_lines_then_error(self):
        raw = '\n\n\n{"error": "failed"}'
        result = ArenaError._parse_body(raw)
        assert result == {"error": "failed"}

    def test_ndjson_skips_whitespace_only_lines(self):
        raw = '   \n\t\n{"error": "line parsed"}'
        result = ArenaError._parse_body(raw)
        assert result == {"error": "line parsed"}

    def test_nested_envelope_non_message_inner_skipped(self):
        inner = json.dumps({"status": 1})
        raw = json.dumps({"detail": inner})
        result = ArenaError._parse_body(raw)
        assert result is not None
        assert result["detail"] == inner


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
        ArenaError._cli_mode = False
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
        parsed = urlparse("https://docs.agilerl.com/errors/123")
        assert parsed.hostname == "docs.agilerl.com"
        assert parsed.geturl() in err.detail


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


class TestArenaInferenceError:
    def test_label_is_inference_error(self):
        err = ArenaInferenceError(detail="bad obs", status_code=400)
        assert "InferenceError" in str(err)

    def test_is_subclass_of_arena_api_error(self):
        assert issubclass(ArenaInferenceError, ArenaAPIError)


class TestArenaTrainingError:
    def test_label_is_training_error(self):
        err = ArenaTrainingError(detail="job failed", status_code=500)
        assert "TrainingError" in str(err)

    def test_is_subclass_of_api_error(self):
        assert issubclass(ArenaTrainingError, ArenaAPIError)


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

    def test_environment_not_found_via_message_key(self):
        body = {"message": "Environment 'X' not found."}
        sdk_hint, cli_hint = ArenaError._generate_hints(body, {})
        assert "list_environments" in sdk_hint
        assert "arena env list" in cli_hint


class TestArenaPackageImport:
    def test_agilerl_arena_package_imports(self):
        import importlib

        arena_mod = importlib.import_module("agilerl.arena")
        assert arena_mod is not None


class TestResolveApiErrorClass:
    def test_training_error_promoted_for_missing_environment(self):
        from agilerl.arena.exceptions import (
            ArenaEnvironmentNotFoundError,
            ArenaTrainingError,
            resolve_api_error_class,
        )

        message = "Environment 'LunarLander-v3' not found. Register custom environments first."
        assert (
            resolve_api_error_class(ArenaTrainingError, message)
            is ArenaEnvironmentNotFoundError
        )

    def test_training_error_unchanged_for_other_failures(self):
        from agilerl.arena.exceptions import ArenaTrainingError, resolve_api_error_class

        assert (
            resolve_api_error_class(
                ArenaTrainingError, "Internal server error during training."
            )
            is ArenaTrainingError
        )

    def test_other_base_classes_unchanged(self):
        from agilerl.arena.exceptions import (
            ArenaAPIError,
            ArenaValidationError,
            resolve_api_error_class,
        )

        message = "Environment 'X' not found."
        assert resolve_api_error_class(ArenaAPIError, message) is ArenaAPIError
        assert (
            resolve_api_error_class(ArenaValidationError, message)
            is ArenaValidationError
        )


class TestArenaEnvironmentNotFoundError:
    def test_promoted_from_training_error(self):
        import json

        from agilerl.arena.exceptions import (
            ArenaEnvironmentNotFoundError,
            ArenaError,
            ArenaTrainingError,
        )

        ArenaError._cli_mode = False
        raw = json.dumps(
            {
                "error": "Environment 'MyEnv-v1' not found. Register custom environments first."
            }
        )
        err = ArenaTrainingError.from_response_body(raw, status_code=400)
        assert isinstance(err, ArenaEnvironmentNotFoundError)

    def test_sdk_hint_shown(self):
        import json

        from agilerl.arena.exceptions import ArenaError, ArenaTrainingError

        ArenaError._cli_mode = False
        raw = json.dumps({"error": "Environment 'X' not found."})
        err = ArenaTrainingError.from_response_body(raw, status_code=400)
        assert "list_environments" in str(err)

    def test_cli_hint_shown(self):
        import json

        from agilerl.arena.exceptions import ArenaError, ArenaTrainingError

        ArenaError._cli_mode = True
        raw = json.dumps({"error": "Environment 'X' not found."})
        err = ArenaTrainingError.from_response_body(raw, status_code=400)
        assert "arena env list" in str(err)
        ArenaError._cli_mode = False

    def test_non_env_error_stays_training_error(self):
        import json

        from agilerl.arena.exceptions import (
            ArenaEnvironmentNotFoundError,
            ArenaError,
            ArenaTrainingError,
        )

        ArenaError._cli_mode = False
        raw = json.dumps({"error": "Internal server error during training."})
        err = ArenaTrainingError.from_response_body(raw, status_code=500)
        assert not isinstance(err, ArenaEnvironmentNotFoundError)
        assert isinstance(err, ArenaTrainingError)
