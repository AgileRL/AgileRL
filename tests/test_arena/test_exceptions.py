"""Tests for agilerl.arena.exceptions."""

from __future__ import annotations

import pytest

from agilerl.arena.exceptions import (
    ArenaAPIError,
    ArenaAuthError,
    ArenaError,
    ArenaTimeoutError,
    ArenaValidationError,
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
        assert str(err) == "Arena API error 500: server error"

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
    def test_stores_errors_list(self):
        errors = [{"msg": "field required"}, {"msg": "invalid value"}]
        err = ArenaValidationError(errors)
        assert err.errors == errors

    def test_summary_message(self):
        errors = [{"msg": "err1"}, {"msg": "err2"}]
        err = ArenaValidationError(errors)
        assert "err1" in str(err)
        assert "err2" in str(err)
        assert "Environment validation failed" in str(err)

    def test_truncates_summary_beyond_five(self):
        errors = [{"msg": f"err{i}"} for i in range(8)]
        err = ArenaValidationError(errors)
        msg = str(err)
        assert "err0" in msg
        assert "err4" in msg
        assert "err5" not in msg
        assert "and 3 more" in msg

    def test_is_subclass_of_arena_error(self):
        assert issubclass(ArenaValidationError, ArenaError)

    def test_errors_without_msg_key(self):
        errors = [{"code": "E001"}]
        err = ArenaValidationError(errors)
        assert "E001" in str(err) or "code" in str(err)
