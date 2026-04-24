"""Tests for agilerl.arena.output — StreamRichRenderer, emit_result, handle_error."""

from __future__ import annotations

import io
import logging
from unittest.mock import MagicMock, patch

import click
import pytest
from rich.console import Console

from agilerl.arena.exceptions import ArenaAPIError, ArenaError, ArenaValidationError
from agilerl.arena.output import (
    StreamRichRenderer,
    StreamRow,
    _looks_like_environment_catalog,
    emit_csv_preview,
    emit_result,
    handle_error,
)
from agilerl.arena.stream import CheckEvent, ErrorEvent, LogEvent, StatusEvent


# ---------------------------------------------------------------------------
# StreamRichRenderer._styled_status
# ---------------------------------------------------------------------------
class TestStyledStatus:
    def test_pass(self):
        assert "green" in StreamRichRenderer._styled_status("PASS")

    def test_fail(self):
        assert "red" in StreamRichRenderer._styled_status("FAIL")

    def test_warning(self):
        result = StreamRichRenderer._styled_status("WARNING")
        assert "dark_orange" in result

    def test_warn_alias(self):
        result = StreamRichRenderer._styled_status("WARN")
        assert "dark_orange" in result

    def test_completed(self):
        assert "green" in StreamRichRenderer._styled_status("COMPLETED")

    def test_unknown_status_passthrough(self):
        assert StreamRichRenderer._styled_status("UNKNOWN") == "UNKNOWN"


# ---------------------------------------------------------------------------
# StreamRichRenderer.handle_event — CheckEvent
# ---------------------------------------------------------------------------
class TestRendererCheckEvent:
    @pytest.fixture()
    def renderer(self):
        r = StreamRichRenderer()
        r._live = MagicMock()
        return r

    def test_check_pass(self, renderer):
        event = CheckEvent(name="imports", success=True, warnings=[], error="", raw={})
        renderer.handle_event(event)

        assert len(renderer._rows) == 1
        assert "PASS" in renderer._rows[0].status
        assert renderer._rows[0].details == ""

    def test_check_fail(self, renderer):
        event = CheckEvent(
            name="entrypoint", success=False, warnings=[], error="Not found", raw={}
        )
        renderer.handle_event(event)

        assert len(renderer._rows) == 1
        assert "FAIL" in renderer._rows[0].status
        assert renderer._rows[0].details == "Not found"

    def test_check_pass_with_warnings(self, renderer):
        event = CheckEvent(
            name="deps",
            success=True,
            warnings=["Deprecated pkg", "Version conflict"],
            error="",
            raw={},
        )
        renderer.handle_event(event)

        assert len(renderer._rows) == 1
        assert "WARNING" in renderer._rows[0].status
        assert "Deprecated pkg" in renderer._rows[0].details
        assert "Version conflict" in renderer._rows[0].details

    def test_check_unknown_success(self, renderer):
        event = CheckEvent(name="seed", success=None, warnings=[], error="", raw={})
        renderer.handle_event(event)

        assert renderer._rows[0].status == "UNKNOWN"


# ---------------------------------------------------------------------------
# StreamRichRenderer.handle_event — StatusEvent
# ---------------------------------------------------------------------------
class TestRendererStatusEvent:
    def test_completed_closes_live(self):
        renderer = StreamRichRenderer()
        mock_live = MagicMock()
        renderer._live = mock_live

        event = StatusEvent(
            stage="validation",
            status="completed",
            message="Done",
            detail={},
            raw={},
        )
        renderer.handle_event(event)

        mock_live.stop.assert_called_once()
        assert renderer._live is None

    @patch("agilerl.arena.output.logger")
    def test_warning_logs_at_warning_level(self, mock_logger):
        renderer = StreamRichRenderer()
        event = StatusEvent(
            stage="submission",
            status="running",
            message="Low credit balance",
            detail={},
            raw={},
            kind="warning",
        )
        renderer.handle_event(event)

        mock_logger.warning.assert_called_once_with("%s", "Low credit balance")

    @patch("agilerl.arena.output.logger")
    def test_info_logs_at_info_level(self, mock_logger):
        renderer = StreamRichRenderer()
        event = StatusEvent(
            stage="upload",
            status="running",
            message="Uploading environment",
            detail={},
            raw={},
        )
        renderer.handle_event(event)

        mock_logger.info.assert_called_once_with("%s", "Uploading environment")

    def test_status_does_not_add_rows(self):
        renderer = StreamRichRenderer()
        event = StatusEvent(
            stage="upload",
            status="running",
            message="Uploading",
            detail={},
            raw={},
        )
        renderer.handle_event(event)
        assert renderer._rows == []


# ---------------------------------------------------------------------------
# StreamRichRenderer.handle_event — ErrorEvent (W3)
# ---------------------------------------------------------------------------
class TestRendererErrorEvent:
    def test_error_without_live_raises(self):
        """When _live is None, ErrorEvent raises the configured error class (W3)."""
        renderer = StreamRichRenderer(error_cls=ArenaValidationError)
        event = ErrorEvent(
            message="Ambiguous entrypoint", extras={"available": ["a:A"]}
        )

        with pytest.raises(ArenaValidationError) as exc_info:
            renderer.handle_event(event)

        assert "Ambiguous entrypoint" in str(exc_info.value)

    def test_error_with_live_appends_row_no_raise(self):
        """When _live exists, ErrorEvent appends a row but does NOT raise (W3)."""
        renderer = StreamRichRenderer()
        renderer._live = MagicMock()

        event = ErrorEvent(
            message="Profiling failed",
            extras={"reason": "timeout"},
        )
        renderer.handle_event(event)

        assert len(renderer._rows) == 1
        assert renderer._rows[0].event_type == "error"
        assert "Profiling failed" in renderer._rows[0].status

    def test_error_row_includes_extras(self):
        renderer = StreamRichRenderer()
        renderer._live = MagicMock()

        event = ErrorEvent(
            message="Bad env",
            extras={"available_entrypoints": ["a:A", "b:B"]},
        )
        renderer.handle_event(event)

        row_status = renderer._rows[0].status
        assert "a:A" in row_status
        assert "b:B" in row_status


# ---------------------------------------------------------------------------
# StreamRichRenderer.handle_event — LogEvent
# ---------------------------------------------------------------------------
class TestRendererLogEvent:
    def test_log_event_adds_row(self):
        renderer = StreamRichRenderer()
        renderer._live = MagicMock()

        event = LogEvent(text="some debug output")
        renderer.handle_event(event)

        assert len(renderer._rows) == 1
        assert renderer._rows[0].event_type == "log"
        assert renderer._rows[0].status == "some debug output"

    def test_empty_log_event_no_row(self):
        renderer = StreamRichRenderer()
        renderer._live = MagicMock()

        event = LogEvent(text="")
        renderer.handle_event(event)

        assert renderer._rows == []


# ---------------------------------------------------------------------------
# StreamRichRenderer.close — idempotency (W4)
# ---------------------------------------------------------------------------
class TestRendererClose:
    def test_close_idempotent(self):
        """Calling close() twice does not raise (W4)."""
        renderer = StreamRichRenderer()
        renderer._live = MagicMock()

        renderer.close()
        renderer.close()

    def test_close_when_never_started(self):
        """close() when _live was never created is a no-op."""
        renderer = StreamRichRenderer()
        renderer.close()

    def test_context_manager(self):
        renderer = StreamRichRenderer()
        renderer._live = MagicMock()
        with renderer:
            pass
        assert renderer._live is None


# ---------------------------------------------------------------------------
# _looks_like_environment_catalog
# ---------------------------------------------------------------------------
class TestLooksLikeEnvironmentCatalog:
    def test_valid_catalog(self):
        catalog = {
            "MyEnv": {
                "v1": {"validated": True, "profiled": True},
                "v2": {"validated": False, "profiled": False},
            }
        }
        assert _looks_like_environment_catalog(catalog) is True

    def test_empty_dict_is_not_catalog(self):
        assert _looks_like_environment_catalog({}) is False

    def test_flat_dict_is_not_catalog(self):
        assert _looks_like_environment_catalog({"key": "value"}) is False

    def test_missing_validated_key(self):
        catalog = {"MyEnv": {"v1": {"profiled": True}}}
        assert _looks_like_environment_catalog(catalog) is False


# ---------------------------------------------------------------------------
# emit_result dispatch
# ---------------------------------------------------------------------------
class TestEmitResult:
    @patch("agilerl.arena.output._print_rich")
    def test_dict_renders_key_value_table(self, mock_print):
        emit_result({"name": "MyEnv", "status": "active"})
        mock_print.assert_called_once()

    @patch("agilerl.arena.output._print_rich")
    def test_list_of_dicts_renders_table(self, mock_print):
        emit_result([{"id": 1, "name": "a"}, {"id": 2, "name": "b"}])
        mock_print.assert_called_once()

    @patch("agilerl.arena.output._print_rich")
    def test_simple_list_renders_table(self, mock_print):
        emit_result(["one", "two", "three"])
        mock_print.assert_called_once()

    @patch("agilerl.arena.output._print_rich")
    def test_catalog_dict_triggers_catalog_renderer(self, mock_print):
        catalog = {"Env": {"v1": {"validated": True, "profiled": False}}}
        emit_result(catalog)
        mock_print.assert_called_once()

    @patch("agilerl.arena.output._print_rich")
    def test_non_dict_list_falls_through_to_str(self, mock_print):
        emit_result(42)
        mock_print.assert_called_once()
        args = mock_print.call_args
        assert args[0][0] == "42"


# ---------------------------------------------------------------------------
# handle_error
# ---------------------------------------------------------------------------
class TestHandleError:
    def test_arena_error_exits_with_code_1(self):
        err = ArenaAPIError(detail="bad request", status_code=400)
        with pytest.raises(click.exceptions.Exit) as exc_info:
            handle_error(err)
        assert exc_info.value.exit_code == 1

    def test_non_arena_error_re_raises(self):
        err = RuntimeError("unexpected")
        with pytest.raises(RuntimeError, match="unexpected"):
            handle_error(err)


# ---------------------------------------------------------------------------
# emit_csv_preview
# ---------------------------------------------------------------------------
class TestEmitCsvPreview:
    @patch("agilerl.arena.output.console")
    def test_csv_preview(self, mock_console):
        csv_data = b"col1,col2\n1,2\n3,4\n5,6\n"
        emit_csv_preview(csv_data, max_rows=2)
        mock_console.print.assert_called_once()

    @patch("agilerl.arena.output.console")
    def test_empty_csv_no_output(self, mock_console):
        emit_csv_preview(b"", max_rows=5)
        mock_console.print.assert_not_called()
