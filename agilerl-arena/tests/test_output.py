# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Tests for arena.output — StreamRichRenderer, emit_result, handle_error."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import click
import pytest

from agilerl.arena.exceptions import (
    ArenaAPIError,
    ArenaTrainingError,
    ArenaValidationError,
)
from agilerl.arena.output import (
    CANCEL,
    DOWN,
    ENTER,
    UP,
    StreamRichRenderer,
    _looks_like_environment_catalog,
    emit_csv_preview,
    emit_result,
    handle_error,
    select_row,
    supports_interactive_selection,
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
    @pytest.fixture
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

    @patch("agilerl.arena.output.logger")
    def test_secondary_info_logs_dimmed_and_indented(self, mock_logger):
        renderer = StreamRichRenderer()
        event = StatusEvent(
            stage="validation",
            status="running",
            message="Downloading pygments",
            detail={"package": "pygments"},
            raw={},
            level="secondary-info",
        )
        renderer.handle_event(event)

        mock_logger.info.assert_called_once_with(
            "  [dim]%s[/dim]", "Downloading pygments"
        )

    @patch("agilerl.arena.output.logger")
    def test_secondary_info_escapes_markup_in_message(self, mock_logger):
        renderer = StreamRichRenderer()
        event = StatusEvent(
            stage="validation",
            status="running",
            message="Downloading uvicorn[standard]",
            detail={},
            raw={},
            level="secondary-info",
        )
        renderer.handle_event(event)

        logged = mock_logger.info.call_args[0][1]
        # The opening bracket is backslash-escaped so Rich treats it literally.
        assert r"\[standard]" in logged

    @patch("agilerl.arena.output.logger")
    def test_install_success_info_level_logs_normally(self, mock_logger):
        renderer = StreamRichRenderer()
        event = StatusEvent(
            stage="validation",
            status="running",
            message="Installed 5 package(s)",
            detail={"installed": 5},
            raw={},
            level="info",
        )
        renderer.handle_event(event)

        mock_logger.info.assert_called_once_with("%s", "Installed 5 package(s)")

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

    def test_error_without_live_promotes_training_env_not_found(self):
        from agilerl.arena.exceptions import (
            ArenaEnvironmentNotFoundError,
            ArenaError,
        )

        ArenaError._cli_mode = False
        renderer = StreamRichRenderer(error_cls=ArenaTrainingError)
        event = ErrorEvent(
            message="Environment 'LunarLander-v3' not found.",
            extras={},
            raw={},
        )

        with pytest.raises(ArenaEnvironmentNotFoundError) as exc_info:
            renderer.handle_event(event)

        assert "list_environments" in exc_info.value.sdk_hint
        assert "arena env list" in exc_info.value.cli_hint

    def test_error_with_live_shows_env_not_found_cli_hint(self):
        renderer = StreamRichRenderer()
        renderer._live = MagicMock()
        event = ErrorEvent(
            message="Environment 'X' not found.",
            extras={},
            raw={},
        )
        renderer.handle_event(event)

        assert "arena env list" in renderer._rows[0].status


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

    def test_check_event_starts_live_renderer(self):
        renderer = StreamRichRenderer()
        event = CheckEvent(name="imports", success=True, warnings=[], error="", raw={})
        renderer.handle_event(event)

        assert renderer._live is not None
        assert len(renderer._rows) == 1

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

    def test_non_dict_version_map(self):
        assert _looks_like_environment_catalog({"MyEnv": "v1"}) is False

    def test_non_dict_metadata(self):
        catalog = {"MyEnv": {"v1": "ready"}}
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

    @patch("agilerl.arena.output.error_console")
    def test_is_error_uses_error_console(self, mock_error_console):
        from agilerl.arena.output import _print_rich

        _print_rich("oops", is_error=True)
        mock_error_console.print.assert_called_once_with("oops")

    @patch("agilerl.arena.output.console")
    def test_is_error_false_uses_console(self, mock_console):
        from agilerl.arena.output import _print_rich

        _print_rich("ok", is_error=False)
        mock_console.print.assert_called_once_with("ok")

    @patch("agilerl.arena.output._print_rich")
    def test_empty_environment_catalog(self, mock_print):
        from agilerl.arena.output import _emit_environment_catalog

        _emit_environment_catalog({})
        mock_print.assert_called_once_with("No environments found.", is_error=False)

    @patch("agilerl.arena.output._print_rich")
    def test_catalog_skips_non_dict_versions(self, mock_print):
        from agilerl.arena.output import _emit_environment_catalog

        catalog = {
            "MyEnv": "not-a-version-map",
            "Other": {"v1": {"validated": True, "profiled": False}},
        }
        _emit_environment_catalog(catalog)
        mock_print.assert_called_once()

    def test_format_cell_json_for_nested_values(self):
        from agilerl.arena.output import _format_cell

        result = _format_cell({"a": 1})
        assert '"a"' in result


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

    @patch("agilerl.arena.output.console")
    def test_wide_csv_is_pivoted(self, mock_console):
        header = ",".join(f"metric_{i}" for i in range(30))
        row = ",".join(str(i) for i in range(30))
        csv_data = f"{header}\n{row}\n{row}\n".encode()
        emit_csv_preview(csv_data, max_rows=2)
        table = mock_console.print.call_args.args[0]
        # Pivoted: one leading "Metric" column plus one column per preview row.
        assert [col.header for col in table.columns] == ["Metric", "Row 1", "Row 2"]
        assert table.row_count == 30

    @patch("agilerl.arena.output.console")
    def test_narrow_csv_stays_flat(self, mock_console):
        csv_data = b"col1,col2\n1,2\n3,4\n5,6\n"
        emit_csv_preview(csv_data, max_rows=3)
        table = mock_console.print.call_args.args[0]
        assert [col.header for col in table.columns] == ["col1", "col2"]
        assert table.row_count == 3


# ---------------------------------------------------------------------------
# Interactive row selection
# ---------------------------------------------------------------------------


ROWS = [
    {"session_id": "s1", "created_at": "2026-08-01", "last_updated": "2026-08-05"},
    {"session_id": "s2", "created_at": "2026-07-11", "last_updated": "2026-08-04"},
    {"session_id": "s3", "created_at": "2026-06-02", "last_updated": "2026-07-30"},
]


def _pick(keys, rows=None, **kwargs):
    """Drive select_row with a scripted sequence of keypresses."""
    presses = iter(keys)
    with patch("agilerl.arena.output._read_key", side_effect=lambda: next(presses)):
        return select_row(rows if rows is not None else ROWS, **kwargs)


def _last_rendered(keys, **kwargs):
    """The final table select_row drew before the user committed."""
    with patch("agilerl.arena.output.Live") as mock_live:
        _pick(keys, **kwargs)
    return mock_live.return_value.__enter__.return_value.update.call_args.args[0]


class TestSelectRow:
    def test_enter_takes_the_first_row(self):
        assert _pick([ENTER])["session_id"] == "s1"

    def test_down_then_enter(self):
        assert _pick([DOWN, ENTER])["session_id"] == "s2"

    def test_down_and_back_up(self):
        assert _pick([DOWN, DOWN, UP, ENTER])["session_id"] == "s2"

    def test_up_from_the_top_wraps_to_the_bottom(self):
        assert _pick([UP, ENTER])["session_id"] == "s3"

    def test_down_from_the_bottom_wraps_to_the_top(self):
        assert _pick([DOWN, DOWN, DOWN, ENTER])["session_id"] == "s1"

    def test_cancel_returns_none(self):
        assert _pick([DOWN, CANCEL]) is None

    def test_unrecognised_keys_are_ignored(self):
        assert _pick([None, None, DOWN, None, ENTER])["session_id"] == "s2"

    def test_selected_sets_the_starting_row(self):
        assert _pick([ENTER], selected=2)["session_id"] == "s3"

    def test_selected_is_clamped_into_range(self):
        assert _pick([ENTER], selected=99)["session_id"] == "s3"
        assert _pick([ENTER], selected=-5)["session_id"] == "s1"

    def test_no_rows_returns_none_without_reading_a_key(self):
        with patch("agilerl.arena.output._read_key") as mock_read:
            assert select_row([]) is None
        mock_read.assert_not_called()

    def test_columns_rename_the_headers(self):
        table = _last_rendered([ENTER], columns=["Session Id", "Created At", "Last"])
        # A leading marker column precedes the caller's columns.
        assert [col.header for col in table.columns] == [
            "",
            "Session Id",
            "Created At",
            "Last",
        ]

    def test_headers_fall_back_to_the_row_keys(self):
        table = _last_rendered([ENTER])
        assert [col.header for col in table.columns] == [
            "",
            "session_id",
            "created_at",
            "last_updated",
        ]

    def test_only_the_selected_row_is_marked(self):
        table = _last_rendered([DOWN, ENTER])
        assert list(table.columns[0]._cells) == [" ", "[cyan]▸[/cyan]", " "]

    def test_the_highlight_follows_the_arrow_keys(self):
        first = _last_rendered([ENTER])
        assert list(first.columns[0]._cells).index("[cyan]▸[/cyan]") == 0
        second = _last_rendered([DOWN, DOWN, ENTER])
        assert list(second.columns[0]._cells).index("[cyan]▸[/cyan]") == 2


class TestSupportsInteractiveSelection:
    @patch("agilerl.arena.output.console")
    @patch("agilerl.arena.output.sys")
    def test_true_for_a_terminal(self, mock_sys, mock_console):
        mock_sys.stdin.isatty.return_value = True
        mock_console.is_terminal = True
        assert supports_interactive_selection() is True

    @patch("agilerl.arena.output.console")
    @patch("agilerl.arena.output.sys")
    def test_false_when_stdin_is_piped(self, mock_sys, mock_console):
        mock_sys.stdin.isatty.return_value = False
        mock_console.is_terminal = True
        assert supports_interactive_selection() is False

    @patch("agilerl.arena.output.console")
    @patch("agilerl.arena.output.sys")
    def test_false_when_output_is_redirected(self, mock_sys, mock_console):
        mock_sys.stdin.isatty.return_value = True
        mock_console.is_terminal = False
        assert supports_interactive_selection() is False

    @patch("agilerl.arena.output.sys")
    def test_false_when_stdin_is_detached(self, mock_sys):
        mock_sys.stdin.isatty.side_effect = ValueError("I/O operation on closed file")
        assert supports_interactive_selection() is False
