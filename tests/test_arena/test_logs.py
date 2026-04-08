"""Tests for agilerl.arena.logs — SSE parsing, EventStream, LogDisplay."""

from __future__ import annotations

import json
import time
from contextlib import contextmanager
from dataclasses import FrozenInstanceError
from unittest.mock import MagicMock, patch, PropertyMock

import httpx
import pytest

from agilerl.arena.exceptions import ArenaAPIError, ArenaAuthError
from agilerl.arena.logs import (
    EventStream,
    LogDisplay,
    LogEvent,
    _frame_to_log_event,
    _parse_sse_events,
)


# ---------------------------------------------------------------------------
# _parse_sse_events
# ---------------------------------------------------------------------------
class TestParseSSEEvents:
    def test_single_event(self):
        lines = iter(["data: hello\n", "\n"])
        events = list(_parse_sse_events(lines))
        assert len(events) == 1
        assert events[0]["data"] == "hello"
        assert events[0]["event"] == "log"

    def test_multiple_events(self):
        lines = iter(
            [
                "data: first\n",
                "\n",
                "data: second\n",
                "\n",
            ]
        )
        events = list(_parse_sse_events(lines))
        assert len(events) == 2
        assert events[0]["data"] == "first"
        assert events[1]["data"] == "second"

    def test_multiline_data(self):
        lines = iter(
            [
                "data: line1\n",
                "data: line2\n",
                "\n",
            ]
        )
        events = list(_parse_sse_events(lines))
        assert len(events) == 1
        assert events[0]["data"] == "line1\nline2"

    def test_comment_lines_skipped(self):
        lines = iter(
            [
                ": this is a comment\n",
                "data: payload\n",
                "\n",
            ]
        )
        events = list(_parse_sse_events(lines))
        assert len(events) == 1
        assert events[0]["data"] == "payload"

    def test_custom_event_and_id(self):
        lines = iter(
            [
                "event: progress\n",
                "id: evt-42\n",
                "data: step\n",
                "\n",
            ]
        )
        events = list(_parse_sse_events(lines))
        assert events[0]["event"] == "progress"
        assert events[0]["id"] == "evt-42"

    def test_trailing_data_without_blank_line(self):
        lines = iter(["data: incomplete"])
        events = list(_parse_sse_events(lines))
        assert len(events) == 0

    def test_empty_lines_between_events(self):
        lines = iter(
            [
                "data: a\n",
                "\n",
                "\n",
                "data: b\n",
                "\n",
            ]
        )
        events = list(_parse_sse_events(lines))
        assert len(events) == 2

    def test_event_type_resets_between_events(self):
        lines = iter(
            [
                "event: complete\n",
                "data: done\n",
                "\n",
                "data: next\n",
                "\n",
            ]
        )
        events = list(_parse_sse_events(lines))
        assert events[0]["event"] == "complete"
        assert events[1]["event"] == "log"


# ---------------------------------------------------------------------------
# _frame_to_log_event
# ---------------------------------------------------------------------------
class TestFrameToLogEvent:
    def test_valid_json_data(self):
        frame = {
            "event": "log",
            "id": "1",
            "data": json.dumps(
                {
                    "level": "INFO",
                    "message": "Training started",
                    "ts": "2024-01-01T00:00:00",
                    "step": 10,
                }
            ),
        }
        event = _frame_to_log_event(frame)
        assert event.type == "log"
        assert event.id == "1"
        assert event.level == "INFO"
        assert event.message == "Training started"
        assert event.timestamp == "2024-01-01T00:00:00"
        assert event.metadata == {"step": 10}

    def test_invalid_json_falls_back_to_raw(self):
        frame = {"event": "log", "id": "", "data": "not json"}
        event = _frame_to_log_event(frame)
        assert event.message == "not json"
        assert event.level == "INFO"

    def test_missing_optional_fields_use_defaults(self):
        frame = {"event": "progress", "data": json.dumps({})}
        event = _frame_to_log_event(frame)
        assert event.id == ""
        assert event.level == "INFO"
        assert event.message == ""
        assert event.timestamp == ""

    def test_uses_msg_fallback(self):
        frame = {
            "event": "log",
            "id": "",
            "data": json.dumps({"msg": "alt message"}),
        }
        event = _frame_to_log_event(frame)
        assert event.message == "alt message"

    def test_uses_timestamp_fallback(self):
        frame = {
            "event": "log",
            "id": "",
            "data": json.dumps({"timestamp": "2024-06-15T12:00:00"}),
        }
        event = _frame_to_log_event(frame)
        assert event.timestamp == "2024-06-15T12:00:00"


# ---------------------------------------------------------------------------
# LogEvent
# ---------------------------------------------------------------------------
class TestLogEvent:
    def test_frozen(self):
        event = LogEvent(id="1", type="log", level="INFO", message="hi", timestamp="")
        with pytest.raises(FrozenInstanceError):
            event.message = "changed"

    def test_default_metadata(self):
        event = LogEvent(id="1", type="log", level="INFO", message="hi", timestamp="")
        assert event.metadata == {}

    def test_metadata_provided(self):
        event = LogEvent(
            id="1",
            type="progress",
            level="INFO",
            message="step",
            timestamp="",
            metadata={"step": 5, "total": 100},
        )
        assert event.metadata["step"] == 5


# ---------------------------------------------------------------------------
# EventStream
# ---------------------------------------------------------------------------
class TestEventStream:
    def _make_sse_response(self, sse_lines, status_code=200):
        """Create a mock httpx response that streams SSE lines."""
        mock_resp = MagicMock()
        mock_resp.status_code = status_code
        mock_resp.is_success = status_code < 400
        mock_resp.iter_lines.return_value = iter(sse_lines)
        mock_resp.iter_text.return_value = iter(["error text"])
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        return mock_resp

    def test_yields_log_events(self):
        sse_lines = [
            "data: " + json.dumps({"level": "INFO", "message": "hello"}),
            "",
            "event: complete",
            "data: " + json.dumps({"message": "done"}),
            "",
        ]
        mock_resp = self._make_sse_response(sse_lines)
        mock_http = MagicMock()
        mock_http.stream.return_value = mock_resp

        stream = EventStream(mock_http, "/events", {"Authorization": "Bearer tok"})
        events = list(stream)
        assert len(events) == 2
        assert events[0].type == "log"
        assert events[1].type == "complete"

    def test_stops_on_complete_event(self):
        sse_lines = [
            "event: complete",
            "data: " + json.dumps({"message": "done"}),
            "",
            "data: " + json.dumps({"message": "should not appear"}),
            "",
        ]
        mock_resp = self._make_sse_response(sse_lines)
        mock_http = MagicMock()
        mock_http.stream.return_value = mock_resp

        stream = EventStream(mock_http, "/events", {})
        events = list(stream)
        assert len(events) == 1
        assert events[0].type == "complete"

    def test_stops_on_fatal_error(self):
        sse_lines = [
            "event: error",
            "data: " + json.dumps({"message": "crash", "fatal": True}),
            "",
        ]
        mock_resp = self._make_sse_response(sse_lines)
        mock_http = MagicMock()
        mock_http.stream.return_value = mock_resp

        stream = EventStream(mock_http, "/events", {})
        events = list(stream)
        assert len(events) == 1
        assert events[0].type == "error"

    def test_non_fatal_error_continues(self):
        sse_lines = [
            "event: error",
            "data: " + json.dumps({"message": "warning", "fatal": False}),
            "",
            "event: complete",
            "data: " + json.dumps({"message": "done"}),
            "",
        ]
        mock_resp = self._make_sse_response(sse_lines)
        mock_http = MagicMock()
        mock_http.stream.return_value = mock_resp

        stream = EventStream(mock_http, "/events", {})
        events = list(stream)
        assert len(events) == 2

    def test_401_raises_auth_error(self):
        mock_resp = self._make_sse_response([], status_code=401)
        mock_http = MagicMock()
        mock_http.stream.return_value = mock_resp

        stream = EventStream(mock_http, "/events", {})
        with pytest.raises(ArenaAuthError, match="401"):
            list(stream)

    def test_non_success_raises_api_error(self):
        mock_resp = self._make_sse_response([], status_code=500)
        mock_http = MagicMock()
        mock_http.stream.return_value = mock_resp

        stream = EventStream(mock_http, "/events", {})
        with pytest.raises(ArenaAPIError) as exc_info:
            list(stream)
        assert exc_info.value.status_code == 500

    def test_reconnects_on_http_error(self):
        sse_lines_final = [
            "event: complete",
            "data: " + json.dumps({"message": "done"}),
            "",
        ]
        good_resp = self._make_sse_response(sse_lines_final)
        mock_http = MagicMock()
        mock_http.stream.side_effect = [
            httpx.ConnectError("lost"),
            good_resp,
        ]

        with patch("agilerl.arena.logs.time.sleep"):
            stream = EventStream(
                mock_http, "/events", {}, max_retries=3, retry_backoff=0.01
            )
            events = list(stream)

        assert len(events) == 1
        assert events[0].type == "complete"
        assert mock_http.stream.call_count == 2

    def test_exceeds_max_retries(self):
        mock_http = MagicMock()
        mock_http.stream.side_effect = httpx.ConnectError("lost")

        with patch("agilerl.arena.logs.time.sleep"):
            stream = EventStream(
                mock_http, "/events", {}, max_retries=2, retry_backoff=0.01
            )
            with pytest.raises(ArenaAPIError, match="retries"):
                list(stream)

        assert mock_http.stream.call_count == 3  # initial + 2 retries

    def test_last_event_id_sent_on_reconnect(self):
        sse_lines_1 = [
            "id: evt-5",
            "data: " + json.dumps({"message": "partial"}),
            "",
        ]
        resp1 = self._make_sse_response(sse_lines_1)
        # After yielding the event, the iterator ends and the stream reconnects
        # Simulate a ConnectionError on second attempt then success on third
        sse_lines_2 = [
            "event: complete",
            "data: " + json.dumps({"message": "done"}),
            "",
        ]
        resp2 = self._make_sse_response(sse_lines_2)

        mock_http = MagicMock()
        mock_http.stream.side_effect = [resp1, resp2]

        with patch("agilerl.arena.logs.time.sleep"):
            stream = EventStream(
                mock_http,
                "/events",
                {"Authorization": "Bearer tok"},
                max_retries=5,
                retry_backoff=0.01,
            )
            events = list(stream)

        # The second call should include Last-Event-ID
        second_call_kwargs = mock_http.stream.call_args_list[1]
        headers = second_call_kwargs.kwargs.get("headers") or second_call_kwargs[1].get(
            "headers", {}
        )
        assert headers.get("Last-Event-ID") == "evt-5"

    def test_close_sets_finished(self):
        mock_http = MagicMock()
        stream = EventStream(mock_http, "/events", {})
        assert not stream._finished
        stream.close()
        assert stream._finished

    def test_context_manager(self):
        mock_http = MagicMock()
        stream = EventStream(mock_http, "/events", {})
        with stream as s:
            assert s is stream
        assert stream._finished


# ---------------------------------------------------------------------------
# LogDisplay
# ---------------------------------------------------------------------------
class TestLogDisplay:
    def _make_event(self, type_="log", **kwargs):
        defaults = {
            "id": "1",
            "type": type_,
            "level": "INFO",
            "message": "test",
            "timestamp": "2024-01-01T00:00:00",
            "metadata": {},
        }
        defaults.update(kwargs)
        return LogEvent(**defaults)

    def test_render_dispatches_by_type(self):
        display = LogDisplay(console=MagicMock())
        for event_type in ("log", "progress", "complete", "error"):
            handler_name = f"_render_{event_type}"
            with patch.object(display, handler_name) as mock_handler:
                event = self._make_event(type_=event_type)
                display.render(event)
                mock_handler.assert_called_once_with(event)

    def test_render_unknown_type_falls_back_to_log(self):
        display = LogDisplay(console=MagicMock())
        with patch.object(display, "_render_log") as mock_log:
            event = self._make_event(type_="unknown")
            display.render(event)
            mock_log.assert_called_once()

    def test_render_log_prints(self):
        console = MagicMock()
        display = LogDisplay(console=console)
        event = self._make_event(type_="log", level="WARN", message="low disk")
        display._render_log(event)
        console.print.assert_called_once()
        printed = console.print.call_args[0][0]
        assert "low disk" in printed

    def test_render_progress_creates_bar(self):
        console = MagicMock()
        display = LogDisplay(console=console)
        event = self._make_event(
            type_="progress",
            metadata={"step": 50, "total": 100, "best_fitness": 0.9},
        )
        display._render_progress(event)
        assert display._progress is not None
        assert display._task_id is not None

    def test_render_complete_stores_result(self):
        console = MagicMock()
        display = LogDisplay(console=console)
        event = self._make_event(
            type_="complete",
            message="Training done",
            metadata={"result": {"accuracy": 0.99}},
        )
        display._render_complete(event)
        assert display.result == {"accuracy": 0.99}

    def test_render_complete_stops_progress(self):
        console = MagicMock()
        display = LogDisplay(console=console)
        # First create a progress bar
        progress_event = self._make_event(
            type_="progress", metadata={"step": 1, "total": 10}
        )
        display._render_progress(progress_event)
        assert display._progress is not None

        complete_event = self._make_event(type_="complete", metadata={"status": "done"})
        display._render_complete(complete_event)
        assert display._progress is None

    def test_render_error_prints_panel(self):
        console = MagicMock()
        display = LogDisplay(console=console)
        event = self._make_event(
            type_="error",
            message="Something went wrong",
            metadata={"fatal": True},
        )
        display._render_error(event)
        console.print.assert_called_once()

    def test_stop_cleans_up(self):
        console = MagicMock()
        display = LogDisplay(console=console)
        progress_event = self._make_event(
            type_="progress", metadata={"step": 1, "total": 10}
        )
        display._render_progress(progress_event)
        display.stop()
        assert display._progress is None
        assert display._task_id is None

    def test_stop_when_no_progress_is_safe(self):
        display = LogDisplay(console=MagicMock())
        display.stop()  # Should not raise

    def test_result_default_empty(self):
        display = LogDisplay()
        assert display.result == {}

    def test_render_complete_without_result_key(self):
        console = MagicMock()
        display = LogDisplay(console=console)
        event = self._make_event(
            type_="complete",
            metadata={"status": "completed", "duration": 120},
        )
        display._render_complete(event)
        assert display.result == {"status": "completed", "duration": 120}
