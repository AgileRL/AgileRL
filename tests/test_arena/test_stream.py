"""Tests for agilerl.arena.stream — event types, parsing, and NdjsonStream."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from agilerl.arena.stream import (
    CheckEvent,
    CompletionEvent,
    LogEvent,
    NdjsonStream,
    StatusEvent,
    StreamEvent,
    parse_ndjson_line,
)


# ---------------------------------------------------------------------------
# parse_ndjson_line
# ---------------------------------------------------------------------------
class TestParseNdjsonLine:
    def test_empty_line(self):
        event = parse_ndjson_line("")
        assert isinstance(event, LogEvent)
        assert event.text == ""

    def test_plain_text(self):
        event = parse_ndjson_line("some log message")
        assert isinstance(event, LogEvent)
        assert event.text == "some log message"

    def test_invalid_json(self):
        event = parse_ndjson_line("{broken json")
        assert isinstance(event, LogEvent)
        assert event.text == "{broken json"

    def test_non_dict_json(self):
        event = parse_ndjson_line("[1, 2, 3]")
        assert isinstance(event, LogEvent)
        assert "1" in event.text

    def test_status_event(self):
        line = json.dumps(
            {
                "kind": "status",
                "stage": "install",
                "status": "running",
                "message": "Installing packages...",
            }
        )
        event = parse_ndjson_line(line)
        assert isinstance(event, StatusEvent)
        assert event.stage == "install"
        assert event.status == "running"
        assert event.message == "Installing packages..."
        assert event.parsed_message is None

    def test_status_event_with_json_message(self):
        inner = json.dumps({"accepted": True, "experiment_id": 42})
        line = json.dumps(
            {
                "kind": "status",
                "stage": "submit",
                "status": "completed",
                "message": inner,
            }
        )
        event = parse_ndjson_line(line)
        assert isinstance(event, StatusEvent)
        assert event.status == "completed"
        assert isinstance(event.parsed_message, dict)
        assert event.parsed_message["accepted"] is True

    def test_check_event_pass(self):
        line = json.dumps(
            {
                "check": "imports",
                "result": {"success": True},
            }
        )
        event = parse_ndjson_line(line)
        assert isinstance(event, CheckEvent)
        assert event.name == "imports"
        assert event.success is True
        assert event.warnings == []
        assert event.error == ""

    def test_check_event_fail_with_error(self):
        line = json.dumps(
            {
                "check": "entrypoint",
                "result": {"success": False, "error": "Not found"},
            }
        )
        event = parse_ndjson_line(line)
        assert isinstance(event, CheckEvent)
        assert event.success is False
        assert event.error == "Not found"

    def test_check_event_with_warnings(self):
        line = json.dumps(
            {
                "check": "dependencies",
                "result": {
                    "success": True,
                    "warnings": ["Deprecated package: foo", "Version conflict: bar"],
                },
            }
        )
        event = parse_ndjson_line(line)
        assert isinstance(event, CheckEvent)
        assert event.success is True
        assert len(event.warnings) == 2

    def test_check_event_error_msg_key(self):
        line = json.dumps(
            {
                "check": "build",
                "result": {"success": False, "error msg": "Build failed"},
            }
        )
        event = parse_ndjson_line(line)
        assert isinstance(event, CheckEvent)
        assert event.error == "Build failed"

    def test_completion_event(self):
        line = json.dumps(
            {
                "complete": True,
                "env_info": {"env_name": "CartPole-v1"},
            }
        )
        event = parse_ndjson_line(line)
        assert isinstance(event, CompletionEvent)
        assert event.payload["complete"] is True
        assert event.payload["env_info"]["env_name"] == "CartPole-v1"

    def test_unrecognized_dict(self):
        line = json.dumps({"foo": "bar", "baz": 123})
        event = parse_ndjson_line(line)
        assert isinstance(event, LogEvent)


# ---------------------------------------------------------------------------
# NdjsonStream
# ---------------------------------------------------------------------------


def _make_mock_response(lines: list[str]) -> MagicMock:
    """Create a mock httpx.Response that yields *lines* as text chunks."""
    body = "\n".join(lines) + "\n"
    mock = MagicMock()
    mock.iter_text.return_value = iter([body])
    mock.close = MagicMock()
    return mock


class TestNdjsonStream:
    def test_iterate_events(self):
        lines = [
            json.dumps(
                {
                    "kind": "status",
                    "stage": "install",
                    "status": "running",
                    "message": "go",
                }
            ),
            json.dumps({"check": "imports", "result": {"success": True}}),
            json.dumps({"complete": True, "env_info": {"env_name": "MyEnv"}}),
        ]
        resp = _make_mock_response(lines)
        stream = NdjsonStream(resp)
        events = list(stream)

        assert len(events) == 3
        assert isinstance(events[0], StatusEvent)
        assert isinstance(events[1], CheckEvent)
        assert isinstance(events[2], CompletionEvent)

    def test_result_from_completion_event(self):
        lines = [
            json.dumps({"check": "imports", "result": {"success": True}}),
            json.dumps({"complete": True, "validated": True}),
        ]
        resp = _make_mock_response(lines)
        stream = NdjsonStream(resp)
        list(stream)

        assert stream.result is not None
        assert stream.result["complete"] is True

    def test_result_from_status_completed(self):
        inner = json.dumps({"accepted": True, "experiment_id": 7})
        lines = [
            json.dumps(
                {
                    "kind": "status",
                    "stage": "submit",
                    "status": "completed",
                    "message": inner,
                }
            ),
        ]
        resp = _make_mock_response(lines)
        stream = NdjsonStream(resp)
        list(stream)

        assert stream.result is not None
        assert stream.result["accepted"] is True

    def test_collect_returns_result(self):
        lines = [
            json.dumps({"complete": True, "answer": 42}),
        ]
        resp = _make_mock_response(lines)
        stream = NdjsonStream(resp)
        result = stream.collect()

        assert result["answer"] == 42

    def test_collect_fallback_single_json(self):
        """Server returns a single JSON response (not NDJSON)."""
        mock = MagicMock()
        mock.iter_text.return_value = iter(['{"valid": true}'])
        mock.close = MagicMock()
        stream = NdjsonStream(mock)
        result = stream.collect()

        assert result == {"valid": True}

    def test_collect_empty_response(self):
        mock = MagicMock()
        mock.iter_text.return_value = iter([""])
        mock.close = MagicMock()
        stream = NdjsonStream(mock)

        assert stream.collect() == {}

    def test_handler_called_for_each_event(self):
        lines = [
            json.dumps(
                {"kind": "status", "stage": "a", "status": "running", "message": "x"}
            ),
            json.dumps({"check": "b", "result": {"success": True}}),
        ]
        resp = _make_mock_response(lines)
        handler = MagicMock()
        stream = NdjsonStream(resp, handler=handler)
        list(stream)

        assert handler.call_count == 2
        args = [call.args[0] for call in handler.call_args_list]
        assert isinstance(args[0], StatusEvent)
        assert isinstance(args[1], CheckEvent)

    def test_context_manager_closes_response(self):
        resp = _make_mock_response([json.dumps({"complete": True})])
        with NdjsonStream(resp) as stream:
            list(stream)
        resp.close.assert_called_once()

    def test_chunked_delivery(self):
        """Data arriving in arbitrary chunks that split across line boundaries."""
        line1 = json.dumps(
            {"kind": "status", "stage": "a", "status": "ok", "message": "m"}
        )
        line2 = json.dumps({"complete": True})
        full = line1 + "\n" + line2 + "\n"
        mid = len(full) // 2
        chunk1, chunk2 = full[:mid], full[mid:]

        mock = MagicMock()
        mock.iter_text.return_value = iter([chunk1, chunk2])
        mock.close = MagicMock()

        stream = NdjsonStream(mock)
        events = list(stream)
        assert len(events) == 2
        assert isinstance(events[0], StatusEvent)
        assert isinstance(events[1], CompletionEvent)
