"""Tests for agilerl.arena.stream — event types, parsing, and NDJsonStream."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from agilerl.arena.stream import (
    CheckEvent,
    ErrorEvent,
    LogEvent,
    NDJsonStream,
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
        assert event.detail == {}

    def test_status_event_with_detail(self):
        line = json.dumps(
            {
                "kind": "status",
                "stage": "submit",
                "status": "completed",
                "message": "Job submitted",
                "detail": {"accepted": True, "experiment_id": 42},
            }
        )
        event = parse_ndjson_line(line)
        assert isinstance(event, StatusEvent)
        assert event.status == "completed"
        assert event.detail["accepted"] is True
        assert event.detail["experiment_id"] == 42

    def test_status_failed_becomes_error_event(self):
        line = json.dumps(
            {
                "kind": "status",
                "stage": "submission",
                "status": "failed",
                "message": "Insufficient resources",
                "detail": {
                    "error_code": "JOB_REJECTED",
                    "response": {"accepted": False},
                },
            }
        )
        event = parse_ndjson_line(line)
        assert isinstance(event, ErrorEvent)
        assert event.message == "Insufficient resources"
        assert "response" in event.extras
        assert "error_code" not in event.extras

    def test_check_event_pass(self):
        line = json.dumps(
            {
                "kind": "check",
                "stage": "validation",
                "status": "running",
                "message": "imports: passed",
                "detail": {
                    "check": "imports",
                    "result": {"success": True},
                },
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
                "kind": "check",
                "stage": "validation",
                "status": "running",
                "message": "entrypoint: failed",
                "detail": {
                    "check": "entrypoint",
                    "result": {"success": False, "error": "Not found"},
                },
            }
        )
        event = parse_ndjson_line(line)
        assert isinstance(event, CheckEvent)
        assert event.success is False
        assert event.error == "Not found"

    def test_check_event_with_warnings(self):
        line = json.dumps(
            {
                "kind": "check",
                "stage": "validation",
                "status": "running",
                "message": "dependencies: passed with warnings",
                "detail": {
                    "check": "dependencies",
                    "result": {
                        "success": True,
                        "warnings": [
                            "Deprecated package: foo",
                            "Version conflict: bar",
                        ],
                    },
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
                "kind": "check",
                "stage": "validation",
                "status": "running",
                "message": "build: failed",
                "detail": {
                    "check": "build",
                    "result": {"success": False, "error msg": "Build failed"},
                },
            }
        )
        event = parse_ndjson_line(line)
        assert isinstance(event, CheckEvent)
        assert event.error == "Build failed"

    def test_unrecognized_dict(self):
        line = json.dumps({"foo": "bar", "baz": 123})
        event = parse_ndjson_line(line)
        assert isinstance(event, LogEvent)


# ---------------------------------------------------------------------------
# NDJsonStream
# ---------------------------------------------------------------------------


def _make_mock_response(lines: list[str]) -> MagicMock:
    """Create a mock httpx.Response that yields *lines* as text chunks."""
    body = "\n".join(lines) + "\n"
    mock = MagicMock()
    mock.iter_text.return_value = iter([body])
    mock.close = MagicMock()
    return mock


class TestNDJsonStream:
    def test_iterate_events(self):
        lines = [
            json.dumps(
                {
                    "kind": "status",
                    "stage": "upload",
                    "status": "started",
                    "message": "Uploading environment",
                }
            ),
            json.dumps(
                {
                    "kind": "check",
                    "stage": "validation",
                    "status": "running",
                    "message": "imports: passed",
                    "detail": {"check": "imports", "result": {"success": True}},
                }
            ),
            json.dumps(
                {
                    "kind": "status",
                    "stage": "validation",
                    "status": "completed",
                    "message": "Environment validated successfully",
                    "detail": {"env_info": {"env_name": "MyEnv"}},
                }
            ),
        ]
        resp = _make_mock_response(lines)
        stream = NDJsonStream(resp)
        events = list(stream)

        assert len(events) == 3
        assert isinstance(events[0], StatusEvent)
        assert isinstance(events[1], CheckEvent)
        assert isinstance(events[2], StatusEvent)
        assert events[2].status == "completed"

    def test_result_from_status_completed_with_detail(self):
        lines = [
            json.dumps(
                {
                    "kind": "status",
                    "stage": "submission",
                    "status": "completed",
                    "message": "Job submitted",
                    "detail": {"accepted": True, "experiment_id": 7},
                }
            ),
        ]
        resp = _make_mock_response(lines)
        stream = NDJsonStream(resp)
        list(stream)

        assert stream.result is not None
        assert stream.result["accepted"] is True

    def test_result_not_set_for_completed_without_detail(self):
        lines = [
            json.dumps(
                {
                    "kind": "status",
                    "stage": "upload",
                    "status": "completed",
                    "message": "Upload done",
                }
            ),
        ]
        resp = _make_mock_response(lines)
        stream = NDJsonStream(resp)
        list(stream)

        assert stream.result is None

    def test_collect_returns_result(self):
        lines = [
            json.dumps(
                {
                    "kind": "status",
                    "stage": "validation",
                    "status": "completed",
                    "message": "Done",
                    "detail": {"answer": 42},
                }
            ),
        ]
        resp = _make_mock_response(lines)
        stream = NDJsonStream(resp)
        result = stream.collect()

        assert result["answer"] == 42

    def test_collect_fallback_single_json(self):
        """Server returns a single JSON response (not NDJSON)."""
        mock = MagicMock()
        mock.iter_text.return_value = iter(['{"valid": true}'])
        mock.close = MagicMock()
        stream = NDJsonStream(mock)
        result = stream.collect()

        assert result == {"valid": True}

    def test_collect_empty_response(self):
        mock = MagicMock()
        mock.iter_text.return_value = iter([""])
        mock.close = MagicMock()
        stream = NDJsonStream(mock)

        assert stream.collect() == {}

    def test_handler_called_for_each_event(self):
        lines = [
            json.dumps(
                {"kind": "status", "stage": "a", "status": "running", "message": "x"}
            ),
            json.dumps(
                {
                    "kind": "check",
                    "stage": "validation",
                    "status": "running",
                    "message": "b: passed",
                    "detail": {"check": "b", "result": {"success": True}},
                }
            ),
        ]
        resp = _make_mock_response(lines)
        handler = MagicMock()
        stream = NDJsonStream(resp, handler=handler)
        list(stream)

        assert handler.call_count == 2
        args = [call.args[0] for call in handler.call_args_list]
        assert isinstance(args[0], StatusEvent)
        assert isinstance(args[1], CheckEvent)

    def test_context_manager_closes_response(self):
        resp = _make_mock_response(
            [
                json.dumps(
                    {
                        "kind": "status",
                        "stage": "done",
                        "status": "completed",
                        "message": "Done",
                    }
                )
            ]
        )
        with NDJsonStream(resp) as stream:
            list(stream)
        resp.close.assert_called_once()

    def test_chunked_delivery(self):
        """Data arriving in arbitrary chunks that split across line boundaries."""
        line1 = json.dumps(
            {"kind": "status", "stage": "a", "status": "running", "message": "m"}
        )
        line2 = json.dumps(
            {
                "kind": "status",
                "stage": "a",
                "status": "completed",
                "message": "done",
                "detail": {"ok": True},
            }
        )
        full = line1 + "\n" + line2 + "\n"
        mid = len(full) // 2
        chunk1, chunk2 = full[:mid], full[mid:]

        mock = MagicMock()
        mock.iter_text.return_value = iter([chunk1, chunk2])
        mock.close = MagicMock()

        stream = NDJsonStream(mock)
        events = list(stream)
        assert len(events) == 2
        assert isinstance(events[0], StatusEvent)
        assert isinstance(events[1], StatusEvent)
        assert events[1].status == "completed"

    def test_failed_status_becomes_error_event(self):
        lines = [
            json.dumps(
                {
                    "kind": "status",
                    "stage": "submission",
                    "status": "failed",
                    "message": "Insufficient resources",
                    "detail": {
                        "error_code": "JOB_REJECTED",
                        "response": {"accepted": False, "minimumResources": [4]},
                    },
                }
            ),
        ]
        resp = _make_mock_response(lines)
        stream = NDJsonStream(resp)
        events = list(stream)

        assert len(events) == 1
        assert isinstance(events[0], ErrorEvent)
        assert events[0].message == "Insufficient resources"
        assert "response" in events[0].extras
