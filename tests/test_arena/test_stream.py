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

    def test_warning_event(self):
        """``kind:"warning"`` produces a StatusEvent with kind="warning"."""
        line = json.dumps(
            {
                "kind": "warning",
                "stage": "submission",
                "status": "running",
                "message": "Low credit balance",
            }
        )
        event = parse_ndjson_line(line)
        assert isinstance(event, StatusEvent)
        assert event.kind == "warning"
        assert event.message == "Low credit balance"
        assert event.stage == "submission"

    def test_warning_failed_becomes_error_event(self):
        """``kind:"warning"`` with ``status:"failed"`` still becomes ErrorEvent."""
        line = json.dumps(
            {
                "kind": "warning",
                "stage": "submission",
                "status": "failed",
                "message": "Fatal warning",
                "detail": {"info": "details"},
            }
        )
        event = parse_ndjson_line(line)
        assert isinstance(event, ErrorEvent)
        assert event.message == "Fatal warning"

    def test_failed_event_sanitizes_internal_url(self):
        """Internal service URLs in failed-status messages are replaced."""
        line = json.dumps(
            {
                "kind": "status",
                "stage": "validation",
                "status": "failed",
                "message": (
                    "Environment creation failed: Failed to call list-entrypoints: "
                    "error sending request for url "
                    "(http://env-validator:8080/api/v1/validations/custom-envs/list-entrypoints)"
                ),
            }
        )
        event = parse_ndjson_line(line)
        assert isinstance(event, ErrorEvent)
        assert "env-validator" not in event.message
        assert "Something went wrong" in event.message

    def test_check_without_dict_result_falls_back_to_status(self):
        """``kind:"check"`` with non-dict ``detail.result`` falls back to StatusEvent."""
        line = json.dumps(
            {
                "kind": "check",
                "stage": "validation",
                "status": "running",
                "message": "Malformed check",
                "detail": {"check": "imports", "result": "not-a-dict"},
            }
        )
        event = parse_ndjson_line(line)
        assert isinstance(event, StatusEvent)
        assert event.stage == "validation"
        assert event.message == "Malformed check"

    def test_check_with_null_success(self):
        """Explicit ``null`` success in JSON produces CheckEvent(success=None)."""
        line = json.dumps(
            {
                "kind": "check",
                "stage": "validation",
                "status": "running",
                "message": "unknown check",
                "detail": {
                    "check": "seed",
                    "result": {"success": None},
                },
            }
        )
        event = parse_ndjson_line(line)
        assert isinstance(event, CheckEvent)
        assert event.success is None
        assert event.name == "seed"

    def test_check_missing_result_key_falls_back_to_status(self):
        """``kind:"check"`` with no ``result`` key falls back to StatusEvent."""
        line = json.dumps(
            {
                "kind": "check",
                "stage": "validation",
                "status": "running",
                "message": "No result",
                "detail": {"check": "imports"},
            }
        )
        event = parse_ndjson_line(line)
        assert isinstance(event, StatusEvent)


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

    def test_double_iteration_yields_nothing_second_time(self):
        """Iterating a consumed stream yields no events (W1 documentation)."""
        lines = [
            json.dumps(
                {"kind": "status", "stage": "a", "status": "running", "message": "x"}
            ),
        ]
        resp = _make_mock_response(lines)
        resp.iter_text.return_value = iter(
            [
                json.dumps(
                    {
                        "kind": "status",
                        "stage": "a",
                        "status": "running",
                        "message": "x",
                    }
                )
                + "\n"
            ]
        )
        stream = NDJsonStream(resp)
        first = list(stream)
        assert len(first) == 1
        assert stream._consumed is True

        resp.iter_text.return_value = iter([])
        second = list(stream)
        assert second == []

    def test_direct_close_without_context_manager(self):
        """Calling close() directly still closes the HTTP response."""
        resp = _make_mock_response([])
        stream = NDJsonStream(resp)
        stream.close()
        resp.close.assert_called_once()

    def test_collect_fallback_on_unparseable_body(self):
        """When raw chunks are non-JSON text and no result tracked, collect returns {}."""
        mock = MagicMock()
        mock.iter_text.return_value = iter(["not valid json at all\n"])
        mock.close = MagicMock()
        stream = NDJsonStream(mock)
        assert stream.collect() == {}

    def test_trailing_buffer_without_newline(self):
        """Server sends JSON without trailing newline — event is still yielded."""
        payload = json.dumps(
            {
                "kind": "status",
                "stage": "upload",
                "status": "completed",
                "message": "Done",
                "detail": {"ok": True},
            }
        )
        mock = MagicMock()
        mock.iter_text.return_value = iter([payload])  # no trailing \n
        mock.close = MagicMock()

        stream = NDJsonStream(mock)
        events = list(stream)

        assert len(events) == 1
        assert isinstance(events[0], StatusEvent)
        assert events[0].status == "completed"
        assert stream.result == {"ok": True}

    def test_error_event_mid_stream_preserves_order(self):
        """ErrorEvent interleaved between CheckEvents — all yielded in order."""
        lines = [
            json.dumps(
                {
                    "kind": "check",
                    "stage": "validation",
                    "status": "running",
                    "message": "check1: passed",
                    "detail": {"check": "check1", "result": {"success": True}},
                }
            ),
            json.dumps(
                {
                    "kind": "status",
                    "stage": "validation",
                    "status": "failed",
                    "message": "Profiling failed",
                    "detail": {"error_code": "PROF_ERR"},
                }
            ),
            json.dumps(
                {
                    "kind": "check",
                    "stage": "validation",
                    "status": "running",
                    "message": "check2: passed",
                    "detail": {"check": "check2", "result": {"success": True}},
                }
            ),
        ]
        resp = _make_mock_response(lines)
        stream = NDJsonStream(resp)
        events = list(stream)

        assert len(events) == 3
        assert isinstance(events[0], CheckEvent)
        assert isinstance(events[1], ErrorEvent)
        assert isinstance(events[2], CheckEvent)

    def test_multiple_completed_events_keeps_last_result(self):
        """Two completed events — result is the *last* one (documents W2)."""
        lines = [
            json.dumps(
                {
                    "kind": "status",
                    "stage": "validation",
                    "status": "completed",
                    "message": "Validation done",
                    "detail": {"phase": "validation", "passed": True},
                }
            ),
            json.dumps(
                {
                    "kind": "status",
                    "stage": "profiling",
                    "status": "completed",
                    "message": "Profiling done",
                    "detail": {"phase": "profiling", "cpu": 0.5},
                }
            ),
        ]
        resp = _make_mock_response(lines)
        stream = NDJsonStream(resp)
        list(stream)

        assert stream.result is not None
        assert stream.result["phase"] == "profiling"

    def test_collect_calls_renderer_close(self):
        """collect() calls renderer.close() after consuming events."""
        lines = [
            json.dumps(
                {"kind": "status", "stage": "a", "status": "running", "message": "m"}
            ),
        ]
        resp = _make_mock_response(lines)
        renderer = MagicMock()
        stream = NDJsonStream(resp, renderer=renderer)
        stream.collect()

        renderer.close.assert_called_once()

    def test_collect_with_renderer_that_was_already_closed(self):
        """collect() sets renderer to None after closing — safe on double call (W4)."""
        lines = [
            json.dumps(
                {
                    "kind": "status",
                    "stage": "a",
                    "status": "completed",
                    "message": "Done",
                    "detail": {"ok": True},
                }
            ),
        ]
        resp = _make_mock_response(lines)
        renderer = MagicMock()
        stream = NDJsonStream(resp, renderer=renderer)
        stream.collect()

        assert stream._renderer is None
        renderer.close.assert_called_once()
