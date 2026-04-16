"""Typed event model and NDJSON stream wrapper for Arena streaming endpoints.

Handles newline-delimited JSON (NDJSON) responses from Arena streaming endpoints.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Generator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Self

if TYPE_CHECKING:
    import httpx

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class StatusEvent:
    """A stage-progression event (``kind: "status"``).

    :param stage: The stage of the event.
    :type stage: str
    :param status: The status of the event.
    :type status: str
    :param message: The message of the event.
    :type message: str
    :param parsed_message: The parsed message of the event.
    :type parsed_message: dict[str, Any] | list[Any] | None
    :param raw: The raw message of the event.
    :type raw: dict[str, Any]
    """

    stage: str
    status: str
    message: str
    parsed_message: dict[str, Any] | list[Any] | None
    raw: dict[str, Any]


@dataclass(frozen=True, slots=True)
class CheckEvent:
    """An individual validation-check result.

    :param name: The name of the check.
    :type name: str
    :param success: The success of the check.
    :type success: bool | None
    :param warnings: The warnings of the check.
    :type warnings: list[str]
    :param error: The error of the check.
    :type error: str
    :param raw: The raw message of the event.
    :type raw: dict[str, Any]
    """

    name: str
    success: bool | None
    warnings: list[str]
    error: str
    raw: dict[str, Any]


@dataclass(frozen=True, slots=True)
class CompletionEvent:
    """Terminal event indicating the operation finished.

    :param payload: The payload of the event.
    :type payload: dict[str, Any]
    """

    payload: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ErrorEvent:
    """An error returned by the server inside the stream.

    :param message: Primary error message.
    :param extras: Supplementary context (e.g. available entrypoints).
    :param raw: The full raw JSON payload.
    """

    message: str
    extras: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class LogEvent:
    """Fallback event for plain text or unrecognised JSON.

    :param text: The text of the event.
    :type text: str
    """

    text: str


StreamEvent = StatusEvent | CheckEvent | CompletionEvent | ErrorEvent | LogEvent


def _try_parse_json_message(message: str) -> dict[str, Any] | list[Any] | None:
    """Try to parse a JSON message into a dictionary or list.

    :param message: The message to parse.
    :type message: str
    :returns: The parsed dictionary or list.
    :rtype: dict[str, Any] | list[Any] | None
    """
    message = message.strip()
    if not message or message[0] not in "{[":
        return None
    try:
        parsed = json.loads(message)
    except json.JSONDecodeError:
        return None
    if isinstance(parsed, (dict, list)):
        return parsed
    return None


def parse_ndjson_line(line: str) -> StreamEvent:
    """Parse a single NDJSON line into a typed :data:`StreamEvent`.

    :param line: The NDJSON line to parse.
    :type line: str
    :returns: The parsed :data:`StreamEvent`.
    :rtype: StreamEvent
    """
    # If the line is empty, return a LogEvent with an empty text
    if not line:
        return LogEvent(text="")

    try:
        payload = json.loads(line)
    except json.JSONDecodeError:
        return LogEvent(text=line)

    if not isinstance(payload, dict):
        return LogEvent(text=json.dumps(payload, default=str))

    # StatusEvent
    if payload.get("kind") == "status":
        stage = str(payload.get("stage", "-"))
        status = str(payload.get("status", "-"))
        raw_message = str(payload.get("message", ""))
        parsed_message = _try_parse_json_message(raw_message)
        return StatusEvent(
            stage=stage,
            status=status,
            message=raw_message,
            parsed_message=parsed_message,
            raw=payload,
        )

    # CheckEvent
    if "check" in payload and isinstance(payload.get("result"), dict):
        result: dict[str, Any] = payload["result"]
        warnings_raw = result.get("warnings")
        warnings = (
            [str(w) for w in warnings_raw] if isinstance(warnings_raw, list) else []
        )
        error = str(result.get("error msg") or result.get("error") or "")
        return CheckEvent(
            name=str(payload["check"]),
            success=result.get("success"),
            warnings=warnings,
            error=error,
            raw=payload,
        )

    # CompletionEvent
    if payload.get("complete") is True:
        return CompletionEvent(payload=payload)

    # ErrorEvent — server-side error embedded in the stream
    if "error" in payload and "kind" not in payload and "check" not in payload:
        _skip = {"error", "error_code", "status", "status_code"}
        extras = {k: v for k, v in payload.items() if k not in _skip and v}
        return ErrorEvent(
            message=str(payload["error"]),
            extras=extras,
            raw=payload,
        )

    # LogEvent
    return LogEvent(text=json.dumps(payload, default=str))


def _log_event(event: StreamEvent) -> None:
    """Default stream handler that logs events at INFO level."""
    if isinstance(event, StatusEvent):
        logger.info("[%s] %s — %s", event.stage, event.status, event.message)
    elif isinstance(event, CheckEvent):
        status = "PASS" if event.success else "FAIL"
        detail = event.error or ", ".join(event.warnings) or ""
        logger.info("[check] %s — %s %s", event.name, status, detail)
    elif isinstance(event, CompletionEvent):
        logger.info("Completed.")
    elif isinstance(event, ErrorEvent):
        logger.error("%s", event.message)
        for key, value in event.extras.items():
            label = key.replace("_", " ").capitalize()
            if isinstance(value, list):
                logger.error("  %s: %s", label, ", ".join(str(v) for v in value))
            else:
                logger.error("  %s: %s", label, value)
    elif isinstance(event, LogEvent) and event.text:
        logger.info("%s", event.text)


class NDJsonStream:
    """Iterator + context-manager over an NDJSON HTTP response.

    Yields :data:`StreamEvent` objects and tracks the final result
    for convenient access after iteration.

    :param response: The HTTP response to iterate over.
    :type response: httpx.Response
    :param handler: Callback invoked for each event.  Defaults to a
        logging-based handler that emits events at ``INFO`` level.
        Pass an explicit callable to override, or ``False`` to disable.
    :type handler: Callable[[StreamEvent], None] | None | bool

    Usage::

        # Iterate events
        with client.validate_environment(name="MyEnv") as stream:
            for event in stream:
                ...
            print(stream.result)

        # Or just collect the final result
        result = client.validate_environment(name="MyEnv").collect()
    """

    _DEFAULT_HANDLER: Callable[[StreamEvent], None] = staticmethod(_log_event)  # type: ignore[assignment]

    def __init__(
        self,
        response: httpx.Response,
        *,
        handler: Callable[[StreamEvent], None] | None | bool = None,
    ) -> None:
        self._response: httpx.Response = response
        if handler is None:
            self._handler: Callable[[StreamEvent], None] | None = self._DEFAULT_HANDLER
        elif handler is False:
            self._handler = None
        else:
            self._handler = handler

        self._result: dict[str, Any] | None = None
        self._raw_chunks: list[str] = []
        self._consumed: bool = False

    def __iter__(self) -> Generator[StreamEvent, None, None]:
        """Iterate over the events in the stream.

        :returns: A generator of :data:`StreamEvent` objects.
        :rtype: Generator[StreamEvent, None, None]
        """
        buffer = ""
        try:
            # Iterate over the chunks in the response
            for chunk in self._response.iter_text():
                if not chunk:
                    continue
                self._raw_chunks.append(chunk)
                buffer += chunk
                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    line = line.strip()
                    if not line:
                        continue
                    event = parse_ndjson_line(line)
                    self._track_result(event)
                    if self._handler is not None:
                        self._handler(event)
                    yield event

            # If there is any remaining buffer, parse it as a single event
            if buffer.strip():
                event = parse_ndjson_line(buffer.strip())
                self._track_result(event)
                if self._handler is not None:
                    self._handler(event)
                yield event
        finally:
            self._consumed = True

    def collect(self) -> dict[str, Any]:
        """Consume all events and return the final result dict.

        :returns: The final result dict.
        :rtype: dict[str, Any]
        """
        # Consume all events
        for _ in self:
            pass

        # If there is a result, return it
        if self._result is not None:
            return self._result

        # If there is no result, join the raw chunks and parse as JSON
        body = "".join(self._raw_chunks).strip()
        if not body:
            return {}

        try:
            parsed = json.loads(body)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}

    @property
    def result(self) -> dict[str, Any] | None:
        """Final result payload, populated after a completion event is yielded.

        :returns: The final result dict.
        :rtype: dict[str, Any] | None
        """
        return self._result

    def close(self) -> None:
        """Close the underlying HTTP response."""
        self._response.close()

    def __enter__(self) -> Self:
        """Enter the context manager.

        :returns: The :data:`NDJsonStream` object.
        :rtype: Self
        """
        return self

    def __exit__(self, *exc: object) -> None:
        """Exit the context manager.

        :param exc: The exception that was raised.
        :type exc: object
        :returns: None
        :rtype: None
        """
        self.close()

    def _track_result(self, event: StreamEvent) -> None:
        """Track the result of the stream.

        :param event: The event to track.
        :type event: StreamEvent
        """
        if isinstance(event, CompletionEvent):
            self._result = event.payload
        elif (
            isinstance(event, StatusEvent)
            and event.status == "completed"
            and isinstance(event.parsed_message, dict)
        ):
            self._result = event.parsed_message
