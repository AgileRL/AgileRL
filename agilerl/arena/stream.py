"""Typed event model and NDJSON stream wrapper for Arena streaming endpoints.

Handles newline-delimited JSON (NDJSON) responses from Arena streaming endpoints.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Generator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Self

from agilerl.arena.exceptions import _sanitize_detail

if TYPE_CHECKING:
    import httpx


@dataclass(frozen=True, slots=True)
class StatusEvent:
    """A stage-progression or warning event (``kind: "status"`` or ``kind: "warning"``).

    :param stage: The stage of the event.
    :type stage: str
    :param status: The status of the event.
    :type status: str
    :param message: Human-readable message.
    :type message: str
    :param detail: Structured payload from the ``detail`` envelope field.
    :type detail: dict[str, Any]
    :param raw: The full raw JSON payload.
    :type raw: dict[str, Any]
    :param kind: The envelope ``kind`` field (``"status"`` or ``"warning"``).
    :type kind: str
    """

    stage: str
    status: str
    message: str
    detail: dict[str, Any]
    raw: dict[str, Any]
    kind: str = "status"


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


StreamEvent = StatusEvent | CheckEvent | ErrorEvent | LogEvent


def parse_ndjson_line(line: str) -> StreamEvent:
    """Parse a single NDJSON line into a typed :data:`StreamEvent`.

    Every line from the backend is a uniform envelope with a ``kind`` field.
    Dispatch order:

    1. ``kind:"status"`` + ``status:"failed"`` -> :class:`ErrorEvent`
    2. ``kind:"status"`` (other statuses) -> :class:`StatusEvent`
    3. ``kind:"check"`` -> :class:`CheckEvent`
    4. Anything else -> :class:`LogEvent`

    :param line: The NDJSON line to parse.
    :type line: str
    :returns: The parsed :data:`StreamEvent`.
    :rtype: StreamEvent
    """
    if not line:
        return LogEvent(text="")

    try:
        payload = json.loads(line)
    except json.JSONDecodeError:
        return LogEvent(text=line)

    if not isinstance(payload, dict):
        return LogEvent(text=json.dumps(payload, default=str))

    kind = payload.get("kind")
    detail: dict[str, Any] = payload.get("detail") or {}

    # Status (INFO) and warning logs
    if kind in ("status", "warning"):
        status = str(payload.get("status", "-"))
        message = str(payload.get("message", ""))

        if status == "failed":
            _skip = {"error_code"}
            extras = {k: v for k, v in detail.items() if k not in _skip and v}
            return ErrorEvent(
                message=_sanitize_detail(message), extras=extras, raw=payload
            )

        return StatusEvent(
            stage=str(payload.get("stage", "-")),
            status=status,
            message=message,
            detail=detail,
            raw=payload,
            kind=kind,
        )

    # This is unique to environment validation checks
    if kind == "check":
        result_raw = detail.get("result")
        if not isinstance(result_raw, dict):
            return StatusEvent(
                stage=str(payload.get("stage", "-")),
                status=str(payload.get("status", "-")),
                message=str(payload.get("message", "")),
                detail=detail,
                raw=payload,
            )
        warnings_raw = result_raw.get("warnings")
        warnings = (
            [str(w) for w in warnings_raw] if isinstance(warnings_raw, list) else []
        )
        error = str(result_raw.get("error msg") or result_raw.get("error") or "")
        return CheckEvent(
            name=str(detail.get("check", "-")),
            success=result_raw.get("success"),
            warnings=warnings,
            error=error,
            raw=payload,
        )

    return LogEvent(text=json.dumps(payload, default=str))


class NDJsonStream:
    """Iterator + context-manager over an NDJSON HTTP response.

    Yields :data:`StreamEvent` objects and tracks the final result
    for convenient access after iteration.

    :param response: The HTTP response to iterate over.
    :type response: httpx.Response
    :param handler: Callback invoked for each event.  ``None`` means silent.
    :type handler: Callable[[StreamEvent], None] | None
    :param renderer: Optional renderer to close when the stream is fully consumed.
    :type renderer: object | None

    Usage::

        # Iterate events
        with client.validate_environment(name="MyEnv") as stream:
            for event in stream:
                ...
            print(stream.result)

        # Or just collect the final result
        result = client.validate_environment(name="MyEnv").collect()
    """

    def __init__(
        self,
        response: httpx.Response,
        *,
        handler: Callable[[StreamEvent], None] | None = None,
        renderer: Any | None = None,
    ) -> None:
        self._response: httpx.Response = response
        self._handler: Callable[[StreamEvent], None] | None = handler
        self._renderer = renderer

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

        self._close_renderer()

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
        """Close the renderer (if any) and the underlying HTTP response."""
        self._close_renderer()
        self._response.close()

    def _close_renderer(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

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
        if (
            isinstance(event, StatusEvent)
            and event.status == "completed"
            and event.detail
        ):
            self._result = event.detail
