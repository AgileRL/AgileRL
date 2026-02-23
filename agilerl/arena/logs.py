from __future__ import annotations

import json
import logging
import time
from collections.abc import Generator, Iterator
from dataclasses import dataclass, field
from typing import Any, Literal, Self

import httpx
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.text import Text

from agilerl.arena.exceptions import ArenaAPIError, ArenaAuthError

logger = logging.getLogger(__name__)

_LEVEL_STYLES: dict[str, str] = {
    "DEBUG": "dim",
    "INFO": "",
    "WARN": "yellow",
    "WARNING": "yellow",
    "ERROR": "bold red",
}


@dataclass(frozen=True, slots=True)
class LogEvent:
    """Structured representation of a single SSE event from Arena.

    :param id: Server-assigned event identifier used for reconnection.
    :type id: str
    :param type: Event category.
    :type type: str
    :param level: Log severity (DEBUG, INFO, WARN, ERROR).
    :type level: str
    :param message: Human-readable event description.
    :type message: str
    :param timestamp: ISO-8601 timestamp from the server.
    :type timestamp: str
    :param metadata: Arbitrary structured data attached to the event.
    :type metadata: dict[str, Any]
    """

    id: str
    type: Literal["log", "progress", "complete", "error"]
    level: str
    message: str
    timestamp: str
    metadata: dict[str, Any] = field(default_factory=dict)


def _parse_sse_events(lines: Iterator[str]) -> Generator[dict[str, str], None, None]:
    """Parse raw SSE text lines into event dicts.

    Yields one dict per complete event frame with keys ``event``,
    ``id``, and ``data``.
    """
    event_type = "log"
    event_id = ""
    data_parts: list[str] = []

    for raw_line in lines:
        line = raw_line.rstrip("\n\r")

        if line.startswith(":"):
            continue

        if not line:
            if data_parts:
                yield {
                    "event": event_type,
                    "id": event_id,
                    "data": "\n".join(data_parts),
                }
                event_type = "log"
                event_id = ""
                data_parts = []
            continue

        if line.startswith("event:"):
            event_type = line[len("event:") :].strip()
        elif line.startswith("id:"):
            event_id = line[len("id:") :].strip()
        elif line.startswith("data:"):
            data_parts.append(line[len("data:") :].strip())


def _frame_to_log_event(frame: dict[str, str]) -> LogEvent:
    """Convert a raw SSE frame dict into a :class:`LogEvent`."""
    try:
        payload = json.loads(frame["data"])
    except (json.JSONDecodeError, KeyError):
        payload = {"message": frame.get("data", "")}

    return LogEvent(
        id=frame.get("id", ""),
        type=frame.get("event", "log"),
        level=payload.get("level", "INFO"),
        message=payload.get("message", payload.get("msg", "")),
        timestamp=payload.get("ts", payload.get("timestamp", "")),
        metadata={
            k: v
            for k, v in payload.items()
            if k not in {"level", "message", "msg", "ts", "timestamp"}
        },
    )


class EventStream:
    """SSE consumer that yields :class:`LogEvent` objects from an Arena
    operations endpoint.

    Handles automatic reconnection with ``Last-Event-ID`` and
    exponential backoff.

    :param http: An authenticated httpx client.
    :type http: httpx.Client
    :param path: API path (e.g. ``/api/v1/operations/{id}/events``).
    :type path: str
    :param auth_headers: Headers dict containing the ``Authorization``
        bearer token.
    :type auth_headers: dict[str, str]
    :param max_retries: Maximum reconnection attempts before giving up.
    :type max_retries: int
    :param retry_backoff: Base backoff in seconds (doubled each attempt,
        capped at 30 s).
    :type retry_backoff: float
    """

    def __init__(
        self,
        http: httpx.Client,
        path: str,
        auth_headers: dict[str, str],
        *,
        max_retries: int = 5,
        retry_backoff: float = 1.0,
    ) -> None:
        self._http = http
        self._path = path
        self._auth_headers = auth_headers
        self._max_retries = max_retries
        self._retry_backoff = retry_backoff
        self._last_event_id: str = ""
        self._stream: httpx.Response | None = None
        self._finished = False

    def __iter__(self) -> Generator[LogEvent, None, None]:
        attempts = 0

        while not self._finished:
            headers = {**self._auth_headers, "Accept": "text/event-stream"}
            if self._last_event_id:
                headers["Last-Event-ID"] = self._last_event_id

            try:
                with self._http.stream(
                    "GET",
                    self._path,
                    headers=headers,
                    timeout=None,
                ) as resp:
                    if resp.status_code == 401:
                        msg = (
                            "SSE stream returned 401 Unauthorized. "
                            "Please re-authenticate with client.login()."
                        )
                        raise ArenaAuthError(msg)

                    if not resp.is_success:
                        detail = ""
                        for chunk in resp.iter_text():
                            detail += chunk
                            if len(detail) > 500:
                                break
                        raise ArenaAPIError(
                            status_code=resp.status_code,
                            detail=detail[:500] or "No details",
                        )

                    self._stream = resp
                    attempts = 0

                    for frame in _parse_sse_events(resp.iter_lines()):
                        event = _frame_to_log_event(frame)

                        if event.id:
                            self._last_event_id = event.id

                        yield event

                        if event.type == "complete":
                            self._finished = True
                            return
                        if event.type == "error" and event.metadata.get("fatal"):
                            self._finished = True
                            return

            except (httpx.HTTPError, ConnectionError) as exc:
                attempts += 1
                if attempts > self._max_retries:
                    raise ArenaAPIError(
                        status_code=0,
                        detail=(
                            f"SSE stream disconnected after {self._max_retries} "
                            f"retries: {exc}"
                        ),
                    ) from exc

                wait = min(self._retry_backoff * (2 ** (attempts - 1)), 30.0)
                logger.debug(
                    "SSE connection lost (attempt %d/%d), reconnecting in %.1fs ...",
                    attempts,
                    self._max_retries,
                    wait,
                )
                time.sleep(wait)
            finally:
                self._stream = None

    def close(self) -> None:
        """Abort the underlying HTTP stream if still open."""
        self._finished = True
        if self._stream is not None:
            self._stream.close()
            self._stream = None

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class LogDisplay:
    """Rich-powered terminal renderer for :class:`LogEvent` streams.

    :param console: Rich console instance.  A default is created when
        ``None``.
    :type console: rich.console.Console or None
    """

    def __init__(self, console: Console | None = None) -> None:
        self._console = console or Console()
        self._progress: Progress | None = None
        self._task_id: int | None = None
        self._result: dict[str, Any] = {}

    @property
    def result(self) -> dict[str, Any]:
        """Final result payload from the ``complete`` event, if any."""
        return self._result

    def render(self, event: LogEvent) -> None:
        """Dispatch *event* to the appropriate type-specific renderer.

        :param event: The event to render.
        :type event: LogEvent
        """
        handler = {
            "log": self._render_log,
            "progress": self._render_progress,
            "complete": self._render_complete,
            "error": self._render_error,
        }.get(event.type, self._render_log)
        handler(event)

    def _render_log(self, event: LogEvent) -> None:
        style = _LEVEL_STYLES.get(event.level.upper(), "")
        ts = event.timestamp[:19] if event.timestamp else ""
        prefix = f"[dim]{ts}[/dim] " if ts else ""
        level_tag = (
            f"[{style}]{event.level:<5}[/{style}] " if style else f"{event.level:<5} "
        )
        self._console.print(f"{prefix}{level_tag}{event.message}", highlight=False)

    def _render_progress(self, event: LogEvent) -> None:
        step = event.metadata.get("step", 0)
        total = event.metadata.get("total", 0)
        best_fitness = event.metadata.get("best_fitness")
        mean_fitness = event.metadata.get("mean_fitness")

        if self._progress is None:
            self._progress = Progress(
                SpinnerColumn(),
                TextColumn("[bold blue]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeElapsedColumn(),
                console=self._console,
            )
            self._progress.start()
            self._task_id = self._progress.add_task("Training", total=total or 100)

        if self._task_id is not None:
            if total:
                self._progress.update(self._task_id, completed=step, total=total)
            else:
                self._progress.update(self._task_id, completed=step)

        parts: list[str] = []
        if best_fitness is not None:
            parts.append(f"best={best_fitness:.4f}")
        if mean_fitness is not None:
            parts.append(f"mean={mean_fitness:.4f}")
        if parts:
            self._console.print(
                f"  [dim]Step {step}/{total}[/dim]  {' | '.join(parts)}",
                highlight=False,
            )

    def _render_complete(self, event: LogEvent) -> None:
        self._stop_progress()
        self._result = event.metadata.get("result", event.metadata)
        status = event.metadata.get("status", "completed")
        body = Text(event.message) if event.message else Text(f"Status: {status}")
        self._console.print(
            Panel(body, title="[bold green]Complete", border_style="green")
        )

    def _render_error(self, event: LogEvent) -> None:
        self._stop_progress()
        fatal = event.metadata.get("fatal", False)
        title = "[bold red]Fatal Error" if fatal else "[bold red]Error"
        self._console.print(Panel(Text(event.message), title=title, border_style="red"))

    def _stop_progress(self) -> None:
        if self._progress is not None:
            self._progress.stop()
            self._progress = None
            self._task_id = None

    def stop(self) -> None:
        """Clean up any active Rich live displays."""
        self._stop_progress()
