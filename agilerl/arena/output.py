from __future__ import annotations

import csv
import io
import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from functools import singledispatch
from typing import Any, Self

import click
from rich.console import Console
from rich.live import Live
from rich.table import Table

from agilerl.arena.exceptions import ArenaError
from agilerl.arena.stream import (
    CheckEvent,
    CompletionEvent,
    ErrorEvent,
    LogEvent,
    StatusEvent,
    StreamEvent,
)

logger = logging.getLogger(__name__)

console = Console()
error_console = Console(stderr=True)


def _print_rich(renderable: Any, *, is_error: bool = False) -> None:
    if is_error:
        error_console.print(renderable)
        return
    console.print(renderable)


@singledispatch
def emit_result(result: Any, *, is_error: bool = False) -> None:
    """Emit *result* to the terminal as a Rich table.

    Dispatches on the runtime type of *result*:

    * ``dict`` — key/value table (or environment catalog if the shape matches).
    * ``list`` — table of dicts or simple value list.
    * anything else — ``str()`` fallback.
    """
    _print_rich(str(result), is_error=is_error)


@emit_result.register(dict)
def _emit_result_dict(result: dict, *, is_error: bool = False) -> None:
    if _looks_like_environment_catalog(result):
        _emit_environment_catalog(result, is_error=is_error)
        return
    _emit_key_value_table(result, is_error=is_error)


@emit_result.register(list)
def _emit_result_list(result: list, *, is_error: bool = False) -> None:
    if result and all(isinstance(item, dict) for item in result):
        _emit_list_of_dicts(result, is_error=is_error)
        return
    _emit_simple_list(result, is_error=is_error)


def _emit_key_value_table(values: dict[str, Any], *, is_error: bool = False) -> None:
    table = Table(show_header=True, header_style="bold")
    table.add_column("Field")
    table.add_column("Value")
    for key, value in values.items():
        table.add_row(str(key), _format_cell(value))
    _print_rich(table, is_error=is_error)


def _emit_simple_list(values: list[Any], *, is_error: bool = False) -> None:
    table = Table(show_header=True, header_style="bold")
    table.add_column("Value")
    for value in values:
        table.add_row(_format_cell(value))
    _print_rich(table, is_error=is_error)


def _emit_list_of_dicts(
    values: list[dict[str, Any]], *, is_error: bool = False
) -> None:
    columns: list[str] = []
    for row in values:
        for key in row:
            if key not in columns:
                columns.append(key)

    table = Table(show_header=True, header_style="bold")
    for column in columns:
        table.add_column(str(column))

    for row in values:
        table.add_row(*[_format_cell(row.get(column)) for column in columns])
    _print_rich(table, is_error=is_error)


def _looks_like_environment_catalog(values: dict[str, Any]) -> bool:
    if not values:
        return False
    for version_map in values.values():
        if not isinstance(version_map, dict):
            return False
        for metadata in version_map.values():
            if not isinstance(metadata, dict):
                return False
            if not {"validated", "profiled"}.issubset(metadata):
                return False
    return True


def _emit_environment_catalog(
    values: dict[str, Any], *, is_error: bool = False
) -> None:
    table = Table(show_header=True, header_style="bold")
    table.add_column("Environment")
    table.add_column("Version")
    table.add_column("Validated")
    table.add_column("Profiled")

    if not values:
        _print_rich("No environments found.", is_error=is_error)
        return

    for env_name, versions in values.items():
        if not isinstance(versions, dict):
            continue
        for version_name, metadata in versions.items():
            metadata_dict = metadata if isinstance(metadata, dict) else {}
            table.add_row(
                f"[bold]{env_name}[/bold]",
                str(version_name),
                "[green]✔[/green]"
                if metadata_dict.get("validated")
                else "[red]✘[/red]",
                "[green]✔[/green]" if metadata_dict.get("profiled") else "[red]✘[/red]",
            )
    _print_rich(table, is_error=is_error)


def _format_cell(value: Any) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(value, default=str)
    return str(value)


def handle_error(exc: Exception) -> None:
    """Print an :class:`ArenaError` and exit, or re-raise unknown exceptions."""
    if isinstance(exc, ArenaError):
        error_console.print(f"[red bold]Error:[/red bold] {exc}")
        raise click.exceptions.Exit(1)
    raise exc


@dataclass(slots=True)
class StreamRow:
    event_type: str
    name: str
    status: str


class StreamTableRenderer:
    """Render :class:`~agilerl.arena.stream.StreamEvent` objects as a live Rich table."""

    def __init__(self, *, is_error: bool = False) -> None:
        self._console = error_console if is_error else console
        self._rows: list[StreamRow] = []
        self._live: Live | None = None

    def handle_event(self, event: StreamEvent) -> None:
        """Dispatch a typed stream event to the appropriate renderer."""
        if isinstance(event, StatusEvent):
            logger.info("%s", event.message)
            return
        if isinstance(event, CompletionEvent):
            self.close()
            self._render_completion(event)
            return

        self._ensure_live()
        if isinstance(event, CheckEvent):
            self._render_check(event)
        elif isinstance(event, ErrorEvent):
            self._render_error(event)
        elif isinstance(event, LogEvent):
            self._render_log(event)
        self._refresh()

    def close(self) -> None:
        """Stop the live table renderer."""
        if self._live is not None:
            self._refresh()
            self._live.stop()
            self._live = None

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _render_check(self, event: CheckEvent) -> None:
        has_warnings = len(event.warnings) > 0
        if event.success is True and has_warnings:
            styled = _styled_status("WARNING")
        elif event.success is True:
            styled = _styled_status("PASS")
        elif event.success is False:
            styled = _styled_status("FAIL")
        else:
            styled = "UNKNOWN"

        self._rows.append(StreamRow("check", f"[bold]{event.name}[/bold]", styled))

    def _render_completion(self, event: CompletionEvent) -> None:
        env_name = (
            event.payload.get("env_info", {}).get("env_name")
            if isinstance(event.payload.get("env_info"), dict)
            else None
        )
        msg = (
            f"Validation completed for {env_name}"
            if env_name
            else "Validation completed"
        )
        logger.info("%s", msg)

    def _render_error(self, event: ErrorEvent) -> None:
        detail = event.message
        for key, value in event.extras.items():
            label = key.replace("_", " ").capitalize()
            if isinstance(value, list):
                detail += f"\n{label}: {', '.join(str(v) for v in value)}"
            else:
                detail += f"\n{label}: {value}"
        self._rows.append(StreamRow("error", "-", f"[red]{detail}[/red]"))

    def _render_log(self, event: LogEvent) -> None:
        if event.text:
            self._rows.append(StreamRow("log", "-", event.text))

    def _ensure_live(self) -> None:
        if self._live is not None:
            return
        self._live = Live(
            self._build_table(),
            console=self._console,
            refresh_per_second=8,
        )
        self._live.start()

    def _refresh(self) -> None:
        if self._live is not None:
            self._live.update(self._build_table())

    def _build_table(self) -> Table:
        table = Table(show_header=True, header_style="bold")
        table.add_column("Check", no_wrap=True)
        table.add_column("Status", no_wrap=True)
        for row in self._rows:
            table.add_row(row.name, row.status)
        return table


def _styled_status(status: str) -> str:
    upper = status.upper()
    if upper == "PASS":
        return "[green]PASS[/green]"
    if upper == "FAIL":
        return "[red]FAIL[/red]"
    if upper in ("WARNING", "WARN"):
        return "[dark_orange]WARNING[/dark_orange]"
    if upper == "COMPLETED":
        return "[green]COMPLETED[/green]"
    return status


def build_stream_handler() -> tuple[Callable[[StreamEvent], None], StreamTableRenderer]:
    """Build a stream-event handler and its backing renderer.

    The renderer must be closed after use (or used as a context manager).
    """
    renderer = StreamTableRenderer()
    return renderer.handle_event, renderer


def emit_csv_preview(payload: bytes, *, max_rows: int) -> None:
    text = payload.decode("utf-8", errors="replace")
    reader = csv.reader(io.StringIO(text))
    rows = list(reader)
    if not rows:
        return

    header = rows[0]
    data_rows = rows[1 : max_rows + 1]
    table = Table(title=f"Metrics Preview (first {len(data_rows)} rows)")
    for column in header:
        table.add_column(column)
    for row in data_rows:
        padded = row + [""] * max(0, len(header) - len(row))
        table.add_row(*padded[: len(header)])
    console.print(table)
