from __future__ import annotations

import csv
import io
import json
import logging
from dataclasses import dataclass
from functools import singledispatch
from typing import Any, Self

import click
from rich.live import Live
from rich.table import Table

from agilerl.arena import console, error_console
from agilerl.arena.exceptions import ArenaAPIError, ArenaError
from agilerl.arena.stream import (
    CheckEvent,
    ErrorEvent,
    LogEvent,
    StatusEvent,
    StreamEvent,
)

logger = logging.getLogger(__name__)


def _print_rich(renderable: Any, *, is_error: bool = False) -> None:
    if is_error:
        error_console.print(renderable)
        return
    console.print(renderable)


@singledispatch
def emit_result(
    result: Any,
    *,
    is_error: bool = False,
    columns: list[str] | None = None,
) -> None:
    """Emit *result* to the terminal as a Rich table.

    Dispatches on the runtime type of *result*:

    * ``dict`` — key/value table (or environment catalog if the shape matches).
    * ``list`` — table of dicts or simple value list.
    * anything else — ``str()`` fallback.

    :param result: The result to emit.
    :type result: Any
    :param is_error: Whether to print the result as an error.
    :type is_error: bool
    :param columns: The columns to display in the table. If None, all columns will be displayed.
    :type columns: list[str] | None
    :returns: None
    """
    _print_rich(str(result), is_error=is_error)


@emit_result.register(dict)
def _emit_result_dict(
    result: dict, *, is_error: bool = False, columns: list[str] | None = None
) -> None:
    if _looks_like_environment_catalog(result):
        _emit_environment_catalog(result, is_error=is_error)
        return
    _emit_key_value_table(result, is_error=is_error, columns=columns)


@emit_result.register(list)
def _emit_result_list(
    result: list, *, is_error: bool = False, columns: list[str] | None = None
) -> None:
    if result and all(isinstance(item, dict) for item in result):
        _emit_list_of_dicts(result, is_error=is_error, columns=columns)
        return
    _emit_simple_list(result, is_error=is_error, columns=columns)


def _emit_key_value_table(
    values: dict[str, Any], *, is_error: bool = False, columns: list[str] | None = None
) -> None:
    table = Table(show_header=True, header_style="bold")
    col_names = columns if columns and len(columns) == 2 else ["Field", "Value"]
    for name in col_names:
        table.add_column(name)
    for key, value in values.items():
        table.add_row(str(key), _format_cell(value))
    _print_rich(table, is_error=is_error)


def _emit_simple_list(
    values: list[Any], *, is_error: bool = False, columns: list[str] | None = None
) -> None:
    table = Table(show_header=True, header_style="bold")
    col_name = columns[0] if columns and len(columns) >= 1 else "Value"
    table.add_column(col_name)
    for value in values:
        table.add_row(_format_cell(value))
    _print_rich(table, is_error=is_error)


def _emit_list_of_dicts(
    values: list[dict[str, Any]],
    *,
    is_error: bool = False,
    columns: list[str] | None = None,
) -> None:
    keys: list[str] = []
    for row in values:
        for key in row:
            if key not in keys:
                keys.append(key)

    headers = columns if columns and len(columns) == len(keys) else keys
    table = Table(show_header=True, header_style="bold")
    for header in headers:
        table.add_column(str(header))

    for row in values:
        table.add_row(*[_format_cell(row.get(key)) for key in keys])
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
    """Print an :class:`ArenaError` and exit, or re-raise unknown exceptions.

    :param exc: The exception to handle.
    :type exc: Exception
    :returns: None
    :raises click.exceptions.Exit: If the exception is an :class:`ArenaError`.
    """
    if isinstance(exc, ArenaError):
        error_console.print(f"[red bold]Error:[/red bold] {exc}")
        raise click.exceptions.Exit(1) from exc
    raise exc


@dataclass(slots=True)
class StreamRow:
    event_type: str
    name: str
    status: str
    details: str = ""


class StreamRichRenderer:
    """Render :class:`~agilerl.arena.stream.StreamEvent` objects as a live Rich table."""

    def __init__(
        self,
        *,
        is_error: bool = False,
        error_cls: type[ArenaAPIError] = ArenaAPIError,
    ) -> None:
        self._console = error_console if is_error else console
        self._rows: list[StreamRow] = []
        self._live: Live | None = None
        self._error_cls = error_cls

    @staticmethod
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

    def handle_event(self, event: StreamEvent) -> None:
        """Dispatch a typed stream event to the appropriate renderer.

        :param event: The stream event to handle.
        :type event: StreamEvent
        :returns: None
        """
        if isinstance(event, StatusEvent):
            if event.status == "completed":
                self.close()
            if event.kind == "warning":
                logger.warning("%s", event.message)
            else:
                logger.info("%s", event.message)
            return

        if isinstance(event, ErrorEvent):
            if self._live is not None:
                self._render_error(event)
                self._refresh()
            else:
                raise self._error_cls(
                    detail=event.message,
                    extras=event.extras,
                )
            return

        self._ensure_live()
        if isinstance(event, CheckEvent):
            self._render_check(event)
        elif isinstance(event, LogEvent):
            self._render_log(event)
        self._refresh()

    def close(self) -> None:
        """Stop the live table renderer.

        :returns: None
        """
        if self._live is not None:
            self._live.update(self._build_table())
            self._live.stop()
            self._live = None

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _render_check(self, event: CheckEvent) -> None:
        """Render an environment validation check event."""
        has_warnings = len(event.warnings) > 0
        if event.success is True and has_warnings:
            styled = self._styled_status("WARNING")
            details = "; ".join(event.warnings)
        elif event.success is True:
            styled = self._styled_status("PASS")
            details = ""
        elif event.success is False:
            styled = self._styled_status("FAIL")
            details = event.error or ""
        else:
            styled = "UNKNOWN"
            details = ""

        self._rows.append(
            StreamRow("check", f"[bold]{event.name}[/bold]", styled, details)
        )

    def _render_error(self, event: ErrorEvent) -> None:
        """Render an error event as a table row."""
        detail = event.message
        for key, value in event.extras.items():
            label = key.replace("_", " ").capitalize()
            if isinstance(value, list):
                detail += f"\n{label}: {', '.join(str(v) for v in value)}"
            else:
                detail += f"\n{label}: {value}"
        self._rows.append(StreamRow("error", "-", f"[red]{detail}[/red]"))

    def _render_log(self, event: LogEvent) -> None:
        """Render a log event."""
        if event.text:
            self._rows.append(StreamRow("log", "-", event.text))

    def _ensure_live(self) -> None:
        """Ensure the live table renderer is started."""
        if self._live is not None:
            return
        self._live = Live(console=self._console, auto_refresh=False)
        self._live.start()

    def _refresh(self) -> None:
        if self._live is not None:
            self._live.update(self._build_table(), refresh=True)

    def _build_table(self) -> Table:
        table = Table(show_header=True, header_style="bold")
        table.add_column("Check", no_wrap=True)
        table.add_column("Result", no_wrap=True)
        table.add_column("Details")
        for row in self._rows:
            table.add_row(row.name, row.status, row.details)
        return table


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
