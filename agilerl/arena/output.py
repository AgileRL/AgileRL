from __future__ import annotations

import csv
import io
import json
from dataclasses import dataclass
from typing import Any, Callable

import click
from rich.console import Console
from rich.live import Live
from rich.table import Table

from agilerl.arena.config import OutputFormat
from agilerl.arena.exceptions import (
    ArenaAPIError,
    ArenaAuthError,
    ArenaError,
    ArenaValidationError,
)

console = Console()
error_console = Console(stderr=True)


def print_rich(renderable: Any, *, is_error: bool = False) -> None:
    if is_error:
        error_console.print(renderable)
        return
    console.print(renderable)


def emit(result: Any, output: OutputFormat, *, is_error: bool = False) -> None:
    if output == "json":
        click.echo(json.dumps(result, indent=2, default=str), err=is_error)
        return

    if isinstance(result, dict):
        if _looks_like_environment_catalog(result):
            emit_environment_catalog(result, is_error=is_error)
            return
        emit_key_value_table(result, is_error=is_error)
        return

    if isinstance(result, list):
        if result and all(isinstance(item, dict) for item in result):
            emit_list_of_dicts(result, is_error=is_error)
            return
        emit_simple_list(result, is_error=is_error)
        return

    print_rich(str(result), is_error=is_error)


def emit_key_value_table(values: dict[str, Any], *, is_error: bool = False) -> None:
    table = Table(show_header=True, header_style="bold")
    table.add_column("Field")
    table.add_column("Value")
    for key, value in values.items():
        table.add_row(str(key), format_cell(value))
    print_rich(table, is_error=is_error)


def emit_simple_list(values: list[Any], *, is_error: bool = False) -> None:
    table = Table(show_header=True, header_style="bold")
    table.add_column("Value")
    for value in values:
        table.add_row(format_cell(value))
    print_rich(table, is_error=is_error)


def emit_list_of_dicts(values: list[dict[str, Any]], *, is_error: bool = False) -> None:
    columns: list[str] = []
    for row in values:
        for key in row:
            if key not in columns:
                columns.append(key)

    table = Table(show_header=True, header_style="bold")
    for column in columns:
        table.add_column(str(column))

    for row in values:
        table.add_row(*[format_cell(row.get(column)) for column in columns])
    print_rich(table, is_error=is_error)


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


def emit_environment_catalog(values: dict[str, Any], *, is_error: bool = False) -> None:
    table = Table(show_header=True, header_style="bold")
    table.add_column("Environment")
    table.add_column("Version")
    table.add_column("Validated")
    table.add_column("Profiled")

    if not values:
        print_rich("No environments found.", is_error=is_error)
        return

    for env_name, versions in values.items():
        if not isinstance(versions, dict):
            continue
        for version_name, metadata in versions.items():
            metadata_dict = metadata if isinstance(metadata, dict) else {}
            table.add_row(
                str(env_name),
                str(version_name),
                "yes" if bool(metadata_dict.get("validated")) else "no",
                "yes" if bool(metadata_dict.get("profiled")) else "no",
            )
    print_rich(table, is_error=is_error)


def format_cell(value: Any) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(value, default=str)
    return str(value)


def handle_error(exc: Exception, output: OutputFormat) -> None:
    if isinstance(exc, ArenaValidationError):
        emit({"error": "validation_failed", "details": exc.errors}, output, is_error=True)
        raise click.exceptions.Exit(1)
    if isinstance(exc, ArenaAuthError):
        emit(
            {"error": "authentication_failed", "details": str(exc)},
            output,
            is_error=True,
        )
        raise click.exceptions.Exit(1)
    if isinstance(exc, ArenaAPIError):
        emit(
            {
                "error": "api_error",
                "status_code": exc.status_code,
                "details": exc.detail,
            },
            output,
            is_error=True,
        )
        raise click.exceptions.Exit(1)
    if isinstance(exc, ArenaError):
        emit({"error": "arena_error", "details": str(exc)}, output, is_error=True)
        raise click.exceptions.Exit(1)

    raise exc


def stream_chunk(chunk: str) -> None:
    click.echo(chunk, nl=False)


@dataclass(slots=True)
class StreamRow:
    event_type: str
    name: str
    status: str
    details: str


class StreamTableRenderer:
    """Incrementally render newline-delimited JSON chunks as a Rich table."""

    def __init__(self, *, is_error: bool = False) -> None:
        self._console = error_console if is_error else console
        self._buffer = ""
        self._rows: list[StreamRow] = []
        self._live: Live | None = None
        self._completion_payload: dict[str, Any] | None = None
        self._final_status_payload: dict[str, Any] | None = None

    def on_chunk(self, chunk: str) -> None:
        if not chunk:
            return
        self._ensure_live()
        self._buffer += chunk
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            self._consume_line(line.strip())
        self._refresh()

    def finalize_result(self, result: Any) -> Any | None:
        self.close()
        if not isinstance(result, dict):
            return result

        cleaned = {
            key: value for key, value in result.items() if key not in {"stream", "events"}
        }
        if self._completion_payload is None:
            return cleaned or None

        merged = {**cleaned, **self._completion_payload}
        if self._final_status_payload is not None:
            merged.setdefault("final_status", self._final_status_payload.get("status"))
            merged.setdefault("final_stage", self._final_status_payload.get("stage"))
            merged.setdefault("final_message", self._final_status_payload.get("message"))
        return merged or None

    def close(self) -> None:
        if self._buffer.strip():
            self._consume_line(self._buffer.strip())
            self._buffer = ""
        if self._live is not None:
            self._refresh()
            self._live.stop()
            self._live = None

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
        table = Table(title="Stream Updates")
        table.add_column("Type", no_wrap=True)
        table.add_column("Name")
        table.add_column("Status", no_wrap=True)
        table.add_column("Details")
        for row in self._rows:
            table.add_row(row.event_type, row.name, row.status, row.details)
        return table

    def _consume_line(self, line: str) -> None:
        if not line:
            return
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            self._rows.append(StreamRow("log", "-", "-", line))
            return

        if not isinstance(payload, dict):
            self._rows.append(StreamRow("event", "-", "-", format_cell(payload)))
            return

        if payload.get("kind") == "status":
            stage = str(payload.get("stage", "-"))
            status = str(payload.get("status", "-"))
            raw_message = payload.get("message", "")
            message = str(raw_message)
            parsed_message = self._parse_json_message(message)
            if parsed_message is not None:
                message = self._summarize_payload(parsed_message)
            self._rows.append(StreamRow("status", stage, status, message))
            if status == "completed":
                self._final_status_payload = payload
                if isinstance(parsed_message, dict):
                    self._completion_payload = parsed_message
            return

        if "check" in payload and isinstance(payload.get("result"), dict):
            check_name = str(payload["check"])
            result = payload["result"]
            success = result.get("success")
            status = "PASS" if success is True else "FAIL" if success is False else "UNKNOWN"
            error_msg = result.get("error msg") or result.get("error") or ""
            warnings = result.get("warnings")
            warning_text = ""
            if isinstance(warnings, list) and warnings:
                warning_text = f"warnings: {', '.join(str(item) for item in warnings)}"
            details = "; ".join(part for part in (str(error_msg), warning_text) if part).strip()
            self._rows.append(StreamRow("check", check_name, status, details or "-"))
            return

        if payload.get("complete") is True:
            self._completion_payload = payload
            env_name = (
                payload.get("env_info", {}).get("env_name")
                if isinstance(payload.get("env_info"), dict)
                else None
            )
            details = f"env: {env_name}" if env_name else "Validation payload received"
            self._rows.append(StreamRow("result", "validation", "complete", details))
            return

        self._rows.append(StreamRow("event", "-", "-", format_cell(payload)))

    @staticmethod
    def _parse_json_message(message: str) -> dict[str, Any] | list[Any] | None:
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

    @staticmethod
    def _summarize_payload(payload: dict[str, Any] | list[Any]) -> str:
        if isinstance(payload, list):
            return f"items: {len(payload)}"
        if payload.get("ok") is True and "data" in payload:
            data = payload.get("data")
            if data in ("", None):
                return "ok"
            if isinstance(data, list):
                return f"ok, items: {len(data)}"
            if isinstance(data, dict):
                payload = data
            else:
                return f"ok, data: {data}"
        accepted = payload.get("accepted")
        experiment_id = payload.get("experimentId") or payload.get("experiment_id")
        submissions = payload.get("submissions")
        error_code = payload.get("error_code")
        error_message = payload.get("error")
        parts: list[str] = []
        if accepted is not None:
            parts.append(f"accepted: {accepted}")
        if experiment_id is not None:
            parts.append(f"experiment_id: {experiment_id}")
        if isinstance(submissions, list):
            parts.append(f"submissions: {len(submissions)}")
        if error_code is not None:
            parts.append(f"error_code: {error_code}")
        if error_message is not None:
            parts.append(f"error: {error_message}")
        if parts:
            return ", ".join(parts)
        keys = ", ".join(str(key) for key in list(payload.keys())[:4])
        return f"result keys: {keys}" if keys else "completed"


def build_stream_handler(
    output: OutputFormat,
) -> tuple[Callable[[str], None], StreamTableRenderer | None]:
    if output == "json":
        return stream_chunk, None
    renderer = StreamTableRenderer()
    return renderer.on_chunk, renderer


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
