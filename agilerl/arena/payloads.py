from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import click


def load_json_payload(
    payload_json: str | None,
    payload_file: str | None,
    *,
    json_option_name: str,
    file_option_name: str,
) -> dict[str, Any]:
    if payload_json is None and payload_file is None:
        msg = f"Provide either {json_option_name} or {file_option_name}."
        raise click.UsageError(msg)
    if payload_json is not None and payload_file is not None:
        msg = f"Use only one of {json_option_name} or {file_option_name}."
        raise click.UsageError(msg)

    try:
        if payload_json is not None:
            parsed = json.loads(payload_json)
        else:
            parsed = json.loads(Path(payload_file).read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        msg = f"Invalid JSON payload: {exc}"
        raise click.UsageError(msg) from exc

    if not isinstance(parsed, dict):
        msg = "Payload must be a JSON object."
        raise click.UsageError(msg)
    return parsed


def resolve_metrics_output_path(
    *,
    experiment_id: int,
    payload: bytes,
    content_type: str | None,
    disposition: str | None,
    output_file: Path | None,
) -> Path:
    if output_file is not None:
        return output_file

    suggested_name = filename_from_disposition(disposition)
    if suggested_name:
        return Path(suggested_name)

    is_zip = payload.startswith(b"PK") or "zip" in (content_type or "").lower()
    suffix = ".zip" if is_zip else ".csv"
    return Path(f"experiment_{experiment_id}_metrics{suffix}")


def filename_from_disposition(disposition: str | None) -> str | None:
    if not disposition:
        return None
    match = re.search(r'filename="?([^";]+)"?', disposition)
    return match.group(1) if match else None
