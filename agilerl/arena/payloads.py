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
    """Load a JSON payload from a JSON string or file.

    :param payload_json: The JSON string to load.
    :type payload_json: str | None
    :param payload_file: The file to load the JSON from.
    :type payload_file: str | None
    :param json_option_name: The name of the JSON option.
    :type json_option_name: str
    :param file_option_name: The name of the file option.
    :type file_option_name: str
    :returns: The loaded JSON payload.
    :rtype: dict[str, Any]
    :raises click.UsageError: If neither payload_json nor payload_file is provided.
    :raises click.UsageError: If both payload_json and payload_file are provided.
    :raises json.JSONDecodeError: If the JSON is invalid.
    """
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
    """Resolve the output path for metrics.

    :param experiment_id: The ID of the experiment.
    :type experiment_id: int
    :param payload: The payload to resolve the output path for.
    :type payload: bytes
    :param content_type: The content type of the payload.
    :type content_type: str | None
    :param disposition: The disposition of the payload.
    :type disposition: str | None
    :param output_file: The output file to resolve the output path for.
    :type output_file: Path | None
    :returns: The resolved output path.
    :rtype: Path
    """
    # If an output file is provided, use it
    if output_file is not None:
        return output_file

    # If a disposition is provided, use the filename from the disposition
    suggested_name = filename_from_disposition(disposition)
    if suggested_name:
        return Path(suggested_name)

    # If the payload is a zip, use the zip suffix
    is_zip = payload.startswith(b"PK") or "zip" in (content_type or "").lower()
    suffix = ".zip" if is_zip else ".csv"

    # Return the output path
    return Path(f"experiment_{experiment_id}_metrics{suffix}")


def filename_from_disposition(disposition: str | None) -> str | None:
    """Get the filename from the disposition.

    :param disposition: The disposition to get the filename from.
    :type disposition: str | None
    :returns: The filename from the disposition.
    :rtype: str | None
    """
    if not disposition:
        return None

    match = re.search(r'filename="?([^";]+)"?', disposition)

    # Return the filename from the disposition
    return match.group(1) if match else None
