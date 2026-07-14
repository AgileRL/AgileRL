from __future__ import annotations

import re
from pathlib import Path


def resolve_metrics_output_path(
    *,
    experiment_id: int | None = None,
    experiment_name: str | None = None,
    payload: bytes,
    content_type: str | None,
    disposition: str | None,
    output_file: Path | None,
) -> Path:
    """Resolve the output path for metrics.

    :param experiment_id: Numeric experiment id used in default filenames.
    :type experiment_id: int | None
    :param experiment_name: Experiment name used in default filenames.
    :type experiment_name: str | None
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

    # Return the output path from id, name, or a generic fallback
    if experiment_id is not None:
        return Path(f"experiment_{experiment_id}_metrics{suffix}")
    if experiment_name is not None:
        safe = re.sub(r"[^\w\-.]", "_", experiment_name)[:200]
        return Path(f"experiment_{safe}_metrics{suffix}")
    return Path(f"experiment_metrics{suffix}")


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
