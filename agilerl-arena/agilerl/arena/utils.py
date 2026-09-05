# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

import os
import tarfile
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import BinaryIO

from agilerl.arena.exceptions import ArenaFileNotFoundError


def extract_filename(disposition: str | None) -> str | None:
    """Parse a filename from a Content-Disposition header value."""
    if not disposition:
        return None
    for part in disposition.split(";"):
        part = part.strip()
        if part.startswith("filename="):
            return part.removeprefix("filename=").strip('"')
    return None


def order_dataset_fields(
    row: dict[str, str | int | None],
) -> dict[str, str | int | None]:
    """Return a copy of a dataset row with ``name`` and ``hf_dataset_id`` first.

    :param row: The dataset row to order.
    :type row: dict[str, str | None | int]
    :returns: The ordered dataset row.
    :rtype: dict[str, str | None | int]
    """
    ordered = {
        "name": row.get("name"),
        "hf_dataset_id": row.get("hf_dataset_id"),
    }
    for key, value in row.items():
        if key not in ordered:
            ordered[key] = value
    return ordered


def sort_dataset_search_by_downloads(
    results: list[dict[str, str | int | None]],
) -> list[dict[str, str | int | None]]:
    """Sort HuggingFace search rows by ``downloads`` descending."""
    return sorted(results, key=lambda row: row.get("downloads") or 0, reverse=True)


def multipart_text_fields(
    fields: dict[str, str | None],
) -> dict[str, tuple[None, str]]:
    """Convert text form fields to httpx multipart ``files`` entries.

    Arena dataset create expects ``multipart/form-data`` even when no file is
    uploaded. Pass the return value as ``files=`` and omit ``data=``.
    """
    return {key: (None, value) for key, value in fields.items() if value is not None}


def _tar_to_tempfile(add_entries: Callable[[tarfile.TarFile], None]) -> BinaryIO:
    """Write a ``.tar.gz`` to an unlinked temporary file and rewind it.

    Spooling to disk instead of memory keeps large artifact uploads from
    holding the whole archive resident; httpx streams the open handle.
    """
    buf = tempfile.TemporaryFile()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        add_entries(tar)
    buf.seek(0)
    return buf


_ENV_CONFIG_NAMES: tuple[str, ...] = (
    "env_config.yaml",
    "env_config.yml",
    "env_config.json",
)


def discover_env_sidecars(
    source: str | os.PathLike[str] | bytes,
    *,
    requirements: str | os.PathLike[str] | None,
    env_config: str | os.PathLike[str] | None,
) -> tuple[str | os.PathLike[str] | None, str | os.PathLike[str] | None]:
    """Auto-detect ``requirements`` and ``env_config`` sidecars in a source dir.

    When *source* is a directory, its top level is inspected for a
    ``requirements.txt`` and an ``env_config.{yaml,yml,json}`` file (in that
    precedence) so callers need not pass them explicitly. An explicitly provided
    argument always wins and short-circuits detection for that slot. Non-directory
    sources (single files, ``.tar.gz`` paths, or raw bytes) are passed through
    unchanged.

    :param source: The environment source.
    :type source: str | os.PathLike[str] | bytes
    :param requirements: Explicit requirements path, or ``None`` to auto-detect.
    :type requirements: str | os.PathLike[str] | None
    :param env_config: Explicit env-config path, or ``None`` to auto-detect.
    :type env_config: str | os.PathLike[str] | None
    :returns: The resolved ``(requirements, env_config)`` pair.
    :rtype: tuple[str | os.PathLike[str] | None, str | os.PathLike[str] | None]
    """
    if requirements is not None and env_config is not None:
        return requirements, env_config

    if isinstance(source, bytes):
        return requirements, env_config

    path = Path(os.fspath(source)).expanduser().resolve()
    if not path.is_dir():
        return requirements, env_config

    if requirements is None:
        candidate = path / "requirements.txt"
        if candidate.is_file():
            requirements = candidate

    if env_config is None:
        for name in _ENV_CONFIG_NAMES:
            candidate = path / name
            if candidate.is_file():
                env_config = candidate
                break

    return requirements, env_config


def prepare_env_upload(
    source: str | os.PathLike[str] | bytes,
) -> tuple[str, BinaryIO | bytes]:
    """Resolve an environment source into an upload-ready ``(name, payload)`` pair.

    *source* may be:

    * A path to a directory — compressed into ``.tar.gz`` automatically.
    * A path to a single file — compressed into ``.tar.gz`` automatically.
    * A path to an existing ``.tar.gz`` file — opened for streaming as-is.
    * Raw ``bytes`` — used directly (assumed to be a valid ``.tar.gz``).

    Path inputs resolve to open binary handles so httpx streams the upload in
    chunks instead of holding the whole artifact in memory; the caller is
    responsible for closing them once the request has been sent.

    :param source: The source of the environment.
    :type source: str | os.PathLike[str] | bytes
    :returns: The name and payload (open handle or bytes) of the environment.
    :rtype: tuple[str, BinaryIO | bytes]
    :raises ArenaFileNotFoundError: If *source* is a path that does not exist.
    """
    if isinstance(source, bytes):
        return ("environment.tar.gz", source)

    path = Path(os.fspath(source)).expanduser().resolve()

    # Directory
    if path.is_dir():

        def _add_dir(tar: tarfile.TarFile) -> None:
            for child in sorted(path.rglob("*")):
                if child.is_file():
                    tar.add(str(child), arcname=child.relative_to(path).as_posix())

        return (f"{path.name}.tar.gz", _tar_to_tempfile(_add_dir))

    # Single file
    if path.is_file():
        if path.suffixes[-2:] == [".tar", ".gz"] or path.suffix == ".tgz":
            return (path.name, path.open("rb"))

        def _add_file(tar: tarfile.TarFile) -> None:
            tar.add(str(path), arcname=path.name)

        return (f"{path.stem}.tar.gz", _tar_to_tempfile(_add_file))

    msg = f"Source path not found: {path}"
    raise ArenaFileNotFoundError(msg)


def prepare_file_upload(
    source: str | os.PathLike[str] | bytes,
    *,
    default_name: str,
    content_type: str,
    filename: str | None = None,
) -> tuple[str, BinaryIO | bytes, str]:
    """Resolve a path or raw bytes into an httpx multipart file tuple.

    Path inputs resolve to open binary handles so httpx streams the upload
    in chunks instead of reading the whole file into memory; the caller is
    responsible for closing them once the request has been sent.

    :param source: File path or raw file contents.
    :type source: str | os.PathLike[str] | bytes
    :param default_name: Filename used when *source* is raw bytes.
    :type default_name: str
    :param content_type: MIME type for the upload part.
    :type content_type: str
    :param filename: Multipart filename override (relative parquet shard path).
    :type filename: str | None
    :returns: ``(filename, payload, content_type)`` for httpx ``files=``.
    :rtype: tuple[str, BinaryIO | bytes, str]
    :raises ArenaFileNotFoundError: If *source* is a path that does not exist.
    """
    if isinstance(source, bytes):
        return (filename or default_name, source, content_type)

    path = Path(os.fspath(source)).expanduser().resolve()
    if not path.is_file():
        msg = f"Upload file not found: {path}"
        raise ArenaFileNotFoundError(msg)
    return (filename or path.name, path.open("rb"), content_type)
