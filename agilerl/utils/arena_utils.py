import io
import os
import tarfile
from pathlib import Path

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
    row: dict[str, str | None | int],
) -> dict[str, str | None | int]:
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
    results: list[dict[str, str | None | int]],
) -> list[dict[str, str | None | int]]:
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


def prepare_env_upload(source: str | os.PathLike[str] | bytes) -> tuple[str, bytes]:
    """Resolve an environment source into an upload-ready ``(name, bytes)`` pair.

    *source* may be:

    * A path to a directory — compressed into ``.tar.gz`` automatically.
    * A path to a single file — compressed into ``.tar.gz`` automatically.
    * A path to an existing ``.tar.gz`` file — read as-is.
    * Raw ``bytes`` — used directly (assumed to be a valid ``.tar.gz``).

    :param source: The source of the environment.
    :type source: str | os.PathLike[str] | bytes
    :returns: The name and bytes of the prepared environment.
    :rtype: tuple[str, bytes]
    :raises ArenaFileNotFoundError: If *source* is a path that does not exist.
    """
    if isinstance(source, bytes):
        return ("environment.tar.gz", source)

    path = Path(os.fspath(source)).expanduser().resolve()

    # Directory
    if path.is_dir():
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tar:
            for child in sorted(path.rglob("*")):
                if child.is_file():
                    tar.add(str(child), arcname=child.relative_to(path).as_posix())
        return (f"{path.name}.tar.gz", buf.getvalue())

    # Single file
    if path.is_file():
        if path.suffixes[-2:] == [".tar", ".gz"] or path.suffix == ".tgz":
            return (path.name, path.read_bytes())
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tar:
            tar.add(str(path), arcname=path.name)
        return (f"{path.stem}.tar.gz", buf.getvalue())

    msg = f"Source path not found: {path}"
    raise ArenaFileNotFoundError(msg)


def prepare_file_upload(
    source: str | os.PathLike[str] | bytes,
    *,
    default_name: str,
    content_type: str,
) -> tuple[str, bytes, str]:
    """Resolve a path or raw bytes into an httpx multipart file tuple.

    :param source: File path or raw file contents.
    :type source: str | os.PathLike[str] | bytes
    :param default_name: Filename used when *source* is raw bytes.
    :type default_name: str
    :param content_type: MIME type for the upload part.
    :type content_type: str
    :returns: ``(filename, contents, content_type)`` for httpx ``files=``.
    :rtype: tuple[str, bytes, str]
    :raises ArenaFileNotFoundError: If *source* is a path that does not exist.
    """
    if isinstance(source, bytes):
        return (default_name, source, content_type)

    path = Path(os.fspath(source)).expanduser().resolve()
    if not path.is_file():
        msg = f"Upload file not found: {path}"
        raise ArenaFileNotFoundError(msg)
    return (path.name, path.read_bytes(), content_type)
