import io
import json
import os
import tarfile
from pathlib import Path
from typing import Any


def file_to_bytes(file: Path | os.PathLike[str]) -> bytes:
    """Convert a file to bytes.

    :param file: Path to the file.
    :returns: Bytes of the file.
    :raises FileNotFoundError: If the path is not found.
    """
    path = Path(os.fspath(file)).resolve()
    if not path.is_file():
        msg = f"File not found: {path}"
        raise FileNotFoundError(msg)
    return path.read_bytes()


def resolve_env_config(
    config: dict[str, Any] | str | os.PathLike[str] | None,
) -> bytes | None:
    """Resolve the environment configuration from a dictionary or file.

    :param config: Environment configuration.
    :returns: Bytes of the configuration.
    :raises FileNotFoundError: If the configuration file is not found.
    """
    if config is None:
        return None
    if isinstance(config, dict):
        return json.dumps(config, indent=2).encode("utf-8")

    return file_to_bytes(config)


def resolve_env_requirements(
    requirements: str | os.PathLike[str] | None,
) -> bytes | None:
    """Resolve the environment requirements from a file.

    :param requirements: Path to the requirements file.
    :returns: Bytes of the requirements.
    :raises FileNotFoundError: If the requirements file is not found.
    """
    if requirements is None:
        return None

    return file_to_bytes(requirements)


def prepare_env_upload(
    source: Path | os.PathLike[str],
    *,
    config: dict[str, Any] | str | os.PathLike[str] | None = None,
    requirements: str | os.PathLike[str] | None = None,
    description: str | None = None,
    exclude_dirs: tuple[str, ...] = (),
) -> bytes:
    """Prepare an environment for upload to Arena.

    :param source: Path to the environment source directory.
    :type source: Path | os.PathLike[str]
    :param config: Environment configuration.
    :type config: dict[str, Any] | str | os.PathLike[str] | None
    :param requirements: Environment requirements.
    :type requirements: str | os.PathLike[str] | None
    :param exclude_dirs: Directories to exclude from the upload.
    :type exclude_dirs: tuple[str, ...]
    :param description: Description of the environment.
    :type description: str | None
    :returns: Bytes of the prepared environment.
    """
    config_bytes = resolve_env_config(config)
    requirements_bytes = resolve_env_requirements(requirements)

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        if source.is_dir():
            for child in sorted(source.rglob("*")):
                if not child.is_file():
                    continue
                rel = child.relative_to(source)
                if rel.name in exclude_dirs:
                    continue
                tar.add(str(child), arcname=rel.as_posix())
        else:
            tar.add(str(source), arcname=source.name)

        if config_bytes is not None:
            info = tarfile.TarInfo(name="config.json")
            info.size = len(config_bytes)
            tar.addfile(info, io.BytesIO(config_bytes))

        if requirements_bytes is not None:
            info = tarfile.TarInfo(name="requirements.txt")
            info.size = len(requirements_bytes)
            tar.addfile(info, io.BytesIO(requirements_bytes))

        if description is not None:
            info = tarfile.TarInfo(name="description.txt")
            info.size = len(description)
            tar.addfile(info, io.BytesIO(description.encode()))

    return buf.getvalue()
