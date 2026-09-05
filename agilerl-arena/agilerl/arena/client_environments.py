# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Environment listing, validation, and version management."""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from dataclasses import dataclass, fields, is_dataclass
from functools import wraps
from inspect import Parameter, signature
from pathlib import Path
from types import UnionType
from typing import Any, TypeVar, Union, cast, get_args, get_origin, get_type_hints

from agilerl.arena.stream import NDJsonStream
from agilerl.arena.utils import (
    discover_env_sidecars,
    prepare_env_upload,
    prepare_file_upload,
)

logger = logging.getLogger("agilerl.arena.client")

F = TypeVar("F", bound=Callable[..., object])


def _dataclass_type(annotation: object) -> type | None:
    """Return the dataclass hidden in an annotation."""
    origin = get_origin(annotation)
    if origin is Union or origin is UnionType:
        candidates = [arg for arg in get_args(annotation) if arg is not type(None)]
        if len(candidates) != 1:
            return None
        return _dataclass_type(candidates[0])
    if isinstance(annotation, type) and is_dataclass(annotation):
        return annotation
    return None


def _assemble_grouped_kwargs(
    fn: Callable[..., object],
    args: tuple[object, ...],
    kwargs: dict[str, object],
) -> dict[str, object]:
    """Map flat kwargs onto *fn*'s dataclass parameters."""
    hints = get_type_hints(fn)
    params = [param for param in signature(fn).parameters.values() if param.name != "self"]
    bound: dict[str, object] = dict(kwargs)
    remaining = list(args)
    fn_name = fn.__name__
    for param in params:
        if remaining and param.name not in bound:
            bound[param.name] = remaining.pop(0)
    if remaining:
        taken = len(args) - len(remaining)
        msg = (
            f"{fn_name}() takes {taken} positional argument(s) "
            f"but {len(args)} were given"
        )
        raise TypeError(msg)

    leftover = dict(bound)
    grouped: dict[str, object] = {}
    for param in params:
        annotation = hints.get(param.name, param.annotation)
        config_cls = _dataclass_type(annotation)
        if config_cls is None:
            if param.name in leftover:
                grouped[param.name] = leftover.pop(param.name)
            continue
        existing = leftover.pop(param.name, None)
        if existing is not None and is_dataclass(existing) and not isinstance(existing, type):
            grouped[param.name] = existing
            continue
        if existing is not None:
            msg = (
                f"{fn_name}.{param.name} must be a {config_cls.__name__} instance "
                f"or omitted; got {type(existing).__name__}."
            )
            raise TypeError(msg)
        field_names = {item.name for item in fields(config_cls)}
        subset = {
            key: leftover.pop(key) for key in list(leftover) if key in field_names
        }
        if not subset and param.default is not Parameter.empty:
            grouped[param.name] = param.default
            continue
        grouped[param.name] = config_cls(**subset)
    if leftover:
        unexpected = next(iter(leftover))
        msg = f"{fn_name}() got an unexpected keyword argument {unexpected!r}"
        raise TypeError(msg)
    return grouped


def accept_flat_kwargs(fn: F) -> F:
    """Map flat kwargs onto *fn*'s dataclass parameters, then call *fn*."""

    @wraps(fn)
    def wrapper(*args: object, **kwargs: object) -> object:
        params = list(signature(fn).parameters)
        if params and params[0] == "self":
            return fn(args[0], **_assemble_grouped_kwargs(fn, args[1:], kwargs))
        return fn(**_assemble_grouped_kwargs(fn, args, kwargs))

    wrapper.__signature__ = signature(fn)
    return cast("F", wrapper)


@dataclass(frozen=True)
class EnvironmentIdentity:
    """Environment name, version, and description."""

    name: str
    version: str | None = None
    description: str | None = None


@dataclass(frozen=True)
class EnvironmentSource:
    """Upload payload and sidecar paths for environment validation."""

    source: str | os.PathLike[str] | bytes | None = None
    env_config: str | os.PathLike[str] | None = None
    requirements: str | os.PathLike[str] | None = None
    entrypoint: str | None = None


@dataclass(frozen=True)
class EnvironmentKind:
    """Multi-agent, language-based, and rollout flags."""

    multi_agent: bool = False
    language_based: bool = False
    do_rollouts: bool = False


DEFAULT_ENVIRONMENT_SOURCE = EnvironmentSource()
DEFAULT_ENVIRONMENT_KIND = EnvironmentKind()


class EnvironmentClientMixin:
    """Arena environment catalog and validation."""

    def list_environments(
        self,
        name: str | None = None,
        include_arena: bool = False,
    ) -> dict[str, dict[str, dict[str, bool]]]:
        """List environments available to the authenticated user.

        :param name: Environment name. If None, list all environments.
        :type name: str | None
        :param include_arena: Whether to include off-the-shelf Gymnasium/PettingZoo environments
            available in Arena.
        :type include_arena: bool
        :returns: A nested dictionary keyed by environment name, then version:

            .. code-block:: python

                {
                    "<name>": {
                        "<version>": {
                            "validated": True,
                            "profiled": True,
                        }
                    }
                }

        :rtype: dict[str, dict[str, dict[str, bool]]]
        """
        return self._request(
            "GET",
            "/api/cli/v1/environments",
            params={"name": name, "include_arena": include_arena},
        )

    def environment_exists(self, name: str, version: str | None = None) -> bool:
        """Check whether an environment name/version is registered.

        :param name: Environment name.
        :type name: str
        :param version: Environment version. Defaults to None, which resolves to the latest version.
        :type version: str | None
        :returns: True if the environment exists, False otherwise.
        :rtype: bool
        """
        resp: dict[str, bool | str] = self._request(
            "GET",
            "/api/cli/v1/environments/exists",
            params={"name": name, "version": version},
        )
        return bool(resp.get("exists", False))

    def list_environment_entrypoints(
        self,
        name: str,
        version: str | None = None,
    ) -> list[str]:
        """List available entrypoints for an environment version.

        :param name: Environment name, as specified in Arena.
        :type name: str
        :param version: Environment version. Defaults to None, which resolves to the latest version.
        :type version: str | None
        :returns: A list of entrypoints.
        :rtype: list[str]
        """
        resp = self._request(
            "GET",
            "/api/cli/v1/environments/entrypoints",
            params={"name": name, "version": version},
        )
        logger.info("Found %d entrypoints for environment %s.", len(resp), name)
        return resp

    @accept_flat_kwargs
    def validate_environment(
        self,
        identity: EnvironmentIdentity,
        upload: EnvironmentSource = DEFAULT_ENVIRONMENT_SOURCE,
        kind: EnvironmentKind = DEFAULT_ENVIRONMENT_KIND,
    ) -> dict[str, Any]:
        """Validate a custom environment on Arena.

        :param identity: Name, version, and description.
        :param upload: Source archive and sidecar files.
        :param kind: Multi-agent, language-based, and rollout flags.
        :returns: Validation result from Arena.
        """
        source = upload.source
        version = identity.version
        if source is not None:
            if version is None:
                logger.info("No version specified, defaulting to v1.")
                version = "v1"
            identity = EnvironmentIdentity(
                name=identity.name,
                version=version,
                description=identity.description,
            )
            return self._create_and_validate(identity, upload, kind).collect()

        payload: dict[str, Any] = {
            "name": identity.name,
            "version": version,
            "do_rollouts": kind.do_rollouts,
        }
        if upload.entrypoint:
            payload["entrypoint"] = upload.entrypoint
        return self._open_stream(
            "POST",
            "/api/cli/v1/environments/validate",
            json=payload,
            timeout=self._upload_timeout,
        ).collect()

    def profile_environment(
        self,
        *,
        name: str,
        version: str | None = None,
    ) -> dict[str, Any]:
        """Profile a validated environment version.

        :param name: Environment name, as specified in Arena.
        :type name: str
        :param version: Environment version. Defaults to None, which resolves to the latest version.
        :type version: str | None
        """
        payload: dict[str, Any] = {
            "name": name,
            "version": version,
        }
        return self._open_stream(
            "POST",
            "/api/cli/v1/environments/profile",
            json=payload,
            timeout=self._upload_timeout,
        ).collect()

    def delete_environment(
        self, *, name: str, version: str | None = None, confirm: bool = False
    ) -> dict[str, Any] | None:
        """Delete an environment version (or all versions if version is None).

        :param name: Environment name, as specified in Arena.
        :type name: str
        :param version: Environment version. If None, delete all environment versions.
        :type version: str | None
        :param confirm: Whether to confirm the deletion.
        :type confirm: bool
        """
        # Fetch existing versions
        versions_data = self.list_environments(name=name)
        if name in versions_data:
            versions_data = versions_data[name]

        version_list = versions_data.keys()

        if not version_list:
            logger.info(
                "No versions found for environment '%s'. Nothing to delete.", name
            )
            return None

        if version is None:
            logger.info(
                "The following versions for '%s' will be deleted: %s",
                name,
                ", ".join(version_list),
            )
        else:
            if version not in version_list:
                logger.info(
                    "Version '%s' not found in environment '%s'. Please specify a valid version from the list: %s.",
                    version,
                    name,
                    ", ".join(version_list),
                )
                return None

            logger.info(
                "The following version for '%s' will be deleted: %s",
                name,
                version,
            )

        if not confirm:
            confirm_prompt = input("Do you wish to continue? [y/N]: ").strip().lower()
            if confirm_prompt not in ("y", "yes"):
                logger.info("No environment was deleted for %s.", name)
                return None

        payload = {"name": name, "version": version}
        result: dict[str, Any] = self._request(
            "DELETE", "/api/cli/v1/environments/delete", json=payload
        )
        deleted_version = result.get("version", version)
        msg_suffix = f":{deleted_version}" if version else ""
        logger.info("Environment %s%s deleted successfully.", name, msg_suffix)
        return result

    def duplicate_environment_version(
        self,
        *,
        name: str,
        new_version: str,
        version: str | None = None,
    ) -> dict[str, Any]:
        """Duplicate a custom environment version to a new version name.

        :param name: Environment name.
        :type name: str
        :param new_version: New ``version`` for the duplicate (e.g. ``v2``).
        :type new_version: str
        :param version: Source version; when omitted, the latest version is used.
        :type version: str | None
        """
        payload: dict[str, Any] = {
            "name": name,
            "new_version_name": new_version,
            "version": version,
        }
        resp = self._request(
            "POST",
            "/api/cli/v1/environments/duplicate",
            json=payload,
        )
        logger.info(
            "Environment %s:%s duplicated to %s:%s.",
            name,
            resp["source_version"],
            name,
            new_version,
        )
        return resp

    def _create_and_validate(
        self,
        identity: EnvironmentIdentity,
        upload: EnvironmentSource,
        kind: EnvironmentKind,
    ) -> NDJsonStream:
        """Upload, create, and validate an environment."""
        source = upload.source
        assert source is not None
        requirements, env_config = discover_env_sidecars(
            source, requirements=upload.requirements, env_config=upload.env_config
        )
        archive_name, archive_payload = prepare_env_upload(source)
        data = self._create_environment_form_data(identity, upload, kind)
        files = self._create_environment_upload_files(
            archive_name, archive_payload, env_config, requirements
        )
        try:
            return self._open_stream(
                "POST",
                "/api/cli/v1/environments/create-and-validate",
                data=data,
                files=files,
                timeout=self._upload_timeout,
            )
        finally:
            self._close_upload_files(files)

    @staticmethod
    def _create_environment_form_data(
        identity: EnvironmentIdentity,
        upload: EnvironmentSource,
        kind: EnvironmentKind,
    ) -> dict[str, str]:
        """Build multipart text fields for create-and-validate."""
        data: dict[str, str] = {
            "name": identity.name,
            "version": identity.version or "v1",
            "multi_agent": str(kind.multi_agent).lower(),
            "language_based": str(kind.language_based).lower(),
            "do_rollouts": str(kind.do_rollouts).lower(),
        }
        if upload.entrypoint:
            data["entrypoint"] = upload.entrypoint
        if identity.description:
            data["description"] = identity.description
        return data

    @staticmethod
    def _create_environment_upload_files(
        archive_name: str,
        archive_payload: object,
        env_config: str | os.PathLike[str] | None,
        requirements: str | os.PathLike[str] | None,
    ) -> dict[str, tuple[str, Any, str]]:
        """Build multipart file parts for create-and-validate."""
        files: dict[str, tuple[str, Any, str]] = {
            "file": (archive_name, archive_payload, "application/gzip"),
        }
        if env_config is not None:
            is_json = Path(os.fspath(env_config)).suffix.lower() == ".json"
            files["env_config"] = prepare_file_upload(
                env_config,
                default_name="env_config.json" if is_json else "env_config.yaml",
                content_type="application/json" if is_json else "application/x-yaml",
            )
        else:
            files["env_config"] = ("env_config.yaml", b"", "application/x-yaml")
        if requirements is not None:
            files["requirements"] = prepare_file_upload(
                requirements,
                default_name="requirements.txt",
                content_type="text/plain",
            )
        else:
            files["requirements"] = ("requirements.txt", b"", "text/plain")
        return files
