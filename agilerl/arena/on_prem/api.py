"""Typed wrapper over the Arena on-prem HTTP endpoints.

:class:`OnPremApi` is the single seam between on-prem orchestration code and the
:class:`~agilerl.arena.client.ArenaClient` manifest machinery, so installers and
commands never construct raw invoke descriptors themselves.
"""

from __future__ import annotations

import logging
from typing import Any

from agilerl.arena.client import ArenaClient
from agilerl.arena.exceptions import ArenaAPIError
from agilerl.arena.on_prem import endpoints
from agilerl.arena.on_prem.endpoints import SetupKind

logger = logging.getLogger("agilerl.arena.on_prem")


def class_by_name(classes: object, name: str) -> dict[str, Any] | None:
    """Return the single resource class named *name*, or ``None`` if absent.

    *classes* is the raw ``classes/list`` response, so it is typed ``object``
    and validated to be a list here.

    :param classes: The raw ``classes/list`` response (expected to be a list).
    :type classes: object
    :param name: The resource class name to look up.
    :type name: str
    :returns: The matching class dictionary, or ``None`` if no class matches.
    :rtype: dict[str, Any] | None
    :raises ArenaAPIError: If more than one class shares the same name.
    """
    if not isinstance(classes, list):
        return None
    matches = [c for c in classes if isinstance(c, dict) and c.get("name") == name]
    if not matches:
        return None
    if len(matches) > 1:
        msg = f"Multiple on-prem classes named {name!r}; resolve duplicates in Arena first."
        raise ArenaAPIError(msg)
    return matches[0]


def resolve_num_nodes(
    existing: dict[str, Any] | None,
    *,
    explicit: int | None,
    default: int,
) -> int:
    """Decide the node count when creating a class.

    Prefers an existing class's ``num_nodes``, then an explicit ``--num-nodes``,
    then the per-flavor *default* supplied by the installer.

    :param existing: The existing resource class, or ``None`` if it does not exist.
    :type existing: dict[str, Any] | None
    :param explicit: The node count passed via ``--num-nodes``, if any.
    :type explicit: int | None
    :param default: The per-flavor fallback node count.
    :type default: int
    :returns: The resolved node count.
    :rtype: int
    """
    if existing is not None:
        raw = existing.get("num_nodes")
        if isinstance(raw, int) and raw > 0:
            return raw
    if explicit is not None:
        return explicit
    return default


class OnPremApi:
    """Talks to the Arena on-prem endpoints through an :class:`ArenaClient`."""

    def __init__(self, client: ArenaClient) -> None:
        """Wrap an :class:`ArenaClient` with the on-prem endpoint operations.

        :param client: The authenticated Arena client to issue requests through.
        :type client: ArenaClient
        """
        self._client = client

    def enable(self) -> None:
        """Enable the on-prem provider for the account.

        :returns: None
        :rtype: None
        """
        logger.info("Enabling on-prem provider…")
        self._client._invoke_manifest_command(endpoints.ENABLE, {})

    def disable(self) -> None:
        """Disable the on-prem provider for the account.

        :returns: None
        :rtype: None
        """
        logger.info("Disabling on-prem provider…")
        self._client._invoke_manifest_command(endpoints.DISABLE, {})

    def list_classes(self) -> object:
        """Return the raw ``classes/list`` response (a list of class dicts).

        :returns: The decoded ``classes/list`` payload (expected to be a list).
        :rtype: object
        """
        return self._client._invoke_manifest_command(endpoints.LIST_CLASSES, {})

    def find_class(self, name: str) -> dict[str, Any] | None:
        """Return the resource class named *name*, or ``None`` if it does not exist.

        :param name: The resource class name to look up.
        :type name: str
        :returns: The matching class dictionary, or ``None`` if absent.
        :rtype: dict[str, Any] | None
        """
        return class_by_name(self.list_classes(), name)

    def ensure_class(self, name: str, *, num_nodes: int) -> dict[str, Any]:
        """Return the existing resource class named *name*, creating it if needed.

        :param name: The resource class name.
        :type name: str
        :param num_nodes: Node count to use if the class must be created.
        :type num_nodes: int
        :returns: The existing or newly created resource class.
        :rtype: dict[str, Any]
        :raises ArenaAPIError: If the create response is not a JSON object.
        """
        existing = self.find_class(name)
        if existing is not None:
            logger.info("Using existing resource class %r.", name)
            return existing

        logger.info("Creating resource class %r (%d nodes)…", name, num_nodes)
        body: dict[str, Any] = {
            "name": name,
            "num_nodes": num_nodes,
            "enabled": True,
            "metadata": endpoints.DEFAULT_METADATA,
        }
        created = self._client._invoke_manifest_command(endpoints.CREATE_CLASS, body)
        if isinstance(created, dict):
            return created
        msg = "Create class response was not an object."
        raise ArenaAPIError(msg)

    def delete_class(self, name: str) -> None:
        """Delete the resource class named *name* if it is registered in Arena.

        :param name: The resource class name to delete.
        :type name: str
        :returns: None
        :rtype: None
        """
        if self.find_class(name) is None:
            logger.info("No Arena resource class %r; skipping API delete.", name)
            return
        logger.info("Deleting on-prem resource class %r from Arena…", name)
        self._client._invoke_manifest_command(endpoints.DELETE_CLASS, {"name": name})

    def fetch_bundle(self, name: str, setup_type: SetupKind) -> bytes:
        """Download the deployment bundle zip for class *name* and return its bytes.

        :param name: The resource class name to download the bundle for.
        :type name: str
        :param setup_type: The bundle flavor (``dockerSwarm`` or ``helm``).
        :type setup_type: SetupKind
        :returns: The raw bytes of the deployment bundle zip.
        :rtype: bytes
        """
        raw_b, _ctype, _disp = self._client._invoke_manifest_command(
            endpoints.BUNDLE,
            {
                "name": name,
                "setupType": setup_type,
                "archivedType": "zip",
            },
        )
        return raw_b
