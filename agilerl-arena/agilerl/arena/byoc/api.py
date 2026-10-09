# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Typed wrapper over the Arena BYOC HTTP endpoints.

:class:`ByocApi` is the single seam between BYOC orchestration code and the
:class:`~agilerl.arena.client.ArenaClient` manifest machinery, so installers and
commands never construct raw invoke descriptors themselves.
"""

from __future__ import annotations

import logging
from typing import Any, TypedDict

from typing_extensions import NotRequired, Unpack

from agilerl.arena.byoc import endpoints
from agilerl.arena.byoc.endpoints import SetupKind
from agilerl.arena.client import ArenaClient
from agilerl.arena.exceptions import ArenaAPIError

logger = logging.getLogger("agilerl.arena.byoc")


def _named_item(items: object, name: str, kind: str) -> dict[str, Any] | None:
    """Return the single dict named *name* from a list payload, or ``None``.

    :param items: Raw list response (expected to be a list of dicts).
    :type items: object
    :param name: The resource name to look up.
    :type name: str
    :param kind: Label used in the duplicate-name error (for example ``classes``).
    :type kind: str
    :returns: The matching dictionary, or ``None`` if absent.
    :rtype: dict[str, Any] | None
    :raises ArenaAPIError: If more than one item shares the same name.
    """
    if not isinstance(items, list):
        return None
    matches = [c for c in items if isinstance(c, dict) and c.get("name") == name]
    if not matches:
        return None
    if len(matches) > 1:
        msg = f"Multiple BYOC {kind} named {name!r}; resolve duplicates in Arena first."
        raise ArenaAPIError(msg)
    return {str(k): v for k, v in matches[0].items()}


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
    return _named_item(classes, name, kind="classes")


def cluster_by_name(clusters: object, name: str) -> dict[str, Any] | None:
    """Return the single active cluster named *name*, or ``None`` if absent.

    :param clusters: The raw ``clusters/list`` response (expected to be a list).
    :type clusters: object
    :param name: The cluster name to look up.
    :type name: str
    :returns: The matching cluster dictionary, or ``None`` if absent.
    :rtype: dict[str, Any] | None
    :raises ArenaAPIError: If more than one cluster shares the same name.
    """
    return _named_item(clusters, name, kind="clusters")


class RegisterClusterFields(TypedDict):
    """Keyword fields for ``ByocApi.register_cluster``."""

    name: str
    install_storage: NotRequired[bool | None]
    narrow_allowed_ips: NotRequired[bool | None]
    storage_endpoint: NotRequired[str | None]
    storage_bucket: NotRequired[str | None]
    storage_prefix: NotRequired[str | None]
    storage_secret_name: NotRequired[str | None]
    ingress_class_name: NotRequired[str | None]
    hostname_template: NotRequired[str | None]
    inference_domain: NotRequired[str | None]
    gateway_api_parent_refs: NotRequired[object | None]
    tls_secret_name: NotRequired[str | None]
    preprocessing_resource_class: NotRequired[str | None]
    ray_data_storage_class_name: NotRequired[str | None]
    ray_data_pvc_size: NotRequired[str | None]
    byoc_provider: NotRequired[dict[str, Any] | None]


class ByocApi:
    """Talks to the Arena BYOC endpoints through an :class:`ArenaClient`."""

    def __init__(self, client: ArenaClient) -> None:
        """Wrap an :class:`ArenaClient` with the BYOC endpoint operations.

        :param client: The authenticated Arena client to issue requests through.
        :type client: ArenaClient
        """
        self._client = client

    def enable(self) -> None:
        """Enable the BYOC provider for the account.

        :returns: None
        :rtype: None
        """
        logger.info("Enabling BYOC provider…")
        self._client._invoke_manifest_command(endpoints.ENABLE, {})

    def disable(self) -> None:
        """Disable the BYOC provider for the account.

        :returns: None
        :rtype: None
        """
        logger.info("Disabling BYOC provider…")
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
        logger.info("Deleting BYOC resource class %r from Arena…", name)
        self._client._invoke_manifest_command(endpoints.DELETE_CLASS, {"name": name})

    def register_cluster(
        self, **fields: Unpack[RegisterClusterFields]
    ) -> dict[str, Any]:
        """Register a BYOC cluster and return the registration bundle.

        :param fields: Cluster name plus optional storage and inference settings.
        :returns: Registration bundle (cluster name, token, Helm values, etc.).
        :rtype: dict[str, Any]
        """
        name = fields["name"]
        install_storage = fields.get("install_storage")
        narrow_allowed_ips = fields.get("narrow_allowed_ips")
        storage_endpoint = fields.get("storage_endpoint")
        storage_bucket = fields.get("storage_bucket")
        storage_prefix = fields.get("storage_prefix")
        storage_secret_name = fields.get("storage_secret_name")
        ingress_class_name = fields.get("ingress_class_name")
        hostname_template = fields.get("hostname_template")
        inference_domain = fields.get("inference_domain")
        gateway_api_parent_refs = fields.get("gateway_api_parent_refs")
        tls_secret_name = fields.get("tls_secret_name")
        preprocessing_resource_class = fields.get("preprocessing_resource_class")
        ray_data_storage_class_name = fields.get("ray_data_storage_class_name")
        ray_data_pvc_size = fields.get("ray_data_pvc_size")
        byoc_provider = fields.get("byoc_provider")
        body: dict[str, Any] = {"name": name}
        optional_fields: list[tuple[str, Any]] = [
            ("installStorage", install_storage),
            ("narrowAllowedIps", narrow_allowed_ips),
            ("storageEndpoint", storage_endpoint),
            ("storageBucket", storage_bucket),
            ("storagePrefix", storage_prefix),
            ("storageSecretName", storage_secret_name),
            ("ingressClassName", ingress_class_name),
            ("hostnameTemplate", hostname_template),
            ("domain", inference_domain),
            ("gatewayApiParentRefs", gateway_api_parent_refs),
            ("tlsSecretName", tls_secret_name),
            ("preprocessingResourceClass", preprocessing_resource_class),
            ("rayDataStorageClassName", ray_data_storage_class_name),
            ("rayDataPvcSize", ray_data_pvc_size),
            ("byocProvider", byoc_provider),
        ]
        for key, value in optional_fields:
            if value is None or value is False:
                continue
            body[key] = value
        logger.info("Registering BYOC cluster %r…", name)
        result = self._client._invoke_manifest_command(
            endpoints.REGISTER_CLUSTER,
            body,
        )
        if not isinstance(result, dict):
            msg = "Unexpected cluster registration response from Arena."
            raise ArenaAPIError(msg)
        return result

    def list_clusters(self) -> object:
        """Return the raw ``clusters/list`` response (active clusters).

        :returns: The decoded ``clusters/list`` payload (expected to be a list).
        :rtype: object
        """
        return self._client._invoke_manifest_command(endpoints.LIST_CLUSTERS, {})

    def find_cluster(self, name: str) -> dict[str, Any] | None:
        """Return the active cluster named *name*, or ``None`` if it does not exist.

        :param name: The cluster name to look up.
        :type name: str
        :returns: The matching cluster dictionary, or ``None`` if absent.
        :rtype: dict[str, Any] | None
        """
        return cluster_by_name(self.list_clusters(), name)

    def on_prem_cluster(self, name: str) -> dict[str, Any] | None:
        """Return the on-prem cluster named *name*, or ``None``.

        :param name: The cluster name to look up.
        :type name: str
        :returns: The cluster record, including ``byoc_provider`` when set.
        :rtype: dict[str, Any] | None
        :raises ArenaAPIError: If more than one cluster shares *name*.
        """
        payload = self._client._request("GET", "/api/on-prem-clusters")
        rows: object = payload
        if isinstance(payload, dict):
            rows = payload.get("data")
        if not isinstance(rows, list):
            return None
        matches = [
            row for row in rows if isinstance(row, dict) and row.get("name") == name
        ]
        if not matches:
            return None
        if len(matches) > 1:
            msg = (
                f"Multiple BYOC clusters named {name!r}; "
                "resolve duplicates in Arena first."
            )
            raise ArenaAPIError(msg)
        return {str(key): value for key, value in matches[0].items()}

    def stored_provider_config(self, name: str, provider: str) -> dict[str, Any] | None:
        """Return the cloud config Arena stored for *name*, or ``None``.

        :param name: The cluster name to look up.
        :type name: str
        :param provider: Cloud provider id, such as ``nebius``.
        :type provider: str
        :returns: ``byoc_provider.config`` when the stored provider matches.
        :rtype: dict[str, Any] | None
        :raises ArenaAPIError: If more than one cluster shares *name*.
        """
        row = self.on_prem_cluster(name)
        if row is None:
            return None
        stored = row.get("byoc_provider")
        if not isinstance(stored, dict):
            stored = row.get("byocProvider")
        if not isinstance(stored, dict) or stored.get("provider") != provider:
            return None
        config = stored.get("config")
        if not isinstance(config, dict):
            return None
        return {str(key): value for key, value in config.items()}

    def unregister_cluster(self, name: str) -> object:
        """Unregister the BYOC cluster named *name*.

        :param name: The registered cluster name.
        :type name: str
        :returns: The decoded unregister response.
        :rtype: object
        """
        logger.info("Unregistering BYOC cluster %r…", name)
        return self._client._invoke_manifest_command(
            endpoints.UNREGISTER_CLUSTER,
            {"name": name},
        )

    def rotate_cluster_token(self, name: str) -> dict[str, Any]:
        """Rotate the agent token for the BYOC cluster named *name*.

        :param name: The registered cluster name.
        :type name: str
        :returns: Rotation bundle with the new token.
        :rtype: dict[str, Any]
        """
        logger.info("Rotating BYOC cluster token for %r…", name)
        result = self._client._invoke_manifest_command(
            endpoints.ROTATE_CLUSTER_TOKEN,
            {"name": name},
        )
        if not isinstance(result, dict):
            msg = "Unexpected cluster token rotation response from Arena."
            raise ArenaAPIError(msg)
        return result

    def create_class(
        self,
        name: str,
        num_nodes: int,
        cluster_name: str,
        node_selector: dict[str, str],
        metadata: dict[str, object],
    ) -> dict[str, Any]:
        """Create a BYOC resource class linked to a registered cluster."""
        logger.info(
            "Creating BYOC resource class %r for cluster %r…",
            name,
            cluster_name,
        )
        result = self._client._invoke_manifest_command(
            endpoints.CREATE_CLASS,
            {
                "name": name,
                "num_nodes": num_nodes,
                "clusterName": cluster_name,
                "nodeSelector": node_selector,
                "metadata": metadata,
            },
        )
        if not isinstance(result, dict):
            msg = "Unexpected resource class create response from Arena."
            raise ArenaAPIError(msg)
        return result

    def download_cluster_install_package(
        self,
        cluster: str,
        agent_helm_values_yaml: str,
        storage_helm_values_yaml: str | None = None,
    ) -> bytes:
        """Download the Helm install package tarball for a registered cluster.

        :param cluster: Registered cluster name.
        :type cluster: str
        :param agent_helm_values_yaml: Agent chart values YAML.
        :type agent_helm_values_yaml: str
        :param storage_helm_values_yaml: Lab storage values YAML.
        :type storage_helm_values_yaml: str | None
        :returns: ``application/gzip`` install package bytes.
        :rtype: bytes
        """
        body: dict[str, Any] = {
            "cluster": cluster,
            "agentHelmValuesYaml": agent_helm_values_yaml,
        }
        if storage_helm_values_yaml is not None:
            body["storageHelmValuesYaml"] = storage_helm_values_yaml
        logger.info("Downloading BYOC install package for cluster %r…", cluster)
        raw_b, _ctype, _disp = self._client._invoke_manifest_command(
            endpoints.INSTALL_CLUSTER_PACKAGE,
            body,
        )
        return raw_b

    def fetch_bundle(self, name: str, setup_type: SetupKind) -> bytes:
        """Download the deployment bundle zip for class *name* and return its bytes.

        :param name: The resource class name to download the bundle for.
        :type name: str
        :param setup_type: The bundle flavor (``helm``).
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
