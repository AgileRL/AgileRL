# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Typed wrapper over the Arena BYOC HTTP endpoints.

:class:`ByocApi` is the single seam between BYOC orchestration code and the
:class:`~agilerl.client.ArenaClient` manifest machinery, so installers and
commands never construct raw invoke descriptors themselves.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from http import HTTPStatus
from typing import Any, TypedDict

from typing_extensions import NotRequired, Unpack

from agilerl.byoc import endpoints
from agilerl.byoc.endpoints import SetupKind
from agilerl.byoc.provisioning.spec import api_byoc_scope, api_placement
from agilerl.client import ArenaClient, ManifestInvoke
from agilerl.exceptions import ArenaAPIError

logger = logging.getLogger("agilerl.byoc")
PLATFORM_STAFF_ONLY = "Only AgileRL staff can manage platform clusters."


class DuplicateNameError(ArenaAPIError):
    """More than one listed BYOC row has the name being looked up."""


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
    :raises DuplicateNameError: If more than one item shares the same name.
    """
    if not isinstance(items, list):
        return None
    matches = [c for c in items if isinstance(c, dict) and c.get("name") == name]
    if not matches:
        return None
    if len(matches) > 1:
        msg = f"Multiple BYOC {kind} named {name!r}; resolve duplicates in Arena first."
        raise DuplicateNameError(msg)
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
    :raises DuplicateNameError: If more than one class shares the same name.
    """
    return _named_item(classes, name, kind="classes")


def class_ref(
    name: str, cluster: str | None, cluster_id: int | None
) -> dict[str, object]:
    """Return the query naming a class, plus its cluster when one is given.

    :param name: The resource class name.
    :type name: str
    :param cluster: Name of the cluster the class belongs to.
    :type cluster: str | None
    :param cluster_id: Id of the cluster the class belongs to.
    :type cluster_id: int | None
    :returns: ``name`` with ``cluster`` and ``cluster_id`` when set.
    :rtype: dict[str, object]
    """
    ref: dict[str, object] = {"name": name}
    if cluster is not None:
        ref["cluster"] = cluster
    if cluster_id is not None:
        ref["cluster_id"] = cluster_id
    return ref


def cluster_by_name(clusters: object, name: str) -> dict[str, Any] | None:
    """Return the single active cluster named *name*, or ``None`` if absent.

    :param clusters: The raw ``clusters/list`` response (expected to be a list).
    :type clusters: object
    :param name: The cluster name to look up.
    :type name: str
    :returns: The matching cluster dictionary, or ``None`` if absent.
    :rtype: dict[str, Any] | None
    :raises DuplicateNameError: If more than one cluster shares the same name.
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
    placement: NotRequired[str | None]
    byoc_scope: NotRequired[str | None]
    byoc_provider: NotRequired[dict[str, Any] | None]
    cluster_uid: NotRequired[str | None]
    rotate_token: NotRequired[bool]


class ByocApi:
    """Talks to the Arena BYOC endpoints through an :class:`ArenaClient`."""

    def __init__(self, client: ArenaClient, platform: bool = False) -> None:
        """Wrap an :class:`ArenaClient` with the BYOC endpoint operations.

        :param client: The authenticated Arena client to issue requests through.
        :type client: ArenaClient
        :param platform: Address AgileRL's platform clusters instead of the
            active org's clusters on every cluster and class call.
        :type platform: bool
        """
        self._client = client
        self._scope: dict[str, str] = {"byocScope": "platform"} if platform else {}
        self._listed_classes: object | None = None

    @contextmanager
    def _platform_access(self) -> Iterator[None]:
        """Name the staff requirement when Arena refuses a platform call."""
        try:
            yield
        except ArenaAPIError as exc:
            if self._scope and exc.status_code == HTTPStatus.FORBIDDEN:
                raise ArenaAPIError(
                    PLATFORM_STAFF_ONLY, status_code=exc.status_code
                ) from exc
            raise

    def _invoke(self, invoke: ManifestInvoke, args: Mapping[str, object]) -> object:
        """Call a cluster or class route and return its decoded JSON."""
        with self._platform_access():
            return self._client._invoke_manifest_command(
                invoke, {**args, **self._scope}
            )

    def _download(self, invoke: ManifestInvoke, args: Mapping[str, object]) -> bytes:
        """Call a binary cluster or class route and return its body."""
        with self._platform_access():
            raw_b, _ctype, _disp = self._client._invoke_manifest_command(
                invoke, {**args, **self._scope}
            )
        return raw_b

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
        return self._invoke(endpoints.LIST_CLASSES, {})

    def load_classes(self) -> None:
        """Fetch resource classes once for later lookups."""
        self._listed_classes = self.list_classes()

    def find_class(self, name: str) -> dict[str, Any] | None:
        """Return the resource class named *name*, or ``None`` if it does not exist.

        :param name: The resource class name to look up.
        :type name: str
        :returns: The matching class dictionary, or ``None`` if absent.
        :rtype: dict[str, Any] | None
        """
        classes = (
            self.list_classes()
            if self._listed_classes is None
            else self._listed_classes
        )
        return class_by_name(classes, name)

    def delete_class(
        self, name: str, cluster: str | None = None, cluster_id: int | None = None
    ) -> None:
        """Delete the resource class named *name* if it is registered in Arena.

        :param name: The resource class name to delete.
        :type name: str
        :param cluster: Name of the class's cluster, needed when the class name
            matches classes on more than one cluster.
        :type cluster: str | None
        :param cluster_id: Id of the class's cluster, needed when several
            clusters share the cluster name.
        :type cluster_id: int | None
        :returns: None
        :rtype: None
        """
        # classes/list rows carry no cluster; with one named, Arena resolves the class.
        if cluster is None and cluster_id is None and self.find_class(name) is None:
            logger.info("No Arena resource class %r; skipping API delete.", name)
            return
        logger.info("Deleting BYOC resource class %r from Arena…", name)
        self._invoke(endpoints.DELETE_CLASS, class_ref(name, cluster, cluster_id))

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
        placement = fields.get("placement")
        byoc_scope = fields.get("byoc_scope")
        byoc_provider = fields.get("byoc_provider")
        cluster_uid = fields.get("cluster_uid")
        rotate_token = fields.get("rotate_token")
        wire_placement = None if placement is None else api_placement(placement)
        wire_scope = None if byoc_scope is None else api_byoc_scope(byoc_scope)
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
            ("placement", wire_placement),
            ("byocScope", wire_scope),
            ("byocProvider", byoc_provider),
            ("clusterUid", cluster_uid),
            ("rotateToken", rotate_token),
        ]
        for key, value in optional_fields:
            if value is None or value is False:
                continue
            body[key] = value
        logger.info("Registering BYOC cluster %r…", name)
        with self._platform_access():
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
        return self._invoke(endpoints.LIST_CLUSTERS, {})

    def find_cluster(self, name: str) -> dict[str, Any] | None:
        """Return the active cluster named *name*, or ``None`` if it does not exist.

        :param name: The cluster name to look up.
        :type name: str
        :returns: The matching cluster dictionary, or ``None`` if absent.
        :rtype: dict[str, Any] | None
        """
        return cluster_by_name(self.list_clusters(), name)

    def byoc_cluster(self, name: str) -> dict[str, Any] | None:
        """Return the BYOC cluster named *name*, or ``None``.

        :param name: The cluster name to look up.
        :type name: str
        :returns: The cluster record, including ``byoc_provider`` when set.
        :rtype: dict[str, Any] | None
        :raises ArenaAPIError: If more than one cluster shares *name*.
        """
        payload = self._client._request("GET", "/api/byoc-clusters")
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
        row = self.byoc_cluster(name)
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

    def unregister_cluster(
        self, name: str, stop_runs: bool = False, force: bool = False
    ) -> dict[str, Any]:
        """Unregister the BYOC cluster named *name*; flags left false are not sent.

        :param name: The registered cluster name.
        :type name: str
        :param stop_runs: Stop the cluster's experiments and deployments first.
        :type stop_runs: bool
        :param force: Archive now without the agent and end its runs in Arena.
        :type force: bool
        :returns: The decoded unregister response.
        :rtype: dict[str, Any]
        :raises ArenaAPIError: If Arena answers with something other than an
            object.
        """
        logger.info("Unregistering BYOC cluster %r…", name)
        args: dict[str, object] = {"name": name}
        if stop_runs:
            args["stopRuns"] = True
        if force:
            args["force"] = True
        result = self._invoke(endpoints.UNREGISTER_CLUSTER, args)
        if not isinstance(result, dict):
            msg = "Unexpected cluster unregister response from Arena."
            raise ArenaAPIError(msg)
        return {str(key): value for key, value in result.items()}

    def renew_cluster_credential(self, name: str) -> dict[str, Any]:
        """Ask the agent of the BYOC cluster named *name* to renew its credential.

        :param name: The registered cluster name.
        :type name: str
        :returns: The cluster's credential state.
        :rtype: dict[str, Any]
        """
        logger.info(
            "Asking the agent of BYOC cluster %r to renew its credential…", name
        )
        return self._credential_call(endpoints.RENEW_CLUSTER_CREDENTIAL, name)

    def revoke_cluster_credentials(self, name: str) -> dict[str, Any]:
        """Revoke every credential of the BYOC cluster named *name* now.

        :param name: The registered cluster name.
        :type name: str
        :returns: The cluster's credential state.
        :rtype: dict[str, Any]
        """
        logger.info("Revoking every credential of BYOC cluster %r…", name)
        return self._credential_call(endpoints.REVOKE_CLUSTER_CREDENTIALS, name)

    def _credential_call(self, invoke: ManifestInvoke, name: str) -> dict[str, Any]:
        """Call a cluster credential route and return the credential state."""
        result = self._invoke(invoke, {"name": name})
        if not isinstance(result, dict):
            msg = "Unexpected cluster credential response from Arena."
            raise ArenaAPIError(msg)
        return result

    def request_cluster_smoke(self, name: str, cluster_uid: str) -> dict[str, Any]:
        """Ask the agent of cluster *name* to deploy and remove a small train release.

        :param name: The registered cluster name.
        :type name: str
        :param cluster_uid: kube-system namespace UID the CLI is pointed at.
            Arena refuses the smoke when it stores a different one.
        :type cluster_uid: str
        :returns: The smoke, or the one already running on the cluster.
        :rtype: dict[str, Any]
        :raises ArenaAPIError: If Arena answers with something other than an
            object.
        """
        result = self._invoke(
            endpoints.REQUEST_CLUSTER_SMOKE, {"name": name, "clusterUid": cluster_uid}
        )
        if not isinstance(result, dict):
            msg = "Unexpected cluster smoke response from Arena."
            raise ArenaAPIError(msg)
        return result

    def cluster_smoke(self, name: str, smoke_id: int) -> dict[str, Any]:
        """Return smoke *smoke_id* of cluster *name*.

        :param name: The registered cluster name.
        :type name: str
        :param smoke_id: Id Arena gave the smoke.
        :type smoke_id: int
        :returns: The smoke with its status and, once failed, the reason.
        :rtype: dict[str, Any]
        :raises ArenaAPIError: If Arena answers with something other than an
            object.
        """
        result = self._invoke(
            endpoints.GET_CLUSTER_SMOKE, {"name": name, "id": smoke_id}
        )
        if not isinstance(result, dict):
            msg = "Unexpected cluster smoke response from Arena."
            raise ArenaAPIError(msg)
        return result

    def create_class(
        self,
        name: str,
        num_nodes: int,
        cluster_name: str,
        node_selector: dict[str, str],
        metadata: dict[str, object],
        description: str = "",
        availabilities: list[dict[str, object]] | None = None,
        enabled: bool | None = None,
    ) -> dict[str, Any]:
        """Create a BYOC resource class linked to a registered cluster."""
        logger.info(
            "Creating BYOC resource class %r for cluster %r…",
            name,
            cluster_name,
        )
        body: dict[str, object] = {
            "name": name,
            "num_nodes": num_nodes,
            "clusterName": cluster_name,
            "nodeSelector": node_selector,
            "metadata": metadata,
        }
        if description:
            body["description"] = description
        if availabilities is not None:
            body["availabilities"] = availabilities
        if enabled is not None:
            body["enabled"] = enabled
        result = self._invoke(endpoints.CREATE_CLASS, body)
        if not isinstance(result, dict):
            msg = "Unexpected resource class create response from Arena."
            raise ArenaAPIError(msg)
        return result

    def suggested_classes(self, cluster: str) -> dict[str, Any]:
        """Return the classes Arena suggests from *cluster*'s node report.

        :param cluster: Registered cluster name.
        :type cluster: str
        :returns: ``suggestions`` (class create bodies), ``classes_without_nodes``,
            ``needs_agent_upgrade`` and ``reported_at``.
        :rtype: dict[str, Any]
        """
        result = self._invoke(endpoints.SUGGESTED_CLASSES, {"name": cluster})
        if not isinstance(result, dict):
            msg = "Unexpected class suggestions response from Arena."
            raise ArenaAPIError(msg)
        return {str(key): value for key, value in result.items()}

    def create_suggested_class(self, suggestion: Mapping[str, object]) -> None:
        """Create the class one of :meth:`suggested_classes`' suggestions describes.

        :param suggestion: A class create body from ``suggestions``.
        :type suggestion: Mapping[str, object]
        """
        body = {key: value for key, value in suggestion.items() if value is not None}
        logger.info("Creating suggested BYOC resource class %r…", body.get("name"))
        self._invoke(endpoints.CREATE_CLASS, body)

    def update_class(
        self,
        name: str,
        num_nodes: int,
        node_selector: dict[str, str],
        metadata: dict[str, object],
        description: str = "",
        availabilities: list[dict[str, object]] | None = None,
        enabled: bool | None = None,
        cluster: str | None = None,
        cluster_id: int | None = None,
    ) -> dict[str, Any]:
        """Update a BYOC resource class, on *cluster* or *cluster_id* when given."""
        logger.info("Updating BYOC resource class %r…", name)
        body: dict[str, object] = {
            **class_ref(name, cluster, cluster_id),
            "num_nodes": num_nodes,
            "node_selector": node_selector,
            "metadata": metadata,
        }
        if description:
            body["description"] = description
        if availabilities is not None:
            body["availabilities"] = availabilities
        if enabled is not None:
            body["enabled"] = enabled
        result = self._invoke(endpoints.UPDATE_CLASS, body)
        if not isinstance(result, dict):
            msg = "Unexpected resource class update response from Arena."
            raise ArenaAPIError(msg)
        return result

    def replace_class_availabilities(
        self,
        name: str,
        availabilities: list[dict[str, object]],
        enabled: bool | None = None,
        cluster: str | None = None,
        cluster_id: int | None = None,
    ) -> dict[str, Any]:
        """Replace Nebius region availability for a resource class."""
        logger.info("Updating availability for resource class %r…", name)
        body: dict[str, object] = {
            **class_ref(name, cluster, cluster_id),
            "availabilities": availabilities,
        }
        if enabled is not None:
            body["enabled"] = enabled
        result = self._invoke(endpoints.UPDATE_CLASS, body)
        if not isinstance(result, dict):
            msg = "Unexpected resource class update response from Arena."
            raise ArenaAPIError(msg)
        return result

    def fetch_cluster_install_artifacts(self, name: str) -> dict[str, Any]:
        """Return current agent Helm values for the cluster named *name*.

        Arena renders the values for the chart it would ship now. The response
        names existing Secrets and does not include a new cluster token.

        :param name: The registered cluster name.
        :type name: str
        :returns: Install-artifacts payload with ``agentHelmValuesYaml``.
        :rtype: dict[str, Any]
        """
        logger.info("Fetching BYOC install artifacts for %r…", name)
        result = self._invoke(endpoints.INSTALL_CLUSTER_ARTIFACTS, {"name": name})
        if not isinstance(result, dict):
            msg = "Unexpected cluster install artifacts response from Arena."
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
        return self._download(endpoints.INSTALL_CLUSTER_PACKAGE, body)

    def fetch_bundle(self, name: str, setup_type: SetupKind) -> bytes:
        """Download the deployment bundle zip for class *name* and return its bytes.

        :param name: The resource class name to download the bundle for.
        :type name: str
        :param setup_type: The bundle flavor (``helm``).
        :type setup_type: SetupKind
        :returns: The raw bytes of the deployment bundle zip.
        :rtype: bytes
        """
        return self._download(
            endpoints.BUNDLE,
            {"name": name, "setupType": setup_type, "archivedType": "zip"},
        )
