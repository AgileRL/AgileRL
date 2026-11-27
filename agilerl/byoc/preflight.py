# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Read-only checks that a Kubernetes cluster can run the Arena BYOC agent.

Every check reads the cluster with kubectl. The Pod Security check sends a
server-side dry run, which Kubernetes admits or refuses without storing anything.
"""

from __future__ import annotations

import json
import subprocess
from collections import Counter
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import click

from agilerl.byoc.cluster_helm import AGENT_RELEASE, normalize_agent_release
from agilerl.byoc.provisioning.kubectl import kubeconfig_env
from agilerl.output import emit_result

GPU_RESOURCE = "nvidia.com/gpu"
GPU_PRODUCT_LABEL = "nvidia.com/gpu.product"


class CheckStatus(str, Enum):
    """Outcome of one preflight check."""

    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"


@dataclass(frozen=True)
class PreflightCheck:
    """One line of the preflight report."""

    name: str
    status: CheckStatus
    detail: str
    fix: str | None = None

    def to_json(self) -> dict[str, str | None]:
        """Return the check as a JSON object."""
        return {
            "name": self.name,
            "status": self.status.value,
            "detail": self.detail,
            "fix": self.fix,
        }


class KubectlError(Exception):
    """kubectl could not read what a check needs."""


def _kubectl(
    env: dict[str, str], args: list[str], stdin: str | None = None
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["kubectl", *args, "--request-timeout=30s"],
        env=env,
        input=stdin,
        check=False,
        capture_output=True,
        text=True,
    )


def _read_json(env: dict[str, str], args: list[str]) -> dict[str, Any]:
    result = _kubectl(env, [*args, "--output", "json"])
    if result.returncode != 0:
        raise KubectlError(result.stderr.strip())
    return json.loads(result.stdout)


def _can_i(env: dict[str, str], verb: str, resource: str, namespace: str | None) -> str:
    """Return kubectl's ``yes`` or ``no``, or its error text when it gave neither."""
    args = ["auth", "can-i", verb, resource]
    if namespace is not None:
        args += ["--namespace", namespace]
    result = _kubectl(env, args)
    answer = result.stdout.strip()
    if answer in ("yes", "no"):
        return answer
    return result.stderr.strip() or answer


def kubernetes_version_check(env: dict[str, str]) -> PreflightCheck:
    """Fail below Kubernetes 1.30, the first release with ValidatingAdmissionPolicy v1."""
    name = "kubernetes_version"
    try:
        server = _read_json(env, ["version"])["serverVersion"]
    except KubectlError as exc:
        return PreflightCheck(
            name, CheckStatus.FAIL, f"Could not read the server version: {exc}"
        )
    # Managed clusters report minors such as "30+".
    version = (int(server["major"]), int(server["minor"].rstrip("+")))
    detail = f"Kubernetes {version[0]}.{version[1]}"
    if version < (1, 30):
        return PreflightCheck(
            name,
            CheckStatus.FAIL,
            detail,
            "Upgrade the cluster to Kubernetes 1.30 or later. The agent chart needs "
            "ValidatingAdmissionPolicy v1.",
        )
    return PreflightCheck(name, CheckStatus.PASS, detail)


def permission_checks(env: dict[str, str]) -> list[PreflightCheck]:
    """Check the caller may create every cluster-scoped object the agent chart has."""
    required = (
        "namespaces",
        "clusterroles.rbac.authorization.k8s.io",
        "clusterrolebindings.rbac.authorization.k8s.io",
        "validatingadmissionpolicies.admissionregistration.k8s.io",
        "validatingadmissionpolicybindings.admissionregistration.k8s.io",
    )
    checks = [_permission_check(env, "create", resource, True) for resource in required]
    checks += [
        _permission_check(env, verb, "clusterroles.rbac.authorization.k8s.io", False)
        for verb in ("escalate", "bind")
    ]
    return checks


def _permission_check(
    env: dict[str, str], verb: str, resource: str, required: bool
) -> PreflightCheck:
    kind = resource.split(".")[0]
    name = f"can_{verb}_{kind}"
    answer = _can_i(env, verb, resource, None)
    if answer == "yes":
        return PreflightCheck(name, CheckStatus.PASS, f"You may {verb} {kind}")
    if answer != "no":
        return PreflightCheck(
            name, CheckStatus.WARN, f"Could not check {verb} {kind}: {answer}"
        )
    if required:
        return PreflightCheck(
            name,
            CheckStatus.FAIL,
            f"You may not {verb} {kind}",
            "Use a kubeconfig with cluster-admin, which the agent install needs once.",
        )
    return PreflightCheck(
        name,
        CheckStatus.WARN,
        f"You may not {verb} {kind}",
        "Without escalate and bind on clusterroles, Helm can create the agent's "
        "ClusterRoles only if you already hold every right they grant. "
        "cluster-admin has both.",
    )


def node_inventory_check(env: dict[str, str]) -> PreflightCheck:
    """List GPU nodes by product and warn when none or some lack a product label."""
    name = "node_inventory"
    try:
        nodes = _read_json(env, ["get", "nodes"])["items"]
    except KubectlError as exc:
        return PreflightCheck(name, CheckStatus.WARN, f"Could not list nodes: {exc}")
    groups: Counter[tuple[str, int]] = Counter()
    unlabelled = []
    for node in nodes:
        gpus = int(node["status"].get("allocatable", {}).get(GPU_RESOURCE, "0"))
        if gpus == 0:
            continue
        product = node["metadata"].get("labels", {}).get(GPU_PRODUCT_LABEL)
        if product is None:
            unlabelled.append(node["metadata"]["name"])
        groups[(product or "unlabelled", gpus)] += 1
    if not groups:
        return PreflightCheck(
            name,
            CheckStatus.WARN,
            f"{len(nodes)} nodes, none with allocatable {GPU_RESOURCE}",
            "Install the NVIDIA GPU Operator or device plugin so GPU nodes advertise "
            f"{GPU_RESOURCE}.",
        )
    described = ", ".join(
        f"{count} x {gpus} GPU {product}"
        for (product, gpus), count in sorted(groups.items())
    )
    detail = f"{len(nodes)} nodes; GPU nodes: {described}"
    if unlabelled:
        return PreflightCheck(
            name,
            CheckStatus.WARN,
            f"{detail}. No {GPU_PRODUCT_LABEL} label on {', '.join(unlabelled)}",
            "Install NVIDIA GPU Feature Discovery (part of the GPU Operator) or label "
            f"those nodes with {GPU_PRODUCT_LABEL}. Resource classes select GPU nodes "
            "by that label.",
        )
    return PreflightCheck(name, CheckStatus.PASS, detail)


def storage_class_check(
    env: dict[str, str], ray_data_storage_class_name: str | None
) -> PreflightCheck:
    """Check the StorageClass Ray data uses: the named one, else the default."""
    name = "storage_class"
    try:
        classes = _read_json(env, ["get", "storageclasses"])["items"]
    except KubectlError as exc:
        return PreflightCheck(
            name, CheckStatus.WARN, f"Could not list StorageClasses: {exc}"
        )
    names = [item["metadata"]["name"] for item in classes]
    if ray_data_storage_class_name is not None:
        if ray_data_storage_class_name in names:
            return PreflightCheck(
                name,
                CheckStatus.PASS,
                f"Ray data uses StorageClass {ray_data_storage_class_name}",
            )
        return PreflightCheck(
            name,
            CheckStatus.FAIL,
            f"StorageClass {ray_data_storage_class_name}, set for Ray data, does not exist",
            "Create that StorageClass or pass another --ray-data-storage-class-name.",
        )
    default_annotations = (
        "storageclass.kubernetes.io/is-default-class",
        "storageclass.beta.kubernetes.io/is-default-class",
    )
    defaults = [
        item["metadata"]["name"]
        for item in classes
        if any(
            item["metadata"].get("annotations", {}).get(key) == "true"
            for key in default_annotations
        )
    ]
    if len(defaults) == 1:
        return PreflightCheck(
            name, CheckStatus.PASS, f"Default StorageClass {defaults[0]}"
        )
    detail = (
        f"Several default StorageClasses: {', '.join(defaults)}"
        if defaults
        else "No default StorageClass"
    )
    return PreflightCheck(
        name,
        CheckStatus.WARN,
        detail,
        "Ray data and the agent's provisioner volume use the default StorageClass. "
        "Mark exactly one default (annotation storageclass.kubernetes.io/"
        'is-default-class: "true") or pass --ray-data-storage-class-name.',
    )


def _dry_run_privileged_pod(
    env: dict[str, str], namespace: str
) -> subprocess.CompletedProcess[str]:
    """Ask the API server to admit, without storing, a pod shaped like the s3fs sidecar."""
    pod = {
        "apiVersion": "v1",
        "kind": "Pod",
        "metadata": {"name": "arena-preflight-privileged"},
        "spec": {
            "containers": [
                {
                    "name": "probe",
                    "image": "registry.k8s.io/pause:3.10",
                    "securityContext": {"privileged": True},
                    "volumeMounts": [{"name": "fuse", "mountPath": "/dev/fuse"}],
                }
            ],
            "volumes": [{"name": "fuse", "hostPath": {"path": "/dev/fuse"}}],
        },
    }
    return _kubectl(
        env,
        [
            "create",
            "--dry-run=server",
            "--namespace",
            namespace,
            "--filename",
            "-",
            "--output",
            "name",
        ],
        stdin=json.dumps(pod),
    )


def privileged_pod_check(env: dict[str, str], agent_namespace: str) -> PreflightCheck:
    """Dry-run a privileged pod where the agent goes to see whether admission allows it.

    The agent's WireGuard container and s3fs sidecar run privileged. Without the
    agent namespace yet, the ``default`` namespace stands in for the cluster's
    defaults that Helm's new namespace gets.
    """
    name = "privileged_pods"
    found = _kubectl(
        env,
        ["get", "namespace", agent_namespace, "--ignore-not-found", "--output", "name"],
    )
    if found.returncode != 0:
        return PreflightCheck(
            name,
            CheckStatus.WARN,
            f"Could not read namespace {agent_namespace}: {found.stderr.strip()}",
        )
    # --ignore-not-found exits 0 with empty stdout when the namespace is missing.
    probe_namespace = agent_namespace if found.stdout.strip() else "default"
    if _can_i(env, "create", "pods", probe_namespace) != "yes":
        return PreflightCheck(
            name,
            CheckStatus.WARN,
            f"Could not check: you may not create pods in namespace {probe_namespace}",
        )
    result = _dry_run_privileged_pod(env, probe_namespace)
    where = f"namespace {probe_namespace}"
    if probe_namespace != agent_namespace:
        where += f" (namespace {agent_namespace} does not exist yet)"
    if result.returncode == 0:
        return PreflightCheck(
            name, CheckStatus.PASS, f"A privileged pod is admitted in {where}"
        )
    error = result.stderr.strip()
    refused_by_admission = (
        "violates PodSecurity" in error
        or "denied the request" in error
        or "denied request" in error
    )
    if refused_by_admission:
        return PreflightCheck(
            name,
            CheckStatus.FAIL,
            f"A privileged pod is refused in {where}: {error}",
            f"Label namespace {agent_namespace} pod-security.kubernetes.io/enforce="
            "privileged (create it first if needed), or exempt it from the policy that "
            "refused the pod.",
        )
    if "(AlreadyExists)" in error:
        return PreflightCheck(
            name,
            CheckStatus.WARN,
            f"Could not check in {where}: {error}",
            "Delete pod arena-preflight-privileged in that namespace and run again. "
            "The dry run uses that name.",
        )
    return PreflightCheck(
        name, CheckStatus.WARN, f"Could not check in {where}: {error}"
    )


def _deployment_release(metadata: dict[str, Any]) -> str:
    """Return the agent release recorded on a Deployment."""
    return metadata.get("labels", {}).get(
        "app.kubernetes.io/instance", metadata["name"]
    )


def own_agent_release(kubeconfig: Path, agent_namespace: str, cluster_name: str) -> str:
    """Return the agent release a reinstall keeps for *cluster_name*.

    ``arena-byoc-agent`` already in *agent_namespace* is that release.
    Otherwise it is the cluster name. Read from the agent's Deployments.

    :param kubeconfig: Kubeconfig whose current context is the cluster.
    :param agent_namespace: Namespace the agent chart installs into.
    :param cluster_name: Registered cluster name.
    :return: Release ``one_agent`` should treat as this cluster's own.
    """
    fallback = normalize_agent_release(cluster_name)
    try:
        deployments = _read_json(
            kubeconfig_env(kubeconfig),
            [
                "get",
                "deployments",
                "--namespace",
                agent_namespace,
                "--selector",
                "app.kubernetes.io/name=arena-byoc-agent",
            ],
        )["items"]
    except KubectlError:
        return fallback
    releases = {
        _deployment_release(deployment["metadata"])
        for deployment in deployments
        if deployment["metadata"].get("namespace", agent_namespace) == agent_namespace
    }
    if AGENT_RELEASE in releases:
        return AGENT_RELEASE
    return fallback


def one_agent_check(
    env: dict[str, str], agent_namespace: str, own_agent_release: str | None
) -> PreflightCheck:
    """Fail when an Arena agent other than this cluster's own is installed.

    :param own_agent_release: Release of the agent this cluster already runs in
        *agent_namespace*, which a reinstall upgrades in place. ``None`` treats
        every installed agent as another one.
    """
    name = "one_agent"
    try:
        deployments = _read_json(
            env,
            [
                "get",
                "deployments",
                "--all-namespaces",
                "--selector",
                "app.kubernetes.io/name=arena-byoc-agent",
            ],
        )["items"]
    except KubectlError as exc:
        return PreflightCheck(
            name, CheckStatus.WARN, f"Could not list agent Deployments: {exc}"
        )
    others = set()
    for deployment in deployments:
        metadata = deployment["metadata"]
        namespace = metadata["namespace"]
        release = _deployment_release(metadata)
        if (namespace, release) != (agent_namespace, own_agent_release):
            others.add(f"{namespace}/{release}")
    if not others:
        return PreflightCheck(name, CheckStatus.PASS, "No other Arena agent")
    return PreflightCheck(
        name,
        CheckStatus.FAIL,
        f"Arena agents already in this cluster: {', '.join(sorted(others))}",
        "Arena runs one agent per Kubernetes cluster. Unregister the cluster that "
        "agent belongs to, or remove it with helm uninstall --namespace <namespace> "
        "<release>.",
    )


def run_preflight(
    kubeconfig: Path,
    agent_namespace: str,
    own_agent_release: str | None,
    ray_data_storage_class_name: str | None,
) -> list[PreflightCheck]:
    """Run every preflight check against the cluster *kubeconfig* points at.

    :param kubeconfig: Kubeconfig whose current context is the cluster.
    :param agent_namespace: Namespace the agent chart installs into.
    :param own_agent_release: Release of this cluster's own agent in
        *agent_namespace*, or ``None`` when none is expected.
    :param ray_data_storage_class_name: StorageClass for Ray data, or ``None``
        for the cluster default.
    :return: The checks in report order.
    """
    env = kubeconfig_env(kubeconfig)
    return [
        kubernetes_version_check(env),
        *permission_checks(env),
        node_inventory_check(env),
        storage_class_check(env, ray_data_storage_class_name),
        privileged_pod_check(env, agent_namespace),
        one_agent_check(env, agent_namespace, own_agent_release),
    ]


def any_failed(checks: list[PreflightCheck]) -> bool:
    """Return whether any check failed."""
    return any(check.status is CheckStatus.FAIL for check in checks)


def echo_checks(checks: list[PreflightCheck]) -> None:
    """Print *checks* as a table with one row per check."""
    emit_result(
        [
            {
                "check": check.name,
                "result": check.status.value.upper(),
                "detail": check.detail,
                "fix": check.fix or "",
            }
            for check in checks
        ],
        columns=["check", "result", "detail", "fix"],
    )


def preflight_report(
    context: str, server: str, cluster_uid: str, checks: list[PreflightCheck]
) -> dict[str, Any]:
    """Return the report ``--json`` prints and the web wizard reads back."""
    return {
        "context": context,
        "server": server,
        "cluster_uid": cluster_uid,
        "passed": not any_failed(checks),
        "checks": [check.to_json() for check in checks],
    }


def require_passed(checks: list[PreflightCheck]) -> None:
    """Stop with the failed check names when any check failed."""
    failed = [check.name for check in checks if check.status is CheckStatus.FAIL]
    if failed:
        msg = (
            f"Preflight failed: {', '.join(failed)}. Fix them and run again; nothing "
            "was changed in Arena or the cluster."
        )
        raise click.ClickException(msg)
