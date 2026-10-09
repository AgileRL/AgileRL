# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Helm install helpers for enterprise BYOC cluster registration."""

from __future__ import annotations

import base64
import json
import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import click
import yaml

from agilerl.arena.byoc.provisioning.kubectl import kubeconfig_env
from agilerl.arena.byoc.provisioning.storage import storage_secret_endpoint_data

logger = logging.getLogger("agilerl.arena.byoc")

STORAGE_RELEASE = "arena-byoc-storage"
STORAGE_NAMESPACE = "storage"
AGENT_RELEASE = "arena-byoc-agent"
AGENT_NAMESPACE = "arena"
STORAGE_CHART = "arena-byoc-storage"
AGENT_CHART = "arena-byoc-agent"
DEFAULT_HELM_WAIT_TIMEOUT = "10m"


def normalize_agent_release(release: str | None) -> str:
    """Return a Helm agent release name, defaulting to ``arena-byoc-agent``."""
    stripped = (release or "").strip()
    return stripped or AGENT_RELEASE


def normalize_agent_namespace(namespace: str | None) -> str:
    """Return a Helm agent namespace, defaulting to ``arena``."""
    stripped = (namespace or "").strip()
    return stripped or AGENT_NAMESPACE


def agent_deployment_selector(release: str = AGENT_RELEASE) -> str:
    """Return the Deployment label selector for an agent Helm release."""
    return f"app.kubernetes.io/instance={release},app.kubernetes.io/name={AGENT_CHART}"


AGENT_DEPLOYMENT_SELECTOR = agent_deployment_selector()


def resolve_helm_charts_root(charts_dir: Path | None) -> Path:
    """Return the platform ``resources/helm-setup`` directory.

    :param charts_dir: Explicit charts root, or ``None`` to use env/default.
    :type charts_dir: Path | None
    :returns: Absolute path to ``resources/helm-setup``.
    :rtype: Path
    :raises click.ClickException: If the directory cannot be resolved.
    """
    if charts_dir is not None:
        root = charts_dir.expanduser().resolve()
        if not root.is_dir():
            msg = f"Charts directory does not exist: {root}"
            raise click.ClickException(msg)
        return root

    env = os.environ.get("ARENA_HELM_CHARTS_DIR", "").strip()
    if env:
        root = Path(env).expanduser().resolve()
        if not root.is_dir():
            msg = f"ARENA_HELM_CHARTS_DIR is not a directory: {root}"
            raise click.ClickException(msg)
        return root

    msg = (
        "Helm charts directory is required when overriding charts. Pass --charts-dir "
        "pointing at agilerl-platform/resources/helm-setup or set "
        "ARENA_HELM_CHARTS_DIR."
    )
    raise click.ClickException(msg)


def _chart_path(charts_root: Path, chart_name: str) -> Path:
    chart = charts_root / chart_name / "chart"
    if not chart.is_dir():
        msg = f"Helm chart not found at {chart}"
        raise click.ClickException(msg)
    return chart


def helm_available() -> bool:
    """Return whether ``helm`` is on PATH."""
    return shutil.which("helm") is not None


def _require_helm() -> None:
    if not helm_available():
        msg = "helm not found on PATH; install Helm 3.x or omit --install."
        raise click.ClickException(msg)


def list_installed_byoc_releases(
    kubeconfig: Path | None,
    cluster_name: str | None = None,
) -> list[tuple[str, str]]:
    """Return installed BYOC Helm ``(release, namespace)`` pairs, agent first.

    The agent release is the cluster ``--name`` when that release exists, else
    ``arena-byoc-agent``. Helm is searched across all namespaces. Missing helm
    or kubeconfig yields an empty list.

    :param kubeconfig: Kubeconfig used as ``KUBECONFIG`` for Helm.
    :type kubeconfig: Path | None
    :param cluster_name: Registered cluster name, used as the Helm release name.
    :type cluster_name: str | None
    :returns: Installed BYOC releases.
    :rtype: list[tuple[str, str]]
    :raises click.ClickException: If ``helm list`` exits non-zero.
    """
    if kubeconfig is None or not helm_available():
        return []
    result = subprocess.run(
        ["helm", "list", "--all-namespaces", "--output", "json"],
        capture_output=True,
        check=False,
        text=True,
        env=kubeconfig_env(kubeconfig),
    )
    if result.returncode != 0:
        msg = (
            f"helm list --all-namespaces failed (exit {result.returncode}): "
            f"{result.stderr.strip()}"
        )
        raise click.ClickException(msg)
    named = (cluster_name or "").strip()
    named_agent: list[tuple[str, str]] = []
    default_agent: list[tuple[str, str]] = []
    storage: list[tuple[str, str]] = []
    for entry in json.loads(result.stdout or "[]"):
        release = entry.get("name")
        namespace = entry.get("namespace")
        if not isinstance(release, str) or not isinstance(namespace, str):
            continue
        if release == STORAGE_RELEASE:
            storage.append((release, namespace))
        elif named and release == named:
            named_agent.append((release, namespace))
        elif release == AGENT_RELEASE:
            default_agent.append((release, namespace))
    agent = named_agent or default_agent
    return agent + storage


def resolve_install_release(cluster_name: str, agent_namespace: str) -> str:
    """Return the Helm release to install the agent under in *agent_namespace*.

    The chart names its ClusterRole after the release, so two agents on one cluster
    need two release names; the cluster name gives that. An ``arena-byoc-agent``
    release already in that namespace is kept so its upgrade stays in place.

    :param cluster_name: Registered cluster name.
    :type cluster_name: str
    :param agent_namespace: Target namespace for the agent chart.
    :type agent_namespace: str
    :returns: Helm release name.
    :rtype: str
    :raises click.ClickException: If ``helm list`` exits non-zero.
    """
    if not helm_available():
        return normalize_agent_release(cluster_name)
    result = subprocess.run(
        ["helm", "list", "--namespace", agent_namespace, "--output", "json"],
        capture_output=True,
        check=False,
        text=True,
    )
    if result.returncode != 0:
        msg = (
            f"helm list --namespace {agent_namespace} failed "
            f"(exit {result.returncode}): {result.stderr.strip()}"
        )
        raise click.ClickException(msg)
    installed = {
        entry.get("name")
        for entry in json.loads(result.stdout or "[]")
        if isinstance(entry.get("name"), str)
    }
    if AGENT_RELEASE in installed:
        return AGENT_RELEASE
    return normalize_agent_release(cluster_name)


def discover_agent_install(
    kubeconfig: Path | None,
    cluster_name: str,
    agent_namespace: str | None = None,
) -> tuple[str, str]:
    """Return the agent Helm ``(release, namespace)`` for *cluster_name*.

    :param kubeconfig: Kubeconfig used as ``KUBECONFIG`` for Helm.
    :type kubeconfig: Path | None
    :param cluster_name: Registered cluster name, used as the Helm release name.
    :type cluster_name: str
    :param agent_namespace: Explicit namespace; used when non-empty.
    :type agent_namespace: str | None
    :returns: Agent release name and namespace.
    :rtype: tuple[str, str]
    :raises click.ClickException: If Helm lists that release in more than one
        namespace, or lists none while Helm can be queried.
    """
    release = normalize_agent_release(cluster_name)
    explicit = (agent_namespace or "").strip()
    installed = list_installed_byoc_releases(kubeconfig, cluster_name=cluster_name)
    agent = [
        (found_release, namespace)
        for found_release, namespace in installed
        if found_release != STORAGE_RELEASE
    ]
    if explicit:
        matching = [pair for pair in agent if pair[1] == explicit]
        if matching:
            return matching[0]
        return release, explicit
    if len(agent) == 1:
        return agent[0]
    if len(agent) > 1:
        listed = ", ".join(
            f"{found_release!r} in namespace {namespace!r}"
            for found_release, namespace in agent
        )
        msg = (
            f"Helm release {release} is installed in multiple namespaces: "
            f"{listed}. Pass --agent-namespace."
        )
        raise click.ClickException(msg)
    if kubeconfig is not None and helm_available():
        msg = (
            f"Helm release {release} was not found on this cluster. "
            "Pass --agent-namespace if the agent is installed without Helm."
        )
        raise click.ClickException(msg)
    return release, AGENT_NAMESPACE


def discover_agent_namespace(
    kubeconfig: Path | None,
    cluster_name: str,
    agent_namespace: str | None = None,
) -> str:
    """Return the agent Helm namespace for *cluster_name*."""
    _release, namespace = discover_agent_install(
        kubeconfig,
        cluster_name=cluster_name,
        agent_namespace=agent_namespace,
    )
    return namespace


def uninstall_byoc_releases(
    releases: list[tuple[str, str]],
    kubeconfig: Path,
) -> None:
    """Uninstall each Helm release, failing the command on the first error.

    :param releases: ``(release, namespace)`` pairs to uninstall.
    :type releases: list[tuple[str, str]]
    :param kubeconfig: Kubeconfig used as ``KUBECONFIG`` for Helm.
    :type kubeconfig: Path
    :returns: None
    :rtype: None
    :raises click.ClickException: If ``helm uninstall`` exits non-zero.
    """
    env = kubeconfig_env(kubeconfig)
    for release, namespace in releases:
        click.echo(f"Uninstalling Helm release {release} (namespace {namespace})…")
        result = subprocess.run(
            ["helm", "uninstall", release, "--namespace", namespace],
            check=False,
            env=env,
        )
        if result.returncode != 0:
            msg = (
                f"helm uninstall {release} --namespace {namespace} failed "
                f"(exit {result.returncode})"
            )
            raise click.ClickException(msg)


def _kubectl_secret_exists(name: str, namespace: str) -> bool:
    if not shutil.which("kubectl"):
        return False
    result = subprocess.run(
        ["kubectl", "get", "secret", name, "-n", namespace],
        capture_output=True,
        check=False,
    )
    return result.returncode == 0


def _ensure_k8s_namespace(namespace: str) -> None:
    exists = subprocess.run(
        ["kubectl", "get", "namespace", namespace],
        capture_output=True,
        check=False,
    )
    if exists.returncode == 0:
        return
    created = subprocess.run(
        ["kubectl", "create", "namespace", namespace],
        capture_output=True,
        check=False,
        text=True,
    )
    if created.returncode != 0:
        msg = f"Failed to create namespace {namespace!r}: {created.stderr.strip()}"
        raise click.ClickException(msg)
    click.echo(f"Created namespace {namespace}.")


def _copy_k8s_secret(name: str, source_namespace: str, target_namespace: str) -> None:
    get = subprocess.run(
        ["kubectl", "get", "secret", name, "-n", source_namespace, "-o", "json"],
        capture_output=True,
        check=False,
        text=True,
    )
    if get.returncode != 0:
        msg = (
            f"Secret {name!r} is missing in namespace {target_namespace!r} and could not "
            f"be copied from {source_namespace!r}. Re-run with --install-storage on first "
            "register, or create the secret manually."
        )
        raise click.ClickException(msg)
    payload = json.loads(get.stdout)
    metadata = payload.setdefault("metadata", {})
    for key in (
        "resourceVersion",
        "uid",
        "creationTimestamp",
        "managedFields",
        "annotations",
    ):
        metadata.pop(key, None)
    metadata["namespace"] = target_namespace
    # Helm creates the agent namespace later; the secret has to land there first.
    _ensure_k8s_namespace(target_namespace)
    apply = subprocess.run(
        ["kubectl", "apply", "-f", "-"],
        input=json.dumps(payload),
        capture_output=True,
        check=False,
        text=True,
    )
    if apply.returncode != 0:
        msg = f"Failed to copy secret {name!r} into {target_namespace!r}: {apply.stderr.strip()}"
        raise click.ClickException(msg)
    click.echo(f"Copied secret {name!r} from {source_namespace} to {target_namespace}.")


def ensure_agent_storage_secret(
    agent_values: dict[str, Any],
    source_namespace: str = STORAGE_NAMESPACE,
    agent_namespace: str = AGENT_NAMESPACE,
) -> None:
    """Copy lab MinIO credentials into the agent namespace when Helm will not create them.

    First register with ``--install-storage`` creates ``arena-storage`` in both the
    ``storage`` and ``arena`` namespaces. Upsert reinstall sets
    ``storage.createSecret: false``, so ``helm uninstall`` on the agent drops only the
    agent-namespace copy while MinIO keeps the ``storage`` namespace secret.

    :param agent_values: Parsed agent Helm values.
    :type agent_values: dict[str, Any]
    :param source_namespace: Namespace of the bundled MinIO release.
    :type source_namespace: str
    :param agent_namespace: Agent Helm release namespace.
    :type agent_namespace: str
    :returns: None
    :rtype: None
    """
    storage = agent_values.get("storage")
    if not isinstance(storage, dict) or storage.get("createSecret"):
        return
    secret_name = storage.get("secretName")
    if not isinstance(secret_name, str) or not secret_name.strip():
        return
    secret_name = secret_name.strip()
    if _kubectl_secret_exists(secret_name, agent_namespace):
        return
    if not _kubectl_secret_exists(secret_name, source_namespace):
        if not shutil.which("kubectl"):
            msg = (
                f"Secret {secret_name!r} is missing in namespace {agent_namespace!r}. "
                "Install kubectl and re-run, or create the secret before installing the agent."
            )
            raise click.ClickException(msg)
        return
    _copy_k8s_secret(secret_name, source_namespace, agent_namespace)


def sync_storage_secret_endpoint(
    agent_values: dict[str, Any],
    agent_namespace: str = AGENT_NAMESPACE,
) -> bool:
    """Write ``storage.endpoint`` into an existing Secret Helm will not recreate.

    :param agent_values: Parsed agent Helm values.
    :type agent_values: dict[str, Any]
    :param agent_namespace: Agent Helm release namespace.
    :type agent_namespace: str
    :returns: ``True`` when the Secret endpoint changed.
    :rtype: bool
    """
    storage = agent_values.get("storage")
    if not isinstance(storage, dict):
        return False
    # Helm renders this Secret only when createSecret is true.
    if storage.get("createSecret"):
        return False
    endpoint = storage.get("endpoint")
    secret_name = storage.get("secretName")
    if not isinstance(endpoint, str) or not endpoint.strip():
        return False
    if not isinstance(secret_name, str) or not secret_name.strip():
        return False
    endpoint = endpoint.strip()
    secret_name = secret_name.strip()
    if not _kubectl_secret_exists(secret_name, agent_namespace):
        return False
    desired = storage_secret_endpoint_data(endpoint)
    current = _storage_secret_endpoints(secret_name, agent_namespace, desired)
    if all(current.get(key) == value for key, value in desired.items()):
        return False
    _patch_storage_secret_endpoint(secret_name, agent_namespace, desired)
    click.echo(
        f"Updated Secret {secret_name!r} in namespace {agent_namespace} "
        f"to endpoint {endpoint}."
    )
    return True


def _storage_secret_endpoints(
    name: str, namespace: str, desired: dict[str, str]
) -> dict[str, str]:
    result = subprocess.run(
        ["kubectl", "get", "secret", name, "-n", namespace, "-o", "json"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        msg = (
            f"Failed to read Secret {name!r} in namespace {namespace!r}: "
            f"{result.stderr.strip()}"
        )
        raise click.ClickException(msg)
    payload = json.loads(result.stdout)
    data = payload.get("data")
    if not isinstance(data, dict):
        return {}
    endpoints: dict[str, str] = {}
    for key in desired:
        raw = data.get(key)
        if isinstance(raw, str) and raw:
            endpoints[key] = base64.b64decode(raw).decode()
    return endpoints


def _patch_storage_secret_endpoint(
    name: str, namespace: str, desired: dict[str, str]
) -> None:
    patch = {"stringData": desired}
    result = subprocess.run(
        [
            "kubectl",
            "patch",
            "secret",
            name,
            "-n",
            namespace,
            "--type",
            "merge",
            "-p",
            json.dumps(patch),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        msg = (
            f"Failed to update Secret {name!r} in namespace {namespace!r}: "
            f"{result.stderr.strip()}"
        )
        raise click.ClickException(msg)


def _restart_agent_deployment(namespace: str, release: str, *, wait: bool) -> None:
    """Roll the agent Deployment so it reloads the storage Secret."""
    selector = agent_deployment_selector(release)
    listed = subprocess.run(
        [
            "kubectl",
            "get",
            "deployment",
            "--namespace",
            namespace,
            "--selector",
            selector,
            "-o",
            "name",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if listed.returncode != 0:
        msg = (
            f"Failed to list agent Deployments in namespace {namespace!r}: "
            f"{listed.stderr.strip()}"
        )
        raise click.ClickException(msg)
    if not listed.stdout.strip():
        return
    click.echo(
        f"Restarting the agent Deployment in namespace {namespace} "
        "so it reloads the storage Secret."
    )
    restarted = subprocess.run(
        [
            "kubectl",
            "rollout",
            "restart",
            "deployment",
            "--namespace",
            namespace,
            "--selector",
            selector,
        ],
        check=False,
    )
    if restarted.returncode != 0:
        msg = (
            f"Failed to restart the agent Deployment in namespace {namespace!r} "
            f"(exit {restarted.returncode})"
        )
        raise click.ClickException(msg)
    if wait:
        wait_for_deployments(selector, namespace)


def _load_agent_values_yaml(values_file: Path) -> dict[str, Any]:
    data = yaml.safe_load(values_file.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        return data
    return {}


def _deep_merge_dict(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


def merge_agent_chart_defaults(
    chart_dir: Path, agent_values: dict[str, Any]
) -> dict[str, Any]:
    """Deep-merge registration values onto chart ``values.yaml`` defaults.

    Registration values win on conflict.

    :param chart_dir: Path to the agent chart directory.
    :type chart_dir: Path
    :param agent_values: Parsed agent Helm values from registration.
    :type agent_values: dict[str, Any]
    :returns: Merged Helm values.
    :rtype: dict[str, Any]
    :raises click.ClickException: If chart defaults are missing.
    """
    defaults_file = chart_dir / "values.yaml"
    if not defaults_file.is_file():
        msg = f"Chart values file not found: {defaults_file}"
        raise click.ClickException(msg)
    defaults = yaml.safe_load(defaults_file.read_text(encoding="utf-8"))
    if not isinstance(defaults, dict):
        defaults = {}
    return _deep_merge_dict(defaults, agent_values)


def write_merged_agent_values(package_root: Path, agent_values: dict[str, Any]) -> Path:
    """Merge chart defaults into *agent_values* and write an install-package values file.

    :param package_root: Extracted install package root directory.
    :type package_root: Path
    :param agent_values: Parsed agent Helm values from registration.
    :type agent_values: dict[str, Any]
    :returns: Path to the merged values file.
    :rtype: Path
    """
    chart_dir = _chart_path_from_package(package_root, AGENT_CHART)
    merged = merge_agent_chart_defaults(chart_dir, agent_values)
    out = package_root / "agent-helm-values-merged.yaml"
    out.write_text(
        yaml.safe_dump(merged, sort_keys=False, default_flow_style=False),
        encoding="utf-8",
    )
    return out


def run_agent_chart_validate(package_root: Path) -> None:
    """Run the agent chart ``validate.sh`` against stock chart values.

    The script helm-templates the chart. Its first check requires empty
    ``storage.bucket`` so validators stay omitted.

    :param package_root: Root containing ``arena-byoc-agent/`` (install package or
        ``resources/helm-setup``).
    :type package_root: Path
    :returns: None
    :rtype: None
    :raises click.ClickException: If validation exits non-zero.
    """
    agent_root = package_root / AGENT_CHART
    validate = agent_root / "validate.sh"
    if not validate.is_file():
        logger.warning("Agent chart has no validate.sh; check pods with kubectl.")
        return
    chart_dir = agent_root / "chart"
    env = os.environ.copy()
    env.pop("VALUES_FILE", None)
    env["CHART_DIR"] = str(chart_dir)
    logger.info("Running agent chart validation…")
    result = subprocess.run(
        [str(validate)],
        cwd=agent_root,
        env=env,
        check=False,
    )
    if result.returncode != 0:
        msg = f"Agent chart validation failed (exit {result.returncode})"
        raise click.ClickException(msg)


def _require_kubectl() -> None:
    if not shutil.which("kubectl"):
        msg = "kubectl not found on PATH; required to wait for the agent Deployment."
        raise click.ClickException(msg)


def wait_for_deployments(
    selector: str,
    namespace: str,
    timeout: str = DEFAULT_HELM_WAIT_TIMEOUT,
) -> None:
    """Block until Deployments matching *selector* report ``Available``.

    :param selector: Kubernetes label selector.
    :type selector: str
    :param namespace: Kubernetes namespace.
    :type namespace: str
    :param timeout: ``kubectl wait --timeout`` value.
    :type timeout: str
    :returns: None
    :rtype: None
    :raises click.ClickException: If kubectl is missing or wait fails.
    """
    _require_kubectl()
    cmd = [
        "kubectl",
        "wait",
        "--for=condition=available",
        "deployment",
        "--selector",
        selector,
        "--namespace",
        namespace,
        f"--timeout={timeout}",
    ]
    logger.info("Running %s", " ".join(cmd))
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        msg = (
            f"Timed out waiting for deployments matching {selector!r} in namespace "
            f"{namespace!r} (exit {result.returncode})"
        )
        raise click.ClickException(msg)


def helm_upgrade_install(
    release: str,
    chart: Path,
    namespace: str,
    values_file: Path,
    wait: bool = True,
    timeout: str = DEFAULT_HELM_WAIT_TIMEOUT,
    extra_sets: tuple[str, ...] = (),
    wait_for_selector: str | None = None,
) -> None:
    """Run ``helm upgrade --install`` for *release*.

    When *wait_for_selector* is set, Helm does not get ``--wait`` (that flag
    waits for every Deployment in the release, including env-validator and
    reward-fn-val). Readiness is checked on matching Deployments only.

    :param release: Helm release name.
    :type release: str
    :param chart: Path to the chart directory.
    :type chart: Path
    :param namespace: Target namespace.
    :type namespace: str
    :param values_file: Values file passed with ``-f``.
    :type values_file: Path
    :param wait: Wait for readiness when ``True``.
    :type wait: bool
    :param timeout: Helm ``--timeout`` / kubectl wait timeout.
    :type timeout: str
    :param extra_sets: Extra ``--set`` overrides applied after ``-f``.
    :type extra_sets: tuple[str, ...]
    :param wait_for_selector: Label selector used instead of Helm ``--wait``.
    :type wait_for_selector: str | None
    :returns: None
    :rtype: None
    :raises click.ClickException: If helm or the readiness wait exits non-zero.
    """
    _require_helm()
    if not values_file.is_file():
        msg = f"Helm values file not found: {values_file}"
        raise click.ClickException(msg)

    cmd = [
        "helm",
        "upgrade",
        "--install",
        release,
        str(chart),
        "--namespace",
        namespace,
        "--create-namespace",
        "-f",
        str(values_file),
    ]
    for helm_set in extra_sets:
        cmd.extend(["--set", helm_set])
    helm_wait = wait and wait_for_selector is None
    if helm_wait:
        cmd.extend(["--wait", "--timeout", timeout])

    logger.info("Running %s", " ".join(cmd))
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        msg = f"helm upgrade --install {release} failed (exit {result.returncode})"
        raise click.ClickException(msg)

    if wait and wait_for_selector is not None:
        wait_for_deployments(
            wait_for_selector,
            namespace,
            timeout=timeout,
        )


def install_from_install_package_root(
    package_root: Path,
    wait: bool = True,
    agent_release: str = AGENT_RELEASE,
    agent_namespace: str = AGENT_NAMESPACE,
    install_agent: bool = True,
) -> None:
    """Install storage (when values exist) then optionally the agent chart.

    :param package_root: Extracted install package root directory.
    :type package_root: Path
    :param wait: Wait for storage Helm resources and the agent Deployment.
    :type wait: bool
    :param agent_release: Helm release name for the agent chart.
    :type agent_release: str
    :param agent_namespace: Kubernetes namespace for the agent chart.
    :type agent_namespace: str
    :param install_agent: Install the agent chart after storage.
    :type install_agent: bool
    :returns: None
    :rtype: None
    """
    storage_values = package_root / "storage-helm-values.yaml"
    if storage_values.is_file():
        helm_upgrade_install(
            release=STORAGE_RELEASE,
            chart=_chart_path_from_package(package_root, STORAGE_CHART),
            namespace=STORAGE_NAMESPACE,
            values_file=storage_values,
            wait=wait,
        )
    if not install_agent:
        return

    agent_values = package_root / "agent-helm-values.yaml"
    if not agent_values.is_file():
        msg = f"Missing {agent_values}; install package is incomplete."
        raise click.ClickException(msg)

    agent_values_dict = _load_agent_values_yaml(agent_values)
    ensure_agent_storage_secret(agent_values_dict, agent_namespace=agent_namespace)
    endpoint_updated = sync_storage_secret_endpoint(
        agent_values_dict, agent_namespace=agent_namespace
    )
    merged_agent_values = write_merged_agent_values(package_root, agent_values_dict)
    helm_upgrade_install(
        release=agent_release,
        chart=_chart_path_from_package(package_root, AGENT_CHART),
        namespace=agent_namespace,
        values_file=merged_agent_values,
        wait=wait,
        wait_for_selector=agent_deployment_selector(agent_release),
    )
    if endpoint_updated:
        _restart_agent_deployment(agent_namespace, agent_release, wait=wait)
    if wait:
        run_agent_chart_validate(package_root)


def _chart_path_from_package(package_root: Path, chart_name: str) -> Path:
    chart = package_root / chart_name / "chart"
    if not chart.is_dir():
        msg = f"Helm chart not found at {chart}"
        raise click.ClickException(msg)
    return chart


def install_lab_cluster_charts(
    output_dir: Path,
    charts_dir: Path | None = None,
    wait: bool = True,
    agent_namespace: str = AGENT_NAMESPACE,
    agent_release: str = AGENT_RELEASE,
    install_agent: bool = True,
) -> None:
    """Install lab MinIO storage, then optionally the BYOC agent.

    :param output_dir: Directory containing ``storage-helm-values.yaml`` and
        ``agent-helm-values.yaml``.
    :type output_dir: Path
    :param charts_dir: Optional explicit ``resources/helm-setup`` root.
    :type charts_dir: Path | None
    :param wait: Wait for storage Helm resources and the agent Deployment.
    :type wait: bool
    :param agent_namespace: Kubernetes namespace for the agent chart.
    :type agent_namespace: str
    :param install_agent: Install the agent chart after storage.
    :type install_agent: bool
    :returns: None
    :rtype: None
    """
    charts_root = resolve_helm_charts_root(charts_dir)
    storage_values = output_dir / "storage-helm-values.yaml"
    if not storage_values.is_file():
        msg = (
            f"Missing {storage_values}; --install-storage requires storage Helm values."
        )
        raise click.ClickException(msg)

    helm_upgrade_install(
        release=STORAGE_RELEASE,
        chart=_chart_path(charts_root, STORAGE_CHART),
        namespace=STORAGE_NAMESPACE,
        values_file=storage_values,
        wait=wait,
    )
    if not install_agent:
        return

    agent_values = output_dir / "agent-helm-values.yaml"
    if not agent_values.is_file():
        msg = f"Missing {agent_values}; registration did not write agent values."
        raise click.ClickException(msg)
    agent_values_dict = _load_agent_values_yaml(agent_values)
    ensure_agent_storage_secret(agent_values_dict, agent_namespace=agent_namespace)
    endpoint_updated = sync_storage_secret_endpoint(
        agent_values_dict, agent_namespace=agent_namespace
    )
    agent_chart = _chart_path(charts_root, AGENT_CHART)
    merged_agent_values = output_dir / "agent-helm-values-merged.yaml"
    merged_agent_values.write_text(
        yaml.safe_dump(
            merge_agent_chart_defaults(agent_chart, agent_values_dict),
            sort_keys=False,
            default_flow_style=False,
        ),
        encoding="utf-8",
    )
    helm_upgrade_install(
        release=agent_release,
        chart=agent_chart,
        namespace=agent_namespace,
        values_file=merged_agent_values,
        wait=wait,
        wait_for_selector=agent_deployment_selector(agent_release),
    )
    if endpoint_updated:
        _restart_agent_deployment(agent_namespace, agent_release, wait=wait)
    if wait:
        run_agent_chart_validate(charts_root)


def install_enterprise_agent_chart(
    output_dir: Path,
    charts_dir: Path | None = None,
    wait: bool = True,
    agent_namespace: str = AGENT_NAMESPACE,
    agent_release: str = AGENT_RELEASE,
) -> None:
    """Install the BYOC agent chart for an enterprise-registered cluster.

    :param output_dir: Directory containing ``agent-helm-values.yaml``.
    :type output_dir: Path
    :param charts_dir: Optional explicit ``resources/helm-setup`` root.
    :type charts_dir: Path | None
    :param wait: Wait for the agent Deployment (not env-validator or reward-fn-val).
    :type wait: bool
    :param agent_namespace: Kubernetes namespace for the agent chart.
    :type agent_namespace: str
    :returns: None
    :rtype: None
    """
    charts_root = resolve_helm_charts_root(charts_dir)
    agent_values = output_dir / "agent-helm-values.yaml"
    if not agent_values.is_file():
        msg = f"Missing {agent_values}; registration did not write agent values."
        raise click.ClickException(msg)

    agent_values_dict = _load_agent_values_yaml(agent_values)
    ensure_agent_storage_secret(agent_values_dict, agent_namespace=agent_namespace)
    endpoint_updated = sync_storage_secret_endpoint(
        agent_values_dict, agent_namespace=agent_namespace
    )
    agent_chart = _chart_path(charts_root, AGENT_CHART)
    merged_agent_values = output_dir / "agent-helm-values-merged.yaml"
    merged_agent_values.write_text(
        yaml.safe_dump(
            merge_agent_chart_defaults(agent_chart, agent_values_dict),
            sort_keys=False,
            default_flow_style=False,
        ),
        encoding="utf-8",
    )
    helm_upgrade_install(
        release=agent_release,
        chart=agent_chart,
        namespace=agent_namespace,
        values_file=merged_agent_values,
        wait=wait,
        wait_for_selector=agent_deployment_selector(agent_release),
    )
    if endpoint_updated:
        _restart_agent_deployment(agent_namespace, agent_release, wait=wait)
    if wait:
        run_agent_chart_validate(charts_root)
