# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Tests for the read-only checks run before a cluster is connected to Arena."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import click
import pytest
from click.testing import CliRunner

from agilerl.byoc.preflight import (
    CheckStatus,
    PreflightCheck,
    any_failed,
    echo_checks,
    kubernetes_version_check,
    node_inventory_check,
    one_agent_check,
    own_agent_release,
    permission_checks,
    preflight_report,
    privileged_pod_check,
    require_passed,
    run_preflight,
    storage_class_check,
)

MODULE = "agilerl.byoc.preflight"
ENV = {"KUBECONFIG": "/pinned/kubeconfig"}


def gpu_node(name: str, gpus: str, product: str | None) -> dict[str, Any]:
    labels = {} if product is None else {"nvidia.com/gpu.product": product}
    return {
        "metadata": {"name": name, "labels": labels},
        "status": {"allocatable": {"cpu": "32", "nvidia.com/gpu": gpus}},
    }


def cpu_node(name: str) -> dict[str, Any]:
    return {"metadata": {"name": name}, "status": {"allocatable": {"cpu": "8"}}}


def storage_class(name: str, default: bool = False) -> dict[str, Any]:
    annotations = (
        {"storageclass.kubernetes.io/is-default-class": "true"} if default else {}
    )
    return {"metadata": {"name": name, "annotations": annotations}}


def agent_deployment(namespace: str, release: str | None) -> dict[str, Any]:
    labels = {"app.kubernetes.io/name": "arena-byoc-agent"}
    if release is not None:
        labels["app.kubernetes.io/instance"] = release
    return {
        "metadata": {
            "name": "arena-byoc-agent",
            "namespace": namespace,
            "labels": labels,
        }
    }


@dataclass
class FakeCluster:
    """Answers the kubectl calls preflight makes, as a healthy 1.31 GPU cluster."""

    version: dict[str, str] | None = field(
        default_factory=lambda: {"major": "1", "minor": "31"}
    )
    denied: set[str] = field(default_factory=set)
    can_i_error: str | None = None
    nodes: list[dict[str, Any]] | None = field(
        default_factory=lambda: [
            gpu_node("gpu-a", "8", "NVIDIA-H100-80GB-HBM3"),
            gpu_node("gpu-b", "8", "NVIDIA-H100-80GB-HBM3"),
            cpu_node("cpu-a"),
        ]
    )
    storage_classes: list[dict[str, Any]] | None = field(
        default_factory=lambda: [storage_class("standard", default=True)]
    )
    namespace_exists: bool = True
    namespace_error: str | None = None
    dry_run: tuple[int, str] = (0, "")
    deployments: list[dict[str, Any]] | None = field(default_factory=list)
    calls: list[list[str]] = field(default_factory=list)
    stdins: list[str | None] = field(default_factory=list)

    def __call__(
        self, argv: list[str], **kwargs: Any
    ) -> subprocess.CompletedProcess[str]:
        assert argv[0] == "kubectl"
        assert argv[-1] == "--request-timeout=30s"
        assert kwargs["env"] == ENV
        args = argv[1:-1]
        self.calls.append(args)
        self.stdins.append(kwargs.get("input"))
        if args[:2] == ["auth", "can-i"]:
            return self._can_i(args)
        if args[0] == "version":
            return self._json(
                None if self.version is None else {"serverVersion": self.version}
            )
        if args[:2] == ["get", "nodes"]:
            return self._items(self.nodes)
        if args[:2] == ["get", "storageclasses"]:
            return self._items(self.storage_classes)
        if args[:2] == ["get", "namespace"]:
            if self.namespace_error is not None:
                return subprocess.CompletedProcess(argv, 1, "", self.namespace_error)
            name = f"namespace/{args[2]}\n" if self.namespace_exists else ""
            return subprocess.CompletedProcess(argv, 0, name, "")
        if args[:2] == ["get", "deployments"]:
            return self._items(self.deployments)
        if args[0] == "create":
            returncode, stderr = self.dry_run
            return subprocess.CompletedProcess(argv, returncode, "", stderr)
        msg = f"unexpected kubectl call {args}"
        raise AssertionError(msg)

    def _can_i(self, args: list[str]) -> subprocess.CompletedProcess[str]:
        if self.can_i_error is not None:
            return subprocess.CompletedProcess(args, 1, "", self.can_i_error)
        key = f"{args[2]} {args[3].split('.')[0]}"
        if key in self.denied:
            return subprocess.CompletedProcess(args, 1, "no\n", "")
        return subprocess.CompletedProcess(args, 0, "yes\n", "")

    def _json(self, payload: object) -> subprocess.CompletedProcess[str]:
        if payload is None:
            return subprocess.CompletedProcess([], 1, "", "error: forbidden")
        return subprocess.CompletedProcess([], 0, json.dumps(payload), "")

    def _items(
        self, items: list[dict[str, Any]] | None
    ) -> subprocess.CompletedProcess[str]:
        return self._json(None if items is None else {"items": items})


@pytest.fixture
def cluster(monkeypatch: pytest.MonkeyPatch) -> FakeCluster:
    fake = FakeCluster()
    monkeypatch.setattr(f"{MODULE}.subprocess.run", fake)
    return fake


class TestKubernetesVersionCheck:
    def test_passes_from_1_30(self, cluster: FakeCluster) -> None:
        cluster.version = {"major": "1", "minor": "30+"}

        check = kubernetes_version_check(ENV)

        assert check == PreflightCheck(
            "kubernetes_version", CheckStatus.PASS, "Kubernetes 1.30"
        )

    def test_fails_below_1_30(self, cluster: FakeCluster) -> None:
        cluster.version = {"major": "1", "minor": "29"}

        check = kubernetes_version_check(ENV)

        assert check.status is CheckStatus.FAIL
        assert check.detail == "Kubernetes 1.29"
        assert check.fix is not None
        assert "ValidatingAdmissionPolicy" in check.fix

    def test_fails_when_the_version_cannot_be_read(self, cluster: FakeCluster) -> None:
        cluster.version = None

        check = kubernetes_version_check(ENV)

        assert check.status is CheckStatus.FAIL
        assert check.detail == "Could not read the server version: error: forbidden"


class TestPermissionChecks:
    def test_cluster_admin_passes_every_check(self, cluster: FakeCluster) -> None:
        checks = permission_checks(ENV)

        assert [check.name for check in checks] == [
            "can_create_namespaces",
            "can_create_clusterroles",
            "can_create_clusterrolebindings",
            "can_create_validatingadmissionpolicies",
            "can_create_validatingadmissionpolicybindings",
            "can_escalate_clusterroles",
            "can_bind_clusterroles",
        ]
        assert {check.status for check in checks} == {CheckStatus.PASS}
        assert [
            "auth",
            "can-i",
            "create",
            "validatingadmissionpolicies.admissionregistration.k8s.io",
        ] in cluster.calls

    def test_missing_create_right_fails(self, cluster: FakeCluster) -> None:
        cluster.denied = {"create clusterroles"}

        checks = {check.name: check for check in permission_checks(ENV)}

        failed = checks["can_create_clusterroles"]
        assert failed.status is CheckStatus.FAIL
        assert failed.detail == "You may not create clusterroles"
        assert failed.fix is not None
        assert "cluster-admin" in failed.fix
        assert checks["can_create_namespaces"].status is CheckStatus.PASS

    def test_missing_escalate_only_warns(self, cluster: FakeCluster) -> None:
        cluster.denied = {"escalate clusterroles", "bind clusterroles"}

        checks = {check.name: check for check in permission_checks(ENV)}

        assert checks["can_escalate_clusterroles"].status is CheckStatus.WARN
        assert checks["can_bind_clusterroles"].status is CheckStatus.WARN
        assert not any_failed(list(checks.values()))

    def test_unanswered_question_warns_with_kubectls_error(
        self, cluster: FakeCluster
    ) -> None:
        cluster.can_i_error = "error: You must be logged in to the server"

        checks = permission_checks(ENV)

        assert {check.status for check in checks} == {CheckStatus.WARN}
        assert checks[0].detail == (
            "Could not check create namespaces: error: You must be logged in to "
            "the server"
        )


class TestNodeInventoryCheck:
    def test_groups_gpu_nodes_by_product_and_count(self, cluster: FakeCluster) -> None:
        cluster.nodes = [
            gpu_node("gpu-a", "8", "NVIDIA-H100-80GB-HBM3"),
            gpu_node("gpu-b", "8", "NVIDIA-H100-80GB-HBM3"),
            gpu_node("gpu-c", "1", "NVIDIA-L4"),
            cpu_node("cpu-a"),
        ]

        check = node_inventory_check(ENV)

        assert check == PreflightCheck(
            "node_inventory",
            CheckStatus.PASS,
            "4 nodes; GPU nodes: 2 x 8 GPU NVIDIA-H100-80GB-HBM3, 1 x 1 GPU NVIDIA-L4",
        )

    def test_warns_without_gpu_nodes(self, cluster: FakeCluster) -> None:
        cluster.nodes = [cpu_node("cpu-a"), gpu_node("gpu-a", "0", "NVIDIA-L4")]

        check = node_inventory_check(ENV)

        assert check.status is CheckStatus.WARN
        assert check.detail == "2 nodes, none with allocatable nvidia.com/gpu"
        assert check.fix is not None
        assert "device plugin" in check.fix

    def test_warns_about_gpu_nodes_without_a_product_label(
        self, cluster: FakeCluster
    ) -> None:
        cluster.nodes = [
            gpu_node("gpu-a", "4", "NVIDIA-A100-SXM4-80GB"),
            gpu_node("gpu-b", "4", None),
        ]

        check = node_inventory_check(ENV)

        assert check.status is CheckStatus.WARN
        assert check.detail == (
            "2 nodes; GPU nodes: 1 x 4 GPU NVIDIA-A100-SXM4-80GB, 1 x 4 GPU "
            "unlabelled. No nvidia.com/gpu.product label on gpu-b"
        )

    def test_warns_when_nodes_cannot_be_listed(self, cluster: FakeCluster) -> None:
        cluster.nodes = None

        check = node_inventory_check(ENV)

        assert check.status is CheckStatus.WARN
        assert check.detail == "Could not list nodes: error: forbidden"


class TestStorageClassCheck:
    def test_one_default_passes(self, cluster: FakeCluster) -> None:
        cluster.storage_classes = [
            storage_class("fast"),
            storage_class("standard", default=True),
        ]

        check = storage_class_check(ENV, None)

        assert check == PreflightCheck(
            "storage_class", CheckStatus.PASS, "Default StorageClass standard"
        )

    def test_beta_default_annotation_counts(self, cluster: FakeCluster) -> None:
        cluster.storage_classes = [
            {
                "metadata": {
                    "name": "legacy",
                    "annotations": {
                        "storageclass.beta.kubernetes.io/is-default-class": "true"
                    },
                }
            }
        ]

        check = storage_class_check(ENV, None)

        assert check.detail == "Default StorageClass legacy"

    def test_no_default_warns(self, cluster: FakeCluster) -> None:
        cluster.storage_classes = [{"metadata": {"name": "fast"}}]

        check = storage_class_check(ENV, None)

        assert check.status is CheckStatus.WARN
        assert check.detail == "No default StorageClass"

    def test_several_defaults_warn(self, cluster: FakeCluster) -> None:
        cluster.storage_classes = [
            storage_class("a", default=True),
            storage_class("b", default=True),
        ]

        check = storage_class_check(ENV, None)

        assert check.status is CheckStatus.WARN
        assert check.detail == "Several default StorageClasses: a, b"

    def test_named_class_that_exists_passes_without_a_default(
        self, cluster: FakeCluster
    ) -> None:
        cluster.storage_classes = [storage_class("fast")]

        check = storage_class_check(ENV, "fast")

        assert check == PreflightCheck(
            "storage_class", CheckStatus.PASS, "Ray data uses StorageClass fast"
        )

    def test_named_class_that_is_missing_fails(self, cluster: FakeCluster) -> None:
        check = storage_class_check(ENV, "fast")

        assert check.status is CheckStatus.FAIL
        assert check.detail == "StorageClass fast, set for Ray data, does not exist"

    def test_warns_when_classes_cannot_be_listed(self, cluster: FakeCluster) -> None:
        cluster.storage_classes = None

        check = storage_class_check(ENV, None)

        assert check.status is CheckStatus.WARN
        assert check.detail == "Could not list StorageClasses: error: forbidden"


class TestPrivilegedPodCheck:
    def test_admitted_in_the_agent_namespace_passes(self, cluster: FakeCluster) -> None:
        check = privileged_pod_check(ENV, "arena")

        assert check == PreflightCheck(
            "privileged_pods",
            CheckStatus.PASS,
            "A privileged pod is admitted in namespace arena",
        )
        create = next(call for call in cluster.calls if call[0] == "create")
        assert "--dry-run=server" in create
        assert create[create.index("--namespace") + 1] == "arena"
        pod = json.loads(next(stdin for stdin in cluster.stdins if stdin))
        assert pod["spec"]["containers"][0]["securityContext"] == {"privileged": True}

    def test_missing_namespace_is_probed_in_default(self, cluster: FakeCluster) -> None:
        cluster.namespace_exists = False

        check = privileged_pod_check(ENV, "arena")

        assert check.status is CheckStatus.PASS
        assert check.detail == (
            "A privileged pod is admitted in namespace default "
            "(namespace arena does not exist yet)"
        )

    def test_pod_security_refusal_fails(self, cluster: FakeCluster) -> None:
        cluster.dry_run = (
            1,
            (
                'Error from server (Forbidden): pods "arena-preflight-privileged" is '
                'forbidden: violates PodSecurity "baseline:latest": privileged'
            ),
        )

        check = privileged_pod_check(ENV, "arena")

        assert check.status is CheckStatus.FAIL
        assert 'violates PodSecurity "baseline:latest"' in check.detail
        assert check.fix is not None
        assert "pod-security.kubernetes.io/enforce=privileged" in check.fix

    def test_webhook_without_dry_run_support_warns(self, cluster: FakeCluster) -> None:
        cluster.dry_run = (
            1,
            'admission webhook "policy.example.com" does not support dry run',
        )

        check = privileged_pod_check(ENV, "arena")

        assert check.status is CheckStatus.WARN
        assert check.detail.startswith("Could not check in namespace arena:")
        assert check.fix is None

    def test_namespace_lookup_failure_warns_without_a_dry_run(
        self, cluster: FakeCluster
    ) -> None:
        cluster.namespace_error = "error: timed out waiting for the condition"

        check = privileged_pod_check(ENV, "arena")

        assert check == PreflightCheck(
            "privileged_pods",
            CheckStatus.WARN,
            "Could not read namespace arena: error: timed out waiting for the condition",
        )
        assert all(call[0] != "create" for call in cluster.calls)

    def test_probe_name_already_exists_warns(self, cluster: FakeCluster) -> None:
        cluster.dry_run = (
            1,
            (
                'Error from server (AlreadyExists): pods "arena-preflight-privileged" '
                "already exists"
            ),
        )

        check = privileged_pod_check(ENV, "arena")

        assert check.status is CheckStatus.WARN
        assert "(AlreadyExists)" in check.detail
        assert check.fix is not None
        assert "arena-preflight-privileged" in check.fix
        assert "pod-security" not in check.fix

    def test_api_error_warns_without_a_policy_fix(self, cluster: FakeCluster) -> None:
        cluster.dry_run = (
            1,
            (
                "Error from server (Timeout): the server was unable to return a "
                "response in the time allotted"
            ),
        )

        check = privileged_pod_check(ENV, "arena")

        assert check.status is CheckStatus.WARN
        assert "Timeout" in check.detail
        assert check.fix is None

    def test_admission_webhook_denial_fails(self, cluster: FakeCluster) -> None:
        cluster.dry_run = (
            1,
            (
                "Error from server (Forbidden): admission webhook "
                '"policy.example.com" denied the request: privileged containers '
                "are not allowed"
            ),
        )

        check = privileged_pod_check(ENV, "arena")

        assert check.status is CheckStatus.FAIL
        assert check.fix is not None
        assert "pod-security.kubernetes.io/enforce=privileged" in check.fix

    def test_validating_admission_policy_denial_fails(
        self, cluster: FakeCluster
    ) -> None:
        cluster.dry_run = (
            1,
            (
                "Error from server (Forbidden): ValidatingAdmissionPolicy "
                "'no-priv' with binding 'no-priv' denied request: privileged is "
                "false"
            ),
        )

        check = privileged_pod_check(ENV, "arena")

        assert check.status is CheckStatus.FAIL
        assert check.fix is not None
        assert "pod-security.kubernetes.io/enforce=privileged" in check.fix

    def test_no_pod_create_right_warns_without_a_dry_run(
        self, cluster: FakeCluster
    ) -> None:
        cluster.denied = {"create pods"}

        check = privileged_pod_check(ENV, "arena")

        assert check == PreflightCheck(
            "privileged_pods",
            CheckStatus.WARN,
            "Could not check: you may not create pods in namespace arena",
        )
        assert all(call[0] != "create" for call in cluster.calls)


class TestOwnAgentRelease:
    def test_keeps_the_default_release_already_in_the_namespace(
        self, cluster: FakeCluster, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(f"{MODULE}.kubeconfig_env", lambda _path: ENV)
        cluster.deployments = [agent_deployment("arena", "arena-byoc-agent")]

        release = own_agent_release(Path("/pinned/kubeconfig"), "arena", "corp-k8s")

        assert release == "arena-byoc-agent"
        assert one_agent_check(ENV, "arena", release).status is CheckStatus.PASS

    def test_uses_the_cluster_name_when_that_is_the_release(
        self, cluster: FakeCluster, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(f"{MODULE}.kubeconfig_env", lambda _path: ENV)
        cluster.deployments = [agent_deployment("arena", "corp-k8s")]

        release = own_agent_release(Path("/pinned/kubeconfig"), "arena", "corp-k8s")

        assert release == "corp-k8s"

    def test_uses_the_cluster_name_when_nothing_is_installed(
        self, cluster: FakeCluster, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(f"{MODULE}.kubeconfig_env", lambda _path: ENV)

        release = own_agent_release(Path("/pinned/kubeconfig"), "arena", "corp-k8s")

        assert release == "corp-k8s"

    def test_ignores_the_default_release_in_another_namespace(
        self, cluster: FakeCluster, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(f"{MODULE}.kubeconfig_env", lambda _path: ENV)
        cluster.deployments = [agent_deployment("other", "arena-byoc-agent")]

        release = own_agent_release(Path("/pinned/kubeconfig"), "arena", "corp-k8s")

        assert release == "corp-k8s"

    def test_uses_the_cluster_name_when_deployments_cannot_be_listed(
        self, cluster: FakeCluster, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(f"{MODULE}.kubeconfig_env", lambda _path: ENV)
        cluster.deployments = None

        release = own_agent_release(Path("/pinned/kubeconfig"), "arena", "corp-k8s")

        assert release == "corp-k8s"

    def test_prefers_the_default_release_when_both_are_installed(
        self, cluster: FakeCluster, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(f"{MODULE}.kubeconfig_env", lambda _path: ENV)
        cluster.deployments = [
            agent_deployment("arena", "corp-k8s"),
            agent_deployment("arena", "arena-byoc-agent"),
        ]

        release = own_agent_release(Path("/pinned/kubeconfig"), "arena", "corp-k8s")

        assert release == "arena-byoc-agent"


class TestOneAgentCheck:
    def test_no_agent_passes(self, cluster: FakeCluster) -> None:
        check = one_agent_check(ENV, "arena", None)

        assert check == PreflightCheck(
            "one_agent", CheckStatus.PASS, "No other Arena agent"
        )

    def test_own_agent_is_not_another(self, cluster: FakeCluster) -> None:
        cluster.deployments = [agent_deployment("arena", "corp-k8s")]

        check = one_agent_check(ENV, "arena", "corp-k8s")

        assert check.status is CheckStatus.PASS

    def test_other_agents_fail_and_are_named_once(self, cluster: FakeCluster) -> None:
        cluster.deployments = [
            agent_deployment("arena", "corp-k8s"),
            agent_deployment("tenant-b", "other"),
            agent_deployment("tenant-b", "other"),
            agent_deployment("legacy", None),
        ]

        check = one_agent_check(ENV, "arena", "corp-k8s")

        assert check.status is CheckStatus.FAIL
        assert check.detail == (
            "Arena agents already in this cluster: legacy/arena-byoc-agent, "
            "tenant-b/other"
        )
        assert check.fix is not None
        assert "one agent per Kubernetes cluster" in check.fix

    def test_without_an_own_release_every_agent_is_another(
        self, cluster: FakeCluster
    ) -> None:
        cluster.deployments = [agent_deployment("arena", "arena-byoc-agent")]

        check = one_agent_check(ENV, "arena", None)

        assert check.status is CheckStatus.FAIL
        assert check.detail == (
            "Arena agents already in this cluster: arena/arena-byoc-agent"
        )

    def test_warns_when_deployments_cannot_be_listed(
        self, cluster: FakeCluster
    ) -> None:
        cluster.deployments = None

        check = one_agent_check(ENV, "arena", None)

        assert check.status is CheckStatus.WARN
        assert check.detail == "Could not list agent Deployments: error: forbidden"


class TestRunPreflight:
    def test_runs_every_check_against_the_kubeconfig(
        self, cluster: FakeCluster, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(f"{MODULE}.kubeconfig_env", lambda _path: ENV)

        checks = run_preflight(Path("/pinned/kubeconfig"), "arena", None, None)

        assert [check.name for check in checks] == [
            "kubernetes_version",
            "can_create_namespaces",
            "can_create_clusterroles",
            "can_create_clusterrolebindings",
            "can_create_validatingadmissionpolicies",
            "can_create_validatingadmissionpolicybindings",
            "can_escalate_clusterroles",
            "can_bind_clusterroles",
            "node_inventory",
            "storage_class",
            "privileged_pods",
            "one_agent",
        ]
        assert not any_failed(checks)


FAILED = PreflightCheck("one_agent", CheckStatus.FAIL, "Another agent", "Remove it")
PASSED = PreflightCheck("kubernetes_version", CheckStatus.PASS, "Kubernetes 1.31")


class TestPreflightReport:
    def test_reports_the_cluster_and_every_check(self) -> None:
        report = preflight_report(
            "corp", "https://k8s.example:6443", "uid-1", [PASSED, FAILED]
        )

        assert report == {
            "context": "corp",
            "server": "https://k8s.example:6443",
            "cluster_uid": "uid-1",
            "passed": False,
            "checks": [
                {
                    "name": "kubernetes_version",
                    "status": "pass",
                    "detail": "Kubernetes 1.31",
                    "fix": None,
                },
                {
                    "name": "one_agent",
                    "status": "fail",
                    "detail": "Another agent",
                    "fix": "Remove it",
                },
            ],
        }


class TestRequirePassed:
    def test_passes_when_nothing_failed(self) -> None:
        require_passed([PASSED])

    def test_names_the_failed_checks(self) -> None:
        with pytest.raises(
            click.ClickException, match=r"Preflight failed: one_agent\."
        ):
            require_passed([PASSED, FAILED])


class TestEchoChecks:
    def test_prints_one_row_per_check(self) -> None:
        @click.command()
        def show() -> None:
            echo_checks([PASSED, FAILED])

        result = CliRunner().invoke(show, [])

        assert result.exit_code == 0
        assert "kubernetes_version" in result.output
        assert "PASS" in result.output
        assert "FAIL" in result.output
        assert "Remove it" in result.output
