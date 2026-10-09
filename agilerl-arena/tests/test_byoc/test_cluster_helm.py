# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Tests for BYOC cluster Helm install helpers."""

from __future__ import annotations

import base64
import json
from collections.abc import Callable
from pathlib import Path
from unittest.mock import MagicMock, patch

import click
import pytest
import yaml

from agilerl.arena.byoc.cluster_helm import (
    AGENT_CHART,
    AGENT_DEPLOYMENT_SELECTOR,
    AGENT_NAMESPACE,
    AGENT_RELEASE,
    STORAGE_CHART,
    STORAGE_NAMESPACE,
    STORAGE_RELEASE,
    _chart_path,
    _chart_path_from_package,
    _copy_k8s_secret,
    _kubectl_secret_exists,
    _load_agent_values_yaml,
    discover_agent_namespace,
    ensure_agent_storage_secret,
    helm_available,
    helm_upgrade_install,
    install_enterprise_agent_chart,
    install_from_install_package_root,
    install_lab_cluster_charts,
    list_installed_byoc_releases,
    merge_agent_chart_defaults,
    resolve_helm_charts_root,
    resolve_install_release,
    run_agent_chart_validate,
    sync_storage_secret_endpoint,
    uninstall_byoc_releases,
    wait_for_deployments,
    write_merged_agent_values,
)
from agilerl.arena.byoc.provisioning.storage import storage_secret_endpoint_data


def _write_agent_chart_defaults(chart_dir: Path) -> None:
    chart_dir.mkdir(parents=True, exist_ok=True)
    (chart_dir / "values.yaml").write_text(
        """storage:
  s3fsOpts: "-o use_path_request_style"
validatorPersistence:
  storageClassName: ''
  size: 50Gi
envValidator:
  enabled: true
  image:
    repository: env-val
    tag: latest
rewardFunctionValidator:
  enabled: true
  image:
    repository: reward-fn-val
    tag: latest""",
        encoding="utf-8",
    )


def test_merge_agent_chart_defaults_loads_chart_values(tmp_path: Path) -> None:
    chart_dir = tmp_path / "chart"
    _write_agent_chart_defaults(chart_dir)

    merged = merge_agent_chart_defaults(
        chart_dir, {"storage": {"endpoint": "http://minio"}}
    )

    assert merged["envValidator"]["enabled"] is True
    assert merged["rewardFunctionValidator"]["enabled"] is True
    assert merged["storage"]["s3fsOpts"] == "-o use_path_request_style"
    assert merged["storage"]["endpoint"] == "http://minio"


def test_run_agent_chart_validate_invokes_validate_sh_with_env(tmp_path: Path) -> None:
    package_root = tmp_path / "package"
    agent_root = package_root / AGENT_CHART
    agent_root.mkdir(parents=True)
    chart_dir = agent_root / "chart"
    _write_agent_chart_defaults(chart_dir)
    validate = agent_root / "validate.sh"
    validate.write_text("#!/bin/sh\n", encoding="utf-8")
    validate.chmod(0o755)

    with patch("agilerl.arena.byoc.cluster_helm.subprocess.run") as run_mock:
        run_mock.return_value = MagicMock(returncode=0)
        run_agent_chart_validate(package_root)

    run_mock.assert_called_once()
    call = run_mock.call_args
    assert call.args[0] == [str(validate)]
    assert call.kwargs["cwd"] == agent_root
    assert call.kwargs["env"]["CHART_DIR"] == str(chart_dir)
    assert "VALUES_FILE" not in call.kwargs["env"]


def test_run_agent_chart_validate_raises_on_nonzero_exit(tmp_path: Path) -> None:
    package_root = tmp_path / "package"
    agent_root = package_root / AGENT_CHART
    agent_root.mkdir(parents=True)
    _write_agent_chart_defaults(agent_root / "chart")
    validate = agent_root / "validate.sh"
    validate.write_text("#!/bin/sh\nexit 1\n", encoding="utf-8")
    validate.chmod(0o755)

    with patch("agilerl.arena.byoc.cluster_helm.subprocess.run") as run_mock:
        run_mock.return_value = MagicMock(returncode=1)
        with pytest.raises(click.ClickException, match="validation failed"):
            run_agent_chart_validate(package_root)


def test_install_from_install_package_root_merges_values_before_helm(
    tmp_path: Path,
) -> None:
    package_root = tmp_path / "arena-byoc-install"
    package_root.mkdir()
    for name in (STORAGE_CHART, AGENT_CHART):
        _write_agent_chart_defaults(package_root / name / "chart")
    (package_root / "storage-helm-values.yaml").write_text(
        "bucket:\n  name: arena-data\n"
    )
    (package_root / "agent-helm-values.yaml").write_text(
        "storage:\n  endpoint: http://minio\n"
    )

    calls: list[dict[str, object]] = []

    def _record(**kwargs: object) -> None:
        calls.append(kwargs)

    with (
        patch(
            "agilerl.arena.byoc.cluster_helm.helm_upgrade_install",
            side_effect=_record,
        ),
        patch(
            "agilerl.arena.byoc.cluster_helm.run_agent_chart_validate",
        ) as validate_mock,
    ):
        install_from_install_package_root(package_root)

    merged = package_root / "agent-helm-values-merged.yaml"
    assert merged.is_file()
    merged_data = yaml.safe_load(merged.read_text(encoding="utf-8"))
    assert merged_data["envValidator"]["enabled"] is True
    assert merged_data["rewardFunctionValidator"]["enabled"] is True
    assert merged_data["storage"]["s3fsOpts"] == "-o use_path_request_style"
    assert merged_data["storage"]["endpoint"] == "http://minio"
    assert calls[0]["release"] == "arena-byoc-storage"
    assert calls[0].get("wait_for_selector") is None
    assert calls[1]["release"] == "arena-byoc-agent"
    assert calls[1]["values_file"] == merged
    assert calls[1]["wait_for_selector"] == AGENT_DEPLOYMENT_SELECTOR
    validate_mock.assert_called_once_with(package_root)


def test_resolve_helm_charts_root_explicit(tmp_path: Path) -> None:
    charts = tmp_path / "helm-setup"
    charts.mkdir()
    assert resolve_helm_charts_root(charts) == charts.resolve()


def test_resolve_helm_charts_root_missing_raises(tmp_path: Path) -> None:
    with pytest.raises(click.ClickException, match="does not exist"):
        resolve_helm_charts_root(tmp_path / "missing")


def test_resolve_helm_charts_root_requires_env_or_flag() -> None:
    with patch.dict("os.environ", {}, clear=True):
        with pytest.raises(click.ClickException, match="ARENA_HELM_CHARTS_DIR"):
            resolve_helm_charts_root(None)


def test_helm_upgrade_install_invokes_helm(tmp_path: Path) -> None:
    chart = tmp_path / "chart"
    chart.mkdir()
    values = tmp_path / "values.yaml"
    values.write_text("bucket:\n  name: arena-data\n", encoding="utf-8")

    with (
        patch(
            "agilerl.arena.byoc.cluster_helm.shutil.which",
            return_value="/usr/bin/helm",
        ),
        patch("agilerl.arena.byoc.cluster_helm.subprocess.run") as run_mock,
    ):
        run_mock.return_value = MagicMock(returncode=0)
        helm_upgrade_install(
            release="arena-byoc-storage",
            chart=chart,
            namespace="storage",
            values_file=values,
        )

    cmd = run_mock.call_args.args[0]
    assert cmd[:4] == ["helm", "upgrade", "--install", "arena-byoc-storage"]
    assert "--wait" in cmd
    assert "rewardFunctionValidator.enabled=false" not in cmd


def test_helm_upgrade_install_waits_only_on_named_deployment(tmp_path: Path) -> None:
    chart = tmp_path / "chart"
    chart.mkdir()
    values = tmp_path / "values.yaml"
    values.write_text("cluster:\n  id: 1\n", encoding="utf-8")

    with (
        patch(
            "agilerl.arena.byoc.cluster_helm.shutil.which",
            return_value="/usr/bin/helm",
        ),
        patch("agilerl.arena.byoc.cluster_helm.subprocess.run") as run_mock,
    ):
        run_mock.return_value = MagicMock(returncode=0)
        helm_upgrade_install(
            release="arena-byoc-agent",
            chart=chart,
            namespace="arena",
            values_file=values,
            wait_for_selector="app.kubernetes.io/name=arena-byoc-agent",
        )

    helm_cmd = run_mock.call_args_list[0].args[0]
    kubectl_cmd = run_mock.call_args_list[1].args[0]
    assert helm_cmd[:4] == ["helm", "upgrade", "--install", "arena-byoc-agent"]
    assert "--wait" not in helm_cmd
    assert kubectl_cmd == [
        "kubectl",
        "wait",
        "--for=condition=available",
        "deployment",
        "--selector",
        "app.kubernetes.io/name=arena-byoc-agent",
        "--namespace",
        "arena",
        "--timeout=10m",
    ]


def test_helm_upgrade_install_skips_deployment_wait_when_wait_false(
    tmp_path: Path,
) -> None:
    chart = tmp_path / "chart"
    chart.mkdir()
    values = tmp_path / "values.yaml"
    values.write_text("cluster:\n  id: 1\n", encoding="utf-8")

    with (
        patch(
            "agilerl.arena.byoc.cluster_helm.shutil.which",
            return_value="/usr/bin/helm",
        ),
        patch("agilerl.arena.byoc.cluster_helm.subprocess.run") as run_mock,
    ):
        run_mock.return_value = MagicMock(returncode=0)
        helm_upgrade_install(
            release="arena-byoc-agent",
            chart=chart,
            namespace="arena",
            values_file=values,
            wait=False,
            wait_for_selector="app.kubernetes.io/name=arena-byoc-agent",
        )

    assert run_mock.call_count == 1
    assert "--wait" not in run_mock.call_args.args[0]


def test_wait_for_deployments_raises_on_timeout() -> None:
    with (
        patch(
            "agilerl.arena.byoc.cluster_helm.shutil.which",
            return_value="/usr/bin/kubectl",
        ),
        patch("agilerl.arena.byoc.cluster_helm.subprocess.run") as run_mock,
    ):
        run_mock.return_value = MagicMock(returncode=1)
        with pytest.raises(
            click.ClickException, match="Timed out waiting for deployments matching"
        ):
            wait_for_deployments(
                "app.kubernetes.io/name=arena-byoc-agent",
                "arena",
            )


def test_helm_upgrade_install_applies_extra_sets(tmp_path: Path) -> None:
    chart = tmp_path / "chart"
    chart.mkdir()
    values = tmp_path / "values.yaml"
    values.write_text("rewardFunctionValidator:\n  enabled: true\n", encoding="utf-8")

    with (
        patch(
            "agilerl.arena.byoc.cluster_helm.shutil.which",
            return_value="/usr/bin/helm",
        ),
        patch("agilerl.arena.byoc.cluster_helm.subprocess.run") as run_mock,
    ):
        run_mock.return_value = MagicMock(returncode=0)
        helm_upgrade_install(
            release="arena-byoc-agent",
            chart=chart,
            namespace="arena",
            values_file=values,
            extra_sets=("rewardFunctionValidator.enabled=false",),
        )

    cmd = run_mock.call_args.args[0]
    assert "--set" in cmd
    assert "rewardFunctionValidator.enabled=false" in cmd


def _write_minimal_chart(
    chart_dir: Path, values_text: str = "bucket:\n  name: arena-data\n"
) -> None:
    chart_dir.mkdir(parents=True, exist_ok=True)
    (chart_dir / "values.yaml").write_text(values_text, encoding="utf-8")


def test_install_lab_cluster_charts_orders_storage_then_agent(tmp_path: Path) -> None:
    charts_root = tmp_path / "helm-setup"
    _write_minimal_chart(charts_root / STORAGE_CHART / "chart")
    _write_agent_chart_defaults(charts_root / AGENT_CHART / "chart")
    out = tmp_path / "out"
    out.mkdir()
    (out / "storage-helm-values.yaml").write_text("bucket:\n  name: arena-data\n")
    (out / "agent-helm-values.yaml").write_text("storage:\n  endpoint: http://minio\n")

    calls: list[str] = []

    def _record(**kwargs: object) -> None:
        calls.append(str(kwargs["release"]))

    with (
        patch(
            "agilerl.arena.byoc.cluster_helm.helm_upgrade_install",
            side_effect=_record,
        ),
        patch(
            "agilerl.arena.byoc.cluster_helm.run_agent_chart_validate",
        ),
    ):
        install_lab_cluster_charts(out, charts_dir=charts_root)

    assert calls == ["arena-byoc-storage", "arena-byoc-agent"]


def test_install_lab_cluster_charts_skips_agent_when_disabled(tmp_path: Path) -> None:
    charts_root = tmp_path / "helm-setup"
    _write_minimal_chart(charts_root / STORAGE_CHART / "chart")
    out = tmp_path / "out"
    out.mkdir()
    (out / "storage-helm-values.yaml").write_text("bucket:\n  name: arena-data\n")

    calls: list[str] = []

    def _record(**kwargs: object) -> None:
        calls.append(str(kwargs["release"]))

    with (
        patch(
            "agilerl.arena.byoc.cluster_helm.helm_upgrade_install",
            side_effect=_record,
        ),
        patch(
            "agilerl.arena.byoc.cluster_helm.run_agent_chart_validate",
        ) as validate_mock,
        patch(
            "agilerl.arena.byoc.cluster_helm.ensure_agent_storage_secret",
        ) as secret_mock,
    ):
        install_lab_cluster_charts(out, charts_dir=charts_root, install_agent=False)

    assert calls == ["arena-byoc-storage"]
    validate_mock.assert_not_called()
    secret_mock.assert_not_called()


def test_install_lab_cluster_charts_requires_storage_values(tmp_path: Path) -> None:
    charts_root = tmp_path / "helm-setup"
    charts_root.mkdir()
    out = tmp_path / "out"
    out.mkdir()

    with pytest.raises(click.ClickException, match="requires storage Helm values"):
        install_lab_cluster_charts(out, charts_dir=charts_root, install_agent=False)


def test_install_lab_cluster_charts_requires_agent_values(tmp_path: Path) -> None:
    charts_root = tmp_path / "helm-setup"
    _write_minimal_chart(charts_root / STORAGE_CHART / "chart")
    out = tmp_path / "out"
    out.mkdir()
    (out / "storage-helm-values.yaml").write_text("bucket:\n  name: arena-data\n")

    with (
        patch("agilerl.arena.byoc.cluster_helm.helm_upgrade_install"),
        pytest.raises(click.ClickException, match="did not write agent values"),
    ):
        install_lab_cluster_charts(out, charts_dir=charts_root)


def test_install_lab_cluster_charts_uses_agent_namespace(tmp_path: Path) -> None:
    charts_root = tmp_path / "helm-setup"
    _write_minimal_chart(charts_root / STORAGE_CHART / "chart")
    _write_agent_chart_defaults(charts_root / AGENT_CHART / "chart")
    out = tmp_path / "out"
    out.mkdir()
    (out / "storage-helm-values.yaml").write_text("bucket:\n  name: arena-data\n")
    (out / "agent-helm-values.yaml").write_text("storage:\n  endpoint: http://minio\n")

    namespaces: list[str] = []

    def _record(**kwargs: object) -> None:
        namespaces.append(str(kwargs["namespace"]))

    with (
        patch(
            "agilerl.arena.byoc.cluster_helm.helm_upgrade_install",
            side_effect=_record,
        ),
        patch(
            "agilerl.arena.byoc.cluster_helm.run_agent_chart_validate",
        ),
    ):
        install_lab_cluster_charts(
            out, charts_dir=charts_root, agent_namespace="arena-agent"
        )

    assert namespaces == [STORAGE_NAMESPACE, "arena-agent"]


def test_install_from_install_package_root_orders_storage_then_agent(
    tmp_path: Path,
) -> None:
    package_root = tmp_path / "arena-byoc-install"
    package_root.mkdir()
    _write_minimal_chart(package_root / STORAGE_CHART / "chart")
    _write_agent_chart_defaults(package_root / AGENT_CHART / "chart")
    (package_root / "storage-helm-values.yaml").write_text(
        "bucket:\n  name: arena-data\n"
    )
    (package_root / "agent-helm-values.yaml").write_text(
        "storage:\n  endpoint: http://minio\n"
    )

    calls: list[str] = []

    def _record(**kwargs: object) -> None:
        calls.append(str(kwargs["release"]))

    with (
        patch(
            "agilerl.arena.byoc.cluster_helm.helm_upgrade_install",
            side_effect=_record,
        ),
        patch(
            "agilerl.arena.byoc.cluster_helm.run_agent_chart_validate",
        ),
    ):
        install_from_install_package_root(package_root)

    assert calls == ["arena-byoc-storage", "arena-byoc-agent"]


def test_install_from_install_package_root_skips_agent_when_disabled(
    tmp_path: Path,
) -> None:
    package_root = tmp_path / "arena-byoc-install"
    package_root.mkdir()
    _write_minimal_chart(package_root / STORAGE_CHART / "chart")
    (package_root / "storage-helm-values.yaml").write_text(
        "bucket:\n  name: arena-data\n"
    )

    calls: list[str] = []

    def _record(**kwargs: object) -> None:
        calls.append(str(kwargs["release"]))

    with (
        patch(
            "agilerl.arena.byoc.cluster_helm.helm_upgrade_install",
            side_effect=_record,
        ),
        patch(
            "agilerl.arena.byoc.cluster_helm.run_agent_chart_validate",
        ) as validate_mock,
        patch(
            "agilerl.arena.byoc.cluster_helm.ensure_agent_storage_secret",
        ) as secret_mock,
    ):
        install_from_install_package_root(package_root, install_agent=False)

    assert calls == ["arena-byoc-storage"]
    validate_mock.assert_not_called()
    secret_mock.assert_not_called()


def test_install_from_install_package_root_requires_agent_values(
    tmp_path: Path,
) -> None:
    package_root = tmp_path / "arena-byoc-install"
    package_root.mkdir()
    _write_minimal_chart(package_root / STORAGE_CHART / "chart")
    (package_root / "storage-helm-values.yaml").write_text(
        "bucket:\n  name: arena-data\n"
    )

    with (
        patch("agilerl.arena.byoc.cluster_helm.helm_upgrade_install"),
        pytest.raises(click.ClickException, match="install package is incomplete"),
    ):
        install_from_install_package_root(package_root)


def test_install_from_install_package_root_uses_agent_namespace(tmp_path: Path) -> None:
    package_root = tmp_path / "arena-byoc-install"
    package_root.mkdir()
    _write_agent_chart_defaults(package_root / AGENT_CHART / "chart")
    (package_root / "agent-helm-values.yaml").write_text("cluster:\n  id: 1\n")

    namespaces: list[str] = []

    def _record(**kwargs: object) -> None:
        namespaces.append(str(kwargs["namespace"]))

    with (
        patch(
            "agilerl.arena.byoc.cluster_helm.helm_upgrade_install",
            side_effect=_record,
        ),
        patch(
            "agilerl.arena.byoc.cluster_helm.run_agent_chart_validate",
        ),
    ):
        install_from_install_package_root(package_root, agent_namespace="arena-agent")

    assert namespaces == ["arena-agent"]


def test_install_from_install_package_root_skips_storage_when_values_missing(
    tmp_path: Path,
) -> None:
    package_root = tmp_path / "arena-byoc-install"
    package_root.mkdir()
    _write_agent_chart_defaults(package_root / AGENT_CHART / "chart")
    (package_root / "agent-helm-values.yaml").write_text("cluster:\n  id: 1\n")

    calls: list[str] = []

    def _record(**kwargs: object) -> None:
        calls.append(str(kwargs["release"]))

    with (
        patch(
            "agilerl.arena.byoc.cluster_helm.helm_upgrade_install",
            side_effect=_record,
        ),
        patch(
            "agilerl.arena.byoc.cluster_helm.run_agent_chart_validate",
        ),
    ):
        install_from_install_package_root(package_root)

    assert calls == ["arena-byoc-agent"]


def test_ensure_agent_storage_secret_skips_when_create_secret_true() -> None:
    with patch("agilerl.arena.byoc.cluster_helm._kubectl_secret_exists") as exists_mock:
        ensure_agent_storage_secret(
            {"storage": {"secretName": "arena-storage", "createSecret": True}}
        )
    exists_mock.assert_not_called()


def test_ensure_agent_storage_secret_copies_from_storage_namespace() -> None:
    agent_values = {"storage": {"secretName": "arena-storage", "createSecret": False}}

    def _exists(name: str, namespace: str) -> bool:
        return namespace == "storage" and name == "arena-storage"

    with (
        patch(
            "agilerl.arena.byoc.cluster_helm._kubectl_secret_exists",
            side_effect=_exists,
        ),
        patch(
            "agilerl.arena.byoc.cluster_helm._copy_k8s_secret",
        ) as copy_mock,
    ):
        ensure_agent_storage_secret(agent_values)

    copy_mock.assert_called_once_with("arena-storage", "storage", "arena")


class TestResolveInstallRelease:
    """Choosing the Helm release name for an agent install."""

    def test_uses_cluster_name_in_an_empty_namespace(self) -> None:
        listed = MagicMock(returncode=0, stdout="[]", stderr="")

        with (
            patch("agilerl.arena.byoc.cluster_helm.helm_available", return_value=True),
            patch(
                "agilerl.arena.byoc.cluster_helm.subprocess.run", return_value=listed
            ),
        ):
            release = resolve_install_release("op-arena-fdh-2", "arena-fdh-2")

        assert release == "op-arena-fdh-2"

    def test_keeps_an_existing_default_release(self) -> None:
        listed = MagicMock(
            returncode=0,
            stdout=json.dumps([{"name": AGENT_RELEASE, "namespace": "arena"}]),
            stderr="",
        )

        with (
            patch("agilerl.arena.byoc.cluster_helm.helm_available", return_value=True),
            patch(
                "agilerl.arena.byoc.cluster_helm.subprocess.run", return_value=listed
            ),
        ):
            release = resolve_install_release("op-arena-fdh-2", "arena")

        assert release == AGENT_RELEASE

    def test_uses_cluster_name_without_helm(self) -> None:
        with patch(
            "agilerl.arena.byoc.cluster_helm.helm_available", return_value=False
        ):
            release = resolve_install_release("op-arena-fdh-2", "arena-fdh-2")

        assert release == "op-arena-fdh-2"

    def test_raises_when_helm_list_fails(self) -> None:
        listed = MagicMock(returncode=1, stdout="", stderr="no context")

        with (
            patch("agilerl.arena.byoc.cluster_helm.helm_available", return_value=True),
            patch(
                "agilerl.arena.byoc.cluster_helm.subprocess.run", return_value=listed
            ),
            pytest.raises(click.ClickException, match="helm list --namespace"),
        ):
            resolve_install_release("op-arena-fdh-2", "arena-fdh-2")


class TestEnsureAgentStorageSecretNamespace:
    """Copying the storage secret into an agent namespace Helm has not created."""

    @staticmethod
    def _kubectl(
        namespace_exists: bool,
    ) -> tuple[
        list[list[str]],
        list[dict[str, object]],
        Callable[..., MagicMock],
    ]:
        argvs: list[list[str]] = []
        kwargs_seen: list[dict[str, object]] = []

        def run(argv: list[str], **kwargs: object) -> MagicMock:
            argvs.append(argv)
            kwargs_seen.append(kwargs)
            if argv[:3] == ["kubectl", "get", "secret"]:
                namespace = argv[argv.index("-n") + 1]
                if namespace != STORAGE_NAMESPACE:
                    return MagicMock(returncode=1, stdout="", stderr="not found")
                secret = {
                    "kind": "Secret",
                    "metadata": {"name": "arena-storage", "namespace": namespace},
                    "data": {"accesskey": "a2V5"},
                }
                return MagicMock(returncode=0, stdout=json.dumps(secret), stderr="")
            if argv[:3] == ["kubectl", "get", "namespace"]:
                code = 0 if namespace_exists else 1
                return MagicMock(returncode=code, stdout="", stderr="")
            return MagicMock(returncode=0, stdout="", stderr="")

        return argvs, kwargs_seen, run

    def test_creates_missing_namespace_before_applying(self) -> None:
        # Arrange
        argvs, kwargs_seen, run = self._kubectl(namespace_exists=False)

        # Act
        with (
            patch("agilerl.arena.byoc.cluster_helm.shutil.which", return_value="k"),
            patch("agilerl.arena.byoc.cluster_helm.subprocess.run", side_effect=run),
        ):
            ensure_agent_storage_secret(
                {"storage": {"secretName": "arena-storage", "createSecret": False}},
                agent_namespace="arena-fdh-2",
            )

        # Assert
        create = ["kubectl", "create", "namespace", "arena-fdh-2"]
        apply = ["kubectl", "apply", "-f", "-"]
        assert create in argvs
        assert argvs.index(create) < argvs.index(apply)
        applied = json.loads(str(kwargs_seen[argvs.index(apply)]["input"]))
        assert applied["metadata"]["namespace"] == "arena-fdh-2"

    def test_keeps_existing_namespace(self) -> None:
        argvs, _kwargs, run = self._kubectl(namespace_exists=True)

        with (
            patch("agilerl.arena.byoc.cluster_helm.shutil.which", return_value="k"),
            patch("agilerl.arena.byoc.cluster_helm.subprocess.run", side_effect=run),
        ):
            ensure_agent_storage_secret(
                {"storage": {"secretName": "arena-storage", "createSecret": False}},
                agent_namespace="arena-fdh-2",
            )

        assert ["kubectl", "create", "namespace", "arena-fdh-2"] not in argvs
        assert ["kubectl", "apply", "-f", "-"] in argvs

    def test_raises_when_namespace_creation_fails(self) -> None:
        def run(argv: list[str], **_: object) -> MagicMock:
            if argv[:3] == ["kubectl", "get", "secret"]:
                namespace = argv[argv.index("-n") + 1]
                if namespace != STORAGE_NAMESPACE:
                    return MagicMock(returncode=1, stdout="", stderr="not found")
                secret = {"kind": "Secret", "metadata": {"name": "arena-storage"}}
                return MagicMock(returncode=0, stdout=json.dumps(secret), stderr="")
            if argv[:3] == ["kubectl", "create", "namespace"]:
                return MagicMock(returncode=1, stdout="", stderr="forbidden")
            return MagicMock(returncode=1, stdout="", stderr="")

        with (
            patch("agilerl.arena.byoc.cluster_helm.shutil.which", return_value="k"),
            patch("agilerl.arena.byoc.cluster_helm.subprocess.run", side_effect=run),
            pytest.raises(click.ClickException, match="Failed to create namespace"),
        ):
            ensure_agent_storage_secret(
                {"storage": {"secretName": "arena-storage", "createSecret": False}},
                agent_namespace="arena-fdh-2",
            )


def test_install_lab_cluster_charts_syncs_storage_secret_before_agent(
    tmp_path: Path,
) -> None:
    charts_root = tmp_path / "helm-setup"
    _write_minimal_chart(charts_root / STORAGE_CHART / "chart")
    _write_agent_chart_defaults(charts_root / AGENT_CHART / "chart")
    out = tmp_path / "out"
    out.mkdir()
    (out / "storage-helm-values.yaml").write_text("bucket:\n  name: arena-data\n")
    (out / "agent-helm-values.yaml").write_text(
        "storage:\n  secretName: arena-storage\n  createSecret: false\n"
    )

    with (
        patch(
            "agilerl.arena.byoc.cluster_helm.helm_upgrade_install",
        ),
        patch(
            "agilerl.arena.byoc.cluster_helm.ensure_agent_storage_secret",
        ) as sync_mock,
        patch(
            "agilerl.arena.byoc.cluster_helm.run_agent_chart_validate",
        ),
    ):
        install_lab_cluster_charts(out, charts_dir=charts_root)

    sync_mock.assert_called_once_with(
        {"storage": {"secretName": "arena-storage", "createSecret": False}},
        agent_namespace=AGENT_NAMESPACE,
    )


def test_list_installed_byoc_releases_empty_without_helm(tmp_path: Path) -> None:
    kubeconfig = tmp_path / "kubeconfig"
    kubeconfig.write_text("apiVersion: v1\n", encoding="utf-8")

    with patch("agilerl.arena.byoc.cluster_helm.helm_available", return_value=False):
        assert list_installed_byoc_releases(kubeconfig) == []


def test_list_installed_byoc_releases_empty_without_kubeconfig() -> None:
    with patch("agilerl.arena.byoc.cluster_helm.helm_available", return_value=True):
        assert list_installed_byoc_releases(None) == []


def _helm_list_result(*releases: tuple[str, str]) -> MagicMock:
    entries = [{"name": name, "namespace": namespace} for name, namespace in releases]
    return MagicMock(returncode=0, stdout=json.dumps(entries), stderr="")


def test_list_installed_byoc_releases_returns_present_releases(
    tmp_path: Path,
) -> None:
    kubeconfig = tmp_path / "kubeconfig"
    kubeconfig.write_text("apiVersion: v1\n", encoding="utf-8")

    with (
        patch("agilerl.arena.byoc.cluster_helm.helm_available", return_value=True),
        patch(
            "agilerl.arena.byoc.cluster_helm.subprocess.run",
            return_value=_helm_list_result(
                ("cert-manager", "cert-manager"),
                (AGENT_RELEASE, AGENT_NAMESPACE),
            ),
        ) as run_mock,
    ):
        installed = list_installed_byoc_releases(kubeconfig)

    assert installed == [(AGENT_RELEASE, AGENT_NAMESPACE)]
    assert run_mock.call_args.args[0] == [
        "helm",
        "list",
        "--all-namespaces",
        "--output",
        "json",
    ]
    assert run_mock.call_args.kwargs["env"]["KUBECONFIG"] == str(kubeconfig.resolve())


def test_list_installed_byoc_releases_finds_agent_outside_default_namespace(
    tmp_path: Path,
) -> None:
    kubeconfig = tmp_path / "kubeconfig"
    kubeconfig.write_text("apiVersion: v1\n", encoding="utf-8")

    with (
        patch("agilerl.arena.byoc.cluster_helm.helm_available", return_value=True),
        patch(
            "agilerl.arena.byoc.cluster_helm.subprocess.run",
            return_value=_helm_list_result(
                (STORAGE_RELEASE, STORAGE_NAMESPACE),
                (AGENT_RELEASE, "arena-agent"),
            ),
        ),
    ):
        installed = list_installed_byoc_releases(kubeconfig)

    assert installed == [
        (AGENT_RELEASE, "arena-agent"),
        (STORAGE_RELEASE, STORAGE_NAMESPACE),
    ]


def test_list_installed_byoc_releases_prefers_cluster_name(
    tmp_path: Path,
) -> None:
    kubeconfig = tmp_path / "kubeconfig"
    kubeconfig.write_text("apiVersion: v1\n", encoding="utf-8")

    with (
        patch("agilerl.arena.byoc.cluster_helm.helm_available", return_value=True),
        patch(
            "agilerl.arena.byoc.cluster_helm.subprocess.run",
            return_value=_helm_list_result(
                (AGENT_RELEASE, AGENT_NAMESPACE),
                ("prod-cluster", "arena-agent"),
            ),
        ),
    ):
        installed = list_installed_byoc_releases(
            kubeconfig, cluster_name="prod-cluster"
        )

    assert installed == [("prod-cluster", "arena-agent")]


def test_list_installed_byoc_releases_raises_when_helm_list_fails(
    tmp_path: Path,
) -> None:
    kubeconfig = tmp_path / "kubeconfig"
    kubeconfig.write_text("apiVersion: v1\n", encoding="utf-8")

    with (
        patch("agilerl.arena.byoc.cluster_helm.helm_available", return_value=True),
        patch(
            "agilerl.arena.byoc.cluster_helm.subprocess.run",
            return_value=MagicMock(returncode=1, stdout="", stderr="forbidden"),
        ),
        pytest.raises(click.ClickException, match="helm list --all-namespaces failed"),
    ):
        list_installed_byoc_releases(kubeconfig)


def test_discover_agent_namespace_uses_explicit_value() -> None:
    with patch(
        "agilerl.arena.byoc.cluster_helm.list_installed_byoc_releases",
        return_value=[],
    ):
        namespace = discover_agent_namespace(
            None, cluster_name="prod-cluster", agent_namespace=" arena-agent "
        )

    assert namespace == "arena-agent"


def test_discover_agent_namespace_uses_helm_list(tmp_path: Path) -> None:
    kubeconfig = tmp_path / "kubeconfig"
    kubeconfig.write_text("apiVersion: v1\n", encoding="utf-8")

    with patch(
        "agilerl.arena.byoc.cluster_helm.list_installed_byoc_releases",
        return_value=[("prod-cluster", "arena-agent")],
    ):
        namespace = discover_agent_namespace(kubeconfig, cluster_name="prod-cluster")

    assert namespace == "arena-agent"


def test_discover_agent_namespace_rejects_duplicates(tmp_path: Path) -> None:
    kubeconfig = tmp_path / "kubeconfig"
    kubeconfig.write_text("apiVersion: v1\n", encoding="utf-8")

    with (
        patch(
            "agilerl.arena.byoc.cluster_helm.list_installed_byoc_releases",
            return_value=[
                ("prod-cluster", "arena"),
                ("prod-cluster", "arena-agent"),
            ],
        ),
        pytest.raises(click.ClickException, match="multiple namespaces"),
    ):
        discover_agent_namespace(kubeconfig, cluster_name="prod-cluster")


def test_discover_agent_namespace_rejects_missing_when_helm_can_list(
    tmp_path: Path,
) -> None:
    kubeconfig = tmp_path / "kubeconfig"
    kubeconfig.write_text("apiVersion: v1\n", encoding="utf-8")

    with (
        patch("agilerl.arena.byoc.cluster_helm.helm_available", return_value=True),
        patch(
            "agilerl.arena.byoc.cluster_helm.list_installed_byoc_releases",
            return_value=[],
        ),
        pytest.raises(click.ClickException, match="was not found"),
    ):
        discover_agent_namespace(kubeconfig, cluster_name="prod-cluster")


def test_discover_agent_namespace_defaults_without_helm() -> None:
    with (
        patch("agilerl.arena.byoc.cluster_helm.helm_available", return_value=False),
        patch(
            "agilerl.arena.byoc.cluster_helm.list_installed_byoc_releases",
            return_value=[],
        ),
    ):
        assert (
            discover_agent_namespace(None, cluster_name="prod-cluster")
            == AGENT_NAMESPACE
        )


def test_uninstall_byoc_releases_runs_helm_uninstall(tmp_path: Path) -> None:
    kubeconfig = tmp_path / "kubeconfig"
    kubeconfig.write_text("apiVersion: v1\n", encoding="utf-8")
    releases = [
        (AGENT_RELEASE, AGENT_NAMESPACE),
        (STORAGE_RELEASE, STORAGE_NAMESPACE),
    ]

    with patch("agilerl.arena.byoc.cluster_helm.subprocess.run") as run_mock:
        run_mock.return_value = MagicMock(returncode=0)
        uninstall_byoc_releases(releases, kubeconfig)

    assert run_mock.call_count == 2
    assert run_mock.call_args_list[0].args[0] == [
        "helm",
        "uninstall",
        AGENT_RELEASE,
        "--namespace",
        AGENT_NAMESPACE,
    ]
    assert run_mock.call_args_list[1].args[0] == [
        "helm",
        "uninstall",
        STORAGE_RELEASE,
        "--namespace",
        STORAGE_NAMESPACE,
    ]
    assert run_mock.call_args_list[0].kwargs["env"]["KUBECONFIG"] == str(
        kubeconfig.resolve()
    )


def test_uninstall_byoc_releases_raises_on_failure(tmp_path: Path) -> None:
    kubeconfig = tmp_path / "kubeconfig"
    kubeconfig.write_text("apiVersion: v1\n", encoding="utf-8")

    with patch("agilerl.arena.byoc.cluster_helm.subprocess.run") as run_mock:
        run_mock.return_value = MagicMock(returncode=1)
        with pytest.raises(click.ClickException, match="helm uninstall"):
            uninstall_byoc_releases(
                [(AGENT_RELEASE, AGENT_NAMESPACE)],
                kubeconfig,
            )


def test_resolve_helm_charts_root_env_must_be_a_directory(tmp_path: Path) -> None:
    env_file = tmp_path / "not-a-dir"
    env_file.write_text("x\n", encoding="utf-8")

    with patch.dict("os.environ", {"ARENA_HELM_CHARTS_DIR": str(env_file)}):
        with pytest.raises(click.ClickException, match="not a directory"):
            resolve_helm_charts_root(None)


def test_run_agent_chart_validate_skips_missing_script(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    package_root = tmp_path / "package"
    (package_root / AGENT_CHART).mkdir(parents=True)

    with caplog.at_level("WARNING"):
        run_agent_chart_validate(package_root)

    assert "no validate.sh" in caplog.text


def test_helm_upgrade_install_requires_helm(tmp_path: Path) -> None:
    values = tmp_path / "values.yaml"
    values.write_text("x: 1\n", encoding="utf-8")

    with patch("agilerl.arena.byoc.cluster_helm.shutil.which", return_value=None):
        with pytest.raises(click.ClickException, match="helm not found"):
            helm_upgrade_install(
                release=STORAGE_RELEASE,
                chart=tmp_path / "chart",
                namespace=STORAGE_NAMESPACE,
                values_file=values,
            )


def test_helm_upgrade_install_requires_values_file(tmp_path: Path) -> None:
    with patch(
        "agilerl.arena.byoc.cluster_helm.shutil.which", return_value="/usr/bin/helm"
    ):
        with pytest.raises(click.ClickException, match="Helm values file not found"):
            helm_upgrade_install(
                release=STORAGE_RELEASE,
                chart=tmp_path / "chart",
                namespace=STORAGE_NAMESPACE,
                values_file=tmp_path / "missing.yaml",
            )


def test_helm_upgrade_install_raises_on_helm_failure(tmp_path: Path) -> None:
    chart = tmp_path / "chart"
    chart.mkdir()
    values = tmp_path / "values.yaml"
    values.write_text("x: 1\n", encoding="utf-8")

    with (
        patch(
            "agilerl.arena.byoc.cluster_helm.shutil.which", return_value="/usr/bin/helm"
        ),
        patch("agilerl.arena.byoc.cluster_helm.subprocess.run") as run_mock,
    ):
        run_mock.return_value = MagicMock(returncode=1)
        with pytest.raises(click.ClickException, match="helm upgrade --install"):
            helm_upgrade_install(
                release=STORAGE_RELEASE,
                chart=chart,
                namespace=STORAGE_NAMESPACE,
                values_file=values,
            )


def test_wait_for_deployments_requires_kubectl() -> None:
    with patch("agilerl.arena.byoc.cluster_helm.shutil.which", return_value=None):
        with pytest.raises(click.ClickException, match="kubectl not found"):
            wait_for_deployments("app=agent", AGENT_NAMESPACE)


def test_merge_agent_chart_defaults_requires_values_file(tmp_path: Path) -> None:
    chart = tmp_path / "chart"
    chart.mkdir()

    with pytest.raises(click.ClickException, match="Chart values file not found"):
        merge_agent_chart_defaults(chart, {"storage": {}})


def test_merge_agent_chart_defaults_treats_non_mapping_as_empty(tmp_path: Path) -> None:
    chart = tmp_path / "chart"
    chart.mkdir()
    (chart / "values.yaml").write_text("- not-a-map\n", encoding="utf-8")

    assert merge_agent_chart_defaults(chart, {"storage": {"bucket": "b"}}) == {
        "storage": {"bucket": "b"}
    }


def test_list_installed_byoc_releases_skips_non_string_entries(tmp_path: Path) -> None:
    kubeconfig = tmp_path / "kubeconfig"
    kubeconfig.write_text("apiVersion: v1\n", encoding="utf-8")
    listed = MagicMock(
        returncode=0,
        stdout='[{"name": 1, "namespace": "arena"}, {"name": "arena-byoc-agent", "namespace": "arena"}]',
        stderr="",
    )

    with (
        patch("agilerl.arena.byoc.cluster_helm.helm_available", return_value=True),
        patch("agilerl.arena.byoc.cluster_helm.subprocess.run", return_value=listed),
    ):
        assert list_installed_byoc_releases(kubeconfig) == [
            ("arena-byoc-agent", "arena")
        ]


def test_discover_agent_install_returns_matching_explicit_namespace(
    tmp_path: Path,
) -> None:
    kubeconfig = tmp_path / "kubeconfig"
    kubeconfig.write_text("apiVersion: v1\n", encoding="utf-8")
    listed = MagicMock(
        returncode=0,
        stdout='[{"name": "prod-cluster", "namespace": "arena-agent"}]',
        stderr="",
    )

    with (
        patch("agilerl.arena.byoc.cluster_helm.helm_available", return_value=True),
        patch("agilerl.arena.byoc.cluster_helm.subprocess.run", return_value=listed),
    ):
        assert (
            discover_agent_namespace(
                kubeconfig,
                cluster_name="prod-cluster",
                agent_namespace="arena-agent",
            )
            == "arena-agent"
        )


def test_kubectl_secret_exists_is_false_without_kubectl() -> None:
    with patch("agilerl.arena.byoc.cluster_helm.shutil.which", return_value=None):
        assert _kubectl_secret_exists("arena-storage", "arena") is False


def test_ensure_agent_storage_secret_requires_kubectl_when_secret_missing() -> None:
    with (
        patch(
            "agilerl.arena.byoc.cluster_helm._kubectl_secret_exists",
            return_value=False,
        ),
        patch("agilerl.arena.byoc.cluster_helm.shutil.which", return_value=None),
    ):
        with pytest.raises(click.ClickException, match="Install kubectl"):
            ensure_agent_storage_secret(
                {"storage": {"secretName": "arena-storage", "createSecret": False}}
            )


def test_ensure_agent_storage_secret_skips_when_source_missing_and_kubectl_present() -> (
    None
):
    with (
        patch(
            "agilerl.arena.byoc.cluster_helm._kubectl_secret_exists",
            return_value=False,
        ),
        patch("agilerl.arena.byoc.cluster_helm.shutil.which", return_value="kubectl"),
        patch("agilerl.arena.byoc.cluster_helm._copy_k8s_secret") as copy_mock,
    ):
        ensure_agent_storage_secret(
            {"storage": {"secretName": "arena-storage", "createSecret": False}}
        )

    copy_mock.assert_not_called()


def test_copy_k8s_secret_raises_when_get_fails() -> None:
    with patch("agilerl.arena.byoc.cluster_helm.subprocess.run") as run_mock:
        run_mock.return_value = MagicMock(returncode=1, stdout="", stderr="missing")
        with pytest.raises(click.ClickException, match="could not"):
            _copy_k8s_secret("arena-storage", "storage", "arena")


def test_install_enterprise_agent_chart_installs_and_validates(tmp_path: Path) -> None:
    charts_root = tmp_path / "helm-setup"
    _write_agent_chart_defaults(charts_root / AGENT_CHART / "chart")
    out = tmp_path / "out"
    out.mkdir()
    (out / "agent-helm-values.yaml").write_text("storage:\n  endpoint: http://minio\n")

    with (
        patch("agilerl.arena.byoc.cluster_helm.helm_upgrade_install") as helm_mock,
        patch(
            "agilerl.arena.byoc.cluster_helm.run_agent_chart_validate"
        ) as validate_mock,
        patch("agilerl.arena.byoc.cluster_helm.ensure_agent_storage_secret"),
    ):
        install_enterprise_agent_chart(out, charts_dir=charts_root)

    helm_mock.assert_called_once()
    validate_mock.assert_called_once_with(charts_root)
    assert (out / "agent-helm-values-merged.yaml").is_file()


def test_install_enterprise_agent_chart_requires_agent_values(tmp_path: Path) -> None:
    charts_root = tmp_path / "helm-setup"
    charts_root.mkdir()
    out = tmp_path / "out"
    out.mkdir()

    with pytest.raises(click.ClickException, match="did not write agent values"):
        install_enterprise_agent_chart(out, charts_dir=charts_root)


def test_chart_path_requires_chart_directory(tmp_path: Path) -> None:
    with pytest.raises(click.ClickException, match="Helm chart not found"):
        _chart_path(tmp_path, AGENT_CHART)
    with pytest.raises(click.ClickException, match="Helm chart not found"):
        _chart_path_from_package(tmp_path, AGENT_CHART)


def test_load_agent_values_yaml_returns_empty_for_non_mapping(tmp_path: Path) -> None:
    values = tmp_path / "values.yaml"
    values.write_text("- item\n", encoding="utf-8")

    assert _load_agent_values_yaml(values) == {}


def test_resolve_helm_charts_root_uses_env_directory(tmp_path: Path) -> None:
    charts = tmp_path / "helm-setup"
    charts.mkdir()

    with patch.dict("os.environ", {"ARENA_HELM_CHARTS_DIR": str(charts)}):
        assert resolve_helm_charts_root(None) == charts.resolve()


def test_resolve_helm_charts_root_requires_a_directory() -> None:
    with patch.dict("os.environ", {"ARENA_HELM_CHARTS_DIR": ""}, clear=False):
        with pytest.raises(
            click.ClickException, match="Helm charts directory is required"
        ):
            resolve_helm_charts_root(None)


def test_helm_available_checks_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "agilerl.arena.byoc.cluster_helm.shutil.which",
        lambda name: "/usr/bin/helm" if name == "helm" else None,
    )

    assert helm_available() is True


def test_wait_for_deployments_succeeds() -> None:
    with (
        patch("agilerl.arena.byoc.cluster_helm.shutil.which", return_value="kubectl"),
        patch("agilerl.arena.byoc.cluster_helm.subprocess.run") as run_mock,
    ):
        run_mock.return_value = MagicMock(returncode=0)
        wait_for_deployments("app=agent", AGENT_NAMESPACE)

    assert run_mock.call_args.args[0][0] == "kubectl"
    assert "--for=condition=available" in run_mock.call_args.args[0]


def test_copy_k8s_secret_raises_when_apply_fails() -> None:
    secret = {
        "kind": "Secret",
        "metadata": {"name": "arena-storage", "namespace": "storage"},
        "data": {"accesskey": "a2V5"},
    }

    def run(argv: list[str], **_kwargs: object) -> MagicMock:
        if argv[:3] == ["kubectl", "get", "secret"]:
            return MagicMock(returncode=0, stdout=json.dumps(secret), stderr="")
        if argv[:3] == ["kubectl", "get", "namespace"]:
            return MagicMock(returncode=0, stdout="", stderr="")
        return MagicMock(returncode=1, stdout="", stderr="apply failed")

    with patch("agilerl.arena.byoc.cluster_helm.subprocess.run", side_effect=run):
        with pytest.raises(click.ClickException, match="Failed to copy secret"):
            _copy_k8s_secret("arena-storage", "storage", "arena")


def test_run_agent_chart_validate_raises_on_failure(tmp_path: Path) -> None:
    package_root = tmp_path / "package"
    agent_root = package_root / AGENT_CHART
    agent_root.mkdir(parents=True)
    (agent_root / "validate.sh").write_text("#!/bin/sh\nexit 1\n", encoding="utf-8")

    with patch("agilerl.arena.byoc.cluster_helm.subprocess.run") as run_mock:
        run_mock.return_value = MagicMock(returncode=1)
        with pytest.raises(click.ClickException, match="Agent chart validation failed"):
            run_agent_chart_validate(package_root)


def test_helm_upgrade_install_waits_with_selector(tmp_path: Path) -> None:
    chart = tmp_path / "chart"
    chart.mkdir()
    values = tmp_path / "values.yaml"
    values.write_text("x: 1\n", encoding="utf-8")

    with (
        patch(
            "agilerl.arena.byoc.cluster_helm.shutil.which", return_value="/usr/bin/helm"
        ),
        patch("agilerl.arena.byoc.cluster_helm.subprocess.run") as run_mock,
        patch("agilerl.arena.byoc.cluster_helm.wait_for_deployments") as wait_mock,
    ):
        run_mock.return_value = MagicMock(returncode=0)
        helm_upgrade_install(
            release=AGENT_RELEASE,
            chart=chart,
            namespace=AGENT_NAMESPACE,
            values_file=values,
            wait_for_selector="app=agent",
        )

    wait_mock.assert_called_once()
    assert "--wait" not in run_mock.call_args.args[0]


def test_write_merged_agent_values(tmp_path: Path) -> None:
    package_root = tmp_path / "package"
    chart = package_root / AGENT_CHART / "chart"
    _write_agent_chart_defaults(chart)

    merged = write_merged_agent_values(
        package_root, {"storage": {"bucket": "arena-data"}}
    )

    assert merged.is_file()
    assert "arena-data" in merged.read_text(encoding="utf-8")


def test_ensure_agent_storage_secret_skips_when_already_present() -> None:
    with (
        patch(
            "agilerl.arena.byoc.cluster_helm._kubectl_secret_exists",
            return_value=True,
        ),
        patch("agilerl.arena.byoc.cluster_helm._copy_k8s_secret") as copy_mock,
    ):
        ensure_agent_storage_secret(
            {"storage": {"secretName": "arena-storage", "createSecret": False}}
        )

    copy_mock.assert_not_called()


def _encoded_endpoint(endpoint: str) -> str:
    return base64.b64encode(endpoint.encode()).decode()


def _encoded_secret_data(values: dict[str, str]) -> dict[str, str]:
    return {key: _encoded_endpoint(value) for key, value in values.items()}


class TestSyncStorageSecretEndpoint:
    """Refreshing an unmanaged storage Secret from Helm values."""

    def test_skips_when_helm_creates_the_secret(self) -> None:
        with patch("agilerl.arena.byoc.cluster_helm.subprocess.run") as run_mock:
            updated = sync_storage_secret_endpoint(
                {
                    "storage": {
                        "secretName": "storage",
                        "endpoint": "https://s3.eu-west-1.amazonaws.com",
                        "createSecret": True,
                    }
                }
            )

        assert updated is False
        run_mock.assert_not_called()

    def test_skips_when_secret_is_missing(self) -> None:
        with patch(
            "agilerl.arena.byoc.cluster_helm._kubectl_secret_exists",
            return_value=False,
        ):
            updated = sync_storage_secret_endpoint(
                {
                    "storage": {
                        "secretName": "storage",
                        "endpoint": "https://s3.eu-west-1.amazonaws.com",
                        "createSecret": False,
                    }
                }
            )

        assert updated is False

    def test_skips_when_endpoint_already_matches(self) -> None:
        endpoint = "https://s3.eu-west-1.amazonaws.com"
        secret = {"data": _encoded_secret_data(storage_secret_endpoint_data(endpoint))}

        def run(argv: list[str], **_kwargs: object) -> MagicMock:
            if argv[:4] == ["kubectl", "get", "secret", "storage"] and "-o" not in argv:
                return MagicMock(returncode=0, stdout="", stderr="")
            if "-o" in argv:
                return MagicMock(returncode=0, stdout=json.dumps(secret), stderr="")
            return MagicMock(returncode=1, stdout="", stderr="unexpected")

        with patch("agilerl.arena.byoc.cluster_helm.subprocess.run", side_effect=run):
            updated = sync_storage_secret_endpoint(
                {
                    "storage": {
                        "secretName": "storage",
                        "endpoint": endpoint,
                        "createSecret": False,
                    }
                }
            )

        assert updated is False

    def test_patches_endpoint_keys_and_leaves_credentials(self) -> None:
        endpoint = "https://s3.eu-west-1.amazonaws.com"
        old = _encoded_endpoint("https://eks-byoc-wei-data.eu-west-1.amazonaws.com")
        secret = {
            "data": {
                "endpoint": old,
                "AWS_ENDPOINT_URL": old,
                "APP_AWS_CLIENT_ENDPOINT": old,
                "AWS_SECRET_ACCESS_KEY": _encoded_endpoint("secret-key"),
            }
        }
        patched: list[str] = []

        def run(argv: list[str], **_kwargs: object) -> MagicMock:
            if argv[:3] == ["kubectl", "patch", "secret"]:
                patched.append(argv[argv.index("-p") + 1])
                return MagicMock(returncode=0, stdout="", stderr="")
            if "-o" in argv:
                return MagicMock(returncode=0, stdout=json.dumps(secret), stderr="")
            return MagicMock(returncode=0, stdout="", stderr="")

        with patch("agilerl.arena.byoc.cluster_helm.subprocess.run", side_effect=run):
            updated = sync_storage_secret_endpoint(
                {
                    "storage": {
                        "secretName": "storage",
                        "endpoint": "https://s3.eu-west-1.amazonaws.com",
                        "createSecret": False,
                    }
                },
                agent_namespace="arena",
            )

        assert updated is True
        assert len(patched) == 1
        payload = json.loads(patched[0])
        assert payload["stringData"] == storage_secret_endpoint_data(endpoint)
        assert payload["stringData"]["AWS_REGION"] == "eu-west-1"
        assert "AWS_SECRET_ACCESS_KEY" not in payload["stringData"]

    def test_raises_when_patch_fails(self) -> None:
        secret = {"data": {}}

        def run(argv: list[str], **_kwargs: object) -> MagicMock:
            if argv[:3] == ["kubectl", "patch", "secret"]:
                return MagicMock(returncode=1, stdout="", stderr="forbidden")
            if "-o" in argv:
                return MagicMock(returncode=0, stdout=json.dumps(secret), stderr="")
            return MagicMock(returncode=0, stdout="", stderr="")

        with (
            patch("agilerl.arena.byoc.cluster_helm.subprocess.run", side_effect=run),
            pytest.raises(click.ClickException, match="Failed to update Secret"),
        ):
            sync_storage_secret_endpoint(
                {
                    "storage": {
                        "secretName": "storage",
                        "endpoint": "https://s3.eu-west-1.amazonaws.com",
                        "createSecret": False,
                    }
                }
            )


def test_install_enterprise_restarts_agent_after_storage_endpoint_change(
    tmp_path: Path,
) -> None:
    charts_root = tmp_path / "helm-setup"
    _write_agent_chart_defaults(charts_root / AGENT_CHART / "chart")
    out = tmp_path / "out"
    out.mkdir()
    (out / "agent-helm-values.yaml").write_text(
        "storage:\n  secretName: storage\n  endpoint: https://s3.example\n"
        "  createSecret: false\n"
    )
    order: list[str] = []

    def helm(**_kwargs: object) -> None:
        order.append("helm")

    def restart(namespace: str, release: str, *, wait: bool) -> None:
        order.append(f"restart:{namespace}:{release}:{wait}")

    with (
        patch("agilerl.arena.byoc.cluster_helm.helm_upgrade_install", side_effect=helm),
        patch("agilerl.arena.byoc.cluster_helm.run_agent_chart_validate"),
        patch("agilerl.arena.byoc.cluster_helm.ensure_agent_storage_secret"),
        patch(
            "agilerl.arena.byoc.cluster_helm.sync_storage_secret_endpoint",
            return_value=True,
        ),
        patch(
            "agilerl.arena.byoc.cluster_helm._restart_agent_deployment",
            side_effect=restart,
        ),
    ):
        install_enterprise_agent_chart(out, charts_dir=charts_root)

    assert order == ["helm", f"restart:{AGENT_NAMESPACE}:{AGENT_RELEASE}:True"]
