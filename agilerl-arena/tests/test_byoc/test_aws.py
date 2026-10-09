# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Tests for the AWS cluster provider."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import click
import pytest
import yaml

from agilerl.arena.byoc.provisioning.providers import get_provider
from agilerl.arena.byoc.provisioning.providers.aws.autoscaler import (
    CLUSTER_AUTOSCALER_CHART,
    CLUSTER_AUTOSCALER_CHART_VERSION,
    CLUSTER_AUTOSCALER_RELEASE,
    CLUSTER_AUTOSCALER_REPO,
    CLUSTER_AUTOSCALER_SERVICE_ACCOUNT,
    ensure_cluster_autoscaler,
)
from agilerl.arena.byoc.provisioning.providers.aws.cilium import (
    api_server_host,
    ensure_cilium,
    ensure_coredns_addon,
)
from agilerl.arena.byoc.provisioning.providers.aws.filesystem import (
    ARENA_SHARED_STORAGE_CLASS,
    build_efs_storage_class_manifest,
)
from agilerl.arena.byoc.provisioning.providers.aws.gateway import (
    NLB_SERVICE_ANNOTATIONS,
    configure_eks_gateway,
    delete_eks_gateway,
)
from agilerl.arena.byoc.provisioning.providers.aws.gpu import (
    GPU_STARTUP_SCRIPT,
    GPU_STARTUP_TAINT_KEY,
    NVIDIA_DEVICE_PLUGIN_CHART,
    NVIDIA_DEVICE_PLUGIN_CHART_VERSION,
    NVIDIA_DEVICE_PLUGIN_RELEASE,
    NVIDIA_DEVICE_PLUGIN_REPO,
    build_gpu_startup_manifests,
    ensure_nvidia_device_plugin,
)
from agilerl.arena.byoc.provisioning.providers.aws.provider import AWS, AwsProvider
from agilerl.arena.byoc.provisioning.providers.aws.spec import (
    AwsSpec,
    resource_classes,
)
from agilerl.arena.byoc.provisioning.providers.aws.terraform import (
    TerraformStateCredentials,
    delete_cluster_terraform_state,
    ensure_terraform_state_bucket,
)
from agilerl.arena.byoc.provisioning.provisioner import ClusterProvisioner
from agilerl.arena.byoc.provisioning.spec import (
    ClusterSpec,
    load_cluster_spec,
    render_default_cluster_spec,
)
from agilerl.arena.byoc.provisioning.terraform import ClusterOutputs, TerraformRunner

EXAMPLE_SPEC = """\
provider: aws
name: eks-byoc-wei
terraform_state:
  bucket: eks-byoc-wei-tfstate
arena:
  storage:
    bucket: eks-byoc-wei-data
    endpoint: https://storage.eu-north1.nebius.cloud
    install: false
  inference:
    domain: eks-byoc-wei-1.agilerl.rlops.ai
    hostname_template: "inference-{deploymentId}"
    tls_secret_name: star.eks-byoc-wei-1.agilerl.rlops.ai-tls
  gateway:
    enable: true
    name: arena
  workloads:
    ray_data_storage_class_name: arena-shared
    ray_data_pvc_size: 32Gi
aws:
  region: eu-west-1
  system:
    min_node_count: 2
    max_node_count: 10
    instance_type: t3.small
  workers: []
  object_storage:
    size_gib: 1024
  shared_filesystem:
    provision: false
"""


def _aws_spec(**overrides: object) -> ClusterSpec:
    raw = dict(overrides)
    storage: dict[str, object] = {
        "bucket": "eks-data",
        "endpoint": "https://s3.eu-west-1.amazonaws.com",
    }
    arena: dict[str, object] = {
        "storage": storage,
        "inference": {"domain": "inference.example.com"},
    }
    if "storage_bucket_name" in raw:
        storage["bucket"] = raw.pop("storage_bucket_name")
    if "storage_endpoint" in raw:
        storage["endpoint"] = raw.pop("storage_endpoint")
    if "inference" in raw:
        arena["inference"] = raw.pop("inference")
    gateway = raw.pop("gateway_api", None)
    if isinstance(gateway, dict):
        mapped: dict[str, object] = {}
        if "enable" in gateway:
            mapped["enable"] = gateway["enable"]
        if "gateway_name" in gateway:
            mapped["name"] = gateway["gateway_name"]
        arena["gateway"] = mapped
    aws: dict[str, object] = {
        "region": "eu-west-1",
        "object_storage": {"size_gib": 1024},
        "shared_filesystem": {"provision": False},
    }
    aws.update(raw)
    return ClusterSpec.model_validate(
        {
            "provider": "aws",
            "name": "eks-byoc",
            "terraform_state": {"bucket": "eks-tfstate"},
            "arena": arena,
            "aws": aws,
        }
    )


def _outputs(
    tmp_path: Path,
    file_system_id: str | None = None,
    vpc_id: str | None = None,
) -> ClusterOutputs:
    return ClusterOutputs(
        cluster_name="eks-byoc",
        context="eks-byoc",
        kubeconfig_path=tmp_path / "kubeconfig",
        storage_access_key_id="AKIA",
        storage_secret_access_key="secret",
        worker_node_group_ids={},
        efs_file_system_id=file_system_id,
        vpc_id=vpc_id,
        cluster_endpoint="https://example.eks.amazonaws.com",
    )


class TestAwsSpec:
    def test_loads_the_example_document(self, tmp_path: Path) -> None:
        path = tmp_path / "eks-byoc.yaml"
        path.write_text(EXAMPLE_SPEC, encoding="utf-8")

        spec = load_cluster_spec(path)

        assert spec.provider == "aws"
        assert isinstance(spec.aws, AwsSpec)
        assert spec.aws.region == "eu-west-1"
        assert spec.aws.system.instance_type == "t3.small"
        assert spec.aws.workers == []
        assert spec.aws.shared_filesystem.provision is False
        assert spec.arena.workloads.ray_data_storage_class_name == "arena-shared"

    def test_shared_filesystem_requests_efs(self) -> None:
        spec = _aws_spec(shared_filesystem={"provision": True})
        values = spec.aws
        assert isinstance(values, AwsSpec)

        assert spec.terraform_values()["provision_shared_filesystem"] is True

    def test_gateway_api_enables_the_load_balancer(self) -> None:
        enabled = _aws_spec()
        disabled = _aws_spec(gateway_api={"enable": False})

        assert isinstance(enabled.aws, AwsSpec)
        assert isinstance(disabled.aws, AwsSpec)
        assert enabled.terraform_values()["enable_gateway_api"] is True
        assert disabled.terraform_values()["enable_gateway_api"] is False

    def test_default_template_loads(self) -> None:
        rendered = render_default_cluster_spec(provider="aws", name="demo")

        spec = ClusterSpec.model_validate(yaml.safe_load(rendered))

        assert spec.provider == "aws"
        assert spec.name == "demo"
        assert spec.aws is not None
        assert spec.aws.system.instance_type == "t3.xlarge"
        assert spec.aws.shared_filesystem.provision is False


class TestEfsStorageClass:
    def test_manifest_targets_the_file_system(self) -> None:
        manifest = build_efs_storage_class_manifest("fs-123")

        assert manifest["metadata"]["name"] == ARENA_SHARED_STORAGE_CLASS
        assert manifest["provisioner"] == "efs.csi.aws.com"
        assert manifest["parameters"]["fileSystemId"] == "fs-123"
        assert manifest["parameters"]["provisioningMode"] == "efs-ap"


class TestAwsProvider:
    def test_registry_returns_the_aws_provider(self) -> None:
        assert get_provider("aws") is AWS

    def test_cluster_exists_waits_for_the_system_node_group(
        self, tmp_path: Path
    ) -> None:
        runner = MagicMock()
        runner.state_list.return_value = ("aws_eks_cluster.arena",)
        provisioner = ClusterProvisioner(runner=runner)

        with patch.object(AWS, "open_state", return_value=tmp_path):
            assert provisioner.cluster_exists(_aws_spec(), state_dir=tmp_path) is False

        runner.state_list.return_value = (
            "aws_eks_cluster.arena",
            "aws_eks_node_group.arena",
        )
        with patch.object(AWS, "open_state", return_value=tmp_path):
            assert provisioner.cluster_exists(_aws_spec(), state_dir=tmp_path) is True

    def test_post_apply_creates_the_storage_class_when_efs_is_enabled(
        self, tmp_path: Path
    ) -> None:
        spec = _aws_spec(
            shared_filesystem={"provision": True},
            gateway_api={"enable": False},
        )
        outputs = _outputs(tmp_path, file_system_id="fs-123")

        with (
            patch(
                "agilerl.arena.byoc.provisioning.providers.aws.provider.ensure_storage_secret"
            ),
            patch(
                "agilerl.arena.byoc.provisioning.providers.aws.provider.ensure_gpu_runtime_class"
            ),
            patch(
                "agilerl.arena.byoc.provisioning.providers.aws.provider.ensure_cilium"
            ),
            patch(
                "agilerl.arena.byoc.provisioning.providers.aws.provider.ensure_coredns_addon"
            ),
            patch(
                "agilerl.arena.byoc.provisioning.providers.aws.provider.configure_eks_gateway"
            ) as gateway,
            patch(
                "agilerl.arena.byoc.provisioning.providers.aws.provider.ensure_efs_storage_class"
            ) as storage_class,
        ):
            AWS.post_apply(spec, outputs)

        storage_class.assert_called_once_with(
            kubeconfig_path=outputs.kubeconfig_path,
            file_system_id="fs-123",
        )
        gateway.assert_not_called()

    def test_post_apply_skips_the_storage_class_when_efs_is_disabled(
        self, tmp_path: Path
    ) -> None:
        spec = _aws_spec(gateway_api={"enable": False})
        outputs = _outputs(tmp_path)

        with (
            patch(
                "agilerl.arena.byoc.provisioning.providers.aws.provider.ensure_storage_secret"
            ),
            patch(
                "agilerl.arena.byoc.provisioning.providers.aws.provider.ensure_gpu_runtime_class"
            ),
            patch(
                "agilerl.arena.byoc.provisioning.providers.aws.provider.ensure_cilium"
            ),
            patch(
                "agilerl.arena.byoc.provisioning.providers.aws.provider.ensure_coredns_addon"
            ),
            patch(
                "agilerl.arena.byoc.provisioning.providers.aws.provider.configure_eks_gateway"
            ) as gateway,
            patch(
                "agilerl.arena.byoc.provisioning.providers.aws.provider.ensure_efs_storage_class"
            ) as storage_class,
        ):
            AWS.post_apply(spec, outputs)

        storage_class.assert_not_called()
        gateway.assert_not_called()

    def test_post_apply_installs_the_cilium_gateway(self, tmp_path: Path) -> None:
        spec = _aws_spec()
        outputs = _outputs(tmp_path, vpc_id="vpc-123")
        parent_refs = [{"name": "arena", "sectionName": "https"}]

        with (
            patch(
                "agilerl.arena.byoc.provisioning.providers.aws.provider.ensure_storage_secret"
            ),
            patch(
                "agilerl.arena.byoc.provisioning.providers.aws.provider.ensure_gpu_runtime_class"
            ),
            patch(
                "agilerl.arena.byoc.provisioning.providers.aws.provider.ensure_cilium"
            ),
            patch(
                "agilerl.arena.byoc.provisioning.providers.aws.provider.ensure_coredns_addon"
            ),
            patch(
                "agilerl.arena.byoc.provisioning.providers.aws.provider.configure_eks_gateway",
                return_value=parent_refs,
            ) as gateway,
        ):
            result = AWS.post_apply(spec, outputs)

        gateway.assert_called_once_with(
            kubeconfig_path=outputs.kubeconfig_path,
            gateway_name="arena",
            domain="inference.example.com",
            cluster_name="eks-byoc",
            region="eu-west-1",
            vpc_id="vpc-123",
            tls_secret_name="arena-inference-tls",
        )
        assert result.gateway_api_parent_refs == parent_refs

    def test_post_apply_requires_the_vpc_id(self, tmp_path: Path) -> None:
        spec = _aws_spec()

        with (
            patch(
                "agilerl.arena.byoc.provisioning.providers.aws.provider.ensure_storage_secret"
            ),
            patch(
                "agilerl.arena.byoc.provisioning.providers.aws.provider.ensure_gpu_runtime_class"
            ),
            patch(
                "agilerl.arena.byoc.provisioning.providers.aws.provider.ensure_cilium"
            ),
            patch(
                "agilerl.arena.byoc.provisioning.providers.aws.provider.ensure_coredns_addon"
            ),
            pytest.raises(click.ClickException, match="VPC id"),
        ):
            AWS.post_apply(spec, _outputs(tmp_path))

    def test_post_apply_requires_the_file_system_id(self, tmp_path: Path) -> None:
        spec = _aws_spec(shared_filesystem={"provision": True})

        with (
            patch(
                "agilerl.arena.byoc.provisioning.providers.aws.provider.ensure_storage_secret"
            ),
            patch(
                "agilerl.arena.byoc.provisioning.providers.aws.provider.ensure_gpu_runtime_class"
            ),
            patch(
                "agilerl.arena.byoc.provisioning.providers.aws.provider.ensure_cilium"
            ),
            patch(
                "agilerl.arena.byoc.provisioning.providers.aws.provider.ensure_coredns_addon"
            ),
            pytest.raises(click.ClickException, match="EFS file system id"),
        ):
            AWS.post_apply(spec, _outputs(tmp_path))


class TestAwsGpuWorkers:
    def test_worker_node_group_uses_the_nvidia_ami(self) -> None:
        module = (
            Path(__file__).resolve().parents[2]
            / "agilerl"
            / "arena"
            / "byoc"
            / "provisioning"
            / "terraform_modules"
            / "aws"
            / "main.tf"
        )
        text = module.read_text(encoding="utf-8")
        worker = text.split('resource "aws_eks_node_group" "worker"')[1]

        assert "AL2023_x86_64_NVIDIA" in worker
        assert "AL2023_x86_64_STANDARD" in worker
        assert "each.value.num_gpus > 0" in worker
        assert '"nvidia.com/gpu.present" = "true"' in worker
        assert '"k8s.io/cluster-autoscaler/enabled"' in worker
        assert '"k8s.io/cluster-autoscaler/${var.cluster_name}"' in worker
        assert "node-template/label/nvidia.com/gpu.present" in worker
        assert "node-template/taint/nvidia.com/gpu" in worker
        assert "node-template/resources/nvidia.com/gpu" in worker
        assert 'resource "aws_autoscaling_group_tag" "worker_autoscaler"' in worker
        assert "launch_template" in worker
        system = text.split('resource "aws_eks_node_group" "worker"')[0]
        assert '"k8s.io/cluster-autoscaler/enabled"' not in system
        assert GPU_STARTUP_TAINT_KEY in system
        assert "registerWithTaints" in system
        worker_group = worker.split('resource "aws_autoscaling_group_tag"')[0]
        assert GPU_STARTUP_TAINT_KEY not in worker_group

    def test_system_and_worker_node_groups_use_100gib_disks(self) -> None:
        module = (
            Path(__file__).resolve().parents[2]
            / "agilerl"
            / "arena"
            / "byoc"
            / "provisioning"
            / "terraform_modules"
            / "aws"
            / "main.tf"
        )
        text = module.read_text(encoding="utf-8")
        arena = text.split('resource "aws_eks_node_group" "arena"')[1].split(
            'resource "aws_eks_node_group" "worker"'
        )[0]
        worker = text.split('resource "aws_eks_node_group" "worker"')[1]

        assert "node_disk_size_gib = 100" in text
        assert "disk_size       = local.node_disk_size_gib" in arena
        assert "volume_size           = local.node_disk_size_gib" in text
        assert "each.value.num_gpus > 0 ? null : local.node_disk_size_gib" in worker

    def test_post_apply_installs_the_device_plugin_for_gpu_workers(
        self, tmp_path: Path
    ) -> None:
        spec = _aws_spec(
            gateway_api={"enable": False},
            workers=[
                {
                    "name": "gpu",
                    "min_node_count": 0,
                    "max_node_count": 1,
                    "instance_type": "g6.xlarge",
                    "num_cpus": 16,
                    "num_gpus": 1,
                    "memory_gib": 64,
                    "gpu_type": "l4",
                    "vram_gib": 24,
                }
            ],
        )
        outputs = _outputs(tmp_path)

        with (
            patch(
                "agilerl.arena.byoc.provisioning.providers.aws.provider.ensure_storage_secret"
            ),
            patch(
                "agilerl.arena.byoc.provisioning.providers.aws.provider.ensure_gpu_runtime_class"
            ),
            patch(
                "agilerl.arena.byoc.provisioning.providers.aws.provider.ensure_cilium"
            ),
            patch(
                "agilerl.arena.byoc.provisioning.providers.aws.provider.ensure_coredns_addon"
            ),
            patch(
                "agilerl.arena.byoc.provisioning.providers.aws.provider.ensure_nvidia_device_plugin"
            ) as plugin,
            patch(
                "agilerl.arena.byoc.provisioning.providers.aws.provider.ensure_gpu_startup_taint"
            ) as startup,
            patch(
                "agilerl.arena.byoc.provisioning.providers.aws.provider.ensure_cluster_autoscaler"
            ) as autoscaler,
        ):
            AWS.post_apply(spec, outputs)

        plugin.assert_called_once_with(kubeconfig_path=outputs.kubeconfig_path)
        startup.assert_called_once_with(kubeconfig_path=outputs.kubeconfig_path)
        autoscaler.assert_called_once_with(
            kubeconfig_path=outputs.kubeconfig_path,
            cluster_name="eks-byoc",
            region="eu-west-1",
        )

    def test_post_apply_skips_the_device_plugin_without_gpu_workers(
        self, tmp_path: Path
    ) -> None:
        spec = _aws_spec(gateway_api={"enable": False})

        with (
            patch(
                "agilerl.arena.byoc.provisioning.providers.aws.provider.ensure_storage_secret"
            ),
            patch(
                "agilerl.arena.byoc.provisioning.providers.aws.provider.ensure_gpu_runtime_class"
            ),
            patch(
                "agilerl.arena.byoc.provisioning.providers.aws.provider.ensure_cilium"
            ),
            patch(
                "agilerl.arena.byoc.provisioning.providers.aws.provider.ensure_coredns_addon"
            ),
            patch(
                "agilerl.arena.byoc.provisioning.providers.aws.provider.ensure_nvidia_device_plugin"
            ) as plugin,
            patch(
                "agilerl.arena.byoc.provisioning.providers.aws.provider.ensure_gpu_startup_taint"
            ) as startup,
            patch(
                "agilerl.arena.byoc.provisioning.providers.aws.provider.ensure_cluster_autoscaler"
            ) as autoscaler,
        ):
            AWS.post_apply(spec, _outputs(tmp_path))

        plugin.assert_not_called()
        startup.assert_not_called()
        autoscaler.assert_not_called()

    def test_resource_class_reserves_kubelet_overhead_for_the_instance_size(
        self,
    ) -> None:
        spec = _aws_spec(
            workers=[
                {
                    "name": "gpu",
                    "instance_type": "g4dn.xlarge",
                    "num_cpus": 4,
                    "num_gpus": 1,
                    "memory_gib": 16,
                    "gpu_type": "t4",
                    "vram_gib": 16,
                }
            ]
        )

        classes = resource_classes(spec, {"gpu": "ng-1"})

        compute = classes[0].metadata["computeResource"]
        assert compute["numCpus"] == 3
        assert compute["memoryBytes"] == "14 GiB"
        assert compute["numGpus"] == 1

    def test_rejects_an_instance_with_nothing_left_after_reservation(self) -> None:
        spec = _aws_spec(
            workers=[
                {
                    "name": "gpu",
                    "instance_type": "t3.micro",
                    "num_cpus": 1,
                    "num_gpus": 1,
                    "memory_gib": 1,
                    "gpu_type": "t4",
                    "vram_gib": 16,
                }
            ]
        )

        with pytest.raises(click.ClickException, match="too small"):
            resource_classes(spec, {"gpu": "ng-1"})


class TestCilium:
    def test_cluster_does_not_bootstrap_kube_proxy(self) -> None:
        module = (
            Path(__file__).resolve().parents[2]
            / "agilerl"
            / "arena"
            / "byoc"
            / "provisioning"
            / "terraform_modules"
            / "aws"
            / "main.tf"
        )
        text = module.read_text(encoding="utf-8")

        assert "bootstrap_self_managed_addons = false" in text
        assert 'addon_name   = "kube-proxy"' not in text
        assert "ec2:DescribeRouteTables" in text

    def test_install_replaces_kube_proxy(self, tmp_path: Path) -> None:
        calls: list[list[str]] = []

        def run(argv: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
            calls.append(argv)
            return subprocess.CompletedProcess(argv, 0, "", "")

        with patch(
            "agilerl.arena.byoc.provisioning.providers.aws.cilium.helm_available",
            return_value=True,
        ):
            ensure_cilium(
                kubeconfig_path=tmp_path / "kubeconfig",
                api_server_host="example.eks.amazonaws.com",
                run=run,
            )

        rendered = " ".join(calls[0])
        assert "kubeProxyReplacement=true" in rendered
        assert "k8sServiceHost=example.eks.amazonaws.com" in rendered
        assert "ipam.mode=eni" in rendered
        assert "eni.enabled=true" in rendered
        assert "enableIPv4Masquerade=true" in rendered
        assert "bpf.masquerade=true" in rendered
        assert "egressMasqueradeInterfaces" not in rendered
        assert "--wait" in rendered

    def test_install_before_nodes_does_not_wait(self, tmp_path: Path) -> None:
        calls: list[list[str]] = []

        def run(argv: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
            calls.append(argv)
            return subprocess.CompletedProcess(argv, 0, "", "")

        with patch(
            "agilerl.arena.byoc.provisioning.providers.aws.cilium.helm_available",
            return_value=True,
        ):
            ensure_cilium(
                kubeconfig_path=tmp_path / "kubeconfig",
                api_server_host="example.eks.amazonaws.com",
                wait=False,
                run=run,
            )

        assert "--wait" not in calls[0]

    def test_api_server_host_rejects_a_bare_endpoint(self) -> None:
        with pytest.raises(click.ClickException, match="no hostname"):
            api_server_host("not a url")

    def test_coredns_addon_is_created_when_missing(self) -> None:
        calls: list[list[str]] = []

        def run(argv: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
            calls.append(argv)
            if "describe-addon" in argv:
                return subprocess.CompletedProcess(
                    argv, 1, "", "ResourceNotFoundException: addon not found"
                )
            return subprocess.CompletedProcess(argv, 0, "", "")

        ensure_coredns_addon(cluster_name="eks-byoc", region="eu-west-1", run=run)

        rendered = [" ".join(argv) for argv in calls]
        assert any("create-addon" in command for command in rendered)
        assert any("addon-active" in command for command in rendered)

    def test_post_apply_installs_cilium_before_coredns(self, tmp_path: Path) -> None:
        spec = _aws_spec(gateway_api={"enable": False})
        calls: list[str] = []

        with (
            patch(
                "agilerl.arena.byoc.provisioning.providers.aws.provider.ensure_storage_secret"
            ),
            patch(
                "agilerl.arena.byoc.provisioning.providers.aws.provider.ensure_gpu_runtime_class"
            ),
            patch(
                "agilerl.arena.byoc.provisioning.providers.aws.provider.ensure_cilium",
                side_effect=lambda **_kwargs: calls.append("cilium"),
            ) as cilium,
            patch(
                "agilerl.arena.byoc.provisioning.providers.aws.provider.ensure_coredns_addon",
                side_effect=lambda **_kwargs: calls.append("coredns"),
            ),
        ):
            AWS.post_apply(spec, _outputs(tmp_path))

        cilium.assert_called_once_with(
            kubeconfig_path=tmp_path / "kubeconfig",
            api_server_host="example.eks.amazonaws.com",
        )
        assert calls == ["cilium", "coredns"]

    def test_install_cni_runs_before_node_groups(self, tmp_path: Path) -> None:
        runner = MagicMock()
        runner.terraform_output.return_value = "https://example.eks.amazonaws.com"
        runner.run.return_value = MagicMock(returncode=0)

        with (
            patch(
                "agilerl.arena.byoc.provisioning.providers.aws.provider.ensure_cilium"
            ) as cilium,
            patch(
                "agilerl.arena.byoc.provisioning.providers.aws.provider.build_eks_kubeconfig_argv",
                return_value=["aws", "eks", "update-kubeconfig"],
            ),
        ):
            AWS.install_cni(_aws_spec(), tmp_path, runner)

        runner.apply_targets.assert_called_once_with(
            tmp_path, ("aws_eks_cluster.arena",)
        )
        cilium.assert_called_once_with(
            kubeconfig_path=tmp_path / "kubeconfig",
            api_server_host="example.eks.amazonaws.com",
            wait=False,
        )


class TestNvidiaDevicePlugin:
    def test_helm_installs_the_pinned_chart(self, tmp_path: Path) -> None:
        calls: list[list[str]] = []

        def run(argv: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
            calls.append(argv)
            return subprocess.CompletedProcess(argv, 0, "", "")

        with patch(
            "agilerl.arena.byoc.provisioning.providers.aws.gpu.helm_available",
            return_value=True,
        ):
            ensure_nvidia_device_plugin(tmp_path / "kubeconfig", run=run)

        assert calls == [
            [
                "helm",
                "upgrade",
                "--install",
                NVIDIA_DEVICE_PLUGIN_RELEASE,
                NVIDIA_DEVICE_PLUGIN_CHART,
                "--repo",
                NVIDIA_DEVICE_PLUGIN_REPO,
                "--namespace",
                "kube-system",
                "--version",
                NVIDIA_DEVICE_PLUGIN_CHART_VERSION,
                "--wait",
                "--timeout",
                "10m",
            ]
        ]

    def test_requires_helm(self, tmp_path: Path) -> None:
        with (
            patch(
                "agilerl.arena.byoc.provisioning.providers.aws.gpu.helm_available",
                return_value=False,
            ),
            pytest.raises(click.ClickException, match="helm not found"),
        ):
            ensure_nvidia_device_plugin(tmp_path / "kubeconfig")

    def test_helm_failure_raises(self, tmp_path: Path) -> None:
        def run(argv: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
            return subprocess.CompletedProcess(argv, 1, "", "chart missing")

        with (
            patch(
                "agilerl.arena.byoc.provisioning.providers.aws.gpu.helm_available",
                return_value=True,
            ),
            pytest.raises(click.ClickException, match="chart missing"),
        ):
            ensure_nvidia_device_plugin(tmp_path / "kubeconfig", run=run)

    def test_helm_failure_uses_stdout_when_stderr_is_empty(
        self, tmp_path: Path
    ) -> None:
        def run(argv: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
            return subprocess.CompletedProcess(argv, 1, "upgrade failed", "")

        with (
            patch(
                "agilerl.arena.byoc.provisioning.providers.aws.gpu.helm_available",
                return_value=True,
            ),
            pytest.raises(click.ClickException, match="upgrade failed"),
        ):
            ensure_nvidia_device_plugin(tmp_path / "kubeconfig", run=run)

    def test_helm_failure_without_output(self, tmp_path: Path) -> None:
        def run(argv: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
            return subprocess.CompletedProcess(argv, 1, "", "")

        with (
            patch(
                "agilerl.arena.byoc.provisioning.providers.aws.gpu.helm_available",
                return_value=True,
            ),
            pytest.raises(click.ClickException, match="helm upgrade --install"),
        ):
            ensure_nvidia_device_plugin(tmp_path / "kubeconfig", run=run)


class TestGpuStartupTaint:
    def test_script_waits_for_an_advertised_gpu(self) -> None:
        namespace: dict[str, object] = {}
        exec(GPU_STARTUP_SCRIPT.split("def main()")[0], namespace)
        patch_body = namespace["patch_body"]
        startup = {
            "key": GPU_STARTUP_TAINT_KEY,
            "value": "true",
            "effect": "NoSchedule",
        }
        gpu_taint = {
            "key": "nvidia.com/gpu",
            "value": "true",
            "effect": "NoSchedule",
        }
        waiting = {
            "spec": {"taints": [startup, gpu_taint]},
            "status": {"allocatable": {}},
        }
        ready = {
            "spec": {"taints": [startup, gpu_taint]},
            "status": {"allocatable": {"nvidia.com/gpu": "1"}},
        }

        assert GPU_STARTUP_TAINT_KEY in GPU_STARTUP_SCRIPT
        assert patch_body(waiting) is None
        assert patch_body(ready) == {"spec": {"taints": [gpu_taint]}}

    def test_manifest_runs_the_controller_in_kube_system(self) -> None:
        deployment = next(
            manifest
            for manifest in build_gpu_startup_manifests()
            if manifest["kind"] == "Deployment"
        )
        container = deployment["spec"]["template"]["spec"]["containers"][0]
        role = next(
            manifest
            for manifest in build_gpu_startup_manifests()
            if manifest["kind"] == "ClusterRole"
        )

        assert container["command"] == ["python", "-c", GPU_STARTUP_SCRIPT]
        assert role["rules"] == [
            {
                "apiGroups": [""],
                "resources": ["nodes"],
                "verbs": ["get", "list", "patch"],
            }
        ]


class TestClusterAutoscaler:
    def test_helm_installs_the_pinned_chart(self, tmp_path: Path) -> None:
        calls: list[list[str]] = []

        def run(argv: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
            calls.append(argv)
            return subprocess.CompletedProcess(argv, 0, "", "")

        with patch(
            "agilerl.arena.byoc.provisioning.providers.aws.autoscaler.helm_available",
            return_value=True,
        ):
            ensure_cluster_autoscaler(
                tmp_path / "kubeconfig",
                cluster_name="eks-byoc",
                region="eu-west-1",
                run=run,
            )

        assert calls == [
            [
                "helm",
                "upgrade",
                "--install",
                CLUSTER_AUTOSCALER_RELEASE,
                CLUSTER_AUTOSCALER_CHART,
                "--repo",
                CLUSTER_AUTOSCALER_REPO,
                "--namespace",
                "kube-system",
                "--version",
                CLUSTER_AUTOSCALER_CHART_VERSION,
                "--set",
                "autoDiscovery.clusterName=eks-byoc",
                "--set",
                "awsRegion=eu-west-1",
                "--set",
                f"rbac.serviceAccount.name={CLUSTER_AUTOSCALER_SERVICE_ACCOUNT}",
                "--wait",
                "--timeout",
                "10m",
            ]
        ]

    def test_requires_helm(self, tmp_path: Path) -> None:
        with (
            patch(
                "agilerl.arena.byoc.provisioning.providers.aws.autoscaler.helm_available",
                return_value=False,
            ),
            pytest.raises(click.ClickException, match="helm not found"),
        ):
            ensure_cluster_autoscaler(
                tmp_path / "kubeconfig",
                cluster_name="eks-byoc",
                region="eu-west-1",
            )

    def test_helm_failure_raises(self, tmp_path: Path) -> None:
        def run(argv: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
            return subprocess.CompletedProcess(argv, 1, "", "chart missing")

        with (
            patch(
                "agilerl.arena.byoc.provisioning.providers.aws.autoscaler.helm_available",
                return_value=True,
            ),
            pytest.raises(click.ClickException, match="chart missing"),
        ):
            ensure_cluster_autoscaler(
                tmp_path / "kubeconfig",
                cluster_name="eks-byoc",
                region="eu-west-1",
                run=run,
            )


class TestCiliumNlbGateway:
    def test_configure_uses_cilium_and_an_nlb(self, tmp_path: Path) -> None:
        calls: list[tuple[list[str], object]] = []

        def run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
            calls.append((argv, kwargs.get("input")))
            return subprocess.CompletedProcess(argv, 0, "", "")

        with (
            patch(
                "agilerl.arena.byoc.provisioning.providers.aws.gateway.helm_available",
                return_value=True,
            ),
            patch(
                "agilerl.arena.byoc.provisioning.gateway_api.require_kubectl",
            ),
        ):
            refs = configure_eks_gateway(
                kubeconfig_path=tmp_path / "kubeconfig",
                gateway_name="arena",
                domain="inference.example.com",
                cluster_name="eks-byoc",
                region="eu-west-1",
                vpc_id="vpc-123",
                tls_secret_name="arena-inference-tls",
                run=run,
            )

        rendered = [" ".join(argv) for argv, _ in calls]
        helm = next(command for command in rendered if command.startswith("helm "))
        assert "ALBGatewayAPI" not in helm
        assert "vpcId=vpc-123" in helm
        gateway = next(
            yaml.safe_load(str(payload))
            for argv, payload in calls
            if payload and "kind: Gateway\n" in str(payload)
        )
        assert gateway["spec"]["gatewayClassName"] == "cilium"
        https = gateway["spec"]["listeners"][1]
        assert https["tls"]["certificateRefs"][0]["name"] == "arena-inference-tls"
        assert (
            gateway["spec"]["infrastructure"]["annotations"] == NLB_SERVICE_ANNOTATIONS
        )
        assert refs[0]["name"] == "arena"
        assert refs[0]["sectionName"] == "https"


class TestDeleteEksGateway:
    def test_deletes_the_gateway_and_waits_for_its_load_balancer(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        calls: list[tuple[list[str], object]] = []

        def run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
            calls.append((argv, kwargs.get("env")))
            return subprocess.CompletedProcess(argv, 0, "", "")

        kubeconfig = tmp_path / "kubeconfig"
        with patch(
            "agilerl.arena.byoc.provisioning.providers.aws.gateway.require_kubectl"
        ):
            delete_eks_gateway(kubeconfig, "inference-gateway", run=run)

        assert [argv for argv, _env in calls] == [
            [
                "kubectl",
                "delete",
                "gateway",
                "inference-gateway",
                "--namespace",
                "arena",
                "--ignore-not-found=true",
                "--wait=true",
                "--cascade=foreground",
                "--timeout",
                "10m",
            ],
            [
                "kubectl",
                "delete",
                "service",
                "cilium-gateway-inference-gateway",
                "--namespace",
                "arena",
                "--ignore-not-found=true",
                "--wait=true",
                "--timeout",
                "10m",
            ],
        ]
        env = calls[0][1]
        assert isinstance(env, dict)
        assert env["KUBECONFIG"] == str(kubeconfig.resolve())
        assert (
            "Deleting Gateway 'inference-gateway' in namespace 'arena'."
            in capsys.readouterr().out
        )

    def test_raises_when_kubectl_delete_fails(self, tmp_path: Path) -> None:
        def run(argv: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
            return subprocess.CompletedProcess(argv, 1, "", "connection refused")

        with (
            patch(
                "agilerl.arena.byoc.provisioning.providers.aws.gateway.require_kubectl"
            ),
            pytest.raises(click.ClickException, match="connection refused"),
        ):
            delete_eks_gateway(tmp_path / "kubeconfig", "inference-gateway", run=run)


class TestAwsProviderDeleteGateway:
    def test_deletes_the_gateway_before_the_cluster_is_destroyed(
        self, tmp_path: Path
    ) -> None:
        spec = _aws_spec(
            gateway_api={"enable": True, "gateway_name": "inference-gateway"}
        )
        runner = MagicMock()
        runner.run.return_value = subprocess.CompletedProcess([], 0, "", "")

        with (
            patch(
                "agilerl.arena.byoc.provisioning.providers.aws.kubeconfig.resolve_aws_executable",
                return_value="aws",
            ),
            patch(
                "agilerl.arena.byoc.provisioning.providers.aws.provider.delete_eks_gateway"
            ) as delete,
        ):
            AWS.delete_gateway(spec, tmp_path, runner)

        argv = runner.run.call_args.args[0]
        assert argv[1:6] == [
            "eks",
            "update-kubeconfig",
            "--name",
            "eks-byoc",
            "--region",
        ]
        assert argv[argv.index("--kubeconfig") + 1] == str(tmp_path / "kubeconfig")
        delete.assert_called_once_with(
            kubeconfig_path=tmp_path / "kubeconfig",
            gateway_name="inference-gateway",
        )

    def test_skips_deletion_when_the_gateway_is_disabled(self, tmp_path: Path) -> None:
        runner = MagicMock()

        with patch(
            "agilerl.arena.byoc.provisioning.providers.aws.provider.delete_eks_gateway"
        ) as delete:
            AWS.delete_gateway(
                _aws_spec(gateway_api={"enable": False}), tmp_path, runner
            )

        runner.run.assert_not_called()
        delete.assert_not_called()

    def test_raises_when_kubeconfig_cannot_be_read(self, tmp_path: Path) -> None:
        runner = MagicMock()
        runner.run.return_value = subprocess.CompletedProcess([], 1, "", "")

        with (
            patch(
                "agilerl.arena.byoc.provisioning.providers.aws.kubeconfig.resolve_aws_executable",
                return_value="aws",
            ),
            pytest.raises(
                click.ClickException, match="Could not retrieve kubeconfig from EKS"
            ),
        ):
            AWS.delete_gateway(_aws_spec(), tmp_path, runner)

    def test_destroy_deletes_the_gateway_before_terraform(self, tmp_path: Path) -> None:
        spec = _aws_spec()
        runner = MagicMock()
        work_dir = tmp_path / "work"
        events: list[str] = []

        def delete_gateway(
            _self: AwsProvider,
            got_spec: object,
            got_dir: Path,
            got_runner: object,
        ) -> None:
            events.append("gateway")
            assert got_spec is spec
            assert got_dir is work_dir
            assert got_runner is runner

        runner.destroy.side_effect = lambda *_args, **_kwargs: events.append(
            "terraform"
        )

        with (
            patch.object(AwsProvider, "prepare", return_value=(spec, work_dir)),
            patch.object(AwsProvider, "delete_gateway", delete_gateway),
            patch.object(AwsProvider, "finish_destroy"),
        ):
            ClusterProvisioner(runner=runner).destroy(
                spec, state_dir=tmp_path / "state"
            )

        assert events == ["gateway", "terraform"]

    def test_destroy_stops_when_the_gateway_cannot_be_deleted(
        self, tmp_path: Path
    ) -> None:
        spec = _aws_spec()
        runner = MagicMock()

        def delete_gateway(
            _self: AwsProvider,
            _spec: object,
            _work_dir: Path,
            _runner: object,
        ) -> None:
            msg = "Could not retrieve kubeconfig from EKS."
            raise click.ClickException(msg)

        with (
            patch.object(AwsProvider, "prepare", return_value=(spec, tmp_path)),
            patch.object(AwsProvider, "delete_gateway", delete_gateway),
            pytest.raises(
                click.ClickException, match="Could not retrieve kubeconfig from EKS"
            ),
        ):
            ClusterProvisioner(runner=runner).destroy(
                spec, state_dir=tmp_path / "state"
            )

        runner.destroy.assert_not_called()


class _AwsCli:
    def __init__(self, bucket_exists: bool = True) -> None:
        self.calls: list[list[str]] = []
        self.envs: list[dict[str, str] | None] = []
        self.bucket_exists = bucket_exists
        self.user_exists = False
        self.keys: list[str] = []
        self.key_rejections = 0

    def __call__(
        self, argv: list[str], **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        self.calls.append(argv)
        env = kwargs.get("env")
        self.envs.append(env if isinstance(env, dict) else None)
        args = argv[1:]
        if args[:2] == ["s3api", "head-bucket"]:
            if self.bucket_exists:
                return subprocess.CompletedProcess(argv, 0, "", "")
            return subprocess.CompletedProcess(argv, 254, "", "404 Not Found")
        if args[:2] == ["s3api", "create-bucket"]:
            self.bucket_exists = True
            return subprocess.CompletedProcess(argv, 0, "", "")
        if args[:2] == ["iam", "get-user"]:
            if self.user_exists:
                return subprocess.CompletedProcess(argv, 0, "{}", "")
            return subprocess.CompletedProcess(argv, 254, "", "NoSuchEntity")
        if args[:2] == ["iam", "create-user"]:
            self.user_exists = True
            return subprocess.CompletedProcess(argv, 0, "{}", "")
        if args[:2] == ["iam", "put-user-policy"]:
            return subprocess.CompletedProcess(argv, 0, "", "")
        if args[:2] == ["iam", "list-access-keys"]:
            metadata = [{"AccessKeyId": key, "Status": "Active"} for key in self.keys]
            body = json.dumps({"AccessKeyMetadata": metadata})
            return subprocess.CompletedProcess(argv, 0, body, "")
        if args[:2] == ["iam", "create-access-key"]:
            key_id = f"AKIA-{len(self.keys) + 1}"
            self.keys.append(key_id)
            body = json.dumps(
                {
                    "AccessKey": {
                        "AccessKeyId": key_id,
                        "SecretAccessKey": f"secret-{key_id}",
                    }
                }
            )
            return subprocess.CompletedProcess(argv, 0, body, "")
        if args[:2] == ["s3api", "delete-object"]:
            return subprocess.CompletedProcess(argv, 0, "", "")
        if args[:2] == ["sts", "get-caller-identity"]:
            if self.key_rejections:
                self.key_rejections -= 1
                return subprocess.CompletedProcess(
                    argv, 254, "", "InvalidClientTokenId"
                )
            return subprocess.CompletedProcess(argv, 0, "{}", "")
        return subprocess.CompletedProcess(argv, 1, "", f"unexpected {' '.join(args)}")


class TestTerraformStateUser:
    def test_creates_a_user_limited_to_the_state_bucket(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "agilerl.arena.byoc.provisioning.providers.aws.terraform.resolve_aws_executable",
            lambda: "aws",
        )
        aws = _AwsCli()

        credentials = ensure_terraform_state_bucket(
            _aws_spec(), tmp_path / "terraform", run=aws
        )

        assert credentials == TerraformStateCredentials("AKIA-1", "secret-AKIA-1")
        policy_call = next(argv for argv in aws.calls if "put-user-policy" in argv)
        document = json.loads(policy_call[policy_call.index("--policy-document") + 1])
        assert document["Statement"][0]["Resource"] == [
            "arn:aws:s3:::eks-tfstate",
            "arn:aws:s3:::eks-tfstate/*",
        ]
        saved = json.loads(
            (tmp_path / "tfstate-credentials.json").read_text(encoding="utf-8")
        )
        assert saved["aws_access_key_id"] == "AKIA-1"
        assert oct((tmp_path / "tfstate-credentials.json").stat().st_mode & 0o777) == (
            "0o600"
        )

    def test_reuses_the_credentials_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "agilerl.arena.byoc.provisioning.providers.aws.terraform.resolve_aws_executable",
            lambda: "aws",
        )
        (tmp_path / "tfstate-credentials.json").write_text(
            json.dumps(
                {
                    "aws_access_key_id": "AKIA-saved",
                    "aws_secret_access_key": "saved-secret",
                }
            ),
            encoding="utf-8",
        )
        aws = _AwsCli()

        credentials = ensure_terraform_state_bucket(
            _aws_spec(), tmp_path / "terraform", run=aws
        )

        assert credentials.access_key_id == "AKIA-saved"
        assert all("iam" not in argv for argv in aws.calls)

    def test_waits_until_aws_accepts_the_new_key(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange
        monkeypatch.setattr(
            "agilerl.arena.byoc.provisioning.providers.aws.terraform.resolve_aws_executable",
            lambda: "aws",
        )
        sleeps: list[float] = []
        monkeypatch.setattr(
            "agilerl.arena.byoc.provisioning.providers.aws.terraform.time.sleep",
            sleeps.append,
        )
        aws = _AwsCli()
        aws.key_rejections = 2

        # Act
        credentials = ensure_terraform_state_bucket(
            _aws_spec(), tmp_path / "terraform", run=aws
        )

        # Assert
        assert credentials.access_key_id == "AKIA-1"
        assert len(sleeps) == 2
        sts_envs = [
            env
            for argv, env in zip(aws.calls, aws.envs, strict=True)
            if "get-caller-identity" in argv
        ]
        assert sts_envs[-1] is not None
        assert sts_envs[-1]["AWS_ACCESS_KEY_ID"] == "AKIA-1"

    def test_fails_when_aws_never_accepts_the_new_key(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "agilerl.arena.byoc.provisioning.providers.aws.terraform.resolve_aws_executable",
            lambda: "aws",
        )
        monkeypatch.setattr(
            "agilerl.arena.byoc.provisioning.providers.aws.terraform.time.sleep",
            lambda _seconds: None,
        )
        aws = _AwsCli()
        aws.key_rejections = 1000

        with pytest.raises(click.ClickException, match="not accepted by AWS"):
            ensure_terraform_state_bucket(_aws_spec(), tmp_path / "terraform", run=aws)

        assert not (tmp_path / "tfstate-credentials.json").exists()

    def test_rejects_a_user_that_already_has_two_keys(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "agilerl.arena.byoc.provisioning.providers.aws.terraform.resolve_aws_executable",
            lambda: "aws",
        )
        aws = _AwsCli()
        aws.user_exists = True
        aws.keys = ["AKIA-old", "AKIA-older"]

        with pytest.raises(click.ClickException, match="two access keys"):
            ensure_terraform_state_bucket(_aws_spec(), tmp_path / "terraform", run=aws)

    def test_requires_the_bucket_when_create_is_false(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "agilerl.arena.byoc.provisioning.providers.aws.terraform.resolve_aws_executable",
            lambda: "aws",
        )

        with pytest.raises(click.ClickException, match="is required"):
            ensure_terraform_state_bucket(
                _aws_spec(),
                tmp_path / "terraform",
                run=_AwsCli(bucket_exists=False),
            )

    def test_deletes_state_with_the_tfstate_key(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "agilerl.arena.byoc.provisioning.providers.aws.terraform.resolve_aws_executable",
            lambda: "aws",
        )
        (tmp_path / "tfstate-credentials.json").write_text(
            json.dumps(
                {
                    "aws_access_key_id": "AKIA-saved",
                    "aws_secret_access_key": "saved-secret",
                }
            ),
            encoding="utf-8",
        )
        aws = _AwsCli()

        delete_cluster_terraform_state(_aws_spec(), tmp_path / "terraform", run=aws)

        deleted = [
            argv[argv.index("--key") + 1]
            for argv in aws.calls
            if "delete-object" in argv
        ]
        assert deleted == [
            "clusters/eks-byoc/terraform.tfstate",
            "clusters/eks-byoc/terraform.tfstate.tflock",
        ]
        assert aws.envs[0] is not None
        assert aws.envs[0]["AWS_ACCESS_KEY_ID"] == "AKIA-saved"

    def test_prepare_keeps_the_tfstate_key_out_of_terraform_env(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "agilerl.arena.byoc.provisioning.providers.aws.provider.resolve_aws_executable",
            lambda: "aws",
        )
        monkeypatch.setattr(
            "agilerl.arena.byoc.provisioning.providers.aws.provider.ensure_terraform_state_bucket",
            lambda spec, state_dir, create=False, prompt=None: (
                TerraformStateCredentials("AKIA-1", "secret-AKIA-1")
            ),
        )
        monkeypatch.setattr(
            "agilerl.arena.byoc.provisioning.terraform.shutil.which",
            lambda _name: "terraform",
        )
        runner = TerraformRunner()

        AWS.prepare(_aws_spec(), runner, tmp_path / "terraform")

        assert "AWS_ACCESS_KEY_ID" not in runner.extra_env
        assert "AWS_SECRET_ACCESS_KEY" not in runner.extra_env

    def test_backend_authenticates_with_the_tfstate_key(self, tmp_path: Path) -> None:
        # Arrange
        work_dir = tmp_path / "terraform"
        work_dir.mkdir()
        (tmp_path / "tfstate-credentials.json").write_text(
            json.dumps(
                {
                    "aws_access_key_id": "AKIA-saved",
                    "aws_secret_access_key": "saved-secret",
                }
            ),
            encoding="utf-8",
        )

        # Act
        AWS.write_backend(work_dir, _aws_spec())

        # Assert
        backend = (work_dir / "backend.tf").read_text(encoding="utf-8")
        assert 'profile      = "arena-tfstate"' in backend
        assert 'shared_credentials_files = ["tfstate-backend-credentials"]' in backend
        assert "AKIA-saved" not in backend
        profile = work_dir / "tfstate-backend-credentials"
        assert profile.read_text(encoding="utf-8") == (
            "[arena-tfstate]\n"
            "aws_access_key_id = AKIA-saved\n"
            "aws_secret_access_key = saved-secret\n"
        )
        assert oct(profile.stat().st_mode & 0o777) == "0o600"

    def test_backend_requires_the_tfstate_key(self, tmp_path: Path) -> None:
        work_dir = tmp_path / "terraform"
        work_dir.mkdir()

        with pytest.raises(click.ClickException, match="credentials are missing"):
            AWS.write_backend(work_dir, _aws_spec())
