# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Tests for the AWS cluster provider."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import click
import pytest
import yaml
from cryptography import x509
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.asymmetric.ec import ECDSA

from agilerl.arena.byoc.provisioning.providers import get_provider
from agilerl.arena.byoc.provisioning.providers.aws.access import (
    CLUSTER_ADMIN_POLICY_ARN,
    ensure_caller_cluster_admin,
)
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
    ensure_efs_storage_class,
)
from agilerl.arena.byoc.provisioning.providers.aws.gateway import (
    NLB_SERVICE_ANNOTATIONS,
    WEBHOOK_CERT_ANNOTATION,
    WEBHOOK_CERT_DIRECTORY,
    WEBHOOK_DNS_NAMES,
    WEBHOOK_HELM_VALUES,
    _webhook_certs_match,
    configure_eks_gateway,
    delete_eks_gateway,
    ensure_load_balancer_webhook_certs,
    install_aws_load_balancer_controller,
    write_load_balancer_webhook_certs,
)
from agilerl.arena.byoc.provisioning.providers.aws.gpu import (
    GPU_STARTUP_SCRIPT,
    GPU_STARTUP_TAINT_KEY,
    NVIDIA_DEVICE_PLUGIN_CHART,
    NVIDIA_DEVICE_PLUGIN_CHART_VERSION,
    NVIDIA_DEVICE_PLUGIN_RELEASE,
    NVIDIA_DEVICE_PLUGIN_REPO,
    build_gpu_startup_manifests,
    ensure_gpu_startup_taint,
    ensure_nvidia_device_plugin,
)
from agilerl.arena.byoc.provisioning.providers.aws.kubeconfig import (
    build_eks_kubeconfig_argv,
    resolve_aws_executable,
)
from agilerl.arena.byoc.provisioning.providers.aws.provider import (
    AWS,
    AWS_ROLE_IAM_RESOURCES,
    AWS_STORAGE_IAM_RESOURCES,
    AwsProvider,
    _aws_values,
    _optional_string,
    _worker_node_group_ids,
)
from agilerl.arena.byoc.provisioning.providers.aws.provisioner import (
    PROVISIONER_POLICY,
    PROVISIONER_PROFILE,
    PROVISIONER_USER,
    _ensure_profile_key,
    _ensure_user,
    _wait_for_profile,
    use_provisioner_profile,
)
from agilerl.arena.byoc.provisioning.providers.aws.spec import (
    AwsSpec,
    AwsWorkerNodeGroupSpec,
    resource_classes,
    spec_from_stored_row,
)
from agilerl.arena.byoc.provisioning.providers.aws.spec import (
    registration_payload as aws_registration_payload,
)
from agilerl.arena.byoc.provisioning.providers.aws.terraform import (
    TerraformStateCredentials,
    _create_access_key,
    credentials_path,
    delete_access_keys,
    delete_cluster_terraform_state,
    delete_terraform_state_bucket,
    delete_tfstate_user,
    ensure_terraform_state_bucket,
    iam_user_exists,
    read_terraform_state_credentials,
    run_aws,
    run_aws_json,
    run_aws_ok,
    terraform_state_bucket_exists,
    write_terraform_state_credentials,
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
        assert spec.values.placement == "in-cluster"
        assert spec.aws is not None
        assert spec.aws.system.instance_type == "t3.xlarge"
        assert spec.aws.shared_filesystem.provision is False


class TestAwsStoredRow:
    @pytest.mark.parametrize(
        ("stored", "expected"),
        [("byoc_ephemeral", "ephemeral"), ("byoc_in_cluster", "in-cluster")],
    )
    def test_keeps_placement(self, stored: str, expected: str) -> None:
        spec = spec_from_stored_row(
            {
                "name": "eks-byoc",
                "placement": stored,
                "byocProvider": {
                    "provider": "aws",
                    "config": {
                        "region": "eu-west-1",
                        "storage_bucket_name": "eks-data",
                        "storage_endpoint": "https://s3.eu-west-1.amazonaws.com",
                        "terraform_state": {"bucket": "eks-tfstate"},
                        "inference": {"domain": "inference.example.com"},
                        "arena": {
                            "min_node_count": 2,
                            "max_node_count": 10,
                            "instance_type": "t3.xlarge",
                        },
                    },
                },
            }
        )

        assert spec.values.placement == expected


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
        assert "--register-with-taints=nvidia.com/gpu=true:NoSchedule," in system
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

    def test_resource_class_uses_the_instance_size(self) -> None:
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
        assert compute["numCpus"] == 4
        assert compute["memoryBytes"] == "16 GiB"
        assert compute["numGpus"] == 1

    def test_resource_class_keeps_a_one_vcpu_instance(self) -> None:
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

        classes = resource_classes(spec, {"gpu": "ng-1"})

        compute = classes[0].metadata["computeResource"]
        assert compute["numCpus"] == 1
        assert compute["memoryBytes"] == "1 GiB"


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
        assert 'authentication_mode = "API_AND_CONFIG_MAP"' in text
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
                "agilerl.arena.byoc.provisioning.providers.aws.provider.ensure_caller_cluster_admin"
            ) as admin,
            patch(
                "agilerl.arena.byoc.provisioning.providers.aws.provider.build_eks_kubeconfig_argv",
                return_value=["aws", "eks", "update-kubeconfig"],
            ),
        ):
            AWS.install_cni(_aws_spec(), tmp_path, runner)

        admin.assert_called_once_with(cluster_name="eks-byoc", region="eu-west-1")

        runner.apply_targets.assert_called_once_with(
            tmp_path, ("aws_eks_cluster.arena",)
        )
        cilium.assert_called_once_with(
            kubeconfig_path=tmp_path / "kubeconfig",
            api_server_host="example.eks.amazonaws.com",
            wait=False,
        )


class TestCallerClusterAdmin:
    def test_grants_the_caller_cluster_admin(self) -> None:
        principal = "arn:aws:iam::761296822462:user/byoc-provisioner"
        calls: list[list[str]] = []

        def run(argv: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
            calls.append(argv)
            if "get-caller-identity" in argv:
                return subprocess.CompletedProcess(
                    argv, 0, json.dumps({"Arn": principal}), ""
                )
            return subprocess.CompletedProcess(argv, 0, "", "")

        with patch(
            "agilerl.arena.byoc.provisioning.providers.aws.access.resolve_aws_executable",
            return_value="aws",
        ):
            ensure_caller_cluster_admin("eks-byoc-wei", "eu-west-1", run=run)

        assert calls == [
            ["aws", "sts", "get-caller-identity", "--output", "json"],
            [
                "aws",
                "eks",
                "create-access-entry",
                "--cluster-name",
                "eks-byoc-wei",
                "--principal-arn",
                principal,
                "--region",
                "eu-west-1",
            ],
            [
                "aws",
                "eks",
                "associate-access-policy",
                "--cluster-name",
                "eks-byoc-wei",
                "--principal-arn",
                principal,
                "--policy-arn",
                CLUSTER_ADMIN_POLICY_ARN,
                "--access-scope",
                "type=cluster",
                "--region",
                "eu-west-1",
            ],
        ]

    def test_keeps_an_existing_access_entry(self) -> None:
        calls: list[list[str]] = []

        def run(argv: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
            calls.append(argv)
            if "get-caller-identity" in argv:
                return subprocess.CompletedProcess(
                    argv,
                    0,
                    json.dumps(
                        {"Arn": "arn:aws:iam::761296822462:user/byoc-provisioner"}
                    ),
                    "",
                )
            if "create-access-entry" in argv:
                return subprocess.CompletedProcess(
                    argv,
                    1,
                    "",
                    "ResourceInUseException: already in use",
                )
            return subprocess.CompletedProcess(argv, 0, "", "")

        with patch(
            "agilerl.arena.byoc.provisioning.providers.aws.access.resolve_aws_executable",
            return_value="aws",
        ):
            ensure_caller_cluster_admin("eks-byoc-wei", "eu-west-1", run=run)

        assert any("associate-access-policy" in argv for argv in calls)

    def test_rejects_the_account_root(self) -> None:
        def run(argv: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
            return subprocess.CompletedProcess(
                argv, 0, json.dumps({"Arn": "arn:aws:iam::761296822462:root"}), ""
            )

        with (
            patch(
                "agilerl.arena.byoc.provisioning.providers.aws.access.resolve_aws_executable",
                return_value="aws",
            ),
            pytest.raises(click.ClickException, match="root user"),
        ):
            ensure_caller_cluster_admin("eks-byoc-wei", "eu-west-1", run=run)


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
                "--set-json",
                "tolerations="
                + json.dumps(
                    [
                        {"key": "CriticalAddonsOnly", "operator": "Exists"},
                        {
                            "key": "nvidia.com/gpu",
                            "operator": "Exists",
                            "effect": "NoSchedule",
                        },
                        {
                            "key": GPU_STARTUP_TAINT_KEY,
                            "operator": "Exists",
                            "effect": "NoSchedule",
                        },
                    ],
                    separators=(",", ":"),
                ),
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


class TestLoadBalancerWebhookCerts:
    def test_serving_cert_verifies_against_the_ca(self, tmp_path: Path) -> None:
        ca_path, cert_path, key_path = write_load_balancer_webhook_certs(tmp_path)
        ca_cert = x509.load_pem_x509_certificate(ca_path.read_bytes())
        serving = x509.load_pem_x509_certificate(cert_path.read_bytes())

        ca_cert.public_key().verify(
            serving.signature,
            serving.tbs_certificate_bytes,
            ECDSA(hashes.SHA256()),
        )
        names = serving.extensions.get_extension_for_class(
            x509.SubjectAlternativeName
        ).value.get_values_for_type(x509.DNSName)

        assert names == list(WEBHOOK_DNS_NAMES)
        assert isinstance(ca_cert.public_key(), ec.EllipticCurvePublicKey)
        if os.name != "nt":
            assert key_path.stat().st_mode & 0o777 == 0o600

    def test_reuses_certs_that_verify(self, tmp_path: Path) -> None:
        first = ensure_load_balancer_webhook_certs(tmp_path)
        ca = first[0].read_bytes()

        second = ensure_load_balancer_webhook_certs(tmp_path)

        assert second[0].read_bytes() == ca

    def test_replaces_certs_that_do_not_verify(self, tmp_path: Path) -> None:
        ensure_load_balancer_webhook_certs(tmp_path)
        ca_path = tmp_path / "ca.crt"
        original = ca_path.read_bytes()
        ca_path.write_bytes(b"not a certificate\n")

        ensure_load_balancer_webhook_certs(tmp_path)

        assert ca_path.read_bytes() != original
        assert b"BEGIN CERTIFICATE" in ca_path.read_bytes()


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
        values = yaml.safe_load(
            (tmp_path / WEBHOOK_CERT_DIRECTORY / WEBHOOK_HELM_VALUES).read_text()
        )
        assert "ALBGatewayAPI" not in helm
        assert "vpcId=vpc-123" in helm
        assert "webhookTLS.caCert=" not in helm
        assert str(tmp_path / WEBHOOK_CERT_DIRECTORY / WEBHOOK_HELM_VALUES) in helm
        assert values["webhookTLS"]["key"].startswith("-----BEGIN")
        assert values["podAnnotations"][WEBHOOK_CERT_ANNOTATION]
        assert values["keepTLSSecret"] is True
        ca_cert = x509.load_pem_x509_certificate(
            values["webhookTLS"]["caCert"].encode()
        )
        serving = x509.load_pem_x509_certificate(values["webhookTLS"]["cert"].encode())
        ca_cert.public_key().verify(
            serving.signature,
            serving.tbs_certificate_bytes,
            ECDSA(hashes.SHA256()),
        )
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
        runner.state_list.return_value = ("aws_eks_cluster.arena",)
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

    def test_skips_deletion_when_the_cluster_is_already_destroyed(
        self, tmp_path: Path
    ) -> None:
        runner = MagicMock()
        runner.state_list.return_value = ("aws_iam_role.cluster",)

        with patch(
            "agilerl.arena.byoc.provisioning.providers.aws.provider.delete_eks_gateway"
        ) as delete:
            AWS.delete_gateway(_aws_spec(), tmp_path, runner)

        runner.run.assert_not_called()
        delete.assert_not_called()

    def test_raises_when_kubeconfig_cannot_be_read(self, tmp_path: Path) -> None:
        runner = MagicMock()
        runner.state_list.return_value = ("aws_eks_cluster.arena",)
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

        destroy_calls: list[dict[str, object]] = []

        def record_destroy(*_args: object, **kwargs: object) -> None:
            events.append("terraform")
            destroy_calls.append(kwargs)

        runner.destroy.side_effect = record_destroy

        with (
            patch.object(AwsProvider, "prepare", return_value=(spec, work_dir)),
            patch.object(AwsProvider, "delete_gateway", delete_gateway),
            patch.object(AwsProvider, "finish_destroy"),
        ):
            ClusterProvisioner(runner=runner).destroy(
                spec, state_dir=tmp_path / "state"
            )

        assert events == ["gateway", "terraform", "terraform"]
        excluded = destroy_calls[0]["exclude"]
        assert isinstance(excluded, tuple)
        assert "aws_iam_role.cluster" in excluded
        assert "aws_s3_bucket.arena_data" in excluded
        assert destroy_calls[1]["only"] == AWS_ROLE_IAM_RESOURCES

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
        if os.name != "nt":
            assert (
                oct((tmp_path / "tfstate-credentials.json").stat().st_mode & 0o777)
                == "0o600"
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
        monkeypatch.setenv("AWS_PROFILE", PROVISIONER_PROFILE)
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
        if os.name != "nt":
            assert oct(profile.stat().st_mode & 0o777) == "0o600"

    def test_backend_requires_the_tfstate_key(self, tmp_path: Path) -> None:
        work_dir = tmp_path / "terraform"
        work_dir.mkdir()

        with pytest.raises(click.ClickException, match="credentials are missing"):
            AWS.write_backend(work_dir, _aws_spec())


PROVISIONER_ARN = f"arn:aws:iam::123456789012:user/{PROVISIONER_USER}"


class _ProvisionerAwsCli:
    """AWS CLI fake for the provisioner user and its ``byoc`` profile."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []
        self.caller_arn = "arn:aws:iam::123456789012:user/human"
        self.user_exists = True
        self.keys: list[str] = []
        self.profile: dict[str, str] = {}

    def __call__(
        self, argv: list[str], **_kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        self.calls.append(argv)
        args = argv[1:]
        user = {"User": {"UserName": PROVISIONER_USER, "Arn": PROVISIONER_ARN}}
        if args[:2] == ["sts", "get-caller-identity"]:
            return subprocess.CompletedProcess(
                argv, 0, json.dumps({"Arn": self.caller_arn}), ""
            )
        if args[:2] == ["iam", "get-user"]:
            if self.user_exists:
                return subprocess.CompletedProcess(argv, 0, json.dumps(user), "")
            return subprocess.CompletedProcess(argv, 254, "", "NoSuchEntity")
        if args[:2] == ["iam", "create-user"]:
            self.user_exists = True
            return subprocess.CompletedProcess(argv, 0, json.dumps(user), "")
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
                {"AccessKey": {"AccessKeyId": key_id, "SecretAccessKey": "secret"}}
            )
            return subprocess.CompletedProcess(argv, 0, body, "")
        if args[:2] == ["configure", "get"]:
            value = self.profile.get(args[2])
            if value is None:
                return subprocess.CompletedProcess(argv, 1, "", "")
            return subprocess.CompletedProcess(argv, 0, f"{value}\n", "")
        if args[:2] == ["configure", "set"]:
            self.profile[args[2]] = args[3]
            return subprocess.CompletedProcess(argv, 0, "", "")
        return subprocess.CompletedProcess(argv, 1, "", f"unexpected {' '.join(args)}")


class TestUseProvisionerProfile:
    @pytest.fixture(autouse=True)
    def _aws_cli(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "agilerl.arena.byoc.provisioning.providers.aws.terraform.resolve_aws_executable",
            lambda: "aws",
        )
        monkeypatch.delenv("AWS_PROFILE", raising=False)
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIA-human")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "human-secret")

    def test_adopts_existing_user_and_profile_key(self) -> None:
        # Arrange
        aws = _ProvisionerAwsCli()
        aws.keys = ["AKIA-existing"]
        aws.profile = {"aws_access_key_id": "AKIA-existing"}

        # Act
        use_provisioner_profile("eu-west-1", run=aws)

        # Assert
        assert os.environ["AWS_PROFILE"] == PROVISIONER_PROFILE
        assert "AWS_ACCESS_KEY_ID" not in os.environ
        assert "AWS_SECRET_ACCESS_KEY" not in os.environ
        assert not any("create-user" in argv for argv in aws.calls)
        assert not any("create-access-key" in argv for argv in aws.calls)
        assert aws.profile == {"aws_access_key_id": "AKIA-existing"}
        policy_call = next(argv for argv in aws.calls if "put-user-policy" in argv)
        assert policy_call[policy_call.index("--policy-name") + 1] == (
            PROVISIONER_POLICY
        )
        document = json.loads(policy_call[policy_call.index("--policy-document") + 1])
        deny = next(s for s in document["Statement"] if s["Effect"] == "Deny")
        assert deny["Resource"] == PROVISIONER_ARN
        assert "iam:CreateAccessKey" in deny["Action"]

    def test_creates_user_and_writes_profile_key(self) -> None:
        aws = _ProvisionerAwsCli()
        aws.user_exists = False

        use_provisioner_profile("eu-west-1", run=aws)

        assert aws.keys == ["AKIA-1"]
        assert aws.profile == {
            "aws_access_key_id": "AKIA-1",
            "aws_secret_access_key": "secret",
            "region": "eu-west-1",
        }
        assert os.environ["AWS_PROFILE"] == PROVISIONER_PROFILE

    def test_skips_setup_when_already_the_provisioner(self) -> None:
        aws = _ProvisionerAwsCli()
        aws.caller_arn = PROVISIONER_ARN

        use_provisioner_profile("eu-west-1", run=aws)

        assert [argv[1:3] for argv in aws.calls] == [["sts", "get-caller-identity"]]
        assert os.environ["AWS_ACCESS_KEY_ID"] == "AKIA-human"

    def test_skips_setup_when_profile_is_selected(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("AWS_PROFILE", PROVISIONER_PROFILE)
        aws = _ProvisionerAwsCli()

        use_provisioner_profile("eu-west-1", run=aws)

        assert aws.calls == []

    def test_rejects_two_keys_outside_the_profile(self) -> None:
        aws = _ProvisionerAwsCli()
        aws.keys = ["AKIA-old", "AKIA-older"]

        with pytest.raises(click.ClickException, match="two access keys"):
            use_provisioner_profile("eu-west-1", run=aws)

        assert os.environ.get("AWS_PROFILE") != PROVISIONER_PROFILE

    def test_restore_puts_back_the_caller_keys(self) -> None:
        aws = _ProvisionerAwsCli()
        aws.keys = ["AKIA-existing"]
        aws.profile = {"aws_access_key_id": "AKIA-existing"}

        restore = use_provisioner_profile("eu-west-1", run=aws)
        assert os.environ["AWS_PROFILE"] == PROVISIONER_PROFILE
        assert "AWS_ACCESS_KEY_ID" not in os.environ

        restore()

        assert "AWS_PROFILE" not in os.environ
        assert os.environ["AWS_ACCESS_KEY_ID"] == "AKIA-human"
        assert os.environ["AWS_SECRET_ACCESS_KEY"] == "human-secret"

    def test_restore_drops_the_byoc_profile(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("AWS_PROFILE", PROVISIONER_PROFILE)
        monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
        monkeypatch.delenv("AWS_SECRET_ACCESS_KEY", raising=False)
        monkeypatch.delenv("AWS_SESSION_TOKEN", raising=False)
        aws = _ProvisionerAwsCli()

        restore = use_provisioner_profile("eu-west-1", run=aws)
        restore()

        assert "AWS_PROFILE" not in os.environ
        assert "AWS_ACCESS_KEY_ID" not in os.environ
        assert aws.calls == []

    def test_restore_puts_back_a_named_profile(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("AWS_PROFILE", "admin")
        aws = _ProvisionerAwsCli()
        aws.keys = ["AKIA-existing"]
        aws.profile = {"aws_access_key_id": "AKIA-existing"}

        restore = use_provisioner_profile("eu-west-1", run=aws)
        assert os.environ["AWS_PROFILE"] == PROVISIONER_PROFILE

        restore()

        assert os.environ["AWS_PROFILE"] == "admin"
        assert os.environ["AWS_ACCESS_KEY_ID"] == "AKIA-human"


class TestDestroyDeferredIam:
    def test_byoc_pass_keeps_iam_and_caller_pass_deletes_roles(self) -> None:
        spec = _aws_spec()

        excluded = AWS.destroy_exclusions(spec, delete_storage=False)

        assert "aws_s3_bucket.arena_data" in excluded
        assert "aws_iam_user.storage" in excluded
        assert "aws_iam_role.cluster" in excluded

    def test_delete_storage_defers_the_storage_user(self) -> None:
        spec = _aws_spec()

        excluded = AWS.destroy_exclusions(spec, delete_storage=True)

        assert "aws_s3_bucket.arena_data" not in excluded
        assert "aws_iam_user.storage" in excluded
        assert set(AWS_STORAGE_IAM_RESOURCES).issubset(excluded)

    def test_second_pass_runs_as_the_caller(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(
            "agilerl.arena.byoc.provisioning.providers.aws.terraform.resolve_aws_executable",
            lambda: "aws",
        )
        monkeypatch.delenv("AWS_PROFILE", raising=False)
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIA-human")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "human-secret")
        aws = _ProvisionerAwsCli()
        aws.keys = ["AKIA-existing"]
        aws.profile = {"aws_access_key_id": "AKIA-existing"}
        provider = AwsProvider()
        provider.restore_caller_profile = use_provisioner_profile("eu-west-1", run=aws)
        iam = _DestroyAwsCli()
        monkeypatch.setattr(
            "agilerl.arena.byoc.provisioning.providers.aws.provider.subprocess.run", iam
        )
        seen: list[tuple[str | None, tuple[str, ...]]] = []

        class Runner:
            def destroy(
                self,
                _work_dir: Path,
                exclude: tuple[str, ...] = (),
                only: tuple[str, ...] = (),
            ) -> None:
                seen.append((os.environ.get("AWS_PROFILE"), only))
                assert os.environ["AWS_ACCESS_KEY_ID"] == "AKIA-human"
                assert exclude == ()
                assert [
                    "iam",
                    "delete-access-key",
                    "--user-name",
                    "eks-byoc-storage",
                    "--access-key-id",
                    "AKIA-state",
                ] in [argv[1:] for argv in iam.calls]

        provider.destroy_deferred(_aws_spec(), tmp_path, Runner(), delete_storage=True)

        assert seen == [
            (
                None,
                (*AWS_ROLE_IAM_RESOURCES, *AWS_STORAGE_IAM_RESOURCES),
            )
        ]

    def test_keeps_storage_keys_when_storage_is_kept(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        iam = _DestroyAwsCli()
        monkeypatch.setattr(
            "agilerl.arena.byoc.provisioning.providers.aws.provider.subprocess.run", iam
        )
        runner = MagicMock()

        AwsProvider().destroy_deferred(
            _aws_spec(), tmp_path, runner, delete_storage=False
        )

        assert iam.calls == []
        runner.destroy.assert_called_once_with(tmp_path, only=AWS_ROLE_IAM_RESOURCES)


class _DestroyAwsCli:
    """AWS CLI fake for deleting the state user and its bucket."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []
        self.user_exists = True
        self.bucket_exists = True

    def __call__(
        self, argv: list[str], **_kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        self.calls.append(argv)
        args = argv[1:]
        if args[:2] == ["iam", "get-user"]:
            if self.user_exists:
                return subprocess.CompletedProcess(argv, 0, "{}", "")
            return subprocess.CompletedProcess(argv, 254, "", "NoSuchEntity")
        if args[:2] == ["iam", "list-access-keys"]:
            body = json.dumps({"AccessKeyMetadata": [{"AccessKeyId": "AKIA-state"}]})
            return subprocess.CompletedProcess(argv, 0, body, "")
        if args[:2] in (
            ["iam", "delete-access-key"],
            ["iam", "delete-user-policy"],
            ["iam", "delete-user"],
        ):
            return subprocess.CompletedProcess(argv, 0, "", "")
        if args[:2] == ["s3", "rb"]:
            if self.bucket_exists:
                return subprocess.CompletedProcess(argv, 0, "", "")
            return subprocess.CompletedProcess(argv, 1, "", "NoSuchBucket")
        return subprocess.CompletedProcess(argv, 1, "", f"unexpected {' '.join(args)}")


class TestDeleteAllAwsState:
    def test_deletes_the_state_user_before_the_bucket(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange
        monkeypatch.setattr(
            "agilerl.arena.byoc.provisioning.providers.aws.terraform.resolve_aws_executable",
            lambda: "aws",
        )
        aws = _DestroyAwsCli()

        # Act
        delete_tfstate_user(_aws_spec(), run=aws)
        delete_terraform_state_bucket(_aws_spec(), run=aws)

        # Assert
        commands = [argv[1:3] for argv in aws.calls]
        assert commands.index(["iam", "delete-user"]) < commands.index(["s3", "rb"])
        assert ["iam", "delete-access-key"] in commands
        bucket_call = next(argv for argv in aws.calls if argv[1:3] == ["s3", "rb"])
        assert "s3://eks-tfstate" in bucket_call
        assert "--force" in bucket_call

    def test_skips_a_missing_user_and_bucket(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "agilerl.arena.byoc.provisioning.providers.aws.terraform.resolve_aws_executable",
            lambda: "aws",
        )
        aws = _DestroyAwsCli()
        aws.user_exists = False
        aws.bucket_exists = False

        delete_tfstate_user(_aws_spec(), run=aws)
        delete_terraform_state_bucket(_aws_spec(), run=aws)

        assert not any(argv[1:3] == ["iam", "delete-user"] for argv in aws.calls)
        assert any(argv[1:3] == ["s3", "rb"] for argv in aws.calls)

    def test_finish_destroy_deletes_state_iam_only_with_delete_all(
        self, tmp_path: Path
    ) -> None:
        seen: list[str] = []
        with (
            patch(
                "agilerl.arena.byoc.provisioning.providers.aws.provider.delete_tfstate_user",
                side_effect=lambda _spec: seen.append("user"),
            ),
            patch(
                "agilerl.arena.byoc.provisioning.providers.aws.provider.delete_terraform_state_bucket",
                side_effect=lambda _spec: seen.append("bucket"),
            ),
        ):
            AWS.finish_destroy(
                _aws_spec(),
                delete_compute_project=False,
                delete_storage_project=False,
                delete_all=False,
                state_dir=tmp_path / "terraform",
            )
            AWS.finish_destroy(
                _aws_spec(),
                delete_compute_project=False,
                delete_storage_project=False,
                delete_all=True,
                state_dir=tmp_path / "terraform",
            )

        assert seen == ["user", "bucket"]

    def test_finish_destroy_deletes_stale_state_credentials(
        self, tmp_path: Path
    ) -> None:
        credentials = tmp_path / "tfstate-credentials.json"
        credentials.write_text("{}", encoding="utf-8")
        with (
            patch(
                "agilerl.arena.byoc.provisioning.providers.aws.provider.delete_tfstate_user"
            ),
            patch(
                "agilerl.arena.byoc.provisioning.providers.aws.provider.delete_terraform_state_bucket"
            ),
        ):
            AWS.finish_destroy(
                _aws_spec(),
                delete_compute_project=False,
                delete_storage_project=False,
                delete_all=True,
                state_dir=tmp_path / "terraform",
            )

        assert not credentials.is_file()


def _nebius_cluster_spec() -> ClusterSpec:
    return ClusterSpec.model_validate(
        {
            "provider": "nebius",
            "name": "arena-nebius",
            "terraform_state": {"bucket": "tfstate"},
            "arena": {
                "storage": {
                    "bucket": "data",
                    "endpoint": "https://storage.example.com",
                },
                "inference": {"domain": "inference.example.com"},
            },
            "nebius": {
                "tenant_id": "tenant",
                "project_id": "project",
                "storage_project_id": "storage-project",
                "object_storage": {"size_gib": 1},
            },
        }
    )


class TestAwsAccessErrors:
    def test_create_access_entry_failure(self) -> None:
        def run(argv: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
            if "get-caller-identity" in argv:
                return subprocess.CompletedProcess(
                    argv,
                    0,
                    json.dumps({"Arn": "arn:aws:iam::1:user/byoc"}),
                    "",
                )
            if "create-access-entry" in argv:
                return subprocess.CompletedProcess(argv, 1, "", "AccessDenied")
            return subprocess.CompletedProcess(argv, 0, "", "")

        with (
            patch(
                "agilerl.arena.byoc.provisioning.providers.aws.access.resolve_aws_executable",
                return_value="aws",
            ),
            pytest.raises(click.ClickException, match="create-access-entry failed"),
        ):
            ensure_caller_cluster_admin("eks", "eu-west-1", run=run)

    def test_caller_identity_must_be_json(self) -> None:
        def run(argv: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
            return subprocess.CompletedProcess(argv, 0, "not-json", "")

        with (
            patch(
                "agilerl.arena.byoc.provisioning.providers.aws.access.resolve_aws_executable",
                return_value="aws",
            ),
            pytest.raises(click.ClickException, match="did not return JSON"),
        ):
            ensure_caller_cluster_admin("eks", "eu-west-1", run=run)

    def test_caller_identity_must_include_arn(self) -> None:
        def run(argv: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
            return subprocess.CompletedProcess(
                argv, 0, json.dumps({"Account": "1"}), ""
            )

        with (
            patch(
                "agilerl.arena.byoc.provisioning.providers.aws.access.resolve_aws_executable",
                return_value="aws",
            ),
            pytest.raises(click.ClickException, match="did not return an ARN"),
        ):
            ensure_caller_cluster_admin("eks", "eu-west-1", run=run)

    def test_missing_aws_cli(self) -> None:
        with (
            patch(
                "agilerl.arena.byoc.provisioning.providers.aws.access.resolve_aws_executable",
                return_value=None,
            ),
            pytest.raises(click.ClickException, match="AWS CLI not found"),
        ):
            ensure_caller_cluster_admin("eks", "eu-west-1", run=MagicMock())

    def test_checked_command_failure(self) -> None:
        def run(argv: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
            if "get-caller-identity" in argv:
                return subprocess.CompletedProcess(
                    argv,
                    0,
                    json.dumps({"Arn": "arn:aws:iam::1:user/byoc"}),
                    "",
                )
            if "create-access-entry" in argv:
                return subprocess.CompletedProcess(argv, 0, "", "")
            return subprocess.CompletedProcess(argv, 1, "", "denied")

        with (
            patch(
                "agilerl.arena.byoc.provisioning.providers.aws.access.resolve_aws_executable",
                return_value="aws",
            ),
            pytest.raises(click.ClickException, match="associate-access-policy"),
        ):
            ensure_caller_cluster_admin("eks", "eu-west-1", run=run)


class TestAwsCiliumErrors:
    def test_requires_helm(self, tmp_path: Path) -> None:
        with (
            patch(
                "agilerl.arena.byoc.provisioning.providers.aws.cilium.helm_available",
                return_value=False,
            ),
            pytest.raises(click.ClickException, match="helm not found"),
        ):
            ensure_cilium(tmp_path / "kubeconfig", "host.example")

    def test_describe_addon_failure(self) -> None:
        def run(argv: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
            return subprocess.CompletedProcess(argv, 1, "", "AccessDenied")

        with pytest.raises(click.ClickException, match="describe-addon failed"):
            ensure_coredns_addon("eks", "eu-west-1", run=run)

    def test_create_addon_failure(self) -> None:
        def run(argv: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
            if "describe-addon" in argv:
                return subprocess.CompletedProcess(
                    argv, 1, "", "ResourceNotFoundException"
                )
            return subprocess.CompletedProcess(argv, 1, "", "create failed")

        with pytest.raises(click.ClickException, match="create-addon"):
            ensure_coredns_addon("eks", "eu-west-1", run=run)


class TestAwsFilesystemAndGpu:
    def test_ensure_efs_storage_class_applies_the_manifest(
        self, tmp_path: Path
    ) -> None:
        with (
            patch(
                "agilerl.arena.byoc.provisioning.providers.aws.filesystem.require_kubectl"
            ) as kubectl,
            patch(
                "agilerl.arena.byoc.provisioning.providers.aws.filesystem.apply_manifests"
            ) as apply,
        ):
            ensure_efs_storage_class(tmp_path / "kubeconfig", "fs-123", run=MagicMock())

        kubectl.assert_called_once()
        apply.assert_called_once()

    def test_ensure_gpu_startup_taint_applies_manifests(self, tmp_path: Path) -> None:
        with (
            patch(
                "agilerl.arena.byoc.provisioning.providers.aws.gpu.require_kubectl"
            ) as kubectl,
            patch(
                "agilerl.arena.byoc.provisioning.providers.aws.gpu.apply_manifests"
            ) as apply,
        ):
            ensure_gpu_startup_taint(tmp_path / "kubeconfig", run=MagicMock())

        kubectl.assert_called_once()
        apply.assert_called_once()


class TestAwsKubeconfig:
    def test_resolve_aws_executable_uses_path(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "agilerl.arena.byoc.provisioning.providers.aws.kubeconfig.shutil.which",
            lambda name: "/usr/bin/aws" if name == "aws" else None,
        )

        assert resolve_aws_executable() == "/usr/bin/aws"

    def test_build_eks_kubeconfig_requires_aws_cli(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(
            "agilerl.arena.byoc.provisioning.providers.aws.kubeconfig.shutil.which",
            lambda _name: None,
        )

        with pytest.raises(FileNotFoundError, match="AWS CLI not found"):
            build_eks_kubeconfig_argv("eks", "eu-west-1", tmp_path / "kubeconfig")


class TestAwsGatewayErrors:
    def test_install_requires_helm(self, tmp_path: Path) -> None:
        with (
            patch(
                "agilerl.arena.byoc.provisioning.providers.aws.gateway.helm_available",
                return_value=False,
            ),
            pytest.raises(click.ClickException, match="helm not found"),
        ):
            install_aws_load_balancer_controller(
                "eks",
                "eu-west-1",
                "vpc-1",
                tmp_path,
                env={},
                run=MagicMock(),
            )

    def test_webhook_certs_reject_a_non_ec_private_key(self, tmp_path: Path) -> None:
        write_load_balancer_webhook_certs(tmp_path)
        ca_path = tmp_path / "ca.crt"
        cert_path = tmp_path / "tls.crt"
        key_path = tmp_path / "tls.key"

        class NotEC:
            pass

        with patch(
            "agilerl.arena.byoc.provisioning.providers.aws.gateway.ec.EllipticCurvePrivateKey",
            NotEC,
        ):
            assert _webhook_certs_match(ca_path, cert_path, key_path) is False

    def test_webhook_certs_reject_a_non_ec_serving_key(self, tmp_path: Path) -> None:
        write_load_balancer_webhook_certs(tmp_path)
        ca_path = tmp_path / "ca.crt"
        cert_path = tmp_path / "tls.crt"
        key_path = tmp_path / "tls.key"

        class NotEC:
            pass

        with patch(
            "agilerl.arena.byoc.provisioning.providers.aws.gateway.ec.EllipticCurvePublicKey",
            NotEC,
        ):
            assert _webhook_certs_match(ca_path, cert_path, key_path) is False


class TestAwsSpecValidators:
    def test_system_min_cannot_exceed_max(self) -> None:
        with pytest.raises(ValueError, match="min_node_count must be less"):
            _aws_spec(system={"min_node_count": 5, "max_node_count": 1})

    def test_worker_name_cannot_be_blank(self) -> None:
        with pytest.raises(ValueError, match="non-empty string"):
            _aws_spec(workers=[{"name": "  ", "instance_type": "g6.xlarge"}])

    def test_blank_gpu_type_is_unset(self) -> None:
        spec = _aws_spec(
            workers=[{"name": "cpu", "instance_type": "m6i.large", "gpu_type": "  "}]
        )
        assert spec.aws is not None
        assert spec.aws.workers[0].gpu_type is None

    def test_null_gpu_type_stays_unset(self) -> None:
        spec = _aws_spec(
            workers=[{"name": "cpu", "instance_type": "m6i.large", "gpu_type": None}]
        )
        assert spec.aws is not None
        assert spec.aws.workers[0].gpu_type is None
        assert AwsWorkerNodeGroupSpec.strip_gpu_type(None) is None

    def test_worker_min_cannot_exceed_max(self) -> None:
        with pytest.raises(ValueError, match="worker min_node_count"):
            _aws_spec(
                workers=[
                    {
                        "name": "gpu",
                        "instance_type": "g6.xlarge",
                        "min_node_count": 4,
                        "max_node_count": 1,
                    }
                ]
            )

    def test_gpu_worker_requires_size_fields(self) -> None:
        with pytest.raises(ValueError, match="requires num_cpus"):
            _aws_spec(
                workers=[{"name": "gpu", "instance_type": "g6.xlarge", "num_gpus": 1}]
            )

    def test_region_cannot_be_blank(self) -> None:
        with pytest.raises(ValueError, match="non-empty string"):
            _aws_spec(region="  ")

    def test_rejects_singular_worker_key(self) -> None:
        with pytest.raises(ValueError, match=r"aws\.workers must be a list"):
            ClusterSpec.model_validate(
                {
                    "provider": "aws",
                    "name": "eks",
                    "terraform_state": {"bucket": "tfstate"},
                    "arena": {
                        "storage": {
                            "bucket": "data",
                            "endpoint": "https://s3.eu-west-1.amazonaws.com",
                        },
                        "inference": {"domain": "inference.example.com"},
                    },
                    "aws": {
                        "region": "eu-west-1",
                        "object_storage": {"size_gib": 1},
                        "worker": {"instance_type": "g6.xlarge"},
                    },
                }
            )

    def test_rejects_duplicate_worker_names(self) -> None:
        with pytest.raises(ValueError, match="names must be unique"):
            _aws_spec(
                workers=[
                    {"name": "gpu", "instance_type": "g6.xlarge"},
                    {"name": "gpu", "instance_type": "g6.2xlarge"},
                ]
            )


class TestAwsRegistrationAndStoredRow:
    def test_registration_payload_includes_state_key_and_endpoint(self) -> None:
        spec = ClusterSpec.model_validate(
            {
                "provider": "aws",
                "name": "eks-byoc",
                "terraform_state": {
                    "bucket": "eks-tfstate",
                    "key": "custom.tfstate",
                    "endpoint": "https://s3.example",
                },
                "arena": {
                    "storage": {
                        "bucket": "eks-data",
                        "endpoint": "https://s3.eu-west-1.amazonaws.com",
                    },
                    "inference": {"domain": "inference.example.com"},
                },
                "aws": {
                    "region": "eu-west-1",
                    "object_storage": {"size_gib": 1024},
                },
            }
        )

        payload = aws_registration_payload(spec)

        assert payload["provider"] == "aws"
        assert payload["config"]["terraform_state"] == {
            "bucket": "eks-tfstate",
            "key": "custom.tfstate",
            "endpoint": "https://s3.example",
        }

    def test_registration_payload_rejects_a_nebius_spec(self) -> None:
        with pytest.raises(click.ClickException, match="is not an AWS spec"):
            aws_registration_payload(_nebius_cluster_spec())

    def test_resource_classes_skip_cpu_workers(self) -> None:
        spec = _aws_spec(
            workers=[{"name": "cpu", "instance_type": "m6i.large", "num_gpus": 0}]
        )

        assert resource_classes(spec, {"cpu": "ng-1"}) == []

    def test_resource_classes_require_an_aws_spec(self) -> None:
        with pytest.raises(click.ClickException, match="is not an AWS spec"):
            resource_classes(_nebius_cluster_spec(), {})

    def test_resource_classes_require_a_node_group_id(self) -> None:
        spec = _aws_spec(
            workers=[
                {
                    "name": "gpu",
                    "instance_type": "g6.xlarge",
                    "num_gpus": 1,
                    "num_cpus": 4,
                    "memory_gib": 16,
                    "gpu_type": "L4",
                    "vram_gib": 24,
                }
            ]
        )

        with pytest.raises(click.ClickException, match="missing node group id"):
            resource_classes(spec, {})

    def test_resource_classes_require_gpu_size_fields(self) -> None:
        spec = _aws_spec(
            workers=[
                {
                    "name": "gpu",
                    "instance_type": "g6.xlarge",
                    "num_gpus": 1,
                    "num_cpus": 4,
                    "memory_gib": 16,
                    "gpu_type": "L4",
                    "vram_gib": 24,
                }
            ]
        )
        assert spec.aws is not None
        spec.aws.workers[0].num_cpus = None

        with pytest.raises(click.ClickException, match="requires num_cpus"):
            resource_classes(spec, {"gpu": "ng-1"})

    def test_stored_row_rejects_a_blank_name(self) -> None:
        with pytest.raises(ValueError, match="missing a name"):
            spec_from_stored_row({"name": "  "})

    def test_stored_row_rejects_a_non_aws_provider(self) -> None:
        with pytest.raises(ValueError, match="no stored AWS settings"):
            spec_from_stored_row(
                {"name": "eks", "byocProvider": {"provider": "nebius"}}
            )

    def test_stored_row_rejects_a_non_mapping_config(self) -> None:
        with pytest.raises(ValueError, match="no stored AWS settings"):
            spec_from_stored_row(
                {"name": "eks", "byoc_provider": {"provider": "aws", "config": "nope"}}
            )

    def test_stored_row_rejects_missing_inference(self) -> None:
        with pytest.raises(ValueError, match="inference or state"):
            spec_from_stored_row(
                {
                    "name": "eks",
                    "byocProvider": {
                        "provider": "aws",
                        "config": {"terraform_state": {"bucket": "tfstate"}},
                    },
                }
            )

    def test_stored_row_rejects_missing_storage(self) -> None:
        with pytest.raises(ValueError, match="object storage"):
            spec_from_stored_row(
                {
                    "name": "eks",
                    "byocProvider": {
                        "provider": "aws",
                        "config": {
                            "terraform_state": {"bucket": "tfstate"},
                            "inference": {"domain": "inference.example.com"},
                            "storage_bucket_name": "  ",
                        },
                    },
                }
            )

    def test_stored_row_keeps_gateway_and_module(self) -> None:
        spec = spec_from_stored_row(
            {
                "name": "eks-byoc",
                "byocProvider": {
                    "provider": "aws",
                    "config": {
                        "region": "eu-west-1",
                        "storage_bucket_name": "eks-data",
                        "storage_endpoint": "https://s3.eu-west-1.amazonaws.com",
                        "terraform_state": {"bucket": "eks-tfstate"},
                        "inference": {"domain": "inference.example.com"},
                        "shared_filesystem": {"provision": True},
                        "gateway_api": {"enable": False, "gateway_name": "arena"},
                        "module": {"version": "1.2.3"},
                    },
                },
            }
        )

        assert spec.module_version == "1.2.3"
        assert spec.arena.gateway.enable is False
        assert spec.arena.gateway.name == "arena"
        assert spec.aws is not None
        assert spec.aws.shared_filesystem.provision is True


class TestAwsProviderMethods:
    def test_aws_values_rejects_a_nebius_spec(self) -> None:
        with pytest.raises(click.ClickException, match="is not an AWS spec"):
            _aws_values(_nebius_cluster_spec())

    def test_worker_node_group_ids_helpers(self) -> None:
        assert _worker_node_group_ids(None) == {}
        with pytest.raises(click.ClickException, match="string map"):
            _worker_node_group_ids(["ng-1"])
        with pytest.raises(click.ClickException, match="string map"):
            _worker_node_group_ids({"": "ng-1"})
        with pytest.raises(click.ClickException, match="string map"):
            _worker_node_group_ids({"gpu": "  "})
        assert _worker_node_group_ids({"gpu": "ng-1"}) == {"gpu": "ng-1"}
        assert _optional_string("  ") is None
        assert _optional_string("vpc-1") == "vpc-1"

    def test_validate_requires_aws_cli(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "agilerl.arena.byoc.provisioning.providers.aws.provider.resolve_aws_executable",
            lambda: None,
        )

        with pytest.raises(click.ClickException, match="AWS CLI not found"):
            AWS.validate(_aws_spec())

    def test_needs_stored_config_and_recover(self) -> None:
        spec = _aws_spec()

        assert AWS.needs_stored_config(spec) is False
        assert AWS.recover(spec, {}) is spec

    def test_provider_wrappers(self) -> None:
        spec = _aws_spec()

        assert AWS.registration_payload(spec)["provider"] == "aws"
        assert AWS.resource_classes(spec, {}) == []
        row = {
            "name": "eks-byoc",
            "byocProvider": {
                "provider": "aws",
                "config": {
                    "region": "eu-west-1",
                    "storage_bucket_name": "eks-data",
                    "storage_endpoint": "https://s3.eu-west-1.amazonaws.com",
                    "terraform_state": {"bucket": "eks-tfstate"},
                    "inference": {"domain": "inference.example.com"},
                },
            },
        }
        assert AWS.spec_from_stored_row(row).name == "eks-byoc"
        assert AWS.import_ids(spec) == {}

    def test_open_state_returns_none_without_a_bucket(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(
            "agilerl.arena.byoc.provisioning.providers.aws.provider.use_provisioner_profile",
            lambda _region: None,
        )
        monkeypatch.setattr(
            "agilerl.arena.byoc.provisioning.providers.aws.provider.terraform_state_bucket_exists",
            lambda _spec: False,
        )

        assert AWS.open_state(_aws_spec(), MagicMock(), tmp_path) is None

    def test_open_state_initializes_when_the_bucket_exists(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        runner = MagicMock()
        runner.prepare.return_value = tmp_path / "work"
        monkeypatch.setattr(
            "agilerl.arena.byoc.provisioning.providers.aws.provider.use_provisioner_profile",
            lambda _region: None,
        )
        monkeypatch.setattr(
            "agilerl.arena.byoc.provisioning.providers.aws.provider.terraform_state_bucket_exists",
            lambda _spec: True,
        )
        monkeypatch.setattr(
            "agilerl.arena.byoc.provisioning.providers.aws.provider.ensure_terraform_state_credentials",
            lambda *_args, **_kwargs: None,
        )

        assert AWS.open_state(_aws_spec(), runner, tmp_path) == tmp_path / "work"
        runner.initialize.assert_called_once()

    def test_write_backend_includes_a_custom_endpoint(self, tmp_path: Path) -> None:
        work_dir = tmp_path / "terraform"
        work_dir.mkdir()
        write_terraform_state_credentials(
            work_dir, TerraformStateCredentials("AKIA", "secret")
        )
        spec = ClusterSpec.model_validate(
            {
                "provider": "aws",
                "name": "eks-byoc",
                "terraform_state": {
                    "bucket": "eks-tfstate",
                    "endpoint": "https://s3.example",
                },
                "arena": {
                    "storage": {
                        "bucket": "eks-data",
                        "endpoint": "https://s3.eu-west-1.amazonaws.com",
                    },
                    "inference": {"domain": "inference.example.com"},
                },
                "aws": {
                    "region": "eu-west-1",
                    "object_storage": {"size_gib": 1024},
                },
            }
        )

        AWS.write_backend(work_dir, spec)

        backend = (work_dir / "backend.tf").read_text(encoding="utf-8")
        assert "https://s3.example" in backend
        assert "skip_credentials_validation = true" in backend

    def test_cluster_outputs_happy_path(self, tmp_path: Path) -> None:
        def run(argv: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
            (tmp_path / "kubeconfig").write_text("apiVersion: v1\n", encoding="utf-8")
            return subprocess.CompletedProcess(argv, 0, "", "")

        with patch(
            "agilerl.arena.byoc.provisioning.providers.aws.provider.resolve_aws_executable",
            return_value="aws",
        ):
            outputs = AWS.cluster_outputs(
                {
                    "cluster_name": "eks-byoc",
                    "context": "eks-byoc",
                    "cluster_id": "id",
                    "region": "eu-west-1",
                    "cluster_endpoint": "https://example.eks.amazonaws.com",
                    "storage_access_key_id": "AKIA",
                    "storage_secret_access_key": "secret",
                    "worker_node_group_ids": {"gpu": "ng-1"},
                    "vpc_id": "vpc-1",
                },
                tmp_path,
                run=run,
            )

        assert outputs.vpc_id == "vpc-1"
        assert outputs.worker_node_group_ids == {"gpu": "ng-1"}

    def test_cluster_outputs_missing_values(self, tmp_path: Path) -> None:
        with pytest.raises(click.ClickException, match="missing required values"):
            AWS.cluster_outputs({"cluster_name": "eks"}, tmp_path, run=MagicMock())

    def test_cluster_outputs_require_string_name_and_region(
        self, tmp_path: Path
    ) -> None:
        with pytest.raises(click.ClickException, match="must be strings"):
            AWS.cluster_outputs(
                {
                    "cluster_name": 1,
                    "context": "ctx",
                    "cluster_id": "id",
                    "region": "eu-west-1",
                    "cluster_endpoint": "https://example",
                    "storage_access_key_id": "AKIA",
                    "storage_secret_access_key": "secret",
                    "worker_node_group_ids": {},
                },
                tmp_path,
                run=MagicMock(),
            )

    def test_cluster_outputs_require_aws_cli(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(
            "agilerl.arena.byoc.provisioning.providers.aws.provider.resolve_aws_executable",
            lambda: None,
        )

        with pytest.raises(click.ClickException, match="AWS CLI not found"):
            AWS.cluster_outputs(
                {
                    "cluster_name": "eks-byoc",
                    "context": "ctx",
                    "cluster_id": "id",
                    "region": "eu-west-1",
                    "cluster_endpoint": "https://example",
                    "storage_access_key_id": "AKIA",
                    "storage_secret_access_key": "secret",
                    "worker_node_group_ids": {},
                },
                tmp_path,
                run=MagicMock(),
            )

    def test_cluster_outputs_wrap_missing_cli(self, tmp_path: Path) -> None:
        def run(argv: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
            msg = "aws"
            raise FileNotFoundError(msg)

        with (
            patch(
                "agilerl.arena.byoc.provisioning.providers.aws.provider.resolve_aws_executable",
                return_value="aws",
            ),
            pytest.raises(click.ClickException, match="AWS CLI not found"),
        ):
            AWS.cluster_outputs(
                {
                    "cluster_name": "eks-byoc",
                    "context": "ctx",
                    "cluster_id": "id",
                    "region": "eu-west-1",
                    "cluster_endpoint": "https://example",
                    "storage_access_key_id": "AKIA",
                    "storage_secret_access_key": "secret",
                    "worker_node_group_ids": {},
                },
                tmp_path,
                run=run,
            )

    def test_cluster_outputs_nonzero_exit(self, tmp_path: Path) -> None:
        def run(argv: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
            return subprocess.CompletedProcess(argv, 1, "", "denied")

        with (
            patch(
                "agilerl.arena.byoc.provisioning.providers.aws.provider.resolve_aws_executable",
                return_value="aws",
            ),
            pytest.raises(click.ClickException, match="Could not retrieve kubeconfig"),
        ):
            AWS.cluster_outputs(
                {
                    "cluster_name": "eks-byoc",
                    "context": "ctx",
                    "cluster_id": "id",
                    "region": "eu-west-1",
                    "cluster_endpoint": "https://example",
                    "storage_access_key_id": "AKIA",
                    "storage_secret_access_key": "secret",
                    "worker_node_group_ids": {},
                },
                tmp_path,
                run=run,
            )

    def test_cluster_outputs_require_a_kubeconfig_file(self, tmp_path: Path) -> None:
        def run(argv: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
            return subprocess.CompletedProcess(argv, 0, "", "")

        with (
            patch(
                "agilerl.arena.byoc.provisioning.providers.aws.provider.resolve_aws_executable",
                return_value="aws",
            ),
            pytest.raises(click.ClickException, match="did not write a kubeconfig"),
        ):
            AWS.cluster_outputs(
                {
                    "cluster_name": "eks-byoc",
                    "context": "ctx",
                    "cluster_id": "id",
                    "region": "eu-west-1",
                    "cluster_endpoint": "https://example",
                    "storage_access_key_id": "AKIA",
                    "storage_secret_access_key": "secret",
                    "worker_node_group_ids": {},
                },
                tmp_path,
                run=run,
            )

    def test_install_cni_requires_an_endpoint(self, tmp_path: Path) -> None:
        runner = MagicMock()
        runner.terraform_output.return_value = None

        with pytest.raises(click.ClickException, match="EKS API endpoint"):
            AWS.install_cni(_aws_spec(), tmp_path, runner)

    def test_install_cni_requires_kubeconfig(self, tmp_path: Path) -> None:
        runner = MagicMock()
        runner.terraform_output.return_value = "https://example.eks.amazonaws.com"
        runner.run.return_value = MagicMock(returncode=1)

        with (
            patch(
                "agilerl.arena.byoc.provisioning.providers.aws.provider.build_eks_kubeconfig_argv",
                return_value=["aws", "eks", "update-kubeconfig"],
            ),
            pytest.raises(click.ClickException, match="Could not retrieve kubeconfig"),
        ):
            AWS.install_cni(_aws_spec(), tmp_path, runner)

    def test_post_apply_requires_an_endpoint(self, tmp_path: Path) -> None:
        outputs = _outputs(tmp_path)
        object.__setattr__(outputs, "cluster_endpoint", None)

        with pytest.raises(click.ClickException, match="EKS API endpoint"):
            AWS.post_apply(_aws_spec(), outputs)

    def test_rejects_delete_storage_project(self) -> None:
        with pytest.raises(click.ClickException, match="no storage project"):
            AWS.assert_can_delete_storage_project(
                _aws_spec(), delete_storage_project=True
            )

    def test_delete_terraform_state_delegates(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        seen: list[Path] = []
        monkeypatch.setattr(
            "agilerl.arena.byoc.provisioning.providers.aws.provider.delete_cluster_terraform_state",
            lambda spec, state_dir: seen.append(state_dir),
        )

        AWS.delete_terraform_state(_aws_spec(), tmp_path)

        assert seen == [tmp_path]


class TestAwsTerraformHelpers:
    def test_run_aws_requires_the_cli(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "agilerl.arena.byoc.provisioning.providers.aws.terraform.resolve_aws_executable",
            lambda: None,
        )

        with pytest.raises(click.ClickException, match="AWS CLI not found"):
            run_aws(["sts", "get-caller-identity"], run=MagicMock())

    def test_bucket_lookup_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "agilerl.arena.byoc.provisioning.providers.aws.terraform.resolve_aws_executable",
            lambda: "aws",
        )

        def run(argv: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
            return subprocess.CompletedProcess(argv, 1, "", "AccessDenied")

        with pytest.raises(click.ClickException, match="Could not look up Terraform"):
            terraform_state_bucket_exists(_aws_spec(), run=run)

    def test_read_credentials_ignores_blank_keys(self, tmp_path: Path) -> None:
        path = credentials_path(tmp_path / "terraform")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"aws_access_key_id": "  ", "aws_secret_access_key": "secret"}),
            encoding="utf-8",
        )

        assert read_terraform_state_credentials(tmp_path / "terraform") is None

    def test_delete_access_keys_skips_bad_entries(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "agilerl.arena.byoc.provisioning.providers.aws.terraform.resolve_aws_executable",
            lambda: "aws",
        )
        calls: list[list[str]] = []

        def run(argv: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
            calls.append(argv)
            if "list-access-keys" in argv:
                return subprocess.CompletedProcess(
                    argv,
                    0,
                    json.dumps(
                        {
                            "AccessKeyMetadata": [
                                "skip",
                                {"AccessKeyId": "  "},
                                {"AccessKeyId": "AKIA-keep"},
                            ]
                        }
                    ),
                    "",
                )
            return subprocess.CompletedProcess(argv, 0, "", "")

        delete_access_keys("user", run=run)

        assert any("delete-access-key" in argv for argv in calls)

    def test_delete_access_keys_rejects_a_bad_list(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "agilerl.arena.byoc.provisioning.providers.aws.terraform.resolve_aws_executable",
            lambda: "aws",
        )

        def run(argv: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
            return subprocess.CompletedProcess(
                argv, 0, json.dumps({"AccessKeyMetadata": "nope"}), ""
            )

        with pytest.raises(click.ClickException, match="unexpected access key list"):
            delete_access_keys("user", run=run)

    def test_delete_tfstate_user_policy_failure(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "agilerl.arena.byoc.provisioning.providers.aws.terraform.resolve_aws_executable",
            lambda: "aws",
        )

        def run(argv: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
            if "get-user" in argv:
                return subprocess.CompletedProcess(argv, 0, "{}", "")
            if "list-access-keys" in argv:
                return subprocess.CompletedProcess(
                    argv, 0, json.dumps({"AccessKeyMetadata": []}), ""
                )
            if "delete-user-policy" in argv:
                return subprocess.CompletedProcess(argv, 1, "", "AccessDenied")
            return subprocess.CompletedProcess(argv, 0, "", "")

        with pytest.raises(click.ClickException, match="Could not delete the policy"):
            delete_tfstate_user(_aws_spec(), run=run)

    def test_delete_bucket_failure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "agilerl.arena.byoc.provisioning.providers.aws.terraform.resolve_aws_executable",
            lambda: "aws",
        )

        def run(argv: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
            return subprocess.CompletedProcess(argv, 1, "", "AccessDenied")

        with pytest.raises(click.ClickException, match="Could not delete Terraform"):
            delete_terraform_state_bucket(_aws_spec(), run=run)

    def test_delete_cluster_state_requires_credentials(self, tmp_path: Path) -> None:
        with pytest.raises(click.ClickException, match="credentials are missing"):
            delete_cluster_terraform_state(_aws_spec(), tmp_path / "terraform")

    def test_delete_cluster_state_object_failure(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(
            "agilerl.arena.byoc.provisioning.providers.aws.terraform.resolve_aws_executable",
            lambda: "aws",
        )
        write_terraform_state_credentials(
            tmp_path / "terraform", TerraformStateCredentials("AKIA", "secret")
        )

        def run(argv: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
            return subprocess.CompletedProcess(argv, 1, "", "AccessDenied")

        with pytest.raises(
            click.ClickException, match="Could not delete Terraform state"
        ):
            delete_cluster_terraform_state(_aws_spec(), tmp_path / "terraform", run=run)

    def test_create_bucket_uses_location_constraint(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(
            "agilerl.arena.byoc.provisioning.providers.aws.terraform.resolve_aws_executable",
            lambda: "aws",
        )
        calls: list[list[str]] = []

        def run(argv: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
            calls.append(argv)
            if "head-bucket" in argv:
                return subprocess.CompletedProcess(argv, 1, "", "404 Not Found")
            if "create-bucket" in argv:
                return subprocess.CompletedProcess(argv, 0, "", "")
            if "get-user" in argv:
                return subprocess.CompletedProcess(argv, 254, "", "NoSuchEntity")
            if "create-user" in argv:
                return subprocess.CompletedProcess(argv, 0, "{}", "")
            if "put-user-policy" in argv:
                return subprocess.CompletedProcess(argv, 0, "", "")
            if "list-access-keys" in argv:
                return subprocess.CompletedProcess(
                    argv, 0, json.dumps({"AccessKeyMetadata": []}), ""
                )
            if "create-access-key" in argv:
                return subprocess.CompletedProcess(
                    argv,
                    0,
                    json.dumps(
                        {
                            "AccessKey": {
                                "AccessKeyId": "AKIA",
                                "SecretAccessKey": "secret",
                            }
                        }
                    ),
                    "",
                )
            if "get-caller-identity" in argv:
                return subprocess.CompletedProcess(argv, 0, "{}", "")
            return subprocess.CompletedProcess(argv, 0, "", "")

        ensure_terraform_state_bucket(
            _aws_spec(), tmp_path / "terraform", create=True, run=run
        )

        create = next(argv for argv in calls if "create-bucket" in argv)
        assert any("LocationConstraint=eu-west-1" in part for part in create)

    def test_iam_user_lookup_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "agilerl.arena.byoc.provisioning.providers.aws.terraform.resolve_aws_executable",
            lambda: "aws",
        )

        def run(argv: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
            return subprocess.CompletedProcess(argv, 1, "", "AccessDenied")

        with pytest.raises(click.ClickException, match="Could not look up IAM user"):
            iam_user_exists("user", run=run)

    def test_create_access_key_errors(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "agilerl.arena.byoc.provisioning.providers.aws.terraform.resolve_aws_executable",
            lambda: "aws",
        )

        def run_bad_list(
            argv: list[str], **_kwargs: object
        ) -> subprocess.CompletedProcess[str]:
            return subprocess.CompletedProcess(
                argv, 0, json.dumps({"AccessKeyMetadata": {}}), ""
            )

        with pytest.raises(click.ClickException, match="unexpected access key list"):
            _create_access_key("user", run=run_bad_list)

        def run_missing_key(
            argv: list[str], **_kwargs: object
        ) -> subprocess.CompletedProcess[str]:
            if "list-access-keys" in argv:
                return subprocess.CompletedProcess(
                    argv, 0, json.dumps({"AccessKeyMetadata": []}), ""
                )
            return subprocess.CompletedProcess(argv, 0, json.dumps({}), "")

        with pytest.raises(click.ClickException, match="did not return an access key"):
            _create_access_key("user", run=run_missing_key)

        def run_non_string(
            argv: list[str], **_kwargs: object
        ) -> subprocess.CompletedProcess[str]:
            if "list-access-keys" in argv:
                return subprocess.CompletedProcess(
                    argv, 0, json.dumps({"AccessKeyMetadata": []}), ""
                )
            return subprocess.CompletedProcess(
                argv,
                0,
                json.dumps({"AccessKey": {"AccessKeyId": 1, "SecretAccessKey": 2}}),
                "",
            )

        with pytest.raises(click.ClickException, match="did not return an access key"):
            _create_access_key("user", run=run_non_string)

    def test_run_aws_ok_and_json_errors(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "agilerl.arena.byoc.provisioning.providers.aws.terraform.resolve_aws_executable",
            lambda: "aws",
        )

        def fail(
            argv: list[str], **_kwargs: object
        ) -> subprocess.CompletedProcess[str]:
            return subprocess.CompletedProcess(argv, 1, "", "boom")

        with pytest.raises(click.ClickException, match="failed"):
            run_aws_ok(["sts", "get-caller-identity"], run=fail)
        with pytest.raises(click.ClickException, match="failed"):
            run_aws_json(["sts", "get-caller-identity"], run=fail)

        def bad_json(
            argv: list[str], **_kwargs: object
        ) -> subprocess.CompletedProcess[str]:
            return subprocess.CompletedProcess(argv, 0, "not-json", "")

        with pytest.raises(click.ClickException, match="invalid JSON"):
            run_aws_json(["sts", "get-caller-identity"], run=bad_json)

        def not_object(
            argv: list[str], **_kwargs: object
        ) -> subprocess.CompletedProcess[str]:
            return subprocess.CompletedProcess(argv, 0, "[]", "")

        with pytest.raises(click.ClickException, match="invalid JSON"):
            run_aws_json(["sts", "get-caller-identity"], run=not_object)


class TestAwsProvisionerErrors:
    def test_lookup_failure_without_nosuchentity(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "agilerl.arena.byoc.provisioning.providers.aws.terraform.resolve_aws_executable",
            lambda: "aws",
        )

        def run(argv: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
            return subprocess.CompletedProcess(argv, 1, "", "AccessDenied")

        with pytest.raises(click.ClickException, match="Could not look up IAM user"):
            _ensure_user(run)

    def test_user_payload_without_arn(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "agilerl.arena.byoc.provisioning.providers.aws.terraform.resolve_aws_executable",
            lambda: "aws",
        )

        def run(argv: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
            return subprocess.CompletedProcess(argv, 0, json.dumps({"User": {}}), "")

        with pytest.raises(click.ClickException, match="did not return an ARN"):
            _ensure_user(run)

    def test_unexpected_access_key_list(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "agilerl.arena.byoc.provisioning.providers.aws.terraform.resolve_aws_executable",
            lambda: "aws",
        )

        def run(argv: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
            if "configure" in argv and "get" in argv:
                return subprocess.CompletedProcess(argv, 1, "", "")
            if "list-access-keys" in argv:
                return subprocess.CompletedProcess(
                    argv, 0, json.dumps({"AccessKeyMetadata": "nope"}), ""
                )
            return subprocess.CompletedProcess(argv, 0, "", "")

        with pytest.raises(click.ClickException, match="unexpected access key list"):
            _ensure_profile_key("eu-west-1", run)

    def test_create_access_key_without_payload(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "agilerl.arena.byoc.provisioning.providers.aws.terraform.resolve_aws_executable",
            lambda: "aws",
        )

        def run(argv: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
            if "configure" in argv and "get" in argv:
                return subprocess.CompletedProcess(argv, 1, "", "")
            if "list-access-keys" in argv:
                return subprocess.CompletedProcess(
                    argv, 0, json.dumps({"AccessKeyMetadata": []}), ""
                )
            if "create-access-key" in argv:
                return subprocess.CompletedProcess(argv, 0, json.dumps({}), "")
            return subprocess.CompletedProcess(argv, 0, "", "")

        with pytest.raises(click.ClickException, match="did not return an access key"):
            _ensure_profile_key("eu-west-1", run)

    def test_wait_for_profile_times_out(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "agilerl.arena.byoc.provisioning.providers.aws.terraform.resolve_aws_executable",
            lambda: "aws",
        )
        monkeypatch.setattr(
            "agilerl.arena.byoc.provisioning.providers.aws.provisioner.time.sleep",
            lambda _seconds: None,
        )

        def run(argv: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
            return subprocess.CompletedProcess(argv, 1, "", "not ready")

        with pytest.raises(click.ClickException, match="was not accepted"):
            _wait_for_profile(run)
