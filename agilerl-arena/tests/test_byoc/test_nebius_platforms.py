# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Tests for registering Nebius GPU presets as Arena resource classes."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from unittest.mock import MagicMock, patch

import click
import pytest
from click.testing import CliRunner

from agilerl.arena.byoc.cluster_register import (
    ClassAvailability,
    ClusterResourceClass,
    ensure_resource_class,
)
from agilerl.arena.byoc.commands import build_cluster_register_classes_command
from agilerl.arena.byoc.provisioning.providers.nebius.platforms import (
    GpuPreset,
    _match_requested,
    append_worker_blocks,
    cuda_drivers_preset,
    drivers_presets_from_matrix,
    insert_worker_blocks,
    list_capacity_advice,
    list_gpu_presets,
    newest_cuda_drivers_preset,
    node_group_ids,
    parse_gpu_presets,
    parse_preset_choice,
    project_id_from_outputs,
    region_capacity_for_preset,
    resolve_gpu_presets,
    worker_spec_for_preset,
)
from agilerl.arena.byoc.provisioning.providers.nebius.spec import (
    NEBIUS_WORKER_NODE_GROUP_LABEL,
    WorkerNodeGroupSpec,
)
from agilerl.arena.byoc.provisioning.spec import ClusterSpec, load_cluster_spec
from agilerl.arena.byoc.provisioning.terraform import ClusterOutputs
from agilerl.arena.config import CommandConfig

H100 = GpuPreset(
    platform="gpu-h100-sxm",
    preset="1gpu-16vcpu-200gb",
    gpu_count=1,
    vcpu_count=16,
    memory_gib=200,
    vram_gib=80,
)
H100_8 = GpuPreset(
    platform="gpu-h100-sxm",
    preset="8gpu-128vcpu-1600gb",
    gpu_count=8,
    vcpu_count=128,
    memory_gib=1600,
    vram_gib=80,
)
RTX = GpuPreset(
    platform="gpu-rtx6000-a",
    preset="1gpu-16vcpu-200gb",
    gpu_count=1,
    vcpu_count=16,
    memory_gib=200,
    vram_gib=96,
)

CLUSTER_SPEC = """\
# keep me
provider: nebius
name: test
terraform_state:
  bucket: test-tfstate
arena:
  storage:
    bucket: bucket
    endpoint: https://storage.example.com
  inference:
    domain: inference.example.com
nebius:
  tenant_id: tenant
  project_id: project-1
  storage_project_id: storage-project
  subnet_id: subnet
  kubernetes_version: "1.35"
  object_storage:
    size_gib: 1024
  workers:
    - name: workers
      min_node_count: 0
      max_node_count: 4
      platform: gpu-h100-sxm
      preset: 1gpu-16vcpu-200gb
      gpu_drivers_preset: cuda13.0
"""

H100_PLATFORM = {
    "metadata": {"name": "gpu-h100-sxm"},
    "spec": {
        "gpu_memory_gibibytes": 80,
        "presets": [
            {
                "name": "1gpu-16vcpu-200gb",
                "resources": {
                    "gpu_count": 1,
                    "vcpu_count": 16,
                    "memory_gibibytes": 200,
                },
            },
            {"name": "2vcpu-8gb", "resources": {"gpu_count": 0}},
        ],
    },
}


def _spec(tmp_path: Path, text: str = CLUSTER_SPEC) -> Path:
    path = tmp_path / "cluster.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def _outputs(spec: ClusterSpec, **_kwargs: object) -> ClusterOutputs:
    return ClusterOutputs(
        cluster_name=spec.name,
        context=spec.name,
        kubeconfig_path=Path("/tmp/kubeconfig"),
        storage_access_key_id="key",
        storage_secret_access_key="secret",
        worker_node_group_ids={
            worker.name: f"ng-{worker.name}" for worker in spec.nebius_cloud().workers
        },
    )


class TestParseGpuPresets:
    def test_keeps_gpu_presets_and_vram(self) -> None:
        presets = parse_gpu_presets(
            [
                H100_PLATFORM,
                {
                    "metadata": {"name": "cpu-e2"},
                    "spec": {"presets": [{"name": "2vcpu-8gb"}]},
                },
                {
                    "metadata": {"name": "gpu-h200-sxm"},
                    "spec": {
                        "gpuMemoryGibibytes": "141",
                        "presets": [
                            {
                                "name": "1gpu-16vcpu-200gb",
                                "resources": {"gpuCount": "1"},
                            }
                        ],
                    },
                },
            ]
        )

        assert [(item.platform, item.preset, item.vram_gib) for item in presets] == [
            ("gpu-h100-sxm", "1gpu-16vcpu-200gb", 80),
            ("gpu-h200-sxm", "1gpu-16vcpu-200gb", 141),
        ]

    def test_rejects_unrecognized_gpu_preset_name(self) -> None:
        with pytest.raises(click.ClickException, match="not a recognized size"):
            parse_gpu_presets(
                [
                    {
                        "metadata": {"name": "gpu-h100-sxm"},
                        "spec": {
                            "presets": [
                                {"name": "odd-gpu", "resources": {"gpu_count": 1}}
                            ]
                        },
                    }
                ]
            )

    def test_skips_platforms_without_a_usable_preset_list(self) -> None:
        presets = parse_gpu_presets(
            [
                {"name": "  "},
                {"name": "gpu-h100-sxm"},
                {
                    "metadata": {"name": "gpu-h100-sxm"},
                    "spec": {"gpu_memory_gibibytes": 0, "presets": "nope"},
                },
                {
                    "name": "gpu-l40s-d",
                    "spec": {
                        "gpuMemoryGibibytes": True,
                        "presets": [
                            "skip",
                            {"name": "  "},
                            {
                                "name": "odd-gpu",
                                "resources": {"gpu_count": True},
                            },
                            {
                                "name": "1gpu-8vcpu-32gb",
                                "resources": {"gpu_count": 0},
                            },
                            {"name": "1gpu-8vcpu-32gb"},
                        ],
                    },
                },
            ]
        )

        assert presets == [
            GpuPreset(
                platform="gpu-l40s-d",
                preset="1gpu-8vcpu-32gb",
                gpu_count=1,
                vcpu_count=8,
                memory_gib=32,
                vram_gib=None,
            )
        ]


class TestListGpuPresets:
    def test_queries_the_project(self, monkeypatch: pytest.MonkeyPatch) -> None:
        seen: dict[str, list[str]] = {}

        def fake(args: list[str], **_kwargs: object) -> list[dict[str, object]]:
            seen["args"] = args
            return [H100_PLATFORM]

        monkeypatch.setattr(
            "agilerl.arena.byoc.provisioning.providers.nebius.platforms._nebius_list_items",
            fake,
        )

        presets = list_gpu_presets("project-1")

        assert seen["args"] == [
            "compute",
            "platform",
            "list",
            "--parent-id",
            "project-1",
        ]
        assert presets == [
            GpuPreset(
                platform="gpu-h100-sxm",
                preset="1gpu-16vcpu-200gb",
                gpu_count=1,
                vcpu_count=16,
                memory_gib=200,
                vram_gib=80,
            )
        ]


class TestCudaDriversPreset:
    def test_picks_the_newest_cuda_preset_for_the_version(self) -> None:
        presets = drivers_presets_from_matrix(
            {
                "versions": [
                    {
                        "kubernetes_version": "1.33",
                        "items": [{"drivers_preset": "cuda12.4"}],
                    },
                    {
                        "kubernetesVersion": "1.35",
                        "items": [
                            {"driversPreset": "cuda12"},
                            {"drivers_preset": "cuda12.8"},
                            {"drivers_preset": "cuda13.0"},
                            {"os": "ubuntu24.04"},
                        ],
                    },
                ]
            },
            kubernetes_version="1.35",
        )

        assert (
            newest_cuda_drivers_preset(
                presets, platform="gpu-h200-sxm", kubernetes_version="1.35"
            )
            == "cuda13.0"
        )

    def test_rejects_a_matrix_without_cuda(self) -> None:
        with pytest.raises(click.ClickException, match="No CUDA drivers preset"):
            newest_cuda_drivers_preset(
                [], platform="gpu-h200-sxm", kubernetes_version="1.35"
            )

    def test_skips_invalid_matrix_entries(self) -> None:
        assert drivers_presets_from_matrix({"versions": "nope"}, "1.35") == []
        assert (
            drivers_presets_from_matrix(
                {
                    "versions": [
                        "nope",
                        {"kubernetes_version": "1.35", "items": "nope"},
                        {
                            "kubernetes_version": "1.35",
                            "items": ["skip", {"os": "ubuntu24.04"}],
                        },
                    ]
                },
                kubernetes_version="1.35",
            )
            == []
        )

    def test_ignores_non_cuda_driver_names(self) -> None:
        with pytest.raises(click.ClickException, match="No CUDA drivers preset"):
            newest_cuda_drivers_preset(
                ["nvidia", "cuda12.x"],
                platform="gpu-h200-sxm",
                kubernetes_version="1.35",
            )

    def test_reads_the_compatibility_matrix(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def fake(args: list[str], **_kwargs: object) -> dict[str, object]:
            assert args == [
                "mk8s",
                "node-group",
                "get-compatibility-matrix",
                "--cluster-kubernetes-version",
                "1.35",
                "--platform",
                "gpu-h200-sxm",
            ]
            return {
                "versions": [
                    {
                        "kubernetes_version": "1.35",
                        "items": [{"drivers_preset": "cuda13.0"}],
                    }
                ]
            }

        monkeypatch.setattr(
            "agilerl.arena.byoc.provisioning.providers.nebius.platforms._run_nebius_json",
            fake,
        )

        assert cuda_drivers_preset("1.35", "gpu-h200-sxm") == "cuda13.0"


class TestParsePresetChoice:
    def test_all_returns_every_preset(self) -> None:
        assert parse_preset_choice([H100, H100_8], "all") == [H100, H100_8]

    def test_numbers_keep_order_and_skip_duplicates(self) -> None:
        assert parse_preset_choice([H100, H100_8], "2, 2, 1") == [H100_8, H100]

    def test_rejects_text_that_is_not_a_choice(self) -> None:
        with pytest.raises(click.ClickException, match="list of numbers"):
            parse_preset_choice([H100], "h100")

    def test_rejects_a_number_outside_the_list(self) -> None:
        with pytest.raises(click.ClickException, match=r"outside 1\.\.1"):
            parse_preset_choice([H100], "2")

    def test_rejects_an_empty_choice(self) -> None:
        with pytest.raises(click.ClickException, match="at least one preset"):
            parse_preset_choice([H100], " , , ")


class TestResolveGpuPresets:
    def test_rejects_all_and_preset_together(self) -> None:
        with pytest.raises(click.ClickException, match="not both"):
            resolve_gpu_presets([H100], register_all=True, requested=("x",))

    def test_all_returns_every_preset(self) -> None:
        assert resolve_gpu_presets([H100, RTX], register_all=True, requested=()) == [
            H100,
            RTX,
        ]

    def test_requested_presets_skip_duplicates(self) -> None:
        key = f"{H100.platform}:{H100.preset}"

        assert resolve_gpu_presets(
            [H100, RTX], register_all=False, requested=(key, key)
        ) == [H100]

    def test_prompts_when_nothing_is_selected(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen: list[str] = []
        monkeypatch.setattr(
            "agilerl.arena.byoc.provisioning.providers.nebius.platforms.click.echo",
            lambda message: seen.append(str(message)),
        )
        monkeypatch.setattr(
            "agilerl.arena.byoc.provisioning.providers.nebius.platforms.click.prompt",
            lambda *_args, **_kwargs: "1",
        )

        assert resolve_gpu_presets([H100], register_all=False, requested=()) == [H100]
        assert any("GPU presets:" in line for line in seen)

    def test_match_requires_at_least_one_preset(self) -> None:
        with pytest.raises(click.ClickException, match="at least one --preset"):
            _match_requested([H100], ())


class TestInsertWorkerBlocks:
    def test_appends_a_worker_and_keeps_the_rest_of_the_file(
        self, tmp_path: Path
    ) -> None:
        path = _spec(tmp_path)
        spec = load_cluster_spec(path)
        worker = worker_spec_for_preset(spec, H100_8, drivers_preset="cuda13.0")

        append_worker_blocks(path, [worker])
        loaded = load_cluster_spec(path)

        assert "# keep me" in path.read_text(encoding="utf-8")
        assert [item.name for item in loaded.nebius_cloud().workers] == [
            "workers",
            "gpu-h100-sxm-8gpu-128vcpu-1600gb",
        ]
        added = loaded.nebius_cloud().workers[1]
        assert added.min_node_count == 0
        assert added.max_node_count == 4
        assert added.gpu_drivers_preset == "cuda13.0"
        assert added.register_node_group is False

    def test_suffixes_a_name_that_is_already_taken(self) -> None:
        spec = ClusterSpec.model_validate(
            {
                "provider": "nebius",
                "name": "test",
                "terraform_state": {"bucket": "test-tfstate"},
                "arena": {
                    "storage": {
                        "bucket": "bucket",
                        "endpoint": "https://storage.example.com",
                    },
                    "inference": {"domain": "inference.example.com"},
                },
                "nebius": {
                    "tenant_id": "tenant",
                    "project_id": "project-1",
                    "storage_project_id": "storage-project",
                    "subnet_id": "subnet",
                    "object_storage": {"size_gib": 1024},
                    "workers": [
                        {
                            "name": "gpu-h100-sxm-8gpu-128vcpu-1600gb",
                            "platform": "gpu-l40s-d",
                            "preset": "1gpu-8vcpu-32gb",
                        }
                    ],
                },
            }
        )

        worker = worker_spec_for_preset(spec, H100_8, drivers_preset="cuda13.0")

        assert worker.name == "gpu-h100-sxm-8gpu-128vcpu-1600gb-2"

    def test_rejects_an_inline_workers_list(self) -> None:
        worker = WorkerNodeGroupSpec(
            name="gpu-h100-sxm-1gpu-16vcpu-200gb",
            platform="gpu-h100-sxm",
            preset="1gpu-16vcpu-200gb",
            gpu_drivers_preset="cuda13.0",
        )

        with pytest.raises(click.ClickException, match="block list"):
            insert_worker_blocks("nebius:\n  workers: []\n", [worker])

    def test_empty_workers_leave_the_spec_unchanged(self) -> None:
        assert (
            insert_worker_blocks("nebius:\n  workers:\n", []) == "nebius:\n  workers:\n"
        )

    def test_adds_a_newline_before_inserting(self) -> None:
        worker = WorkerNodeGroupSpec(
            name="gpu-h100-sxm-1gpu-16vcpu-200gb",
            platform="gpu-h100-sxm",
            preset="1gpu-16vcpu-200gb",
            gpu_drivers_preset="cuda13.0",
        )

        text = insert_worker_blocks(
            "nebius:\n  workers:\n    # existing",
            [worker],
        )

        assert text.endswith("register_node_group: false\n")
        assert "  workers:\n    # existing\n    - name:" in text

    def test_rejects_a_spec_without_workers(self) -> None:
        worker = WorkerNodeGroupSpec(
            name="gpu-h100-sxm-1gpu-16vcpu-200gb",
            platform="gpu-h100-sxm",
            preset="1gpu-16vcpu-200gb",
            gpu_drivers_preset="cuda13.0",
        )

        with pytest.raises(click.ClickException, match=r"no nebius\.workers list"):
            insert_worker_blocks("provider: nebius\n", [worker])

    def test_inserts_workers_before_the_next_nebius_key(self) -> None:
        worker = WorkerNodeGroupSpec(
            name="gpu-h100-sxm-1gpu-16vcpu-200gb",
            platform="gpu-h100-sxm",
            preset="1gpu-16vcpu-200gb",
            gpu_drivers_preset="cuda13.0",
        )

        text = insert_worker_blocks(
            "nebius:\n"
            "  workers:\n"
            "    - name: workers\n"
            "      platform: gpu-h100-sxm\n"
            "      preset: 1gpu-16vcpu-200gb\n"
            "  subnet_id: subnet\n",
            [worker],
        )

        assert text.index("gpu-h100-sxm-1gpu-16vcpu-200gb") < text.index(
            "subnet_id: subnet"
        )


class TestNodeGroupIdsAndProject:
    def test_node_group_ids_require_a_string_map(self) -> None:
        with pytest.raises(click.ClickException, match="non-empty string map"):
            node_group_ids(None)
        with pytest.raises(click.ClickException, match="non-empty string map"):
            node_group_ids({"": "ng-1"})
        with pytest.raises(click.ClickException, match="non-empty string map"):
            node_group_ids({"gpu": "  "})
        assert node_group_ids({"gpu": "ng-1"}) == {"gpu": "ng-1"}

    def test_project_id_from_outputs(self) -> None:
        with pytest.raises(click.ClickException, match="project_id is missing"):
            project_id_from_outputs({})
        assert project_id_from_outputs({"project_id": " project-1 "}) == "project-1"


class TestRegionCapacityForPreset:
    def test_enables_regions_with_on_demand_capacity(self) -> None:
        items = [
            {
                "spec": {
                    "region": "eu-north1",
                    "fabric": "fabric-7",
                    "compute_instance": {
                        "platform": "gpu-l40s-a",
                        "preset": {"name": "1gpu-16vcpu-64gb"},
                    },
                },
                "status": {"on_demand": {"available": 4}},
            },
            {
                "spec": {
                    "region": "eu-north1",
                    "fabric": "fabric-2",
                    "compute_instance": {
                        "platform": "gpu-l40s-a",
                        "preset": {"name": "1gpu-16vcpu-64gb"},
                    },
                },
                "status": {"onDemand": {"available": 10}},
            },
            {
                "spec": {
                    "region": "us-central1",
                    "fabric": "fabric-3",
                    "compute_instance": {
                        "platform": "gpu-l40s-a",
                        "preset": {"name": "1gpu-16vcpu-64gb"},
                    },
                },
                "status": {"on_demand": {"available": 0}},
            },
            {
                "spec": {
                    "region": "me-west1",
                    "fabric": "fabric-1",
                    "compute_instance": {
                        "platform": "gpu-h100-sxm",
                        "preset": {"name": "1gpu-16vcpu-200gb"},
                    },
                },
                "status": {"on_demand": {"available": 8}},
            },
        ]

        capacities = region_capacity_for_preset(items, "gpu-l40s-a", "1gpu-16vcpu-64gb")

        assert [
            (row.region, row.zones, row.enabled, row.available) for row in capacities
        ] == [
            ("eu-north1", ("fabric-2", "fabric-7"), True, 14),
            ("us-central1", (), False, 0),
        ]

    def test_skips_advice_rows_that_are_not_usable(self) -> None:
        items = [
            {"spec": "nope"},
            {"spec": {"region": "eu-north1"}},
            {
                "spec": {
                    "region": "  ",
                    "compute_instance": {
                        "platform": "gpu-l40s-a",
                        "preset": {"name": "1gpu-16vcpu-64gb"},
                    },
                }
            },
            {
                "spec": {
                    "region": "eu-west1",
                    "fabric": True,
                    "computeInstance": {
                        "platform": "gpu-l40s-a",
                        "preset": "1gpu-16vcpu-64gb",
                    },
                },
                "status": "nope",
            },
            {
                "spec": {
                    "region": "eu-west1",
                    "compute_instance": {
                        "platform": "gpu-l40s-a",
                        "preset": {"name": "1gpu-16vcpu-64gb"},
                    },
                },
                "status": {"on_demand": {"available": True}},
            },
        ]

        assert [
            (row.region, row.zones, row.enabled, row.available)
            for row in region_capacity_for_preset(
                items, "gpu-l40s-a", "1gpu-16vcpu-64gb"
            )
        ] == [("eu-west1", (), False, 0)]

    def test_list_capacity_advice_queries_the_tenant(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen: dict[str, list[str]] = {}

        def fake(args: list[str], **_kwargs: object) -> list[dict[str, object]]:
            seen["args"] = args
            return [{"metadata": {"id": "advice-1"}}]

        monkeypatch.setattr(
            "agilerl.arena.byoc.provisioning.providers.nebius.platforms._nebius_list_items",
            fake,
        )

        assert list_capacity_advice("tenant-1") == [{"metadata": {"id": "advice-1"}}]
        assert seen["args"] == [
            "capacity",
            "resource-advice",
            "list",
            "--parent-id",
            "tenant-1",
        ]


class TestRegisterClassesCommand:
    @pytest.fixture(autouse=True)
    def no_region_capacity(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "agilerl.arena.byoc.commands.list_capacity_advice",
            lambda _tenant_id: [],
        )

    def test_requires_all_or_preset_when_not_interactive(
        self, command_config: CommandConfig, tmp_path: Path
    ) -> None:
        result = CliRunner().invoke(
            build_cluster_register_classes_command(),
            ["--spec", str(_spec(tmp_path))],
            obj=command_config,
        )

        assert result.exit_code != 0
        assert "--preset platform:preset" in result.output

    def test_rejects_all_and_preset_together(
        self, command_config: CommandConfig, tmp_path: Path
    ) -> None:
        result = CliRunner().invoke(
            build_cluster_register_classes_command(),
            [
                "--spec",
                str(_spec(tmp_path)),
                "--all",
                "--preset",
                f"{H100.platform}:{H100.preset}",
            ],
            obj=command_config,
        )

        assert result.exit_code != 0
        assert "not both" in result.output

    def test_registers_one_new_preset(
        self,
        command_config: CommandConfig,
        tmp_path: Path,
        client_context: Callable[[MagicMock], MagicMock],
    ) -> None:
        path = _spec(tmp_path)
        api = MagicMock()
        api.find_cluster.return_value = {"name": "test"}
        api.find_class.return_value = None
        api.create_class.side_effect = lambda **kwargs: {"name": kwargs["name"]}
        provisioner = MagicMock()
        provisioner.provision.side_effect = _outputs

        with (
            patch(
                "agilerl.arena.byoc.commands.arena_client",
                return_value=client_context(MagicMock()),
            ),
            patch("agilerl.arena.byoc.commands.ByocApi", return_value=api),
            patch(
                "agilerl.arena.byoc.commands.ClusterProvisioner",
                return_value=provisioner,
            ),
            patch(
                "agilerl.arena.byoc.commands.list_gpu_presets",
                return_value=[H100, RTX],
            ),
            patch(
                "agilerl.arena.byoc.commands.cuda_drivers_preset",
                return_value="cuda13.0",
            ) as drivers,
        ):
            result = CliRunner().invoke(
                build_cluster_register_classes_command(),
                [
                    "--spec",
                    str(path),
                    "--preset",
                    f"{RTX.platform}:{RTX.preset}",
                    "--yes",
                ],
                obj=command_config,
            )

        assert result.exit_code == 0, result.output
        assert path.read_text(encoding="utf-8") == CLUSTER_SPEC
        drivers.assert_not_called()
        provisioner.status.assert_not_called()
        provisioner.provision.assert_not_called()
        created = api.create_class.call_args.kwargs
        assert created["name"] == "test-gpu-rtx6000-a-1gpu-16vcpu-200gb"
        assert created["cluster_name"] == "test"
        assert created["node_selector"] == {
            "nebius.com/platform": "gpu-rtx6000-a",
            "nebius.com/preset": "1gpu-16vcpu-200gb",
        }
        compute = created["metadata"]["computeResource"]
        assert compute["numCpus"] == 16
        assert compute["numGpus"] == 1
        assert compute["memoryBytes"] == "200 GB"
        assert created["description"] == (
            "NVIDIA® RTX PRO™ 6000 1 GPU - 16 CPUs - 200 GB RAM."
        )
        assert compute["gpu"]["type"] == "gpu-rtx6000-a"
        assert compute["gpu"]["count"] == 1
        assert compute["gramPerGpu"] == 96
        assert api.create_class.call_count == 1
        assert created["availabilities"] == []

    def test_registers_region_capacity_for_the_preset(
        self,
        command_config: CommandConfig,
        tmp_path: Path,
        client_context: Callable[[MagicMock], MagicMock],
    ) -> None:
        path = _spec(tmp_path)
        api = MagicMock()
        api.find_cluster.return_value = {"name": "test"}
        api.find_class.return_value = None
        api.create_class.side_effect = lambda **kwargs: {"name": kwargs["name"]}
        advice = [
            {
                "spec": {
                    "region": "eu-north1",
                    "fabric": "fabric-2",
                    "compute_instance": {
                        "platform": RTX.platform,
                        "preset": {"name": RTX.preset},
                    },
                },
                "status": {"on_demand": {"available": 24}},
            },
            {
                "spec": {
                    "region": "us-central1",
                    "fabric": "fabric-1",
                    "compute_instance": {
                        "platform": RTX.platform,
                        "preset": {"name": RTX.preset},
                    },
                },
                "status": {"on_demand": {"available": 0}},
            },
        ]

        with (
            patch(
                "agilerl.arena.byoc.commands.arena_client",
                return_value=client_context(MagicMock()),
            ),
            patch("agilerl.arena.byoc.commands.ByocApi", return_value=api),
            patch(
                "agilerl.arena.byoc.commands.ClusterProvisioner",
                return_value=MagicMock(),
            ),
            patch(
                "agilerl.arena.byoc.commands.list_gpu_presets",
                return_value=[RTX],
            ),
            patch(
                "agilerl.arena.byoc.commands.list_capacity_advice",
                return_value=advice,
            ) as listed,
        ):
            result = CliRunner().invoke(
                build_cluster_register_classes_command(),
                [
                    "--spec",
                    str(path),
                    "--preset",
                    f"{RTX.platform}:{RTX.preset}",
                    "--yes",
                ],
                obj=command_config,
            )

        assert result.exit_code == 0, result.output
        assert listed.call_args.args == ("tenant",)
        created = api.create_class.call_args.kwargs
        assert created["availabilities"] == [
            {
                "region": "eu-north1",
                "zones": ["fabric-2"],
                "enabled": True,
            },
            {
                "region": "us-central1",
                "zones": [],
                "enabled": False,
            },
        ]
        assert (
            "Availability for 'test-gpu-rtx6000-a-1gpu-16vcpu-200gb': eu-north1 (24)."
            in result.output
        )

    def test_refreshes_availability_when_the_class_already_matches(self) -> None:
        resource_class = ClusterResourceClass(
            name="pool",
            num_nodes=1,
            node_selector={"nebius.com/platform": "gpu-l40s-a"},
            metadata={"computeResource": {"numCpus": 16}},
            availabilities=(
                ClassAvailability(
                    region="eu-north1",
                    zones=("fabric-2",),
                    enabled=True,
                    available=24,
                ),
            ),
        )
        api = MagicMock()
        api.find_class.return_value = {
            "name": "pool",
            "num_nodes": 1,
            "node_selector": {"nebius.com/platform": "gpu-l40s-a"},
            "metadata": {"computeResource": {"numCpus": 16}},
        }

        name = ensure_resource_class(
            api, cluster_name="test", resource_class=resource_class, yes=True
        )

        assert name == "pool"
        api.create_class.assert_not_called()
        api.update_class.assert_not_called()
        api.replace_class_availabilities.assert_called_once_with(
            "pool",
            [{"region": "eu-north1", "zones": ["fabric-2"], "enabled": True}],
        )

    def test_recovers_stored_project_ids_instead_of_preparing(
        self,
        command_config: CommandConfig,
        tmp_path: Path,
        client_context: Callable[[MagicMock], MagicMock],
    ) -> None:
        path = _spec(
            tmp_path,
            CLUSTER_SPEC.replace(
                "  project_id: project-1\n  storage_project_id: storage-project\n  subnet_id: subnet\n",
                "  region: eu-north1\n",
            ),
        )
        api = MagicMock()
        api.find_cluster.return_value = {"name": "test"}
        api.find_class.return_value = None
        api.create_class.side_effect = lambda **kwargs: {"name": kwargs["name"]}
        provisioner = MagicMock()
        listed: list[str] = []

        def recover(_config: CommandConfig, spec: ClusterSpec) -> ClusterSpec:
            return spec.model_copy(
                update={
                    "nebius": spec.nebius_cloud().model_copy(
                        update={"project_id": "project-stored"}
                    )
                }
            )

        with (
            patch(
                "agilerl.arena.byoc.commands.arena_client",
                return_value=client_context(MagicMock()),
            ),
            patch("agilerl.arena.byoc.commands.ByocApi", return_value=api),
            patch(
                "agilerl.arena.byoc.commands.ClusterProvisioner",
                return_value=provisioner,
            ),
            patch(
                "agilerl.arena.byoc.commands._recover_spec",
                side_effect=recover,
            ),
            patch(
                "agilerl.arena.byoc.commands.list_gpu_presets",
                side_effect=lambda project_id: listed.append(project_id) or [RTX],
            ),
        ):
            result = CliRunner().invoke(
                build_cluster_register_classes_command(),
                [
                    "--spec",
                    str(path),
                    "--preset",
                    f"{RTX.platform}:{RTX.preset}",
                    "--yes",
                ],
                obj=command_config,
            )

        assert result.exit_code == 0, result.output
        assert listed == ["project-stored"]
        provisioner.status.assert_not_called()
        provisioner.provision.assert_not_called()

    def test_register_node_group_adds_the_worker_and_applies(
        self,
        command_config: CommandConfig,
        tmp_path: Path,
        client_context: Callable[[MagicMock], MagicMock],
    ) -> None:
        path = _spec(tmp_path)
        api = MagicMock()
        api.find_cluster.return_value = {"name": "test"}
        api.find_class.return_value = None
        api.create_class.side_effect = lambda **kwargs: {"name": kwargs["name"]}
        provisioner = MagicMock()
        provisioner.provision.side_effect = _outputs

        with (
            patch(
                "agilerl.arena.byoc.commands.arena_client",
                return_value=client_context(MagicMock()),
            ),
            patch("agilerl.arena.byoc.commands.ByocApi", return_value=api),
            patch(
                "agilerl.arena.byoc.commands.ClusterProvisioner",
                return_value=provisioner,
            ),
            patch(
                "agilerl.arena.byoc.commands.list_gpu_presets",
                return_value=[H100, RTX],
            ),
            patch(
                "agilerl.arena.byoc.commands.cuda_drivers_preset",
                return_value="cuda13.0",
            ) as drivers,
        ):
            result = CliRunner().invoke(
                build_cluster_register_classes_command(),
                [
                    "--spec",
                    str(path),
                    "--preset",
                    f"{RTX.platform}:{RTX.preset}",
                    "--register-node-group",
                    "--yes",
                ],
                obj=command_config,
            )

        assert result.exit_code == 0, result.output
        assert "# keep me" in path.read_text(encoding="utf-8")
        loaded = load_cluster_spec(path)
        added = loaded.nebius_cloud().workers[1]
        assert added.platform == "gpu-rtx6000-a"
        assert added.min_node_count == 0
        assert added.register_node_group is True
        drivers.assert_called_once_with("1.35", "gpu-rtx6000-a")
        provisioner.status.assert_not_called()
        created = api.create_class.call_args.kwargs
        assert created["name"] == "test-gpu-rtx6000-a-1gpu-16vcpu-200gb"
        assert created["node_selector"] == {
            NEBIUS_WORKER_NODE_GROUP_LABEL: "ng-gpu-rtx6000-a-1gpu-16vcpu-200gb"
        }
        assert api.create_class.call_count == 1

    def test_registers_an_existing_node_group_without_applying(
        self,
        command_config: CommandConfig,
        tmp_path: Path,
        client_context: Callable[[MagicMock], MagicMock],
    ) -> None:
        path = _spec(tmp_path)
        original = path.read_text(encoding="utf-8")
        api = MagicMock()
        api.find_cluster.return_value = {"name": "test"}
        api.find_class.return_value = None
        api.create_class.side_effect = lambda **kwargs: {"name": kwargs["name"]}
        provisioner = MagicMock()
        provisioner.status.return_value = {
            "worker_node_group_ids": {"workers": "ng-workers"}
        }

        with (
            patch(
                "agilerl.arena.byoc.commands.arena_client",
                return_value=client_context(MagicMock()),
            ),
            patch("agilerl.arena.byoc.commands.ByocApi", return_value=api),
            patch(
                "agilerl.arena.byoc.commands.ClusterProvisioner",
                return_value=provisioner,
            ),
            patch(
                "agilerl.arena.byoc.commands.list_gpu_presets",
                return_value=[H100, H100_8],
            ),
            patch("agilerl.arena.byoc.commands.cuda_drivers_preset") as drivers,
        ):
            result = CliRunner().invoke(
                build_cluster_register_classes_command(),
                [
                    "--spec",
                    str(path),
                    "--preset",
                    f"{H100.platform}:{H100.preset}",
                    "--yes",
                ],
                obj=command_config,
            )

        assert result.exit_code == 0, result.output
        assert path.read_text(encoding="utf-8") == original
        drivers.assert_not_called()
        provisioner.provision.assert_not_called()
        created = api.create_class.call_args.kwargs
        assert created["name"] == "test-workers"
        assert created["num_nodes"] == 4
        assert created["node_selector"] == {
            NEBIUS_WORKER_NODE_GROUP_LABEL: "ng-workers"
        }
        compute = created["metadata"]["computeResource"]
        assert compute["numCpus"] == 16
        assert compute["memoryBytes"] == "200 GB"
        assert created["description"] == ("NVIDIA® H100 1 GPU - 16 CPUs - 200 GB RAM.")

    def test_all_registers_every_preset_and_skips_an_existing_class(
        self,
        command_config: CommandConfig,
        tmp_path: Path,
        client_context: Callable[[MagicMock], MagicMock],
    ) -> None:
        path = _spec(tmp_path)
        api = MagicMock()
        api.find_cluster.return_value = {"name": "test"}
        api.find_class.side_effect = lambda name: (
            {"name": name} if name == "test-workers" else None
        )
        api.create_class.side_effect = lambda **kwargs: {"name": kwargs["name"]}
        provisioner = MagicMock()
        provisioner.provision.side_effect = _outputs

        with (
            patch(
                "agilerl.arena.byoc.commands.arena_client",
                return_value=client_context(MagicMock()),
            ),
            patch("agilerl.arena.byoc.commands.ByocApi", return_value=api),
            patch(
                "agilerl.arena.byoc.commands.ClusterProvisioner",
                return_value=provisioner,
            ),
            patch(
                "agilerl.arena.byoc.commands.list_gpu_presets",
                return_value=[H100, H100_8],
            ),
            patch(
                "agilerl.arena.byoc.commands.cuda_drivers_preset",
                return_value="cuda13.0",
            ),
        ):
            result = CliRunner().invoke(
                build_cluster_register_classes_command(),
                ["--spec", str(path), "--all", "--yes"],
                obj=command_config,
            )

        assert result.exit_code == 0, result.output
        assert "already exists" in result.output
        updated = api.update_class.call_args.kwargs
        assert updated["name"] == "test-workers"
        assert updated["num_nodes"] == 4
        assert updated["metadata"]["computeResource"]["numCpus"] == 16
        assert updated["metadata"]["computeResource"]["memoryBytes"] == "200 GB"
        assert updated["description"] == ("NVIDIA® H100 1 GPU - 16 CPUs - 200 GB RAM.")
        created = api.create_class.call_args.kwargs
        assert created["name"] == "test-gpu-h100-sxm-8gpu-128vcpu-1600gb"
        compute = created["metadata"]["computeResource"]
        assert compute["numCpus"] == 128
        assert compute["numGpus"] == 8
        assert compute["memoryBytes"] == "1600 GB"
        assert compute["gpu"]["type"] == "gpu-h100-sxm"
        assert compute["gpu"]["count"] == 8
        assert created["node_selector"] == {
            "nebius.com/platform": "gpu-h100-sxm",
            "nebius.com/preset": "8gpu-128vcpu-1600gb",
        }
        assert api.create_class.call_count == 1
        provisioner.provision.assert_not_called()
        assert [
            worker.preset for worker in load_cluster_spec(path).nebius_cloud().workers
        ] == ["1gpu-16vcpu-200gb"]

    def test_reads_project_id_from_terraform_when_the_spec_omits_it(
        self,
        command_config: CommandConfig,
        tmp_path: Path,
        client_context: Callable[[MagicMock], MagicMock],
    ) -> None:
        path = _spec(
            tmp_path,
            CLUSTER_SPEC.replace(
                "  project_id: project-1\n", "  region: eu-north1\n"
            ).replace("  subnet_id: subnet\n", ""),
        )
        api = MagicMock()
        api.find_cluster.return_value = {"name": "test"}
        api.find_class.return_value = None
        api.create_class.return_value = {"name": "test-workers"}
        provisioner = MagicMock()
        provisioner.status.return_value = {
            "project_id": "project-created",
            "worker_node_group_ids": {"workers": "ng-workers"},
        }

        with (
            patch(
                "agilerl.arena.byoc.commands.arena_client",
                return_value=client_context(MagicMock()),
            ),
            patch("agilerl.arena.byoc.commands.ByocApi", return_value=api),
            patch(
                "agilerl.arena.byoc.commands.ClusterProvisioner",
                return_value=provisioner,
            ),
            patch(
                "agilerl.arena.byoc.commands.list_gpu_presets",
                return_value=[H100],
            ) as listed,
        ):
            result = CliRunner().invoke(
                build_cluster_register_classes_command(),
                [
                    "--spec",
                    str(path),
                    "--preset",
                    f"{H100.platform}:{H100.preset}",
                    "--yes",
                ],
                obj=command_config,
            )

        assert result.exit_code == 0, result.output
        assert listed.call_args.args[0] == "project-created"
        provisioner.provision.assert_not_called()
        assert provisioner.status.call_count == 1

    def test_missing_node_group_id_registers_the_class_without_applying(
        self,
        command_config: CommandConfig,
        tmp_path: Path,
        client_context: Callable[[MagicMock], MagicMock],
    ) -> None:
        path = _spec(tmp_path)
        api = MagicMock()
        api.find_cluster.return_value = {"name": "test"}
        api.find_class.return_value = None
        api.create_class.return_value = {"name": "test-workers"}
        provisioner = MagicMock()
        provisioner.status.return_value = {
            "worker_node_group_ids": {"other": "ng-other"}
        }
        provisioner.provision.side_effect = _outputs

        with (
            patch(
                "agilerl.arena.byoc.commands.arena_client",
                return_value=client_context(MagicMock()),
            ),
            patch("agilerl.arena.byoc.commands.ByocApi", return_value=api),
            patch(
                "agilerl.arena.byoc.commands.ClusterProvisioner",
                return_value=provisioner,
            ),
            patch(
                "agilerl.arena.byoc.commands.list_gpu_presets",
                return_value=[H100],
            ),
        ):
            result = CliRunner().invoke(
                build_cluster_register_classes_command(),
                [
                    "--spec",
                    str(path),
                    "--preset",
                    f"{H100.platform}:{H100.preset}",
                    "--yes",
                ],
                obj=command_config,
            )

        assert result.exit_code == 0, result.output
        provisioner.provision.assert_not_called()
        assert api.create_class.call_args.kwargs["node_selector"] == {
            "nebius.com/platform": "gpu-h100-sxm",
            "nebius.com/preset": "1gpu-16vcpu-200gb",
        }
        created = api.create_class.call_args.kwargs["metadata"]["computeResource"]
        assert created["numGpus"] == 1
        assert created["gpu"]["type"] == "gpu-h100-sxm"
        assert created["gpu"]["count"] == 1

    def test_register_node_group_applies_when_the_id_is_missing(
        self,
        command_config: CommandConfig,
        tmp_path: Path,
        client_context: Callable[[MagicMock], MagicMock],
    ) -> None:
        path = _spec(tmp_path)
        api = MagicMock()
        api.find_cluster.return_value = {"name": "test"}
        api.find_class.return_value = None
        api.create_class.return_value = {"name": "test-workers"}
        provisioner = MagicMock()
        provisioner.status.return_value = {
            "worker_node_group_ids": {"other": "ng-other"}
        }
        provisioner.provision.side_effect = _outputs

        with (
            patch(
                "agilerl.arena.byoc.commands.arena_client",
                return_value=client_context(MagicMock()),
            ),
            patch("agilerl.arena.byoc.commands.ByocApi", return_value=api),
            patch(
                "agilerl.arena.byoc.commands.ClusterProvisioner",
                return_value=provisioner,
            ),
            patch(
                "agilerl.arena.byoc.commands.list_gpu_presets",
                return_value=[H100],
            ),
        ):
            result = CliRunner().invoke(
                build_cluster_register_classes_command(),
                [
                    "--spec",
                    str(path),
                    "--preset",
                    f"{H100.platform}:{H100.preset}",
                    "--register-node-group",
                    "--yes",
                ],
                obj=command_config,
            )

        assert result.exit_code == 0, result.output
        provisioner.provision.assert_called_once()
        assert api.create_class.call_args.kwargs["node_selector"] == {
            NEBIUS_WORKER_NODE_GROUP_LABEL: "ng-workers"
        }

    def test_spec_register_node_group_applies_without_the_flag(
        self,
        command_config: CommandConfig,
        tmp_path: Path,
        client_context: Callable[[MagicMock], MagicMock],
    ) -> None:
        path = _spec(
            tmp_path,
            CLUSTER_SPEC.replace(
                "gpu_drivers_preset: cuda13.0\n",
                "gpu_drivers_preset: cuda13.0\n      register_node_group: true\n",
            ),
        )
        api = MagicMock()
        api.find_cluster.return_value = {"name": "test"}
        api.find_class.return_value = None
        api.create_class.return_value = {"name": "test-workers"}
        provisioner = MagicMock()
        provisioner.status.return_value = {
            "worker_node_group_ids": {"other": "ng-other"}
        }
        provisioner.provision.side_effect = _outputs

        with (
            patch(
                "agilerl.arena.byoc.commands.arena_client",
                return_value=client_context(MagicMock()),
            ),
            patch("agilerl.arena.byoc.commands.ByocApi", return_value=api),
            patch(
                "agilerl.arena.byoc.commands.ClusterProvisioner",
                return_value=provisioner,
            ),
            patch(
                "agilerl.arena.byoc.commands.list_gpu_presets",
                return_value=[H100],
            ),
        ):
            result = CliRunner().invoke(
                build_cluster_register_classes_command(),
                [
                    "--spec",
                    str(path),
                    "--preset",
                    f"{H100.platform}:{H100.preset}",
                    "--yes",
                ],
                obj=command_config,
            )

        assert result.exit_code == 0, result.output
        provisioner.provision.assert_called_once()
        assert api.create_class.call_args.kwargs["node_selector"] == {
            NEBIUS_WORKER_NODE_GROUP_LABEL: "ng-workers"
        }

    def test_stops_when_the_cluster_is_not_registered(
        self,
        command_config: CommandConfig,
        tmp_path: Path,
        client_context: Callable[[MagicMock], MagicMock],
    ) -> None:
        path = _spec(tmp_path)
        api = MagicMock()
        api.find_cluster.return_value = None
        provisioner = MagicMock()

        with (
            patch(
                "agilerl.arena.byoc.commands.arena_client",
                return_value=client_context(MagicMock()),
            ),
            patch("agilerl.arena.byoc.commands.ByocApi", return_value=api),
            patch(
                "agilerl.arena.byoc.commands.ClusterProvisioner",
                return_value=provisioner,
            ),
            patch("agilerl.arena.byoc.commands.list_gpu_presets") as listed,
        ):
            result = CliRunner().invoke(
                build_cluster_register_classes_command(),
                ["--spec", str(path), "--all", "--yes"],
                obj=command_config,
            )

        assert result.exit_code != 0
        assert "not registered" in result.output
        listed.assert_not_called()
        provisioner.provision.assert_not_called()

    def test_abort_leaves_the_spec_unchanged(
        self,
        command_config: CommandConfig,
        tmp_path: Path,
        client_context: Callable[[MagicMock], MagicMock],
    ) -> None:
        path = _spec(tmp_path)
        original = path.read_text(encoding="utf-8")
        api = MagicMock()
        api.find_cluster.return_value = {"name": "test"}
        provisioner = MagicMock()

        with (
            patch(
                "agilerl.arena.byoc.commands.arena_client",
                return_value=client_context(MagicMock()),
            ),
            patch("agilerl.arena.byoc.commands.ByocApi", return_value=api),
            patch(
                "agilerl.arena.byoc.commands.ClusterProvisioner",
                return_value=provisioner,
            ),
            patch(
                "agilerl.arena.byoc.commands.list_gpu_presets",
                return_value=[H100, H100_8],
            ),
            patch(
                "agilerl.arena.byoc.commands.cuda_drivers_preset",
                return_value="cuda13.0",
            ),
        ):
            result = CliRunner().invoke(
                build_cluster_register_classes_command(),
                ["--spec", str(path), "--all"],
                input="n\n",
                obj=command_config,
            )

        assert result.exit_code == 0, result.output
        assert "Aborted." in result.output
        assert path.read_text(encoding="utf-8") == original
        provisioner.provision.assert_not_called()
        api.create_class.assert_not_called()

    def test_unknown_preset_does_not_change_the_spec(
        self,
        command_config: CommandConfig,
        tmp_path: Path,
        client_context: Callable[[MagicMock], MagicMock],
    ) -> None:
        path = _spec(tmp_path)
        original = path.read_text(encoding="utf-8")
        api = MagicMock()
        api.find_cluster.return_value = {"name": "test"}

        with (
            patch(
                "agilerl.arena.byoc.commands.arena_client",
                return_value=client_context(MagicMock()),
            ),
            patch("agilerl.arena.byoc.commands.ByocApi", return_value=api),
            patch(
                "agilerl.arena.byoc.commands.list_gpu_presets",
                return_value=[H100],
            ),
        ):
            result = CliRunner().invoke(
                build_cluster_register_classes_command(),
                [
                    "--spec",
                    str(path),
                    "--preset",
                    "gpu-b200-sxm:1gpu-16vcpu-200gb",
                    "--yes",
                ],
                obj=command_config,
            )

        assert result.exit_code != 0
        assert "Unknown GPU preset" in result.output
        assert path.read_text(encoding="utf-8") == original
