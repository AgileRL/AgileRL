# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Nebius GPU platforms and presets for a registered cluster."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import click

from agilerl.arena.byoc.provisioning.providers.nebius.spec import (
    NEBIUS_GPU_WORKER_PRESET,
    WorkerNodeGroupSpec,
)
from agilerl.arena.byoc.provisioning.providers.nebius.terraform import (
    NebiusRun,
    _nebius_list_items,
    _run_nebius_json,
)
from agilerl.arena.byoc.provisioning.spec import ClusterSpec


@dataclass(frozen=True)
class GpuPreset:
    """One GPU preset on a Nebius compute platform."""

    platform: str
    preset: str
    gpu_count: int
    vcpu_count: int
    memory_gib: int
    vram_gib: int | None


@dataclass(frozen=True)
class RegionCapacity:
    """On-demand capacity for one preset in one Nebius region."""

    region: str
    zones: tuple[str, ...]
    enabled: bool
    available: int


def list_gpu_presets(
    project_id: str, run: NebiusRun = subprocess.run
) -> list[GpuPreset]:
    """Return GPU presets in a Nebius project."""
    items = _nebius_list_items(
        ["compute", "platform", "list", "--parent-id", project_id],
        run=run,
    )
    return parse_gpu_presets(items)


def list_capacity_advice(
    tenant_id: str, run: NebiusRun = subprocess.run
) -> list[dict[str, Any]]:
    """Return capacity advisor rows for a Nebius tenant."""
    return _nebius_list_items(
        ["capacity", "resource-advice", "list", "--parent-id", tenant_id],
        run=run,
    )


def region_capacity_for_preset(
    items: list[dict[str, Any]], platform: str, preset: str
) -> list[RegionCapacity]:
    """Return per-region on-demand capacity for one platform and preset."""
    available_by_zone: dict[tuple[str, str], int] = {}
    for item in items:
        parsed = _advice_row(item)
        if parsed is None:
            continue
        row_platform, row_preset, region, fabric, available = parsed
        if row_platform != platform or row_preset != preset:
            continue
        key = (region, fabric)
        available_by_zone[key] = available_by_zone.get(key, 0) + available

    by_region: dict[str, list[tuple[str, int]]] = {}
    for (region, fabric), available in available_by_zone.items():
        by_region.setdefault(region, []).append((fabric, available))

    capacities: list[RegionCapacity] = []
    for region in sorted(by_region):
        fabrics = by_region[region]
        total = sum(count for _, count in fabrics)
        zones = tuple(
            sorted(fabric for fabric, count in fabrics if fabric and count > 0)
        )
        # on_demand.available is how many VMs the quota and physical capacity allow.
        capacities.append(
            RegionCapacity(
                region=region,
                zones=zones,
                enabled=total > 0,
                available=total,
            )
        )
    return capacities


def parse_gpu_presets(items: list[dict[str, Any]]) -> list[GpuPreset]:
    """Return GPU presets from a Nebius compute platform list."""
    presets: list[GpuPreset] = []
    for item in items:
        platform = _platform_name(item)
        if platform is None:
            continue
        spec = item.get("spec")
        if not isinstance(spec, dict):
            continue
        vram_gib = _non_negative_int(
            _field(spec, "gpu_memory_gibibytes", "gpuMemoryGibibytes")
        )
        if vram_gib == 0:
            vram_gib = None
        raw_presets = spec.get("presets")
        if not isinstance(raw_presets, list):
            continue
        for raw in raw_presets:
            preset = _gpu_preset(platform, raw, vram_gib=vram_gib)
            if preset is not None:
                presets.append(preset)
    return presets


def cuda_drivers_preset(
    kubernetes_version: str,
    platform: str,
    run: NebiusRun = subprocess.run,
) -> str:
    """Return the newest CUDA drivers preset for a platform and Kubernetes version."""
    payload = _run_nebius_json(
        [
            "mk8s",
            "node-group",
            "get-compatibility-matrix",
            "--cluster-kubernetes-version",
            kubernetes_version,
            "--platform",
            platform,
        ],
        run=run,
    )
    return newest_cuda_drivers_preset(
        drivers_presets_from_matrix(payload, kubernetes_version=kubernetes_version),
        platform=platform,
        kubernetes_version=kubernetes_version,
    )


def drivers_presets_from_matrix(
    payload: dict[str, Any], kubernetes_version: str
) -> list[str]:
    """Return drivers presets for one Kubernetes version in a compatibility matrix."""
    versions = payload.get("versions")
    if not isinstance(versions, list):
        return []
    found: list[str] = []
    for version in versions:
        if not isinstance(version, dict):
            continue
        raw_version = _field(version, "kubernetes_version", "kubernetesVersion")
        if (
            not isinstance(raw_version, str)
            or raw_version.strip() != kubernetes_version
        ):
            continue
        items = version.get("items")
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            preset = _field(item, "drivers_preset", "driversPreset")
            if isinstance(preset, str) and preset.strip():
                found.append(preset.strip())
    return found


def newest_cuda_drivers_preset(
    presets: list[str], platform: str, kubernetes_version: str
) -> str:
    """Return the highest cuda* preset, or raise when none are listed."""
    ranked = [preset for preset in presets if _cuda_rank(preset)]
    if not ranked:
        msg = (
            f"No CUDA drivers preset for {platform} on Kubernetes {kubernetes_version}."
        )
        raise click.ClickException(msg)
    return max(ranked, key=_cuda_rank)


def resolve_gpu_presets(
    presets: list[GpuPreset],
    register_all: bool,
    requested: tuple[str, ...],
) -> list[GpuPreset]:
    """Return the presets selected by ``--all``, ``--preset``, or a prompt."""
    if register_all and requested:
        msg = "Pass --all or --preset, not both."
        raise click.ClickException(msg)
    if register_all:
        return list(presets)
    if requested:
        return _match_requested(presets, requested)
    _echo_presets(presets)
    answer = click.prompt(
        "Register all, or a comma-separated list of numbers",
        type=str,
    )
    return parse_preset_choice(presets, answer)


def parse_preset_choice(presets: list[GpuPreset], answer: str) -> list[GpuPreset]:
    """Return presets chosen by ``all`` or a comma-separated list of numbers."""
    text = answer.strip().lower()
    if text == "all":
        return list(presets)
    indexes: list[int] = []
    for part in text.replace(" ", ",").split(","):
        piece = part.strip()
        if not piece:
            continue
        if not piece.isdigit():
            msg = f"Choose all or a list of numbers, not {answer!r}."
            raise click.ClickException(msg)
        indexes.append(int(piece))
    if not indexes:
        msg = "Choose all or at least one preset."
        raise click.ClickException(msg)
    chosen: list[GpuPreset] = []
    seen: set[int] = set()
    for index in indexes:
        if index in seen:
            continue
        seen.add(index)
        if index < 1 or index > len(presets):
            msg = f"Preset number {index} is outside 1..{len(presets)}."
            raise click.ClickException(msg)
        chosen.append(presets[index - 1])
    return chosen


def worker_for_preset(
    spec: ClusterSpec, preset: GpuPreset
) -> WorkerNodeGroupSpec | None:
    """Return the worker that already uses this platform and preset."""
    for worker in spec.nebius_cloud().workers:
        if worker.platform == preset.platform and worker.preset == preset.preset:
            return worker
    return None


def worker_spec_for_preset(
    spec: ClusterSpec,
    preset: GpuPreset,
    drivers_preset: str,
    register_node_group: bool = False,
) -> WorkerNodeGroupSpec:
    """Return a new worker node group for a GPU preset."""
    cloud = spec.nebius_cloud()
    taken = {worker.name for worker in cloud.workers}
    base = f"{preset.platform}-{preset.preset}"
    name = base
    suffix = 2
    while name in taken:
        name = f"{base}-{suffix}"
        suffix += 1
    return WorkerNodeGroupSpec(
        name=name,
        min_node_count=0,
        max_node_count=cloud.workers[0].max_node_count,
        platform=preset.platform,
        preset=preset.preset,
        gpu_drivers_preset=drivers_preset,
        register_node_group=register_node_group,
    )


def append_worker_blocks(path: Path, workers: list[WorkerNodeGroupSpec]) -> None:
    """Append worker node groups to a cluster spec without rewriting the rest."""
    text = path.read_text(encoding="utf-8")
    path.write_text(insert_worker_blocks(text, workers), encoding="utf-8")


def insert_worker_blocks(text: str, workers: list[WorkerNodeGroupSpec]) -> str:
    """Return spec text with worker blocks inserted at the end of ``nebius.workers``."""
    if not workers:
        return text
    lines = text.splitlines(keepends=True)
    workers_index, workers_indent = _workers_list_start(lines)
    item_indent, end = _workers_list_end(lines, workers_index, workers_indent)
    if end > 0 and not lines[end - 1].endswith("\n"):
        lines[end - 1] = f"{lines[end - 1]}\n"
    lines[end:end] = [_worker_block(worker, item_indent) for worker in workers]
    return "".join(lines)


def node_group_ids(value: object) -> dict[str, str]:
    """Return Terraform ``worker_node_group_ids`` as a name-to-id map."""
    if not isinstance(value, dict) or not value:
        msg = "Terraform output worker_node_group_ids must be a non-empty string map."
        raise click.ClickException(msg)
    ids: dict[str, str] = {}
    for name, group_id in value.items():
        if not isinstance(name, str) or not name.strip():
            msg = (
                "Terraform output worker_node_group_ids must be a non-empty string map."
            )
            raise click.ClickException(msg)
        if not isinstance(group_id, str) or not group_id.strip():
            msg = (
                "Terraform output worker_node_group_ids must be a non-empty string map."
            )
            raise click.ClickException(msg)
        ids[name] = group_id
    return ids


def project_id_from_outputs(outputs: dict[str, object]) -> str:
    """Return the Nebius project id from Terraform outputs."""
    value = outputs.get("project_id")
    if not isinstance(value, str) or not value.strip():
        msg = (
            "Terraform output project_id is missing. Set nebius.project_id in the spec."
        )
        raise click.ClickException(msg)
    return value.strip()


def _gpu_preset(platform: str, raw: object, vram_gib: int | None) -> GpuPreset | None:
    if not isinstance(raw, dict):
        return None
    name = raw.get("name")
    if not isinstance(name, str) or not name.strip():
        return None
    preset = name.strip()
    match = NEBIUS_GPU_WORKER_PRESET.fullmatch(preset.lower())
    resources = raw.get("resources")
    gpu_count = None
    if isinstance(resources, dict):
        gpu_count = _non_negative_int(_field(resources, "gpu_count", "gpuCount"))
    if gpu_count == 0:
        return None
    if match is None:
        if gpu_count is not None and gpu_count > 0:
            msg = f"GPU preset {preset!r} on {platform!r} is not a recognized size."
            raise click.ClickException(msg)
        return None
    return GpuPreset(
        platform=platform,
        preset=preset,
        gpu_count=int(match["gpus"]),
        vcpu_count=int(match["cpus"]),
        memory_gib=int(match["memory_gb"]),
        vram_gib=vram_gib,
    )


def _platform_name(item: dict[str, Any]) -> str | None:
    metadata = item.get("metadata")
    if isinstance(metadata, dict):
        name = metadata.get("name")
        if isinstance(name, str) and name.strip():
            return name.strip()
    name = item.get("name")
    if isinstance(name, str) and name.strip():
        return name.strip()
    return None


def _match_requested(
    presets: list[GpuPreset], requested: tuple[str, ...]
) -> list[GpuPreset]:
    by_key = {f"{item.platform}:{item.preset}": item for item in presets}
    chosen: list[GpuPreset] = []
    seen: set[str] = set()
    missing: list[str] = []
    for raw in requested:
        key = raw.strip()
        if key in seen:
            continue
        seen.add(key)
        item = by_key.get(key)
        if item is None:
            missing.append(key)
        else:
            chosen.append(item)
    if missing:
        available = ", ".join(sorted(by_key))
        listed = ", ".join(missing)
        msg = f"Unknown GPU preset {listed}. Available: {available}."
        raise click.ClickException(msg)
    if not chosen:
        msg = "Pass at least one --preset platform:preset."
        raise click.ClickException(msg)
    return chosen


def _echo_presets(presets: list[GpuPreset]) -> None:
    click.echo("GPU presets:")
    for index, preset in enumerate(presets, start=1):
        click.echo(
            f"  {index}. {preset.platform}:{preset.preset} "
            f"({preset.gpu_count} GPU, {preset.vcpu_count} vCPU, "
            f"{preset.memory_gib} GB)"
        )


def _workers_list_start(lines: list[str]) -> tuple[int, str]:
    for index, line in enumerate(lines):
        stripped = line.lstrip(" ")
        if stripped.startswith("#") or not stripped.startswith("workers:"):
            continue
        rest = stripped[len("workers:") :].split("#", 1)[0].strip()
        if rest:
            msg = "nebius.workers must be a block list so new presets can be added."
            raise click.ClickException(msg)
        return index, line[: len(line) - len(stripped)]
    msg = "Cluster spec has no nebius.workers list."
    raise click.ClickException(msg)


def _workers_list_end(
    lines: list[str], workers_index: int, workers_indent: str
) -> tuple[str, int]:
    item_indent: str | None = None
    end = len(lines)
    for index in range(workers_index + 1, len(lines)):
        line = lines[index]
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        indent = len(line) - len(line.lstrip(" "))
        if indent <= len(workers_indent):
            end = index
            break
        if item_indent is None and line.lstrip().startswith("- "):
            item_indent = line[:indent]
    if item_indent is None:
        item_indent = f"{workers_indent}  "
    return item_indent, end


def _worker_block(worker: WorkerNodeGroupSpec, item_indent: str) -> str:
    body = f"{item_indent}  "
    return (
        f"{item_indent}- name: {worker.name}\n"
        f"{body}min_node_count: {worker.min_node_count}\n"
        f"{body}max_node_count: {worker.max_node_count}\n"
        f"{body}platform: {worker.platform}\n"
        f"{body}preset: {worker.preset}\n"
        f"{body}gpu_drivers_preset: {worker.gpu_drivers_preset}\n"
        f"{body}register_node_group: {str(worker.register_node_group).lower()}\n"
    )


def _cuda_rank(name: str) -> tuple[int, ...]:
    if not name.startswith("cuda"):
        return ()
    parts: list[int] = []
    for piece in name.removeprefix("cuda").split("."):
        if not piece.isdigit():
            return ()
        parts.append(int(piece))
    return tuple(parts)


def _advice_row(item: dict[str, Any]) -> tuple[str, str, str, str, int] | None:
    spec = item.get("spec")
    if not isinstance(spec, dict):
        return None
    compute = _field(spec, "compute_instance", "computeInstance")
    if not isinstance(compute, dict):
        return None
    platform = compute.get("platform")
    preset_value = compute.get("preset")
    preset = preset_value.get("name") if isinstance(preset_value, dict) else None
    region = spec.get("region")
    if (
        not isinstance(platform, str)
        or not isinstance(preset, str)
        or not isinstance(region, str)
        or not platform
        or not preset
        or not region.strip()
    ):
        return None
    fabric = spec.get("fabric")
    fabric_name = fabric.strip() if isinstance(fabric, str) else ""
    available = 0
    status = item.get("status")
    if isinstance(status, dict):
        on_demand = _field(status, "on_demand", "onDemand")
        if isinstance(on_demand, dict):
            parsed = _non_negative_int(on_demand.get("available"))
            if parsed is not None:
                available = parsed
    return platform, preset, region.strip(), fabric_name, available


def _field(payload: dict[str, Any], *names: str) -> object:
    for name in names:
        if name in payload:
            return payload[name]
    return None


def _non_negative_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value >= 0 else None
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None
