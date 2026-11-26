# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0
"""Predicted step seconds for SFT and DPO runs, which generate no rollouts."""

from __future__ import annotations

from agilerl.arena.memory.resolve.split import learn_gpu_seconds
from agilerl.arena.memory.specs import DeviceSpec, ModelSpec
from agilerl.arena.models.algorithms.dpo import DPOSpec
from agilerl.arena.models.algorithms.sft import SFTSpec
from agilerl.arena.models.manifest import TrainingManifest


def supervised_step_seconds(
    manifest: TrainingManifest, model: ModelSpec, device: DeviceSpec
) -> float | None:
    """Seconds one SFT or DPO optimizer step takes on the manifest's trainers.

    Rows are ``max_model_len`` long, as packed rows are. A DPO row is a chosen
    and a rejected sequence, and the reference scores both once. Each update
    epoch is a forward and a backward.

    :param manifest: Validated manifest with ``training_gpus_per_agent`` set.
    :param model: Model the trainers train.
    :param device: One trainer GPU.
    :return: Step seconds, or ``None`` for other algorithms or a GPU with no
        peak bf16 figure.
    """
    spec = manifest.algorithm
    peak_flops = device.peak_bf16_flops
    if peak_flops is None or not isinstance(spec, SFTSpec | DPOSpec):
        return None
    sequences = 2 if isinstance(spec, DPOSpec) else 1
    # DPO.learn scores the reference before the epoch loop at every beta.
    reference = int(isinstance(spec, DPOSpec))
    gpu_seconds = learn_gpu_seconds(
        model,
        device,
        peak_flops,
        spec.batch_size * sequences,
        spec.max_model_len,
        2 * spec.update_epochs + reference,
    )
    return gpu_seconds / manifest.training.training_gpus_per_agent
