# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0
"""Whether an omitted PPO actor-critic pass is fused."""

from __future__ import annotations

from agilerl.arena.memory.estimator import estimate_training
from agilerl.arena.memory.specs import DeviceSpec, ModelSpec, TrainingSettings


def resolve_fuse_actor_critic_pass(
    configured: bool | None,
    fused_fits: bool | None,
) -> bool:
    """Whether the actor and critic share one gradient pass.

    :param configured: Manifest value. ``None`` leaves the choice to the fit.
    :param fused_fits: Whether the fused peak is inside the device budget.
        ``None`` means there is no device budget.
    :return: The explicit value, or fused when that peak fits or there is no
        budget.
    """
    if configured is not None:
        return configured
    if fused_fits is None:
        return True
    return fused_fits


def training_settings_with_fuse(
    configured: bool | None,
    settings: TrainingSettings,
    model: ModelSpec | None,
    device: DeviceSpec | None,
) -> TrainingSettings:
    """Training settings with the fuse bit from :func:`resolve_fuse_actor_critic_pass`.

    :param configured: Manifest ``fuse_actor_critic_pass``. Read for PPO only.
    :param settings: Training settings, including the micro-batch to price.
    :param model: Model the fused peak is estimated on.
    :param device: Training GPU. Without one, an omitted PPO field fuses.
    :return: Settings whose fuse bit is the shared decision.
    """
    if settings.algorithm != "ppo":
        return settings
    if configured is not None or model is None or device is None:
        return settings.model_copy(
            update={
                "fuse_actor_critic_pass": resolve_fuse_actor_critic_pass(
                    configured, None
                )
            }
        )
    trial = settings.model_copy(update={"fuse_actor_critic_pass": True})
    fused_fits = estimate_training(model, device, trial, orchestrated=True).fits
    fuse = resolve_fuse_actor_critic_pass(None, fused_fits)
    if fuse:
        return trial
    return settings.model_copy(update={"fuse_actor_critic_pass": False})
