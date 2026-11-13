# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Qwen3.5 layers: GatedDeltaNet mixer wraps and the fused gated RMSNorm.

The catalog names the mixer classes to wrap.
"""

from __future__ import annotations

import importlib
import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from agilerl.utils.patching import try_import

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GdnKernel:
    """A fused GatedDeltaNet kernel: where it imports from, and its pip package."""

    module: str
    function: str
    package: str


def _kernel_import_error(kernel: GdnKernel) -> str | None:
    try:
        getattr(importlib.import_module(kernel.module), kernel.function)
    except (ImportError, AttributeError) as error:
        return f"{kernel.package} ({type(error).__name__}: {error})"
    return None


def gdn_kernel_import_errors(kernels: tuple[GdnKernel, ...]) -> list[str]:
    """Import each kernel and describe every one that fails.

    transformers falls back to its torch kernels on any import error, so a
    wheel that is installed but broken has to be imported to be caught.

    :param kernels: Kernels to import.
    :return: ``"<package> (<error>)"`` for each kernel that did not import.
    """
    return [
        error
        for kernel in kernels
        if (error := _kernel_import_error(kernel)) is not None
    ]


def clone_gdn_conv_output(fn: Callable[..., Any] | None) -> Callable[..., Any] | None:
    """Return ``fn`` with a cloned output so later in-place kernels do not alias it."""
    if fn is None or getattr(fn, "_agilerl_clone_output", False):
        return fn

    def wrapped(*args: object, **kwargs: object) -> object:
        out = fn(*args, **kwargs)
        if isinstance(out, torch.Tensor) and out.requires_grad:
            return out.clone()
        return out

    wrapped.__dict__["_agilerl_clone_output"] = True
    return wrapped


def install_gdn_conv_clone(mixer: nn.Module) -> bool:
    """Clone ``causal_conv1d_fn`` output on ``mixer`` once.

    :param mixer: A GatedDeltaNet module.
    :return: True when this call wrapped the mixer.
    """
    if getattr(mixer, "_agilerl_gdn_conv_clone", False):
        return False
    conv = getattr(mixer, "causal_conv1d_fn", None)
    wrapped = clone_gdn_conv_output(conv)
    if wrapped is None or wrapped is conv:
        return False
    object.__setattr__(mixer, "causal_conv1d_fn", wrapped)
    object.__setattr__(mixer, "_agilerl_gdn_conv_clone", True)
    return True


def apply_gdn_conv_clone(model: nn.Module, mixers: tuple[str, ...]) -> int:
    """Wrap each catalog GatedDeltaNet on ``model``.

    :param model: Module tree that may contain GatedDeltaNet mixers.
    :param mixers: Dotted GatedDeltaNet class paths (``GdnMixerConfig.mixer``).
    :return: How many mixers were wrapped.
    """
    classes = mixer_classes(mixers)
    if not classes:
        return 0
    return sum(
        int(install_gdn_conv_clone(module))
        for module in model.modules()
        if isinstance(module, classes)
    )


def mixer_classes(paths: tuple[str, ...]) -> tuple[type, ...]:
    """Resolve dotted mixer class paths, skipping any that cannot be imported."""
    classes: list[type] = []
    for path in paths:
        module_path, _, class_name = path.rpartition(".")
        module = try_import(module_path)
        if module is None:
            continue
        mixer_cls = getattr(module, class_name, None)
        if mixer_cls is not None:
            classes.append(mixer_cls)
    return tuple(classes)


GATED_NORM_TARGETS = (
    ("transformers.models.qwen3_5_moe.modeling_qwen3_5_moe", "Qwen3_5MoeRMSNormGated"),
    ("transformers.models.qwen3_5.modeling_qwen3_5", "Qwen3_5RMSNormGated"),
)


def install_fused_gated_norm() -> None:
    """Point every Qwen GDN gated RMSNorm at fla's fused kernel when fla imports."""
    fused_norm_gate = try_import("fla.modules.fused_norm_gate")
    if fused_norm_gate is None:
        return
    rms_norm_gated = fused_norm_gate.rms_norm_gated

    def fused_gated_rms_norm_forward(
        self: nn.Module, hidden_states: torch.Tensor, gate: torch.Tensor
    ) -> torch.Tensor:
        return rms_norm_gated(
            hidden_states,
            gate,
            self.weight,
            None,
            activation=self.activation,
            eps=self.variance_epsilon,
        )

    for module_path, class_name in GATED_NORM_TARGETS:
        norm_cls = getattr(importlib.import_module(module_path), class_name)
        norm_cls.forward = fused_gated_rms_norm_forward
        logger.info("[qwen-gdn] %s runs fla's fused gated RMSNorm", class_name)
