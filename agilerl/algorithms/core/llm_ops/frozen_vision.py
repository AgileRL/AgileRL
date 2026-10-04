# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Run frozen vision towers without autograd.

Hugging Face ``enable_input_require_grads`` (called by
``gradient_checkpointing_enable``) hooks the input embeddings of every
sub-model, the vision tower's patch embedding included. A tower with no
trainable parameter then still records autograd, saves its activations,
and is recomputed under activation checkpointing, although no gradient
it computes is ever used.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from functools import partial

import torch
import torch.nn as nn
from torch.utils._pytree import tree_map_only
from transformers import PreTrainedModel

logger = logging.getLogger(__name__)


def _no_grad_forward(
    forward: Callable[..., object], *args: object, **kwargs: object
) -> object:
    """Run ``forward`` without autograd and detach every output tensor."""
    with torch.no_grad():
        output = forward(*args, **kwargs)
    return tree_map_only(torch.Tensor, torch.Tensor.detach, output)


def install_frozen_vision_no_grad(model: nn.Module) -> str | None:
    """Run the vision tower under ``torch.no_grad`` when none of its params train.

    The tower is the outermost Hugging Face model's
    ``get_encoder(modality="image")``. A tower with any parameter that requires
    grad is left unchanged. Idempotent.

    :param model: Actor (PEFT or value-head wrappers included).
    :type model: nn.Module
    :return: Module path of the tower that runs without autograd, or ``None``.
    :rtype: str | None
    """
    hf_model = next(
        (module for module in model.modules() if isinstance(module, PreTrainedModel)),
        None,
    )
    if hf_model is None:
        return None
    tower = hf_model.get_encoder(modality="image")
    if tower is hf_model:
        return None
    path = next(name for name, module in model.named_modules() if module is tower)
    if any(param.requires_grad for param in tower.parameters()):
        logger.info(
            "Vision tower %s has trainable params; it runs with autograd.", path
        )
        return None
    if getattr(tower.forward, "func", None) is not _no_grad_forward:
        tower.forward = partial(_no_grad_forward, tower.forward)
    logger.info("Vision tower %s is frozen; it runs without autograd.", path)
    return path
