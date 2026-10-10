# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Run frozen vision towers without autograd, and reuse their outputs within a learn step.

Hugging Face ``enable_input_require_grads`` (called by
``gradient_checkpointing_enable``) hooks the input embeddings of every
sub-model, the vision tower's patch embedding included. A tower with no
trainable parameter then still records autograd, saves its activations,
and is recomputed under activation checkpointing, although no gradient
it computes is ever used.

A learn step runs the same images through a frozen tower several times
(reference, old-policy and gradient forwards). :class:`VisionFeatureCache`
keeps each image's tower output for the rest of the step.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Hashable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial

import torch
import torch.nn as nn
from peft.tuners.tuners_utils import BaseTunerLayer
from torch.utils._pytree import TreeSpec, tree_flatten, tree_map_only, tree_unflatten
from transformers import PreTrainedModel

logger = logging.getLogger(__name__)

MODE_DEPENDENT_MODULES = (
    nn.BatchNorm1d,
    nn.BatchNorm2d,
    nn.BatchNorm3d,
    nn.SyncBatchNorm,
)
DROPOUT_MODULES = (
    nn.Dropout,
    nn.Dropout1d,
    nn.Dropout2d,
    nn.Dropout3d,
    nn.AlphaDropout,
    nn.FeatureAlphaDropout,
)
CALL_ARG_TYPES = (bool, int, float, str, type(None))


@dataclass(frozen=True)
class TowerCall:
    """Output layout of tower calls that share their non-image arguments.

    :param spec: Pytree structure of the output.
    :param device: Device the output tensors live on.
    """

    spec: TreeSpec
    device: torch.device


class VisionFeatureCache:
    """Frozen vision tower outputs of one learn step, one entry per image.

    Callers name the images of the next tower calls with :meth:`images`; keys
    only mean something inside one :meth:`step`. A call whose images all have
    entries for the same non-image arguments reuses them. Any other call runs
    the tower and stores each image's slice of every output tensor in host
    memory.

    :param mode_dependent: Whether train mode can change the tower output, so
        entries from eval-mode calls are kept apart from train-mode ones.
    """

    def __init__(self, mode_dependent: bool = True) -> None:
        self.mode_dependent = mode_dependent
        self.entries: dict[tuple[Hashable, int], list[torch.Tensor]] | None = None
        self.calls: dict[Hashable, TowerCall] = {}
        self.keys: list[int] | None = None

    @contextmanager
    def step(self) -> Iterator[None]:
        """Keep tower outputs until the context exits."""
        self.entries = {}
        self.calls = {}
        try:
            yield
        finally:
            self.entries = None
            self.calls = {}

    @contextmanager
    def images(self, keys: torch.Tensor | None) -> Iterator[None]:
        """Name the images of tower calls inside the context, in batch order.

        :param keys: ``(N,)`` integer key per image, or ``None`` to bypass the cache.
        :type keys: torch.Tensor | None
        """
        previous = self.keys
        self.keys = None if keys is None else keys.tolist()
        try:
            yield
        finally:
            self.keys = previous

    def run(
        self,
        tower: nn.Module,
        forward: Callable[..., object],
        args: tuple[object, ...],
        kwargs: dict[str, object],
    ) -> object:
        """Tower output for ``forward(*args, **kwargs)``, from the cache when every image has an entry."""
        entries = self.entries
        keys = self.keys
        pixel_values = args[0] if args else None
        if (
            entries is None
            or keys is None
            or not isinstance(pixel_values, torch.Tensor)
            or pixel_values.shape[0] != len(keys)
        ):
            return forward(*args, **kwargs)
        call = self._call_key(tower, args, kwargs)
        if call is None:
            return forward(*args, **kwargs)
        stored = [entries.get((call, key)) for key in keys]
        hits = [entry for entry in stored if entry is not None]
        layout = self.calls.get(call)
        if layout is not None and len(hits) == len(keys):
            return tree_unflatten(
                [
                    torch.cat(
                        [
                            entry[leaf].to(layout.device, non_blocking=True)
                            for entry in hits
                        ]
                    )
                    for leaf in range(len(hits[0]))
                ],
                layout.spec,
            )
        output = forward(*args, **kwargs)
        self._store(entries, call, keys, output)
        return output

    def _call_key(
        self,
        tower: nn.Module,
        args: tuple[object, ...],
        kwargs: dict[str, object],
    ) -> Hashable | None:
        """Key of a call's non-image arguments, or ``None`` when they cannot key a cache."""
        extra = (*args[1:], *kwargs.values())
        if not all(isinstance(value, CALL_ARG_TYPES) for value in extra):
            return None
        mode = tower.training if self.mode_dependent else None
        return (args[1:], tuple(sorted(kwargs.items())), mode)

    def _store(
        self,
        entries: dict[tuple[Hashable, int], list[torch.Tensor]],
        call: Hashable,
        keys: list[int],
        output: object,
    ) -> None:
        """Keep each image's slice of every output tensor in host memory."""
        leaves, spec = tree_flatten(output)
        if not leaves or not all(
            isinstance(leaf, torch.Tensor)
            and leaf.dim() > 0
            and leaf.shape[0] == len(keys)
            for leaf in leaves
        ):
            return
        device = leaves[0].device
        self.calls[call] = TowerCall(spec, device)
        for image, key in enumerate(keys):
            entries[(call, key)] = [
                _host_copy(leaf[image : image + 1]) for leaf in leaves
            ]


def _host_copy(tensor: torch.Tensor) -> torch.Tensor:
    """Copy of ``tensor`` in host memory, pinned and asynchronous for a CUDA source."""
    host = torch.empty(tensor.shape, dtype=tensor.dtype, pin_memory=tensor.is_cuda)
    return host.copy_(tensor, non_blocking=tensor.is_cuda)


def _output_depends_on_mode(tower: nn.Module) -> bool:
    """Whether train mode can change the tower output (batch norm, dropout, stochastic depth)."""
    if any(
        isinstance(module, MODE_DEPENDENT_MODULES)
        or (isinstance(module, DROPOUT_MODULES) and module.p > 0)
        for module in tower.modules()
    ):
        return True
    if not isinstance(tower, PreTrainedModel):
        return True
    # Functional dropout and stochastic depth only show up in the config.
    return any(
        ("dropout" in name or "drop_path" in name)
        and isinstance(rate, (int, float))
        and rate > 0
        for name, rate in tower.config.to_dict().items()
    )


def _no_grad_forward(
    forward: Callable[..., object],
    tower: nn.Module,
    cache: VisionFeatureCache | None,
    *args: object,
    **kwargs: object,
) -> object:
    """Run ``forward`` without autograd and detach every output tensor."""
    with torch.no_grad():
        output = (
            forward(*args, **kwargs)
            if cache is None
            else cache.run(tower, forward, args, kwargs)
        )
    return tree_map_only(torch.Tensor, torch.Tensor.detach, output)


def install_frozen_vision_no_grad(
    model: nn.Module, cache: VisionFeatureCache
) -> str | None:
    """Run the vision tower under ``torch.no_grad`` when none of its params train.

    The tower is the outermost Hugging Face model's
    ``get_encoder(modality="image")``. A tower with any parameter that requires
    grad is left unchanged. A tower with adapter layers runs without
    ``cache``, since its output depends on the routed adapter. Idempotent.

    :param model: Actor (PEFT or value-head wrappers included).
    :type model: nn.Module
    :param cache: Cache the tower reuses its outputs from within a learn step.
        Its ``mode_dependent`` flag is set from the tower.
    :type cache: VisionFeatureCache
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
        logger.debug(
            "Vision tower %s has trainable params; it runs with autograd.", path
        )
        return None
    tower_cache = (
        None
        if any(isinstance(module, BaseTunerLayer) for module in tower.modules())
        else cache
    )
    if tower_cache is not None:
        tower_cache.mode_dependent = _output_depends_on_mode(tower)
    forward = tower.forward
    if isinstance(forward, partial) and forward.func is _no_grad_forward:
        if forward.args[2] is tower_cache:
            return path
        forward = forward.args[0]
    tower.forward = partial(_no_grad_forward, forward, tower, tower_cache)
    logger.debug("Vision tower %s is frozen; it runs without autograd.", path)
    return path
