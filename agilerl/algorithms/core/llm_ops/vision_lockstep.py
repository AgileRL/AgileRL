# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Keep every rank's vision tower calls in step during a learn that has images on some rank.

FSDP2 shards vision blocks as their own units. A forward that skips the tower
skips their all-gathers, and with a trainable tower their gradient
reduce-scatters, while another rank's forward enters them, so both ranks hang.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import Any

import torch
from torch import nn
from transformers import PreTrainedModel
from transformers.utils import ModelOutput

from agilerl.algorithms.core.llm_ops.frozen_vision import VisionFeatureCache
from agilerl.distributed.process import allreduce_sum_ints, any_rank

__all__ = [
    "FILLER_METHOD",
    "PlannedImages",
    "VisionLockstep",
    "vision_language_module",
    "vision_lockstep",
]

#: Method a patched VL model class runs its vision tower on one filler image with.
FILLER_METHOD = "vision_filler_features"

ForwardOutput = torch.Tensor | tuple[torch.Tensor, ...] | ModelOutput
#: Cache keys of a planned forward's images, ``[]`` for none, or ``None`` for unkeyed images.
PlannedImages = list[int] | None


def vision_language_module(model: nn.Module) -> PreTrainedModel | None:
    """Submodule whose forward takes ``pixel_values`` and can run a filler image, or ``None``.

    Its forward runs the vision tower before calling its ``get_decoder()``.
    """
    return next(
        (
            module
            for module in model.modules()
            if isinstance(module, PreTrainedModel) and hasattr(module, FILLER_METHOD)
        ),
        None,
    )


def _tensors(value: object) -> Iterator[torch.Tensor]:
    """Every tensor in a nest of model outputs, tuples, lists and dicts."""
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _tensors(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _tensors(item)


def _plus_zero(output: ForwardOutput, zero: torch.Tensor) -> ForwardOutput:
    """``output`` with ``zero`` added to its first tensor."""
    if isinstance(output, torch.Tensor):
        return output + zero
    if isinstance(output, tuple):
        return (output[0] + zero, *output[1:])
    key = next(iter(output.keys()))
    output[key] = output[key] + zero
    return output


class VisionLockstep:
    """Whether each forward must run the vision tower on some rank, as every rank agrees.

    :param cache: Vision feature cache of the learn step.
    """

    def __init__(self, cache: VisionFeatureCache) -> None:
        self.cache = cache
        self.planned: list[PlannedImages] = []
        self.agreed_runs: list[bool] = []
        self.forwards = 0

    @contextmanager
    def plan(self, forwards: Sequence[PlannedImages]) -> Iterator[None]:
        """Agree on the next forwards' tower calls in one all-reduce per run of new images; any later forward agrees alone.

        :param forwards: Images of each forward, in run order; every rank names as many.
        """
        self.planned, self.agreed_runs, self.forwards = list(forwards), [], 0
        try:
            self._agree()
            yield
        finally:
            self.planned, self.agreed_runs, self.forwards = [], [], 0

    def reuses(self, forward: int) -> bool:
        """Whether every rank already agreed that planned forward ``forward`` reuses cached features.

        :param forward: Index of the forward in the plan.
        """
        return forward < len(self.agreed_runs) and not self.agreed_runs[forward]

    def must_run(self, has_images: bool) -> bool:
        """Whether any rank must run the tower on the forward starting now.

        :param has_images: Whether this rank's forward has images.
        """
        if self.forwards == len(self.planned):
            return any_rank(has_images and not self.cache.covers())
        if self.forwards == len(self.agreed_runs):
            self._agree()
        self.forwards += 1
        return self.agreed_runs[self.forwards - 1]

    def _agree(self) -> None:
        """Agree on the next planned forwards, up to one whose images an earlier one stores."""
        remaining = self.planned[len(self.agreed_runs) :]
        if not remaining:
            return
        stored: set[int] = set()
        needs: list[int] = []
        repeats: list[int] = []
        for images in remaining:
            repeats.append(int(images is not None and not stored.isdisjoint(images)))
            covered = bool(images) and self.cache.reuse_dtype(images) is not None
            needs.append(int(images is None or (bool(images) and not covered)))
            if images and not covered and self.cache.tower is not None:
                stored.update(images)
        totals = allreduce_sum_ints(needs + repeats)
        count = len(remaining)
        end = next((k for k in range(1, count) if totals[count + k]), count)
        self.agreed_runs.extend(total > 0 for total in totals[:end])


@contextmanager
def vision_lockstep(
    module: PreTrainedModel, cache: VisionFeatureCache
) -> Iterator[VisionLockstep]:
    """Run ``module``'s vision tower on every rank's forward, or on none.

    Each forward, the ranks agree whether any must run the tower: one has
    images that ``cache`` lacks (:class:`VisionLockstep`). If none must, ranks
    with images reuse their cached features. Otherwise ranks with images run
    the tower, and a forward without images runs it on one zero image and adds
    zero times its features to the output, so the tower's forward and backward
    run there too. An empty ``pixel_values`` counts as no images.

    The zero image runs as ``module.get_decoder()`` starts, where a forward
    with images has already run the tower, so every rank gathers FSDP units
    computed before the tower (Gemma4 per-layer embeddings) in the same order.

    :param module: Module :func:`vision_language_module` returns.
    :param cache: Vision feature cache of the learn step.
    :return: The agreement, which callers can :meth:`VisionLockstep.plan` forwards on.
    """
    lockstep = VisionLockstep(cache)
    filler_devices: list[torch.device] = []
    pending: list[object] = []

    def agree(
        _module: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> tuple[tuple[Any, ...], dict[str, Any]] | None:
        pixel_values = kwargs.get("pixel_values")
        has_images = pixel_values is not None and pixel_values.shape[0] > 0
        if not has_images:
            kwargs["pixel_values"] = None
        if not lockstep.must_run(has_images):
            cache.agreed = "reuse"
            return args, kwargs
        cache.agreed = "run"
        if has_images:
            return args, kwargs
        anchor = kwargs.get("input_ids")
        if anchor is None:
            anchor = kwargs["inputs_embeds"]
        filler_devices.append(anchor.device)
        return args, kwargs

    def run_filler(_module: nn.Module, _args: tuple[Any, ...]) -> None:
        if not filler_devices:
            return
        with cache.images(None):
            pending.append(getattr(module, FILLER_METHOD)(filler_devices.pop()))

    def add_filler(
        _module: nn.Module,
        _args: tuple[Any, ...],
        _kwargs: dict[str, Any],
        output: ForwardOutput,
    ) -> ForwardOutput | None:
        cache.agreed = None
        if not pending:
            return None
        features = [
            tensor.sum().float()
            for tensor in _tensors(pending.pop())
            if tensor.is_floating_point()
        ]
        return _plus_zero(output, torch.stack(features).sum() * 0)

    handles = [
        module.register_forward_pre_hook(agree, with_kwargs=True),
        module.get_decoder().register_forward_pre_hook(run_filler),
        module.register_forward_hook(add_filler, with_kwargs=True),
    ]
    try:
        yield lockstep
    finally:
        cache.agreed = None
        for handle in handles:
            handle.remove()
