# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""INFO timing logs for FSDP2 forward, backward, and collectives."""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager

import torch
from torch import nn
from torch.distributed.fsdp import FSDPModule

from agilerl.distributed.process import get_rank

logger = logging.getLogger("agilerl.fsdp_profile")

# Thread-local so FSDP autograd hooks still see the Python stage name.
_STAGE = threading.local()

COMM_METHODS: tuple[tuple[str, str], ...] = (
    ("unshard", "all_gather_unshard"),
    ("wait_for_unshard", "wait_unshard"),
    ("reshard", "reshard"),
    ("post_backward", "reduce_scatter"),
)


def _sync_cuda() -> None:
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        torch.cuda.synchronize()


def _field_str(key: str, value: object) -> str | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return f"{key}={str(value).lower()}"
    if isinstance(value, float):
        return f"{key}={value:.1f}"
    return f"{key}={value}"


def log_fsdp_stage(stage: str, elapsed_ms: float, **fields: object) -> None:
    """Emit one ``fsdp_profile`` INFO line."""
    parts = [
        "fsdp_profile",
        f"stage={stage}",
        f"elapsed_ms={elapsed_ms:.1f}",
        f"rank={get_rank()}",
    ]
    for key, value in fields.items():
        piece = _field_str(key, value)
        if piece is not None:
            parts.append(piece)
    logger.info("%s", " ".join(parts))


@contextmanager
def timed_fsdp(stage: str, *, sync: bool = True, **fields: object) -> Iterator[None]:
    """Time a stage. ``sync=True`` waits for queued CUDA work first."""
    if sync:
        _sync_cuda()
    parent = getattr(_STAGE, "name", "")
    _STAGE.name = stage
    start = time.perf_counter()
    try:
        yield
    finally:
        if sync:
            _sync_cuda()
        extra = dict(fields)
        if parent:
            extra["parent"] = parent
        log_fsdp_stage(stage, (time.perf_counter() - start) * 1000.0, **extra)
        _STAGE.name = parent


def _wrap_named_method(owner: object, method_name: str, stage: str, unit: str) -> None:
    orig = getattr(owner, method_name, None)
    if orig is None or not callable(orig):
        return

    def wrapped(*args: object, **kwargs: object) -> object:
        with timed_fsdp(stage, sync=False, unit=unit):
            return orig(*args, **kwargs)

    setattr(owner, method_name, wrapped)


def install_fsdp_comm_timers(model: nn.Module) -> None:
    """Wrap FSDP2 unshard / reshard / reduce-scatter with INFO timers.

    Collectives are timed without an extra CUDA sync so prefetch can overlap.
    """
    seen: set[int] = set()
    for name, module in model.named_modules():
        if not isinstance(module, FSDPModule):
            continue
        unit = name or type(module).__name__
        try:
            group = module._get_fsdp_state()._fsdp_param_group
        except (AttributeError, RuntimeError):
            continue
        if group is None or id(group) in seen:
            continue
        seen.add(id(group))
        if getattr(group, "_agilerl_fsdp_profiled", False):
            continue
        for method_name, stage in COMM_METHODS:
            _wrap_named_method(group, method_name, stage, unit)
        group._agilerl_fsdp_profiled = True
