# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""CUDA peak-memory logs for learn-phase boundaries.

Enable with ``AGILERL_LOG_GRAM_PHASES=1``. Each rank logs its local GPU.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterator
from contextlib import contextmanager

import torch

logger = logging.getLogger("agilerl.gram_phase")

GRAM_PHASE_LOG_ENV = "AGILERL_LOG_GRAM_PHASES"
_GIB = 1024**3


def gram_phase_logging_enabled() -> bool:
    """Return whether phase GRAM logs should emit."""
    return os.environ.get(GRAM_PHASE_LOG_ENV, "") == "1"


def _bytes_to_gib(num_bytes: int) -> float:
    return num_bytes / _GIB


def _device_index() -> int | None:
    if not torch.cuda.is_available():
        return None
    return torch.cuda.current_device()


def snapshot_gram(phase: str) -> None:
    """Log current allocated/reserved bytes (not a peak window)."""
    if not gram_phase_logging_enabled():
        return
    device = _device_index()
    if device is None:
        return
    torch.cuda.synchronize(device)
    logger.info(
        "GRAM snapshot phase=%s rank=%s local_rank=%s device=%s "
        "alloc_gb=%.4f reserved_gb=%.4f",
        phase,
        os.environ.get("RANK", "?"),
        os.environ.get("LOCAL_RANK", "?"),
        device,
        _bytes_to_gib(torch.cuda.memory_allocated(device)),
        _bytes_to_gib(torch.cuda.memory_reserved(device)),
    )


@contextmanager
def log_gram_phase(phase: str) -> Iterator[None]:
    """Reset CUDA peak stats, run the block, log peak and end-of-phase resident."""
    if not gram_phase_logging_enabled():
        yield
        return
    device = _device_index()
    if device is None:
        yield
        return
    torch.cuda.synchronize(device)
    before_alloc = torch.cuda.memory_allocated(device)
    before_reserved = torch.cuda.memory_reserved(device)
    torch.cuda.reset_peak_memory_stats(device)
    try:
        yield
    finally:
        torch.cuda.synchronize(device)
        logger.info(
            "GRAM phase=%s rank=%s local_rank=%s device=%s "
            "peak_alloc_gb=%.4f peak_reserved_gb=%.4f "
            "after_alloc_gb=%.4f after_reserved_gb=%.4f "
            "before_alloc_gb=%.4f before_reserved_gb=%.4f",
            phase,
            os.environ.get("RANK", "?"),
            os.environ.get("LOCAL_RANK", "?"),
            device,
            _bytes_to_gib(torch.cuda.max_memory_allocated(device)),
            _bytes_to_gib(torch.cuda.max_memory_reserved(device)),
            _bytes_to_gib(torch.cuda.memory_allocated(device)),
            _bytes_to_gib(torch.cuda.memory_reserved(device)),
            _bytes_to_gib(before_alloc),
            _bytes_to_gib(before_reserved),
        )
