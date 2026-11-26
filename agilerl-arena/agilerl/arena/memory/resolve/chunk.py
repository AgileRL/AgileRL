# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0
"""Fused-logprob chunk rows."""

from __future__ import annotations

# Bounds on fused-logprob chunk rows. Floor: ``lm_head`` is re-read once per
# chunk. Ceiling: the fp32 logit tile grows linearly with rows.
FUSED_CHUNK_ROWS_MIN = 128
FUSED_CHUNK_ROWS_MAX = 4096
FUSED_LOGIT_TILE_BYTES = 256 * 1024 * 1024


def resolve_chunk_rows(vocab_size: int) -> int:
    """Fused-logprob chunk rows for a 256 MiB fp32 logit tile.

    :param vocab_size: ``lm_head`` width.
    :return: The tile size clamped to 128..4096.
    """
    rows = FUSED_LOGIT_TILE_BYTES // max(1, vocab_size * 4)
    return max(FUSED_CHUNK_ROWS_MIN, min(FUSED_CHUNK_ROWS_MAX, rows))
