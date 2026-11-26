# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""GSPO algorithm variant built on top of GRPO."""

from __future__ import annotations

from typing import Any

from agilerl.algorithms.grpo import GRPO
from agilerl.utils.algo_utils import inherit_init_signature


@inherit_init_signature(
    GRPO,
    fixed={
        "loss_type",
        "off_policy_token_mask_bounds",
        "off_policy_sequence_mask_threshold",
    },
)
class GSPO(GRPO):
    """GSPO loss variant of :class:`agilerl.algorithms.grpo.GRPO`.

    Paper: https://arxiv.org/abs/2507.18071

    The sequence-level ratio has no per-token weight, so the off-policy masks
    stay unset.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize a GSPO agent with the off-policy masks unset."""
        configured = [
            name
            for name in (
                "off_policy_token_mask_bounds",
                "off_policy_sequence_mask_threshold",
            )
            if name in kwargs and kwargs[name] is not None
        ]
        if configured:
            msg = "GSPO does not take " + " or ".join(configured)
            raise ValueError(msg)
        kwargs.pop("off_policy_token_mask_bounds", None)
        kwargs.pop("off_policy_sequence_mask_threshold", None)
        super().__init__(
            *args,
            loss_type="gspo",
            off_policy_token_mask_bounds=None,
            off_policy_sequence_mask_threshold=None,
            **kwargs,
        )
