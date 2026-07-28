# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""GSPO algorithm variant built on top of GRPO."""

from __future__ import annotations

from typing import Any

from agilerl.algorithms.grpo import GRPO
from agilerl.utils.algo_utils import inherit_init_signature


@inherit_init_signature(GRPO, fixed={"loss_type"})
class GSPO(GRPO):
    """GSPO loss variant of :class:`agilerl.algorithms.grpo.GRPO`.

    Paper: https://arxiv.org/abs/2507.18071
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize a GSPO agent with fixed ``loss_type``."""
        super().__init__(*args, loss_type="gspo", **kwargs)
