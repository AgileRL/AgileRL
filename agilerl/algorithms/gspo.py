# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""GSPO algorithm variant built on top of GRPO."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from agilerl.algorithms.configs import GRPOObjective
from agilerl.algorithms.grpo import GRPO
from agilerl.utils.algo_utils import inherit_init_signature
from agilerl.utils.constructor_kwargs import assemble_init_kwargs


@inherit_init_signature(GRPO, fixed={"loss_type"})
class GSPO(GRPO):
    """GSPO loss variant of :class:`agilerl.algorithms.grpo.GRPO`.

    Paper: https://arxiv.org/abs/2507.18071
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize a GSPO agent with GSPO ``loss_type``."""
        assembled = assemble_init_kwargs(GRPO, args, kwargs)
        objective = assembled.get("objective") or GRPOObjective()
        assembled["objective"] = replace(objective, loss_type="gspo")
        super().__init__(**assembled)
