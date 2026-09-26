# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Recurrent PPO algorithm variant of PPO."""

from __future__ import annotations

from typing import Any

from agilerl.algorithms.ppo import PPO
from agilerl.utils.algo_utils import inherit_init_signature


@inherit_init_signature(PPO, defaults={"recurrent": True, "learn_step": 8192})
class RecurrentPPO(PPO):
    """PPO with a recurrent encoder."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs.setdefault("recurrent", True)
        kwargs.setdefault("learn_step", 8192)
        super().__init__(*args, **kwargs)
