# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

import pytest

from agilerl.arena.models.algorithms import PPOSpec, RainbowDQNSpec
from agilerl.utils.trainer_utils import create_population_from_spec


def test_num_envs_algo_without_env_raises():
    # PPO carries ``num_envs``; resolving it needs a live env.
    with pytest.raises(ValueError, match=r"requires an .*environment to resolve"):
        create_population_from_spec(1, PPOSpec(net_config=None), None, None, None)


def test_classic_rl_without_env_raises():
    with pytest.raises(ValueError, match="require an instantiated environment"):
        create_population_from_spec(
            1, RainbowDQNSpec(net_config=None), None, None, None
        )
