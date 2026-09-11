# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

from agilerl.algorithms.core import (
    MultiAgentAlgorithm,
    MultiAgentRLAlgorithm,
    RLAlgorithm,
    SingleAgentAlgorithm,
)


class TestAlgorithmNameAliases:
    def test_rl_algorithm_is_single_agent_algorithm(self) -> None:
        assert RLAlgorithm is SingleAgentAlgorithm

    def test_multi_agent_rl_algorithm_is_multi_agent_algorithm(self) -> None:
        assert MultiAgentRLAlgorithm is MultiAgentAlgorithm
