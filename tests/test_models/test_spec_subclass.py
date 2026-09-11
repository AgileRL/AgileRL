# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Framework algorithm specs subclass the arena field models."""

from __future__ import annotations

import pytest

from agilerl import HAS_ARENA_DEPENDENCIES, HAS_LLM_DEPENDENCIES, AgentType
from agilerl.models.algorithms.dqn import DQNSpec
from agilerl.models.algorithms.neural_ts import NeuralTSSpec
from agilerl.models.algorithms.neural_ucb import NeuralUCBSpec
from agilerl.models.algorithms.ppo import PPOSpec
from agilerl.models.env_types import LLMEnvType

requires_arena = pytest.mark.skipif(
    not HAS_ARENA_DEPENDENCIES, reason="agilerl-arena is not installed"
)


@requires_arena
class TestFrameworkSpecSubclassesArena:
    def test_dqn_spec_subclasses_arena_and_keeps_construction(self) -> None:
        from agilerl.arena.models.algorithms.dqn import DQNSpec as ArenaDQNSpec

        assert issubclass(DQNSpec, ArenaDQNSpec)
        assert DQNSpec is not ArenaDQNSpec
        assert callable(DQNSpec.build_algorithm)

    def test_ppo_spec_subclasses_arena(self) -> None:
        from agilerl.arena.models.algorithms.ppo import PPOSpec as ArenaPPOSpec

        assert issubclass(PPOSpec, ArenaPPOSpec)
        assert PPOSpec is not ArenaPPOSpec
        assert PPOSpec.model_fields["learn_step"].default == 2048

    def test_neural_bandit_specs_keep_arena_learn_step_defaults(self) -> None:
        assert NeuralTSSpec.model_fields["learn_step"].default == 2
        assert NeuralUCBSpec.model_fields["learn_step"].default == 2

    def test_arena_dqn_spec_has_no_build_algorithm(self) -> None:
        from agilerl.arena.models.algorithms.dqn import DQNSpec as ArenaDQNSpec

        assert "build_algorithm" not in ArenaDQNSpec.__dict__


@requires_arena
class TestAgentTypeAndLLMEnvTypeIdentity:
    def test_agilerl_agent_type_is_arena_agent_type(self) -> None:
        from agilerl.arena import AgentType as ArenaAgentType

        assert AgentType is ArenaAgentType
        assert AgentType.OfflineAgent.value == "offline_agent"
        assert AgentType.BanditAgent.value == "bandit_agent"

    def test_llm_env_type_is_arena_enum(self) -> None:
        from agilerl.arena.models.env import LLMEnvType as ArenaLLMEnvType

        assert LLMEnvType is ArenaLLMEnvType


@requires_arena
class TestNetworkSpecIdentity:
    def test_encoder_and_actor_specs_are_arena_classes(self) -> None:
        from agilerl.arena.models.networks import MlpSpec as ArenaMlpSpec
        from agilerl.arena.models.networks import NetworkSpec as ArenaNetworkSpec
        from agilerl.arena.models.networks import (
            StochasticActorSpec as ArenaStochasticActorSpec,
        )
        from agilerl.models.networks import MlpSpec, NetworkSpec, StochasticActorSpec

        assert NetworkSpec is ArenaNetworkSpec
        assert MlpSpec is ArenaMlpSpec
        assert StochasticActorSpec is ArenaStochasticActorSpec

    def test_finetuning_network_spec_subclasses_arena(self) -> None:
        from agilerl.arena.models.networks import (
            FinetuningNetworkSpec as ArenaFinetuningNetworkSpec,
        )
        from agilerl.models.networks import FinetuningNetworkSpec

        assert issubclass(FinetuningNetworkSpec, ArenaFinetuningNetworkSpec)
        assert FinetuningNetworkSpec is not ArenaFinetuningNetworkSpec


@pytest.mark.skipif(not HAS_LLM_DEPENDENCIES, reason="agilerl[llm] is not installed")
class TestDPOSpecObjective:
    def test_dpo_spec_does_not_declare_objective(self) -> None:
        from agilerl.arena.models.algorithms.dpo import DPOSpec as ArenaDPOSpec
        from agilerl.models.algorithms.dpo import DPOSpec

        assert "objective" not in DPOSpec.__dict__
        assert "objective" not in ArenaDPOSpec.__dict__
        assert DPOSpec.objective is None
        assert ArenaDPOSpec.objective is None


@pytest.mark.skipif(not HAS_LLM_DEPENDENCIES, reason="agilerl[llm] is not installed")
class TestCISPOSpecClipCoef:
    def test_accepts_asymmetric_clip_bounds(self) -> None:
        from agilerl.models.algorithms.cispo import CISPOSpec

        spec = CISPOSpec(group_size=2, clip_coef=[0.8, 2.0])

        assert spec.clip_coef == [0.8, 2.0]


@requires_arena
class TestConstructionBasesOmitArenaFields:
    def test_rl_and_multi_agent_bases_have_no_net_config_field(self) -> None:
        from agilerl.models.algo import MultiAgentRLAlgorithmSpec, RLAlgorithmSpec

        assert "net_config" not in RLAlgorithmSpec.model_fields
        assert "net_config" not in MultiAgentRLAlgorithmSpec.model_fields

    def test_llm_base_has_no_arena_llm_fields(self) -> None:
        from agilerl.models.algo import LLMAlgorithmSpec

        for name in (
            "pretrained_model_name_or_path",
            "max_model_len",
            "seed",
            "env_type",
            "objective",
        ):
            assert name not in LLMAlgorithmSpec.model_fields


@pytest.mark.skipif(not HAS_LLM_DEPENDENCIES, reason="agilerl[llm] is not installed")
class TestSFTSpecObjective:
    def test_sft_spec_keeps_sft_objective(self) -> None:
        from agilerl.models.algorithms.sft import SFTSpec

        assert SFTSpec.objective == "sft"
