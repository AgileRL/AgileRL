# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""The strategy layer: which loop trains a contract spec, and with what."""

from typing import ClassVar

import pytest

from agilerl.arena.models.algorithms import (
    CQNSpec,
    DPOSpec,
    DQNSpec,
    GRPOSpec,
    IPPOSpec,
    LLMAlgorithmSpec,
    MADDPGSpec,
    NeuralTSSpec,
    PPOSpec,
    SFTSpec,
    SingleAgentAlgorithmSpec,
)
from agilerl.arena.models.registry import MANIFEST_REGISTRY
from agilerl.strategies import (
    BANDIT,
    LLM_DATASET,
    LLM_ROLLOUT,
    MULTI_AGENT_OFF_POLICY,
    MULTI_AGENT_ON_POLICY,
    OFFLINE,
    SINGLE_AGENT_OFF_POLICY,
    SINGLE_AGENT_ON_POLICY,
    TrainingStrategy,
    select_strategy,
)


class TestStrategyFor:
    @pytest.mark.parametrize(
        ("spec_cls", "expected"),
        [
            (PPOSpec, SINGLE_AGENT_ON_POLICY),
            (DQNSpec, SINGLE_AGENT_OFF_POLICY),
            (CQNSpec, OFFLINE),
            (NeuralTSSpec, BANDIT),
            (IPPOSpec, MULTI_AGENT_ON_POLICY),
            (MADDPGSpec, MULTI_AGENT_OFF_POLICY),
            (GRPOSpec, LLM_ROLLOUT),
            (DPOSpec, LLM_DATASET),
            (SFTSpec, LLM_DATASET),
        ],
    )
    def test_dispatches_on_the_contract_flags(self, spec_cls, expected):
        assert select_strategy(spec_cls.model_construct()) is expected

    def test_rejects_a_non_spec(self):
        with pytest.raises(TypeError, match="not an algorithm spec"):
            select_strategy(object())  # type: ignore[arg-type]

    def test_a_user_subclass_dispatches_like_its_parent(self):
        class MyPPOSpec(PPOSpec):
            pass

        assert select_strategy(MyPPOSpec()) is SINGLE_AGENT_ON_POLICY

    def test_a_flag_on_a_subclass_is_honoured(self):
        # The flags are what dispatch reads, so a subclass that flips one moves.
        class _OffSpec(SingleAgentAlgorithmSpec):
            offline: ClassVar[bool] = True

        assert select_strategy(_OffSpec()) is OFFLINE

    def test_a_bare_llm_base_has_no_strategy(self):
        with pytest.raises(KeyError, match="No training strategy"):
            select_strategy(LLMAlgorithmSpec.model_construct())

    def test_every_registered_spec_has_a_strategy(self):
        for spec_cls in dict(MANIFEST_REGISTRY.items()).values():
            assert isinstance(
                select_strategy(spec_cls.model_construct()), TrainingStrategy
            )


class TestTrainingLoops:
    def test_every_registered_spec_has_a_loop(self):
        for spec_cls in dict(MANIFEST_REGISTRY.items()).values():
            spec = spec_cls.model_construct()
            assert select_strategy(spec).get_training_loop(spec) is not None

    def test_each_paradigm_names_its_loop(self):
        from agilerl.training.train_bandits import train_bandits
        from agilerl.training.train_multi_agent_off_policy import (
            train_multi_agent_off_policy,
        )
        from agilerl.training.train_multi_agent_on_policy import (
            train_multi_agent_on_policy,
        )
        from agilerl.training.train_off_policy import train_off_policy
        from agilerl.training.train_offline import train_offline
        from agilerl.training.train_on_policy import train_on_policy

        assert SINGLE_AGENT_ON_POLICY.get_training_loop(PPOSpec()) is train_on_policy
        assert SINGLE_AGENT_OFF_POLICY.get_training_loop(DQNSpec()) is train_off_policy
        assert OFFLINE.get_training_loop(CQNSpec()) is train_offline
        assert BANDIT.get_training_loop(NeuralTSSpec()) is train_bandits
        assert (
            MULTI_AGENT_ON_POLICY.get_training_loop(IPPOSpec())
            is train_multi_agent_on_policy
        )
        assert (
            MULTI_AGENT_OFF_POLICY.get_training_loop(MADDPGSpec())
            is train_multi_agent_off_policy
        )

    def test_rollout_and_dataset_loops(self):
        from agilerl.training.llm import train_llm_dataset, train_llm_rollout

        assert (
            LLM_ROLLOUT.get_training_loop(GRPOSpec.model_construct())
            is train_llm_rollout
        )
        assert (
            LLM_DATASET.get_training_loop(DPOSpec.model_construct())
            is train_llm_dataset
        )
        assert (
            LLM_DATASET.get_training_loop(SFTSpec.model_construct())
            is train_llm_dataset
        )

    def test_a_user_subclass_trains_like_its_parent(self):
        from agilerl.training.train_on_policy import train_on_policy

        class MyPPOSpec(PPOSpec):
            pass

        spec = MyPPOSpec()
        assert select_strategy(spec).get_training_loop(spec) is train_on_policy


class TestStrategyDefensiveBranches:
    def test_a_strategy_without_a_loop_says_so(self):
        class LooplessStrategy(TrainingStrategy):
            def get_trainer_kwargs(
                self, spec, *, training, env_spec, memory=None, n_step_memory=None
            ):
                return {}

        with pytest.raises(NotImplementedError, match="must set default_loop"):
            LooplessStrategy().get_training_loop(PPOSpec())

    def test_multi_agent_on_policy_kwargs_carry_sum_scores(self):
        from agilerl.models.env import GymEnvSpec
        from agilerl.models.training import TrainingSpec

        kwargs = MULTI_AGENT_ON_POLICY.get_trainer_kwargs(
            IPPOSpec(),
            training=TrainingSpec(max_steps=10, sum_scores=False),
            env_spec=GymEnvSpec(name="mpe2.simple_spread_v3"),
        )
        assert kwargs["sum_scores"] is False
