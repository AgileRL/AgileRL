# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

import copy
import os
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, Mock, call, patch

import gymnasium as gym
import numpy as np
import pytest
import torch
from accelerate import Accelerator, DeepSpeedPlugin
from gymnasium import spaces
from peft import LoraConfig

from agilerl import HAS_DEEPSPEED, HAS_LLM_DEPENDENCIES, HAS_VLLM
from agilerl.algorithms import (
    CQN,
    DDPG,
    DQN,
    IPPO,
    MADDPG,
    MATD3,
    PPO,
    TD3,
    NeuralTS,
    NeuralUCB,
    RainbowDQN,
)
from agilerl.algorithms.core import EvolvableAlgorithm, LLMAlgorithm

if HAS_LLM_DEPENDENCIES or TYPE_CHECKING:
    from agilerl.algorithms import GRPO, LLMPPO, LLMREINFORCE
from agilerl.hpo.mutation import Mutations
from agilerl.hpo.tournament import TournamentSelection
from agilerl.typing import BatchDimension
from agilerl.utils.algo_utils import CosineLRScheduleConfig
from agilerl.utils.llm_utils import (
    aggregate_metrics_across_gpus,
    gather_tensor,
)
from agilerl.utils.utils import (
    calculate_vectorized_scores,
    consolidate_mutations,
    create_population,
    default_progress_bar,
    get_env_defined_actions,
    init_loggers,
    init_wandb,
    make_multi_agent_vect_envs,
    make_skill_vect_envs,
    make_vect_envs,
    print_hyperparams,
    resolve_selection_strategy,
    run_selection_and_mutation,
    save_llm_checkpoint,
    save_population_checkpoint,
    suppress_verbose_logging,
    tournament_selection_and_mutation,
)
from agilerl.wrappers.learning import Skill
from tests.helper_functions import (
    FakeSelectionAgent,
    make_multi_frequency_selection,
    new_agents,
)

create_module = None
if HAS_DEEPSPEED and HAS_VLLM:
    from tests.test_algorithms.test_llms.test_grpo import create_module

# Shared HP dict that can be used by any algorithm
SHARED_INIT_HP = {
    "POPULATION_SIZE": 4,
    "DOUBLE": True,
    "BATCH_SIZE": 128,
    "CUDAGRAPHS": False,
    "LR": 1e-3,
    "LR_ACTOR": 1e-4,
    "LR_CRITIC": 1e-3,
    "GAMMA": 0.99,
    "LEARN_STEP": 1,
    "TAU": 1e-3,
    "BETA": 0.4,
    "PRIOR_EPS": 0.000001,
    "NUM_ATOMS": 51,
    "V_MIN": 0,
    "V_MAX": 200,
    "N_STEP": 3,
    "POLICY_FREQ": 10,
    "GAE_LAMBDA": 0.95,
    "ACTION_STD_INIT": 0.6,
    "CLIP_COEF": 0.2,
    "ENT_COEF": 0.01,
    "VF_COEF": 0.5,
    "MAX_GRAD_NORM": 0.5,
    "TARGET_KL": None,
    "UPDATE_EPOCHS": 4,
    "AGENT_IDS": ["agent1", "agent2"],
    "LAMBDA": 1.0,
    "REG": 0.000625,
    "O_U_NOISE": True,
    "EXPL_NOISE": 0.1,
    "MEAN_NOISE": 0.0,
    "THETA": 0.15,
    "DT": 0.01,
}


class TestMakeVectEnvs:
    def test_requires_env_or_make_env(self):
        with pytest.raises(
            ValueError, match="Either env_name or make_env must be provided"
        ):
            make_vect_envs()

    def test_with_make_env(self):
        def make_env():
            return gym.make("CartPole-v1")

        env = make_vect_envs(make_env=make_env, num_envs=2)
        assert isinstance(env, gym.vector.AsyncVectorEnv)
        assert env.num_envs == 2

    def test_sync_vector(self):
        env = make_vect_envs("CartPole-v1", num_envs=2, should_async_vector=False)
        assert isinstance(env, gym.vector.SyncVectorEnv)
        assert env.num_envs == 2

    def test_extra_wrappers_applied(self):
        class MarkingWrapper(gym.Wrapper):
            def __init__(self, env):
                super().__init__(env)
                self.marked = True

        env = make_vect_envs(
            "CartPole-v1",
            num_envs=1,
            should_async_vector=False,
            extra_wrappers=[MarkingWrapper],
        )
        try:
            assert isinstance(env.envs[0], MarkingWrapper)
            assert env.envs[0].marked is True
        finally:
            env.close()

    # Returns an AsyncVectorEnv object when given a valid environment name and number of environments
    def test_returns_asyncvectorenv_object(self):
        num_envs = 3
        env = make_vect_envs("CartPole-v1", num_envs=num_envs)
        assert isinstance(env, gym.vector.AsyncVectorEnv)
        assert env.num_envs == num_envs


def test_suppress_verbose_logging():
    suppress_verbose_logging()


class TestDefaultProgressBar:
    def test_no_accelerator(self):
        bar = default_progress_bar(10, accelerator=None)
        assert bar.total == 10

    def test_with_accelerator(self):
        acc = Accelerator()
        bar = default_progress_bar(10, accelerator=acc)
        assert bar.total == 10


class TestGetEnvDefinedActions:
    def test_all_none(self):
        info = {"a": {"env_defined_action": None}, "b": {"env_defined_action": None}}
        assert get_env_defined_actions(info, ["a", "b"]) is None

    def test_some_defined(self):
        info = {"a": {"env_defined_action": 1}, "b": {"env_defined_action": None}}
        result = get_env_defined_actions(info, ["a", "b"])
        assert result == {"a": 1, "b": None}


def test_batch_dimension_repr():
    assert repr(BatchDimension()) == "BatchDimension"


class TestSavePopulationCheckpoint:
    def test_no_accelerator(self, tmp_path):
        pop = [
            MagicMock(spec=EvolvableAlgorithm),
            MagicMock(spec=EvolvableAlgorithm),
        ]
        for _i, agent in enumerate(pop):
            agent.steps = 200
            agent.save_checkpoint = MagicMock()
        save_path = str(tmp_path / "ckpt")
        save_population_checkpoint(pop, save_path, overwrite_checkpoints=True)
        assert pop[0].save_checkpoint.called
        assert pop[1].save_checkpoint.called

    def test_with_accelerator(self, tmp_path):
        pop = [MagicMock(spec=EvolvableAlgorithm), MagicMock(spec=EvolvableAlgorithm)]
        for agent in pop:
            agent.steps = 200
            agent.save_checkpoint = MagicMock()
            agent.unwrap_models = MagicMock()
            agent.wrap_models = MagicMock()
        accel = MagicMock(spec=Accelerator)
        accel.wait_for_everyone = MagicMock()
        accel.is_main_process = True
        save_path = str(tmp_path / "ckpt")
        save_population_checkpoint(
            pop, save_path, overwrite_checkpoints=True, accelerator=accel
        )
        assert accel.wait_for_everyone.call_count >= 3
        for agent in pop:
            agent.unwrap_models.assert_called()
            agent.wrap_models.assert_called()
            agent.save_checkpoint.assert_called_once()


# Returns an AsyncVectorEnv object when given a valid environment name and number of environments
def test_make_multi_agent_vect_envs_returns_asyncvectorenv_object():
    # ``speaker_listener_like_env`` mirrors the MPE speaker/listener API but
    # imports trivially, so workers spawn far faster than with the real
    # ``simple_speaker_listener_v4.parallel_env`` (which pulls PettingZoo
    # MPE + PyGame on each subprocess startup).
    from tests.pz_vector_test_utils import speaker_listener_like_env

    num_envs = 2
    env_kwargs = {"continuous_actions": False}
    env = make_multi_agent_vect_envs(
        speaker_listener_like_env, num_envs=num_envs, **env_kwargs
    )
    env.close()
    assert env.num_envs == num_envs


def test_make_multi_agent_vect_envs_extra_wrappers():
    from pettingzoo.utils.wrappers import BaseWrapper

    from agilerl.vector.pz_async_vec_env import AsyncPettingZooVecEnv
    from tests.pz_vector_test_utils import speaker_listener_like_env

    class MarkingWrapper(BaseWrapper):
        def __init__(self, env):
            super().__init__(env)
            self.marked = True

    with patch(
        "agilerl.utils.utils.AsyncPettingZooVecEnv",
        wraps=AsyncPettingZooVecEnv,
    ) as mock_cls:
        env = make_multi_agent_vect_envs(
            speaker_listener_like_env,
            num_envs=1,
            extra_wrappers=[MarkingWrapper],
            continuous_actions=False,
        )
        try:
            built = mock_cls.call_args[1]["env_fns"][0]()
            assert isinstance(built, MarkingWrapper)
            assert built.marked is True
        finally:
            env.close()


# Returns an AsyncVectorEnv object when given a valid environment name and number of environments
def test_make_skill_vect_envs_returns_asyncvectorenv_object():
    num_envs = 3
    skill = Skill
    env = make_skill_vect_envs("CartPole-v1", skill=skill, num_envs=num_envs)
    assert isinstance(env, gym.vector.AsyncVectorEnv)
    assert env.num_envs == num_envs


class TestCreatePopulation:
    # Can create a population of agent for each single agent algorithm
    def test_initial_population_single_agent(self):
        observation_space = spaces.Box(0, 1, shape=(4,))
        continuous_action_space = spaces.Box(0, 1, shape=(2,))
        discrete_action_space = spaces.Discrete(2)
        net_config = {"encoder_config": {"hidden_size": [8, 8]}}
        population_size = 4
        device = "cpu"
        accelerator = None

        algo_classes = {
            "DQN": DQN,
            "Rainbow DQN": RainbowDQN,
            "DDPG": DDPG,
            "TD3": TD3,
            "PPO": PPO,
            "CQN": CQN,
        }

        for algo, algo_cls in algo_classes.items():
            if algo in ["TD3", "DDPG"]:
                action_space = continuous_action_space
            else:
                action_space = discrete_action_space

            population = create_population(
                algo=algo,
                observation_space=observation_space,
                action_space=action_space,
                net_config=net_config,
                INIT_HP=SHARED_INIT_HP,
                population_size=population_size,
                device=device,
                accelerator=accelerator,
            )
            assert len(population) == population_size
            for agent in population:
                assert isinstance(agent, algo_cls)
                assert agent.observation_space == observation_space
                assert agent.action_space == action_space
                assert agent.device == "cpu"
                assert agent.accelerator is None

    # Can create a population of agent for bandit algorithms
    def test_initial_population_bandits(self):
        observation_space = spaces.Box(0, 1, shape=(4,))
        action_space = spaces.Discrete(5)
        net_config = {"encoder_config": {"hidden_size": [8]}}
        init_hp = {**SHARED_INIT_HP, "LAMBDA": 1.0, "REG": 0.000625}

        for algo, algo_cls in [("NeuralUCB", NeuralUCB), ("NeuralTS", NeuralTS)]:
            population = create_population(
                algo=algo,
                observation_space=observation_space,
                action_space=action_space,
                net_config=net_config,
                INIT_HP=init_hp,
                population_size=2,
                device="cpu",
            )
            assert len(population) == 2
            for agent in population:
                assert isinstance(agent, algo_cls)

    # Can create a population of agent for each multi agent algorithm
    def test_initial_population_multi_agent(self):
        observation_space = [spaces.Box(0, 1, shape=(4,)) for _ in range(2)]
        action_space = [spaces.Discrete(2) for _ in range(2)]
        net_config = {"encoder_config": {"hidden_size": [8], "min_mlp_nodes": 2}}
        population_size = 4
        device = "cpu"
        accelerator = None

        algo_classes = {
            "MADDPG": MADDPG,
            "MATD3": MATD3,
            "IPPO": IPPO,
        }

        for algo, algo_cls in algo_classes.items():
            population = create_population(
                algo=algo,
                observation_space=observation_space,
                action_space=action_space,
                net_config=net_config,
                INIT_HP=SHARED_INIT_HP,
                population_size=population_size,
                device=device,
                accelerator=accelerator,
            )
            assert len(population) == population_size
            for agent in population:
                assert isinstance(agent, algo_cls)
                assert agent.observation_spaces == observation_space
                assert agent.action_spaces == action_space
                assert agent.device == "cpu"
                assert agent.accelerator is None

    @pytest.mark.skipif(
        not (HAS_DEEPSPEED and HAS_VLLM),
        reason="Need to install agilerl with deepspeed + vllm",
    )
    @pytest.mark.parametrize(
        ("algo", "expected_type"),
        [
            ("GRPO", GRPO),
            ("LLMPPO", LLMPPO),
            ("llmppo", LLMPPO),
            ("LLMREINFORCE", LLMREINFORCE),
        ],
    )
    def test_llm_policy_gradient_algorithms(self, vector_space, algo, expected_type):
        init_hp = {
            "PAD_TOKEN_ID": 1000 - 1,
            "PAD_TOKEN": "<pad>",
            "BATCH_SIZE": 2,
            "BETA": 0.001,
            "LR": 0.001,
            "MAX_GRAD_NORM": 0.5,
            "UPDATE_EPOCHS": 1,
            "MAX_MODEL_LEN": 100,
            "GRADIENT_CHECKPOINTING": False,
        }
        device = "cuda" if torch.cuda.is_available() else "cpu"
        population_size = 1

        lora_kw = {
            "r": 16,
            "lora_alpha": 64,
            "target_modules": ["linear_1"],
            "task_type": "CAUSAL_LM",
            "lora_dropout": 0.05,
        }
        actor = create_module(
            input_size=10,
            max_tokens=20,
            vocab_size=1000,
            device=device,
        )
        common_kw = {
            "algo": algo,
            "observation_space": vector_space,
            "action_space": copy.deepcopy(vector_space),
            "net_config": None,
            "INIT_HP": init_hp,
            "hp_config": None,
            "population_size": population_size,
            "device": device,
            "accelerator": None,
            "actor_network": actor,
            "algo_kwargs": {
                "lora_config": LoraConfig(**lora_kw),
                "pad_token_id": 1000 - 1,
                "pad_token": "<pad>",
                "use_vllm": False,
            },
        }

        if expected_type is LLMPPO:
            mock_agent = MagicMock(spec=LLMPPO)
            with patch(
                "agilerl.utils.utils.LLMPPO", return_value=mock_agent
            ) as mock_cls:
                population = create_population(**common_kw)
            assert len(population) == population_size
            assert population[0] is mock_agent
            mock_cls.assert_called_once()
            call_kw = mock_cls.call_args.kwargs
            assert call_kw["batch_size"] == init_hp["BATCH_SIZE"]
            assert call_kw["beta"] == init_hp["BETA"]
            assert call_kw["vf_coef"] == SHARED_INIT_HP["VF_COEF"]
            assert call_kw["lr_actor"] == init_hp["LR"]
        else:
            population = create_population(**common_kw)
            assert len(population) == population_size
            for agent in population:
                assert isinstance(agent, expected_type)
                assert agent.accelerator is None
                assert agent.batch_size == init_hp["BATCH_SIZE"]

    @pytest.mark.skipif(
        not HAS_LLM_DEPENDENCIES,
        reason="agilerl[llm] not installed",
    )
    def test_llmppo_uses_clone_and_generation_defaults(self, vector_space):
        init_hp = {
            "BATCH_SIZE": 2,
            "LR": 7e-5,
            "BETA": 0.01,
            "MAX_GRAD_NORM": 0.5,
            "UPDATE_EPOCHS": 1,
            "MAX_MODEL_LEN": 96,
            "MAX_OUTPUT_TOKENS": 12,
            "USE_VLLM": True,
            "GRADIENT_CHECKPOINTING": False,
            "COSINE_lR_SCHEDULER": {"num_epochs": 10, "warmup_proportion": 0.1},
        }
        actor = MagicMock(name="actor_network")
        actor.state_dict.return_value = {"w": torch.tensor([1.0])}
        cloned_actor = MagicMock(name="cloned_actor")
        vllm_cfg = object()
        a0 = MagicMock(name="ppo_agent_0")
        a1 = MagicMock(name="ppo_agent_1")

        with (
            patch(
                "agilerl.utils.utils.clone_llm", return_value=cloned_actor
            ) as mock_clone,
            patch("agilerl.utils.utils.LLMPPO", side_effect=[a0, a1]) as mock_llmppo,
        ):
            population = create_population(
                algo="LLMPPO",
                observation_space=vector_space,
                action_space=copy.deepcopy(vector_space),
                net_config=None,
                INIT_HP=init_hp,
                hp_config=None,
                population_size=2,
                device="cpu",
                accelerator=None,
                actor_network=actor,
                vllm_config=vllm_cfg,
                algo_kwargs={"pad_token_id": 999, "pad_token": "<pad>"},
            )

        assert population == [a0, a1]
        mock_clone.assert_called_once()
        first_kw = mock_llmppo.call_args_list[0].kwargs
        second_kw = mock_llmppo.call_args_list[1].kwargs
        assert first_kw["actor_network"] is actor
        assert second_kw["actor_network"] is cloned_actor
        assert first_kw["use_vllm"] is True
        assert first_kw["vllm_config"] is vllm_cfg
        assert first_kw["lr_actor"] == init_hp["LR"]
        assert first_kw["cosine_lr_schedule_config"] is not None
        assert isinstance(first_kw["cosine_lr_schedule_config"], CosineLRScheduleConfig)

    @pytest.mark.skipif(
        not HAS_LLM_DEPENDENCIES,
        reason="agilerl[llm] not installed",
    )
    def test_llmreinforce_normalized_name_and_kwargs_overrides(
        self,
        vector_space,
    ):
        init_hp = {
            "BATCH_SIZE": 3,
            "LR": 5e-6,
            "BETA": 0.02,
            "MAX_GRAD_NORM": 0.7,
            "UPDATE_EPOCHS": 2,
            "MAX_MODEL_LEN": 80,
            "USE_VLLM": False,
            "GRADIENT_CHECKPOINTING": False,
            "COSINE_lR_SCHEDULER": {"num_epochs": 8, "warmup_proportion": 0.2},
        }
        actor = MagicMock(name="actor_network")
        actor.state_dict.return_value = {"w": torch.tensor([2.0])}
        cloned_actor = MagicMock(name="cloned_actor")
        pop0 = MagicMock(name="reinforce_agent_0")
        pop1 = MagicMock(name="reinforce_agent_1")
        global_vllm_cfg = object()
        local_vllm_cfg = object()

        with (
            patch(
                "agilerl.utils.utils.clone_llm", return_value=cloned_actor
            ) as mock_clone,
            patch(
                "agilerl.utils.utils.LLMREINFORCE",
                side_effect=[pop0, pop1],
            ) as mock_reinforce,
        ):
            population = create_population(
                algo="llmreinforce",
                observation_space=vector_space,
                action_space=copy.deepcopy(vector_space),
                net_config=None,
                INIT_HP=init_hp,
                hp_config=None,
                population_size=2,
                device="cpu",
                accelerator=None,
                actor_network=actor,
                vllm_config=global_vllm_cfg,
                torch_compiler="inductor",
                algo_kwargs={
                    "pad_token_id": 999,
                    "pad_token": "<pad>",
                    "use_vllm": True,
                    "vllm_config": local_vllm_cfg,
                },
            )

        assert population == [pop0, pop1]
        mock_clone.assert_called_once()
        first_kw = mock_reinforce.call_args_list[0].kwargs
        second_kw = mock_reinforce.call_args_list[1].kwargs
        assert first_kw["actor_network"] is actor
        assert second_kw["actor_network"] is cloned_actor
        assert first_kw["use_vllm"] is True
        assert first_kw["vllm_config"] is local_vllm_cfg
        assert first_kw["torch_compiler"] == "inductor"
        assert first_kw["lr"] == init_hp["LR"]
        assert isinstance(first_kw["cosine_lr_schedule_config"], CosineLRScheduleConfig)

    @pytest.mark.skipif(
        not HAS_LLM_DEPENDENCIES,
        reason="agilerl[llm] not installed",
    )
    def test_llmppo_uses_unique_per_agent_accelerators(self, vector_space):
        init_hp = {
            "BATCH_SIZE": 2,
            "LR": 7e-5,
            "BETA": 0.01,
            "MAX_GRAD_NORM": 0.5,
            "UPDATE_EPOCHS": 1,
            "MAX_MODEL_LEN": 96,
            "MAX_OUTPUT_TOKENS": 12,
            "USE_VLLM": False,
            "GRADIENT_CHECKPOINTING": False,
        }
        actor = MagicMock(name="actor_network")
        actor.state_dict.return_value = {"w": torch.tensor([1.0])}
        cloned_actor = MagicMock(name="cloned_actor")
        a0 = MagicMock(name="ppo_agent_0")
        a1 = MagicMock(name="ppo_agent_1")
        base_accelerator = MagicMock(name="base_accelerator")
        acc0 = MagicMock(name="agent_accel_0")
        acc1 = MagicMock(name="agent_accel_1")

        with (
            patch("agilerl.utils.utils.clone_llm", return_value=cloned_actor),
            patch(
                "agilerl.utils.utils.get_state_dict",
                return_value={"w": torch.tensor([1.0])},
            ),
            patch(
                "agilerl.utils.utils.get_llm_accelerator", side_effect=[acc0, acc1]
            ) as mock_get_accel,
            patch("agilerl.utils.utils.LLMPPO", side_effect=[a0, a1]) as mock_llmppo,
        ):
            population = create_population(
                algo="LLMPPO",
                observation_space=vector_space,
                action_space=copy.deepcopy(vector_space),
                net_config=None,
                INIT_HP=init_hp,
                hp_config=None,
                population_size=2,
                device="cpu",
                accelerator=base_accelerator,
                actor_network=actor,
                algo_kwargs={
                    "pad_token_id": 999,
                    "pad_token": "<pad>",
                    "use_vllm": False,
                },
            )

        assert population == [a0, a1]
        assert mock_get_accel.call_args_list == [
            call(base_accelerator, 0),
            call(base_accelerator, 1),
        ]
        first_kw = mock_llmppo.call_args_list[0].kwargs
        second_kw = mock_llmppo.call_args_list[1].kwargs
        assert first_kw["accelerator"] is acc0
        assert second_kw["accelerator"] is acc1

    @pytest.mark.skipif(
        not HAS_LLM_DEPENDENCIES,
        reason="agilerl[llm] not installed",
    )
    @pytest.mark.parametrize(
        ("algo", "patch_target"),
        [
            ("GRPO", "agilerl.utils.utils.GRPO"),
            ("LLMPPO", "agilerl.utils.utils.LLMPPO"),
            ("LLMREINFORCE", "agilerl.utils.utils.LLMREINFORCE"),
        ],
    )
    def test_llm_population_threads_liger_and_logprob_flags(
        self, vector_space, algo, patch_target
    ):
        """``USE_LIGER_LOSS`` / ``CAST_LOGPROBS_TO_FP32`` are forwarded from
        ``INIT_HP`` to the algo constructor for every LLM RL branch in
        ``create_population`` (GRPO/CISPO/GSPO, LLMPPO, LLMREINFORCE).
        """
        init_hp = {
            "BATCH_SIZE": 2,
            "LR": 1e-5,
            "BETA": 0.01,
            "MAX_GRAD_NORM": 0.5,
            "UPDATE_EPOCHS": 1,
            "MAX_MODEL_LEN": 64,
            "USE_VLLM": False,
            "GRADIENT_CHECKPOINTING": False,
            "USE_LIGER_LOSS": True,
            "CAST_LOGPROBS_TO_FP32": False,
        }
        actor = MagicMock(name="actor_network")
        actor.state_dict.return_value = {"w": torch.tensor([1.0])}
        mock_agent = MagicMock(name=f"{algo}_agent")

        with patch(patch_target, return_value=mock_agent) as mock_cls:
            create_population(
                algo=algo,
                observation_space=vector_space,
                action_space=copy.deepcopy(vector_space),
                net_config=None,
                INIT_HP=init_hp,
                hp_config=None,
                population_size=1,
                device="cpu",
                accelerator=None,
                actor_network=actor,
                algo_kwargs={
                    "pad_token_id": 999,
                    "pad_token": "<pad>",
                    "use_vllm": False,
                },
            )

        call_kw = mock_cls.call_args.kwargs
        assert call_kw["use_liger_loss"] is True
        assert call_kw["cast_logprobs_to_fp32"] is False

    @pytest.mark.skipif(
        not (HAS_DEEPSPEED and HAS_VLLM),
        reason="Need to install agilerl with deepspeed + vllm",
    )
    def test_sft_cpu(self):
        """Exercise ``create_population`` SFT branch (clone after first agent)."""
        pytest.importorskip("peft")
        from peft import LoraConfig

        from agilerl.algorithms.sft import SFT

        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["linear_1"],
            task_type="CAUSAL_LM",
            lora_dropout=0.05,
        )
        actor = create_module(5, 10, 30, "cpu")
        pop = create_population(
            algo="SFT",
            net_config=None,
            INIT_HP=SHARED_INIT_HP,
            population_size=2,
            actor_network=actor,
            algo_kwargs={
                "pad_token_id": 29,
                "pad_token": "<pad>",
                "lora_config": lora_config,
            },
        )
        assert len(pop) == 2
        assert all(isinstance(agent, SFT) for agent in pop)

    @pytest.mark.skipif(
        not (HAS_DEEPSPEED and HAS_VLLM),
        reason="Need to install agilerl with deepspeed + vllm",
    )
    def test_dpo_cpu(self):
        """Exercise ``create_population`` DPO branch (clone after first agent)."""
        pytest.importorskip("peft")
        from peft import LoraConfig

        from agilerl.algorithms.dpo import DPO

        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["linear_1"],
            task_type="CAUSAL_LM",
            lora_dropout=0.05,
        )
        actor = create_module(5, 10, 30, "cpu")
        pop = create_population(
            algo="DPO",
            net_config=None,
            INIT_HP=SHARED_INIT_HP,
            population_size=2,
            actor_network=actor,
            algo_kwargs={
                "pad_token_id": 29,
                "pad_token": "<pad>",
                "lora_config": lora_config,
            },
        )
        assert len(pop) == 2
        assert all(isinstance(agent, DPO) for agent in pop)


class TestCalculateVectorizedScores:
    # The function returns a list of episode rewards from the first episode in each parallel environment.
    def test_returns_list_of_episode_rewards(self):
        rewards = np.array([[1, 2, 3, 4, 5], [4, 5, 6, 7, 8]])
        terminations = np.array([[0, 0, 1, 0, 1], [0, 1, 0, 0, 0]])
        expected_rewards = [6, 9]

        result = calculate_vectorized_scores(
            rewards,
            terminations,
            include_unterminated=False,
            only_first_episode=True,
        )

        assert result == expected_rewards

    # The function returns a list of episode rewards including all episodes.
    def test_returns_list_of_episode_rewards_including_unterminated(self):
        rewards = np.array([[1, 2, 3], [4, 5, 6]])
        terminations = np.array([[0, 0, 1], [0, 1, 0]])
        expected_rewards = [6, 9, 6]

        result = calculate_vectorized_scores(
            rewards,
            terminations,
            include_unterminated=True,
            only_first_episode=False,
        )

        assert result == expected_rewards

    # The function returns a list of episode rewards including all terminated episodes.
    def test_returns_list_of_episode_rewards_all_terminated_episodes(self):
        rewards = np.array([[1, 2, 3, 4, 5], [4, 5, 6, 7, 8]])
        terminations = np.array([[0, 0, 1, 0, 1], [0, 1, 0, 0, 0]])
        expected_rewards = [6, 9, 9]

        result = calculate_vectorized_scores(
            rewards,
            terminations,
            include_unterminated=False,
            only_first_episode=False,
        )

        assert result == expected_rewards

    # The function returns a list of episode rewards including all terminated episodes.
    def test_returns_list_of_episode_rewards_including_all_terminated_episodes(self):
        rewards = np.array([[1, 2, 3, 4, 5], [4, 5, 6, 7, 8]])
        terminations = np.array([[0, 0, 1, 0, 1], [0, 1, 0, 0, 0]])
        expected_rewards = [6, 9, 9]

        result = calculate_vectorized_scores(
            rewards,
            terminations,
            include_unterminated=False,
            only_first_episode=False,
        )

        assert result == expected_rewards

    # The function returns a list of episode rewards containing no terminated episodes.
    def test_returns_list_of_episode_rewards_with_no_terminations(self):
        rewards = np.array([[1, 2, 3, 4, 5], [4, 5, 6, 7, 8]])
        terminations = np.array([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0]])
        expected_rewards = [15, 30]

        result = calculate_vectorized_scores(
            rewards,
            terminations,
            include_unterminated=True,
            only_first_episode=False,
        )

        assert result == expected_rewards


class TestPrintHyperparams:
    def test_empty_fitness(self):
        pop = create_population(
            algo="DQN",
            observation_space=spaces.Box(0, 1, shape=(4,)),
            action_space=spaces.Discrete(2),
            net_config={"encoder_config": {"hidden_size": [8]}},
            INIT_HP=SHARED_INIT_HP,
            population_size=1,
        )
        pop[0].fitness = []
        with patch("builtins.print") as mock_print:
            print_hyperparams(pop)
            call_args = mock_print.call_args[0][0]
            assert "nan" in call_args.lower() or "nan" in str(call_args)

    # The function prints the hyperparameters and fitnesses of all agents in the population.
    def test_prints_hyperparams(self):
        # Arrange
        observation_space = spaces.Box(0, 1, shape=(4,))
        action_space = spaces.Discrete(2)
        net_config = {"encoder_config": {"hidden_size": [8]}}
        population_size = 1
        device = "cpu"
        accelerator = None
        algo = "DQN"

        pop = create_population(
            algo=algo,
            observation_space=observation_space,
            action_space=action_space,
            net_config=net_config,
            INIT_HP=SHARED_INIT_HP,
            population_size=population_size,
            device=device,
            accelerator=accelerator,
        )

        # Manually set attributes
        pop[0].fitness = [1, 2, 3]
        pop[0].lr = 0.01
        pop[0].batch_size = 32

        agent = pop[0]
        mean_fitness = np.mean(agent.fitness[-5:]).item()
        attrs = EvolvableAlgorithm.inspect_attributes(agent)
        expected_lines = [
            f"Agent ID: {agent.index}  |  Mean 5 Fitness: {mean_fitness:.2f}",
            "Attributes:",
            *[f"  {k}: {v}" for k, v in sorted(attrs.items())],
        ]
        expected_output = "\n".join(expected_lines) + "\n"

        with patch("builtins.print") as mock_print:
            print_hyperparams(pop)
            mock_print.assert_called_once_with(expected_output)


class TestSaveLlmCheckpoint:
    def test_save_with_accelerator(self, tmp_path):
        """Test saving checkpoint when agent has an accelerator."""
        agent = Mock()
        agent.actor = Mock()
        agent.accelerator = Mock()
        agent.accelerator.wait_for_everyone = Mock()
        agent.algo = "grpo"
        save_llm_checkpoint(agent, str(tmp_path))
        agent.save_checkpoint.assert_called_once_with(str(tmp_path))
        agent.accelerator.wait_for_everyone.assert_called()

    def test_save_without_accelerator(self, tmp_path):
        """Test saving checkpoint when agent has no accelerator."""
        agent = Mock()
        agent.actor = Mock()
        agent.algo = "grpo"
        agent.accelerator = None
        save_llm_checkpoint(agent, str(tmp_path))
        agent.save_checkpoint.assert_called_once_with(str(tmp_path))

    def test_with_path(self, tmp_path):
        agent = Mock()
        agent.actor = Mock()
        agent.algo = "grpo"
        agent.accelerator = None
        path = str(tmp_path / "my_ckpt")
        save_llm_checkpoint(agent, path)
        agent.save_checkpoint.assert_called_once_with(path)


class TestInitWandb:
    def test_addl_args(self):
        with patch("agilerl.utils.utils.wandb") as mock_wandb:
            mock_wandb.api = MagicMock()
            init_wandb(
                algo="DQN",
                env_name="CartPole-v1",
                addl_args={"tags": ["test"]},
            )
            mock_wandb.init.assert_called_once()
            assert mock_wandb.init.call_args[1].get("tags") == ["test"]

    def test_default_name_generated_when_wandb_name_unset(self, monkeypatch):
        monkeypatch.delenv("WANDB_NAME", raising=False)
        with patch("agilerl.utils.utils.wandb") as mock_wandb:
            mock_wandb.api = MagicMock()
            init_wandb(algo="DQN", env_name="CartPole-v1")
            name = mock_wandb.init.call_args.kwargs["name"]
            assert name.startswith("CartPole-v1-EvoHPO-DQN-")

    def test_wandb_name_env_var_wins(self, monkeypatch):
        monkeypatch.setenv("WANDB_NAME", "my-run")
        with patch("agilerl.utils.utils.wandb") as mock_wandb:
            mock_wandb.api = MagicMock()
            init_wandb(algo="DQN", env_name="CartPole-v1")
            assert "name" not in mock_wandb.init.call_args.kwargs

    def test_no_api_warns(self, monkeypatch):
        monkeypatch.delenv("WANDB_API_KEY", raising=False)

        class FakeWandb:
            def init(self, **kwargs):
                pass

        wandb_no_api = FakeWandb()
        with patch("agilerl.utils.utils.wandb", wandb_no_api):
            with pytest.warns(UserWarning, match="API key"):
                init_wandb(algo="DQN", env_name="CartPole-v1")

    def test_with_accelerator_main_process(self):
        with patch("agilerl.utils.utils.wandb") as mock_wandb:
            mock_wandb.api = MagicMock()
            mock_accel = MagicMock(spec=Accelerator)
            mock_accel.is_main_process = True
            mock_accel.wait_for_everyone = Mock()
            init_wandb(algo="DQN", env_name="CartPole-v1", accelerator=mock_accel)
            mock_accel.wait_for_everyone.assert_called()
            mock_wandb.init.assert_called_once()

    def test_with_api_key(self):
        class NoApiWandb:
            login = Mock()
            init = Mock()

        with patch("agilerl.utils.utils.wandb", NoApiWandb):
            init_wandb(algo="DQN", env_name="CartPole-v1", wandb_api_key="test-key")
            NoApiWandb.login.assert_called_once_with(key="test-key")

    def test_mutation_hyperparams(self):
        with patch("agilerl.utils.utils.wandb") as mock_wandb:
            mock_wandb.api = MagicMock()
            init_wandb(
                algo="DQN",
                env_name="CartPole-v1",
                mutation_hyperparams={"MUT_1": 0.5},
            )
            assert mock_wandb.init.call_args[1]["config"].get("MUT_1") == 0.5


class TestInitLoggers:
    def test_tensorboard_and_csv_loggers(self, tmp_path):
        from agilerl.logger import CSVLogger, TensorboardLogger

        mock_sw = MagicMock()
        pbar = MagicMock()
        with patch("agilerl.logger.SummaryWriter", mock_sw):
            loggers = init_loggers(
                algo="PPO",
                env_name="CartPole-v1",
                pbar=pbar,
                verbose=False,
                tensorboard=True,
                csv=True,
                tensorboard_log_dir=str(tmp_path / "tb"),
                csv_log_dir=str(tmp_path / "csv"),
            )
        assert len(loggers) == 2
        assert isinstance(loggers[0], TensorboardLogger)
        assert isinstance(loggers[1], CSVLogger)
        mock_sw.assert_called_once()

    def test_csv_without_log_dir_raises(self):
        with pytest.raises(ValueError, match="csv_log_dir must be provided"):
            init_loggers(
                algo="PPO",
                env_name="CartPole-v1",
                pbar=MagicMock(),
                verbose=False,
                csv=True,
                csv_log_dir=None,
            )

    def test_stdout_logger_receives_accelerator(self):
        """StdOutLogger must get the accelerator so non-main ranks skip prints
        when report_metrics is called on every rank.
        """
        from agilerl.logger import StdOutLogger

        acc = MagicMock()
        pbar = MagicMock()
        loggers = init_loggers(
            algo="GRPO",
            env_name="gsm8k",
            pbar=pbar,
            verbose=True,
            accelerator=acc,
        )
        assert len(loggers) == 1
        assert isinstance(loggers[0], StdOutLogger)
        assert loggers[0]._accelerator is acc


class FakeAgent(FakeSelectionAgent):
    """The operator's agent stand-in plus the accelerator round-trip bookkeeping."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.unwrap_calls = 0
        self.wrap_calls = 0
        self.saved: list[str] = []
        self.loaded: list[str] = []

    def save_checkpoint(self, path):
        self.saved.append(path)
        super().save_checkpoint(path)

    def load_checkpoint(self, path):
        self.loaded.append(path)

    def unwrap_models(self):
        self.unwrap_calls += 1

    def wrap_models(self):
        self.wrap_calls += 1


class FakeAccelerator:
    """Minimal stand-in for a HuggingFace accelerator."""

    def __init__(self, is_main_process):
        self.is_main_process = is_main_process
        self.wait_count = 0

    def wait_for_everyone(self):
        self.wait_count += 1


class FakeMutations:
    mutate_elite = False

    def mutation(self, population, pre_training_mut=False, indices=None):
        return population


class FakeStrategy:
    """Minimal selection strategy exposing the unified select contract."""

    def __init__(self, elite, new_population, indices):
        self._result = (elite, new_population, indices)
        self.select_calls: list = []

    def select(self, population):
        self.select_calls.append(population)
        return self._result


class RecordingMutations:
    """Mutation stub that records the indices argument of every call."""

    mutate_elite = False

    def __init__(self, result=None):
        self._result = result
        self.indices_seen: list = []

    def mutation(self, population, pre_training_mut=False, indices=None):
        self.indices_seen.append(indices)
        return self._result if self._result is not None else population


def make_selection_population(subpop_fitnesses):
    """Build a population of the accelerator-aware FakeAgent with unique indices."""
    population = []
    idx = 0
    for subpop, fitnesses in subpop_fitnesses.items():
        for fit in fitnesses:
            population.append(FakeAgent(idx, subpop, fit))
            idx += 1
    return population


class TestRunSelectionAndMutation:
    def test_none_returns_population_unchanged(self):
        pop = [1, 2, 3]
        out = run_selection_and_mutation(
            None, population=pop, mutation=RecordingMutations(), env_name="env"
        )
        assert out is pop

    def test_selects_then_mutates_with_reported_indices(self):
        pop = [object()]
        evolved = [object(), object()]
        strategy = FakeStrategy(elite=None, new_population=evolved, indices=[7])
        mutation = RecordingMutations(result=["mutated"])

        out = run_selection_and_mutation(
            strategy, population=pop, mutation=mutation, env_name="env"
        )

        assert strategy.select_calls == [pop]
        assert mutation.indices_seen == [[7]]
        assert out == ["mutated"]

    def test_tournament_style_indices_none_mutates_whole_population(self):
        strategy = FakeStrategy(elite=None, new_population=[1], indices=None)
        mutation = RecordingMutations()

        run_selection_and_mutation(
            strategy, population=[1], mutation=mutation, env_name="env"
        )

        assert mutation.indices_seen == [None]

    def test_saves_elite_from_select_result(self, tmp_path):
        elite = FakeAgent(0, 0, 5.0)
        strategy = FakeStrategy(elite=elite, new_population=[elite], indices=None)
        elite_path = str(tmp_path / "elite.pt")

        run_selection_and_mutation(
            strategy,
            population=[elite],
            mutation=RecordingMutations(),
            env_name="env",
            save_elite=True,
            elite_path=elite_path,
        )

        assert elite.saved == [elite_path]

    def test_multi_frequency_language_model_dispatches_through_llm_branch(
        self, monkeypatch, tmp_path
    ):
        strategy = make_multi_frequency_selection()
        elite = MagicMock(spec=LLMAlgorithm)
        monkeypatch.setattr(strategy, "select", lambda pop: (elite, ["evolved"], [3]))
        saved: list = []
        monkeypatch.setattr(
            "agilerl.utils.utils.save_llm_checkpoint",
            lambda agent, path: saved.append((agent, path)),
        )
        mutation = RecordingMutations(result=["mutated"])
        elite_path = str(tmp_path / "elite")

        out = run_selection_and_mutation(
            strategy,
            population=[1],
            mutation=mutation,
            env_name="env",
            language_model=True,
            save_elite=True,
            elite_path=elite_path,
        )

        assert out == ["mutated"]
        assert mutation.indices_seen == [[3]]  # only the winner clones are perturbed
        assert saved == [(elite, elite_path)]

    def test_multi_frequency_language_model_consolidates_under_accelerator(
        self, monkeypatch
    ):
        strategy = make_multi_frequency_selection()
        monkeypatch.setattr(
            strategy,
            "select",
            lambda pop: (MagicMock(spec=LLMAlgorithm), ["evolved"], [3]),
        )
        consolidated: list = []
        monkeypatch.setattr(
            "agilerl.utils.utils.consolidate_mutations",
            lambda pop: consolidated.append(pop),
        )
        accelerator = FakeAccelerator(is_main_process=True)
        # consolidate_mutations only receives the LLMAlgorithm members
        mutated = MagicMock(spec=LLMAlgorithm)
        mutation = RecordingMutations(result=[mutated])

        run_selection_and_mutation(
            strategy,
            population=[1],
            mutation=mutation,
            env_name="env",
            language_model=True,
            accelerator=accelerator,
        )

        assert mutation.indices_seen == [[3]]
        assert consolidated == [[mutated]]  # mutation decisions broadcast to workers

    def test_no_accelerator(self):
        population = [MagicMock(spec=EvolvableAlgorithm) for _ in range(3)]
        for agent in population:
            agent.steps = 100
        tournament = MagicMock(spec=TournamentSelection)
        tournament.select = Mock(return_value=(population[0], population, None))
        mutation = MagicMock(spec=Mutations)
        mutation.mutation = Mock(return_value=population)
        result = run_selection_and_mutation(
            tournament,
            population=population,
            mutation=mutation,
            env_name="CartPole-v1",
            algo="DQN",
        )
        tournament.select.assert_called_once()
        # A tournament reports indices=None
        mutation.mutation.assert_called_once_with(population, indices=None)
        assert len(result) == 3

    def test_worker_loads_checkpoint(self):
        """Worker process loads checkpoints saved by main process."""
        population = [MagicMock(spec=EvolvableAlgorithm) for _ in range(2)]
        for agent in population:
            agent.steps = 100
            agent.load_checkpoint = Mock()
            agent.unwrap_models = Mock()
            agent.wrap_models = Mock()
        tournament = MagicMock(spec=TournamentSelection)
        tournament.select = Mock(return_value=(population[0], population, None))
        mutation = MagicMock(spec=Mutations)
        mutation.mutation = Mock(return_value=population)
        accel = MagicMock(spec=Accelerator)
        accel.wait_for_everyone = Mock()
        accel.is_main_process = False

        with patch("agilerl.utils.utils.Path") as mock_path:
            mock_path.return_value.mkdir = Mock()
            run_selection_and_mutation(
                tournament,
                population=population,
                mutation=mutation,
                env_name="CartPole-v1",
                algo="DQN",
                accelerator=accel,
            )
        for agent in population:
            agent.load_checkpoint.assert_called()

    def test_save_elite_with_path(self):
        population = [MagicMock(spec=EvolvableAlgorithm) for _ in range(2)]
        elite = population[0]
        elite.steps = 100
        elite.save_checkpoint = Mock()
        tournament = MagicMock(spec=TournamentSelection)
        tournament.select = Mock(return_value=(elite, population, None))
        mutation = MagicMock(spec=Mutations)
        mutation.mutation = Mock(return_value=population)
        run_selection_and_mutation(
            tournament,
            population=population,
            mutation=mutation,
            env_name="CartPole-v1",
            algo="DQN",
            elite_path="/tmp/elite",
            save_elite=True,
        )
        elite.save_checkpoint.assert_called_once_with("/tmp/elite.pt")

    def test_language_model(self):
        """Test run_selection_and_mutation with a language model population."""
        population = [MagicMock(spec=LLMAlgorithm) for _ in range(3)]
        for agent in population:
            agent.mut = "lr"
            agent.lr = 0.01
            agent.optimizer = Mock()
            agent.optimizer.param_groups = [{"lr": 0.01}]
            agent.accelerator = MagicMock(spec=Accelerator)
            agent.actor = MagicMock()
            agent.actor.save_checkpoint = Mock()
        tournament = MagicMock(spec=TournamentSelection)
        mutation = MagicMock(spec=Mutations)
        mutation.mutation = Mock(return_value=population)
        tournament.select = Mock(return_value=(population[0], population, None))
        env_name = "CartPole-v1"
        elite_path = None
        accelerator = MagicMock(spec=Accelerator)
        accelerator.is_main_process = True
        accelerator.wait_for_everyone = Mock()

        with (
            patch(
                "agilerl.utils.utils.save_llm_checkpoint"
            ) as mock_save_llm_checkpoint,
            patch(
                "agilerl.utils.utils.consolidate_mutations"
            ) as mock_consolidate_mutations,
        ):
            output_pop = run_selection_and_mutation(
                tournament,
                population=population,
                mutation=mutation,
                env_name=env_name,
                elite_path=elite_path,
                save_elite=True,
                accelerator=accelerator,
                language_model=True,
            )
            mock_save_llm_checkpoint.assert_called_once_with(population[0], elite_path)
            mock_consolidate_mutations.assert_called_once_with(output_pop)

        tournament.select.assert_called_once_with(population)
        mutation.mutation.assert_called_once_with(population, indices=None)
        accelerator.wait_for_everyone.assert_called()


class TestTournamentSelectionAndMutationDeprecatedShim:
    def test_forwards_to_run_selection_and_mutation_with_warning(self):
        population = [MagicMock(spec=EvolvableAlgorithm) for _ in range(3)]
        for agent in population:
            agent.steps = 100
        tournament = MagicMock(spec=TournamentSelection)
        tournament.select = Mock(return_value=(population[0], population, None))
        mutation = MagicMock(spec=Mutations)
        mutation.mutation = Mock(return_value=population)

        with pytest.warns(DeprecationWarning, match="deprecated"):
            result = tournament_selection_and_mutation(
                population, tournament, mutation, "CartPole-v1", algo="DQN"
            )

        tournament.select.assert_called_once()
        mutation.mutation.assert_called_once_with(population, indices=None)
        assert len(result) == 3


class TestRunSelectionAndMutationMultiFrequency:
    """The shared entry point orchestrates a real multi-frequency selection."""

    def test_orchestration_schedules_subpops_at_their_frequencies(self):
        strategy = make_multi_frequency_selection(
            n_subpop=3, population_size=12, ratios=[1, 2, 3]
        )
        pop = make_selection_population(
            {0: [4, 3, 2, 1], 1: [8, 7, 6, 5], 2: [12, 11, 10, 9]}
        )
        fired = []  # (cycle, subpop)

        for cycle in range(1, 7):
            new_pop = run_selection_and_mutation(
                strategy, population=pop, mutation=FakeMutations(), env_name="env"
            )
            fired.extend(
                (cycle, subpop)
                for subpop in sorted(
                    {a.subpopulation_id for a in new_agents(pop, new_pop)}
                )
            )
            pop = new_pop

        assert [c for c, s in fired if s == 0] == [1, 2, 3, 4, 5, 6]  # delta=1
        assert [c for c, s in fired if s == 1] == [2, 4, 6]  # delta=2
        assert [c for c, s in fired if s == 2] == [3, 6]  # delta=3

    def test_orchestration_saves_global_elite(self, tmp_path):
        strategy = make_multi_frequency_selection(
            n_subpop=2, population_size=8, ratios=[1, 2]
        )
        pop = make_selection_population({0: [4, 3, 2, 1], 1: [8, 7, 6, 5]})
        elite_path = str(tmp_path / "best.pt")

        run_selection_and_mutation(
            strategy,
            population=pop,
            mutation=FakeMutations(),
            env_name="env",
            save_elite=True,
            elite_path=elite_path,
        )

        assert os.path.exists(elite_path)

    def test_orchestration_accelerator_main_process_evolves_and_saves(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.chdir(tmp_path)
        strategy = make_multi_frequency_selection(
            n_subpop=2, population_size=8, ratios=[1, 2]
        )
        pop = make_selection_population({0: [4, 3, 2, 1], 1: [8, 7, 6, 5]})
        accel = FakeAccelerator(is_main_process=True)
        elite_path = str(tmp_path / "best.pt")

        out = run_selection_and_mutation(
            strategy,
            population=pop,
            mutation=FakeMutations(),
            env_name="env",
            accelerator=accel,
            save_elite=True,
            elite_path=elite_path,
        )

        # select() ran on the main process: subpop 0 (delta 1) fired and reset its counter,
        # subpop 1 (delta 2) has not fired yet
        assert strategy.counters == [0, 1]
        assert os.path.exists(elite_path)
        for agent in pop:
            assert agent.unwrap_calls == 1
        for agent in out:
            assert agent.wrap_calls == 1
            assert len(agent.saved) == 1
            assert agent.loaded == []
        assert accel.wait_count == 4

    def test_orchestration_accelerator_worker_loads_without_evolving(self):
        strategy = make_multi_frequency_selection(
            n_subpop=2, population_size=8, ratios=[1, 2]
        )
        pop = make_selection_population({0: [4, 3, 2, 1], 1: [8, 7, 6, 5]})
        accel = FakeAccelerator(is_main_process=False)

        out = run_selection_and_mutation(
            strategy,
            population=pop,
            mutation=FakeMutations(),
            env_name="env",
            accelerator=accel,
            algo="DQN",
        )

        assert out is pop
        assert strategy.counters == [0, 0]  # counters untouched -> select() did not run
        for i, agent in enumerate(out):
            assert agent.unwrap_calls == 1
            assert agent.wrap_calls == 1
            assert agent.loaded == [f"models/env/DQN_{i}.pt"]
            assert agent.saved == []
        assert accel.wait_count == 4

    def test_orchestration_assigns_missing_subpopulations(self):
        strategy = make_multi_frequency_selection(
            n_subpop=2, population_size=8, ratios=[1, 2]
        )
        pop = make_selection_population({None: [8, 7, 6, 5, 4, 3, 2, 1]})
        for agent in pop:
            agent.subpopulation_id = None

        run_selection_and_mutation(
            strategy, population=pop, mutation=FakeMutations(), env_name="env"
        )

        assert sorted(a.subpopulation_id for a in pop) == [0, 0, 0, 0, 1, 1, 1, 1]


class TestResolveSelectionStrategy:
    def test_resolve_prefers_new_argument_without_warning(self, recwarn):
        strategy = make_multi_frequency_selection()
        assert resolve_selection_strategy(strategy, None) is strategy
        assert len(recwarn) == 0

    def test_resolve_folds_deprecated_tournament_with_warning(self):
        tournament = TournamentSelection(
            tournament_size=2, elitism=True, population_size=4
        )
        with pytest.warns(DeprecationWarning, match="deprecated"):
            resolved = resolve_selection_strategy(None, tournament)
        assert resolved is tournament

    def test_resolve_conflict_prefers_selection_strategy(self):
        strategy = make_multi_frequency_selection()
        tournament = TournamentSelection(
            tournament_size=2, elitism=True, population_size=4
        )
        with pytest.warns(DeprecationWarning, match="deprecated"):
            resolved = resolve_selection_strategy(strategy, tournament)
        assert resolved is strategy


class TestGatherTensor:
    def test_with_tensor_input(self):
        """Test gather_tensor with tensor input"""
        accelerator = Accelerator()

        input_tensor = torch.tensor([1, 2, 3], device=accelerator.device)

        gathered = gather_tensor(input_tensor, accelerator)

        assert isinstance(gathered, torch.Tensor)

        assert torch.equal(gathered, input_tensor)

    def test_with_non_tensor_input(self):
        """Test gather_tensor with non-tensor input"""
        input_list = [1, 2, 3]

        accelerator = Accelerator()

        gathered = gather_tensor(input_list, accelerator)

        assert isinstance(gathered, torch.Tensor)

        assert torch.equal(gathered, torch.tensor(input_list).to(accelerator.device))

    def test_device(self):
        """Test that tensor is moved to accelerator device"""
        input_tensor = torch.tensor([1, 2, 3])

        accelerator = Accelerator()

        gathered = gather_tensor(input_tensor, accelerator)

        assert gathered.device.type == accelerator.device.type

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_distributed(self):
        """Test gather_tensor in distributed setting"""
        accelerator = Accelerator()

        rank = accelerator.process_index
        input_tensor = torch.tensor([rank], device=accelerator.device)

        gathered = gather_tensor(input_tensor, accelerator)

        assert len(gathered) == accelerator.num_processes
        assert torch.equal(
            gathered,
            torch.arange(accelerator.num_processes, device=accelerator.device),
        )


class TestAggregateMetricsAcrossGpus:
    def test_single_process(self):
        """Test aggregate_metrics_across_gpus with single process"""
        accelerator = Accelerator()

        metric_tensor = torch.tensor([1.0, 2.0, 3.0], device=accelerator.device)

        result = aggregate_metrics_across_gpus(accelerator, metric_tensor)

        assert result == 2.0  # (1 + 2 + 3) / 3 = 2.0
        assert isinstance(result, float)

    def test_with_scalar(self):
        """Test aggregate_metrics_across_gpus with scalar input"""
        accelerator = Accelerator()

        metric_tensor = torch.tensor(5.0, device=accelerator.device)

        result = aggregate_metrics_across_gpus(accelerator, metric_tensor)

        assert result == 5.0
        assert isinstance(result, float)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_distributed(self):
        """Test aggregate_metrics_across_gpus in distributed setting"""
        accelerator = Accelerator()

        rank = accelerator.process_index
        metric_tensor = torch.tensor([rank + 1.0], device=accelerator.device)

        result = aggregate_metrics_across_gpus(accelerator, metric_tensor)

        expected_mean = (
            sum(range(1, accelerator.num_processes + 1)) / accelerator.num_processes
        )
        assert (
            abs(result - expected_mean) < 1e-6
        )  # Allow for small floating point differences

    def test_with_negative_values(self):
        """Test aggregate_metrics_across_gpus with negative values"""
        accelerator = Accelerator()

        metric_tensor = torch.tensor([-1.0, -2.0, -3.0], device=accelerator.device)

        result = aggregate_metrics_across_gpus(accelerator, metric_tensor)

        assert result == -2.0  # (-1 + -2 + -3) / 3 = -2.0
        assert isinstance(result, float)

    def test_with_zero_values(self):
        """Test aggregate_metrics_across_gpus with zero values"""
        accelerator = Accelerator()

        metric_tensor = torch.tensor([0.0, 0.0, 0.0], device=accelerator.device)

        result = aggregate_metrics_across_gpus(accelerator, metric_tensor)

        assert result == 0.0
        assert isinstance(result, float)


class TestConsolidateMutations:
    def test_warning_if_not_llm_algorithm(self):
        """Test consolidate_mutations"""
        population = [Mock() for _ in range(3)]
        with pytest.warns(UserWarning, match="Consolidate mutations is only supported"):
            consolidate_mutations(population)

    @pytest.mark.skipif(
        not HAS_LLM_DEPENDENCIES, reason="LLM dependencies not installed"
    )
    def test_consolidate_mutations(self):
        population = [MagicMock(spec=GRPO) for _ in range(3)]
        for agent in population:
            agent.mut = "lr"
            agent.lr = 0.01
            agent.lr_critic = None
            agent.optimizer = Mock()
            agent.optimizer.param_groups = [{"lr": 0.01}]
            agent.cosine_lr_schedule_config = {"warmup_steps": 0, "total_steps": 100}
            agent.accelerator = MagicMock(spec=Accelerator)
            agent.accelerator.is_main_process = True
            agent.accelerator.wait_for_everyone = Mock()
            agent.accelerator.state = MagicMock()
            agent.accelerator.state.deepspeed_plugin = MagicMock(spec=DeepSpeedPlugin)
            agent.accelerator.state.deepspeed_plugin.deepspeed_config = {}
            agent.actor = MagicMock()
        consolidate_mutations(population)
        for agent in population:
            assert agent.mut == "lr"
            assert agent.lr == 0.01
            assert agent.optimizer.param_groups[0]["lr"] == 0.01


def test_check_box2d_available_raises_when_box2d_missing(monkeypatch):
    """Covers the ImportError path when Box2D is required but not installed."""
    import builtins

    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "Box2D":
            raise ImportError
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    from agilerl.utils.utils import _check_box2d_available

    with pytest.raises(ImportError, match="Box2D physics engine"):
        _check_box2d_available("LunarLander-v2")

    real_import = builtins.__import__


class TestCheckBox2dAvailable:
    def test_non_box2d_env_short_circuits(self):
        from agilerl.utils.utils import _check_box2d_available

        # No exception even with no Box2D installed because the prefix doesn't match.
        _check_box2d_available("CartPole-v1")


class TestNormalizeAlgoName:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("grpo", "GRPO"),
            ("LLM-PPO", "LLM_PPO"),
            ("llm reinforce", "LLMREINFORCE"),
            ("Dpo", "DPO"),
        ],
    )
    def test_normalizes_case_and_separators(self, raw, expected):
        from agilerl.utils.utils import _normalize_algo_name

        assert _normalize_algo_name(raw) == expected


class TestPrepareLlmAlgoKwargs:
    def _init_hp(self, **overrides):
        base = {"BATCH_SIZE": 8}
        base.update(overrides)
        return base

    def test_tokenizer_defaults_apply_when_not_set(self):
        from agilerl.utils.utils import _prepare_llm_algo_kwargs

        tokenizer = MagicMock(pad_token_id=7, pad_token="<pad>")
        merged = _prepare_llm_algo_kwargs(
            {},
            tokenizer=tokenizer,
            model_name="foo/bar",
            lora_config=None,
            vllm_config=None,
            INIT_HP=self._init_hp(),
        )
        assert merged["pad_token_id"] == 7
        assert merged["pad_token"] == "<pad>"
        assert merged["model_name"] == "foo/bar"
        assert merged["use_vllm"] is False
        assert merged["use_separate_reference_adapter"] is True
        assert merged["micro_batch_size_per_gpu"] == 8

    def test_existing_kwargs_take_priority_over_init_hp(self):
        from agilerl.utils.utils import _prepare_llm_algo_kwargs

        merged = _prepare_llm_algo_kwargs(
            {"model_name": "explicit/model", "micro_batch_size_per_gpu": 1},
            tokenizer=None,
            model_name=None,
            lora_config=None,
            vllm_config=None,
            INIT_HP=self._init_hp(MODEL_NAME="ignored/name"),
        )
        assert merged["model_name"] == "explicit/model"
        assert merged["micro_batch_size_per_gpu"] == 1

    def test_dpo_path_skips_generation_defaults(self):
        from agilerl.utils.utils import _prepare_llm_algo_kwargs

        merged = _prepare_llm_algo_kwargs(
            {},
            tokenizer=None,
            model_name="foo/bar",
            lora_config=None,
            vllm_config=MagicMock(),
            INIT_HP=self._init_hp(),
            with_generation_defaults=False,
        )
        assert "use_vllm" not in merged
        assert "vllm_config" not in merged
        assert merged["use_separate_reference_adapter"] is False

    def test_attn_implementation_injected_into_model_config(self):
        """A non-"auto" ATTN_IMPLEMENTATION is written to model_config so the
        algorithm's create_model treats it as authoritative.
        """
        from agilerl.utils.utils import _prepare_llm_algo_kwargs

        merged = _prepare_llm_algo_kwargs(
            {},
            tokenizer=None,
            model_name="foo",
            lora_config=None,
            vllm_config=None,
            INIT_HP=self._init_hp(ATTN_IMPLEMENTATION="flash_attention_2"),
        )
        assert merged["model_config"] == {"attn_implementation": "flash_attention_2"}

    @pytest.mark.parametrize("attn_impl", ["auto", None])
    def test_attn_implementation_auto_or_absent_leaves_model_config_alone(
        self, attn_impl
    ):
        r"""\"auto\" (or no key) must not create model_config - the algorithm's
        auto-pick path stays in charge.
        """
        from agilerl.utils.utils import _prepare_llm_algo_kwargs

        init_hp = self._init_hp()
        if attn_impl is not None:
            init_hp["ATTN_IMPLEMENTATION"] = attn_impl
        merged = _prepare_llm_algo_kwargs(
            {},
            tokenizer=None,
            model_name="foo",
            lora_config=None,
            vllm_config=None,
            INIT_HP=init_hp,
        )
        assert "model_config" not in merged

    def test_attn_implementation_does_not_override_explicit_model_config(self):
        """A caller-supplied model_config attn_implementation wins over the
        INIT_HP value; sibling model_config keys are preserved.
        """
        from agilerl.utils.utils import _prepare_llm_algo_kwargs

        merged = _prepare_llm_algo_kwargs(
            {"model_config": {"attn_implementation": "sdpa", "use_cache": False}},
            tokenizer=None,
            model_name="foo",
            lora_config=None,
            vllm_config=None,
            INIT_HP=self._init_hp(ATTN_IMPLEMENTATION="flash_attention_2"),
        )
        assert merged["model_config"]["attn_implementation"] == "sdpa"
        assert merged["model_config"]["use_cache"] is False


class TestValidateLlmKwargs:
    def test_raises_when_pad_token_missing(self):
        from agilerl.utils.utils import _validate_llm_kwargs

        with pytest.raises(ValueError, match="pad_token_id and pad_token"):
            _validate_llm_kwargs({"model_name": "x"}, actor_network=None)

    def test_raises_when_no_model_or_network(self):
        from agilerl.utils.utils import _validate_llm_kwargs

        with pytest.raises(ValueError, match="model_name or actor_network"):
            _validate_llm_kwargs(
                {"pad_token_id": 0, "pad_token": "<pad>"},
                actor_network=None,
            )

    def test_accepts_actor_network_without_model_name(self):
        from agilerl.utils.utils import _validate_llm_kwargs

        # No raise.
        _validate_llm_kwargs(
            {"pad_token_id": 0, "pad_token": "<pad>"},
            actor_network=MagicMock(),
        )


class TestLoraConfigFromInitHp:
    """Cover the ``_lora_config_from_init_hp`` helper branches.

    The default tests pass a fully-built ``lora_config`` so the helper's
    string-normalization and HAS_LLM_DEPENDENCIES guard are otherwise unhit.
    """

    def test_returns_none_when_no_modules(self):
        from agilerl.utils.utils import _lora_config_from_init_hp

        assert _lora_config_from_init_hp({}) is None

    def test_returns_none_when_llm_deps_missing(self):
        from agilerl.utils import utils as utils_mod

        with patch.object(utils_mod, "HAS_LLM_DEPENDENCIES", False):
            assert (
                utils_mod._lora_config_from_init_hp({"LORA_TARGET_MODULES": "linear_1"})
                is None
            )

    @pytest.mark.skipif(
        not HAS_LLM_DEPENDENCIES, reason="LoraConfig requires agilerl[llm]."
    )
    def test_string_modules_are_wrapped_into_list(self):
        from agilerl.utils.utils import _lora_config_from_init_hp

        cfg = _lora_config_from_init_hp({"LORA_TARGET_MODULES": "linear_1"})
        assert cfg is not None
        assert list(cfg.target_modules) == ["linear_1"]

    @pytest.mark.skipif(
        not HAS_LLM_DEPENDENCIES, reason="LoraConfig requires agilerl[llm]."
    )
    def test_target_modules_alias_is_accepted(self):
        from agilerl.utils.utils import _lora_config_from_init_hp

        cfg = _lora_config_from_init_hp(
            {"TARGET_MODULES": ["q_proj", "k_proj"], "LORA_R": 4}
        )
        assert cfg is not None
        assert set(cfg.target_modules) == {"q_proj", "k_proj"}
        assert cfg.r == 4


@pytest.mark.skipif(
    not HAS_LLM_DEPENDENCIES, reason="LoraConfig requires agilerl[llm]."
)
class TestPrepareLlmAlgoKwargsLoraDefaults:
    def test_init_hp_lora_modules_build_default_lora_config(self):
        """When no ``lora_config`` is supplied and INIT_HP carries Lora keys, the
        helper should build a fresh ``LoraConfig`` and stash it under
        ``lora_config``.
        """
        from agilerl.utils.utils import _prepare_llm_algo_kwargs

        merged = _prepare_llm_algo_kwargs(
            {},
            tokenizer=None,
            model_name="foo/bar",
            lora_config=None,
            vllm_config=None,
            INIT_HP={
                "BATCH_SIZE": 4,
                "LORA_TARGET_MODULES": ["linear_1"],
                "LORA_R": 8,
            },
        )
        assert merged.get("lora_config") is not None
        assert list(merged["lora_config"].target_modules) == ["linear_1"]
        assert merged["lora_config"].r == 8

    def test_explicit_lora_config_takes_precedence_over_init_hp(self):
        from agilerl.utils.utils import _prepare_llm_algo_kwargs

        explicit = LoraConfig(
            r=16,
            target_modules=["explicit"],
            task_type="CAUSAL_LM",
        )
        merged = _prepare_llm_algo_kwargs(
            {},
            tokenizer=None,
            model_name="foo/bar",
            lora_config=explicit,
            vllm_config=None,
            INIT_HP={
                "BATCH_SIZE": 4,
                "LORA_TARGET_MODULES": ["from_init_hp"],
            },
        )
        # The explicit config is preserved; the INIT_HP fallback is not used.
        assert merged["lora_config"] is explicit


@pytest.mark.skipif(
    not (HAS_DEEPSPEED and HAS_VLLM),
    reason="Need to install agilerl with deepspeed + vllm",
)
class TestCreatePopulationLlmTorchCompiler:
    """``create_population`` should forward ``torch_compiler`` into every LLM
    branch's kwargs (GRPO/CISPO/GSPO, SFT, DPO, LLMPPO, LLMREINFORCE).
    """

    @pytest.fixture
    def actor(self):
        return create_module(input_size=5, max_tokens=10, vocab_size=30, device="cpu")

    @pytest.fixture
    def init_hp(self):
        return {
            "BATCH_SIZE": 2,
            "LR": 1e-5,
            "BETA": 0.01,
            "MAX_GRAD_NORM": 0.5,
            "UPDATE_EPOCHS": 1,
            "MAX_MODEL_LEN": 64,
            "USE_VLLM": False,
            "GRADIENT_CHECKPOINTING": False,
        }

    @pytest.mark.parametrize(
        ("algo", "patch_target"),
        [
            ("GRPO", "agilerl.utils.utils.GRPO"),
            ("CISPO", "agilerl.utils.utils.CISPO"),
            ("GSPO", "agilerl.utils.utils.GSPO"),
            ("SFT", "agilerl.utils.utils.SFT"),
            ("DPO", "agilerl.utils.utils.DPO"),
            ("LLMPPO", "agilerl.utils.utils.LLMPPO"),
            ("LLMREINFORCE", "agilerl.utils.utils.LLMREINFORCE"),
        ],
    )
    def test_torch_compiler_threaded_through(
        self, vector_space, actor, init_hp, algo, patch_target
    ):
        mock_agent = MagicMock(name=f"{algo}_agent")
        with patch(patch_target, return_value=mock_agent) as mock_cls:
            create_population(
                algo=algo,
                observation_space=vector_space,
                action_space=copy.deepcopy(vector_space),
                net_config=None,
                INIT_HP=init_hp,
                hp_config=None,
                population_size=1,
                device="cpu",
                accelerator=None,
                actor_network=actor,
                torch_compiler="inductor",
                algo_kwargs={
                    "pad_token_id": 29,
                    "pad_token": "<pad>",
                    "use_vllm": False,
                },
            )
        call_kw = mock_cls.call_args.kwargs
        assert call_kw["torch_compiler"] == "inductor"

    @pytest.mark.parametrize("algo", ["CISPO", "GSPO"])
    def test_cispo_gspo_drop_loss_type_before_construction(
        self, vector_space, actor, init_hp, algo
    ):
        """Both CISPO and GSPO sit under the GRPO branch but should never see
        ``loss_type`` forwarded; the branch must ``pop`` it before instantiating
        the algo (otherwise the constructor receives a duplicate ``loss_type``).
        """
        patch_target = f"agilerl.utils.utils.{algo}"
        mock_agent = MagicMock(name=f"{algo}_agent")
        with patch(patch_target, return_value=mock_agent) as mock_cls:
            create_population(
                algo=algo,
                observation_space=vector_space,
                action_space=copy.deepcopy(vector_space),
                net_config=None,
                INIT_HP={**init_hp, "LOSS_TYPE": "should-be-dropped"},
                hp_config=None,
                population_size=1,
                device="cpu",
                accelerator=None,
                actor_network=actor,
                algo_kwargs={
                    "pad_token_id": 29,
                    "pad_token": "<pad>",
                    "use_vllm": False,
                },
            )
        call_kw = mock_cls.call_args.kwargs
        assert "loss_type" not in call_kw


class TestCreatePopulationLlmDepGuard:
    """When ``agilerl[llm]`` is not installed, every LLM-algo branch in
    ``create_population`` should raise a clear ImportError instead of failing
    deep inside the algorithm import path.
    """

    @pytest.mark.parametrize(
        ("algo", "match"),
        [
            ("GRPO", "GRPO/CISPO/GSPO require optional LLM dependencies"),
            ("CISPO", "GRPO/CISPO/GSPO require optional LLM dependencies"),
            ("GSPO", "GRPO/CISPO/GSPO require optional LLM dependencies"),
            ("SFT", "SFT requires optional LLM dependencies"),
            ("DPO", "DPO requires optional LLM dependencies"),
            ("LLMPPO", "LLMPPO requires optional LLM dependencies"),
            ("LLMREINFORCE", "LLMREINFORCE requires optional LLM dependencies"),
        ],
    )
    def test_llm_branches_raise_when_dependencies_missing(
        self, algo, match, vector_space
    ):
        from agilerl.utils import utils as utils_mod

        with patch.object(utils_mod, "HAS_LLM_DEPENDENCIES", False):
            with pytest.raises(ImportError, match=match):
                utils_mod.create_population(
                    algo=algo,
                    observation_space=vector_space,
                    action_space=copy.deepcopy(vector_space),
                    net_config=None,
                    INIT_HP={"BATCH_SIZE": 2},
                    hp_config=None,
                    population_size=1,
                    device="cpu",
                    accelerator=None,
                    actor_network=None,
                )


class TestAggregateMetricsNoAccelerator:
    """Cover the ``accelerator is None`` branch of ``aggregate_metrics_across_gpus``
    and the polymorphic float/ndarray/tensor branches of ``safe_aggregate_metrics``.
    """

    def test_aggregate_with_none_accelerator_and_tensor(self):
        result = aggregate_metrics_across_gpus(None, torch.tensor([1.0, 2.0, 3.0]))
        assert result == pytest.approx(2.0)

    def test_aggregate_with_none_accelerator_and_scalar_passthrough(self):
        # When accelerator is None and the metric is a plain scalar, the helper
        # short-circuits and returns the scalar unchanged.
        result = aggregate_metrics_across_gpus(None, 1.5)
        assert result == 1.5

    def test_safe_aggregate_with_tensor_no_accelerator(self):
        from agilerl.utils.llm_utils import safe_aggregate_metrics

        result = safe_aggregate_metrics(None, torch.tensor([2.0, 4.0]))
        assert isinstance(result, float)
        assert result == pytest.approx(3.0)

    def test_safe_aggregate_with_ndarray_no_accelerator(self):
        from agilerl.utils.llm_utils import safe_aggregate_metrics

        result = safe_aggregate_metrics(None, np.array([3.0, 5.0, 7.0]))
        assert isinstance(result, float)
        assert result == pytest.approx(5.0)

    def test_safe_aggregate_with_plain_float_no_accelerator(self):
        from agilerl.utils.llm_utils import safe_aggregate_metrics

        assert safe_aggregate_metrics(None, 2.5) == 2.5

    def test_safe_aggregate_with_accelerator_delegates(self):
        from agilerl.utils.llm_utils import safe_aggregate_metrics

        accelerator = Accelerator()
        result = safe_aggregate_metrics(
            accelerator,
            torch.tensor([1.0, 3.0], device=accelerator.device),
        )
        assert result == pytest.approx(2.0)


class TestDistributedHelpers:
    """World size / rank helpers: Accelerate, torch.distributed, single-process."""

    def test_world_size_prefers_accelerator(self):
        from agilerl.utils.utils import _distributed_world_size

        accelerator = MagicMock(num_processes=4)
        assert _distributed_world_size(accelerator) == 4

    def test_rank_prefers_accelerator(self):
        from agilerl.utils.utils import _distributed_rank

        accelerator = MagicMock(process_index=2)
        assert _distributed_rank(accelerator) == 2

    def test_world_size_and_rank_fall_back_to_single_process(self):
        from agilerl.utils.utils import _distributed_rank, _distributed_world_size

        with patch("torch.distributed.is_available", return_value=False):
            assert _distributed_world_size(None) == 1
            assert _distributed_rank(None) == 0

    def test_topology_is_the_process_group_without_tensor_parallelism(self):
        from agilerl.utils.utils import data_parallel_topology

        accelerator = MagicMock(num_processes=4, process_index=2)
        assert data_parallel_topology(accelerator, 1) == (2, 4)

    @pytest.mark.parametrize(
        ("process_index", "expected_rank"),
        [(0, 0), (1, 0), (2, 1), (3, 1), (6, 3), (7, 3)],
    )
    def test_tensor_parallel_ranks_share_one_replica_index(
        self, process_index, expected_rank
    ):
        """Both processes of a TP pair must get the same shard, or they generate different data."""
        from agilerl.utils.utils import data_parallel_topology

        accelerator = MagicMock(num_processes=8, process_index=process_index)
        assert data_parallel_topology(accelerator, 2) == (expected_rank, 4)

    def test_a_replica_straddling_the_process_group_is_rejected(self):
        from agilerl.utils.utils import data_parallel_topology

        accelerator = MagicMock(num_processes=6, process_index=0)
        with pytest.raises(ValueError, match="does not divide"):
            data_parallel_topology(accelerator, 4)

    def test_world_size_and_rank_use_torch_distributed(self):
        from agilerl.utils.utils import _distributed_rank, _distributed_world_size

        with (
            patch("torch.distributed.is_available", return_value=True),
            patch("torch.distributed.is_initialized", return_value=True),
            patch("torch.distributed.get_world_size", return_value=8),
            patch("torch.distributed.get_rank", return_value=3),
        ):
            assert _distributed_world_size(None) == 8
            assert _distributed_rank(None) == 3
