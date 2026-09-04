# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

import itertools
from collections import Counter
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import ANY, MagicMock, Mock, call, patch

import pytest
import torch
from accelerate import Accelerator

from agilerl import HAS_LLM_DEPENDENCIES

if not HAS_LLM_DEPENDENCIES:
    pytest.skip("LLM dependencies not installed", allow_module_level=True)

from agilerl.algorithms import DPO, GRPO, LLMPPO, LLMREINFORCE
from agilerl.algorithms.core import ActionResult
from agilerl.algorithms.core.base import MultiAgentAlgorithm
from agilerl.algorithms.sft import SFT
from agilerl.hpo.multi_frequency import MultiFrequencySelection
from agilerl.hpo.mutation import Mutations
from agilerl.population import Population
from agilerl.rollouts.on_policy import collect_rollouts_llm
from agilerl.training.llm import (
    train_llm_dataset,
    train_llm_rollout,
)
from agilerl.training.llm.rollout import _any_rank_empty_batch
from agilerl.utils.utils import run_selection_and_mutation
from tests.helper_functions import (
    rank_population_by_subpopulation,
    weakest_agent_index,
)

pytestmark = pytest.mark.llm


@contextmanager
def _population_init_skip_per_mock_class():
    """Bypass Population's homogeneous type() check for multiple MagicMock(spec=…) agents.

    Python 3.13+ assigns a distinct type object to each MagicMock(spec=GRPO|DPO|SFT)
    instance; those mocks still satisfy isinstance(..., GRPO|DPO|SFT) for train_llm.
    """

    def _init(self, agents, min_evo_steps=10, accelerator=None, loggers=None):
        if not agents:
            msg = "Population requires at least one agent."
            raise ValueError(msg)
        sample_agent = agents[0]
        self._agents = agents
        self.sample_agent = sample_agent
        self.min_evo_steps = min_evo_steps
        self.accelerator = accelerator
        self.loggers = loggers or []
        self.last_fitnesses = []
        self.evo_steps = 0
        self.is_multi_agent = all(
            isinstance(agent, MultiAgentAlgorithm) for agent in agents
        )
        self.additional_metric_names = self.sample_agent.metrics.additional_metrics
        self.nonscalar_metric_names = self.sample_agent.metrics.nonscalar_metrics
        self.agent_ids = (
            self.sample_agent.metrics.agent_ids if self.is_multi_agent else None
        )

    with patch.object(Population, "__init__", _init):
        yield


# ---------------------------------------------------------------------------
# Helpers: Reasoning / Preference / SFT mock agents (Population-compatible)
# ---------------------------------------------------------------------------


def _mock_grpo_agent(**overrides):
    """Build a mock GRPO agent with proper metrics interface."""
    agent = MagicMock(spec=GRPO)
    agent.algo = "GRPO"
    agent.fitness = [0.0]
    agent.local_rank = "0"
    agent.get_action.return_value = ActionResult(
        [torch.ones(1, 100) for _ in range(2)],
        Mock(),
        None,
    )
    agent.learn.return_value = (0.5, 0.2)
    agent.test.return_value = torch.tensor([0.8])
    agent.batch_size_per_process = 32
    agent.batch_size = 32
    agent.pretrained_model_name_or_path = "Qwen/Qwen2.5-0.5B"
    agent.lr = 0.01
    agent.index = 0
    agent.mut = None

    metrics = MagicMock()
    metrics.steps = 0
    metrics.steps_per_second = 0.0
    metrics.scores = []
    metrics.additional_metrics = [
        "loss",
        "kl",
        "mean_reward",
        "completion_length",
        "accuracy",
    ]
    metrics.nonscalar_metrics = []
    agent.metrics = metrics

    agent.registry = MagicMock()
    agent.registry.hp_config = MagicMock()
    agent.registry.hp_config.config = {"lr": 0.01, "batch_size": 32}
    agent.registry.hp_config.names.return_value = ["lr", "batch_size"]

    for key, val in overrides.items():
        setattr(agent, key, val)
    return agent


def _mock_dpo_agent(**overrides):
    """Build a mock DPO agent with proper metrics interface."""
    agent = MagicMock(spec=DPO)
    agent.algo = "DPO"
    agent.fitness = [0.0]
    agent.local_rank = "0"
    agent.learn.return_value = {
        "loss": 0.5,
        "chosen_reward": 0.2,
        "rejected_reward": 0.1,
    }
    agent.test.return_value = 0.87
    agent.batch_size_per_process = 32
    agent.batch_size = 32
    agent.pretrained_model_name_or_path = "Qwen/Qwen2.5-0.5B"
    agent.lr = 0.001
    agent.index = 0
    agent.mut = None

    metrics = MagicMock()
    metrics.steps = 0
    metrics.steps_per_second = 0.0
    metrics.scores = []
    metrics.additional_metrics = [
        "loss",
        "chosen_reward",
        "rejected_reward",
        "reward_margin",
    ]
    metrics.nonscalar_metrics = []
    agent.metrics = metrics

    agent.registry = MagicMock()
    agent.registry.hp_config = MagicMock()
    agent.registry.hp_config.config = {"lr": 0.001, "batch_size": 32}
    agent.registry.hp_config.names.return_value = ["lr", "batch_size"]

    for key, val in overrides.items():
        setattr(agent, key, val)
    return agent


def _mock_sft_agent(**overrides):
    """Build a mock SFT agent with proper metrics interface."""
    agent = MagicMock(spec=SFT)
    agent.algo = "SFT"
    agent.fitness = [0.0]
    agent.local_rank = "0"
    agent.learn.return_value = {"loss": 0.5, "perplexity": 1.65}
    agent.test.return_value = -0.4
    agent.batch_size_per_process = 32
    agent.batch_size = 32
    agent.pretrained_model_name_or_path = "Qwen/Qwen2.5-0.5B"
    agent.lr = 5e-5
    agent.index = 0
    agent.mut = None

    metrics = MagicMock()
    metrics.steps = 0
    metrics.steps_per_second = 0.0
    metrics.scores = []
    metrics.additional_metrics = ["loss", "perplexity"]
    metrics.nonscalar_metrics = []
    agent.metrics = metrics

    agent.registry = MagicMock()
    agent.registry.hp_config = MagicMock()
    agent.registry.hp_config.config = {"lr": 5e-5, "batch_size": 32}
    agent.registry.hp_config.names.return_value = ["lr", "batch_size"]

    for key, val in overrides.items():
        setattr(agent, key, val)
    return agent


def _preference_batch():
    """Minimal collated preference batch that satisfies is_preference_prompts."""
    ids = torch.ones(1, 4, dtype=torch.long)
    return {
        "prompt": ["prompt"],
        "prompt_lengths": [4],
        "chosen": ["chosen"],
        "rejected": ["rejected"],
        "chosen_input_ids": ids,
        "chosen_attention_mask": ids,
        "rejected_input_ids": ids,
        "rejected_attention_mask": ids,
    }


def _sft_batch():
    """Minimal collated SFT batch that satisfies is_sft_prompts."""
    ids = torch.ones(1, 4, dtype=torch.long)
    return {
        "prompt": ["prompt"],
        "prompt_lengths": [4],
        "response": ["response"],
        "input_ids": ids,
        "attention_mask": ids,
    }


class _IncrementingPreferenceEnv:
    """Stub DatasetEnv: rewind restarts at batch 0; otherwise yields the next batch."""

    def __init__(self):
        self.name = "incrementing"
        self.data_batch_size_per_gpu = 1
        self.world_size = 1
        self.num_epochs = 0
        self._next_id = 0
        self.reset_dataloaders_calls: list[bool] = []

    def __len__(self) -> int:
        return 4

    def reset(self, reset_dataloaders: bool = False) -> dict:
        self.reset_dataloaders_calls.append(reset_dataloaders)
        if reset_dataloaders:
            self._next_id = 0
        batch_id = self._next_id
        self._next_id += 1
        batch = _preference_batch()
        batch["chosen_input_ids"] = torch.full((1, 4), batch_id, dtype=torch.long)
        return batch


# ---------------------------------------------------------------------------
# Helpers: Rollout mock agents and environments
# ---------------------------------------------------------------------------


def _make_rollout_mock_agent(*, spec=LLMPPO):
    """Build a rollout agent mock (LLMPPO/LLMREINFORCE/GRPO)."""
    mock_agent = MagicMock(spec=spec)
    mock_agent.fitness = [0.0]
    if spec is LLMPPO:
        mock_agent.algo = "LLMPPO"
    elif spec is LLMREINFORCE:
        mock_agent.algo = "LLMREINFORCE"
    elif spec is GRPO:
        mock_agent.algo = "GRPO"
        mock_agent.group_size = 1
    else:
        mock_agent.algo = getattr(spec, "__name__", "MOCK")

    mock_agent.learn.return_value = {
        "loss": 0.5,
        "kl": 0.2,
        "pg_loss": 0.1,
        "vf_loss": 0.1,
        "entropy": 1.0,
    }
    mock_agent.batch_size = 16
    mock_agent.batch_size_per_process = 16
    mock_agent.max_model_len = 1024
    mock_agent.pretrained_model_name_or_path = "Qwen/Qwen2.5-0.5B"
    mock_agent.lr = 0.01
    mock_agent.index = 0
    mock_agent.mut = 0
    mock_agent.seed = 42
    mock_agent.device = torch.device("cpu")

    metrics = MagicMock()
    metrics.steps = 0
    metrics.steps_per_second = 0.0
    metrics.scores = []
    metrics.additional_metrics = [
        "loss",
        "kl",
        "mean_reward",
        "completion_length",
        "accuracy",
    ]
    metrics.nonscalar_metrics = []
    mock_agent.metrics = metrics

    mock_agent.registry = MagicMock()
    mock_agent.registry.hp_config = MagicMock()
    mock_agent.registry.hp_config.config = {"lr": 0.01, "batch_size": 16}
    mock_agent.registry.hp_config.names.return_value = ["lr", "batch_size"]

    return mock_agent


def _rollout_collect_return(*, batch_steps=3, n_trajectories=1, seq_len=8):
    """Standard return value for a mocked collect_rollouts_llm call.

    Masks and turn ids are ``seq_len - 1``: they describe the transitions between
    tokens, not the tokens, and :class:`Trajectory` rejects any other relationship.
    """
    return (
        [torch.ones(1, seq_len, dtype=torch.long) for _ in range(n_trajectories)],
        [torch.ones(1, seq_len - 1, dtype=torch.bool) for _ in range(n_trajectories)],
        [torch.zeros(1, seq_len - 1, dtype=torch.long) for _ in range(n_trajectories)],
        [torch.ones(2, dtype=torch.float32) for _ in range(n_trajectories)],
        batch_steps,
        42,
        None,  # all_sampling_logps
    )


# ---------------------------------------------------------------------------
# TestFinetuneLlmReasoning
# ---------------------------------------------------------------------------


class TestTrainLlmDatasetPreference:
    def _pref_env(self, *, length=6):
        mock_env = MagicMock()
        mock_env.__len__.return_value = length
        example = _preference_batch()
        mock_env.reset.return_value = example
        mock_env.step.return_value = example
        mock_env.data_batch_size_per_gpu = 1
        mock_env.world_size = 1
        return mock_env

    def test_train_llm_dataset_preference_saves_elite_at_end(self):
        weaker = _mock_dpo_agent()
        weaker.fitness = [0.2]
        stronger = _mock_dpo_agent()
        stronger.fitness = [0.8]
        mock_env = self._pref_env(length=1)

        with (
            patch("agilerl.training.llm.dataset.default_progress_bar") as mock_pbar_fn,
            patch("agilerl.training.llm.dataset.save_llm_checkpoint") as mock_save,
            patch("agilerl.training.llm.dataset.init_loggers", return_value=[]),
            _population_init_skip_per_mock_class(),
        ):
            mock_pbar_fn.return_value = MagicMock()
            train_llm_dataset(
                pop=[weaker, stronger],
                env=mock_env,
                evaluation_interval=10,
                save_elite=True,
                elite_path="/tmp/dpo-elite",
            )

            assert mock_save.call_args_list[-1] == call(stronger, "/tmp/dpo-elite")

    @pytest.mark.parametrize("use_accelerator", [True, False])
    def test_train_llm_dataset_preference_basic_training_loop(self, use_accelerator):
        mock_agent = _mock_dpo_agent()
        mock_env = self._pref_env()

        with (
            patch("agilerl.training.llm.dataset.default_progress_bar") as mock_pbar_fn,
            patch("agilerl.training.llm.dataset.save_llm_checkpoint"),
            patch("agilerl.training.llm.dataset.init_loggers", return_value=[]),
        ):
            mock_pbar_fn.return_value = MagicMock()
            train_llm_dataset(
                pop=[mock_agent],
                env=mock_env,
                evaluation_interval=2,
                accelerator=None if use_accelerator else Accelerator(),
            )
            assert mock_env.reset.call_count == 6
            assert mock_agent.get_action.call_count == 0
            assert mock_env.step.call_count == 0
            assert mock_agent.learn.call_count == 6
            assert mock_agent.test.call_count == 3

    @pytest.mark.parametrize("use_accelerator", [True, False])
    def test_train_llm_dataset_preference_with_wandb_and_checkpoints(
        self, use_accelerator
    ):
        mock_agent = _mock_dpo_agent()
        mock_env = self._pref_env()

        with (
            patch("agilerl.training.llm.dataset.default_progress_bar") as mock_pbar_fn,
            patch("agilerl.training.llm.dataset.init_loggers") as mock_init_loggers,
            patch("agilerl.training.llm.dataset.save_llm_checkpoint") as mock_save,
        ):
            mock_pbar_fn.return_value = MagicMock()
            mock_init_loggers.return_value = []

            train_llm_dataset(
                pop=[mock_agent],
                env=mock_env,
                save_elite=True,
                wb=True,
                wandb_api_key="fake_key",
                evaluation_interval=3,
                accelerator=None if use_accelerator else Accelerator(),
                checkpoint_steps=6,
            )

            mock_init_loggers.assert_called_once()
            assert mock_init_loggers.call_args.kwargs["wb"] is True
            assert mock_save.call_count == 1
            assert mock_agent.test.call_count == 2

    @pytest.mark.parametrize("use_accelerator", [True, False])
    def test_train_llm_dataset_preference_evolvable_training_loop(
        self, use_accelerator
    ):
        mock_agent = _mock_dpo_agent()
        mock_env = self._pref_env()

        mutation = MagicMock()
        mutation.architecture_mut = 0
        mutation.new_layer_prob = 0
        mutation.parameters_mut = 0
        mutation.activation_mut = 0
        mutation = MagicMock()
        mutation.architecture_mut = 0
        mutation.new_layer_prob = 0
        mutation.parameters_mut = 0
        mutation.activation_mut = 0

        with (
            patch("agilerl.training.llm.dataset.default_progress_bar") as mock_pbar_fn,
            patch("agilerl.training.llm.dataset.save_llm_checkpoint"),
            patch("agilerl.training.llm.dataset.init_loggers", return_value=[]),
            patch(
                "agilerl.training.llm.dataset.run_selection_and_mutation"
            ) as mock_tsm,
        ):
            mock_pbar_fn.return_value = MagicMock()
            mock_tsm.return_value = [mock_agent]

            train_llm_dataset(
                pop=[mock_agent],
                env=mock_env,
                evaluation_interval=2,
                evo_steps=1,
                accelerator=None if use_accelerator else Accelerator(),
                selection_strategy=Mock(),
                mutation=mutation,
            )
            assert mock_env.reset.call_count == 6
            assert mock_env.step.call_count == 0
            assert mock_agent.learn.call_count == 6
            assert mock_agent.test.call_count == 3
            assert mock_tsm.call_count == 6

    def test_train_llm_dataset_preference_warning_num_epochs_and_max_steps(self):
        mock_agent = _mock_dpo_agent()
        mock_env = self._pref_env()

        with (
            patch("agilerl.training.llm.dataset.default_progress_bar") as mock_pbar_fn,
            patch("agilerl.training.llm.dataset.save_llm_checkpoint"),
            patch("agilerl.training.llm.dataset.init_loggers", return_value=[]),
            patch(
                "agilerl.training.llm.dataset.run_selection_and_mutation"
            ) as mock_tsm,
        ):
            mock_pbar_fn.return_value = MagicMock()
            mock_tsm.return_value = [mock_agent]

            with pytest.warns(UserWarning, match="num_epochs"):
                train_llm_dataset(
                    pop=[mock_agent],
                    env=mock_env,
                    evaluation_interval=2,
                    num_epochs=10,
                    max_steps=100,
                    evo_steps=None,
                )

    def test_train_llm_dataset_preference_break_on_num_epochs(self):
        mock_agent = _mock_dpo_agent()
        mock_env = self._pref_env()

        with (
            patch("agilerl.training.llm.dataset.default_progress_bar") as mock_pbar_fn,
            patch("agilerl.training.llm.dataset.init_loggers", return_value=[]),
            patch("agilerl.training.llm.dataset.save_llm_checkpoint"),
        ):
            mock_pbar_fn.return_value = MagicMock()
            mock_env.num_epochs = 2
            train_llm_dataset(
                pop=[mock_agent],
                env=mock_env,
                evaluation_interval=2,
                evo_steps=1,
                accelerator=None,
                num_epochs=2,
                checkpoint_steps=3,
            )

    def test_train_llm_dataset_preference_value_error_if_algo_not_dpo(self):
        mock_agent = MagicMock(spec=GRPO)
        mock_agent.algo = "GRPO"
        mock_agent.pretrained_model_name_or_path = "Qwen/Qwen2.5-0.5B"
        mock_agent.batch_size_per_process = 32
        mock_agent.batch_size = 32
        with pytest.raises(ValueError, match="DPO"):
            train_llm_dataset(
                pop=[mock_agent],
                env=MagicMock(),
                evaluation_interval=2,
                accelerator=None,
            )

    def test_train_llm_dataset_preference_env_fn_uses_distinct_env_instances(self):
        agent_a = _mock_dpo_agent(index=0)
        agent_b = _mock_dpo_agent(index=1)

        env_a = MagicMock()
        env_a.__len__.return_value = 1
        env_a.name = "env_a"
        env_a.data_batch_size_per_gpu = 1
        env_a.world_size = 1
        env_a.num_epochs = 0
        env_a.reset.return_value = _preference_batch()
        env_a.step.return_value = _preference_batch()

        env_b = MagicMock()
        env_b.__len__.return_value = 1
        env_b.name = "env_b"
        env_b.data_batch_size_per_gpu = 1
        env_b.world_size = 1
        env_b.num_epochs = 0
        env_b.reset.return_value = _preference_batch()
        env_b.step.return_value = _preference_batch()

        env_fn = MagicMock(side_effect=[env_a, env_b])

        with (
            _population_init_skip_per_mock_class(),
            patch("agilerl.training.llm.dataset.default_progress_bar") as mock_pbar_fn,
            patch(
                "agilerl.training.llm.rollout.safe_aggregate_metrics", return_value=0.5
            ),
            patch("agilerl.training.llm.dataset.save_llm_checkpoint"),
            patch("agilerl.training.llm.dataset.init_loggers", return_value=[]),
        ):
            mock_pbar_fn.return_value = MagicMock()
            train_llm_dataset(
                pop=[agent_a, agent_b],
                env_fn=env_fn,
                max_steps=2,
                evaluation_interval=100,
                verbose=False,
                accelerator=None,
            )

        assert env_fn.call_count == 2
        assert env_a.reset.call_count >= 1
        assert env_b.reset.call_count >= 1


# ---------------------------------------------------------------------------
# TestFinetuneLlmSft
# ---------------------------------------------------------------------------


class TestTrainLlmDatasetSft:
    def test_train_llm_dataset_sft_saves_elite_at_end(self):
        weaker = _mock_sft_agent()
        weaker.fitness = [0.2]
        stronger = _mock_sft_agent()
        stronger.fitness = [0.8]

        mock_env = MagicMock()
        mock_env.__len__.return_value = 1
        mock_env.reset.return_value = _sft_batch()
        mock_env.step.return_value = "next_prompts"
        mock_env.data_batch_size_per_gpu = 1
        mock_env.world_size = 1
        mock_env.num_epochs = 1

        with (
            patch("agilerl.training.llm.dataset.default_progress_bar") as mock_pbar_fn,
            patch("agilerl.training.llm.dataset.save_llm_checkpoint") as mock_save,
            patch("agilerl.training.llm.dataset.init_loggers", return_value=[]),
            _population_init_skip_per_mock_class(),
        ):
            mock_pbar_fn.return_value = MagicMock()
            train_llm_dataset(
                pop=[weaker, stronger],
                env=mock_env,
                evaluation_interval=10,
                save_elite=True,
                elite_path="/tmp/sft-elite",
            )

            assert mock_save.call_args_list[-1] == call(stronger, "/tmp/sft-elite")

    @pytest.mark.parametrize("use_accelerator", [True, False])
    def test_train_llm_dataset_sft_basic_training_loop(self, use_accelerator):
        mock_agent = _mock_sft_agent()

        mock_env = MagicMock()
        mock_env.__len__.return_value = 6
        mock_env.reset.return_value = _sft_batch()
        mock_env.step.return_value = "next_prompts"
        mock_env.data_batch_size_per_gpu = 1
        mock_env.world_size = 1

        with (
            patch("agilerl.training.llm.dataset.default_progress_bar") as mock_pbar_fn,
            patch("agilerl.training.llm.dataset.save_llm_checkpoint"),
            patch("agilerl.training.llm.dataset.init_loggers", return_value=[]),
        ):
            mock_pbar_fn.return_value = MagicMock()
            train_llm_dataset(
                pop=[mock_agent],
                env=mock_env,
                evaluation_interval=2,
                accelerator=None if use_accelerator else Accelerator(),
            )
            assert mock_env.reset.call_count == 6
            assert mock_env.step.call_count == 0
            assert mock_agent.learn.call_count == 6
            assert mock_agent.test.call_count == 3

    @pytest.mark.parametrize("use_accelerator", [True, False])
    def test_train_llm_dataset_sft_with_wandb_and_checkpoints(self, use_accelerator):
        mock_agent = _mock_sft_agent()

        mock_env = MagicMock()
        mock_env.__len__.return_value = 6
        mock_env.reset.return_value = _sft_batch()
        mock_env.step.return_value = "next_prompts"
        mock_env.data_batch_size_per_gpu = 1
        mock_env.world_size = 1

        with (
            patch("agilerl.training.llm.dataset.default_progress_bar") as mock_pbar_fn,
            patch("agilerl.training.llm.dataset.init_loggers") as mock_init_loggers,
            patch("agilerl.training.llm.dataset.save_llm_checkpoint") as mock_save,
        ):
            mock_pbar_fn.return_value = MagicMock()
            mock_init_loggers.return_value = []

            train_llm_dataset(
                pop=[mock_agent],
                env=mock_env,
                save_elite=True,
                wb=True,
                wandb_api_key="fake_key",
                evaluation_interval=3,
                accelerator=None if use_accelerator else Accelerator(),
                checkpoint_steps=6,
            )

            mock_init_loggers.assert_called_once()
            assert mock_init_loggers.call_args.kwargs["wb"] is True
            assert mock_save.call_count == 1
            assert mock_agent.test.call_count == 2

    @pytest.mark.parametrize("use_accelerator", [True, False])
    def test_train_llm_dataset_sft_evolvable_training_loop(self, use_accelerator):
        mock_agent = _mock_sft_agent()

        mock_env = MagicMock()
        mock_env.__len__.return_value = 6
        mock_env.reset.return_value = _sft_batch()
        mock_env.step.return_value = "next_prompts"
        mock_env.data_batch_size_per_gpu = 1
        mock_env.world_size = 1

        mutation = MagicMock()
        mutation.architecture_mut = 0
        mutation.new_layer_prob = 0
        mutation.parameters_mut = 0
        mutation.activation_mut = 0
        mutation = MagicMock()
        mutation.architecture_mut = 0
        mutation.new_layer_prob = 0
        mutation.parameters_mut = 0
        mutation.activation_mut = 0

        with (
            patch("agilerl.training.llm.dataset.default_progress_bar") as mock_pbar_fn,
            patch("agilerl.training.llm.dataset.save_llm_checkpoint"),
            patch("agilerl.training.llm.dataset.init_loggers", return_value=[]),
            patch(
                "agilerl.training.llm.dataset.run_selection_and_mutation"
            ) as mock_tsm,
        ):
            mock_pbar_fn.return_value = MagicMock()
            mock_tsm.return_value = [mock_agent]

            train_llm_dataset(
                pop=[mock_agent],
                env=mock_env,
                evaluation_interval=2,
                evo_steps=1,
                accelerator=None if use_accelerator else Accelerator(),
                selection_strategy=Mock(),
                mutation=mutation,
            )
            assert mock_env.reset.call_count == 6
            assert mock_env.step.call_count == 0
            assert mock_agent.learn.call_count == 6
            assert mock_agent.test.call_count == 3
            assert mock_tsm.call_count == 6

    def test_train_llm_dataset_sft_warning_num_epochs_and_max_steps(self):
        mock_agent = _mock_sft_agent()

        mock_env = MagicMock()
        mock_env.__len__.return_value = 6
        mock_env.reset.return_value = _sft_batch()
        mock_env.step.return_value = "next_prompts"
        mock_env.data_batch_size_per_gpu = 1
        mock_env.world_size = 1

        with (
            patch("agilerl.training.llm.dataset.default_progress_bar") as mock_pbar_fn,
            patch("agilerl.training.llm.dataset.save_llm_checkpoint"),
            patch("agilerl.training.llm.dataset.init_loggers", return_value=[]),
        ):
            mock_pbar_fn.return_value = MagicMock()
            with pytest.warns(UserWarning, match="num_epochs"):
                train_llm_dataset(
                    pop=[mock_agent],
                    env=mock_env,
                    evaluation_interval=2,
                    num_epochs=10,
                    max_steps=100,
                    evo_steps=None,
                )

    def test_train_llm_dataset_sft_break_on_num_epochs(self):
        mock_agent = _mock_sft_agent()

        mock_env = MagicMock()
        mock_env.__len__.return_value = 3
        mock_env.reset.return_value = _sft_batch()
        mock_env.step.return_value = "next_prompts"
        mock_env.data_batch_size_per_gpu = 1
        mock_env.world_size = 1

        with (
            patch("agilerl.training.llm.dataset.default_progress_bar") as mock_pbar_fn,
            patch("agilerl.training.llm.dataset.init_loggers", return_value=[]),
            patch("agilerl.training.llm.dataset.save_llm_checkpoint"),
        ):
            mock_pbar_fn.return_value = MagicMock()
            mock_env.num_epochs = 2
            train_llm_dataset(
                pop=[mock_agent],
                env=mock_env,
                evaluation_interval=2,
                evo_steps=1,
                accelerator=None,
                num_epochs=2,
                checkpoint_steps=3,
            )

    def test_train_llm_dataset_sft_value_error_if_algo_not_sft(self):
        mock_agent = MagicMock(spec=GRPO)
        mock_agent.algo = "GRPO"
        mock_agent.pretrained_model_name_or_path = "Qwen/Qwen2.5-0.5B"
        mock_agent.batch_size_per_process = 32
        mock_agent.batch_size = 32
        with pytest.raises(ValueError, match="SFT"):
            train_llm_dataset(
                pop=[mock_agent],
                env=MagicMock(),
                evaluation_interval=2,
                accelerator=None,
            )

    def test_train_llm_dataset_sft_evo_steps_not_set(self):
        with pytest.raises(ValueError, match="evo_steps"):
            train_llm_dataset(
                pop=[MagicMock(spec=SFT)],
                env=MagicMock(),
                evo_steps=None,
                accelerator=None,
                selection_strategy=MagicMock(),
                mutation=MagicMock(),
            )


@pytest.mark.parametrize(
    ("make_agent", "make_batch", "error_match"),
    [
        (_mock_dpo_agent, _preference_batch, None),
        (_mock_dpo_agent, _sft_batch, "preference"),
        (_mock_sft_agent, _sft_batch, None),
        (_mock_sft_agent, _preference_batch, r"objective='sft'"),
    ],
)
def test_train_llm_dataset_requires_batch_matching_algorithm(
    make_agent, make_batch, error_match
):
    agent = make_agent()
    batch = make_batch()
    env = MagicMock()
    env.__len__.return_value = 1
    env.reset.return_value = batch
    env.data_batch_size_per_gpu = 1
    env.world_size = 1
    env.num_epochs = 0
    env.name = "mock_env"

    with (
        patch("agilerl.training.llm.dataset.default_progress_bar") as mock_pbar_fn,
        patch("agilerl.training.llm.dataset.init_loggers", return_value=[]),
        patch("agilerl.training.llm.dataset.save_llm_checkpoint"),
    ):
        mock_pbar_fn.return_value = MagicMock()
        if error_match is not None:
            with pytest.raises(ValueError, match=error_match):
                train_llm_dataset(
                    pop=[agent],
                    env=env,
                    max_steps=1,
                    evaluation_interval=100,
                    verbose=False,
                    accelerator=None,
                )
            agent.learn.assert_not_called()
            return

        train_llm_dataset(
            pop=[agent],
            env=env,
            max_steps=1,
            evaluation_interval=100,
            verbose=False,
            accelerator=None,
        )

    agent.learn.assert_called_with(batch)


# ---------------------------------------------------------------------------
# TestFinetuneLlmRollout
# ---------------------------------------------------------------------------


class TestTrainLlmRollout:
    @pytest.mark.parametrize("agent_spec", [LLMPPO, LLMREINFORCE, GRPO])
    @pytest.mark.parametrize("use_accelerator", [True, False])
    def test_train_llm_rollout_basic_training_loop(self, agent_spec, use_accelerator):
        mock_agent = _make_rollout_mock_agent(spec=agent_spec)
        batch_steps = 3
        max_steps = 9

        with (
            patch("agilerl.training.llm.rollout.default_progress_bar") as mock_pbar_fn,
            patch("agilerl.training.llm.rollout.init_loggers", return_value=[]),
            patch("agilerl.training.llm.rollout.safe_aggregate_metrics") as mock_agg,
            patch("agilerl.training.llm.rollout.save_llm_checkpoint") as mock_save,
            patch("agilerl.training.llm.rollout.RolloutCollector"),
            patch("agilerl.training.llm.rollout.collect_rollouts_llm") as mock_collect,
        ):
            mock_pbar_fn.return_value = MagicMock()
            mock_agg.return_value = 0.5
            mock_collect.return_value = _rollout_collect_return(batch_steps=batch_steps)

            train_llm_rollout(
                pop=[mock_agent],
                env_factory=MagicMock(),
                max_turns=2,
                init_hp={"BATCH_SIZE": 1, "ALGO": mock_agent.algo},
                max_steps=max_steps,
                evaluation_interval=100,
                verbose=False,
                accelerator=None if use_accelerator else Accelerator(),
            )

        num_outer = max_steps // batch_steps
        assert mock_collect.call_count == num_outer
        assert mock_agent.learn.call_count == num_outer
        assert mock_agent.test.call_count == 0
        # The rollout loop passes turn_ids and sampling_logps unconditionally;
        # every rollout algorithm's learn() accepts both.
        mock_agent.learn.assert_called_with(ANY, turn_ids=ANY, sampling_logps=ANY)
        # No checkpoint/elite path was configured, so nothing may be written.
        assert mock_save.call_count == 0

    def test_train_llm_rollout_labels_run_from_population(self):
        """A manifest-style init_hp (no flat ALGO key) still names the run after
        the actual algorithm and env, not the LLMPPO/rollout defaults.
        """
        mock_agent = _make_rollout_mock_agent(spec=GRPO)

        with (
            patch("agilerl.training.llm.rollout.default_progress_bar"),
            patch(
                "agilerl.training.llm.rollout.init_loggers", return_value=[]
            ) as mock_init_loggers,
            patch("agilerl.training.llm.rollout.safe_aggregate_metrics"),
            patch("agilerl.training.llm.rollout.save_llm_checkpoint"),
            patch("agilerl.training.llm.rollout.RolloutCollector"),
            patch("agilerl.training.llm.rollout.collect_rollouts_llm") as mock_collect,
        ):
            mock_collect.return_value = _rollout_collect_return(batch_steps=3)

            train_llm_rollout(
                pop=[mock_agent],
                env_factory=MagicMock(),
                max_turns=2,
                init_hp={"env_name": "game:Sudoku-v0-hard"},
                max_steps=3,
                evaluation_interval=100,
                verbose=False,
                accelerator=None,
            )

        assert mock_init_loggers.call_args.kwargs["algo"] == "GRPO"
        assert mock_init_loggers.call_args.kwargs["env_name"] == "game:Sudoku-v0-hard"

    def test_train_llm_rollout_final_checkpoint_needs_a_configured_path(self):
        """The end-of-run checkpoint fires only when a save target is configured."""
        mock_agent = _make_rollout_mock_agent(spec=GRPO)
        with (
            patch(
                "agilerl.training.llm.rollout.default_progress_bar",
                return_value=MagicMock(),
            ),
            patch("agilerl.training.llm.rollout.init_loggers", return_value=[]),
            patch(
                "agilerl.training.llm.rollout.safe_aggregate_metrics",
                return_value=0.5,
            ),
            patch("agilerl.training.llm.rollout.save_llm_checkpoint") as mock_save,
            patch("agilerl.training.llm.rollout.RolloutCollector"),
            patch(
                "agilerl.training.llm.rollout.collect_rollouts_llm",
                return_value=_rollout_collect_return(batch_steps=3),
            ),
        ):
            train_llm_rollout(
                pop=[mock_agent],
                env_factory=MagicMock(),
                max_turns=2,
                init_hp={"BATCH_SIZE": 1, "ALGO": mock_agent.algo},
                max_steps=3,
                checkpoint_path="ckpts",
                evaluation_interval=100,
                verbose=False,
                accelerator=None,
            )
        mock_save.assert_called_once_with(mock_agent, "ckpts")

    def test_train_llm_rollout_aborts_when_no_progress(self):
        """A rollout that never yields turns (``batch_steps == 0``) leaves
        ``total_steps`` unchanged, so the loop would spin forever. It must warn
        and then abort with a clear error after a bounded number of consecutive
        stalls instead.
        """
        mock_agent = _make_rollout_mock_agent(spec=GRPO)

        with (
            patch(
                "agilerl.training.llm.rollout.default_progress_bar",
                return_value=MagicMock(),
            ),
            patch("agilerl.training.llm.rollout.init_loggers", return_value=[]),
            patch(
                "agilerl.training.llm.rollout.safe_aggregate_metrics",
                return_value=0.5,
            ),
            patch("agilerl.training.llm.rollout.save_llm_checkpoint"),
            patch("agilerl.training.llm.rollout.RolloutCollector"),
            patch(
                "agilerl.training.llm.rollout.collect_rollouts_llm",
                return_value=_rollout_collect_return(batch_steps=0, n_trajectories=0),
            ) as mock_collect,
            pytest.warns(UserWarning, match="no usable turns"),
            pytest.raises(RuntimeError, match="made no progress"),
        ):
            train_llm_rollout(
                pop=[mock_agent],
                env_factory=MagicMock(),
                max_turns=2,
                init_hp={"BATCH_SIZE": 1, "ALGO": mock_agent.algo},
                # Large max_steps so only the stall guard — not completion — can
                # end the loop.
                max_steps=1000,
                evaluation_interval=100,
                verbose=False,
                accelerator=None,
            )

        # Bounded: aborted after the stall threshold rather than looping forever.
        assert mock_collect.call_count == 8
        mock_agent.learn.assert_not_called()

    def test_train_llm_rollout_forwards_sampling_logps_to_learn(self):
        """When the rollout captures sampling logps, they're forwarded to
        ``learn(..., sampling_logps=...)`` for GRPO/PPO/REINFORCE agents.
        """
        mock_agent = _make_rollout_mock_agent(spec=GRPO)
        sampling_logps = [torch.zeros(1, 7)]
        rollout_return = (
            [torch.ones(1, 8, dtype=torch.long)],  # token_ids_list
            [torch.ones(1, 7, dtype=torch.bool)],  # action_masks_list
            [torch.zeros(1, 7, dtype=torch.long)],  # all_turn_ids
            [torch.ones(2, dtype=torch.float32)],  # all_rewards
            3,  # batch_steps
            123,  # group_seed
            sampling_logps,  # all_sampling_logps (non-None)
        )
        with (
            patch(
                "agilerl.training.llm.rollout.default_progress_bar",
                return_value=MagicMock(),
            ),
            patch("agilerl.training.llm.rollout.init_loggers", return_value=[]),
            patch(
                "agilerl.training.llm.rollout.safe_aggregate_metrics",
                return_value=0.5,
            ),
            patch("agilerl.training.llm.rollout.save_llm_checkpoint"),
            patch("agilerl.training.llm.rollout.RolloutCollector"),
            patch(
                "agilerl.training.llm.rollout.collect_rollouts_llm",
                return_value=rollout_return,
            ),
        ):
            train_llm_rollout(
                pop=[mock_agent],
                env_factory=MagicMock(),
                max_turns=2,
                init_hp={"BATCH_SIZE": 1, "ALGO": mock_agent.algo},
                max_steps=3,  # one outer iteration (batch_steps=3)
                evaluation_interval=100,
                verbose=False,
                accelerator=None,
            )

        _, learn_kwargs = mock_agent.learn.call_args
        forwarded = learn_kwargs.get("sampling_logps")
        assert forwarded is not None
        assert forwarded[0] is sampling_logps[0]

    def test_train_llm_rollout_derives_group_seed_from_agent_seed(self):
        """The first rollout seeds from the agent's configured seed (rank 0)."""
        mock_agent = _make_rollout_mock_agent()
        mock_agent.seed = 1234
        with (
            patch(
                "agilerl.training.llm.rollout.default_progress_bar",
                return_value=MagicMock(),
            ),
            patch("agilerl.training.llm.rollout.init_loggers", return_value=[]),
            patch(
                "agilerl.training.llm.rollout.safe_aggregate_metrics",
                return_value=0.5,
            ),
            patch("agilerl.training.llm.rollout.save_llm_checkpoint"),
            patch("agilerl.training.llm.rollout.RolloutCollector"),
            patch("agilerl.training.llm.rollout.collect_rollouts_llm") as mock_collect,
        ):
            mock_collect.return_value = _rollout_collect_return(batch_steps=3)
            train_llm_rollout(
                pop=[mock_agent],
                env_factory=MagicMock(),
                max_turns=2,
                init_hp={"BATCH_SIZE": 1, "ALGO": mock_agent.algo},
                max_steps=3,
                evaluation_interval=100,
                verbose=False,
                accelerator=None,
            )

        assert mock_collect.call_args_list[0].kwargs["group_seed"] == 1234

    def test_train_llm_rollout_allows_batch_size_indivisible_by_group_size(self):
        """The batch>group case is unconstrained too: batch_size=3, group_size=2
        (three prompts, two completions each) must pass startup validation rather
        than being rejected. Patch the rollout to a sentinel and assert the call
        reaches it — i.e. it gets past the (now-removed) divisibility guard.
        """
        mock_agent = _make_rollout_mock_agent(spec=GRPO)
        mock_agent.group_size = 2
        mock_agent.batch_size = 16
        mock_agent.batch_size_per_process = 16
        with (
            patch(
                "agilerl.training.llm.rollout.default_progress_bar",
                return_value=MagicMock(),
            ),
            patch("agilerl.training.llm.rollout.init_loggers", return_value=[]),
            patch("agilerl.training.llm.rollout.RolloutCollector"),
            patch(
                "agilerl.training.llm.rollout.collect_rollouts_llm",
                side_effect=RuntimeError("reached rollout"),
            ),
            pytest.raises(RuntimeError, match="reached rollout"),
        ):
            train_llm_rollout(
                pop=[mock_agent],
                max_turns=1,
                env_factory=MagicMock(),
                init_hp={"BATCH_SIZE": 3, "ALGO": "GRPO"},
                max_steps=100,
                accelerator=None,
                verbose=False,
            )

    @pytest.mark.parametrize("use_accelerator", [True, False])
    def test_train_llm_rollout_with_wandb_and_checkpoints(self, use_accelerator):
        mock_agent = _make_rollout_mock_agent()

        with (
            patch("agilerl.training.llm.rollout.default_progress_bar") as mock_pbar_fn,
            patch("agilerl.training.llm.rollout.init_loggers") as mock_init_loggers,
            patch("agilerl.training.llm.rollout.safe_aggregate_metrics") as mock_agg,
            patch("agilerl.training.llm.rollout.save_llm_checkpoint") as mock_save,
            patch("agilerl.training.llm.rollout.RolloutCollector"),
            patch("agilerl.training.llm.rollout.collect_rollouts_llm") as mock_collect,
        ):
            mock_pbar_fn.return_value = MagicMock()
            mock_init_loggers.return_value = []
            mock_agg.return_value = 0.5
            mock_collect.return_value = _rollout_collect_return(batch_steps=3)

            train_llm_rollout(
                pop=[mock_agent],
                env_factory=MagicMock(),
                max_turns=2,
                init_hp={"BATCH_SIZE": 1, "ALGO": "LLMPPO"},
                max_steps=18,
                evaluation_interval=100,
                verbose=False,
                wb=True,
                wandb_api_key="fake_key",
                checkpoint_steps=2,
                accelerator=None if use_accelerator else Accelerator(),
            )

        mock_init_loggers.assert_called_once()
        assert mock_init_loggers.call_args.kwargs["wb"] is True
        assert mock_save.call_count >= 1

    @pytest.mark.parametrize("use_accelerator", [True, False])
    def test_train_llm_rollout_evolvable_training_loop(self, use_accelerator):
        mock_agent = _make_rollout_mock_agent()
        mutation = MagicMock()
        mutation.architecture_mut = 0
        mutation.new_layer_prob = 0
        mutation.parameters_mut = 0
        mutation.activation_mut = 0

        with (
            patch("agilerl.training.llm.rollout.default_progress_bar") as mock_pbar_fn,
            patch("agilerl.training.llm.rollout.init_loggers", return_value=[]),
            patch("agilerl.training.llm.rollout.safe_aggregate_metrics") as mock_agg,
            patch("agilerl.training.llm.rollout.save_llm_checkpoint") as mock_save,
            patch("agilerl.training.llm.rollout.RolloutCollector"),
            patch("agilerl.training.llm.rollout.collect_rollouts_llm") as mock_collect,
            patch(
                "agilerl.training.llm.rollout.run_selection_and_mutation"
            ) as mock_tourn,
        ):
            mock_pbar_fn.return_value = MagicMock()
            mock_agg.return_value = 0.5
            mock_collect.return_value = _rollout_collect_return(batch_steps=3)
            mock_tourn.return_value = [mock_agent]

            train_llm_rollout(
                pop=[mock_agent],
                env_factory=MagicMock(),
                max_turns=2,
                init_hp={"BATCH_SIZE": 1, "ALGO": "LLMPPO"},
                max_steps=9,
                evaluation_interval=100,
                verbose=False,
                evo_steps=1,
                selection_strategy=Mock(),
                mutation=mutation,
                accelerator=None if use_accelerator else Accelerator(),
            )

        assert mock_tourn.call_count == 3
        assert mock_save.call_count == 0

    def test_train_llm_rollout_refreshes_collector_geometry_after_evolution(self):
        mock_agent = _make_rollout_mock_agent(spec=GRPO)
        mock_agent.group_size = 1
        evolved = _make_rollout_mock_agent(spec=GRPO)
        evolved.group_size = 4
        mutation = MagicMock()
        mutation.architecture_mut = 0
        mutation.new_layer_prob = 0
        mutation.parameters_mut = 0
        mutation.activation_mut = 0

        with (
            patch(
                "agilerl.training.llm.rollout.default_progress_bar",
                return_value=MagicMock(),
            ),
            patch("agilerl.training.llm.rollout.init_loggers", return_value=[]),
            patch(
                "agilerl.training.llm.rollout.safe_aggregate_metrics",
                return_value=0.5,
            ),
            patch("agilerl.training.llm.rollout.save_llm_checkpoint"),
            patch(
                "agilerl.training.llm.rollout.RolloutCollector"
            ) as mock_collector_cls,
            patch(
                "agilerl.training.llm.rollout.collect_rollouts_llm",
                return_value=_rollout_collect_return(batch_steps=3),
            ),
            patch(
                "agilerl.training.llm.rollout.run_selection_and_mutation",
                return_value=[evolved],
            ),
        ):
            collector = mock_collector_cls.return_value
            train_llm_rollout(
                pop=[mock_agent],
                env_factory=MagicMock(),
                max_turns=2,
                init_hp={"BATCH_SIZE": 1, "ALGO": mock_agent.algo},
                max_steps=3,
                evaluation_interval=100,
                verbose=False,
                evo_steps=1,
                selection_strategy=Mock(),
                mutation=mutation,
                accelerator=None,
            )

        collector.update_rollout_geometry.assert_any_call(
            rollout_batch_size=1,
            group_size=4,
        )

    def test_train_llm_rollout_value_error_when_evo_steps_missing_with_tournament(
        self,
    ):
        mutation = MagicMock()
        mutation.architecture_mut = 0
        mutation.new_layer_prob = 0
        mutation.parameters_mut = 0
        mutation.activation_mut = 0
        mock_agent = _make_rollout_mock_agent()
        with pytest.raises(ValueError, match="evo_steps"):
            train_llm_rollout(
                pop=[mock_agent],
                env_factory=MagicMock(),
                max_turns=1,
                init_hp={"BATCH_SIZE": 1, "ALGO": "LLMPPO"},
                max_steps=1,
                evo_steps=None,
                selection_strategy=MagicMock(),
                mutation=mutation,
                accelerator=None,
            )

    def test_train_llm_rollout_warns_when_evo_steps_without_tournament(self):
        mock_agent = _make_rollout_mock_agent()
        with pytest.warns(UserWarning, match="evo_steps"):
            train_llm_rollout(
                pop=[mock_agent],
                env_factory=MagicMock(),
                max_turns=1,
                init_hp={"BATCH_SIZE": 1, "ALGO": "LLMPPO"},
                max_steps=0,
                evo_steps=3,
                selection_strategy=None,
                mutation=None,
                accelerator=None,
                verbose=False,
            )

    def test_train_llm_rollout_value_error_if_algo_not_supported(self):
        mock_agent = MagicMock(spec=DPO)
        mock_agent.algo = "DPO"
        mock_agent.batch_size = 16
        mock_agent.batch_size_per_process = 16
        with pytest.raises(ValueError, match=r"LLMPPO.*LLMREINFORCE.*GRPO"):
            train_llm_rollout(
                pop=[mock_agent],
                env_factory=MagicMock(),
                max_turns=1,
                init_hp={"BATCH_SIZE": 1, "ALGO": "DPO"},
                max_steps=0,
                accelerator=None,
                verbose=False,
            )

    def test_train_llm_rollout_max_reward_adds_accuracy_metric(self):
        mock_agent = _make_rollout_mock_agent()

        with (
            patch("agilerl.training.llm.rollout.default_progress_bar") as mock_pbar_fn,
            patch("agilerl.training.llm.rollout.init_loggers", return_value=[]),
            patch("agilerl.training.llm.rollout.safe_aggregate_metrics") as mock_agg,
            patch("agilerl.training.llm.rollout.save_llm_checkpoint"),
            patch("agilerl.training.llm.rollout.RolloutCollector"),
            patch("agilerl.training.llm.rollout.collect_rollouts_llm") as mock_collect,
        ):
            mock_pbar_fn.return_value = MagicMock()
            mock_agg.return_value = 0.5
            mock_collect.return_value = _rollout_collect_return(batch_steps=3)
            train_llm_rollout(
                pop=[mock_agent],
                env_factory=MagicMock(),
                max_turns=2,
                init_hp={"BATCH_SIZE": 1, "ALGO": "LLMPPO"},
                max_steps=9,
                evaluation_interval=100,
                max_reward=1.0,
                verbose=False,
                accelerator=None,
            )

        num_outer = 3
        # agg called for: mean_score (1) + accuracy (1) per outer iteration
        assert mock_agg.call_count == num_outer * 2

    def test_train_llm_rollout_registers_accuracy_metric(self):
        mock_agent = _make_rollout_mock_agent()
        mock_agent.metrics.additional_metrics = ["loss", "mean_reward"]

        with (
            patch("agilerl.training.llm.rollout.default_progress_bar") as mock_pbar_fn,
            patch("agilerl.training.llm.rollout.init_loggers", return_value=[]),
            patch("agilerl.training.llm.rollout.safe_aggregate_metrics") as mock_agg,
            patch("agilerl.training.llm.rollout.save_llm_checkpoint"),
            patch("agilerl.training.llm.rollout.RolloutCollector"),
            patch("agilerl.training.llm.rollout.collect_rollouts_llm") as mock_collect,
        ):
            mock_pbar_fn.return_value = MagicMock()
            mock_agg.return_value = 0.5
            mock_collect.return_value = _rollout_collect_return(batch_steps=3)
            train_llm_rollout(
                pop=[mock_agent],
                env_factory=MagicMock(),
                max_turns=2,
                init_hp={"BATCH_SIZE": 1, "ALGO": "LLMPPO"},
                max_steps=9,
                evaluation_interval=100,
                max_reward=1.0,
                verbose=False,
                accelerator=None,
            )

        mock_agent.metrics.register.assert_called_with("accuracy")

    def test_train_llm_rollout_logs_rubric_component_metrics(self):
        mock_agent = _make_rollout_mock_agent()
        mock_agent.metrics.additional_metrics = ["loss", "mean_reward"]

        mock_rollout_env = MagicMock()
        mock_rollout_env.num_epochs = 0
        mock_rollout_env.get_rubric_score_means.return_value = {"fmt": 0.75}

        def _agg(_accelerator, value):
            return (
                value if isinstance(value, torch.Tensor) else torch.tensor(float(value))
            )

        with (
            patch("agilerl.training.llm.rollout.default_progress_bar") as mock_pbar_fn,
            patch("agilerl.training.llm.rollout.init_loggers", return_value=[]),
            patch(
                "agilerl.training.llm.rollout.safe_aggregate_metrics",
                side_effect=_agg,
            ),
            patch("agilerl.training.llm.rollout.save_llm_checkpoint"),
            patch(
                "agilerl.training.llm.rollout.RolloutCollector",
                return_value=mock_rollout_env,
            ),
            patch("agilerl.training.llm.rollout.collect_rollouts_llm") as mock_collect,
        ):
            mock_pbar_fn.return_value = MagicMock()
            mock_collect.return_value = _rollout_collect_return(batch_steps=3)
            train_llm_rollout(
                pop=[mock_agent],
                env_factory=MagicMock(),
                max_turns=2,
                init_hp={"BATCH_SIZE": 1, "ALGO": "LLMPPO"},
                max_steps=3,
                evaluation_interval=100,
                verbose=False,
                accelerator=None,
            )

        mock_agent.metrics.register.assert_any_call("reward_fmt")
        mock_agent.metrics.log.assert_any_call("reward_fmt", ANY)

    def test_train_llm_rollout_stops_at_wall_clock_limit(self, capsys):
        mock_agent = _make_rollout_mock_agent()

        with (
            patch("agilerl.training.llm.rollout.default_progress_bar") as mock_pbar_fn,
            patch("agilerl.training.llm.rollout.init_loggers", return_value=[]),
            patch("agilerl.training.llm.rollout.safe_aggregate_metrics") as mock_agg,
            patch("agilerl.training.llm.rollout.save_llm_checkpoint"),
            patch("agilerl.training.llm.rollout.RolloutCollector"),
            patch("agilerl.training.llm.rollout.collect_rollouts_llm") as mock_collect,
            patch(
                "agilerl.training.llm.rollout.time.monotonic",
                side_effect=itertools.count(100, 100).__next__,
            ),
        ):
            mock_pbar_fn.return_value = MagicMock()
            mock_agg.return_value = 0.5
            mock_collect.return_value = _rollout_collect_return(batch_steps=3)
            train_llm_rollout(
                pop=[mock_agent],
                env_factory=MagicMock(),
                max_turns=2,
                init_hp={"BATCH_SIZE": 1, "ALGO": "LLMPPO"},
                max_steps=100,
                max_wall_seconds=50,
                evaluation_interval=100,
                verbose=False,
                accelerator=None,
            )

        assert "wall time limit (50s) reached" in capsys.readouterr().out
        mock_collect.assert_not_called()

    def test_train_llm_rollout_wall_clock_stop_is_rank_aligned(self, capsys):
        mock_agent = _make_rollout_mock_agent()
        accelerator = _two_rank_accelerator(peer_empty=True)

        with (
            patch("agilerl.training.llm.rollout.default_progress_bar") as mock_pbar_fn,
            patch("agilerl.training.llm.rollout.init_loggers", return_value=[]),
            patch("agilerl.training.llm.rollout.safe_aggregate_metrics") as mock_agg,
            patch("agilerl.training.llm.rollout.save_llm_checkpoint"),
            patch("agilerl.training.llm.rollout.RolloutCollector"),
            patch("agilerl.training.llm.rollout.collect_rollouts_llm") as mock_collect,
            patch(
                "agilerl.training.llm.rollout.time.monotonic",
                return_value=100,
            ),
        ):
            mock_pbar_fn.return_value = MagicMock()
            mock_agg.return_value = 0.5
            mock_collect.return_value = _rollout_collect_return(batch_steps=3)
            train_llm_rollout(
                pop=[mock_agent],
                env_factory=MagicMock(),
                max_turns=2,
                init_hp={"BATCH_SIZE": 1, "ALGO": "LLMPPO"},
                max_steps=100,
                max_wall_seconds=50,
                evaluation_interval=100,
                verbose=False,
                accelerator=accelerator,
            )

        assert "wall time limit (50s) reached" in capsys.readouterr().out
        mock_collect.assert_not_called()

    def test_train_llm_rollout_eval_interval_calls_test(self):
        mock_agent = _make_rollout_mock_agent()
        batch_steps = 3
        max_steps = 9

        with (
            patch("agilerl.training.llm.rollout.default_progress_bar") as mock_pbar_fn,
            patch("agilerl.training.llm.rollout.init_loggers", return_value=[]),
            patch("agilerl.training.llm.rollout.safe_aggregate_metrics") as mock_agg,
            patch("agilerl.training.llm.rollout.save_llm_checkpoint"),
            patch("agilerl.training.llm.rollout.RolloutCollector"),
            patch("agilerl.training.llm.rollout.collect_rollouts_llm") as mock_collect,
        ):
            mock_pbar_fn.return_value = MagicMock()
            mock_agg.return_value = 0.5
            mock_collect.return_value = _rollout_collect_return(batch_steps=batch_steps)
            train_llm_rollout(
                pop=[mock_agent],
                env_factory=MagicMock(),
                max_turns=2,
                init_hp={"BATCH_SIZE": 1, "ALGO": "LLMPPO"},
                max_steps=max_steps,
                evaluation_interval=1,
                verbose=False,
                accelerator=None,
            )

        num_outer = max_steps // batch_steps
        assert mock_agent.test.call_count == num_outer

    def test_train_llm_rollout_saves_elite_at_end(self):
        weaker = _make_rollout_mock_agent()
        weaker.fitness = [0.1]
        stronger = _make_rollout_mock_agent()
        stronger.fitness = [0.9]

        with (
            patch("agilerl.training.llm.rollout.default_progress_bar") as mock_pbar_fn,
            patch("agilerl.training.llm.rollout.init_loggers", return_value=[]),
            patch("agilerl.training.llm.rollout.safe_aggregate_metrics") as mock_agg,
            patch("agilerl.training.llm.rollout.save_llm_checkpoint") as mock_save,
            patch("agilerl.training.llm.rollout.RolloutCollector"),
            patch("agilerl.training.llm.rollout.collect_rollouts_llm") as mock_collect,
            _population_init_skip_per_mock_class(),
        ):
            mock_pbar_fn.return_value = MagicMock()
            mock_agg.return_value = 0.5
            mock_collect.return_value = _rollout_collect_return(batch_steps=3)
            train_llm_rollout(
                pop=[weaker, stronger],
                env_factory=MagicMock(),
                max_turns=2,
                init_hp={"BATCH_SIZE": 1, "ALGO": "LLMPPO"},
                max_steps=9,
                evaluation_interval=100,
                save_elite=True,
                elite_path="/tmp/rollout-elite",
                verbose=False,
                accelerator=None,
            )

        assert mock_save.call_args_list[-1] == call(stronger, "/tmp/rollout-elite")

    def test_train_llm_rollout_init_hp_none_uses_agent_fields(self):
        mock_agent = _make_rollout_mock_agent()
        mock_agent.batch_size_per_process = 7
        mock_agent.algo = "LLMPPO"

        with (
            patch("agilerl.training.llm.rollout.default_progress_bar") as mock_pbar_fn,
            patch("agilerl.training.llm.rollout.init_loggers") as mock_init_loggers,
            patch("agilerl.training.llm.rollout.safe_aggregate_metrics") as mock_agg,
            patch("agilerl.training.llm.rollout.save_llm_checkpoint"),
            patch("agilerl.training.llm.rollout.RolloutCollector"),
            patch("agilerl.training.llm.rollout.collect_rollouts_llm") as mock_collect,
        ):
            mock_pbar_fn.return_value = MagicMock()
            mock_init_loggers.return_value = []
            mock_agg.return_value = 0.5
            mock_collect.return_value = _rollout_collect_return(batch_steps=3)
            train_llm_rollout(
                pop=[mock_agent],
                env_factory=MagicMock(),
                max_turns=2,
                init_hp=None,
                max_steps=0,
                wb=True,
                wandb_api_key="fake",
                verbose=False,
                accelerator=None,
            )

        init_hp_passed = mock_init_loggers.call_args.kwargs["init_hyperparams"]
        assert init_hp_passed["BATCH_SIZE_PER_GPU"] == 7
        assert init_hp_passed["ALGO"] == "LLMPPO"

    def test_train_llm_rollout_allows_group_size_indivisible_by_batch_size(
        self,
    ):
        """batch_size and group_size need not divide each other for GRPO.

        The rollout vec env keeps each prompt's group whole (group-contiguous),
        so e.g. batch_size=2, group_size=3 (two prompts, three completions each)
        is valid and must pass startup validation rather than being rejected.
        We patch the rollout to raise a sentinel and assert the call reaches it
        — i.e. it gets past the (now-removed) divisibility guard.
        """
        agent = _make_rollout_mock_agent(spec=GRPO)
        agent.group_size = 3
        agent.batch_size = 2
        agent.batch_size_per_process = 2

        with (
            patch(
                "agilerl.training.llm.rollout.default_progress_bar",
                return_value=MagicMock(),
            ),
            patch("agilerl.training.llm.rollout.init_loggers", return_value=[]),
            patch("agilerl.training.llm.rollout.RolloutCollector"),
            patch(
                "agilerl.training.llm.rollout.collect_rollouts_llm",
                side_effect=RuntimeError("reached rollout"),
            ),
            pytest.raises(RuntimeError, match="reached rollout"),
        ):
            train_llm_rollout(
                pop=[agent],
                max_turns=2,
                env_factory=MagicMock(),
                init_hp={"BATCH_SIZE": 2, "BATCH_SIZE_PER_GPU": 2, "ALGO": "GRPO"},
                max_steps=8,
                accelerator=None,
                wb=False,
                verbose=False,
            )


# ---------------------------------------------------------------------------
# Module-level: env/env_fn validation tests
# ---------------------------------------------------------------------------


def test_train_llm_dataset_env_and_env_fn_mutually_exclusive():
    agent = MagicMock(spec=DPO)
    agent.algo = "DPO"
    agent.batch_size_per_process = 1
    agent.batch_size = 1
    agent.pretrained_model_name_or_path = "Qwen/Qwen2.5-0.5B"
    agent.fitness = [0.0]

    env = MagicMock()
    env.__len__.return_value = 1
    env.name = "mock_env"
    env.data_batch_size_per_gpu = 1
    env.world_size = 1
    env.num_epochs = 0
    env.reset.return_value = "prompts"
    env.step.return_value = "prompts"

    with pytest.raises(ValueError, match="Provide exactly one of 'env' or 'env_fn'"):
        train_llm_dataset(
            pop=[agent],
            env=env,
            env_fn=lambda: env,
            max_steps=0,
            verbose=False,
            accelerator=None,
        )


def test_train_llm_dataset_requires_env_or_env_fn():
    with pytest.raises(ValueError, match="Either 'env' or 'env_fn' must be provided"):
        train_llm_dataset(
            pop=[MagicMock()],
            env=None,
            env_fn=None,
            max_steps=0,
            verbose=False,
            accelerator=None,
        )


def test_train_llm_dataset_warns_on_shared_env_with_population():
    agents = [
        _mock_dpo_agent(index=i, batch_size_per_process=1, batch_size=1)
        for i in range(2)
    ]

    env = MagicMock()
    env.__len__.return_value = 1
    env.name = "mock_env"
    env.data_batch_size_per_gpu = 1
    env.world_size = 1
    env.num_epochs = 0
    env.reset.return_value = "prompts"
    env.step.return_value = "prompts"

    with (
        _population_init_skip_per_mock_class(),
        pytest.warns(UserWarning, match="fairness bias"),
    ):
        train_llm_dataset(
            pop=agents,
            env=env,
            max_steps=0,
            verbose=False,
            accelerator=None,
        )


def test_train_llm_dataset_shared_env_first_iteration_yields_successive_batches():
    agent_a = _mock_dpo_agent(index=0, batch_size_per_process=1, batch_size=1)
    agent_b = _mock_dpo_agent(index=1, batch_size_per_process=1, batch_size=1)
    env = _IncrementingPreferenceEnv()

    with (
        _population_init_skip_per_mock_class(),
        patch("agilerl.training.llm.dataset.default_progress_bar") as mock_pbar_fn,
        patch("agilerl.training.llm.dataset.save_llm_checkpoint"),
        patch("agilerl.training.llm.dataset.init_loggers", return_value=[]),
    ):
        mock_pbar_fn.return_value = MagicMock()
        with pytest.warns(UserWarning, match="fairness bias"):
            train_llm_dataset(
                pop=[agent_a, agent_b],
                env=env,
                max_steps=2,
                evaluation_interval=100,
                verbose=False,
                accelerator=None,
            )

    assert env.reset_dataloaders_calls == [True, False]
    batch_a = agent_a.learn.call_args.args[0]
    batch_b = agent_b.learn.call_args.args[0]
    assert batch_a["chosen_input_ids"][0, 0].item() == 0
    assert batch_b["chosen_input_ids"][0, 0].item() == 1


def test_train_llm_dataset_env_fn_rewinds_each_env_on_first_iteration():
    agent_a = _mock_dpo_agent(index=0, batch_size_per_process=1, batch_size=1)
    agent_b = _mock_dpo_agent(index=1, batch_size_per_process=1, batch_size=1)
    env_a = _IncrementingPreferenceEnv()
    env_b = _IncrementingPreferenceEnv()
    env_fn = MagicMock(side_effect=[env_a, env_b])

    with (
        _population_init_skip_per_mock_class(),
        patch("agilerl.training.llm.dataset.default_progress_bar") as mock_pbar_fn,
        patch("agilerl.training.llm.dataset.save_llm_checkpoint"),
        patch("agilerl.training.llm.dataset.init_loggers", return_value=[]),
    ):
        mock_pbar_fn.return_value = MagicMock()
        train_llm_dataset(
            pop=[agent_a, agent_b],
            env_fn=env_fn,
            max_steps=2,
            evaluation_interval=100,
            verbose=False,
            accelerator=None,
        )

    assert env_a.reset_dataloaders_calls == [True]
    assert env_b.reset_dataloaders_calls == [True]
    assert agent_a.learn.call_args.args[0]["chosen_input_ids"][0, 0].item() == 0
    assert agent_b.learn.call_args.args[0]["chosen_input_ids"][0, 0].item() == 0


def test_train_llm_checkpoint_triggering_non_divisible_steps():
    agent = _mock_dpo_agent(batch_size_per_process=1, batch_size=1)
    env = MagicMock()
    env.reset.return_value = _preference_batch()

    env.__len__.return_value = 10
    env.name = "mock_env"
    env.data_batch_size_per_gpu = 1
    env.world_size = 1
    env.num_epochs = 0

    with (
        patch("agilerl.training.llm.dataset.default_progress_bar") as mock_pbar_fn,
        patch("agilerl.training.llm.rollout.safe_aggregate_metrics", return_value=0.5),
        patch("agilerl.training.llm.dataset.save_llm_checkpoint") as mock_save,
        patch("agilerl.training.llm.dataset.init_loggers", return_value=[]),
    ):
        mock_pbar_fn.return_value = MagicMock()
        train_llm_dataset(
            pop=[agent],
            env=env,
            max_steps=5,
            checkpoint_steps=2,
            evaluation_interval=100,
            verbose=False,
            accelerator=None,
        )

    assert mock_save.call_count == 3


@pytest.mark.parametrize("agent_spec", [DPO, SFT])
def test_inner_loop_breaks_after_max_steps_first_agent(agent_spec):
    if agent_spec is DPO:
        agent0 = _mock_dpo_agent(index=0, batch_size_per_process=1, batch_size=1)
        agent1 = _mock_dpo_agent(index=1, batch_size_per_process=1, batch_size=1)
        env = MagicMock()
        env.__len__.return_value = 2
        env.reset.return_value = _preference_batch()
    else:
        agent0 = _mock_sft_agent(index=0, batch_size_per_process=1, batch_size=1)
        agent1 = _mock_sft_agent(index=1, batch_size_per_process=1, batch_size=1)
        env = MagicMock()
        env.__len__.return_value = 2
        env.reset.return_value = _sft_batch()

    env.name = "mock_env"
    env.num_epochs = 0
    env.data_batch_size_per_gpu = 1
    env.world_size = 1

    with (
        _population_init_skip_per_mock_class(),
        patch("agilerl.training.llm.dataset.default_progress_bar") as mock_pbar_fn,
        patch("agilerl.training.llm.dataset.save_llm_checkpoint"),
        patch("agilerl.training.llm.rollout.safe_aggregate_metrics", return_value=0.5),
        patch("agilerl.training.llm.dataset.init_loggers", return_value=[]),
    ):
        mock_pbar_fn.return_value = MagicMock()
        train_llm_dataset(
            pop=[agent0, agent1],
            env=env,
            accelerator=None,
            max_steps=1,
            evaluation_interval=100,
            verbose=False,
        )
    assert agent0.learn.call_count == 1


def test_collect_rollouts_llm_breaks_when_vector_env_has_no_active_prompts():
    mock_agent = _make_rollout_mock_agent()

    def _mock_get_action(obs, training=True, **kwargs):
        if isinstance(obs, dict):
            input_ids = obs.get("input_ids")
            batch = int(input_ids.shape[0]) if hasattr(input_ids, "shape") else 1
        else:
            batch = len(obs)
        return ActionResult(
            [torch.ones(1, 5, dtype=torch.long) for _ in range(batch)], None, None
        )

    mock_agent.get_action.side_effect = _mock_get_action

    prompt = {
        "input_ids": torch.ones(1, 3, dtype=torch.long),
        "attention_mask": torch.ones(1, 3, dtype=torch.long),
    }
    mock_env = MagicMock(spec=["reset", "step", "get_trajectories"])
    mock_env.reset.return_value = [prompt]
    mock_env.step.return_value = None
    mock_env.get_trajectories.return_value = (
        [torch.ones(1, 8, dtype=torch.long)],
        [torch.ones(1, 7, dtype=torch.bool)],
        [torch.zeros(1, 7, dtype=torch.long)],
        [torch.ones(2, dtype=torch.float32)],
        1,
        None,  # all_sampling_logps (added to get_trajectories' return)
    )

    _ = collect_rollouts_llm(
        agent=mock_agent,
        env=mock_env,
        n_steps=5,
        batch_size=1,
        group_seed=123,
    )

    assert mock_agent.get_action.call_count == 1
    assert mock_env.step.call_count == 1


def test_train_llm_rollout_closes_envs_on_teardown():
    """The batch rollout env and a lazily-built eval env are closed on teardown."""
    mock_agent = _make_rollout_mock_agent(spec=LLMPPO)
    test_env = MagicMock()
    env_factory = MagicMock(return_value=test_env)

    with (
        patch(
            "agilerl.training.llm.rollout.default_progress_bar",
            return_value=MagicMock(),
        ),
        patch("agilerl.training.llm.rollout.init_loggers", return_value=[]),
        patch("agilerl.training.llm.rollout.safe_aggregate_metrics", return_value=0.5),
        patch("agilerl.training.llm.rollout.save_llm_checkpoint"),
        patch("agilerl.training.llm.rollout.RolloutCollector") as mock_batch_env_cls,
        patch("agilerl.training.llm.rollout.collect_rollouts_llm") as mock_collect,
    ):
        mock_collect.return_value = _rollout_collect_return(batch_steps=3)
        rollout_env = mock_batch_env_cls.return_value

        train_llm_rollout(
            pop=[mock_agent],
            env_factory=env_factory,
            max_turns=2,
            init_hp={"BATCH_SIZE": 1, "ALGO": mock_agent.algo},
            max_steps=3,
            evaluation_interval=1,
            verbose=False,
            accelerator=None,
        )

    rollout_env.close.assert_called_once()
    test_env.close.assert_called_once()


def test_validate_finetune_args_warns_when_checkpoint_steps_ignored():
    """Periodic checkpoints are skipped while evolution is active; say so."""
    from agilerl.training.llm.common import _validate_finetune_args

    with pytest.warns(
        UserWarning,
        match="'checkpoint_steps' is set, but evolution is active",
    ):
        _validate_finetune_args(
            2,
            MagicMock(
                architecture_mut=0, new_layer_prob=0, parameters_mut=0, activation_mut=0
            ),
            MagicMock(
                architecture_mut=0, new_layer_prob=0, parameters_mut=0, activation_mut=0
            ),
            None,
            None,
            [_mock_dpo_agent()],
            DPO,
            "unused",
            checkpoint_steps=10,
        )


def test_train_llm_rollout_syncs_ranks_after_evaluation():
    """Distributed evaluation must rendezvous before training continues."""
    mock_agent = _make_rollout_mock_agent(spec=LLMPPO)
    accelerator = Accelerator()

    with (
        patch(
            "agilerl.training.llm.rollout.default_progress_bar",
            return_value=MagicMock(),
        ),
        patch("agilerl.training.llm.rollout.init_loggers", return_value=[]),
        patch("agilerl.training.llm.rollout.safe_aggregate_metrics", return_value=0.5),
        patch("agilerl.training.llm.rollout.save_llm_checkpoint"),
        patch("agilerl.training.llm.rollout.RolloutCollector"),
        patch("agilerl.training.llm.rollout.collect_rollouts_llm") as mock_collect,
        patch.object(accelerator, "wait_for_everyone") as mock_wait,
    ):
        mock_collect.return_value = _rollout_collect_return(batch_steps=3)

        train_llm_rollout(
            pop=[mock_agent],
            env_factory=MagicMock(),
            max_turns=2,
            init_hp={"BATCH_SIZE": 1, "ALGO": mock_agent.algo},
            max_steps=3,
            evaluation_interval=1,
            verbose=False,
            accelerator=accelerator,
        )

    assert mock_agent.test.call_count == 1
    mock_wait.assert_called()


def _non_main_accelerator():
    """Single-process stand-in for a non-main DP rank."""
    return SimpleNamespace(
        is_main_process=False,
        num_processes=1,
        process_index=0,
        device=torch.device("cpu"),
        wait_for_everyone=lambda: None,
        gather=lambda t: t,
    )


def _two_rank_accelerator(*, peer_empty: bool):
    """Two-rank stand-in whose gather reports a peer empty/non-empty flag."""

    def gather(t):
        peer = torch.tensor([int(peer_empty)], device=t.device, dtype=t.dtype)
        return torch.cat([t.reshape(-1), peer])

    return SimpleNamespace(
        is_main_process=True,
        num_processes=2,
        process_index=0,
        device=torch.device("cpu"),
        wait_for_everyone=lambda: None,
        gather=gather,
    )


class TestAnyRankEmptyBatch:
    def test_returns_local_flag_without_accelerator(self):
        assert _any_rank_empty_batch(True, None) is True
        assert _any_rank_empty_batch(False, None) is False

    def test_is_true_when_any_peer_is_empty(self):
        acc = _two_rank_accelerator(peer_empty=True)

        assert _any_rank_empty_batch(False, acc) is True

    def test_is_false_when_every_rank_has_data(self):
        acc = _two_rank_accelerator(peer_empty=False)

        assert _any_rank_empty_batch(False, acc) is False

    def test_is_true_when_local_and_peer_are_empty(self):
        acc = _two_rank_accelerator(peer_empty=True)

        assert _any_rank_empty_batch(True, acc) is True


def test_train_llm_dataset_rejects_an_unsharded_env_on_distributed_runs():
    mock_agent = _mock_sft_agent()
    mock_env = MagicMock()
    mock_env.__len__.return_value = 2
    mock_env.name = "mock_env"
    mock_env.data_batch_size_per_gpu = 1
    mock_env.world_size = 2  # env sharded for 2 ranks, run has 1
    with pytest.raises(ValueError, match="data-parallel ranks"):
        train_llm_dataset(pop=[mock_agent], env=mock_env, max_steps=1)


def test_train_llm_dataset_non_main_ranks_clear_metrics():
    mock_agent = _mock_sft_agent()
    mock_env = MagicMock()
    mock_env.__len__.return_value = 1
    mock_env.name = "mock_env"
    mock_env.data_batch_size_per_gpu = 1
    mock_env.world_size = 1
    mock_env.num_epochs = 0
    mock_env.reset.return_value = _sft_batch()
    with (
        _population_init_skip_per_mock_class(),
        patch("agilerl.training.llm.dataset.default_progress_bar") as mock_pbar_fn,
        patch("agilerl.training.llm.dataset.init_loggers", return_value=[]),
        patch(
            "agilerl.training.llm.dataset.Population.clear_agent_metrics"
        ) as mock_clear,
        patch("agilerl.training.llm.dataset.Population.report_metrics") as mock_report,
    ):
        mock_pbar_fn.return_value = MagicMock()
        train_llm_dataset(
            pop=[mock_agent],
            env=mock_env,
            max_steps=1,
            accelerator=_non_main_accelerator(),
        )
    mock_clear.assert_called()
    mock_report.assert_not_called()


class TestTrainLlmRolloutDistributedBranches:
    def test_non_main_ranks_clear_metrics(self):
        mock_agent = _make_rollout_mock_agent(spec=GRPO)
        with (
            patch(
                "agilerl.training.llm.rollout.default_progress_bar",
                return_value=MagicMock(),
            ),
            patch("agilerl.training.llm.rollout.init_loggers", return_value=[]),
            patch(
                "agilerl.training.llm.rollout.safe_aggregate_metrics",
                return_value=0.5,
            ),
            patch("agilerl.training.llm.rollout.save_llm_checkpoint"),
            patch("agilerl.training.llm.rollout.RolloutCollector"),
            patch(
                "agilerl.training.llm.rollout.collect_rollouts_llm",
                return_value=_rollout_collect_return(batch_steps=3),
            ),
            patch(
                "agilerl.training.llm.rollout.Population.clear_agent_metrics"
            ) as mock_clear,
            patch(
                "agilerl.training.llm.rollout.Population.report_metrics"
            ) as mock_report,
        ):
            train_llm_rollout(
                pop=[mock_agent],
                env_factory=MagicMock(),
                max_turns=2,
                init_hp={"BATCH_SIZE": 1, "ALGO": mock_agent.algo},
                max_steps=3,
                evaluation_interval=100,
                verbose=False,
                accelerator=_non_main_accelerator(),
            )
        mock_clear.assert_called()
        mock_report.assert_not_called()

    def test_cross_rank_alignment_pads_turn_ids_to_the_global_width(self):
        mock_agent = _make_rollout_mock_agent(spec=GRPO)
        mock_agent.pad_token_id = 0
        rect_ids = torch.ones(1, 10, dtype=torch.long)
        rect_masks = torch.ones(1, 9, dtype=torch.bool)
        rect_rewards = torch.ones(1, 2)
        with (
            patch(
                "agilerl.training.llm.rollout.default_progress_bar",
                return_value=MagicMock(),
            ),
            patch("agilerl.training.llm.rollout.init_loggers", return_value=[]),
            patch(
                "agilerl.training.llm.rollout.safe_aggregate_metrics",
                return_value=0.5,
            ),
            patch("agilerl.training.llm.rollout.save_llm_checkpoint"),
            patch("agilerl.training.llm.rollout.RolloutCollector"),
            patch(
                "agilerl.training.llm.rollout.collect_rollouts_llm",
                return_value=_rollout_collect_return(batch_steps=3),
            ),
            patch(
                "agilerl.training.llm.rollout.needs_cross_rank_seq_padding",
                return_value=True,
            ),
            patch(
                "agilerl.training.llm.rollout.align_completion_batch_shapes_across_ranks",
                return_value=(rect_ids, rect_masks, rect_rewards),
            ) as mock_align,
            patch(
                "agilerl.training.llm.rollout.data_parallel_topology",
                return_value=(0, 2),
            ),
        ):
            train_llm_rollout(
                pop=[mock_agent],
                env_factory=MagicMock(),
                max_turns=2,
                init_hp={"BATCH_SIZE": 1, "ALGO": mock_agent.algo},
                max_steps=3,
                evaluation_interval=100,
                verbose=False,
                accelerator=_non_main_accelerator(),
            )
        mock_align.assert_called_once()
        experiences, kwargs = (
            mock_agent.learn.call_args.args[0],
            mock_agent.learn.call_args.kwargs,
        )
        assert experiences == (rect_ids, rect_masks, rect_rewards)
        # turn_ids padded from the collected width up to the aligned mask width.
        assert kwargs["turn_ids"].shape == (1, 9)
        assert kwargs["turn_ids"][0, -1].item() == -1

    def test_skips_learn_when_peer_rank_is_empty(self):
        mock_agent = _make_rollout_mock_agent(spec=GRPO)

        with (
            patch(
                "agilerl.training.llm.rollout.default_progress_bar",
                return_value=MagicMock(),
            ),
            patch("agilerl.training.llm.rollout.init_loggers", return_value=[]),
            patch(
                "agilerl.training.llm.rollout.safe_aggregate_metrics",
                return_value=0.5,
            ),
            patch("agilerl.training.llm.rollout.save_llm_checkpoint"),
            patch("agilerl.training.llm.rollout.RolloutCollector"),
            patch(
                "agilerl.training.llm.rollout.collect_rollouts_llm",
                return_value=_rollout_collect_return(batch_steps=3),
            ) as mock_collect,
            pytest.warns(UserWarning, match="no usable turns"),
            pytest.raises(RuntimeError, match="made no progress"),
        ):
            train_llm_rollout(
                pop=[mock_agent],
                env_factory=MagicMock(),
                max_turns=2,
                init_hp={"BATCH_SIZE": 1, "ALGO": mock_agent.algo},
                max_steps=1000,
                evaluation_interval=100,
                verbose=False,
                accelerator=_two_rank_accelerator(peer_empty=True),
            )

        mock_agent.learn.assert_not_called()
        assert mock_collect.call_count == 8

    def test_learns_when_every_rank_has_data(self):
        mock_agent = _make_rollout_mock_agent(spec=GRPO)

        with (
            patch(
                "agilerl.training.llm.rollout.default_progress_bar",
                return_value=MagicMock(),
            ),
            patch("agilerl.training.llm.rollout.init_loggers", return_value=[]),
            patch(
                "agilerl.training.llm.rollout.safe_aggregate_metrics",
                return_value=0.5,
            ),
            patch("agilerl.training.llm.rollout.save_llm_checkpoint"),
            patch("agilerl.training.llm.rollout.RolloutCollector"),
            patch(
                "agilerl.training.llm.rollout.collect_rollouts_llm",
                return_value=_rollout_collect_return(batch_steps=3),
            ),
        ):
            train_llm_rollout(
                pop=[mock_agent],
                env_factory=MagicMock(),
                max_turns=2,
                init_hp={"BATCH_SIZE": 1, "ALGO": mock_agent.algo},
                max_steps=6,
                evaluation_interval=100,
                verbose=False,
                accelerator=_two_rank_accelerator(peer_empty=False),
            )

        mock_agent.learn.assert_called()


class _LLMHPParam:
    def __init__(self, value):
        self.value = value


class _LLMHPConfig:
    """Minimal HyperparameterConfig stand-in for the LLM migration path."""

    def __init__(self, values):
        self._params = {name: _LLMHPParam(v) for name, v in values.items()}

    def __iter__(self):
        return iter(self._params)

    def __bool__(self):
        return bool(self._params)

    def names(self):
        return list(self._params)

    def __getitem__(self, key):
        return self._params[key]


class _LLMFinetuneAgent:
    """Fake :class:`LLMAlgorithm`."""

    def __init__(self, index, subpopulation_id, fitness, lr=1e-3):
        self.index = index
        self.subpopulation_id = subpopulation_id
        self.fitness = [fitness]
        self.lr = lr
        self.accelerator = None
        self.mut = "stale-mut"
        self.registry = SimpleNamespace(
            hp_config=_LLMHPConfig({"lr": lr}), optimizers=[]
        )
        self.clean_up_calls = 0

    def clone(self, index=None, wrap=False):
        twin = _LLMFinetuneAgent(
            self.index if index is None else index,
            self.subpopulation_id,
            self.fitness[-1],
            lr=self.lr,
        )
        twin.mut = self.mut
        for name in self.registry.hp_config.names():
            twin.registry.hp_config[name].value = self.registry.hp_config[name].value
        return twin

    def clean_up(self):
        self.clean_up_calls += 1

    def mutation_hook(self):
        pass

    def reinit_optimizers(self, optimizer=None):
        pass


class TestMultiFrequencyLLMEvolution:
    """The LLM finetuners' evolution entry point drives the real operator end to end."""

    def test_run_selection_and_mutation_drives_the_real_operator(
        self, monkeypatch, tmp_path
    ):
        monkeypatch.setattr(
            "agilerl.hpo.multi_frequency.LLMAlgorithm", _LLMFinetuneAgent
        )
        monkeypatch.setattr("agilerl.utils.utils.LLMAlgorithm", _LLMFinetuneAgent)
        saved: list = []
        monkeypatch.setattr(
            "agilerl.utils.utils.save_llm_checkpoint",
            lambda agent, path: saved.append(agent),
        )
        population = [_LLMFinetuneAgent(i, i // 4, 0.0) for i in range(8)]
        strategy = MultiFrequencySelection(
            population_size=8,
            n_subpopulations=2,
            evolution_frequency_ratios=[1, 2],
            n_winners=1,
            n_survivors=1,
            n_open_for_migration=1,
            n_losers=1,
            seed=0,
        )
        mutation = Mutations(
            no_mutation=1.0,
            architecture=0.0,
            new_layer_prob=0.0,
            parameters=0.0,
            activation=0.0,
            rl_hp=0.0,
            mutation_sd=0.1,
            rand_seed=0,
            device="cpu",
        )

        for cycle in range(3):
            rank_population_by_subpopulation(population)
            doomed = {weakest_agent_index(population, subpop=0)}
            if cycle % 2 == 1:
                doomed.add(weakest_agent_index(population, subpop=1))

            population = run_selection_and_mutation(
                strategy,
                population=population,
                mutation=mutation,
                env_name="Env",
                algo="GRPO",
                language_model=True,
                save_elite=True,
                elite_path=str(tmp_path / "elite"),
            )

            assert not (doomed & {a.index for a in population})
            assert len(population) == 8
            assert Counter(a.subpopulation_id for a in population) == Counter(
                {0: 4, 1: 4}
            )
            assert len({a.index for a in population}) == 8  # indices stay unique
            assert all(a.mut == "None" for a in population)

        assert len(saved) == 3  # the live elite is checkpointed every cycle
        assert all(isinstance(a, _LLMFinetuneAgent) for a in population)


class TestTrainLlmRolloutCrossRankPadding:
    """Multi-rank Liger losses pad every rank to one global sequence length.

    The padding path re-pads ``turn_ids`` to the widened mask, so it asserts the
    turn ids it was handed actually exist rather than indexing ``None``.
    """

    @staticmethod
    def _batch(*, turn_ids):
        batch = MagicMock()
        batch.is_empty = False
        batch.rewards = torch.ones(1, 2, dtype=torch.float32)
        batch.token_ids = torch.ones(1, 8, dtype=torch.long)
        batch.action_masks = torch.ones(1, 7, dtype=torch.bool)
        batch.turn_ids = turn_ids
        batch.experiences.return_value = (
            batch.token_ids,
            batch.action_masks,
            batch.rewards,
        )
        return batch

    def _run(self, batch):
        mock_agent = _make_rollout_mock_agent(spec=GRPO)
        mock_agent.pad_token_id = 0  # read to build the alignment call
        aligned = (
            torch.ones(1, 9, dtype=torch.long),
            torch.ones(1, 9, dtype=torch.bool),
            torch.ones(1, 2, dtype=torch.float32),
        )
        with (
            patch(
                "agilerl.training.llm.rollout.default_progress_bar",
                return_value=MagicMock(),
            ),
            patch("agilerl.training.llm.rollout.init_loggers", return_value=[]),
            patch(
                "agilerl.training.llm.rollout.safe_aggregate_metrics",
                return_value=0.5,
            ),
            patch("agilerl.training.llm.rollout.save_llm_checkpoint"),
            patch("agilerl.training.llm.rollout.RolloutCollector"),
            patch(
                "agilerl.training.llm.rollout.collect_rollouts_llm",
                return_value=_rollout_collect_return(batch_steps=3),
            ),
            patch(
                "agilerl.training.llm.rollout.collate_llm_rollouts",
                return_value=batch,
            ),
            patch(
                "agilerl.training.llm.rollout.needs_cross_rank_seq_padding",
                return_value=True,
            ),
            patch(
                "agilerl.training.llm.rollout.align_completion_batch_shapes_across_ranks",
                return_value=aligned,
            ),
        ):
            train_llm_rollout(
                pop=[mock_agent],
                env_factory=MagicMock(),
                max_turns=2,
                init_hp={"BATCH_SIZE": 1, "ALGO": mock_agent.algo},
                max_steps=3,
                evaluation_interval=100,
                verbose=False,
                accelerator=Accelerator(),
            )
        return mock_agent

    def test_turn_ids_are_padded_out_to_the_aligned_mask(self):
        mock_agent = self._run(
            self._batch(turn_ids=torch.zeros(1, 7, dtype=torch.long))
        )

        _args, learn_kwargs = mock_agent.learn.call_args
        padded = learn_kwargs["turn_ids"]
        # Widened to the aligned mask length, the new positions marked -1.
        assert padded.shape[1] == 9
        assert padded[0, -1].item() == -1

    def test_a_batch_without_turn_ids_is_rejected(self):
        with pytest.raises(RuntimeError, match="aligned batches are non-empty"):
            self._run(self._batch(turn_ids=None))
