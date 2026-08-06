# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

import itertools
from collections import Counter
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
from unittest.mock import ANY, MagicMock, Mock, call, patch

import pytest
import torch

from agilerl import HAS_LLM_DEPENDENCIES

if not HAS_LLM_DEPENDENCIES:
    pytest.skip("LLM dependencies not installed", allow_module_level=True)

from agilerl.algorithms import DPO, GRPO, LLMPPO, LLMREINFORCE
from agilerl.algorithms.core import ActionResult
from agilerl.algorithms.core.base import MultiAgentRLAlgorithm
from agilerl.algorithms.sft import SFT
from agilerl.hpo.multi_frequency import MultiFrequencySelection
from agilerl.hpo.mutation import Mutations
from agilerl.population import Population
from agilerl.rollouts.on_policy import collect_rollouts_llm
from agilerl.training.llm import (
    finetune_llm_multiturn,
    finetune_llm_preference,
    finetune_llm_reasoning,
    finetune_llm_sft,
)
from agilerl.utils.utils import run_selection_and_mutation
from tests.helper_functions import (
    rank_population_by_subpopulation,
    weakest_agent_index,
)

pytestmark = pytest.mark.llm


def test_train_llm_module_emits_deprecation_warning():
    import importlib
    import sys

    sys.modules.pop("agilerl.training.train_llm", None)
    with pytest.warns(FutureWarning, match="agilerl.training.train_llm is deprecated"):
        importlib.import_module("agilerl.training.train_llm")


def _finetune_module_path(finetune_fn):
    return {
        finetune_llm_reasoning: "agilerl.training.llm.reasoning",
        finetune_llm_preference: "agilerl.training.llm.preference",
        finetune_llm_sft: "agilerl.training.llm.sft",
        finetune_llm_multiturn: "agilerl.training.llm.multiturn",
    }[finetune_fn]


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
            isinstance(agent, MultiAgentRLAlgorithm) for agent in agents
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


# ---------------------------------------------------------------------------
# Helpers: Multiturn mock agents and environments
# ---------------------------------------------------------------------------


def _make_multiturn_mock_agent(*, spec=LLMPPO):
    """Build a multiturn agent mock (LLMPPO/LLMREINFORCE/GRPO)."""
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


def _multiturn_collect_return(*, batch_steps=3):
    """Standard return value for a mocked collect_rollouts_llm call."""
    return (
        [torch.ones(1, 8, dtype=torch.long)],
        [torch.ones(1, 8, dtype=torch.bool)],
        [torch.zeros(1, 8, dtype=torch.long)],
        [torch.ones(2, dtype=torch.float32)],
        batch_steps,
        42,
        None,  # all_sampling_logps
    )


# ---------------------------------------------------------------------------
# TestFinetuneLlmReasoning
# ---------------------------------------------------------------------------


class TestFinetuneLlmReasoning:
    def test_finetune_llm_reasoning_basic_training_loop(self):
        mock_agent = _mock_grpo_agent()

        mock_env = MagicMock()
        mock_env.__len__.return_value = 6
        mock_env.reset.return_value = "initial_prompts"
        mock_env.step.return_value = ("next_prompts", torch.tensor([2.0, 3.0]))
        mock_env.data_batch_size_per_gpu = 1

        with (
            patch(
                "agilerl.training.llm.reasoning.default_progress_bar"
            ) as mock_pbar_fn,
            patch("agilerl.training.llm.reasoning.aggregate_metrics_across_gpus") as mock_agg,
            patch("agilerl.training.llm.reasoning.save_llm_checkpoint"),
            patch("agilerl.training.llm.reasoning.init_loggers", return_value=[]),
        ):
            mock_pbar_fn.return_value = MagicMock()
            mock_agg.return_value = 0.5
            result = finetune_llm_reasoning(
                pop=[mock_agent],
                env=mock_env,
                evaluation_interval=2,
                max_reward=2.0,

            )
            # finetune_llm_* must return (population, fitnesses) — same contract
            # as the non-LLM train fns — so the `agilerl train` CLI can unpack
            # the result (otherwise it raises ValueError on a 1-element return).
            assert isinstance(result, tuple)
            assert len(result) == 2
            agents, fitnesses = result
            assert mock_agent in agents
            assert isinstance(fitnesses, list)
            assert mock_env.reset.call_count == 1
            assert mock_env.reset.call_args == call(reset_dataloaders=True)
            assert mock_agent.get_action.call_count == 6
            assert mock_env.step.call_count == 6
            assert mock_agent.learn.call_count == 6
            assert mock_agent.test.call_count == 3

    def test_finetune_llm_reasoning_with_wandb_and_checkpoints(self):
        mock_agent = _mock_grpo_agent()

        mock_env = MagicMock()
        mock_env.__len__.return_value = 6
        mock_env.reset.return_value = "initial_prompts"
        mock_env.step.return_value = ("next_prompts", torch.tensor([2.0, 3.0]))
        mock_env.data_batch_size_per_gpu = 1

        with (
            patch(
                "agilerl.training.llm.reasoning.default_progress_bar"
            ) as mock_pbar_fn,
            patch("agilerl.training.llm.reasoning.init_loggers") as mock_init_loggers,
            patch("agilerl.training.llm.reasoning.aggregate_metrics_across_gpus") as mock_agg,
            patch("agilerl.training.llm.reasoning.save_llm_checkpoint") as mock_save,
        ):
            mock_pbar_fn.return_value = MagicMock()
            mock_init_loggers.return_value = []
            mock_agg.return_value = 0.5

            finetune_llm_reasoning(
                pop=[mock_agent],
                env=mock_env,
                save_elite=True,
                wb=True,
                wandb_api_key="fake_key",
                evaluation_interval=3,

                max_reward=2.0,
                checkpoint_steps=6,
            )

            mock_init_loggers.assert_called_once()
            assert mock_init_loggers.call_args.kwargs["wb"] is True
            assert mock_save.call_count == 1
            assert mock_agent.test.call_count == 2

    def test_finetune_llm_reasoning_periodic_checkpoints_use_checkpoint_path(self):
        mock_agent = _mock_grpo_agent()

        mock_env = MagicMock()
        mock_env.__len__.return_value = 6
        mock_env.reset.return_value = "initial_prompts"
        mock_env.step.return_value = ("next_prompts", torch.tensor([2.0, 3.0]))
        mock_env.data_batch_size_per_gpu = 1

        with (
            patch("agilerl.training.llm.reasoning.default_progress_bar"),
            patch("agilerl.training.llm.reasoning.init_loggers") as mock_init_loggers,
            patch("agilerl.training.llm.reasoning.aggregate_metrics_across_gpus") as mock_agg,
            patch("agilerl.training.llm.reasoning.save_llm_checkpoint") as mock_save,
        ):
            mock_init_loggers.return_value = []
            mock_agg.return_value = 0.5

            finetune_llm_reasoning(
                pop=[mock_agent],
                env=mock_env,
                save_elite=False,
                evaluation_interval=3,

                checkpoint_steps=6,
                checkpoint_path="/tmp/llm_ckpts",
            )

            assert mock_save.call_count == 1
            assert mock_save.call_args.args == (mock_agent, "/tmp/llm_ckpts")

    def test_finetune_llm_reasoning_evolvable_training_loop(self):
        mock_agent = _mock_grpo_agent()

        mock_env = MagicMock()
        mock_env.__len__.return_value = 6
        mock_env.reset.return_value = "initial_prompts"
        mock_env.step.return_value = ("next_prompts", torch.tensor([2.0, 3.0]))
        mock_env.data_batch_size_per_gpu = 1

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
            patch(
                "agilerl.training.llm.reasoning.default_progress_bar"
            ) as mock_pbar_fn,
            patch("agilerl.training.llm.reasoning.aggregate_metrics_across_gpus") as mock_agg,
            patch("agilerl.training.llm.reasoning.save_llm_checkpoint"),
            patch("agilerl.training.llm.reasoning.init_loggers", return_value=[]),
            patch(
                "agilerl.training.llm.reasoning.run_selection_and_mutation"
            ) as mock_run_selection_and_mutation,
        ):
            mock_pbar_fn.return_value = MagicMock()
            mock_run_selection_and_mutation.return_value = [mock_agent]
            mock_agg.return_value = 0.5

            finetune_llm_reasoning(
                pop=[mock_agent],
                env=mock_env,
                evaluation_interval=2,
                max_reward=2.0,
                evo_steps=1,

                selection_strategy=Mock(),
                mutation=mutation,
            )
            assert mock_env.reset.call_count == 1
            assert mock_env.reset.call_args == call(reset_dataloaders=True)
            assert mock_agent.get_action.call_count == 6
            assert mock_env.step.call_count == 6
            assert mock_agent.learn.call_count == 6
            assert mock_agent.test.call_count == 3
            assert mock_run_selection_and_mutation.call_count == 6

    def test_finetune_llm_reasoning_deprecated_tournament_argument(self):
        mock_agent = _mock_grpo_agent()

        mock_env = MagicMock()
        mock_env.__len__.return_value = 6
        mock_env.reset.return_value = "initial_prompts"
        mock_env.step.return_value = ("next_prompts", torch.tensor([2.0, 3.0]))
        mock_env.data_batch_size_per_gpu = 1

        strategy = Mock()

        mutation = MagicMock()
        mutation.architecture_mut = 0
        mutation.new_layer_prob = 0
        mutation.parameters_mut = 0
        mutation.activation_mut = 0

        with (
            patch(
                "agilerl.training.llm.reasoning.default_progress_bar"
            ) as mock_pbar_fn,
            patch("agilerl.training.llm.reasoning.aggregate_metrics_across_gpus") as mock_agg,
            patch("agilerl.training.llm.reasoning.save_llm_checkpoint"),
            patch("agilerl.training.llm.reasoning.init_loggers", return_value=[]),
            patch(
                "agilerl.training.llm.reasoning.run_selection_and_mutation"
            ) as mock_run_selection_and_mutation,
        ):
            mock_pbar_fn.return_value = MagicMock()
            mock_run_selection_and_mutation.return_value = [mock_agent]
            mock_agg.return_value = 0.5

            with pytest.warns(DeprecationWarning, match="'tournament' argument"):
                finetune_llm_reasoning(
                    pop=[mock_agent],
                    env=mock_env,
                    evaluation_interval=2,
                    max_reward=2.0,
                    evo_steps=1,

                    tournament=strategy,
                    mutation=mutation,
                )

        assert mock_run_selection_and_mutation.call_count == 6
        assert mock_run_selection_and_mutation.call_args.args[0] is strategy

    def test_finetune_llm_reasoning_warns_checkpoint_steps_during_evolution(self):
        mock_agent = _mock_grpo_agent()
        mock_env = MagicMock()
        mock_env.__len__.return_value = 3
        mock_env.reset.return_value = "initial_prompts"
        mock_env.step.return_value = ("next_prompts", torch.tensor([2.0, 3.0]))
        mock_env.data_batch_size_per_gpu = 1

        mutation = MagicMock()
        mutation.architecture_mut = 0
        mutation.new_layer_prob = 0
        mutation.parameters_mut = 0
        mutation.activation_mut = 0

        with (
            patch(
                "agilerl.training.llm.reasoning.default_progress_bar"
            ) as mock_pbar_fn,
            patch("agilerl.training.llm.reasoning.aggregate_metrics_across_gpus") as mock_agg,
            patch("agilerl.training.llm.reasoning.save_llm_checkpoint"),
            patch("agilerl.training.llm.reasoning.init_loggers", return_value=[]),
            patch(
                "agilerl.training.llm.reasoning.run_selection_and_mutation",
                return_value=[mock_agent],
            ),
        ):
            mock_pbar_fn.return_value = MagicMock()
            mock_agg.return_value = 0.5
            with pytest.warns(
                UserWarning,
                match=r"checkpoint_steps.*evolution is active",
            ):
                finetune_llm_reasoning(
                    pop=[mock_agent],
                    env=mock_env,
                    evaluation_interval=2,
                    evo_steps=1,
                    selection_strategy=Mock(),
                    mutation=mutation,
                    checkpoint_steps=3,
                )

    def test_finetune_llm_reasoning_saves_elite_at_end(self):
        weaker = _mock_grpo_agent()
        weaker.fitness = [0.1]
        stronger = _mock_grpo_agent()
        stronger.fitness = [0.9]

        mock_env = MagicMock()
        mock_env.__len__.return_value = 1
        mock_env.reset.return_value = "initial_prompts"
        mock_env.step.return_value = ("next_prompts", torch.tensor([2.0]))
        mock_env.data_batch_size_per_gpu = 1

        with (
            patch(
                "agilerl.training.llm.reasoning.default_progress_bar"
            ) as mock_pbar_fn,
            patch("agilerl.training.llm.reasoning.aggregate_metrics_across_gpus") as mock_agg,
            patch("agilerl.training.llm.reasoning.save_llm_checkpoint") as mock_save,
            patch("agilerl.training.llm.reasoning.init_loggers", return_value=[]),
            _population_init_skip_per_mock_class(),
        ):
            mock_pbar_fn.return_value = MagicMock()
            mock_agg.return_value = 0.5
            finetune_llm_reasoning(
                pop=[weaker, stronger],
                env=mock_env,
                evaluation_interval=10,
                save_elite=True,
                elite_path="/tmp/elite",
            )

            assert mock_save.call_args_list[-1] == call(stronger, "/tmp/elite")

    @pytest.mark.parametrize(
        "finetune_fn",
        [finetune_llm_reasoning, finetune_llm_preference],
    )
    def test_finetune_llm_reasoning_evo_steps_not_set(self, finetune_fn):
        with pytest.raises(ValueError, match="evo_steps"):
            finetune_fn(
                pop=[
                    MagicMock(
                        spec=(GRPO if finetune_fn == finetune_llm_reasoning else DPO),
                    ),
                ],
                env=MagicMock(),
                evo_steps=None,

                selection_strategy=MagicMock(),
                mutation=MagicMock(),
            )

    def test_finetune_llm_reasoning_warning_num_epochs_and_max_steps(self):
        mock_agent = _mock_grpo_agent()

        mock_env = MagicMock()
        mock_env.__len__.return_value = 6
        mock_env.reset.return_value = "initial_prompts"
        mock_env.step.return_value = ("next_prompts", torch.tensor([2.0, 3.0]))
        mock_env.data_batch_size_per_gpu = 1

        with (
            patch(
                "agilerl.training.llm.reasoning.default_progress_bar"
            ) as mock_pbar_fn,
            patch("agilerl.training.llm.reasoning.aggregate_metrics_across_gpus") as mock_agg,
            patch("agilerl.training.llm.reasoning.save_llm_checkpoint"),
            patch("agilerl.training.llm.reasoning.init_loggers", return_value=[]),
            patch(
                "agilerl.training.llm.reasoning.run_selection_and_mutation"
            ) as mock_tsm,
        ):
            mock_pbar_fn.return_value = MagicMock()
            mock_tsm.return_value = [mock_agent]
            mock_agg.return_value = 0.5

            with pytest.warns(UserWarning, match="num_epochs"):
                finetune_llm_reasoning(
                    pop=[mock_agent],
                    env=mock_env,
                    evaluation_interval=2,
                    max_reward=2.0,
                    num_epochs=10,
                    max_steps=100,
                    evo_steps=None,
                )

    def test_finetune_llm_reasoning_max_steps_set_from_num_epochs(self):
        mock_agent = _mock_grpo_agent()

        mock_env = MagicMock()
        mock_env.__len__.return_value = 3
        mock_env.reset.return_value = "initial_prompts"
        mock_env.step.return_value = ("next_prompts", torch.tensor([2.0, 3.0]))
        mock_env.data_batch_size_per_gpu = 1

        with (
            patch(
                "agilerl.training.llm.reasoning.default_progress_bar"
            ) as mock_pbar_fn,
            patch("agilerl.training.llm.reasoning.init_loggers", return_value=[]),
            patch("agilerl.training.llm.reasoning.aggregate_metrics_across_gpus") as mock_agg,
            patch("agilerl.training.llm.reasoning.save_llm_checkpoint") as mock_save,
        ):
            mock_pbar_fn.return_value = MagicMock()
            mock_agg.return_value = 0.5
            finetune_llm_reasoning(
                pop=[mock_agent],
                env=mock_env,
                evaluation_interval=2,
                max_reward=2.0,
                evo_steps=1,

                num_epochs=2,
                checkpoint_steps=3,
            )
            assert mock_save.call_count == 2

    def test_finetune_llm_reasoning_break_on_num_epochs(self):
        mock_agent = _mock_grpo_agent()

        mock_env = MagicMock()
        mock_env.__len__.return_value = 3
        mock_env.reset.return_value = "initial_prompts"
        mock_env.step.return_value = ("next_prompts", torch.tensor([2.0, 3.0]))
        mock_env.data_batch_size_per_gpu = 1

        with (
            patch(
                "agilerl.training.llm.reasoning.default_progress_bar"
            ) as mock_pbar_fn,
            patch("agilerl.training.llm.reasoning.init_loggers", return_value=[]),
            patch("agilerl.training.llm.reasoning.aggregate_metrics_across_gpus") as mock_agg,
            patch("agilerl.training.llm.reasoning.save_llm_checkpoint"),
        ):
            mock_pbar_fn.return_value = MagicMock()
            mock_env.num_epochs = 2
            mock_agg.return_value = 0.5
            finetune_llm_reasoning(
                pop=[mock_agent],
                env=mock_env,
                evaluation_interval=2,
                max_reward=2.0,
                evo_steps=1,

                num_epochs=2,
                checkpoint_steps=3,
            )

    def test_finetune_llm_reasoning_value_error_if_algo_not_grpo(self):
        mock_agent = MagicMock(spec=DPO)
        mock_agent.algo = "DPO"
        mock_agent.pretrained_model_name_or_path = "Qwen/Qwen2.5-0.5B"
        mock_agent.batch_size_per_process = 32
        mock_agent.batch_size = 32
        with pytest.raises(ValueError, match="reasoning"):
            finetune_llm_reasoning(
                pop=[mock_agent],
                env=MagicMock(),
                evaluation_interval=2,

            )

    def test_finetune_llm_reasoning_env_fn_uses_distinct_env_instances(self):
        agent_a = _mock_grpo_agent(index=0)
        agent_b = _mock_grpo_agent(index=1)

        env_a = MagicMock()
        env_a.__len__.return_value = 1
        env_a.name = "env_a"
        env_a.data_batch_size_per_gpu = 1
        env_a.num_epochs = 0
        env_a.reset.return_value = "prompts_a"
        env_a.step.return_value = ("next_a", torch.tensor([1.0]))

        env_b = MagicMock()
        env_b.__len__.return_value = 1
        env_b.name = "env_b"
        env_b.data_batch_size_per_gpu = 1
        env_b.num_epochs = 0
        env_b.reset.return_value = "prompts_b"
        env_b.step.return_value = ("next_b", torch.tensor([1.0]))

        env_fn = MagicMock(side_effect=[env_a, env_b])

        with (
            _population_init_skip_per_mock_class(),
            patch(
                "agilerl.training.llm.reasoning.default_progress_bar"
            ) as mock_pbar_fn,
            patch(
                "agilerl.training.llm.reasoning.aggregate_metrics_across_gpus",
                return_value=0.5,
            ),
            patch("agilerl.training.llm.reasoning.save_llm_checkpoint"),
            patch("agilerl.training.llm.reasoning.init_loggers", return_value=[]),
        ):
            mock_pbar_fn.return_value = MagicMock()
            finetune_llm_reasoning(
                pop=[agent_a, agent_b],
                env_fn=env_fn,
                max_steps=2,
                evaluation_interval=100,
                verbose=False,

            )

        assert env_fn.call_count == 2
        assert env_a.step.call_count >= 1
        assert env_b.step.call_count >= 1

    def test_finetune_llm_reasoning_max_reward_none_skips_accuracy(self):
        mock_agent = _mock_grpo_agent()

        mock_env = MagicMock()
        mock_env.__len__.return_value = 4
        mock_env.reset.return_value = "prompts"
        mock_env.step.return_value = ("next", torch.tensor([1.0]))
        mock_env.data_batch_size_per_gpu = 1

        with (
            patch(
                "agilerl.training.llm.reasoning.default_progress_bar"
            ) as mock_pbar_fn,
            patch("agilerl.training.llm.reasoning.aggregate_metrics_across_gpus") as mock_agg,
            patch("agilerl.training.llm.reasoning.save_llm_checkpoint"),
            patch("agilerl.training.llm.reasoning.init_loggers", return_value=[]),
        ):
            mock_pbar_fn.return_value = MagicMock()
            mock_agg.return_value = 0.5
            finetune_llm_reasoning(
                pop=[mock_agent],
                env=mock_env,
                evaluation_interval=2,
                max_reward=None,
                verbose=False,

            )

    def test_finetune_llm_reasoning_registers_accuracy_metric(self):
        mock_agent = _mock_grpo_agent()
        mock_agent.metrics.additional_metrics = [
            "loss",
            "kl",
            "mean_reward",
            "completion_length",
        ]

        mock_env = MagicMock()
        mock_env.__len__.return_value = 2
        mock_env.reset.return_value = "prompts"
        mock_env.step.return_value = ("next", torch.tensor([2.0, 0.0]))
        mock_env.data_batch_size_per_gpu = 1

        with (
            patch(
                "agilerl.training.llm.reasoning.default_progress_bar"
            ) as mock_pbar_fn,
            patch("agilerl.training.llm.reasoning.save_llm_checkpoint"),
            patch("agilerl.training.llm.reasoning.init_loggers", return_value=[]),
        ):
            mock_pbar_fn.return_value = MagicMock()
            finetune_llm_reasoning(
                pop=[mock_agent],
                env=mock_env,
                evaluation_interval=100,
                max_reward=2.0,
                verbose=False,

            )

        mock_agent.metrics.register.assert_called_with("accuracy")
        # rewards [2.0, 0.0] with max_reward=2.0 → accuracy 1/2
        mock_agent.metrics.log.assert_any_call("accuracy", 0.5)


# ---------------------------------------------------------------------------
# TestFinetuneLlmPreference
# ---------------------------------------------------------------------------


class TestFinetuneLlmPreference:
    def _pref_env(self, *, length=6):
        mock_env = MagicMock()
        mock_env.__len__.return_value = length
        example = {
            "prompt": ["This is a mock prompt"],
            "prompt_lengths": [10],
            "chosen": ["This is a mock chosen prompt"],
            "rejected": ["This is a mock rejected prompt"],
            "chosen_input_ids": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "chosen_attention_mask": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            "rejected_input_ids": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "rejected_attention_mask": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        }
        mock_env.reset.return_value = example
        mock_env.step.return_value = example
        mock_env.data_batch_size_per_gpu = 1
        return mock_env

    def test_finetune_llm_preference_saves_elite_at_end(self):
        weaker = _mock_dpo_agent()
        weaker.fitness = [0.2]
        stronger = _mock_dpo_agent()
        stronger.fitness = [0.8]
        mock_env = self._pref_env(length=1)

        with (
            patch(
                "agilerl.training.llm.preference.default_progress_bar"
            ) as mock_pbar_fn,
            patch("agilerl.training.llm.preference.save_llm_checkpoint") as mock_save,
            patch("agilerl.training.llm.preference.init_loggers", return_value=[]),
            _population_init_skip_per_mock_class(),
        ):
            mock_pbar_fn.return_value = MagicMock()
            finetune_llm_preference(
                pop=[weaker, stronger],
                env=mock_env,
                evaluation_interval=10,
                save_elite=True,
                elite_path="/tmp/dpo-elite",
            )

            assert mock_save.call_args_list[-1] == call(stronger, "/tmp/dpo-elite")

    def test_finetune_llm_preference_basic_training_loop(self):
        mock_agent = _mock_dpo_agent()
        mock_env = self._pref_env()

        with (
            patch(
                "agilerl.training.llm.preference.default_progress_bar"
            ) as mock_pbar_fn,
            patch("agilerl.training.llm.preference.save_llm_checkpoint"),
            patch("agilerl.training.llm.preference.init_loggers", return_value=[]),
        ):
            mock_pbar_fn.return_value = MagicMock()
            finetune_llm_preference(
                pop=[mock_agent],
                env=mock_env,
                evaluation_interval=2,

            )
            assert mock_env.reset.call_count == 1
            assert mock_env.reset.call_args == call(reset_dataloaders=True)
            assert mock_agent.get_action.call_count == 0
            assert mock_env.step.call_count == 6
            assert mock_agent.learn.call_count == 6
            assert mock_agent.test.call_count == 3

    def test_finetune_llm_preference_with_wandb_and_checkpoints(self):
        mock_agent = _mock_dpo_agent()
        mock_env = self._pref_env()

        with (
            patch(
                "agilerl.training.llm.preference.default_progress_bar"
            ) as mock_pbar_fn,
            patch("agilerl.training.llm.preference.init_loggers") as mock_init_loggers,
            patch("agilerl.training.llm.preference.save_llm_checkpoint") as mock_save,
        ):
            mock_pbar_fn.return_value = MagicMock()
            mock_init_loggers.return_value = []

            finetune_llm_preference(
                pop=[mock_agent],
                env=mock_env,
                save_elite=True,
                wb=True,
                wandb_api_key="fake_key",
                evaluation_interval=3,

                checkpoint_steps=6,
            )

            mock_init_loggers.assert_called_once()
            assert mock_init_loggers.call_args.kwargs["wb"] is True
            assert mock_save.call_count == 1
            assert mock_agent.test.call_count == 2

    def test_finetune_llm_preference_evolvable_training_loop(self):
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
            patch(
                "agilerl.training.llm.preference.default_progress_bar"
            ) as mock_pbar_fn,
            patch("agilerl.training.llm.preference.save_llm_checkpoint"),
            patch("agilerl.training.llm.preference.init_loggers", return_value=[]),
            patch(
                "agilerl.training.llm.preference.run_selection_and_mutation"
            ) as mock_tsm,
        ):
            mock_pbar_fn.return_value = MagicMock()
            mock_tsm.return_value = [mock_agent]

            finetune_llm_preference(
                pop=[mock_agent],
                env=mock_env,
                evaluation_interval=2,
                evo_steps=1,

                selection_strategy=Mock(),
                mutation=mutation,
            )
            assert mock_env.reset.call_count == 1
            assert mock_env.step.call_count == 6
            assert mock_agent.learn.call_count == 6
            assert mock_agent.test.call_count == 3
            assert mock_tsm.call_count == 6

    def test_finetune_llm_preference_warning_num_epochs_and_max_steps(self):
        mock_agent = _mock_dpo_agent()
        mock_env = self._pref_env()

        with (
            patch(
                "agilerl.training.llm.preference.default_progress_bar"
            ) as mock_pbar_fn,
            patch("agilerl.training.llm.preference.save_llm_checkpoint"),
            patch("agilerl.training.llm.preference.init_loggers", return_value=[]),
            patch(
                "agilerl.training.llm.preference.run_selection_and_mutation"
            ) as mock_tsm,
        ):
            mock_pbar_fn.return_value = MagicMock()
            mock_tsm.return_value = [mock_agent]

            with pytest.warns(UserWarning, match="num_epochs"):
                finetune_llm_preference(
                    pop=[mock_agent],
                    env=mock_env,
                    evaluation_interval=2,
                    num_epochs=10,
                    max_steps=100,
                    evo_steps=None,
                )

    def test_finetune_llm_preference_break_on_num_epochs(self):
        mock_agent = _mock_dpo_agent()
        mock_env = self._pref_env()

        with (
            patch(
                "agilerl.training.llm.preference.default_progress_bar"
            ) as mock_pbar_fn,
            patch("agilerl.training.llm.preference.init_loggers", return_value=[]),
            patch("agilerl.training.llm.preference.save_llm_checkpoint"),
        ):
            mock_pbar_fn.return_value = MagicMock()
            mock_env.num_epochs = 2
            finetune_llm_preference(
                pop=[mock_agent],
                env=mock_env,
                evaluation_interval=2,
                evo_steps=1,

                num_epochs=2,
                checkpoint_steps=3,
            )

    def test_finetune_llm_preference_value_error_if_algo_not_dpo(self):
        mock_agent = MagicMock(spec=GRPO)
        mock_agent.algo = "GRPO"
        mock_agent.pretrained_model_name_or_path = "Qwen/Qwen2.5-0.5B"
        mock_agent.batch_size_per_process = 32
        mock_agent.batch_size = 32
        with pytest.raises(ValueError, match="DPO"):
            finetune_llm_preference(
                pop=[mock_agent],
                env=MagicMock(),
                evaluation_interval=2,

            )

    def test_finetune_llm_preference_env_fn_uses_distinct_env_instances(self):
        agent_a = _mock_dpo_agent(index=0)
        agent_b = _mock_dpo_agent(index=1)

        env_a = MagicMock()
        env_a.__len__.return_value = 1
        env_a.name = "env_a"
        env_a.data_batch_size_per_gpu = 1
        env_a.num_epochs = 0
        env_a.reset.return_value = {"prompt": ["a"]}
        env_a.step.return_value = {"prompt": ["a_next"]}

        env_b = MagicMock()
        env_b.__len__.return_value = 1
        env_b.name = "env_b"
        env_b.data_batch_size_per_gpu = 1
        env_b.num_epochs = 0
        env_b.reset.return_value = {"prompt": ["b"]}
        env_b.step.return_value = {"prompt": ["b_next"]}

        env_fn = MagicMock(side_effect=[env_a, env_b])

        with (
            _population_init_skip_per_mock_class(),
            patch(
                "agilerl.training.llm.preference.default_progress_bar"
            ) as mock_pbar_fn,
            patch("agilerl.training.llm.preference.save_llm_checkpoint"),
            patch("agilerl.training.llm.preference.init_loggers", return_value=[]),
        ):
            mock_pbar_fn.return_value = MagicMock()
            finetune_llm_preference(
                pop=[agent_a, agent_b],
                env_fn=env_fn,
                max_steps=2,
                evaluation_interval=100,
                verbose=False,

            )

        assert env_fn.call_count == 2
        assert env_a.step.call_count >= 1
        assert env_b.step.call_count >= 1


# ---------------------------------------------------------------------------
# TestFinetuneLlmSft
# ---------------------------------------------------------------------------


class TestFinetuneLlmSft:
    def test_finetune_llm_sft_saves_elite_at_end(self):
        weaker = _mock_sft_agent()
        weaker.fitness = [0.2]
        stronger = _mock_sft_agent()
        stronger.fitness = [0.8]

        mock_env = MagicMock()
        mock_env.__len__.return_value = 1
        mock_env.reset.return_value = "initial_prompts"
        mock_env.step.return_value = "next_prompts"
        mock_env.data_batch_size_per_gpu = 1
        mock_env.num_epochs = 1

        with (
            patch("agilerl.training.llm.sft.default_progress_bar") as mock_pbar_fn,
            patch("agilerl.training.llm.sft.save_llm_checkpoint") as mock_save,
            patch("agilerl.training.llm.sft.init_loggers", return_value=[]),
            _population_init_skip_per_mock_class(),
        ):
            mock_pbar_fn.return_value = MagicMock()
            finetune_llm_sft(
                pop=[weaker, stronger],
                env=mock_env,
                evaluation_interval=10,
                save_elite=True,
                elite_path="/tmp/sft-elite",
            )

            assert mock_save.call_args_list[-1] == call(stronger, "/tmp/sft-elite")

    def test_finetune_llm_sft_basic_training_loop(self):
        mock_agent = _mock_sft_agent()

        mock_env = MagicMock()
        mock_env.__len__.return_value = 6
        mock_env.reset.return_value = "initial_prompts"
        mock_env.step.return_value = "next_prompts"
        mock_env.data_batch_size_per_gpu = 1

        with (
            patch("agilerl.training.llm.sft.default_progress_bar") as mock_pbar_fn,
            patch("agilerl.training.llm.sft.save_llm_checkpoint"),
            patch("agilerl.training.llm.sft.init_loggers", return_value=[]),
        ):
            mock_pbar_fn.return_value = MagicMock()
            finetune_llm_sft(
                pop=[mock_agent],
                env=mock_env,
                evaluation_interval=2,

            )
            assert mock_env.reset.call_count == 1
            assert mock_env.reset.call_args == call(reset_dataloaders=True)
            assert mock_env.step.call_count == 6
            assert mock_agent.learn.call_count == 6
            assert mock_agent.test.call_count == 3

    def test_finetune_llm_sft_with_wandb_and_checkpoints(self):
        mock_agent = _mock_sft_agent()

        mock_env = MagicMock()
        mock_env.__len__.return_value = 6
        mock_env.reset.return_value = "initial_prompts"
        mock_env.step.return_value = "next_prompts"
        mock_env.data_batch_size_per_gpu = 1

        with (
            patch("agilerl.training.llm.sft.default_progress_bar") as mock_pbar_fn,
            patch("agilerl.training.llm.sft.init_loggers") as mock_init_loggers,
            patch("agilerl.training.llm.sft.save_llm_checkpoint") as mock_save,
        ):
            mock_pbar_fn.return_value = MagicMock()
            mock_init_loggers.return_value = []

            finetune_llm_sft(
                pop=[mock_agent],
                env=mock_env,
                save_elite=True,
                wb=True,
                wandb_api_key="fake_key",
                evaluation_interval=3,

                checkpoint_steps=6,
            )

            mock_init_loggers.assert_called_once()
            assert mock_init_loggers.call_args.kwargs["wb"] is True
            assert mock_save.call_count == 1
            assert mock_agent.test.call_count == 2

    def test_finetune_llm_sft_evolvable_training_loop(self):
        mock_agent = _mock_sft_agent()

        mock_env = MagicMock()
        mock_env.__len__.return_value = 6
        mock_env.reset.return_value = "initial_prompts"
        mock_env.step.return_value = "next_prompts"
        mock_env.data_batch_size_per_gpu = 1

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
            patch("agilerl.training.llm.sft.default_progress_bar") as mock_pbar_fn,
            patch("agilerl.training.llm.sft.save_llm_checkpoint"),
            patch("agilerl.training.llm.sft.init_loggers", return_value=[]),
            patch("agilerl.training.llm.sft.run_selection_and_mutation") as mock_tsm,
        ):
            mock_pbar_fn.return_value = MagicMock()
            mock_tsm.return_value = [mock_agent]

            finetune_llm_sft(
                pop=[mock_agent],
                env=mock_env,
                evaluation_interval=2,
                evo_steps=1,

                selection_strategy=Mock(),
                mutation=mutation,
            )
            assert mock_env.reset.call_count == 1
            assert mock_env.step.call_count == 6
            assert mock_agent.learn.call_count == 6
            assert mock_agent.test.call_count == 3
            assert mock_tsm.call_count == 6

    def test_finetune_llm_sft_warning_num_epochs_and_max_steps(self):
        mock_agent = _mock_sft_agent()

        mock_env = MagicMock()
        mock_env.__len__.return_value = 6
        mock_env.reset.return_value = "initial_prompts"
        mock_env.step.return_value = "next_prompts"
        mock_env.data_batch_size_per_gpu = 1

        with (
            patch("agilerl.training.llm.sft.default_progress_bar") as mock_pbar_fn,
            patch("agilerl.training.llm.sft.save_llm_checkpoint"),
            patch("agilerl.training.llm.sft.init_loggers", return_value=[]),
        ):
            mock_pbar_fn.return_value = MagicMock()
            with pytest.warns(UserWarning, match="num_epochs"):
                finetune_llm_sft(
                    pop=[mock_agent],
                    env=mock_env,
                    evaluation_interval=2,
                    num_epochs=10,
                    max_steps=100,
                    evo_steps=None,
                )

    def test_finetune_llm_sft_break_on_num_epochs(self):
        mock_agent = _mock_sft_agent()

        mock_env = MagicMock()
        mock_env.__len__.return_value = 3
        mock_env.reset.return_value = "initial_prompts"
        mock_env.step.return_value = "next_prompts"
        mock_env.data_batch_size_per_gpu = 1

        with (
            patch("agilerl.training.llm.sft.default_progress_bar") as mock_pbar_fn,
            patch("agilerl.training.llm.sft.init_loggers", return_value=[]),
            patch("agilerl.training.llm.sft.save_llm_checkpoint"),
        ):
            mock_pbar_fn.return_value = MagicMock()
            mock_env.num_epochs = 2
            finetune_llm_sft(
                pop=[mock_agent],
                env=mock_env,
                evaluation_interval=2,
                evo_steps=1,

                num_epochs=2,
                checkpoint_steps=3,
            )

    def test_finetune_llm_sft_value_error_if_algo_not_sft(self):
        mock_agent = MagicMock(spec=GRPO)
        mock_agent.algo = "GRPO"
        mock_agent.pretrained_model_name_or_path = "Qwen/Qwen2.5-0.5B"
        mock_agent.batch_size_per_process = 32
        mock_agent.batch_size = 32
        with pytest.raises(ValueError, match="SFT"):
            finetune_llm_sft(
                pop=[mock_agent],
                env=MagicMock(),
                evaluation_interval=2,

            )

    def test_finetune_llm_sft_evo_steps_not_set(self):
        with pytest.raises(ValueError, match="evo_steps"):
            finetune_llm_sft(
                pop=[MagicMock(spec=SFT)],
                env=MagicMock(),
                evo_steps=None,

                selection_strategy=MagicMock(),
                mutation=MagicMock(),
            )


# ---------------------------------------------------------------------------
# TestFinetuneLlmMultiturn
# ---------------------------------------------------------------------------


class TestFinetuneLlmMultiturn:
    @pytest.mark.parametrize("agent_spec", [LLMPPO, LLMREINFORCE, GRPO])
    def test_finetune_llm_multiturn_basic_training_loop(
        self, agent_spec
    ):
        mock_agent = _make_multiturn_mock_agent(spec=agent_spec)
        batch_steps = 3
        max_steps = 9

        with (
            patch(
                "agilerl.training.llm.multiturn.default_progress_bar"
            ) as mock_pbar_fn,
            patch("agilerl.training.llm.multiturn.init_loggers", return_value=[]),
            patch("agilerl.training.llm.multiturn.aggregate_metrics_across_gpus") as mock_agg,
            patch("agilerl.training.llm.multiturn.save_llm_checkpoint") as mock_save,
            patch("agilerl.training.llm.multiturn.SyncMultiTurnVecEnv"),
            patch(
                "agilerl.training.llm.multiturn.collect_rollouts_llm"
            ) as mock_collect,
            patch(
                "agilerl.training.llm.multiturn.stack_and_pad_experiences"
            ) as mock_stack,
        ):
            mock_pbar_fn.return_value = MagicMock()
            mock_agg.return_value = 0.5
            mock_collect.return_value = _multiturn_collect_return(
                batch_steps=batch_steps
            )
            mock_stack.return_value = (torch.zeros(1, 8, dtype=torch.long),)

            finetune_llm_multiturn(
                pop=[mock_agent],
                env_factory=MagicMock(),
                max_turns=2,
                init_hp={"BATCH_SIZE": 1, "ALGO": mock_agent.algo},
                max_steps=max_steps,
                evaluation_interval=100,
                verbose=False,

            )

        num_outer = max_steps // batch_steps
        assert mock_collect.call_count == num_outer
        assert mock_agent.learn.call_count == num_outer
        assert mock_agent.test.call_count == 0
        mock_agent.learn.assert_called_with(ANY, turn_ids=ANY)
        assert mock_save.call_count == 0

    def test_finetune_llm_multiturn_labels_run_from_population(self):
        """A manifest-style init_hp (no flat ALGO key) still names the run after
        the actual algorithm and env, not the LLMPPO/multiturn defaults.
        """
        mock_agent = _make_multiturn_mock_agent(spec=GRPO)

        with (
            patch("agilerl.training.llm.multiturn.default_progress_bar"),
            patch(
                "agilerl.training.llm.multiturn.init_loggers", return_value=[]
            ) as mock_init_loggers,
            patch("agilerl.training.llm.multiturn.aggregate_metrics_across_gpus"),
            patch("agilerl.training.llm.multiturn.save_llm_checkpoint"),
            patch("agilerl.training.llm.multiturn.SyncMultiTurnVecEnv"),
            patch(
                "agilerl.training.llm.multiturn.collect_rollouts_llm"
            ) as mock_collect,
            patch(
                "agilerl.training.llm.multiturn.stack_and_pad_experiences"
            ) as mock_stack,
        ):
            mock_collect.return_value = _multiturn_collect_return(batch_steps=3)
            mock_stack.return_value = (torch.zeros(1, 8, dtype=torch.long),)

            finetune_llm_multiturn(
                pop=[mock_agent],
                env_factory=MagicMock(),
                max_turns=2,
                init_hp={"env_name": "game:Sudoku-v0-hard"},
                max_steps=3,
                evaluation_interval=100,
                verbose=False,

            )

        assert mock_init_loggers.call_args.kwargs["algo"] == "GRPO"
        assert mock_init_loggers.call_args.kwargs["env_name"] == "game:Sudoku-v0-hard"

    def test_finetune_llm_multiturn_forwards_sampling_logps_to_learn(self):
        """When the rollout captures sampling logps, they're forwarded to
        ``learn(..., sampling_logps=...)`` for GRPO/PPO/REINFORCE agents.
        """
        mock_agent = _make_multiturn_mock_agent(spec=GRPO)
        sampling_logps = [torch.zeros(1, 8)]
        rollout_return = (
            [torch.ones(1, 8, dtype=torch.long)],  # completion_ids_list
            [torch.ones(1, 8, dtype=torch.bool)],  # action_masks_list
            [torch.zeros(1, 8, dtype=torch.long)],  # all_turn_ids
            [torch.ones(2, dtype=torch.float32)],  # all_rewards
            3,  # batch_steps
            123,  # group_seed
            sampling_logps,  # all_sampling_logps (non-None)
        )
        with (
            patch(
                "agilerl.training.llm.multiturn.default_progress_bar",
                return_value=MagicMock(),
            ),
            patch("agilerl.training.llm.multiturn.init_loggers", return_value=[]),
            patch(
                "agilerl.training.llm.multiturn.aggregate_metrics_across_gpus",
                return_value=0.5,
            ),
            patch("agilerl.training.llm.multiturn.save_llm_checkpoint"),
            patch("agilerl.training.llm.multiturn.SyncMultiTurnVecEnv"),
            patch(
                "agilerl.training.llm.multiturn.collect_rollouts_llm",
                return_value=rollout_return,
            ),
            patch(
                "agilerl.training.llm.multiturn.stack_and_pad_experiences"
            ) as mock_stack,
        ):
            mock_stack.return_value = (torch.zeros(1, 8, dtype=torch.long),)
            finetune_llm_multiturn(
                pop=[mock_agent],
                env_factory=MagicMock(),
                max_turns=2,
                init_hp={"BATCH_SIZE": 1, "ALGO": mock_agent.algo},
                max_steps=3,  # one outer iteration (batch_steps=3)
                evaluation_interval=100,
                verbose=False,

            )

        _, learn_kwargs = mock_agent.learn.call_args
        assert learn_kwargs.get("sampling_logps") is sampling_logps

    def test_finetune_llm_multiturn_allows_batch_size_indivisible_by_group_size(self):
        """The batch>group case is unconstrained too: batch_size=3, group_size=2
        (three prompts, two completions each) must pass startup validation rather
        than being rejected. Patch the rollout to a sentinel and assert the call
        reaches it — i.e. it gets past the (now-removed) divisibility guard.
        """
        mock_agent = _make_multiturn_mock_agent(spec=GRPO)
        mock_agent.group_size = 2
        mock_agent.batch_size = 16
        mock_agent.batch_size_per_process = 16
        with (
            patch(
                "agilerl.training.llm.multiturn.default_progress_bar",
                return_value=MagicMock(),
            ),
            patch("agilerl.training.llm.multiturn.init_loggers", return_value=[]),
            patch("agilerl.training.llm.multiturn.SyncMultiTurnVecEnv"),
            patch(
                "agilerl.training.llm.multiturn.collect_rollouts_llm",
                side_effect=RuntimeError("reached rollout"),
            ),
            pytest.raises(RuntimeError, match="reached rollout"),
        ):
            finetune_llm_multiturn(
                pop=[mock_agent],
                max_turns=1,
                env_factory=MagicMock(),
                init_hp={"BATCH_SIZE": 3, "ALGO": "GRPO"},
                max_steps=100,

                verbose=False,
            )

    def test_finetune_llm_multiturn_with_wandb_and_checkpoints(self):
        mock_agent = _make_multiturn_mock_agent()

        with (
            patch(
                "agilerl.training.llm.multiturn.default_progress_bar"
            ) as mock_pbar_fn,
            patch("agilerl.training.llm.multiturn.init_loggers") as mock_init_loggers,
            patch("agilerl.training.llm.multiturn.aggregate_metrics_across_gpus") as mock_agg,
            patch("agilerl.training.llm.multiturn.save_llm_checkpoint") as mock_save,
            patch("agilerl.training.llm.multiturn.SyncMultiTurnVecEnv"),
            patch(
                "agilerl.training.llm.multiturn.collect_rollouts_llm"
            ) as mock_collect,
            patch(
                "agilerl.training.llm.multiturn.stack_and_pad_experiences"
            ) as mock_stack,
        ):
            mock_pbar_fn.return_value = MagicMock()
            mock_init_loggers.return_value = []
            mock_agg.return_value = 0.5
            mock_collect.return_value = _multiturn_collect_return(batch_steps=3)
            mock_stack.return_value = (torch.zeros(1, 8, dtype=torch.long),)

            finetune_llm_multiturn(
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

            )

        mock_init_loggers.assert_called_once()
        assert mock_init_loggers.call_args.kwargs["wb"] is True
        assert mock_save.call_count >= 1

    def test_finetune_llm_multiturn_evolvable_training_loop(self):
        mock_agent = _make_multiturn_mock_agent()
        mutation = MagicMock()
        mutation.architecture_mut = 0
        mutation.new_layer_prob = 0
        mutation.parameters_mut = 0
        mutation.activation_mut = 0

        with (
            patch(
                "agilerl.training.llm.multiturn.default_progress_bar"
            ) as mock_pbar_fn,
            patch("agilerl.training.llm.multiturn.init_loggers", return_value=[]),
            patch("agilerl.training.llm.multiturn.aggregate_metrics_across_gpus") as mock_agg,
            patch("agilerl.training.llm.multiturn.save_llm_checkpoint") as mock_save,
            patch("agilerl.training.llm.multiturn.SyncMultiTurnVecEnv"),
            patch(
                "agilerl.training.llm.multiturn.collect_rollouts_llm"
            ) as mock_collect,
            patch(
                "agilerl.training.llm.multiturn.stack_and_pad_experiences"
            ) as mock_stack,
            patch(
                "agilerl.training.llm.multiturn.run_selection_and_mutation"
            ) as mock_tourn,
        ):
            mock_pbar_fn.return_value = MagicMock()
            mock_agg.return_value = 0.5
            mock_collect.return_value = _multiturn_collect_return(batch_steps=3)
            mock_stack.return_value = (torch.zeros(1, 8, dtype=torch.long),)
            mock_tourn.return_value = [mock_agent]

            finetune_llm_multiturn(
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

            )

        assert mock_tourn.call_count == 3
        assert mock_save.call_count == 0

    def test_finetune_llm_multiturn_value_error_when_evo_steps_missing_with_selection_strategy(
        self,
    ):
        mutation = MagicMock()
        mutation.architecture_mut = 0
        mutation.new_layer_prob = 0
        mutation.parameters_mut = 0
        mutation.activation_mut = 0
        mock_agent = _make_multiturn_mock_agent()
        with pytest.raises(ValueError, match="evo_steps"):
            finetune_llm_multiturn(
                pop=[mock_agent],
                env_factory=MagicMock(),
                max_turns=1,
                init_hp={"BATCH_SIZE": 1, "ALGO": "LLMPPO"},
                max_steps=1,
                evo_steps=None,
                selection_strategy=MagicMock(),
                mutation=mutation,

            )

    def test_finetune_llm_multiturn_warns_when_evo_steps_without_selection_strategy(
        self,
    ):
        mock_agent = _make_multiturn_mock_agent()
        with pytest.warns(UserWarning, match="evo_steps"):
            finetune_llm_multiturn(
                pop=[mock_agent],
                env_factory=MagicMock(),
                max_turns=1,
                init_hp={"BATCH_SIZE": 1, "ALGO": "LLMPPO"},
                max_steps=0,
                evo_steps=3,
                selection_strategy=None,
                mutation=None,

                verbose=False,
            )

    def test_finetune_llm_multiturn_value_error_if_algo_not_supported(self):
        mock_agent = MagicMock(spec=DPO)
        mock_agent.algo = "DPO"
        mock_agent.batch_size = 16
        mock_agent.batch_size_per_process = 16
        with pytest.raises(ValueError, match=r"LLMPPO.*LLMREINFORCE.*GRPO"):
            finetune_llm_multiturn(
                pop=[mock_agent],
                env_factory=MagicMock(),
                max_turns=1,
                init_hp={"BATCH_SIZE": 1, "ALGO": "DPO"},
                max_steps=0,

                verbose=False,
            )

    def test_finetune_llm_multiturn_max_reward_adds_accuracy_metric(self):
        mock_agent = _make_multiturn_mock_agent()

        with (
            patch(
                "agilerl.training.llm.multiturn.default_progress_bar"
            ) as mock_pbar_fn,
            patch("agilerl.training.llm.multiturn.init_loggers", return_value=[]),
            patch("agilerl.training.llm.multiturn.aggregate_metrics_across_gpus") as mock_agg,
            patch("agilerl.training.llm.multiturn.save_llm_checkpoint"),
            patch("agilerl.training.llm.multiturn.SyncMultiTurnVecEnv"),
            patch(
                "agilerl.training.llm.multiturn.collect_rollouts_llm"
            ) as mock_collect,
            patch(
                "agilerl.training.llm.multiturn.stack_and_pad_experiences"
            ) as mock_stack,
        ):
            mock_pbar_fn.return_value = MagicMock()
            mock_agg.return_value = 0.5
            mock_collect.return_value = _multiturn_collect_return(batch_steps=3)
            mock_stack.return_value = (torch.zeros(1, 8, dtype=torch.long),)
            finetune_llm_multiturn(
                pop=[mock_agent],
                env_factory=MagicMock(),
                max_turns=2,
                init_hp={"BATCH_SIZE": 1, "ALGO": "LLMPPO"},
                max_steps=9,
                evaluation_interval=100,
                max_reward=1.0,
                verbose=False,

            )

        num_outer = 3
        # agg called for: mean_score (1) + accuracy (1) per outer iteration
        assert mock_agg.call_count == num_outer * 2

    def test_finetune_llm_multiturn_registers_accuracy_metric(self):
        mock_agent = _make_multiturn_mock_agent()
        mock_agent.metrics.additional_metrics = ["loss", "mean_reward"]

        with (
            patch(
                "agilerl.training.llm.multiturn.default_progress_bar"
            ) as mock_pbar_fn,
            patch("agilerl.training.llm.multiturn.init_loggers", return_value=[]),
            patch("agilerl.training.llm.multiturn.aggregate_metrics_across_gpus") as mock_agg,
            patch("agilerl.training.llm.multiturn.save_llm_checkpoint"),
            patch("agilerl.training.llm.multiturn.SyncMultiTurnVecEnv"),
            patch(
                "agilerl.training.llm.multiturn.collect_rollouts_llm"
            ) as mock_collect,
            patch(
                "agilerl.training.llm.multiturn.stack_and_pad_experiences"
            ) as mock_stack,
        ):
            mock_pbar_fn.return_value = MagicMock()
            mock_agg.return_value = 0.5
            mock_collect.return_value = _multiturn_collect_return(batch_steps=3)
            mock_stack.return_value = (torch.zeros(1, 8, dtype=torch.long),)
            finetune_llm_multiturn(
                pop=[mock_agent],
                env_factory=MagicMock(),
                max_turns=2,
                init_hp={"BATCH_SIZE": 1, "ALGO": "LLMPPO"},
                max_steps=9,
                evaluation_interval=100,
                max_reward=1.0,
                verbose=False,

            )

        mock_agent.metrics.register.assert_called_with("accuracy")

    def test_finetune_llm_multiturn_stops_at_wall_clock_limit(self, capsys):
        mock_agent = _make_multiturn_mock_agent()

        with (
            patch(
                "agilerl.training.llm.multiturn.default_progress_bar"
            ) as mock_pbar_fn,
            patch("agilerl.training.llm.multiturn.init_loggers", return_value=[]),
            patch("agilerl.training.llm.multiturn.aggregate_metrics_across_gpus") as mock_agg,
            patch("agilerl.training.llm.multiturn.save_llm_checkpoint"),
            patch("agilerl.training.llm.multiturn.SyncMultiTurnVecEnv"),
            patch(
                "agilerl.training.llm.multiturn.collect_rollouts_llm"
            ) as mock_collect,
            patch(
                "agilerl.training.llm.multiturn.stack_and_pad_experiences"
            ) as mock_stack,
            patch(
                "agilerl.training.llm.multiturn.time.monotonic",
                side_effect=itertools.count(100, 100).__next__,
            ),
        ):
            mock_pbar_fn.return_value = MagicMock()
            mock_agg.return_value = 0.5
            mock_collect.return_value = _multiturn_collect_return(batch_steps=3)
            mock_stack.return_value = (torch.zeros(1, 8, dtype=torch.long),)
            finetune_llm_multiturn(
                pop=[mock_agent],
                env_factory=MagicMock(),
                max_turns=2,
                init_hp={"BATCH_SIZE": 1, "ALGO": "LLMPPO"},
                max_steps=100,
                max_wall_seconds=50,
                evaluation_interval=100,
                verbose=False,

            )

        assert "wall time limit (50s) reached" in capsys.readouterr().out
        mock_collect.assert_not_called()

    def test_finetune_llm_multiturn_eval_interval_calls_test(self):
        mock_agent = _make_multiturn_mock_agent()
        batch_steps = 3
        max_steps = 9

        with (
            patch(
                "agilerl.training.llm.multiturn.default_progress_bar"
            ) as mock_pbar_fn,
            patch("agilerl.training.llm.multiturn.init_loggers", return_value=[]),
            patch("agilerl.training.llm.multiturn.aggregate_metrics_across_gpus") as mock_agg,
            patch("agilerl.training.llm.multiturn.save_llm_checkpoint"),
            patch("agilerl.training.llm.multiturn.SyncMultiTurnVecEnv"),
            patch(
                "agilerl.training.llm.multiturn.collect_rollouts_llm"
            ) as mock_collect,
            patch(
                "agilerl.training.llm.multiturn.stack_and_pad_experiences"
            ) as mock_stack,
        ):
            mock_pbar_fn.return_value = MagicMock()
            mock_agg.return_value = 0.5
            mock_collect.return_value = _multiturn_collect_return(
                batch_steps=batch_steps
            )
            mock_stack.return_value = (torch.zeros(1, 8, dtype=torch.long),)
            finetune_llm_multiturn(
                pop=[mock_agent],
                env_factory=MagicMock(),
                max_turns=2,
                init_hp={"BATCH_SIZE": 1, "ALGO": "LLMPPO"},
                max_steps=max_steps,
                evaluation_interval=1,
                verbose=False,

            )

        num_outer = max_steps // batch_steps
        assert mock_agent.test.call_count == num_outer

    def test_finetune_llm_multiturn_saves_elite_at_end(self):
        weaker = _make_multiturn_mock_agent()
        weaker.fitness = [0.1]
        stronger = _make_multiturn_mock_agent()
        stronger.fitness = [0.9]

        with (
            patch(
                "agilerl.training.llm.multiturn.default_progress_bar"
            ) as mock_pbar_fn,
            patch("agilerl.training.llm.multiturn.init_loggers", return_value=[]),
            patch("agilerl.training.llm.multiturn.aggregate_metrics_across_gpus") as mock_agg,
            patch("agilerl.training.llm.multiturn.save_llm_checkpoint") as mock_save,
            patch("agilerl.training.llm.multiturn.SyncMultiTurnVecEnv"),
            patch(
                "agilerl.training.llm.multiturn.collect_rollouts_llm"
            ) as mock_collect,
            patch(
                "agilerl.training.llm.multiturn.stack_and_pad_experiences"
            ) as mock_stack,
            _population_init_skip_per_mock_class(),
        ):
            mock_pbar_fn.return_value = MagicMock()
            mock_agg.return_value = 0.5
            mock_collect.return_value = _multiturn_collect_return(batch_steps=3)
            mock_stack.return_value = (torch.zeros(1, 8, dtype=torch.long),)
            finetune_llm_multiturn(
                pop=[weaker, stronger],
                env_factory=MagicMock(),
                max_turns=2,
                init_hp={"BATCH_SIZE": 1, "ALGO": "LLMPPO"},
                max_steps=9,
                evaluation_interval=100,
                save_elite=True,
                elite_path="/tmp/multiturn-elite",
                verbose=False,

            )

        assert mock_save.call_args_list[-1] == call(stronger, "/tmp/multiturn-elite")

    def test_finetune_llm_multiturn_init_hp_none_uses_agent_fields(self):
        mock_agent = _make_multiturn_mock_agent()
        mock_agent.batch_size_per_process = 7
        mock_agent.algo = "LLMPPO"

        with (
            patch(
                "agilerl.training.llm.multiturn.default_progress_bar"
            ) as mock_pbar_fn,
            patch("agilerl.training.llm.multiturn.init_loggers") as mock_init_loggers,
            patch("agilerl.training.llm.multiturn.aggregate_metrics_across_gpus") as mock_agg,
            patch("agilerl.training.llm.multiturn.save_llm_checkpoint"),
            patch("agilerl.training.llm.multiturn.SyncMultiTurnVecEnv"),
            patch(
                "agilerl.training.llm.multiturn.collect_rollouts_llm"
            ) as mock_collect,
            patch(
                "agilerl.training.llm.multiturn.stack_and_pad_experiences"
            ) as mock_stack,
        ):
            mock_pbar_fn.return_value = MagicMock()
            mock_init_loggers.return_value = []
            mock_agg.return_value = 0.5
            mock_collect.return_value = _multiturn_collect_return(batch_steps=3)
            mock_stack.return_value = (torch.zeros(1, 8, dtype=torch.long),)
            finetune_llm_multiturn(
                pop=[mock_agent],
                env_factory=MagicMock(),
                max_turns=2,
                init_hp=None,
                max_steps=0,
                wb=True,
                wandb_api_key="fake",
                verbose=False,

            )

        init_hp_passed = mock_init_loggers.call_args.kwargs["init_hyperparams"]
        assert init_hp_passed["BATCH_SIZE_PER_GPU"] == 7
        assert init_hp_passed["ALGO"] == "LLMPPO"

    def test_finetune_llm_multiturn_allows_group_size_indivisible_by_batch_size(
        self,
    ):
        """batch_size and group_size need not divide each other for GRPO.

        The rollout vec env keeps each prompt's group whole (group-contiguous),
        so e.g. batch_size=2, group_size=3 (two prompts, three completions each)
        is valid and must pass startup validation rather than being rejected.
        We patch the rollout to raise a sentinel and assert the call reaches it
        — i.e. it gets past the (now-removed) divisibility guard.
        """
        agent = _make_multiturn_mock_agent(spec=GRPO)
        agent.group_size = 3
        agent.batch_size = 2
        agent.batch_size_per_process = 2

        with (
            patch(
                "agilerl.training.llm.multiturn.default_progress_bar",
                return_value=MagicMock(),
            ),
            patch("agilerl.training.llm.multiturn.init_loggers", return_value=[]),
            patch("agilerl.training.llm.multiturn.SyncMultiTurnVecEnv"),
            patch(
                "agilerl.training.llm.multiturn.collect_rollouts_llm",
                side_effect=RuntimeError("reached rollout"),
            ),
            pytest.raises(RuntimeError, match="reached rollout"),
        ):
            finetune_llm_multiturn(
                pop=[agent],
                max_turns=2,
                env_factory=MagicMock(),
                init_hp={"BATCH_SIZE": 2, "BATCH_SIZE_PER_GPU": 2, "ALGO": "GRPO"},
                max_steps=8,

                wb=False,
                verbose=False,
            )


# ---------------------------------------------------------------------------
# Distributed: report_metrics must run on every rank
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "loop",
    ["reasoning", "preference", "sft", "multiturn"],
)
def test_report_metrics_called_on_non_main_process(loop):
    """WandbLogger / Logger.on_main_process issue wait_for_everyone barriers.
    report_metrics must therefore run on every rank — calling it only on the
    main process desyncs NCCL (hang after the first metrics table).
    """
    acc = MagicMock()
    acc.is_main_process = False
    acc.num_processes = 2

    with patch.object(Population, "report_metrics", autospec=True) as mock_report:
        if loop == "reasoning":
            mock_agent = _mock_grpo_agent()
            mock_env = MagicMock()
            mock_env.__len__.return_value = 2
            mock_env.reset.return_value = "initial_prompts"
            mock_env.step.return_value = ("next_prompts", torch.tensor([2.0, 3.0]))
            mock_env.data_batch_size_per_gpu = 1
            with (
                patch(
                    "agilerl.training.llm.reasoning.default_progress_bar",
                    return_value=MagicMock(),
                ),
                patch("agilerl.training.llm.reasoning.init_loggers", return_value=[]),
                patch(
                    "agilerl.training.llm.reasoning.aggregate_metrics_across_gpus",
                    return_value=0.5,
                ),
                patch("agilerl.training.llm.reasoning.save_llm_checkpoint"),
            ):
                finetune_llm_reasoning(
                    pop=[mock_agent],
                    env=mock_env,
                    max_steps=2,
                    evaluation_interval=100,
                    verbose=False,

                )
        elif loop == "preference":
            mock_agent = _mock_dpo_agent()
            mock_env = MagicMock()
            mock_env.__len__.return_value = 2
            mock_env.reset.return_value = "batch"
            mock_env.step.return_value = "batch"
            mock_env.data_batch_size_per_gpu = 1
            with (
                patch(
                    "agilerl.training.llm.preference.default_progress_bar",
                    return_value=MagicMock(),
                ),
                patch("agilerl.training.llm.preference.init_loggers", return_value=[]),
                patch("agilerl.training.llm.preference.save_llm_checkpoint"),
            ):
                finetune_llm_preference(
                    pop=[mock_agent],
                    env=mock_env,
                    max_steps=2,
                    evaluation_interval=100,
                    verbose=False,

                )
        elif loop == "sft":
            mock_agent = _mock_sft_agent()
            mock_env = MagicMock()
            mock_env.__len__.return_value = 2
            mock_env.reset.return_value = "batch"
            mock_env.step.return_value = "batch"
            mock_env.data_batch_size_per_gpu = 1
            with (
                patch(
                    "agilerl.training.llm.sft.default_progress_bar",
                    return_value=MagicMock(),
                ),
                patch("agilerl.training.llm.sft.init_loggers", return_value=[]),
                patch("agilerl.training.llm.sft.save_llm_checkpoint"),
            ):
                finetune_llm_sft(
                    pop=[mock_agent],
                    env=mock_env,
                    max_steps=2,
                    evaluation_interval=100,
                    verbose=False,

                )
        else:
            mock_agent = _make_multiturn_mock_agent(spec=GRPO)
            with (
                patch(
                    "agilerl.training.llm.multiturn.default_progress_bar",
                    return_value=MagicMock(),
                ),
                patch("agilerl.training.llm.multiturn.init_loggers", return_value=[]),
                patch(
                    "agilerl.training.llm.multiturn.aggregate_metrics_across_gpus",
                    return_value=0.5,
                ),
                patch("agilerl.training.llm.multiturn.save_llm_checkpoint"),
                patch("agilerl.training.llm.multiturn.SyncMultiTurnVecEnv"),
                patch(
                    "agilerl.training.llm.multiturn.collect_rollouts_llm",
                    return_value=_multiturn_collect_return(batch_steps=2),
                ),
                patch(
                    "agilerl.training.llm.multiturn.stack_and_pad_experiences",
                    return_value=(torch.zeros(1, 8, dtype=torch.long),),
                ),
            ):
                finetune_llm_multiturn(
                    pop=[mock_agent],
                    env_factory=MagicMock(),
                    max_turns=2,
                    init_hp={"BATCH_SIZE": 1, "ALGO": mock_agent.algo},
                    max_steps=2,
                    evaluation_interval=100,
                    verbose=False,

                )

        assert mock_report.call_count >= 1, (
            f"{loop}: report_metrics must run on non-main ranks "
            "(logger collectives require all ranks)"
        )


def test_finetune_llm_reasoning_aligns_completion_shapes_before_learn():
    """When Liger token-IS needs cross-rank T sync, align before learn()."""
    mock_agent = _mock_grpo_agent()
    mock_agent.pad_token_id = 0
    mock_agent.use_liger_loss = True
    mock_agent.importance_sampling_level = "token"
    mock_agent.get_action.return_value = ActionResult(
        completion_ids=[torch.ones(1, 4, dtype=torch.long)],
        action_masks=[torch.ones(1, 3, dtype=torch.bool)],
        sampling_logps=None,
    )

    mock_env = MagicMock()
    mock_env.__len__.return_value = 1
    mock_env.reset.return_value = "prompts"
    mock_env.step.return_value = ("next", torch.tensor([1.0]))
    mock_env.data_batch_size_per_gpu = 1
    mock_env.num_epochs = 0

    acc = MagicMock()
    acc.is_main_process = True
    acc.num_processes = 2

    aligned = (
        torch.ones(1, 6, dtype=torch.long),
        torch.ones(1, 5, dtype=torch.bool),
        torch.tensor([1.0]),
    )
    with (
        patch(
            "agilerl.training.llm.reasoning.default_progress_bar",
            return_value=MagicMock(),
        ),
        patch("agilerl.training.llm.reasoning.init_loggers", return_value=[]),
        patch(
            "agilerl.training.llm.reasoning.aggregate_metrics_across_gpus", return_value=0.5
        ),
        patch("agilerl.training.llm.reasoning.save_llm_checkpoint"),
        patch(
            "agilerl.training.llm.reasoning.needs_cross_rank_seq_padding",
            return_value=True,
        ) as mock_needs,
        patch(
            "agilerl.training.llm.reasoning.align_completion_batch_shapes_across_ranks",
            return_value=aligned,
        ) as mock_align,
        patch.object(Population, "report_metrics", autospec=True),
    ):
        finetune_llm_reasoning(
            pop=[mock_agent],
            env=mock_env,
            max_steps=1,
            evaluation_interval=100,
            verbose=False,

        )

    mock_needs.assert_called()
    mock_align.assert_called()
    learn_batch = mock_agent.learn.call_args.args[0]
    assert learn_batch[0].shape == (1, 6)
    assert learn_batch[1].shape == (1, 5)


def test_finetune_llm_multiturn_aligns_and_pads_turn_ids():
    """Cross-rank T pad must also extend turn_ids to the padded mask length."""
    mock_agent = _make_multiturn_mock_agent(spec=GRPO)
    mock_agent.pad_token_id = 0
    mock_agent.use_liger_loss = True
    mock_agent.importance_sampling_level = "token"

    aligned_ids = torch.ones(1, 10, dtype=torch.long)
    aligned_masks = torch.ones(1, 9, dtype=torch.bool)
    aligned_rewards = torch.ones(1, 2, dtype=torch.float32)
    short_turn_ids = torch.zeros(1, 7, dtype=torch.long)
    rewards_2d = torch.ones(1, 2, dtype=torch.float32)

    acc = MagicMock()
    acc.is_main_process = True
    acc.num_processes = 2

    with (
        patch(
            "agilerl.training.llm.multiturn.default_progress_bar",
            return_value=MagicMock(),
        ),
        patch("agilerl.training.llm.multiturn.init_loggers", return_value=[]),
        patch(
            "agilerl.training.llm.multiturn.aggregate_metrics_across_gpus", return_value=0.5
        ),
        patch("agilerl.training.llm.multiturn.save_llm_checkpoint"),
        patch("agilerl.training.llm.multiturn.SyncMultiTurnVecEnv"),
        patch(
            "agilerl.training.llm.multiturn.collect_rollouts_llm",
            return_value=_multiturn_collect_return(batch_steps=2),
        ),
        patch(
            "agilerl.training.llm.multiturn.stack_and_pad_experiences",
            side_effect=[
                (short_turn_ids,),
                (rewards_2d,),
            ],
        ),
        patch(
            "agilerl.training.llm.multiturn.needs_cross_rank_seq_padding",
            return_value=True,
        ),
        patch(
            "agilerl.training.llm.multiturn.align_completion_batch_shapes_across_ranks",
            return_value=(aligned_ids, aligned_masks, aligned_rewards),
        ),
        patch.object(Population, "report_metrics", autospec=True),
    ):
        finetune_llm_multiturn(
            pop=[mock_agent],
            env_factory=MagicMock(),
            max_turns=2,
            init_hp={"BATCH_SIZE": 1, "ALGO": mock_agent.algo},
            max_steps=2,
            evaluation_interval=100,
            verbose=False,

        )

    assert mock_agent.learn.call_count >= 1
    turn_ids = mock_agent.learn.call_args.kwargs["turn_ids"]
    assert turn_ids.shape == (1, 9)
    assert torch.all(turn_ids[:, 7:] == -1)


# ---------------------------------------------------------------------------
# Module-level: env/env_fn validation tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("finetune_fn", "agent_spec"),
    [
        (finetune_llm_reasoning, GRPO),
        (finetune_llm_preference, DPO),
    ],
)
def test_finetune_llm_env_and_env_fn_mutually_exclusive(finetune_fn, agent_spec):
    agent = MagicMock(spec=agent_spec)
    agent.algo = "GRPO" if agent_spec is GRPO else "DPO"
    agent.batch_size_per_process = 1
    agent.batch_size = 1
    agent.pretrained_model_name_or_path = "Qwen/Qwen2.5-0.5B"
    agent.fitness = [0.0]

    env = MagicMock()
    env.__len__.return_value = 1
    env.name = "mock_env"
    env.data_batch_size_per_gpu = 1
    env.num_epochs = 0
    env.reset.return_value = "prompts"
    env.step.return_value = "prompts"

    with pytest.raises(ValueError, match="Provide exactly one of 'env' or 'env_fn'"):
        finetune_fn(
            pop=[agent],
            env=env,
            env_fn=lambda: env,
            max_steps=0,
            verbose=False,

        )


@pytest.mark.parametrize(
    "finetune_fn",
    [finetune_llm_reasoning, finetune_llm_preference],
)
def test_finetune_llm_requires_env_or_env_fn(finetune_fn):
    with pytest.raises(ValueError, match="Either 'env' or 'env_fn' must be provided"):
        finetune_fn(
            pop=[MagicMock()],
            env=None,
            env_fn=None,
            max_steps=0,
            verbose=False,

        )


@pytest.mark.parametrize(
    ("finetune_fn", "agent_spec"),
    [
        (finetune_llm_reasoning, GRPO),
        (finetune_llm_preference, DPO),
    ],
)
def test_finetune_llm_warns_on_shared_env_with_population(finetune_fn, agent_spec):
    if agent_spec is GRPO:
        agents = [
            _mock_grpo_agent(index=i, batch_size_per_process=1, batch_size=1)
            for i in range(2)
        ]
    else:
        agents = [
            _mock_dpo_agent(index=i, batch_size_per_process=1, batch_size=1)
            for i in range(2)
        ]

    env = MagicMock()
    env.__len__.return_value = 1
    env.name = "mock_env"
    env.data_batch_size_per_gpu = 1
    env.num_epochs = 0
    env.reset.return_value = "prompts"
    env.step.return_value = "prompts"

    with (
        _population_init_skip_per_mock_class(),
        pytest.warns(UserWarning, match="fairness bias"),
    ):
        finetune_fn(
            pop=agents,
            env=env,
            max_steps=0,
            verbose=False,

        )


@pytest.mark.parametrize(
    "finetune_fn", [finetune_llm_reasoning, finetune_llm_preference]
)
def test_finetune_llm_checkpoint_triggering_non_divisible_steps(finetune_fn):
    if finetune_fn is finetune_llm_reasoning:
        agent = _mock_grpo_agent(batch_size_per_process=1, batch_size=1)
        env = MagicMock()
        env.reset.return_value = "prompts"
        env.step.return_value = ("next", torch.tensor([1.0]))
    else:
        agent = _mock_dpo_agent(batch_size_per_process=1, batch_size=1)
        env = MagicMock()
        env.reset.return_value = {"prompt": ["x"]}
        env.step.return_value = {"prompt": ["y"]}

    env.__len__.return_value = 10
    env.name = "mock_env"
    env.data_batch_size_per_gpu = 1
    env.num_epochs = 0

    mod = _finetune_module_path(finetune_fn)
    with (
        patch(f"{mod}.default_progress_bar") as mock_pbar_fn,
        patch(f"{mod}.save_llm_checkpoint") as mock_save,
        patch(f"{mod}.init_loggers", return_value=[]),
        (
            patch(f"{mod}.aggregate_metrics_across_gpus", return_value=0.5)
            if mod == "agilerl.training.llm.reasoning"
            else nullcontext()
        ),
    ):
        mock_pbar_fn.return_value = MagicMock()
        finetune_fn(
            pop=[agent],
            env=env,
            max_steps=5,
            checkpoint_steps=2,
            evaluation_interval=100,
            verbose=False,

        )

    assert mock_save.call_count == 3


@pytest.mark.parametrize(
    ("finetune_fn", "agent_spec"),
    [
        (finetune_llm_reasoning, GRPO),
        (finetune_llm_preference, DPO),
        (finetune_llm_sft, SFT),
    ],
)
def test_inner_loop_breaks_after_max_steps_first_agent(finetune_fn, agent_spec):
    if agent_spec is GRPO:
        agent0 = _mock_grpo_agent(index=0, batch_size_per_process=1, batch_size=1)
        agent1 = _mock_grpo_agent(index=1, batch_size_per_process=1, batch_size=1)
        env = MagicMock()
        env.__len__.return_value = 2
        env.reset.return_value = "initial_prompts"
        env.step.return_value = ("next_prompts", torch.tensor([1.0]))
    elif agent_spec is DPO:
        agent0 = _mock_dpo_agent(index=0, batch_size_per_process=1, batch_size=1)
        agent1 = _mock_dpo_agent(index=1, batch_size_per_process=1, batch_size=1)
        example = {
            "prompt": ["p"],
            "prompt_lengths": [1],
            "chosen": ["c"],
            "rejected": ["r"],
            "chosen_input_ids": [1],
            "chosen_attention_mask": [1],
            "rejected_input_ids": [1],
            "rejected_attention_mask": [1],
        }
        env = MagicMock()
        env.__len__.return_value = 2
        env.reset.return_value = example
        env.step.return_value = example
    else:
        agent0 = _mock_sft_agent(index=0, batch_size_per_process=1, batch_size=1)
        agent1 = _mock_sft_agent(index=1, batch_size_per_process=1, batch_size=1)
        env = MagicMock()
        env.__len__.return_value = 2
        env.reset.return_value = "initial_prompts"
        env.step.return_value = "next_prompts"

    env.data_batch_size_per_gpu = 1

    mod = _finetune_module_path(finetune_fn)
    with (
        _population_init_skip_per_mock_class(),
        patch(f"{mod}.default_progress_bar") as mock_pbar_fn,
        patch(f"{mod}.save_llm_checkpoint"),
        patch(f"{mod}.init_loggers", return_value=[]),
        (
            patch(f"{mod}.aggregate_metrics_across_gpus", return_value=0.5)
            if mod == "agilerl.training.llm.reasoning"
            else nullcontext()
        ),
    ):
        mock_pbar_fn.return_value = MagicMock()
        finetune_fn(
            pop=[agent0, agent1],
            env=env,

            max_steps=1,
            evaluation_interval=100,
            verbose=False,
        )
    assert agent0.learn.call_count == 1


def test_collect_rollouts_llm_breaks_when_vector_env_has_no_active_prompts():
    mock_agent = _make_multiturn_mock_agent()

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
        group_size=1,
    )

    assert mock_agent.get_action.call_count == 1
    assert mock_env.step.call_count == 1


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
