from unittest.mock import MagicMock, Mock, call, patch

import pytest
import torch
from accelerate import Accelerator

from agilerl.algorithms import DPO, GRPO
from agilerl.training.train_llm import (
    finetune_llm_preference,
    finetune_llm_reasoning,
)

pytestmark = pytest.mark.llm


def _mock_grpo_agent(**overrides):
    """Build a mock GRPO agent with proper metrics interface."""
    agent = MagicMock(spec=GRPO)
    agent.algo = "GRPO"
    agent.fitness = [0.0]
    agent.local_rank = "0"
    agent.get_action.return_value = (
        [torch.ones(1, 100) for _ in range(2)],
        Mock(),
    )
    agent.learn.return_value = (0.5, 0.2)
    agent.test.return_value = torch.tensor([0.8])
    agent.batch_size_per_process = 32
    agent.batch_size = 32
    agent.pretrained_model_name_or_path = "Qwen/Qwen2.5-0.5B"
    agent.lr = 0.01
    agent.index = 0
    agent.mut = None

    # Proper metrics mock for Population
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

    # hp_config for Population._collect_hyperparameters
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
    agent.learn.return_value = (0.5, 0.2, 0.1)
    agent.test.return_value = 0.87
    agent.batch_size_per_process = 32
    agent.batch_size = 32
    agent.pretrained_model_name_or_path = "Qwen/Qwen2.5-0.5B"
    agent.lr = 0.001
    agent.index = 0
    agent.mut = None

    # Proper metrics mock for Population
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

    # hp_config for Population._collect_hyperparameters
    agent.registry = MagicMock()
    agent.registry.hp_config = MagicMock()
    agent.registry.hp_config.config = {"lr": 0.001, "batch_size": 32}
    agent.registry.hp_config.names.return_value = ["lr", "batch_size"]

    for key, val in overrides.items():
        setattr(agent, key, val)
    return agent


@pytest.mark.parametrize("use_accelerator", [True, False])
def test_finetune_llm_reasoning_basic_training_loop(use_accelerator):
    """Test the basic training loop in finetune_llm_reasoning."""
    mock_agent = _mock_grpo_agent()

    mock_env = MagicMock()
    mock_env.__len__.return_value = 6
    mock_env.reset.return_value = "initial_prompts"
    mock_env.step.return_value = ("next_prompts", torch.tensor([2.0, 3.0]))
    mock_env.data_batch_size_per_gpu = 1

    with (
        patch("agilerl.training.train_llm.default_progress_bar") as mock_pbar_fn,
        patch("agilerl.training.train_llm.aggregate_metrics_across_gpus") as mock_agg,
        patch("agilerl.training.train_llm.save_llm_checkpoint"),
        patch("agilerl.training.train_llm.init_loggers", return_value=[]),
    ):
        mock_pbar_fn.return_value = MagicMock()
        mock_agg.return_value = 0.5
        finetune_llm_reasoning(
            pop=[mock_agent],
            env=mock_env,
            evaluation_interval=2,
            max_reward=2.0,
            accelerator=None if use_accelerator else Accelerator(),
        )
        assert mock_env.reset.call_count == 1
        assert mock_env.reset.call_args == call(reset_dataloaders=True)
        assert mock_agent.get_action.call_count == 6
        assert mock_env.step.call_count == 6
        assert mock_agent.learn.call_count == 6
        assert mock_agent.test.call_count == 3


@pytest.mark.parametrize("use_accelerator", [True, False])
def test_finetune_llm_reasoning_with_wandb_and_checkpoints(use_accelerator):
    """Test finetune_llm_reasoning with wandb logging and checkpointing enabled."""
    mock_agent = _mock_grpo_agent()

    mock_env = MagicMock()
    mock_env.__len__.return_value = 6
    mock_env.reset.return_value = "initial_prompts"
    mock_env.step.return_value = ("next_prompts", torch.tensor([2.0, 3.0]))
    mock_env.data_batch_size_per_gpu = 1

    with (
        patch("agilerl.training.train_llm.default_progress_bar") as mock_pbar_fn,
        patch("agilerl.training.train_llm.init_loggers") as mock_init_loggers,
        patch("agilerl.training.train_llm.aggregate_metrics_across_gpus") as mock_agg,
        patch("agilerl.training.train_llm.save_llm_checkpoint") as mock_save,
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
            accelerator=None if use_accelerator else Accelerator(),
            max_reward=2.0,
            checkpoint_steps=6,
        )

        # Verify loggers were initialized with wb=True
        mock_init_loggers.assert_called_once()
        assert mock_init_loggers.call_args.kwargs["wb"] is True
        # Verify checkpointing
        assert mock_save.call_count == 1
        # Verify evaluation was called at the right intervals
        assert mock_agent.test.call_count == 2


@pytest.mark.parametrize("use_accelerator", [True, False])
def test_finetune_llm_reasoning_evolvable_training_loop(use_accelerator):
    """Test the basic training loop in finetune_llm_reasoning with evolution."""
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

    with (
        patch("agilerl.training.train_llm.default_progress_bar") as mock_pbar_fn,
        patch("agilerl.training.train_llm.aggregate_metrics_across_gpus") as mock_agg,
        patch("agilerl.training.train_llm.save_llm_checkpoint"),
        patch("agilerl.training.train_llm.init_loggers", return_value=[]),
        patch(
            "agilerl.training.train_llm.tournament_selection_and_mutation"
        ) as mock_tournament_selection_and_mutation,
    ):
        mock_pbar_fn.return_value = MagicMock()
        mock_tournament_selection_and_mutation.return_value = [mock_agent]
        mock_agg.return_value = 0.5

        finetune_llm_reasoning(
            pop=[mock_agent],
            env=mock_env,
            evaluation_interval=2,
            max_reward=2.0,
            evo_steps=1,
            accelerator=None if use_accelerator else Accelerator(),
            tournament=Mock(),
            mutation=mutation,
        )
        assert mock_env.reset.call_count == 1
        assert mock_env.reset.call_args == call(reset_dataloaders=True)
        assert mock_agent.get_action.call_count == 6
        assert mock_env.step.call_count == 6
        assert mock_agent.learn.call_count == 6
        assert mock_agent.test.call_count == 3
        assert mock_tournament_selection_and_mutation.call_count == 6


@pytest.mark.parametrize(
    "finetune_fn",
    [finetune_llm_reasoning, finetune_llm_preference],
)
def test_finetune_llm_reasoning_evo_steps_not_set(finetune_fn):
    """Test that finetune_llm raises a ValueError if evo_steps is not set."""
    with pytest.raises(ValueError) as evo_steps_not_set_error:
        finetune_fn(
            pop=[
                MagicMock(
                    spec=(GRPO if finetune_fn == finetune_llm_reasoning else DPO),
                ),
            ],
            env=MagicMock(),
            evo_steps=None,
            accelerator=None,
            tournament=MagicMock(),
            mutation=MagicMock(),
        )
        assert (
            "'evo_steps' is set but at least one of 'tournament' or 'mutation' is set to None. Evolution will not take place."
            in str(evo_steps_not_set_error.value)
        )


@pytest.mark.parametrize(
    "finetune_fn",
    [finetune_llm_reasoning, finetune_llm_preference],
)
def test_finetune_llm_reasoning_value_error_if_evo_steps_not_set(finetune_fn):
    """Test that finetune_llm raises a ValueError if evo_steps is not set."""
    with pytest.raises(ValueError) as evo_steps_not_set_error:
        finetune_llm_reasoning(
            pop=[
                MagicMock(
                    spec=(GRPO if finetune_fn == finetune_llm_reasoning else DPO),
                ),
            ],
            env=MagicMock(),
            evo_steps=None,
            accelerator=None,
            tournament=MagicMock(),
            mutation=MagicMock(),
        )
        assert (
            "'evo_steps' must be set if 'tournament' and 'mutation' are not None."
            in str(evo_steps_not_set_error.value)
        )


def test_finetune_llm_reasoning_warning_num_epochs_and_max_steps():
    """Test that finetune_llm_reasoning raises a warning if num_epochs and max_steps are set."""
    mock_agent = _mock_grpo_agent()

    mock_env = MagicMock()
    mock_env.__len__.return_value = 6
    mock_env.reset.return_value = "initial_prompts"
    mock_env.step.return_value = ("next_prompts", torch.tensor([2.0, 3.0]))
    mock_env.data_batch_size_per_gpu = 1

    with (
        patch("agilerl.training.train_llm.default_progress_bar") as mock_pbar_fn,
        patch("agilerl.training.train_llm.aggregate_metrics_across_gpus") as mock_agg,
        patch("agilerl.training.train_llm.save_llm_checkpoint"),
        patch("agilerl.training.train_llm.init_loggers", return_value=[]),
        patch(
            "agilerl.training.train_llm.tournament_selection_and_mutation"
        ) as mock_tournament_selection_and_mutation,
    ):
        mock_pbar_fn.return_value = MagicMock()
        mock_tournament_selection_and_mutation.return_value = [mock_agent]
        mock_agg.return_value = 0.5

        with pytest.warns(UserWarning) as num_epochs_and_max_steps_warning:
            finetune_llm_reasoning(
                pop=[mock_agent],
                env=mock_env,
                evaluation_interval=2,
                max_reward=2.0,
                num_epochs=10,
                max_steps=100,
                evo_steps=None,
            )
            assert (
                "'num_epochs' is set but 'max_steps' is also set. 'num_epochs' will take precedence over 'max_steps'."
                in str(num_epochs_and_max_steps_warning[0].message)
            )


def test_finetune_llm_reasoning_max_steps_set_from_num_epochs():
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
        patch("agilerl.training.train_llm.default_progress_bar") as mock_pbar_fn,
        patch("agilerl.training.train_llm.init_loggers", return_value=[]),
        patch("agilerl.training.train_llm.aggregate_metrics_across_gpus") as mock_agg,
        patch("agilerl.training.train_llm.save_llm_checkpoint") as mock_save,
    ):
        mock_pbar_fn.return_value = MagicMock()
        mock_agg.return_value = 0.5
        finetune_llm_reasoning(
            pop=[mock_agent],
            env=mock_env,
            evaluation_interval=2,
            max_reward=2.0,
            evo_steps=1,
            accelerator=None,
            num_epochs=2,
            checkpoint_steps=3,
        )
        # Verify 2 checkpoints as 2 epochs
        assert mock_save.call_count == 2


def test_finetune_llm_reasoning_break_on_num_epochs():
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
        patch("agilerl.training.train_llm.default_progress_bar") as mock_pbar_fn,
        patch("agilerl.training.train_llm.init_loggers", return_value=[]),
        patch("agilerl.training.train_llm.aggregate_metrics_across_gpus") as mock_agg,
        patch("agilerl.training.train_llm.save_llm_checkpoint"),
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
            accelerator=None,
            num_epochs=2,
            checkpoint_steps=3,
        )


# Preference loop tests
@pytest.mark.parametrize("use_accelerator", [True, False])
def test_finetune_llm_preference_basic_training_loop(use_accelerator):
    """Test the basic training loop in finetune_llm_preference."""
    mock_agent = _mock_dpo_agent()

    mock_env = MagicMock()
    mock_env.__len__.return_value = 6
    example_prefernce_env_return = {
        "prompt": ["This is a mock prompt"],
        "prompt_lengths": [10],
        "chosen": ["This is a mock chosen prompt"],
        "rejected": ["This is a mock rejected prompt"],
        "chosen_input_ids": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "chosen_attention_mask": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        "rejected_input_ids": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "rejected_attention_mask": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    }
    mock_env.reset.return_value = example_prefernce_env_return
    mock_env.step.return_value = example_prefernce_env_return
    mock_env.data_batch_size_per_gpu = 1

    with (
        patch("agilerl.training.train_llm.default_progress_bar") as mock_pbar_fn,
        patch("agilerl.training.train_llm.aggregate_metrics_across_gpus") as mock_agg,
        patch("agilerl.training.train_llm.save_llm_checkpoint"),
        patch("agilerl.training.train_llm.init_loggers", return_value=[]),
    ):
        mock_pbar_fn.return_value = MagicMock()
        mock_agg.return_value = 0.5
        finetune_llm_preference(
            pop=[mock_agent],
            env=mock_env,
            evaluation_interval=2,
            accelerator=None if use_accelerator else Accelerator(),
        )
        assert mock_env.reset.call_count == 1
        assert mock_env.reset.call_args == call(reset_dataloaders=True)
        assert mock_agent.get_action.call_count == 0
        assert mock_env.step.call_count == 6
        assert mock_agent.learn.call_count == 6
        assert mock_agent.test.call_count == 3


@pytest.mark.parametrize("use_accelerator", [True, False])
def test_finetune_llm_preference_with_wandb_and_checkpoints(use_accelerator):
    """Test finetune_llm_preference with wandb logging and checkpointing enabled."""
    mock_agent = _mock_dpo_agent()

    mock_env = MagicMock()
    mock_env.__len__.return_value = 6
    example_prefernce_env_return = {
        "prompt": ["This is a mock prompt"],
        "prompt_lengths": [10],
        "chosen": ["This is a mock chosen prompt"],
        "rejected": ["This is a mock rejected prompt"],
        "chosen_input_ids": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "chosen_attention_mask": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        "rejected_input_ids": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "rejected_attention_mask": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    }
    mock_env.reset.return_value = example_prefernce_env_return
    mock_env.step.return_value = example_prefernce_env_return
    mock_env.data_batch_size_per_gpu = 1

    with (
        patch("agilerl.training.train_llm.default_progress_bar") as mock_pbar_fn,
        patch("agilerl.training.train_llm.init_loggers") as mock_init_loggers,
        patch("agilerl.training.train_llm.aggregate_metrics_across_gpus") as mock_agg,
        patch("agilerl.training.train_llm.save_llm_checkpoint") as mock_save,
    ):
        mock_pbar_fn.return_value = MagicMock()
        mock_init_loggers.return_value = []
        mock_agg.return_value = 0.5

        finetune_llm_preference(
            pop=[mock_agent],
            env=mock_env,
            save_elite=True,
            wb=True,
            wandb_api_key="fake_key",
            evaluation_interval=3,
            accelerator=None if use_accelerator else Accelerator(),
            checkpoint_steps=6,
        )

        # Verify loggers were initialized with wb=True
        mock_init_loggers.assert_called_once()
        assert mock_init_loggers.call_args.kwargs["wb"] is True
        # Verify checkpointing
        assert mock_save.call_count == 1
        # Verify evaluation was called at the right intervals
        assert mock_agent.test.call_count == 2


@pytest.mark.parametrize("use_accelerator", [True, False])
def test_finetune_llm_preference_evolvable_training_loop(use_accelerator):
    """Test the preference training loop with evolution."""
    mock_agent = _mock_dpo_agent()

    mock_env = MagicMock()
    mock_env.__len__.return_value = 6
    example_prefernce_env_return = {
        "prompt": ["This is a mock prompt"],
        "prompt_lengths": [10],
        "chosen": ["This is a mock chosen prompt"],
        "rejected": ["This is a mock rejected prompt"],
        "chosen_input_ids": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "chosen_attention_mask": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        "rejected_input_ids": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "rejected_attention_mask": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    }
    mock_env.reset.return_value = example_prefernce_env_return
    mock_env.step.return_value = example_prefernce_env_return
    mock_env.data_batch_size_per_gpu = 1

    mutation = MagicMock()
    mutation.architecture_mut = 0
    mutation.new_layer_prob = 0
    mutation.parameters_mut = 0
    mutation.activation_mut = 0

    with (
        patch("agilerl.training.train_llm.default_progress_bar") as mock_pbar_fn,
        patch("agilerl.training.train_llm.aggregate_metrics_across_gpus") as mock_agg,
        patch("agilerl.training.train_llm.save_llm_checkpoint"),
        patch("agilerl.training.train_llm.init_loggers", return_value=[]),
        patch(
            "agilerl.training.train_llm.tournament_selection_and_mutation"
        ) as mock_tournament_selection_and_mutation,
    ):
        mock_pbar_fn.return_value = MagicMock()
        mock_tournament_selection_and_mutation.return_value = [mock_agent]
        mock_agg.return_value = 0.5

        finetune_llm_preference(
            pop=[mock_agent],
            env=mock_env,
            evaluation_interval=2,
            evo_steps=1,
            accelerator=None if use_accelerator else Accelerator(),
            tournament=Mock(),
            mutation=mutation,
        )
        assert mock_env.reset.call_count == 1
        assert mock_env.reset.call_args == call(reset_dataloaders=True)
        assert mock_env.step.call_count == 6
        assert mock_agent.learn.call_count == 6
        assert mock_agent.test.call_count == 3
        assert mock_tournament_selection_and_mutation.call_count == 6


def test_finetune_llm_preference_warning_num_epochs_and_max_steps():
    """Test that finetune_llm_preference raises a warning if num_epochs and max_steps are set."""
    mock_agent = _mock_dpo_agent()

    mock_env = MagicMock()
    mock_env.__len__.return_value = 6
    example_prefernce_env_return = {
        "prompt": ["This is a mock prompt"],
        "prompt_lengths": [10],
        "chosen": ["This is a mock chosen prompt"],
        "rejected": ["This is a mock rejected prompt"],
        "chosen_input_ids": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "chosen_attention_mask": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        "rejected_input_ids": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "rejected_attention_mask": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    }
    mock_env.reset.return_value = example_prefernce_env_return
    mock_env.step.return_value = example_prefernce_env_return
    mock_env.data_batch_size_per_gpu = 1

    with (
        patch("agilerl.training.train_llm.default_progress_bar") as mock_pbar_fn,
        patch("agilerl.training.train_llm.aggregate_metrics_across_gpus") as mock_agg,
        patch("agilerl.training.train_llm.save_llm_checkpoint"),
        patch("agilerl.training.train_llm.init_loggers", return_value=[]),
        patch(
            "agilerl.training.train_llm.tournament_selection_and_mutation"
        ) as mock_tournament_selection_and_mutation,
    ):
        mock_pbar_fn.return_value = MagicMock()
        mock_tournament_selection_and_mutation.return_value = [mock_agent]
        mock_agg.return_value = 0.5

        with pytest.warns(UserWarning) as num_epochs_and_max_steps_warning:
            finetune_llm_preference(
                pop=[mock_agent],
                env=mock_env,
                evaluation_interval=2,
                num_epochs=10,
                max_steps=100,
                evo_steps=None,
            )
            assert (
                "'num_epochs' is set but 'max_steps' is also set. 'num_epochs' will take precedence over 'max_steps'."
                in str(num_epochs_and_max_steps_warning[0].message)
            )


def test_finetune_llm_preference_break_on_num_epochs():
    mock_agent = _mock_dpo_agent()

    mock_env = MagicMock()
    mock_env.__len__.return_value = 6
    example_prefernce_env_return = {
        "prompt": ["This is a mock prompt"],
        "prompt_lengths": [10],
        "chosen": ["This is a mock chosen prompt"],
        "rejected": ["This is a mock rejected prompt"],
        "chosen_input_ids": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "chosen_attention_mask": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        "rejected_input_ids": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "rejected_attention_mask": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    }
    mock_env.reset.return_value = example_prefernce_env_return
    mock_env.step.return_value = example_prefernce_env_return
    mock_env.data_batch_size_per_gpu = 1

    mutation = MagicMock()
    mutation.architecture_mut = 0
    mutation.new_layer_prob = 0
    mutation.parameters_mut = 0
    mutation.activation_mut = 0

    with (
        patch("agilerl.training.train_llm.default_progress_bar") as mock_pbar_fn,
        patch("agilerl.training.train_llm.init_loggers", return_value=[]),
        patch("agilerl.training.train_llm.aggregate_metrics_across_gpus") as mock_agg,
        patch("agilerl.training.train_llm.save_llm_checkpoint"),
    ):
        mock_pbar_fn.return_value = MagicMock()
        mock_env.num_epochs = 2
        mock_agg.return_value = 0.5
        finetune_llm_preference(
            pop=[mock_agent],
            env=mock_env,
            evaluation_interval=2,
            evo_steps=1,
            accelerator=None,
            num_epochs=2,
            checkpoint_steps=3,
        )


def test_finetune_llm_preference_value_error_if_algo_not_dpo():
    mock_agent = MagicMock(spec=GRPO)
    mock_agent.algo = "GRPO"
    mock_agent.pretrained_model_name_or_path = "Qwen/Qwen2.5-0.5B"
    mock_agent.batch_size_per_process = 32
    mock_agent.batch_size = 32
    with pytest.raises(
        ValueError,
        match="The algorithm must be DPO for preference-based reinforcement learning.",
    ):
        finetune_llm_preference(
            pop=[mock_agent],
            env=MagicMock(),
            evaluation_interval=2,
            accelerator=None,
        )


def test_finetune_llm_reasoning_value_error_if_algo_not_grpo():
    mock_agent = MagicMock(spec=DPO)
    mock_agent.algo = "DPO"
    mock_agent.pretrained_model_name_or_path = "Qwen/Qwen2.5-0.5B"
    mock_agent.batch_size_per_process = 32
    mock_agent.batch_size = 32
    with pytest.raises(
        ValueError,
        match="The algorithm must be GRPO for reasoning-based reinforcement learning.",
    ):
        finetune_llm_reasoning(
            pop=[mock_agent],
            env=MagicMock(),
            evaluation_interval=2,
            accelerator=None,
        )
