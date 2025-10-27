from unittest.mock import MagicMock, Mock, call, patch

import pytest
import torch
from accelerate import Accelerator

from agilerl.algorithms import DPO, GRPO
from agilerl.training.train_llm import (
    finetune_llm,
    finetune_llm_preference,
)


@pytest.mark.parametrize("use_accelerator", [True, False])
def test_finetune_llm_basic_training_loop(use_accelerator):
    """Test the basic training loop in finetune_llm."""
    # Create mock agent
    mock_agent = MagicMock()
    mock_agent.fitness = [0.0]
    mock_agent.local_rank = "0"  # Main process
    mock_agent.get_action.return_value = (
        [torch.ones(1, 100) for _ in range(2)],
        Mock(),
    )
    mock_agent.learn.return_value = (0.5, 0.2)
    mock_agent.test.return_value = torch.tensor([0.8])

    # Create mock environment - use MagicMock for special methods
    mock_env = MagicMock()
    mock_env.__len__.return_value = 6
    mock_env.reset.return_value = "initial_prompts"
    mock_env.step.return_value = ("next_prompts", torch.tensor([2.0, 3.0]))
    mock_env.data_batch_size_per_gpu = 1

    # Mock other dependencies
    with patch("agilerl.training.train_llm.trange"), patch(
        "agilerl.training.train_llm.aggregate_metrics_across_gpus"
    ) as mock_agg, patch("agilerl.training.train_llm.save_llm_checkpoint"):

        mock_agg.return_value = 0.5
        finetune_llm(
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
        assert mock_agg.call_count == 36
        assert mock_agent.test.call_count == 3  # Should be called at step 2


@pytest.mark.parametrize(
    "use_accelerator",
    [
        True,
        # False
    ],
)
def test_finetune_llm_with_wandb_and_checkpoints(use_accelerator):
    """Test finetune_llm with wandb logging and checkpointing enabled."""
    # Create mock agent
    mock_agent = MagicMock()
    mock_agent.algo = "GRPO"
    mock_agent.registry = MagicMock()
    mock_agent.registry.hp_config = MagicMock()
    mock_agent.registry.hp_config.config = {"lr": 0.001, "batch_size": 32}
    mock_agent.fitness = [0.0]
    mock_agent.local_rank = "0"  # Main process
    mock_agent.get_action.return_value = (
        [torch.ones(1, 100) for _ in range(2)],
        Mock(),
    )
    mock_agent.learn.return_value = (0.5, 0.2)
    mock_agent.test.return_value = torch.tensor([0.8])

    # Create mock environment - use MagicMock for special methods
    mock_env = MagicMock()
    mock_env.__len__.return_value = 6
    mock_env.reset.return_value = "initial_prompts"
    mock_env.step.return_value = ("next_prompts", torch.tensor([2.0, 3.0]))
    mock_env.data_batch_size_per_gpu = 1

    # Mock dependencies
    with patch("agilerl.training.train_llm.trange") as mock_trange, patch(
        "agilerl.training.train_llm.init_wandb"
    ) as mock_init_wandb, patch(
        "agilerl.training.train_llm.wandb"
    ) as mock_wandb, patch(
        "agilerl.training.train_llm.aggregate_metrics_across_gpus"
    ) as mock_agg, patch(
        "agilerl.training.train_llm.save_llm_checkpoint"
    ) as mock_save:

        # Configure mocks
        mock_pbar = Mock()
        mock_trange.return_value = mock_pbar
        mock_agg.return_value = 0.5

        # Run the function with wandb and checkpointing enabled
        finetune_llm(
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

        # Verify wandb was initialized
        mock_init_wandb.assert_called_once()
        # Verify wandb logging
        assert mock_wandb.log.call_count >= 5
        # Verify checkpointing
        assert mock_save.call_count == 1

        # Verify evaluation was called at the right intervals (steps 3)
        assert mock_agent.test.call_count == 2


@pytest.mark.parametrize("use_accelerator", [True, False])
def test_finetune_llm_evolvable_training_loop(use_accelerator):
    """Test the basic training loop in finetune_llm."""
    # Create mock agent
    mock_agent = MagicMock()
    mock_agent.fitness = [0.0]
    mock_agent.local_rank = "0"  # Main process
    mock_agent.get_action.return_value = (
        [torch.ones(1, 100) for _ in range(2)],
        Mock(),
    )
    mock_agent.learn.return_value = (0.5, 0.2)
    mock_agent.test.return_value = torch.tensor([0.8])

    # Create mock environment - use MagicMock for special methods
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

    # Mock other dependencies
    with patch("agilerl.training.train_llm.trange"), patch(
        "agilerl.training.train_llm.aggregate_metrics_across_gpus"
    ) as mock_agg, patch("agilerl.training.train_llm.save_llm_checkpoint"), patch(
        "agilerl.training.train_llm.tournament_selection_and_mutation"
    ) as mock_tournament_selection_and_mutation:

        mock_tournament_selection_and_mutation.return_value = [mock_agent]

        mock_agg.return_value = 0.5
        finetune_llm(
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
        assert mock_agg.call_count == 36
        assert mock_agent.test.call_count == 3  # Should be called at step 2
        assert (
            mock_tournament_selection_and_mutation.call_count == 6
        )  # Should be called at step 2


@pytest.mark.parametrize("finetune_fn", [finetune_llm, finetune_llm_preference])
def test_finetune_llm_evo_steps_not_set(finetune_fn):
    """Test that finetune_llm raises a ValueError if evo_steps is not set."""
    with pytest.raises(ValueError) as evo_steps_not_set_error:
        finetune_fn(
            pop=[MagicMock()],
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


@pytest.mark.parametrize("finetune_fn", [finetune_llm, finetune_llm_preference])
def test_finetune_llm_value_error_if_evo_steps_not_set(finetune_fn):
    """Test that finetune_llm raises a warning if evo_steps is not set."""

    with pytest.raises(ValueError) as evo_steps_not_set_error:
        finetune_llm(
            pop=[MagicMock()],
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


def test_finetune_llm_warning_num_epochs_and_max_steps():
    """Test that finetune_llm raises a warning if evo_steps is not set."""
    # Create mock agent
    mock_agent = MagicMock()
    mock_agent.fitness = [0.0]
    mock_agent.local_rank = "0"  # Main process
    mock_agent.get_action.return_value = (
        [torch.ones(1, 100) for _ in range(2)],
        Mock(),
    )
    mock_agent.learn.return_value = (0.5, 0.2)
    mock_agent.test.return_value = torch.tensor([0.8])

    # Create mock environment - use MagicMock for special methods
    mock_env = MagicMock()
    mock_env.__len__.return_value = 6
    mock_env.reset.return_value = "initial_prompts"
    mock_env.step.return_value = ("next_prompts", torch.tensor([2.0, 3.0]))
    mock_env.data_batch_size_per_gpu = 1

    # Mock other dependencies
    with patch("agilerl.training.train_llm.trange"), patch(
        "agilerl.training.train_llm.aggregate_metrics_across_gpus"
    ) as mock_agg, patch("agilerl.training.train_llm.save_llm_checkpoint"), patch(
        "agilerl.training.train_llm.tournament_selection_and_mutation"
    ) as mock_tournament_selection_and_mutation:

        mock_tournament_selection_and_mutation.return_value = [mock_agent]

        mock_agg.return_value = 0.5
        with pytest.warns(UserWarning) as num_epochs_and_max_steps_warning:
            finetune_llm(
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


def test_finetune_llm_max_steps_set_from_num_epochs():
    # Create mock agent
    mock_agent = MagicMock()
    mock_agent.fitness = [0.0]
    mock_agent.local_rank = "0"  # Main process
    mock_agent.get_action.return_value = (
        [torch.ones(1, 100) for _ in range(2)],
        Mock(),
    )
    mock_agent.learn.return_value = (0.5, 0.2)
    mock_agent.test.return_value = torch.tensor([0.8])

    # Create mock environment - use MagicMock for special methods
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

    # Mock other dependencies
    with patch("agilerl.training.train_llm.trange"), patch(
        "agilerl.training.train_llm.init_wandb"
    ), patch("agilerl.training.train_llm.wandb"), patch(
        "agilerl.training.train_llm.aggregate_metrics_across_gpus"
    ) as mock_agg, patch(
        "agilerl.training.train_llm.save_llm_checkpoint"
    ) as mock_save:

        mock_agg.return_value = 0.5
        finetune_llm(
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


def test_finetune_llm_break_on_num_epochs():
    # Create mock agent
    # Create mock agent
    mock_agent = MagicMock()
    mock_agent.fitness = [0.0]
    mock_agent.local_rank = "0"  # Main process
    mock_agent.get_action.return_value = (
        [torch.ones(1, 100) for _ in range(2)],
        Mock(),
    )
    mock_agent.learn.return_value = (0.5, 0.2)
    mock_agent.test.return_value = torch.tensor([0.8])

    # Create mock environment - use MagicMock for special methods
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

    # Mock other dependencies
    with patch("agilerl.training.train_llm.trange"), patch(
        "agilerl.training.train_llm.init_wandb"
    ), patch("agilerl.training.train_llm.wandb"), patch(
        "agilerl.training.train_llm.aggregate_metrics_across_gpus"
    ) as mock_agg, patch(
        "agilerl.training.train_llm.save_llm_checkpoint"
    ):

        mock_env.num_epochs = 2
        mock_agg.return_value = 0.5
        finetune_llm(
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
    """Test the basic training loop in finetune_llm."""
    # Create mock agent
    mock_agent = MagicMock()
    mock_agent.fitness = [0.0]
    mock_agent.local_rank = "0"  # Main process
    mock_agent.get_action = MagicMock()
    mock_agent.learn.return_value = (0.5, 0.2, 0.1)
    mock_agent.test.return_value = 0.87

    # Create mock environment - use MagicMock for special methods
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

    # Mock other dependencies
    with patch("agilerl.training.train_llm.trange"), patch(
        "agilerl.training.train_llm.aggregate_metrics_across_gpus"
    ) as mock_agg, patch("agilerl.training.train_llm.save_llm_checkpoint"):

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
        assert mock_agg.call_count == 21
        assert mock_agent.test.call_count == 3  # Should be called at step 2


@pytest.mark.parametrize(
    "use_accelerator",
    [True, False],
)
def test_finetune_llm_preference_with_wandb_and_checkpoints(use_accelerator):
    """Test finetune_llm with wandb logging and checkpointing enabled."""

    # Create mock agent
    mock_agent = MagicMock()
    mock_agent.algo = "DPO"
    mock_agent.registry = MagicMock()
    mock_agent.registry.hp_config = MagicMock()
    mock_agent.registry.hp_config.config = {"lr": 0.001, "batch_size": 32}
    mock_agent.fitness = [0.0]
    mock_agent.learn.return_value = (0.5, 0.2, 0.1)
    mock_agent.test.return_value = 0.87

    # Create mock environment - use MagicMock for special methods
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

    # Mock dependencies
    with patch("agilerl.training.train_llm.trange") as mock_trange, patch(
        "agilerl.training.train_llm.init_wandb"
    ) as mock_init_wandb, patch(
        "agilerl.training.train_llm.wandb"
    ) as mock_wandb, patch(
        "agilerl.training.train_llm.aggregate_metrics_across_gpus"
    ) as mock_agg, patch(
        "agilerl.training.train_llm.save_llm_checkpoint"
    ) as mock_save:

        # Configure mocks
        mock_pbar = Mock()
        mock_trange.return_value = mock_pbar
        mock_agg.return_value = 0.5

        # Run the function with wandb and checkpointing enabled
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

        # Verify wandb was initialized
        mock_init_wandb.assert_called_once()
        # Verify wandb logging
        assert mock_wandb.log.call_count >= 5
        # Verify checkpointing
        assert mock_save.call_count == 1

        # Verify evaluation was called at the right intervals (steps 3)
        assert mock_agent.test.call_count == 2


@pytest.mark.parametrize("use_accelerator", [True, False])
def test_finetune_llm_preference_evolvable_training_loop(use_accelerator):
    """Test the basic training loop in finetune_llm."""
    # Create mock agent
    mock_agent = MagicMock()
    mock_agent.algo = "DPO"
    mock_agent.registry = MagicMock()
    mock_agent.registry.hp_config = MagicMock()
    mock_agent.registry.hp_config.config = {"lr": 0.001, "batch_size": 32}
    mock_agent.fitness = [0.0]
    mock_agent.learn.return_value = (0.5, 0.2, 0.1)
    mock_agent.test.return_value = 0.87

    # Create mock environment - use MagicMock for special methods
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

    # Mock other dependencies
    with patch("agilerl.training.train_llm.trange"), patch(
        "agilerl.training.train_llm.aggregate_metrics_across_gpus"
    ) as mock_agg, patch("agilerl.training.train_llm.save_llm_checkpoint"), patch(
        "agilerl.training.train_llm.tournament_selection_and_mutation"
    ) as mock_tournament_selection_and_mutation:

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
        assert mock_agg.call_count == 21
        assert mock_agent.test.call_count == 3  # Should be called at step 2
        assert (
            mock_tournament_selection_and_mutation.call_count == 6
        )  # Should be called at step 2


def test_finetune_llm_preference_warning_num_epochs_and_max_steps():
    """Test that finetune_llm raises a warning if evo_steps is not set."""
    # Create mock agent
    mock_agent = MagicMock()
    mock_agent.algo = "DPO"
    mock_agent.registry = MagicMock()
    mock_agent.registry.hp_config = MagicMock()
    mock_agent.registry.hp_config.config = {"lr": 0.001, "batch_size": 32}
    mock_agent.fitness = [0.0]
    mock_agent.learn.return_value = (0.5, 0.2, 0.1)
    mock_agent.test.return_value = 0.87

    # Create mock environment - use MagicMock for special methods
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

    # Mock other dependencies
    with patch("agilerl.training.train_llm.trange"), patch(
        "agilerl.training.train_llm.aggregate_metrics_across_gpus"
    ) as mock_agg, patch("agilerl.training.train_llm.save_llm_checkpoint"), patch(
        "agilerl.training.train_llm.tournament_selection_and_mutation"
    ) as mock_tournament_selection_and_mutation:

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
    # Create mock agent
    mock_agent = MagicMock()
    mock_agent.algo = "DPO"
    mock_agent.registry = MagicMock()
    mock_agent.registry.hp_config = MagicMock()
    mock_agent.registry.hp_config.config = {"lr": 0.001, "batch_size": 32}
    mock_agent.fitness = [0.0]
    mock_agent.learn.return_value = (0.5, 0.2, 0.1)
    mock_agent.test.return_value = 0.87

    # Create mock environment - use MagicMock for special methods
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

    # Mock other dependencies
    with patch("agilerl.training.train_llm.trange"), patch(
        "agilerl.training.train_llm.init_wandb"
    ), patch("agilerl.training.train_llm.wandb"), patch(
        "agilerl.training.train_llm.aggregate_metrics_across_gpus"
    ) as mock_agg, patch(
        "agilerl.training.train_llm.save_llm_checkpoint"
    ):

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
    # Create mock agent
    mock_agent = MagicMock(spec=DPO)
    mock_agent.algo = "DPO"

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


def test_finetune_llm_value_error_if_algo_not_grpo():
    # Create mock agent
    mock_agent = MagicMock(spec=GRPO)
    mock_agent.algo = "DPO"

    with pytest.raises(
        ValueError,
        match="The algorithm must be GRPO for preference-based reinforcement learning.",
    ):
        finetune_llm(
            pop=[mock_agent],
            env=MagicMock(),
            evaluation_interval=2,
            accelerator=None,
        )
