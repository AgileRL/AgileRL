from unittest.mock import MagicMock, Mock, call, patch

import pytest
import torch
from accelerate import Accelerator

pytest.importorskip("transformers", reason="LLM dependencies not installed")

from agilerl.algorithms import DPO, GRPO
from agilerl.algorithms.sft import SFT
from agilerl.training.train_llm import (
    finetune_llm_preference,
    finetune_llm_reasoning,
    finetune_llm_sft,
)

pytestmark = pytest.mark.llm


@pytest.mark.parametrize("use_accelerator", [True, False])
def test_finetune_llm_reasoning_basic_training_loop(use_accelerator):
    """Test the basic training loop in finetune_llm_reasoning."""
    # Create mock agent
    mock_agent = MagicMock(spec=GRPO)
    mock_agent.fitness = [0.0]
    mock_agent.local_rank = "0"  # Main process
    mock_agent.get_action.return_value = (
        [torch.ones(1, 100) for _ in range(2)],
        Mock(),
    )
    mock_agent.learn.return_value = (0.5, 0.2)
    mock_agent.test.return_value = torch.tensor([0.8])
    mock_agent.algo = "GRPO"
    mock_agent.batch_size_per_process = 32
    mock_agent.batch_size = 32
    mock_agent.steps = [10]
    mock_agent.scores = [0.0]
    mock_agent.pretrained_model_name_or_path = "Qwen/Qwen2.5-0.5B"

    # Create mock environment - use MagicMock for special methods
    mock_env = MagicMock()
    mock_env.__len__.return_value = 6
    mock_env.reset.return_value = "initial_prompts"
    mock_env.step.return_value = ("next_prompts", torch.tensor([2.0, 3.0]))
    mock_env.data_batch_size_per_gpu = 1

    # Mock other dependencies
    with (
        patch("agilerl.training.train_llm.trange"),
        patch("agilerl.utils.utils.aggregate_metrics_across_gpus") as mock_agg,
        patch("agilerl.training.train_llm.save_llm_checkpoint"),
    ):
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
        # aggregate_metrics_across_gpus is only used when an Accelerator is passed;
        # safe_aggregate_metrics handles the no-accelerator case locally.
        expected_agg_calls = 36 if not use_accelerator else 0
        assert mock_agg.call_count == expected_agg_calls
        assert mock_agent.test.call_count == 3  # Should be called at step 2


@pytest.mark.parametrize(
    "use_accelerator",
    [
        True,
        # False
    ],
)
def test_finetune_llm_reasoning_with_wandb_and_checkpoints(use_accelerator):
    """Test finetune_llm_reasoning with wandb logging and checkpointing enabled."""
    # Create mock agent
    mock_agent = MagicMock(spec=GRPO)
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
    mock_agent.batch_size_per_process = 32
    mock_agent.batch_size = 32
    mock_agent.steps = [10]
    mock_agent.scores = [0.0]
    mock_agent.pretrained_model_name_or_path = "Qwen/Qwen2.5-0.5B"
    mock_agent.lr = 0.01

    # Create mock environment - use MagicMock for special methods
    mock_env = MagicMock()
    mock_env.__len__.return_value = 6
    mock_env.reset.return_value = "initial_prompts"
    mock_env.step.return_value = ("next_prompts", torch.tensor([2.0, 3.0]))
    mock_env.data_batch_size_per_gpu = 1

    # Mock dependencies
    with (
        patch("agilerl.training.train_llm.trange") as mock_trange,
        patch("agilerl.training.train_llm.init_wandb") as mock_init_wandb,
        patch("agilerl.training.train_llm.wandb") as mock_wandb,
        patch("agilerl.utils.utils.aggregate_metrics_across_gpus") as mock_agg,
        patch("agilerl.training.train_llm.save_llm_checkpoint") as mock_save,
    ):
        # Configure mocks
        mock_pbar = Mock()
        mock_trange.return_value = mock_pbar
        mock_agg.return_value = 0.5

        # Run the function with wandb and checkpointing enabled
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

        # Verify wandb was initialized
        mock_init_wandb.assert_called_once()
        # Verify wandb logging
        assert mock_wandb.log.call_count >= 5
        # Verify checkpointing
        assert mock_save.call_count == 1

        # Verify evaluation was called at the right intervals (steps 3)
        assert mock_agent.test.call_count == 2


@pytest.mark.parametrize("use_accelerator", [True, False])
def test_finetune_llm_reasoning_evolvable_training_loop(use_accelerator):
    """Test the basic training loop in finetune_llm_reasoning."""
    # Create mock agent
    mock_agent = MagicMock(spec=GRPO)
    mock_agent.algo = "GRPO"
    mock_agent.fitness = [0.0]
    mock_agent.local_rank = "0"  # Main process
    mock_agent.get_action.return_value = (
        [torch.ones(1, 100) for _ in range(2)],
        Mock(),
    )
    mock_agent.learn.return_value = (0.5, 0.2)
    mock_agent.test.return_value = torch.tensor([0.8])
    mock_agent.batch_size_per_process = 32
    mock_agent.batch_size = 32
    mock_agent.steps = [10]
    mock_agent.scores = [0.0]

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
    with (
        patch("agilerl.training.train_llm.trange"),
        patch("agilerl.utils.utils.aggregate_metrics_across_gpus") as mock_agg,
        patch("agilerl.training.train_llm.save_llm_checkpoint"),
        patch(
            "agilerl.training.train_llm.tournament_selection_and_mutation"
        ) as mock_tournament_selection_and_mutation,
    ):
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
        expected_agg_calls = 36 if not use_accelerator else 0
        assert mock_agg.call_count == expected_agg_calls
        assert mock_agent.test.call_count == 3  # Should be called at step 2
        assert (
            mock_tournament_selection_and_mutation.call_count == 6
        )  # Should be called at step 2


@pytest.mark.parametrize(
    "finetune_fn",
    [finetune_llm_reasoning, finetune_llm_preference],
)
def test_finetune_llm_reasoning_evo_steps_not_set(finetune_fn):
    """Test that finetune_llm_reasoning raises a ValueError if evo_steps is not set."""
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
    """Test that finetune_llm_reasoning raises a warning if evo_steps is not set."""
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
    """Test that finetune_llm_reasoning raises a warning if evo_steps is not set."""
    # Create mock agent
    mock_agent = MagicMock(spec=GRPO)
    mock_agent.algo = "GRPO"
    mock_agent.fitness = [0.0]
    mock_agent.local_rank = "0"  # Main process
    mock_agent.get_action.return_value = (
        [torch.ones(1, 100) for _ in range(2)],
        Mock(),
    )
    mock_agent.learn.return_value = (0.5, 0.2)
    mock_agent.test.return_value = torch.tensor([0.8])
    mock_agent.batch_size_per_process = 32
    mock_agent.batch_size = 32
    mock_agent.steps = [10]
    mock_agent.scores = [0.0]

    # Create mock environment - use MagicMock for special methods
    mock_env = MagicMock()
    mock_env.__len__.return_value = 6
    mock_env.reset.return_value = "initial_prompts"
    mock_env.step.return_value = ("next_prompts", torch.tensor([2.0, 3.0]))
    mock_env.data_batch_size_per_gpu = 1

    # Mock other dependencies
    with (
        patch("agilerl.training.train_llm.trange"),
        patch("agilerl.utils.utils.aggregate_metrics_across_gpus") as mock_agg,
        patch("agilerl.training.train_llm.save_llm_checkpoint"),
        patch(
            "agilerl.training.train_llm.tournament_selection_and_mutation"
        ) as mock_tournament_selection_and_mutation,
    ):
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
    # Create mock agent
    mock_agent = MagicMock(spec=GRPO)
    mock_agent.algo = "GRPO"
    mock_agent.fitness = [0.0]
    mock_agent.local_rank = "0"  # Main process
    mock_agent.get_action.return_value = (
        [torch.ones(1, 100) for _ in range(2)],
        Mock(),
    )
    mock_agent.learn.return_value = (0.5, 0.2)
    mock_agent.test.return_value = torch.tensor([0.8])
    mock_agent.batch_size_per_process = 32
    mock_agent.batch_size = 32
    mock_agent.steps = [10]
    mock_agent.scores = [0.0]

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
    with (
        patch("agilerl.training.train_llm.trange"),
        patch("agilerl.training.train_llm.init_wandb"),
        patch("agilerl.training.train_llm.wandb"),
        patch("agilerl.utils.utils.aggregate_metrics_across_gpus") as mock_agg,
        patch("agilerl.training.train_llm.save_llm_checkpoint") as mock_save,
    ):
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
    # Create mock agent
    # Create mock agent
    mock_agent = MagicMock(spec=GRPO)
    mock_agent.algo = "GRPO"
    mock_agent.fitness = [0.0]
    mock_agent.local_rank = "0"  # Main process
    mock_agent.get_action.return_value = (
        [torch.ones(1, 100) for _ in range(2)],
        Mock(),
    )
    mock_agent.learn.return_value = (0.5, 0.2)
    mock_agent.test.return_value = torch.tensor([0.8])
    mock_agent.batch_size_per_process = 32
    mock_agent.batch_size = 32
    mock_agent.steps = [10]
    mock_agent.scores = [0.0]

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
    with (
        patch("agilerl.training.train_llm.trange"),
        patch("agilerl.training.train_llm.init_wandb"),
        patch("agilerl.training.train_llm.wandb"),
        patch("agilerl.utils.utils.aggregate_metrics_across_gpus") as mock_agg,
        patch("agilerl.training.train_llm.save_llm_checkpoint"),
    ):
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
    """Test the basic training loop in finetune_llm."""
    # Create mock agent
    mock_agent = MagicMock(spec=DPO)
    mock_agent.algo = "DPO"
    mock_agent.fitness = [0.0]
    mock_agent.local_rank = "0"  # Main process
    mock_agent.get_action = MagicMock()
    mock_agent.learn.return_value = (0.5, 0.2, 0.1)
    mock_agent.test.return_value = 0.87
    mock_agent.batch_size = 32
    mock_agent.batch_size_per_process = 32
    mock_agent.steps = [10]
    mock_agent.scores = [0.0]

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
    with (
        patch("agilerl.training.train_llm.trange"),
        patch("agilerl.utils.utils.aggregate_metrics_across_gpus") as mock_agg,
        patch("agilerl.training.train_llm.save_llm_checkpoint"),
    ):
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
        expected_agg_calls = 0 if use_accelerator else 21
        assert mock_agg.call_count == expected_agg_calls
        assert mock_agent.test.call_count == 3  # Should be called at step 2


@pytest.mark.parametrize(
    "use_accelerator",
    [True, False],
)
def test_finetune_llm_preference_with_wandb_and_checkpoints(use_accelerator):
    """Test finetune_llm with wandb logging and checkpointing enabled."""
    # Create mock agent
    mock_agent = MagicMock(spec=DPO)
    mock_agent.algo = "DPO"
    mock_agent.registry = MagicMock()
    mock_agent.registry.hp_config = MagicMock()
    mock_agent.registry.hp_config.config = {"lr": 0.001, "batch_size": 32}
    mock_agent.fitness = [0.0]
    mock_agent.learn.return_value = (0.5, 0.2, 0.1)
    mock_agent.test.return_value = 0.87
    mock_agent.batch_size = 32
    mock_agent.batch_size_per_process = 32
    mock_agent.steps = [10]
    mock_agent.scores = [0.0]
    mock_agent.pretrained_model_name_or_path = "Qwen/Qwen2.5-0.5B"
    mock_agent.lr = 0.001

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
    with (
        patch("agilerl.training.train_llm.trange") as mock_trange,
        patch("agilerl.training.train_llm.init_wandb") as mock_init_wandb,
        patch("agilerl.training.train_llm.wandb") as mock_wandb,
        patch("agilerl.utils.utils.aggregate_metrics_across_gpus") as mock_agg,
        patch("agilerl.training.train_llm.save_llm_checkpoint") as mock_save,
    ):
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
    mock_agent = MagicMock(spec=DPO)
    mock_agent.algo = "DPO"
    mock_agent.registry = MagicMock()
    mock_agent.registry.hp_config = MagicMock()
    mock_agent.registry.hp_config.config = {"lr": 0.001, "batch_size": 32}
    mock_agent.fitness = [0.0]
    mock_agent.learn.return_value = (0.5, 0.2, 0.1)
    mock_agent.test.return_value = 0.87
    mock_agent.batch_size = 32
    mock_agent.batch_size_per_process = 32
    mock_agent.steps = [10]
    mock_agent.scores = [0.0]

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
    with (
        patch("agilerl.training.train_llm.trange"),
        patch("agilerl.utils.utils.aggregate_metrics_across_gpus") as mock_agg,
        patch("agilerl.training.train_llm.save_llm_checkpoint"),
        patch(
            "agilerl.training.train_llm.tournament_selection_and_mutation"
        ) as mock_tournament_selection_and_mutation,
    ):
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
        expected_agg_calls = 0 if use_accelerator else 21
        assert mock_agg.call_count == expected_agg_calls
        assert mock_agent.test.call_count == 3  # Should be called at step 2
        assert (
            mock_tournament_selection_and_mutation.call_count == 6
        )  # Should be called at step 2


def test_finetune_llm_preference_warning_num_epochs_and_max_steps():
    """Test that finetune_llm raises a warning if evo_steps is not set."""
    # Create mock agent
    mock_agent = MagicMock(spec=DPO)
    mock_agent.algo = "DPO"
    mock_agent.registry = MagicMock()
    mock_agent.registry.hp_config = MagicMock()
    mock_agent.registry.hp_config.config = {"lr": 0.001, "batch_size": 32}
    mock_agent.fitness = [0.0]
    mock_agent.learn.return_value = (0.5, 0.2, 0.1)
    mock_agent.test.return_value = 0.87
    mock_agent.batch_size = 32
    mock_agent.batch_size_per_process = 32
    mock_agent.steps = [10]
    mock_agent.scores = [0.0]

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
    with (
        patch("agilerl.training.train_llm.trange"),
        patch("agilerl.utils.utils.aggregate_metrics_across_gpus") as mock_agg,
        patch("agilerl.training.train_llm.save_llm_checkpoint"),
        patch(
            "agilerl.training.train_llm.tournament_selection_and_mutation"
        ) as mock_tournament_selection_and_mutation,
    ):
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
    mock_agent = MagicMock(spec=DPO)
    mock_agent.algo = "DPO"
    mock_agent.registry = MagicMock()
    mock_agent.registry.hp_config = MagicMock()
    mock_agent.registry.hp_config.config = {"lr": 0.001, "batch_size": 32}
    mock_agent.fitness = [0.0]
    mock_agent.learn.return_value = (0.5, 0.2, 0.1)
    mock_agent.test.return_value = 0.87
    mock_agent.batch_size = 32
    mock_agent.batch_size_per_process = 32
    mock_agent.steps = [10]
    mock_agent.scores = [0.0]

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
    with (
        patch("agilerl.training.train_llm.trange"),
        patch("agilerl.training.train_llm.init_wandb"),
        patch("agilerl.training.train_llm.wandb"),
        patch("agilerl.utils.utils.aggregate_metrics_across_gpus") as mock_agg,
        patch("agilerl.training.train_llm.save_llm_checkpoint"),
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
    mock_agent = MagicMock(spec=GRPO)
    mock_agent.algo = "GRPO"
    mock_agent.pretrained_model_name_or_path = "Qwen/Qwen2.5-0.5B"
    mock_agent.batch_size_per_process = 32
    mock_agent.batch_size = 32
    mock_agent.steps = [10]
    mock_agent.scores = [0.0]
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
    # Create mock agent
    mock_agent = MagicMock(spec=DPO)
    mock_agent.algo = "DPO"
    mock_agent.pretrained_model_name_or_path = "Qwen/Qwen2.5-0.5B"
    mock_agent.batch_size_per_process = 32
    mock_agent.batch_size = 32
    mock_agent.steps = [10]
    mock_agent.scores = [0.0]
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


@pytest.mark.parametrize("use_accelerator", [True, False])
def test_finetune_llm_sft_basic_training_loop(use_accelerator):
    """Test the basic training loop in finetune_llm_sft."""
    mock_agent = MagicMock(spec=SFT)
    mock_agent.algo = "SFT"
    mock_agent.fitness = [0.0]
    mock_agent.learn.return_value = (0.5, 1.65)
    mock_agent.test.return_value = -0.4
    mock_agent.batch_size_per_process = 32
    mock_agent.batch_size = 32
    mock_agent.steps = [10]
    mock_agent.scores = [0.0]

    mock_env = MagicMock()
    mock_env.__len__.return_value = 6
    mock_env.reset.return_value = "initial_prompts"
    mock_env.step.return_value = "next_prompts"
    mock_env.data_batch_size_per_gpu = 1

    with (
        patch("agilerl.training.train_llm.trange"),
        patch("agilerl.utils.utils.safe_aggregate_metrics") as mock_safe_agg,
        patch("agilerl.training.train_llm.save_llm_checkpoint"),
    ):
        mock_safe_agg.side_effect = lambda acc, val: (
            float(val) if not isinstance(val, float) else val
        )
        finetune_llm_sft(
            pop=[mock_agent],
            env=mock_env,
            evaluation_interval=2,
            accelerator=None if use_accelerator else Accelerator(),
        )
        assert mock_env.reset.call_count == 1
        assert mock_env.reset.call_args == call(reset_dataloaders=True)
        assert mock_env.step.call_count == 6
        assert mock_agent.learn.call_count == 6
        assert mock_agent.test.call_count == 3


@pytest.mark.parametrize("use_accelerator", [True, False])
def test_finetune_llm_sft_with_wandb_and_checkpoints(use_accelerator):
    """Test finetune_llm_sft with wandb logging and checkpointing enabled."""
    mock_agent = MagicMock(spec=SFT)
    mock_agent.algo = "SFT"
    mock_agent.registry = MagicMock()
    mock_agent.registry.hp_config = MagicMock()
    mock_agent.registry.hp_config.config = {"lr": 0.001, "batch_size": 32}
    mock_agent.fitness = [0.0]
    mock_agent.learn.return_value = (0.5, 1.65)
    mock_agent.test.return_value = -0.4
    mock_agent.batch_size_per_process = 32
    mock_agent.batch_size = 32
    mock_agent.steps = [10]
    mock_agent.scores = [0.0]
    mock_agent.pretrained_model_name_or_path = "Qwen/Qwen2.5-0.5B"
    mock_agent.lr = 0.001

    mock_env = MagicMock()
    mock_env.__len__.return_value = 6
    mock_env.reset.return_value = "initial_prompts"
    mock_env.step.return_value = "next_prompts"
    mock_env.data_batch_size_per_gpu = 1

    with (
        patch("agilerl.training.train_llm.trange") as mock_trange,
        patch("agilerl.training.train_llm.init_wandb") as mock_init_wandb,
        patch("agilerl.training.train_llm.wandb") as mock_wandb,
        patch("agilerl.utils.utils.safe_aggregate_metrics") as mock_safe_agg,
        patch("agilerl.training.train_llm.save_llm_checkpoint") as mock_save,
    ):
        mock_pbar = Mock()
        mock_trange.return_value = mock_pbar
        mock_safe_agg.side_effect = lambda acc, val: (
            float(val) if not isinstance(val, float) else val
        )

        finetune_llm_sft(
            pop=[mock_agent],
            env=mock_env,
            save_elite=True,
            wb=True,
            wandb_api_key="fake_key",
            evaluation_interval=3,
            accelerator=None if use_accelerator else Accelerator(),
            checkpoint_steps=6,
        )

        mock_init_wandb.assert_called_once()
        assert mock_wandb.log.call_count >= 5
        assert mock_save.call_count == 1
        assert mock_agent.test.call_count == 2


@pytest.mark.parametrize("use_accelerator", [True, False])
def test_finetune_llm_sft_evolvable_training_loop(use_accelerator):
    """Test the evolvable training loop in finetune_llm_sft."""
    mock_agent = MagicMock(spec=SFT)
    mock_agent.algo = "SFT"
    mock_agent.fitness = [0.0]
    mock_agent.learn.return_value = (0.5, 1.65)
    mock_agent.test.return_value = -0.4
    mock_agent.batch_size_per_process = 32
    mock_agent.batch_size = 32
    mock_agent.steps = [10]
    mock_agent.scores = [0.0]

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

    with (
        patch("agilerl.training.train_llm.trange"),
        patch("agilerl.utils.utils.safe_aggregate_metrics") as mock_safe_agg,
        patch("agilerl.training.train_llm.save_llm_checkpoint"),
        patch(
            "agilerl.training.train_llm.tournament_selection_and_mutation"
        ) as mock_tournament_selection_and_mutation,
    ):
        mock_tournament_selection_and_mutation.return_value = [mock_agent]
        mock_safe_agg.side_effect = lambda acc, val: (
            float(val) if not isinstance(val, float) else val
        )

        finetune_llm_sft(
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


def test_finetune_llm_sft_warning_num_epochs_and_max_steps():
    """Test that finetune_llm_sft warns when both num_epochs and max_steps are set."""
    mock_agent = MagicMock(spec=SFT)
    mock_agent.algo = "SFT"
    mock_agent.fitness = [0.0]
    mock_agent.learn.return_value = (0.5, 1.65)
    mock_agent.test.return_value = -0.4
    mock_agent.batch_size_per_process = 32
    mock_agent.batch_size = 32
    mock_agent.steps = [10]
    mock_agent.scores = [0.0]

    mock_env = MagicMock()
    mock_env.__len__.return_value = 6
    mock_env.reset.return_value = "initial_prompts"
    mock_env.step.return_value = "next_prompts"
    mock_env.data_batch_size_per_gpu = 1

    with (
        patch("agilerl.training.train_llm.trange"),
        patch("agilerl.utils.utils.safe_aggregate_metrics") as mock_safe_agg,
        patch("agilerl.training.train_llm.save_llm_checkpoint"),
    ):
        mock_safe_agg.side_effect = lambda acc, val: (
            float(val) if not isinstance(val, float) else val
        )
        with pytest.warns(UserWarning) as num_epochs_and_max_steps_warning:
            finetune_llm_sft(
                pop=[mock_agent],
                env=mock_env,
                evaluation_interval=2,
                num_epochs=10,
                max_steps=100,
                evo_steps=None,
            )
            assert "'num_epochs' overrides 'max_steps'." in str(
                num_epochs_and_max_steps_warning[0].message
            )


def test_finetune_llm_sft_break_on_num_epochs():
    """Test that finetune_llm_sft breaks when num_epochs is reached."""
    mock_agent = MagicMock(spec=SFT)
    mock_agent.algo = "SFT"
    mock_agent.fitness = [0.0]
    mock_agent.learn.return_value = (0.5, 1.65)
    mock_agent.test.return_value = -0.4
    mock_agent.batch_size_per_process = 32
    mock_agent.batch_size = 32
    mock_agent.steps = [10]
    mock_agent.scores = [0.0]

    mock_env = MagicMock()
    mock_env.__len__.return_value = 3
    mock_env.reset.return_value = "initial_prompts"
    mock_env.step.return_value = "next_prompts"
    mock_env.data_batch_size_per_gpu = 1

    mutation = MagicMock()
    mutation.architecture_mut = 0
    mutation.new_layer_prob = 0
    mutation.parameters_mut = 0
    mutation.activation_mut = 0

    with (
        patch("agilerl.training.train_llm.trange"),
        patch("agilerl.training.train_llm.init_wandb"),
        patch("agilerl.training.train_llm.wandb"),
        patch("agilerl.utils.utils.safe_aggregate_metrics") as mock_safe_agg,
        patch("agilerl.training.train_llm.save_llm_checkpoint"),
    ):
        mock_env.num_epochs = 2
        mock_safe_agg.side_effect = lambda acc, val: (
            float(val) if not isinstance(val, float) else val
        )
        finetune_llm_sft(
            pop=[mock_agent],
            env=mock_env,
            evaluation_interval=2,
            evo_steps=1,
            accelerator=None,
            num_epochs=2,
            checkpoint_steps=3,
        )


def test_finetune_llm_sft_value_error_if_algo_not_sft():
    """Test that finetune_llm_sft raises ValueError if agent is not SFT."""
    mock_agent = MagicMock(spec=GRPO)
    mock_agent.algo = "GRPO"
    mock_agent.pretrained_model_name_or_path = "Qwen/Qwen2.5-0.5B"
    mock_agent.batch_size_per_process = 32
    mock_agent.batch_size = 32
    mock_agent.steps = [10]
    mock_agent.scores = [0.0]
    with pytest.raises(
        ValueError,
        match="Population must contain SFT agents.",
    ):
        finetune_llm_sft(
            pop=[mock_agent],
            env=MagicMock(),
            evaluation_interval=2,
            accelerator=None,
        )


def test_finetune_llm_sft_evo_steps_not_set():
    """Test that finetune_llm_sft raises ValueError if evo_steps not set with tournament/mutation."""
    with pytest.raises(ValueError) as evo_steps_not_set_error:
        finetune_llm_sft(
            pop=[MagicMock(spec=SFT)],
            env=MagicMock(),
            evo_steps=None,
            accelerator=None,
            tournament=MagicMock(),
            mutation=MagicMock(),
        )
        assert (
            "'evo_steps' must be set when 'tournament' and 'mutation' are not None."
            in str(evo_steps_not_set_error.value)
        )


def test_finetune_llm_reasoning_csv_logging_without_wandb(tmp_path):
    """csv_check True, wb_check False: aggregate block runs; CSV written; wandb.log unused."""
    mock_agent = MagicMock(spec=GRPO)
    mock_agent.registry = MagicMock()
    mock_agent.registry.hp_config = MagicMock()
    mock_agent.registry.hp_config.config = {}
    mock_agent.fitness = [0.0]
    mock_agent.get_action.return_value = (
        [torch.ones(1, 100) for _ in range(2)],
        Mock(),
    )
    mock_agent.learn.return_value = (0.5, 0.2)
    mock_agent.test.return_value = torch.tensor([0.8])
    mock_agent.algo = "GRPO"
    mock_agent.batch_size_per_process = 32
    mock_agent.batch_size = 32
    mock_agent.steps = [10]
    mock_agent.scores = [0.0]
    mock_agent.pretrained_model_name_or_path = "Qwen/Qwen2.5-0.5B"

    mock_env = MagicMock()
    mock_env.__len__.return_value = 4
    mock_env.reset.return_value = "initial_prompts"
    mock_env.step.return_value = ("next_prompts", torch.tensor([2.0, 3.0]))
    mock_env.data_batch_size_per_gpu = 1

    with (
        patch("agilerl.training.train_llm.trange"),
        patch("agilerl.training.train_llm.aggregate_metrics_across_gpus") as mock_agg,
        patch("agilerl.training.train_llm.save_llm_checkpoint"),
        patch("agilerl.training.train_llm.wandb") as mock_wandb,
    ):
        mock_agg.return_value = 0.5
        finetune_llm_reasoning(
            pop=[mock_agent],
            env=mock_env,
            evaluation_interval=2,
            max_reward=2.0,
            accelerator=None,
            elite_path=str(tmp_path),
            wb=False,
            verbose=False,
        )
    mock_wandb.log.assert_not_called()
    metrics_csv = tmp_path / "metrics.csv"
    assert metrics_csv.is_file()
    assert "Train/Best reward" in metrics_csv.read_text()


def test_finetune_llm_reasoning_wandb_and_csv_both(tmp_path):
    """wb_check and csv_check True: wandb.log and CSV row logging both run."""
    mock_agent = MagicMock(spec=GRPO)
    mock_agent.registry = MagicMock()
    mock_agent.registry.hp_config = MagicMock()
    mock_agent.registry.hp_config.config = {}
    mock_agent.fitness = [0.0]
    mock_agent.get_action.return_value = (
        [torch.ones(1, 100) for _ in range(2)],
        Mock(),
    )
    mock_agent.learn.return_value = (0.5, 0.2)
    mock_agent.test.return_value = torch.tensor([0.8])
    mock_agent.algo = "GRPO"
    mock_agent.batch_size_per_process = 32
    mock_agent.batch_size = 32
    mock_agent.steps = [10]
    mock_agent.scores = [0.0]
    mock_agent.pretrained_model_name_or_path = "Qwen/Qwen2.5-0.5B"
    mock_agent.lr = 0.01

    mock_env = MagicMock()
    mock_env.__len__.return_value = 4
    mock_env.reset.return_value = "initial_prompts"
    mock_env.step.return_value = ("next_prompts", torch.tensor([2.0, 3.0]))
    mock_env.data_batch_size_per_gpu = 1

    with (
        patch("agilerl.training.train_llm.trange"),
        patch("agilerl.training.train_llm.init_wandb"),
        patch("agilerl.training.train_llm.wandb") as mock_wandb,
        patch("agilerl.training.train_llm.aggregate_metrics_across_gpus") as mock_agg,
        patch("agilerl.training.train_llm.save_llm_checkpoint"),
    ):
        mock_agg.return_value = 0.5
        finetune_llm_reasoning(
            pop=[mock_agent],
            env=mock_env,
            evaluation_interval=2,
            max_reward=2.0,
            accelerator=None,
            elite_path=str(tmp_path),
            wb=True,
            wandb_api_key="fake_key",
            verbose=False,
        )
    assert mock_wandb.log.call_count >= 1
    assert "Train/Best reward" in (tmp_path / "metrics.csv").read_text()


def test_finetune_llm_reasoning_aggregate_skips_eval_when_never_evaluates(tmp_path):
    """agg_test_metrics stays None: inner eval merge in wb/csv block is skipped."""
    mock_agent = MagicMock(spec=GRPO)
    mock_agent.registry = MagicMock()
    mock_agent.registry.hp_config = MagicMock()
    mock_agent.registry.hp_config.config = {}
    mock_agent.fitness = [0.0]
    mock_agent.get_action.return_value = (
        [torch.ones(1, 100) for _ in range(2)],
        Mock(),
    )
    mock_agent.learn.return_value = (0.5, 0.2)
    mock_agent.test.return_value = torch.tensor([0.8])
    mock_agent.algo = "GRPO"
    mock_agent.batch_size_per_process = 32
    mock_agent.batch_size = 32
    mock_agent.steps = [10]
    mock_agent.scores = [0.0]
    mock_agent.pretrained_model_name_or_path = "x"

    mock_env = MagicMock()
    mock_env.__len__.return_value = 4
    mock_env.reset.return_value = "initial_prompts"
    mock_env.step.return_value = ("next_prompts", torch.tensor([2.0, 3.0]))
    mock_env.data_batch_size_per_gpu = 1

    with (
        patch("agilerl.training.train_llm.trange"),
        patch("agilerl.training.train_llm.aggregate_metrics_across_gpus") as mock_agg,
        patch("agilerl.training.train_llm.save_llm_checkpoint"),
        patch("agilerl.training.train_llm.wandb") as mock_wandb,
    ):
        mock_agg.return_value = 0.5
        finetune_llm_reasoning(
            pop=[mock_agent],
            env=mock_env,
            evaluation_interval=100,
            max_reward=2.0,
            accelerator=None,
            elite_path=str(tmp_path),
            wb=False,
            verbose=False,
        )
    mock_agent.test.assert_not_called()
    mock_wandb.log.assert_not_called()


def test_finetune_llm_reasoning_max_reward_none_skips_accuracy_in_aggregate(tmp_path):
    """max_reward None: train/accuracy population keys omitted in aggregate block."""
    mock_agent = MagicMock(spec=GRPO)
    mock_agent.registry = MagicMock()
    mock_agent.registry.hp_config = MagicMock()
    mock_agent.registry.hp_config.config = {}
    mock_agent.fitness = [0.0]
    mock_agent.get_action.return_value = (
        [torch.ones(1, 100) for _ in range(2)],
        Mock(),
    )
    mock_agent.learn.return_value = (0.5, 0.2)
    mock_agent.test.return_value = torch.tensor([0.8])
    mock_agent.algo = "GRPO"
    mock_agent.batch_size_per_process = 32
    mock_agent.batch_size = 32
    mock_agent.steps = [10]
    mock_agent.scores = [0.0]
    mock_agent.pretrained_model_name_or_path = "x"

    mock_env = MagicMock()
    mock_env.__len__.return_value = 4
    mock_env.reset.return_value = "initial_prompts"
    mock_env.step.return_value = ("next_prompts", torch.tensor([2.0, 3.0]))
    mock_env.data_batch_size_per_gpu = 1

    with (
        patch("agilerl.training.train_llm.trange"),
        patch("agilerl.training.train_llm.aggregate_metrics_across_gpus") as mock_agg,
        patch("agilerl.training.train_llm.save_llm_checkpoint"),
        patch("agilerl.training.train_llm.wandb") as mock_wandb,
    ):
        mock_agg.return_value = 0.5
        finetune_llm_reasoning(
            pop=[mock_agent],
            env=mock_env,
            evaluation_interval=2,
            max_reward=None,
            accelerator=None,
            elite_path=str(tmp_path),
            wb=False,
            verbose=False,
        )
    text = (tmp_path / "metrics.csv").read_text()
    assert "Train/Best reward" in text
    mock_wandb.log.assert_not_called()


def test_finetune_llm_preference_csv_logging_without_wandb(tmp_path, capsys):
    """DPO: csv_check only path; teardown closes CSV and prints path (train_llm.py ~858–860)."""
    mock_agent = MagicMock(spec=DPO)
    mock_agent.algo = "DPO"
    mock_agent.fitness = [0.0]
    mock_agent.get_action = MagicMock()
    mock_agent.learn.return_value = (0.5, 0.2, 0.1)
    mock_agent.test.return_value = 0.87
    mock_agent.batch_size = 32
    mock_agent.batch_size_per_process = 32
    mock_agent.steps = [10]
    mock_agent.scores = [0.0]
    mock_agent.pretrained_model_name_or_path = "x"

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
    mock_env = MagicMock()
    mock_env.__len__.return_value = 4
    mock_env.reset.return_value = example
    mock_env.step.return_value = example
    mock_env.data_batch_size_per_gpu = 1

    with (
        patch("agilerl.training.train_llm.trange"),
        patch("agilerl.training.train_llm.aggregate_metrics_across_gpus") as mock_agg,
        patch("agilerl.training.train_llm.save_llm_checkpoint"),
        patch("agilerl.training.train_llm.wandb") as mock_wandb,
    ):
        mock_agg.return_value = 0.5
        finetune_llm_preference(
            pop=[mock_agent],
            env=mock_env,
            evaluation_interval=2,
            accelerator=None,
            elite_path=str(tmp_path),
            wb=False,
            verbose=False,
        )
    mock_wandb.log.assert_not_called()
    csv_path = tmp_path / "metrics.csv"
    assert csv_path.is_file()
    assert "Train/Best reward margin" in csv_path.read_text()
    out = capsys.readouterr().out
    assert "Training metrics saved to" in out
    assert "metrics.csv" in out


def test_finetune_llm_preference_aggregate_skips_eval_when_never_evaluates(
    tmp_path, capsys
):
    """DPO: agg_test_metrics None skips eval keys in aggregate block."""
    mock_agent = MagicMock(spec=DPO)
    mock_agent.algo = "DPO"
    mock_agent.fitness = [0.0]
    mock_agent.get_action = MagicMock()
    mock_agent.learn.return_value = (0.5, 0.2, 0.1)
    mock_agent.test.return_value = 0.87
    mock_agent.batch_size = 32
    mock_agent.batch_size_per_process = 32
    mock_agent.steps = [10]
    mock_agent.scores = [0.0]

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
    mock_env = MagicMock()
    mock_env.__len__.return_value = 4
    mock_env.reset.return_value = example
    mock_env.step.return_value = example
    mock_env.data_batch_size_per_gpu = 1

    with (
        patch("agilerl.training.train_llm.trange"),
        patch("agilerl.training.train_llm.aggregate_metrics_across_gpus") as mock_agg,
        patch("agilerl.training.train_llm.save_llm_checkpoint"),
        patch("agilerl.training.train_llm.wandb") as mock_wandb,
    ):
        mock_agg.return_value = 0.5
        finetune_llm_preference(
            pop=[mock_agent],
            env=mock_env,
            evaluation_interval=100,
            accelerator=None,
            elite_path=str(tmp_path),
            wb=False,
            verbose=False,
        )
    mock_agent.test.assert_not_called()
    assert "Training metrics saved to" in capsys.readouterr().out


def test_finetune_llm_sft_csv_logging_without_wandb(tmp_path, capsys):
    """SFT: csv_check only; teardown closes CSV and prints path (train_llm.py ~1094–1096)."""
    mock_agent = MagicMock(spec=SFT)
    mock_agent.algo = "SFT"
    mock_agent.fitness = [0.0]
    mock_agent.learn.return_value = (0.5, 1.65)
    mock_agent.test.return_value = -0.4
    mock_agent.batch_size_per_process = 32
    mock_agent.batch_size = 32
    mock_agent.steps = [10]
    mock_agent.scores = [0.0]
    mock_agent.pretrained_model_name_or_path = "x"

    mock_env = MagicMock()
    mock_env.__len__.return_value = 4
    mock_env.reset.return_value = "initial_prompts"
    mock_env.step.return_value = "next_prompts"
    mock_env.data_batch_size_per_gpu = 1

    with (
        patch("agilerl.training.train_llm.trange"),
        patch("agilerl.utils.utils.safe_aggregate_metrics") as mock_safe_agg,
        patch("agilerl.training.train_llm.save_llm_checkpoint"),
        patch("agilerl.training.train_llm.wandb") as mock_wandb,
    ):
        mock_safe_agg.side_effect = lambda acc, val: (
            float(val) if not isinstance(val, float) else val
        )
        finetune_llm_sft(
            pop=[mock_agent],
            env=mock_env,
            evaluation_interval=2,
            accelerator=None,
            elite_path=str(tmp_path),
            wb=False,
            verbose=False,
        )
    mock_wandb.log.assert_not_called()
    metrics_csv = tmp_path / "metrics.csv"
    assert metrics_csv.is_file()
    assert "Train/Best loss" in metrics_csv.read_text(encoding="utf-8")
    out = capsys.readouterr().out
    assert "Training metrics saved to" in out
    assert "metrics.csv" in out


def test_finetune_llm_sft_aggregate_skips_eval_fitness_when_never_evaluates(tmp_path):
    """SFT: agg_test_metrics None skips Eval/Best fitness in aggregate block."""
    mock_agent = MagicMock(spec=SFT)
    mock_agent.algo = "SFT"
    mock_agent.fitness = [0.0]
    mock_agent.learn.return_value = (0.5, 1.65)
    mock_agent.test.return_value = -0.4
    mock_agent.batch_size_per_process = 32
    mock_agent.batch_size = 32
    mock_agent.steps = [10]
    mock_agent.scores = [0.0]

    mock_env = MagicMock()
    mock_env.__len__.return_value = 4
    mock_env.reset.return_value = "initial_prompts"
    mock_env.step.return_value = "next_prompts"
    mock_env.data_batch_size_per_gpu = 1

    with (
        patch("agilerl.training.train_llm.trange"),
        patch("agilerl.utils.utils.safe_aggregate_metrics") as mock_safe_agg,
        patch("agilerl.training.train_llm.save_llm_checkpoint"),
        patch("agilerl.training.train_llm.wandb") as mock_wandb,
    ):
        mock_safe_agg.side_effect = lambda acc, val: (
            float(val) if not isinstance(val, float) else val
        )
        finetune_llm_sft(
            pop=[mock_agent],
            env=mock_env,
            evaluation_interval=100,
            accelerator=None,
            elite_path=str(tmp_path),
            wb=False,
            verbose=False,
        )
    mock_agent.test.assert_not_called()


def test_create_pbar_custom_bar_format():
    from agilerl.training.train_llm import _create_pbar

    with patch("agilerl.training.train_llm.trange") as mock_trange:
        _create_pbar(None, 3, bar_format="x")
    assert mock_trange.call_args.kwargs["bar_format"] == "x"


def test_wandb_extend_hpo_skips_empty_config():
    from agilerl.training.train_llm import _wandb_extend_hpo_hyperparams

    wandb_dict: dict[str, object] = {}
    agent = MagicMock()
    agent.registry.hp_config.config = {}
    _wandb_extend_hpo_hyperparams(wandb_dict, [agent])
    assert wandb_dict == {}


def test_wandb_extend_hpo_adds_keys():
    from agilerl.training.train_llm import _wandb_extend_hpo_hyperparams

    wandb_dict: dict[str, object] = {}
    agent = MagicMock()
    agent.lr = 0.1
    agent.registry.hp_config.config = {"lr": 0.1}
    _wandb_extend_hpo_hyperparams(wandb_dict, [agent])
    assert wandb_dict["HPO_agent_0/lr"] == 0.1


def test_save_elite_checkpoint_picks_best_agent(tmp_path):
    from agilerl.training.train_llm import _save_elite_checkpoint

    with patch("agilerl.training.train_llm.save_llm_checkpoint") as save:
        worse = MagicMock()
        worse.fitness = [1.0]
        better = MagicMock()
        better.fitness = [3.0]
        elite_dir = str(tmp_path / "elite")
        _save_elite_checkpoint([worse, better], True, elite_dir, None)
    save.assert_called_once_with(better, elite_dir)


def test_open_csv_log_and_log_row(tmp_path):
    from agilerl.training.train_llm import _log_csv_row, _open_csv_log

    csv_file, writer = _open_csv_log(str(tmp_path), ["step"], None)
    assert csv_file is not None and writer is not None
    _log_csv_row(writer, csv_file, {"step": 1})
    csv_file.close()
