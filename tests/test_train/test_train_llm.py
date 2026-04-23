from unittest.mock import ANY, MagicMock, Mock, call, patch

import pytest
import torch
from accelerate import Accelerator

pytest.importorskip("transformers", reason="LLM dependencies not installed")

from agilerl.algorithms import DPO, GRPO, LLMPPO, LLMREINFORCE
from agilerl.algorithms.sft import SFT
from agilerl.rollouts.on_policy import collect_rollouts_llm
from agilerl.training.train_llm import (
    _format_prefixed_metrics,
    _normalize_learn_metrics,
    build_eval_wandb_dict,
    build_train_wandb_dict,
    finetune_llm_multiturn,
    finetune_llm_preference,
    finetune_llm_reasoning,
    finetune_llm_sft,
)

pytestmark = pytest.mark.llm


def _make_multiturn_mock_env(*, turn_boundaries_len: int = 3):
    """GEM-style env: reset/step/get_episode_data + turn_boundaries for step accounting."""
    mock_env = MagicMock(
        spec=["reset", "step", "get_episode_data", "turn_boundaries"],
    )
    prompt_dict: dict = {
        "input_ids": torch.ones(1, 4, dtype=torch.long),
        "attention_mask": torch.ones(1, 4, dtype=torch.long),
    }
    mock_env.reset.return_value = (prompt_dict, {})
    mock_env.step.return_value = (prompt_dict, 0.0, False, False, {})
    mock_env.turn_boundaries = list(range(turn_boundaries_len))
    L = 8
    T = 2
    mock_env.get_episode_data.return_value = (
        torch.ones(1, L, dtype=torch.long),
        torch.ones(1, L, dtype=torch.long),
        torch.zeros(1, L, dtype=torch.long),
        torch.ones(T, dtype=torch.float32),
    )
    return mock_env


def _make_pop_for_wandb_dict(size: int = 2):
    pop = []
    for idx in range(size):
        agent = MagicMock()
        agent.index = idx
        agent.registry = MagicMock()
        agent.registry.hp_config = MagicMock()
        agent.registry.hp_config.config = {"lr": 1e-4}
        agent.lr = 1e-4 + idx * 1e-5
        pop.append(agent)
    return pop


def _make_multiturn_mock_agent(*, spec=LLMPPO):
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

    def _mock_get_action(obs, training=True, **kwargs):
        if isinstance(obs, dict):
            input_ids = obs.get("input_ids")
            batch = int(input_ids.shape[0]) if hasattr(input_ids, "shape") else 1
        else:
            batch = len(obs)
        return ([torch.ones(1, 5, dtype=torch.long) for _ in range(batch)], None)

    mock_agent.get_action.side_effect = _mock_get_action
    if spec is GRPO:
        mock_agent.learn.return_value = {"mean_loss": 0.5, "mean_kl": 0.2}
    else:
        mock_agent.learn.return_value = {
            "mean_loss": 0.5,
            "mean_kl": 0.2,
            "mean_pg_loss": 0.1,
            "mean_vf_loss": 0.1,
            "mean_entropy": 1.0,
        }
    mock_agent.batch_size = 16
    mock_agent.batch_size_per_process = 16
    mock_agent.max_model_len = 1024
    mock_agent.pretrained_model_name_or_path = "Qwen/Qwen2.5-0.5B"
    mock_agent.steps = [10]
    mock_agent.scores = [0.0]
    mock_agent.index = 0
    mock_agent.mut = 0
    mock_agent.device = torch.device("cpu")
    mock_agent.set_reference_policy = MagicMock()
    return mock_agent


def _make_multiturn_env_factory(*, turn_boundaries_len: int = 3):
    return lambda: _make_multiturn_mock_env(turn_boundaries_len=turn_boundaries_len)


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
        patch("agilerl.training.train_llm.aggregate_metrics_across_gpus") as mock_agg,
        patch("agilerl.training.train_llm.save_llm_checkpoint"),
    ):
        mock_agg.return_value = 0.5
        finetune_llm_reasoning(
            pop=[mock_agent],
            env=mock_env,
            evaluation_interval=2,
            max_reward=2.0,
            accelerator=Accelerator() if use_accelerator else None,
        )
        assert mock_env.reset.call_count == 1
        assert mock_env.reset.call_args == call(reset_dataloaders=True)
        assert mock_agent.get_action.call_count == 6
        assert mock_env.step.call_count == 6
        assert mock_agent.learn.call_count == 6
        expected_agg_calls = 36
        assert mock_agg.call_count == expected_agg_calls
        if not use_accelerator:
            assert all(
                call_args.args[0] is None for call_args in mock_agg.call_args_list
            )
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
        patch("agilerl.training.train_llm.aggregate_metrics_across_gpus") as mock_agg,
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
        patch("agilerl.training.train_llm.aggregate_metrics_across_gpus") as mock_agg,
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
            accelerator=Accelerator() if use_accelerator else None,
            tournament=Mock(),
            mutation=mutation,
        )
        assert mock_env.reset.call_count == 1
        assert mock_env.reset.call_args == call(reset_dataloaders=True)
        assert mock_agent.get_action.call_count == 6
        assert mock_env.step.call_count == 6
        assert mock_agent.learn.call_count == 6
        expected_agg_calls = 36
        assert mock_agg.call_count == expected_agg_calls
        if not use_accelerator:
            assert all(
                call_args.args[0] is None for call_args in mock_agg.call_args_list
            )
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
        patch("agilerl.training.train_llm.aggregate_metrics_across_gpus") as mock_agg,
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
        patch("agilerl.training.train_llm.aggregate_metrics_across_gpus") as mock_agg,
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
        patch("agilerl.training.train_llm.aggregate_metrics_across_gpus") as mock_agg,
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
        patch("agilerl.training.train_llm.aggregate_metrics_across_gpus") as mock_agg,
        patch("agilerl.training.train_llm.save_llm_checkpoint"),
    ):
        mock_agg.return_value = 0.5
        finetune_llm_preference(
            pop=[mock_agent],
            env=mock_env,
            evaluation_interval=2,
            accelerator=Accelerator() if use_accelerator else None,
        )
        assert mock_env.reset.call_count == 1
        assert mock_env.reset.call_args == call(reset_dataloaders=True)
        assert mock_agent.get_action.call_count == 0
        assert mock_env.step.call_count == 6
        assert mock_agent.learn.call_count == 6
        expected_agg_calls = 21
        assert mock_agg.call_count == expected_agg_calls
        if not use_accelerator:
            assert all(
                call_args.args[0] is None for call_args in mock_agg.call_args_list
            )
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
        patch("agilerl.training.train_llm.aggregate_metrics_across_gpus") as mock_agg,
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
            accelerator=Accelerator() if use_accelerator else None,
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
        patch("agilerl.training.train_llm.aggregate_metrics_across_gpus") as mock_agg,
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
            accelerator=Accelerator() if use_accelerator else None,
            tournament=Mock(),
            mutation=mutation,
        )
        assert mock_env.reset.call_count == 1
        assert mock_env.reset.call_args == call(reset_dataloaders=True)
        assert mock_env.step.call_count == 6
        assert mock_agent.learn.call_count == 6
        expected_agg_calls = 21
        assert mock_agg.call_count == expected_agg_calls
        if not use_accelerator:
            assert all(
                call_args.args[0] is None for call_args in mock_agg.call_args_list
            )
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
        patch("agilerl.training.train_llm.aggregate_metrics_across_gpus") as mock_agg,
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
        patch("agilerl.training.train_llm.aggregate_metrics_across_gpus") as mock_agg,
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
        match="The algorithm must be GRPO, LLMPPO, or LLMREINFORCE for reasoning-based reinforcement learning",
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
    mock_agent.batch_size_per_process = 1
    mock_agent.batch_size = 1
    mock_agent.steps = [10]
    mock_agent.scores = [0.0]

    mock_env = MagicMock()
    mock_env.__len__.return_value = 6
    mock_env.name = "mock_sft"
    mock_env.num_epochs = 0
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
    mock_agent.batch_size_per_process = 1
    mock_agent.batch_size = 1
    mock_agent.steps = [10]
    mock_agent.scores = [0.0]
    mock_agent.pretrained_model_name_or_path = "Qwen/Qwen2.5-0.5B"
    mock_agent.lr = 0.001

    mock_env = MagicMock()
    mock_env.__len__.return_value = 6
    mock_env.name = "mock_sft"
    mock_env.num_epochs = 0
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
    mock_agent.batch_size_per_process = 1
    mock_agent.batch_size = 1
    mock_agent.steps = [10]
    mock_agent.scores = [0.0]

    mock_env = MagicMock()
    mock_env.__len__.return_value = 6
    mock_env.name = "mock_sft"
    mock_env.num_epochs = 0
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
            assert "num_epochs" in str(num_epochs_and_max_steps_warning[0].message)
            assert "max_steps" in str(num_epochs_and_max_steps_warning[0].message)


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
        match="The algorithm must be SFT",
    ):
        finetune_llm_sft(
            pop=[mock_agent],
            env=MagicMock(),
            evaluation_interval=2,
            accelerator=None,
        )


def test_finetune_llm_sft_evo_steps_not_set():
    """Test that finetune_llm_sft raises ValueError if evo_steps not set with tournament/mutation."""
    with pytest.raises(
        ValueError,
        match="'evo_steps' must be set if 'tournament' and 'mutation' are not None",
    ):
        finetune_llm_sft(
            pop=[MagicMock(spec=SFT)],
            env=MagicMock(),
            evo_steps=None,
            accelerator=None,
            tournament=MagicMock(),
            mutation=MagicMock(),
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
            log_csv=True,
            verbose=False,
        )
    mock_wandb.log.assert_not_called()
    metrics_csv = tmp_path / "metrics.csv"
    assert metrics_csv.is_file()
    assert "Train/Best Reward" in metrics_csv.read_text()


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
            log_csv=True,
            verbose=False,
        )
    assert mock_wandb.log.call_count >= 1
    assert "Train/Best Reward" in (tmp_path / "metrics.csv").read_text()


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
            log_csv=True,
            verbose=False,
        )
    mock_agent.test.assert_not_called()
    mock_wandb.log.assert_not_called()


@pytest.mark.parametrize(
    ("finetune_fn", "agent_spec"),
    [
        (finetune_llm_reasoning, GRPO),
        (finetune_llm_preference, DPO),
        (finetune_llm_sft, SFT),
    ],
)
def test_finetune_llm_env_and_env_fn_mutually_exclusive(finetune_fn, agent_spec):
    agent = MagicMock(spec=agent_spec)
    if agent_spec is GRPO:
        agent.algo = "GRPO"
    elif agent_spec is DPO:
        agent.algo = "DPO"
    else:
        agent.algo = "SFT"
    agent.batch_size_per_process = 1
    agent.batch_size = 1
    agent.steps = [0]
    agent.scores = [0.0]
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
            accelerator=None,
        )


@pytest.mark.parametrize(
    "finetune_fn",
    [finetune_llm_reasoning, finetune_llm_preference, finetune_llm_sft],
)
def test_finetune_llm_requires_env_or_env_fn(finetune_fn):
    with pytest.raises(ValueError, match="Either 'env' or 'env_fn' must be provided"):
        finetune_fn(
            pop=[
                MagicMock(spec=SFT) if finetune_fn is finetune_llm_sft else MagicMock()
            ],
            env=None,
            env_fn=None,
            max_steps=0,
            verbose=False,
            accelerator=None,
        )


@pytest.mark.parametrize(
    ("finetune_fn", "agent_spec"),
    [
        (finetune_llm_reasoning, GRPO),
        (finetune_llm_preference, DPO),
        (finetune_llm_sft, SFT),
    ],
)
def test_finetune_llm_warns_on_shared_env_with_population(finetune_fn, agent_spec):
    agents = []
    for algo_name in ("a0", "a1"):
        agent = MagicMock(spec=agent_spec)
        if agent_spec is GRPO:
            agent.algo = "GRPO"
        elif agent_spec is DPO:
            agent.algo = "DPO"
        else:
            agent.algo = "SFT"
        agent.batch_size_per_process = 1
        agent.batch_size = 1
        agent.steps = [0]
        agent.scores = [0.0]
        agent.pretrained_model_name_or_path = "Qwen/Qwen2.5-0.5B"
        agent.fitness = [0.0]
        agent.index = algo_name
        if agent_spec is SFT:
            agent.learn.return_value = (0.5, 1.0)
            agent.test.return_value = -0.1
        agents.append(agent)

    env = MagicMock()
    env.__len__.return_value = 1
    env.name = "mock_env"
    env.data_batch_size_per_gpu = 1
    env.num_epochs = 0
    env.reset.return_value = "prompts"
    env.step.return_value = "prompts"

    with pytest.warns(UserWarning, match="fairness bias"):
        finetune_fn(
            pop=agents,
            env=env,
            max_steps=0,
            verbose=False,
            accelerator=None,
        )


def test_finetune_llm_reasoning_env_fn_uses_distinct_env_instances():
    agent_a = MagicMock(spec=GRPO)
    agent_a.algo = "GRPO"
    agent_a.fitness = [0.0]
    agent_a.get_action.return_value = ([torch.ones(1, 4)], Mock())
    agent_a.learn.return_value = (0.5, 0.2)
    agent_a.batch_size_per_process = 1
    agent_a.batch_size = 1
    agent_a.steps = [0]
    agent_a.scores = [0.0]
    agent_a.pretrained_model_name_or_path = "Qwen/Qwen2.5-0.5B"

    agent_b = MagicMock(spec=GRPO)
    agent_b.algo = "GRPO"
    agent_b.fitness = [0.0]
    agent_b.get_action.return_value = ([torch.ones(1, 4)], Mock())
    agent_b.learn.return_value = (0.5, 0.2)
    agent_b.batch_size_per_process = 1
    agent_b.batch_size = 1
    agent_b.steps = [0]
    agent_b.scores = [0.0]
    agent_b.pretrained_model_name_or_path = "Qwen/Qwen2.5-0.5B"

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
    env_b.step.return_value = ("next_b", torch.tensor([2.0]))

    env_fn = MagicMock(side_effect=[env_a, env_b])

    with (
        patch("agilerl.training.train_llm.trange"),
        patch(
            "agilerl.training.train_llm.aggregate_metrics_across_gpus", return_value=0.5
        ),
        patch("agilerl.training.train_llm.save_llm_checkpoint"),
    ):
        finetune_llm_reasoning(
            pop=[agent_a, agent_b],
            env_fn=env_fn,
            max_steps=2,
            evaluation_interval=100,
            verbose=False,
            accelerator=None,
        )

    assert env_fn.call_count == 2
    assert env_a.step.call_count == 1
    assert env_b.step.call_count == 1
    assert agent_a.get_action.call_args.args[0] == "prompts_a"
    assert agent_b.get_action.call_args.args[0] == "prompts_b"


def test_finetune_llm_preference_env_fn_uses_distinct_env_instances():
    agent_a = MagicMock(spec=DPO)
    agent_a.algo = "DPO"
    agent_a.fitness = [0.0]
    agent_a.learn.return_value = (0.5, 0.2, 0.1)
    agent_a.batch_size_per_process = 1
    agent_a.batch_size = 1
    agent_a.steps = [0]
    agent_a.scores = [0.0]
    agent_a.pretrained_model_name_or_path = "Qwen/Qwen2.5-0.5B"

    agent_b = MagicMock(spec=DPO)
    agent_b.algo = "DPO"
    agent_b.fitness = [0.0]
    agent_b.learn.return_value = (0.5, 0.2, 0.1)
    agent_b.batch_size_per_process = 1
    agent_b.batch_size = 1
    agent_b.steps = [0]
    agent_b.scores = [0.0]
    agent_b.pretrained_model_name_or_path = "Qwen/Qwen2.5-0.5B"

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
        patch("agilerl.training.train_llm.trange"),
        patch(
            "agilerl.training.train_llm.aggregate_metrics_across_gpus", return_value=0.5
        ),
        patch("agilerl.training.train_llm.save_llm_checkpoint"),
    ):
        finetune_llm_preference(
            pop=[agent_a, agent_b],
            env_fn=env_fn,
            max_steps=2,
            evaluation_interval=100,
            verbose=False,
            accelerator=None,
        )

    assert env_fn.call_count == 2
    assert env_a.step.call_count == 1
    assert env_b.step.call_count == 1
    assert agent_a.learn.call_args.args[0] == {"prompt": ["a"]}
    assert agent_b.learn.call_args.args[0] == {"prompt": ["b"]}


@pytest.mark.parametrize(
    "finetune_fn", [finetune_llm_reasoning, finetune_llm_preference]
)
def test_finetune_llm_checkpoint_triggering_non_divisible_steps(finetune_fn):
    if finetune_fn is finetune_llm_reasoning:
        agent = MagicMock(spec=GRPO)
        agent.algo = "GRPO"
        agent.get_action.return_value = ([torch.ones(1, 4)], Mock())
        agent.learn.return_value = (0.5, 0.2)
        env = MagicMock()
        env.reset.return_value = "prompts"
        env.step.return_value = ("next", torch.tensor([1.0]))
    else:
        agent = MagicMock(spec=DPO)
        agent.algo = "DPO"
        agent.learn.return_value = (0.5, 0.2, 0.1)
        env = MagicMock()
        env.reset.return_value = {"prompt": ["x"]}
        env.step.return_value = {"prompt": ["y"]}

    agent.fitness = [0.0]
    agent.batch_size_per_process = 1
    agent.batch_size = 1
    agent.steps = [0]
    agent.scores = [0.0]
    agent.pretrained_model_name_or_path = "Qwen/Qwen2.5-0.5B"
    env.__len__.return_value = 10
    env.name = "mock_env"
    env.data_batch_size_per_gpu = 1
    env.num_epochs = 0

    with (
        patch("agilerl.training.train_llm.trange"),
        patch(
            "agilerl.training.train_llm.aggregate_metrics_across_gpus", return_value=0.5
        ),
        patch("agilerl.training.train_llm.save_llm_checkpoint") as mock_save,
    ):
        finetune_fn(
            pop=[agent],
            env=env,
            max_steps=5,
            checkpoint_steps=2,
            evaluation_interval=100,
            verbose=False,
            accelerator=None,
        )

    assert mock_save.call_count == 3


# --- finetune_llm_multiturn ---


@pytest.mark.parametrize("agent_spec", [LLMPPO, LLMREINFORCE, GRPO])
@pytest.mark.parametrize("use_accelerator", [True, False])
def test_finetune_llm_multiturn_basic_training_loop(agent_spec, use_accelerator):
    """Smoke: episode collection, learn with turn_ids, step accounting; no agent.test."""
    mock_agent = _make_multiturn_mock_agent(spec=agent_spec)
    mock_env = _make_multiturn_mock_env(turn_boundaries_len=3)
    max_turns = 2
    batch_size = 1
    batch_steps_per_iter = len(mock_env.turn_boundaries)
    max_steps = 9
    num_outer = max_steps // batch_steps_per_iter

    with (
        patch("agilerl.training.train_llm.trange"),
        patch("agilerl.training.train_llm.stack_and_pad_experiences") as mock_stack,
        patch("agilerl.training.train_llm.aggregate_metrics_across_gpus") as mock_agg,
        patch("agilerl.training.train_llm.save_llm_checkpoint") as mock_save,
    ):
        mock_stack.return_value = (torch.zeros(1, 8, dtype=torch.long),)
        mock_agg.return_value = 0.5
        finetune_llm_multiturn(
            pop=[mock_agent],
            env_factory=lambda: mock_env,
            max_turns=max_turns,
            init_hp={"BATCH_SIZE": batch_size, "ALGO": mock_agent.algo},
            max_steps=max_steps,
            evaluation_interval=100,
            verbose=False,
            accelerator=None if use_accelerator else Accelerator(),
        )

    assert mock_env.reset.call_count == num_outer * batch_size
    assert mock_agent.get_action.call_count == num_outer * batch_size * max_turns
    assert mock_env.step.call_count == num_outer * batch_size * max_turns
    assert mock_env.get_episode_data.call_count == num_outer * batch_size
    assert mock_agent.learn.call_count == num_outer
    assert mock_agent.test.call_count == 0
    n_metrics = 4 if agent_spec is GRPO else 7
    assert mock_agg.call_count == num_outer * n_metrics
    if agent_spec is GRPO:
        mock_agent.learn.assert_called_with(ANY)
    else:
        mock_agent.learn.assert_called_with(ANY, turn_ids=ANY)
    assert mock_save.call_count == 1


def test_finetune_llm_multiturn_grpo_requires_batch_multiple_of_group_size():
    mock_agent = _make_multiturn_mock_agent(spec=GRPO)
    mock_agent.group_size = 2
    mock_agent.batch_size = 16
    mock_agent.batch_size_per_process = 16
    mock_env = _make_multiturn_mock_env(turn_boundaries_len=3)
    with pytest.raises(ValueError, match="divisible by"):
        finetune_llm_multiturn(
            pop=[mock_agent],
            max_turns=1,
            env_factory=_make_multiturn_env_factory(turn_boundaries_len=3),
            init_hp={"BATCH_SIZE": 3, "ALGO": "GRPO"},
            max_steps=100,
            accelerator=None,
            verbose=False,
        )


@pytest.mark.parametrize("use_accelerator", [True, False])
def test_finetune_llm_multiturn_with_wandb_and_checkpoints(use_accelerator):
    mock_agent = _make_multiturn_mock_agent()
    mock_agent.registry = MagicMock()
    mock_agent.registry.hp_config = MagicMock()
    mock_agent.registry.hp_config.config = {"lr": 0.001, "batch_size": 16}
    mock_agent.lr = 0.01
    mock_env = _make_multiturn_mock_env(turn_boundaries_len=3)

    with (
        patch("agilerl.training.train_llm.trange") as mock_trange,
        patch("agilerl.training.train_llm.stack_and_pad_experiences") as mock_stack,
        patch("agilerl.training.train_llm.init_wandb") as mock_init_wandb,
        patch("agilerl.training.train_llm.wandb") as mock_wandb,
        patch("agilerl.training.train_llm.aggregate_metrics_across_gpus") as mock_agg,
        patch("agilerl.training.train_llm.save_llm_checkpoint") as mock_save,
    ):
        mock_trange.return_value = Mock()
        mock_stack.return_value = (torch.zeros(1, 8, dtype=torch.long),)
        mock_agg.return_value = 0.5

        finetune_llm_multiturn(
            pop=[mock_agent],
            env_factory=lambda: mock_env,
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

    mock_init_wandb.assert_called_once()
    assert mock_wandb.log.call_count >= 2
    assert mock_save.call_count >= 1


@pytest.mark.parametrize("use_accelerator", [True, False])
def test_finetune_llm_multiturn_evolvable_training_loop(use_accelerator):
    mock_agent = _make_multiturn_mock_agent()
    mock_env = _make_multiturn_mock_env(turn_boundaries_len=3)
    mutation = MagicMock()
    mutation.architecture_mut = 0
    mutation.new_layer_prob = 0
    mutation.parameters_mut = 0
    mutation.activation_mut = 0

    with (
        patch("agilerl.training.train_llm.trange"),
        patch("agilerl.training.train_llm.stack_and_pad_experiences") as mock_stack,
        patch("agilerl.training.train_llm.aggregate_metrics_across_gpus") as mock_agg,
        patch("agilerl.training.train_llm.save_llm_checkpoint") as mock_save,
        patch(
            "agilerl.training.train_llm.tournament_selection_and_mutation"
        ) as mock_tourn,
    ):
        mock_stack.return_value = (torch.zeros(1, 8, dtype=torch.long),)
        mock_agg.return_value = 0.5
        mock_tourn.return_value = [mock_agent]

        finetune_llm_multiturn(
            pop=[mock_agent],
            env_factory=lambda: mock_env,
            max_turns=2,
            init_hp={"BATCH_SIZE": 1, "ALGO": "LLMPPO"},
            max_steps=9,
            evaluation_interval=100,
            verbose=False,
            evo_steps=1,
            tournament=Mock(),
            mutation=mutation,
            accelerator=None if use_accelerator else Accelerator(),
        )

    assert mock_tourn.call_count == 3
    assert mock_save.call_count == 0


def test_finetune_llm_multiturn_value_error_when_evo_steps_missing_with_tournament():
    mutation = MagicMock()
    mutation.architecture_mut = 0
    mutation.new_layer_prob = 0
    mutation.parameters_mut = 0
    mutation.activation_mut = 0
    mock_agent = _make_multiturn_mock_agent()
    with pytest.raises(ValueError, match="'evo_steps' must be set"):
        finetune_llm_multiturn(
            pop=[mock_agent],
            env_factory=MagicMock,
            max_turns=1,
            init_hp={"BATCH_SIZE": 1, "ALGO": "LLMPPO"},
            max_steps=1,
            evo_steps=None,
            tournament=MagicMock(),
            mutation=mutation,
            accelerator=None,
        )


def test_finetune_llm_multiturn_warns_when_evo_steps_without_tournament():
    mock_agent = _make_multiturn_mock_agent()
    with pytest.warns(UserWarning, match="evo_steps"):
        finetune_llm_multiturn(
            pop=[mock_agent],
            env_factory=MagicMock,
            max_turns=1,
            init_hp={"BATCH_SIZE": 1, "ALGO": "LLMPPO"},
            max_steps=0,
            evo_steps=3,
            tournament=None,
            mutation=None,
            accelerator=None,
            verbose=False,
        )


def test_finetune_llm_multiturn_value_error_if_algo_not_supported():
    mock_agent = MagicMock(spec=DPO)
    mock_agent.algo = "DPO"
    mock_agent.batch_size = 16
    mock_agent.batch_size_per_process = 16
    with pytest.raises(
        ValueError,
        match="The algorithm must be LLMPPO, LLMREINFORCE, or GRPO for multi-turn GEM",
    ):
        finetune_llm_multiturn(
            pop=[mock_agent],
            env_factory=MagicMock,
            max_turns=1,
            init_hp={"BATCH_SIZE": 1, "ALGO": "DPO"},
            max_steps=0,
            accelerator=None,
            verbose=False,
        )


def test_finetune_llm_multiturn_eval_fn_interval():
    mock_agent = _make_multiturn_mock_agent()
    mock_env = _make_multiturn_mock_env(turn_boundaries_len=3)
    eval_fn = MagicMock(return_value=0.42)

    with (
        patch("agilerl.training.train_llm.trange"),
        patch("agilerl.training.train_llm.stack_and_pad_experiences") as mock_stack,
        patch("agilerl.training.train_llm.aggregate_metrics_across_gpus") as mock_agg,
        patch("agilerl.training.train_llm.save_llm_checkpoint"),
    ):
        mock_stack.return_value = (torch.zeros(1, 8, dtype=torch.long),)
        mock_agg.return_value = 0.5
        finetune_llm_multiturn(
            pop=[mock_agent],
            env_factory=lambda: mock_env,
            max_turns=2,
            init_hp={"BATCH_SIZE": 1, "ALGO": "LLMPPO"},
            max_steps=9,
            evaluation_interval=1,
            eval_fn=eval_fn,
            verbose=False,
            accelerator=None,
        )

    assert eval_fn.call_count == 3
    n_metrics = 7
    n_eval_agg = 3
    assert mock_agg.call_count == 3 * n_metrics + n_eval_agg


def test_finetune_llm_multiturn_max_reward_adds_accuracy_metric():
    mock_agent = _make_multiturn_mock_agent()
    mock_env = _make_multiturn_mock_env(turn_boundaries_len=3)

    with (
        patch("agilerl.training.train_llm.trange"),
        patch("agilerl.training.train_llm.stack_and_pad_experiences") as mock_stack,
        patch("agilerl.training.train_llm.aggregate_metrics_across_gpus") as mock_agg,
        patch("agilerl.training.train_llm.save_llm_checkpoint"),
    ):
        mock_stack.return_value = (torch.zeros(1, 8, dtype=torch.long),)
        mock_agg.return_value = 0.5
        finetune_llm_multiturn(
            pop=[mock_agent],
            env_factory=lambda: mock_env,
            max_turns=2,
            init_hp={"BATCH_SIZE": 1, "ALGO": "LLMPPO"},
            max_steps=9,
            evaluation_interval=100,
            max_reward=1.0,
            verbose=False,
            accelerator=None,
        )

    num_outer = 3
    assert mock_agg.call_count == num_outer * 8


def test_finetune_llm_multiturn_init_hp_none_uses_agent_fields():
    """Covers init_hp branch that copies BATCH_SIZE_PER_GPU and ALGO from the agent."""
    mock_agent = _make_multiturn_mock_agent()
    mock_agent.batch_size_per_process = 7
    mock_agent.algo = "LLMPPO"
    mock_env = _make_multiturn_mock_env(turn_boundaries_len=3)

    with (
        patch("agilerl.training.train_llm.trange"),
        patch("agilerl.training.train_llm.stack_and_pad_experiences") as mock_stack,
        patch("agilerl.training.train_llm.aggregate_metrics_across_gpus") as mock_agg,
        patch("agilerl.training.train_llm.save_llm_checkpoint"),
        patch("agilerl.training.train_llm.init_wandb") as mock_init_wandb,
        patch("agilerl.training.train_llm.wandb"),
    ):
        mock_stack.return_value = (torch.zeros(1, 8, dtype=torch.long),)
        mock_agg.return_value = 0.5
        finetune_llm_multiturn(
            pop=[mock_agent],
            env_factory=lambda: mock_env,
            max_turns=2,
            init_hp=None,
            max_steps=0,
            wb=True,
            wandb_api_key="fake",
            verbose=False,
            accelerator=None,
        )

    init_kw = mock_init_wandb.call_args.kwargs["init_hyperparams"]
    assert init_kw["BATCH_SIZE_PER_GPU"] == 7
    assert init_kw["ALGO"] == "LLMPPO"


def test_finetune_llm_multiturn_sliding_window_max_model_len_assert_passes():
    """Covers getattr(env, '_sw_max_model_len') when it matches agent.max_model_len."""
    mock_agent = _make_multiturn_mock_agent()
    mock_agent.max_model_len = 1024
    mock_env = MagicMock()
    prompt: dict = {
        "input_ids": torch.ones(1, 4, dtype=torch.long),
        "attention_mask": torch.ones(1, 4, dtype=torch.long),
    }
    L, T = 8, 2
    mock_env.reset.return_value = (prompt, {})
    mock_env.step.return_value = (prompt, 0.0, False, False, {})
    mock_env.turn_boundaries = [0, 1, 2]
    mock_env._sw_max_model_len = 1024
    mock_env.get_episode_data.return_value = (
        torch.ones(1, L, dtype=torch.long),
        torch.ones(1, L, dtype=torch.long),
        torch.zeros(1, L, dtype=torch.long),
        torch.ones(T, dtype=torch.float32),
    )

    with (
        patch("agilerl.training.train_llm.trange"),
        patch("agilerl.training.train_llm.stack_and_pad_experiences") as mock_stack,
        patch("agilerl.training.train_llm.aggregate_metrics_across_gpus") as mock_agg,
        patch("agilerl.training.train_llm.save_llm_checkpoint"),
    ):
        mock_stack.return_value = (torch.zeros(1, 8, dtype=torch.long),)
        mock_agg.return_value = 0.5
        finetune_llm_multiturn(
            pop=[mock_agent],
            env_factory=lambda: mock_env,
            max_turns=2,
            init_hp={"BATCH_SIZE": 1, "ALGO": "LLMPPO"},
            max_steps=3,
            evaluation_interval=100,
            verbose=False,
            accelerator=None,
        )


def test_finetune_llm_multiturn_breaks_turn_loop_when_terminated():
    """Covers early exit from the max_turns loop when env.step sets terminated."""
    mock_agent = _make_multiturn_mock_agent()
    mock_env = _make_multiturn_mock_env(turn_boundaries_len=3)
    prompt: dict = {
        "input_ids": torch.ones(1, 4, dtype=torch.long),
        "attention_mask": torch.ones(1, 4, dtype=torch.long),
    }
    mock_env.reset.return_value = (prompt, {})
    mock_env.step.return_value = (prompt, 1.0, True, False, {})
    max_turns = 5

    with (
        patch("agilerl.training.train_llm.trange"),
        patch("agilerl.training.train_llm.stack_and_pad_experiences") as mock_stack,
        patch("agilerl.training.train_llm.aggregate_metrics_across_gpus") as mock_agg,
        patch("agilerl.training.train_llm.save_llm_checkpoint"),
    ):
        mock_stack.return_value = (torch.zeros(1, 8, dtype=torch.long),)
        mock_agg.return_value = 0.5
        finetune_llm_multiturn(
            pop=[mock_agent],
            env_factory=lambda: mock_env,
            max_turns=max_turns,
            init_hp={"BATCH_SIZE": 1, "ALGO": "LLMPPO"},
            max_steps=3,
            evaluation_interval=100,
            verbose=False,
            accelerator=None,
        )

    assert mock_agent.get_action.call_count == 1


def test_finetune_llm_multiturn_wandb_accuracy_and_eval_scores_with_verbose_banner():
    """W&B max_reward keys, Eval/Best score from eval_fn, HPO keys, verbose pbar.write paths."""
    mock_pbar = Mock()
    mock_agent = _make_multiturn_mock_agent()
    mock_agent.registry = MagicMock()
    mock_agent.registry.hp_config = MagicMock()
    mock_agent.registry.hp_config.config = {"lr": 0.01}
    mock_agent.lr = 0.01
    mock_env = _make_multiturn_mock_env(turn_boundaries_len=3)
    eval_fn = MagicMock(return_value=0.33)

    with (
        patch("agilerl.training.train_llm.trange", return_value=mock_pbar),
        patch("agilerl.training.train_llm.stack_and_pad_experiences") as mock_stack,
        patch("agilerl.training.train_llm.aggregate_metrics_across_gpus") as mock_agg,
        patch("agilerl.training.train_llm.save_llm_checkpoint"),
        patch("agilerl.training.train_llm.init_wandb"),
        patch("agilerl.training.train_llm.wandb") as mock_wandb,
    ):
        mock_stack.return_value = (torch.zeros(1, 8, dtype=torch.long),)
        mock_agg.return_value = 0.5
        finetune_llm_multiturn(
            pop=[mock_agent],
            env_factory=lambda: mock_env,
            max_turns=2,
            init_hp={"BATCH_SIZE": 1, "ALGO": "LLMPPO", "env_name": "gem_test"},
            max_steps=9,
            evaluation_interval=1,
            eval_fn=eval_fn,
            max_reward=0.5,
            wb=True,
            wandb_api_key="fake",
            verbose=True,
            accelerator=None,
        )

    assert mock_pbar.write.call_count >= 2
    eval_logged = any(
        "Eval/Best Score" in c.args[0] for c in mock_wandb.log.call_args_list
    )
    assert eval_logged
    hpo_logged = any(
        "HPO_agent_0/lr" in c.args[0] for c in mock_wandb.log.call_args_list
    )
    assert hpo_logged
    acc_logged = any(
        "Train/Best Accuracy" in c.args[0] for c in mock_wandb.log.call_args_list
    )
    assert acc_logged


def test_finetune_llm_multiturn_accelerator_syncs_after_eval_fn():
    """Covers accelerator.wait_for_everyone() after distributed eval aggregation."""
    mock_agent = _make_multiturn_mock_agent()
    mock_env = _make_multiturn_mock_env(turn_boundaries_len=3)
    acc = MagicMock(spec=Accelerator)
    acc.is_main_process = True
    acc.wait_for_everyone = MagicMock()

    with (
        patch("agilerl.training.train_llm.trange"),
        patch("agilerl.training.train_llm.stack_and_pad_experiences") as mock_stack,
        patch("agilerl.training.train_llm.aggregate_metrics_across_gpus") as mock_agg,
        patch("agilerl.training.train_llm.save_llm_checkpoint"),
        patch(
            "agilerl.training.train_llm._distributed_world_size",
            return_value=1,
        ),
    ):
        mock_stack.return_value = (torch.zeros(1, 8, dtype=torch.long),)
        mock_agg.return_value = 0.5
        finetune_llm_multiturn(
            pop=[mock_agent],
            env_factory=lambda: mock_env,
            max_turns=2,
            init_hp={"BATCH_SIZE": 1, "ALGO": "LLMPPO"},
            max_steps=3,
            evaluation_interval=1,
            eval_fn=lambda _a: 0.1,
            verbose=False,
            accelerator=acc,
        )

    assert acc.wait_for_everyone.call_count >= 1


def test_collect_rollouts_llm_breaks_when_vector_env_has_no_active_prompts():
    mock_agent = _make_multiturn_mock_agent()
    prompt = {
        "input_ids": torch.ones(1, 3, dtype=torch.long),
        "attention_mask": torch.ones(1, 3, dtype=torch.long),
    }
    mock_env = MagicMock(spec=["reset", "step", "get_trajectories"])
    mock_env.reset.return_value = prompt
    mock_env.step.return_value = None
    mock_env.get_trajectories.return_value = (
        [torch.ones(1, 8, dtype=torch.long)],
        [torch.ones(1, 7, dtype=torch.bool)],
        [torch.zeros(1, 7, dtype=torch.long)],
        [torch.ones(2, dtype=torch.float32)],
        1,
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


def test_build_train_wandb_dict_reasoning_llmppo_uses_fallback_pg_and_entropy_keys():
    pop = _make_pop_for_wandb_dict(size=2)
    agent = MagicMock(spec=LLMPPO)
    agent_metrics_dict = {
        "agent_0/train_metrics": {
            "Train/Rewards": 1.0,
            "Train/Mean Loss": 0.5,
            "Train/Mean KL": 0.1,
            "Train/Completion Length": 8.0,
            "Train/PG Loss": 0.2,
            "Train/Entropy": 1.2,
            "Train/Mean VF Loss": 0.3,
            "Train/Accuracy": 0.5,
        },
        "agent_1/train_metrics": {
            "Train/Rewards": 3.0,
            "Train/Mean Loss": 0.7,
            "Train/Mean KL": 0.3,
            "Train/Completion Length": 10.0,
            "Train/PG Loss": 0.4,
            "Train/Entropy": 1.0,
            "Train/Mean VF Loss": 0.5,
            "Train/Accuracy": 0.75,
        },
    }

    out = build_train_wandb_dict(
        agent_metrics_dict=agent_metrics_dict,
        pop=pop,
        agent=agent,
        max_reward=4.0,
        mode="reasoning",
    )
    assert out["Train/Best Reward"] == 3.0
    assert out["Train/Mean Population Reward"] == pytest.approx(2.0)
    assert out["Train/Mean Population PG Loss"] == pytest.approx(0.3)
    assert out["Train/Mean Population Entropy"] == pytest.approx(1.1)
    assert out["Train/Mean Population Critic Loss"] == pytest.approx(0.4)
    assert out["Train/Best Accuracy"] == pytest.approx(0.75)
    assert "HPO_agent_0/lr" in out


def test_build_eval_wandb_dict_preference_and_multiturn_score_modes():
    pop = _make_pop_for_wandb_dict(size=2)
    pref_metrics = {
        "agent_0/test_metrics": {"Eval/Mean Reward Margin": 0.2},
        "agent_1/test_metrics": {"Eval/Mean Reward Margin": 0.6},
    }
    pref_out = build_eval_wandb_dict(pref_metrics, pop=pop, mode="preference")
    assert pref_out["Eval/Best Reward Margin"] == pytest.approx(0.6)
    assert pref_out["Eval/Mean Population Reward Margin"] == pytest.approx(0.4)

    score_metrics = {
        "agent_0/test_metrics": {"Eval/Score": 0.9},
        "agent_1/test_metrics": {"Eval/Score": 0.7},
    }
    score_out = build_eval_wandb_dict(
        score_metrics, pop=pop, mode="multiturn", eval_score_mode=True
    )
    assert score_out["Eval/Best Score"] == pytest.approx(0.9)
    assert score_out["Eval/Mean Population Score"] == pytest.approx(0.8)


def test_build_train_and_eval_wandb_dict_sft_mode():
    pop = _make_pop_for_wandb_dict(size=2)
    agent = MagicMock(spec=SFT)
    train_metrics = {
        "agent_0/train_metrics": {
            "Train/Loss": 0.5,
            "Train/Perplexity": 10.0,
        },
        "agent_1/train_metrics": {
            "Train/Loss": 0.3,
            "Train/Perplexity": 12.0,
        },
    }
    train_out = build_train_wandb_dict(
        agent_metrics_dict=train_metrics,
        pop=pop,
        agent=agent,
        mode="sft",
    )
    assert train_out["Train/Best Loss"] == pytest.approx(0.3)
    assert train_out["Train/Mean Population Loss"] == pytest.approx(0.4)
    assert train_out["Train/Mean Population Perplexity"] == pytest.approx(11.0)
    assert "HPO_agent_0/lr" in train_out

    eval_metrics = {
        "agent_0/test_metrics": {"Eval/Negative loss (fitness)": -0.5},
        "agent_1/test_metrics": {"Eval/Negative loss (fitness)": -0.3},
    }
    eval_out = build_eval_wandb_dict(eval_metrics, pop=pop, mode="sft")
    assert eval_out["Eval/Best Fitness"] == pytest.approx(-0.3)
    assert eval_out["Eval/Mean Population Fitness"] == pytest.approx(-0.4)


def test_train_metric_format_and_learn_output_normalization_helpers():
    formatted = _format_prefixed_metrics({"mean_kl": 0.2, "mean_pg_loss": 0.1}, "Train")
    assert formatted["Train/Mean KL"] == 0.2
    assert formatted["Train/Mean PG Loss"] == 0.1

    agent = MagicMock(spec=LLMREINFORCE)
    metrics = _normalize_learn_metrics(agent, (1.0, 0.5, 0.2, 0.3), mode="multiturn")
    assert metrics["mean_loss"] == 1.0
    assert metrics["mean_kl"] == 0.5
    assert metrics["pg_loss"] == 0.2
    assert metrics["entropy"] == 0.3


def test_normalize_learn_metrics_error_paths_and_multiturn_len5():
    agent = MagicMock(spec=LLMPPO)

    with pytest.raises(TypeError, match="Expected learn\\(\\) to return dict or tuple"):
        _normalize_learn_metrics(agent, 1.23, mode="reasoning")

    with pytest.raises(
        ValueError, match="Preference learn\\(\\) tuple output must have 3 values"
    ):
        _normalize_learn_metrics(agent, (1.0, 2.0), mode="preference")

    mt_metrics = _normalize_learn_metrics(
        agent,
        (1.0, 0.5, 0.2, 0.3, 0.1),
        mode="multiturn",
    )
    assert mt_metrics["mean_vf_loss"] == 0.3
    assert mt_metrics["mean_entropy"] == 0.1

    with pytest.raises(
        ValueError,
        match="Reasoning/multi-turn learn\\(\\) tuple output has an unsupported shape",
    ):
        _normalize_learn_metrics(agent, (1.0, 0.5, 0.2), mode="reasoning")


def test_validate_evolution_args_warns_when_checkpoint_steps_ignored():
    from agilerl.training.train_llm import _validate_llm_evolution_args

    with pytest.warns(
        UserWarning,
        match="'checkpoint_steps' is set, but evolution is active",
    ):
        _validate_llm_evolution_args(
            evo_steps=2,
            tournament=MagicMock(),
            mutation=MagicMock(),
            checkpoint_steps=10,
        )


def test_init_llm_wandb_passes_entity_and_run_name():
    from agilerl.training.train_llm import _init_llm_wandb

    agent = MagicMock()
    agent.batch_size = 8
    agent.pretrained_model_name_or_path = "mock-model"
    pop = [agent]
    init_hp = {"ALGO": "GRPO"}

    with patch("agilerl.training.train_llm.init_wandb") as mock_init:
        _init_llm_wandb(
            init_hp=init_hp,
            pop=pop,
            env_name="mock-env",
            effective_data_batch_size=8,
            wb=True,
            wandb_api_key="fake-key",
            accelerator=None,
            wandb_entity="acme",
            wandb_run_name="run-1",
        )

    assert mock_init.call_args.kwargs["addl_args"] == {
        "entity": "acme",
        "name": "run-1",
    }


@pytest.mark.parametrize(
    "finetune_fn, agent_spec",
    [
        (finetune_llm_reasoning, GRPO),
        (finetune_llm_preference, DPO),
        (finetune_llm_sft, SFT),
    ],
)
def test_inner_loop_breaks_after_max_steps_first_agent(finetune_fn, agent_spec):
    if agent_spec is GRPO:
        agent0 = MagicMock(spec=GRPO)
        agent1 = MagicMock(spec=GRPO)
        for agent in (agent0, agent1):
            agent.algo = "GRPO"
            agent.get_action.return_value = ([torch.ones(1, 5)], Mock())
            agent.learn.return_value = (0.5, 0.2)
            agent.test.return_value = torch.tensor([0.8])
            agent.batch_size = 1
            agent.batch_size_per_process = 1
            agent.steps = [0]
            agent.scores = [0.0]
            agent.pretrained_model_name_or_path = "x"
            agent.fitness = [0.0]
        env = MagicMock()
        env.__len__.return_value = 2
        env.reset.return_value = "initial_prompts"
        env.step.return_value = ("next_prompts", torch.tensor([1.0]))
    elif agent_spec is DPO:
        agent0 = MagicMock(spec=DPO)
        agent1 = MagicMock(spec=DPO)
        for agent in (agent0, agent1):
            agent.algo = "DPO"
            agent.learn.return_value = (0.5, 0.2, 0.1)
            agent.test.return_value = 0.7
            agent.batch_size = 1
            agent.batch_size_per_process = 1
            agent.steps = [0]
            agent.scores = [0.0]
            agent.pretrained_model_name_or_path = "x"
            agent.fitness = [0.0]
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
        agent0 = MagicMock(spec=SFT)
        agent1 = MagicMock(spec=SFT)
        for agent in (agent0, agent1):
            agent.algo = "SFT"
            agent.learn.return_value = (0.5, 1.2)
            agent.test.return_value = -0.3
            agent.batch_size = 1
            agent.batch_size_per_process = 1
            agent.steps = [0]
            agent.scores = [0.0]
            agent.pretrained_model_name_or_path = "x"
            agent.fitness = [0.0]
            agent.registry = MagicMock()
            agent.registry.hp_config = MagicMock()
            agent.registry.hp_config.config = {}
        env = MagicMock()
        env.__len__.return_value = 2
        env.reset.return_value = "initial_prompts"
        env.step.return_value = "next_prompts"

    env.data_batch_size_per_gpu = 1

    with (
        patch("agilerl.training.train_llm.trange"),
        patch("agilerl.training.train_llm.save_llm_checkpoint"),
        patch(
            "agilerl.training.train_llm.aggregate_metrics_across_gpus", return_value=0.5
        ),
        patch(
            "agilerl.training.train_llm.safe_aggregate_metrics",
            side_effect=lambda _a, v: float(v),
        ),
    ):
        finetune_fn(
            pop=[agent0, agent1],
            env=env,
            accelerator=None,
            max_steps=1,
            evaluation_interval=100,
            verbose=False,
        )
    assert agent0.learn.call_count == 1
    assert agent1.learn.call_count == 0


def test_finetune_llm_sft_env_fn_updates_prompts_by_agent():
    """SFT env_fn path initializes and updates per-agent prompts."""
    agent0 = MagicMock(spec=SFT)
    agent1 = MagicMock(spec=SFT)
    for agent in (agent0, agent1):
        agent.algo = "SFT"
        agent.learn.return_value = (0.5, 1.2)
        agent.test.return_value = -0.3
        agent.batch_size = 1
        agent.batch_size_per_process = 1
        agent.steps = [0]
        agent.scores = [0.0]
        agent.pretrained_model_name_or_path = "x"
        agent.fitness = [0.0]
        agent.registry = MagicMock()
        agent.registry.hp_config = MagicMock()
        agent.registry.hp_config.config = {}

    def _mk_env():
        env = MagicMock()
        env.__len__.return_value = 2
        env.reset.return_value = "initial_prompts"
        env.step.return_value = "next_prompts"
        env.data_batch_size_per_gpu = 1
        return env

    with (
        patch("agilerl.training.train_llm.trange"),
        patch("agilerl.training.train_llm.save_llm_checkpoint"),
        patch(
            "agilerl.training.train_llm.safe_aggregate_metrics",
            side_effect=lambda _a, v: float(v),
        ),
    ):
        finetune_llm_sft(
            pop=[agent0, agent1],
            env_fn=_mk_env,
            accelerator=None,
            max_steps=2,
            evaluation_interval=100,
            verbose=False,
        )
    assert agent0.learn.call_count >= 1
    assert agent1.learn.call_count >= 1


def test_finetune_llm_multiturn_raises_when_group_size_not_divisible_by_batch_size():
    agent = _make_multiturn_mock_agent(spec=GRPO)
    agent.group_size = 3
    agent.batch_size = 2
    agent.batch_size_per_process = 2

    with pytest.raises(
        ValueError,
        match="Group size \\(3\\) must be divisible by batch size \\(2\\)",
    ):
        finetune_llm_multiturn(
            pop=[agent],
            max_turns=2,
            env_factory=_make_multiturn_env_factory(),
            init_hp={"BATCH_SIZE": 2, "BATCH_SIZE_PER_GPU": 2, "ALGO": "GRPO"},
            max_steps=8,
            accelerator=None,
            wb=False,
            verbose=False,
        )


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
            log_csv=True,
            verbose=False,
        )
    text = (tmp_path / "metrics.csv").read_text()
    assert "Train/Best Reward" in text
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
            log_csv=True,
            verbose=False,
        )
    mock_wandb.log.assert_not_called()
    csv_path = tmp_path / "metrics.csv"
    assert csv_path.is_file()
    assert "Train/Best Reward Margin" in csv_path.read_text()
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
            log_csv=True,
            verbose=False,
        )
    mock_agent.test.assert_not_called()
    assert "Training metrics saved to" in capsys.readouterr().out


def test_finetune_llm_sft_csv_logging_without_wandb(tmp_path, capsys):
    """SFT: csv_check only; teardown closes CSV and prints path (train_llm.py ~1094–1096)."""
    mock_agent = MagicMock(spec=SFT)
    mock_agent.algo = "SFT"
    mock_agent.registry = MagicMock()
    mock_agent.registry.hp_config = MagicMock()
    mock_agent.registry.hp_config.config = {}
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
            log_csv=True,
            verbose=False,
        )
    mock_wandb.log.assert_not_called()
    metrics_csv = tmp_path / "metrics.csv"
    assert metrics_csv.is_file()
    assert "Train/Best Loss" in metrics_csv.read_text(encoding="utf-8")
    out = capsys.readouterr().out
    assert "Training metrics saved to" in out
    assert "metrics.csv" in out


def test_finetune_llm_sft_aggregate_skips_eval_fitness_when_never_evaluates(tmp_path):
    """SFT: no eval skips Eval/Best Fitness keys in aggregate block."""
    mock_agent = MagicMock(spec=SFT)
    mock_agent.algo = "SFT"
    mock_agent.registry = MagicMock()
    mock_agent.registry.hp_config = MagicMock()
    mock_agent.registry.hp_config.config = {}
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
            log_csv=True,
            verbose=False,
        )
    mock_agent.test.assert_not_called()


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


def test_save_elite_checkpoint_waits_but_skips_non_main_process(tmp_path):
    from agilerl.training.train_llm import _save_elite_checkpoint

    acc = MagicMock()
    acc.is_main_process = False
    with patch("agilerl.training.train_llm.save_llm_checkpoint") as save:
        agent = MagicMock()
        agent.fitness = [1.0]
        _save_elite_checkpoint([agent], True, str(tmp_path / "elite"), acc)
    acc.wait_for_everyone.assert_called_once()
    save.assert_not_called()


def test_open_csv_log_and_log_row(tmp_path):
    from agilerl.training.train_llm import _log_csv_row, _open_csv_log

    csv_file, writer = _open_csv_log(str(tmp_path), ["step"], None)
    assert csv_file is not None and writer is not None
    _log_csv_row(writer, csv_file, {"step": 1}, None)
    csv_file.close()

    non_main = MagicMock()
    non_main.is_main_process = False
    csv_file_none, writer_none = _open_csv_log(str(tmp_path), ["step"], non_main)
    assert csv_file_none is None and writer_none is None

    writer_mock = MagicMock()
    file_mock = MagicMock()
    _log_csv_row(writer_mock, file_mock, {"step": 2}, non_main)
    writer_mock.writerow.assert_not_called()
