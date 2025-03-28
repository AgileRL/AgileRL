from unittest.mock import MagicMock, Mock, call, patch

import pytest
import torch
from accelerate import Accelerator

from agilerl.training.train_llm import (
    aggregate_metrics_across_gpus,
    finetune_llm,
    gather_tensor,
    save_llm_checkpoint,
)


@pytest.fixture
def mock_agent():
    agent = MagicMock()
    agent.local_rank = 0
    return agent


@patch("torch.distributed.get_world_size")
@patch("torch.distributed.all_gather")
def test_gather_tensor_with_tensor_input(
    mock_all_gather, mock_get_world_size, mock_agent
):
    mock_get_world_size.return_value = 3
    input_tensor = torch.tensor([1.0, 2.0, 3.0], device=f"cuda:{mock_agent.local_rank}")

    def mock_gather(output_list, input_tensor):
        output_list[0].copy_(
            torch.tensor([1.0, 2.0, 3.0], device=f"cuda:{mock_agent.local_rank}")
        )
        output_list[1].copy_(
            torch.tensor([4.0, 5.0, 6.0], device=f"cuda:{mock_agent.local_rank}")
        )
        output_list[2].copy_(
            torch.tensor([7.0, 8.0, 9.0], device=f"cuda:{mock_agent.local_rank}")
        )

    mock_all_gather.side_effect = mock_gather
    mock_agent.device = f"cuda:{mock_agent.local_rank}"
    result = gather_tensor(input_tensor, mock_agent)
    expected = torch.tensor(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
        device=f"cuda:{mock_agent.local_rank}",
    )
    assert torch.allclose(result, expected)
    mock_get_world_size.assert_called_once()
    mock_all_gather.assert_called_once()


@patch("torch.distributed.get_world_size")
@patch("torch.distributed.all_gather")
def test_gather_tensor_with_scalar_input(
    mock_all_gather, mock_get_world_size, mock_agent
):
    mock_get_world_size.return_value = 2
    input_scalar = 42.0

    def mock_gather(output_list, input_tensor):
        output_list[0].copy_(torch.tensor(42.0, device=f"cuda:{mock_agent.local_rank}"))
        output_list[1].copy_(torch.tensor(84.0, device=f"cuda:{mock_agent.local_rank}"))

    mock_all_gather.side_effect = mock_gather
    mock_agent.device = f"cuda:{mock_agent.local_rank}"
    result = gather_tensor(input_scalar, mock_agent)
    expected = torch.tensor([42.0, 84.0], device=f"cuda:{mock_agent.local_rank}")
    assert torch.allclose(result, expected)
    mock_get_world_size.assert_called_once()
    mock_all_gather.assert_called_once()


@pytest.fixture
def setup_test_data():
    agent = Mock()
    agent.device = torch.device("cpu")
    agent.world_size = 4
    loss = torch.tensor([[2.5]])
    kl = torch.tensor([[1.2]])
    rewards = torch.tensor([3.0, 4.0, 5.0])

    return agent, loss, kl, rewards


def mock_gather_tensor(tensor, agent):
    if not isinstance(tensor, torch.Tensor):
        tensor = torch.tensor(tensor, device=f"cuda:{agent.local_rank}")
    tensor = tensor.detach().clone()
    world_size = agent.world_size
    gathered_tensors = []
    for i in range(world_size):
        gathered_tensors.append(tensor)
    return torch.stack(gathered_tensors)


@patch("agilerl.training.train_llm.gather_tensor", side_effect=mock_gather_tensor)
def test_basic_aggregation(mock_gather, setup_test_data):
    """Test basic aggregation functionality."""
    agent, *data = setup_test_data
    avg_loss, avg_kl, avg_reward = (
        aggregate_metrics_across_gpus(agent, metric) for metric in data
    )
    mock_gather.assert_called()
    assert avg_loss == 2.5
    assert pytest.approx(avg_kl) == 1.2
    assert avg_reward == 4.0
    assert mock_gather.call_count == 3
    mock_gather.assert_any_call(data[0], agent)
    mock_gather.assert_any_call(data[1], agent)
    assert mock_gather.call_args_list[2][0][0].mean() == 4.0


def test_save_with_accelerator():
    """Test saving checkpoint when agent has an accelerator."""
    agent = Mock()
    agent.actor = Mock()
    agent.accelerator = Mock()
    unwrapped_model = Mock()
    agent.accelerator.unwrap_model = Mock(return_value=unwrapped_model)
    save_llm_checkpoint(agent, "test_checkpoint", 100)
    agent.accelerator.unwrap_model.assert_called_once_with(agent.actor)
    unwrapped_model.save_pretrained.assert_called_once_with("test_checkpoint/step_100")


def test_save_without_accelerator():
    """Test saving checkpoint when agent has no accelerator."""
    agent = Mock()
    agent.actor = Mock()
    agent.accelerator = None
    save_llm_checkpoint(agent, None, 42)
    agent.actor.save_pretrained.assert_called_once_with("./saved_checkpoints/step_42")


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


def test_finetune_llm_with_wandb_and_checkpoints():
    """Test finetune_llm with wandb logging and checkpointing enabled."""
    # Create mock agent
    mock_agent = MagicMock()
    mock_agent.algo = "GRPO"
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
            checkpoint_interval=2,
            checkpoint_path="test_checkpoint",
            wb=True,
            wandb_api_key="fake_key",
            evaluation_interval=3,
            accelerator=None,
            max_reward=2.0,
        )

        # Verify wandb was initialized
        mock_init_wandb.assert_called_once()

        # Verify wandb logging
        assert mock_wandb.log.call_count >= 5  # At least once per step

        # Verify checkpointing
        assert mock_save.call_count == 3  # Should be called at steps 2 and 4
        mock_save.assert_has_calls(
            [
                call(mock_agent, "test_checkpoint", 1),  # i=1 for step 2
                call(mock_agent, "test_checkpoint", 3),  # i=3 for step 4
            ]
        )

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
            mutation=Mock(),
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
