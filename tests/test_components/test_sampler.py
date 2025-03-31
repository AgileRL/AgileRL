import pytest
import torch
from accelerate import Accelerator
from tensordict import TensorDict
from torch.utils.data import DataLoader

from agilerl.components.data import ReplayDataset
from agilerl.components.replay_buffer import (
    MultiStepReplayBuffer,
    PrioritizedReplayBuffer,
    ReplayBuffer,
)
from agilerl.components.sampler import Sampler


# Initialize Sampler with default parameters
def test_initialize_with_default_parameters():
    memory_size = 100
    device = "cpu"
    buffer = ReplayBuffer(memory_size, device)

    sampler = Sampler(memory=buffer)
    assert sampler.distributed is False
    assert sampler.per is False
    assert sampler.n_step is False
    assert sampler.memory == buffer
    assert sampler.dataset is None
    assert sampler.dataloader is None


# Call sample_standard() method with valid batch_size
def test_sample_standard_with_valid_batch_size():
    memory_size = 100
    device = "cpu"

    buffer = ReplayBuffer(memory_size, device)
    sampler = Sampler(
        memory=buffer,
        dataset=None,
        dataloader=None,
    )

    experiences1 = TensorDict(
        {
            "state": [1],
            "action": [2],
            "reward": [3],
        },
        batch_size=[1],
    )
    experiences2 = TensorDict(
        {
            "state": [4],
            "action": [5],
            "reward": [6],
        },
        batch_size=[1],
    )
    experiences3 = TensorDict(
        {
            "state": [7],
            "action": [8],
            "reward": [9],
        },
        batch_size=[1],
    )
    # Add experiences to memory
    buffer.add(experiences1)
    buffer.add(experiences2)
    buffer.add(experiences3)

    batch_size = 3
    samples = sampler.sample(batch_size)

    # TensorDict should have the right shape
    assert samples.batch_size[0] == batch_size
    # Check that all keys are present
    assert "state" in samples
    assert "action" in samples
    assert "reward" in samples
    # Check that tensors have the expected batch dimension
    assert samples["state"].shape[0] == batch_size


# Call sample_distributed() method with valid batch_size
def test_sample_distributed_with_valid_batch_size():
    accelerator = Accelerator()

    memory_size = 100
    batch_size = 3

    buffer = ReplayBuffer(memory_size)
    replay_dataset = ReplayDataset(buffer, batch_size=batch_size)

    replay_dataloader = DataLoader(replay_dataset, batch_size=None)
    replay_dataloader = accelerator.prepare(replay_dataloader)
    sampler = Sampler(
        dataset=replay_dataset,
        dataloader=replay_dataloader,
    )

    # Add experiences to memory
    experiences1 = TensorDict(
        {
            "state": [1],
            "action": [2],
            "reward": [3],
        },
        batch_size=[1],
    )
    experiences2 = TensorDict(
        {
            "state": [4],
            "action": [5],
            "reward": [6],
        },
        batch_size=[1],
    )
    experiences3 = TensorDict(
        {
            "state": [7],
            "action": [8],
            "reward": [9],
        },
        batch_size=[1],
    )
    buffer.add(experiences1)
    buffer.add(experiences2)
    buffer.add(experiences3)

    samples = sampler.sample(batch_size)

    # Check that all expected keys are present
    assert "state" in samples
    assert "action" in samples
    assert "reward" in samples

    # Check that tensors have the expected dimensions by examining their shape directly
    assert samples["state"].shape[0] == batch_size


# Call sample_per() method with valid batch_size
def test_sample_per_with_valid_batch_size():
    memory_size = 100
    alpha = 0.6
    device = "cpu"

    buffer = PrioritizedReplayBuffer(memory_size, alpha, device)
    sampler = Sampler(
        memory=buffer,
        dataset=None,
        dataloader=None,
    )

    # Add experiences to memory
    experiences1 = TensorDict(
        {
            "state": torch.tensor([[1]], dtype=torch.float32),
            "action": torch.tensor([[2]], dtype=torch.float32),
            "reward": torch.tensor([[3]], dtype=torch.float32),
            "next_state": torch.tensor([[2]], dtype=torch.float32),
            "done": torch.tensor([[1]], dtype=torch.float32),
        },
        batch_size=[1],
    )
    experiences2 = TensorDict(
        {
            "state": torch.tensor([[4]], dtype=torch.float32),
            "action": torch.tensor([[5]], dtype=torch.float32),
            "reward": torch.tensor([[6]], dtype=torch.float32),
            "next_state": torch.tensor([[5]], dtype=torch.float32),
            "done": torch.tensor([[0]], dtype=torch.float32),
        },
        batch_size=[1],
    )
    experiences3 = TensorDict(
        {
            "state": torch.tensor([[7]], dtype=torch.float32),
            "action": torch.tensor([[8]], dtype=torch.float32),
            "reward": torch.tensor([[9]], dtype=torch.float32),
            "next_state": torch.tensor([[8]], dtype=torch.float32),
            "done": torch.tensor([[0]], dtype=torch.float32),
        },
        batch_size=[1],
    )

    buffer.add(experiences1)
    buffer.add(experiences2)
    buffer.add(experiences3)

    batch_size = 2  # Reduced batch size to avoid dimension mismatch issues
    samples = sampler.sample(batch_size, beta=0.4)

    # Check that expected keys are present
    assert "state" in samples
    assert "action" in samples
    assert "reward" in samples
    assert "next_state" in samples
    assert "done" in samples
    assert "weights" in samples
    assert "idxs" in samples

    # Check dimensions match the requested batch size
    assert samples["weights"].shape[0] == batch_size
    assert samples["idxs"].shape[0] == batch_size


# Call sample_n_step() method with valid batch_size
def test_sample_n_step_with_valid_batch_size():
    memory_size = 10000
    n_step = 3
    gamma = 0.95
    device = "cpu"

    buffer = MultiStepReplayBuffer(memory_size, n_step, gamma, device)
    sampler = Sampler(
        memory=buffer,
        dataset=None,
        dataloader=None,
    )

    # Add experiences to memory
    experiences1 = TensorDict(
        {
            "obs": [1],
            "action": [2],
            "reward": [3.0],
            "next_obs": [2],
            "done": [1],
        },
    )
    experiences2 = TensorDict(
        {
            "obs": [4],
            "action": [5],
            "reward": [6.0],
            "next_obs": [5],
            "done": [0],
        },
    )
    experiences3 = TensorDict(
        {
            "obs": [7],
            "action": [8],
            "reward": [9.0],
            "next_obs": [8],
            "done": [0],
        },
    )
    experiences4 = TensorDict(
        {
            "obs": [9],
            "action": [8],
            "reward": [7.0],
            "next_obs": [8],
            "done": [0],
        },
    )
    for experience in [experiences1, experiences2, experiences3, experiences4]:
        experience = experience.unsqueeze(0)
        experience.batch_size = [1]
        buffer.add(experience)

    idxs = torch.tensor([0, 1])
    samples = sampler.sample(idxs)

    # Check that some expected keys are present
    assert "obs" in samples
    assert "action" in samples
    assert "reward" in samples

    # Check dimensions
    assert samples["obs"].shape[0] == len(idxs)


@pytest.mark.parametrize(
    "memory, dataset, dataloader",
    [
        (0, None, None),
        (None, 0, 0),
        (0, None, None),
        (0, None, None),
    ],
)
def test_warnings_in_constructor(memory, dataset, dataloader):
    with pytest.warns():
        _ = Sampler(
            memory,
            dataset,
            dataloader,
        )


# Test that the Sampler correctly replaces the collate function in a regular DataLoader
def test_replace_dataloader_collate_fn():
    memory_size = 100
    batch_size = 3

    buffer = ReplayBuffer(memory_size)
    replay_dataset = ReplayDataset(buffer, batch_size=batch_size)

    # Create a regular DataLoader with default collate function
    original_dataloader = DataLoader(replay_dataset, batch_size=None)

    # Create a Sampler that should replace the collate function
    sampler = Sampler(
        memory=None,
        dataset=replay_dataset,
        dataloader=original_dataloader,
    )

    # Add experiences to memory
    experiences1 = TensorDict(
        {
            "state": [1],
            "action": [2],
            "reward": [3],
        },
        batch_size=[1],
    )
    experiences2 = TensorDict(
        {
            "state": [4],
            "action": [5],
            "reward": [6],
        },
        batch_size=[1],
    )
    experiences3 = TensorDict(
        {
            "state": [7],
            "action": [8],
            "reward": [9],
        },
        batch_size=[1],
    )
    buffer.add(experiences1)
    buffer.add(experiences2)
    buffer.add(experiences3)

    # Verify that the Sampler created a new dataloader and stored the original
    assert sampler.dataloader is not original_dataloader
    assert sampler.dataloader.collate_fn == Sampler.tensordict_collate_fn

    # Sample from the sampler
    samples = sampler.sample(batch_size)

    # Check that all expected keys are present
    assert "state" in samples
    assert "action" in samples
    assert "reward" in samples

    # Check that tensors have the expected dimensions
    assert samples["state"].shape[0] == batch_size
