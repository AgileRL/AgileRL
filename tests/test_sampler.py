import pytest
from accelerate import Accelerator
from torch.utils.data import DataLoader

from agilerl.components.replay_buffer import (
    MultiStepReplayBuffer,
    PrioritizedReplayBuffer,
    ReplayBuffer,
)
from agilerl.components.replay_data import ReplayDataset
from agilerl.components.sampler import Sampler


# Initialize Sampler with default parameters
def test_initialize_with_default_parameters():
    action_dim = 1
    memory_size = 100
    field_names = ["state", "action", "reward"]
    device = "cpu"

    buffer = ReplayBuffer(action_dim, memory_size, field_names, device)

    sampler = Sampler(memory=buffer)
    assert sampler.distributed is False
    assert sampler.per is False
    assert sampler.n_step is False
    assert sampler.memory == buffer
    assert sampler.dataset is None
    assert sampler.dataloader is None


# Call sample_standard() method with valid batch_size
def test_sample_standard_with_valid_batch_size():
    action_dim = 1
    memory_size = 100
    field_names = ["state", "action", "reward"]
    device = "cpu"

    buffer = ReplayBuffer(action_dim, memory_size, field_names, device)
    sampler = Sampler(
        distributed=False,
        per=False,
        n_step=False,
        memory=buffer,
        dataset=None,
        dataloader=None,
    )

    # Add experiences to memory
    buffer.save2memorySingleEnv(1, 2, 3)
    buffer.save2memorySingleEnv(4, 5, 6)
    buffer.save2memorySingleEnv(7, 8, 9)

    batch_size = 3
    samples = sampler.sample(batch_size)

    assert len(samples) == len(field_names)
    assert len(samples[0]) == batch_size
    assert len(samples[1]) == batch_size
    assert len(samples[2]) == batch_size


# Call sample_distributed() method with valid batch_size
def test_sample_distributed_with_valid_batch_size():
    accelerator = Accelerator()

    action_dim = 1
    memory_size = 100
    field_names = ["state", "action", "reward"]
    batch_size = 3

    buffer = ReplayBuffer(action_dim, memory_size, field_names)
    replay_dataset = ReplayDataset(buffer, batch_size=batch_size)
    replay_dataloader = DataLoader(replay_dataset, batch_size=None)
    replay_dataloader = accelerator.prepare(replay_dataloader)

    sampler = Sampler(
        distributed=True,
        per=False,
        n_step=False,
        memory=None,
        dataset=replay_dataset,
        dataloader=replay_dataloader,
    )

    # Add experiences to memory
    buffer.save2memorySingleEnv(1, 2, 3)
    buffer.save2memorySingleEnv(4, 5, 6)
    buffer.save2memorySingleEnv(7, 8, 9)

    samples = sampler.sample(batch_size)

    assert len(samples) == len(field_names)
    assert len(samples[0]) == batch_size
    assert len(samples[1]) == batch_size
    assert len(samples[2]) == batch_size


# Call sample_per() method with valid batch_size
def test_sample_per_with_valid_batch_size():
    action_dim = 1
    memory_size = 100
    field_names = ["state", "action", "reward", "next_state", "done"]
    num_envs = 1
    alpha = 0.6
    n_step = 1
    gamma = 0.99
    device = "cpu"

    buffer = PrioritizedReplayBuffer(
        action_dim, memory_size, field_names, num_envs, alpha, n_step, gamma, device
    )
    sampler = Sampler(
        distributed=False,
        per=True,
        n_step=False,
        memory=buffer,
        dataset=None,
        dataloader=None,
    )

    # Add experiences to memory
    buffer.save2memorySingleEnv(1, 2, 3, 2, 1)
    buffer.save2memorySingleEnv(4, 5, 6, 5, 4)
    buffer.save2memorySingleEnv(7, 8, 9, 8, 7)

    batch_size = 3
    samples = sampler.sample(batch_size, beta=0.4)

    assert len(samples) == len(field_names) + 2  # Fields, + weights and idxs
    assert len(samples[0]) == batch_size
    assert len(samples[1]) == batch_size
    assert len(samples[2]) == batch_size


# Call sample_n_step() method with valid batch_size
def test_sample_n_step_with_valid_batch_size():
    action_dim = 4
    memory_size = 10000
    field_names = ["state", "action", "reward", "next_state", "done"]
    num_envs = 1
    n_step = 3
    gamma = 0.95
    device = "cpu"

    buffer = MultiStepReplayBuffer(
        action_dim, memory_size, field_names, num_envs, n_step, gamma, device
    )
    sampler = Sampler(
        distributed=False,
        per=False,
        n_step=True,
        memory=buffer,
        dataset=None,
        dataloader=None,
    )

    # Add experiences to memory
    buffer.save2memorySingleEnv(1, 2, 3, 2, 1)
    buffer.save2memorySingleEnv(4, 5, 6, 5, 4)
    buffer.save2memorySingleEnv(7, 8, 9, 8, 7)
    buffer.save2memorySingleEnv(9, 8, 7, 8, 9)

    idxs = [0, 1]
    samples = sampler.sample(idxs)

    assert len(samples) == len(field_names)
    assert len(samples[0]) == len(idxs)
    assert len(samples[1]) == len(idxs)
    assert len(samples[2]) == len(idxs)


@pytest.mark.parametrize(
    "distributed, per, n_step, memory, dataset, dataloader",
    [
        (False, False, False, 0, None, None),
        (True, False, False, None, 0, 0),
        (False, True, False, 0, None, None),
        (False, False, True, 0, None, None),
    ],
)
def test_warnings_in_constructor(distributed, per, n_step, memory, dataset, dataloader):
    with pytest.warns():
        _ = Sampler(
            distributed,
            per,
            n_step,
            memory,
            dataset,
            dataloader,
        )
