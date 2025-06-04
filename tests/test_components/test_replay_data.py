import numpy as np
import torch

from agilerl.components.data import ReplayDataset, Transition
from agilerl.components.replay_buffer import ReplayBuffer


# The dataset can be initialized with a buffer and batch size
def test_initialization_with_buffer_and_batch_size():
    buffer = ReplayBuffer(
        max_size=1000,
    )
    batch_size = 2
    dataset = ReplayDataset(buffer, batch_size=batch_size)
    assert dataset.buffer == buffer
    assert dataset.batch_size == batch_size


# Sampling a batch of experiences from the buffer works correctly
def test_sampling_batch_from_buffer():
    buffer = ReplayBuffer(
        max_size=1000,
    )

    state1 = np.array([1, 2, 3])
    action1 = np.array([0])
    reward1 = np.array([1])
    next_state1 = np.array([4, 5, 6])
    done1 = np.array([False])

    transition1 = Transition(
        obs=state1,
        action=action1,
        reward=reward1,
        next_obs=next_state1,
        done=done1,
    ).to_tensordict()

    transition1 = transition1.unsqueeze(0)
    transition1.batch_size = [1]
    buffer.add(transition1)
    buffer.add(transition1)

    batch_size = 2
    dataset = ReplayDataset(buffer, batch_size=batch_size)
    iterator = iter(dataset)
    batch = next(iterator)

    assert len(batch) == batch_size
    assert len(batch["obs"]) == batch_size
    assert torch.equal(batch["obs"][0], torch.from_numpy(state1).float())
    assert torch.equal(batch["obs"][1], torch.from_numpy(state1).float())
    assert torch.equal(batch["action"][0], torch.from_numpy(action1).float())
    assert torch.equal(batch["action"][1], torch.from_numpy(action1).float())
    assert torch.equal(batch["reward"][0], torch.from_numpy(reward1).float())
    assert torch.equal(batch["reward"][1], torch.from_numpy(reward1).float())
    assert torch.equal(batch["next_obs"][0], torch.from_numpy(next_state1).float())
    assert torch.equal(batch["next_obs"][1], torch.from_numpy(next_state1).float())
    assert torch.equal(batch["done"][0], torch.from_numpy(done1).float())
    assert torch.equal(batch["done"][1], torch.from_numpy(done1).float())
