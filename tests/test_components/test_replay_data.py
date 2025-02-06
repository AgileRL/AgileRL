import numpy as np
import torch

from agilerl.components.replay_buffer import ReplayBuffer
from agilerl.components.replay_data import ReplayDataset


# The dataset can be initialized with a buffer and batch size
def test_initialization_with_buffer_and_batch_size():
    buffer = ReplayBuffer(
        memory_size=1000,
        field_names=["state", "action", "reward", "next_state", "done"],
    )
    batch_size = 2
    dataset = ReplayDataset(buffer, batch_size=batch_size)
    assert dataset.buffer == buffer
    assert dataset.batch_size == batch_size


# Sampling a batch of experiences from the buffer works correctly
def test_sampling_batch_from_buffer():
    field_names = ["state", "action", "reward", "next_state", "done"]
    buffer = ReplayBuffer(
        memory_size=1000,
        field_names=field_names,
    )

    state1 = np.array([1, 2, 3])
    action1 = np.array([0])
    reward1 = np.array([1])
    next_state1 = np.array([4, 5, 6])
    done1 = np.array([False])

    buffer.save_to_memory(state1, action1, reward1, next_state1, done1)
    buffer.save_to_memory(state1, action1, reward1, next_state1, done1)

    batch_size = 2
    dataset = ReplayDataset(buffer, batch_size=batch_size)
    iterator = iter(dataset)
    batch = next(iterator)

    assert len(batch) == len(field_names)
    assert len(batch[0]) == batch_size
    assert torch.equal(batch[0][0], torch.from_numpy(state1).float())
    assert torch.equal(batch[0][1], torch.from_numpy(state1).float())
    assert torch.equal(batch[1][0], torch.from_numpy(action1).float())
    assert torch.equal(batch[1][1], torch.from_numpy(action1).float())
    assert torch.equal(batch[2][0], torch.from_numpy(reward1).float())
    assert torch.equal(batch[2][1], torch.from_numpy(reward1).float())
    assert torch.equal(batch[3][0], torch.from_numpy(next_state1).float())
    assert torch.equal(batch[3][1], torch.from_numpy(next_state1).float())
    assert torch.equal(batch[4][0], torch.from_numpy(done1).float())
    assert torch.equal(batch[4][1], torch.from_numpy(done1).float())
