import numpy as np
import pytest
import torch
from tensordict import TensorDict

from agilerl.components.data import MultiAgentTransition
from agilerl.components.replay_buffer import (
    MultiAgentReplayBuffer,
    MultiStepReplayBuffer,
    PrioritizedReplayBuffer,
    ReplayBuffer,
)
from agilerl.components.segment_tree import MinSegmentTree, SumSegmentTree


##### ReplayBuffer class tests #####
def test_create_instance_with_valid_arguments():
    """Test that a ReplayBuffer can be created with valid arguments."""
    max_size = 100
    device = "cpu"
    dtype = torch.float32

    buffer = ReplayBuffer(max_size, device, dtype)

    assert buffer.max_size == max_size
    assert buffer.device == device
    assert buffer.dtype == dtype
    assert buffer.counter == 0
    assert buffer.initialized is False
    assert buffer._cursor == 0
    assert buffer._size == 0
    assert buffer._storage is None


def test_get_length_of_memory():
    """Test that the length of the buffer can be retrieved with __len__ method."""
    max_size = 100
    buffer = ReplayBuffer(max_size)

    # Add experiences to memory
    data = TensorDict(
        {
            "state": torch.tensor([1, 2, 3]),
            "action": torch.tensor([0]),
            "reward": torch.tensor([1.0]),
        },
    )
    data = data.unsqueeze(0)
    data.batch_size = [1]
    buffer.add(data)

    data = TensorDict(
        {
            "state": torch.tensor([4, 5, 6]),
            "action": torch.tensor([1]),
            "reward": torch.tensor([2.0]),
        },
    )
    data = data.unsqueeze(0)
    data.batch_size = [1]
    buffer.add(data)

    assert len(buffer) == 2
    assert buffer.size == 2


def test_buffer_initialization():
    """Test that the buffer correctly initializes storage when adding first data."""
    buffer = ReplayBuffer(max_size=1000)

    # Create data
    data = TensorDict(
        {
            "state": torch.tensor([1, 2, 3]),
            "action": torch.tensor([0]),
            "reward": torch.tensor([1.0]),
        },
    )
    data = data.unsqueeze(0)
    data.batch_size = [1]

    # Add to buffer
    buffer.add(data)

    # Check initialization status
    assert buffer.initialized is True
    assert buffer._storage is not None
    assert len(buffer) == 1
    assert buffer.counter == 1


def test_add_experience_when_buffer_full():
    """Test that when buffer is full, old experiences are overwritten."""
    buffer = ReplayBuffer(max_size=2)

    # Add three experiences (more than buffer size)
    data1 = TensorDict(
        {
            "state": torch.tensor([1, 2, 3]),
            "action": torch.tensor([0]),
            "reward": torch.tensor([1.0]),
        },
    )
    data1 = data1.unsqueeze(0)
    data1.batch_size = [1]
    buffer.add(data1)

    data2 = TensorDict(
        {
            "state": torch.tensor([4, 5, 6]),
            "action": torch.tensor([1]),
            "reward": torch.tensor([2.0]),
        },
    )
    data2 = data2.unsqueeze(0)
    data2.batch_size = [1]
    buffer.add(data2)

    data3 = TensorDict(
        {
            "state": torch.tensor([7, 8, 9]),
            "action": torch.tensor([2]),
            "reward": torch.tensor([3.0]),
        },
    )
    data3 = data3.unsqueeze(0)
    data3.batch_size = [1]
    buffer.add(data3)

    # Check that buffer has max_size elements
    assert len(buffer) == 2
    assert buffer.is_full

    # Check that oldest experience was overwritten
    sample = buffer.sample(2)
    assert sample.keys() == data1.keys()

    # Either data2 and data3 should be in the buffer, data1 should be overwritten
    assert any(
        torch.all(sample["state"][i] == data2["state"])
        or torch.all(sample["state"][i] == data3["state"])
        for i in range(2)
    )


def test_add_experience_when_cursor_wraps():
    """Test add() wrap-around path when end > max_size."""
    buffer = ReplayBuffer(max_size=3)

    # Add 2 transitions: cursor=2, size=2
    data1 = TensorDict(
        {
            "state": torch.tensor([[1, 2, 3], [4, 5, 6]]),
            "action": torch.tensor([[0], [1]]),
            "reward": torch.tensor([[1.0], [2.0]]),
        },
    )
    data1.batch_size = [2]
    buffer.add(data1)
    assert len(buffer) == 2
    assert buffer._cursor == 2

    # Add 2 more: start=2, end=4 > 3, triggers wrap-around split
    data2 = TensorDict(
        {
            "state": torch.tensor([[7, 8, 9], [10, 11, 12]]),
            "action": torch.tensor([[2], [3]]),
            "reward": torch.tensor([[3.0], [4.0]]),
        },
    )
    data2.batch_size = [2]
    buffer.add(data2)
    assert len(buffer) == 3
    assert buffer._cursor == 1  # (2+2) % 3 = 1

    # Verify storage layout deterministically: [data2[1], data1[1], data2[0]]
    storage = buffer.storage[: buffer.size]
    expected = torch.tensor([[10.0, 11.0, 12.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    assert torch.allclose(storage["state"].float(), expected)


def test_add_with_1d_tensors_reshaped_to_batch_1():
    buffer = ReplayBuffer(max_size=10)

    # Batch of 2 with 1D reward/action (shape (2,) instead of (2,1))
    data = TensorDict(
        {
            "state": torch.tensor([[1, 2, 3], [4, 5, 6]]),
            "action": torch.tensor([0.0, 1.0]),
            "reward": torch.tensor([1.0, 2.0]),
        },
    )
    data.batch_size = [2]
    buffer.add(data)
    assert len(buffer) == 2
    sample = buffer.sample(2)
    assert sample["reward"].shape == (2, 1)
    assert sample["action"].shape == (2, 1)


def test_add_with_nested_tensordict_1d_values_reshaped():
    """Test add() reshapes 1D values inside nested TensorDict."""
    buffer = ReplayBuffer(max_size=10)

    # Nested TensorDict: inner "scalar" has shape (2,) -> reshaped to (2, 1)
    inner = TensorDict({"scalar": torch.tensor([1.0, 2.0])}, batch_size=[2])
    data = TensorDict(
        {
            "state": inner,
            "action": torch.tensor([[0], [1]]),
            "reward": torch.tensor([[1.0], [2.0]]),
        },
    )
    data.batch_size = [2]
    buffer.add(data)
    assert len(buffer) == 2
    sample = buffer.sample(2)
    assert sample["state"]["scalar"].shape == (2, 1)


def test_add_vectorized_experiences():
    """Test adding vectorized experiences."""
    buffer = ReplayBuffer(max_size=10)

    # Create batched data
    batch_data = TensorDict(
        {
            "state": torch.tensor([[1, 2, 3], [4, 5, 6]]),
            "action": torch.tensor([[0], [1]]),
            "reward": torch.tensor([[1.0], [2.0]]),
        },
    )

    # Add batch to buffer
    batch_data.batch_size = [2]
    buffer.add(batch_data)

    # Check buffer state
    assert len(buffer) == 2
    assert buffer.counter == 2


def test_sample_experiences():
    """Test sampling experiences from the buffer."""
    buffer = ReplayBuffer(max_size=100)

    # Add multiple experiences
    for i in range(10):
        data = TensorDict(
            {
                "state": torch.tensor([i, i + 1, i + 2]),
                "action": torch.tensor([i % 3]),
                "reward": torch.tensor([float(i)]),
            },
        )
        data = data.unsqueeze(0)
        data.batch_size = [1]
        buffer.add(data)

    # Sample experiences
    batch_size = 5
    samples = buffer.sample(batch_size)

    # Verify sample structure
    assert isinstance(samples, TensorDict)
    assert samples.batch_size == torch.Size([batch_size])
    assert "state" in samples
    assert "action" in samples
    assert "reward" in samples
    assert samples["state"].shape == (batch_size, 3)
    assert samples["action"].shape == (batch_size, 1)
    assert samples["reward"].shape == (batch_size, 1)


def test_sample_experiences_with_images():
    """Test sampling experiences with image observations."""
    buffer = ReplayBuffer(max_size=100)

    # Add multiple experiences with image observations
    for i in range(10):
        # Create image-like tensor (C, H, W)
        img = torch.ones((3, 84, 84)) * i
        data = TensorDict(
            {
                "state": img,
                "action": torch.tensor([i % 3]),
                "reward": torch.tensor([float(i)]),
            },
        )
        data = data.unsqueeze(0)
        data.batch_size = [1]
        buffer.add(data)

    # Sample experiences
    batch_size = 5
    samples = buffer.sample(batch_size)

    # Verify sample structure for image observations
    assert isinstance(samples, TensorDict)
    assert samples.batch_size == torch.Size([batch_size])
    assert "state" in samples
    assert samples["state"].shape == (batch_size, 3, 84, 84)
    assert samples["action"].shape == (batch_size, 1)
    assert samples["reward"].shape == (batch_size, 1)


def test_sample_experiences_with_dict_obs():
    """Test sampling experiences with dictionary observations."""
    buffer = ReplayBuffer(max_size=100)

    # Add multiple experiences with dictionary observations
    for i in range(10):
        # Create dictionary observation
        obs = TensorDict(
            {
                "image": torch.ones((3, 84, 84)) * i,
                "vector": torch.tensor([i, i + 1, i + 2]),
            },
        )
        data = TensorDict(
            {
                "state": obs,
                "action": torch.tensor([i % 3]),
                "reward": torch.tensor([float(i)]),
            },
        )
        data = data.unsqueeze(0)
        data.batch_size = [1]
        buffer.add(data)

    # Sample experiences
    batch_size = 5
    samples = buffer.sample(batch_size)

    # Verify sample structure for dictionary observations
    assert isinstance(samples, TensorDict)
    assert samples.batch_size == torch.Size([batch_size])
    assert "state" in samples
    assert isinstance(samples["state"], TensorDict)
    assert "image" in samples["state"]
    assert "vector" in samples["state"]
    assert samples["state"]["image"].shape == (batch_size, 3, 84, 84)
    assert samples["state"]["vector"].shape == (batch_size, 3)
    assert samples["action"].shape == (batch_size, 1)
    assert samples["reward"].shape == (batch_size, 1)


def test_sample_with_indices():
    """Test sampling experiences with indices returned."""
    buffer = ReplayBuffer(max_size=100)

    # Add multiple experiences
    for i in range(10):
        data = TensorDict(
            {
                "state": torch.tensor([i, i + 1, i + 2]),
                "action": torch.tensor([i % 3]),
                "reward": torch.tensor([float(i)]),
            },
        )
        data = data.unsqueeze(0)
        data.batch_size = [1]
        buffer.add(data)

    # Sample experiences with indices
    batch_size = 5
    samples = buffer.sample(batch_size, return_idx=True)

    # Verify indices are returned
    assert "idxs" in samples
    assert samples["idxs"].shape == (batch_size,)
    assert all(0 <= idx < len(buffer) for idx in samples["idxs"])


def test_clear_buffer():
    """Test clearing the buffer."""
    buffer = ReplayBuffer(max_size=100)

    # Add experiences
    data = TensorDict(
        {
            "state": torch.tensor([1, 2, 3]),
            "action": torch.tensor([0]),
            "reward": torch.tensor([1.0]),
        },
    )
    data = data.unsqueeze(0)
    data.batch_size = [1]
    buffer.add(data)

    # Clear buffer
    buffer.clear()

    # Check buffer state
    assert len(buffer) == 0
    assert buffer._size == 0
    assert buffer._cursor == 0
    assert buffer._storage is None
    assert not buffer.initialized


##### MultiStepReplayBuffer class tests #####
def test_nstep_buffer_initialization():
    """Test initialization of MultiStepReplayBuffer."""
    max_size = 1000
    n_step = 3
    gamma = 0.99

    buffer = MultiStepReplayBuffer(max_size, n_step, gamma)

    assert buffer.max_size == max_size
    assert buffer.n_step == n_step
    assert buffer.gamma == gamma
    assert len(buffer.n_step_buffer) == 0


def test_nstep_buffer_add():
    """Test adding transitions to MultiStepReplayBuffer."""
    buffer = MultiStepReplayBuffer(max_size=1000, n_step=3, gamma=0.99)

    # Define transition fields required for n-step returns
    buffer.reward_key = "reward"
    buffer.done_key = "done"
    buffer.ns_key = "next_state"

    # Add a transition
    data = TensorDict(
        {
            "state": torch.tensor([1, 2, 3]),
            "action": torch.tensor([0]),
            "reward": torch.tensor([1.0]),
            "next_state": torch.tensor([4, 5, 6]),
            "done": torch.tensor([False]),
        },
    )
    data = data.unsqueeze(0)
    data.batch_size = [1]
    buffer.add(data)

    # Check that transition is in n-step buffer but not in main buffer yet
    assert len(buffer.n_step_buffer) == 1
    assert len(buffer) == 0

    # Add more transitions to complete n-step return
    for i in range(2):
        data = TensorDict(
            {
                "state": torch.tensor([i + 4, i + 5, i + 6]),
                "action": torch.tensor([i + 1]),
                "reward": torch.tensor([float(i + 2)]),
                "next_state": torch.tensor([i + 7, i + 8, i + 9]),
                "done": torch.tensor([False]),
            },
        )
        data = data.unsqueeze(0)
        data.batch_size = [1]
        buffer.add(data)

    # Now a transition should be in the main buffer
    assert len(buffer) == 1
    assert len(buffer.n_step_buffer) == 3


def test_nstep_buffer_add_with_images():
    """Test adding image transitions to MultiStepReplayBuffer."""
    buffer = MultiStepReplayBuffer(max_size=1000, n_step=3, gamma=0.99)

    # Define transition fields required for n-step returns
    buffer.reward_key = "reward"
    buffer.done_key = "done"
    buffer.ns_key = "next_state"

    # Add transitions with image observations
    for i in range(4):  # Add more than n_step
        # Create image-like tensor (C, H, W)
        img = torch.ones((3, 84, 84)) * i
        next_img = torch.ones((3, 84, 84)) * (i + 1)

        data = TensorDict(
            {
                "state": img,
                "action": torch.tensor([i % 3]),
                "reward": torch.tensor([float(i)]),
                "next_state": next_img,
                "done": torch.tensor([False]),
            },
        )
        data = data.unsqueeze(0)
        data.batch_size = [1]
        buffer.add(data)

    # Should have processed at least one n-step transition
    assert len(buffer) >= 1

    # Sample and verify image dimensions
    if len(buffer) > 0:
        samples = buffer.sample(1)
        assert samples["state"].shape[1:] == (3, 84, 84)


def test_nstep_reward_calculation():
    """Test n-step reward calculation."""
    buffer = MultiStepReplayBuffer(max_size=1000, n_step=3, gamma=0.9)

    # Define required keys
    buffer.reward_key = "reward"
    buffer.done_key = "done"
    buffer.ns_key = "next_state"

    # Initialize buffer storage to avoid None reference
    buffer._storage = TensorDict({}, batch_size=[0, 0])

    # Fill n-step buffer with transitions
    for i in range(3):
        data = TensorDict(
            {
                "state": torch.tensor([i, i + 1, i + 2]),
                "action": torch.tensor([i]),
                "reward": torch.tensor([float(i + 1)]),
                "next_state": torch.tensor([i + 3, i + 4, i + 5]),
                "done": torch.tensor([False]),
            },
        )
        data = data.unsqueeze(0)
        data.batch_size = [1]
        buffer.n_step_buffer.append(data)

    # Calculate n-step return
    n_step_data = buffer._get_n_step_info()

    # Expected reward: r1 + gamma*r2 + gamma^2*r3
    expected_reward = 1.0 + 0.9 * 2.0 + 0.9 * 0.9 * 3.0

    assert torch.isclose(n_step_data["reward"], torch.tensor(expected_reward))
    assert torch.all(n_step_data["next_state"] == buffer.n_step_buffer[2]["next_state"])
    assert torch.all(n_step_data["done"] == buffer.n_step_buffer[2]["done"])


def test_nstep_early_termination():
    """Test n-step calculation with early termination."""
    buffer = MultiStepReplayBuffer(max_size=1000, n_step=3, gamma=0.9)

    # Define required keys
    buffer.reward_key = "reward"
    buffer.done_key = "done"
    buffer.ns_key = "next_state"

    # Fill n-step buffer with transitions where second one terminates
    data1 = TensorDict(
        {
            "state": torch.tensor([1, 2, 3]),
            "action": torch.tensor([0]),
            "reward": torch.tensor([1.0]),
            "next_state": torch.tensor([4, 5, 6]),
            "done": torch.tensor([False]),
        },
    )
    data1 = data1.unsqueeze(0)
    data1.batch_size = [1]
    buffer.n_step_buffer.append(data1)

    data2 = TensorDict(
        {
            "state": torch.tensor([4, 5, 6]),
            "action": torch.tensor([1]),
            "reward": torch.tensor([2.0]),
            "next_state": torch.tensor([7, 8, 9]),
            "done": torch.tensor([True]),
        },
    )
    data2 = data2.unsqueeze(0)
    data2.batch_size = [1]
    buffer.n_step_buffer.append(data2)

    data3 = TensorDict(
        {
            "state": torch.tensor([7, 8, 9]),
            "action": torch.tensor([2]),
            "reward": torch.tensor([3.0]),
            "next_state": torch.tensor([10, 11, 12]),
            "done": torch.tensor([False]),
        },
    )
    data3 = data3.unsqueeze(0)
    data3.batch_size = [1]
    buffer.n_step_buffer.append(data3)

    # Calculate n-step return
    n_step_data = buffer._get_n_step_info()

    # Expected reward: r1 + gamma*r2 (no r3 because episode terminates)
    expected_reward = 1.0 + 0.9 * 2.0

    assert torch.isclose(n_step_data["reward"], torch.tensor(expected_reward))
    assert torch.all(n_step_data["next_state"] == data2["next_state"])
    assert torch.all(n_step_data["done"] == data2["done"])


def test_nstep_sampling_with_indices():
    """Test sampling from MultiStepReplayBuffer with indices."""
    buffer = MultiStepReplayBuffer(max_size=1000, n_step=3, gamma=0.99)

    # Define transition fields required for n-step returns
    buffer.reward_key = "reward"
    buffer.done_key = "done"
    buffer.ns_key = "next_state"

    # Add transitions to fill buffer
    for i in range(10):
        data = TensorDict(
            {
                "state": torch.tensor([i, i + 1, i + 2]),
                "action": torch.tensor([i % 3]),
                "reward": torch.tensor([float(i)]),
                "next_state": torch.tensor([i + 3, i + 4, i + 5]),
                "done": torch.tensor([i % 5 == 0]),  # Some episodes terminate
            },
        )
        data = data.unsqueeze(0)
        data.batch_size = [1]
        buffer.add(data)

    # Skip test if buffer is empty
    if len(buffer) == 0:
        pytest.skip("Buffer is empty, cannot test sampling")

    # Sample with indices
    samples = buffer.sample(min(3, len(buffer)), return_idx=True)

    # Verify indices are returned
    assert "idxs" in samples
    assert len(samples["idxs"]) == min(3, len(buffer))
    assert all(0 <= idx < len(buffer) for idx in samples["idxs"])


##### PrioritizedReplayBuffer class tests #####
def test_prioritized_buffer_initialization():
    """Test initialization of PrioritizedReplayBuffer."""
    max_size = 1000
    alpha = 0.6

    buffer = PrioritizedReplayBuffer(max_size, alpha)

    assert buffer.max_size == max_size
    assert buffer.alpha == alpha
    assert buffer.max_priority == 1.0
    assert buffer.tree_ptr == 0
    assert isinstance(buffer.sum_tree, SumSegmentTree)
    assert isinstance(buffer.min_tree, MinSegmentTree)


def test_prioritized_buffer_add():
    """Test adding experience to PrioritizedReplayBuffer."""
    buffer = PrioritizedReplayBuffer(max_size=100, alpha=0.6)

    # Add a transition
    data = TensorDict(
        {
            "state": torch.tensor([1, 2, 3]),
            "action": torch.tensor([0]),
            "reward": torch.tensor([1.0]),
        },
    )
    data = data.unsqueeze(0)
    data.batch_size = [1]
    buffer.add(data)

    assert len(buffer) == 1
    assert buffer.tree_ptr == 1
    assert buffer.sum_tree[0] == buffer.max_priority**buffer.alpha
    assert buffer.min_tree[0] == buffer.max_priority**buffer.alpha


def test_prioritized_buffer_add_batch():
    """Test adding batch of experiences to PrioritizedReplayBuffer."""
    buffer = PrioritizedReplayBuffer(max_size=100, alpha=0.6)

    # Add batch
    batch_data = TensorDict(
        {
            "state": torch.tensor([[1, 2, 3], [4, 5, 6]]),
            "action": torch.tensor([[0], [1]]),
            "reward": torch.tensor([[1.0], [2.0]]),
        },
    )
    batch_data.batch_size = [2]
    buffer.add(batch_data)

    assert len(buffer) == 2
    assert buffer.tree_ptr == 2
    assert buffer.sum_tree[0] == buffer.max_priority**buffer.alpha
    assert buffer.sum_tree[1] == buffer.max_priority**buffer.alpha


def test_prioritized_buffer_with_image_observations():
    """Test PrioritizedReplayBuffer with image observations."""
    buffer = PrioritizedReplayBuffer(max_size=100, alpha=0.6)

    # Add experiences with image observations
    for i in range(5):
        # Create image-like tensor (C, H, W)
        img = torch.ones((3, 84, 84)) * i

        data = TensorDict(
            {
                "state": img,
                "action": torch.tensor([i % 3]),
                "reward": torch.tensor([float(i)]),
            },
        )
        data = data.unsqueeze(0)
        data.batch_size = [1]
        buffer.add(data)

    # Sample experiences
    batch_size = 3
    beta = 0.4
    samples = buffer.sample(batch_size, beta)

    # Verify sample structure for image observations
    assert isinstance(samples, TensorDict)
    assert "state" in samples
    assert samples["state"].shape == (batch_size, 3, 84, 84)
    assert "weights" in samples
    assert "idxs" in samples


def test_prioritized_buffer_with_dict_observations():
    """Test PrioritizedReplayBuffer with dictionary observations."""
    buffer = PrioritizedReplayBuffer(max_size=100, alpha=0.6)

    # Add experiences with dictionary observations
    for i in range(5):
        # Create dictionary observation
        obs = TensorDict(
            {
                "image": torch.ones((3, 84, 84)) * i,
                "vector": torch.tensor([i, i + 1, i + 2]),
            },
        )
        data = TensorDict(
            {
                "state": obs,
                "action": torch.tensor([i % 3]),
                "reward": torch.tensor([float(i)]),
            },
        )
        data = data.unsqueeze(0)
        data.batch_size = [1]
        buffer.add(data)

    # Sample experiences
    batch_size = 3
    beta = 0.4
    samples = buffer.sample(batch_size, beta)

    # Verify sample structure for dictionary observations
    assert isinstance(samples, TensorDict)
    assert "state" in samples
    assert isinstance(samples["state"], TensorDict)
    assert "image" in samples["state"]
    assert "vector" in samples["state"]
    assert samples["state"]["image"].shape == (batch_size, 3, 84, 84)
    assert samples["state"]["vector"].shape == (batch_size, 3)
    assert "weights" in samples
    assert "idxs" in samples


def test_prioritized_sampling():
    """Test sampling from PrioritizedReplayBuffer with priority."""
    buffer = PrioritizedReplayBuffer(max_size=100, alpha=0.6)

    # Add multiple experiences
    for i in range(10):
        data = TensorDict(
            {
                "state": torch.tensor([i, i + 1, i + 2]),
                "action": torch.tensor([i % 3]),
                "reward": torch.tensor([float(i)]),
            },
        )
        data = data.unsqueeze(0)
        data.batch_size = [1]
        buffer.add(data)

    # Sample experiences
    batch_size = 5
    beta = 0.4
    samples = buffer.sample(batch_size, beta)

    # Verify sample structure
    assert isinstance(samples, TensorDict)
    assert "state" in samples
    assert "action" in samples
    assert "reward" in samples
    assert "weights" in samples
    assert "idxs" in samples
    assert samples.batch_size == torch.Size([batch_size])
    assert samples["weights"].shape == (batch_size, 1)
    assert samples["idxs"].shape == (batch_size, 1)


def test_prioritized_buffer_update_priorities():
    """Test updating priorities in PrioritizedReplayBuffer."""
    buffer = PrioritizedReplayBuffer(max_size=100, alpha=0.6)

    # Add experiences
    for i in range(5):
        data = TensorDict(
            {
                "state": torch.tensor([i, i + 1, i + 2]),
                "action": torch.tensor([i % 3]),
                "reward": torch.tensor([float(i)]),
            },
        )
        data = data.unsqueeze(0)
        data.batch_size = [1]
        buffer.add(data)

    # 'Sample' experiences
    batch_size = 3
    indices = np.arange(batch_size)

    # Update priorities
    new_priorities = torch.tensor([2.0, 3.0, 4.0])
    buffer.update_priorities(indices, new_priorities)

    # Verify priorities were updated
    for idx, priority in zip(indices, new_priorities, strict=False):
        assert buffer.sum_tree[idx.item()] == priority.item() ** buffer.alpha, (
            priority.item() ** buffer.alpha
        )
        assert buffer.min_tree[idx.item()] == priority.item() ** buffer.alpha, (
            priority.item() ** buffer.alpha
        )

    # Max priority should be updated
    assert buffer.max_priority == 4.0


def test_proportional_sampling():
    """Test proportional sampling based on priorities."""
    buffer = PrioritizedReplayBuffer(max_size=100, alpha=0.6)

    # Add experiences with different priorities
    for i in range(5):
        data = TensorDict(
            {
                "state": torch.tensor([i, i + 1, i + 2]),
                "action": torch.tensor([i % 3]),
                "reward": torch.tensor([float(i)]),
            },
        )
        data = data.unsqueeze(0)
        data.batch_size = [1]
        buffer.add(data)

        # Update priorities to make them different
        buffer._update_priority(i, float(i + 1))

    # Mock random sampling to test proportional selection
    original_rand = torch.rand
    try:
        # Use deterministic values for testing
        predetermined_values = [0.1, 0.3, 0.6, 0.8]
        counter = 0

        def mock_rand(size):
            nonlocal counter
            val = predetermined_values[counter % len(predetermined_values)]
            counter += 1
            return torch.tensor([val])

        torch.rand = mock_rand

        # Sample indices
        batch_size = 4
        indices = buffer._sample_proportional(batch_size)

        # Verify indices are within range
        assert all(0 <= idx < len(buffer) for idx in indices)

    finally:
        # Restore original function
        torch.rand = original_rand


def test_weight_calculation():
    """Test weight calculation for importance sampling."""
    buffer = PrioritizedReplayBuffer(max_size=100, alpha=0.6)

    # Add experiences
    for i in range(5):
        data = TensorDict(
            {
                "state": torch.tensor([i, i + 1, i + 2]),
                "action": torch.tensor([i % 3]),
                "reward": torch.tensor([float(i)]),
            },
        )
        data = data.unsqueeze(0)
        data.batch_size = [1]
        buffer.add(data)

    # Update priorities to known values
    priorities = [0.5, 1.0, 1.5, 2.0, 2.5]
    for i, priority in enumerate(priorities):
        buffer._update_priority(i, priority)

    # Test weight calculation with beta=0.4
    beta = 0.4
    for i, priority in enumerate(priorities):
        weight = buffer._calculate_weights(torch.tensor([i]), beta)[0]

        # Calculate expected weight:
        # w = (p * N)^(-beta) / max_weight where max_weight = (p_min * N)^(-beta)
        p_sample = (priority**buffer.alpha) / buffer.sum_tree.sum()
        p_min = buffer.min_tree.min() / buffer.sum_tree.sum()
        expected_weight = ((p_sample * len(buffer)) ** -beta) / (
            (p_min * len(buffer)) ** -beta
        )

        assert torch.isclose(weight, torch.tensor(expected_weight), rtol=1e-5)


def test_replay_buffer_size_setter():
    buffer = ReplayBuffer(max_size=10)
    data = TensorDict(
        {
            "state": torch.tensor([1]),
            "action": torch.tensor([0]),
            "reward": torch.tensor([1.0]),
        },
    )
    data = data.unsqueeze(0)
    data.batch_size = [1]
    buffer.add(data)
    buffer.size = 5
    assert buffer._size == 5


def test_replay_buffer_storage_property_after_init():
    buffer = ReplayBuffer(max_size=10)
    data = TensorDict(
        {
            "state": torch.tensor([1]),
            "action": torch.tensor([0]),
            "reward": torch.tensor([1.0]),
        },
    )
    data = data.unsqueeze(0)
    data.batch_size = [1]
    buffer.add(data)
    assert buffer.storage is not None
    assert buffer.storage.batch_size[0] == 10


def test_prioritized_replay_buffer_update_priority_assert():
    buffer = PrioritizedReplayBuffer(max_size=10, alpha=0.6)
    data = TensorDict(
        {
            "state": torch.tensor([1]),
            "action": torch.tensor([0]),
            "reward": torch.tensor([1.0]),
        },
    )
    data = data.unsqueeze(0)
    data.batch_size = [1]
    buffer.add(data)
    with pytest.raises(AssertionError):
        buffer._update_priority(100, 1.0)


def test_multistep_buffer_sample_from_indices():
    buffer = MultiStepReplayBuffer(max_size=100, n_step=3, gamma=0.99)
    buffer.reward_key = "reward"
    buffer.done_key = "done"
    buffer.ns_key = "next_state"
    for i in range(6):
        data = TensorDict(
            {
                "state": torch.tensor([i]),
                "action": torch.tensor([i % 2]),
                "reward": torch.tensor([float(i)]),
                "next_state": torch.tensor([i + 1]),
                "done": torch.tensor([i == 5]),
            },
        )
        data = data.unsqueeze(0)
        data.batch_size = [1]
        buffer.add(data)
    if len(buffer) > 0:
        idxs = torch.tensor([0])
        samples = buffer.sample_from_indices(idxs)
        assert "state" in samples
        assert samples.batch_size[0] == 1


def test_multistep_buffer_get_n_step_info_termination_key():
    buffer = MultiStepReplayBuffer(max_size=100, n_step=3, gamma=0.99)
    buffer.reward_key = "reward"
    buffer.ns_key = "next_obs"
    for i in range(3):
        data = TensorDict(
            {
                "state": torch.tensor([i]),
                "action": torch.tensor([i]),
                "reward": torch.tensor([float(i + 1)]),
                "next_obs": torch.tensor([i + 1]),
                "termination": torch.tensor([False]),
            },
        )
        data = data.unsqueeze(0)
        data.batch_size = [1]
        buffer.add(data)
    assert buffer.done_key == "termination"


def test_multistep_buffer_get_n_step_info_terminated_key():
    buffer = MultiStepReplayBuffer(max_size=100, n_step=3, gamma=0.99)
    buffer.reward_key = "reward"
    buffer.ns_key = "next_obs"
    for i in range(3):
        data = TensorDict(
            {
                "state": torch.tensor([i]),
                "action": torch.tensor([i]),
                "reward": torch.tensor([float(i + 1)]),
                "next_obs": torch.tensor([i + 1]),
                "terminated": torch.tensor([False]),
            },
        )
        data = data.unsqueeze(0)
        data.batch_size = [1]
        buffer.add(data)
    assert buffer.done_key == "terminated"


def test_prioritized_buffer_update_priorities_small_priority_clamped():
    buffer = PrioritizedReplayBuffer(max_size=10, alpha=0.6)
    for i in range(3):
        data = TensorDict(
            {
                "state": torch.tensor([i]),
                "action": torch.tensor([0]),
                "reward": torch.tensor([1.0]),
            },
        )
        data = data.unsqueeze(0)
        data.batch_size = [1]
        buffer.add(data)
    buffer.update_priorities(torch.tensor([0, 1]), torch.tensor([1e-10, 1e-10]))
    assert buffer.sum_tree[0] >= 1e-5**buffer.alpha


# =====================================================================
# MultiAgentReplayBuffer tests
# =====================================================================

MA_AGENTS = ["agent_0", "agent_1"]


def _make_ma_td(
    agent_ids: list[str],
    batch_size: int = 1,
    obs_size: int = 3,
    act_size: int = 1,
) -> TensorDict:
    """Nested TensorDict: field -> agent_id -> Tensor."""
    bs = [batch_size]
    return TensorDict(
        {
            "obs": TensorDict(
                {a: torch.randn(*bs, obs_size) for a in agent_ids},
                batch_size=bs,
            ),
            "action": TensorDict(
                {a: torch.randn(*bs, act_size) for a in agent_ids},
                batch_size=bs,
            ),
            "reward": TensorDict(
                {a: torch.rand(*bs, 1) for a in agent_ids},
                batch_size=bs,
            ),
            "next_obs": TensorDict(
                {a: torch.randn(*bs, obs_size) for a in agent_ids},
                batch_size=bs,
            ),
            "done": TensorDict(
                {a: torch.zeros(*bs, 1) for a in agent_ids},
                batch_size=bs,
            ),
        },
        batch_size=bs,
    )


def _make_deterministic_ma_td(
    agent_ids: list[str],
    batch_size: int,
    offset: float = 0.0,
    obs_size: int = 2,
) -> TensorDict:
    """Deterministic nested TensorDict for round-trip checks."""
    bs = [batch_size]
    return TensorDict(
        {
            "obs": TensorDict(
                {
                    a: torch.arange(batch_size, dtype=torch.float32)
                    .unsqueeze(1)
                    .expand(-1, obs_size)
                    + offset
                    + i
                    for i, a in enumerate(agent_ids)
                },
                batch_size=bs,
            ),
            "reward": TensorDict(
                {
                    a: (
                        torch.arange(batch_size, dtype=torch.float32) + offset + i * 100
                    ).unsqueeze(1)
                    for i, a in enumerate(agent_ids)
                },
                batch_size=bs,
            ),
        },
        batch_size=bs,
    )


##### MultiAgentReplayBuffer — Initialisation #####


class TestMAInit:
    def test_empty_defaults(self):
        buf = MultiAgentReplayBuffer(100)
        assert len(buf) == 0
        assert buf.size == 0
        assert buf.max_size == 100
        assert buf.counter == 0
        assert not buf.is_full
        assert not buf.initialized
        assert buf.storage is None

    def test_size_setter(self):
        buf = MultiAgentReplayBuffer(50)
        buf.size = 7
        assert buf.size == 7
        assert len(buf) == 7


##### MultiAgentReplayBuffer — add() #####


class TestMAAdd:
    def test_single_transition(self):
        buf = MultiAgentReplayBuffer(100)
        buf.add(_make_ma_td(MA_AGENTS, batch_size=1))
        assert len(buf) == 1
        assert buf.counter == 1
        assert buf.initialized
        assert buf.storage is not None

    def test_batch_add(self):
        buf = MultiAgentReplayBuffer(100)
        buf.add(_make_ma_td(MA_AGENTS, batch_size=8))
        assert len(buf) == 8
        assert buf.counter == 8

    def test_multiple_sequential_adds(self):
        buf = MultiAgentReplayBuffer(100)
        for _ in range(5):
            buf.add(_make_ma_td(MA_AGENTS, batch_size=3))
        assert len(buf) == 15
        assert buf.counter == 15

    def test_storage_batch_dim_equals_max_size(self):
        buf = MultiAgentReplayBuffer(20)
        buf.add(_make_ma_td(MA_AGENTS, batch_size=1))
        assert buf.storage.shape[0] == 20


##### MultiAgentReplayBuffer — _normalize_dims (key differentiator) #####


class TestMANormalizeDims:
    def test_level1_flat_scalar(self):
        """Top-level 1-D tensor is reshaped to (batch, 1)."""
        bs = 4
        td = TensorDict({"global_reward": torch.rand(bs)}, batch_size=[bs])
        buf = MultiAgentReplayBuffer(50)
        buf.add(td)
        assert buf.storage["global_reward"].shape == (50, 1)

    def test_level2_agent_scalar(self):
        """field -> agent_id with 1-D tensors are reshaped."""
        bs = 4
        td = TensorDict(
            {
                "reward": TensorDict(
                    {a: torch.rand(bs) for a in MA_AGENTS},
                    batch_size=[bs],
                ),
            },
            batch_size=[bs],
        )
        buf = MultiAgentReplayBuffer(50)
        buf.add(td)
        for a in MA_AGENTS:
            assert buf.storage["reward", a].shape == (50, 1)

    def test_level3_nested_sub_td_scalar(self):
        """field -> agent_id -> sub_td -> 1-D tensor."""
        bs = 3
        td = TensorDict(
            {
                "info": TensorDict(
                    {
                        "agent_0": TensorDict(
                            {"health": torch.rand(bs)},
                            batch_size=[bs],
                        ),
                    },
                    batch_size=[bs],
                ),
            },
            batch_size=[bs],
        )
        buf = MultiAgentReplayBuffer(50)
        buf.add(td)
        assert buf.storage["info", "agent_0", "health"].shape == (50, 1)

    def test_already_2d_not_changed(self):
        bs, obs_size = 4, 5
        td = TensorDict(
            {
                "state": TensorDict(
                    {a: torch.randn(bs, obs_size) for a in MA_AGENTS},
                    batch_size=[bs],
                ),
            },
            batch_size=[bs],
        )
        buf = MultiAgentReplayBuffer(50)
        buf.add(td)
        for a in MA_AGENTS:
            assert buf.storage["state", a].shape == (50, obs_size)

    def test_high_dim_untouched(self):
        """3-D+ tensors (images) are never reshaped."""
        bs, img = 2, (3, 64, 64)
        td = TensorDict(
            {
                "obs": TensorDict(
                    {a: torch.randn(bs, *img) for a in MA_AGENTS},
                    batch_size=[bs],
                ),
            },
            batch_size=[bs],
        )
        buf = MultiAgentReplayBuffer(50)
        buf.add(td)
        for a in MA_AGENTS:
            assert buf.storage["obs", a].shape == (50, *img)

    def test_mixed_scalar_and_vector_leaves(self):
        bs = 4
        td = TensorDict(
            {
                "reward": TensorDict(
                    {a: torch.rand(bs) for a in MA_AGENTS},
                    batch_size=[bs],
                ),
                "state": TensorDict(
                    {a: torch.randn(bs, 6) for a in MA_AGENTS},
                    batch_size=[bs],
                ),
            },
            batch_size=[bs],
        )
        buf = MultiAgentReplayBuffer(50)
        buf.add(td)
        for a in MA_AGENTS:
            assert buf.storage["reward", a].shape == (50, 1)
            assert buf.storage["state", a].shape == (50, 6)

    def test_done_flag_scalar_gets_unsqueezed(self):
        """Binary done flags passed as 1-D tensors are correctly reshaped."""
        bs = 3
        td = TensorDict(
            {
                "done": TensorDict(
                    {a: torch.tensor([True, False, True]) for a in MA_AGENTS},
                    batch_size=[bs],
                ),
            },
            batch_size=[bs],
        )
        buf = MultiAgentReplayBuffer(50)
        buf.add(td)
        for a in MA_AGENTS:
            assert buf.storage["done", a].shape == (50, 1)


##### MultiAgentReplayBuffer — Circular overwrite #####


class TestMACircular:
    def test_overwrites_oldest(self):
        buf = MultiAgentReplayBuffer(4)
        for _ in range(6):
            buf.add(_make_ma_td(MA_AGENTS, batch_size=1))
        assert len(buf) == 4
        assert buf.counter == 6
        assert buf.is_full

    def test_wrap_around_single_batch(self):
        buf = MultiAgentReplayBuffer(5)
        buf.add(_make_ma_td(MA_AGENTS, batch_size=3))
        buf.add(_make_ma_td(MA_AGENTS, batch_size=4))
        assert len(buf) == 5
        assert buf.is_full
        assert buf.counter == 7

    def test_exact_fill(self):
        buf = MultiAgentReplayBuffer(4)
        buf.add(_make_ma_td(MA_AGENTS, batch_size=4))
        assert len(buf) == 4
        assert buf.is_full

    def test_storage_shape_constant_after_overflow(self):
        buf = MultiAgentReplayBuffer(10)
        for _ in range(20):
            buf.add(_make_ma_td(MA_AGENTS, batch_size=3))
        assert buf.storage.shape[0] == 10
        assert len(buf) == 10

    def test_overwritten_values_are_newest(self):
        buf = MultiAgentReplayBuffer(3)
        old = _make_deterministic_ma_td(MA_AGENTS, 3, offset=0.0)
        new = _make_deterministic_ma_td(MA_AGENTS, 3, offset=100.0)
        buf.add(old)
        buf.add(new)
        sampled = buf.sample(3)
        for a in MA_AGENTS:
            for row in sampled["reward", a]:
                assert row.item() >= 100.0

    def test_cursor_position_after_wrap(self):
        buf = MultiAgentReplayBuffer(5)
        buf.add(_make_ma_td(MA_AGENTS, batch_size=3))
        assert buf._cursor == 3
        buf.add(_make_ma_td(MA_AGENTS, batch_size=4))
        assert buf._cursor == 2  # (3+4) % 5


##### MultiAgentReplayBuffer — sample() #####


class TestMASample:
    def test_returns_tensordict(self):
        buf = MultiAgentReplayBuffer(100)
        buf.add(_make_ma_td(MA_AGENTS, batch_size=20))
        s = buf.sample(8)
        assert isinstance(s, TensorDict)
        assert s.shape[0] == 8

    def test_nested_structure_preserved(self):
        buf = MultiAgentReplayBuffer(100)
        buf.add(_make_ma_td(MA_AGENTS, batch_size=10))
        s = buf.sample(4)
        for field in ("obs", "action", "reward", "next_obs", "done"):
            assert field in s.keys()
            sub = s[field]
            assert isinstance(sub, TensorDict)
            for aid in MA_AGENTS:
                assert aid in sub.keys()
                assert isinstance(sub[aid], torch.Tensor)

    def test_sample_shapes(self):
        obs_size, act_size = 5, 2
        buf = MultiAgentReplayBuffer(100)
        buf.add(
            _make_ma_td(MA_AGENTS, batch_size=20, obs_size=obs_size, act_size=act_size)
        )
        s = buf.sample(8)
        for aid in MA_AGENTS:
            assert s["obs", aid].shape == (8, obs_size)
            assert s["action", aid].shape == (8, act_size)
            assert s["reward", aid].shape == (8, 1)
            assert s["next_obs", aid].shape == (8, obs_size)
            assert s["done", aid].shape == (8, 1)

    def test_return_idx(self):
        buf = MultiAgentReplayBuffer(100)
        buf.add(_make_ma_td(MA_AGENTS, batch_size=20))
        s = buf.sample(5, return_idx=True)
        assert "idxs" in s.keys()
        assert s["idxs"].shape == (5,)

    def test_no_idx_by_default(self):
        buf = MultiAgentReplayBuffer(100)
        buf.add(_make_ma_td(MA_AGENTS, batch_size=10))
        s = buf.sample(3)
        assert "idxs" not in s.keys()

    def test_sampled_indices_within_bounds(self):
        buf = MultiAgentReplayBuffer(50)
        buf.add(_make_ma_td(MA_AGENTS, batch_size=20))
        s = buf.sample(15, return_idx=True)
        assert (s["idxs"] >= 0).all()
        assert (s["idxs"] < buf.size).all()


##### MultiAgentReplayBuffer — Value round-trip #####


class TestMAValues:
    def test_exact_recovery_single_transition(self):
        buf = MultiAgentReplayBuffer(100)
        td = TensorDict(
            {
                "obs": TensorDict(
                    {
                        "a0": torch.tensor([[1.0, 2.0, 3.0]]),
                        "a1": torch.tensor([[4.0, 5.0, 6.0]]),
                    },
                    batch_size=[1],
                ),
                "reward": TensorDict(
                    {
                        "a0": torch.tensor([[10.0]]),
                        "a1": torch.tensor([[20.0]]),
                    },
                    batch_size=[1],
                ),
            },
            batch_size=[1],
        )
        buf.add(td)
        s = buf.sample(1)
        torch.testing.assert_close(s["obs", "a0"], torch.tensor([[1.0, 2.0, 3.0]]))
        torch.testing.assert_close(s["obs", "a1"], torch.tensor([[4.0, 5.0, 6.0]]))
        torch.testing.assert_close(s["reward", "a0"], torch.tensor([[10.0]]))
        torch.testing.assert_close(s["reward", "a1"], torch.tensor([[20.0]]))

    def test_batch_values_belong_to_original(self):
        buf = MultiAgentReplayBuffer(100)
        obs_a0 = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        td = TensorDict(
            {
                "obs": TensorDict(
                    {"a0": obs_a0},
                    batch_size=[3],
                ),
            },
            batch_size=[3],
        )
        buf.add(td)
        s = buf.sample(3)
        for row in s["obs", "a0"]:
            assert any(torch.allclose(row, obs_a0[i]) for i in range(3))

    def test_scalar_reward_round_trip(self):
        """1-D rewards (auto-unsqueezed) are recovered correctly."""
        buf = MultiAgentReplayBuffer(100)
        td = TensorDict(
            {
                "reward": TensorDict(
                    {"a0": torch.tensor([42.0])},
                    batch_size=[1],
                ),
            },
            batch_size=[1],
        )
        buf.add(td)
        s = buf.sample(1)
        assert s["reward", "a0"].item() == pytest.approx(42.0)

    def test_multi_field_deterministic_round_trip(self):
        """Full 5-field transition round-trips correctly with a single stored item."""
        buf = MultiAgentReplayBuffer(10)
        td = TensorDict(
            {
                "obs": TensorDict(
                    {
                        "a0": torch.tensor([[1.0, 2.0]]),
                        "a1": torch.tensor([[3.0, 4.0]]),
                    },
                    batch_size=[1],
                ),
                "action": TensorDict(
                    {"a0": torch.tensor([[0.5]]), "a1": torch.tensor([[0.7]])},
                    batch_size=[1],
                ),
                "reward": TensorDict(
                    {"a0": torch.tensor([[10.0]]), "a1": torch.tensor([[20.0]])},
                    batch_size=[1],
                ),
                "next_obs": TensorDict(
                    {
                        "a0": torch.tensor([[5.0, 6.0]]),
                        "a1": torch.tensor([[7.0, 8.0]]),
                    },
                    batch_size=[1],
                ),
                "done": TensorDict(
                    {"a0": torch.tensor([[0.0]]), "a1": torch.tensor([[1.0]])},
                    batch_size=[1],
                ),
            },
            batch_size=[1],
        )
        buf.add(td)
        s = buf.sample(1)
        torch.testing.assert_close(s["obs", "a0"], torch.tensor([[1.0, 2.0]]))
        torch.testing.assert_close(s["action", "a1"], torch.tensor([[0.7]]))
        torch.testing.assert_close(s["reward", "a0"], torch.tensor([[10.0]]))
        torch.testing.assert_close(s["next_obs", "a1"], torch.tensor([[7.0, 8.0]]))
        torch.testing.assert_close(s["done", "a1"], torch.tensor([[1.0]]))


##### MultiAgentReplayBuffer — Image observations #####


class TestMAImages:
    def test_image_obs_shapes(self):
        img = (3, 64, 64)
        td = TensorDict(
            {
                "obs": TensorDict(
                    {a: torch.randn(1, *img) for a in MA_AGENTS},
                    batch_size=[1],
                ),
            },
            batch_size=[1],
        )
        buf = MultiAgentReplayBuffer(50)
        buf.add(td)
        s = buf.sample(1)
        for a in MA_AGENTS:
            assert s["obs", a].shape == (1, *img)

    def test_vectorized_image_batch(self):
        img = (3, 32, 32)
        n = 4
        td = TensorDict(
            {
                "obs": TensorDict(
                    {a: torch.randn(n, *img) for a in MA_AGENTS},
                    batch_size=[n],
                ),
            },
            batch_size=[n],
        )
        buf = MultiAgentReplayBuffer(50)
        buf.add(td)
        s = buf.sample(2)
        for a in MA_AGENTS:
            assert s["obs", a].shape == (2, *img)


##### MultiAgentReplayBuffer — Heterogeneous obs per agent #####


class TestMAHeterogeneous:
    def test_different_obs_and_act_sizes(self):
        td = TensorDict(
            {
                "obs": TensorDict(
                    {"big": torch.randn(2, 10), "small": torch.randn(2, 3)},
                    batch_size=[2],
                ),
                "action": TensorDict(
                    {"big": torch.randn(2, 4), "small": torch.randn(2, 1)},
                    batch_size=[2],
                ),
            },
            batch_size=[2],
        )
        buf = MultiAgentReplayBuffer(50)
        buf.add(td)
        s = buf.sample(2)
        assert s["obs", "big"].shape == (2, 10)
        assert s["obs", "small"].shape == (2, 3)
        assert s["action", "big"].shape == (2, 4)
        assert s["action", "small"].shape == (2, 1)

    def test_image_and_vector_agents(self):
        """One agent observes images, another observes vectors (mirrors old test)."""
        td = TensorDict(
            {
                "obs": TensorDict(
                    {
                        "visual_agent": torch.randn(2, 3, 128, 128),
                        "sensor_agent": torch.randn(2, 4),
                    },
                    batch_size=[2],
                ),
                "action": TensorDict(
                    {
                        "visual_agent": torch.randn(2, 2),
                        "sensor_agent": torch.randn(2, 1),
                    },
                    batch_size=[2],
                ),
                "reward": TensorDict(
                    {
                        "visual_agent": torch.rand(2, 1),
                        "sensor_agent": torch.rand(2, 1),
                    },
                    batch_size=[2],
                ),
            },
            batch_size=[2],
        )
        buf = MultiAgentReplayBuffer(50)
        buf.add(td)
        s = buf.sample(2)
        assert s["obs", "visual_agent"].shape == (2, 3, 128, 128)
        assert s["obs", "sensor_agent"].shape == (2, 4)
        assert s["action", "visual_agent"].shape == (2, 2)


##### MultiAgentReplayBuffer — Variable agent counts #####


class TestMAAgentCounts:
    def test_single_agent(self):
        agents = ["solo"]
        td = _make_ma_td(agents, batch_size=5)
        buf = MultiAgentReplayBuffer(20)
        buf.add(td)
        s = buf.sample(3)
        assert "solo" in s["obs"].keys()
        assert s["obs", "solo"].shape[0] == 3

    def test_three_agents(self):
        agents = ["a", "b", "c"]
        td = _make_ma_td(agents, batch_size=5, obs_size=4)
        buf = MultiAgentReplayBuffer(20)
        buf.add(td)
        s = buf.sample(3)
        for a in agents:
            assert a in s["obs"].keys()
            assert s["obs", a].shape == (3, 4)

    def test_five_agents(self):
        agents = [f"agent_{i}" for i in range(5)]
        td = _make_ma_td(agents, batch_size=4, obs_size=2, act_size=3)
        buf = MultiAgentReplayBuffer(30)
        buf.add(td)
        s = buf.sample(4)
        assert len(s["obs"].keys()) == 5
        for a in agents:
            assert s["action", a].shape == (4, 3)


##### MultiAgentReplayBuffer — Device placement #####


class TestMADevice:
    def test_cpu_explicit(self):
        buf = MultiAgentReplayBuffer(50, device="cpu")
        buf.add(_make_ma_td(MA_AGENTS, batch_size=2))
        s = buf.sample(1)
        for a in MA_AGENTS:
            assert s["obs", a].device == torch.device("cpu")

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda(self):
        buf = MultiAgentReplayBuffer(50, device="cuda")
        buf.add(_make_ma_td(MA_AGENTS, batch_size=2))
        s = buf.sample(1)
        for a in MA_AGENTS:
            assert s["obs", a].is_cuda


##### MultiAgentReplayBuffer — clear() #####


class TestMAClear:
    def test_resets_all_state(self):
        buf = MultiAgentReplayBuffer(50)
        buf.add(_make_ma_td(MA_AGENTS, batch_size=10))
        buf.clear()
        assert len(buf) == 0
        assert buf.size == 0
        assert buf.storage is None
        assert not buf.initialized
        assert not buf.is_full

    def test_usable_after_clear(self):
        buf = MultiAgentReplayBuffer(50)
        buf.add(_make_ma_td(MA_AGENTS, batch_size=5))
        buf.clear()
        buf.add(_make_ma_td(MA_AGENTS, batch_size=3))
        assert len(buf) == 3
        s = buf.sample(2)
        assert s.shape[0] == 2

    def test_counter_persists_after_clear(self):
        buf = MultiAgentReplayBuffer(50)
        buf.add(_make_ma_td(MA_AGENTS, batch_size=5))
        assert buf.counter == 5
        buf.clear()
        assert buf.counter == 5
        buf.add(_make_ma_td(MA_AGENTS, batch_size=2))
        assert buf.counter == 7


##### MultiAgentReplayBuffer — Edge cases #####


class TestMAEdgeCases:
    def test_add_exactly_max_size(self):
        buf = MultiAgentReplayBuffer(8)
        buf.add(_make_ma_td(MA_AGENTS, batch_size=8))
        assert len(buf) == 8
        assert buf.is_full
        s = buf.sample(8)
        assert s.shape[0] == 8

    def test_add_larger_than_max_size(self):
        buf = MultiAgentReplayBuffer(4)
        buf.add(_make_ma_td(MA_AGENTS, batch_size=6))
        assert len(buf) == 4
        assert buf.is_full

    def test_sample_entire_buffer(self):
        buf = MultiAgentReplayBuffer(10)
        buf.add(_make_ma_td(MA_AGENTS, batch_size=10))
        s = buf.sample(10)
        assert s.shape[0] == 10

    def test_is_full_transitions_at_boundary(self):
        buf = MultiAgentReplayBuffer(5)
        assert not buf.is_full
        buf.add(_make_ma_td(MA_AGENTS, batch_size=4))
        assert not buf.is_full
        buf.add(_make_ma_td(MA_AGENTS, batch_size=1))
        assert buf.is_full

    def test_add_batch_size_one_repeatedly(self):
        buf = MultiAgentReplayBuffer(3)
        for i in range(5):
            buf.add(_make_ma_td(MA_AGENTS, batch_size=1))
        assert len(buf) == 3
        assert buf.counter == 5


##### MultiAgentTransition — helper #####


def _build_ma_transition(num_envs: int, **field_dicts) -> TensorDict:
    """Construct a MultiAgentTransition and return the resulting TensorDict.

    Mirrors the single-agent pattern:
    ``Transition(...).to_tensordict()`` + ``batch_size = [num_envs]``.
    """
    t = MultiAgentTransition(**field_dicts)
    td = t.to_tensordict()
    td.batch_size = [num_envs]
    return td


##### MultiAgentTransition — flat observations #####


class TestMultiAgentTransitionFlat:
    def test_basic_construction(self):
        n = 4
        agents = ["a0", "a1"]
        td = _build_ma_transition(
            num_envs=n,
            obs={a: np.random.randn(n, 3) for a in agents},
            action={a: np.random.randn(n, 1) for a in agents},
            reward={a: np.random.randn(n) for a in agents},
            next_obs={a: np.random.randn(n, 3) for a in agents},
            done={a: np.zeros(n) for a in agents},
        )
        assert td.shape[0] == n
        for field in ("obs", "action", "reward", "next_obs", "done"):
            assert field in td.keys()
            sub = td[field]
            assert isinstance(sub, TensorDict)
            for a in agents:
                assert a in sub.keys()
                assert isinstance(sub[a], torch.Tensor)

    def test_dtype_is_float32(self):
        n = 2
        td = _build_ma_transition(
            num_envs=n,
            obs={"a0": np.ones((n, 3), dtype=np.float64)},
            action={"a0": np.zeros((n, 1), dtype=np.int32)},
            reward={"a0": np.ones(n, dtype=np.float64)},
            next_obs={"a0": np.ones((n, 3), dtype=np.float64)},
            done={"a0": np.zeros(n, dtype=np.bool_)},
        )
        assert td["obs", "a0"].dtype == torch.float32
        assert td["action", "a0"].dtype == torch.float32

    def test_round_trip_through_buffer(self):
        td = _build_ma_transition(
            num_envs=1,
            obs={"a0": np.array([[1.0, 2.0]])},
            action={"a0": np.array([[0.5]])},
            reward={"a0": np.array([10.0])},
            next_obs={"a0": np.array([[3.0, 4.0]])},
            done={"a0": np.array([0.0])},
        )
        buf = MultiAgentReplayBuffer(10)
        buf.add(td)
        s = buf.sample(1)
        torch.testing.assert_close(s["obs", "a0"], torch.tensor([[1.0, 2.0]]))
        torch.testing.assert_close(s["next_obs", "a0"], torch.tensor([[3.0, 4.0]]))
        torch.testing.assert_close(s["reward", "a0"], torch.tensor([[10.0]]))


##### MultiAgentTransition — dict observation spaces #####


class TestMultiAgentTransitionDictObs:
    """Tests for when per-agent observations are dicts (gymnasium.spaces.Dict)."""

    def test_dict_obs_produces_nested_tensordict(self):
        n = 2
        td = _build_ma_transition(
            num_envs=n,
            obs={
                "a0": {
                    "image": np.random.randn(n, 3, 8, 8),
                    "vector": np.random.randn(n, 5),
                },
                "a1": {
                    "image": np.random.randn(n, 3, 8, 8),
                    "vector": np.random.randn(n, 5),
                },
            },
            action={"a0": np.random.randn(n, 2), "a1": np.random.randn(n, 2)},
            reward={"a0": np.random.randn(n), "a1": np.random.randn(n)},
            next_obs={
                "a0": {
                    "image": np.random.randn(n, 3, 8, 8),
                    "vector": np.random.randn(n, 5),
                },
                "a1": {
                    "image": np.random.randn(n, 3, 8, 8),
                    "vector": np.random.randn(n, 5),
                },
            },
            done={"a0": np.zeros(n), "a1": np.zeros(n)},
        )
        assert td.shape[0] == n
        for a in ("a0", "a1"):
            agent_obs = td["obs", a]
            assert isinstance(agent_obs, TensorDict)
            assert "image" in agent_obs.keys()
            assert "vector" in agent_obs.keys()
            assert agent_obs["image"].shape == (n, 3, 8, 8)
            assert agent_obs["vector"].shape == (n, 5)

    def test_dict_obs_round_trip_through_buffer(self):
        n = 3
        td = _build_ma_transition(
            num_envs=n,
            obs={
                "a0": {
                    "cam": np.random.randn(n, 3, 4, 4),
                    "lidar": np.random.randn(n, 10),
                },
            },
            action={"a0": np.random.randn(n, 2)},
            reward={"a0": np.random.randn(n)},
            next_obs={
                "a0": {
                    "cam": np.random.randn(n, 3, 4, 4),
                    "lidar": np.random.randn(n, 10),
                },
            },
            done={"a0": np.zeros(n)},
        )
        buf = MultiAgentReplayBuffer(20)
        buf.add(td)
        s = buf.sample(2)
        assert s["obs", "a0", "cam"].shape == (2, 3, 4, 4)
        assert s["obs", "a0", "lidar"].shape == (2, 10)
        assert s["next_obs", "a0", "cam"].shape == (2, 3, 4, 4)

    def test_mixed_dict_and_flat_agents(self):
        """One agent has dict obs, another has flat obs."""
        n = 2
        td = _build_ma_transition(
            num_envs=n,
            obs={
                "visual": {
                    "image": np.random.randn(n, 3, 8, 8),
                    "depth": np.random.randn(n, 1, 8, 8),
                },
                "simple": np.random.randn(n, 4),
            },
            action={"visual": np.random.randn(n, 2), "simple": np.random.randn(n, 1)},
            reward={"visual": np.random.randn(n), "simple": np.random.randn(n)},
            next_obs={
                "visual": {
                    "image": np.random.randn(n, 3, 8, 8),
                    "depth": np.random.randn(n, 1, 8, 8),
                },
                "simple": np.random.randn(n, 4),
            },
            done={"visual": np.zeros(n), "simple": np.zeros(n)},
        )
        assert isinstance(td["obs", "visual"], TensorDict)
        assert isinstance(td["obs", "simple"], torch.Tensor)
        assert td["obs", "visual", "image"].shape == (n, 3, 8, 8)
        assert td["obs", "simple"].shape == (n, 4)

    def test_dict_obs_deterministic_values(self):
        """Verify exact values survive the conversion."""
        img = np.array([[[[1.0, 2.0], [3.0, 4.0]]]])  # (1, 1, 2, 2)
        vec = np.array([[5.0, 6.0, 7.0]])  # (1, 3)
        td = _build_ma_transition(
            num_envs=1,
            obs={"a0": {"image": img, "vector": vec}},
            action={"a0": np.array([[0.1]])},
            reward={"a0": np.array([99.0])},
            next_obs={"a0": {"image": img * 2, "vector": vec * 3}},
            done={"a0": np.array([0.0])},
        )
        torch.testing.assert_close(
            td["obs", "a0", "vector"], torch.tensor([[5.0, 6.0, 7.0]])
        )
        torch.testing.assert_close(
            td["next_obs", "a0", "image"],
            torch.tensor([[[[2.0, 4.0], [6.0, 8.0]]]]),
        )

    def test_dict_obs_buffer_circular_overwrite(self):
        """Dict-obs transitions survive circular buffer overwrites."""
        buf = MultiAgentReplayBuffer(3)
        for i in range(5):
            td = _build_ma_transition(
                num_envs=1,
                obs={"a0": {"x": np.full((1, 2), float(i))}},
                action={"a0": np.zeros((1, 1))},
                reward={"a0": np.array([float(i)])},
                next_obs={"a0": {"x": np.full((1, 2), float(i + 1))}},
                done={"a0": np.zeros(1)},
            )
            buf.add(td)
        assert len(buf) == 3
        s = buf.sample(3)
        assert s["obs", "a0", "x"].shape == (3, 2)
        for row in s["reward", "a0"]:
            assert row.item() >= 2.0  # oldest values (0, 1) have been overwritten

    def test_tuple_obs_produces_tensordict(self):
        """Tuple observations are converted via to_tensordict with tuple_obs_N keys."""
        n = 2
        td = _build_ma_transition(
            num_envs=n,
            obs={
                "a0": (np.random.randn(n, 3), np.random.randn(n, 5)),
            },
            action={"a0": np.random.randn(n, 1)},
            reward={"a0": np.random.randn(n)},
            next_obs={
                "a0": (np.random.randn(n, 3), np.random.randn(n, 5)),
            },
            done={"a0": np.zeros(n)},
        )
        agent_obs = td["obs", "a0"]
        assert isinstance(agent_obs, TensorDict)
        assert "tuple_obs_0" in agent_obs.keys()
        assert "tuple_obs_1" in agent_obs.keys()
        assert agent_obs["tuple_obs_0"].shape == (n, 3)
        assert agent_obs["tuple_obs_1"].shape == (n, 5)
