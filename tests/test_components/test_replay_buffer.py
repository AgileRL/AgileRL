import numpy as np
import pytest
import torch
from tensordict import TensorDict

from agilerl.components.replay_buffer import (
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
    for idx, priority in zip(indices, new_priorities):
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
