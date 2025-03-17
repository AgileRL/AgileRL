import torch
from tensordict import TensorDict

from agilerl.buffers.multi_agent_replay_buffer import MultiAgentReplayBuffer


def test_create_instance_with_valid_arguments():
    """Test creating an instance with valid arguments"""
    # Test parameters
    max_size = 1000
    agent_ids = ["agent1", "agent2"]
    device = "cpu"
    dtype = torch.float32

    # Create buffer instance
    buffer = MultiAgentReplayBuffer(max_size, agent_ids, device, dtype)

    # Check if buffer properties are set correctly
    assert buffer.max_size == max_size
    assert buffer.agent_ids == agent_ids
    assert buffer.device == device
    assert buffer.dtype == dtype
    assert buffer.counter == 0
    assert buffer.initialized is False
    assert buffer._cursor == 0
    assert buffer._size == 0
    assert buffer._storage is None


def test_add_single_transition():
    """Test adding a single transition"""
    buffer = MultiAgentReplayBuffer(max_size=10, agent_ids=["agent1", "agent2"])

    # Create sample transition
    transition = TensorDict(
        {
            "obs": TensorDict(
                {
                    "agent1": torch.ones(4),
                    "agent2": torch.zeros(4),
                },
                batch_size=[],
            ),
            "action": TensorDict(
                {
                    "agent1": torch.tensor(1),
                    "agent2": torch.tensor(0),
                },
                batch_size=[],
            ),
            "reward": TensorDict(
                {
                    "agent1": torch.tensor(1.0),
                    "agent2": torch.tensor(0.5),
                },
                batch_size=[],
            ),
            "next_obs": TensorDict(
                {
                    "agent1": torch.ones(4) * 2,
                    "agent2": torch.ones(4),
                },
                batch_size=[],
            ),
            "done": torch.tensor(False),
        },
        batch_size=[],
    )

    # Add transition to buffer
    buffer.add(transition)

    # Check if buffer size increased
    assert buffer._size == 1
    assert buffer._cursor == 1
    assert buffer.counter == 1

    # Check if storage contains correct data
    assert (buffer.storage["obs"]["agent1"][0] == torch.ones(4)).all()
    assert (buffer.storage["obs"]["agent2"][0] == torch.zeros(4)).all()
    assert buffer.storage["action"]["agent1"][0] == 1
    assert buffer.storage["action"]["agent2"][0] == 0
    assert buffer.storage["reward"]["agent1"][0] == 1.0
    assert buffer.storage["reward"]["agent2"][0] == 0.5
    assert (buffer.storage["next_obs"]["agent1"][0] == torch.ones(4) * 2).all()
    assert (buffer.storage["next_obs"]["agent2"][0] == torch.ones(4)).all()
    assert not buffer.storage["done"][0]


def test_add_vectorized_transitions():
    """Test adding vectorized transitions"""
    buffer = MultiAgentReplayBuffer(max_size=10, agent_ids=["agent1", "agent2"])

    # Create vectorized transitions
    batch_size = 3
    transitions = TensorDict(
        {
            "obs": TensorDict(
                {
                    "agent1": torch.ones(batch_size, 4),
                    "agent2": torch.zeros(batch_size, 4),
                },
                batch_size=[batch_size],
            ),
            "action": TensorDict(
                {
                    "agent1": torch.tensor([1, 2, 3]),
                    "agent2": torch.tensor([0, 1, 2]),
                },
                batch_size=[batch_size],
            ),
            "reward": TensorDict(
                {
                    "agent1": torch.tensor([1.0, 1.5, 2.0]),
                    "agent2": torch.tensor([0.5, 1.0, 1.5]),
                },
                batch_size=[batch_size],
            ),
            "next_obs": TensorDict(
                {
                    "agent1": torch.ones(batch_size, 4) * 2,
                    "agent2": torch.ones(batch_size, 4),
                },
                batch_size=[batch_size],
            ),
            "done": torch.tensor([False, False, True]),
        },
        batch_size=[batch_size],
    )

    # Add transitions to buffer
    buffer.add(transitions, is_vectorised=True)

    # Check if buffer size increased
    assert buffer._size == 3
    assert buffer._cursor == 3
    assert buffer.counter == 3

    # Check if storage contains correct data
    assert (buffer.storage["obs"]["agent1"][0] == torch.ones(4)).all()
    assert (buffer.storage["obs"]["agent2"][0] == torch.zeros(4)).all()
    assert buffer.storage["action"]["agent1"][0] == 1
    assert buffer.storage["action"]["agent2"][0] == 0
    assert buffer.storage["reward"]["agent1"][0] == 1.0
    assert buffer.storage["reward"]["agent2"][0] == 0.5
    assert (buffer.storage["next_obs"]["agent1"][0] == torch.ones(4) * 2).all()
    assert (buffer.storage["next_obs"]["agent2"][0] == torch.ones(4)).all()
    assert not buffer.storage["done"][0]


def test_buffer_wrapping():
    """Test buffer wrapping when adding more transitions than max_size"""
    buffer = MultiAgentReplayBuffer(max_size=5, agent_ids=["agent1", "agent2"])

    # Create vectorized transitions
    batch_size = 7
    transitions = TensorDict(
        {
            "obs": TensorDict(
                {
                    "agent1": torch.arange(batch_size)
                    .unsqueeze(1)
                    .expand(-1, 4)
                    .float(),
                    "agent2": torch.arange(batch_size, 2 * batch_size)
                    .unsqueeze(1)
                    .expand(-1, 4)
                    .float(),
                },
                batch_size=[batch_size],
            ),
            "action": TensorDict(
                {
                    "agent1": torch.arange(batch_size),
                    "agent2": torch.arange(batch_size, 2 * batch_size),
                },
                batch_size=[batch_size],
            ),
            "reward": TensorDict(
                {
                    "agent1": torch.arange(batch_size).float(),
                    "agent2": torch.arange(batch_size, 2 * batch_size).float(),
                },
                batch_size=[batch_size],
            ),
            "next_obs": TensorDict(
                {
                    "agent1": torch.arange(batch_size)
                    .unsqueeze(1)
                    .expand(-1, 4)
                    .float()
                    + 1,
                    "agent2": torch.arange(batch_size, 2 * batch_size)
                    .unsqueeze(1)
                    .expand(-1, 4)
                    .float()
                    + 1,
                },
                batch_size=[batch_size],
            ),
            "done": torch.tensor([False] * batch_size),
        },
        batch_size=[batch_size],
    )

    # Add transitions to buffer
    buffer.add(transitions, is_vectorised=True)

    # Check if buffer size is capped at max_size
    assert buffer._size == 5
    assert buffer._cursor == 2  # 5 + 7 = 12, 12 % 5 = 2
    assert buffer.counter == 7

    # Check if storage contains wrapped data (should now contain indices 2,3,4,5,6)
    # First position (index 0) contains transition 5 (6th in batch)
    assert (buffer.storage["obs"]["agent1"][0][0] == 5).all()
    assert (buffer.storage["obs"]["agent2"][0][0] == 12).all()
    assert buffer.storage["action"]["agent1"][0] == 5
    assert buffer.storage["action"]["agent2"][0] == 12
    assert buffer.storage["reward"]["agent1"][0] == 5.0
    assert buffer.storage["reward"]["agent2"][0] == 12.0

    # Last position (index 4) contains transition 4 (5th in batch)
    assert (buffer.storage["obs"]["agent1"][4][0] == 4).all()
    assert (buffer.storage["obs"]["agent2"][4][0] == 11).all()
    assert buffer.storage["action"]["agent1"][4] == 4
    assert buffer.storage["action"]["agent2"][4] == 11
    assert buffer.storage["reward"]["agent1"][4] == 4.0
    assert buffer.storage["reward"]["agent2"][4] == 11.0


def test_sampling():
    """Test sampling from buffer"""
    buffer = MultiAgentReplayBuffer(max_size=10, agent_ids=["agent1", "agent2"])

    # Create and add 5 vectorized transitions
    batch_size = 5
    transitions = TensorDict(
        {
            "obs": TensorDict(
                {
                    "agent1": torch.arange(batch_size)
                    .unsqueeze(1)
                    .expand(-1, 4)
                    .float(),
                    "agent2": torch.arange(batch_size, 2 * batch_size)
                    .unsqueeze(1)
                    .expand(-1, 4)
                    .float(),
                },
                batch_size=[batch_size],
            ),
            "action": TensorDict(
                {
                    "agent1": torch.arange(batch_size),
                    "agent2": torch.arange(batch_size, 2 * batch_size),
                },
                batch_size=[batch_size],
            ),
            "reward": TensorDict(
                {
                    "agent1": torch.arange(batch_size).float(),
                    "agent2": torch.arange(batch_size, 2 * batch_size).float(),
                },
                batch_size=[batch_size],
            ),
            "next_obs": TensorDict(
                {
                    "agent1": torch.arange(batch_size)
                    .unsqueeze(1)
                    .expand(-1, 4)
                    .float()
                    + 1,
                    "agent2": torch.arange(batch_size, 2 * batch_size)
                    .unsqueeze(1)
                    .expand(-1, 4)
                    .float()
                    + 1,
                },
                batch_size=[batch_size],
            ),
            "done": torch.tensor([False, False, True, False, False]),
        },
        batch_size=[batch_size],
    )

    buffer.add(transitions, is_vectorised=True)

    # Sample from buffer
    sample_size = 3
    samples = buffer.sample(sample_size)

    # Check if sample structure is correct
    assert isinstance(samples, TensorDict)
    assert torch.Size([sample_size]) == samples.batch_size

    # Check if all required keys are present
    assert "obs" in samples
    assert "action" in samples
    assert "reward" in samples
    assert "next_obs" in samples
    assert "done" in samples

    # Check if agent-specific keys are present
    assert "agent1" in samples["obs"]
    assert "agent2" in samples["obs"]
    assert "agent1" in samples["action"]
    assert "agent2" in samples["action"]
    assert "agent1" in samples["reward"]
    assert "agent2" in samples["reward"]
    assert "agent1" in samples["next_obs"]
    assert "agent2" in samples["next_obs"]

    # Check if shapes are correct
    assert samples["obs"]["agent1"].shape == (sample_size, 4)
    assert samples["obs"]["agent2"].shape == (sample_size, 4)
    assert samples["action"]["agent1"].shape == (sample_size,)
    assert samples["action"]["agent2"].shape == (sample_size,)
    assert samples["reward"]["agent1"].shape == (sample_size,)
    assert samples["reward"]["agent2"].shape == (sample_size,)
    assert samples["next_obs"]["agent1"].shape == (sample_size, 4)
    assert samples["next_obs"]["agent2"].shape == (sample_size, 4)
    assert samples["done"].shape == (sample_size,)


def test_sample_with_indices():
    """Test sampling from buffer with indices"""
    buffer = MultiAgentReplayBuffer(max_size=10, agent_ids=["agent1", "agent2"])

    # Create and add 5 vectorized transitions
    batch_size = 5
    transitions = TensorDict(
        {
            "obs": TensorDict(
                {
                    "agent1": torch.arange(batch_size)
                    .unsqueeze(1)
                    .expand(-1, 4)
                    .float(),
                    "agent2": torch.arange(batch_size, 2 * batch_size)
                    .unsqueeze(1)
                    .expand(-1, 4)
                    .float(),
                },
                batch_size=[batch_size],
            ),
            "action": TensorDict(
                {
                    "agent1": torch.arange(batch_size),
                    "agent2": torch.arange(batch_size, 2 * batch_size),
                },
                batch_size=[batch_size],
            ),
            "reward": TensorDict(
                {
                    "agent1": torch.arange(batch_size).float(),
                    "agent2": torch.arange(batch_size, 2 * batch_size).float(),
                },
                batch_size=[batch_size],
            ),
            "next_obs": TensorDict(
                {
                    "agent1": torch.arange(batch_size)
                    .unsqueeze(1)
                    .expand(-1, 4)
                    .float()
                    + 1,
                    "agent2": torch.arange(batch_size, 2 * batch_size)
                    .unsqueeze(1)
                    .expand(-1, 4)
                    .float()
                    + 1,
                },
                batch_size=[batch_size],
            ),
            "done": torch.tensor([False, False, True, False, False]),
        },
        batch_size=[batch_size],
    )

    buffer.add(transitions, is_vectorised=True)

    # Sample from buffer with indices
    sample_size = 3
    samples = buffer.sample(sample_size, return_idx=True)

    # Check if indices are present
    assert "idxs" in samples
    assert samples["idxs"].shape == (sample_size,)

    # Check if all indices are valid
    assert (samples["idxs"] >= 0).all()
    assert (samples["idxs"] < buffer._size).all()


def test_legacy_save_methods():
    """Test legacy save methods for compatibility"""
    buffer = MultiAgentReplayBuffer(max_size=10, agent_ids=["agent1", "agent2"])

    # Create transition
    transition = TensorDict(
        {
            "obs": TensorDict(
                {
                    "agent1": torch.ones(4),
                    "agent2": torch.zeros(4),
                },
                batch_size=[],
            ),
            "action": TensorDict(
                {
                    "agent1": torch.tensor(1),
                    "agent2": torch.tensor(0),
                },
                batch_size=[],
            ),
            "reward": TensorDict(
                {
                    "agent1": torch.tensor(1.0),
                    "agent2": torch.tensor(0.5),
                },
                batch_size=[],
            ),
            "next_obs": TensorDict(
                {
                    "agent1": torch.ones(4) * 2,
                    "agent2": torch.ones(4),
                },
                batch_size=[],
            ),
            "done": torch.tensor(False),
        },
        batch_size=[],
    )

    # Test save_to_memory_single_env
    buffer.save_to_memory_single_env(transition)
    assert buffer._size == 1

    # Clear buffer
    buffer.clear()
    assert buffer._size == 0
    assert buffer._storage is None

    # Test save_to_memory method
    buffer.save_to_memory(transition)
    assert buffer._size == 1

    # Create vectorized transitions
    batch_size = 3
    transitions = TensorDict(
        {
            "obs": TensorDict(
                {
                    "agent1": torch.ones(batch_size, 4),
                    "agent2": torch.zeros(batch_size, 4),
                },
                batch_size=[batch_size],
            ),
            "action": TensorDict(
                {
                    "agent1": torch.tensor([1, 2, 3]),
                    "agent2": torch.tensor([0, 1, 2]),
                },
                batch_size=[batch_size],
            ),
            "reward": TensorDict(
                {
                    "agent1": torch.tensor([1.0, 1.5, 2.0]),
                    "agent2": torch.tensor([0.5, 1.0, 1.5]),
                },
                batch_size=[batch_size],
            ),
            "next_obs": TensorDict(
                {
                    "agent1": torch.ones(batch_size, 4) * 2,
                    "agent2": torch.ones(batch_size, 4),
                },
                batch_size=[batch_size],
            ),
            "done": torch.tensor([False, False, True]),
        },
        batch_size=[batch_size],
    )

    # Clear buffer again
    buffer.clear()

    # Test save_to_memory_vect_envs
    buffer.save_to_memory_vect_envs(transitions)
    assert buffer._size == 3


def test_clear_buffer():
    """Test clearing the buffer"""
    buffer = MultiAgentReplayBuffer(max_size=10, agent_ids=["agent1", "agent2"])

    # Add a transition
    transition = TensorDict(
        {
            "obs": TensorDict(
                {
                    "agent1": torch.ones(4),
                    "agent2": torch.zeros(4),
                },
                batch_size=[],
            ),
            "action": TensorDict(
                {
                    "agent1": torch.tensor(1),
                    "agent2": torch.tensor(0),
                },
                batch_size=[],
            ),
            "reward": TensorDict(
                {
                    "agent1": torch.tensor(1.0),
                    "agent2": torch.tensor(0.5),
                },
                batch_size=[],
            ),
            "next_obs": TensorDict(
                {
                    "agent1": torch.ones(4) * 2,
                    "agent2": torch.ones(4),
                },
                batch_size=[],
            ),
            "done": torch.tensor(False),
        },
        batch_size=[],
    )

    buffer.add(transition)
    assert buffer._size == 1

    # Clear buffer
    buffer.clear()

    # Check if buffer is empty
    assert buffer._size == 0
    assert buffer._cursor == 0
    assert buffer._storage is None
    assert buffer.initialized is False


def test_image_observations():
    """Test buffer with image observations"""
    buffer = MultiAgentReplayBuffer(max_size=10, agent_ids=["agent1", "agent2"])

    # Create transition with image observations
    obs_shape = (3, 84, 84)  # C x H x W
    transition = TensorDict(
        {
            "obs": TensorDict(
                {
                    "agent1": torch.zeros(obs_shape),
                    "agent2": torch.ones(obs_shape),
                },
                batch_size=[],
            ),
            "action": TensorDict(
                {
                    "agent1": torch.tensor(1),
                    "agent2": torch.tensor(0),
                },
                batch_size=[],
            ),
            "reward": TensorDict(
                {
                    "agent1": torch.tensor(1.0),
                    "agent2": torch.tensor(0.5),
                },
                batch_size=[],
            ),
            "next_obs": TensorDict(
                {
                    "agent1": torch.zeros(obs_shape) + 0.5,
                    "agent2": torch.ones(obs_shape) + 0.5,
                },
                batch_size=[],
            ),
            "done": torch.tensor(False),
        },
        batch_size=[],
    )

    # Add transition and sample
    buffer.add(transition)
    samples = buffer.sample(1)

    # Check if image shapes are preserved
    assert samples["obs"]["agent1"].shape == (1,) + obs_shape
    assert samples["obs"]["agent2"].shape == (1,) + obs_shape
    assert samples["next_obs"]["agent1"].shape == (1,) + obs_shape
    assert samples["next_obs"]["agent2"].shape == (1,) + obs_shape


def test_mixed_observation_spaces():
    """Test buffer with mixed observation spaces"""
    buffer = MultiAgentReplayBuffer(max_size=10, agent_ids=["agent1", "agent2"])

    # Create transition with mixed observation spaces
    # Agent1: dictionary with image and vector
    # Agent2: vector only
    transition = TensorDict(
        {
            "obs": TensorDict(
                {
                    "agent1": TensorDict(
                        {
                            "image": torch.zeros(3, 84, 84),
                            "vector": torch.tensor([1.0, 2.0, 3.0, 4.0]),
                        },
                        batch_size=[],
                    ),
                    "agent2": torch.ones(4),
                },
                batch_size=[],
            ),
            "action": TensorDict(
                {
                    "agent1": torch.tensor(1),
                    "agent2": torch.tensor(0),
                },
                batch_size=[],
            ),
            "reward": TensorDict(
                {
                    "agent1": torch.tensor(1.0),
                    "agent2": torch.tensor(0.5),
                },
                batch_size=[],
            ),
            "next_obs": TensorDict(
                {
                    "agent1": TensorDict(
                        {
                            "image": torch.zeros(3, 84, 84) + 0.5,
                            "vector": torch.tensor([2.0, 3.0, 4.0, 5.0]),
                        },
                        batch_size=[],
                    ),
                    "agent2": torch.ones(4) + 1,
                },
                batch_size=[],
            ),
            "done": torch.tensor(False),
        },
        batch_size=[],
    )

    # Add transition and sample
    buffer.add(transition)
    samples = buffer.sample(1)

    # Check structure and shapes
    assert isinstance(samples["obs"]["agent1"], TensorDict)
    assert "image" in samples["obs"]["agent1"]
    assert "vector" in samples["obs"]["agent1"]
    assert samples["obs"]["agent1"]["image"].shape == (1, 3, 84, 84)
    assert samples["obs"]["agent1"]["vector"].shape == (1, 4)
    assert samples["obs"]["agent2"].shape == (1, 4)

    assert isinstance(samples["next_obs"]["agent1"], TensorDict)
    assert "image" in samples["next_obs"]["agent1"]
    assert "vector" in samples["next_obs"]["agent1"]
    assert samples["next_obs"]["agent1"]["image"].shape == (1, 3, 84, 84)
    assert samples["next_obs"]["agent1"]["vector"].shape == (1, 4)
    assert samples["next_obs"]["agent2"].shape == (1, 4)
