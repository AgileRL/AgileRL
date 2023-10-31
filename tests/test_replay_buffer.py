import numpy as np
import torch

from agilerl.components.replay_buffer import ReplayBuffer


# Can create an instance of ReplayBuffer with valid arguments
def test_create_instance_with_valid_arguments():
    action_dim = 2
    memory_size = 100
    field_names = ["state", "action", "reward"]
    device = "cpu"

    buffer = ReplayBuffer(action_dim, memory_size, field_names, device)

    assert buffer.n_actions == action_dim
    assert buffer.memory_size == memory_size
    assert buffer.field_names == field_names
    assert buffer.device == device


# Can get length of memory with __len__ method
def test_get_length_of_memory():
    action_dim = 2
    memory_size = 100
    field_names = ["state", "action", "reward"]
    device = "cpu"

    buffer = ReplayBuffer(action_dim, memory_size, field_names, device)

    # Add experiences to memory
    buffer.save2memorySingleEnv(1, 2, 3)
    buffer.save2memorySingleEnv(4, 5, 6)
    buffer.save2memorySingleEnv(7, 8, 9)

    assert len(buffer) == 3


# Can add experiences to memory and appends to end of deque
def test_append_to_memory_deque():
    buffer = ReplayBuffer(
        action_dim=4,
        memory_size=1000,
        field_names=["state", "action", "reward", "next_state", "done"],
    )
    buffer._add([0, 0, 0, 0], [1, 1, 1, 1], 1, [0, 0, 0, 0], False)
    buffer._add([1, 1, 1, 1], [2, 2, 2, 2], 2, [1, 1, 1, 1], True)
    assert len(buffer.memory) == 2
    assert buffer.memory[0].state == [0, 0, 0, 0]
    assert buffer.memory[0].action == [1, 1, 1, 1]
    assert buffer.memory[0].reward == 1
    assert buffer.memory[0].next_state == [0, 0, 0, 0]
    assert buffer.memory[0].done is False
    assert buffer.memory[1].state == [1, 1, 1, 1]
    assert buffer.memory[1].action == [2, 2, 2, 2]
    assert buffer.memory[1].reward == 2
    assert buffer.memory[1].next_state == [1, 1, 1, 1]
    assert buffer.memory[1].done is True


# Can add an experience when memory is full and maxlen is reached
def test_add_experience_when_memory_full():
    buffer = ReplayBuffer(
        action_dim=4,
        memory_size=2,
        field_names=["state", "action", "reward", "next_state", "done"],
    )
    buffer._add([0, 0, 0, 0], [1, 1, 1, 1], 1, [0, 0, 0, 0], False)
    buffer._add([1, 1, 1, 1], [2, 2, 2, 2], 2, [1, 1, 1, 1], True)
    buffer._add([2, 2, 2, 2], [3, 3, 3, 3], 3, [2, 2, 2, 2], False)
    assert len(buffer.memory) == 2
    assert buffer.memory[0].state == [1, 1, 1, 1]
    assert buffer.memory[0].action == [2, 2, 2, 2]
    assert buffer.memory[0].reward == 2
    assert buffer.memory[0].next_state == [1, 1, 1, 1]
    assert buffer.memory[0].done is True
    assert buffer.memory[1].state == [2, 2, 2, 2]
    assert buffer.memory[1].action == [3, 3, 3, 3]
    assert buffer.memory[1].reward == 3
    assert buffer.memory[1].next_state == [2, 2, 2, 2]
    assert buffer.memory[1].done is False


# Can add single experiences to memory with save2memorySingleEnv method
def test_add_single_experiences_to_memory():
    action_dim = 2
    memory_size = 100
    field_names = ["state", "action", "reward"]
    device = "cpu"

    buffer = ReplayBuffer(action_dim, memory_size, field_names, device)

    state = np.array([1, 2])
    action = np.array([0])
    reward = np.array([0])

    buffer.save2memorySingleEnv(state, action, reward)

    assert len(buffer.memory) == 1
    assert buffer.memory[0].state.tolist() == state.tolist()
    assert buffer.memory[0].action.tolist() == action.tolist()
    assert buffer.memory[0].reward.tolist() == reward.tolist()


# Can add multiple experiences to memory with save2memoryVectEnvs method
def test_add_multiple_experiences_to_memory():
    action_dim = 2
    memory_size = 100
    field_names = ["state", "action", "reward", "next_state", "done"]
    device = "cpu"

    buffer = ReplayBuffer(action_dim, memory_size, field_names, device)

    states = np.array([[1, 2], [3, 4]])
    actions = np.array([[0], [1]])
    rewards = np.array([[0], [1]])
    next_states = np.array([[5, 6], [7, 8]])
    dones = np.array([[False], [True]])

    buffer.save2memoryVectEnvs(states, actions, rewards, next_states, dones)

    assert len(buffer.memory) == 2
    assert buffer.memory[0].state.tolist() == states[0].tolist()
    assert buffer.memory[0].action.tolist() == actions[0].tolist()
    assert buffer.memory[0].reward.tolist() == rewards[0].tolist()
    assert buffer.memory[0].next_state.tolist() == next_states[0].tolist()
    assert buffer.memory[0].done.tolist() == dones[0].tolist()
    assert buffer.memory[1].state.tolist() == states[1].tolist()
    assert buffer.memory[1].action.tolist() == actions[1].tolist()
    assert buffer.memory[1].reward.tolist() == rewards[1].tolist()
    assert buffer.memory[1].next_state.tolist() == next_states[1].tolist()
    assert buffer.memory[1].done.tolist() == dones[1].tolist()


# Can handle vectorized and un-vectorized experiences from environment
def test_add_any_experiences_to_memory():
    action_dim = 2
    memory_size = 100
    field_names = ["state", "action", "reward", "next_state", "done"]
    device = "cpu"

    buffer = ReplayBuffer(action_dim, memory_size, field_names, device)

    states = np.array([[1, 2], [3, 4]])
    actions = np.array([[0], [1]])
    rewards = np.array([[0], [1]])
    next_states = np.array([[5, 6], [7, 8]])
    dones = np.array([[False], [True]])

    buffer.save2memory(states, actions, rewards, next_states, dones, is_vectorised=True)

    assert len(buffer.memory) == 2
    assert buffer.memory[0].state.tolist() == states[0].tolist()
    assert buffer.memory[0].action.tolist() == actions[0].tolist()
    assert buffer.memory[0].reward.tolist() == rewards[0].tolist()
    assert buffer.memory[0].next_state.tolist() == next_states[0].tolist()
    assert buffer.memory[0].done.tolist() == dones[0].tolist()
    assert buffer.memory[1].state.tolist() == states[1].tolist()
    assert buffer.memory[1].action.tolist() == actions[1].tolist()
    assert buffer.memory[1].reward.tolist() == rewards[1].tolist()
    assert buffer.memory[1].next_state.tolist() == next_states[1].tolist()
    assert buffer.memory[1].done.tolist() == dones[1].tolist()

    new_state = np.array([1, 2])
    new_action = np.array([0])
    new_reward = np.array([0])
    new_next_state = np.array([3, 4])
    new_done = np.array([False])

    buffer.save2memory(
        new_state, new_action, new_reward, new_next_state, new_done, is_vectorised=False
    )

    assert len(buffer.memory) == 3
    assert buffer.memory[2].state.tolist() == new_state.tolist()
    assert buffer.memory[2].action.tolist() == new_action.tolist()
    assert buffer.memory[2].reward.tolist() == new_reward.tolist()
    assert buffer.memory[2].next_state.tolist() == new_next_state.tolist()
    assert buffer.memory[2].done.tolist() == new_done.tolist()


# Can sample experiences from memory of desired batch size with sample method
def test_sample_experiences_from_memory():
    action_dim = 1
    memory_size = 100
    field_names = ["state", "action", "reward"]
    device = "cpu"

    buffer = ReplayBuffer(action_dim, memory_size, field_names, device)

    # Add experiences to memory
    buffer.save2memorySingleEnv(np.array([1, 1]), 2, 3)
    buffer.save2memorySingleEnv(np.array([4, 4]), 5, 6)
    buffer.save2memorySingleEnv(np.array([7, 7]), 8, 9)

    # Sample experiences from memory
    batch_size = 2
    experiences = buffer.sample(batch_size)

    assert len(experiences[0]) == batch_size
    assert len(experiences[1]) == batch_size
    assert len(experiences[2]) == batch_size
    assert isinstance(experiences[0], torch.Tensor)
    assert experiences[0].shape == (batch_size, 2)
    assert isinstance(experiences[1], torch.Tensor)
    assert experiences[1].shape == (batch_size, action_dim)
    assert isinstance(experiences[2], torch.Tensor)
    assert experiences[2].shape == (batch_size, 1)


# Can process transition from experiences with _process_transition method
def test_process_transition_from_experiences():
    action_dim = 1
    memory_size = 100
    field_names = ["state", "action", "reward"]
    device = "cpu"

    buffer = ReplayBuffer(action_dim, memory_size, field_names, device)

    # Create experiences
    experience1 = buffer.experience(1, 2, 3)
    experience2 = buffer.experience(4, 5, 6)
    experiences = [experience1, experience2]

    # Process transition from experiences
    transition = buffer._process_transition(experiences)

    assert isinstance(transition["state"], torch.Tensor)
    assert transition["state"].shape == (len(experiences), 1)
    assert isinstance(transition["action"], torch.Tensor)
    assert transition["action"].shape == (len(experiences), action_dim)
    assert isinstance(transition["reward"], torch.Tensor)
    assert transition["reward"].shape == (len(experiences), 1)


# Can process single transition from experiences with _process_transition method
def test_process_single_transition_from_experiences():
    action_dim = 1
    memory_size = 100
    field_names = ["state", "action", "reward"]
    device = "cpu"

    buffer = ReplayBuffer(action_dim, memory_size, field_names, device)

    # Create experiences
    experience1 = buffer.experience(np.array([1, 1]), 2, 3)
    experiences = [experience1]

    # Process transition from experiences
    transition = buffer._process_transition(experiences)

    assert isinstance(transition["state"], torch.Tensor)
    assert transition["state"].shape == (len(experiences), 2)
    assert isinstance(transition["action"], torch.Tensor)
    assert transition["action"].shape == (len(experiences), action_dim)
    assert isinstance(transition["reward"], torch.Tensor)
    assert transition["reward"].shape == (len(experiences), 1)
