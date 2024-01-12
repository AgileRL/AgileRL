import numpy as np
import torch

from agilerl.components.multi_agent_replay_buffer import MultiAgentReplayBuffer


# Can initialize an instance of MultiAgentReplayBuffer with valid arguments
def test_initialize_instance_with_valid_arguments():
    memory_size = 100
    field_names = ["state", "action", "reward"]
    agent_ids = ["agent1", "agent2"]

    buffer = MultiAgentReplayBuffer(memory_size, field_names, agent_ids)

    assert len(buffer) == 0
    assert buffer.memory.maxlen == memory_size
    assert buffer.field_names == field_names
    assert buffer.agent_ids == agent_ids
    assert buffer.counter == 0
    assert buffer.device is None


# Can get length of memory with __len__ method
def test_get_length_of_memory():
    memory_size = 100
    field_names = ["state", "action", "reward"]
    agent_ids = ["agent1", "agent2"]

    buffer = MultiAgentReplayBuffer(memory_size, field_names, agent_ids)

    state = {"agent1": np.array([1, 2, 3]), "agent2": np.array([1, 2, 3])}
    action = {"agent1": np.array([4, 5]), "agent2": np.array([4, 5])}
    reward = {"agent1": np.array([6]), "agent2": np.array([7])}

    # Add experiences to memory
    buffer.save2memory(state, action, reward)
    buffer.save2memory(state, action, reward)
    buffer.save2memory(state, action, reward)

    assert len(buffer) == 3


# Can add experiences to memory and appends to end of deque
def test_append_to_memory_deque():
    memory_size = 100
    field_names = ["state", "action", "reward"]
    agent_ids = ["agent1", "agent2"]

    buffer = MultiAgentReplayBuffer(memory_size, field_names, agent_ids)

    state = {"agent1": np.array([1, 2, 3]), "agent2": np.array([1, 2, 3])}
    action = {"agent1": np.array([4, 5]), "agent2": np.array([4, 5])}
    reward = {"agent1": np.array([6]), "agent2": np.array([7])}
    state2 = {"agent1": np.array([3, 2, 1]), "agent2": np.array([3, 2, 1])}
    action2 = {"agent1": np.array([5, 4]), "agent2": np.array([5, 4])}
    reward2 = {"agent1": np.array([7]), "agent2": np.array([6])}

    buffer._add(state, action, reward)
    buffer._add(state2, action2, reward2)

    assert len(buffer.memory) == 2
    assert buffer.memory[0].state == state
    assert buffer.memory[0].action == action
    assert buffer.memory[0].reward == reward
    assert buffer.memory[1].state == state2
    assert buffer.memory[1].action == action2
    assert buffer.memory[1].reward == reward2


# Can add an experience when memory is full and maxlen is reached
def test_add_experience_when_memory_full():
    memory_size = 2
    field_names = ["state", "action", "reward"]
    agent_ids = ["agent1", "agent2"]

    buffer = MultiAgentReplayBuffer(memory_size, field_names, agent_ids)

    state = {"agent1": np.array([1, 2, 3]), "agent2": np.array([1, 2, 3])}
    action = {"agent1": np.array([4, 5]), "agent2": np.array([4, 5])}
    reward = {"agent1": np.array([6]), "agent2": np.array([7])}
    state2 = {"agent1": np.array([3, 2, 1]), "agent2": np.array([3, 2, 1])}
    action2 = {"agent1": np.array([5, 4]), "agent2": np.array([5, 4])}
    reward2 = {"agent1": np.array([7]), "agent2": np.array([6])}
    state3 = {"agent1": np.array([7, 8, 9]), "agent2": np.array([9, 8, 7])}
    action3 = {"agent1": np.array([6, 5]), "agent2": np.array([5, 6])}
    reward3 = {"agent1": np.array([4]), "agent2": np.array([3])}

    buffer._add(state, action, reward)
    buffer._add(state2, action2, reward2)
    buffer._add(state3, action3, reward3)

    assert len(buffer.memory) == 2
    assert buffer.memory[0].state == state2
    assert buffer.memory[0].action == action2
    assert buffer.memory[0].reward == reward2
    assert buffer.memory[1].state == state3
    assert buffer.memory[1].action == action3
    assert buffer.memory[1].reward == reward3


# Can add experiences to memory using save2memory method
def test_add_experiences_to_memory():
    memory_size = 100
    field_names = ["state", "action", "reward"]
    agent_ids = ["agent1", "agent2"]

    buffer = MultiAgentReplayBuffer(memory_size, field_names, agent_ids)

    state = {"agent1": np.array([1, 2, 3]), "agent2": np.array([1, 2, 3])}
    action = {"agent1": np.array([4, 5]), "agent2": np.array([4, 5])}
    reward = {"agent1": np.array([6]), "agent2": np.array([7])}

    buffer.save2memory(state, action, reward)

    assert len(buffer) == 1
    assert buffer.memory[0].state == state
    assert buffer.memory[0].action == action
    assert buffer.memory[0].reward == reward


# Can add single experiences to memory with save2memorySingleEnv method
def test_add_single_experiences_to_memory():
    memory_size = 100
    field_names = ["state", "action", "reward"]
    agent_ids = ["agent1", "agent2"]
    device = "cpu"

    buffer = MultiAgentReplayBuffer(memory_size, field_names, agent_ids, device)

    state = {"agent1": np.array([1, 2, 3]), "agent2": np.array([1, 2, 3])}
    action = {"agent1": np.array([4, 5]), "agent2": np.array([4, 5])}
    reward = {"agent1": np.array([6]), "agent2": np.array([7])}

    buffer.save2memorySingleEnv(state, action, reward)

    assert len(buffer.memory) == 1
    assert buffer.memory[0].state == state
    assert buffer.memory[0].action == action
    assert buffer.memory[0].reward == reward


# Can add multiple experiences to memory with save2memoryVectEnvs method
def test_add_multiple_experiences_to_memory():
    memory_size = 100
    field_names = ["state", "action", "reward"]
    agent_ids = ["agent1", "agent2"]
    device = "cpu"

    buffer = MultiAgentReplayBuffer(memory_size, field_names, agent_ids, device)

    states = {
        "agent1": np.array([[1, 2], [3, 4]]),
        "agent2": np.array([[1, 2], [3, 4]]),
    }
    actions = {"agent1": np.array([[0], [1]]), "agent2": np.array([[0], [1]])}
    rewards = {"agent1": np.array([[0], [1]]), "agent2": np.array([[0], [1]])}

    buffer.save2memoryVectEnvs(states, actions, rewards)

    state1 = {"agent1": np.array([1, 2]), "agent2": np.array([1, 2])}
    action1 = {"agent1": np.array([0]), "agent2": np.array([0])}
    reward1 = {"agent1": np.array([0]), "agent2": np.array([0])}

    state2 = {"agent1": np.array([3, 4]), "agent2": np.array([3, 4])}
    action2 = {"agent1": np.array([1]), "agent2": np.array([1])}
    reward2 = {"agent1": np.array([1]), "agent2": np.array([1])}

    assert len(buffer.memory) == 2
    assert str(buffer.memory[0].state) == str(state1)
    assert str(buffer.memory[0].action) == str(action1)
    assert str(buffer.memory[0].reward) == str(reward1)
    assert str(buffer.memory[1].state) == str(state2)
    assert str(buffer.memory[1].action) == str(action2)
    assert str(buffer.memory[1].reward) == str(reward2)


# Can handle vectorized and un-vectorized experiences from environment
def test_add_any_experiences_to_memory():
    memory_size = 100
    field_names = ["state", "action", "reward"]
    agent_ids = ["agent1", "agent2"]
    device = "cpu"

    buffer = MultiAgentReplayBuffer(memory_size, field_names, agent_ids, device)

    states = {
        "agent1": np.array([[1, 2], [3, 4]]),
        "agent2": np.array([[1, 2], [3, 4]]),
    }
    actions = {"agent1": np.array([[0], [1]]), "agent2": np.array([[0], [1]])}
    rewards = {"agent1": np.array([[0], [1]]), "agent2": np.array([[0], [1]])}

    buffer.save2memory(states, actions, rewards, is_vectorised=True)

    state1 = {"agent1": np.array([1, 2]), "agent2": np.array([1, 2])}
    action1 = {"agent1": np.array([0]), "agent2": np.array([0])}
    reward1 = {"agent1": np.array([0]), "agent2": np.array([0])}

    state2 = {"agent1": np.array([3, 4]), "agent2": np.array([3, 4])}
    action2 = {"agent1": np.array([1]), "agent2": np.array([1])}
    reward2 = {"agent1": np.array([1]), "agent2": np.array([1])}

    assert len(buffer.memory) == 2
    assert str(buffer.memory[0].state) == str(state1)
    assert str(buffer.memory[0].action) == str(action1)
    assert str(buffer.memory[0].reward) == str(reward1)
    assert str(buffer.memory[1].state) == str(state2)
    assert str(buffer.memory[1].action) == str(action2)
    assert str(buffer.memory[1].reward) == str(reward2)

    new_state = {"agent1": np.array([1, 2]), "agent2": np.array([1, 2])}
    new_action = {"agent1": np.array([0]), "agent2": np.array([0])}
    new_reward = {"agent1": np.array([0]), "agent2": np.array([0])}

    buffer.save2memory(new_state, new_action, new_reward, is_vectorised=False)

    assert len(buffer.memory) == 3
    assert str(buffer.memory[2].state) == str(new_state)
    assert str(buffer.memory[2].action) == str(new_action)
    assert str(buffer.memory[2].reward) == str(new_reward)


# Can sample experiences from memory using sample method
def test_sample_experiences_from_memory():
    memory_size = 100
    field_names = ["state", "action", "reward"]
    agent_ids = ["agent1", "agent2"]

    buffer = MultiAgentReplayBuffer(memory_size, field_names, agent_ids)

    state = {"agent1": np.array([1, 2, 3]), "agent2": np.array([1, 2, 3])}
    action = {"agent1": np.array([4, 5]), "agent2": np.array([4, 5])}
    reward = {"agent1": np.array([6]), "agent2": np.array([7])}

    buffer.save2memory(state, action, reward)

    batch_size = 1
    transition = buffer.sample(batch_size)

    assert len(transition) == len(field_names)
    assert transition[0]["agent1"].shape == (batch_size,) + state["agent1"].shape
    assert transition[1]["agent1"].shape == (batch_size,) + action["agent1"].shape
    assert transition[2]["agent1"].shape == (batch_size,) + reward["agent1"].shape


# Can sample experiences from memory using sample method
def test_sample_experiences_from_memory_images():
    memory_size = 100
    field_names = ["state", "action", "reward"]
    agent_ids = ["agent1", "agent2"]

    buffer = MultiAgentReplayBuffer(memory_size, field_names, agent_ids)

    state = {
        "agent1": np.random.rand(3, 128, 128),
        "agent2": np.random.rand(3, 128, 128),
    }
    action = {"agent1": np.array([4, 5]), "agent2": np.array([4, 5])}
    reward = {"agent1": np.array([6]), "agent2": np.array([7])}

    buffer.save2memory(state, action, reward)

    batch_size = 1
    transition = buffer.sample(batch_size)

    assert len(transition) == len(field_names)
    assert transition[0]["agent1"].shape == (batch_size,) + state["agent1"].shape
    assert transition[1]["agent1"].shape == (batch_size,) + action["agent1"].shape
    assert transition[2]["agent1"].shape == (batch_size,) + reward["agent1"].shape


# Can process a transition from experiences and return a dictionary of numpy arrays.
def test_returns_np_transition_dictionary():
    memory_size = 100
    field_names = ["state", "action", "reward", "next_state", "done"]
    agent_ids = ["agent1", "agent2"]

    # Initialize the class object
    buffer = MultiAgentReplayBuffer(
        memory_size=memory_size, field_names=field_names, agent_ids=agent_ids
    )

    # Create some dummy experiences
    experience1 = buffer.experience(
        state={"agent1": np.array([1]), "agent2": np.array([4])},
        action={"agent1": np.array([0]), "agent2": np.array([1])},
        reward={"agent1": np.array([0]), "agent2": np.array([1])},
        next_state={"agent1": np.array([3]), "agent2": np.array([6])},
        done={"agent1": np.array([True]), "agent2": np.array([False])},
    )
    experience2 = buffer.experience(
        state={"agent1": np.array([4]), "agent2": np.array([7])},
        action={"agent1": np.array([1]), "agent2": np.array([0])},
        reward={"agent1": np.array([1]), "agent2": np.array([0])},
        next_state={"agent1": np.array([6]), "agent2": np.array([9])},
        done={"agent1": np.array([False]), "agent2": np.array([True])},
    )
    experiences = [experience1, experience2]

    # Call the method under test
    transition = buffer._process_transition(experiences, np_array=True)

    # Check the transition dictionary returned
    assert isinstance(transition, dict)
    assert set(transition.keys()) == set(field_names)
    assert isinstance(transition["state"], dict)
    assert isinstance(transition["action"], dict)
    assert isinstance(transition["reward"], dict)
    assert isinstance(transition["next_state"], dict)
    assert isinstance(transition["done"], dict)
    assert set(transition["state"].keys()) == set(agent_ids)
    assert set(transition["action"].keys()) == set(agent_ids)
    assert set(transition["reward"].keys()) == set(agent_ids)
    assert set(transition["next_state"].keys()) == set(agent_ids)
    assert set(transition["done"].keys()) == set(agent_ids)
    assert isinstance(transition["state"]["agent1"], np.ndarray)
    assert isinstance(transition["state"]["agent2"], np.ndarray)
    assert isinstance(transition["action"]["agent1"], np.ndarray)
    assert isinstance(transition["action"]["agent2"], np.ndarray)
    assert isinstance(transition["reward"]["agent1"], np.ndarray)
    assert isinstance(transition["reward"]["agent2"], np.ndarray)
    assert isinstance(transition["next_state"]["agent1"], np.ndarray)
    assert isinstance(transition["next_state"]["agent2"], np.ndarray)
    assert isinstance(transition["done"]["agent1"], np.ndarray)
    assert isinstance(transition["done"]["agent2"], np.ndarray)
    assert transition["state"]["agent1"].shape == (len(experiences), 1)
    assert transition["action"]["agent1"].shape == (len(experiences), 1)
    assert transition["reward"]["agent1"].shape == (len(experiences), 1)
    assert transition["next_state"]["agent1"].shape == (len(experiences), 1)
    assert transition["done"]["agent1"].shape == (len(experiences), 1)
    assert np.array_equal(transition["state"]["agent1"], np.array([[1], [4]]))
    assert np.array_equal(transition["state"]["agent2"], np.array([[4], [7]]))
    assert np.array_equal(transition["action"]["agent1"], np.array([[0], [1]]))
    assert np.array_equal(transition["action"]["agent2"], np.array([[1], [0]]))
    assert np.array_equal(transition["reward"]["agent1"], np.array([[0], [1]]))
    assert np.array_equal(transition["reward"]["agent2"], np.array([[1], [0]]))
    assert np.array_equal(transition["next_state"]["agent1"], np.array([[3], [6]]))
    assert np.array_equal(transition["next_state"]["agent2"], np.array([[6], [9]]))
    assert np.array_equal(transition["done"]["agent1"], np.array([[1], [0]]))
    assert np.array_equal(transition["done"]["agent2"], np.array([[0], [1]]))


# Can process a transition from a single experience and return a dictionary of numpy arrays.
def test_returns_np_transition_asingle_experience_dictionary():
    memory_size = 100
    field_names = ["state", "action", "reward", "next_state", "done"]
    agent_ids = ["agent1", "agent2"]

    # Initialize the class object
    buffer = MultiAgentReplayBuffer(
        memory_size=memory_size, field_names=field_names, agent_ids=agent_ids
    )

    # Create some dummy experiences
    experience1 = buffer.experience(
        state={"agent1": np.array([1]), "agent2": np.array([4])},
        action={"agent1": np.array([0]), "agent2": np.array([1])},
        reward={"agent1": np.array([0]), "agent2": np.array([1])},
        next_state={"agent1": np.array([3]), "agent2": np.array([6])},
        done={"agent1": np.array([True]), "agent2": np.array([False])},
    )
    experiences = [experience1]

    # Call the method under test
    transition = buffer._process_transition(experiences, np_array=True)

    # Check the transition dictionary returned
    assert isinstance(transition, dict)
    assert set(transition.keys()) == set(field_names)
    assert isinstance(transition["state"], dict)
    assert isinstance(transition["action"], dict)
    assert isinstance(transition["reward"], dict)
    assert isinstance(transition["next_state"], dict)
    assert isinstance(transition["done"], dict)
    assert set(transition["state"].keys()) == set(agent_ids)
    assert set(transition["action"].keys()) == set(agent_ids)
    assert set(transition["reward"].keys()) == set(agent_ids)
    assert set(transition["next_state"].keys()) == set(agent_ids)
    assert set(transition["done"].keys()) == set(agent_ids)
    assert isinstance(transition["state"]["agent1"], np.ndarray)
    assert isinstance(transition["state"]["agent2"], np.ndarray)
    assert isinstance(transition["action"]["agent1"], np.ndarray)
    assert isinstance(transition["action"]["agent2"], np.ndarray)
    assert isinstance(transition["reward"]["agent1"], np.ndarray)
    assert isinstance(transition["reward"]["agent2"], np.ndarray)
    assert isinstance(transition["next_state"]["agent1"], np.ndarray)
    assert isinstance(transition["next_state"]["agent2"], np.ndarray)
    assert isinstance(transition["done"]["agent1"], np.ndarray)
    assert isinstance(transition["done"]["agent2"], np.ndarray)
    assert transition["state"]["agent1"].shape == (len(experiences), 1)
    assert transition["action"]["agent1"].shape == (len(experiences), 1)
    assert transition["reward"]["agent1"].shape == (len(experiences), 1)
    assert transition["next_state"]["agent1"].shape == (len(experiences), 1)
    assert transition["done"]["agent1"].shape == (len(experiences), 1)
    assert np.array_equal(transition["state"]["agent1"], np.array([[1]]))
    assert np.array_equal(transition["state"]["agent2"], np.array([[4]]))
    assert np.array_equal(transition["action"]["agent1"], np.array([[0]]))
    assert np.array_equal(transition["action"]["agent2"], np.array([[1]]))
    assert np.array_equal(transition["reward"]["agent1"], np.array([[0]]))
    assert np.array_equal(transition["reward"]["agent2"], np.array([[1]]))
    assert np.array_equal(transition["next_state"]["agent1"], np.array([[3]]))
    assert np.array_equal(transition["next_state"]["agent2"], np.array([[6]]))
    assert np.array_equal(transition["done"]["agent1"], np.array([[1]]))
    assert np.array_equal(transition["done"]["agent2"], np.array([[0]]))


# Can process a transition from experiences and return a dictionary of numpy arrays.
def test_returns_np_transition_dictionary_images():
    memory_size = 100
    field_names = ["state", "action", "reward", "next_state", "done"]
    agent_ids = ["agent1", "agent2"]

    # Initialize the class object
    buffer = MultiAgentReplayBuffer(
        memory_size=memory_size, field_names=field_names, agent_ids=agent_ids
    )

    # Create some dummy experiences
    experience1 = buffer.experience(
        state={"agent1": np.random.rand(3, 128, 128), "agent2": np.array([4])},
        action={"agent1": np.array([0]), "agent2": np.array([1])},
        reward={"agent1": np.array([0]), "agent2": np.array([1])},
        next_state={"agent1": np.random.rand(3, 128, 128), "agent2": np.array([6])},
        done={"agent1": np.array([True]), "agent2": np.array([False])},
    )
    experience2 = buffer.experience(
        state={"agent1": np.random.rand(3, 128, 128), "agent2": np.array([7])},
        action={"agent1": np.array([1]), "agent2": np.array([0])},
        reward={"agent1": np.array([1]), "agent2": np.array([0])},
        next_state={"agent1": np.random.rand(3, 128, 128), "agent2": np.array([9])},
        done={"agent1": np.array([False]), "agent2": np.array([True])},
    )
    experiences = [experience1, experience2]

    # Call the method under test
    transition = buffer._process_transition(experiences, np_array=True)

    # Check the transition dictionary returned
    assert isinstance(transition, dict)
    assert set(transition.keys()) == set(field_names)
    assert isinstance(transition["state"], dict)
    assert isinstance(transition["action"], dict)
    assert isinstance(transition["reward"], dict)
    assert isinstance(transition["next_state"], dict)
    assert isinstance(transition["done"], dict)
    assert set(transition["state"].keys()) == set(agent_ids)
    assert set(transition["action"].keys()) == set(agent_ids)
    assert set(transition["reward"].keys()) == set(agent_ids)
    assert set(transition["next_state"].keys()) == set(agent_ids)
    assert set(transition["done"].keys()) == set(agent_ids)
    assert isinstance(transition["state"]["agent1"], np.ndarray)
    assert isinstance(transition["state"]["agent2"], np.ndarray)
    assert isinstance(transition["action"]["agent1"], np.ndarray)
    assert isinstance(transition["action"]["agent2"], np.ndarray)
    assert isinstance(transition["reward"]["agent1"], np.ndarray)
    assert isinstance(transition["reward"]["agent2"], np.ndarray)
    assert isinstance(transition["next_state"]["agent1"], np.ndarray)
    assert isinstance(transition["next_state"]["agent2"], np.ndarray)
    assert isinstance(transition["done"]["agent1"], np.ndarray)
    assert isinstance(transition["done"]["agent2"], np.ndarray)
    assert transition["state"]["agent1"].shape == (len(experiences), 3, 128, 128)
    assert transition["action"]["agent1"].shape == (len(experiences), 1)
    assert transition["reward"]["agent1"].shape == (len(experiences), 1)
    assert transition["next_state"]["agent1"].shape == (len(experiences), 3, 128, 128)
    assert transition["done"]["agent1"].shape == (len(experiences), 1)
    assert np.array_equal(transition["action"]["agent1"], np.array([[0], [1]]))
    assert np.array_equal(transition["action"]["agent2"], np.array([[1], [0]]))
    assert np.array_equal(transition["reward"]["agent1"], np.array([[0], [1]]))
    assert np.array_equal(transition["reward"]["agent2"], np.array([[1], [0]]))
    assert np.array_equal(transition["done"]["agent1"], np.array([[1], [0]]))
    assert np.array_equal(transition["done"]["agent2"], np.array([[0], [1]]))


# Can process a transition from experiences and return a dictionary of torch tensors.
def test_returns_torch_transition_dictionary():
    memory_size = 100
    field_names = ["state", "action", "reward", "next_state", "done"]
    agent_ids = ["agent1", "agent2"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the class object
    buffer = MultiAgentReplayBuffer(
        memory_size=memory_size,
        field_names=field_names,
        agent_ids=agent_ids,
        device=device,
    )

    # Create some dummy experiences
    experience1 = buffer.experience(
        state={"agent1": np.array([1, 2, 3]), "agent2": np.array([4, 5, 6])},
        action={"agent1": np.array([0]), "agent2": np.array([1])},
        reward={"agent1": np.array([0]), "agent2": np.array([1])},
        next_state={"agent1": np.array([3, 2, 1]), "agent2": np.array([6, 5, 4])},
        done={"agent1": np.array([True]), "agent2": np.array([False])},
    )
    experience2 = buffer.experience(
        state={"agent1": np.array([4, 5, 6]), "agent2": np.array([7, 8, 9])},
        action={"agent1": np.array([1]), "agent2": np.array([0])},
        reward={"agent1": np.array([1]), "agent2": np.array([0])},
        next_state={"agent1": np.array([6, 5, 4]), "agent2": np.array([9, 8, 7])},
        done={"agent1": np.array([False]), "agent2": np.array([True])},
    )
    experiences = [experience1, experience2]

    # Call the method under test
    transition = buffer._process_transition(experiences)

    # Check the transition dictionary returned
    assert isinstance(transition, dict)
    assert set(transition.keys()) == set(field_names)
    assert isinstance(transition["state"], dict)
    assert isinstance(transition["action"], dict)
    assert isinstance(transition["reward"], dict)
    assert isinstance(transition["next_state"], dict)
    assert isinstance(transition["done"], dict)
    assert set(transition["state"].keys()) == set(agent_ids)
    assert set(transition["action"].keys()) == set(agent_ids)
    assert set(transition["reward"].keys()) == set(agent_ids)
    assert set(transition["next_state"].keys()) == set(agent_ids)
    assert set(transition["done"].keys()) == set(agent_ids)
    assert isinstance(transition["state"]["agent1"], torch.Tensor)
    assert isinstance(transition["state"]["agent2"], torch.Tensor)
    assert isinstance(transition["action"]["agent1"], torch.Tensor)
    assert isinstance(transition["action"]["agent2"], torch.Tensor)
    assert isinstance(transition["reward"]["agent1"], torch.Tensor)
    assert isinstance(transition["reward"]["agent2"], torch.Tensor)
    assert isinstance(transition["next_state"]["agent1"], torch.Tensor)
    assert isinstance(transition["next_state"]["agent2"], torch.Tensor)
    assert isinstance(transition["done"]["agent1"], torch.Tensor)
    assert isinstance(transition["done"]["agent2"], torch.Tensor)
    assert transition["state"]["agent1"].shape == (len(experiences), 3)
    assert transition["action"]["agent1"].shape == (len(experiences), 1)
    assert transition["reward"]["agent1"].shape == (len(experiences), 1)
    assert transition["next_state"]["agent1"].shape == (len(experiences), 3)
    assert transition["done"]["agent1"].shape == (len(experiences), 1)
    assert torch.equal(
        transition["state"]["agent1"],
        torch.from_numpy(np.array([[1, 2, 3], [4, 5, 6]])).to(device),
    )
    assert torch.equal(
        transition["state"]["agent2"],
        torch.from_numpy(np.array([[4, 5, 6], [7, 8, 9]])).to(device),
    )
    assert torch.equal(
        transition["action"]["agent1"],
        torch.from_numpy(np.array([[0], [1]])).to(device),
    )
    assert torch.equal(
        transition["action"]["agent2"],
        torch.from_numpy(np.array([[1], [0]])).to(device),
    )
    assert torch.equal(
        transition["reward"]["agent1"],
        torch.from_numpy(np.array([[0], [1]])).to(device),
    )
    assert torch.equal(
        transition["reward"]["agent2"],
        torch.from_numpy(np.array([[1], [0]])).to(device),
    )
    assert torch.equal(
        transition["next_state"]["agent1"],
        torch.from_numpy(np.array([[3, 2, 1], [6, 5, 4]])).to(device),
    )
    assert torch.equal(
        transition["next_state"]["agent2"],
        torch.from_numpy(np.array([[6, 5, 4], [9, 8, 7]])).to(device),
    )
    assert torch.equal(
        transition["done"]["agent1"], torch.from_numpy(np.array([[1], [0]])).to(device)
    )
    assert torch.equal(
        transition["done"]["agent2"], torch.from_numpy(np.array([[0], [1]])).to(device)
    )
