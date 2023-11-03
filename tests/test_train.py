from unittest.mock import ANY, MagicMock, patch

import numpy as np
import pytest
from accelerate import Accelerator

import agilerl.training.train
import agilerl.training.train_on_policy
from agilerl.algorithms.ddpg import DDPG
from agilerl.algorithms.dqn import DQN
from agilerl.algorithms.dqn_rainbow import RainbowDQN
from agilerl.algorithms.ppo import PPO
from agilerl.algorithms.td3 import TD3
from agilerl.training.train import train
from agilerl.training.train_on_policy import train_on_policy


class DummyEnv:
    def __init__(self, state_size, action_size, vect=True, num_envs=2):
        self.state_size = state_size
        self.action_size = action_size
        self.vect = vect
        if self.vect:
            self.state_size = (num_envs,) + self.state_size
            self.n_envs = num_envs
            self.num_envs = num_envs
        else:
            self.n_envs = 1

    def reset(self):
        return np.random.rand(*self.state_size), "info_string"

    def step(self, action):
        return (
            np.random.rand(*self.state_size),
            np.random.randint(0, 5, self.n_envs),
            np.random.randint(0, 2, self.n_envs),
            np.random.randint(0, 2, self.n_envs),
            "info_string",
        )


class DummyAgentOffPolicy:
    def __init__(self, batch_size, env, beta=None):
        self.state_size = env.state_size
        self.action_size = env.action_size
        self.batch_size = batch_size
        self.beta = beta
        self.learn_step = 1
        self.scores = []
        self.steps = [0]
        self.fitness = []
        self.mut = "mutation"
        self.index = 1

    def getAction(self, *args):
        return np.random.randn(self.action_size)

    def learn(self, experiences, n_step=False, per=False):
        if n_step and per:
            return None, None
        else:
            return

    def test(self, env, swap_channels, max_steps, loop):
        rand_int = np.random.uniform(0, 400)
        self.fitness.append(rand_int)
        return rand_int

    def saveCheckpoint(self, *args):
        return

    def loadCheckpoint(self, *args):
        return

    def wrap_models(self, *args):
        return

    def unwrap_models(self, *args):
        return


class DummyAgentOnPolicy(DummyAgentOffPolicy):
    def __init__(self, batch_size, env):
        super().__init__(batch_size, env)

    def getAction(self, *args):
        return tuple(np.random.randn(self.action_size) for _ in range(4))

    def test(self, env, swap_channels, max_steps, loop):
        return super().test(env, swap_channels, max_steps, loop)

    def saveCheckpoint(self, *args):
        return

    def loadCheckpoint(self, *args):
        return

    def wrap_models(self, *args):
        return

    def unwrap_models(self, *args):
        return


class DummyTournament:
    def __init__(self):
        pass

    def select(self, pop):
        return pop[0], pop


class DummyMutations:
    def __init__(self):
        pass

    def mutation(self, pop, pre_training_mut=False):
        return pop


class DummyMemory:
    def __init__(self):
        self.counter = 0
        self.state_size = None
        self.action_size = None
        self.next_state_size = None

    def save2memoryVectEnvs(self, states, actions, rewards, next_states, dones):
        if self.state_size is None:
            self.state_size, *_ = (state.shape for state in states)
            self.action_size, *_ = (action.shape for action in actions)
            self.next_state_size, *_ = (next_state.shape for next_state in next_states)
            self.counter += 1

        one_step_transition = (
            np.random.randn(*self.state_size),
            np.random.randn(*self.action_size),
            np.random.uniform(0, 400),
            np.random.choice([True, False]),
            np.random.randn(*self.next_state_size),
        )

        return one_step_transition

    def save2memory(self, state, action, reward, next_state, done, is_vectorised):
        if is_vectorised:
            self.save2memoryVectEnvs(state, action, reward, next_state, done)
        else:
            self.state_size = state.shape
            self.action_size = action.shape
            self.next_state_size = next_state.shape
            self.counter += 1

    def __len__(self):
        return 1000

    def sample(self, batch_size, beta=None, *args):
        # Account for sample_from_indices
        if isinstance(batch_size, list):
            batch_size = len(batch_size)

        if batch_size == 1:
            states = np.random.randn(*self.state_size)
            actions = np.random.randn(*self.action_size)
            rewards = np.random.uniform(0, 400)
            dones = np.random.choice([True, False])
            next_states = np.random.randn(*self.next_state_size)
        else:
            states = np.array(
                [np.random.randn(*self.state_size) for _ in range(batch_size)]
            )
            actions = np.array(
                [np.random.randn(*self.action_size) for _ in range(batch_size)]
            )
            rewards = np.array([np.random.uniform(0, 400) for _ in range(batch_size)])
            dones = np.array(
                [np.random.choice([True, False]) for _ in range(batch_size)]
            )
            next_states = np.array(
                [np.random.randn(*self.next_state_size) for _ in range(batch_size)]
            )

        if beta is None:
            return states, actions, rewards, dones, next_states

        idxs = [np.random.randn(1) for _ in range(batch_size)]
        weights = [i for i in range(batch_size)]

        return states, actions, rewards, dones, next_states, weights, idxs

    def update_priorities(self, idxs, priorities):
        return


class DummyNStepMemory(DummyMemory):
    def __init__(self):
        super().__init__()
        self.counter = 0

    def save2memory(self, state, action, reward, next_state, done, is_vectorised):
        return super().save2memory(state, action, reward, next_state, done)

    def save2memoryVectEnvs(self, state, action, reward, next_state, done):
        self.state_size = state.shape
        self.action_size = action.shape
        self.next_state_size = next_state.shape
        self.counter += 1

        one_step_transition = (
            np.random.randn(*self.state_size),
            np.random.randn(*self.action_size),
            np.random.uniform(0, 400),
            np.random.randn(*self.next_state_size),
            np.random.choice([True, False]),
        )

        return one_step_transition

    def __len__(self):
        return super().__len__()

    def sample_n_step(self, *args):
        return super().sample(*args)

    def sample_per(self, *args):
        return super().sample(*args)

    def sample_from_indices(self, *args):
        return super().sample(*args)


@pytest.fixture
def env(state_size, action_size, vect):
    return DummyEnv(state_size, action_size, vect)


@pytest.fixture
def population_off_policy(env):
    return [DummyAgentOffPolicy(5, env, 0.4) for _ in range(6)]


@pytest.fixture
def population_on_policy(env):
    return [DummyAgentOnPolicy(5, env) for _ in range(6)]


@pytest.fixture
def tournament():
    return DummyTournament()


@pytest.fixture
def mutations():
    return DummyMutations()


@pytest.fixture
def memory():
    return DummyMemory()


@pytest.fixture
def n_step_memory():
    return DummyNStepMemory()


@pytest.fixture
def mocked_agent_off_policy(env, algo):
    mock_agent = MagicMock(spec=algo)
    mock_agent.learn_step = 1
    mock_agent.batch_size = 5
    mock_agent.state_size = env.state_size
    mock_agent.action_size = env.action_size
    mock_agent.beta = 0.4
    mock_agent.scores = []
    mock_agent.steps = [0]
    mock_agent.fitness = []
    mock_agent.mut = "mutation"
    mock_agent.index = 1
    mock_agent.getAction.side_effect = lambda state: np.random.randn(env.action_size)
    mock_agent.test.side_effect = lambda *args, **kwargs: np.random.uniform(0, 400)
    mock_agent.learn.side_effect = lambda experiences: None
    mock_agent.saveCheckpoint.side_effect = lambda *args, **kwargs: None
    mock_agent.loadCheckpoint.side_effect = lambda *args, **kwargs: None
    mock_agent.wrap_models.side_effect = lambda *args, **kwargs: None
    mock_agent.unwrap_models.side_effect = lambda *args, **kwargs: None

    return mock_agent


@pytest.fixture
def mocked_agent_on_policy(env, algo):
    mock_agent = MagicMock(spec=algo)
    mock_agent.learn_step = 1
    mock_agent.batch_size = 5
    mock_agent.state_size = env.state_size
    mock_agent.action_size = env.action_size
    mock_agent.beta = 0.4
    mock_agent.scores = []
    mock_agent.steps = [0]
    mock_agent.fitness = []
    mock_agent.mut = "mutation"
    mock_agent.index = 1
    mock_agent.getAction.side_effect = lambda state: tuple(
        np.random.randn(env.action_size) for _ in range(4)
    )
    mock_agent.test.side_effect = lambda *args, **kwargs: np.random.uniform(0, 400)
    mock_agent.learn.side_effect = lambda experiences: None
    mock_agent.saveCheckpoint.side_effect = lambda *args, **kwargs: None
    mock_agent.loadCheckpoint.side_effect = lambda *args, **kwargs: None
    mock_agent.wrap_models.side_effect = lambda *args, **kwargs: None
    mock_agent.unwrap_models.side_effect = lambda *args, **kwargs: None

    return mock_agent


@pytest.fixture
def mocked_memory():
    mock_memory = MagicMock()
    mock_memory.counter = 0
    mock_memory.state_size = None
    mock_memory.action_size = None
    mock_memory.next_state_size = None
    mock_memory.__len__.return_value = 10

    def save2memoryVectEnvs(states, actions, rewards, next_states, dones):
        if mock_memory.state_size is None:
            mock_memory.state_size, *_ = (state.shape for state in states)
            mock_memory.action_size, *_ = (action.shape for action in actions)
            mock_memory.next_state_size, *_ = (
                next_state.shape for next_state in next_states
            )
            mock_memory.counter += 1

        one_step_transition = (
            np.random.randn(*mock_memory.state_size),
            np.random.randn(*mock_memory.action_size),
            np.random.uniform(0, 400),
            np.random.choice([True, False]),
            np.random.randn(*mock_memory.next_state_size),
        )
        return one_step_transition

    mock_memory.save2memoryVectEnvs.side_effect = save2memoryVectEnvs

    def save2memory(state, action, reward, next_state, done, is_vectorised):
        if is_vectorised:
            mock_memory.save2memoryVectEnvs(state, action, reward, next_state, done)
        else:
            mock_memory.state_size = state.shape
            mock_memory.action_size = action.shape
            mock_memory.next_state_size = next_state.shape
            mock_memory.counter += 1

    # Assigning the save2memory function to the MagicMock
    mock_memory.save2memory.side_effect = save2memory

    def sample(batch_size, beta=None, *args):
        # Account for sample_from_indices
        if isinstance(batch_size, list):
            batch_size = len(batch_size)

        if batch_size == 1:
            states = np.random.randn(*mock_memory.state_size)
            actions = np.random.randn(*mock_memory.action_size)
            rewards = np.random.uniform(0, 400)
            dones = np.random.choice([True, False])
            next_states = np.random.randn(*mock_memory.next_state_size)
        else:
            states = np.array(
                [np.random.randn(*mock_memory.state_size) for _ in range(batch_size)]
            )
            actions = np.array(
                [np.random.randn(*mock_memory.action_size) for _ in range(batch_size)]
            )
            rewards = np.array([np.random.uniform(0, 400) for _ in range(batch_size)])
            dones = np.array(
                [np.random.choice([True, False]) for _ in range(batch_size)]
            )
            next_states = np.array(
                [
                    np.random.randn(*mock_memory.next_state_size)
                    for _ in range(batch_size)
                ]
            )
        if beta is None:
            return states, actions, rewards, dones, next_states

        idxs = [np.random.randn(1) for _ in range(batch_size)]
        weights = [i for i in range(batch_size)]

        return states, actions, rewards, dones, next_states, weights, idxs

    # Assigning the sample function to the MagicMock
    mock_memory.sample.side_effect = sample

    def update_priorities(idxs, priorities):
        return None

    mock_memory.update_priorities.side_effect = update_priorities

    return mock_memory


@pytest.fixture
def mocked_n_step_memory():
    mock_memory = MagicMock()
    mock_memory.counter = 0
    mock_memory.state_size = None
    mock_memory.action_size = None
    mock_memory.next_state_size = None
    mock_memory.__len__.return_value = 10000

    def save2memoryVectEnvs(state, action, reward, next_state, done):
        mock_memory.state_size = state.shape
        mock_memory.action_size = action.shape
        mock_memory.next_state_size = next_state.shape
        mock_memory.counter += 1

        one_step_transition = (
            np.random.randn(*mock_memory.state_size),
            np.random.randn(*mock_memory.action_size),
            np.random.uniform(0, 400),
            np.random.randn(*mock_memory.next_state_size),
            np.random.choice([True, False]),
        )
        return one_step_transition

    mock_memory.save2memoryVectEnvs.side_effect = save2memoryVectEnvs

    def save2memory(state, action, reward, next_state, done, is_vectorised):
        if is_vectorised:
            mock_memory.save2memoryVectEnvs(state, action, reward, next_state, done)
        else:
            mock_memory.state_size = state.shape
            mock_memory.action_size = action.shape
            mock_memory.next_state_size = next_state.shape
            mock_memory.counter += 1

    # Assigning the save2memory function to the MagicMock
    mock_memory.save2memory.side_effect = save2memory

    def sample(batch_size, beta=None, *args):
        # Account for sample_from_indices
        if isinstance(batch_size, list):
            batch_size = len(batch_size)

        if batch_size == 1:
            states = np.random.randn(*mock_memory.state_size)
            actions = np.random.randn(*mock_memory.action_size)
            rewards = np.random.uniform(0, 400)
            dones = np.random.choice([True, False])
            next_states = np.random.randn(*mock_memory.next_state_size)
        else:
            states = np.array(
                [np.random.randn(*mock_memory.state_size) for _ in range(batch_size)]
            )
            actions = np.array(
                [np.random.randn(*mock_memory.action_size) for _ in range(batch_size)]
            )
            rewards = np.array([np.random.uniform(0, 400) for _ in range(batch_size)])
            dones = np.array(
                [np.random.choice([True, False]) for _ in range(batch_size)]
            )
            next_states = np.array(
                [
                    np.random.randn(*mock_memory.next_state_size)
                    for _ in range(batch_size)
                ]
            )
        if beta is None:
            return states, actions, rewards, dones, next_states

        idxs = [np.random.randn(1) for _ in range(batch_size)]
        weights = [i for i in range(batch_size)]

        return states, actions, rewards, dones, next_states, weights, idxs

    # Assigning the sample function to the MagicMock
    mock_memory.sample.side_effect = sample
    mock_memory.sample_n_step.side_effect = sample
    mock_memory.sample_per.side_effect = sample
    mock_memory.sample_from_indices.side_effect = sample

    return mock_memory


@pytest.fixture
def mocked_env(state_size, action_size, vect=True, num_envs=2):
    mock_env = MagicMock()
    mock_env.state_size = state_size
    mock_env.action_size = action_size
    mock_env.vect = vect
    if mock_env.vect:
        mock_env.state_size = (num_envs,) + mock_env.state_size
        mock_env.n_envs = num_envs
        mock_env.num_envs = num_envs
    else:
        mock_env.n_envs = 1

    def reset():
        return np.random.rand(*mock_env.state_size), "info_string"

    mock_env.reset.side_effect = reset

    def step(action):
        return (
            np.random.rand(*mock_env.state_size),
            np.random.randint(0, 5, mock_env.n_envs),
            np.random.randint(0, 2, mock_env.n_envs),
            np.random.randint(0, 2, mock_env.n_envs),
            "info_string",
        )

    mock_env.step.side_effect = step

    return mock_env


@pytest.fixture
def mocked_mutations():
    mock_mutations = MagicMock()

    def mutation(pop, pre_training_mut=False):
        return pop

    mock_mutations.mutation.side_effect = mutation
    return mock_mutations


@pytest.fixture
def mocked_tournament():
    mock_tournament = MagicMock()

    def select(pop):
        return pop[0], pop

    mock_tournament.select.side_effect = select
    return mock_tournament


@pytest.mark.parametrize(
    "state_size, action_size, vect", [((6,), 2, True), ((6,), 2, False)]
)
def test_train(env, population_off_policy, tournament, mutations, memory):
    pop, pop_fitnesses = train(
        env,
        "env_name",
        "algo",
        population_off_policy,
        memory,
        INIT_HP=None,
        MUT_P=None,
        swap_channels=False,
        n_episodes=10,
        max_steps=5,
        evo_epochs=5,
        evo_loop=1,
        n_step=False,
        per=False,
        noisy=True,
        n_step_memory=None,
        tournament=tournament,
        mutation=mutations,
        wb=False,
    )

    assert len(pop) == len(population_off_policy)


@pytest.mark.parametrize(
    "state_size, action_size, vect, per, n_step, algo",
    [
        ((6,), 2, True, False, False, DQN),
        ((6,), 2, False, False, False, DDPG),
        ((6,), 2, False, False, False, TD3),
        ((6,), 2, True, True, True, RainbowDQN),
    ],
)
def test_train_agent_calls_made(
    env,
    algo,
    mocked_agent_off_policy,
    tournament,
    mutations,
    memory,
    per,
    n_step,
    n_step_memory,
):
    for accelerator_flag in [True, False]:
        if accelerator_flag:
            accelerator = Accelerator()
        else:
            accelerator = None
        if not isinstance(algo, RainbowDQN):
            n_step_memory = None
            per = False
            n_step = False
        mock_population = [mocked_agent_off_policy for _ in range(6)]

        pop, pop_fitnesses = train(
            env,
            "env_name",
            "algo",
            mock_population,
            memory,
            INIT_HP=None,
            MUT_P=None,
            swap_channels=False,
            n_episodes=10,
            max_steps=5,
            evo_epochs=5,
            evo_loop=1,
            n_step=n_step,
            per=per,
            noisy=True,
            n_step_memory=n_step_memory,
            tournament=tournament,
            mutation=mutations,
            wb=False,
            accelerator=accelerator,
        )

        mocked_agent_off_policy.getAction.assert_called()
        mocked_agent_off_policy.learn.assert_called()
        mocked_agent_off_policy.test.assert_called()
        if accelerator is not None:
            mocked_agent_off_policy.saveCheckpoint.assert_called()
            mocked_agent_off_policy.wrap_models.assert_called()
            mocked_agent_off_policy.unwrap_models.assert_called()


@pytest.mark.parametrize("state_size, action_size, vect", [((6,), 2, True)])
def test_train_replay_buffer_calls(
    mocked_memory, env, population_off_policy, tournament, mutations
):
    pop, pop_fitnesses = train(
        env,
        "env_name",
        "algo",
        population_off_policy,
        mocked_memory,
        INIT_HP=None,
        MUT_P=None,
        swap_channels=False,
        n_episodes=10,
        max_steps=5,
        evo_epochs=5,
        evo_loop=1,
        n_step=False,
        per=False,
        noisy=True,
        n_step_memory=None,
        tournament=tournament,
        mutation=mutations,
        wb=False,
    )
    mocked_memory.save2memory.assert_called()
    mocked_memory.sample.assert_called()


@pytest.mark.parametrize(
    "state_size, action_size, vect, per",
    [((6,), 2, True, True), ((6,), 2, True, False)],
)
def test_train_alternate_buffer_calls(
    env,
    mocked_memory,
    population_off_policy,
    tournament,
    mutations,
    mocked_n_step_memory,
    per,
):
    pop, pop_fitnesses = train(
        env,
        "env_name",
        "algo",
        population_off_policy,
        memory=mocked_memory,
        INIT_HP=None,
        MUT_P=None,
        swap_channels=False,
        n_episodes=10,
        max_steps=5,
        evo_epochs=5,
        evo_loop=1,
        n_step=True,
        per=per,
        noisy=False,
        n_step_memory=mocked_n_step_memory,
        tournament=tournament,
        mutation=mutations,
        wb=False,
    )
    mocked_n_step_memory.save2memoryVectEnvs.assert_called()
    mocked_memory.save2memoryVectEnvs.assert_called()
    if per:
        mocked_n_step_memory.sample_from_indices.assert_called()
        mocked_memory.update_priorities.assert_called()
    else:
        mocked_memory.sample.assert_called()


@pytest.mark.parametrize(
    "state_size, action_size, vect",
    [
        ((6,), 2, True),
    ],
)
def test_train_env_calls(
    mocked_env, memory, population_off_policy, tournament, mutations
):
    pop, pop_fitnesses = train(
        mocked_env,
        "env_name",
        "algo",
        population_off_policy,
        memory,
        INIT_HP=None,
        MUT_P=None,
        swap_channels=False,
        n_episodes=10,
        max_steps=5,
        evo_epochs=5,
        evo_loop=1,
        n_step=False,
        per=False,
        noisy=True,
        n_step_memory=None,
        tournament=tournament,
        mutation=mutations,
        wb=False,
    )
    mocked_env.step.assert_called()
    mocked_env.reset.assert_called()


@pytest.mark.parametrize(
    "state_size, action_size, vect",
    [
        ((6,), 2, True),
    ],
)
def test_train_tourn_mut_calls(
    env, memory, population_off_policy, mocked_tournament, mocked_mutations
):
    pop, pop_fitnesses = train(
        env,
        "env_name",
        "algo",
        population_off_policy,
        memory,
        INIT_HP=None,
        MUT_P=None,
        swap_channels=False,
        n_episodes=10,
        max_steps=5,
        evo_epochs=5,
        evo_loop=1,
        n_step=False,
        per=False,
        noisy=True,
        n_step_memory=None,
        tournament=mocked_tournament,
        mutation=mocked_mutations,
        wb=False,
    )
    mocked_mutations.mutation.assert_called()
    mocked_tournament.select.assert_called()


@pytest.mark.parametrize("state_size, action_size, vect", [((250, 160, 3), 2, False)])
def test_train_rgb_input(env, population_off_policy, tournament, mutations, memory):
    pop, pop_fitnesses = train(
        env,
        "env_name",
        "algo",
        population_off_policy,
        memory,
        INIT_HP=None,
        MUT_P=None,
        swap_channels=True,
        n_episodes=10,
        max_steps=5,
        evo_epochs=5,
        evo_loop=1,
        n_step=False,
        per=False,
        noisy=False,
        n_step_memory=None,
        tournament=tournament,
        mutation=mutations,
        wb=False,
    )

    assert len(pop) == len(population_off_policy)


@pytest.mark.parametrize(
    "state_size, action_size, vect, per",
    [((6,), 2, True, True), ((6,), 2, True, False)],
)
def test_train_using_alternate_buffers(
    env, memory, population_off_policy, tournament, mutations, n_step_memory, per
):
    pop, pop_fitnesses = train(
        env,
        "env_name",
        "algo",
        population_off_policy,
        memory=memory,
        INIT_HP=None,
        MUT_P=None,
        swap_channels=False,
        n_episodes=10,
        max_steps=5,
        evo_epochs=5,
        evo_loop=1,
        n_step=True,
        per=per,
        noisy=False,
        n_step_memory=n_step_memory,
        tournament=tournament,
        mutation=mutations,
        wb=False,
    )

    assert len(pop) == len(population_off_policy)


@pytest.mark.parametrize("state_size, action_size, vect", [((3, 64, 64), 2, True)])
def test_train_using_alternate_buffers_rgb(
    env, memory, population_off_policy, tournament, mutations, n_step_memory
):
    pop, pop_fitnesses = train(
        env,
        "env_name",
        "algo",
        population_off_policy,
        memory=memory,
        INIT_HP=None,
        MUT_P=None,
        swap_channels=True,
        n_episodes=10,
        max_steps=5,
        evo_epochs=5,
        evo_loop=1,
        n_step=True,
        per=True,
        noisy=False,
        n_step_memory=n_step_memory,
        tournament=tournament,
        mutation=mutations,
        wb=False,
    )

    assert len(pop) == len(population_off_policy)


@pytest.mark.parametrize(
    "state_size, action_size, vect", [((6,), 2, True), ((6,), 2, False)]
)
def test_train_distributed(env, population_off_policy, tournament, mutations, memory):
    accelerator = Accelerator()
    pop, pop_fitnesses = train(
        env,
        "env_name",
        "algo",
        population_off_policy,
        memory,
        INIT_HP=None,
        MUT_P=None,
        swap_channels=False,
        n_episodes=10,
        max_steps=5,
        evo_epochs=5,
        evo_loop=1,
        n_step=False,
        per=False,
        noisy=True,
        n_step_memory=None,
        tournament=tournament,
        mutation=mutations,
        wb=False,
        accelerator=accelerator,
    )

    assert len(pop) == len(population_off_policy)


@pytest.mark.parametrize("state_size, action_size, vect", [((6,), 2, True)])
def test_wandb_init_log(env, population_off_policy, tournament, mutations, memory):
    INIT_HP = {
        "BATCH_SIZE": 128,
        "LR": 1e-3,
        "GAMMA": 0.99,
        "LEARN_STEP": 1,
        "TAU": 1e-3,
        "CHANNELS_LAST": False,
        "POP_SIZE": 6,
        "MEMORY_SIZE": 20000,
    }
    MUT_P = {
        "NO_MUT": 0.4,
        "ARCH_MUT": 0.2,
        "PARAMS_MUT": 0.2,
        "ACT_MUT": 0.2,
        "RL_HP_MUT": 0.2,
    }
    with patch("agilerl.training.train.wandb.init") as mock_wandb_init, patch(
        "agilerl.training.train.wandb.log"
    ) as mock_wandb_log, patch(
        "agilerl.training.train.wandb.finish"
    ) as mock_wandb_finish:
        # Call the function that should trigger wandb.init
        agilerl.training.train.train(
            env,
            "env_name",
            "algo",
            population_off_policy,
            memory,
            INIT_HP=INIT_HP,
            MUT_P=MUT_P,
            swap_channels=False,
            n_episodes=10,
            max_steps=5,
            evo_epochs=1,
            evo_loop=1,
            n_step=False,
            per=False,
            noisy=True,
            n_step_memory=None,
            tournament=tournament,
            mutation=mutations,
            wb=True,
        )

        # Assert that wandb.init was called with expected arguments
        mock_wandb_init.assert_called_once_with(
            project=ANY,
            name=ANY,
            config=ANY,
        )
        # Assert that wandb.log was called with expected log parameters
        mock_wandb_log.assert_called_with(
            {
                "global_step": ANY,
                "train/mean_score": ANY,
                "eval/mean_fitness": ANY,
                "eval/best_fitness": ANY,
            }
        )
        # Assert that wandb.finish was called
        mock_wandb_finish.assert_called()


@pytest.mark.parametrize(
    "state_size, action_size, vect, checkpoint, accelerator",
    [((6,), 2, True, 5, True), ((6,), 2, True, 5, False), ((6,), 2, True, None, True)],
)
def test_wandb_init_log_distributed(
    env, population_off_policy, tournament, mutations, memory, checkpoint, accelerator
):
    if accelerator:
        accelerator = Accelerator()
    else:
        accelerator = None
    INIT_HP = {
        "BATCH_SIZE": 128,
        "LR": 1e-3,
        "GAMMA": 0.99,
        "LEARN_STEP": 1,
        "TAU": 1e-3,
        "CHANNELS_LAST": False,
        "POP_SIZE": 6,
        "MEMORY_SIZE": 20000,
    }
    MUT_P = {
        "NO_MUT": 0.4,
        "ARCH_MUT": 0.2,
        "PARAMS_MUT": 0.2,
        "ACT_MUT": 0.2,
        "RL_HP_MUT": 0.2,
    }
    with patch("agilerl.training.train.wandb.init") as mock_wandb_init, patch(
        "agilerl.training.train.wandb.log"
    ) as mock_wandb_log, patch(
        "agilerl.training.train.wandb.finish"
    ) as mock_wandb_finish:
        # Call the function that should trigger wandb.init
        agilerl.training.train.train(
            env,
            "env_name",
            "algo",
            population_off_policy,
            memory,
            INIT_HP=INIT_HP,
            MUT_P=MUT_P,
            swap_channels=False,
            n_episodes=10,
            max_steps=5,
            evo_epochs=1,
            evo_loop=1,
            n_step=False,
            per=False,
            noisy=True,
            n_step_memory=None,
            tournament=tournament,
            mutation=mutations,
            wb=True,
            accelerator=accelerator,
            checkpoint=checkpoint,
        )

        # Assert that wandb.init was called with expected arguments
        mock_wandb_init.assert_called_once_with(
            project=ANY,
            name=ANY,
            config=ANY,
        )
        # Assert that wandb.log was called with expected log parameters
        mock_wandb_log.assert_called_with(
            {
                "global_step": ANY,
                "train/mean_score": ANY,
                "eval/mean_fitness": ANY,
                "eval/best_fitness": ANY,
            }
        )
        # Assert that wandb.finish was called
        mock_wandb_finish.assert_called()


@pytest.mark.parametrize("state_size, action_size, vect", [((6,), 2, True)])
def test_early_stop_wandb(env, population_off_policy, tournament, mutations, memory):
    INIT_HP = {
        "BATCH_SIZE": 128,
        "LR": 1e-3,
        "GAMMA": 0.99,
        "LEARN_STEP": 1,
        "TAU": 1e-3,
        "CHANNELS_LAST": False,
        "POP_SIZE": 6,
        "MEMORY_SIZE": 20000,
    }
    MUT_P = {
        "NO_MUT": 0.4,
        "ARCH_MUT": 0.2,
        "PARAMS_MUT": 0.2,
        "ACT_MUT": 0.2,
        "RL_HP_MUT": 0.2,
    }
    with patch("agilerl.training.train.wandb.finish") as mock_wandb_finish:
        # Call the function that should trigger wandb.init
        agilerl.training.train.train(
            env,
            "env_name",
            "algo",
            population_off_policy,
            memory,
            INIT_HP=INIT_HP,
            MUT_P=MUT_P,
            target=-10000,
            swap_channels=False,
            n_episodes=110,
            max_steps=5,
            evo_epochs=1,
            evo_loop=1,
            n_step=False,
            per=False,
            noisy=True,
            n_step_memory=None,
            tournament=tournament,
            mutation=mutations,
            wb=True,
        )
        # Assert that wandb.finish was called
        mock_wandb_finish.assert_called()


@pytest.mark.parametrize("state_size, action_size, vect, algo", [((6,), 2, True, PPO)])
def test_train_on_policy_agent_calls_made(
    env, algo, mocked_agent_on_policy, tournament, mutations
):
    for accelerator_flag in [True, False]:
        if accelerator_flag:
            accelerator = Accelerator()
        else:
            accelerator = None
        mock_population = [mocked_agent_on_policy for _ in range(6)]
        pop, pop_fitnesses = train_on_policy(
            env,
            "env_name",
            "algo",
            mock_population,
            INIT_HP=None,
            MUT_P=None,
            swap_channels=False,
            n_episodes=10,
            max_steps=5,
            evo_epochs=5,
            evo_loop=1,
            tournament=tournament,
            mutation=mutations,
            wb=False,
            accelerator=accelerator,
        )

        mocked_agent_on_policy.getAction.assert_called()
        mocked_agent_on_policy.learn.assert_called()
        mocked_agent_on_policy.test.assert_called()
        if accelerator is not None:
            mocked_agent_on_policy.saveCheckpoint.assert_called()
            mocked_agent_on_policy.wrap_models.assert_called()
            mocked_agent_on_policy.unwrap_models.assert_called()


@pytest.mark.parametrize(
    "state_size, action_size, vect",
    [
        ((6,), 2, True),
    ],
)
def test_train_on_policy_env_calls(
    mocked_env, population_on_policy, tournament, mutations
):
    pop, pop_fitnesses = train_on_policy(
        mocked_env,
        "env_name",
        "algo",
        population_on_policy,
        INIT_HP=None,
        MUT_P=None,
        swap_channels=False,
        n_episodes=10,
        max_steps=5,
        evo_epochs=5,
        evo_loop=1,
        tournament=tournament,
        mutation=mutations,
        wb=False,
    )
    mocked_env.step.assert_called()
    mocked_env.reset.assert_called()


@pytest.mark.parametrize(
    "state_size, action_size, vect",
    [
        ((6,), 2, True),
    ],
)
def test_train_on_policy_tourn_mut_calls(
    env, population_on_policy, mocked_tournament, mocked_mutations
):
    pop, pop_fitnesses = train_on_policy(
        env,
        "env_name",
        "algo",
        population_on_policy,
        INIT_HP=None,
        MUT_P=None,
        swap_channels=False,
        n_episodes=10,
        max_steps=5,
        evo_epochs=5,
        evo_loop=1,
        tournament=mocked_tournament,
        mutation=mocked_mutations,
        wb=False,
    )
    mocked_mutations.mutation.assert_called()
    mocked_tournament.select.assert_called()


@pytest.mark.parametrize(
    "state_size, action_size, vect", [((6,), 2, True), ((6,), 2, False)]
)
def test_train_on_policy(env, population_on_policy, tournament, mutations):
    pop, pop_fitnesses = train_on_policy(
        env,
        "env_name",
        "algo",
        population_on_policy,
        INIT_HP=None,
        MUT_P=None,
        swap_channels=False,
        n_episodes=10,
        max_steps=5,
        evo_epochs=5,
        evo_loop=1,
        tournament=tournament,
        mutation=mutations,
        wb=False,
    )

    assert len(pop) == len(population_on_policy)


@pytest.mark.parametrize("state_size, action_size, vect", [((250, 160, 3), 2, False)])
def test_train_on_policy_rgb_input(env, population_on_policy, tournament, mutations):
    pop, pop_fitnesses = train_on_policy(
        env,
        "env_name",
        "algo",
        population_on_policy,
        INIT_HP=None,
        MUT_P=None,
        swap_channels=True,
        n_episodes=10,
        max_steps=5,
        evo_epochs=5,
        evo_loop=1,
        tournament=tournament,
        mutation=mutations,
        wb=False,
    )

    assert len(pop) == len(population_on_policy)


@pytest.mark.parametrize(
    "state_size, action_size, vect", [((6,), 2, True), ((6,), 2, False)]
)
def test_train_on_policy_distributed(env, population_on_policy, tournament, mutations):
    accelerator = Accelerator()
    pop, pop_fitnesses = train_on_policy(
        env,
        "env_name",
        "algo",
        population_on_policy,
        INIT_HP=None,
        MUT_P=None,
        swap_channels=False,
        n_episodes=10,
        max_steps=5,
        evo_epochs=5,
        evo_loop=1,
        tournament=tournament,
        mutation=mutations,
        wb=False,
        accelerator=accelerator,
    )

    assert len(pop) == len(population_on_policy)


@pytest.mark.parametrize(
    "state_size, action_size, vect, accelerator, checkpoint",
    [((6,), 2, True, False, 5), ((6,), 2, True, False, None), ((6,), 2, True, True, 5)],
)
def test_wandb_init_log_on_policy(
    env, population_on_policy, tournament, mutations, accelerator, checkpoint
):
    if accelerator:
        accelerator = Accelerator()
    else:
        accelerator = None
    INIT_HP = {
        "BATCH_SIZE": 128,
        "LR": 1e-3,
        "GAMMA": 0.99,
        "LEARN_STEP": 1,
        "TAU": 1e-3,
        "CHANNELS_LAST": False,
        "POP_SIZE": 6,
    }
    MUT_P = {
        "NO_MUT": 0.4,
        "ARCH_MUT": 0.2,
        "PARAMS_MUT": 0.2,
        "ACT_MUT": 0.2,
        "RL_HP_MUT": 0.2,
    }
    with patch("agilerl.training.train_on_policy.wandb.init") as mock_wandb_init, patch(
        "agilerl.training.train_on_policy.wandb.log"
    ) as mock_wandb_log, patch(
        "agilerl.training.train.wandb.finish"
    ) as mock_wandb_finish:
        # Call the function that should trigger wandb.init
        agilerl.training.train_on_policy.train_on_policy(
            env,
            "env_name",
            "algo",
            population_on_policy,
            INIT_HP=INIT_HP,
            MUT_P=MUT_P,
            swap_channels=False,
            n_episodes=10,
            max_steps=5,
            evo_epochs=1,
            evo_loop=1,
            tournament=tournament,
            mutation=mutations,
            wb=True,
            accelerator=accelerator,
            checkpoint=checkpoint,
        )

        # Assert that wandb.init was called with expected arguments
        mock_wandb_init.assert_called_once_with(
            project=ANY,
            name=ANY,
            config=ANY,
        )
        # Assert that wandb.log was called with expected log parameters
        mock_wandb_log.assert_called_with(
            {
                "global_step": ANY,
                "train/mean_score": ANY,
                "eval/mean_fitness": ANY,
                "eval/best_fitness": ANY,
            }
        )
        # Assert that wandb.finish was called
        mock_wandb_finish.assert_called()


@pytest.mark.parametrize("state_size, action_size, vect", [((6,), 2, True)])
def test_early_stop_wandb_on_policy(env, population_on_policy, tournament, mutations):
    INIT_HP = {
        "BATCH_SIZE": 128,
        "LR": 1e-3,
        "GAMMA": 0.99,
        "LEARN_STEP": 1,
        "TAU": 1e-3,
        "CHANNELS_LAST": False,
        "POP_SIZE": 6,
        "MEMORY_SIZE": 20000,
    }
    MUT_P = {
        "NO_MUT": 0.4,
        "ARCH_MUT": 0.2,
        "PARAMS_MUT": 0.2,
        "ACT_MUT": 0.2,
        "RL_HP_MUT": 0.2,
    }
    with patch("agilerl.training.train_on_policy.wandb.finish") as mock_wandb_finish:
        # Call the function that should trigger wandb.init
        agilerl.training.train_on_policy.train_on_policy(
            env,
            "env_name",
            "algo",
            population_on_policy,
            INIT_HP=INIT_HP,
            MUT_P=MUT_P,
            target=-10000,
            swap_channels=False,
            n_episodes=110,
            max_steps=5,
            evo_epochs=1,
            evo_loop=1,
            tournament=tournament,
            mutation=mutations,
            wb=True,
        )
        # Assert that wandb.finish was called
        mock_wandb_finish.assert_called()
