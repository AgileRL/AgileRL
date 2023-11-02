from unittest.mock import ANY, patch

import numpy as np
import pytest
from accelerate import Accelerator

import agilerl.training.train
from agilerl.training.train import train


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


class DummyAgent:
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
def population(env):
    return [DummyAgent(5, env, 0.4) for _ in range(6)]


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


@pytest.mark.parametrize(
    "state_size, action_size, vect", [((6,), 2, True), ((6,), 2, False)]
)
def test_train(env, population, tournament, mutations, memory):
    pop, pop_fitnesses = train(
        env,
        "env_name",
        "algo",
        population,
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

    assert len(pop) == len(population)


@pytest.mark.parametrize("state_size, action_size, vect", [((250, 160, 3), 2, False)])
def test_train_rgb_input(env, population, tournament, mutations, memory):
    pop, pop_fitnesses = train(
        env,
        "env_name",
        "algo",
        population,
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

    assert len(pop) == len(population)


@pytest.mark.parametrize(
    "state_size, action_size, vect, per",
    [((6,), 2, True, True), ((6,), 2, True, False)],
)
def test_train_using_alternate_buffers(
    env, memory, population, tournament, mutations, n_step_memory, per
):
    pop, pop_fitnesses = train(
        env,
        "env_name",
        "algo",
        population,
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

    assert len(pop) == len(population)


@pytest.mark.parametrize("state_size, action_size, vect", [((3, 64, 64), 2, True)])
def test_train_using_alternate_buffers_rgb(
    env, memory, population, tournament, mutations, n_step_memory
):
    pop, pop_fitnesses = train(
        env,
        "env_name",
        "algo",
        population,
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

    assert len(pop) == len(population)


@pytest.mark.parametrize(
    "state_size, action_size, vect", [((6,), 2, True), ((6,), 2, False)]
)
def test_train_distributed(env, population, tournament, mutations, memory):
    accelerator = Accelerator()
    pop, pop_fitnesses = train(
        env,
        "env_name",
        "algo",
        population,
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

    assert len(pop) == len(population)


@pytest.mark.parametrize("state_size, action_size, vect", [((6,), 2, True)])
def test_wandb_init_log(env, population, tournament, mutations, memory):
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
            population,
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
    env, population, tournament, mutations, memory, checkpoint, accelerator
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
            population,
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
def test_early_stop_wandb(env, population, tournament, mutations, memory):
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
            population,
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
