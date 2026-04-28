import types
from collections.abc import Callable

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.spaces import Box, Discrete, MultiDiscrete
from pettingzoo import ParallelEnv


class SpeakerListenerLikeEnv(ParallelEnv):
    """Lightweight stand-in for ``simple_speaker_listener_v4.parallel_env``.

    Mirrors the agent names and observation/action shapes of the MPE speaker /
    listener task without pulling in the heavy MPE/PyGame dependency tree.
    Workers spawned for ``AsyncPettingZooVecEnv`` only need to import this file
    instead of the full pettingzoo MPE module, which is the dominant cost in
    most ``test_vector`` tests.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "name": "speaker_listener_like"}

    def __init__(self, render_mode=None, continuous_actions=True):
        self.possible_agents = ["speaker_0", "listener_0"]
        self.agents = self.possible_agents.copy()
        self.render_mode = render_mode
        self._continuous = continuous_actions
        self._obs_shape = {"speaker_0": (3,), "listener_0": (11,)}

    def reset(self, seed=None, options=None):
        self.agents = self.possible_agents.copy()
        observations = {
            agent: np.zeros(self._obs_shape[agent], dtype=np.float32)
            for agent in self.agents
        }
        infos = {agent: {} for agent in self.agents}
        return observations, infos

    def step(self, actions):
        observations = {
            agent: np.zeros(self._obs_shape[agent], dtype=np.float32)
            for agent in self.agents
        }
        rewards = dict.fromkeys(self.agents, 0.0)
        terminations = dict.fromkeys(self.agents, False)
        truncations = dict.fromkeys(self.agents, False)
        infos = {agent: {} for agent in self.agents}
        return observations, rewards, terminations, truncations, infos

    def observation_space(self, agent):
        return Box(low=-np.inf, high=np.inf, shape=self._obs_shape[agent], dtype=np.float32)

    def action_space(self, agent):
        if self._continuous:
            return Box(low=0.0, high=1.0, shape=(5,), dtype=np.float32)
        return Discrete(5)

    def render(self):
        if self.render_mode == "rgb_array":
            return np.zeros((4, 4, 3), dtype=np.uint8)
        return None

    def close(self):
        pass


def speaker_listener_like_env(render_mode=None, continuous_actions=True):
    """Factory matching ``simple_speaker_listener_v4.parallel_env``'s call shape."""
    return SpeakerListenerLikeEnv(
        render_mode=render_mode, continuous_actions=continuous_actions
    )


class SyncMultiAgentVecEnv:
    """In-process stand-in for ``AsyncPettingZooVecEnv`` for tests.

    Spawning even one ``AsyncPettingZooVecEnv`` worker subprocess costs ~5-40s
    in CI (Python re-import + shared-memory setup). The vast majority of
    multi-agent algorithm/training tests only need an env-shaped object that
    responds to ``reset``/``step`` with the correct vectorised shapes; they do
    not rely on real cross-process parallelism. This helper provides exactly
    that interface synchronously, eliminating the subprocess startup cost.

    Mimics the subset of ``AsyncPettingZooVecEnv``'s public API required by
    ``train_multi_agent_*`` and ``algorithm.test()`` (``num_envs``,
    ``agents``, ``possible_agents``, ``single_action_space``,
    ``single_observation_space``, ``reset``, ``step``, ``close``).

    Like ``AsyncPettingZooVecEnv``, missing agents (those that aren't returned
    by an env's ``reset``/``step`` for a given env index — the async-agent
    case) are filled with NaN placeholders via ``get_placeholder_value`` so
    downstream algorithm code that masks via ``np.isnan`` keeps working.
    """

    def __init__(self, env_fns: list[Callable[[], ParallelEnv]]):
        self.envs = [fn() for fn in env_fns]
        self.num_envs = len(self.envs)
        dummy = self.envs[0]
        self.possible_agents = list(dummy.possible_agents)
        self.agents = list(self.possible_agents)
        self._single_action_spaces = {
            agent: dummy.action_space(agent) for agent in self.possible_agents
        }
        self._single_observation_spaces = {
            agent: dummy.observation_space(agent) for agent in self.possible_agents
        }
        self.metadata = getattr(dummy, "metadata", None)
        self.render_mode = getattr(dummy, "render_mode", None)

    def _fill_missing(self, per_env_dicts, transition_name):
        from agilerl.vector.pz_async_vec_env import get_placeholder_value

        filled = []
        for env_dict in per_env_dicts:
            new = {}
            for agent in self.possible_agents:
                if agent in env_dict:
                    new[agent] = env_dict[agent]
                else:
                    new[agent] = get_placeholder_value(
                        agent,
                        transition_name,
                        self._single_observation_spaces,
                    )
            filled.append(new)
        return filled

    def single_action_space(self, agent):
        return self._single_action_spaces[agent]

    def single_observation_space(self, agent):
        return self._single_observation_spaces[agent]

    @staticmethod
    def _stack(per_env_dicts, agents):
        return {
            agent: np.stack(
                [np.asarray(d[agent]) for d in per_env_dicts], axis=0
            )
            for agent in agents
        }

    @staticmethod
    def _stack_as_float(per_env_dicts, agents):
        # ``AsyncPettingZooVecEnv`` represents per-agent ``terminated`` /
        # ``truncated`` / ``reward`` arrays as floats so callers can use NaN
        # to mark inactive agents. Replicate that here so downstream code
        # (e.g. ``algorithm.test`` calling ``np.isnan``) works identically.
        return {
            agent: np.stack(
                [np.asarray(d[agent], dtype=np.float64) for d in per_env_dicts],
                axis=0,
            )
            for agent in agents
        }

    def _add_info(self, vector_infos, env_info, env_num):
        """Compile a vectorised info dict.

        Mirrors ``AsyncPettingZooVecEnv._add_info`` so callers see the same
        per-info-key arrays + ``_{key}`` masks shape regardless of which vec
        env implementation they got.
        """
        for key, value in env_info.items():
            if isinstance(value, dict):
                array = self._add_info(vector_infos.get(key, {}), value, env_num)
            else:
                if key not in vector_infos:
                    if type(value) in [int, float, bool] or issubclass(
                        type(value), np.number
                    ):
                        array = np.zeros(self.num_envs, dtype=type(value))
                    elif isinstance(value, np.ndarray):
                        array = np.zeros(
                            (self.num_envs, *value.shape), dtype=value.dtype
                        )
                    elif value is None:
                        array = np.full(
                            self.num_envs, fill_value=np.nan, dtype=np.float32
                        )
                    else:
                        array = np.full(
                            self.num_envs, fill_value=None, dtype=object
                        )
                else:
                    array = vector_infos[key]
                array[env_num] = value
            array_mask = vector_infos.get(
                f"_{key}", np.zeros(self.num_envs, dtype=bool)
            )
            array_mask[env_num] = True
            vector_infos[key], vector_infos[f"_{key}"] = array, array_mask
        return vector_infos

    def reset(self, seed=None, options=None):
        per_env = [env.reset() for env in self.envs]
        obs_dicts = self._fill_missing([out[0] for out in per_env], "observation")
        info_dicts = [out[1] for out in per_env]
        obs = self._stack(obs_dicts, self.possible_agents)
        infos: dict = {}
        for env_idx, info in enumerate(info_dicts):
            infos = self._add_info(infos, info, env_idx)
        return obs, infos

    def step(self, actions):
        per_env_results = []
        for env_idx, env in enumerate(self.envs):
            env_actions = {
                agent: np.asarray(actions[agent])[env_idx]
                for agent in actions
                if agent in self.possible_agents
            }
            per_env_results.append(env.step(env_actions))
        obs_list = self._fill_missing(
            [r[0] for r in per_env_results], "observation"
        )
        reward_list = self._fill_missing(
            [r[1] for r in per_env_results], "reward"
        )
        term_list = self._fill_missing(
            [r[2] for r in per_env_results], "terminated"
        )
        trunc_list = self._fill_missing(
            [r[3] for r in per_env_results], "truncated"
        )
        info_list = [r[4] for r in per_env_results]
        obs = self._stack(obs_list, self.possible_agents)
        rewards = self._stack_as_float(reward_list, self.possible_agents)
        terms = self._stack_as_float(term_list, self.possible_agents)
        truncs = self._stack_as_float(trunc_list, self.possible_agents)
        infos: dict = {}
        for env_idx, info in enumerate(info_list):
            infos = self._add_info(infos, info, env_idx)
        return obs, rewards, terms, truncs, infos

    def render(self):
        return [env.render() for env in self.envs]

    def close(self, **_kwargs):
        for env in self.envs:
            close = getattr(env, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass


def make_sync_multi_agent_vec_env(
    env_cls: Callable[..., ParallelEnv], num_envs: int = 2, **env_kwargs
) -> SyncMultiAgentVecEnv:
    """Convenience constructor mirroring ``make_multi_agent_vect_envs``."""
    env_fns = [lambda: env_cls(**env_kwargs) for _ in range(num_envs)]
    return SyncMultiAgentVecEnv(env_fns=env_fns)

ROCK = 0
PAPER = 1
SCISSORS = 2
NONE = 3
MOVES = ["ROCK", "PAPER", "SCISSORS", "None"]
NUM_ITERS = 100
REWARD_MAP = {
    (ROCK, ROCK): (0, 0),
    (ROCK, PAPER): (-1, 1),
    (ROCK, SCISSORS): (1, -1),
    (PAPER, ROCK): (1, -1),
    (PAPER, PAPER): (0, 0),
    (PAPER, SCISSORS): (-1, 1),
    (SCISSORS, ROCK): (-1, 1),
    (SCISSORS, PAPER): (1, -1),
    (SCISSORS, SCISSORS): (0, 0),
}


class term_env(ParallelEnv):
    metadata = {"render_modes": ["human"], "name": "rps_v2"}

    def __init__(self, render_mode=None):
        self.possible_agents = ["player_" + str(r) for r in range(2)]
        self.agent_name_mapping = dict(
            zip(
                self.possible_agents,
                list(range(len(self.possible_agents))),
                strict=False,
            ),
        )
        self.render_mode = render_mode
        self.seed = self.dummy_seed

    def dummy_seed(self, seed):
        pass

    def observation_space(self, agent):
        return Discrete(2)  # 0, 2, shape=(2,), dtype=float)

    def action_space(self, agent):
        return Box(0, 2, shape=(1,), dtype=float)

    def close(self):
        pass

    def reset(self, seed=None, options=None):
        self.agents = self.possible_agents[:]
        self.num_moves = 0
        observations = dict.fromkeys(self.agents, NONE)
        infos = {agent: {} for agent in self.agents}
        self.state = observations
        return observations, infos

    def step(self, actions):
        if not actions:
            self.agents = []
            return {}, {}, {}, {}, {}
        rewards = {}
        rewards[self.agents[0]], rewards[self.agents[1]] = (0, 0)
        terminations = dict.fromkeys(self.agents, True)
        self.num_moves += 1
        env_truncation = self.num_moves >= NUM_ITERS
        truncations = dict.fromkeys(self.agents, env_truncation)
        observations = {
            self.agents[i]: actions[self.agents[1 - i]].astype(int)
            for i in range(len(self.agents))
        }
        self.state = observations
        infos = {agent: {} for agent in self.agents}
        if env_truncation:
            self.agents = []
        return observations, rewards, terminations, truncations, infos


def basic_reset_func(
    self,
    *,
    seed: int | None = None,
    options: dict | None = None,
):
    """A basic reset function that will pass the environment check using random actions from the observation space."""
    return {
        agent: self.observation_space(agent).sample() for agent in self.possible_agents
    }, {agent: {"options": options} for agent in self.possible_agents}


def old_reset_func(self):
    """An old reset function that will pass the environment check using random actions from the observation space."""
    super(GenericTestEnv, self).reset()
    return {
        agent: self.observation_space(agent).sample() for agent in self.possible_agent
    }


def basic_step_func(self, action):
    """A step function that follows the basic step api that will pass the environment check using random actions from the observation space."""

    def pz_dict(transition, agents):
        return dict.fromkeys(self.possible_agents, transition)

    return (
        {
            agent: self.observation_space(agent).sample()
            for agent in self.possible_agents
        },
        pz_dict(0, self.possible_agents),
        pz_dict(False, self.possible_agents),
        pz_dict(False, self.possible_agents),
        pz_dict({}, self.possible_agents),
    )


def old_step_func(self, action):
    """A step function that follows the old step api that will pass the environment check using random actions from the observation space."""

    def pz_dict(transition, agents):
        return dict.fromkeys(self.possible_agents, transition)

    return (
        {
            agent: self.observation_space(agent).sample()
            for agent in self.possible_agents
        },
        pz_dict(0, self.possible_agents),
        pz_dict(False, self.possible_agents),
        pz_dict(False, self.possible_agents),
        pz_dict({}, self.possible_agents),
    )


def basic_render_func(self):
    """Basic render fn that does nothing."""


class GenericTestEnv(ParallelEnv):
    """A generic testing environment for use in testing with modified environments are required."""

    def __init__(
        self,
        action_space: spaces.Space | None = None,
        observation_space: spaces.Space | None = None,
        reset_func: Callable = basic_reset_func,
        step_func: Callable = basic_step_func,
        render_func: Callable = basic_render_func,
        metadata=None,
        render_mode=None,
    ):
        """Generic testing environment constructor.

        Args:
            action_space: The environment action space
            observation_space: The environment observation space
            reset_func: The environment reset function
            step_func: The environment step function
            render_func: The environment render function
            metadata: The environment metadata
            render_mode: The render mode of the environment
            spec: The environment spec

        """
        if action_space is None:
            action_space = spaces.Box(0, 1, (1,))
        if observation_space is None:
            observation_space = spaces.Box(0, 1, (1,))
        if metadata is None:
            metadata = {"render_modes": []}
        self.metadata = metadata
        self.render_mode = render_mode
        self.possible_agents = ["agent_0"]
        self.action_space_dict = {"agent_0": action_space}
        self.observation_space_dict = {"agent_0": observation_space}

        if observation_space is not None:
            self.observation_space = self.action_space_func
        if action_space is not None:
            self.action_space = self.observation_space_func

        if reset_func is not None:
            self.reset = types.MethodType(reset_func, self)
        if step_func is not None:
            self.step = types.MethodType(step_func, self)
        if render_func is not None:
            self.render = types.MethodType(render_func, self)

    def action_space_func(self, agent):
        return self.action_space_dict[agent]

    def observation_space_func(self, agent):
        return self.observation_space_dict[agent]

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ):
        """Resets the environment."""
        # If you need a default working reset function, use `basic_reset_fn` above
        msg = "TestingEnv reset_fn is not set."
        raise NotImplementedError(msg)

    def step(self, action):
        """Steps through the environment."""
        msg = "TestingEnv step_fn is not set."
        raise NotImplementedError(msg)

    def render(self):
        """Renders the environment."""
        msg = "testingEnv render_fn is not set."
        raise NotImplementedError(msg)


class CustomSpace(gym.Space):
    """Minimal custom observation space."""

    __hash__ = None  # mutable space, not hashable

    shape = (4,)
    _dtype = np.float16

    def sample(self):
        """Generates a sample from the custom space."""
        return self.np_random.integers(0, 10, ())

    def contains(self, x):
        """Check if the element `x` is contained within the space."""
        return 0 <= x <= 10

    def __eq__(self, other):
        """Check if the two spaces are equal."""
        return isinstance(other, CustomSpace)

    @property
    def dtype(self):
        return self._dtype
