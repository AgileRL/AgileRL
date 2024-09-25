"""An async vector pettingzoo environment"""

import copy
import multiprocessing as mp
import operator
import sys
import time
import traceback
from collections import defaultdict
from enum import Enum
from itertools import accumulate
from multiprocessing import Queue
from multiprocessing.connection import Connection
from typing import Any, Dict, TypeVar

import numpy as np
from gymnasium import logger
from gymnasium.error import (
    AlreadyPendingCallError,
    ClosedEnvironmentError,
    CustomSpaceError,
    NoAsyncCallError,
)
from gymnasium.vector.utils import CloudpickleWrapper, batch_space, clear_mpi_env_vars

from agilerl.vector.pz_vec_env import VecEnv

AgentID = TypeVar("AgentID")
ObsType = TypeVar("ObsType")


class AsyncState(Enum):
    """The AsyncVectorEnv possible states given the different actions/"""

    DEFAULT = "default"
    WAITING_RESET = "reset"
    WAITING_STEP = "step"
    WAITING_CALL = "call"


class AsyncVectorPZEnv(VecEnv):
    """Vectorized PettingZoo environment that runs multiple environments in parallel"""

    def __init__(
        self,
        env_fns,
        shared_memory=True,
        optimize=False,  # Optimize used for Ray framework
        daemon=True,
        copy=True,
    ):
        # Core class attributes
        self.env_fns = env_fns
        self.shared_memory = shared_memory
        self.num_envs = len(env_fns)
        dummy_env = env_fns[0]()
        self.metadata = dummy_env.metadata
        self.render_mode = dummy_env.render_mode
        self.possible_agents = dummy_env.possible_agents
        self.copy = copy

        # Collect action space data
        self.single_action_space_dict = {
            agent: dummy_env.action_space(agent) for agent in dummy_env.possible_agents
        }
        self.action_space_dict = {
            agent: batch_space(self.single_action_space_dict[agent], self.num_envs)
            for agent in dummy_env.possible_agents
        }
        self.single_action_space = self._get_single_action_space
        self.action_space = self._get_action_space

        # Collect observation space data
        self.single_observation_space_dict = {
            agent: dummy_env.observation_space(agent)
            for agent in dummy_env.possible_agents
        }
        self.observation_space_dict = {
            agent: batch_space(self.single_observation_space_dict[agent], self.num_envs)
            for agent in dummy_env.possible_agents
        }
        self.single_observation_space = self._get_single_observation_space
        self.observation_space = self._get_observation_space

        self.indi_obs_widths = {
            agent: int(np.prod(obs.shape))
            for agent, obs in self.single_observation_space_dict.items()
        }
        self.obs_boundaries = [0] + list(
            accumulate(self.indi_obs_widths.values(), operator.add)
        )
        self.obs_space_width = int(np.sum(list(self.indi_obs_widths.values())))

        print("-----------------------------")
        print(self.single_observation_space_dict)
        print(self.indi_obs_widths)
        print(self.obs_boundaries)
        print(self.obs_space_width)

        dummy_env.close()
        del dummy_env

        ctx = mp.get_context()
        # Do we need to also do this for env defined actions and action masks
        if self.shared_memory:
            try:
                # _obs_buffers = {agent : create_shared_memory(
                #         num_envs=self.num_envs, total_obs_width=self.indi_obs_widths[agent], ctx=ctx, dtype=space.dtype.char
                #     ) for agent, space in self.single_observation_space_dict.items()
                # }
                _obs_buffer = create_shared_memory(
                    self.num_envs, width=self.obs_space_width, dtype="d", ctx=ctx
                )
                self.observations = read_from_shared_memory(
                    self.obs_space_width, _obs_buffer, self.num_envs, float
                )
            except CustomSpaceError as e:
                raise ValueError(
                    "Using `AsyncVector(..., shared_memory=True)` caused an error, you can disable this feature with `shared_memory=False` however this is slower."
                ) from e
        else:
            _obs_buffers = None
            self.observations = np.zeros((self.num_envs, self.obs_space_width))

        # FIXME these should probably all be Wei's special dict interface class -> ArrayDict - dictionary that uses an array for storage
        self.rewards = np.zeros((self.num_envs, 1))
        self.terminations = np.zeros((self.num_envs, 1))
        self.truncations = np.zeros((self.num_envs, 1))
        self.infos = {}

        self.parent_pipes, self.processes = [], []
        self.error_queue = ctx.Queue()

        target = _async_worker
        with clear_mpi_env_vars():
            for idx, env_fn in enumerate(self.env_fns):
                parent_pipe, child_pipe = ctx.Pipe()
                process = ctx.Process(
                    target=target,
                    name=f"Worker<{type(self).__name__}>-{idx}",
                    args=(
                        idx,
                        CloudpickleWrapper(env_fn),
                        child_pipe,
                        parent_pipe,
                        _obs_buffer,
                        self.error_queue,
                        self.obs_space_width,
                    ),
                )
                self.parent_pipes.append(parent_pipe)
                self.processes.append(process)
                process.daemon = daemon
                process.start()
                child_pipe.close()
        self._state = AsyncState.DEFAULT

        super().__init__(
            len(env_fns),
            self.possible_agents,
        )

    @property
    def np_random_seed(self) -> tuple[int, ...]:
        """Returns a tuple of np_random seeds for all the wrapped envs."""
        return self.get_attr("np_random_seed")

    @property
    def np_random(self) -> tuple[np.random.Generator, ...]:
        """Returns the tuple of the numpy random number generators for the wrapped envs."""
        return self.get_attr("np_random")

    def _get_single_action_space(self, agent):
        return self.single_action_space_dict[agent]

    def _get_action_space(self, agent):
        return self.action_space_dict[agent]

    def _get_single_observation_space(self, agent):
        return self.single_observation_space_dict[agent]

    def _get_observation_space(self, agent):
        return self.observation_space_dict[agent]

    def reset(
        self,
        *,
        seed: int | list[int] | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[AgentID, ObsType], dict[AgentID, dict]]:
        """Resets all sub-environments in parallel and return a batch of concatenated observations and info.

        Args:
            seed: The environment reset seeds
            options: If to return the options

        Returns:
            A batch of observations and info from the vectorized environment.
        """
        self.reset_async(seed=seed, options=options)
        return self.reset_wait()

    def reset_async(
        self,
        seed: int | list[int] | None = None,
        options: dict | None = None,
    ):
        """Send calls to the :obj:`reset` methods of the sub-environments.

        To get the results of these calls, you may invoke :meth:`reset_wait`.
        """
        self._assert_is_running()
        if seed is None:
            seed = [None for _ in range(self.num_envs)]
        elif isinstance(seed, int):
            seed = [seed + i for i in range(self.num_envs)]
        assert (
            len(seed) == self.num_envs
        ), f"If seeds are passed as a list the length must match num_envs={self.num_envs} but got length={len(seed)}."

        if self._state != AsyncState.DEFAULT:
            raise AlreadyPendingCallError(
                f"Calling `reset_async` while waiting for a pending call to `{self._state.value}` to complete",
                str(self._state.value),
            )

        for pipe, env_seed in zip(self.parent_pipes, seed):
            env_kwargs = {"seed": env_seed, "options": options}
            pipe.send(("reset", env_kwargs))
        self._state = AsyncState.WAITING_RESET

    def reset_wait(
        self, timeout: int | float | None = 60
    ) -> tuple[dict[AgentID, ObsType], dict[AgentID, dict]]:
        """Waits for the calls triggered by :meth:`reset_async` to finish and returns the results."""
        self._assert_is_running()
        if self._state != AsyncState.WAITING_RESET:
            raise NoAsyncCallError(
                "Calling `reset_wait` without any prior " "call to `reset_async`.",
                AsyncState.WAITING_RESET.value,
            )

        if not self._poll_pipe_envs(timeout):
            self._state = AsyncState.DEFAULT
            raise mp.TimeoutError(
                f"The call to `reset_wait` has timed out after {timeout} second(s)."
            )

        results, successes = zip(*[pipe.recv() for pipe in self.parent_pipes])
        self._raise_if_errors(successes)

        infos = {}
        results, info_data = zip(*results)
        for i, info in enumerate(info_data):
            pass
            # infos = self._add_info(infos, info, i)

        if not self.shared_memory:
            pass
        self._state = AsyncState.DEFAULT
        return self.pack_observation(self.observations), infos

    def step_async(self, actions):
        self._assert_is_running()
        if self._state != AsyncState.DEFAULT:
            raise AlreadyPendingCallError(
                f"Calling `step_async` while waiting for a pending call to `{self._state.value}` to complete.",
                str(self._state.value),
            )
        for pipe, action in zip(self.parent_pipes, actions):
            pipe.send(("step", action))
        self._state = AsyncState.WAITING_STEP

    def step_wait(self, timeout: int | None = 60):
        self._assert_is_running()
        if self._state != AsyncState.WAITING_STEP:
            raise NoAsyncCallError(
                "Calling `step_wait` without any prior call " "to `step_async`.",
                AsyncState.WAITING_STEP.value,
            )

        if not self._poll_pipe_envs(timeout):
            self._state = AsyncState.DEFAULT
            raise mp.TimeoutError(
                f"The call to `step_wait` has timed out after {timeout} second(s)."
            )

        rewards, terminations, truncations, infos = (
            defaultdict(list) for _ in range(4)
        )
        successes = []
        for env_idx, pipe in enumerate(self.parent_pipes):
            env_step_return, success = pipe.recv()
            successes.append(success)
            for agent in self.agents:
                if success:
                    rewards[agent].append(env_step_return[1][agent])
                    terminations[agent].append(env_step_return[2][agent])
                    truncations[agent].append(env_step_return[3][agent])
                    infos[agent] = self._add_info(
                        infos, env_step_return[4][agent], None
                    )

        self._raise_if_errors(successes)
        self._state = AsyncState.DEFAULT
        return (
            self.pack_observation(self.observations),
            {agent: np.array(rew) for agent, rew in rewards.items()},
            {agent: np.array(term) for agent, term in terminations.items()},
            {agent: np.array(trunc) for agent, trunc in truncations.items()},
            infos,
        )

    def close_extras(self, timeout=60, terminate=False):
        timeout = 0 if terminate else timeout

        try:
            if self._state != AsyncState.DEFAULT:
                logger.warn(
                    f"Calling `close` while waiting for a pending call to `{self._state.value}` to complete."
                )
                function = getattr(self, f"{self._state.value}_wait")
                function(timeout)
        except mp.TimeoutError:
            terminate = True

        if terminate:
            for process in self.processes:
                if process.is_alive():
                    process.terminate()
        else:
            for pipe in self.parent_pipes:
                if (pipe is not None) and (not pipe.closed):
                    pipe.send(("close", None))
            for pipe in self.parent_pipes:
                if (pipe is not None) and (not pipe.closed):
                    pipe.recv()

        for pipe in self.parent_pipes:
            if pipe is not None:
                pipe.close()
        for process in self.processes:
            process.join()

    def pack_observation(self, observation):
        obs_dict = {}
        obs_copy = copy.deepcopy(observation) if self.copy else observation
        for idx, agent in enumerate(self.agents):
            obs_dict[agent] = obs_copy[
                :, self.obs_boundaries[idx] : self.obs_boundaries[idx + 1]
            ]
        return obs_dict

    def _poll_pipe_envs(self, timeout: int | None = None):
        self._assert_is_running()

        if timeout is None:
            return True

        end_time = time.perf_counter() + timeout
        for pipe in self.parent_pipes:
            delta = max(end_time - time.perf_counter(), 0)

            if pipe is None:
                return False
            if pipe.closed or (not pipe.poll(delta)):
                return False
        return True

    def _raise_if_errors(self, successes: list[bool] | tuple[bool]):
        if all(successes):
            return

        num_errors = self.num_envs - sum(successes)
        assert num_errors > 0
        for i in range(num_errors):
            index, exctype, value, trace = self.error_queue.get()

            logger.error(
                f"Received the following error from Worker-{index} - Shutting it down"
            )
            logger.error(f"{trace}")

            self.parent_pipes[index].close()
            self.parent_pipes[index] = None

            if i == num_errors - 1:
                logger.error("Raising the last exception back to the main process.")
                self._state = AsyncState.DEFAULT
                raise exctype(value)

    def _assert_is_running(self):
        if self.closed:
            raise ClosedEnvironmentError(
                f"Trying to operate on `{type(self).__name__}`, after a call to `close()`."
            )


def _async_worker(
    index: int,
    env_fn: callable,
    pipe: Connection,
    parent_pipe: Connection,
    shared_memory: Dict[str, mp.Array],
    error_queue: Queue,
    obs_width: int,
):
    env = env_fn()
    # observation_space = env.observation_space
    # action_space = env.action_space
    agents = env.possible_agents[:]
    autoreset = False
    parent_pipe.close()

    # TODO Ensure the order of the agents is preserved
    try:
        while True:
            command, data = pipe.recv()
            if command == "reset":
                observation, info = env.reset(**data)
                if shared_memory:
                    write_to_shared_memory(
                        index=index,
                        width=obs_width,
                        shared_memory=shared_memory,
                        data=observation,
                    )
                    observation = None
                    autoreset = False
                pipe.send(((observation, info), True))
            elif command == "step":
                if autoreset:
                    observation, info = env.reset()
                    reward = {agent: 0 for agent in agents}
                    terminated = {agent: False for agent in agents}
                    truncated = {agent: False for agent in agents}
                else:
                    data = {
                        possible_agent: np.array(data[idx]).squeeze()
                        for idx, possible_agent in enumerate(agents)
                    }
                    observation, reward, terminated, truncated, info = env.step(data)
                autoreset = all(
                    [
                        term | trunc
                        for term, trunc in zip(terminated.values(), truncated.values())
                    ]
                )
                if shared_memory:
                    write_to_shared_memory(
                        index=index,
                        width=obs_width,
                        shared_memory=shared_memory,
                        data=observation,
                    )
                    observation = None
                pipe.send(((observation, reward, terminated, truncated, info), True))

    except (KeyboardInterrupt, Exception):
        error_type, error_message, _ = sys.exc_info()
        trace = traceback.format_exc()

        error_queue.put((index, error_type, error_message, trace))
        pipe.send((None, False))
    finally:
        env.close()


def create_shared_memory(num_envs, width, dtype, ctx=mp):
    return ctx.Array(dtype, num_envs * int(width))


def read_from_shared_memory(width, shared_memory, num_envs, dtype):
    return np.frombuffer(shared_memory.get_obj(), dtype=dtype).reshape(
        (num_envs, width)
    )


def write_to_shared_memory(
    index,
    width,
    shared_memory,
    data,
):
    destination = np.frombuffer(shared_memory.get_obj(), dtype=float)
    np.copyto(destination[index * width : (index + 1) * width], dict_to_1d_array(data))


def dict_to_1d_array(input_dict):
    input_dict_copy = input_dict.copy()
    result = []
    for value in input_dict_copy.values():
        if isinstance(value, np.ndarray):
            result.extend(value.flatten())
        elif isinstance(value, list):
            result.extend(value)
        elif isinstance(value, (int, float)):
            result.append(value)
        else:
            raise TypeError(f"Unsupported type: {type(value)}")
    return np.array(result)


# class ExperienceHandler:
#     """Class to handle observation/action reshaping/reformatting"""

#     def __init__(self, env, experience_type):
#         if experience_type == "action":
