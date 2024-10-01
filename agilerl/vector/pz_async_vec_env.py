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
from typing import Any, TypeVar

import numpy as np
from gymnasium import logger
from gymnasium.error import (
    AlreadyPendingCallError,
    ClosedEnvironmentError,
    NoAsyncCallError,
)
from gymnasium.vector.utils import CloudpickleWrapper, batch_space, clear_mpi_env_vars

from agilerl.vector.pz_vec_env import PettingZooVecEnv

AgentID = TypeVar("AgentID")
ObsType = TypeVar("ObsType")


class AsyncState(Enum):
    """The AsyncVectorEnv possible states given the different actions/"""

    DEFAULT = "default"
    WAITING_RESET = "reset"
    WAITING_STEP = "step"
    WAITING_CALL = "call"


class AsyncVectorPettingZooEnv(PettingZooVecEnv):
    """Vectorized PettingZoo environment that runs multiple environments in parallel

    :param env_fns: Functions that create the environment
    :type env_fns: List[Callable]
    """

    def __init__(self, env_fns, experience_handler=None):
        # Core class attributes
        self.env_fns = env_fns
        self.shared_memory = True
        self.num_envs = len(env_fns)
        dummy_env = env_fns[0]()
        self.metadata = dummy_env.metadata
        self.render_mode = dummy_env.render_mode
        self.possible_agents = dummy_env.possible_agents

        if experience_handler is None:
            env = env_fns[0]()
            self.experience_handler = PettingZooExperienceHandler(
                env, self.num_envs, copy_obs=True
            )
            del env
        else:
            self.experience_handler = experience_handler

        ctx = mp.get_context()
        dummy_env = env_fns[0]()
        self.experience_handler.detect_space_info(dummy_env)
        del dummy_env

        self.action_space = self._get_action_space
        self.observation_space = self._get_observation_space
        self.single_action_space = self._get_single_action_space
        self.single_observation_space = self._get_single_observation_space

        # Create the shared memory for sharing observations between subprocesses
        self._obs_buffer = create_shared_memory(
            self.num_envs,
            width=self.experience_handler.total_observation_width,
            dtype="d",
            ctx=ctx,
        )
        self.observations = read_from_shared_memory(
            self.experience_handler.total_observation_width,
            self._obs_buffer,
            self.num_envs,
            float,
        )
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
                        self._obs_buffer,
                        self.error_queue,
                        self.experience_handler,
                    ),
                )
                self.parent_pipes.append(parent_pipe)
                self.processes.append(process)
                process.daemon = True
                process.start()
                child_pipe.close()
        self._state = AsyncState.DEFAULT

        super().__init__(
            len(env_fns),
            self.possible_agents,
        )

    def _get_single_action_space(self, agent):
        return self.experience_handler.single_action_space_dict[agent]

    def _get_action_space(self, agent):
        return self.experience_handler.action_space_dict[agent]

    def _get_single_observation_space(self, agent):
        return self.experience_handler.single_observation_space_dict[agent]

    def _get_observation_space(self, agent):
        return self.experience_handler.observation_space_dict[agent]

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
        # Convert info data to list before passing into key_search function
        for i, info in enumerate(info_data):
            infos = self._add_info(infos, info, i)

        self._state = AsyncState.DEFAULT
        return self.experience_handler.pack_observation(self.observations), infos

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
        infos = {}
        for env_idx, pipe in enumerate(self.parent_pipes):
            env_step_return, success = pipe.recv()
            successes.append(success)
            if success:
                for agent in self.agents:
                    rewards[agent].append(env_step_return[1][agent])
                    terminations[agent].append(env_step_return[2][agent])
                    truncations[agent].append(env_step_return[3][agent])
                infos = self._add_info(infos, env_step_return[4], env_idx)

        self._raise_if_errors(successes)
        self._state = AsyncState.DEFAULT
        return (
            self.experience_handler.pack_observation(self.observations),
            {agent: np.array(rew) for agent, rew in rewards.items()},
            {agent: np.array(term) for agent, term in terminations.items()},
            {agent: np.array(trunc) for agent, trunc in truncations.items()},
            infos,
        )

    def render(self):
        return self.call("render")

    def call(self, name, *args, **kwargs):
        self.call_async(name, *args, **kwargs)
        return self.call_wait()

    def call_async(self, name, *args, **kwargs):
        self._assert_is_running()
        if self._state != AsyncState.DEFAULT:
            raise AlreadyPendingCallError(
                f"Calling `call_async` while waiting for a pending call to `{self._state.value}` to complete.",
                str(self._state.value),
            )

        for pipe in self.parent_pipes:
            pipe.send(("_call", (name, args, kwargs)))
        self._state = AsyncState.WAITING_CALL

    def call_wait(self, timeout: int | float | None = 60):
        """Error: The call to :meth:`call_wait` has timed out after timeout second(s)."""
        self._assert_is_running()
        if self._state != AsyncState.WAITING_CALL:
            raise NoAsyncCallError(
                "Calling `call_wait` without any prior call to `call_async`.",
                AsyncState.WAITING_CALL.value,
            )

        if not self._poll_pipe_envs(timeout):
            self._state = AsyncState.DEFAULT
            raise mp.TimeoutError(
                f"The call to `call_wait` has timed out after {timeout} second(s)."
            )

        results, successes = zip(*[pipe.recv() for pipe in self.parent_pipes])
        self._raise_if_errors(successes)
        self._state = AsyncState.DEFAULT
        return results

    def get_attr(self, name: str):
        return self.call(name)

    def set_attr(self, name: str, values: list[Any] | tuple[Any] | object):
        """Sets an attribute of the sub-environments."""
        self._assert_is_running()
        if not isinstance(values, (list, tuple)):
            values = [values for _ in range(self.num_envs)]
        if len(values) != self.num_envs:
            raise ValueError(
                "Values must be a list or tuple with length equal to the number of environments. "
                f"Got `{len(values)}` values for {self.num_envs} environments."
            )

        if self._state != AsyncState.DEFAULT:
            raise AlreadyPendingCallError(
                f"Calling `set_attr` while waiting for a pending call to `{self._state.value}` to complete.",
                str(self._state.value),
            )

        for pipe, value in zip(self.parent_pipes, values):
            pipe.send(("_setattr", (name, value)))
        _, successes = zip(*[pipe.recv() for pipe in self.parent_pipes])
        self._raise_if_errors(successes)

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

    def _add_info(self, vector_infos, env_info, env_num):
        """
        Compile a vectorized information dictionary
        """

        for key, value in env_info.items():
            # If value is a dictionary, then we apply the `_add_info` recursively.
            if isinstance(value, dict):
                array = self._add_info(vector_infos.get(key, {}), value, env_num)
            # Otherwise, we are a base case to group the data
            else:
                # If the key doesn't exist in the vector infos, then we can create an array of that batch type
                if key not in vector_infos:
                    if type(value) in [int, float, bool] or issubclass(
                        type(value), np.number
                    ):
                        array = np.zeros(self.num_envs, dtype=type(value))
                    elif isinstance(value, np.ndarray):
                        # We assume that all instances of the np.array info are of the same shape
                        array = np.zeros(
                            (self.num_envs, *value.shape), dtype=value.dtype
                        )
                    else:
                        # For unknown objects, we use a Numpy object array
                        array = np.full(self.num_envs, fill_value=None, dtype=object)
                # Otherwise, just use the array that already exists
                else:
                    array = vector_infos[key]

                # Assign the data in the `env_num` position
                #   We only want to run this for the base-case data (not recursive data forcing the ugly function structure)
                array[env_num] = value

            # Get the array mask and if it doesn't already exist then create a zero bool array
            array_mask = vector_infos.get(
                f"_{key}", np.zeros(self.num_envs, dtype=np.bool_)
            )
            array_mask[env_num] = True

            # Update the vector info with the updated data and mask information
            vector_infos[key], vector_infos[f"_{key}"] = array, array_mask

        return vector_infos


class PettingZooExperienceHandler:
    """Class for formatting experiences when being returned by a vectorized environment

    :param env_fn: Function that returns environment instance when called
    :type env_fn: Callable
    :param num_envs: Number of environments to vectorize
    :type num_envs: int
    :param copy_obs: Boolean flag to indicate whether to copy observations or not, defaults to True
    :type copy_obs: bool, optional
    """

    def __init__(self, env, num_envs, copy_obs=True):
        self.num_envs = num_envs
        self.copy = copy_obs
        self.detect_space_info(env)

    def detect_space_info(self, dummy_env):

        self.metadata = dummy_env.metadata
        self.render_mode = dummy_env.render_mode
        self.possible_agents = dummy_env.possible_agents

        try:
            # Collect action space data
            self.single_action_space_dict = {
                agent: dummy_env.action_space(agent)
                for agent in dummy_env.possible_agents
            }
            self.action_space_dict = {
                agent: batch_space(self.single_action_space_dict[agent], self.num_envs)
                for agent in dummy_env.possible_agents
            }

            # Collect observation space data
            self.single_observation_space_dict = {
                agent: dummy_env.observation_space(agent)
                for agent in dummy_env.possible_agents
            }
            self.observation_space_dict = {
                agent: batch_space(
                    self.single_observation_space_dict[agent], self.num_envs
                )
                for agent in dummy_env.possible_agents
            }

            # Width of each agents flattened observations
            self.observation_widths = {
                agent: int(np.prod(obs.shape))
                for agent, obs in self.single_observation_space_dict.items()
            }
            # Cumulative widths of the flattened observations
            self.observation_boundaries = [0] + list(
                accumulate(self.observation_widths.values(), operator.add)
            )
        except Exception as e:
            raise ValueError(
                "Unable to calculate observation dimensions from current observation space type."
            ) from e

        # Total width of all agents observations, flattened and concatenated
        self.total_observation_width = int(
            np.sum(list(self.observation_widths.values()))
        )
        dummy_env.reset()

        # This is the gospel order of agents
        self.agents = dummy_env.possible_agents[:]
        self.agent_index_map = {agent: i for i, agent in enumerate(self.agents)}

        dummy_env.close()

    def get_placeholder_value(self, index, agent, transition_name, shared_memory=None):
        """When an agent is killed, used to obtain a placeholder value to return for associated experience.

        :param index: Subprocess index
        :type index: int
        :param agent: Agent ID
        :type agent: str
        :param transition_name: Name of the transition
        :type transition_name: str
        :param shared_memory: Shared memory object
        :type shared_memory: mp.RawArray
        """
        if transition_name == "reward":
            return 0
        if transition_name == "truncation" or transition_name == "termination":
            return True
        if transition_name == "info":
            return {}
        if transition_name == "observation":
            return self.read_obs_from_shared_memory(shared_memory, agent, index)

    def pack_observation(self, observation):
        """Pack the observation into a dictionary before returning back to the training loop"""
        obs_dict = {}
        obs_copy = copy.deepcopy(observation) if self.copy else observation
        for idx, agent in enumerate(self.agents):
            obs_dict[agent] = obs_copy[
                :,
                self.observation_boundaries[idx] : self.observation_boundaries[idx + 1],
            ].astype(self.observation_space_dict[agent].dtype)
        return obs_dict

    def process_transition(self, index, transitions, shared_memory, transition_names):
        transition_list = list(transitions)
        for transition, name in zip(transition_list, transition_names):
            transition = {
                agent: (
                    transition[agent]
                    if agent in transition.keys()
                    else self.get_placeholder_value(index, agent, name, shared_memory)
                )
                for agent in self.agents
            }
        return transition_list

    def read_obs_from_shared_memory(self, shared_memory, agent, index):
        """
        Function to read previous timesteps state from the shared memory buffer if a sub-agent has been 'killed'
        in the environment. This then acts as a placeholder when returning vectorized observations.
        """
        agent_idx = self.agent_index_map[agent]
        start_index = (
            index * self.total_observation_width
        ) + self.observation_boundaries[agent_idx]
        end_index = (
            index * self.total_observation_width
        ) + self.observation_boundaries[agent_idx + 1]
        buffer = np.frombuffer(shared_memory).astype(
            self.observation_space_dict[agent].dtype
        )
        return buffer[start_index:end_index]


def _async_worker(
    index,
    env_fn,
    pipe,
    parent_pipe,
    shared_memory,
    error_queue,
    experience_handler,
):
    """
    :param index: Subprocess index
    :type index: int
    :param env_fn: Function to call environment
    :type env_fn: callable
    :param pipe: Child pipe object for sending data to the main process
    :type pipe: Connection
    :param parent_pipe: Parent pipe object
    :type parent_pipe: Connection
    :param shared_memory: Shared memory object
    :shared_memory: mp.RawArray
    :param error_queue: Queue object for collecting subprocess errors to communicate back to the main process
    :type error_queue: mp.Queue
    :param experience_handler: Experience handler object to handle and format experiences
    :type experience_handler: PettingZooExperienceHandler,
    """
    env = env_fn()
    agents = env.possible_agents[:]
    autoreset = False
    parent_pipe.close()

    try:
        while True:
            command, data = pipe.recv()
            if command == "reset":
                observation, info = env.reset(**data)
                if shared_memory:
                    write_to_shared_memory(
                        index=index,
                        width=experience_handler.total_observation_width,
                        shared_memory=shared_memory,
                        data=observation,
                    )
                    observation = None
                    autoreset = False
                pipe.send(((observation, info), True))
            elif command == "step":
                if autoreset:
                    observation, info = experience_handler.process_transition(
                        index, env.reset(), shared_memory, ["observation", "info"]
                    )
                    reward = {agent: 0 for agent in agents}
                    terminated = {agent: False for agent in agents}
                    truncated = {agent: False for agent in agents}
                else:
                    data = {
                        possible_agent: np.array(data[idx]).squeeze()
                        for idx, possible_agent in enumerate(agents)
                    }
                    transition = env.step(data)
                    observation, reward, terminated, truncated, info = (
                        experience_handler.process_transition(
                            index,
                            transition,
                            shared_memory,
                            [
                                "observation",
                                "reward",
                                "terminated",
                                "truncated",
                                "info",
                            ],
                        )
                    )
                autoreset = all(
                    [
                        term | trunc
                        for term, trunc in zip(terminated.values(), truncated.values())
                    ]
                )
                if shared_memory:
                    # Add in logic here to separate observations and action masks
                    write_to_shared_memory(
                        index=index,
                        width=experience_handler.total_observation_width,
                        shared_memory=shared_memory,
                        data=observation,
                    )
                    observation = None

                pipe.send(((observation, reward, terminated, truncated, info), True))

            elif command == "close":
                pipe.send((None, True))
                break
            elif command == "_call":
                name, args, kwargs = data
                if name in ["reset", "step", "close", "_setattr", "_check_spaces"]:
                    raise ValueError(
                        f"Trying to call function `{name}` with `call`, use `{name}` directly instead."
                    )
                attr = getattr(env, name)
                if callable(attr):
                    pipe.send((attr(*args, **kwargs), True))
                else:
                    pipe.send((attr, True))
            elif command == "_setattr":
                name, value = data
                setattr(env, name, value)
                pipe.send((None, True))
            else:
                raise RuntimeError(
                    f"Received unknown command `{command}`. Must be one of [`reset`, `step`, `close`, `_call`, `_setattr`, `_check_spaces`]."
                )

    except (KeyboardInterrupt, Exception):
        error_type, error_message, _ = sys.exc_info()
        trace = traceback.format_exc()
        error_queue.put((index, error_type, error_message, trace))
        pipe.send((None, False))

    finally:
        env.close()


def create_shared_memory(num_envs, width, dtype, ctx=mp):
    """
    Create a RawArray to write observations to.

    :param num_envs: Number of environments to vectorise
    :type num_envs: int
    :param width: Width of the array
    :type width: int
    :param dtype: Array data type
    :type dtype: str
    :param ctx: Multiprocessing context
    :type ctx: Context # FIXME
    """
    return ctx.RawArray(dtype, num_envs * int(width))


def read_from_shared_memory(width, shared_memory, num_envs, dtype):
    return np.frombuffer(shared_memory).reshape((num_envs, width)).astype(dtype)


def write_to_shared_memory(
    index,
    width,
    shared_memory,
    data,
):
    destination = np.frombuffer(shared_memory, dtype=np.float32)
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
    return np.array(result, dtype=np.float32)
