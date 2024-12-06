"""An async vector pettingzoo environment"""

import multiprocessing as mp
import operator
import sys
import time
import traceback
from collections import defaultdict
from copy import deepcopy
from enum import Enum
from itertools import accumulate
from typing import TypeVar

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
    """The AsyncVectorEnv possible states given the different actions."""

    DEFAULT = "default"
    WAITING_RESET = "reset"
    WAITING_STEP = "step"
    WAITING_CALL = "call"


class AsyncPettingZooVecEnv(PettingZooVecEnv):
    """Vectorized PettingZoo environment that runs multiple environments in parallel

    :param env_fns: Functions that create the environment
    :type env_fns: list[Callable]
    :param copy: Boolean flag to copy the observation data when it is returned with either .step() or .reset(), recommended, defaults to True
    :type copy: bool, optional
    :param context: Context for multiprocessing
    """

    def __init__(self, env_fns, experience_spec=None, copy=True, context=None):

        # Core class attributes
        self.env_fns = env_fns
        self.num_envs = len(env_fns)
        dummy_env = env_fns[0]()
        self.metadata = (
            dummy_env.metadata
            if hasattr(dummy_env, "metadata")
            else dummy_env.unwrapped.metadata
        )
        self.render_mode = (
            dummy_env.render_mode
            if hasattr(dummy_env, "render_mode")
            else dummy_env.unwrapped.render_mode
        )
        self.possible_agents = dummy_env.possible_agents
        self.copy = copy

        if experience_spec is None:
            self.experience_spec = PettingZooExperienceSpec(self.num_envs)
        else:
            self.experience_spec = experience_spec

        self.experience_spec.detect_space_info(dummy_env)
        del dummy_env
        ctx = mp.get_context(context)

        self.action_space = self._get_action_space
        self.observation_space = self._get_observation_space
        self.single_action_space = self._get_single_action_space
        self.single_observation_space = self._get_single_observation_space

        # Create the shared memory for sharing observations between subprocesses
        self._obs_buffer = SharedMemory(
            num_envs=self.num_envs, exp_spec=self.experience_spec, context=ctx
        )
        self.observations = Observations(
            shared_memory=self._obs_buffer.shared_memory,
            exp_spec=self.experience_spec,
            num_envs=self.num_envs,
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
                        self._obs_buffer.shared_memory,
                        self.error_queue,
                        self.experience_spec.observation_shapes,
                        self.experience_spec.observation_widths,
                        self.experience_spec.observation_dtypes,
                        self.experience_spec.agents,
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
        """Get an agents single action space

        :param agent: Name of agent
        :type agent: str
        """
        return self.experience_spec.single_action_space[agent]

    def _get_action_space(self, agent):
        """Get an agents action space

        :param agent: Name of agent
        :type agent: str
        """
        return self.experience_spec.action_space[agent]

    def _get_single_observation_space(self, agent):
        """Get an agents single observation space

        :param agent: Name of agent
        :type agent: str
        """
        return self.experience_spec.single_observation_space[agent]

    def _get_observation_space(self, agent):
        """Get an agents observation space

        :param agent: Name of agent
        :type agent: str
        """
        return self.experience_spec.observation_space[agent]

    def reset(
        self,
        *,
        seed=None,
        options=None,
    ):
        """
        Reset all the environments and return two dictionaries of batched observations and infos.

        :param seed: Random seed, defaults to None
        :type seed: None | int, optional
        :param options: Options dictionary
        :type options: dict[str, Any]
        """
        self.reset_async(seed=seed, options=options)
        return self.reset_wait()

    def reset_async(
        self,
        seed=None,
        options=None,
    ):
        """Send calls to the :obj:`reset` methods of the sub-environments.

        To get the results of these calls, you may invoke :meth:`reset_wait`.

        :param seed: Random seed, defaults to None
        :type seed: None | int, optional
        :param options: Options dictionary
        :type options: dict[str, Any]
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

    def reset_wait(self, timeout=None):
        """Waits for the calls triggered by :meth:`reset_async` to finish and returns the results.

        :param timeout: Number of seconds before the call to ``reset_wait`` times out. If `None`, the call to ``reset_wait`` never times out, defaults to 0
        :type timeout: int | float | None, optional
        """
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

        info_data, successes = zip(*[pipe.recv() for pipe in self.parent_pipes])
        self._raise_if_errors(successes)

        infos = {}
        # Convert info data to list before passing into key_search function
        for i, info in enumerate(info_data):
            infos = self._add_info(infos, info, i)

        self._state = AsyncState.DEFAULT
        return (
            (
                {
                    agent: deepcopy(self.observations[agent])
                    for agent in self.observations.keys()
                }
                if self.copy
                else self.observations
            ),
            infos,
        )

    def step_async(self, actions):
        """
        Tell all the environments to start taking a step
        with the given actions.
        Call step_wait() to get the results of the step.
        You should not call this if a step_async run is
        already pending.

        :param actions: List of lists of length num_envs, each sub list contains actions for each agent in a given environment
        :type actions: list[list[int | float | np.ndarray]]
        """
        self._assert_is_running()
        if self._state != AsyncState.DEFAULT:
            raise AlreadyPendingCallError(
                f"Calling `step_async` while waiting for a pending call to `{self._state.value}` to complete.",
                str(self._state.value),
            )
        for pipe, action in zip(self.parent_pipes, actions):
            pipe.send(("step", action))
        self._state = AsyncState.WAITING_STEP

    def step_wait(self, timeout=None):
        """
        Wait for the calls to :obj:`step` in each sub-environment to finish.

        :param timeout: Number of seconds before the call to ``step_wait`` times out. If `None`, the call to ``step_wait`` never times out, defaults to 0
        :type timeout: int | float | None, optional
        """

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
                    rewards[agent].append(env_step_return[0][agent])
                    terminations[agent].append(env_step_return[1][agent])
                    truncations[agent].append(env_step_return[2][agent])
                infos = self._add_info(infos, env_step_return[3], env_idx)

        self._raise_if_errors(successes)
        self._state = AsyncState.DEFAULT
        return (
            (
                {
                    agent: deepcopy(self.observations[agent])
                    for agent in self.observations.keys()
                }
                if self.copy
                else self.observations
            ),
            {agent: np.array(rew) for agent, rew in rewards.items()},
            {agent: np.array(term) for agent, term in terminations.items()},
            {agent: np.array(trunc) for agent, trunc in truncations.items()},
            infos,
        )

    def render(self):
        """
        Returns the rendered frames from the parallel environments.

        """

        return self.call("render")

    def call(self, name, *args, **kwargs):
        """
        Call a method from each parallel environment with args and kwargs.

        :param name: Name of the method or property to call
        :type name: str
        :param *args: Position arguments to apply to the method call.
        :type *args: Any
        :param **kwargs: Keyword arguments to apply to the method call.
        :type **kwargs: Any
        """
        self.call_async(name, *args, **kwargs)
        return self.call_wait()

    def call_async(self, name, *args, **kwargs):
        """
        Calls the method with name asynchronously and apply args and kwargs to the method.

        :param name: Name of the method or property to call
        :type name: str
        :param *args: Position arguments to apply to the method call.
        :type *args: Any
        :param **kwargs: Keyword arguments to apply to the method call.
        :type **kwargs: Any
        """
        self._assert_is_running()
        if self._state != AsyncState.DEFAULT:
            raise AlreadyPendingCallError(
                f"Calling `call_async` while waiting for a pending call to `{self._state.value}` to complete.",
                str(self._state.value),
            )

        for pipe in self.parent_pipes:
            pipe.send(("_call", (name, args, kwargs)))
        self._state = AsyncState.WAITING_CALL

    def call_wait(self, timeout=None):
        """Calls all parent pipes and waits for the results.

        :param timeout: Number of seconds before the call to :meth:`call_wait` times out. If ``None`` (default), the call to :meth:`call_wait` never times out, defaults to 0
        :type timeout: int | float | None, optional
        """
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

    def get_attr(self, name):
        """
        Get a property from each parallel environment.

        :param name: Name of property to get from each individual environment
        :type name: str
        """
        return self.call(name)

    def set_attr(self, name, values):
        """Sets an attribute of the sub-environments.

        :param name: Name of the property to be set in each individual environment.
        :type name: str
        :param values: Values of the property to be set to. If ``values`` is a list or
            tuple, then it corresponds to the values for each individual
            environment, otherwise a single value is set for all environments.
        :type values: list[Any] | tuple[Any] | object
        """
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

    def close_extras(self, timeout=None, terminate=False):
        """
        Close the environments & clean up the extra resources (processes and pipes).

        :param timeout: Number of seconds before the call to :meth:`close` times out. If ``None``,
                the call to :meth:`close` never times out. If the call to :meth:`close`
                times out, then all processes are terminated, defaults to 0
        :type timeout: int | float | None, optional
        :param terminate: If ``True``, then the :meth:`close` operation is forced and all processes are terminated, defaults to False
        :type terminate: bool, optional
        """
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
                    elif value is None:
                        array = np.full(
                            self.num_envs, fill_value=np.nan, dtype=np.float32
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

    def __del__(self):
        """On deleting the object, checks that the vector environment is closed."""
        if not getattr(self, "closed", True) and hasattr(self, "_state"):
            self.close(terminate=True)
        if hasattr(self, "_obs_buffer"):
            del self._obs_buffer
        if hasattr(self, "observations"):
            del self.observations
        if hasattr(self, "experience_spec"):
            del self.experience_spec


class PettingZooExperienceSpec:
    """Class for formatting experiences when being returned by a vectorized environment

    :param num_envs: Number of environments to vectorize
    :type num_envs: int
    """

    def __init__(self, num_envs):
        self.num_envs = num_envs

    def detect_space_info(self, dummy_env):

        self.metadata = (
            dummy_env.metadata
            if hasattr(dummy_env, "metadata")
            else dummy_env.unwrapped.metadata
        )
        self.render_mode = (
            dummy_env.render_mode
            if hasattr(dummy_env, "render_mode")
            else dummy_env.unwrapped.render_mode
        )
        self.possible_agents = dummy_env.possible_agents

        try:
            # Collect action space data
            self.single_action_space = {
                agent: dummy_env.action_space(agent)
                for agent in dummy_env.possible_agents
            }
            self.action_space = {
                agent: batch_space(self.single_action_space[agent], self.num_envs)
                for agent in dummy_env.possible_agents
            }

            # Collect observation space data
            self.single_observation_space = {
                agent: dummy_env.observation_space(agent)
                for agent in dummy_env.possible_agents
            }

            self.observation_dtypes = {
                agent: dummy_env.observation_space(agent).dtype
                for agent in dummy_env.possible_agents
            }

            self.observation_space = {
                agent: batch_space(self.single_observation_space[agent], self.num_envs)
                for agent in dummy_env.possible_agents
            }

            self.observation_shapes = {
                agent: space.shape if space.shape != () else (1,)
                for agent, space in self.single_observation_space.items()
            }

            # Width of each agents flattened observations
            self.observation_widths = {
                agent: int(np.prod(obs.shape))
                for agent, obs in self.single_observation_space.items()
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
        del dummy_env


class Observations:
    """
    Class for storing observations with a dictionary interface

    :param shared_memory: A RawArray that all envs write observations to.
    :type shared_memory: multiprocessing.RawArray
    :param exp_spec: Experience specification
    :type exp_spec: agilerl.vector.pz_async_vec_env.PettingZooExperienceSpec
    :param num_envs: Number of environments
    :type num_envs: int

    """

    def __init__(self, shared_memory, exp_spec, num_envs):
        self.exp_spec = exp_spec
        self.num_envs = num_envs
        self.shared_memory = shared_memory
        self.obs_view = []
        for shm, agent in zip(shared_memory, exp_spec.agents):
            self.obs_view.append(
                np.frombuffer(
                    shm.get_obj(), dtype=exp_spec.single_observation_space[agent].dtype
                )
            )

    def __getitem__(self, key):
        """
        Get agent observation given a key (agent_id)
        """
        agent_idx = self.exp_spec.agent_index_map[key]
        return self.obs_view[agent_idx].reshape(
            (self.num_envs, *self.exp_spec.observation_shapes[key])
        )

    def __str__(self):
        """"""
        my_dic = {agent: obs for agent, obs in zip(self.exp_spec.agents, self.obs_view)}
        return f"{my_dic}"

    def __repr__(self):
        return self.__str__()

    def __contains__(self, key):  # key is agent_id
        return key in self.exp_spec.agent_index_map.keys()

    def __len__(self):
        return len(self.exp_spec.agents)

    def __iterate_kv(self):
        for key in self.exp_spec.agents:
            yield (key, self.__getitem__(key))

    def __iter__(self):
        return self.__iterate_kv()

    def keys(self):
        for k, _ in self.__iterate_kv():
            yield k

    def values(self):
        for _, v in self.__iterate_kv():
            yield v

    def items(self):
        return self.__iterate_kv()

    def get(self, key):
        try:
            return self.__getitem__(key)
        except KeyError:
            return None


class SharedMemory:
    """Class to hold the shared memory object that each of the subprocesses will write their observation to.

    :param num_envs: Number of environments to vectorize
    :type num_envs: int
    :param exp_spec: Experience specification
    :type exp_spec: PettingZooExperienceSpec
    :param context: Multiprocessing context
    :type context: BaseContext

    """

    def __init__(self, num_envs, exp_spec, context):
        self.shared_memory = []
        for agent in exp_spec.agents:
            shared_memory = context.Array(
                exp_spec.single_observation_space[agent].dtype.char,
                exp_spec.observation_widths[agent] * num_envs,
            )
            self.shared_memory.append(shared_memory)


def get_placeholder_value(agent, transition_name, observation_shapes=None):
    """When an agent is killed, used to obtain a placeholder value to return for associated experience.

    :param agent: Agent ID
    :type agent: str
    :param transition_name: Name of the transition
    :type transition_name: str
    :param observations: Observations numpy array backed by RawArray, defaults to None
    :type observations: agilerl.vector.pz_async_vec_env.Observations, optional
    """
    match transition_name:
        case "reward":
            return 0
        case "truncated":
            return False
        case "terminated":
            return True
        case "info":
            return {}
        case "observation":
            return -np.ones_like(observation_shapes[agent])


def process_transition(transitions, observation_shapes, transition_names, agents):
    """Process transition, adds in placeholder values for killed sub-agents

    :param transitions: Tuple of environment transition
    :type transitions: Tuple[Any]
    :param observation_shapes: Observation shapes
    :type observation_shapes: Dict[str, Tuple[int]]
    :param transition_names: Names associated to transitions
    :type transition_names: List[str]
    :param agents: List of sub-agent names
    :type agents: List[str]
    """
    transition_list = list(transitions)
    for transition, name in zip(transition_list, transition_names):
        transition = {
            agent: (
                transition[agent]
                if agent in transition.keys()
                else get_placeholder_value(agent, name, observation_shapes)
            )
            for agent in agents
        }
    return transition_list


def set_env_obs(index, observation, shared_memory, widths, dtypes):
    """Set the observation for a given environment

    :param index: Environment index
    :type index: int
    :param observation: Observation from env.step or env.reset
    :type observation: Dict[str, np.ndarray]
    :param shared_memory: Shared memory
    :type shared_memory: mp.Array
    :param widths: Flattened observation widths
    :type widths: Dict[str, int]
    :param dtypes: Observation dtypes
    :type dtypes: Dict[str, np.dtype]
    """
    for agent_idx, (agent, obs) in enumerate(observation.items()):
        dest = np.frombuffer(shared_memory[agent_idx].get_obj(), dtype=dtypes[agent])
        np.copyto(
            dest[index * widths[agent] : (index + 1) * widths[agent]],
            np.asarray(obs, dtype=dtypes[agent]).flatten(),
        )


def _async_worker(
    index,
    env_fn,
    pipe,
    parent_pipe,
    shared_memory,
    error_queue,
    observation_shapes,
    observation_widths,
    observation_dtypes,
    agents,
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
    :param shared_memory: List of shared memories.
    :type shared_memory: List[multiprocessing.Array]
    :param error_queue: Queue object for collecting subprocess errors to communicate back to the main process
    :type error_queue: mp.Queue
    :param observation_shapes: Shapes of observations
    :type observation_shapes: Dict[str, Tuple[int]]
    :param observation_widths: Flattened observation widths
    :type observation_widths: Dict[str, int]
    :param observation_dtypes: Observation dtypes
    :type observation_dtypes: Dict[str, np.dtype]
    :param agents: Sub-agent names
    :type agents: str

    """
    env = env_fn()
    autoreset = False
    parent_pipe.close()

    try:
        while True:
            command, data = pipe.recv()
            if command == "reset":
                observation, info = process_transition(
                    env.reset(**data),
                    observation_shapes,
                    ["observation", "info"],
                    agents,
                )
                set_env_obs(
                    index,
                    observation,
                    shared_memory,
                    observation_widths,
                    observation_dtypes,
                )
                autoreset = False
                pipe.send(((info), True))
            elif command == "step":
                if autoreset:
                    observation, info = process_transition(
                        env.reset(), observation_shapes, ["observation", "info"], agents
                    )
                    reward = {agent: 0 for agent in agents}
                    terminated = {agent: False for agent in agents}
                    truncated = {agent: False for agent in agents}
                else:
                    data = {
                        possible_agent: (
                            np.array(data[idx]).squeeze()
                            if not isinstance(data[idx], int)
                            else data[idx]
                        )
                        for idx, possible_agent in enumerate(agents)
                    }
                    transition = env.step(data)
                    observation, reward, terminated, truncated, info = transition
                    observation, reward, terminated, truncated, info = (
                        process_transition(
                            transition,
                            observation_shapes,
                            [
                                "observation",
                                "reward",
                                "terminated",
                                "truncated",
                                "info",
                            ],
                            agents,
                        )
                    )
                autoreset = all(
                    [
                        term | trunc
                        for term, trunc in zip(terminated.values(), truncated.values())
                    ]
                )
                set_env_obs(
                    index,
                    observation,
                    shared_memory,
                    observation_widths,
                    observation_dtypes,
                )
                pipe.send(((reward, terminated, truncated, info), True))

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
