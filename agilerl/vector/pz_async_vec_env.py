"""An async vector pettingzoo environment"""

import multiprocessing as mp
import sys
import time
import traceback
from collections import OrderedDict, defaultdict
from copy import deepcopy
from enum import Enum
from multiprocessing.connection import Connection
from multiprocessing.sharedctypes import RawArray
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeAlias, TypeVar, Union

import numpy as np
from gymnasium import logger, spaces
from gymnasium.error import (
    AlreadyPendingCallError,
    ClosedEnvironmentError,
    NoAsyncCallError,
)
from gymnasium.vector.utils import CloudpickleWrapper, clear_mpi_env_vars
from pettingzoo import ParallelEnv

from agilerl.typing import ActionType, GymSpaceType, NumpyObsType, PzStepReturn
from agilerl.vector.pz_vec_env import PettingZooVecEnv

AgentID = TypeVar("AgentID")
ObsType = TypeVar("ObsType")
PzEnvType = Union[PettingZooVecEnv, ParallelEnv]
SharedMemoryType: TypeAlias = Union[RawArray, Tuple[RawArray, ...], Dict[str, RawArray]]


def reshape_observation(
    raw_data: NumpyObsType, space: spaces.Space, num_envs: int
) -> Any:
    """Reshape the raw data to the correct shape for the observation space.

    :param raw_data: The raw data to reshape
    :type raw_data: np.ndarray, Dict[str, np.ndarray], Tuple[np.ndarray, ...]
    :param space: The observation space
    :type space: gymnasium.spaces.Space
    :param num_envs: The number of environments
    :type num_envs: int

    :return: The reshaped data
    :rtype: Any
    """
    if isinstance(space, spaces.Dict):
        result = OrderedDict()
        for key, subspace in space.spaces.items():
            result[key] = reshape_observation(raw_data[key], subspace, num_envs)

    elif isinstance(space, spaces.Tuple):
        result = []
        for i, subspace in enumerate(space.spaces):
            result.append(reshape_observation(raw_data[i], subspace, num_envs))

        result = tuple(result)
    else:
        # Reshape to [num_envs, *shape]
        shape = space.shape if space.shape != () else (1,)
        reshaped = raw_data.reshape((num_envs, *shape))
        result = reshaped.astype(space.dtype)

    return result


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
    :param context: Context for multiprocessing. Choose between "spawn", "fork", or "forkserver".
    :type context: str, optional
    """

    processes: List[mp.Process]
    parent_pipes: List[Connection]
    error_queue: mp.Queue
    _state: AsyncState

    def __init__(
        self,
        env_fns: List[Callable[[], PzEnvType]],
        copy: bool = True,
        context: Optional[str] = None,
    ):
        # Core class attributes
        ctx = mp.get_context(context)
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
        self.active_agents = None
        self.previous_active = None
        self.copy = copy
        self.agents = dummy_env.possible_agents[:]
        action_spaces = {
            agent: dummy_env.action_space(agent) for agent in dummy_env.possible_agents
        }
        observation_spaces = {
            agent: dummy_env.observation_space(agent)
            for agent in dummy_env.possible_agents
        }
        dummy_env.close()
        del dummy_env
        super().__init__(
            len(env_fns),
            observation_spaces,
            action_spaces,
            self.possible_agents,
        )

        # Create the shared memory for sharing observations between subprocesses
        self._obs_buffer = create_shared_memory(
            num_envs=self.num_envs,
            obs_spaces=self._single_observation_spaces,
            context=ctx,
        )
        self.observations = Observations(
            shared_memory=self._obs_buffer,
            obs_spaces=self._single_observation_spaces,
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
                        self._obs_buffer,
                        self.error_queue,
                        self.agents,
                    ),
                )
                self.parent_pipes.append(parent_pipe)
                self.processes.append(process)
                process.daemon = True
                process.start()
                child_pipe.close()

        self._state = AsyncState.DEFAULT

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, NumpyObsType], Dict[str, Any]]:
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
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> None:
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

    def get_observations(self) -> Dict[str, NumpyObsType]:
        """Get the observations from the environments.

        :return: Observations from the environments
        :rtype: Dict[str, NumpyObsType]
        """
        return (
            {
                agent: deepcopy(self.observations[agent])
                for agent in self.observations.keys()
            }
            if self.copy
            else self.observations
        )

    def reset_wait(
        self, timeout: Optional[float] = None
    ) -> Tuple[Dict[str, NumpyObsType], Dict[str, Any]]:
        """Waits for the calls triggered by :meth:`reset_async` to finish and returns the results.

        :param timeout: Number of seconds before the call to ``reset_wait`` times out. If `None`, the call to
        ``reset_wait`` never times out, defaults to 0
        :type timeout: int | float | None, optional

        :return: Tuple of observations and infos
        :rtype: Tuple[Dict[str, NumpyObsType], Dict[str, Any]]
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
            self.get_observations(),
            infos,
        )

    def step_async(self, actions: List[List[ActionType]]) -> None:
        """
        Tell all the environments to start taking a step with the given actions.
        Call step_wait() to get the results of the step.
        You should not call this if a step_async run is already pending.

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

    def step_wait(self, timeout: Optional[float] = None) -> PzStepReturn:
        """
        Wait for the calls to :obj:`step` in each sub-environment to finish.

        :param timeout: Number of seconds before the call to ``step_wait`` times out. If `None`, the call to ``step_wait`` never times out, defaults to 0
        :type timeout: int | float | None, optional

        :return: Tuple of observations, rewards, dones, and infos
        :rtype: Tuple[Dict[str, NumpyObsType], Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, Any]]
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
                reward, term, trunc, info = env_step_return
                for agent in self.agents:
                    rewards[agent].append(reward[agent])
                    terminations[agent].append(term[agent])
                    truncations[agent].append(trunc[agent])

                infos = self._add_info(infos, info, env_idx)

        self._raise_if_errors(successes)
        self._state = AsyncState.DEFAULT

        rewards = {agent: np.array(rew) for agent, rew in rewards.items()}
        terminations = {agent: np.array(term) for agent, term in terminations.items()}
        truncations = {agent: np.array(trunc) for agent, trunc in truncations.items()}

        return (
            self.get_observations(),
            rewards,
            terminations,
            truncations,
            infos,
        )

    def render(self) -> Any:
        """
        Returns the rendered frames from the parallel environments.

        """
        return self.call("render")

    def call(self, name: str, *args: Any, **kwargs: Any) -> Any:
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

    def call_async(self, name: str, *args: Any, **kwargs: Any) -> None:
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

    def call_wait(self, timeout: Optional[float] = None) -> Any:
        """Calls all parent pipes and waits for the results.

        :param timeout: Number of seconds before the call to :meth:`call_wait` times out. If ``None`` (default),
        the call to :meth:`call_wait` never times out, defaults to 0
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

    def get_attr(self, name: str) -> Any:
        """
        Get a property from each parallel environment.

        :param name: Name of property to get from each individual environment
        :type name: str
        """
        return self.call(name)

    def set_attr(self, name: str, values: Any) -> None:
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

    def close_extras(
        self, timeout: Optional[float] = None, terminate: bool = False
    ) -> None:
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

    def _poll_pipe_envs(self, timeout: Optional[float] = None) -> bool:
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

    def _raise_if_errors(self, successes: List[bool]) -> None:
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

    def _assert_is_running(self) -> None:
        if self.closed:
            raise ClosedEnvironmentError(
                f"Trying to operate on `{type(self).__name__}`, after a call to `close()`."
            )

    def _add_info(
        self, vector_infos: Dict[str, Any], env_info: Dict[str, Any], env_num: int
    ) -> Dict[str, Any]:
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

    def __del__(self) -> None:
        """On deleting the object, checks that the vector environment is closed."""
        if not getattr(self, "closed", True) and hasattr(self, "_state"):
            self.close(terminate=True)
        if hasattr(self, "_obs_buffer"):
            del self._obs_buffer
        if hasattr(self, "observations"):
            del self.observations


class Observations:
    """
    Class for storing observations with a dictionary interface

    :param shared_memory: A RawArray that all envs write observations to.
    :type shared_memory: multiprocessing.RawArray
    :param obs_spaces: Dictionary of gymnasium observation spaces
    :type obs_spaces: Dict[str, gymnasiums.spaces.Space]
    :param num_envs: Number of environments
    :type num_envs: int
    """

    obs_view: Dict[str, NumpyObsType]

    def __init__(
        self,
        shared_memory: Dict[str, SharedMemoryType],
        obs_spaces: Dict[str, spaces.Space],
        num_envs: int,
    ):
        self.num_envs = num_envs
        self.shared_memory = shared_memory
        self.obs_view = {}
        self.agents = list(self.shared_memory.keys())
        self.obs_spaces = obs_spaces
        for agent, shm in shared_memory.items():
            agent_space = obs_spaces[agent]
            if isinstance(agent_space, spaces.Dict):
                obs_view = OrderedDict()
                for key, subspace in agent_space.spaces.items():
                    obs_view[key] = np.frombuffer(
                        shm[key].get_obj(), dtype=subspace.dtype
                    )
                self.obs_view[agent] = obs_view
            elif isinstance(agent_space, spaces.Tuple):
                obs_view = tuple(
                    np.frombuffer(shm[i].get_obj(), dtype=agent_space.spaces[i].dtype)
                    for i in range(len(agent_space.spaces))
                )
            else:
                obs_view = np.frombuffer(shm.get_obj(), dtype=agent_space.dtype)

            self.obs_view[agent] = obs_view

    def __getitem__(self, agent: str) -> NumpyObsType:
        """
        Get agent observation given a key (agent_id)
        """
        return reshape_observation(
            self.obs_view[agent], self.obs_spaces[agent], self.num_envs
        )

    def __str__(self) -> str:
        return f"{self.obs_view}"

    def __repr__(self) -> str:
        return self.__str__()

    def __contains__(self, key: str) -> bool:
        return key in self.obs_view.keys()

    def __len__(self) -> int:
        return len(self.agents)

    def __iterate_kv(self) -> Any:
        for key in self.obs_view:
            yield (key, self.__getitem__(key))

    def __iter__(self) -> Any:
        return self.__iterate_kv()

    def keys(self) -> Any:
        for k, _ in self.__iterate_kv():
            yield k

    def values(self) -> Any:
        for _, v in self.__iterate_kv():
            yield v

    def items(self) -> Any:
        return self.__iterate_kv()

    def get(self, key: str) -> Optional[np.ndarray]:
        try:
            return self.__getitem__(key)
        except KeyError:
            return None


def _create_memory_array(
    num_envs: int, obs_space: spaces.Space, context: Any
) -> RawArray:
    """Create a shared memory array for a given observation space.

    :param num_envs: Number of environments
    :type num_envs: int
    :param obs_space: Observation space
    :type obs_space: gymnasium.spaces.Space
    :param context: Multiprocessing context
    :type context: Any
    """
    return context.Array(obs_space.dtype.char, num_envs * int(np.prod(obs_space.shape)))


def create_shared_memory(
    num_envs: int, obs_spaces: Dict[str, spaces.Space], context: Any
) -> Dict[str, RawArray]:
    """
    Create shared memory for multi-agent observations.

    :param num_envs: Number of environments
    :type num_envs: int
    :param obs_spaces: Dictionary of gymnasium observation spaces
    :type obs_spaces: Dict[str, gymnasiums.spaces.Space]
    :param context: Multiprocessing context
    :type context: Any
    """
    shared_memory = {}
    for agent, obs_space in obs_spaces.items():
        if isinstance(obs_space, spaces.Dict):
            shm = OrderedDict()
            for key, subspace in obs_space.spaces.items():
                shm[key] = _create_memory_array(num_envs, subspace, context)
        elif isinstance(obs_space, spaces.Tuple):
            shm = tuple(
                _create_memory_array(num_envs, subspace, context)
                for subspace in obs_space.spaces
            )
        else:
            shm = _create_memory_array(num_envs, obs_space, context)

        shared_memory[agent] = shm

    return shared_memory


def get_placeholder_value(
    agent: str,
    transition_name: str,
    obs_spaces: Optional[Dict[str, spaces.Space]] = None,
) -> Any:
    """Used to obtain a placeholder value to return for associated experience when an
    agent is killed or is inactive for the current step.

    :param agent: Agent ID
    :type agent: str
    :param transition_name: Name of the transition
    :type transition_name: str
    :param obs_spaces: Observation spaces
    :type obs_spaces: Dict[str, gymnasium.spaces.Space]

    :return: Placeholder value
    :rtype: Any
    """
    match transition_name:
        case "reward":
            return np.nan
        case "truncated":
            return np.nan
        case "terminated":
            return np.nan
        case "info":
            return {}
        case "observation":
            if obs_spaces is None:
                return None

            agent_space = obs_spaces[agent]
            if isinstance(agent_space, spaces.Dict):
                # For Dict spaces, create a dictionary of -1 arrays
                return {k: np.full(v.shape, np.nan) for k, v in agent_space.items()}
            elif isinstance(agent_space, spaces.Tuple):
                # For Tuple spaces, create a tuple of -1 arrays
                return tuple(np.full(s.shape, np.nan) for s in agent_space)
            else:
                # For normal spaces
                return np.full(agent_space.shape, np.nan)


def process_transition(
    transitions: Tuple[Dict[str, NumpyObsType], ...],
    obs_spaces: Dict[str, spaces.Space],
    transition_names: List[str],
    agents: List[str],
) -> List[Dict[str, NumpyObsType]]:
    """Process transition, adds in placeholder values for killed agents.

    :param transitions: Tuple of environment transition
    :type transitions: Tuple[Dict[str, NumpyObsType], ...]
    :param obs_spaces: Observation spaces
    :type obs_spaces: Dict[str, gymnasium.spaces.Space]
    :param transition_names: Names associated to transitions
    :type transition_names: List[str]
    :param agents: List of agent names
    :type agents: List[str]
    """
    transition_list = []
    for transition, name in zip(list(transitions), transition_names):
        transition = {
            agent: (
                transition[agent]
                if agent in transition.keys()
                else get_placeholder_value(agent, name, obs_spaces)
            )
            for agent in agents
        }
        transition_list.append(transition)
    return transition_list


def write_vector_observation(
    index: int,
    observation: np.ndarray,
    shared_memory: RawArray,
    obs_space: spaces.Space,
) -> None:
    """Write a vector observation to the shared memory.

    :param index: Environment index
    :type index: int
    :param observation: Observation from env.step or env.reset
    :type observation: np.ndarray
    :param shared_memory: Shared memory
    :type shared_memory: multiprocessing.RawArray
    :param obs_space: Observation space
    :type obs_space: gymnasium.spaces.Space
    """
    size = int(np.prod(obs_space.shape))
    dtype = obs_space.dtype
    dest = np.frombuffer(shared_memory.get_obj(), dtype=dtype)
    np.copyto(
        dest[index * size : (index + 1) * size],
        np.asarray(observation, dtype=dtype).flatten(),
    )


def write_to_shared_memory(
    index: int,
    observation: Dict[str, NumpyObsType],
    shared_memory: Dict[str, RawArray],
    obs_space: Dict[str, GymSpaceType],
) -> None:
    """Set the observation for a given environment. Handles Dict and Tuple spaces.

    :param index: Environment index
    :type index: int
    :param observation: Observation from env.step or env.reset
    :type observation: Dict[str, np.ndarray]
    :param shared_memory: Shared memory
    :type shared_memory: Dict[str, mp.Array | Tuple[mp.Array, ...] | Dict[str, mp.Array]]
    :param obs_space: Observation space dictionary
    :type obs_space: Dict[str, gymnasium.spaces.Space]
    """
    for agent in shared_memory.keys():
        agent_space = obs_space[agent]
        obs = observation[agent]
        if isinstance(agent_space, spaces.Dict):
            for key, subspace in agent_space.spaces.items():
                write_vector_observation(
                    index, obs[key], shared_memory[agent][key], subspace
                )

        elif isinstance(agent_space, spaces.Tuple):
            for i, subspace in enumerate(agent_space.spaces):
                write_vector_observation(
                    index, obs[i], shared_memory[agent][i], subspace
                )
        else:
            write_vector_observation(index, obs, shared_memory[agent], agent_space)


def _async_worker(
    index: int,
    env_fn: Callable[[], ParallelEnv],
    pipe: Connection,
    parent_pipe: Connection,
    shared_memory: Dict[str, RawArray],
    error_queue: mp.Queue,
    agents: List[str],
) -> None:
    """
    Worker function to run the environment in a subprocess.

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
    :param agents: Agent names
    :type agents: str

    """
    env = env_fn()
    observation_space = {agent: env.observation_space(agent) for agent in agents}
    parent_pipe.close()

    # Need to keep track of the active agents in the environment
    try:
        while True:
            command, data = pipe.recv()
            if command == "reset":
                obs, info = env.reset(**data)
                transition = obs, info
                observation, info = process_transition(
                    transition,
                    observation_space,
                    ["observation", "info"],
                    agents,
                )
                write_to_shared_memory(
                    index, observation, shared_memory, observation_space
                )
                pipe.send((info, True))
            elif command == "step":
                data = {
                    active_agent: (
                        np.array(data[active_agent]).squeeze()
                        if not isinstance(data[active_agent], int)
                        else data[active_agent]
                    )
                    for active_agent in data
                }
                observation, reward, terminated, truncated, info = env.step(data)
                if all(
                    [
                        term | trunc
                        for term, trunc in zip(terminated.values(), truncated.values())
                    ]
                ):
                    observation, info = env.reset()

                transition = observation, reward, terminated, truncated, info
                observation, reward, terminated, truncated, info = process_transition(
                    transition,
                    observation_space,
                    [
                        "observation",
                        "reward",
                        "terminated",
                        "truncated",
                        "info",
                    ],
                    agents,
                )
                write_to_shared_memory(
                    index, observation, shared_memory, observation_space
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
