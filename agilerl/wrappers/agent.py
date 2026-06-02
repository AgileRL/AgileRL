import copy
from abc import ABC
from collections import OrderedDict
from collections.abc import Callable
from functools import partial
from typing import Any, Generic, TypeVar

import dill
import numpy as np
import torch
from gymnasium import spaces
from tensordict import TensorDictBase

from agilerl.algorithms import PPO
from agilerl.algorithms.core import (
    EvolvableAlgorithm,
    MultiAgentRLAlgorithm,
    RLAlgorithm,
)
from agilerl.algorithms.core.base import get_checkpoint_dict
from agilerl.typing import (
    ActionReturnType,
    DeviceType,
    ExperiencesType,
    ObservationType,
)
from agilerl.utils.algo_utils import obs_to_tensor, stack_experiences
from agilerl.wrappers.utils import RunningMeanStd

AgentType = TypeVar("AgentType", bound=RLAlgorithm | MultiAgentRLAlgorithm)
MARLObservationType = dict[str, ObservationType]
SelfAgentWrapper = TypeVar("SelfAgentWrapper", bound="AgentWrapper")


class AgentWrapper(ABC, Generic[AgentType]):
    """Base class for all agent wrappers. Agent wrappers are used to apply an
    additional functionality to the ``get_action()`` and ``learn()`` methods of
    an ``EvolvableAlgorithm`` instance.

    :param agent: Agent to be wrapped
    :type agent: AgentType
    """

    wrapped_get_action: Callable
    wrapped_learn: Callable

    def __init__(self, agent: AgentType) -> None:
        self.agent = agent
        self.observation_space = agent.observation_space
        self.action_space = agent.action_space
        self.multi_agent = isinstance(agent, MultiAgentRLAlgorithm)

        # Wrap the agent's methods
        self.wrapped_get_action = agent.get_action
        self.wrapped_learn = agent.learn

        self.agent.get_action = partial(self.get_action)
        self.agent.learn = partial(self.learn)

    @property
    def training(self) -> bool:
        """Return the training status of the agent.

        :return: Training status of the agent
        :rtype: bool
        """
        return self.agent.training

    @property
    def device(self) -> DeviceType:
        """Return the device of the agent.

        :return: Device of the agent
        :rtype: DeviceType
        """
        return self.agent.device if not hasattr(self.agent, "rollout_buffer") else "cpu"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.agent})"

    def __getstate__(self) -> dict[str, Any]:
        return {
            "agent": self.agent,
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "multi_agent": self.multi_agent,
        }

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.agent: AgentType = state["agent"]
        self.observation_space = state["observation_space"]
        self.action_space = state["action_space"]
        self.multi_agent = state["multi_agent"]

        self.wrapped_get_action = self.agent.get_action
        self.wrapped_learn = self.agent.learn

        self.agent.get_action = partial(self.get_action)
        self.agent.learn = partial(self.learn)

    def __getattr__(self, name: str) -> Any:
        """Get attribute of the wrapper.

        :param name: The name of the attribute.
        :type name: str
        :return: The attribute of the network.
        :rtype: Any
        """
        try:
            return super().__getattribute__(name)
        except AttributeError:
            return getattr(self.agent, name)

    def __setattr__(self, name: str, value: Any | AgentType) -> None:
        if name == "agent" or not hasattr(self, "agent"):
            super().__setattr__(name, value)
        if hasattr(self.agent, name):
            object.__setattr__(self.agent, name, value)
        else:
            super().__setattr__(name, value)

    def clone(self, index: int | None = None, wrap: bool = True) -> SelfAgentWrapper:
        """Clone the wrapper with the underlying agent.

        :param index: Index of the agent in a population, defaults to None
        :type index: int | None, optional
        :param wrap: If True, wrap the models in the clone with the accelerator, defaults to False
        :type wrap: bool, optional

        :return: Cloned agent wrapper
        :rtype: SelfAgentWrapper
        """
        agent_clone = self.agent.clone(index, wrap)

        input_args = EvolvableAlgorithm.inspect_attributes(self, input_args_only=True)
        input_args.pop("agent", None)

        clone = self.__class__(agent_clone, **input_args)
        return EvolvableAlgorithm.copy_attributes(self, clone)

    def save_checkpoint(self, path: str) -> None:
        """Save a checkpoint of agent properties and network weights to path.

        :param path: Location to save checkpoint at
        :type path: string
        """
        checkpoint = get_checkpoint_dict(self.agent)

        del checkpoint["learn"]
        del checkpoint["get_action"]

        # Add wrapper attributes to checkpoint
        checkpoint["wrapper_cls"] = self.__class__
        checkpoint["wrapper_init_dict"] = EvolvableAlgorithm.inspect_attributes(
            self,
            input_args_only=True,
        )
        checkpoint["wrapper_attrs"] = EvolvableAlgorithm.inspect_attributes(self)

        checkpoint["wrapper_init_dict"].pop("agent")
        checkpoint["wrapper_attrs"].pop("agent")

        # Save checkpoint
        torch.save(
            checkpoint,
            path,
            pickle_module=dill,
        )

    def load_checkpoint(self, path: str) -> None:
        """Load a checkpoint of agent properties and network weights from path.

        :param path: Location to load checkpoint from
        :type path: string
        """
        checkpoint = torch.load(path, pickle_module=dill, weights_only=False)

        # Load agent properties and network weights
        self.agent.load_checkpoint(path)

        # Load wrapper attributes
        for key, value in checkpoint["wrapper_attrs"].items():
            setattr(self, key, value)

    def get_action(
        self,
        obs: ObservationType | MARLObservationType,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Return the action from the agent.

        :param obs: Observation from the environment
        :type obs: ObservationType | MARLObservationType
        :param args: Additional positional arguments
        :type args: Any
        :param kwargs: Additional keyword arguments
        :type kwargs: Any

        :return: Action from the agent
        :rtype: Any
        """
        return self.wrapped_get_action(obs, *args, **kwargs)

    def learn(
        self, experiences: ExperiencesType | None = None, *args: Any, **kwargs: Any
    ) -> Any:
        """Learns from the experiences.

        :param experiences: Experiences from the environment
        :type experiences: ExperiencesType
        :param args: Additional positional arguments
        :type args: Any
        :param kwargs: Additional keyword arguments
        :type kwargs: Any

        :return: Learning information
        :rtype: Any
        """
        if experiences is None:
            return self.wrapped_learn(*args, **kwargs)

        return self.wrapped_learn(experiences, *args, **kwargs)


RunningStatsType = (
    RunningMeanStd | dict[str, RunningMeanStd] | tuple[RunningMeanStd, ...]
)


class RSNorm(AgentWrapper[AgentType]):
    """Wrapper to normalize observations such that each coordinate is centered with unit variance.
    Handles both single and multi-agent settings, as well as Dict and Tuple observation spaces.

    The normalization statistics are only updated when the agent is in training mode. This can be
    disabled during inference through ``agent.set_training_mode(False)``.

    .. warning::
        This wrapper is currently only supported for off-policy algorithms since it relies on
        experiences passed as a :class:`tensordict.TensorDict` (as produced by the replay
        buffer). For on-policy PPO, experiences may be omitted so the wrapper normalizes
        observations stored in the rollout buffer before learning.

    :param agent: Agent to be wrapped
    :type agent: RLAlgorithm, MultiAgentRLAlgorithm
    :param epsilon: Small value to avoid division by zero, defaults to 1e-4
    :type epsilon: float, optional
    :param norm_obs_keys: List of observation keys to normalize, defaults to None
    :type norm_obs_keys: List | None
    """

    obs_rms: RunningStatsType | dict[str, RunningStatsType]

    def __init__(
        self,
        agent: AgentType,
        epsilon: float = 1e-4,
        norm_obs_keys: list[str] | None = None,
    ) -> None:
        super().__init__(agent)

        self.norm_obs_keys = norm_obs_keys
        if self.multi_agent:
            self.obs_rms = OrderedDict()
            for agent_id, obs_space in self.observation_space.items():
                self.obs_rms[agent_id] = RSNorm.build_rms(
                    obs_space,
                    epsilon,
                    norm_obs_keys,
                    self.device,
                )
        else:
            self.obs_rms = RSNorm.build_rms(
                self.observation_space,
                epsilon,
                norm_obs_keys,
                self.device,
            )

    @staticmethod
    def build_rms(
        observation_space: spaces.Space,
        epsilon: float = 1e-4,
        norm_obs_keys: list[str] | None = None,
        device: DeviceType = "cpu",
    ) -> RunningMeanStd | dict[str, RunningMeanStd] | tuple[RunningMeanStd, ...]:
        """Build the RunningMeanStd object(s) based on the observation space.

        :param observation_space: Observation space of the agent
        :type observation_space: spaces.Space
        :return: RunningMeanStd object(s)
        :rtype: RunningMeanStd | dict[str, RunningMeanStd] | tuple[RunningMeanStd, ...]
        """
        if isinstance(observation_space, spaces.Dict):
            spaces_map = observation_space.spaces
            if norm_obs_keys is not None:
                spaces_map = {
                    key: value
                    for key, value in spaces_map.items()
                    if key in norm_obs_keys
                }

            return {
                key: RunningMeanStd(epsilon, shape=value.shape, device=device)
                for key, value in spaces_map.items()
            }

        if isinstance(observation_space, spaces.Tuple):
            return tuple(
                RunningMeanStd(epsilon, shape=value.shape, device=device)
                for value in observation_space.spaces
            )

        return RunningMeanStd(epsilon, shape=observation_space.shape, device=device)

    def _normalize_observation(
        self,
        observation: ObservationType,
        *,
        obs_rms: RunningStatsType | None = None,
    ) -> ObservationType:
        """Normalize the observation using the RunningMeanStd object(s).

        :param observation: Observation from the environment
        :type observation: ObservationType
        :param obs_rms: Optional running-statistics object(s) to use instead of
            ``self.obs_rms`` (required for per-agent stats in multi-agent mode).

        :return: Normalized observation
        :rtype: ObservationType
        """
        obs_rms = self.obs_rms if obs_rms is None else obs_rms
        if isinstance(obs_rms, dict):
            norm_observation = {}
            for key, rms in obs_rms.items():
                norm_observation[key] = (observation[key] - rms.mean) / (
                    rms.var + rms.epsilon
                ).sqrt()

            observation = norm_observation
        elif isinstance(obs_rms, tuple):
            norm_observation = []
            for i, rms in enumerate(obs_rms):
                norm_obs = (observation[i] - rms.mean) / (rms.var + rms.epsilon).sqrt()
                norm_observation.append(norm_obs)

            observation = tuple(norm_observation)
        else:
            observation = (observation - obs_rms.mean) / (
                obs_rms.var + obs_rms.epsilon
            ).sqrt()

        return observation

    def normalize_observation(self, observation: ObservationType) -> ObservationType:
        """Normalize the observation using the RunningMeanStd object(s).

        :param observation: Observation from the environment
        :type observation: ObservationType

        :return: Normalized observation
        :rtype: ObservationType
        """
        if self.multi_agent:
            for agent_id, obs in observation.items():
                agent_rms = self.obs_rms[agent_id]
                observation[agent_id] = self._normalize_observation(
                    obs, obs_rms=agent_rms
                )
            return observation

        return self._normalize_observation(observation)

    def _update_statistics(
        self,
        observation: ObservationType,
        *,
        obs_rms: RunningStatsType | None = None,
    ) -> None:
        """Update the running statistics using the observation.

        :param observation: Observation from the environment
        :type observation: ObservationType
        :param obs_rms: Optional running-statistics object(s) to use instead of
            ``self.obs_rms`` (required for per-agent stats in multi-agent mode).
        """
        obs_rms = self.obs_rms if obs_rms is None else obs_rms
        if isinstance(obs_rms, dict):
            for key, rms in obs_rms.items():
                rms.update(observation[key])
        elif isinstance(obs_rms, tuple):
            for i, rms in enumerate(obs_rms):
                rms.update(observation[i])
        else:
            obs_rms.update(observation)

    def update_statistics(self, observation: ObservationType) -> None:
        """Update the running statistics using the observation.

        :param observation: Observation from the environment
        :type observation: ObservationType
        """
        if self.multi_agent:
            for agent_id, obs in observation.items():
                self._update_statistics(obs, obs_rms=self.obs_rms[agent_id])
        else:
            self._update_statistics(observation)

    def get_action(self, obs: ObservationType, *args: Any, **kwargs: Any) -> Any:
        """Return the action from the agent after normalizing the observation.

        :param obs: Observation from the environment
        :type obs: ObservationType

        :return: Action from the agent
        :rtype: Any
        """
        obs = obs_to_tensor(obs, self.device)

        # Update running statistics only when in training mode
        if self.training:
            self.update_statistics(obs)

        obs = self.normalize_observation(obs)
        return self.wrapped_get_action(obs, *args, **kwargs)

    def learn(
        self,
        experiences: TensorDictBase | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Learns from the experiences after normalizing the observations.

        :param experiences: Experiences from the environment
        :type experiences: TensorDictBase
        :param args: Additional positional arguments
        :type args: Any
        :param kwargs: Additional keyword arguments
        :type kwargs: Any

        :return: Learning information
        :rtype: Any
        """
        if experiences is None:
            if not isinstance(self.agent, PPO):
                msg = "Experiences must be provided unless the wrapped agent is PPO."
                raise ValueError(
                    msg,
                )

            buffer_size = (
                self.agent.rollout_buffer.capacity
                if not self.agent.rollout_buffer.full
                else self.agent.rollout_buffer.pos
            )
            valid_data = self.agent.rollout_buffer.buffer[:buffer_size]
            valid_data["observations"] = self.normalize_observation(
                valid_data["observations"],
            )
            valid_data["next_observations"] = self.normalize_observation(
                valid_data["next_observations"],
            )
            self.agent.rollout_buffer.buffer[:buffer_size] = valid_data

            return self.wrapped_learn(*args, **kwargs)

        # NOTE: All AgileRL off-policy algorithms now expect experiences to be a TensorDict.
        if not isinstance(experiences, TensorDictBase):
            msg = "Experiences must be a TensorDict."
            raise ValueError(msg)

        experiences["obs"] = self.normalize_observation(experiences["obs"])
        experiences["next_obs"] = self.normalize_observation(experiences["next_obs"])
        return self.wrapped_learn(experiences, *args, **kwargs)


class AsyncAgentsWrapper(AgentWrapper[MultiAgentRLAlgorithm]):
    """Wrapper for multi-agent algorithms that solve environments with asynchronous agents (i.e. environments
    where agents don't return observations with the same frequency).

    .. note::
        This currently supports IPPO, MADDPG, and MATD3.

    :param agent: MultiAgentRLAlgorithm instance to be wrapped.
    :type agent: MultiAgentRLAlgorithm
    """

    def __init__(self, agent: MultiAgentRLAlgorithm) -> None:
        super().__init__(agent)

        assert self.agent.algo in {"IPPO", "MADDPG", "MATD3"}, (
            "AsyncAgentsWrapper is currently only supported for IPPO, MADDPG, and MATD3."
        )

        # State caches for Zero-Order Hold (ZOH) off-policy tracking
        self._last_obs: dict[str, ObservationType] = {}
        self._last_env_actions: dict[str, np.ndarray] = {}
        self._last_raw_actions: dict[str, np.ndarray] = {}

    def _is_off_policy(self) -> bool:
        """Check if the wrapped algorithm is an off-policy algorithm.

        :return: True if the algorithm is MADDPG or MATD3, False otherwise.
        :rtype: bool
        """
        return self.agent.algo in {"MADDPG", "MATD3"}

    def _get_active_mask(self, agent_obs: ObservationType) -> np.ndarray:
        """Returns a 1D boolean array indicating which environment vector rows are active.

        An environment index is considered inactive if all values within its
        observation row are NaN.

        :param agent_obs: Observation structure for a single agent type.
        :type agent_obs: ObservationType
        :return: Boolean array where True means active and False means inactive.
        :rtype: np.ndarray
        """
        if isinstance(agent_obs, dict):
            sample = next(iter(agent_obs.values()))
        elif isinstance(agent_obs, tuple):
            sample = agent_obs[0]
        else:
            sample = agent_obs

        sample_arr = np.asarray(sample)
        if sample_arr.ndim == 1:  # Non-vectorized environment instance
            return np.array([not np.isnan(sample_arr).all()])

        # Vectorized context: True if any value in the row is not NaN
        return ~np.isnan(sample_arr).all(axis=1)

    def _merge_with_cache(
        self, current: Any, cached: Any, active_mask: np.ndarray
    ) -> Any:
        """Recursively merges current step outputs with historical cached entries.

        Utilizes clean NumPy boolean indexing to overwrite historical, zero-order hold values
        with active decision payloads wherever the active_mask evaluates to True.

        :param current: The modern data dictionary, tuple, or array payload.
        :type current: Any
        :param cached: The historical zero-order hold structural equivalent.
        :type cached: Any
        :param active_mask: 1D boolean evaluation mask.
        :type active_mask: np.ndarray
        :return: Merged structural object filled across all agent indices.
        :rtype: Any
        """
        if isinstance(current, dict):
            return {
                k: self._merge_with_cache(current[k], cached[k], active_mask)
                for k in current
            }
        if isinstance(current, tuple):
            return tuple(
                self._merge_with_cache(c, ca, active_mask)
                for c, ca in zip(current, cached, strict=True)
            )

        curr_arr = np.asarray(current)
        cache_arr = np.asarray(cached)

        # Ensure array dimensions match vector specifications smoothly
        if curr_arr.ndim == 1:
            curr_arr = curr_arr.reshape(1, -1)
        if cache_arr.ndim == 1:
            cache_arr = cache_arr.reshape(1, -1)

        merged = cache_arr.copy()
        merged[active_mask] = curr_arr[active_mask]
        return merged

    def _get_action_off_policy(
        self, obs: MARLObservationType, *args: Any, **kwargs: Any
    ) -> tuple[
        dict[str, np.ndarray],
        dict[str, np.ndarray],
        dict[str, np.ndarray],
        MARLObservationType,
    ]:
        """Executes zero-order hold and generates decision masking parameters for off-policy tracking.

        Filters out inactive spaces before polling wrapped inference models, filling in missing historical
        actions and observations to supply a globally valid step representation to centralized multi-agent critics.

        :param obs: Raw observation dictionary containing optional NaN entities.
        :type obs: MARLObservationType
        :return: Tuple containing joint env actions, raw actions, active replay masks, and synthesized joint observations.
        :rtype: tuple
        """
        obs_for_policy = copy.deepcopy(obs)
        _, filtered_obs = self.extract_inactive_agents(obs_for_policy)

        action_return = self.wrapped_get_action(filtered_obs, *args, **kwargs)
        if not isinstance(action_return, tuple) or len(action_return) < 2:
            msg = "MADDPG/MATD3 must return a tuple of (env_actions, raw_actions)."
            raise TypeError(msg)

        policy_env = dict(action_return[0])
        policy_raw = dict(action_return[1])

        joint_obs: MARLObservationType = {}
        joint_env_actions: dict[str, np.ndarray] = {}
        joint_raw_actions: dict[str, np.ndarray] = {}
        active_masks: dict[str, np.ndarray] = {}

        for agent_id in self.agent.agent_ids:
            if agent_id in obs:
                active_mask = self._get_active_mask(obs[agent_id])
            elif agent_id in self._last_obs:
                batch_size = len(np.atleast_1d(self._last_obs[agent_id]))
                active_mask = np.zeros(batch_size, dtype=bool)
            else:
                active_mask = np.array([False])

            active_masks[agent_id] = active_mask.astype(np.float32).reshape(-1, 1)

            # 2. Synchronize Joint Observation spaces
            if agent_id not in self._last_obs:
                space_shape = self.agent.possible_observation_spaces[agent_id].shape
                joint_obs[agent_id] = obs.get(agent_id, np.zeros(space_shape))
            else:
                current_obs = obs.get(agent_id, self._last_obs[agent_id])
                joint_obs[agent_id] = self._merge_with_cache(
                    current_obs, self._last_obs[agent_id], active_mask
                )

            self._last_obs[agent_id] = copy.deepcopy(joint_obs[agent_id])

            if agent_id not in self._last_env_actions:
                action_shape = self.agent.possible_action_spaces[agent_id].shape
                action_dim = np.prod(action_shape) if action_shape else 1
                self._last_env_actions[agent_id] = np.zeros(
                    (len(active_mask), action_dim)
                )
                self._last_raw_actions[agent_id] = np.zeros(
                    (len(active_mask), action_dim)
                )

            current_env = policy_env.get(agent_id, self._last_env_actions[agent_id])
            current_raw = policy_raw.get(agent_id, self._last_raw_actions[agent_id])

            joint_env_actions[agent_id] = self._merge_with_cache(
                current_env, self._last_env_actions[agent_id], active_mask
            )
            joint_raw_actions[agent_id] = self._merge_with_cache(
                current_raw, self._last_raw_actions[agent_id], active_mask
            )

            self._last_env_actions[agent_id] = joint_env_actions[agent_id].copy()
            self._last_raw_actions[agent_id] = joint_raw_actions[agent_id].copy()

        return joint_env_actions, joint_raw_actions, active_masks, joint_obs

    def extract_inactive_agents(
        self,
        obs: MARLObservationType,
    ) -> tuple[dict[str, np.ndarray], MARLObservationType]:
        """Extract the inactive agents from an observation. Inspects each key in the
        observation dictionary and, if all the values are `np.nan` (as set by
        ``AsyncPettingZooVecEnv``), the agent is considered inactive and removed from
        the observation dictionary.

        :param obs: Observation dictionary
        :type obs: MARLObservationType

        :return: Tuple of inactive agents and filtered observations
        :rtype: tuple[dict[str, np.ndarray], MARLObservationType]
        """
        inactive_agents = {}
        agents_to_remove = []

        for agent_id, agent_obs in obs.items():
            if isinstance(agent_obs, dict):
                sample = next(iter(agent_obs.values()))
            elif isinstance(agent_obs, tuple):
                sample = agent_obs[0]
            else:
                sample = agent_obs

            if len(sample.shape) == 1:
                continue

            active_mask: np.ndarray = ~np.isnan(sample).all(axis=1)

            if active_mask.all():
                continue

            inactive_agent_indices = np.where(~active_mask)[0]

            if not active_mask.any():
                agents_to_remove.append(agent_id)
                continue

            if isinstance(agent_obs, dict):
                obs[agent_id] = {k: v[active_mask] for k, v in agent_obs.items()}
            elif isinstance(agent_obs, tuple):
                obs[agent_id] = tuple(v[active_mask] for v in agent_obs)
            else:
                obs[agent_id] = agent_obs[active_mask]

            inactive_agents[agent_id] = inactive_agent_indices

        for agent_id in agents_to_remove:
            obs.pop(agent_id)

        return inactive_agents, obs

    def stack_experiences(
        self, experiences: ExperiencesType | list[ExperiencesType]
    ) -> ExperiencesType:
        """Stacks the collected experiences.

        :param experiences: Experiences from the environment
        :type experiences: ExperiencesType | list[ExperiencesType]
        :return: Stacked experiences
        :rtype: ExperiencesType
        """
        if isinstance(experiences, dict):
            return {
                key: self.stack_experiences(val) for key, val in experiences.items()
            }

        if not isinstance(experiences, list):
            return experiences

        if len(experiences) > 0:
            return stack_experiences(experiences, to_torch=False)[0]

        return None

    def _insert_placeholder_actions(
        self,
        action_dict: dict[str, np.ndarray],
        inactive_agents: dict[str, np.ndarray],
    ) -> dict[str, np.ndarray]:
        """Insert placeholder actions for inactive agents back into action dict for IPPO formatting."""
        for agent_id, inactive_array in inactive_agents.items():
            if agent_id not in action_dict:
                continue

            agent_action = action_dict[agent_id]
            if agent_action is None:
                continue

            if len(agent_action.shape) == 1:
                placeholder_shape = ()
            else:
                placeholder_shape = agent_action.shape[1:]

            if np.issubdtype(agent_action.dtype, np.integer):
                placeholder = np.zeros(placeholder_shape, dtype=agent_action.dtype)
            else:
                placeholder = np.full(
                    placeholder_shape, np.nan, dtype=agent_action.dtype
                )

            action_dict[agent_id] = np.insert(
                agent_action,
                inactive_array,
                placeholder,
                axis=0,
            )

        return action_dict

    def get_action(
        self,
        obs: MARLObservationType,
        *args: Any,
        **kwargs: Any,
    ) -> ActionReturnType:
        """Return the calculated actions matching target structural expectations.

        Evaluates based on algorithmic branch dependencies, returning either a masked
        on-policy vector dictionary or a composite tuple detailing multi-agent tracking
        dependencies for off-policy calculations.

        :param obs: Observation from the environment
        :type obs: MARLObservationType
        :return: Action structure matching policy configurations
        :rtype: ActionReturnType
        """
        if self._is_off_policy():
            return self._get_action_off_policy(obs, *args, **kwargs)

        # Retained on-policy execution path (IPPO)
        inactive_agents, obs = self.extract_inactive_agents(obs)
        action_return = self.wrapped_get_action(obs, *args, **kwargs)

        action_dict = (
            action_return[0] if isinstance(action_return, tuple) else action_return
        )
        action_dict = dict(action_dict)
        action_dict = self._insert_placeholder_actions(action_dict, inactive_agents)

        if isinstance(action_return, tuple):
            action_return = (action_dict, *action_return[1:])
        else:
            action_return = action_dict

        return action_return

    def learn(
        self,
        experiences: TensorDictBase | ExperiencesType,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Learns from the collected experiences.

        For MADDPG/MATD3, *experiences* must be a replay-buffer
        :class:`tensordict.TensorDict` that includes an ``active_mask`` field.
        Actor gradients are masked inside the native loss functions when this wrapper is applied.

        For IPPO, *experiences* matches structural rollout sequences across independent networks.

        :param experiences: Experiences from the environment
        :type experiences: TensorDictBase | ExperiencesType
        :return: Learning optimization details
        :rtype: Any
        """
        if self._is_off_policy():
            if not isinstance(experiences, TensorDictBase):
                msg = "MADDPG/MATD3 require TensorDict experiences from the replay buffer."
                raise ValueError(msg)
            if "active_mask" not in experiences.keys():
                msg = "Async off-policy samples must include 'active_mask' generated during environment tracking steps."
                raise ValueError(msg)
            return self.wrapped_learn(experiences, *args, **kwargs)

        # Retained on-policy IPPO calculation updates
        states, actions, log_probs, rewards, dones, values, next_state, next_done = map(
            self.stack_experiences,
            experiences,
        )

        for agent_id in self.agent.agent_ids:
            agent_next_state: np.ndarray | None = next_state.get(agent_id, None)

            if agent_next_state is None or np.isnan(agent_next_state).all():
                agent_states = states[agent_id]
                agent_dones = dones[agent_id]
                agent_rewards = rewards[agent_id]

                next_state[agent_id] = agent_states[-1]
                next_done[agent_id] = agent_dones[-1]
                states[agent_id] = agent_states[:-1]
                dones[agent_id] = agent_dones[:-1]
                rewards[agent_id] = agent_rewards[:-1]
                actions[agent_id] = actions[agent_id][:-1]
                log_probs[agent_id] = log_probs[agent_id][:-1]
                values[agent_id] = values[agent_id][:-1]

        experiences = (
            states,
            actions,
            log_probs,
            rewards,
            dones,
            values,
            next_state,
            next_done,
        )
        return self.wrapped_learn(experiences, *args, **kwargs)
