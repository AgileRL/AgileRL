# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

from abc import ABC
from collections import OrderedDict
from collections.abc import Callable, Mapping
from functools import partial
from typing import Any, Generic, TypeGuard, TypeVar

import dill
import numpy as np
import numpy.typing as npt
import torch
from gymnasium import spaces
from tensordict import TensorDict, TensorDictBase
from typing_extensions import Self, TypeIs

from agilerl.algorithms import PPO
from agilerl.algorithms.core import (
    EvolvableAlgorithm,
    MultiAgentAlgorithm,
    SingleAgentAlgorithm,
)
from agilerl.algorithms.core.base import get_checkpoint_dict
from agilerl.protocols import EvolvableAttributeDict
from agilerl.typing import (
    ActionReturn,
    ArrayDict,
    ArrayTuple,
    DeviceType,
    ExperiencesType,
    MultiAgentObservationType,
    MultiAgentTensorObsType,
    NumpyObsType,
    ObservationType,
    TorchObsType,
)
from agilerl.utils.algo_utils import obs_to_tensor, stack_experiences
from agilerl.wrappers.utils import RunningMeanStd

AgentT = TypeVar("AgentT", bound=SingleAgentAlgorithm | MultiAgentAlgorithm)


class AgentWrapper(ABC, Generic[AgentT]):
    """Base class for all agent wrappers. Agent wrappers are used to apply an
    additional functionality to the ``get_action()`` and ``learn()`` methods of
    an ``EvolvableAlgorithm`` instance.

    :param agent: Agent to be wrapped
    :type agent: AgentT
    """

    wrapped_get_action: Callable
    wrapped_learn: Callable

    def __init__(self, agent: AgentT) -> None:
        self.agent = agent
        self.observation_space = agent.observation_space
        self.action_space = agent.action_space
        self.multi_agent = isinstance(agent, MultiAgentAlgorithm)

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

    def evolvable_attributes(
        self,
        networks_only: bool = False,
    ) -> EvolvableAttributeDict:
        """Delegate attribute inspection to the wrapped agent."""
        return self.agent.evolvable_attributes(networks_only)

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
        self.agent: AgentT = state["agent"]
        self.observation_space = state["observation_space"]
        self.action_space = state["action_space"]
        self.multi_agent = state["multi_agent"]

        self.wrapped_get_action = self.agent.get_action
        self.wrapped_learn = self.agent.learn

        self.agent.get_action = partial(self.get_action)
        self.agent.learn = partial(self.learn)

    def __getattr__(self, name: str) -> Any:  # noqa: ANN401 -- proxies arbitrary attributes from the wrapped agent
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

    def __setattr__(self, name: str, value: object) -> None:
        if name == "agent" or not hasattr(self, "agent"):
            super().__setattr__(name, value)
        if hasattr(self.agent, name):
            object.__setattr__(self.agent, name, value)
        else:
            super().__setattr__(name, value)

    def clone(self, index: int | None = None, wrap: bool = True) -> Self:
        """Clone the wrapper with the underlying agent.

        :param index: Index of the agent in a population, defaults to None
        :type index: int | None, optional
        :param wrap: If True, wrap the models in the clone with the accelerator, defaults to False
        :type wrap: bool, optional

        :return: Cloned agent wrapper
        :rtype: Self
        """
        agent_clone = self.agent.clone(index, wrap)

        input_args = EvolvableAlgorithm.inspect_attributes(
            self,
            input_args_only=True,
        )
        input_args.pop("agent", None)

        clone = self.__class__(agent_clone, **input_args)

        # `copy_attributes` populates `clone` in place and returns it.
        EvolvableAlgorithm.copy_attributes(
            self,
            clone,
        )
        return clone

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
        checkpoint["wrapper_attrs"] = EvolvableAlgorithm.inspect_attributes(
            self,
        )

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
        obs: ObservationType | MultiAgentObservationType,
        *args: Any,
        **kwargs: Any,
    ) -> ActionReturn:
        """Return the action from the agent.

        :param obs: Observation from the environment
        :type obs: ObservationType | MultiAgentObservationType
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
    ) -> Any:  # noqa: ANN401 -- learn return varies across wrapped algorithms (loss dict, etc.)
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


def _space_shape(space: spaces.Space) -> tuple[int, ...]:
    """Return the shape of a space, treating a shapeless space as a scalar.

    :param space: Space to inspect
    :type space: spaces.Space

    :return: Shape of the space
    :rtype: tuple[int, ...]
    """
    return space.shape if space.shape is not None else ()


def _is_tensor_tuple(
    obs: TorchObsType | MultiAgentTensorObsType,
) -> TypeIs[tuple[torch.Tensor, ...]]:
    """Narrow a tensorised observation to a tuple of tensors."""
    return isinstance(obs, tuple)


def _narrow_obs_entry(value: object) -> TorchObsType:
    """Narrow a raw ``TensorDict`` entry to a tensorised observation."""
    assert isinstance(value, (torch.Tensor, TensorDict)), (
        f"Expected a tensor observation entry, got {type(value).__name__}."
    )
    return value


def _is_marl_obs(
    obs: TorchObsType | MultiAgentTensorObsType,
) -> TypeIs[MultiAgentTensorObsType]:
    """Narrow a tensorised observation to the multi-agent per-agent mapping."""
    return isinstance(obs, dict)


def _is_tensor_mapping(
    obs: TorchObsType | MultiAgentTensorObsType,
) -> TypeIs[dict[str, torch.Tensor]]:
    """Narrow a tensorised observation to a per-key tensor mapping."""
    return isinstance(obs, Mapping)


class RSNorm(AgentWrapper[AgentT]):
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
    :type agent: SingleAgentAlgorithm, MultiAgentAlgorithm
    :param epsilon: Small value to avoid division by zero, defaults to 1e-4
    :type epsilon: float, optional
    :param norm_obs_keys: List of observation keys to normalize, defaults to None
    :type norm_obs_keys: List | None
    """

    obs_rms: RunningStatsType | dict[str, RunningStatsType]

    def __init__(
        self,
        agent: AgentT,
        epsilon: float = 1e-4,
        norm_obs_keys: list[str] | None = None,
    ) -> None:
        super().__init__(agent)

        self.norm_obs_keys = norm_obs_keys
        # The single- and multi-agent statistics containers are structurally
        # indistinguishable to the type checker (both are dict-like); ``multi_agent``
        # is the runtime discriminator, so keep a typed handle to whichever one is
        # populated and expose the union through ``obs_rms``.
        self._single_obs_rms: RunningStatsType | None = None
        self._multi_obs_rms: dict[str, RunningStatsType] | None = None
        # ``multi_agent`` is exactly ``isinstance(agent, MultiAgentAlgorithm)``
        # (see the base constructor); narrowing the agent yields its precisely-typed
        # observation space(s) rather than the delegated, widened attribute.
        if isinstance(self.agent, MultiAgentAlgorithm):
            # Multi-agent algorithms expose `observation_space` as a per-agent mapping
            # (either a `spaces.Dict` or an `OrderedDict` of the unique agent spaces).
            multi_stats: dict[str, RunningStatsType] = OrderedDict(
                (
                    agent_id,
                    RSNorm.build_rms(obs_space, epsilon, norm_obs_keys, self.device),
                )
                for agent_id, obs_space in self.agent.observation_space.items()
            )
            self._multi_obs_rms = multi_stats
            self.obs_rms = multi_stats
        else:
            assert isinstance(self.agent, SingleAgentAlgorithm)
            single_stats = RSNorm.build_rms(
                self.agent.observation_space,
                epsilon,
                norm_obs_keys,
                self.device,
            )
            self._single_obs_rms = single_stats
            self.obs_rms = single_stats

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
                key: RunningMeanStd(epsilon, shape=_space_shape(value), device=device)
                for key, value in spaces_map.items()
            }

        if isinstance(observation_space, spaces.Tuple):
            return tuple(
                RunningMeanStd(epsilon, shape=_space_shape(value), device=device)
                for value in observation_space.spaces
            )

        return RunningMeanStd(
            epsilon, shape=_space_shape(observation_space), device=device
        )

    def _leaf_obs_rms(self) -> RunningStatsType:
        """Single-agent running statistics."""
        assert self._single_obs_rms is not None, (
            "single-agent statistics are only available on a single-agent wrapper"
        )
        return self._single_obs_rms

    def _agent_obs_rms(self) -> dict[str, RunningStatsType]:
        """Per-agent running statistics used in multi-agent mode (see"""
        assert self._multi_obs_rms is not None, (
            "per-agent statistics are only available on a multi-agent wrapper"
        )
        return self._multi_obs_rms

    def _normalize_observation(
        self,
        observation: TorchObsType | MultiAgentTensorObsType,
        *,
        obs_rms: RunningStatsType | None = None,
    ) -> TorchObsType:
        """Normalize the observation using the RunningMeanStd object(s).

        :param observation: Tensorised observation from the environment
        :type observation: TorchObsType
        :param obs_rms: Optional running-statistics object(s) to use instead of
            ``self.obs_rms`` (required for per-agent stats in multi-agent mode).

        :return: Normalized observation
        :rtype: TorchObsType
        """
        obs_rms = self._leaf_obs_rms() if obs_rms is None else obs_rms

        # The statistics mirror the (tensorised) observation's structure, so narrowing
        # the stats tells us the observation's structure too.
        if isinstance(obs_rms, RunningMeanStd):
            assert isinstance(observation, torch.Tensor)
            return (observation - obs_rms.mean) / (obs_rms.var + obs_rms.epsilon).sqrt()

        if isinstance(obs_rms, tuple):
            assert _is_tensor_tuple(observation)
            return tuple(
                (observation[i] - rms.mean) / (rms.var + rms.epsilon).sqrt()
                for i, rms in enumerate(obs_rms)
            )

        assert _is_tensor_mapping(observation)
        return {
            key: (observation[key] - rms.mean) / (rms.var + rms.epsilon).sqrt()
            for key, rms in obs_rms.items()
        }

    def normalize_observation(
        self, observation: TorchObsType | MultiAgentTensorObsType
    ) -> TorchObsType | MultiAgentTensorObsType:
        """Normalize the observation using the RunningMeanStd object(s).

        :param observation: Tensorised observation from the environment
        :type observation: TorchObsType | MultiAgentTensorObsType

        :return: Normalized observation
        :rtype: TorchObsType | MultiAgentTensorObsType
        """
        # A single-agent Dict observation is itself a ``dict[str, Tensor]``, so it is
        # indistinguishable by type from the multi-agent mapping; ``multi_agent`` is the
        # runtime discriminator that narrows to the per-agent mapping below.
        if self.multi_agent:
            assert _is_marl_obs(observation), (
                "Multi-agent observations must be a per-agent mapping."
            )
            agent_rms = self._agent_obs_rms()
            return {
                agent_id: self._normalize_observation(obs, obs_rms=agent_rms[agent_id])
                for agent_id, obs in observation.items()
            }

        return self._normalize_observation(observation)

    def _update_statistics(
        self,
        observation: TorchObsType | MultiAgentTensorObsType,
        *,
        obs_rms: RunningStatsType | None = None,
    ) -> None:
        """Update the running statistics using the observation.

        :param observation: Tensorised observation from the environment
        :type observation: TorchObsType
        :param obs_rms: Optional running-statistics object(s) to use instead of
            ``self.obs_rms`` (required for per-agent stats in multi-agent mode).
        """
        obs_rms = self._leaf_obs_rms() if obs_rms is None else obs_rms

        # The statistics mirror the (tensorised) observation's structure, so narrowing
        # the stats tells us the observation's structure too.
        if isinstance(obs_rms, RunningMeanStd):
            assert isinstance(observation, torch.Tensor)
            obs_rms.update(observation)
        elif isinstance(obs_rms, tuple):
            assert _is_tensor_tuple(observation)
            for i, rms in enumerate(obs_rms):
                rms.update(observation[i])
        else:
            assert _is_tensor_mapping(observation)
            for key, rms in obs_rms.items():
                rms.update(observation[key])

    def update_statistics(
        self, observation: TorchObsType | MultiAgentTensorObsType
    ) -> None:
        """Update the running statistics using the observation.

        :param observation: Tensorised observation from the environment
        :type observation: TorchObsType | MultiAgentTensorObsType
        """
        # ``multi_agent`` is the runtime discriminator (see `normalize_observation`).
        if self.multi_agent:
            assert _is_marl_obs(observation), (
                "Multi-agent observations must be a per-agent mapping."
            )
            agent_rms = self._agent_obs_rms()
            for agent_id, obs in observation.items():
                self._update_statistics(obs, obs_rms=agent_rms[agent_id])
        else:
            self._update_statistics(observation)

    def get_action(
        self,
        obs: ObservationType | MultiAgentObservationType,
        *args: Any,
        **kwargs: Any,
    ) -> ActionReturn:
        """Return the action from the agent after normalizing the observation.

        :param obs: Observation from the environment
        :type obs: ObservationType | MultiAgentObservationType

        :return: Action from the agent
        :rtype: Any
        """
        # `obs_to_tensor` is overloaded on single-agent obs; the MARL branch is
        # dispatched by the same runtime code but has no matching overload.
        tensor_obs = obs_to_tensor(obs, self.device)

        # Update running statistics only when in training mode
        if self.training:
            self.update_statistics(tensor_obs)

        norm_obs = self.normalize_observation(tensor_obs)
        return self.wrapped_get_action(norm_obs, *args, **kwargs)

    def learn(
        self,
        experiences: TensorDictBase | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> Any:  # noqa: ANN401 -- learn return varies across wrapped algorithms (loss dict, etc.)
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
            # Slicing a TensorDict yields a TensorDict; its entries are tensors (or
            # nested tensor collections) that keep their structure once normalized.
            valid_data = self.agent.rollout_buffer.buffer[:buffer_size]
            assert isinstance(valid_data, TensorDictBase)
            valid_data["observations"] = self.normalize_observation(
                _narrow_obs_entry(valid_data["observations"]),
            )
            valid_data["next_observations"] = self.normalize_observation(
                _narrow_obs_entry(valid_data["next_observations"]),
            )
            self.agent.rollout_buffer.buffer[:buffer_size] = valid_data

            return self.wrapped_learn(*args, **kwargs)

        # NOTE: All AgileRL off-policy algorithms now expect experiences to be a TensorDict.
        if not isinstance(experiences, TensorDictBase):
            msg = "Experiences must be a TensorDict."
            raise ValueError(msg)

        experiences["obs"] = self.normalize_observation(
            _narrow_obs_entry(experiences["obs"]),
        )
        experiences["next_obs"] = self.normalize_observation(
            _narrow_obs_entry(experiences["next_obs"]),
        )
        return self.wrapped_learn(experiences, *args, **kwargs)


def _is_array_dict(obs: NumpyObsType) -> TypeIs[ArrayDict]:
    """Narrow a numpy observation leaf to a per-key array mapping."""
    return isinstance(obs, dict)


def _is_array_tuple(obs: NumpyObsType) -> TypeIs[ArrayTuple]:
    """Narrow a numpy observation leaf to a tuple of arrays (see :func:`_is_array_dict`)."""
    return isinstance(obs, tuple)


def _is_numpy_obs_mapping(
    obs: MultiAgentObservationType,
) -> TypeGuard[dict[str, NumpyObsType]]:
    """Confirm every per-agent leaf is a numpy observation."""
    return all(isinstance(leaf, (np.ndarray, dict, tuple)) for leaf in obs.values())


def _mask_rows(array: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Select the rows of ``array`` where ``mask`` is true."""
    return array[mask]


class AsyncAgentsWrapper(AgentWrapper[MultiAgentAlgorithm]):
    """Wrapper for multi-agent algorithms that solve environments with asynchronous agents (i.e. environments
    where agents don't return observations with the same frequency).

    .. note::
        This currently supports IPPO, MADDPG, and MATD3.

    :param agent: MultiAgentAlgorithm instance to be wrapped.
    :type agent: MultiAgentAlgorithm
    """

    def __init__(self, agent: MultiAgentAlgorithm) -> None:
        super().__init__(agent)

        assert self.agent.algo in {"IPPO", "MADDPG", "MATD3"}, (
            "AsyncAgentsWrapper is currently only supported for IPPO, MADDPG, and MATD3."
        )

    def extract_inactive_agents(
        self,
        obs: dict[str, NumpyObsType],
    ) -> tuple[dict[str, npt.NDArray], dict[str, NumpyObsType]]:
        """Extract the inactive agents from an observation. Inspects each key in the
        observation dictionary and, if all the values are `np.nan` (as set by
        ``AsyncPettingZooVecEnv``), the agent is considered inactive and removed from
        the observation dictionary.

        :param obs: Observation dictionary
        :type obs: dict[str, NumpyObsType]

        :return: Tuple of inactive agents and filtered observations
        :rtype: tuple[dict[str, npt.NDArray], dict[str, NumpyObsType]]
        """
        inactive_agents: dict[str, npt.NDArray] = {}
        agents_to_remove: list[str] = []

        # Process each agent's observations
        for agent_id, agent_obs in obs.items():
            # Get a representative array leaf based on the container's structure.
            if _is_array_dict(agent_obs):
                sample = next(iter(agent_obs.values()))
            elif _is_array_tuple(agent_obs):
                sample = agent_obs[0]
            else:
                sample = agent_obs

            # Skip non-vectorized environments, assuming env doesn't return
            # observations for inactive agents
            if len(sample.shape) == 1:
                continue

            # Create boolean mask for active agents. Reducing over a single axis of a
            # multi-dimensional sample always yields an array, but numpy's ``.all``
            # overloads widen the result to a scalar-or-array union.
            active_mask = np.asarray(~np.isnan(sample).all(axis=1))

            # If all agents are active, skip
            if active_mask.all():
                continue

            # Get indices of inactive agents
            inactive_agent_indices = np.where(~active_mask)[0]

            # If all agents are inactive, mark for removal
            if not active_mask.any():
                agents_to_remove.append(agent_id)
                continue

            # Apply mask to filter observations
            if _is_array_dict(agent_obs):
                obs[agent_id] = {
                    k: _mask_rows(v, active_mask) for k, v in agent_obs.items()
                }
            elif _is_array_tuple(agent_obs):
                obs[agent_id] = tuple(_mask_rows(v, active_mask) for v in agent_obs)
            else:
                obs[agent_id] = _mask_rows(agent_obs, active_mask)

            inactive_agents[agent_id] = inactive_agent_indices

        # Remove completely inactive agents
        for agent_id in agents_to_remove:
            obs.pop(agent_id)

        return inactive_agents, obs

    def stack_experiences(self, experiences: Any) -> Any:  # noqa: ANN401 -- arbitrarily nested runtime container (per-agent dicts/lists/arrays), preserved as-is
        """Stacks the experiences, preserving the structure of the container.

        :param experiences: Experiences from the environment
        :type experiences: Any

        :return: Stacked experiences, with the same structure as the input
        :rtype: Any
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
        action_dict: dict[str, npt.NDArray],
        inactive_agents: dict[str, npt.NDArray],
    ) -> dict[str, npt.NDArray]:
        """Insert placeholder actions for inactive agents back into action dict."""
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

    def _align_async_off_policy_experiences(
        self,
        experiences: Mapping[str, Mapping[str, npt.NDArray]],
    ) -> TensorDict:
        """Align async off-policy experiences.

        :param experiences: Stacked experiences, keyed by field and then by agent ID
        :type experiences: Mapping[str, Mapping[str, npt.NDArray]]

        :return: Experiences with all fields aligned per agent
        :rtype: ExperiencesType
        """
        obs = experiences["obs"]
        actions = experiences["action"]
        rewards = experiences["reward"]
        next_obs = experiences["next_obs"]
        dones = experiences["done"]

        all_agent_ids = (
            set(obs.keys())
            | set(actions.keys())
            | set(rewards.keys())
            | set(next_obs.keys())
            | set(dones.keys())
        )

        aligned_obs: dict[str, Any] = {}
        aligned_actions: dict[str, Any] = {}
        aligned_rewards: dict[str, Any] = {}
        aligned_next_obs: dict[str, Any] = {}
        aligned_dones: dict[str, Any] = {}

        for agent_id in all_agent_ids:
            agent_obs = obs.get(agent_id)
            agent_actions = actions.get(agent_id)
            agent_rewards = rewards.get(agent_id)
            agent_next_obs = next_obs.get(agent_id)
            agent_dones = dones.get(agent_id)

            if (
                agent_obs is None
                or agent_actions is None
                or agent_rewards is None
                or agent_dones is None
            ):
                continue

            # If next_states is missing or all NaN, infer it from the state sequence.
            missing_next_obs = agent_next_obs is None or (
                isinstance(agent_next_obs, np.ndarray)
                and np.isnan(agent_next_obs).all()
            )

            if missing_next_obs:
                if len(agent_obs) <= 1:
                    continue

                aligned_obs[agent_id] = agent_obs[:-1]
                aligned_actions[agent_id] = agent_actions[:-1]
                aligned_rewards[agent_id] = agent_rewards[:-1]
                aligned_dones[agent_id] = agent_dones[:-1]
                aligned_next_obs[agent_id] = agent_obs[1:]
            else:
                # If lengths differ, trim all fields to the shortest length.
                min_len = min(
                    len(agent_obs),
                    len(agent_actions),
                    len(agent_rewards),
                    len(agent_next_obs),
                    len(agent_dones),
                )
                if min_len == 0:
                    continue

                aligned_obs[agent_id] = agent_obs[:min_len]
                aligned_actions[agent_id] = agent_actions[:min_len]
                aligned_rewards[agent_id] = agent_rewards[:min_len]
                aligned_next_obs[agent_id] = agent_next_obs[:min_len]
                aligned_dones[agent_id] = agent_dones[:min_len]

        # MADDPG/MATD3 ``learn`` consume a TensorDict of nested per-agent fields.
        def _nested_td(fields: dict[str, Any]) -> TensorDict:
            td = TensorDict({}, batch_size=[])
            td.update(fields)
            return td

        return _nested_td(
            {
                "obs": _nested_td(aligned_obs),
                "action": _nested_td(aligned_actions),
                "reward": _nested_td(aligned_rewards),
                "next_obs": _nested_td(aligned_next_obs),
                "done": _nested_td(aligned_dones),
            }
        )

    def get_action(
        self,
        obs: MultiAgentObservationType,
        *args: Any,
        **kwargs: Any,
    ) -> ActionReturn:
        """Return the action from the agent.

        Since the environments may not return observations for all agents at the
        same time, we extract inactive agents from the observation and fill in
        placeholder values for their actions.

        :param obs: Observation from the environment
        :type obs: MultiAgentObservationType

        :return: Action from the agent
        :rtype: Any
        """
        # Async vectorised MARL environments always emit numpy-array observations, which
        # `extract_inactive_agents` requires to NaN-mask inactive agents.
        assert _is_numpy_obs_mapping(obs), (
            "AsyncAgentsWrapper expects numpy-array observations from the environment."
        )
        inactive_agents, numpy_obs = self.extract_inactive_agents(obs)
        action_return = self.wrapped_get_action(numpy_obs, *args, **kwargs)

        # Off-policy MARL: MADDPG / MATD3 return (env_actions, raw_actions)
        if self.agent.algo in {"MADDPG", "MATD3"}:
            if not isinstance(action_return, tuple) or len(action_return) < 2:
                return action_return

            env_action_dict = dict(action_return[0])
            raw_action_dict = dict(action_return[1])

            env_action_dict = self._insert_placeholder_actions(
                env_action_dict,
                inactive_agents,
            )
            raw_action_dict = self._insert_placeholder_actions(
                raw_action_dict,
                inactive_agents,
            )

            return (env_action_dict, raw_action_dict, *action_return[2:])

        # Existing on-policy path (IPPO)
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

    def learn(self, experiences: ExperiencesType, *args: Any, **kwargs: Any) -> Any:  # noqa: ANN401 -- learn return varies across wrapped algorithms (loss dict, etc.)
        """Learns from the collected experiences.

        :param experiences: Experiences from the environment
        :type experiences: ExperiencesType
        :param args: Additional positional arguments
        :type args: Any
        :param kwargs: Additional keyword arguments
        :type kwargs: Any

        :return: Learning information
        :rtype: Any
        """
        # Off-policy branch for MADDPG / MATD3
        if self.agent.algo in {"MADDPG", "MATD3"}:
            stacked = self.stack_experiences(experiences)
            aligned = self._align_async_off_policy_experiences(stacked)
            return self.wrapped_learn(aligned, *args, **kwargs)

        # Existing IPPO branch
        states, actions, log_probs, rewards, dones, values, next_state, next_done = map(
            self.stack_experiences,
            experiences,
        )

        # Handle case where we haven't collected a next state for each sub-agent
        for agent_id in self.agent.agent_ids:
            agent_next_state: npt.NDArray | None = next_state.get(agent_id, None)

            # If we haven't collected a next state for this agent yet, we need to use
            # last collected state as next_state
            if agent_next_state is None or np.isnan(agent_next_state).all():
                agent_states = states[agent_id]
                agent_dones = dones[agent_id]
                agent_rewards = rewards[agent_id]

                # Update to use last collected state as next_state
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
