from abc import ABC
from collections import OrderedDict
from functools import partial
from typing import Any, Callable, Dict, Generic, List, Optional, Tuple, TypeVar, Union

import dill
import numpy as np
import torch
from gymnasium import spaces
from tensordict import is_tensor_collection

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

AgentType = TypeVar("AgentType", bound=Union[RLAlgorithm, MultiAgentRLAlgorithm])
MARLObservationType = Dict[str, ObservationType]
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
        """Returns the training status of the agent.

        :return: Training status of the agent
        :rtype: bool
        """
        return self.agent.training

    @property
    def device(self) -> DeviceType:
        """Returns the device of the agent.

        :return: Device of the agent
        :rtype: DeviceType
        """
        return self.agent.device

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.agent})"

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

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "agent" or not hasattr(self, "agent"):
            super().__setattr__(name, value)
        elif hasattr(self.agent, name):
            object.__setattr__(self.agent, name, value)
        else:
            super().__setattr__(name, value)

    def clone(self, index: Optional[int] = None, wrap: bool = True) -> SelfAgentWrapper:
        """Clones the wrapper with the underlying agent.

        :param index: Index of the agent in a population, defaults to None
        :type index: Optional[int], optional
        :param wrap: If True, wrap the models in the clone with the accelerator, defaults to False
        :type wrap: bool, optional

        :return: Cloned agent wrapper
        :rtype: SelfAgentWrapper
        """
        agent_clone = self.agent.clone(index, wrap)

        input_args = EvolvableAlgorithm.inspect_attributes(self, input_args_only=True)
        input_args.pop("agent", None)

        clone = self.__class__(agent_clone, **input_args)
        clone = EvolvableAlgorithm.copy_attributes(self, clone)
        return clone

    def save_checkpoint(self, path: str) -> None:
        """Saves a checkpoint of agent properties and network weights to path.

        :param path: Location to save checkpoint at
        :type path: string
        """
        checkpoint = get_checkpoint_dict(self.agent)

        del checkpoint["learn"]
        del checkpoint["get_action"]

        # Add wrapper attributes to checkpoint
        checkpoint["wrapper_cls"] = self.__class__
        checkpoint["wrapper_init_dict"] = EvolvableAlgorithm.inspect_attributes(
            self, input_args_only=True
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
        """Loads a checkpoint of agent properties and network weights from path.

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
        obs: Union[ObservationType, MARLObservationType],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Returns the action from the agent.

        :param obs: Observation from the environment
        :type obs: Union[ObservationType, MARLObservationType]
        :param args: Additional positional arguments
        :type args: Any
        :param kwargs: Additional keyword arguments
        :type kwargs: Any

        :return: Action from the agent
        :rtype: Any
        """
        return self.wrapped_get_action(obs, *args, **kwargs)

    def learn(self, experiences: ExperiencesType, *args: Any, **kwargs: Any) -> Any:
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
        return self.wrapped_learn(experiences, *args, **kwargs)


RunningStatsType = Union[
    RunningMeanStd, Dict[str, RunningMeanStd], Tuple[RunningMeanStd, ...]
]


class RSNorm(AgentWrapper[AgentType]):
    """Wrapper to normalize observations such that each coordinate is centered with unit variance.
    Handles both single and multi-agent settings, as well as Dict and Tuple observation spaces.

    The normalization statistics are only updated when the agent is in training mode. This can be
    disabled during inference through ``agent.set_training_mode(False)``.

    .. warning::
        This wrapper is currently only supported for off-policy algorithms since it relies on
        passed experiences to be formatted as a tuple of PyTorch tensors. Currently
        AgileRL does not use a Buffer class to store experiences for on-policy algorithms, albeit this
        will be released in a soon-to-come update!

    :param agent: Agent to be wrapped
    :type agent: RLAlgorithm, MultiAgentRLAlgorithm
    :param epsilon: Small value to avoid division by zero, defaults to 1e-4
    :type epsilon: float, optional
    :param norm_obs_keys: List of observation keys to normalize, defaults to None
    :type norm_obs_keys: Optional[List]
    """

    obs_rms: Union[RunningStatsType, Dict[str, RunningStatsType]]

    def __init__(
        self,
        agent: AgentType,
        epsilon: float = 1e-4,
        norm_obs_keys: Optional[List[str]] = None,
    ) -> None:
        super().__init__(agent)

        self.norm_obs_keys = norm_obs_keys
        if self.multi_agent:
            self.obs_rms = OrderedDict()
            for agent_id, obs_space in self.observation_space.items():
                self.obs_rms[agent_id] = RSNorm.build_rms(
                    obs_space, epsilon, norm_obs_keys, self.device
                )
        else:
            self.obs_rms = RSNorm.build_rms(
                self.observation_space, epsilon, norm_obs_keys, self.device
            )

    @staticmethod
    def build_rms(
        observation_space: spaces.Space,
        epsilon: float = 1e-4,
        norm_obs_keys: Optional[List[str]] = None,
        device: DeviceType = "cpu",
    ) -> Union[RunningMeanStd, Dict[str, RunningMeanStd], Tuple[RunningMeanStd, ...]]:
        """Builds the RunningMeanStd object(s) based on the observation space.

        :param observation_space: Observation space of the agent
        :type observation_space: spaces.Space
        :return: RunningMeanStd object(s)
        :rtype: Union[RunningMeanStd, Dict[str, RunningMeanStd], Tuple[RunningMeanStd, ...]]
        """
        if isinstance(observation_space, spaces.Dict):
            if norm_obs_keys is not None:
                observation_space = {
                    key: value
                    for key, value in observation_space.spaces.items()
                    if key in norm_obs_keys
                }

            return {
                key: RunningMeanStd(epsilon, shape=value.shape, device=device)
                for key, value in observation_space.spaces.items()
            }

        elif isinstance(observation_space, spaces.Tuple):
            return tuple(
                RunningMeanStd(epsilon, shape=value.shape, device=device)
                for value in observation_space.spaces
            )

        else:
            return RunningMeanStd(epsilon, shape=observation_space.shape, device=device)

    def _normalize_observation(self, observation: ObservationType) -> ObservationType:
        """Normalizes the observation using the RunningMeanStd object(s).

        :param observation: Observation from the environment
        :type observation: ObservationType

        :return: Normalized observation
        :rtype: ObservationType
        """
        if isinstance(self.obs_rms, dict):
            norm_observation = {}
            for key, rms in self.obs_rms.items():
                norm_observation[key] = (observation[key] - rms.mean) / (
                    rms.var + rms.epsilon
                ).sqrt()

            observation = norm_observation
        elif isinstance(self.obs_rms, tuple):
            norm_observation = []
            for i, rms in enumerate(self.obs_rms):
                norm_obs = (observation[i] - rms.mean) / (rms.var + rms.epsilon).sqrt()
                norm_observation.append(norm_obs)

            observation = tuple(norm_observation)
        else:
            observation = (observation - self.obs_rms.mean) / (
                self.obs_rms.var + self.obs_rms.epsilon
            ).sqrt()

        return observation

    def normalize_observation(self, observation: ObservationType) -> ObservationType:
        """Normalizes the observation using the RunningMeanStd object(s).

        :param observation: Observation from the environment
        :type observation: ObservationType

        :return: Normalized observation
        :rtype: ObservationType
        """
        if self.multi_agent:
            for agent_id, obs in observation.items():
                observation[agent_id] = self._normalize_observation(obs)
            return observation

        return self._normalize_observation(observation)

    def _update_statistics(self, observation: ObservationType) -> None:
        """Updates the running statistics using the observation.

        :param observation: Observation from the environment
        :type observation: ObservationType
        """
        if isinstance(self.obs_rms, dict):
            for key, rms in self.obs_rms.items():
                rms.update(observation[key])
        elif isinstance(self.obs_rms, tuple):
            for i, rms in enumerate(self.obs_rms):
                rms.update(observation[i])
        else:
            self.obs_rms.update(observation)

    def update_statistics(self, observation: ObservationType) -> None:
        """Updates the running statistics using the observation.

        :param observation: Observation from the environment
        :type observation: ObservationType
        """
        if self.multi_agent:
            for _, obs in observation.items():
                self._update_statistics(obs)
        else:
            self._update_statistics(observation)

    def get_action(self, obs: ObservationType, *args: Any, **kwargs: Any) -> Any:
        """Returns the action from the agent after normalizing the observation.

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

    def learn(self, experiences: ExperiencesType, *args: Any, **kwargs: Any) -> Any:
        """Learns from the experiences after normalizing the observations.

        :param experiences: Experiences from the environment
        :type experiences: ExperiencesType
        :param args: Additional positional arguments
        :type args: Any
        :param kwargs: Additional keyword arguments
        :type kwargs: Any

        :return: Learning information
        :rtype: Any
        """
        # NOTE: We want to move towards always passing experiences as TensorDict objects
        if is_tensor_collection(experiences):
            experiences["obs"] = self.normalize_observation(experiences["obs"])
            experiences["next_obs"] = self.normalize_observation(
                experiences["next_obs"]
            )
        else:
            experiences = (
                self.normalize_observation(experiences[0]),  # State
                experiences[1],
                experiences[2],
                self.normalize_observation(experiences[3]),  # Next state
                *experiences[4:],
            )

        return self.wrapped_learn(experiences, *args, **kwargs)


class AsyncAgentsWrapper(AgentWrapper[MultiAgentRLAlgorithm]):
    """Wrapper for multi-agent algorithms that solve environments with asynchronous agents (i.e. environments
    where agents don't return observations with the same frequency).

    .. warning::
        This is currently only supported for on-policy multi-agent algorithms such as IPPO.

    :param agent: MultiAgentRLAlgorithm instance to be wrapped.
    :type agent: MultiAgentRLAlgorithm
    """

    def __init__(self, agent: MultiAgentRLAlgorithm) -> None:
        super().__init__(agent)

        assert (
            self.agent.algo == "IPPO"
        ), "AsyncAgentsWrapper is currently only supported for IPPO."

    def extract_inactive_agents(
        self, obs: Dict[str, ObservationType]
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, ObservationType]]:
        """Extract the inactive agents from an observation. Inspects each key in the
        observation dictionary and, if all the values are `np.nan` (as set by
        ``AsyncPettingZooVecEnv``), the agent is considered inactive and removed from
        the observation dictionary.

        :param obs: Observation dictionary
        :type obs: Dict[str, ObservationType]

        :return: Tuple of inactive agents and filtered observations
        :rtype: Tuple[Dict[str, np.ndarray], Dict[str, ObservationType]]
        """
        inactive_agents = {}
        agents_to_remove = []

        # Process each agent's observations
        for agent_id, agent_obs in obs.items():
            # Get the sample observation based on type
            if isinstance(agent_obs, dict):
                sample = next(iter(agent_obs.values()))
            elif isinstance(agent_obs, tuple):
                sample = agent_obs[0]
            else:
                sample = agent_obs

            # Skip non-vectorized environments, assuming env doesn't return
            # observations for inactive agents
            if len(sample.shape) == 1:
                continue

            # Create boolean mask for active agents
            active_mask: np.ndarray = ~np.isnan(sample).all(axis=1)

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
            if isinstance(agent_obs, dict):
                obs[agent_id] = {k: v[active_mask] for k, v in agent_obs.items()}
            elif isinstance(agent_obs, tuple):
                obs[agent_id] = tuple(v[active_mask] for v in agent_obs)
            else:
                obs[agent_id] = agent_obs[active_mask]

            inactive_agents[agent_id] = inactive_agent_indices

        # Remove completely inactive agents
        for agent_id in agents_to_remove:
            obs.pop(agent_id)

        return inactive_agents, obs

    def stack_experiences(self, experience: ExperiencesType) -> ExperiencesType:
        """Stacks the experiences.

        :param experiences: Experiences from the environment
        :type experiences: ExperiencesType
        """
        stacked_experience = {}
        for agent_id, inp in experience.items():
            if isinstance(inp, list):
                stacked_exp = (
                    stack_experiences(inp, to_torch=False)[0] if len(inp) > 0 else None
                )
            else:
                stacked_exp = inp

            stacked_experience[agent_id] = stacked_exp

        return stacked_experience

    def get_action(
        self, obs: ObservationType, *args: Any, **kwargs: Any
    ) -> ActionReturnType:
        """Returns the action from the agent. Since the environments may not return observations for all agents
        at the same time, we need to extract the inactive agents from the observation and fill in placeholder
        values for their actions.

        :param obs: Observation from the environment
        :type obs: ObservationType

        :return: Action from the agent
        :rtype: Any
        """
        inactive_agents, obs = self.extract_inactive_agents(obs)
        action_return: ActionReturnType = self.wrapped_get_action(obs, *args, **kwargs)

        # Need to fill in placeholder np.nan for inactive agents
        action_dict = (
            action_return[0] if isinstance(action_return, tuple) else action_return
        )
        for agent_id, inactive_array in inactive_agents.items():
            placeholder = (
                int(np.nan)
                if np.issubdtype(action_dict[agent_id].dtype, np.integer)
                else np.nan
            )

            # Insert placeholder values for inactive agents
            action_dict[agent_id] = np.insert(
                action_dict[agent_id], inactive_array, placeholder, axis=0
            )

        if isinstance(action_return, tuple):
            action_return = (action_dict,) + action_return[1:]
        else:
            action_return = action_dict

        return action_return

    def learn(self, experiences: ExperiencesType, *args: Any, **kwargs: Any) -> Any:
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
        (states, actions, log_probs, rewards, dones, values, next_state, next_done) = (
            map(self.stack_experiences, experiences)
        )

        # Handle case where we haven't collected a next state for each sub-agent
        for agent_id in self.agent.agent_ids:
            agent_next_state: Optional[np.ndarray] = next_state.get(agent_id, None)

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
