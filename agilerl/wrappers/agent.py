from abc import ABC, abstractmethod
from collections import OrderedDict
from functools import partial
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from gymnasium import spaces

from agilerl.algorithms.core import MultiAgentRLAlgorithm, RLAlgorithm
from agilerl.typing import DeviceType, ExperiencesType, ObservationType
from agilerl.utils.algo_utils import obs_to_tensor

AgentType = Union[RLAlgorithm, MultiAgentRLAlgorithm]
MARLObservationType = Dict[str, ObservationType]


class AgentWrapper(ABC):
    """Base class for all agent wrappers. Agent wrappers are used to apply an
    additional functionality to an ``EvolvableAlgorithm`` instance.

    :param agent: Agent to be wrapped
    :type agent: EvolvableAlgorithm
    """

    def __init__(self, agent: AgentType) -> None:
        self.agent = agent
        self.observation_space = agent.observation_space
        self.action_space = agent.action_space
        self.multi_agent = isinstance(agent, MultiAgentRLAlgorithm)

        # Wrap the agent's methods
        self.agent_get_action = agent.get_action
        self.agent_learn = agent.learn

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
        return getattr(self.agent, name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "agent":
            super().__setattr__(name, value)
        else:
            setattr(self.agent, name, value)

    @abstractmethod
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
        raise NotImplementedError

    @abstractmethod
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
        raise NotImplementedError


class RunningMeanStd:
    """Tracks mean, variance, and count of values using torch tensors.

    :param epsilon: Small value to avoid division by zero, defaults to 1e-4
    :type epsilon: float, optional
    :param shape: Shape of the tensor, defaults to ()
    :type shape: tuple[int, ...], optional
    :param device: Device to store the tensors, defaults to "cpu"
    :type device: DeviceType, optional
    :param dtype: Data type of the tensor, defaults to torch.float32
    :type dtype: torch.dtype, optional
    """

    def __init__(
        self,
        epsilon: float = 1e-4,
        shape: tuple[int, ...] = (),
        device: DeviceType = "cpu",
        dtype=torch.float32,
    ) -> None:

        self.epsilon = epsilon
        self.device = device
        self.mean = torch.zeros(shape, dtype=dtype, device=device)
        self.var = torch.ones(shape, dtype=dtype, device=device)
        self.count = torch.tensor(epsilon, dtype=dtype, device=device)

    def update(self, x: torch.Tensor) -> None:
        """Updates mean, variance, and count using a batch of samples."""
        batch_mean = torch.mean(x, dim=0)
        batch_var = torch.var(x, dim=0, unbiased=False)  # Matches NumPy's default
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(
        self, batch_mean: torch.Tensor, batch_var: torch.Tensor, batch_count: int
    ) -> None:
        """Updates mean and variance using batch moments.

        :param batch_mean: Mean of the batch
        :type batch_mean: torch.Tensor
        :param batch_var: Variance of the batch
        :type batch_var: torch.Tensor
        :param batch_count: Number of samples in the batch
        :type batch_count: int
        """
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        self.mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + (delta**2) * (self.count * batch_count / tot_count)
        self.var = M2 / tot_count
        self.count = tot_count


RunningStatsType = Union[
    RunningMeanStd, Dict[str, RunningMeanStd], Tuple[RunningMeanStd, ...]
]


class RSNorm(AgentWrapper):
    """Wrapper to normalize observations such that each coordinate is centered with unit variance.
    Handles both single and multi-agent settings, as well as Dict and Tuple observation spaces.

    The normalization statistics are only updated when the agent is in training mode. This can be
    disabled during inference through ``agent.set_training_mode(False)``.

    .. note::
        For a population of agents, each individual has its own normalization statistics. Ideally
        this should be shared across all agents, but this is not implemented yet.

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
            for key, rms in self.obs_rms.items():
                observation[key] = (observation[key] - rms.mean) / (
                    rms.var + rms.epsilon
                ).sqrt()
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
        return self.agent_get_action(obs, *args, **kwargs)

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
        # NOTE: Hard assumption that the first element is the current observation
        # and the fourth element is the next observation
        experiences = (
            self.normalize_observation(experiences[0]),  # State
            experiences[1],
            experiences[2],
            self.normalize_observation(experiences[3]),  # Next state
            *experiences[4:],
        )
        return self.agent_learn(experiences, *args, **kwargs)
