import copy
import warnings
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from gymnasium import spaces

from agilerl.algorithms.core import MultiAgentRLAlgorithm, OptimizerWrapper
from agilerl.algorithms.core.registry import HyperparameterConfig, NetworkGroup
from agilerl.modules.base import EvolvableModule, ModuleDict
from agilerl.modules.configs import MlpNetConfig
from agilerl.networks.actors import DeterministicActor
from agilerl.networks.q_networks import ContinuousQNetwork
from agilerl.typing import (
    ArrayDict,
    InfosDict,
    MultiAgentModule,
    ObservationType,
    PzEnvType,
    StandardTensorDict,
)
from agilerl.utils.algo_utils import (
    concatenate_spaces,
    format_shared_critic_encoder,
    get_deepest_head_config,
    key_in_nested_dict,
    make_safe_deepcopies,
    obs_channels_to_first,
)


class MATD3(MultiAgentRLAlgorithm):
    """Multi-Agent Twin Delayed Deep Deterministic Policy Gradient (MATD3) algorithm.

    Paper: https://arxiv.org/abs/1910.01465

    :param observation_spaces: Observation space for each agent
    :type observation_spaces: Union[List[spaces.Space], spaces.Dict]
    :param action_spaces: Action space for each agent
    :type action_spaces: Union[List[spaces.Space], spaces.Dict]
    :param agent_ids: Agent ID for each agent
    :type agent_ids: Optional[List[str]], optional
    :param O_U_noise: Use Ornstein Uhlenbeck action noise for exploration. If False, uses Gaussian noise. Defaults to True
    :type O_U_noise: bool, optional
    :param expl_noise: Scale for Ornstein Uhlenbeck action noise, or standard deviation for Gaussian exploration noise
    :type expl_noise: float, optional
    :param vect_noise_dim: Vectorization dimension of environment for action noise, defaults to 1
    :type vect_noise_dim: int, optional
    :param mean_noise: Mean of exploration noise, defaults to 0.0
    :type mean_noise: float, optional
    :param theta: Rate of mean reversion in Ornstein Uhlenbeck action noise, defaults to 0.15
    :type theta: float, optional
    :param dt: Timestep for Ornstein Uhlenbeck action noise update, defaults to 1e-2
    :type dt: float, optional
    :param index: Index to keep track of object instance during tournament selection and mutation, defaults to 0
    :type index: int, optional
    :param hp_config: RL hyperparameter mutation configuration, defaults to None, whereby algorithm mutations are disabled.
    :type hp_config: HyperparameterConfig, optional
    :param policy_freq: Policy update frequency, defaults to 2
    :type policy_freq: int, optional
    :param net_config: Network configuration, defaults to None
    :type net_config: Optional[Dict[str, Any]], optional
    :param batch_size: Size of batched sample from replay buffer for learning, defaults to 64
    :type batch_size: int, optional
    :param lr_actor: Learning rate for actor optimizer, defaults to 0.001
    :type lr_actor: float, optional
    :param lr_critic: Learning rate for critic optimizer, defaults to 0.01
    :type lr_critic: float, optional
    :param learn_step: Learning frequency, defaults to 5
    :type learn_step: int, optional
    :param gamma: Discount factor, defaults to 0.95
    :type gamma: float, optional
    :param tau: For soft update of target network parameters, defaults to 0.01
    :type tau: float, optional
    :param normalize_images: Normalize image observations, defaults to True
    :type normalize_images: bool, optional
    :param mut: Most recent mutation to agent, defaults to None
    :type mut: Optional[str], optional
    :param actor_networks: List of custom actor networks, defaults to None
    :type actor_networks: Optional[ModuleDict], optional
    :param critic_networks: List containing two lists of custom critic networks, defaults to None
    :type critic_networks: Optional[List[ModuleDict]], optional
    :param device: Device for accelerated computing, 'cpu' or 'cuda', defaults to 'cpu'
    :type device: str, optional
    :param accelerator: Accelerator for distributed computing, defaults to None
    :type accelerator: accelerate.Accelerator(), optional
    :param torch_compiler: The torch compile mode 'default', 'reduce-overhead' or 'max-autotune', defaults to None
    :type torch_compiler: Optional[str], optional
    :param wrap: Wrap models for distributed training upon creation, defaults to True
    :type wrap: bool, optional
    """

    possible_action_spaces: Dict[str, Union[spaces.Box, spaces.Discrete]]

    actors: MultiAgentModule[DeterministicActor]
    actor_targets: MultiAgentModule[DeterministicActor]
    critics_1: MultiAgentModule[ContinuousQNetwork]
    critic_targets_1: MultiAgentModule[ContinuousQNetwork]
    critics_2: MultiAgentModule[ContinuousQNetwork]
    critic_targets_2: MultiAgentModule[ContinuousQNetwork]

    def __init__(
        self,
        observation_spaces: Union[List[spaces.Space], spaces.Dict],
        action_spaces: Union[List[spaces.Space], spaces.Dict],
        agent_ids: Optional[List[str]] = None,
        O_U_noise: bool = True,
        expl_noise: float = 0.1,
        vect_noise_dim: int = 1,
        mean_noise: float = 0.0,
        theta: float = 0.15,
        dt: float = 1e-2,
        index: int = 0,
        hp_config: Optional[HyperparameterConfig] = None,
        policy_freq: int = 2,
        net_config: Optional[Dict[str, Any]] = None,
        batch_size: int = 64,
        lr_actor: float = 0.001,
        lr_critic: float = 0.01,
        learn_step: int = 5,
        gamma: float = 0.95,
        tau: float = 0.01,
        normalize_images: bool = True,
        mut: Optional[str] = None,
        actor_networks: Optional[ModuleDict] = None,
        critic_networks: Optional[List[ModuleDict]] = None,
        device: str = "cpu",
        accelerator: Optional[Any] = None,
        torch_compiler: Optional[str] = None,
        wrap: bool = True,
    ):
        super().__init__(
            observation_spaces,
            action_spaces,
            index=index,
            agent_ids=agent_ids,
            hp_config=hp_config,
            device=device,
            accelerator=accelerator,
            normalize_images=normalize_images,
            torch_compiler=torch_compiler,
            name="MATD3",
        )

        assert learn_step >= 1, "Learn step must be greater than or equal to one."
        assert isinstance(learn_step, int), "Learn step rate must be an integer."
        assert isinstance(batch_size, int), "Batch size must be an integer."
        assert batch_size >= 1, "Batch size must be greater than or equal to one."
        assert isinstance(lr_actor, float), "Actor learning rate must be a float."
        assert lr_actor > 0, "Actor learning rate must be greater than zero."
        assert isinstance(lr_critic, float), "Critic learning rate must be a float."
        assert lr_critic > 0, "Critic learning rate must be greater than zero."
        assert isinstance(learn_step, int), "Learn step rate must be an integer."
        assert learn_step >= 1, "Learn step must be greater than or equal to one."
        assert isinstance(gamma, float), "Gamma must be a float."
        assert isinstance(tau, float), "Tau must be a float."
        assert tau > 0, "Tau must be greater than zero."
        assert isinstance(policy_freq, int), "Policy frequency must be an integer."
        assert policy_freq > 0, "Policy frequency must be greater than zero."
        if (actor_networks is not None) != (critic_networks is not None):
            warnings.warn(
                "Actor and critic network must both be supplied to use custom networks. Defaulting to net config."
            )
        assert isinstance(
            wrap, bool
        ), "Wrap models flag must be boolean value True or False."

        self.batch_size = batch_size
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.learn_step = learn_step
        self.gamma = gamma
        self.net_config = net_config
        self.tau = tau
        self.mut = mut
        self.policy_freq = policy_freq
        self.learn_counter = {agent: 0 for agent in self.agent_ids}
        self.O_U_noise = O_U_noise
        self.vect_noise_dim = vect_noise_dim
        self.theta = theta
        self.dt = dt
        self.sqdt = dt ** (0.5)

        # Initialise noise for exploration
        self.sample_gaussian = {
            agent_id: torch.zeros(*(vect_noise_dim, action_dim), device=self.device)
            for agent_id, action_dim in self.action_dims.items()
        }
        self.expl_noise = (
            expl_noise
            if isinstance(expl_noise, dict)
            else {
                agent_id: expl_noise
                * torch.ones(*(vect_noise_dim, action_dim), device=self.device)
                for agent_id, action_dim in self.action_dims.items()
            }
        )
        self.mean_noise = (
            mean_noise
            if isinstance(mean_noise, dict)
            else {
                agent_id: mean_noise
                * torch.ones(*(vect_noise_dim, action_dim), device=self.device)
                for agent_id, action_dim in self.action_dims.items()
            }
        )
        self.current_noise = {
            agent_id: torch.zeros(*(vect_noise_dim, action_dim), device=self.device)
            for agent_id, action_dim in self.action_dims.items()
        }

        if actor_networks is not None and critic_networks is not None:
            assert isinstance(
                critic_networks, list
            ), "critic_networks must be a list containing the two critics in MATD3."

            if isinstance(actor_networks, list):
                assert len(actor_networks) == len(
                    self.agent_ids
                ), "actor_networks must be a list of the same length as the number of agents"
                actor_networks = ModuleDict(
                    {
                        self.agent_ids[i]: actor_networks[i]
                        for i in range(len(self.agent_ids))
                    }
                )
            if isinstance(critic_networks[0], list):
                assert len(critic_networks[0]) == len(
                    self.agent_ids
                ), "critic_networks at index 0 must be a list of the same length as the number of agents"
                assert len(critic_networks[1]) == len(
                    self.agent_ids
                ), "critic_networks at index 1 must be a list of the same length as the number of agents"

                critic_networks[0] = ModuleDict(
                    {
                        self.agent_ids[i]: critic_networks[0][i]
                        for i in range(len(self.agent_ids))
                    }
                )
                critic_networks[1] = ModuleDict(
                    {
                        self.agent_ids[i]: critic_networks[1][i]
                        for i in range(len(self.agent_ids))
                    }
                )

            actors_list = list(actor_networks.values())
            critics_list = list(critic_networks[0].values()) + list(
                critic_networks[1].values()
            )
            assert all(
                isinstance(net, actors_list[0].__class__) for net in actors_list
            ), "'actor_networks' must all be the same type"
            assert all(
                isinstance(net, critics_list[0].__class__) for net in critics_list
            ), "'critic_networks' must all be the same type"

            if not all(
                isinstance(net, EvolvableModule) for net in actor_networks.values()
            ):
                raise TypeError(
                    "All actor networks must be instances of EvolvableModule"
                )
            if not all(
                isinstance(net, EvolvableModule) for net in critic_networks[0].values()
            ):
                raise TypeError(
                    "All critic networks must be instances of EvolvableModule"
                )
            if not all(
                isinstance(net, EvolvableModule) for net in critic_networks[1].values()
            ):
                raise TypeError(
                    "All critic networks must be instances of EvolvableModule"
                )
            self.actors, self.critics_1, self.critics_2 = make_safe_deepcopies(
                actor_networks, critic_networks[0], critic_networks[1]
            )
            self.actor_targets, self.critic_targets_1, self.critic_targets_2 = (
                make_safe_deepcopies(
                    actor_networks, critic_networks[0], critic_networks[1]
                )
            )
        else:
            agent_configs, encoder_configs = self.build_net_config(
                net_config, return_encoders=True
            )

            # Iterate over actor configs and modify accordingly
            for agent_id, space in self.possible_action_spaces.items():
                agent_config = agent_configs[agent_id]
                head_config = agent_config.get("head_config", None)

                # Determine actor output activation from action space
                discrete_actions = isinstance(space, spaces.Discrete)
                if head_config is not None:
                    if discrete_actions:
                        head_config["output_activation"] = "GumbelSoftmax"
                else:
                    output_activation = "GumbelSoftmax" if discrete_actions else None
                    head_config = MlpNetConfig(
                        hidden_size=[64], output_activation=output_activation
                    )
                    if head_config.output_activation is None:
                        head_config.pop("output_activation")

                agent_config["head_config"] = head_config
                agent_configs[agent_id] = agent_config

            # Format critic net config from actor net configs
            latent_dim = max(
                [
                    agent_configs[agent_id].get("latent_dim", 32)
                    for agent_id in self.agent_ids
                ]
            )
            min_latent_dim = min(
                [
                    agent_configs[agent_id].get("min_latent_dim", 8)
                    for agent_id in self.agent_ids
                ]
            )
            max_latent_dim = max(
                [
                    agent_configs[agent_id].get("max_latent_dim", 128)
                    for agent_id in self.agent_ids
                ]
            )
            critic_encoder_config = format_shared_critic_encoder(encoder_configs)
            critic_head_config = get_deepest_head_config(agent_configs, self.agent_ids)
            critic_net_config = {
                "encoder_config": critic_encoder_config,
                "head_config": critic_head_config,
                "latent_dim": latent_dim,
                "min_latent_dim": min_latent_dim,
                "max_latent_dim": max_latent_dim,
            }
            clip_actions = self.torch_compiler is None

            def create_actor(agent_id):
                return DeterministicActor(
                    self.possible_observation_spaces[agent_id],
                    self.possible_action_spaces[agent_id],
                    device=self.device,
                    clip_actions=clip_actions,
                    **copy.deepcopy(agent_configs[agent_id]),
                )

            # Critic uses observations + actions of all agents to predict Q-value
            def create_critic():
                return ContinuousQNetwork(
                    observation_space=self.possible_observation_spaces,
                    action_space=concatenate_spaces(
                        list(self.possible_action_spaces.values())
                    ),
                    device=self.device,
                    **copy.deepcopy(critic_net_config),
                )

            self.actors = ModuleDict(
                {agent_id: create_actor(agent_id) for agent_id in self.agent_ids}
            )
            self.critics_1 = ModuleDict(
                {agent_id: create_critic() for agent_id in self.agent_ids}
            )
            self.critics_2 = ModuleDict(
                {agent_id: create_critic() for agent_id in self.agent_ids}
            )
            self.actor_targets = ModuleDict(
                {agent_id: create_actor(agent_id) for agent_id in self.agent_ids}
            )
            self.critic_targets_1 = ModuleDict(
                {agent_id: create_critic() for agent_id in self.agent_ids}
            )
            self.critic_targets_2 = ModuleDict(
                {agent_id: create_critic() for agent_id in self.agent_ids}
            )

        # Initialise target network parameters
        for agent_id in self.agent_ids:
            self.actor_targets[agent_id].load_state_dict(
                self.actors[agent_id].state_dict()
            )
            self.critic_targets_1[agent_id].load_state_dict(
                self.critics_1[agent_id].state_dict()
            )
            self.critic_targets_2[agent_id].load_state_dict(
                self.critics_2[agent_id].state_dict()
            )

        # Optimizers
        self.actor_optimizers = OptimizerWrapper(
            optim.Adam,
            networks=self.actors,
            lr=lr_actor,
        )

        self.critic_1_optimizers = OptimizerWrapper(
            optim.Adam,
            networks=self.critics_1,
            lr=lr_critic,
        )

        self.critic_2_optimizers = OptimizerWrapper(
            optim.Adam,
            networks=self.critics_2,
            lr=lr_critic,
        )

        if self.accelerator is not None and wrap:
            self.wrap_models()
        elif self.torch_compiler:
            if (
                any(
                    actor.output_activation == "GumbelSoftmax"
                    for actor in self.actors.values()
                )
                and self.torch_compiler != "default"
            ):
                warnings.warn(
                    f"{self.torch_compiler} compile mode is not compatible with GumbelSoftmax activation, changing to 'default' mode."
                )
                self.torch_compiler = "default"

            torch.set_float32_matmul_precision("high")
            self.recompile()

        self.criterion = nn.MSELoss()

        # Register network groups for mutations
        self.register_network_group(
            NetworkGroup(
                eval_network=self.actors,
                shared_networks=self.actor_targets,
                policy=True,
            )
        )
        self.register_network_group(
            NetworkGroup(
                eval_network=self.critics_1,
                shared_networks=self.critic_targets_1,
            )
        )
        self.register_network_group(
            NetworkGroup(
                eval_network=self.critics_2,
                shared_networks=self.critic_targets_2,
            )
        )

    def process_infos(
        self, infos: Optional[InfosDict] = None
    ) -> Tuple[ArrayDict, ArrayDict, ArrayDict]:
        """
        Process the information, extract env_defined_actions, action_masks and agent_masks

        :param infos: Info dict
        :type infos: Dict[str, Dict[...]]
        :return: Action masks, env defined actions, agent masks
        :rtype: Tuple[ArrayDict, ArrayDict, ArrayDict]
        """
        if infos is None:
            infos = {agent: {} for agent in self.agent_ids}

        env_defined_actions, agent_masks = self.extract_agent_masks(infos)
        action_masks = self.extract_action_masks(infos)
        return action_masks, env_defined_actions, agent_masks

    def get_action(
        self, obs: Dict[str, ObservationType], infos: Optional[InfosDict] = None
    ) -> Tuple[ArrayDict, ArrayDict]:
        """Returns the next action to take in the environment.
        Epsilon is the probability of taking a random action, used for exploration.
        For epsilon-greedy behaviour, set epsilon to 0.

        :param obs: Environment observations: {'agent_0': state_dim_0, ..., 'agent_n': state_dim_n}
        :type obs: Dict[str, numpy.Array]
        :param infos: Information dictionary from environment, defaults to None
        :type infos: Dict[str, Dict[...]], optional

        :return: Processed actions for each agent, raw actions for each agent
        :rtype: Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]
        """
        assert not key_in_nested_dict(
            obs, "action_mask"
        ), "AgileRL requires action masks to be defined in the information dictionary."

        action_masks, env_defined_actions, agent_masks = self.process_infos(infos)

        # Preprocess observations
        preprocessed_states = self.preprocess_observation(obs)

        action_dict: Dict[str, np.ndarray] = {}
        for agent_id, obs in preprocessed_states.items():
            actor = self.actors[agent_id]
            actor.eval()
            if self.accelerator is not None:
                with actor.no_sync(), torch.no_grad():
                    actions = actor(obs)
            else:
                with torch.no_grad():
                    actions = actor(obs)

            # Need to rescale actions outside of forward pass if using torch.compile
            if self.torch_compiler is not None and isinstance(
                self.possible_action_spaces[agent_id], spaces.Box
            ):
                actions = DeterministicActor.rescale_action(
                    action=actions,
                    low=actor.action_low,
                    high=actor.action_high,
                    output_activation=actor.output_activation,
                )

            actor.train()
            if self.training:
                if isinstance(self.possible_action_spaces[agent_id], spaces.Discrete):
                    min_action, max_action = 0, 1
                else:
                    min_action = self.possible_action_spaces[agent_id].low
                    max_action = self.possible_action_spaces[agent_id].high

                # Add noise to actions for exploration
                actions = torch.clamp(
                    actions + self.action_noise(agent_id),
                    torch.as_tensor(min_action, device=actions.device),
                    torch.as_tensor(max_action, device=actions.device),
                )

            action_dict[agent_id] = actions.cpu().numpy()

        # Process agents with discrete actions
        processed_action_dict: ArrayDict = OrderedDict()
        for agent_id, action_space in self.possible_action_spaces.items():
            if isinstance(action_space, spaces.Discrete):
                action = action_dict[agent_id]
                mask = (
                    1 - np.array(action_masks[agent_id])
                    if action_masks[agent_id] is not None
                    else None
                )
                action: np.ndarray = np.ma.array(action, mask=mask)
                processed_action_dict[agent_id] = action.argmax(axis=-1)

                if (
                    len(processed_action_dict[agent_id].shape) == 1
                    and env_defined_actions
                ):
                    env_defined_actions = {
                        agent: action.squeeze(1) if len(action.shape) > 1 else action
                        for agent, action in env_defined_actions.items()
                    }
                    agent_masks = {
                        agent: mask.squeeze(1) if len(mask.shape) > 1 else mask
                        for agent, mask in agent_masks.items()
                    }
            else:
                processed_action_dict[agent_id] = action_dict[agent_id]

        # If using env_defined_actions replace actions
        if env_defined_actions is not None:
            for agent in self.agent_ids:
                processed_action_dict[agent][agent_masks[agent]] = env_defined_actions[
                    agent
                ][agent_masks[agent]]

        return processed_action_dict, action_dict

    def action_noise(self, agent_id: str) -> torch.Tensor:
        """Create action noise for exploration, either Ornstein Uhlenbeck or
            from a normal distribution.

        :param agent_id: Agent ID for action dims
        :type agent_id: str
        :return: Action noise
        :rtype: torch.Tensor
        """
        if self.O_U_noise:
            noise = (
                self.current_noise[agent_id]
                + self.theta
                * (self.mean_noise[agent_id] - self.current_noise[agent_id])
                * self.dt
                + self.expl_noise[agent_id]
                * self.sqdt
                * self.sample_gaussian[agent_id].normal_()
            )
            self.current_noise[agent_id] = noise
        else:
            torch.normal(
                self.mean_noise[agent_id],
                self.expl_noise[agent_id],
                out=self.sample_gaussian[agent_id],
            )
            noise = self.sample_gaussian[agent_id]
        return noise

    def reset_action_noise(self, indices: List[int]) -> None:
        """Reset action noise.

        :param indices: List of indices to reset noise for
        :type indices: List[int]
        """
        for agent_id in self.agent_ids:
            for idx in indices:
                self.current_noise[agent_id][idx, :] = 0

    def learn(self, experiences: Tuple[StandardTensorDict, ...]) -> Dict[str, float]:
        """Updates agent network parameters to learn from experiences.

        :param experience: Tuple of dictionaries containing batched states, actions,
            rewards, next_states, dones in that order for each individual agent.
        :type experience: Tuple[Dict[str, torch.Tensor]]

        :return: Losses for each agent
        :rtype: Dict[str, float]
        """
        states, actions, rewards, next_states, dones = experiences

        actions = {
            agent_id: agent_actions.to(self.device)
            for agent_id, agent_actions in actions.items()
        }
        rewards = {
            agent_id: agent_rewards.to(self.device)
            for agent_id, agent_rewards in rewards.items()
        }
        dones = {
            agent_id: agent_dones.to(self.device)
            for agent_id, agent_dones in dones.items()
        }

        # Preprocess observations
        states = self.preprocess_observation(states)
        next_states = self.preprocess_observation(next_states)

        next_actions = []
        with torch.no_grad():
            for agent_id in self.agent_ids:
                next_actions.append(self.actor_targets[agent_id](next_states[agent_id]))

        # Stack states and actions
        stacked_actions = torch.cat(list(actions.values()), dim=1)
        stacked_next_actions = torch.cat(next_actions, dim=1)

        loss_dict = {}
        for agent_id in self.agent_ids:
            loss_dict[f"{agent_id}"] = self.learn_individual(
                agent_id,
                stacked_actions=stacked_actions,
                stacked_next_actions=stacked_next_actions,
                states=states,
                next_states=next_states,
                actions=actions,
                rewards=rewards,
                dones=dones,
            )

        if self.learn_counter[agent_id] % self.policy_freq == 0:
            for agent_id in self.agent_ids:
                self.soft_update(self.actors[agent_id], self.actor_targets[agent_id])
                self.soft_update(
                    self.critics_1[agent_id], self.critic_targets_1[agent_id]
                )
                self.soft_update(
                    self.critics_2[agent_id], self.critic_targets_2[agent_id]
                )

        return loss_dict

    def learn_individual(
        self,
        agent_id: str,
        stacked_actions: torch.Tensor,
        stacked_next_actions: torch.Tensor,
        states: StandardTensorDict,
        next_states: StandardTensorDict,
        actions: StandardTensorDict,
        rewards: StandardTensorDict,
        dones: StandardTensorDict,
    ) -> Tuple[Optional[float], float]:
        """
        Inner call to each agent for the learning/algo training steps, up until the soft updates.
        Applies all forward/backward props.

        :param agent_id: ID of the agent
        :type agent_id: str

        :param stacked_actions: Stacked actions tensor for CNN architecture
        :type stacked_actions: Optional[torch.Tensor]
        :param stacked_next_actions: Stacked next actions tensor for CNN architecture
        :type stacked_next_actions: Optional[torch.Tensor]
        :param states: Dictionary of current states for each agent
        :type states: dict[str, torch.Tensor]
        :param actions: Dictionary of actions taken by each agent
        :type actions: dict[str, torch.Tensor]
        :param rewards: Dictionary of rewards received by each agent
        :type rewards: dict[str, torch.Tensor]
        :param dones: Dictionary of done flags for each agent
        :type dones: dict[str, torch.Tensor]

        :return: Tuple containing actor loss (if applicable) and critic loss
        :rtype: Tuple[Optional[float], float]
        """
        actor = self.actors[agent_id]
        critic_1 = self.critics_1[agent_id]
        critic_target_1 = self.critic_targets_1[agent_id]
        critic_2 = self.critics_2[agent_id]
        critic_target_2 = self.critic_targets_2[agent_id]
        actor_optimizer = self.actor_optimizers[agent_id]
        critic_1_optimizer = self.critic_1_optimizers[agent_id]
        critic_2_optimizer = self.critic_2_optimizers[agent_id]

        if self.accelerator is not None:
            with critic_1.no_sync():
                q_value_1 = critic_1(states, stacked_actions)
            with critic_2.no_sync():
                q_value_2 = critic_2(states, stacked_actions)
        else:
            q_value_1 = critic_1(states, stacked_actions)
            q_value_2 = critic_2(states, stacked_actions)

        with torch.no_grad():
            if self.accelerator is not None:
                with critic_target_1.no_sync():
                    q_value_next_state_1 = critic_target_1(
                        next_states, stacked_next_actions
                    )
                with critic_target_2.no_sync():
                    q_value_next_state_2 = critic_target_2(
                        next_states, stacked_next_actions
                    )
            else:
                q_value_next_state_1 = critic_target_1(
                    next_states, stacked_next_actions
                )
                q_value_next_state_2 = critic_target_2(
                    next_states, stacked_next_actions
                )

        q_value_next_state = torch.min(q_value_next_state_1, q_value_next_state_2)

        # Replace NaN rewards with 0 and dones with True
        rewards[agent_id] = torch.where(
            torch.isnan(rewards[agent_id]),
            torch.full_like(rewards[agent_id], 0),
            rewards[agent_id],
        ).to(torch.float32)

        dones[agent_id] = torch.where(
            torch.isnan(dones[agent_id]),
            torch.full_like(dones[agent_id], 1),
            dones[agent_id],
        ).to(torch.uint8)

        y_j = (
            rewards[agent_id] + (1 - dones[agent_id]) * self.gamma * q_value_next_state
        )

        critic_loss = self.criterion(q_value_1, y_j) + self.criterion(q_value_2, y_j)

        # critic loss backprop
        critic_1_optimizer.zero_grad()
        critic_2_optimizer.zero_grad()
        if self.accelerator is not None:
            self.accelerator.backward(critic_loss)
        else:
            critic_loss.backward()

        critic_1_optimizer.step()
        critic_2_optimizer.step()

        actor_loss = None

        # Calculate actions and actor loss
        if self.accelerator is not None:
            with actor.no_sync():
                action = actor(states[agent_id])
        else:
            action = actor(states[agent_id])

        detached_actions = copy.deepcopy(actions)
        detached_actions[agent_id] = action

        # update actor and targets every policy_freq learn steps
        self.learn_counter[agent_id] += 1
        if self.learn_counter[agent_id] % self.policy_freq == 0:
            stacked_detached_actions = torch.cat(list(detached_actions.values()), dim=1)
            if self.accelerator is not None:
                with critic_1.no_sync():
                    actor_loss = -critic_1(states, stacked_detached_actions).mean()
            else:
                actor_loss = -critic_1(states, stacked_detached_actions).mean()

            # actor loss backprop
            actor_optimizer.zero_grad()
            if self.accelerator is not None:
                self.accelerator.backward(actor_loss)
            else:
                actor_loss.backward()
            actor_optimizer.step()

        return actor_loss.item() if actor_loss is not None else None, critic_loss.item()

    def soft_update(self, net: nn.Module, target: nn.Module) -> None:
        """Soft updates target network.

        :param net: Network to be updated
        :type net: nn.Module
        :param target: Target network
        :type target: nn.Module
        """
        for eval_param, target_param in zip(net.parameters(), target.parameters()):
            target_param.data.copy_(
                self.tau * eval_param.data + (1.0 - self.tau) * target_param.data
            )

    def test(
        self,
        env: PzEnvType,
        swap_channels: bool = False,
        max_steps: Optional[int] = None,
        loop: int = 3,
        sum_scores: bool = True,
    ) -> float:
        """Returns mean test score of agent in environment with epsilon-greedy policy.

        :param env: The environment to be tested in
        :type env: Gym-style environment
        :param swap_channels: Swap image channels dimension from last to first [H, W, C] -> [C, H, W], defaults to False
        :type swap_channels: bool, optional
        :param max_steps: Maximum number of testing steps, defaults to None
        :type max_steps: int, optional
        :param loop: Number of testing loops/episodes to complete. The returned score is the mean. Defaults to 3
        :type loop: int, optional
        :param sum_scores: Boolean flag to indicate whether to sum sub-agent scores, defaults to True
        :type sum_scores: book, optional
        """
        self.set_training_mode(False)
        with torch.no_grad():
            rewards = []
            if hasattr(env, "num_envs"):
                num_envs = env.num_envs
                is_vectorised = True
            else:
                num_envs = 1
                is_vectorised = False

            for _ in range(loop):
                obs, info = env.reset()
                scores = (
                    np.zeros((num_envs, 1))
                    if sum_scores
                    else np.zeros((num_envs, len(self.agent_ids)))
                )
                completed_episode_scores = (
                    np.zeros((num_envs, 1))
                    if sum_scores
                    else np.zeros((num_envs, len(self.agent_ids)))
                )
                finished = np.zeros(num_envs)
                step = 0
                while not np.all(finished):
                    step += 1
                    if swap_channels:
                        expand_dims = not is_vectorised
                        obs = {
                            agent_id: obs_channels_to_first(s, expand_dims)
                            for agent_id, s in obs.items()
                        }

                    action, _ = self.get_action(
                        obs,
                        infos=info,
                    )

                    if not is_vectorised:
                        action = {agent: act[0] for agent, act in action.items()}

                    obs, reward, term, trunc, info = env.step(action)

                    # Compute score increment (replace NaNs representing inactive agents with 0)
                    agent_rewards = np.array(list(reward.values())).transpose()
                    agent_rewards = np.where(np.isnan(agent_rewards), 0, agent_rewards)
                    score_increment = (
                        (
                            np.sum(agent_rewards, axis=-1)[:, np.newaxis]
                            if is_vectorised
                            else np.sum(agent_rewards, axis=-1)
                        )
                        if sum_scores
                        else agent_rewards
                    )
                    scores += score_increment

                    dones = {}
                    for agent_id in self.agent_ids:
                        terminated = term.get(agent_id, True)
                        truncated = trunc.get(agent_id, False)

                        # Replace NaNs with True (indicate killed agent)
                        terminated = np.where(
                            np.isnan(terminated), True, terminated
                        ).astype(bool)
                        truncated = np.where(
                            np.isnan(truncated), False, truncated
                        ).astype(bool)

                        dones[agent_id] = terminated | truncated

                    if not is_vectorised:
                        dones = {
                            agent: np.array([dones[agent_id]])
                            for agent in self.agent_ids
                        }

                    for idx, agent_dones in enumerate(zip(*dones.values())):
                        if (
                            np.all(agent_dones)
                            or (max_steps is not None and step == max_steps)
                        ) and not finished[idx]:
                            completed_episode_scores[idx] = scores[idx]
                            finished[idx] = 1

                rewards.append(np.mean(completed_episode_scores, axis=0))

        mean_fit = np.mean(rewards, axis=0)
        mean_fit = mean_fit[0] if sum_scores else mean_fit
        self.fitness.append(mean_fit)
        return mean_fit
