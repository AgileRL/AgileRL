import copy
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from gymnasium import spaces

from agilerl.algorithms.core import MultiAgentRLAlgorithm
from agilerl.algorithms.core.registry import HyperparameterConfig, NetworkGroup
from agilerl.algorithms.core.wrappers import OptimizerWrapper
from agilerl.modules.base import EvolvableModule
from agilerl.modules.configs import MlpNetConfig
from agilerl.networks.actors import DeterministicActor
from agilerl.networks.q_networks import ContinuousQNetwork
from agilerl.typing import (
    ArrayDict,
    InfosDict,
    ObservationType,
    PzEnvType,
    TensorDict,
)
from agilerl.utils.algo_utils import (
    concatenate_spaces,
    contains_image_space,
    key_in_nested_dict,
    make_safe_deepcopies,
    multi_agent_sample_tensor_from_space,
    obs_channels_to_first,
)
from agilerl.utils.evolvable_networks import get_default_encoder_config


class MATD3(MultiAgentRLAlgorithm):
    """Multi-Agent Twin Delayed Deep Deterministic Policy Gradient (MATD3) algorithm.

    Paper: https://arxiv.org/abs/1910.01465

    :param observation_spaces: Observation space for each agent
    :type observation_spaces: List[spaces.Space]
    :param action_spaces: Action space for each agent
    :type action_spaces: List[spaces.Space]
    :param agent_ids: Agent ID for each agent
    :type agent_ids: List[str]
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
    :type actor_networks: Optional[List[EvolvableModule]], optional
    :param critic_networks: List containing two lists of custom critic networks, defaults to None
    :type critic_networks: Optional[List[List[EvolvableModule]]], optional
    :param device: Device for accelerated computing, 'cpu' or 'cuda', defaults to 'cpu'
    :type device: str, optional
    :param accelerator: Accelerator for distributed computing, defaults to None
    :type accelerator: accelerate.Accelerator(), optional
    :param torch_compiler: The torch compile mode 'default', 'reduce-overhead' or 'max-autotune', defaults to None
    :type torch_compiler: Optional[str], optional
    :param wrap: Wrap models for distributed training upon creation, defaults to True
    :type wrap: bool, optional
    """

    actors: List[Union[nn.Module, DeterministicActor]]
    actor_targets: List[Union[nn.Module, DeterministicActor]]
    critics_1: List[Union[nn.Module, ContinuousQNetwork]]
    critic_targets_1: List[Union[nn.Module, ContinuousQNetwork]]
    critics_2: List[Union[nn.Module, ContinuousQNetwork]]
    critic_targets_2: List[Union[nn.Module, ContinuousQNetwork]]

    def __init__(
        self,
        observation_spaces: List[spaces.Space],
        action_spaces: List[spaces.Space],
        agent_ids: List[str],
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
        actor_networks: Optional[list[EvolvableModule]] = None,
        critic_networks: Optional[List[List[nn.Module]]] = None,
        device: str = "cpu",
        accelerator: Optional[Any] = None,
        torch_compiler: Optional[str] = None,
        wrap: bool = True,
    ):
        super().__init__(
            observation_spaces,
            action_spaces,
            agent_ids,
            index=index,
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
                "Actor and critic network lists must both be supplied to use custom networks. Defaulting to net config."
            )
        assert isinstance(
            wrap, bool
        ), "Wrap models flag must be boolean value True or False."

        self.is_image_space = contains_image_space(self.single_space)
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
        self.sample_gaussian = [
            torch.zeros(*(vect_noise_dim, self.action_dims[idx]), device=self.device)
            for idx in range(self.n_agents)
        ]
        self.expl_noise = (
            expl_noise
            if isinstance(expl_noise, list)
            else [
                expl_noise
                * torch.ones(*(vect_noise_dim, action_dim), device=self.device)
                for action_dim in self.action_dims
            ]
        )
        self.mean_noise = (
            mean_noise
            if isinstance(mean_noise, list)
            else [
                mean_noise
                * torch.ones(*(vect_noise_dim, action_dim), device=self.device)
                for action_dim in self.action_dims
            ]
        )
        self.current_noise = [
            torch.zeros(*(vect_noise_dim, action_dim), device=self.device)
            for action_dim in self.action_dims
        ]
        self.theta = theta
        self.dt = dt
        self.sqdt = dt ** (0.5)

        if actor_networks is not None and critic_networks is not None:
            assert (
                len({type(net) for net in actor_networks}) == 1
            ), "'actor_networks' must all be the same type"
            assert (
                len({type(net) for net in critic_networks}) == 1
            ), "'critic_networks' must all be the same type"

            if not all(isinstance(net, EvolvableModule) for net in actor_networks):
                raise TypeError(
                    "All actor networks must be instances of EvolvableModule"
                )
            if not all(isinstance(net, EvolvableModule) for net in critic_networks[0]):
                raise TypeError(
                    "All critic networks must be instances of EvolvableModule"
                )
            if not all(isinstance(net, EvolvableModule) for net in critic_networks[1]):
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
            net_config = {} if net_config is None else net_config
            simba = net_config.get("simba", False)
            critic_net_config = copy.deepcopy(net_config)

            encoder_config = net_config.get("encoder_config", None)
            critic_encoder_config = critic_net_config.get("encoder_config", None)
            head_config = net_config.get("head_config", None)

            # Determine actor output activation from action space
            if head_config is not None:
                if self.discrete_actions:
                    head_config["output_activation"] = "GumbelSoftmax"

                critic_head_config = copy.deepcopy(head_config)
                critic_head_config["output_activation"] = None
            else:
                output_activation = "GumbelSoftmax" if self.discrete_actions else None
                head_config = MlpNetConfig(
                    hidden_size=[64], output_activation=output_activation
                )
                if head_config.output_activation is None:
                    head_config.pop("output_activation")

                critic_head_config = MlpNetConfig(hidden_size=[64])

            if encoder_config is None:
                encoder_config = get_default_encoder_config(self.single_space, simba)
                critic_encoder_config = get_default_encoder_config(
                    self.single_space, simba
                )

            # For image spaces we need to give a sample input tensor to
            # build networks with Conv3d blocks appropriately
            # NOTE: Currently AgileRL only supports Dict observation spaces
            # with a unique image space (i.e. all agents and all subspaces
            # contain the same image space).
            if self.is_image_space:
                actor_sample_input = multi_agent_sample_tensor_from_space(
                    self.single_space, self.n_agents, device=self.device
                )
                critic_sample_input = multi_agent_sample_tensor_from_space(
                    self.single_space,
                    self.n_agents,
                    device=self.device,
                    critic=True,
                )

                def get_first_sample_input(
                    sample_input: Union[
                        Tuple[torch.Tensor, ...], Dict[str, torch.Tensor]
                    ],
                ) -> torch.Tensor:
                    if isinstance(self.single_space, spaces.Dict):
                        return list(sample_input.values())[0]
                    elif isinstance(self.single_space, spaces.Tuple):
                        return sample_input[0]

                    return sample_input

                actor_sample_input = get_first_sample_input(actor_sample_input)
                critic_sample_input = get_first_sample_input(critic_sample_input)

                if isinstance(self.single_space, (spaces.Dict, spaces.Tuple)):
                    encoder_config["cnn_config"]["sample_input"] = actor_sample_input
                    critic_encoder_config["cnn_config"][
                        "sample_input"
                    ] = critic_sample_input
                else:
                    encoder_config["sample_input"] = actor_sample_input
                    critic_encoder_config["sample_input"] = critic_sample_input

            net_config["encoder_config"] = encoder_config
            net_config["head_config"] = head_config

            critic_net_config["encoder_config"] = critic_encoder_config
            critic_net_config["head_config"] = critic_head_config

            clip_actions = self.torch_compiler is None

            def create_actor(idx):
                return DeterministicActor(
                    self.observation_spaces[idx],
                    self.action_spaces[idx],
                    n_agents=self.n_agents,
                    device=self.device,
                    clip_actions=clip_actions,
                    **copy.deepcopy(net_config),
                )

            # NOTE: Critic uses observations + actions of all agents to predict Q-value
            concatenated_obs_space = concatenate_spaces(observation_spaces)
            concatenated_action_space = concatenate_spaces(action_spaces)

            def create_critic():
                return ContinuousQNetwork(
                    observation_space=concatenated_obs_space,
                    action_space=concatenated_action_space,
                    n_agents=self.n_agents,
                    device=self.device,
                    **copy.deepcopy(critic_net_config),
                )

            self.actors = [create_actor(idx) for idx in range(self.n_agents)]
            self.critics_1 = [create_critic() for _ in range(self.n_agents)]
            self.critics_2 = [create_critic() for _ in range(self.n_agents)]
            self.actor_targets = [create_actor(idx) for idx in range(self.n_agents)]
            self.critic_targets_1 = [create_critic() for _ in range(self.n_agents)]
            self.critic_targets_2 = [create_critic() for _ in range(self.n_agents)]

        # Initialise target network parameters
        for actor, actor_target in zip(self.actors, self.actor_targets):
            actor_target.load_state_dict(actor.state_dict())

        for critic_1, critic_2, critic_target_1, critic_target_2 in zip(
            self.critics_1, self.critics_2, self.critic_targets_1, self.critic_targets_2
        ):
            critic_target_1.load_state_dict(critic_1.state_dict())
            critic_target_2.load_state_dict(critic_2.state_dict())

        # Optimizers
        self.actor_optimizers = OptimizerWrapper(
            optim.Adam,
            networks=self.actors,
            lr=lr_actor,
            multiagent=True,
        )

        self.critic_1_optimizers = OptimizerWrapper(
            optim.Adam,
            networks=self.critics_1,
            lr=lr_critic,
            multiagent=True,
        )

        self.critic_2_optimizers = OptimizerWrapper(
            optim.Adam,
            networks=self.critics_2,
            lr=lr_critic,
            multiagent=True,
        )

        if self.accelerator is not None and wrap:
            self.wrap_models()
        elif self.torch_compiler:
            if (
                any(actor.output_activation == "GumbelSoftmax" for actor in self.actors)
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
                eval=self.actors,
                shared=self.actor_targets,
                policy=True,
                multiagent=True,
            )
        )
        self.register_network_group(
            NetworkGroup(
                eval=self.critics_1, shared=self.critic_targets_1, multiagent=True
            )
        )
        self.register_network_group(
            NetworkGroup(
                eval=self.critics_2, shared=self.critic_targets_2, multiagent=True
            )
        )

    def process_infos(self, infos: InfosDict) -> Tuple[ArrayDict, ArrayDict, ArrayDict]:
        """
        Process the information, extract env_defined_actions, action_masks and agent_masks

        :param infos: Info dict
        :type infos: Dict[str, Dict[...]]
        """
        if infos is None:
            infos = {agent: {} for agent in self.agent_ids}

        env_defined_actions, agent_masks = self.extract_agent_masks(infos)
        action_masks = self.extract_action_masks(infos)
        return action_masks, env_defined_actions, agent_masks

    def get_action(
        self,
        obs: Dict[str, ObservationType],
        training: bool = True,
        infos: Optional[InfosDict] = None,
    ) -> Tuple[ArrayDict, ArrayDict]:
        """Returns the next action to take in the environment.
        Epsilon is the probability of taking a random action, used for exploration.
        For epsilon-greedy behaviour, set epsilon to 0.

        :param obs: Environment observations: {'agent_0': state_dim_0, ..., 'agent_n': state_dim_n}
        :type obs: Dict[str, numpy.Array]
        :param training: Agent is training, use exploration noise, defaults to True
        :type training: bool, optional
        :param infos: Information dictionary from environment, defaults to None
        :type infos: Dict[str, Dict[...]], optional

        :return: Action to take in the environment
        :rtype: Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]
        """
        assert not key_in_nested_dict(
            obs, "action_mask"
        ), "AgileRL requires action masks to be defined in the information dictionary."

        action_masks, env_defined_actions, agent_masks = self.process_infos(infos)

        # Preprocess observations
        preprocessed_states = list(self.preprocess_observation(obs).values())

        action_dict = {}
        for idx, (agent_id, obs, actor) in enumerate(
            zip(self.agent_ids, preprocessed_states, self.actors)
        ):
            actor.eval()
            if self.accelerator is not None:
                with actor.no_sync(), torch.no_grad():
                    actions = actor(obs)
            else:
                with torch.no_grad():
                    actions = actor(obs)

            if self.torch_compiler is not None and isinstance(
                self.action_spaces[idx], spaces.Box
            ):
                actions = DeterministicActor.rescale_action(
                    action=actions,
                    low=actor.action_low,
                    high=actor.action_high,
                    output_activation=actor.output_activation,
                )

            actor.train()
            if training:
                if self.discrete_actions:
                    min_action, max_action = 0, 1
                else:
                    min_action, max_action = (
                        self.min_action[idx][0],
                        self.max_action[idx][0],
                    )

                # Add noise to actions for exploration
                actions = torch.clamp(
                    actions + self.action_noise(idx),
                    min_action,
                    max_action,
                )

            action_dict[agent_id] = actions.cpu().numpy()

        discrete_action_dict = None
        if self.discrete_actions:
            discrete_action_dict = {}
            for agent, action in action_dict.items():
                mask = (
                    1 - np.array(action_masks[agent])
                    if action_masks[agent] is not None
                    else None
                )
                action: np.ndarray = np.ma.array(action, mask=mask)

                discrete_action_dict[agent] = action.argmax(axis=-1)
                if len(discrete_action_dict[agent].shape) == 1:
                    discrete_action_dict[agent] = discrete_action_dict[agent][
                        :, np.newaxis
                    ]
        else:
            discrete_action_dict = None

        # If using env_defined_actions replace actions
        if env_defined_actions is not None:
            for agent in self.agent_ids:
                if self.discrete_actions:
                    discrete_action_dict[agent][agent_masks[agent]] = (
                        env_defined_actions[agent][agent_masks[agent]]
                    )
                else:
                    action_dict[agent][agent_masks[agent]] = env_defined_actions[agent][
                        agent_masks[agent]
                    ]

        return (action_dict, discrete_action_dict)

    def action_noise(self, idx: int) -> torch.Tensor:
        """Create action noise for exploration, either Ornstein Uhlenbeck or
            from a normal distribution.

        :param idx: Agent index for action dims
        :type idx: int
        :return: Action noise
        :rtype: torch.Tensor
        """
        if self.O_U_noise:
            noise = (
                self.current_noise[idx]
                + self.theta
                * (self.mean_noise[idx] - self.current_noise[idx])
                * self.dt
                + self.expl_noise[idx] * self.sqdt * self.sample_gaussian[idx].normal_()
            )
            self.current_noise[idx] = noise
        else:
            torch.normal(
                self.mean_noise[idx],
                self.expl_noise[idx],
                out=self.sample_gaussian[idx],
            )
            noise = self.sample_gaussian[idx]
        return noise

    def reset_action_noise(self, indices: List[int]) -> None:
        """Reset action noise.

        :param indices: List of indices to reset noise for
        :type indices: List[int]
        """
        for i in range(len(self.current_noise)):
            for idx in indices:
                self.current_noise[i][idx, :] = 0

    def learn(self, experiences: Tuple[TensorDict, ...]) -> Dict[str, float]:
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
            for i, agent_id_label in enumerate(self.agent_ids):
                next_actions.append(self.actor_targets[i](next_states[agent_id_label]))

        # Stack states and actions
        stacked_states = self.stack_critic_observations(states)
        stacked_next_states = self.stack_critic_observations(next_states)
        stacked_actions = torch.cat(list(actions.values()), dim=1)
        stacked_next_actions = torch.cat(next_actions, dim=1)

        loss_dict = {}
        for idx, (
            agent_id,
            actor,
            critic_1,
            critic_target_1,
            critic_2,
            critic_target_2,
            actor_optimizer,
            critic_1_optimizer,
            critic_2_optimizer,
        ) in enumerate(
            zip(
                self.agent_ids,
                self.actors,
                self.critics_1,
                self.critic_targets_1,
                self.critics_2,
                self.critic_targets_2,
                self.actor_optimizers,
                self.critic_1_optimizers,
                self.critic_2_optimizers,
            )
        ):
            loss_dict[f"{agent_id}"] = self.learn_individual(
                idx,
                agent_id,
                actor,
                critic_1,
                critic_target_1,
                critic_2,
                critic_target_2,
                actor_optimizer,
                critic_1_optimizer,
                critic_2_optimizer,
                stacked_states,
                stacked_actions,
                stacked_next_states,
                stacked_next_actions,
                states,
                actions,
                rewards,
                dones,
            )

        if self.learn_counter[agent_id] % self.policy_freq == 0:
            for (
                actor,
                actor_target,
                critic_1,
                critic_target_1,
                critic_2,
                critic_target_2,
            ) in zip(
                self.actors,
                self.actor_targets,
                self.critics_1,
                self.critic_targets_1,
                self.critics_2,
                self.critic_targets_2,
            ):
                self.soft_update(actor, actor_target)
                self.soft_update(critic_1, critic_target_1)
                self.soft_update(critic_2, critic_target_2)

        return loss_dict

    def learn_individual(
        self,
        idx: int,
        agent_id: str,
        actor: nn.Module,
        critic_1: nn.Module,
        critic_target_1: nn.Module,
        critic_2: nn.Module,
        critic_target_2: nn.Module,
        actor_optimizer: optim.Optimizer,
        critic_1_optimizer: optim.Optimizer,
        critic_2_optimizer: optim.Optimizer,
        stacked_states: torch.Tensor,
        stacked_actions: torch.Tensor,
        stacked_next_states: torch.Tensor,
        stacked_next_actions: torch.Tensor,
        states: TensorDict,
        actions: TensorDict,
        rewards: TensorDict,
        dones: TensorDict,
    ) -> Tuple[Optional[float], float]:
        """
        Inner call to each agent for the learning/algo training steps, up until the soft updates.
        Applies all forward/backward props.

        :param idx: Index of the agent
        :type idx: int
        :param agent_id: ID of the agent
        :type agent_id: str
        :param actor: Actor network of the agent
        :type actor: nn.Module
        :param critic_1: First critic network of the agent
        :type critic_1: nn.Module
        :param critic_target_1: Target network for the first critic
        :type critic_target_1: nn.Module
        :param critic_2: Second critic network of the agent
        :type critic_2: nn.Module
        :param critic_target_2: Target network for the second critic
        :type critic_target_2: nn.Module
        :param actor_optimizer: Optimizer for the actor network
        :type actor_optimizer: optim.Optimizer
        :param critic_1_optimizer: Optimizer for the first critic network
        :type critic_1_optimizer: optim.Optimizer
        :param critic_2_optimizer: Optimizer for the second critic network
        :type critic_2_optimizer: optim.Optimizer
        :param stacked_states: Stacked states tensor for CNN architecture
        :type stacked_states: Optional[torch.Tensor]
        :param stacked_actions: Stacked actions tensor for CNN architecture
        :type stacked_actions: Optional[torch.Tensor]
        :param stacked_next_states: Stacked next states tensor for CNN architecture
        :type stacked_next_states: Optional[torch.Tensor]
        :param stacked_next_actions: Stacked next actions tensor for CNN architecture
        :type stacked_next_actions: Optional[torch.Tensor]
        :param states: Dictionary of current states for each agent
        :type states: TensorDict
        :param actions: Dictionary of actions taken by each agent
        :type actions: TensorDict
        :param rewards: Dictionary of rewards received by each agent
        :type rewards: TensorDict
        :param dones: Dictionary of done flags for each agent
        :type dones: TensorDict

        :return: Tuple containing actor loss (if applicable) and critic loss
        :rtype: Tuple[Optional[float], float]
        """
        if self.accelerator is not None:
            with critic_1.no_sync():
                q_value_1 = critic_1(stacked_states, stacked_actions)
            with critic_2.no_sync():
                q_value_2 = critic_2(stacked_states, stacked_actions)
        else:
            q_value_1 = critic_1(stacked_states, stacked_actions)
            q_value_2 = critic_2(stacked_states, stacked_actions)

        with torch.no_grad():
            if self.accelerator is not None:
                with critic_target_1.no_sync():
                    q_value_next_state_1 = critic_target_1(
                        stacked_next_states, stacked_next_actions
                    )
                with critic_target_2.no_sync():
                    q_value_next_state_2 = critic_target_2(
                        stacked_next_states, stacked_next_actions
                    )
            else:
                q_value_next_state_1 = critic_target_1(
                    stacked_next_states, stacked_next_actions
                )
                q_value_next_state_2 = critic_target_2(
                    stacked_next_states, stacked_next_actions
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
                    actor_loss = -critic_1(
                        stacked_states, stacked_detached_actions
                    ).mean()
            else:
                actor_loss = -critic_1(stacked_states, stacked_detached_actions).mean()

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
                    cont_actions, discrete_action = self.get_action(
                        obs,
                        training=False,
                        infos=info,
                    )

                    action = discrete_action if self.discrete_actions else cont_actions
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
