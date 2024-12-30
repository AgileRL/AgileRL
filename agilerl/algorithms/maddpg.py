from typing import Optional, Dict, Any, List, Tuple
import copy
import inspect
import warnings

import dill
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from gymnasium import spaces

from agilerl.typing import NumpyObsType, ArrayDict, InfosDict, TensorDict, ExperiencesType, ArrayLike
from agilerl.algorithms.core import MultiAgentAlgorithm
from agilerl.algorithms.core.wrappers import OptimizerWrapper
from agilerl.algorithms.core.registry import NetworkGroup
from agilerl.modules.configs import MlpNetConfig
from agilerl.networks.q_networks import ContinuousQNetwork
from agilerl.networks.actors import DeterministicActor
from agilerl.modules.base import EvolvableModule
from agilerl.utils.algo_utils import (
    key_in_nested_dict,
    make_safe_deepcopies
)

class MADDPG(MultiAgentAlgorithm):
    """The MADDPG algorithm class. MADDPG paper: https://arxiv.org/abs/1706.02275

    :param observation_spaces: Observation space for each agent
    :type observation_spaces: list[spaces.Space]
    :param action_spaces: Action space for each agent
    :type action_spaces: list[spaces.Space]
    :param n_agents: Number of agents
    :type n_agents: int
    :param agent_ids: Agent ID for each agent
    :type agent_ids: list[str]
    :param O_U_noise: Use Ornstein Uhlenbeck action noise for exploration. If False, uses Gaussian noise. Defaults to True
    :type O_U_noise: bool, optional
    :param vect_noise_dim: Vectorization dimension of environment for action noise, defaults to 1
    :type vect_noise_dim: int, optional
    :param expl_noise: Scale for Ornstein Uhlenbeck action noise, or standard deviation for Gaussian exploration noise
    :type expl_noise: float, optional
    :param mean_noise: Mean of exploration noise, defaults to 0.0
    :type mean_noise: float, optional
    :param theta: Rate of mean reversion in Ornstein Uhlenbeck action noise, defaults to 0.15
    :type theta: float, optional
    :param dt: Timestep for Ornstein Uhlenbeck action noise update, defaults to 1e-2
    :type dt: float, optional
    :param index: Index to keep track of object instance during tournament selection and mutation, defaults to 0
    :type index: int, optional
    :param net_config: Encoder configuration, defaults to mlp with hidden size [64,64]
    :type net_config: dict, optional
    :param head_config: Head configuration for actors and critics network, defaults to None
    :type head_config: dict, optional
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
    :param mutation: Most recent mutation to agent, defaults to None
    :type mutation: str, optional
    :param actor_networks: List of custom actor networks, defaults to None
    :type actor_networks: list[nn.Module], optional
    :param critic_networks: List of custom critic networks, defaults to None
    :type critic_networks: list[nn.Module], optional
    :param device: Device for accelerated computing, 'cpu' or 'cuda', defaults to 'cpu'
    :type device: str, optional
    :param accelerator: Accelerator for distributed computing, defaults to None
    :type accelerator: accelerate.Accelerator(), optional
    :param torch_compile: the torch compile mode 'default', 'reduce-overhead' or 'max-autotune', defaults to None
    :type torch_compile: str, optional
    :param wrap: Wrap models for distributed training upon creation, defaults to True
    :type wrap: bool, optional
    """
    actors: List[EvolvableModule]
    actor_targets: List[EvolvableModule]
    critics: List[EvolvableModule]
    critic_targets: List[EvolvableModule]

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
        net_config: Optional[Dict[str, Any]] = None,
        head_config: Optional[Dict[str, Any]] = None,
        batch_size: int = 64,
        lr_actor: float = 0.001,
        lr_critic: float = 0.01,
        learn_step: int = 5,
        gamma: float = 0.95,
        tau: float = 0.01,
        mut: Optional[str] = None,
        normalize_images: bool = True,
        actor_networks: Optional[List[EvolvableModule]] = None,
        critic_networks: Optional[List[EvolvableModule]] = None,
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
            learn_step=learn_step,
            device=device,
            accelerator=accelerator,
            normalize_images=normalize_images,
            torch_compiler=torch_compiler,
            name="MADDPG",
        )

        assert isinstance(batch_size, int), "Batch size must be an integer."
        assert batch_size >= 1, "Batch size must be greater than or equal to one."
        assert isinstance(lr_actor, float), "Actor learning rate must be a float."
        assert lr_actor > 0, "Actor learning rate must be greater than zero."
        assert isinstance(lr_critic, float), "Critic learning rate must be a float."
        assert lr_critic > 0, "Critic learning rate must be greater than zero."
        assert isinstance(gamma, float), "Gamma must be a float."
        assert isinstance(tau, float), "Tau must be a float."
        assert tau > 0, "Tau must be greater than zero."
        if (actor_networks is not None) != (critic_networks is not None):
            warnings.warn(
                "Actor and critic network lists must both be supplied to use custom networks. Defaulting to net config."
            )
        assert isinstance(
            wrap, bool
        ), "Wrap models flag must be boolean value True or False."

        if self.max_action is not None and self.min_action is not None:
            for x, n in zip(self.max_action, self.min_action):
                x, n = x[0], n[0]
                assert x > n, "Max action must be greater than min action."
                assert x > 0, "Max action must be greater than zero."
                assert n <= 0, "Min action must be less than or equal to zero."

        self.batch_size = batch_size
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.gamma = gamma
        self.tau = tau
        self.mut = mut
        self.learn_counter = 0
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
                expl_noise * torch.ones(*(vect_noise_dim, action_dim), device=self.device)
                for action_dim in self.action_dims
            ]
        )
        self.mean_noise = (
            mean_noise
            if isinstance(mean_noise, list)
            else [
                mean_noise * torch.ones(*(vect_noise_dim, action_dim), device=self.device)
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

        # if "mlp_output_activation" not in self.net_config.keys():
        #     if not self.discrete_actions:
        #         if self.min_action[idx][0] < 0:
        #             self.net_config["mlp_output_activation"] = "Tanh"
        #         else:
        #             self.net_config["mlp_output_activation"] = "Sigmoid"
        #     else:
        #         self.net_config["mlp_output_activation"] = "GumbelSoftmax"

        if actor_networks is not None and critic_networks is not None:
            assert (
                len({type(net) for net in actor_networks}) == 1
            ), "'actor_networks' must all be the same type"
            assert (
                len({type(net) for net in critic_networks}) == 1
            ), "'critic_networks' must all be the same type"

            if not all(isinstance(net, EvolvableModule) for net in actor_networks):
                raise ValueError(
                    "All actor networks must be instances of EvolvableModule"
                )
            if not all(isinstance(net, EvolvableModule) for net in critic_networks):
                raise ValueError(
                    "All critic networks must be instances of EvolvableModule"
                )
            self.actors, self.critics = make_safe_deepcopies(actor_networks, critic_networks)
        else:
            # model
            self.actors = []
            if head_config is not None:
                critic_head_config = copy.deepcopy(head_config)
                critic_head_config["output_activation"] = None
            else:
                critic_head_config = MlpNetConfig(hidden_size=[64])

            # create_actor = lambda: 

            # create_critic = lambda: 


        # Create target networks
        self.actor_targets = copy.deepcopy(self.actors)
        self.critic_targets = copy.deepcopy(self.critics)

        # Initialise target network parameters
        for actor, actor_target in zip(self.actors, self.actor_targets):
            actor_target.load_state_dict(actor.state_dict())
        for critic, critic_target in zip(self.critics, self.critic_targets):
            critic_target.load_state_dict(critic.state_dict())

        # Optimizers
        self.actor_optimizers = OptimizerWrapper(
            optim.Adam,
            networks=self.actors,
            optimizer_kwargs={"lr": self.lr_actor},
            multiagent=True
        )
        self.critic_optimizers = OptimizerWrapper(
            optim.Adam,
            networks=self.critics,
            optimizer_kwargs={"lr": self.lr_critic},
            multiagent=True
        )

        if self.accelerator is not None and wrap:
            self.wrap_models()
        else:
            if self.torch_compiler:
                if (
                    any(
                        actor.output_activation == "GumbelSoftmax"
                        for actor in self.actors
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
                eval=self.actors,
                shared=self.actor_targets,
                policy=True,
                multiagent=True
            )
        )
        self.register_network_group(
            NetworkGroup(
                eval=self.critics,
                shared=self.critic_targets,
                multiagent=True
            )
        )

    def scale_to_action_space(self, action: ArrayLike, idx: int) -> torch.Tensor:
        """Scales actions to action space defined by self.min_action and self.max_action.

        :param action: Action to be scaled
        :type action: numpy.ndarray
        """
        mlp_output_activation = self.actors[idx].output_activation
        if mlp_output_activation in ["Tanh"]:
            pre_scaled_min = -1
            pre_scaled_max = 1
        elif mlp_output_activation in ["Sigmoid", "Softmax", "GumbelSoftmax"]:
            pre_scaled_min = 0
            pre_scaled_max = 1
        else:
            return torch.where(
                action > 0,
                action * self.max_action[idx][0],
                action * -self.min_action[idx][0],
            )

        if not (
            isinstance(self.min_action[idx][0], (np.ndarray, torch.Tensor))
            or isinstance(self.max_action[idx][0], (np.ndarray, torch.Tensor))
        ):
            if (
                pre_scaled_min == self.min_action[idx][0]
                and pre_scaled_max == self.max_action[idx][0]
            ):
                return action

        return self.min_action[idx][0] + (
            self.max_action[idx][0] - self.min_action[idx][0]
        ) * (action - pre_scaled_min) / (pre_scaled_max - pre_scaled_min)

    def extract_action_masks(self, infos: InfosDict) -> ArrayDict:
        """Extract action masks from info dictionary

        :param infos: Info dict
        :type infos: Dict[str, Dict[...]]

        :return: Action masks
        :rtype: Dict[str, np.ndarray]
        """
        action_masks = {
            agent: info.get("action_mask", None) if isinstance(info, dict) else None
            for agent, info in infos.items()
            if agent in self.agent_ids
        }  # Get dict of form {"agent_id" : [1, 0, 0, 0]...} etc

        return action_masks

    def extract_agent_masks(
            self,
            infos: InfosDict
            ) -> Tuple[ArrayDict, ArrayDict]:
        """Extract env_defined_actions from info dictionary and determine agent masks

        :param infos: Info dict
        :type infos: Dict[str, Dict[...]]

        :return: env_defined_actions, agent_masks
        :rtype: Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]
        """
        if all(not info for agent, info in infos.items() if agent in self.agent_ids):
            return None, None
        env_defined_actions = {
            agent: (
                info.get("env_defined_actions", None)
                if isinstance(info, dict)
                else None
            )
            for agent, info in infos.items()
            if agent in self.agent_ids
        }
        agent_masks = None
        if env_defined_actions is not None:
            agent_masks = {}
            for idx, agent in enumerate(env_defined_actions.keys()):
                # Handle None if environment isn't vectorized
                if env_defined_actions[agent] is None:
                    if not self.discrete_actions:
                        nan_arr = np.empty(self.action_dims[idx])
                        nan_arr[:] = np.nan
                    else:
                        nan_arr = np.array([[np.nan]])
                    env_defined_actions[agent] = nan_arr

                # Handle discrete actions + env not vectorized
                if isinstance(env_defined_actions[agent], (int, float)):
                    env_defined_actions[agent] = np.array(
                        [[env_defined_actions[agent]]]
                    )

                # Ensure additional dimension is added in so shapes align for masking
                if len(env_defined_actions[agent].shape) == 1:
                    env_defined_actions[agent] = (
                        env_defined_actions[agent][:, np.newaxis]
                        if self.discrete_actions
                        else env_defined_actions[agent][np.newaxis, :]
                    )
                agent_masks[agent] = np.where(
                    np.isnan(env_defined_actions[agent]), 0, 1
                ).astype(bool)

        return env_defined_actions, agent_masks

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
            states: Dict[str, NumpyObsType],
            training: bool = True,
            infos: Optional[InfosDict] = None
            ) -> Tuple[TensorDict, TensorDict]:
        """Returns the next action to take in the environment.
        Epsilon is the probability of taking a random action, used for exploration.
        For epsilon-greedy behaviour, set epsilon to 0.

        :param state: Environment observations: {'agent_0': state_dim_0, ..., 'agent_n': state_dim_n}
        :type state: Dict[str, numpy.Array]
        :param training: Agent is training, use exploration noise, defaults to True
        :type training: bool, optional
        :param infos: Information dictionary returned by env.step(actions)
        :type infos: Dict[str, Dict[str, ...]]
        :return: Tuple of actions for each agent
        :rtype: Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]
        """
        assert not key_in_nested_dict(
            states, "action_mask"
        ), "AgileRL requires action masks to be defined in the information dictionary."

        action_masks, env_defined_actions, agent_masks = self.process_infos(infos)

        # Preprocess observations
        preprocessed_states = list(self.preprocess_observation(states).values())
        if self.arch == "cnn":
            preprocessed_states = [state.unsqueeze(2) for state in preprocessed_states]

        action_dict = {}
        for idx, (agent_id, state, actor) in enumerate(
            zip(self.agent_ids, preprocessed_states, self.actors)
        ):
            actor.eval()
            if self.accelerator is not None:
                with actor.no_sync(), torch.no_grad():
                    actions = actor(state)
            else:
                with torch.no_grad():
                    actions = actor(state)
            actor.train()
            if self.discrete_actions and training:
                actions = torch.clamp(actions + self.action_noise(idx), 0, 1)
            elif not self.discrete_actions:
                actions = self.scale_to_action_space(actions, idx)
                if training:
                    actions = torch.clamp(
                        actions + self.action_noise(idx),
                        self.min_action[idx][0],
                        self.max_action[idx][0],
                    )
            action_dict[agent_id] = actions.cpu().numpy()

        if self.discrete_actions:
            discrete_action_dict = {}
            for agent, action in action_dict.items():
                mask = (
                    1 - np.array(action_masks[agent])
                    if action_masks[agent] is not None
                    else None
                )
                action = np.ma.array(action, mask=mask)
                if self.one_hot:
                    discrete_action_dict[agent] = action.argmax(axis=-1)
                else:
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

    def reset_action_noise(self, indices: List[int]):
        """Reset action noise."""
        for i in range(len(self.current_noise)):
            for idx in indices:
                self.current_noise[i][idx, :] = 0

    def learn(self, experiences: ExperiencesType) -> TensorDict:
        """Updates agent network parameters to learn from experiences.

        :param experience: Tuple of dictionaries containing batched states, actions, rewards, next_states,
        dones in that order for each individual agent.
        :type experience: Tuple[Dict[str, torch.Tensor]]

        :return: Loss dictionary
        :rtype: Dict[str, torch.Tensor]
        """
        states, actions, rewards, next_states, dones = experiences

        actions = {agent_id: agent_actions.to(self.device) for agent_id, agent_actions in actions.items()}
        rewards = {agent_id: agent_rewards.to(self.device) for agent_id, agent_rewards in rewards.items()}
        dones = {agent_id: agent_dones.to(self.device) for agent_id, agent_dones in dones.items()}

        # Preprocess observations
        states = self.preprocess_observation(states)
        next_states = self.preprocess_observation(next_states)

        # Need to unsqueeze next states for Conv3d
        if self.arch == "cnn":
            next_states = {
                agent_id: next_state.unsqueeze(2)
                for agent_id, next_state in next_states.items()
            }

        # Get next actions
        next_actions = []
        with torch.no_grad():
            for i, agent_id_label in enumerate(self.agent_ids):
                unscaled_actions = self.actor_targets[i](
                    next_states[agent_id_label]
                )
                if not self.discrete_actions:
                    scaled_actions = torch.where(
                        unscaled_actions > 0,
                        unscaled_actions * self.max_action[i][0],
                        unscaled_actions * -self.min_action[i][0],
                    )
                    next_actions.append(scaled_actions)
                else:
                    next_actions.append(unscaled_actions)

        # Stack values from different agents
        if self.arch == "mlp":
            action_values = list(actions.values())
            state_values = list(states.values())
            input_combined = torch.cat(state_values + action_values, dim=1)
            next_input_combined = torch.cat(
                list(next_states.values()) + next_actions, dim=1
            )
        elif self.arch == "cnn":
            next_states = {
                agent_id: next_state.squeeze(2) 
                for agent_id, next_state in next_states.items()
                }

            stacked_states = torch.stack(list(states.values()), dim=2)
            stacked_actions = torch.cat(list(actions.values()), dim=1)
            stacked_next_states = torch.stack(list(next_states.values()), dim=2)
            stacked_next_actions = torch.cat(next_actions, dim=1)
            
            # Need to unsqueeze states for Conv3d
            states = {
                agent_id: state.unsqueeze(2) for agent_id, state in states.items()
            }
        
        loss_dict = {}
        for idx, (
            agent_id,
            actor,
            critic,
            critic_target,
            actor_optimizer,
            critic_optimizer,
        ) in enumerate(
            zip(
                self.agent_ids,
                self.actors,
                self.critics,
                self.critic_targets,
                self.actor_optimizers,
                self.critic_optimizers,
            )
        ):
            loss_dict[f"{agent_id}"] = self._learn_individual(
                idx,
                agent_id,
                actor,
                critic,
                critic_target,
                actor_optimizer,
                critic_optimizer,
                input_combined if self.arch == "mlp" else None,
                stacked_states if self.arch == "cnn" else None,
                stacked_actions if self.arch == "cnn" else None,
                next_input_combined if self.arch == "mlp" else None,
                stacked_next_states if self.arch == "cnn" else None,
                stacked_next_actions if self.arch == "cnn" else None,
                states,
                actions,
                rewards,
                dones,
            )

        for actor, actor_target, critic, critic_target in zip(
            self.actors, self.actor_targets, self.critics, self.critic_targets
        ):
            self.soft_update(actor, actor_target)
            self.soft_update(critic, critic_target)

        return loss_dict

    def _learn_individual(
        self,
        idx: int,
        agent_id: str,
        actor: nn.Module,
        critic: nn.Module,
        critic_target: nn.Module,
        actor_optimizer: optim.Optimizer,
        critic_optimizer: optim.Optimizer,
        input_combined: Optional[torch.Tensor],
        stacked_states: Optional[torch.Tensor],
        stacked_actions: Optional[torch.Tensor],
        next_input_combined: Optional[torch.Tensor],
        stacked_next_states: Optional[torch.Tensor],
        stacked_next_actions: Optional[torch.Tensor],
        states: TensorDict,
        actions: TensorDict,
        rewards: TensorDict,
        dones: TensorDict,
    ) -> Tuple[float, float]:
        """
        Inner call to each agent for the learning/algo training steps, up until the soft updates.
        Applies all forward/backward props.

        :param idx: Index of the agent
        :type idx: int
        :param agent_id: ID of the agent
        :type agent_id: str
        :param actor: Actor network of the agent
        :type actor: nn.Module
        :param critic: Critic network of the agent
        :type critic: nn.Module
        :param critic_target: Target critic network of the agent
        :type critic_target: nn.Module
        :param actor_optimizer: Optimizer for the actor network
        :type actor_optimizer: optim.Optimizer
        :param critic_optimizer: Optimizer for the critic network
        :type critic_optimizer: optim.Optimizer
        :param input_combined: Combined input tensor for MLP architecture
        :type input_combined: Optional[torch.Tensor]
        :param stacked_states: Stacked states tensor for CNN architecture
        :type stacked_states: Optional[torch.Tensor]
        :param stacked_actions: Stacked actions tensor for CNN architecture
        :type stacked_actions: Optional[torch.Tensor]
        :param next_input_combined: Combined input tensor for next state in MLP architecture
        :type next_input_combined: Optional[torch.Tensor]
        :param stacked_next_states: Stacked next states tensor for CNN architecture
        :type stacked_next_states: Optional[torch.Tensor]
        :param stacked_next_actions: Stacked next actions tensor for CNN architecture
        :type stacked_next_actions: Optional[torch.Tensor]
        :param states: Dictionary of current states for each agent
        :type states: TensorDict
        :param actions: Dictionary of actions for each agent
        :type actions: TensorDict
        :param rewards: Dictionary of rewards for each agent
        :type rewards: TensorDict
        :param dones: Dictionary of done flags for each agent
        :type dones: TensorDict
        :return: Tuple containing actor loss and critic loss
        :rtype: Tuple[float, float]
        """
        if self.arch == "mlp":
            if self.accelerator is not None:
                with critic.no_sync():
                    q_value = critic(input_combined)
            else:
                q_value = critic(input_combined)
        elif self.arch == "cnn":
            if self.accelerator is not None:
                with critic.no_sync():
                    q_value = critic(stacked_states, stacked_actions)
            else:
                q_value = critic(stacked_states, stacked_actions)

        with torch.no_grad():
            if self.arch == "mlp":
                if self.accelerator is not None:
                    with critic_target.no_sync():
                        q_value_next_state = critic_target(next_input_combined)
                else:
                    q_value_next_state = critic_target(next_input_combined)
            elif self.arch == "cnn":
                if self.accelerator is not None:
                    with critic_target.no_sync():
                        q_value_next_state = critic_target(
                            stacked_next_states, stacked_next_actions
                        )
                else:
                    q_value_next_state = critic_target(
                        stacked_next_states, stacked_next_actions
                    )

        y_j = (
            rewards[agent_id] + (1 - dones[agent_id]) * self.gamma * q_value_next_state
        )

        critic_loss = self.criterion(q_value, y_j)

        # critic loss backprop
        critic_optimizer.zero_grad()
        if self.accelerator is not None:
            self.accelerator.backward(critic_loss)
        else:
            critic_loss.backward()
        critic_optimizer.step()

        # Get actions from actor
        if self.accelerator is not None:
            with actor.no_sync():
                action = actor(states[agent_id])
        else:
            action = actor(states[agent_id])
        if not self.discrete_actions:
            action = torch.where(
                action > 0,
                action * self.max_action[idx][0],
                action * -self.min_action[idx][0],
            )
        detached_actions = copy.deepcopy(actions)
        detached_actions[agent_id] = action

        # update actor and targets
        if self.arch == "mlp":
            input_combined = torch.cat(
                list(states.values()) + list(detached_actions.values()), 1
            )
            if self.accelerator is not None:
                with critic.no_sync():
                    actor_loss = -critic(input_combined).mean()
            else:
                actor_loss = -critic(input_combined).mean()

        elif self.arch == "cnn":
            stacked_detached_actions = torch.cat(list(detached_actions.values()), dim=1)
            if self.accelerator is not None:
                with critic.no_sync():
                    actor_loss = -critic(
                        stacked_states, stacked_detached_actions
                    ).mean()
            else:
                actor_loss = -critic(stacked_states, stacked_detached_actions).mean()

        # actor loss backprop
        actor_optimizer.zero_grad()
        if self.accelerator is not None:
            self.accelerator.backward(actor_loss)
        else:
            actor_loss.backward()
        actor_optimizer.step()

        return actor_loss.item(), critic_loss.item()

    def soft_update(self, net, target):
        """Soft updates target network."""
        for eval_param, target_param in zip(net.parameters(), target.parameters()):
            target_param.data.copy_(
                self.tau * eval_param.data + (1.0 - self.tau) * target_param.data
            )

    def test(self, env, swap_channels=False, max_steps=None, loop=3, sum_scores=True):
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
        with torch.no_grad():
            rewards = []
            if hasattr(env, "num_envs"):
                num_envs = env.num_envs
                is_vectorised = True
            else:
                num_envs = 1
                is_vectorised = False

            for i in range(loop):
                state, info = env.reset()
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
                        if is_vectorised:
                            state = {
                                agent_id: np.moveaxis(s, [-1], [-3])
                                for agent_id, s in state.items()
                            }
                        else:
                            state = {
                                agent_id: np.moveaxis(np.expand_dims(s, 0), [-1], [-3])
                                for agent_id, s in state.items()
                            }
                    cont_actions, discrete_action = self.get_action(
                        state,
                        training=False,
                        infos=info,
                    )
                    if self.discrete_actions:
                        action = discrete_action
                    else:
                        action = cont_actions
                    if not is_vectorised:
                        action = {agent: act[0] for agent, act in action.items()}
                    state, reward, term, trunc, info = env.step(action)
                    score_increment = (
                        (
                            np.sum(
                                np.array(list(reward.values())).transpose(), axis=-1
                            )[:, np.newaxis]
                            if is_vectorised
                            else np.sum(
                                np.array(list(reward.values())).transpose(), axis=-1
                            )
                        )
                        if sum_scores
                        else np.array(list(reward.values())).transpose()
                    )
                    scores += score_increment
                    dones = {
                        agent: term[agent] | trunc[agent] for agent in self.agent_ids
                    }
                    if not is_vectorised:
                        dones = {
                            agent: np.array([done]) for agent, done in dones.items()
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

