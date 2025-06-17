import copy
import warnings
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from gymnasium import spaces
from torch.nn.utils import clip_grad_norm_

from agilerl.algorithms.core import MultiAgentRLAlgorithm, OptimizerWrapper
from agilerl.algorithms.core.registry import HyperparameterConfig, NetworkGroup
from agilerl.modules import EvolvableModule, ModuleDict
from agilerl.modules.configs import MlpNetConfig
from agilerl.networks.actors import StochasticActor
from agilerl.networks.value_networks import ValueNetwork
from agilerl.typing import (
    ArrayDict,
    ExperiencesType,
    InfosDict,
    MultiAgentModule,
    NetConfigType,
    ObservationType,
    PzEnvType,
    StandardTensorDict,
    TorchObsType,
)
from agilerl.utils.algo_utils import (
    concatenate_experiences_into_batches,
    concatenate_tensors,
    get_experiences_samples,
    get_vect_dim,
    key_in_nested_dict,
    make_safe_deepcopies,
    obs_channels_to_first,
    preprocess_observation,
    vectorize_experiences_by_agent,
)


class IPPO(MultiAgentRLAlgorithm):
    """Independent Proximal Policy Optimization (IPPO) algorithm.

    Paper: https://arxiv.org/pdf/2011.09533

    :param observation_spaces: Observation space for each agent
    :type observation_spaces: Union[list[spaces.Space], spaces.Dict]
    :param action_spaces: Action space for each agent
    :type action_spaces: Union[list[spaces.Space], spaces.Dict]
    :param agent_ids: Agent ID for each agent
    :type agent_ids: Optional[list[str]], optional
    :param index: Index to keep track of object instance during tournament selection and mutation, defaults to 0
    :type index: int, optional
    :param hp_config: RL hyperparameter mutation configuration, defaults to None, whereby algorithm mutations are disabled.
    :type hp_config: HyperparameterConfig, optional
    :param net_config: Network configuration, defaults to None
    :type net_config: dict, optional
    :param batch_size: Size of batched sample from replay buffer for learning, defaults to 64
    :type batch_size: int, optional
    :param lr: Learning rate for optimizer, defaults to 1e-4
    :type lr: float, optional
    :param learn_step: Learning frequency, defaults to 2048
    :type learn_step: int, optional
    :param gamma: Discount factor, defaults to 0.99
    :type gamma: float, optional
    :param gae_lambda: Lambda for general advantage estimation, defaults to 0.95
    :type gae_lambda: float, optional
    :param mut: Most recent mutation to agent, defaults to None
    :type mut: str, optional
    :param action_std_init: Initial action standard deviation, defaults to 0.0
    :type action_std_init: float, optional
    :param clip_coef: Surrogate clipping coefficient, defaults to 0.2
    :type clip_coef: float, optional
    :param ent_coef: Entropy coefficient, defaults to 0.01
    :type ent_coef: float, optional
    :param vf_coef: Value function coefficient, defaults to 0.5
    :type vf_coef: float, optional
    :param max_grad_norm: Maximum norm for gradient clipping, defaults to 0.5
    :type max_grad_norm: float, optional
    :param target_kl: Target KL divergence threshold, defaults to None
    :type target_kl: float, optional
    :param normalize_images: Flag to normalize images, defaults to True
    :type normalize_images: bool, optional
    :param update_epochs: Number of policy update epochs, defaults to 4
    :type update_epochs: int, optional
    :param actor_networks: List of custom actor networks, defaults to None
    :type actor_networks: agilerl.modules.ModuleDict, optional
    :param critic_networks: List of custom critic networks, defaults to None
    :type critic_networks: agilerl.modules.ModuleDict, optional
    :param action_batch_size: Size of batches to use when getting an action for stepping in the environment.
        Defaults to None, whereby the entire observation is used at once.
    :type action_batch_size: int, optional
    :param device: Device for accelerated computing, 'cpu' or 'cuda', defaults to 'cpu'
    :type device: str, optional
    :param accelerator: Accelerator for distributed computing, defaults to None
    :type accelerator: accelerate.Accelerator(), optional
    :param torch_compiler: The torch compile mode 'default', 'reduce-overhead' or 'max-autotune', defaults to None
    :type torch_compiler: str, optional
    :param wrap: Wrap models for distributed training upon creation, defaults to True
    :type wrap: bool, optional
    """

    actors: MultiAgentModule[StochasticActor]
    critics: MultiAgentModule[ValueNetwork]

    def __init__(
        self,
        observation_spaces: Union[List[spaces.Space], spaces.Dict],
        action_spaces: Union[List[spaces.Space], spaces.Dict],
        agent_ids: Optional[List[str]] = None,
        index: int = 0,
        hp_config: Optional[HyperparameterConfig] = None,
        net_config: Optional[Dict[str, Any]] = None,
        batch_size: int = 64,
        lr: float = 1e-4,
        learn_step: int = 2048,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        mut: Optional[str] = None,
        action_std_init: float = 0.0,
        clip_coef: float = 0.2,
        ent_coef: float = 0.01,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        target_kl: Optional[float] = None,
        normalize_images: bool = True,
        update_epochs: int = 4,
        actor_networks: Optional[ModuleDict] = None,
        critic_networks: Optional[ModuleDict] = None,
        action_batch_size: Optional[int] = None,
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
            torch_compiler=torch_compiler,
            normalize_images=normalize_images,
            placeholder_value=None,
            name="IPPO",
        )

        assert learn_step >= 1, "Learn step must be greater than or equal to one."
        assert isinstance(learn_step, int), "Learn step rate must be an integer."
        assert isinstance(batch_size, int), "Batch size must be an integer."
        assert batch_size >= 1, "Batch size must be greater than or equal to one."
        assert isinstance(lr, float), "Learning rate must be a float."
        assert lr > 0, "Learning rate must be greater than zero."
        assert isinstance(gamma, (float, int, torch.Tensor)), "Gamma must be a float."
        assert isinstance(gae_lambda, (float, int)), "Lambda must be a float."
        assert gae_lambda >= 0, "Lambda must be greater than or equal to zero."
        assert isinstance(
            action_std_init, (float, int)
        ), "Action standard deviation must be a float."
        assert (
            action_std_init >= 0
        ), "Action standard deviation must be greater than or equal to zero."
        assert isinstance(
            clip_coef, (float, int)
        ), "Clipping coefficient must be a float."
        assert (
            clip_coef >= 0
        ), "Clipping coefficient must be greater than or equal to zero."
        assert isinstance(
            ent_coef, (float, int)
        ), "Entropy coefficient must be a float."
        assert (
            ent_coef >= 0
        ), "Entropy coefficient must be greater than or equal to zero."
        assert isinstance(
            vf_coef, (float, int)
        ), "Value function coefficient must be a float."
        assert (
            vf_coef >= 0
        ), "Value function coefficient must be greater than or equal to zero."
        assert isinstance(
            max_grad_norm, (float, int)
        ), "Maximum norm for gradient clipping must be a float."
        assert (
            max_grad_norm >= 0
        ), "Maximum norm for gradient clipping must be greater than or equal to zero."
        assert (
            isinstance(target_kl, (float, int)) or target_kl is None
        ), "Target KL divergence threshold must be a float."
        if target_kl is not None:
            assert (
                target_kl >= 0
            ), "Target KL divergence threshold must be greater than or equal to zero."
        assert isinstance(
            update_epochs, int
        ), "Policy update epochs must be an integer."
        assert (
            update_epochs >= 1
        ), "Policy update epochs must be greater than or equal to one."
        assert isinstance(
            wrap, bool
        ), "Wrap models flag must be boolean value True or False."
        if (actor_networks is not None) != (critic_networks is not None):
            warnings.warn(
                "Actor and critic network lists must both be supplied to use custom networks. Defaulting to net config."
            )

        self.batch_size = batch_size
        self.lr = lr
        self.gamma = gamma
        self.learn_step = learn_step
        self.mut = mut
        self.gae_lambda = gae_lambda
        self.action_std_init = action_std_init
        self.net_config = net_config
        self.clip_coef = clip_coef
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.target_kl = target_kl
        self.update_epochs = update_epochs
        self.action_batch_size = action_batch_size

        if actor_networks is not None and critic_networks is not None:
            if isinstance(actor_networks, list):
                assert len(actor_networks) == len(
                    self.observation_space
                ), "actor_networks must be a list of the same length as the number of homogeneous agents"
                actor_networks = ModuleDict(
                    {
                        agent_id: actor_networks[idx]
                        for idx, agent_id in enumerate(self.observation_space)
                    }
                )
            if isinstance(critic_networks, list):
                assert len(critic_networks) == len(
                    self.observation_space
                ), "critic_networks must be a list of the same length as the number of homogeneous agents"

                critic_networks = ModuleDict(
                    {
                        agent_id: critic_networks[idx]
                        for idx, agent_id in enumerate(self.observation_space)
                    }
                )

            actors_list = list(actor_networks.values())
            critics_list = list(critic_networks.values())
            if not all(isinstance(net, EvolvableModule) for net in actors_list):
                raise TypeError(
                    "All actor networks must be instances of EvolvableModule"
                )
            if not all(isinstance(net, EvolvableModule) for net in critics_list):
                raise TypeError(
                    "All critic networks must be instances of EvolvableModule"
                )

            assert len(actor_networks) == self.n_unique_agents, (
                f"Length of actor_networks ({len(actor_networks)}) does not match number of unique "
                f"agents defined in environment ({self.n_unique_agents}: {list(self.observation_space.keys())})"
            )
            assert len(critic_networks) == self.n_unique_agents, (
                f"Length of critic_networks ({len(critic_networks)}) does not match number of unique "
                f"agents defined in environment ({self.n_unique_agents}: {list(self.observation_space.keys())})"
            )

            self.actors, self.critics = make_safe_deepcopies(
                actor_networks, critic_networks
            )
        else:
            net_config: NetConfigType = self.build_net_config(net_config, flatten=False)

            self.actors = ModuleDict()
            self.critics = ModuleDict()
            for agent_id in self.observation_space:
                obs_space = self.observation_space[agent_id]
                action_space = self.action_space[agent_id]

                agent_config = net_config[agent_id]
                critic_net_config = copy.deepcopy(agent_config)
                head_config = agent_config.get("head_config", None)
                if head_config is not None:
                    critic_head_config = copy.deepcopy(head_config)
                    critic_head_config["output_activation"] = None
                else:
                    critic_head_config = MlpNetConfig(hidden_size=[64])

                critic_net_config["head_config"] = critic_head_config

                # Create one actor and critic per group of homogeneous agents,
                # which will be used by all homogeneous (identical) agents of that group
                actor = StochasticActor(
                    obs_space,
                    action_space,
                    action_std_init=self.action_std_init,
                    device=self.device,
                    **copy.deepcopy(agent_config),
                )

                critic = ValueNetwork(
                    observation_space=obs_space,
                    device=self.device,
                    **copy.deepcopy(critic_net_config),
                )

                self.actors[agent_id] = actor
                self.critics[agent_id] = critic

        # Optimizers
        self.actor_optimizers = OptimizerWrapper(
            optim.Adam, networks=self.actors, lr=self.lr
        )
        self.critic_optimizers = OptimizerWrapper(
            optim.Adam, networks=self.critics, lr=self.lr
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
                policy=True,
            )
        )
        self.register_network_group(
            NetworkGroup(
                eval_network=self.critics,
            )
        )

    def process_infos(
        self, infos: Optional[InfosDict]
    ) -> Tuple[ArrayDict, ArrayDict, ArrayDict]:
        """
        Process the information, extract env_defined_actions, action_masks and agent_masks

        :param infos: Info dict
        :type infos: Dict[str, Dict[...]]
        :return: Tuple of action_masks, env_defined_actions, agent_masks
        :rtype: Tuple[ArrayDict, ArrayDict, ArrayDict]
        """
        if infos is None:
            infos = {agent: {} for agent in self.agent_ids}
            action_masks = {agent: None for agent in self.observation_space}
        else:
            action_masks = self.extract_action_masks(infos)

        env_defined_actions, agent_masks = self.extract_agent_masks(infos)

        return action_masks, env_defined_actions, agent_masks

    def extract_action_masks(self, infos: InfosDict) -> ArrayDict:
        """Extract action masks from info dictionary

        :param infos: Info dict
        :type infos: Dict[str, Dict[...]]

        :return: Action masks
        :rtype: Dict[str, np.ndarray]
        """
        # Get dict of form {"agent_id" : [1, 0, 0, 0]...} etc
        action_masks = {group_id: [] for group_id in self.observation_space}
        for agent_id, info in infos.items():
            if isinstance(info, dict):
                agent_id = (
                    self.get_group_id(agent_id)
                    if self.has_grouped_agents()
                    else agent_id
                )
                action_masks[agent_id].append(
                    info.get("action_mask", None) if isinstance(info, dict) else None
                )

        # Check and stack masks
        for group_id in self.observation_space:
            if None in action_masks[group_id] or not action_masks[group_id]:
                assert all(mask is None for mask in action_masks[group_id]), (
                    f"If action masks are provided for any agents, they must be provided for all agents. "
                    "Action masks can be defined as an array with the shape of the action space "
                    f"({self.action_space}), where 1=legal and 0=illegal."
                )

                action_masks[group_id] = None
            else:
                action_masks[group_id] = torch.Tensor(action_masks[group_id])

        return action_masks

    def preprocess_observation(
        self, observation: ObservationType, group_ids: List[str]
    ) -> Dict[str, TorchObsType]:
        """Preprocesses observations for forward pass through neural network.

        :param observation: Observations of environment
        :type observation: numpy.ndarray[float] or dict[str, numpy.ndarray[float]]
        :param group_ids: List of group IDs
        :type group_ids: List[str]

        :return: Preprocessed observations
        :rtype: torch.Tensor[float] or dict[str, torch.Tensor[float]] or Tuple[torch.Tensor[float], ...]
        """
        preprocessed = {group_id: [] for group_id in group_ids}
        for agent_id, agent_obs in observation.items():
            group_id = (
                self.get_group_id(agent_id) if self.has_grouped_agents() else agent_id
            )
            preprocessed[group_id].append(
                preprocess_observation(
                    self.observation_space.get(group_id),
                    observation=agent_obs,
                    device=self.device,
                    normalize_images=self.normalize_images,
                )
            )

        # Need to concatenate / stack observations for each group of homogeneous agents
        for group_id in group_ids:
            if not preprocessed[group_id]:
                continue

            preprocessed[group_id] = concatenate_tensors(preprocessed[group_id])

        return preprocessed

    def _get_action_and_values(
        self,
        obs: TorchObsType,
        actor: StochasticActor,
        critic: ValueNetwork,
        action_mask: Optional[torch.Tensor] = None,
        batch_size: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get actions and values for a batch of grouped observations.

        :param obs: Observations of environment
        :type obs: torch.Tensor[float] or dict[str, torch.Tensor[float]] or Tuple[torch.Tensor[float], ...]
        :param actor: Actor network
        :type actor: StochasticActor
        :param critic: Critic network
        :type critic: ValueNetwork
        :param action_mask: Action mask
        :type action_mask: torch.Tensor, optional
        :param batch_size: Batch size
        :type batch_size: int, optional
        :return: Tuple of actions, log probabilities, entropies, values
        :rtype: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        """
        # Process in batches
        if batch_size is not None and obs.shape[0] > batch_size:
            actions = []
            log_probs = []
            entropies = []
            values = []

            num_batches = int(np.ceil(obs.shape[0] / batch_size))
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, obs.shape[0])

                minibatch_indices = np.arange(start_idx, end_idx)
                batch_obs = get_experiences_samples(minibatch_indices, obs)[0]
                batch_mask = None
                if action_mask is not None:
                    batch_mask = action_mask[minibatch_indices]

                batch_action, batch_log_prob, batch_entropy = actor(
                    batch_obs, action_mask=batch_mask
                )
                batch_state_values = critic(batch_obs).squeeze(-1)

                actions.append(batch_action)
                log_probs.append(batch_log_prob)
                entropies.append(batch_entropy)
                values.append(batch_state_values)

            # Concatenate results
            action = torch.cat(actions)
            log_prob = torch.cat(log_probs)
            entropy = torch.cat(entropies)
            values = torch.cat(values)
        else:
            with torch.no_grad():
                action, log_prob, entropy = actor(obs, action_mask=action_mask)
                values = critic(obs).squeeze(-1)

        return action, log_prob, entropy, values

    def get_action(
        self,
        obs: Dict[str, ObservationType],
        infos: Optional[InfosDict] = None,
    ) -> Tuple[ArrayDict, ArrayDict, ArrayDict, ArrayDict]:
        """Returns the next action to take in the environment.

        :param obs: Environment observations: {'agent_0': state_dim_0, ..., 'agent_n': state_dim_n}
        :type obs: Dict[str, numpy.Array | Dict[str, numpy.Array] | Tuple[numpy.Array, ...]]
        :param infos: Information dictionary returned by env.step(actions)
        :type infos: Dict[str, Dict[str, ...]]
        :return: Tuple of actions, log probabilities, entropies, values
        :rtype: Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray]]
        """
        assert not key_in_nested_dict(
            obs, "action_mask"
        ), "AgileRL requires action masks to be defined in the information dictionary."

        action_masks, env_defined_actions, agent_masks = self.process_infos(infos)
        vect_dim = get_vect_dim(obs, self.possible_observation_spaces)

        # Groups to extract actions from in observation
        unique_agents_ids = list(obs.keys())
        grouped_agents = defaultdict(list)
        for agent_id in unique_agents_ids:
            group_id = (
                self.get_group_id(agent_id) if self.has_grouped_agents() else agent_id
            )
            grouped_agents[group_id].append(agent_id)

        # Preprocess observations
        preprocessed = self.preprocess_observation(obs, list(grouped_agents.keys()))

        action_dict = {}
        action_logprob_dict = {}
        dist_entropy_dict = {}
        state_values_dict = {}
        for agent_id, obs in preprocessed.items():
            action_mask = action_masks[agent_id]
            actor = self.actors[agent_id]
            critic = self.critics[agent_id]

            with torch.no_grad():
                action, log_prob, entropy, values = self._get_action_and_values(
                    obs=obs,
                    actor=actor,
                    critic=critic,
                    action_mask=action_mask,
                    batch_size=self.action_batch_size,
                )

            # Clip to action space during inference
            agent_space = self.action_space[agent_id]
            action = action.cpu().data.numpy()
            if not self.training and isinstance(agent_space, spaces.Box):
                if actor.squash_output:
                    action = actor.scale_action(action)
                else:
                    action = np.clip(action, agent_space.low, agent_space.high)

            action_dict[agent_id] = action
            action_logprob_dict[agent_id] = log_prob.cpu().data.numpy()
            dist_entropy_dict[agent_id] = entropy.cpu().data.numpy()
            state_values_dict[agent_id] = values.cpu().data.numpy()

        action_dict = self.disassemble_grouped_outputs(
            action_dict, vect_dim, grouped_agents
        )

        # If using env_defined_actions replace actions
        if env_defined_actions is not None:
            for agent_id in unique_agents_ids:
                action_dict[agent_id][agent_masks[agent_id]] = env_defined_actions[
                    agent_id
                ][agent_masks[agent_id]]

        return (
            action_dict,
            self.disassemble_grouped_outputs(
                action_logprob_dict, vect_dim, grouped_agents
            ),
            self.disassemble_grouped_outputs(
                dist_entropy_dict, vect_dim, grouped_agents
            ),
            self.disassemble_grouped_outputs(
                state_values_dict, vect_dim, grouped_agents
            ),
        )

    def learn(self, experiences: ExperiencesType) -> StandardTensorDict:
        """Updates agent network parameters to learn from experiences.

        :param experiences: Tuple of dictionaries containing batched states, actions,
            rewards, next_states, dones in that order for each individual agent.
        :type experiences: Tuple[Dict[str, torch.Tensor]]

        :return: Loss dictionary
        :rtype: Dict[str, torch.Tensor]
        """
        # Process experiences
        states, actions, log_probs, rewards, dones, values, next_states, next_dones = (
            map(self.assemble_shared_inputs, experiences)
        )

        loss_dict = {}
        for agent_id, state in states.items():
            actor = self.actors[agent_id]
            critic = self.critics[agent_id]
            actor_optimizer = self.actor_optimizers[agent_id]
            critic_optimizer = self.critic_optimizers[agent_id]
            obs_space = self.observation_space[agent_id]
            action_space = self.action_space[agent_id]

            loss_dict[f"{agent_id}"] = self._learn_individual(
                experiences=(
                    state,
                    actions[agent_id],
                    log_probs[agent_id],
                    rewards[agent_id],
                    dones[agent_id],
                    values[agent_id],
                    next_states[agent_id],
                    next_dones[agent_id],
                ),
                actor=actor,
                critic=critic,
                actor_optimizer=actor_optimizer,
                critic_optimizer=critic_optimizer,
                obs_space=obs_space,
                action_space=action_space,
            )

        return loss_dict

    def _learn_individual(
        self,
        experiences: ExperiencesType,
        actor: Union[EvolvableModule, StochasticActor],
        critic: Union[EvolvableModule, ValueNetwork],
        actor_optimizer: OptimizerWrapper,
        critic_optimizer: OptimizerWrapper,
        obs_space: spaces,
        action_space: spaces,
    ) -> float:
        """Inner call to each agent for the learning/algo training steps,
        essentially the PPO learn method. Applies all forward/backward props.

        :param experience: States, actions, log_probs, rewards, dones, values, next_state, next_done in
            that order, organised by shared agent id
        :type experience: Tuple[Union[numpy.ndarray, Dict[str, numpy.ndarray]], ...]
        :param actor: Actor network
        :type actor: EvolvableModule
        :param critic: Critic network
        :type critic: EvolvableModule
        :param actor_optimizer: Optimizer specific to the actor
        :type actor_optimizer: OptimizerWrapper
        :param critic_optimizer: Optimizer specific to the critic
        :type critic_optimzer: OptimizerWrapper
        :param obs_space: Observation space for the agent
        :type obs_space: gymnasium.spaces
        :param action_space: Action space for the agent
        :type action_space: gymnasium.spaces
        """
        (states, actions, log_probs, rewards, dones, values, next_state, next_done) = (
            experiences
        )

        log_probs, rewards, dones, values = map(
            vectorize_experiences_by_agent, (log_probs, rewards, dones, values)
        )
        log_probs = log_probs.squeeze()
        rewards = rewards.squeeze()
        dones = dones.squeeze()
        values = values.squeeze()
        next_state = vectorize_experiences_by_agent(next_state, dim=0)
        next_done = vectorize_experiences_by_agent(next_done, dim=0)

        with torch.no_grad():
            num_steps = rewards.size(0)
            rewards = rewards.reshape(num_steps, -1)
            dones = dones.reshape(num_steps, -1)
            values = values.reshape(num_steps, -1)
            next_done = next_done.reshape(1, -1)

            next_state = preprocess_observation(
                obs_space, next_state, self.device, self.normalize_images
            )
            next_value = critic(next_state).reshape(1, -1).cpu()
            advantages = torch.zeros_like(rewards).float()
            last_gae_lambda = 0
            for t in reversed(range(num_steps)):
                if t == num_steps - 1:
                    next_non_terminal = 1.0 - next_done
                    nextvalue = next_value.squeeze()
                else:
                    next_non_terminal = 1.0 - dones[t + 1]
                    nextvalue = values[t + 1]

                # Calculate delta (TD error)
                delta = (
                    rewards[t] + self.gamma * nextvalue * next_non_terminal - values[t]
                )

                # Use recurrence relation to compute advantage
                advantages[t] = last_gae_lambda = (
                    delta
                    + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lambda
                )

            advantages = advantages.reshape((-1,))
            values = values.reshape((-1,))
            returns = advantages + values

        states = concatenate_experiences_into_batches(states, obs_space)
        actions = concatenate_experiences_into_batches(actions, action_space)
        log_probs = log_probs.reshape((-1,))
        experiences = (states, actions, log_probs, advantages, returns, values)

        # Move experiences to algo device
        experiences = self.to_device(*experiences)

        num_samples = experiences[4].size(0)
        batch_idxs = np.arange(num_samples)
        mean_loss = 0
        for _ in range(self.update_epochs):
            np.random.shuffle(batch_idxs)
            for start in range(0, num_samples, self.batch_size):
                minibatch_idxs = batch_idxs[start : start + self.batch_size]
                (
                    batch_states,
                    batch_actions,
                    batch_log_probs,
                    batch_advantages,
                    batch_returns,
                    batch_values,
                ) = get_experiences_samples(minibatch_idxs, *experiences)

                batch_actions = batch_actions.squeeze()
                batch_returns = batch_returns.squeeze()
                batch_log_probs = batch_log_probs.squeeze()
                batch_advantages = batch_advantages.squeeze()
                batch_values = batch_values.squeeze()

                if len(minibatch_idxs) > 1:
                    batch_states = preprocess_observation(
                        obs_space, batch_states, self.device, self.normalize_images
                    )
                    _, _, entropy = actor(batch_states)
                    value = critic(batch_states).squeeze(-1)

                    log_prob = actor.action_log_prob(batch_actions)

                    if isinstance(action_space, spaces.Box) and action_space.shape == (
                        1,
                    ):
                        batch_actions = batch_actions.unsqueeze(1)

                    logratio = log_prob - batch_log_probs
                    ratio = logratio.exp()

                    with torch.no_grad():
                        approx_kl = ((ratio - 1) - logratio).mean()

                    minibatch_advs = batch_advantages
                    minibatch_advs = (minibatch_advs - minibatch_advs.mean()) / (
                        minibatch_advs.std() + 1e-8
                    )

                    # Policy loss
                    pg_loss1 = -minibatch_advs * ratio
                    pg_loss2 = -minibatch_advs * torch.clamp(
                        ratio, 1 - self.clip_coef, 1 + self.clip_coef
                    )
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # Value loss
                    value = value.view(-1)
                    v_loss_unclipped = (value - batch_returns) ** 2
                    v_clipped = batch_values + torch.clamp(
                        value - batch_values, -self.clip_coef, self.clip_coef
                    )

                    v_loss_clipped = (v_clipped - batch_returns) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()

                    entropy_loss = entropy.mean()

                    actor_loss = pg_loss - self.ent_coef * entropy_loss
                    critic_loss = v_loss * self.vf_coef

                    # loss backprop
                    actor_optimizer.zero_grad()
                    if self.accelerator is not None:
                        self.accelerator.backward(actor_loss)
                    else:
                        actor_loss.backward()

                    clip_grad_norm_(actor.parameters(), self.max_grad_norm)
                    actor_optimizer.step()

                    critic_optimizer.zero_grad()
                    if self.accelerator is not None:
                        self.accelerator.backward(critic_loss)
                    else:
                        critic_loss.backward()
                    clip_grad_norm_(critic.parameters(), self.max_grad_norm)
                    critic_optimizer.step()

                    mean_loss += actor_loss.item() + critic_loss.item()

            if self.target_kl is not None:
                if approx_kl > self.target_kl:
                    break

        mean_loss /= num_samples * self.update_epochs
        return mean_loss

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
        :type env: PettingZoo environment
        :param swap_channels: Swap image channels dimension from last to first [H, W, C] -> [C, H, W], defaults to False
        :type swap_channels: bool, optional
        :param max_steps: Maximum number of testing steps, defaults to None
        :type max_steps: int, optional
        :param loop: Number of testing loops/episodes to complete. The returned score is the mean. Defaults to 3
        :type loop: int, optional
        :param sum_scores: Boolean flag to indicate whether to sum sub-agent scores, defaults to True
        :type sum_scores: bool, optional
        :return: Mean test score
        :rtype: float
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
                    else np.zeros((num_envs, len(self.observation_space)))
                )
                completed_episode_scores = (
                    np.zeros((num_envs, 1))
                    if sum_scores
                    else np.zeros((num_envs, len(self.observation_space)))
                )
                finished = np.zeros(num_envs)
                step = 0
                while not np.all(finished):
                    step += 1
                    if swap_channels:
                        obs = {
                            agent_id: obs_channels_to_first(s)
                            for agent_id, s in obs.items()
                        }

                    # Get next action from agent
                    action, _, _, _ = self.get_action(obs=obs, infos=info)

                    if not is_vectorised:
                        action = {agent: act[0] for agent, act in action.items()}

                    obs, reward, term, trunc, info = env.step(action)
                    reward = self.sum_shared_rewards(reward)

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
