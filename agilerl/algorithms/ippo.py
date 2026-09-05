# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

import copy
import warnings
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import replace
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
from gymnasium import spaces
from pettingzoo import ParallelEnv
from torch import nn, optim
from torch.nn.utils import clip_grad_norm_
from torch.optim import Optimizer

from agilerl.algorithms.configs import (
    AlgorithmRuntime,
    IPPOAgentSetup,
    IPPONetworkSetup,
    PopulationIndex,
    PPOLearnConfig,
)
from agilerl.algorithms.core import MultiAgentRLAlgorithm, OptimizerWrapper
from agilerl.algorithms.core.registry import (
    NetworkGroup,
    make_default_hp_config,
)
from agilerl.modules import EvolvableModule, ModuleDict
from agilerl.modules.configs import MlpNetConfig
from agilerl.networks.actors import StochasticActor
from agilerl.networks.value_networks import ValueNetwork
from agilerl.typing import (
    ArrayDict,
    InfosDict,
    IPPOActionMasks,
    IPPOProcessInfosReturn,
    MaybeActionMask,
    ObservationType,
    SupportedObservationSpace,
    TorchObsType,
    coerce_action_mask,
)
from agilerl.utils.algo_utils import (
    apply_env_defined_actions,
    concatenate_experiences_into_batches,
    configure_tf32_precision,
    get_experiences_samples,
    get_num_envs,
    get_vect_dim,
    key_in_nested_dict,
    make_safe_deepcopies,
    vectorize_agent_experiences_flat,
    vectorize_experiences_by_agent,
)
from agilerl.utils.algo_utils import (
    preprocess_observation as preprocess_observation_fn,
)
from agilerl.vector.pz_vec_env import PettingZooVecEnv


class IPPO(MultiAgentRLAlgorithm[tuple[Mapping[str, Any], ...]]):
    """Independent Proximal Policy Optimization (IPPO).

    Paper: https://arxiv.org/pdf/2011.09533

    :param observation_spaces: Observation space for each agent
    :type observation_spaces: list[SupportedObservationSpace] | spaces.Dict
    :param action_spaces: Action space for each agent
    :type action_spaces: list[spaces.Space] | spaces.Dict
    :param agent_ids: Agent ID for each agent
    :type agent_ids: list[str] | None, optional
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
    :type actor_networks: list[EvolvableModule] | ModuleDict | None, optional
    :param critic_networks: List of custom critic networks, defaults to None
    :type critic_networks: list[EvolvableModule] | ModuleDict | None, optional
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

    # Values are StochasticActor/ValueNetwork instances; torch.compile and
    # Accelerator wrappers proxy the same interface at runtime.
    actors: ModuleDict[StochasticActor]
    critics: ModuleDict[ValueNetwork]

    def __init__(
        self,
        observation_spaces: list[SupportedObservationSpace] | spaces.Dict,
        action_spaces: list[spaces.Space] | spaces.Dict,
        member: PopulationIndex | None = None,
        learn: PPOLearnConfig | None = None,
        network: IPPONetworkSetup | None = None,
        agents: IPPOAgentSetup | None = None,
        runtime: AlgorithmRuntime | None = None,
    ) -> None:
        member = member or PopulationIndex()
        learn = learn or PPOLearnConfig()
        network = network or IPPONetworkSetup()
        agents = agents or IPPOAgentSetup()
        runtime = runtime or AlgorithmRuntime()
        mut = member.mut
        batch_size = learn.batch_size
        lr = learn.lr
        learn_step = learn.learn_step
        gamma = learn.gamma
        gae_lambda = learn.gae_lambda
        clip_coef = learn.clip_coef
        ent_coef = learn.ent_coef
        vf_coef = learn.vf_coef
        max_grad_norm = learn.max_grad_norm
        target_kl = learn.target_kl
        update_epochs = learn.update_epochs
        net_config = network.net_config
        actor_networks = network.actor_networks
        critic_networks = network.critic_networks
        action_std_init = network.action_std_init
        action_batch_size = network.action_batch_size
        wrap = runtime.wrap

        super().__init__(
            observation_spaces,
            action_spaces,
            member=member,
            agents=agents,
            runtime=replace(runtime, name=runtime.name or "IPPO"),
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
            action_std_init,
            (float, int),
        ), "Action standard deviation must be a float."
        assert action_std_init >= 0, (
            "Action standard deviation must be greater than or equal to zero."
        )
        assert isinstance(
            clip_coef,
            (float, int),
        ), "Clipping coefficient must be a float."
        assert clip_coef >= 0, (
            "Clipping coefficient must be greater than or equal to zero."
        )
        assert isinstance(
            ent_coef,
            (float, int),
        ), "Entropy coefficient must be a float."
        assert ent_coef >= 0, (
            "Entropy coefficient must be greater than or equal to zero."
        )
        assert isinstance(
            vf_coef,
            (float, int),
        ), "Value function coefficient must be a float."
        assert vf_coef >= 0, (
            "Value function coefficient must be greater than or equal to zero."
        )
        assert isinstance(
            max_grad_norm,
            (float, int),
        ), "Maximum norm for gradient clipping must be a float."
        assert max_grad_norm >= 0, (
            "Maximum norm for gradient clipping must be greater than or equal to zero."
        )
        assert isinstance(target_kl, (float, int)) or target_kl is None, (
            "Target KL divergence threshold must be a float."
        )
        if target_kl is not None:
            assert target_kl >= 0, (
                "Target KL divergence threshold must be greater than or equal to zero."
            )
        assert isinstance(
            update_epochs,
            int,
        ), "Policy update epochs must be an integer."
        assert update_epochs >= 1, (
            "Policy update epochs must be greater than or equal to one."
        )
        assert isinstance(
            wrap,
            bool,
        ), "Wrap models flag must be boolean value True or False."
        if (actor_networks is not None) != (critic_networks is not None):
            warnings.warn(
                "Actor and critic network lists must both be supplied to use custom networks. Defaulting to net config.",
                stacklevel=2,
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

        # Default RL hyperparameters to mutate when doing Evo-HPO
        self.hp_config = self.hp_config or make_default_hp_config(
            lr=self.lr,
            batch_size=self.batch_size,
            learn_step=self.learn_step,
        )

        if actor_networks is not None and critic_networks is not None:
            if isinstance(actor_networks, list):
                assert len(actor_networks) == len(
                    self.observation_space,
                ), (
                    "actor_networks must be a list of the same length as the number of homogeneous agents"
                )
                # isinstance(list) leaves a list & ModuleDict intersection; pin it.
                actor_list = actor_networks
                actor_networks = ModuleDict(
                    {
                        agent_id: actor_list[idx]
                        for idx, agent_id in enumerate(self.observation_space)
                    },
                )
            if isinstance(critic_networks, list):
                assert len(critic_networks) == len(
                    self.observation_space,
                ), (
                    "critic_networks must be a list of the same length as the number of homogeneous agents"
                )

                # isinstance(list) leaves a list & ModuleDict intersection; pin it.
                critic_list = critic_networks
                critic_networks = ModuleDict(
                    {
                        agent_id: critic_list[idx]
                        for idx, agent_id in enumerate(self.observation_space)
                    },
                )

            actors_list = list(actor_networks.values())
            critics_list = list(critic_networks.values())
            if not all(isinstance(net, EvolvableModule) for net in actors_list):
                msg = "All actor networks must be instances of EvolvableModule"
                raise TypeError(
                    msg,
                )
            if not all(isinstance(net, EvolvableModule) for net in critics_list):
                msg = "All critic networks must be instances of EvolvableModule"
                raise TypeError(
                    msg,
                )

            assert len(actor_networks) == self.n_unique_agents, (
                f"Length of actor_networks ({len(actor_networks)}) does not match number of unique "
                f"agents defined in environment ({self.n_unique_agents}: {list(self.observation_space.keys())})"
            )
            assert len(critic_networks) == self.n_unique_agents, (
                f"Length of critic_networks ({len(critic_networks)}) does not match number of unique "
                f"agents defined in environment ({self.n_unique_agents}: {list(self.observation_space.keys())})"
            )

            actors_copy, critics_copy = make_safe_deepcopies(
                actor_networks,
                critic_networks,
            )
            self.actors = actors_copy
            self.critics = critics_copy
        else:
            built_net_config = self.build_net_config(net_config, flatten=False)

            self.actors = ModuleDict()
            self.critics = ModuleDict()
            for agent_id in self.observation_space:
                obs_space = self.observation_space[agent_id]
                action_space = self.action_space[agent_id]

                agent_config = built_net_config[agent_id]
                critic_net_config = copy.deepcopy(agent_config)
                head_config = agent_config.get("head_config", None)
                if head_config is not None:
                    critic_head_config = copy.deepcopy(head_config)
                    critic_head_config["output_activation"] = None
                    critic_net_config.pop("squash_output", None)
                else:
                    critic_head_config = MlpNetConfig(hidden_size=[64])

                critic_net_config["head_config"] = critic_head_config

                # Create one actor and critic per group of homogeneous agents,
                # which will be used by all homogeneous (identical) agents of
                # that group.
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
            optim.Adam,
            networks=self.actors,
            lr=self.lr,
        )
        self.critic_optimizers = OptimizerWrapper(
            optim.Adam,
            networks=self.critics,
            lr=self.lr,
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
                    f"{self.torch_compiler} compile mode is not compatible with GumbelSoftmax activation, changing to 'default' mode.",
                    stacklevel=2,
                )
                self.torch_compiler = "default"

            configure_tf32_precision()
            self.recompile()

        self.criterion = nn.MSELoss()

        # Register network groups for mutations
        self.register_network_group(
            NetworkGroup(
                eval_network=self.actors,
                policy=True,
            ),
        )
        self.register_network_group(
            NetworkGroup(
                eval_network=self.critics,
            ),
        )

        # Register metrics to keep track of during training
        for metric_name in ("loss", "policy_loss", "value_loss", "entropy_loss"):
            self.metrics.register(metric_name)

    def process_infos(
        self,
        infos: InfosDict | None,
    ) -> IPPOProcessInfosReturn:
        """Process the information, extract env_defined_actions, action_masks and agent_masks.

        :param infos: Info dict
        :type infos: InfosDict | None
        :return: Tuple of action_masks, env_defined_actions, agent_masks (the
            latter two are ``None`` when the info dict defines no actions)
        :rtype: IPPOProcessInfosReturn
        """
        if infos is None:
            infos = {agent: {} for agent in self.agent_ids}
            action_masks: IPPOActionMasks = dict.fromkeys(
                self.observation_space,
            )
        else:
            action_masks = self.extract_action_masks(infos)

        env_defined_actions, agent_masks = self.extract_agent_masks(infos)

        return action_masks, env_defined_actions, agent_masks

    def extract_action_masks(
        self,
        infos: InfosDict,
    ) -> IPPOActionMasks:
        """Extract action masks from info dictionary.

        :param infos: Info dict
        :type infos: InfosDict

        :return: Action masks per group (``None`` when no masks are provided)
        :rtype: IPPOActionMasks
        """
        # Get dict of form {"agent_id" : [1, 0, 0, 0]...} etc
        collected_masks: dict[str, list[MaybeActionMask]] = {
            group_id: [] for group_id in self.observation_space
        }
        for agent_id, info in infos.items():
            if isinstance(info, dict):
                group_id = (
                    self.get_group_id(agent_id)
                    if self.has_grouped_agents()
                    else agent_id
                )
                collected_masks[group_id].append(
                    coerce_action_mask(info.get("action_mask", None))
                )

        # Check and stack masks
        action_masks: dict[str, torch.Tensor | None] = {}
        for group_id in self.observation_space:
            group_masks = collected_masks[group_id]
            if None in group_masks or not group_masks:
                assert all(mask is None for mask in group_masks), (
                    f"If action masks are provided for any agents, they must be provided for all agents. "
                    "Action masks can be defined as an array with the shape of the action space "
                    f"({self.action_space}), where 1=legal and 0=illegal."
                )

                action_masks[group_id] = None
            else:
                action_masks[group_id] = torch.Tensor(np.array(group_masks))

        return action_masks

    def _get_action_and_values(
        self,
        obs: TorchObsType,
        actor: StochasticActor,
        critic: ValueNetwork,
        action_mask: torch.Tensor | None = None,
        batch_size: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get actions and values for a batch of grouped observations.

        :param obs: Observations of environment
        :type obs: torch.Tensor[float] or dict[str, torch.Tensor[float]] or tuple[torch.Tensor[float], ...]
        :param actor: Actor network
        :type actor: StochasticActor
        :param critic: Critic network
        :type critic: ValueNetwork
        :param action_mask: Action mask
        :type action_mask: torch.Tensor, optional
        :param batch_size: Batch size
        :type batch_size: int, optional
        :return: Tuple of actions, log probabilities, entropies, values
        :rtype: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        """
        # Process in batches. Mapping/tuple observations carry no direct
        # shape; they always take the single-pass branch (checking the leaf
        # arms first keeps the narrowing exact).
        if (
            batch_size is not None
            and not isinstance(obs, (dict, tuple))
            and obs.shape[0] > batch_size
        ):
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
                    batch_obs,
                    action_mask=batch_mask,
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
        obs: Mapping[str, ObservationType],
        infos: InfosDict | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[ArrayDict, ArrayDict, ArrayDict, ArrayDict]:
        """Return the next action to take in the environment.

        :param obs: Environment observations: {'agent_0': state_dim_0, ..., 'agent_n': state_dim_n}
        :type obs: Mapping[str, numpy.Array | dict[str, numpy.Array] | tuple[numpy.Array, ...]]
        :param infos: Information dictionary returned by env.step(actions)
        :type infos: InfosDict | None
        :return: Tuple of actions, log probabilities, entropies, values
        :rtype: tuple[ArrayDict, ArrayDict, ArrayDict, ArrayDict]
        """
        assert not key_in_nested_dict(
            obs,
            "action_mask",
        ), "AgileRL requires action masks to be defined in the information dictionary."

        action_masks, env_defined_actions, agent_masks = self.process_infos(infos)
        # Environment observations are numpy containers at this boundary.
        vect_dim = get_vect_dim(
            obs,
            self.possible_observation_spaces,
        )

        # Groups to extract actions from in observation
        unique_agents_ids = list(obs.keys())
        grouped_agents = defaultdict(list)
        for agent_id in unique_agents_ids:
            group_id = (
                self.get_group_id(agent_id) if self.has_grouped_agents() else agent_id
            )
            grouped_agents[group_id].append(agent_id)

        # Preprocess observations for each active agent group.
        preprocessed = self.preprocess_observation(
            obs,
            list(grouped_agents.keys()),
        )

        action_dict = {}
        action_logprob_dict = {}
        dist_entropy_dict = {}
        state_values_dict = {}
        for agent_id, agent_obs in preprocessed.items():
            action_mask = action_masks[agent_id]
            actor = self.actors[agent_id]
            critic = self.critics[agent_id]

            with torch.no_grad():
                action, log_prob, entropy, values = self._get_action_and_values(
                    obs=agent_obs,
                    actor=actor,
                    critic=critic,
                    action_mask=action_mask,
                    batch_size=self.action_batch_size,
                )

            # Clip to action space during inference
            agent_space = self.action_space[agent_id]
            action = action.cpu().data.numpy()
            if not self.training and isinstance(agent_space, spaces.Box):
                action = np.clip(action, agent_space.low, agent_space.high)

            action_dict[agent_id] = action
            action_logprob_dict[agent_id] = log_prob.cpu().data.numpy()
            dist_entropy_dict[agent_id] = entropy.cpu().data.numpy()
            state_values_dict[agent_id] = values.cpu().data.numpy()

        action_dict = self.disassemble_grouped_outputs(
            action_dict,
            vect_dim,
            grouped_agents,
        )

        # If using env_defined_actions replace actions
        if env_defined_actions is not None:
            # extract_agent_masks returns the masks alongside the actions.
            assert agent_masks is not None
            action_dict = apply_env_defined_actions(
                unique_agents_ids,
                action_dict,
                env_defined_actions,
                agent_masks,
                discrete_actions=isinstance(
                    next(iter(self.action_space.values())), spaces.Discrete
                ),
            )

        return (
            action_dict,
            self.disassemble_grouped_outputs(
                action_logprob_dict,
                vect_dim,
                grouped_agents,
            ),
            self.disassemble_grouped_outputs(
                dist_entropy_dict,
                vect_dim,
                grouped_agents,
            ),
            self.disassemble_grouped_outputs(
                state_values_dict,
                vect_dim,
                grouped_agents,
            ),
        )

    def learn(self, experiences: tuple[Mapping[str, Any], ...]) -> dict[str, float]:
        """Update agent network parameters to learn from experiences.

        :param experiences: 8-tuple of per-agent field maps holding batched
            states, actions, log_probs, rewards, dones, values, next_states and
            next_dones in that order.
        :type experiences: tuple[Mapping[str, Any], ...]

        :return: Loss dictionary
        :rtype: dict[str, float]
        """
        states, actions, log_probs, rewards, dones, values, next_states, next_dones = (
            map(self.assemble_shared_inputs, experiences)
        )

        loss_dict: dict[str, float] = {}
        for agent_id, state in states.items():
            actor = self.actors[agent_id]
            critic = self.critics[agent_id]
            actor_optimizer = self.actor_optimizers[agent_id]
            critic_optimizer = self.critic_optimizers[agent_id]
            obs_space = self.observation_space[agent_id]
            action_space = self.action_space[agent_id]

            loss_dict[f"{agent_id}"] = self._learn_individual(
                agent_id=agent_id,
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
        agent_id: str,
        experiences: tuple[dict[str, Any], ...],
        actor: StochasticActor,
        critic: ValueNetwork,
        actor_optimizer: Optimizer,
        critic_optimizer: Optimizer,
        obs_space: spaces.Space,
        action_space: spaces.Space,
    ) -> float:
        """Inner call to each agent for the learning/algo training steps,
        essentially the PPO learn method. Applies all forward/backward props.

        :param agent_id: ID of the agent
        :type agent_id: str
        :param experiences: States, actions, log_probs, rewards, dones, values, next_state, next_done in
            that order, organised by shared agent id
        :type experiences: tuple[dict[str, Any], ...]
        :param actor: Actor network
        :type actor: StochasticActor
        :param critic: Critic network
        :type critic: ValueNetwork
        :param actor_optimizer: Optimizer specific to the actor
        :type actor_optimizer: torch.optim.Optimizer
        :param critic_optimizer: Optimizer specific to the critic
        :type critic_optimizer: torch.optim.Optimizer
        :param obs_space: Observation space for the agent
        :type obs_space: gymnasium.spaces.Space
        :param action_space: Action space for the agent
        :type action_space: gymnasium.spaces.Space
        """
        obs, actions, log_probs, rewards, dones, values, next_obs, next_done = (
            experiences
        )

        # These fields are per-agent scalars, so they always vectorize to a flat
        # tensor (the dict/tuple arms only arise for structured observations,
        # which next_obs below may still be).
        log_probs = vectorize_agent_experiences_flat(log_probs)
        rewards = vectorize_agent_experiences_flat(rewards)
        dones = vectorize_agent_experiences_flat(dones)
        values = vectorize_agent_experiences_flat(values)
        log_probs = log_probs.squeeze()
        rewards = rewards.squeeze()
        dones = dones.squeeze()
        values = values.squeeze()
        vect_next_obs = vectorize_experiences_by_agent(next_obs, dim=0)
        next_done = vectorize_agent_experiences_flat(next_done, dim=0)

        with torch.no_grad():
            num_steps = rewards.size(0)
            rewards = rewards.reshape(num_steps, -1)
            dones = dones.reshape(num_steps, -1)
            values = values.reshape(num_steps, -1)
            next_done = next_done.reshape(1, -1)

            preprocessed_next_obs = preprocess_observation_fn(
                obs_space,
                vect_next_obs,
                self.device,
                self.normalize_images,
                swap_channels=self.swap_channels,
            )
            next_value = critic(preprocessed_next_obs).reshape(1, -1).cpu()
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

        flat_obs = concatenate_experiences_into_batches(obs, obs_space)
        flat_actions = concatenate_experiences_into_batches(
            actions,
            action_space,
            actions=True,
        )
        log_probs = log_probs.reshape((-1,))

        # Move experiences to algo device
        flat_experiences = self.to_device(
            flat_obs,
            flat_actions,
            log_probs,
            advantages,
            returns,
            values,
        )

        # The returns entry is a flat tensor in this layout.
        returns_flat = flat_experiences[4]
        assert isinstance(returns_flat, torch.Tensor)
        num_samples = returns_flat.size(0)
        batch_idxs = np.arange(num_samples)
        learn_metrics = {
            "loss": 0.0,
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy_loss": 0.0,
        }
        approx_kl = torch.tensor(float("inf"))
        for _ in range(self.update_epochs):
            np.random.shuffle(batch_idxs)
            for start in range(0, num_samples, self.batch_size):
                minibatch_idxs = batch_idxs[start : start + self.batch_size]
                (
                    batch_obs_raw,
                    batch_actions_raw,
                    batch_log_probs_raw,
                    batch_advantages_raw,
                    batch_returns_raw,
                    batch_values_raw,
                ) = get_experiences_samples(minibatch_idxs, *flat_experiences)

                # Non-observation fields are flat tensors in this layout.
                batch_actions = batch_actions_raw.squeeze()
                batch_returns = batch_returns_raw.squeeze()
                batch_log_probs = batch_log_probs_raw.squeeze()
                batch_advantages = batch_advantages_raw.squeeze()
                batch_values = batch_values_raw.squeeze()

                if len(minibatch_idxs) > 1:
                    batch_obs = preprocess_observation_fn(
                        obs_space,
                        # Sampled from the non-None observation batch above.
                        batch_obs_raw,
                        self.device,
                        self.normalize_images,
                        swap_channels=self.swap_channels,
                    )
                    _, _, entropy = actor(batch_obs)
                    value = critic(batch_obs).squeeze(-1)

                    log_prob = actor.action_log_prob(batch_actions)

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
                        ratio,
                        1 - self.clip_coef,
                        1 + self.clip_coef,
                    )
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # Value loss
                    value = value.view(-1)
                    v_loss_unclipped = (value - batch_returns) ** 2
                    v_clipped = batch_values + torch.clamp(
                        value - batch_values,
                        -self.clip_coef,
                        self.clip_coef,
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

                    learn_metrics["loss"] += actor_loss.item() + critic_loss.item()
                    learn_metrics["policy_loss"] += pg_loss.item()
                    learn_metrics["value_loss"] += v_loss.item()
                    learn_metrics["entropy_loss"] += entropy_loss.item()

            # Early stopping for the epoch if KL divergence target is exceeded
            if self.target_kl is not None and approx_kl > self.target_kl:
                break

        # Log metrics
        learn_metrics = {
            k: v / (num_samples * self.update_epochs) for k, v in learn_metrics.items()
        }
        for key, value in learn_metrics.items():
            self.metrics.log(key, value, agent_id=agent_id)

        return learn_metrics["loss"]

    def test(
        self,
        env: ParallelEnv | PettingZooVecEnv,
        max_steps: int | None = None,
        loop: int = 3,
        sum_scores: bool = True,
    ) -> float | npt.NDArray:
        """Return mean test score of agent in environment with epsilon-greedy policy.

        :param env: The environment to be tested in
        :type env: ParallelEnv | PettingZooVecEnv
        :param max_steps: Maximum number of testing steps, defaults to None
        :type max_steps: int, optional
        :param loop: Number of testing loops/episodes to complete. The returned score is the mean. Defaults to 3
        :type loop: int, optional
        :param sum_scores: Boolean flag to indicate whether to sum sub-agent scores, defaults to True
        :type sum_scores: bool, optional
        :return: Mean test score, or per-agent scores when ``sum_scores`` is False
        :rtype: float | npt.NDArray
        """
        self.set_training_mode(False)
        with torch.no_grad():
            rewards = []
            num_envs = get_num_envs(env)
            is_vectorised = hasattr(env, "num_envs")

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
                            np.isnan(terminated),
                            True,
                            terminated,
                        ).astype(bool)
                        truncated = np.where(
                            np.isnan(truncated),
                            False,
                            truncated,
                        ).astype(bool)

                        dones[agent_id] = terminated | truncated

                    if not is_vectorised:
                        dones = {
                            agent: np.array([dones[agent_id]])
                            for agent in self.agent_ids
                        }

                    for idx, agent_dones in enumerate(
                        zip(*dones.values(), strict=False)
                    ):
                        if (
                            np.all(agent_dones)
                            or (max_steps is not None and step == max_steps)
                        ) and not finished[idx]:
                            completed_episode_scores[idx] = scores[idx]
                            finished[idx] = 1

                rewards.append(np.mean(completed_episode_scores, axis=0))

        mean_fit_row = np.mean(rewards, axis=0)
        if sum_scores:
            fitness = float(mean_fit_row[0])
            self.metrics.add_fitness(fitness)
            return fitness

        # Per-agent fitness rows are stored as-is by BaseMetrics.add_fitness.
        self.metrics.add_fitness(mean_fit_row)
        return mean_fit_row
