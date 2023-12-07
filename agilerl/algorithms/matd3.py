import copy
import random
import warnings

import dill
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from agilerl.networks.evolvable_cnn import EvolvableCNN
from agilerl.networks.evolvable_mlp import EvolvableMLP
from agilerl.wrappers.make_evolvable import MakeEvolvable


class MATD3:
    """The MATD3 algorithm class. MATD3 paper: https://arxiv.org/abs/1910.01465

    :param state_dims: State observation dimensions for each agent
    :type state_dims: list[tuple]
    :param action_dims: Action dimensions for each agent
    :type action_dims: list[int]
    :param one_hot: One-hot encoding, used with discrete observation spaces
    :type one_hot: bool
    :param n_agents: Number of agents
    :type n_agents: int
    :param agent_ids: Agent ID for each agent
    :type agent_ids: list[str]
    :param max_action: Upper bound of the action space for each agent
    :type max_action: list[float]
    :param min_action: Lower bound of the action space for each agent
    :type min_action: list[float]
    :param discrete_actions: Boolean flag to indicate a discrete action space
    :type discrete_actions: bool, optional
    :param expl_noise: Standard deviation for Gaussian exploration noise, defaults to 0.1
    :type expl_noise: float, optional
    :param policy_freq: Policy update frequency, defaults to 2
    :type policy_freq: int, optional
    :param index: Index to keep track of object instance during tournament selection and mutation, defaults to 0
    :type index: int, optional
    :param net_config: Network configuration, defaults to mlp with hidden size [64,64]
    :type net_config: dict, optional
    :param batch_size: Size of batched sample from replay buffer for learning, defaults to 64
    :type batch_size: int, optional
    :param lr: Learning rate for optimizer, defaults to 0.01
    :type lr: float, optional
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
    :param critic_networks: List containing two lists of custom critic networks, defaults to None
    :type critic_networks: list[list[nn.Module]], optional
    :param device: Device for accelerated computing, 'cpu' or 'cuda', defaults to 'cpu'
    :type device: str, optional
    :param accelerator: Accelerator for distributed computing, defaults to None
    :type accelerator: accelerate.Accelerator(), optional
    :param wrap: Wrap models for distributed training upon creation, defaults to True
    :type wrap: bool, optional
    """

    def __init__(
        self,
        state_dims,
        action_dims,
        one_hot,
        n_agents,
        agent_ids,
        max_action,
        min_action,
        discrete_actions,
        expl_noise=0.1,
        index=0,
        policy_freq=2,
        net_config={"arch": "mlp", "h_size": [64, 64]},
        batch_size=64,
        lr=0.01,
        learn_step=5,
        gamma=0.95,
        tau=0.01,
        mutation=None,
        actor_networks=None,
        critic_networks=None,
        device="cpu",
        accelerator=None,
        wrap=True,
    ):
        assert isinstance(state_dims, list), "State dimensions must be a list."
        assert isinstance(action_dims, list), "Action dimensions must be a list."
        assert isinstance(
            one_hot, bool
        ), "One-hot encoding flag must be boolean value True or False."
        assert isinstance(n_agents, int), "Number of agents must be an integer."
        assert isinstance(
            agent_ids, (tuple, list)
        ), "Agent IDs must be stores in a tuple or list."
        assert isinstance(
            discrete_actions, bool
        ), "Discrete actions flag must be a boolean value True or False."
        assert (
            isinstance(max_action, list) or max_action is None
        ), "Max action must be a list."
        assert (
            isinstance(min_action, list) or min_action is None
        ), "Min action must be a list."
        assert (max_action is not None) == (
            min_action is not None
        ), "Max and min actions must both be supplied, or both be None."
        if max_action is not None and min_action is not None:
            for x, n in zip(max_action, min_action):
                x, n = x[0], n[0]
                assert x > n, "Max action must be greater than min action."
                assert x > 0, "Max action must be greater than zero."
                assert n <= 0, "Min action must be less than or equal to zero."
        assert isinstance(index, int), "Agent index must be an integer."
        assert isinstance(batch_size, int), "Batch size must be an integer."
        assert batch_size >= 1, "Batch size must be greater than or equal to one."
        assert isinstance(lr, float), "Learning rate must be a float."
        assert lr > 0, "Learning rate must be greater than zero."
        assert isinstance(learn_step, int), "Learn step rate must be an integer."
        assert learn_step >= 1, "Learn step must be greater than or equal to one."
        assert isinstance(gamma, float), "Gamma must be a float."
        assert isinstance(tau, float), "Tau must be a float."
        assert tau > 0, "Tau must be greater than zero."
        assert isinstance(policy_freq, int), "Policy frequency must be an integer."
        assert policy_freq > 0, "Policy frequency must be grateer than zero."
        assert n_agents == len(
            agent_ids
        ), "Number of agents must be equal to the length of the agent IDs list."
        if actor_networks is not None:
            assert all(
                isinstance(actor, nn.Module) for actor in actor_networks
            ), "Actor networks must be an nn.Module or None."
        if critic_networks is not None:
            for critic_network in critic_networks:
                assert all(
                    isinstance(critic, nn.Module) for critic in critic_network
                ), "Critic networks must be an nn.Module or None."
        if (actor_networks is not None) != (critic_networks is not None):
            warnings.warn(
                "Actor and critic network lists must both be supplied to use custom networks. Defaulting to net config."
            )
        assert isinstance(
            wrap, bool
        ), "Wrap models flag must be boolean value True or False."
        self.algo = "MATD3"
        self.state_dims = state_dims
        self.total_state_dims = sum(state_dim[0] for state_dim in self.state_dims)
        self.action_dims = action_dims
        self.one_hot = one_hot
        self.n_agents = n_agents
        self.multi = True
        self.agent_ids = agent_ids
        self.net_config = net_config
        self.batch_size = batch_size
        self.lr = lr
        self.learn_step = learn_step
        self.gamma = gamma
        self.tau = tau
        self.mut = mutation
        self.device = device
        self.accelerator = accelerator
        self.index = index
        self.policy_freq = policy_freq
        self.scores = []
        self.fitness = []
        self.steps = [0]
        self.max_action = max_action
        self.expl_noise = expl_noise
        self.min_action = min_action
        self.discrete_actions = discrete_actions
        self.total_actions = sum(self.action_dims)
        self.actor_networks = actor_networks
        self.critic_networks = critic_networks

        if self.actor_networks is not None and self.critic_networks is not None:
            self.actors = actor_networks
            self.critics_1 = critic_networks[0]
            self.critics_2 = critic_networks[-1]
            self.net_config = None
        else:
            if "output_activation" in self.net_config.keys():
                pass
            else:
                if self.discrete_actions:
                    self.net_config["output_activation"] = "GumbelSoftmax"

            # model
            if self.net_config["arch"] == "mlp":  # Multi-layer Perceptron
                self.actors = []
                for idx, (action_dim, state_dim) in enumerate(
                    zip(self.action_dims, self.state_dims)
                ):
                    if not self.discrete_actions:
                        if self.min_action[idx][0] < 0:
                            self.net_config["output_activation"] = "Tanh"
                        else:
                            self.net_config["output_activation"] = "Sigmoid"
                    self.actors.append(
                        EvolvableMLP(
                            num_inputs=state_dim[0],
                            num_outputs=action_dim,
                            hidden_size=self.net_config["h_size"],
                            mlp_output_activation=self.net_config["output_activation"],
                            device=self.device,
                            accelerator=self.accelerator,
                        )
                    )
                self.critics_1 = [
                    EvolvableMLP(
                        num_inputs=self.total_state_dims + self.total_actions,
                        num_outputs=1,
                        hidden_size=self.net_config["h_size"],
                        device=self.device,
                        accelerator=self.accelerator,
                    )
                    for _ in range(self.n_agents)
                ]
                self.critics_2 = [
                    EvolvableMLP(
                        num_inputs=self.total_state_dims + self.total_actions,
                        num_outputs=1,
                        hidden_size=self.net_config["h_size"],
                        device=self.device,
                        accelerator=self.accelerator,
                    )
                    for _ in range(self.n_agents)
                ]
            elif self.net_config["arch"] == "cnn":  # Convolutional Neural Network
                self.actors = []
                for idx, (action_dim, state_dim) in enumerate(
                    zip(self.action_dims, self.state_dims)
                ):
                    if not self.discrete_actions:
                        if self.min_action[idx][0] < 0:
                            self.net_config["output_activation"] = "Tanh"
                        else:
                            self.net_config["output_activation"] = "Sigmoid"
                    self.actors.append(
                        EvolvableCNN(
                            input_shape=state_dim,
                            num_actions=action_dim,
                            channel_size=self.net_config["c_size"],
                            kernel_size=self.net_config["k_size"],
                            stride_size=self.net_config["s_size"],
                            hidden_size=self.net_config["h_size"],
                            normalize=self.net_config["normalize"],
                            mlp_output_activation=self.net_config["output_activation"],
                            multi=self.multi,
                            n_agents=self.n_agents,
                            device=self.device,
                            accelerator=self.accelerator,
                        )
                    )
                self.critics_1 = [
                    EvolvableCNN(
                        input_shape=state_dim,
                        num_actions=self.total_actions,
                        channel_size=self.net_config["c_size"],
                        kernel_size=self.net_config["k_size"],
                        stride_size=self.net_config["s_size"],
                        hidden_size=self.net_config["h_size"],
                        normalize=self.net_config["normalize"],
                        mlp_activation="Tanh",
                        mlp_output_activation=None,
                        critic=True,
                        n_agents=self.n_agents,
                        multi=self.multi,
                        device=self.device,
                        accelerator=self.accelerator,
                    )
                    for state_dim in self.state_dims
                ]
                self.critics_2 = [
                    EvolvableCNN(
                        input_shape=state_dim,
                        num_actions=self.total_actions,
                        channel_size=self.net_config["c_size"],
                        kernel_size=self.net_config["k_size"],
                        stride_size=self.net_config["s_size"],
                        hidden_size=self.net_config["h_size"],
                        normalize=self.net_config["normalize"],
                        mlp_activation="Tanh",
                        mlp_output_activation=None,
                        critic=True,
                        n_agents=self.n_agents,
                        multi=self.multi,
                        device=self.device,
                        accelerator=self.accelerator,
                    )
                    for state_dim in self.state_dims
                ]

        # Assign architecture
        self.arch = (
            self.net_config["arch"]
            if self.net_config is not None
            else self.actors[0].arch
        )

        # Create target networks
        self.actor_targets = copy.deepcopy(self.actors)
        self.critic_targets_1 = copy.deepcopy(self.critics_1)
        self.critic_targets_2 = copy.deepcopy(self.critics_2)

        # Initialise target network parameters
        for actor, actor_target in zip(self.actors, self.actor_targets):
            actor_target.load_state_dict(actor.state_dict())
        for critic_1, critic_2, critic_target_1, critic_target_2 in zip(
            self.critics_1, self.critics_2, self.critic_targets_1, self.critic_targets_2
        ):
            critic_target_1.load_state_dict(critic_1.state_dict())
            critic_target_2.load_state_dict(critic_2.state_dict())

        self.actor_optimizers_type = [
            optim.Adam(actor.parameters(), lr=self.lr) for actor in self.actors
        ]
        self.critic_1_optimizers_type = [
            optim.Adam(critic.parameters(), lr=self.lr) for critic in self.critics_1
        ]
        self.critic_2_optimizers_type = [
            optim.Adam(critic.parameters(), lr=self.lr) for critic in self.critics_2
        ]

        if self.accelerator is not None:
            self.actor_optimizers = self.actor_optimizers_type
            self.critic_1_optimizers = self.critic_1_optimizers_type
            self.critic_2_optimizers = self.critic_2_optimizers_type
            if wrap:
                self.wrap_models()
        else:
            self.actors = [actor.to(self.device) for actor in self.actors]
            self.actor_targets = [
                actor_target.to(self.device) for actor_target in self.actor_targets
            ]
            self.critics_1 = [critic.to(self.device) for critic in self.critics_1]
            self.critic_targets_1 = [
                critic_target.to(self.device) for critic_target in self.critic_targets_1
            ]
            self.critics_2 = [critic.to(self.device) for critic in self.critics_2]
            self.critic_targets_2 = [
                critic_target.to(self.device) for critic_target in self.critic_targets_2
            ]
            self.actor_optimizers = self.actor_optimizers_type
            self.critic_1_optimizers = self.critic_1_optimizers_type
            self.critic_2_optimizers = self.critic_2_optimizers_type

        self.criterion = nn.MSELoss()

    def scale_to_action_space(self, action, idx):
        """Scales actions to action space defined by self.min_action and self.max_action.

        :param action: Action to be scaled
        :type action: numpy.ndarray
        """
        return np.where(
            action > 0,
            action * self.max_action[idx][0],
            action * -self.min_action[idx][0],
        )

    def getAction(self, states, epsilon=0, agent_mask=None, env_defined_actions=None):
        """Returns the next action to take in the environment.
        Epsilon is the probability of taking a random action, used for exploration.
        For epsilon-greedy behaviour, set epsilon to 0.

        :param state: Environment observations: {'agent_0': state_dim_0, ..., 'agent_n': state_dim_n}
        :type state: Dict[str, numpy.Array]
        :param epsilon: Probablilty of taking a random action for exploration, defaults to 0
        :type epsilon: float, optional
        :param agent_mask: Mask of agents to return actions for: {'agent_0': True, ..., 'agent_n': False}
        :type agent_mask: Dict[str, bool]
        :param env_defined_actions: Dictionary of actions defined by the environment: {'agent_0': np.array, ..., 'agent_n': np.array}
        :type env_defined_actions: Dict[str, np.array]
        """
        # Get agents, states and actions we want to take actions for at this timestep according to agent_mask
        if agent_mask is None:
            agent_ids = self.agent_ids
            actors = self.actors
            state_dims = self.state_dims
        else:
            agent_ids = [agent for agent in agent_mask.keys() if agent_mask[agent]]
            state_dims = [
                state_dim
                for state_dim, mask_flag in zip(self.state_dims, agent_mask.keys())
                if mask_flag
            ]
            action_dims_dict = {
                agent: action_dim
                for agent, action_dim in zip(self.agent_ids, self.action_dims)
            }
            states = {
                agent: states[agent] for agent in agent_mask.keys() if agent_mask[agent]
            }
            actors = [
                actor
                for agent, actor in zip(agent_mask.keys(), self.actors)
                if agent_mask[agent]
            ]

        # Convert states to a list of torch tensors
        states = [torch.from_numpy(state).float() for state in states.values()]

        # Configure accelerator
        if self.accelerator is None:
            states = [state.to(self.device) for state in states]

        if self.one_hot:
            states = [
                nn.functional.one_hot(state.long(), num_classes=state_dim[0])
                .float()
                .squeeze()
                for state, state_dim in zip(states, state_dims)
            ]

        if self.arch == "mlp":
            states = [
                state.unsqueeze(0) if len(state.size()) < 2 else state
                for state in states
            ]
        elif self.arch == "cnn":
            states = [state.unsqueeze(2) for state in states]

        action_dict = {}
        for idx, (agent_id, state, actor) in enumerate(zip(agent_ids, states, actors)):
            if random.random() < epsilon:
                if self.discrete_actions:
                    action = (
                        np.random.dirichlet(
                            np.ones(self.action_dims[idx]), state.size()[0]
                        )
                        .astype("float32")
                        .squeeze()
                    )
                else:
                    action = (
                        (self.max_action[idx][0] - self.min_action[idx][0])
                        * np.random.rand(state.size()[0], self.action_dims[idx])
                        .astype("float32")
                        .squeeze()
                    ) + self.min_action[idx][0]
            else:
                actor.eval()
                if self.accelerator is not None:
                    with actor.no_sync(), torch.no_grad():
                        action_values = actor(state)
                else:
                    with torch.no_grad():
                        action_values = actor(state)
                actor.train()
                if self.discrete_actions:
                    action = action_values.cpu().data.numpy().squeeze()
                else:
                    action = self.scale_to_action_space(
                        action_values.cpu().data.numpy().squeeze(), idx
                    )
                    action = (
                        action
                        + np.random.normal(
                            0, self.expl_noise, size=self.action_dims[idx]
                        ).astype(np.float32)
                    ).clip(self.min_action[idx][0], self.max_action[idx][0])
            action_dict[agent_id] = action

        if self.discrete_actions:
            discrete_action_dict = {}
            for agent, action in action_dict.items():
                if self.one_hot:
                    discrete_action_dict[agent] = action.argmax(axis=1)
                else:
                    discrete_action_dict[agent] = action.argmax().item()
        else:
            discrete_action_dict = None

        if env_defined_actions is not None:
            for agent in env_defined_actions.keys():
                if not agent_mask[agent]:
                    if self.discrete_actions:
                        discrete_action_dict.update({agent: env_defined_actions[agent]})
                        action = env_defined_actions[agent]
                        action = (
                            nn.functional.one_hot(
                                torch.tensor(action).long(),
                                num_classes=action_dims_dict[agent],
                            )
                            .float()
                            .squeeze()
                        )
                        action_dict.update({agent: action})
                    else:
                        action_dict.update({agent: env_defined_actions[agent]})

        return (
            action_dict,
            discrete_action_dict,
        )

    def learn(self, experiences):
        """Updates agent network parameters to learn from experiences.

        :param experience: Tuple of dictionaries containing batched states, actions, rewards, next_states,
        dones in that order for each individual agent.
        :type experience: Tuple[Dict[str, torch.Tensor]]
        """

        for idx, (
            agent_id,
            actor,
            actor_target,
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
                self.actor_targets,
                self.critics_1,
                self.critic_targets_1,
                self.critics_2,
                self.critic_targets_2,
                self.actor_optimizers,
                self.critic_1_optimizers,
                self.critic_2_optimizers,
            )
        ):
            states, actions, rewards, next_states, dones = experiences

            if self.one_hot:
                states = {
                    agent_id: nn.functional.one_hot(
                        state.long(), num_classes=state_dim[0]
                    )
                    .float()
                    .squeeze()
                    for (agent_id, state), state_dim in zip(
                        states.items(), self.state_dims
                    )
                }
                next_states = {
                    agent_id: nn.functional.one_hot(
                        next_state.long(), num_classes=state_dim[0]
                    )
                    .float()
                    .squeeze()
                    for (agent_id, next_state), state_dim in zip(
                        next_states.items(), self.state_dims
                    )
                }

            if self.arch == "mlp":
                action_values = list(actions.values())
                input_combined = torch.cat(list(states.values()) + action_values, 1)
                if self.accelerator is not None:
                    with critic_1.no_sync():
                        q_value_1 = critic_1(input_combined)
                    with critic_2.no_sync():
                        q_value_2 = critic_2(input_combined)
                else:
                    q_value_1 = critic_1(input_combined)
                    q_value_2 = critic_2(input_combined)
                next_actions = []
                for i, agent_id_label in enumerate(self.agent_ids):
                    unscaled_actions = self.actor_targets[i](
                        next_states[agent_id_label]
                    ).detach_()
                    if not self.discrete_actions:
                        scaled_actions = torch.where(
                            unscaled_actions > 0,
                            unscaled_actions * self.max_action[i][0],
                            unscaled_actions * -self.min_action[i][0],
                        )
                        next_actions.append(scaled_actions)
                    else:
                        next_actions.append(unscaled_actions)

            elif self.arch == "cnn":
                stacked_states = torch.stack(list(states.values()), dim=2)
                stacked_actions = torch.cat(list(actions.values()), dim=1)
                if self.accelerator is not None:
                    with critic_1.no_sync():
                        q_value_1 = critic_1(stacked_states, stacked_actions)
                    with critic_2.no_sync():
                        q_value_2 = critic_2(stacked_states, stacked_actions)
                else:
                    q_value_1 = critic_1(stacked_states, stacked_actions)
                    q_value_2 = critic_2(stacked_states, stacked_actions)
                next_actions = []
                for i, agent_id_label in enumerate(self.agent_ids):
                    unscaled_actions = self.actor_targets[i](
                        next_states[agent_id_label].unsqueeze(2)
                    ).detach_()
                    if not self.discrete_actions:
                        scaled_actions = torch.where(
                            unscaled_actions > 0,
                            unscaled_actions * self.max_action[i][0],
                            unscaled_actions * -self.min_action[i][0],
                        )
                        next_actions.append(scaled_actions)
                    else:
                        next_actions.append(unscaled_actions)

            if self.arch == "mlp":
                next_input_combined = torch.cat(
                    list(next_states.values()) + next_actions, 1
                )
                if self.accelerator is not None:
                    with critic_target_1.no_sync():
                        q_value_next_state_1 = critic_target_1(next_input_combined)
                    with critic_target_2.no_sync():
                        q_value_next_state_2 = critic_target_2(next_input_combined)
                else:
                    q_value_next_state_1 = critic_target_1(next_input_combined)
                    q_value_next_state_2 = critic_target_2(next_input_combined)
            elif self.arch == "cnn":
                stacked_next_states = torch.stack(list(next_states.values()), dim=2)
                stacked_next_actions = torch.cat(next_actions, dim=1)
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

            y_j = (
                rewards[agent_id]
                + (1 - dones[agent_id]) * self.gamma * q_value_next_state
            )

            critic_loss = self.criterion(q_value_1, y_j.detach_()) + self.criterion(
                q_value_2, y_j.detach_()
            )

            # critic loss backprop
            critic_1_optimizer.zero_grad()
            critic_2_optimizer.zero_grad()
            if self.accelerator is not None:
                self.accelerator.backward(critic_loss)
            else:
                critic_loss.backward()
            critic_1_optimizer.step()
            critic_2_optimizer.step()

            # update actor and targets every policy_freq episodes
            if len(self.scores) % self.policy_freq == 0:
                if self.arch == "mlp":
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
                    input_combined = torch.cat(
                        list(states.values()) + list(detached_actions.values()), 1
                    )
                    if self.accelerator is not None:
                        with critic_1.no_sync():
                            actor_loss = -critic_1(input_combined).mean()
                    else:
                        actor_loss = -critic_1(input_combined).mean()

                elif self.arch == "cnn":
                    if self.accelerator is not None:
                        with actor.no_sync():
                            action = actor(states[agent_id].unsqueeze(2))
                    else:
                        action = actor(states[agent_id].unsqueeze(2))
                    if not self.discrete_actions:
                        action = torch.where(
                            action > 0,
                            action * self.max_action[idx][0],
                            action * -self.min_action[idx][0],
                        )
                    detached_actions = copy.deepcopy(actions)
                    detached_actions[agent_id] = action
                    stacked_detached_actions = torch.cat(
                        list(detached_actions.values()), dim=1
                    )
                    if self.accelerator is not None:
                        with critic_1.no_sync():
                            actor_loss = -critic_1(
                                stacked_states, stacked_detached_actions
                            ).mean()
                    else:
                        actor_loss = -critic_1(
                            stacked_states, stacked_detached_actions
                        ).mean()

                # actor loss backprop
                actor_optimizer.zero_grad()
                if self.accelerator is not None:
                    self.accelerator.backward(actor_loss)
                else:
                    actor_loss.backward()
                actor_optimizer.step()

        if len(self.scores) % self.policy_freq == 0:
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
                self.softUpdate(actor, actor_target)
                self.softUpdate(critic_1, critic_target_1)
                self.softUpdate(critic_2, critic_target_2)

            return actor_loss.item(), critic_loss.item()
        else:
            return None, critic_loss.item()

    def softUpdate(self, net, target):
        """Soft updates target network."""
        for eval_param, target_param in zip(net.parameters(), target.parameters()):
            target_param.data.copy_(
                self.tau * eval_param.data + (1.0 - self.tau) * target_param.data
            )

    def test(self, env, swap_channels=False, max_steps=500, loop=3):
        """Returns mean test score of agent in environment with epsilon-greedy policy.

        :param env: The environment to be tested in
        :type env: Gym-style environment
        :param swap_channels: Swap image channels dimension from last to first [H, W, C] -> [C, H, W], defaults to False
        :type swap_channels: bool, optional
        :param max_steps: Maximum number of testing steps, defaults to 500
        :type max_steps: int, optional
        :param loop: Number of testing loops/episodes to complete. The returned score is the mean. Defaults to 3
        :type loop: int, optional
        """
        with torch.no_grad():
            rewards = []
            for i in range(loop):
                state, info = env.reset()
                agent_reward = {agent_id: 0 for agent_id in self.agent_ids}
                score = 0
                for _ in range(max_steps):
                    if swap_channels:
                        state = {
                            agent_id: np.moveaxis(np.expand_dims(s, 0), [3], [1])
                            for agent_id, s in state.items()
                        }
                    agent_mask = (
                        info["agent_mask"] if "agent_mask" in info.keys() else None
                    )
                    env_defined_actions = (
                        info["env_defined_actions"]
                        if "env_defined_actions" in info.keys()
                        else None
                    )
                    cont_actions, discrete_action = self.getAction(
                        state,
                        epsilon=0,
                        agent_mask=agent_mask,
                        env_defined_actions=env_defined_actions,
                    )
                    if self.discrete_actions:
                        action = discrete_action
                    else:
                        action = cont_actions
                    state, reward, done, trunc, info = env.step(action)
                    for agent_id, r in reward.items():
                        agent_reward[agent_id] += r
                    score = sum(agent_reward.values())
                rewards.append(score)
        mean_fit = np.mean(rewards)
        self.fitness.append(mean_fit)
        return mean_fit

    def clone(self, index=None, wrap=True):
        """Returns cloned agent identical to self.

        :param index: Index to keep track of agent for tournament selection and mutation, defaults to None
        :type index: int, optional
        """
        if index is None:
            index = self.index

        clone = type(self)(
            state_dims=self.state_dims,
            action_dims=self.action_dims,
            one_hot=self.one_hot,
            n_agents=self.n_agents,
            agent_ids=self.agent_ids,
            max_action=self.max_action,
            min_action=self.min_action,
            expl_noise=self.expl_noise,
            discrete_actions=self.discrete_actions,
            index=index,
            net_config=self.net_config,
            batch_size=self.batch_size,
            policy_freq=self.policy_freq,
            lr=self.lr,
            learn_step=self.learn_step,
            gamma=self.gamma,
            tau=self.tau,
            mutation=self.mut,
            actor_networks=self.actor_networks,
            critic_networks=self.critic_networks,
            device=self.device,
            accelerator=self.accelerator,
            wrap=wrap,
        )

        if self.accelerator is not None:
            self.unwrap_models()
        actors = [actor.clone() for actor in self.actors]
        actor_targets = [actor_target.clone() for actor_target in self.actor_targets]
        critics_1 = [critic.clone() for critic in self.critics_1]
        critic_targets_1 = [
            critic_target.clone() for critic_target in self.critic_targets_1
        ]
        critics_2 = [critic.clone() for critic in self.critics_2]
        critic_targets_2 = [
            critic_target.clone() for critic_target in self.critic_targets_2
        ]
        actor_optimizers = [
            optim.Adam(actor.parameters(), lr=clone.lr) for actor in actors
        ]
        critic_1_optimizers = [
            optim.Adam(critic.parameters(), lr=clone.lr) for critic in critics_1
        ]
        critic_2_optimizers = [
            optim.Adam(critic.parameters(), lr=clone.lr) for critic in critics_2
        ]
        clone.actor_optimizers_type = actor_optimizers
        clone.critic_1_optimizers_type = critic_1_optimizers
        clone.critic_2_optimizers_type = critic_2_optimizers

        if self.accelerator is not None:
            if wrap:
                clone.actors = [self.accelerator.prepare(actor) for actor in actors]
                clone.actor_targets = [
                    self.accelerator.prepare(actor_target)
                    for actor_target in actor_targets
                ]
                clone.critics_1 = [
                    self.accelerator.prepare(critic) for critic in critics_1
                ]
                clone.critic_targets_1 = [
                    self.accelerator.prepare(critic_target)
                    for critic_target in critic_targets_1
                ]
                clone.critics_2 = [
                    self.accelerator.prepare(critic) for critic in critics_2
                ]
                clone.critic_targets_2 = [
                    self.accelerator.prepare(critic_target)
                    for critic_target in critic_targets_2
                ]
                clone.actor_optimizers = [
                    self.accelerator.prepare(actor_optimizer)
                    for actor_optimizer in actor_optimizers
                ]
                clone.critic_1_optimizers = [
                    self.accelerator.prepare(critic_optimizer)
                    for critic_optimizer in critic_1_optimizers
                ]
                clone.critic_2_optimizers = [
                    self.accelerator.prepare(critic_optimizer)
                    for critic_optimizer in critic_2_optimizers
                ]
            else:
                (
                    clone.actors,
                    clone.actor_targets,
                    clone.critics_1,
                    clone.critic_targets_1,
                    clone.critics_2,
                    clone.critic_targets_2,
                    clone.actor_optimizers,
                    clone.critic_1_optimizers,
                    clone.critic_2_optimizers,
                ) = (
                    actors,
                    actor_targets,
                    critics_1,
                    critic_targets_1,
                    critics_2,
                    critic_targets_2,
                    actor_optimizers,
                    critic_1_optimizers,
                    critic_2_optimizers,
                )
        else:
            clone.actors = [actor.to(self.device) for actor in actors]
            clone.actor_targets = [
                actor_target.to(self.device) for actor_target in actor_targets
            ]
            clone.critics_1 = [critic.to(self.device) for critic in critics_1]
            clone.critic_targets_1 = [
                critic_target.to(self.device) for critic_target in critic_targets_1
            ]
            clone.critics_2 = [critic.to(self.device) for critic in critics_2]
            clone.critic_targets_2 = [
                critic_target.to(self.device) for critic_target in critic_targets_2
            ]
            clone.actor_optimizers = actor_optimizers
            clone.critic_1_optimizers = critic_1_optimizers
            clone.critic_2_optimizers = critic_2_optimizers

        clone.fitness = copy.deepcopy(self.fitness)
        clone.steps = copy.deepcopy(self.steps)
        clone.scores = copy.deepcopy(self.scores)

        return clone

    def wrap_models(self):
        if self.accelerator is not None:
            self.actors = [self.accelerator.prepare(actor) for actor in self.actors]
            self.actor_targets = [
                self.accelerator.prepare(actor_target)
                for actor_target in self.actor_targets
            ]
            self.critics_1 = [
                self.accelerator.prepare(critic) for critic in self.critics_1
            ]
            self.critic_targets_1 = [
                self.accelerator.prepare(critic_target)
                for critic_target in self.critic_targets_1
            ]
            self.critics_2 = [
                self.accelerator.prepare(critic) for critic in self.critics_2
            ]
            self.critic_targets_2 = [
                self.accelerator.prepare(critic_target)
                for critic_target in self.critic_targets_2
            ]
            self.actor_optimizers = [
                self.accelerator.prepare(actor_optimizer)
                for actor_optimizer in self.actor_optimizers_type
            ]
            self.critic_1_optimizers = [
                self.accelerator.prepare(critic_optimizer)
                for critic_optimizer in self.critic_1_optimizers_type
            ]
            self.critic_2_optimizers = [
                self.accelerator.prepare(critic_optimizer)
                for critic_optimizer in self.critic_2_optimizers_type
            ]

    def unwrap_models(self):
        if self.accelerator is not None:
            self.actors = [
                self.accelerator.unwrap_model(actor) for actor in self.actors
            ]
            self.actor_targets = [
                self.accelerator.unwrap_model(actor_target)
                for actor_target in self.actor_targets
            ]
            self.critics_1 = [
                self.accelerator.unwrap_model(critic) for critic in self.critics_1
            ]
            self.critic_targets_1 = [
                self.accelerator.unwrap_model(critic_target)
                for critic_target in self.critic_targets_1
            ]
            self.critics_2 = [
                self.accelerator.unwrap_model(critic) for critic in self.critics_2
            ]
            self.critic_targets_2 = [
                self.accelerator.unwrap_model(critic_target)
                for critic_target in self.critic_targets_2
            ]
            self.actor_optimizers = [
                self.accelerator.unwrap_model(actor_optimizer)
                for actor_optimizer in self.actor_optimizers
            ]
            self.critic_1_optimizers = [
                self.accelerator.unwrap_model(critic_optimizer)
                for critic_optimizer in self.critic_1_optimizers
            ]
            self.critic_2_optimizers = [
                self.accelerator.unwrap_model(critic_optimizer)
                for critic_optimizer in self.critic_2_optimizers
            ]

    def saveCheckpoint(self, path):
        """Saves a checkpoint of agent properties and network weights to path.

        :param path: Location to save checkpoint at
        :type path: string
        """

        torch.save(
            {
                "state_dims": self.state_dims,
                "action_dims": self.action_dims,
                "one_hot": self.one_hot,
                "n_agents": self.n_agents,
                "agent_ids": self.agent_ids,
                "min_action": self.min_action,
                "max_action": self.max_action,
                "discrete_actions": self.discrete_actions,
                "actors_init_dict": [actor.init_dict for actor in self.actors],
                "actors_state_dict": [actor.state_dict() for actor in self.actors],
                "actor_targets_init_dict": [
                    actor_target.init_dict for actor_target in self.actor_targets
                ],
                "actor_targets_state_dict": [
                    actor_target.state_dict() for actor_target in self.actor_targets
                ],
                "critics_1_init_dict": [
                    critic_1.init_dict for critic_1 in self.critics_1
                ],
                "critics_1_state_dict": [
                    critic_1.state_dict() for critic_1 in self.critics_1
                ],
                "critic_targets_1_init_dict": [
                    critic_target_1.init_dict
                    for critic_target_1 in self.critic_targets_1
                ],
                "critic_targets_1_state_dict": [
                    critic_target_1.state_dict()
                    for critic_target_1 in self.critic_targets_1
                ],
                "critics_2_init_dict": [
                    critic_2.init_dict for critic_2 in self.critics_2
                ],
                "critics_2_state_dict": [
                    critic_2.state_dict() for critic_2 in self.critics_2
                ],
                "critic_targets_2_init_dict": [
                    critic_target_2.init_dict
                    for critic_target_2 in self.critic_targets_2
                ],
                "critic_targets_2_state_dict": [
                    critic_target_2.state_dict()
                    for critic_target_2 in self.critic_targets_2
                ],
                "actor_optimizers_state_dict": [
                    actor_optimizer.state_dict()
                    for actor_optimizer in self.actor_optimizers
                ],
                "critic_1_optimizers_state_dict": [
                    critic_1_optimizer.state_dict()
                    for critic_1_optimizer in self.critic_1_optimizers
                ],
                "critic_2_optimizers_state_dict": [
                    critic_2_optimizer.state_dict()
                    for critic_2_optimizer in self.critic_2_optimizers
                ],
                "expl_noise": self.expl_noise,
                "net_config": self.net_config,
                "batch_size": self.batch_size,
                "lr": self.lr,
                "learn_step": self.learn_step,
                "policy_freq": self.policy_freq,
                "gamma": self.gamma,
                "tau": self.tau,
                "mutation": self.mut,
                "index": self.index,
                "scores": self.scores,
                "fitness": self.fitness,
                "steps": self.steps,
            },
            path,
            pickle_module=dill,
        )

    def loadCheckpoint(self, path):
        """Loads saved agent properties and network weights from checkpoint.

        :param path: Location to load checkpoint from
        :type path: string
        """
        checkpoint = torch.load(path, pickle_module=dill)
        self.net_config = checkpoint["net_config"]
        if self.net_config is not None:
            self.arch = checkpoint["net_config"]["arch"]
            if self.arch == "mlp":
                self.actors = [
                    EvolvableMLP(**checkpoint["actors_init_dict"][idx])
                    for idx, _ in enumerate(self.agent_ids)
                ]
                self.actor_targets = [
                    EvolvableMLP(**checkpoint["actor_targets_init_dict"][idx])
                    for idx, _ in enumerate(self.agent_ids)
                ]
                self.critics_1 = [
                    EvolvableMLP(**checkpoint["critics_1_init_dict"][idx])
                    for idx, _ in enumerate(self.agent_ids)
                ]
                self.critic_targets_1 = [
                    EvolvableMLP(**checkpoint["critic_targets_1_init_dict"][idx])
                    for idx, _ in enumerate(self.agent_ids)
                ]
                self.critics_2 = [
                    EvolvableMLP(**checkpoint["critics_2_init_dict"][idx])
                    for idx, _ in enumerate(self.agent_ids)
                ]
                self.critic_targets_2 = [
                    EvolvableMLP(**checkpoint["critic_targets_2_init_dict"][idx])
                    for idx, _ in enumerate(self.agent_ids)
                ]
            elif self.arch == "cnn":
                self.actors = [
                    EvolvableCNN(**checkpoint["actors_init_dict"][idx])
                    for idx, _ in enumerate(self.agent_ids)
                ]
                self.actor_targets = [
                    EvolvableCNN(**checkpoint["actor_targets_init_dict"][idx])
                    for idx, _ in enumerate(self.agent_ids)
                ]
                self.critics_1 = [
                    EvolvableCNN(**checkpoint["critics_1_init_dict"][idx])
                    for idx, _ in enumerate(self.agent_ids)
                ]
                self.critic_targets_1 = [
                    EvolvableCNN(**checkpoint["critic_targets_1_init_dict"][idx])
                    for idx, _ in enumerate(self.agent_ids)
                ]
                self.critics_2 = [
                    EvolvableCNN(**checkpoint["critics_2_init_dict"][idx])
                    for idx, _ in enumerate(self.agent_ids)
                ]
                self.critic_targets_2 = [
                    EvolvableCNN(**checkpoint["critic_targets_2_init_dict"][idx])
                    for idx, _ in enumerate(self.agent_ids)
                ]
        else:
            self.actors = [
                MakeEvolvable(**checkpoint["actors_init_dict"][idx])
                for idx, _ in enumerate(self.agent_ids)
            ]
            self.actor_targets = [
                MakeEvolvable(**checkpoint["actor_targets_init_dict"][idx])
                for idx, _ in enumerate(self.agent_ids)
            ]
            self.critics_1 = [
                MakeEvolvable(**checkpoint["critics_1_init_dict"][idx])
                for idx, _ in enumerate(self.agent_ids)
            ]
            self.critic_targets_1 = [
                MakeEvolvable(**checkpoint["critic_targets_1_init_dict"][idx])
                for idx, _ in enumerate(self.agent_ids)
            ]
            self.critics_2 = [
                MakeEvolvable(**checkpoint["critics_2_init_dict"][idx])
                for idx, _ in enumerate(self.agent_ids)
            ]
            self.critic_targets_2 = [
                MakeEvolvable(**checkpoint["critic_targets_2_init_dict"][idx])
                for idx, _ in enumerate(self.agent_ids)
            ]
        self.lr = checkpoint["lr"]
        self.actor_optimizers = [
            optim.Adam(actor.parameters(), lr=self.lr) for actor in self.actors
        ]
        self.critic_1_optimizers = [
            optim.Adam(critic_1.parameters(), lr=self.lr) for critic_1 in self.critics_1
        ]
        self.critic_2_optimizers = [
            optim.Adam(critic_2.parameters(), lr=self.lr) for critic_2 in self.critics_2
        ]
        actor_list = []
        critic_1_list = []
        critic_2_list = []
        actor_target_list = []
        critic_target_1_list = []
        critic_target_2_list = []
        actor_optimizer_list = []
        critic_1_optimizer_list = []
        critic_2_optimizer_list = []

        for idx, (
            actor,
            actor_target,
            critic_1,
            critic_target_1,
            critic_2,
            critic_target_2,
            actor_optimizer,
            critic_1_optimizer,
            critic_2_optimizer,
        ) in enumerate(
            zip(
                self.actors,
                self.actor_targets,
                self.critics_1,
                self.critic_targets_1,
                self.critics_2,
                self.critic_targets_2,
                self.actor_optimizers,
                self.critic_1_optimizers,
                self.critic_2_optimizers,
            )
        ):
            actor.load_state_dict(checkpoint["actors_state_dict"][idx])
            actor_list.append(actor)
            actor_target.load_state_dict(checkpoint["actor_targets_state_dict"][idx])
            actor_target_list.append(actor_target)
            critic_1.load_state_dict(checkpoint["critics_1_state_dict"][idx])
            critic_1_list.append(critic_1)
            critic_2.load_state_dict(checkpoint["critics_2_state_dict"][idx])
            critic_2_list.append(critic_2)
            critic_target_1.load_state_dict(
                checkpoint["critic_targets_1_state_dict"][idx]
            )
            critic_target_1_list.append(critic_target_1)
            critic_target_2.load_state_dict(
                checkpoint["critic_targets_2_state_dict"][idx]
            )
            critic_target_2_list.append(critic_target_2)
            actor_optimizer.load_state_dict(
                checkpoint["actor_optimizers_state_dict"][idx]
            )
            actor_optimizer_list.append(actor_optimizer)
            critic_1_optimizer.load_state_dict(
                checkpoint["critic_1_optimizers_state_dict"][idx]
            )
            critic_1_optimizer_list.append(critic_1_optimizer)
            critic_2_optimizer.load_state_dict(
                checkpoint["critic_2_optimizers_state_dict"][idx]
            )
            critic_2_optimizer_list.append(critic_2_optimizer)

        self.actors = actor_list
        self.actor_targets = actor_target_list
        self.critics_1 = critic_1_list
        self.critic_targets_1 = critic_target_1_list
        self.critics_2 = critic_2_list
        self.critic_targets_2 = critic_target_2_list
        self.actor_optimizers = actor_optimizer_list
        self.critic_1_optimizers = critic_1_optimizer_list
        self.critic_2_optimizers = critic_2_optimizer_list
        self.expl_noise = checkpoint["expl_noise"]
        self.batch_size = checkpoint["batch_size"]
        self.learn_step = checkpoint["learn_step"]
        self.policy_freq = checkpoint["policy_freq"]
        self.gamma = checkpoint["gamma"]
        self.tau = checkpoint["tau"]
        self.mut = checkpoint["mutation"]
        self.index = checkpoint["index"]
        self.scores = checkpoint["scores"]
        self.fitness = checkpoint["fitness"]
        self.steps = checkpoint["steps"]

    @classmethod
    def load(cls, path, device="cpu", accelerator=None):
        """Creates agent with properties and network weights loaded from path.

        :param path: Location to load checkpoint from
        :type path: string
        :param device: Device for accelerated computing, 'cpu' or 'cuda', defaults to 'cpu'
        :type device: str, optional
        :param accelerator: Accelerator for distributed computing, defaults to None
        :type accelerator: accelerate.Accelerator(), optional
        """
        checkpoint = torch.load(path, pickle_module=dill)
        for idx, _ in enumerate(checkpoint["agent_ids"]):
            checkpoint["actors_init_dict"][idx]["device"] = device
            checkpoint["actor_targets_init_dict"][idx]["device"] = device
            checkpoint["critics_1_init_dict"][idx]["device"] = device
            checkpoint["critic_targets_1_init_dict"][idx]["device"] = device
            checkpoint["critics_2_init_dict"][idx]["device"] = device
            checkpoint["critic_targets_2_init_dict"][idx]["device"] = device

        if checkpoint["net_config"] is not None:
            agent = cls(
                state_dims=checkpoint["state_dims"],
                action_dims=checkpoint["action_dims"],
                one_hot=checkpoint["one_hot"],
                n_agents=checkpoint["n_agents"],
                agent_ids=checkpoint["agent_ids"],
                discrete_actions=checkpoint["discrete_actions"],
                min_action=checkpoint["min_action"],
                max_action=checkpoint["max_action"],
                expl_noise=checkpoint["expl_noise"],
                index=checkpoint["index"],
                net_config=checkpoint["net_config"],
                policy_freq=checkpoint["policy_freq"],
                batch_size=checkpoint["batch_size"],
                lr=checkpoint["lr"],
                learn_step=checkpoint["learn_step"],
                gamma=checkpoint["gamma"],
                tau=checkpoint["tau"],
                mutation=checkpoint["mutation"],
                device=device,
                accelerator=accelerator,
            )
            agent.arch = checkpoint["net_config"]["arch"]
            if agent.arch == "mlp":
                agent.actors = [
                    EvolvableMLP(**checkpoint["actors_init_dict"][idx])
                    for idx, _ in enumerate(agent.agent_ids)
                ]
                agent.actor_targets = [
                    EvolvableMLP(**checkpoint["actor_targets_init_dict"][idx])
                    for idx, _ in enumerate(agent.agent_ids)
                ]
                agent.critics_1 = [
                    EvolvableMLP(**checkpoint["critics_1_init_dict"][idx])
                    for idx, _ in enumerate(agent.agent_ids)
                ]
                agent.critic_targets_1 = [
                    EvolvableMLP(**checkpoint["critic_targets_1_init_dict"][idx])
                    for idx, _ in enumerate(agent.agent_ids)
                ]
                agent.critics_2 = [
                    EvolvableMLP(**checkpoint["critics_2_init_dict"][idx])
                    for idx, _ in enumerate(agent.agent_ids)
                ]
                agent.critic_targets_2 = [
                    EvolvableMLP(**checkpoint["critic_targets_2_init_dict"][idx])
                    for idx, _ in enumerate(agent.agent_ids)
                ]
            elif agent.arch == "cnn":
                agent.actors = [
                    EvolvableCNN(**checkpoint["actors_init_dict"][idx])
                    for idx, _ in enumerate(agent.agent_ids)
                ]
                agent.actor_targets = [
                    EvolvableCNN(**checkpoint["actor_targets_init_dict"][idx])
                    for idx, _ in enumerate(agent.agent_ids)
                ]
                agent.critics_1 = [
                    EvolvableCNN(**checkpoint["critics_1_init_dict"][idx])
                    for idx, _ in enumerate(agent.agent_ids)
                ]
                agent.critic_targets_1 = [
                    EvolvableCNN(**checkpoint["critic_targets_1_init_dict"][idx])
                    for idx, _ in enumerate(agent.agent_ids)
                ]
                agent.critics_2 = [
                    EvolvableCNN(**checkpoint["critics_2_init_dict"][idx])
                    for idx, _ in enumerate(agent.agent_ids)
                ]
                agent.critic_targets_2 = [
                    EvolvableCNN(**checkpoint["critic_targets_2_init_dict"][idx])
                    for idx, _ in enumerate(agent.agent_ids)
                ]
        else:
            agent = cls(
                state_dims=checkpoint["state_dims"],
                action_dims=checkpoint["action_dims"],
                one_hot=checkpoint["one_hot"],
                n_agents=checkpoint["n_agents"],
                agent_ids=checkpoint["agent_ids"],
                discrete_actions=checkpoint["discrete_actions"],
                min_action=checkpoint["min_action"],
                max_action=checkpoint["max_action"],
                expl_noise=checkpoint["expl_noise"],
                index=checkpoint["index"],
                net_config=checkpoint["net_config"],
                policy_freq=checkpoint["policy_freq"],
                batch_size=checkpoint["batch_size"],
                lr=checkpoint["lr"],
                learn_step=checkpoint["learn_step"],
                gamma=checkpoint["gamma"],
                tau=checkpoint["tau"],
                mutation=checkpoint["mutation"],
                actor_networks=[
                    MakeEvolvable(**checkpoint["actors_init_dict"][idx])
                    for idx, _ in enumerate(checkpoint["agent_ids"])
                ],
                critic_networks=[
                    [
                        MakeEvolvable(**checkpoint["critics_1_init_dict"][idx])
                        for idx, _ in enumerate(checkpoint["agent_ids"])
                    ],
                    [
                        MakeEvolvable(**checkpoint["critics_2_init_dict"][idx])
                        for idx, _ in enumerate(checkpoint["agent_ids"])
                    ],
                ],
                device=device,
                accelerator=accelerator,
            )
            agent.actor_targets = [
                MakeEvolvable(**checkpoint["actor_targets_init_dict"][idx])
                for idx, _ in enumerate(checkpoint["agent_ids"])
            ]
            agent.critic_targets = [
                [
                    MakeEvolvable(**checkpoint["critic_targets_1_init_dict"][idx])
                    for idx, _ in enumerate(checkpoint["agent_ids"])
                ],
                [
                    MakeEvolvable(**checkpoint["critic_targets_2_init_dict"][idx])
                    for idx, _ in enumerate(checkpoint["agent_ids"])
                ],
            ]

        agent.actor_optimizers = [
            optim.Adam(actor.parameters(), lr=agent.lr) for actor in agent.actors
        ]
        agent.critic_1_optimizers = [
            optim.Adam(critic_1.parameters(), lr=agent.lr)
            for critic_1 in agent.critics_1
        ]
        agent.critic_2_optimizers = [
            optim.Adam(critic_2.parameters(), lr=agent.lr)
            for critic_2 in agent.critics_2
        ]
        actor_list = []
        critic_1_list = []
        critic_2_list = []
        actor_target_list = []
        critic_target_1_list = []
        critic_target_2_list = []
        actor_optimizer_list = []
        critic_1_optimizer_list = []
        critic_2_optimizer_list = []

        for idx, (
            actor,
            actor_target,
            critic_1,
            critic_target_1,
            critic_2,
            critic_target_2,
            actor_optimizer,
            critic_1_optimizer,
            critic_2_optimizer,
        ) in enumerate(
            zip(
                agent.actors,
                agent.actor_targets,
                agent.critics_1,
                agent.critic_targets_1,
                agent.critics_2,
                agent.critic_targets_2,
                agent.actor_optimizers,
                agent.critic_1_optimizers,
                agent.critic_2_optimizers,
            )
        ):
            actor.load_state_dict(checkpoint["actors_state_dict"][idx])
            actor_list.append(actor)
            actor_target.load_state_dict(checkpoint["actor_targets_state_dict"][idx])
            actor_target_list.append(actor_target)
            critic_1.load_state_dict(checkpoint["critics_1_state_dict"][idx])
            critic_1_list.append(critic_1)
            critic_2.load_state_dict(checkpoint["critics_2_state_dict"][idx])
            critic_2_list.append(critic_2)
            critic_target_1.load_state_dict(
                checkpoint["critic_targets_1_state_dict"][idx]
            )
            critic_target_1_list.append(critic_target_1)
            critic_target_2.load_state_dict(
                checkpoint["critic_targets_2_state_dict"][idx]
            )
            critic_target_2_list.append(critic_target_2)
            actor_optimizer.load_state_dict(
                checkpoint["actor_optimizers_state_dict"][idx]
            )
            actor_optimizer_list.append(actor_optimizer)
            critic_1_optimizer.load_state_dict(
                checkpoint["critic_1_optimizers_state_dict"][idx]
            )
            critic_1_optimizer_list.append(critic_1_optimizer)
            critic_2_optimizer.load_state_dict(
                checkpoint["critic_2_optimizers_state_dict"][idx]
            )
            critic_2_optimizer_list.append(critic_2_optimizer)

        agent.actors = actor_list
        agent.actor_targets = actor_target_list
        agent.critics_1 = critic_1_list
        agent.critic_targets_1 = critic_target_1_list
        agent.critics_2 = critic_2_list
        agent.critic_targets_2 = critic_target_2_list
        agent.actor_optimizers = actor_optimizer_list
        agent.critic_1_optimizers = critic_1_optimizer_list
        agent.critic_2_optimizers = critic_2_optimizer_list

        if accelerator is not None:
            agent.wrap_models()

        agent.scores = checkpoint["scores"]
        agent.fitness = checkpoint["fitness"]
        agent.steps = checkpoint["steps"]

        return agent
