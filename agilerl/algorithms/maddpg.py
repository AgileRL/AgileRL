import copy
import random

import dill
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from agilerl.networks.evolvable_cnn import EvolvableCNN
from agilerl.networks.evolvable_mlp import EvolvableMLP


class MADDPG:
    """The MADDPG algorithm class. MADDPG paper: https://arxiv.org/abs/1706.02275

    :param state_dims: State observation dimensions for each agent
    :type state_dims: List[tuple]
    :param action_dims: Action dimensions for each agent
    :type action_dims: List[int]
    :param one_hot: One-hot encoding, used with discrete observation spaces
    :type one_hot: bool
    :param n_agents: Number of agents
    :type n_agents: int
    :param agent_ids: Agent ID for each agent
    :type agent_ids: List[str]
    :param max_action: Upper bound of the action space
    :type max_action: float
    :param min_action: Lower bound of the action space
    :type min_action: float
    :param discrete_actions: Boolean flag to indicate a discrete action space
    :type discrete_actions: bool, optional
    :param expl_noise: Standard deviation for Gaussian exploration noise, defaults to 0.1
    :type expl_noise: float, optional
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
    :param device: Device for accelerated computing, 'cpu' or 'cuda', defaults to 'cpu'
    :type device: str, optional
    :param accelerator: Accelerator for distributed computing, defaults to None
    :type accelerator: Hugging Face accelerate.Accelerator(), optional
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
        net_config={"arch": "mlp", "h_size": [64, 64]},
        batch_size=64,
        lr=0.01,
        learn_step=5,
        gamma=0.95,
        tau=0.01,
        mutation=None,
        device="cpu",
        accelerator=None,
        wrap=True,
    ):
        self.algo = "MADDPG"
        self.state_dims = state_dims
        self.total_state_dims = sum(state_dim[0] for state_dim in self.state_dims)
        self.action_dims = action_dims
        self.one_hot = one_hot
        self.n_agents = n_agents
        self.multi = True if n_agents > 1 else False
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
        self.scores = []
        self.fitness = []
        self.steps = [0]
        self.max_action = max_action
        self.expl_noise = expl_noise
        self.min_action = min_action
        self.discrete_actions = discrete_actions
        self.total_actions = (
            sum(self.action_dims)
            if not self.discrete_actions
            else len(self.action_dims)
        )

        if "output_activation" in self.net_config.keys():
            pass
        else:
            if self.discrete_actions:
                self.net_config["output_activation"] = "gumbel_softmax"
            else:
                self.net_config["output_activation"] = "softmax"

        # model
        if self.net_config["arch"] == "mlp":  # Multi-layer Perceptron
            self.actors = [
                EvolvableMLP(
                    num_inputs=state_dim[0],
                    num_outputs=action_dim,
                    hidden_size=self.net_config["h_size"],
                    output_activation=self.net_config["output_activation"],
                    device=self.device,
                    accelerator=self.accelerator,
                )
                for (action_dim, state_dim) in zip(self.action_dims, self.state_dims)
            ]
            self.actor_targets = copy.deepcopy(self.actors)

            self.critics = [
                EvolvableMLP(
                    num_inputs=self.total_state_dims + self.total_actions,
                    num_outputs=1,
                    hidden_size=self.net_config["h_size"],
                    device=self.device,
                    accelerator=self.accelerator,
                )
                for _ in range(self.n_agents)
            ]
            self.critic_targets = copy.deepcopy(self.critics)

        elif self.net_config["arch"] == "cnn":  # Convolutional Neural Network
            self.actors = [
                EvolvableCNN(
                    input_shape=state_dim,
                    num_actions=action_dim,
                    channel_size=self.net_config["c_size"],
                    kernel_size=self.net_config["k_size"],
                    stride_size=self.net_config["s_size"],
                    hidden_size=self.net_config["h_size"],
                    normalize=self.net_config["normalize"],
                    mlp_activation=self.net_config["output_activation"],
                    multi=self.multi,
                    n_agents=self.n_agents,
                    device=self.device,
                    accelerator=self.accelerator,
                )
                for (action_dim, state_dim) in zip(self.action_dims, self.state_dims)
            ]
            self.actor_targets = copy.deepcopy(self.actors)

            self.critics = [
                EvolvableCNN(
                    input_shape=state_dim,
                    num_actions=self.total_actions,
                    channel_size=self.net_config["c_size"],
                    kernel_size=self.net_config["k_size"],
                    stride_size=self.net_config["s_size"],
                    hidden_size=self.net_config["h_size"],
                    normalize=self.net_config["normalize"],
                    mlp_activation="tanh",
                    critic=True,
                    n_agents=self.n_agents,
                    multi=self.multi,
                    device=self.device,
                    accelerator=self.accelerator,
                )
                for state_dim in self.state_dims
            ]
            self.critic_targets = copy.deepcopy(self.critics)

        self.actor_optimizers_type = [
            optim.Adam(actor.parameters(), lr=self.lr) for actor in self.actors
        ]
        self.critic_optimizers_type = [
            optim.Adam(critic.parameters(), lr=self.lr) for critic in self.critics
        ]

        if self.accelerator is not None:
            self.actor_optimizers = self.actor_optimizers_type
            self.critic_optimizers = self.critic_optimizers_type
            if wrap:
                self.wrap_models()
        else:
            self.actors = [actor.to(self.device) for actor in self.actors]
            self.actor_targets = [
                actor_target.to(self.device) for actor_target in self.actor_targets
            ]
            self.critics = [critic.to(self.device) for critic in self.critics]
            self.critic_targets = [
                critic_target.to(self.device) for critic_target in self.critic_targets
            ]
            self.actor_optimizers = self.actor_optimizers_type
            self.critic_optimizers = self.critic_optimizers_type

        self.criterion = nn.MSELoss()

    def getAction(self, states, epsilon=0):
        """Returns the next action to take in the environment.
        Epsilon is the probability of taking a random action, used for exploration.
        For epsilon-greedy behaviour, set epsilon to 0.

        :param state: Environment observations: {'agent_0': state_dim_0, ..., 'agent_n': state_dim_n}
        :type state: Dict[str, numpy.Array]
        :param epsilon: Probablilty of taking a random action for exploration, defaults to 0
        :type epsilon: float, optional
        """
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
                for state, state_dim in zip(states, self.state_dims)
            ]

        if self.net_config["arch"] == "mlp":
            states = [
                state.unsqueeze(0) if len(state.size()) < 2 else state
                for state in states
            ]
        elif self.net_config["arch"] == "cnn":
            states = [state.unsqueeze(2) for state in states]

        actions = {}
        for idx, (agent_id, state, actor) in enumerate(
            zip(self.agent_ids, states, self.actors)
        ):
            if random.random() < epsilon:
                if self.discrete_actions:
                    action = np.random.randint(0, self.action_dims[idx])
                else:
                    action = (
                        np.random.rand(state.size()[0], self.action_dims[idx])
                        .astype("float32")
                        .squeeze()
                    )
            else:
                actor.eval()
                if self.accelerator is not None:
                    with actor.no_sync():
                        action_values = actor(state)
                else:
                    with torch.no_grad():
                        action_values = actor(state)
                actor.train()
                if self.discrete_actions:
                    action = action_values.squeeze(0).argmax().item()
                else:
                    action = (
                        action_values.cpu().data.numpy().squeeze()
                        + np.random.normal(
                            0,
                            self.max_action[idx][0] * self.expl_noise,
                            size=self.action_dims[idx],
                        ).astype(np.float32)
                    )
                    action = np.clip(
                        action, self.min_action[idx][0], self.max_action[idx][0]
                    )
            actions[agent_id] = action

        return actions

    def _squeeze_exp(self, experiences):
        """Remove first dim created by dataloader.

        :param experiences: List of batched states, actions, rewards, next_states, dones in that order.
        :type state: List[torch.Tensor[float]]
        """
        st, ac, re, ne, do = experiences
        return st.squeeze(0), ac.squeeze(0), re.squeeze(0), ne.squeeze(0), do.squeeze(0)

    def learn(self, experiences):
        """Updates agent network parameters to learn from experiences.

        :param experience: Tuple of dictionaries containing batched states, actions, rewards, next_states,
        dones in that order for each individual agent.
        :type experience: Tuple[Dict[str, torch.Tensor]]
        """

        for (
            agent_id,
            actor,
            actor_target,
            critic,
            critic_target,
            actor_optimizer,
            critic_optimizer,
        ) in zip(
            self.agent_ids,
            self.actors,
            self.actor_targets,
            self.critics,
            self.critic_targets,
            self.actor_optimizers,
            self.critic_optimizers,
        ):
            states, actions, rewards, next_states, dones = experiences

            if self.one_hot:
                states = {
                    agent_id: nn.functional.one_hot(
                        state.long(), num_classes=state_dim[0]
                    )
                    .float()
                    .squeeze()
                    for agent_id, state, state_dim in zip(
                        states.items(), self.state_dims
                    )
                }

            if self.net_config["arch"] == "mlp":
                if self.discrete_actions:
                    action_values = [a.unsqueeze(1) for a in actions.values()]
                else:
                    action_values = list(actions.values())
                input_combined = torch.cat(list(states.values()) + action_values, 1)
                if self.accelerator is not None:
                    with critic.no_sync():
                        q_value = critic(input_combined)
                else:
                    q_value = critic(input_combined)
                next_actions = [
                    self.actor_targets[idx](next_states[agent_id]).detach_()
                    for idx, agent_id in enumerate(self.agent_ids)
                ]

            elif self.net_config["arch"] == "cnn":
                stacked_states = torch.stack(list(states.values()), dim=2)
                stacked_actions = torch.stack(list(actions.values()), dim=1)
                if self.accelerator is not None:
                    with critic.no_sync():
                        q_value = critic(stacked_states, stacked_actions)
                else:
                    q_value = critic(stacked_states, stacked_actions)
                next_actions = [
                    self.actor_targets[idx](
                        next_states[agent_id].unsqueeze(2)
                    ).detach_()
                    for idx, agent_id in enumerate(self.agent_ids)
                ]

            if self.discrete_actions:
                next_actions = [
                    torch.argmax(agent_actions, dim=1).unsqueeze(1)
                    if self.net_config["arch"] == "mlp"
                    else torch.argmax(agent_actions, dim=1)
                    for agent_actions in next_actions
                ]

            if self.net_config["arch"] == "mlp":
                next_input_combined = torch.cat(
                    list(next_states.values()) + next_actions, 1
                )
                if self.accelerator is not None:
                    with critic_target.no_sync():
                        q_value_next_state = critic_target(next_input_combined)
                else:
                    q_value_next_state = critic_target(next_input_combined)
            elif self.net_config["arch"] == "cnn":
                stacked_next_states = torch.stack(list(next_states.values()), dim=2)
                stacked_next_actions = torch.stack(next_actions, dim=1)
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
                rewards[agent_id]
                + (1 - dones[agent_id]) * self.gamma * q_value_next_state
            )

            critic_loss = self.criterion(q_value, y_j.detach_())

            # critic loss backprop
            critic_optimizer.zero_grad()
            if self.accelerator is not None:
                self.accelerator.backward(critic_loss)
            else:
                critic_loss.backward()
            critic_optimizer.step()

            # update actor and targets
            if self.net_config["arch"] == "mlp":
                if self.accelerator is not None:
                    with actor.no_sync():
                        action = actor(states[agent_id])
                else:
                    action = actor(states[agent_id])
                detached_actions = copy.deepcopy(actions)
                if self.discrete_actions:
                    action = action.argmax(1).unsqueeze(1)
                    detached_actions = {
                        agent_id: d.unsqueeze(1)
                        for agent_id, d in detached_actions.items()
                    }
                detached_actions[agent_id] = action
                input_combined = torch.cat(
                    list(states.values()) + list(detached_actions.values()), 1
                )
                if self.accelerator is not None:
                    with critic.no_sync():
                        actor_loss = -critic(input_combined).mean()
                else:
                    actor_loss = -critic(input_combined).mean()

            elif self.net_config["arch"] == "cnn":
                if self.accelerator is not None:
                    with actor.no_sync():
                        action = actor(states[agent_id].unsqueeze(2))
                else:
                    action = actor(states[agent_id].unsqueeze(2))
                if self.discrete_actions:
                    action = action.argmax(1)
                detached_actions = copy.deepcopy(actions)
                detached_actions[agent_id] = action
                stacked_detached_actions = torch.stack(
                    list(detached_actions.values()), dim=1
                )
                if self.accelerator is not None:
                    with critic.no_sync():
                        actor_loss = -critic(
                            stacked_states, stacked_detached_actions
                        ).mean()
                else:
                    actor_loss = -critic(
                        stacked_states, stacked_detached_actions
                    ).mean()

            # actor loss backprop
            actor_optimizer.zero_grad()
            if self.accelerator is not None:
                self.accelerator.backward(actor_loss)
            else:
                actor_loss.backward()
            actor_optimizer.step()

        for actor, actor_target, critic, critic_target in zip(
            self.actors, self.actor_targets, self.critics, self.critic_targets
        ):
            self.softUpdate(actor, actor_target)
            self.softUpdate(critic, critic_target)

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
        :param loop: Number of testing loops/epsiodes to complete. The returned score is the mean. Defaults to 3
        :type loop: int, optional
        """
        with torch.no_grad():
            rewards = []
            for i in range(loop):
                state, _ = env.reset()
                agent_reward = {agent_id: 0 for agent_id in self.agent_ids}
                score = 0
                for _ in range(max_steps):
                    if swap_channels:
                        state = {
                            agent_id: np.moveaxis(np.expand_dims(s, 0), [3], [1])
                            for agent_id, s in state.items()
                        }
                    action = self.getAction(state, epsilon=0)
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
            lr=self.lr,
            learn_step=self.learn_step,
            gamma=self.gamma,
            tau=self.tau,
            mutation=self.mut,
            device=self.device,
            accelerator=self.accelerator,
            wrap=wrap,
        )

        if self.accelerator is not None:
            self.unwrap_models()
        actors = [actor.clone() for actor in self.actors]
        actor_targets = [actor_target.clone() for actor_target in self.actor_targets]
        critics = [critic.clone() for critic in self.critics]
        critic_targets = [
            critic_target.clone() for critic_target in self.critic_targets
        ]
        actor_optimizers = [
            optim.Adam(actor.parameters(), lr=clone.lr) for actor in actors
        ]
        critic_optimizers = [
            optim.Adam(critic.parameters(), lr=clone.lr) for critic in critics
        ]
        clone.actor_optimizers_type = actor_optimizers
        clone.critic_optimizers_type = critic_optimizers

        if self.accelerator is not None:
            if wrap:
                clone.actors = [self.accelerator.prepare(actor) for actor in actors]
                clone.actor_targets = [
                    self.accelerator.prepare(actor_target)
                    for actor_target in actor_targets
                ]
                clone.critics = [self.accelerator.prepare(critic) for critic in critics]
                clone.critic_targets = [
                    self.accelerator.prepare(critic_target)
                    for critic_target in critic_targets
                ]
                clone.actor_optimizers = [
                    self.accelerator.prepare(actor_optimizer)
                    for actor_optimizer in actor_optimizers
                ]
                clone.critic_optimizers = [
                    self.accelerator.prepare(critic_optimizer)
                    for critic_optimizer in critic_optimizers
                ]
            else:
                (
                    clone.actors,
                    clone.actor_targets,
                    clone.critics,
                    clone.critic_targets,
                    clone.actor_optimizer,
                    clone.critic_optimizer,
                ) = (
                    actors,
                    actor_targets,
                    critics,
                    critic_targets,
                    actor_optimizers,
                    critic_optimizers,
                )
        else:
            clone.actors = [actor.to(self.device) for actor in actors]
            clone.actor_targets = [
                actor_target.to(self.device) for actor_target in actor_targets
            ]
            clone.critics = [critic.to(self.device) for critic in critics]
            clone.critic_targets = [
                critic_target.to(self.device) for critic_target in critic_targets
            ]
            clone.actor_optimizers = actor_optimizers
            clone.critic_optimizers = critic_optimizers

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
            self.critics = [self.accelerator.prepare(critic) for critic in self.critics]
            self.critic_targets = [
                self.accelerator.prepare(critic_target)
                for critic_target in self.critic_targets
            ]
            self.actor_optimizers = [
                self.accelerator.prepare(actor_optimizer)
                for actor_optimizer in self.actor_optimizers_type
            ]
            self.critic_optimizers = [
                self.accelerator.prepare(critic_optimizer)
                for critic_optimizer in self.critic_optimizers_type
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
            self.critics = [
                self.accelerator.unwrap_model(critic) for critic in self.critics
            ]
            self.critic_targets = [
                self.accelerator.unwrap_model(critic_target)
                for critic_target in self.critic_targets
            ]
            self.actor_optimizers = [
                self.accelerator.unwrap_model(actor_optimizer)
                for actor_optimizer in self.actor_optimizers
            ]
            self.critic_optimizers = [
                self.accelerator.unwrap_model(critic_optimizer)
                for critic_optimizer in self.critic_optimizers
            ]

    def saveCheckpoint(self, path):
        """Saves a checkpoint of agent properties and network weights to path.

        :param path: Location to save checkpoint at
        :type path: string
        """

        torch.save(
            {
                "actors_init_dict": [actor.init_dict for actor in self.actors],
                "actors_state_dict": [actor.state_dict() for actor in self.actors],
                "actor_targets_init_dict": [
                    actor_target.init_dict for actor_target in self.actor_targets
                ],
                "actor_targets_state_dict": [
                    actor_target.state_dict() for actor_target in self.actor_targets
                ],
                "critics_init_dict": [critic.init_dict for critic in self.critics],
                "critics_state_dict": [critic.state_dict() for critic in self.critics],
                "critic_targets_init_dict": [
                    critic_target.init_dict for critic_target in self.critic_targets
                ],
                "critic_targets_state_dict": [
                    critic_target.state_dict() for critic_target in self.critic_targets
                ],
                "actor_optimizers_state_dict": [
                    actor_optimizer.state_dict()
                    for actor_optimizer in self.actor_optimizers
                ],
                "critic_optimizers_state_dict": [
                    critic_optimizer.state_dict()
                    for critic_optimizer in self.critic_optimizers
                ],
                "net_config": self.net_config,
                "batch_size": self.batch_size,
                "lr": self.lr,
                "learn_step": self.learn_step,
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
        if self.net_config["arch"] == "mlp":
            self.actors = [
                EvolvableMLP(**checkpoint["actors_init_dict"][idx])
                for idx, _ in enumerate(self.agent_ids)
            ]
            self.actor_targets = [
                EvolvableMLP(**checkpoint["actor_targets_init_dict"][idx])
                for idx, _ in enumerate(self.agent_ids)
            ]
            self.critics = [
                EvolvableMLP(**checkpoint["critics_init_dict"][idx])
                for idx, _ in enumerate(self.agent_ids)
            ]
            self.critic_targets = [
                EvolvableMLP(**checkpoint["critic_targets_init_dict"][idx])
                for idx, _ in enumerate(self.agent_ids)
            ]
        elif self.net_config["arch"] == "cnn":
            self.actors = [
                EvolvableCNN(**checkpoint["actors_init_dict"][idx])
                for idx, _ in enumerate(self.agent_ids)
            ]
            self.actor_targets = [
                EvolvableCNN(**checkpoint["actor_targets_init_dict"][idx])
                for idx, _ in enumerate(self.agent_ids)
            ]
            self.critics = [
                EvolvableCNN(**checkpoint["critics_init_dict"][idx])
                for idx, _ in enumerate(self.agent_ids)
            ]
            self.critic_targets = [
                EvolvableCNN(**checkpoint["critic_targets_init_dict"][idx])
                for idx, _ in enumerate(self.agent_ids)
            ]

        self.lr = checkpoint["lr"]
        self.actor_optimizers = [
            optim.Adam(actor.parameters(), lr=self.lr) for actor in self.actors
        ]
        self.critic_optimizers = [
            optim.Adam(critic.parameters(), lr=self.lr) for critic in self.critics
        ]
        actor_list = []
        critic_list = []
        actor_target_list = []
        critic_target_list = []
        actor_optimizer_list = []
        critic_optimizer_list = []

        for idx, (
            actor,
            actor_target,
            critic,
            critic_target,
            actor_optimizer,
            critic_optimizer,
        ) in enumerate(
            zip(
                self.actors,
                self.actor_targets,
                self.critics,
                self.critic_targets,
                self.actor_optimizers,
                self.critic_optimizers,
            )
        ):
            actor.load_state_dict(checkpoint["actors_state_dict"][idx])
            actor_list.append(actor)
            actor_target.load_state_dict(checkpoint["actor_targets_state_dict"][idx])
            actor_target_list.append(actor_target)
            critic.load_state_dict(checkpoint["critics_state_dict"][idx])
            critic_list.append(critic)
            critic_target.load_state_dict(checkpoint["critic_targets_state_dict"][idx])
            critic_target_list.append(critic_target)
            actor_optimizer.load_state_dict(
                checkpoint["actor_optimizers_state_dict"][idx]
            )
            actor_optimizer_list.append(actor_optimizer)
            critic_optimizer.load_state_dict(
                checkpoint["critic_optimizers_state_dict"][idx]
            )
            critic_optimizer_list.append(critic_optimizer)

        self.actors = actor_list
        self.actor_targets = actor_target_list
        self.critics = critic_list
        self.critic_targets = critic_target_list
        self.actor_optimizers = actor_optimizer_list
        self.critic_optimizers = critic_optimizer_list
        self.batch_size = checkpoint["batch_size"]
        self.learn_step = checkpoint["learn_step"]
        self.gamma = checkpoint["gamma"]
        self.tau = checkpoint["tau"]
        self.mut = checkpoint["mutation"]
        self.index = checkpoint["index"]
        self.scores = checkpoint["scores"]
        self.fitness = checkpoint["fitness"]
        self.steps = checkpoint["steps"]
