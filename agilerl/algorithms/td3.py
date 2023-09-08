import copy
import random

import dill
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from agilerl.networks.evolvable_cnn import EvolvableCNN
from agilerl.networks.evolvable_mlp import EvolvableMLP


class TD3:
    """The TD3 algorithm class. TD3 paper: https://arxiv.org/abs/1802.09477

    :param state_dim: State observation dimension
    :type state_dim: int
    :param action_dim: Action dimension
    :type action_dim: int
    :param one_hot: One-hot encoding, used with discrete observation spaces
    :type one_hot: bool
    :param max_action: Upper bound of the action space
    :type max_action: float
    :param expl_noise: Standard deviation for Gaussian exploration noise
    :param expl_noise: float, optional
    :param index: Index to keep track of object instance during tournament selection and mutation, defaults to 0
    :type index: int, optional
    :param net_config: Network configuration, defaults to mlp with hidden size [64,64]
    :type net_config: dict, optional
    :param batch_size: Size of batched sample from replay buffer for learning, defaults to 64
    :type batch_size: int, optional
    :param lr: Learning rate for optimizer, defaults to 1e-4
    :type lr: float, optional
    :param learn_step: Learning frequency, defaults to 5
    :type learn_step: int, optional
    :param gamma: Discount factor, defaults to 0.99
    :type gamma: float, optional
    :param tau: For soft update of target network parameters, defaults to 1e-3
    :type tau: float, optional
    :param mutation: Most recent mutation to agent, defaults to None
    :type mutation: str, optional
    :param policy_freq: Frequency of target network updates compared to policy network, defaults to 2
    :type policy_freq: int, optional
    :param device: Device for accelerated computing, 'cpu' or 'cuda', defaults to 'cpu'
    :type device: str, optional
    :param accelerator: Accelerator for distributed computing, defaults to None
    :type accelerator: Hugging Face accelerate.Accelerator(), optional
    :param wrap: Wrap models for distributed training upon creation, defaults to True
    :type wrap: bool, optional
    """

    def __init__(
        self,
        state_dim,
        action_dim,
        one_hot,
        max_action,
        expl_noise=0.1,
        index=0,
        net_config={"arch": "mlp", "h_size": [64, 64]},
        batch_size=64,
        lr=1e-4,
        learn_step=5,
        gamma=0.99,
        tau=0.005,
        mutation=None,
        policy_freq=2,
        device="cpu",
        accelerator=None,
        wrap=True,
    ):
        self.algo = "TD3"
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.one_hot = one_hot
        self.net_config = net_config
        self.batch_size = batch_size
        self.lr = lr
        self.learn_step = learn_step
        self.gamma = gamma
        self.tau = tau
        self.mut = mutation
        self.policy_freq = policy_freq
        self.max_action = max_action
        self.expl_noise = expl_noise
        self.device = device
        self.accelerator = accelerator

        self.index = index
        self.scores = []
        self.fitness = []
        self.steps = [0]

        # model
        # TD3 employs two critic networks
        if self.net_config["arch"] == "mlp":  # Multi-layer Perceptron
            self.actor = EvolvableMLP(
                num_inputs=state_dim[0],
                num_outputs=action_dim,
                hidden_size=self.net_config["h_size"],
                output_activation="tanh",
                device=self.device,
                accelerator=self.accelerator,
            )
            self.actor_target = EvolvableMLP(
                num_inputs=state_dim[0],
                num_outputs=action_dim,
                hidden_size=self.net_config["h_size"],
                output_activation="tanh",
                device=self.device,
                accelerator=self.accelerator,
            )
            self.actor_target.load_state_dict(self.actor.state_dict())

            self.critic_1 = EvolvableMLP(
                num_inputs=state_dim[0] + action_dim,
                num_outputs=1,
                hidden_size=self.net_config["h_size"],
                device=self.device,
                accelerator=self.accelerator,
            )
            self.critic_target_1 = EvolvableMLP(
                num_inputs=state_dim[0] + action_dim,
                num_outputs=1,
                hidden_size=self.net_config["h_size"],
                device=self.device,
                accelerator=self.accelerator,
            )

            self.critic_2 = EvolvableMLP(
                num_inputs=state_dim[0] + action_dim,
                num_outputs=1,
                hidden_size=self.net_config["h_size"],
                device=self.device,
                accelerator=self.accelerator,
            )
            self.critic_target_2 = EvolvableMLP(
                num_inputs=state_dim[0] + action_dim,
                num_outputs=1,
                hidden_size=self.net_config["h_size"],
                device=self.device,
                accelerator=self.accelerator,
            )

            self.critic_target_1.load_state_dict(self.critic_1.state_dict())
            self.critic_target_2.load_state_dict(self.critic_2.state_dict())

        elif self.net_config["arch"] == "cnn":  # Convolutional Neural Network
            self.actor = EvolvableCNN(
                input_shape=state_dim,
                num_actions=action_dim,
                channel_size=self.net_config["c_size"],
                kernel_size=self.net_config["k_size"],
                stride_size=self.net_config["s_size"],
                hidden_size=self.net_config["h_size"],
                normalize=self.net_config["normalize"],
                mlp_activation="tanh",
                device=self.device,
                accelerator=self.accelerator,
            )
            self.actor_target = EvolvableCNN(
                input_shape=state_dim,
                num_actions=action_dim,
                channel_size=self.net_config["c_size"],
                kernel_size=self.net_config["k_size"],
                stride_size=self.net_config["s_size"],
                hidden_size=self.net_config["h_size"],
                normalize=self.net_config["normalize"],
                mlp_activation="tanh",
                device=self.device,
                accelerator=self.accelerator,
            )
            self.actor_target.load_state_dict(self.actor.state_dict())

            self.critic_1 = EvolvableCNN(
                input_shape=state_dim,
                num_actions=action_dim,
                channel_size=self.net_config["c_size"],
                kernel_size=self.net_config["k_size"],
                stride_size=self.net_config["s_size"],
                hidden_size=self.net_config["h_size"],
                normalize=self.net_config["normalize"],
                mlp_activation="tanh",
                critic=True,
                device=self.device,
                accelerator=self.accelerator,
            )
            self.critic_target_1 = EvolvableCNN(
                input_shape=state_dim,
                num_actions=action_dim,
                channel_size=self.net_config["c_size"],
                kernel_size=self.net_config["k_size"],
                stride_size=self.net_config["s_size"],
                hidden_size=self.net_config["h_size"],
                normalize=self.net_config["normalize"],
                mlp_activation="tanh",
                critic=True,
                device=self.device,
                accelerator=self.accelerator,
            )

            self.critic_2 = EvolvableCNN(
                input_shape=state_dim,
                num_actions=action_dim,
                channel_size=self.net_config["c_size"],
                kernel_size=self.net_config["k_size"],
                stride_size=self.net_config["s_size"],
                hidden_size=self.net_config["h_size"],
                normalize=self.net_config["normalize"],
                mlp_activation="tanh",
                critic=True,
                device=self.device,
                accelerator=self.accelerator,
            )
            self.critic_target_2 = EvolvableCNN(
                input_shape=state_dim,
                num_actions=action_dim,
                channel_size=self.net_config["c_size"],
                kernel_size=self.net_config["k_size"],
                stride_size=self.net_config["s_size"],
                hidden_size=self.net_config["h_size"],
                normalize=self.net_config["normalize"],
                mlp_activation="tanh",
                critic=True,
                device=self.device,
                accelerator=self.accelerator,
            )

            self.critic_target_1.load_state_dict(self.critic_1.state_dict())
            self.critic_target_2.load_state_dict(self.critic_2.state_dict())

        self.actor_optimizer_type = optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic_1_optimizer_type = optim.Adam(
            self.critic_1.parameters(), lr=self.lr
        )
        self.critic_2_optimizer_type = optim.Adam(
            self.critic_2.parameters(), lr=self.lr
        )

        if self.accelerator is not None:
            self.actor_optimizer = self.actor_optimizer_type
            self.critic_1_optimizer = self.critic_1_optimizer_type
            self.critic_2_optimizer = self.critic_2_optimizer_type
            if wrap:
                self.wrap_models()
        else:
            self.actor = self.actor.to(self.device)
            self.actor_target = self.actor_target.to(self.device)
            self.critic_1 = self.critic_1.to(self.device)
            self.critic_target_1 = self.critic_target_1.to(self.device)
            self.critic_2 = self.critic_2.to(self.device)
            self.critic_target_2 = self.critic_target_2.to(self.device)
            self.actor_optimizer = self.actor_optimizer_type
            self.critic_1_optimizer = self.critic_1_optimizer_type
            self.critic_2_optimizer = self.critic_2_optimizer_type

        self.criterion = nn.MSELoss()

    def getAction(self, state, epsilon=0):
        """Returns the next action to take in the environment, noise is added to aid exploration.
        Epsilon is the probability of taking a random action, used for exploration.
        For epsilon-greedy behaviour, set epsilon to 0.

        :param state: Environment observation, or multiple observations in a batch
        :type state: float or List[float]
        :param epsilon: Probablilty of taking a random action for exploration, defaults to 0
        :type epsilon: float, optional
        """
        state = torch.from_numpy(state).float().to(self.device)

        if self.one_hot:
            state = (
                nn.functional.one_hot(state.long(), num_classes=self.state_dim[0])
                .float()
                .squeeze()
            )

        if len(state.size()) < 2:
            state = state.unsqueeze(0)

        # epsilon-greedy, Gaussian noise added to aid exploration
        if random.random() < epsilon:
            action = (
                np.random.rand(state.size()[0], self.action_dim).astype("float32") - 0.5
            ) * 2
        else:
            self.actor.eval()
            with torch.no_grad():
                action_values = self.actor(state)
            self.actor.train()

            action = action_values.cpu().data.numpy() + np.random.normal(
                0, self.max_action * self.expl_noise, size=self.action_dim
            ).astype(np.float32).clip(-self.max_action, self.max_action)
        return action

    def learn(self, experiences, noise_clip=0.5, policy_noise=0.2):
        """Updates agent network parameters to learn from experiences.

        :param experience: List of batched states, actions, rewards, next_states, dones in that order.
        :type experience: List[torch.Tensor[float]]
        :param noise_clip: Maximum noise limit to apply to actions, defaults to 0.5
        :type noise_clip: float, optional
        :param policy_noise: Standard deviation of noise applied to policy, defaults to 0.2
        :type policy_noise: float, optional
        """
        states, actions, rewards, next_states, dones = experiences

        if self.one_hot:
            states = (
                nn.functional.one_hot(states.long(), num_classes=self.state_dim[0])
                .float()
                .squeeze()
            )
            next_states = (
                nn.functional.one_hot(next_states.long(), num_classes=self.state_dim[0])
                .float()
                .squeeze()
            )

        if self.net_config["arch"] == "mlp":
            input_combined = torch.cat([states, actions], 1)
            q_value_1 = self.critic_1(input_combined)
            q_value_2 = self.critic_2(input_combined)
        elif self.net_config["arch"] == "cnn":
            q_value_1 = self.critic_1(states, actions)
            q_value_2 = self.critic_2(states, actions)

        next_actions = self.actor_target(next_states)
        noise = actions.data.normal_(0, policy_noise).to(self.device)
        noise = noise.clamp(-noise_clip, noise_clip)
        next_actions = next_actions + noise

        # Compute the target, y_j, making use of twin critic networks
        if self.net_config["arch"] == "mlp":
            next_input_combined = torch.cat([next_states, next_actions], 1)
            q_value_next_state_1 = self.critic_target_1(next_input_combined)
            q_value_next_state_2 = self.critic_target_2(next_input_combined)
        elif self.net_config["arch"] == "cnn":
            q_value_next_state_1 = self.critic_target_1(next_states, next_actions)
            q_value_next_state_2 = self.critic_target_2(next_states, next_actions)
        q_value_next_state = torch.min(q_value_next_state_1, q_value_next_state_2)
        y_j = rewards + ((1 - dones) * self.gamma * q_value_next_state).detach()

        # Loss equation needs to be updated to account for two q_values from two critics
        critic_loss = self.criterion(q_value_1, y_j) + self.criterion(q_value_2, y_j)

        # critic loss backprop
        self.critic_1_optimizer.zero_grad()
        self.critic_2_optimizer.zero_grad()
        if self.accelerator is not None:
            self.accelerator.backward(critic_loss)
        else:
            critic_loss.backward()
        self.critic_1_optimizer.step()
        self.critic_2_optimizer.step()

        # update actor and targets every policy_freq episodes
        if len(self.scores) % self.policy_freq == 0:
            if self.net_config["arch"] == "mlp":
                input_combined = torch.cat([states, self.actor.forward(states)], 1)
                actor_loss = -self.critic_1(input_combined).mean()
            elif self.net_config["arch"] == "cnn":
                actor_loss = -self.critic_1(states, self.actor.forward(states)).mean()

            # actor loss backprop
            self.actor_optimizer.zero_grad()
            if self.accelerator is not None:
                self.accelerator.backward(actor_loss)
            else:
                actor_loss.backward()
            self.actor_optimizer.step()

            # Add in a soft update for both critic_targets
            self.softUpdate(self.actor, self.actor_target)
            self.softUpdate(self.critic_1, self.critic_target_1)
            self.softUpdate(self.critic_2, self.critic_target_2)

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
                state = env.reset()[0]
                score = 0
                for idx_step in range(max_steps):
                    if swap_channels:
                        state = np.moveaxis(state, [3], [1])
                    action = self.getAction(state, epsilon=0)
                    state, reward, done, _, _ = env.step(action)
                    score += reward
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
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            one_hot=self.one_hot,
            max_action=self.max_action,
            expl_noise=self.expl_noise,
            index=index,
            net_config=self.net_config,
            batch_size=self.batch_size,
            lr=self.lr,
            learn_step=self.learn_step,
            gamma=self.gamma,
            tau=self.tau,
            mutation=self.mut,
            policy_freq=self.policy_freq,
            device=self.device,
            accelerator=self.accelerator,
            wrap=wrap,
        )

        if self.accelerator is not None:
            self.unwrap_models()
        actor = self.actor.clone()
        actor_target = self.actor_target.clone()
        critic_1 = self.critic_1.clone()
        critic_target_1 = self.critic_target_1.clone()
        critic_2 = self.critic_2.clone()
        critic_target_2 = self.critic_target_2.clone()

        actor_optimizer = optim.Adam(clone.actor.parameters(), lr=clone.lr)
        critic_1_optimizer = optim.Adam(clone.critic_1.parameters(), lr=clone.lr)
        critic_2_optimizer = optim.Adam(clone.critic_2.parameters(), lr=clone.lr)

        clone.actor_optimizer_type = actor_optimizer
        clone.critic_1_optimizer_type = critic_1_optimizer
        clone.critic_2_optimizer_type = critic_2_optimizer

        if self.accelerator is not None:
            if wrap:
                (
                    clone.actor,
                    clone.actor_target,
                    clone.critic_1,
                    clone.critic_target_1,
                    clone.critic_2,
                    clone.critic_target_2,
                    clone.actor_optimizer,
                    clone.critic_1_optimizer,
                    clone.critic_2_optimizer,
                ) = self.accelerator.prepare(
                    actor,
                    actor_target,
                    critic_1,
                    critic_target_1,
                    critic_2,
                    critic_target_2,
                    actor_optimizer,
                    critic_1_optimizer,
                    critic_2_optimizer,
                )
            else:
                (
                    clone.actor,
                    clone.actor_target,
                    clone.critic_1,
                    clone.critic_target_1,
                    clone.critic_2,
                    clone.critic_target_2,
                    clone.actor_optimizer,
                    clone.critic_1_optimizer,
                    clone.critic_1_optimizer,
                ) = (
                    actor,
                    actor_target,
                    critic_1,
                    critic_target_1,
                    critic_2,
                    critic_target_2,
                    actor_optimizer,
                    critic_1_optimizer,
                    critic_2_optimizer,
                )
        else:
            clone.actor = actor.to(self.device)
            clone.actor_target = actor_target.to(self.device)
            clone.critic_1 = critic_1.to(self.device)
            clone.critic_target_1 = critic_target_1.to(self.device)
            clone.critic_2 = critic_2.to(self.device)
            clone.critic_target_2 = critic_target_2.to(self.device)
            clone.actor_optimizer = actor_optimizer
            clone.critic_1_optimizer = critic_1_optimizer
            clone.critic_2_optimizer = critic_2_optimizer

        clone.fitness = copy.deepcopy(self.fitness)
        clone.steps = copy.deepcopy(self.steps)
        clone.scores = copy.deepcopy(self.scores)

        return clone

    def wrap_models(self):
        if self.accelerator is not None:
            (
                self.actor,
                self.actor_target,
                self.critic_1,
                self.critic_target_1,
                self.critic_2,
                self.critic_target_2,
                self.actor_optimizer,
                self.critic_1_optimizer,
                self.critic_2_optimizer,
            ) = self.accelerator.prepare(
                self.actor,
                self.actor_target,
                self.critic_1,
                self.critic_target_1,
                self.critic_2,
                self.critic_target_2,
                self.actor_optimizer_type,
                self.critic_1_optimizer_type,
                self.critic_2_optimizer_type,
            )

    def unwrap_models(self):
        if self.accelerator is not None:
            self.actor = self.accelerator.unwrap_model(self.actor)
            self.actor_target = self.accelerator.unwrap_model(self.actor_target)
            self.critic_1 = self.accelerator.unwrap_model(self.critic_1)
            self.critic_target_1 = self.accelerator.unwrap_model(self.critic_target_1)
            self.critic_2 = self.accelerator.unwrap_model(self.critic_2)
            self.critic_target_2 = self.accelerator.unwrap_model(self.critic_target_2)
            self.actor_optimizer = self.accelerator.unwrap_model(self.actor_optimizer)
            self.critic_1_optimizer = self.accelerator.unwrap_model(
                self.critic_1_optimizer
            )
            self.critic_2_optimizer = self.accelerator.unwrap_model(
                self.critic_2_optimizer
            )

    def saveCheckpoint(self, path):
        """Saves a checkpoint of agent properties and network weights to path.

        :param path: Location to save checkpoint at
        :type path: string
        """
        torch.save(
            {
                "actor_init_dict": self.actor.init_dict,
                "actor_state_dict": self.actor.state_dict(),
                "actor_target_init_dict": self.actor_target.init_dict,
                "actor_target_state_dict": self.actor_target.state_dict(),
                "critic_1_init_dict": self.critic_1.init_dict,
                "critic_1_state_dict": self.critic_1.state_dict(),
                "critic_target_1_init_dict": self.critic_target_1.init_dict,
                "critic_target_1_state_dict": self.critic_target_1.state_dict(),
                "critic_2_init_dict": self.critic_2.init_dict,
                "critic_2_state_dict": self.critic_2.state_dict(),
                "critic_target_2_init_dict": self.critic_target_2.init_dict,
                "critic_target_2_state_dict": self.critic_target_2.state_dict(),
                "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
                "critic_1_optimizer_state_dict": self.critic_1_optimizer.state_dict(),
                "critic_2_optimizer_state_dict": self.critic_2_optimizer.state_dict(),
                "net_config": self.net_config,
                "batch_size": self.batch_size,
                "lr": self.lr,
                "learn_step": self.learn_step,
                "gamma": self.gamma,
                "tau": self.tau,
                "mutation": self.mut,
                "max_action": self.max_action,
                "expl_noise": self.expl_noise,
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
            self.actor = EvolvableMLP(**checkpoint["actor_init_dict"])
            self.actor_target = EvolvableMLP(**checkpoint["actor_target_init_dict"])
            self.critic_1 = EvolvableMLP(**checkpoint["critic_1_init_dict"])
            self.critic_target_1 = EvolvableMLP(
                **checkpoint["critic_target_1_init_dict"]
            )
            self.critic_2 = EvolvableMLP(**checkpoint["critic_2_init_dict"])
            self.critic_target_2 = EvolvableMLP(
                **checkpoint["critic_target_2_init_dict"]
            )
        elif self.net_config["arch"] == "cnn":
            self.actor = EvolvableCNN(**checkpoint["actor_init_dict"])
            self.actor_target = EvolvableCNN(**checkpoint["actor_target_init_dict"])
            self.critic_1 = EvolvableCNN(**checkpoint["critic_1_init_dict"])
            self.critic_target_1 = EvolvableCNN(
                **checkpoint["critic_target_1_init_dict"]
            )
            self.critic_2 = EvolvableCNN(**checkpoint["critic_2_init_dict"])
            self.critic_target_2 = EvolvableCNN(
                **checkpoint["critic_target_2_init_dict"]
            )
        self.lr = checkpoint["lr"]
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic_1_optimizer = optim.Adam(self.critic_1.parameters(), lr=self.lr)
        self.critic_2_optimizer = optim.Adam(self.critic_2.parameters(), lr=self.lr)
        self.actor.load_state_dict(checkpoint["actor_state_dict"])
        self.actor_target.load_state_dict(checkpoint["actor_target_state_dict"])
        self.critic_1.load_state_dict(checkpoint["critic_1_state_dict"])
        self.critic_target_1.load_state_dict(checkpoint["critic_target_1_state_dict"])
        self.critic_2.load_state_dict(checkpoint["critic_2_state_dict"])
        self.critic_target_2.load_state_dict(checkpoint["critic_target_2_state_dict"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer_state_dict"])
        self.critic_1_optimizer.load_state_dict(
            checkpoint["critic_1_optimizer_state_dict"]
        )
        self.critic_2_optimizer.load_state_dict(
            checkpoint["critic_2_optimizer_state_dict"]
        )
        self.batch_size = checkpoint["batch_size"]
        self.learn_step = checkpoint["learn_step"]
        self.gamma = checkpoint["gamma"]
        self.tau = checkpoint["tau"]
        self.mut = checkpoint["mutation"]
        self.max_action = checkpoint["max_action"]
        self.expl_noise = checkpoint["expl_noise"]
        self.index = checkpoint["index"]
        self.scores = checkpoint["scores"]
        self.fitness = checkpoint["fitness"]
        self.steps = checkpoint["steps"]
