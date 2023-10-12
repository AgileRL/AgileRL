import copy
import random

import dill
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_

from agilerl.networks.evolvable_cnn import EvolvableCNN
from agilerl.networks.evolvable_mlp import EvolvableMLP


class CQN:
    """The CQN algorithm class. CQN paper: https://arxiv.org/abs/2006.04779

    :param state_dim: State observation dimension
    :type state_dim: int
    :param action_dim: Action dimension
    :type action_dim: int
    :param one_hot: One-hot encoding, used with discrete observation spaces
    :type one_hot: bool
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
    :param double: Use double Q-learning, defaults to False
    :type double: bool, optional
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
        index=0,
        net_config={"arch": "mlp", "h_size": [64, 64]},
        batch_size=64,
        lr=1e-4,
        learn_step=5,
        gamma=0.99,
        tau=1e-3,
        mutation=None,
        double=False,
        device="cpu",
        accelerator=None,
        wrap=True,
    ):
        self.algo = "CQN"
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
        self.device = device
        self.accelerator = accelerator

        self.index = index
        self.scores = []
        self.fitness = []
        self.steps = [0]

        self.double = double

        # model
        if self.net_config["arch"] == "mlp":  # Multi-layer Perceptron
            self.actor = EvolvableMLP(
                num_inputs=state_dim[0],
                num_outputs=action_dim,
                hidden_size=self.net_config["h_size"],
                device=self.device,
                accelerator=self.accelerator,
            )
            self.actor_target = EvolvableMLP(
                num_inputs=state_dim[0],
                num_outputs=action_dim,
                hidden_size=self.net_config["h_size"],
                device=self.device,
                accelerator=self.accelerator,
            )
            self.actor_target.load_state_dict(self.actor.state_dict())

        elif self.net_config["arch"] == "cnn":  # Convolutional Neural Network
            self.actor = EvolvableCNN(
                input_shape=state_dim,
                num_actions=action_dim,
                channel_size=self.net_config["c_size"],
                kernel_size=self.net_config["k_size"],
                stride_size=self.net_config["s_size"],
                hidden_size=self.net_config["h_size"],
                normalize=self.net_config["normalize"],
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
                device=self.device,
                accelerator=self.accelerator,
            )
            self.actor_target.load_state_dict(self.actor.state_dict())

        self.optimizer_type = optim.Adam(self.actor.parameters(), lr=self.lr)

        if self.accelerator is not None:
            self.optimizer = self.optimizer_type
            if wrap:
                self.wrap_models()
        else:
            self.actor = self.actor.to(self.device)
            self.actor_target = self.actor_target.to(self.device)
            self.optimizer = self.optimizer_type

        self.criterion = nn.MSELoss()

    def getAction(self, state, epsilon=0, action_mask=None):
        """Returns the next action to take in the environment. Epsilon is the
        probability of taking a random action, used for exploration.
        For epsilon-greedy behaviour, set epsilon to 0.

        :param state: State observation, or multiple observations in a batch
        :type state: float or List[float]
        :param epsilon: Probablilty of taking a random action for exploration, defaults to 0
        :type epsilon: float, optional
        :param action_mask: Mask of legal actions 1=legal 0=illegal, defaults to None
        :type action_mask: List, optional
        """
        state = torch.from_numpy(state).float()
        if self.accelerator is None:
            state = state.to(self.device)

        if self.one_hot:
            state = (
                nn.functional.one_hot(state.long(), num_classes=self.state_dim[0])
                .float()
                .squeeze()
            )

        if len(state.size()) < 2:
            state = state.unsqueeze(0)

        # epsilon-greedy
        if random.random() < epsilon:
            if action_mask is None:
                action = np.random.randint(0, self.action_dim, size=state.size()[0])
            else:
                inv_mask = 1 - action_mask

                available_actions = np.ma.array(
                    np.arange(0, self.action_dim), mask=inv_mask
                ).compressed()
                action = np.random.choice(available_actions, size=state.size()[0])
        else:
            self.actor.eval()
            with torch.no_grad():
                action_values = self.actor(state)
            self.actor.train()
            if action_mask is None:
                action = np.argmax(action_values.cpu().data.numpy(), axis=-1)
            else:
                inv_mask = 1 - action_mask
                masked_action_values = np.ma.array(
                    action_values.cpu().data.numpy(), mask=inv_mask
                )
                action = np.argmax(masked_action_values, axis=-1)
        return action

    def _squeeze_exp(self, experiences):
        """Remove first dim created by dataloader.

        :param experiences: List of batched states, actions, rewards, next_states, dones in that order.
        :type state: List[torch.Tensor[float]]
        """
        st, ac, re, ne, do = experiences
        return st.squeeze(0), ac.squeeze(0), re.squeeze(0), ne.squeeze(0), do.squeeze(0)

    def learn(self, experiences):
        """Updates agent network parameters to learn from experiences.

        :param experiences: List of batched states, actions, rewards, next_states, dones in that order.
        :type state: List[torch.Tensor[float]]
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

        if self.double:  # Double Q-learning
            q_idx = self.actor_target(next_states).argmax(dim=1).unsqueeze(1)
            q_target_next = self.actor(next_states).gather(dim=1, index=q_idx).detach()
        else:
            q_target_next = (
                self.actor_target(next_states).detach().max(axis=1)[0].unsqueeze(1)
            )

        # target, if terminal then y_j = rewards
        q_target = rewards + self.gamma * q_target_next * (1 - dones)
        q_a_s = self.actor(states)
        q_eval = q_a_s.gather(1, actions.long())

        # loss backprop
        cql1_loss = torch.logsumexp(q_a_s, dim=1).mean() - q_a_s.mean()
        loss = self.criterion(q_eval, q_target)
        q1_loss = cql1_loss + 0.5 * loss
        self.optimizer.zero_grad()
        if self.accelerator is not None:
            self.accelerator.backward(q1_loss)
        else:
            q1_loss.backward()
        clip_grad_norm_(self.actor.parameters(), 1)
        self.optimizer.step()

        # soft update target network
        self.softUpdate()

    def softUpdate(self):
        """Soft updates target network."""
        for eval_param, target_param in zip(
            self.actor.parameters(), self.actor_target.parameters()
        ):
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
                    if done:
                        break
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

        actor = self.actor.clone()
        actor_target = self.actor_target.clone()
        optimizer = optim.Adam(actor.parameters(), lr=clone.lr)
        clone.optimizer_type = optimizer
        if self.accelerator is not None:
            if wrap:
                (
                    clone.actor,
                    clone.actor_target,
                    clone.optimizer,
                ) = self.accelerator.prepare(actor, actor_target, optimizer)
            else:
                clone.actor, clone.actor_target, clone.optimizer = (
                    actor,
                    actor_target,
                    optimizer,
                )
        else:
            clone.actor = actor.to(self.device)
            clone.actor_target = actor_target.to(self.device)
            clone.optimizer = optimizer
        clone.fitness = copy.deepcopy(self.fitness)
        clone.steps = copy.deepcopy(self.steps)
        clone.scores = copy.deepcopy(self.scores)

        return clone

    def wrap_models(self):
        if self.accelerator is not None:
            self.actor, self.actor_target, self.optimizer = self.accelerator.prepare(
                self.actor, self.actor_target, self.optimizer
            )

    def unwrap_models(self):
        if self.accelerator is not None:
            self.actor = self.accelerator.unwrap_model(self.actor)
            self.actor_target = self.accelerator.unwrap_model(self.actor_target)
            self.optimizer = self.accelerator.unwrap_model(self.optimizer)

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
                "optimizer_state_dict": self.optimizer.state_dict(),
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
            self.actor = EvolvableMLP(**checkpoint["actor_init_dict"])
            self.actor_target = EvolvableMLP(**checkpoint["actor_target_init_dict"])
        elif self.net_config["arch"] == "cnn":
            self.actor = EvolvableCNN(**checkpoint["actor_init_dict"])
            self.actor_target = EvolvableCNN(**checkpoint["actor_target_init_dict"])
        self.lr = checkpoint["lr"]
        self.optimizer = optim.Adam(self.actor.parameters(), lr=self.lr)
        self.actor.load_state_dict(checkpoint["actor_state_dict"])
        self.actor_target.load_state_dict(checkpoint["actor_target_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.batch_size = checkpoint["batch_size"]
        self.learn_step = checkpoint["learn_step"]
        self.gamma = checkpoint["gamma"]
        self.tau = checkpoint["tau"]
        self.mut = checkpoint["mutation"]
        self.index = checkpoint["index"]
        self.scores = checkpoint["scores"]
        self.fitness = checkpoint["fitness"]
        self.steps = checkpoint["steps"]
