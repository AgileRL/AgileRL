import copy

import dill
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_

from agilerl.networks.evolvable_cnn import EvolvableCNN
from agilerl.networks.evolvable_mlp import EvolvableMLP


class RainbowDQN:
    """The Rainbow DQN algorithm class. Rainbow DQN paper: https://arxiv.org/abs/1710.02298

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
    :param beta: Importance sampling coefficient, defaults to 0.4
    :type beta: float, optional
    :param prior_eps: Minimum priority for sampling, defaults to 1e-6
    :type prior_eps: float, optional
    :param num_atoms: Unit number of support, defaults to 51
    :type num_atoms: int, optional
    :param v_min: Minimum value of support, defaults to 0
    :type v_min: float, optional
    :param v_max: Maximum value of support, defaults to 200
    :type v_max: float, optional
    :param n_step: Step number to calculate n-step td error, defaults to 3
    :type n_step: int, optional
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
        beta=0.4,
        prior_eps=1e-6,
        num_atoms=51,
        v_min=0.0,
        v_max=200.0,
        n_step=3,
        mutation=None,
        device="cpu",
        accelerator=None,
        wrap=True,
    ):
        self.algo = "Rainbow DQN"
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.one_hot = one_hot
        self.net_config = net_config
        self.batch_size = batch_size
        self.lr = lr
        self.learn_step = learn_step
        self.gamma = gamma
        self.tau = tau
        self.beta = beta
        self.prior_eps = prior_eps
        self.num_atoms = num_atoms
        self.v_min = v_min
        self.v_max = v_max
        self.n_step = n_step
        self.mut = mutation
        self.device = device
        self.accelerator = accelerator
        self.index = index
        self.scores = []
        self.fitness = []
        self.steps = [0]

        self.support = torch.linspace(self.v_min, self.v_max, self.num_atoms).to(
            self.device
        )

        # model
        if self.net_config["arch"] == "mlp":  # Multi-layer Perceptron
            self.actor = EvolvableMLP(
                num_inputs=state_dim[0],
                num_outputs=action_dim,
                hidden_size=self.net_config["h_size"],
                output_activation="relu",
                output_vanish=False,
                init_layers=False,
                num_atoms=self.num_atoms,
                support=self.support,
                rainbow=True,
                device=self.device,
                accelerator=self.accelerator,
            )
            self.actor_target = EvolvableMLP(
                num_inputs=state_dim[0],
                num_outputs=action_dim,
                hidden_size=self.net_config["h_size"],
                output_activation="relu",
                output_vanish=False,
                init_layers=False,
                num_atoms=self.num_atoms,
                support=self.support,
                rainbow=True,
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
                mlp_output_activation="relu",
                num_atoms=self.num_atoms,
                support=self.support,
                rainbow=True,
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
                mlp_output_activation="relu",
                num_atoms=self.num_atoms,
                support=self.support,
                rainbow=True,
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

    def getAction(self, state, action_mask=None):
        """Returns the next action to take in the environment.

        :param state: State observation, or multiple observations in a batch
        :type state: float or List[float]
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

    def _dqn_loss(self, states, actions, rewards, next_states, dones, gamma):
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

        delta_z = float(self.v_max - self.v_min) / (self.num_atoms - 1)

        with torch.no_grad():
            # Double Q-learning
            next_dist = self.actor_target(next_states, q=False) * self.support
            next_action = next_dist.sum(2).max(1)[1]
            next_action = (
                next_action.unsqueeze(1)
                .unsqueeze(1)
                .expand(next_dist.size(0), 1, next_dist.size(2))
            )
            next_dist = next_dist.gather(1, next_action).squeeze(1)

            t_z = rewards + (1 - dones) * gamma * self.support
            t_z = t_z.clamp(min=self.v_min, max=self.v_max)
            b = (t_z - self.v_min) / delta_z
            L = b.floor().long()
            u = b.ceil().long()

            offset = (
                torch.linspace(
                    0, (self.batch_size - 1) * self.num_atoms, self.batch_size
                )
                .long()
                .unsqueeze(1)
                .expand(self.batch_size, self.num_atoms)
            )

            proj_dist = torch.zeros(next_dist.size())

            if self.accelerator is None:
                offset = offset.to(self.device)
                proj_dist = proj_dist.to(self.device)

            proj_dist.view(-1).index_add_(
                0, (L + offset).view(-1), (next_dist * (u.float() - b)).view(-1)
            )
            proj_dist.view(-1).index_add_(
                0, (u + offset).view(-1), (next_dist * (b - L.float())).view(-1)
            )

        dist = self.actor(states, q=False)
        actions = actions.unsqueeze(1).expand(actions.size(0), 1, self.num_atoms)
        dist = dist.gather(1, actions).squeeze(1)
        dist.data.clamp_(0.01, 0.99)
        log_p = torch.log(dist)

        # loss
        elementwise_loss = -(proj_dist * log_p).sum(1)
        return elementwise_loss

    def learn(self, experiences, n_step=True, per=False):
        """Updates agent network parameters to learn from experiences.

        :param experiences: List of batched states, actions, rewards, next_states, dones in that order.
        :type state: List[torch.Tensor[float]]
        :param n_step: Use multi-step learning, defaults to True
        :type n_step: bool, optional
        :param per: Use prioritized experience replay buffer, defaults to True
        :type per: bool, optional
        """
        if per:
            if n_step:
                (
                    states,
                    actions,
                    rewards,
                    next_states,
                    dones,
                    weights,
                    idxs,
                    n_states,
                    n_actions,
                    n_rewards,
                    n_next_states,
                    n_dones,
                ) = experiences
            else:
                (
                    states,
                    actions,
                    rewards,
                    next_states,
                    dones,
                    weights,
                    idxs,
                ) = experiences
            elementwise_loss = self._dqn_loss(
                states, actions, rewards, next_states, dones, self.gamma
            )
            if n_step:
                n_gamma = self.gamma**self.n_step
                n_step_elementwise_loss = self._dqn_loss(
                    n_states, n_actions, n_rewards, n_next_states, n_dones, n_gamma
                )
                elementwise_loss += n_step_elementwise_loss
            loss = torch.mean(elementwise_loss * weights)

        else:
            (
                states,
                actions,
                rewards,
                next_states,
                dones,
            ) = experiences
            idxs, new_priorities = None, None
            if n_step:
                n_gamma = self.gamma**self.n_step
            else:
                n_gamma = self.gamma
            elementwise_loss = self._dqn_loss(
                states, actions, rewards, next_states, dones, n_gamma
            )
            loss = torch.mean(elementwise_loss)

        self.optimizer.zero_grad()
        if self.accelerator is not None:
            self.accelerator.backward(loss)
        else:
            loss.backward()
        clip_grad_norm_(self.actor.parameters(), 10.0)
        self.optimizer.step()

        # soft update target network
        self.softUpdate()

        self.actor.reset_noise()
        self.actor_target.reset_noise()

        if per:
            loss_for_prior = elementwise_loss.detach().cpu().numpy()
            new_priorities = loss_for_prior + self.prior_eps

        return idxs, new_priorities

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
        :param loop: Number of testing loops/epsiodes to complete. The returned score is the mean over these tests. Defaults to 3
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
                    action = self.getAction(state)
                    state, reward, done, trunc, _ = env.step(action)
                    score += reward[0]
                    if done[0] or trunc[0]:
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
            beta=self.beta,
            prior_eps=self.prior_eps,
            num_atoms=self.num_atoms,
            v_min=self.v_min,
            v_max=self.v_max,
            n_step=self.n_step,
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
                "beta": self.beta,
                "prior_eps": self.prior_eps,
                "num_atoms": self.num_atoms,
                "v_min": self.v_min,
                "v_max": self.v_max,
                "n_step": self.n_step,
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
        self.beta = checkpoint["beta"]
        self.prior_eps = checkpoint["prior_eps"]
        self.num_atoms = checkpoint["num_atoms"]
        self.v_min = checkpoint["v_min"]
        self.v_max = checkpoint["v_max"]
        self.n_step = checkpoint["n_step"]
        self.mut = checkpoint["mutation"]
        self.index = checkpoint["index"]
        self.scores = checkpoint["scores"]
        self.fitness = checkpoint["fitness"]
        self.steps = checkpoint["steps"]
