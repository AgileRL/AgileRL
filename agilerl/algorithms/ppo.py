import copy

import dill
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical, MultivariateNormal

from agilerl.networks.evolvable_cnn import EvolvableCNN
from agilerl.networks.evolvable_mlp import EvolvableMLP


class PPO:
    """The PPO algorithm class. PPO paper: https://arxiv.org/abs/1707.06347v2

    :param state_dim: State observation dimension
    :type state_dim: int
    :param action_dim: Action dimension
    :type action_dim: int
    :param one_hot: One-hot encoding, used with discrete observation spaces
    :type one_hot: bool
    :param discrete_actions: Boolean flag to indicate a discrete action space
    :type discrete_actions: bool, optional
    :param index: Index to keep track of object instance during tournament selection and mutation, defaults to 0
    :type index: int, optional
    :param net_config: Network configuration, defaults to mlp with hidden size [64,64]
    :type net_config: dict, optional
    :param batch_size: Size of batched sample from replay buffer for learning, defaults to 64
    :type batch_size: int, optional
    :param lr: Learning rate for optimizer, defaults to 1e-4
    :type lr: float, optional
    :param gamma: Discount factor, defaults to 0.99
    :type gamma: float, optional
    :param gae_lambda: Lambda for general advantage estimation, defaults to 0.95
    :type gae_lambda: float, optional
    :param mutation: Most recent mutation to agent, defaults to None
    :type mutation: str, optional
    :param action_std_init: Initial action standard deviation, defaults to 0.6
    :type action_std_init: float, optional
    :param clip_coef: Surrogate clipping coefficient, defaults to 0.2
    :type clip coef: float, optional
    :param ent_coef: Entropy coefficient, defaults to 0.01
    :type ent_coef: float, optional
    :param vf_coef: Value function coefficient, defaults to 0.5
    :type vf_coef: float, optional
    :param max_grad_norm: Maximum norm for gradient clipping, defaults to 0.5
    :type max_grad_norm: float, optional
    :param target_kl: Target KL divergence threshold, defaults to None
    :type target_kl: float, optional
    :param update_epochs: Number of policy update epochs, defaults to 4
    :type update_epochs: int, optional
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
        discrete_actions,
        index=0,
        net_config={"arch": "mlp", "h_size": [64, 64]},
        batch_size=64,
        lr=1e-4,
        gamma=0.99,
        gae_lambda=0.95,
        mutation=None,
        action_std_init=0.6,
        clip_coef=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        target_kl=None,
        update_epochs=4,
        device="cpu",
        accelerator=None,
        wrap=True,
    ):
        self.algo = "PPO"
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.one_hot = one_hot
        self.discrete_actions = discrete_actions
        self.net_config = net_config
        self.batch_size = batch_size
        self.lr = lr
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.mut = mutation
        self.action_std_init = action_std_init
        self.clip_coef = clip_coef
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.target_kl = target_kl
        self.update_epochs = update_epochs
        self.device = device
        self.accelerator = accelerator

        self.index = index
        self.scores = []
        self.fitness = []
        self.steps = [0]

        # Set up network output activations
        if "output_activation" in self.net_config.keys():
            pass
        else:
            if self.discrete_actions:
                self.net_config["output_activation"] = "softmax"
            else:
                self.net_config["output_activation"] = "tanh"

        # For continuous action spaces
        if not self.discrete_actions:
            self.action_var = torch.full((action_dim,), action_std_init**2)
            if self.accelerator is None:
                self.action_var = self.action_var.to(self.device)

        # model
        if self.net_config["arch"] == "mlp":  # Multi-layer Perceptron
            self.actor = EvolvableMLP(
                num_inputs=state_dim[0],
                num_outputs=action_dim,
                hidden_size=self.net_config["h_size"],
                output_activation=self.net_config["output_activation"],
                device=self.device,
                accelerator=self.accelerator,
            )
            self.critic = EvolvableMLP(
                num_inputs=state_dim[0],
                num_outputs=1,
                hidden_size=self.net_config["h_size"],
                device=self.device,
                accelerator=self.accelerator,
            )

        elif self.net_config["arch"] == "cnn":  # Convolutional Neural Network
            self.actor = EvolvableCNN(
                input_shape=state_dim,
                num_actions=action_dim,
                channel_size=self.net_config["c_size"],
                kernel_size=self.net_config["k_size"],
                stride_size=self.net_config["s_size"],
                hidden_size=self.net_config["h_size"],
                normalize=self.net_config["normalize"],
                mlp_activation=self.net_config["output_activation"],
                device=self.device,
                accelerator=self.accelerator,
            )
            self.critic = EvolvableCNN(
                input_shape=state_dim,
                num_actions=1,
                channel_size=self.net_config["c_size"],
                kernel_size=self.net_config["k_size"],
                stride_size=self.net_config["s_size"],
                hidden_size=self.net_config["h_size"],
                normalize=self.net_config["normalize"],
                mlp_activation="tanh",
                device=self.device,
                accelerator=self.accelerator,
            )

        self.optimizer_type = optim.Adam(
            [
                {"params": self.actor.parameters(), "lr": self.lr},
                {"params": self.critic.parameters(), "lr": self.lr},
            ]
        )

        if self.accelerator is not None:
            self.optimizer = self.optimizer_type
            if wrap:
                self.wrap_models()
        else:
            self.actor = self.actor.to(self.device)
            self.critic = self.critic.to(self.device)
            self.optimizer = self.optimizer_type

        self.criterion = nn.MSELoss()

    def prepare_state(self, state):
        """Prepares state for forward pass through neural network.

        :param state: Observation of environment
        :type state: np.Array() or List
        """
        if not isinstance(state, torch.Tensor):
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

        return state.float()

    def getAction(self, state, action=None, grad=False):
        """Returns the next action to take in the environment.
        Epsilon is the probability of taking a random action, used for exploration.
        For epsilon-greedy behaviour, set epsilon to 0.

        :param state: Environment observation, or multiple observations in a batch
        :type state: float or List[float]
        :param action: Action in environment to evaluate, defaults to None
        :type action: torch.Tensor(), optional
        :param grad: Calculate gradients on actions, defaults to False
        :type grad: bool, optional
        """
        state = self.prepare_state(state)

        if not grad:
            self.actor.eval()
            self.critic.eval()
            with torch.no_grad():
                action_values = self.actor(state)
                state_values = self.critic(state).squeeze(-1)
            self.actor.train()
            self.critic.train()

        else:
            action_values = self.actor(state)
            state_values = self.critic(state).squeeze(-1)

        if self.discrete_actions:
            dist = Categorical(action_values)
        else:
            cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
            dist = MultivariateNormal(action_values, cov_mat)

        return_tensors = True
        if action is None:
            action = dist.sample()
            return_tensors = False
        elif self.accelerator is None:
            action = action.to(self.device)

        action_logprob = dist.log_prob(action)
        dist_entropy = dist.entropy()

        if return_tensors:
            return action, action_logprob, dist_entropy, state_values
        else:
            return (
                action.cpu().data.numpy(),
                action_logprob.cpu().data.numpy(),
                dist_entropy.cpu().data.numpy(),
                state_values.cpu().data.numpy(),
            )

    def learn(self, experiences, noise_clip=0.5, policy_noise=0.2):
        """Updates agent network parameters to learn from experiences.

        :param experience: List of batched states, actions, rewards, next_states, dones in that order.
        :type experience: List[torch.Tensor[float]]
        :param noise_clip: Maximum noise limit to apply to actions, defaults to 0.5
        :type noise_clip: float, optional
        :param policy_noise: Standard deviation of noise applied to policy, defaults to 0.2
        :type policy_noise: float, optional
        """
        experiences = [torch.from_numpy(np.array(exp)) for exp in experiences]
        states, actions, log_probs, rewards, dones, values, next_state = experiences
        dones = dones.long()

        # Bootstrapping
        with torch.no_grad():
            num_steps = rewards.size(0)
            next_state = self.prepare_state(next_state)
            next_value = self.critic(next_state).reshape(1, -1).cpu()
            advantages = torch.zeros_like(rewards)
            last_gae_lambda = 0
            for t in reversed(range(num_steps)):
                if t == num_steps - 1:
                    nextnonterminal = 1.0 - dones[-1]
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = (
                    rewards[t] + self.gamma * nextvalues * nextnonterminal - values[t]
                )
                advantages[t] = last_gae_lambda = (
                    delta
                    + self.gamma * self.gae_lambda * nextnonterminal * last_gae_lambda
                )
            returns = advantages + values

        states = states.reshape((-1,) + self.state_dim)
        if self.discrete_actions:
            actions = actions.reshape(-1)
        else:
            actions = actions.reshape((-1, self.action_dim))
        log_probs = log_probs.reshape(-1)
        advantages = advantages.reshape(-1)
        returns = returns.reshape(-1)
        values = values.reshape(-1)

        if self.accelerator is None:
            states, actions, log_probs, advantages, returns, values = (
                states.to(self.device),
                actions.to(self.device),
                log_probs.to(self.device),
                advantages.to(self.device),
                returns.to(self.device),
                values.to(self.device),
            )

        num_samples = returns.size(0)
        batch_idxs = np.arange(num_samples)

        clipfracs = []

        for epoch in range(self.update_epochs):
            np.random.shuffle(batch_idxs)
            for start in range(0, num_samples, self.batch_size):
                minibatch_idxs = batch_idxs[start : start + self.batch_size]

                _, log_prob, entropy, value = self.getAction(
                    state=states[minibatch_idxs],
                    action=actions[minibatch_idxs],
                    grad=True,
                )

                logratio = log_prob - log_probs[minibatch_idxs]
                ratio = logratio.exp()

                with torch.no_grad():
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [
                        ((ratio - 1.0).abs() > self.clip_coef).float().mean().item()
                    ]

                minibatch_advs = advantages[minibatch_idxs]
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
                v_loss_unclipped = (value - returns[minibatch_idxs]) ** 2
                v_clipped = values[minibatch_idxs] + torch.clamp(
                    value - values[minibatch_idxs], -self.clip_coef, self.clip_coef
                )
                v_loss_clipped = (v_clipped - returns[minibatch_idxs]) ** 2
                v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                v_loss = 0.5 * v_loss_max.mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - self.ent_coef * entropy_loss + v_loss * self.vf_coef

                # actor loss backprop
                self.optimizer.zero_grad()
                if self.accelerator is not None:
                    self.accelerator.backward(loss)
                else:
                    loss.backward()
                self.optimizer.step()

            if self.target_kl is not None:
                if approx_kl > self.target_kl:
                    break

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
                    action, _, _, _ = self.getAction(state)
                    state, reward, done, trunc, info = env.step(action)
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
            discrete_actions=self.discrete_actions,
            index=index,
            net_config=self.net_config,
            batch_size=self.batch_size,
            lr=self.lr,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            action_std_init=self.action_std_init,
            clip_coef=self.clip_coef,
            ent_coef=self.ent_coef,
            vf_coef=self.vf_coef,
            max_grad_norm=self.max_grad_norm,
            target_kl=self.target_kl,
            update_epochs=self.update_epochs,
            mutation=self.mut,
            device=self.device,
            accelerator=self.accelerator,
            wrap=wrap,
        )

        if self.accelerator is not None:
            self.unwrap_models()
        actor = self.actor.clone()
        critic = self.critic.clone()
        optimizer = optim.Adam(
            [
                {"params": actor.parameters(), "lr": self.lr},
                {"params": critic.parameters(), "lr": self.lr},
            ]
        )
        clone.optimizer_type = optimizer

        if self.accelerator is not None:
            if wrap:
                (
                    clone.actor,
                    clone.critic,
                    clone.optimizer,
                ) = self.accelerator.prepare(actor, critic, optimizer)
            else:
                (
                    clone.actor,
                    clone.critic,
                    clone.optimizer,
                ) = (
                    actor,
                    critic,
                    optimizer,
                )
        else:
            clone.actor = actor.to(self.device)
            clone.critic = critic.to(self.device)
            clone.optimizer = optimizer

        clone.fitness = copy.deepcopy(self.fitness)
        clone.steps = copy.deepcopy(self.steps)
        clone.scores = copy.deepcopy(self.scores)

        return clone

    def wrap_models(self):
        if self.accelerator is not None:
            self.actor, self.critic, self.optimizer = self.accelerator.prepare(
                self.actor, self.critic, self.optimizer_type
            )

    def unwrap_models(self):
        if self.accelerator is not None:
            self.actor = self.accelerator.unwrap_model(self.actor)
            self.critic = self.accelerator.unwrap_model(self.critic)
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
                "critic_init_dict": self.critic.init_dict,
                "critic_state_dict": self.critic.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "discrete_actions": self.discrete_actions,
                "net_config": self.net_config,
                "batch_size": self.batch_size,
                "lr": self.lr,
                "gamma": self.gamma,
                "gae_lambda": self.gae_lambda,
                "action_std_init": self.action_std_init,
                "clip_coef": self.clip_coef,
                "ent_coef": self.ent_coef,
                "vf_coef": self.vf_coef,
                "max_grad_norm": self.max_grad_norm,
                "target_kl": self.target_kl,
                "update_epochs": self.update_epochs,
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
            self.critic = EvolvableMLP(**checkpoint["critic_init_dict"])
        elif self.net_config["arch"] == "cnn":
            self.actor = EvolvableCNN(**checkpoint["actor_init_dict"])
            self.critic = EvolvableCNN(**checkpoint["critic_init_dict"])
        self.lr = checkpoint["lr"]
        self.optimizer = optim.Adam(
            [
                {"params": self.actor.parameters(), "lr": self.lr},
                {"params": self.critic.parameters(), "lr": self.lr},
            ]
        )
        self.actor.load_state_dict(checkpoint["actor_state_dict"])
        self.critic.load_state_dict(checkpoint["critic_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.discrete_actions = checkpoint["discrete_actions"]
        self.batch_size = checkpoint["batch_size"]
        self.gamma = checkpoint["gamma"]
        self.gae_lambda = checkpoint["gae_lambda"]
        self.action_std_init = checkpoint["action_std_init"]
        self.clip_coef = checkpoint["clip_coef"]
        self.ent_coef = checkpoint["ent_coef"]
        self.vf_coef = checkpoint["vf_coef"]
        self.max_grad_norm = checkpoint["max_grad_norm"]
        self.target_kl = checkpoint["target_kl"]
        self.update_epochs = checkpoint["update_epochs"]
        self.mut = checkpoint["mutation"]
        self.index = checkpoint["index"]
        self.scores = checkpoint["scores"]
        self.fitness = checkpoint["fitness"]
        self.steps = checkpoint["steps"]
