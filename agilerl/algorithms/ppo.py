import copy
import inspect

import dill
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical, MultivariateNormal
from torch.nn.utils import clip_grad_norm_

from agilerl.networks.evolvable_cnn import EvolvableCNN
from agilerl.networks.evolvable_mlp import EvolvableMLP
from agilerl.utils.algo_utils import chkpt_attribute_to_device, unwrap_optimizer
from agilerl.wrappers.make_evolvable import MakeEvolvable


class PPO:
    """The PPO algorithm class. PPO paper: https://arxiv.org/abs/1707.06347v2

    :param state_dim: State observation dimension
    :type state_dim: list[int]
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
    :param learn_step: Learning frequency, defaults to 2048
    :type learn_step: int, optional
    :param gamma: Discount factor, defaults to 0.99
    :type gamma: float, optional
    :param gae_lambda: Lambda for general advantage estimation, defaults to 0.95
    :type gae_lambda: float, optional
    :param mut: Most recent mutation to agent, defaults to None
    :type mut: str, optional
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
    :param actor_network: Custom actor network, defaults to None
    :type actor_network: nn.Module, optional
    :param critic_network: Custom critic network, defaults to None
    :type critic_network: nn.Module, optional
    :param device: Device for accelerated computing, 'cpu' or 'cuda', defaults to 'cpu'
    :type device: str, optional
    :param accelerator: Accelerator for distributed computing, defaults to None
    :type accelerator: accelerate.Accelerator(), optional
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
        net_config={"arch": "mlp", "hidden_size": [64, 64]},
        batch_size=64,
        lr=1e-4,
        learn_step=2048,
        gamma=0.99,
        gae_lambda=0.95,
        mut=None,
        action_std_init=0.6,
        clip_coef=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        target_kl=None,
        update_epochs=4,
        actor_network=None,
        critic_network=None,
        device="cpu",
        accelerator=None,
        wrap=True,
    ):
        assert isinstance(
            state_dim, (list, tuple)
        ), "State dimension must be a list or tuple."
        assert isinstance(
            action_dim, (int, np.integer)
        ), "Action dimension must be an integer."
        assert isinstance(
            one_hot, bool
        ), "One-hot encoding flag must be boolean value True or False."
        assert isinstance(
            discrete_actions, bool
        ), "Discrete actions flag must be boolean value True or False."
        assert isinstance(index, int), "Agent index must be an integer."
        assert isinstance(batch_size, int), "Batch size must be an integer."
        assert batch_size >= 1, "Batch size must be greater than or equal to one."
        assert isinstance(lr, float), "Learning rate must be a float."
        assert lr > 0, "Learning rate must be greater than zero."
        assert isinstance(gamma, (float, int)), "Gamma must be a float."
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

        self.algo = "PPO"
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.one_hot = one_hot
        self.discrete_actions = discrete_actions
        self.net_config = net_config
        self.batch_size = batch_size
        self.lr = lr
        self.learn_step = learn_step
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.mut = mut
        self.action_std_init = action_std_init
        self.clip_coef = clip_coef
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.target_kl = target_kl
        self.update_epochs = update_epochs
        self.actor_network = actor_network
        self.critic_network = critic_network
        self.device = device
        self.accelerator = accelerator

        self.index = index
        self.scores = []
        self.fitness = []
        self.steps = [0]

        # For continuous action spaces
        if not self.discrete_actions:
            self.action_var = torch.full((action_dim,), action_std_init**2)
            if self.accelerator is None:
                self.action_var = self.action_var.to(self.device)
            else:
                self.action_var = self.action_var.to(self.accelerator.device)

        if self.actor_network is not None and self.critic_network is not None:
            assert type(actor_network) == type(
                critic_network
            ), "'actor_network' and 'critic_network' must be the same type."
            self.actor = actor_network
            self.critic = critic_network
            if isinstance(self.actor, (EvolvableMLP, EvolvableCNN)) and isinstance(
                self.critic, (EvolvableMLP, EvolvableCNN)
            ):
                self.net_config = self.actor.net_config
            elif isinstance(self.actor, MakeEvolvable) and isinstance(
                self.critic, MakeEvolvable
            ):
                self.net_config = None
            else:
                assert (
                    False
                ), f"'actor_network' argument is of type {type(actor_network)} and 'critic_network' of type {type(critic_network)}, \
                                both must be the same type and be of type EvolvableMLP, EvolvableCNN or MakeEvolvable"

        else:
            assert isinstance(self.net_config, dict), "Net config must be a dictionary."
            assert (
                "arch" in self.net_config.keys()
            ), "Net config must contain arch: 'mlp' or 'cnn'."

            # Set up network output activations
            if "mlp_output_activation" not in self.net_config.keys():
                if self.discrete_actions:
                    self.net_config["mlp_output_activation"] = "Softmax"
                else:
                    self.net_config["mlp_output_activation"] = "Tanh"

            if "mlp_activation" not in self.net_config.keys():
                self.net_config["mlp_activation"] = "Tanh"

            critic_net_config = copy.deepcopy(self.net_config)
            critic_net_config["mlp_output_activation"] = (
                None  # Critic must have no output activation
            )

            # model
            if self.net_config["arch"] == "mlp":  # Multi-layer Perceptron
                assert (
                    "hidden_size" in self.net_config.keys()
                ), "Net config must contain hidden_size: int."
                assert isinstance(
                    self.net_config["hidden_size"], list
                ), "Net config hidden_size must be a list."
                assert (
                    len(self.net_config["hidden_size"]) > 0
                ), "Net config hidden_size must contain at least one element."
                self.actor = EvolvableMLP(
                    num_inputs=state_dim[0],
                    num_outputs=action_dim,
                    device=self.device,
                    accelerator=self.accelerator,
                    **self.net_config,
                )
                self.critic = EvolvableMLP(
                    num_inputs=state_dim[0],
                    num_outputs=1,
                    device=self.device,
                    accelerator=self.accelerator,
                    **critic_net_config,
                )
            elif self.net_config["arch"] == "cnn":  # Convolutional Neural Network
                for key in [
                    "channel_size",
                    "kernel_size",
                    "stride_size",
                    "hidden_size",
                ]:
                    assert (
                        key in self.net_config.keys()
                    ), f"Net config must contain {key}: int."
                    assert isinstance(
                        self.net_config[key], list
                    ), f"Net config {key} must be a list."
                    assert (
                        len(self.net_config[key]) > 0
                    ), f"Net config {key} must contain at least one element."
                assert (
                    "normalize" in self.net_config.keys()
                ), "Net config must contain normalize: True or False."
                assert isinstance(
                    self.net_config["normalize"], bool
                ), "Net config normalize must be boolean value True or False."
                self.actor = EvolvableCNN(
                    input_shape=state_dim,
                    num_actions=action_dim,
                    device=self.device,
                    accelerator=self.accelerator,
                    **self.net_config,
                )
                self.critic = EvolvableCNN(
                    input_shape=state_dim,
                    num_actions=1,
                    device=self.device,
                    accelerator=self.accelerator,
                    **critic_net_config,
                )

        self.arch = (
            self.net_config["arch"] if self.net_config is not None else self.actor.arch
        )

        self.optimizer = optim.Adam(
            [
                {"params": self.actor.parameters(), "lr": self.lr},
                {"params": self.critic.parameters(), "lr": self.lr},
            ]
        )

        if self.accelerator is not None:
            if wrap:
                self.wrap_models()
        else:
            self.actor = self.actor.to(self.device)
            self.critic = self.critic.to(self.device)

    def prepare_state(self, state):
        """Prepares state for forward pass through neural network.

        :param state: Observation of environment
        :type state: np.Array() or list
        """
        if not isinstance(state, torch.Tensor):
            state = torch.from_numpy(state).float()

        if self.accelerator is None:
            state = state.to(self.device)
        else:
            state = state.to(self.accelerator.device)

        if self.one_hot:
            state = (
                nn.functional.one_hot(state.long(), num_classes=self.state_dim[0])
                .float()
                .squeeze()
            )

        if (self.arch == "mlp" and len(state.size()) < 2) or (
            self.arch == "cnn" and len(state.size()) < 4
        ):
            state = state.unsqueeze(0)

        return state.float()

    def get_action(self, state, action=None, grad=False):
        """Returns the next action to take in the environment.

        :param state: Environment observation, or multiple observations in a batch
        :type state: numpy.ndarray[float]
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
        else:
            action = action.to(self.accelerator.device)

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

        :param experience: List of batched states, actions, log_probs, rewards, dones, values, next_state in that order.
        :type experience: list[torch.Tensor[float]]
        :param noise_clip: Maximum noise limit to apply to actions, defaults to 0.5
        :type noise_clip: float, optional
        :param policy_noise: Standard deviation of noise applied to policy, defaults to 0.2
        :type policy_noise: float, optional
        """
        experiences = [torch.from_numpy(np.array(exp)) for exp in experiences]
        states, actions, log_probs, rewards, dones, values, next_state = experiences
        if self.accelerator is not None:
            next_state = next_state.to(self.accelerator.device)
        dones = dones.long()

        # Bootstrapping
        with torch.no_grad():
            num_steps = rewards.size(0)
            next_state = self.prepare_state(next_state)
            next_value = self.critic(next_state).reshape(1, -1).cpu()
            advantages = torch.zeros_like(rewards).float()
            for t in range(num_steps):
                discount = 1
                a_t = 0
                for k in range(t, num_steps):
                    if k != num_steps - 1:
                        nextvalue = values[k + 1]
                    else:
                        nextvalue = next_value.squeeze()

                    a_t += discount * (
                        rewards[k]
                        + self.gamma * nextvalue * (1.0 - dones[k])
                        - values[k]
                    )
                    discount *= self.gamma * self.gae_lambda * (1.0 - dones[k])

                advantages[t] = a_t
            returns = advantages + values

        if self.one_hot:
            states = states.reshape(-1)
        else:
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
        else:
            states = states.to(self.accelerator.device)
            actions = actions.to(self.accelerator.device)
            log_probs = log_probs.to(self.accelerator.device)
            advantages = advantages.to(self.accelerator.device)
            returns = returns.to(self.accelerator.device)
            values = values.to(self.accelerator.device)

        num_samples = returns.size(0)
        batch_idxs = np.arange(num_samples)
        clipfracs = []

        mean_loss = 0
        for epoch in range(self.update_epochs):
            np.random.shuffle(batch_idxs)
            for start in range(0, num_samples, self.batch_size):
                minibatch_idxs = batch_idxs[start : start + self.batch_size]
                _, log_prob, entropy, value = self.get_action(
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

                # actor + critic loss backprop
                self.optimizer.zero_grad()
                if self.accelerator is not None:
                    self.accelerator.backward(loss)
                else:
                    loss.backward()
                clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.optimizer.step()

                mean_loss += loss.item()

            if self.target_kl is not None:
                if approx_kl > self.target_kl:
                    break

        mean_loss /= num_samples * self.update_epochs
        return mean_loss

    def test(self, env, swap_channels=False, max_steps=None, loop=3):
        """Returns mean test score of agent in environment with epsilon-greedy policy.

        :param env: The environment to be tested in
        :type env: Gym-style environment
        :param swap_channels: Swap image channels dimension from last to first [H, W, C] -> [C, H, W], defaults to False
        :type swap_channels: bool, optional
        :param max_steps: Maximum number of testing steps, defaults to None
        :type max_steps: int, optional
        :param loop: Number of testing loops/episodes to complete. The returned score is the mean. Defaults to 3
        :type loop: int, optional
        """
        with torch.no_grad():
            rewards = []
            num_envs = env.num_envs if hasattr(env, "num_envs") else 1
            for i in range(loop):
                state, _ = env.reset()
                scores = np.zeros(num_envs)
                completed_episode_scores = np.zeros(num_envs)
                finished = np.zeros(num_envs)
                step = 0
                while not np.all(finished):
                    if swap_channels:
                        state = np.moveaxis(state, [-1], [-3])
                    action, _, _, _ = self.get_action(state)
                    state, reward, done, trunc, _ = env.step(action)
                    step += 1
                    scores += np.array(reward)
                    for idx, (d, t) in enumerate(zip(done, trunc)):
                        if (
                            d or t or (max_steps is not None and step == max_steps)
                        ) and not finished[idx]:
                            completed_episode_scores[idx] = scores[idx]
                            finished[idx] = 1
                rewards.append(np.mean(completed_episode_scores))
        mean_fit = np.mean(rewards)
        self.fitness.append(mean_fit)
        return mean_fit

    def clone(self, index=None, wrap=True):
        """Returns cloned agent identical to self.

        :param index: Index to keep track of agent for tournament selection and mutation, defaults to None
        :type index: int, optional
        """
        input_args = self.inspect_attributes(input_args_only=True)
        input_args["wrap"] = wrap
        clone = type(self)(**input_args)

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
        optimizer.load_state_dict(self.optimizer.state_dict())

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

        for attribute in self.inspect_attributes().keys():
            if hasattr(self, attribute) and hasattr(clone, attribute):
                attr, clone_attr = getattr(self, attribute), getattr(clone, attribute)
                if isinstance(attr, torch.Tensor) or isinstance(
                    clone_attr, torch.Tensor
                ):
                    if not torch.equal(attr, clone_attr):
                        setattr(
                            clone, attribute, copy.deepcopy(getattr(self, attribute))
                        )
                else:
                    if attr != clone_attr:
                        setattr(
                            clone, attribute, copy.deepcopy(getattr(self, attribute))
                        )
            else:
                setattr(clone, attribute, copy.deepcopy(getattr(self, attribute)))

        if index is not None:
            clone.index = index

        return clone

    def inspect_attributes(self, input_args_only=False):
        # Get all attributes of the current object
        attributes = inspect.getmembers(self, lambda a: not (inspect.isroutine(a)))
        guarded_attributes = ["actor", "critic", "optimizer"]

        # Exclude private and built-in attributes
        attributes = [
            a for a in attributes if not (a[0].startswith("__") and a[0].endswith("__"))
        ]

        if input_args_only:
            constructor_params = inspect.signature(self.__init__).parameters.keys()
            attributes = {
                k: v
                for k, v in attributes
                if k not in guarded_attributes and k in constructor_params
            }
        else:
            # Remove the algo specific guarded variables
            attributes = {k: v for k, v in attributes if k not in guarded_attributes}

        return attributes

    def wrap_models(self):
        if self.accelerator is not None:
            self.actor, self.critic, self.optimizer = self.accelerator.prepare(
                self.actor, self.critic, self.optimizer
            )

    def unwrap_models(self):
        if self.accelerator is not None:
            self.actor = self.accelerator.unwrap_model(self.actor)
            self.critic = self.accelerator.unwrap_model(self.critic)
            self.optimizer = unwrap_optimizer(
                self.optimizer, [self.actor, self.critic], self.lr
            )

    def save_checkpoint(self, path):
        """Saves a checkpoint of agent properties and network weights to path.

        :param path: Location to save checkpoint at
        :type path: string
        """

        attribute_dict = self.inspect_attributes()

        network_info = {
            "actor_init_dict": self.actor.init_dict,
            "actor_state_dict": self.actor.state_dict(),
            "critic_init_dict": self.critic.init_dict,
            "critic_state_dict": self.critic.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }

        attribute_dict.update(network_info)

        torch.save(
            attribute_dict,
            path,
            pickle_module=dill,
        )

    def load_checkpoint(self, path):
        """Loads saved agent properties and network weights from checkpoint.

        :param path: Location to load checkpoint from
        :type path: string
        """
        network_info = [
            "actor_state_dict",
            "critic_state_dict",
            "optimizer_state_dict",
            "actor_init_dict",
            "critic_init_dict",
            "net_config",
            "lr",
        ]

        checkpoint = torch.load(path, map_location=self.device, pickle_module=dill)
        self.net_config = checkpoint["net_config"]
        if self.net_config is not None:
            self.arch = checkpoint["net_config"]["arch"]
            if self.arch == "mlp":
                network_class = EvolvableMLP
            elif self.arch == "cnn":
                network_class = EvolvableCNN
        else:
            network_class = MakeEvolvable
        self.actor = network_class(**checkpoint["actor_init_dict"])
        self.critic = network_class(**checkpoint["critic_init_dict"])
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

        for attribute in checkpoint.keys():
            if attribute not in network_info:
                setattr(self, attribute, checkpoint[attribute])

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
        checkpoint = torch.load(path, map_location=device, pickle_module=dill)
        checkpoint["actor_init_dict"]["device"] = device
        checkpoint["critic_init_dict"]["device"] = device

        actor_init_dict = chkpt_attribute_to_device(
            checkpoint.pop("actor_init_dict"), device
        )
        actor_state_dict = chkpt_attribute_to_device(
            checkpoint.pop("actor_state_dict"), device
        )
        critic_init_dict = chkpt_attribute_to_device(
            checkpoint.pop("critic_init_dict"), device
        )
        critic_state_dict = chkpt_attribute_to_device(
            checkpoint.pop("critic_state_dict"), device
        )
        optimizer_state_dict = chkpt_attribute_to_device(
            checkpoint.pop("optimizer_state_dict"), device
        )

        checkpoint["device"] = device
        checkpoint["accelerator"] = accelerator
        checkpoint = chkpt_attribute_to_device(checkpoint, device)

        constructor_params = inspect.signature(cls.__init__).parameters.keys()
        class_init_dict = {
            k: v for k, v in checkpoint.items() if k in constructor_params
        }

        if checkpoint["net_config"] is not None:
            agent = cls(**class_init_dict)
            agent.arch = checkpoint["net_config"]["arch"]
            if agent.arch == "mlp":
                agent.actor = EvolvableMLP(**actor_init_dict)
                agent.critic = EvolvableMLP(**critic_init_dict)
            elif agent.arch == "cnn":
                agent.actor = EvolvableCNN(**actor_init_dict)
                agent.critic = EvolvableCNN(**critic_init_dict)
        else:
            class_init_dict["actor_network"] = MakeEvolvable(**actor_init_dict)
            class_init_dict["critic_network"] = MakeEvolvable(**critic_init_dict)
            agent = cls(**class_init_dict)

        agent.actor.load_state_dict(actor_state_dict)
        agent.critic.load_state_dict(critic_state_dict)
        agent.optimizer = optim.Adam(
            [
                {"params": agent.actor.parameters(), "lr": agent.lr},
                {"params": agent.critic.parameters(), "lr": agent.lr},
            ]
        )
        agent.optimizer.load_state_dict(optimizer_state_dict)

        if accelerator is not None:
            agent.wrap_models()

        for attribute in agent.inspect_attributes().keys():
            setattr(agent, attribute, checkpoint[attribute])

        return agent
