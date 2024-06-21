import copy
import inspect

import dill
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_

from agilerl.networks.evolvable_cnn import EvolvableCNN
from agilerl.networks.evolvable_mlp import EvolvableMLP
from agilerl.utils.algo_utils import chkpt_attribute_to_device, unwrap_optimizer
from agilerl.wrappers.make_evolvable import MakeEvolvable


class RainbowDQN:
    """The Rainbow DQN algorithm class. Rainbow DQN paper: https://arxiv.org/abs/1710.02298

    :param state_dim: State observation dimension
    :type state_dim: list[int]
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
    :param noise_std: Noise standard deviation, defaults to 0.5
    :type noise_std: float, optional
    :param n_step: Step number to calculate n-step td error, defaults to 3
    :type n_step: int, optional
    :param mut: Most recent mutation to agent, defaults to None
    :type mut: str, optional
    :param combined_reward: Boolean flag indicating whether to use combined 1-step and n-step reward, defaults to False
    :type combined_reward: bool, optional
    :param actor_network: Custom actor network, defaults to None
    :type actor_network: nn.Module, optional
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
        index=0,
        net_config={"arch": "mlp", "hidden_size": [64, 64]},
        batch_size=64,
        lr=1e-4,
        learn_step=5,
        gamma=0.99,
        tau=1e-3,
        beta=0.4,
        prior_eps=1e-6,
        num_atoms=51,
        v_min=-10,
        v_max=10,
        noise_std=0.5,
        n_step=3,
        mut=None,
        combined_reward=False,
        actor_network=None,
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
        assert isinstance(index, int), "Agent index must be an integer."
        assert isinstance(batch_size, int), "Batch size must be an integer."
        assert batch_size >= 1, "Batch size must be greater than or equal to one."
        assert isinstance(lr, float), "Learning rate must be a float."
        assert lr > 0, "Learning rate must be greater than zero."
        assert isinstance(learn_step, int), "Learn step rate must be an integer."
        assert learn_step >= 1, "Learn step must be greater than or equal to one."
        assert isinstance(gamma, (float, int)), "Gamma must be a float."
        assert isinstance(tau, float), "Tau must be a float."
        assert tau > 0, "Tau must be greater than zero."
        assert isinstance(
            prior_eps, float
        ), "Minimum priority for sampling must be a float."
        assert prior_eps > 0, "Minimum priority for sampling must be greater than zero."
        assert isinstance(num_atoms, int), "Number of atoms must be an integer."
        assert num_atoms >= 1, "Number of atoms must be greater than or equal to one."
        assert isinstance(
            v_min, (float, int)
        ), "Minimum value of support must be a float."
        assert isinstance(
            v_max, (float, int)
        ), "Maximum value of support must be a float."
        assert (
            v_max >= v_min
        ), "Maximum value of support must be greater than or equal to minimum value."
        assert isinstance(n_step, int), "Step number must be an integer."
        assert n_step >= 1, "Step number must be greater than or equal to one."
        assert isinstance(
            wrap, bool
        ), "Wrap models flag must be boolean value True or False."
        if net_config is not None:
            if "hidden_size" in net_config.keys():
                assert (
                    len(net_config["hidden_size"]) > 1
                ), f"Length of hidden size list must be greater than 1, currently {len(net_config['hidden_size'])}"

            if "min_hidden_layers" in net_config.keys():
                assert (
                    net_config["min_hidden_layers"] > 1
                ), f"Minimum number of hidden layers must be greater than 1 for Rainbow DQN, currently {net_config['min_hidden_layers']}"
            else:
                net_config["min_hidden_layers"] = 2

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
        self.mut = mut
        self.actor_network = actor_network
        self.device = device
        self.accelerator = accelerator
        self.index = index
        self.scores = []
        self.fitness = []
        self.steps = [0]
        self.combined_reward = combined_reward
        self.noise_std = noise_std

        self.support = torch.linspace(self.v_min, self.v_max, self.num_atoms)
        self.delta_z = (self.v_max - self.v_min) / (self.num_atoms - 1)
        if self.accelerator is None:
            self.support = self.support.to(self.device)
        else:
            self.support = self.support.to(self.accelerator.device)

        if self.actor_network is not None:
            self.actor = actor_network
            if isinstance(self.actor, (EvolvableMLP, EvolvableCNN)):
                self.net_config = self.actor.net_config
                self.actor_network = None
            elif isinstance(self.actor, MakeEvolvable):
                self.net_config = None
                self.actor.rainbow = True
                self.actor_network = actor_network
                self.actor.support = self.support
                self.actor.num_atoms = self.num_atoms
                self.actor = MakeEvolvable(**self.actor.init_dict)
                self.actor.load_state_dict(self.actor.state_dict())
            else:
                assert (
                    False
                ), f"'actor_network' argument is of type {type(actor_network)}, but must be of type EvolvableMLP, EvolvableCNN or MakeEvolvable"

        else:
            assert isinstance(self.net_config, dict), "Net config must be a dictionary."
            assert (
                "arch" in self.net_config.keys()
            ), "Net config must contain arch: 'mlp' or 'cnn'."
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

                if "mlp_output_activation" not in self.net_config.keys():
                    self.net_config["mlp_output_activation"] = "ReLU"

                self.actor = EvolvableMLP(
                    num_inputs=state_dim[0],
                    num_outputs=action_dim,
                    output_vanish=True,
                    init_layers=False,
                    layer_norm=True,
                    num_atoms=self.num_atoms,
                    support=self.support,
                    rainbow=True,
                    noise_std=noise_std,
                    device=self.device,
                    accelerator=self.accelerator,
                    **self.net_config,
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
                    num_atoms=self.num_atoms,
                    support=self.support,
                    rainbow=True,
                    noise_std=noise_std,
                    device=self.device,
                    accelerator=self.accelerator,
                    **self.net_config,
                )

        # Create the target network by copying the actor network
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.optimizer = optim.Adam(self.actor.parameters(), lr=self.lr)

        self.arch = (
            self.net_config["arch"] if self.net_config is not None else self.actor.arch
        )

        if self.accelerator is not None:
            if wrap:
                self.wrap_models()
        else:
            self.actor = self.actor.to(self.device)
            self.actor_target = self.actor_target.to(self.device)

        # Put the nets into training mode
        self.actor.train()
        self.actor_target.train()

    def get_action(self, state, action_mask=None, training=True):
        """Returns the next action to take in the environment.

        :param state: State observation, or multiple observations in a batch
        :type state: numpy.ndarray[float]
        :param action_mask: Mask of legal actions 1=legal 0=illegal, defaults to None
        :type action_mask: numpy.ndarray, optional
        """
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

        self.actor.train(mode=training)
        with torch.no_grad():
            action_values = self.actor(state)
        if action_mask is None:
            action = np.argmax(action_values.cpu().data.numpy(), axis=-1)
        else:
            inv_mask = 1 - action_mask
            masked_action_values = np.ma.array(
                action_values.cpu().data.numpy(), mask=inv_mask
            )
            action = np.argmax(masked_action_values, axis=-1)

        self.actor.train()

        return action

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

        with torch.no_grad():

            # Predict next actions from next_states
            next_actions = self.actor(next_states).argmax(1)

            # Predict the target q distribution for the same next states
            target_q_dist = self.actor_target(next_states, q=False)

            # Index the target q_dist to select the distributions corresponding to next_actions
            target_q_dist = target_q_dist[range(self.batch_size), next_actions]

            # Determine the target z values
            t_z = rewards + (1 - dones) * gamma * self.support
            t_z = t_z.clamp(min=self.v_min, max=self.v_max)

            # Finds closest support element index value
            b = (t_z - self.v_min) / self.delta_z

            # Find the neighbouring indices of b
            L = b.floor().long()
            u = b.ceil().long()

            # Shape of projected q distribution is (batch_size, num_atoms) as we have argmaxed over actions
            # Fix disappearing probability mass
            L[(u > 0) * (L == u)] -= 1
            u[(L < (self.num_atoms - 1)) * (L == u)] += 1
            offset = (
                torch.linspace(
                    0, (self.batch_size - 1) * self.num_atoms, self.batch_size
                )
                .long()
                .unsqueeze(1)
                .expand(self.batch_size, self.num_atoms)
            )
            proj_dist = torch.zeros(target_q_dist.size())
            if self.accelerator is None:
                offset = offset.to(self.device)
                proj_dist = proj_dist.to(self.device)
            else:
                offset = offset.to(self.accelerator.device)
                proj_dist = proj_dist.to(self.accelerator.device)
            proj_dist.view(-1).index_add_(
                0, (L + offset).view(-1), (target_q_dist * (u.float() - b)).view(-1)
            )
            proj_dist.view(-1).index_add_(
                0, (u + offset).view(-1), (target_q_dist * (b - L.float())).view(-1)
            )

        # Calculate the current state
        log_q_dist = self.actor(states, q=False, log=True)
        log_p = log_q_dist[range(self.batch_size), actions.squeeze().long()]

        # loss
        elementwise_loss = -(proj_dist * log_p).sum(1)
        return elementwise_loss

    def learn(self, experiences, n_step=False, per=False):
        """Updates agent network parameters to learn from experiences.

        :param experiences: List of batched states, actions, rewards, next_states, dones in that order.
        :type state: list[torch.Tensor[float]]
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
                if self.accelerator is not None:
                    states = states.to(self.accelerator.device)
                    actions = actions.to(self.accelerator.device)
                    rewards = rewards.to(self.accelerator.device)
                    next_states = next_states.to(self.accelerator.device)
                    dones = dones.to(self.accelerator.device)
                    weights = weights.to(self.accelerator.device)
                    n_states = n_states.to(self.accelerator.device)
                    n_actions = n_actions.to(self.accelerator.device)
                    n_rewards = n_rewards.to(self.accelerator.device)
                    n_next_states = n_next_states.to(self.accelerator.device)
                    n_dones = n_dones.to(self.accelerator.device)
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
                if self.accelerator is not None:
                    states = states.to(self.accelerator.device)
                    actions = actions.to(self.accelerator.device)
                    rewards = rewards.to(self.accelerator.device)
                    next_states = next_states.to(self.accelerator.device)
                    dones = dones.to(self.accelerator.device)
                    weights = weights.to(self.accelerator.device)
            if self.combined_reward or not n_step:
                elementwise_loss = self._dqn_loss(
                    states, actions, rewards, next_states, dones, self.gamma
                )
            if n_step:
                n_gamma = self.gamma**self.n_step
                n_step_elementwise_loss = self._dqn_loss(
                    n_states, n_actions, n_rewards, n_next_states, n_dones, n_gamma
                )
                if self.combined_reward:
                    elementwise_loss += n_step_elementwise_loss
                else:
                    elementwise_loss = n_step_elementwise_loss
            loss = torch.mean(elementwise_loss * weights)

        else:
            if n_step:
                (
                    states,
                    actions,
                    rewards,
                    next_states,
                    dones,
                    idxs,
                    n_states,
                    n_actions,
                    n_rewards,
                    n_next_states,
                    n_dones,
                ) = experiences
                if self.accelerator is not None:
                    states = states.to(self.accelerator.device)
                    actions = actions.to(self.accelerator.device)
                    rewards = rewards.to(self.accelerator.device)
                    next_states = next_states.to(self.accelerator.device)
                    dones = dones.to(self.accelerator.device)
                    n_states = n_states.to(self.accelerator.device)
                    n_actions = n_actions.to(self.accelerator.device)
                    n_rewards = n_rewards.to(self.accelerator.device)
                    n_next_states = n_next_states.to(self.accelerator.device)
                    n_dones = n_dones.to(self.accelerator.device)
            else:
                (
                    states,
                    actions,
                    rewards,
                    next_states,
                    dones,
                ) = experiences
                if self.accelerator is not None:
                    states = states.to(self.accelerator.device)
                    actions = actions.to(self.accelerator.device)
                    rewards = rewards.to(self.accelerator.device)
                    next_states = next_states.to(self.accelerator.device)
                    dones = dones.to(self.accelerator.device)
                idxs = None
            new_priorities = None

            if self.combined_reward or not n_step:
                elementwise_loss = self._dqn_loss(
                    states, actions, rewards, next_states, dones, self.gamma
                )

            if n_step:
                n_gamma = self.gamma**self.n_step
                n_step_elementwise_loss = self._dqn_loss(
                    n_states, n_actions, n_rewards, n_next_states, n_dones, n_gamma
                )
                if self.combined_reward:
                    elementwise_loss += n_step_elementwise_loss
                else:
                    elementwise_loss = n_step_elementwise_loss
            loss = torch.mean(elementwise_loss)

        self.optimizer.zero_grad()
        if self.accelerator is not None:
            self.accelerator.backward(loss)
        else:
            loss.backward()
        clip_grad_norm_(self.actor.parameters(), 10.0)
        self.optimizer.step()

        # soft update target network
        self.soft_update()
        self.actor.reset_noise()
        self.actor_target.reset_noise()

        if per:
            loss_for_prior = elementwise_loss.detach().cpu().numpy()
            new_priorities = loss_for_prior + self.prior_eps

        return loss.item(), idxs, new_priorities

    def soft_update(self):
        """Soft updates target network."""
        for eval_param, target_param in zip(
            self.actor.parameters(), self.actor_target.parameters()
        ):
            target_param.data.copy_(
                self.tau * eval_param.data + (1.0 - self.tau) * target_param.data
            )

    def test(self, env, swap_channels=False, max_steps=None, loop=3):
        """Returns mean test score of agent in environment with epsilon-greedy policy.

        :param env: The environment to be tested in
        :type env: Gym-style environment
        :param swap_channels: Swap image channels dimension from last to first [H, W, C] -> [C, H, W], defaults to False
        :type swap_channels: bool, optional
        :param max_steps: Maximum number of testing steps, defaults to None
        :type max_steps: int, optional
        :param loop: Number of testing loops/episodes to complete. The returned score is the mean over these tests. Defaults to 3
        :type loop: int, optional
        """
        with torch.no_grad():
            rewards = []
            num_envs = env.num_envs if hasattr(env, "num_envs") else 1
            for _ in range(loop):
                state, _ = env.reset()
                scores = np.zeros(num_envs)
                completed_episode_scores = np.zeros(num_envs)
                finished = np.zeros(num_envs)
                step = 0
                while not np.all(finished):
                    if swap_channels:
                        state = np.moveaxis(state, [-1], [-3])
                    action = self.get_action(state, training=False)
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

        actor = self.actor.clone()
        actor_target = self.actor_target.clone()
        optimizer = optim.Adam(actor.parameters(), lr=clone.lr)
        optimizer.load_state_dict(self.optimizer.state_dict())

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
        guarded_attributes = ["actor", "actor_target", "optimizer"]

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
            self.actor, self.actor_target, self.optimizer = self.accelerator.prepare(
                self.actor, self.actor_target, self.optimizer
            )

    def unwrap_models(self):
        if self.accelerator is not None:
            self.actor = self.accelerator.unwrap_model(self.actor)
            self.actor_target = self.accelerator.unwrap_model(self.actor_target)
            self.optimizer = unwrap_optimizer(self.optimizer, self.actor, self.lr)

    def save_checkpoint(self, path):
        """Saves a checkpoint of agent properties and network weights to path.

        :param path: Location to save checkpoint at
        :type path: string
        """
        attribute_dict = self.inspect_attributes()

        network_info = {
            "actor_init_dict": self.actor.init_dict,
            "actor_state_dict": self.actor.state_dict(),
            "actor_target_init_dict": self.actor_target.init_dict,
            "actor_target_state_dict": self.actor_target.state_dict(),
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
            "actor_target_state_dict",
            "optimizer_state_dict",
            "actor_init_dict",
            "actor_target_init_dict",
            "net_config",
            "lr",
        ]
        checkpoint = torch.load(path, map_location=self.device, pickle_module=dill)
        self.net_config = checkpoint["net_config"]
        if self.net_config is not None:
            self.arch = checkpoint["net_config"]["arch"]
            if self.net_config["arch"] == "mlp":
                network_class = EvolvableMLP
            elif self.net_config["arch"] == "cnn":
                network_class = EvolvableCNN
        else:
            network_class = MakeEvolvable
        self.actor = network_class(**checkpoint["actor_init_dict"])
        self.actor_target = network_class(**checkpoint["actor_target_init_dict"])

        self.lr = checkpoint["lr"]
        self.optimizer = optim.Adam(self.actor.parameters(), lr=self.lr)
        self.actor.load_state_dict(checkpoint["actor_state_dict"])
        self.actor_target.load_state_dict(checkpoint["actor_target_state_dict"])
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
        checkpoint["actor_target_init_dict"]["device"] = device

        actor_init_dict = chkpt_attribute_to_device(
            checkpoint.pop("actor_init_dict"), device
        )
        actor_target_init_dict = chkpt_attribute_to_device(
            checkpoint.pop("actor_target_init_dict"), device
        )
        actor_state_dict = chkpt_attribute_to_device(
            checkpoint.pop("actor_state_dict"), device
        )
        actor_target_state_dict = chkpt_attribute_to_device(
            checkpoint.pop("actor_target_state_dict"), device
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
                agent.actor_target = EvolvableMLP(**actor_target_init_dict)
            elif agent.arch == "cnn":
                agent.actor = EvolvableCNN(**actor_init_dict)
                agent.actor_target = EvolvableCNN(**actor_target_init_dict)
        else:
            class_init_dict["actor_network"] = MakeEvolvable(**actor_init_dict)
            agent = cls(**class_init_dict)
            agent.actor_target = MakeEvolvable(**actor_target_init_dict)

        agent.optimizer = optim.Adam(agent.actor.parameters(), lr=agent.lr)
        agent.actor.load_state_dict(actor_state_dict)
        agent.actor_target.load_state_dict(actor_target_state_dict)
        agent.optimizer.load_state_dict(optimizer_state_dict)

        if accelerator is not None:
            agent.wrap_models()

        for attribute in agent.inspect_attributes().keys():
            setattr(agent, attribute, checkpoint[attribute])

        return agent
