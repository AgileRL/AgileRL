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
from agilerl.utils.algo_utils import unwrap_optimizer
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
    :param n_step: Step number to calculate n-step td error, defaults to 3
    :type n_step: int, optional
    :param mut: Most recent mutation to agent, defaults to None
    :type mut: str, optional
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
        n_step=3,
        mut=None,
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

        self.support = torch.linspace(self.v_min, self.v_max, self.num_atoms)
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
            # model
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
                    output_vanish=False,
                    init_layers=False,
                    layer_norm=False,
                    num_atoms=self.num_atoms,
                    support=self.support,
                    rainbow=True,
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

    def getAction(self, state, action_mask=None, training=True):
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

        if len(state.size()) < 2:
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
        delta_z = float(self.v_max - self.v_min) / (self.num_atoms - 1)

        self.actor.eval()
        self.actor_target.eval()
        with torch.no_grad():
            next_actions = self.actor(next_states).argmax(1)
            next_dist = self.actor_target(next_states, q=False)
            next_dist = next_dist[range(self.batch_size), next_actions]

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
            else:
                offset = offset.to(self.accelerator.device)
                proj_dist = proj_dist.to(self.accelerator.device)

            proj_dist.view(-1).index_add_(
                0, (L + offset).view(-1), (next_dist * (u.float() - b)).view(-1)
            )
            proj_dist.view(-1).index_add_(
                0, (u + offset).view(-1), (next_dist * (b - L.float())).view(-1)
            )

        dist = self.actor(states, q=False)
        log_p = torch.log(dist[range(self.batch_size), actions.squeeze().long()])

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
            elementwise_loss = self._dqn_loss(
                states, actions, rewards, next_states, dones, self.gamma
            )
            if n_step:
                n_gamma = self.gamma**self.n_step
                n_step_elementwise_loss = self._dqn_loss(
                    n_states, n_actions, n_rewards, n_next_states, n_dones, n_gamma
                )
                elementwise_loss += n_step_elementwise_loss
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

        return loss.item(), idxs, new_priorities

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
        :param loop: Number of testing loops/episodes to complete. The returned score is the mean over these tests. Defaults to 3
        :type loop: int, optional
        """
        with torch.no_grad():
            rewards = []
            for i in range(loop):
                state = env.reset()[0]
                score = 0
                finished = False
                while not finished:
                    if swap_channels:
                        # Handle unvectorised image environment
                        if not hasattr(env, "num_envs"):
                            state = np.expand_dims(state, 0)
                        state = np.moveaxis(state, [-1], [-3])
                    action = self.getAction(state, training=False)
                    state, reward, done, trunc, _ = env.step(action)
                    score += reward[0]
                    if done[0] or trunc[0]:
                        finished = True
                rewards.append(score)
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

    def saveCheckpoint(self, path):
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

    def loadCheckpoint(self, path):
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
        checkpoint = torch.load(path, pickle_module=dill)
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
        checkpoint = torch.load(path, pickle_module=dill)
        checkpoint["actor_init_dict"]["device"] = device
        checkpoint["actor_target_init_dict"]["device"] = device

        actor_init_dict = checkpoint.pop("actor_init_dict")
        actor_target_init_dict = checkpoint.pop("actor_target_init_dict")
        actor_state_dict = checkpoint.pop("actor_state_dict")
        actor_target_state_dict = checkpoint.pop("actor_target_state_dict")
        optimizer_state_dict = checkpoint.pop("optimizer_state_dict")

        checkpoint["device"] = device
        checkpoint["accelerator"] = accelerator

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
