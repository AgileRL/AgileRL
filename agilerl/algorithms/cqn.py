import copy
import inspect
import random

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


class CQN:
    """The CQN algorithm class. CQN paper: https://arxiv.org/abs/2006.04779

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
    :param mut: Most recent mutation to agent, defaults to None
    :type mut: str, optional
    :param double: Use double Q-learning, defaults to False
    :type double: bool, optional
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
        mut=None,
        double=False,
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
            double, bool
        ), "Double Q-learning flag must be boolean value True or False."
        assert isinstance(
            wrap, bool
        ), "Wrap models flag must be boolean value True or False."

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
        self.mut = mut
        self.device = device
        self.accelerator = accelerator

        self.index = index
        self.scores = []
        self.fitness = []
        self.steps = [0]

        self.double = double
        self.actor_network = actor_network

        if self.actor_network is not None:
            self.actor = actor_network
            if isinstance(self.actor, (EvolvableMLP, EvolvableCNN)):
                self.net_config = self.actor.net_config
                self.actor_network = None
            elif isinstance(self.actor, MakeEvolvable):
                self.net_config = None
                self.actor_network = actor_network
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
                self.actor = EvolvableMLP(
                    num_inputs=state_dim[0],
                    num_outputs=action_dim,
                    hidden_size=self.net_config["hidden_size"],
                    device=self.device,
                    accelerator=self.accelerator,
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
                    channel_size=self.net_config["channel_size"],
                    kernel_size=self.net_config["kernel_size"],
                    stride_size=self.net_config["stride_size"],
                    hidden_size=self.net_config["hidden_size"],
                    normalize=self.net_config["normalize"],
                    device=self.device,
                    accelerator=self.accelerator,
                )

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

        self.criterion = nn.MSELoss()

    def get_action(self, state, epsilon=0, action_mask=None):
        """Returns the next action to take in the environment. Epsilon is the
        probability of taking a random action, used for exploration.
        For epsilon-greedy behaviour, set epsilon to 0.

        :param state: State observation, or multiple observations in a batch
        :type state: numpy.ndarray[float]
        :param epsilon: Probablilty of taking a random action for exploration, defaults to 0
        :type epsilon: float, optional
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

    def learn(self, experiences):
        """Updates agent network parameters to learn from experiences.

        :param experiences: List of batched states, actions, rewards, next_states, dones in that order.
        :type state: list[torch.Tensor[float]]
        """
        states, actions, rewards, next_states, dones = experiences
        if self.accelerator is not None:
            states = states.to(self.accelerator.device)
            actions = actions.to(self.accelerator.device)
            rewards = rewards.to(self.accelerator.device)
            next_states = next_states.to(self.accelerator.device)
            dones = dones.to(self.accelerator.device)

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
        self.soft_update()

        return q1_loss.item()

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
        :param max_steps: Maximum number of testing steps, defaults to 500
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
                    action = self.get_action(state, epsilon=0)
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
