import copy
import inspect

import dill
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from agilerl.networks.evolvable_cnn import EvolvableCNN
from agilerl.networks.evolvable_mlp import EvolvableMLP
from agilerl.wrappers.make_evolvable import MakeEvolvable


class NeuralTS:
    """The NeuralTS algorithm class. NeuralTS paper: https://arxiv.org/abs/1911.04462

    :param state_dim: State observation (context) dimension
    :type state_dim: list[int]
    :param action_dim: Action dimension
    :type action_dim: int
    :param index: Index to keep track of object instance during tournament selection and mutation, defaults to 0
    :type index: int, optional
    :param net_config: Network configuration, defaults to mlp with hidden size [64,64]
    :type net_config: dict, optional
    :param gamma: Positive scaling factor, defaults to 1.0
    :type gamma: float, optional
    :param lamb: Regularization parameter lambda, defaults to 1.0
    :type lamb: float, optional
    :param reg: Loss regularization parameter, defaults to 0.000625
    :type reg: float, optional
    :param batch_size: Size of batched sample from replay buffer for learning, defaults to 64
    :type batch_size: int, optional
    :param lr: Learning rate for optimizer, defaults to 1e-4
    :type lr: float, optional
    :param learn_step: Learning frequency, defaults to 1
    :type learn_step: int, optional
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
        index=0,
        net_config={"arch": "mlp", "h_size": [128]},
        gamma=1.0,
        lamb=1.0,
        reg=0.000625,
        batch_size=64,
        lr=3e-4,
        learn_step=1,
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
        assert isinstance(index, int), "Agent index must be an integer."
        assert isinstance(
            gamma, (float, int)
        ), "Scaling factor must be a float or integer."
        assert gamma > 0, "Scaling factor must be positive."
        assert isinstance(
            lamb, (float, int)
        ), "Regularization parameter lambda must be a float or integer."
        assert lamb > 0, "Regularization parameter lambda must be greater than zero."
        assert isinstance(reg, float), "Loss regularization parameter must be a float."
        assert reg > 0, "Loss regularization parameter must be greater than zero."
        assert isinstance(batch_size, int), "Batch size must be an integer."
        assert batch_size >= 1, "Batch size must be greater than or equal to one."
        assert isinstance(lr, float), "Learning rate must be a float."
        assert lr > 0, "Learning rate must be greater than zero."
        assert isinstance(learn_step, int), "Learn step rate must be an integer."
        assert learn_step >= 1, "Learn step must be greater than or equal to one."
        assert (
            isinstance(actor_network, nn.Module) or actor_network is None
        ), "Actor network must be an nn.Module or None."
        assert isinstance(
            wrap, bool
        ), "Wrap models flag must be boolean value True or False."

        self.algo = "NeuralTS"
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.net_config = net_config
        self.gamma = gamma
        self.lamb = lamb
        self.reg = reg
        self.batch_size = batch_size
        self.lr = lr
        self.learn_step = learn_step
        self.mut = mut
        self.device = device
        self.accelerator = accelerator
        self.index = index
        self.scores = []
        self.regret = [0]
        self.fitness = []
        self.steps = [0]
        self.actor_network = actor_network

        if self.actor_network is not None:
            self.actor = actor_network
            self.net_config = None
        else:
            # model
            assert isinstance(self.net_config, dict), "Net config must be a dictionary."
            assert (
                "arch" in self.net_config.keys()
            ), "Net config must contain arch: 'mlp' or 'cnn'."
            if self.net_config["arch"] == "mlp":  # Multi-layer Perceptron
                assert (
                    "h_size" in self.net_config.keys()
                ), "Net config must contain h_size: int."
                assert isinstance(
                    self.net_config["h_size"], list
                ), "Net config h_size must be a list."
                assert (
                    len(self.net_config["h_size"]) > 0
                ), "Net config h_size must contain at least one element."
                self.actor = EvolvableMLP(
                    num_inputs=state_dim[0],
                    num_outputs=1,
                    hidden_size=self.net_config["h_size"],
                    device=self.device,
                    accelerator=self.accelerator,
                )
            elif self.net_config["arch"] == "cnn":  # Convolutional Neural Network
                for key in ["c_size", "k_size", "s_size", "h_size"]:
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
                    num_actions=1,
                    channel_size=self.net_config["c_size"],
                    kernel_size=self.net_config["k_size"],
                    stride_size=self.net_config["s_size"],
                    hidden_size=self.net_config["h_size"],
                    normalize=self.net_config["normalize"],
                    device=self.device,
                    accelerator=self.accelerator,
                )

        self.numel = sum(w.numel() for w in self.actor.parameters() if w.requires_grad)
        self.sigma_inv = lamb * np.eye(self.numel)
        self.theta_0 = torch.cat(
            [w.flatten() for w in self.actor.parameters() if w.requires_grad]
        )

        self.optimizer_type = optim.Adam(self.actor.parameters(), lr=self.lr)

        self.arch = (
            self.net_config["arch"] if self.net_config is not None else self.actor.arch
        )

        if self.accelerator is not None:
            self.optimizer = self.optimizer_type
            if wrap:
                self.wrap_models()
        else:
            self.actor = self.actor.to(self.device)
            self.optimizer = self.optimizer_type

        self.criterion = nn.MSELoss()

    def getAction(self, state, action_mask=None):
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

        if len(state.size()) < 2:
            state = state.unsqueeze(0)

        mu = self.actor(state)
        g = np.zeros((self.action_dim, self.numel))
        for k, fx in enumerate(mu):
            self.optimizer.zero_grad()
            fx.backward(retain_graph=True)
            g[k] = (
                torch.cat(
                    [
                        w.grad.detach().flatten() / np.sqrt(self.actor.hidden_size[-1])
                        for w in self.actor.parameters()
                        if w.requires_grad
                    ]
                )
                .cpu()
                .numpy()
            )

        with torch.no_grad():
            action_values = np.random.normal(
                loc=self.actor(state).cpu().numpy(),
                scale=self.gamma
                * np.sqrt(
                    np.matmul(np.matmul(g[:, None, :], self.sigma_inv), g[:, :, None])[
                        :, 0, :
                    ]
                ),
            )

        if action_mask is None:
            action = np.argmax(action_values)
        else:
            inv_mask = 1 - action_mask
            masked_action_values = np.ma.array(action_values, mask=inv_mask)
            action = np.argmax(masked_action_values)

        # Sherman-Morrison-Woodbury Update
        v = np.expand_dims(g[action], -1)
        self.sigma_inv -= (self.sigma_inv @ v @ v.T @ self.sigma_inv) / (
            1 + v.T @ self.sigma_inv @ v
        )

        return action

    def learn(self, experiences):
        """Updates agent network parameters to learn from experiences.

        :param experiences: Batched states, rewards in that order.
        :type state: list[torch.Tensor[float]]
        """
        states, rewards = experiences
        if self.accelerator is not None:
            states = states.to(self.accelerator.device)
            rewards = rewards.to(self.accelerator.device)

        pred_rewards = self.actor(states)

        # loss backprop
        loss = self.criterion(rewards, pred_rewards)
        loss += (
            self.reg
            * torch.norm(
                torch.cat(
                    [w.flatten() for w in self.actor.parameters() if w.requires_grad]
                )
                - self.theta_0
            )
            ** 2
        )
        self.optimizer.zero_grad()
        if self.accelerator is not None:
            self.accelerator.backward(loss)
        else:
            loss.backward()
        self.optimizer.step()

        return loss.item()

    def test(self, env, swap_channels=False, max_steps=100, loop=1):
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
                state = env.reset()
                score = 0
                for idx_step in range(max_steps):
                    if swap_channels:
                        state = np.moveaxis(state, [-1], [-3])
                    action = np.argmax(self.actor(state).cpu().numpy())
                    state, reward = env.step(action)
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
        input_args = self.inspect_attributes(input_args_only=True)
        input_args["wrap"] = wrap

        if index is None:
            input_args["index"] = self.index
        clone = type(self)(**input_args)

        actor = self.actor.clone()
        optimizer = optim.Adam(actor.parameters(), lr=clone.lr)
        clone.optimizer_type = optimizer
        if self.accelerator is not None:
            if wrap:
                (
                    clone.actor,
                    clone.optimizer,
                ) = self.accelerator.prepare(actor, optimizer)
            else:
                clone.actor, clone.optimizer = (
                    actor,
                    optimizer,
                )
        else:
            clone.actor = actor.to(self.device)
            clone.optimizer = optimizer

        for attribute in self.inspect_attributes().keys():
            if hasattr(self, attribute) and hasattr(clone, attribute):
                attr, clone_attr = getattr(self, attribute), getattr(clone, attribute)
                if isinstance(attr, torch.Tensor) or isinstance(
                    clone_attr, torch.Tensor
                ):
                    if not torch.equal(attr, clone_attr):
                        setattr(clone, attribute, torch.clone(getattr(self, attribute)))
                elif isinstance(attr, np.ndarray) or isinstance(clone_attr, np.ndarray):
                    if not np.array_equal(attr, clone_attr):
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

        clone.numel = sum(
            w.numel() for w in clone.actor.parameters() if w.requires_grad
        )
        clone.theta_0 = torch.cat(
            [w.flatten() for w in clone.actor.parameters() if w.requires_grad]
        )

        return clone

    def inspect_attributes(self, input_args_only=False):
        # Get all attributes of the current object
        attributes = inspect.getmembers(self, lambda a: not (inspect.isroutine(a)))
        guarded_attributes = ["actor", "optimizer", "optimizer_type"]

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
            self.actor, self.optimizer = self.accelerator.prepare(
                self.actor, self.optimizer
            )

    def unwrap_models(self):
        if self.accelerator is not None:
            self.actor = self.accelerator.unwrap_model(self.actor)
            self.optimizer = self.accelerator.unwrap_model(self.optimizer)

    def saveCheckpoint(self, path):
        """Saves a checkpoint of agent properties and network weights to path.

        :param path: Location to save checkpoint at
        :type path: string
        """

        attribute_dict = self.inspect_attributes()

        network_info = {
            "actor_init_dict": self.actor.init_dict,
            "actor_state_dict": self.actor.state_dict(),
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
            "optimizer_state_dict",
            "actor_init_dict",
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

        self.lr = checkpoint["lr"]
        self.optimizer = optim.Adam(self.actor.parameters(), lr=self.lr)
        self.actor.load_state_dict(checkpoint["actor_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        for attribute in checkpoint.keys():
            if attribute not in network_info:
                setattr(self, attribute, checkpoint[attribute])

        self.numel = sum(w.numel() for w in self.actor.parameters() if w.requires_grad)
        self.theta_0 = torch.cat(
            [w.flatten() for w in self.actor.parameters() if w.requires_grad]
        )

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

        actor_init_dict = checkpoint.pop("actor_init_dict")
        actor_state_dict = checkpoint.pop("actor_state_dict")
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
            elif agent.arch == "cnn":
                agent.actor = EvolvableCNN(**actor_init_dict)
        else:
            class_init_dict["actor_network"] = MakeEvolvable(**actor_init_dict)
            agent = cls(**class_init_dict)

        agent.optimizer = optim.Adam(agent.actor.parameters(), lr=agent.lr)
        agent.actor.load_state_dict(actor_state_dict)
        agent.optimizer.load_state_dict(optimizer_state_dict)

        if accelerator is not None:
            agent.wrap_models()

        for attribute in agent.inspect_attributes().keys():
            setattr(agent, attribute, checkpoint[attribute])

        agent.numel = sum(
            w.numel() for w in agent.actor.parameters() if w.requires_grad
        )
        agent.theta_0 = torch.cat(
            [w.flatten() for w in agent.actor.parameters() if w.requires_grad]
        )

        return agent
