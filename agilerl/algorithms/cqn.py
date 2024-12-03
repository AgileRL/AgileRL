from typing import Optional, Dict, Any, Tuple
import copy
import inspect
import random
import dill
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from gymnasium import spaces

from agilerl.modules.multi_input import EvolvableMultiInput
from agilerl.modules.cnn import EvolvableCNN
from agilerl.modules.mlp import EvolvableMLP
from agilerl.algorithms.base import RLAlgorithm
from agilerl.utils.algo_utils import chkpt_attribute_to_device, unwrap_optimizer, obs_channels_to_first
from agilerl.wrappers.make_evolvable import MakeEvolvable
from agilerl.typing import NumpyObsType, TorchObsType, ObservationType

class CQN(RLAlgorithm):
    """The CQN algorithm class. CQN paper: https://arxiv.org/abs/2006.04779

    :param observation_space: The observation space of the environment.
    :type observation_space: spaces.Space
    :param action_space: The action space of the environment.
    :type action_space: spaces.Space
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
        observation_space: spaces.Space,
        action_space: spaces.Space,
        index: int = 0,
        net_config: Dict[str, Any] = {"arch": "mlp", "hidden_size": [64, 64]},
        batch_size: int = 64,
        lr: float = 1e-4,
        learn_step: int = 5,
        gamma: float = 0.99,
        tau: float = 1e-3,
        double: bool = False,
        normalize_images: bool = True,
        mut: Optional[str] = None,
        actor_network: Optional[nn.Module] = None,
        device: str = "cpu",
        accelerator: Optional[Any] = None,
        wrap: bool = True
    ) -> None:
        super().__init__(
            observation_space,
            action_space,
            index=index,
            net_config=net_config,
            learn_step=learn_step,
            device=device,
            accelerator=accelerator,
            normalize_images=normalize_images,
            name="CQN"
            )

        assert isinstance(batch_size, int), "Batch size must be an integer."
        assert batch_size >= 1, "Batch size must be greater than or equal to one."
        assert isinstance(lr, float), "Learning rate must be a float."
        assert lr > 0, "Learning rate must be greater than zero."
        assert isinstance(gamma, (float, int)), "Gamma must be a float."
        assert isinstance(tau, float), "Tau must be a float."
        assert tau > 0, "Tau must be greater than zero."
        assert isinstance(
            double, bool
        ), "Double Q-learning flag must be boolean value True or False."
        assert isinstance(
            wrap, bool
        ), "Wrap models flag must be boolean value True or False."

        self.batch_size = batch_size
        self.lr = lr
        self.gamma = gamma
        self.tau = tau
        self.double = double
        self.actor_network = actor_network
        self.mut = mut

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
            ), "Net config must contain arch: 'mlp', 'cnn', or 'composed'."
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
                    num_inputs=self.state_dim[0],
                    num_outputs=self.action_dim,
                    hidden_size=self.net_config["hidden_size"],
                    device='cpu', # Use CPU since we will make deepcopy for target
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

                self.actor = EvolvableCNN(
                    input_shape=self.state_dim,
                    num_outputs=self.action_dim,
                    channel_size=self.net_config["channel_size"],
                    kernel_size=self.net_config["kernel_size"],
                    stride_size=self.net_config["stride_size"],
                    hidden_size=self.net_config["hidden_size"],
                    device='cpu', # Use CPU since we will make deepcopy for target
                    accelerator=self.accelerator,
                )
            elif self.net_config["arch"] == "composed":  # Composed network
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
                    "latent_dim" in self.net_config.keys()
                ), "Net config must contain latent_dim: int."

                self.actor = EvolvableMultiInput(
                    observation_space=self.observation_space,
                    num_outputs=self.action_dim,
                    channel_size=self.net_config["channel_size"],
                    kernel_size=self.net_config["kernel_size"],
                    stride_size=self.net_config["stride_size"],
                    hidden_size=self.net_config["hidden_size"],
                    normalize=self.net_config["normalize"],
                    latent_dim=self.net_config["latent_dim"],
                    device='cpu', # Use CPU since we will make deepcopy for target
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

    def get_action(
            self,
            state: np.ndarray,
            epsilon: float = 0,
            action_mask: Optional[np.ndarray] = None
            ) -> np.ndarray:
        """Returns the next action to take in the environment. Epsilon is the
        probability of taking a random action, used for exploration.
        For epsilon-greedy behaviour, set epsilon to 0.

        :param state: State observation, or multiple observations in a batch
        :type state: numpy.ndarray[float]
        :param epsilon: Probablilty of taking a random action for exploration, defaults to 0
        :type epsilon: float, optional
        :param action_mask: Mask of legal actions 1=legal 0=illegal, defaults to None
        :type action_mask: numpy.ndarray, optional

        :return: Action to take in the environment
        :rtype: numpy.ndarray[int]
        """
        state = self.preprocess_observation(state)

        # epsilon-greedy
        if random.random() < epsilon:
            if action_mask is None:
                action = np.random.randint(0, self.action_dim, size=len(state))
            else:
                action = np.argmax(
                    (
                        np.random.uniform(0, 1, (len(state), self.action_dim))
                        * action_mask
                    ),
                    axis=1,
                )

        else:
            self.actor.eval()
            with torch.no_grad():
                action_values = self.actor(state).cpu().data.numpy()
            self.actor.train()

            if action_mask is None:
                action = np.argmax(action_values, axis=-1)
            else:
                inv_mask = 1 - action_mask
                masked_action_values = np.ma.array(action_values, mask=inv_mask)
                action = np.argmax(masked_action_values, axis=-1)

        return action

    def learn(self, experiences: Tuple[torch.Tensor, ...]) -> float:
        """Updates agent network parameters to learn from experiences.

        :param experiences: List of batched states, actions, rewards, next_states, dones in that order.
        :type state: list[torch.Tensor[float]]

        :return: Loss from learning
        :rtype: float
        """
        states, actions, rewards, next_states, dones = experiences
        if self.accelerator is not None:
            states = states.to(self.accelerator.device)
            actions = actions.to(self.accelerator.device)
            rewards = rewards.to(self.accelerator.device)
            next_states = next_states.to(self.accelerator.device)
            dones = dones.to(self.accelerator.device)

        states = self.preprocess_observation(states)
        next_states = self.preprocess_observation(next_states)

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

    def soft_update(self) -> None:
        """Soft updates target network."""
        for eval_param, target_param in zip(
            self.actor.parameters(), self.actor_target.parameters()
        ):
            target_param.data.copy_(
                self.tau * eval_param.data + (1.0 - self.tau) * target_param.data
            )

    def test(self, env, swap_channels: bool = False, max_steps: Optional[int] = None, loop: int=3):
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
                state, info = env.reset()
                scores = np.zeros(num_envs)
                completed_episode_scores = np.zeros(num_envs)
                finished = np.zeros(num_envs)
                step = 0
                while not np.all(finished):
                    if swap_channels:
                        state = obs_channels_to_first(state)

                    action_mask = info.get("action_mask", None)
                    action = self.get_action(state, epsilon=0, action_mask=action_mask)
                    state, reward, done, trunc, info = env.step(action)
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

    def clone(self, index: Optional[int] = None, wrap: bool = True) -> "CQN":
        """Returns cloned agent identical to self.

        :param index: Index to keep track of agent for tournament selection and mutation, defaults to None
        :type index: int, optional
        :param wrap: Wrap models for distributed training upon creation, defaults to True
        :type wrap: bool, optional
        """
        input_args = self.inspect_attributes(input_args_only=True)
        input_args["wrap"] = wrap
        clone = type(self)(**input_args)

        actor = self.actor.clone()
        actor_target = self.actor_target.clone()
        optimizer = optim.Adam(actor.parameters(), lr=clone.lr)
        if self.accelerator is not None and wrap:
            (
                clone.actor,
                clone.actor_target,
                clone.optimizer,
            ) = self.accelerator.prepare(actor, actor_target, optimizer)
        else:
            clone.actor = actor
            clone.actor_target = actor_target
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

    def unwrap_models(self):
        if self.accelerator is not None:
            self.actor = self.accelerator.unwrap_model(self.actor)
            self.actor_target = self.accelerator.unwrap_model(self.actor_target)
            self.optimizer = unwrap_optimizer(self.optimizer, self.actor, self.lr)

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
