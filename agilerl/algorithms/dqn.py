from typing import Optional, Dict, Any, List
import copy
import inspect
import dill
import numpy as np
from numpy.typing import ArrayLike
import torch
import torch.nn as nn
import torch.optim as optim
from gymnasium import spaces
from tensordict import TensorDict, from_module
from tensordict.nn import CudaGraphModule

from agilerl.typing import TorchObsType, ExperiencesType, NumpyObsType, GymEnvType
from agilerl.algorithms.core import RLAlgorithm
from agilerl.algorithms.core.wrappers import OptimizerWrapper
from agilerl.algorithms.core.registry import NetworkGroup
from agilerl.modules.cnn import EvolvableCNN
from agilerl.modules.mlp import EvolvableMLP
from agilerl.modules.multi_input import EvolvableMultiInput
from agilerl.modules.base import EvolvableModule
from agilerl.wrappers.make_evolvable import MakeEvolvable
from agilerl.utils.algo_utils import (
    chkpt_attribute_to_device,
    unwrap_optimizer,
    obs_channels_to_first,
    make_safe_deepcopies
)

class DQN(RLAlgorithm):
    """The DQN algorithm class. DQN paper: https://arxiv.org/abs/1312.5602

    :param observation_space: Observation space of the environment
    :type observation_space: gym.spaces.Space
    :param action_space: Action space of the environment
    :type action_space: gym.spaces.Space
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
        net_config: Optional[Dict[str, Any]] = {"arch": "mlp", "hidden_size": [64, 64]},
        batch_size: int = 64,
        lr: float = 1e-4,
        learn_step: int = 5,
        gamma: float = 0.99,
        tau: float = 1e-3,
        mut: Optional[str] = None,
        double: bool = False,
        actor_network: Optional[EvolvableModule] = None,
        device: str = "cpu",
        accelerator: Optional[Any] = None,
        cudagraphs: bool = False,
        wrap: bool = True,
    ):
        super().__init__(
            observation_space,
            action_space,
            index=index,
            net_config=net_config,
            learn_step=learn_step,
            device=device,
            accelerator=accelerator,
            name="DQN"
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
        self.mut = mut
        self.double = double
        self.cudagraphs = cudagraphs
        self.capturable = cudagraphs

        if actor_network is not None:
            if isinstance(actor_network, (EvolvableMLP, EvolvableCNN)):
                self.net_config = actor_network.net_config
            elif isinstance(actor_network, MakeEvolvable):
                self.net_config = None
            else:
                assert (
                    False
                ), (f"'actor_network' argument is of type {type(actor_network)}, but "
                     "must be of type EvolvableMLP, EvolvableCNN or MakeEvolvable")
            
            # Need to make deepcopies 
            self.actor, self.actor_target = make_safe_deepcopies(actor_network, actor_network)
            self.actor_detached = make_safe_deepcopies(actor_network) if cudagraphs else None
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

                create_actor = lambda: EvolvableMLP(
                    num_inputs=self.state_dim[0],
                    num_outputs=self.action_dim,
                    device=self.device,
                    accelerator=self.accelerator,
                    **self.net_config,
                )
                self.actor = create_actor()
                self.actor_target = create_actor()
                self.actor_detached = create_actor() if cudagraphs else None

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
                
                create_actor = lambda: EvolvableCNN(
                    input_shape=self.state_dim,
                    num_outputs=self.action_dim,
                    device=self.device,
                    accelerator=self.accelerator,
                    **self.net_config,
                )

                self.actor = create_actor()
                self.actor_target = create_actor()
                self.actor_detached = create_actor() if cudagraphs else None

            elif self.net_config["arch"] == "composed": # Dict observations
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

                create_actor = lambda: EvolvableMultiInput(
                    observation_space=self.observation_space,
                    num_outputs=self.action_dim,
                    device=self.device,
                    accelerator=self.accelerator,
                    **self.net_config,
                )

                self.actor = create_actor()
                self.actor_target = create_actor()
                self.actor_detached = create_actor() if cudagraphs else None
        
        # Copy over actor weights to target
        self.init_hook()

        # Initialize optimizer with OptimizerWrapper
        self.optimizer = OptimizerWrapper(
            optim.Adam,
            networks=self.actor,
            optimizer_kwargs={"lr": self.lr, "capturable": self.capturable}
        )

        # TODO: Ideally refactor to remove this
        self.arch = (
            self.net_config["arch"] if self.net_config is not None else self.actor.arch
        )

        if self.accelerator is not None and wrap:
            self.wrap_models()

        self.criterion = nn.MSELoss()
        
        if self.cudagraphs:
            # torch.compile and cuda graph optimizations
            self.update = torch.compile(self.update, mode=None)
            self._get_action = torch.compile(self._get_action, mode=None, fullgraph=True)
            self.update = CudaGraphModule(self.update)
            self._get_action = CudaGraphModule(self._get_action)
        
        # Register DQN network groups and mutation hook
        shared_networks = [self.actor_target]
        if self.actor_detached is not None:
            shared_networks += [self.actor_detached]

        self.register_network_group(
            NetworkGroup(
                eval=self.actor,
                shared=shared_networks,
                policy=True
            )
        )
        self.register_init_hook(self.init_hook)
    
    def init_hook(self) -> None:
        """Resets module parameters for the detached and target networks."""
        self.param_vals: TensorDict = from_module(self.actor).detach()

        if self.actor_detached is not None:
            self.param_vals.to_module(self.actor_detached)

        # NOTE: This removes the target params from the computation graph which 
        # reduces memory overhead and speeds up training, however these won't 
        # appear in the module's parameters
        self.target_params: TensorDict = self.param_vals.clone().lock_()
        self.target_params.to_module(self.actor_target)

    def get_action(
            self,
            obs: NumpyObsType,
            epsilon: float = 0.0,
            action_mask: Optional[ArrayLike] = None
            ) -> ArrayLike:
        """Returns the next action to take in the environment.

        :param obs: The current observation from the environment
        :type obs: np.ndarray, dict[str, np.ndarray], tuple[np.ndarray]
        :param epsilon: Probablilty of taking a random action for exploration, defaults to 0
        :type epsilon: float, optional
        :param action_mask: Mask of legal actions 1=legal 0=illegal, defaults to None
        :type action_mask: numpy.ndarray, optional
        """
        # Preprocess observations and convert inputs to torch tensors
        torch_obs = self.preprocess_observation(obs)
        device = self.device if self.accelerator is None else self.accelerator.device
        epsilon = torch.tensor(epsilon, device=device)
        if action_mask is not None:
            action_mask = torch.as_tensor(action_mask, device=device)
        else:
            action_mask = torch.ones((torch_obs.shape[0], self.action_dim), device=device)

        return self._get_action(torch_obs, epsilon, action_mask).cpu().numpy()

    def _get_action(
            self,
            obs: TorchObsType,
            epsilon: Optional[torch.Tensor] = None,
            action_mask: Optional[torch.Tensor] = None
            ) -> torch.Tensor:
        """Returns the next action to take in the environment.
        Epsilon is the probability of taking a random action, used for exploration.
        For epsilon-greedy behaviour, set epsilon to 0.

        :param obs: The current observation from the environment
        :type obs: torch.Tensor, dict[str, torch.Tensor], tuple[torch.Tensor]
        :param epsilon: Probablilty of taking a random action for exploration, defaults to 0
        :type epsilon: float, optional
        :param action_mask: Mask of legal actions 1=legal 0=illegal, defaults to None
        :type action_mask: numpy.ndarray, optional
        """
        if not self.cudagraphs:
            with torch.no_grad():
                q_values = self.actor(obs)
        else:
            q_values = self.actor_detached(obs)

        # Masked random actions
        masked_random_values = torch.rand_like(q_values) * action_mask
        masked_random_actions = torch.argmax(masked_random_values, dim=-1)

        # Masked policy actions
        masked_q_values = q_values.masked_fill((1 - action_mask).bool(), float("-inf"))
        masked_policy_actions = torch.argmax(masked_q_values, dim=-1)

        # actions_random = torch.randint_like(actions, n_act)
        use_policy = torch.rand(masked_policy_actions.shape, device=q_values.device).gt(epsilon)

        # Recompute actions with masking
        actions = torch.where(use_policy, masked_policy_actions, masked_random_actions)

        return actions

    def update(
            self,
            obs: TorchObsType,
            actions: torch.Tensor,
            rewards: torch.Tensor,
            next_obs: TorchObsType,
            dones: torch.Tensor
            ) -> torch.Tensor:
        """Updates agent network parameters to learn from experiences.

        :param obs: List of batched states
        :type obs: torch.Tensor[float], dict[str, torch.Tensor[float]], tuple[torch.Tensor[float]]
        :param actions: List of batched actions
        :type actions: torch.Tensor[int]
        :param rewards: List of batched rewards
        :type rewards: torch.Tensor[float]
        :param next_obs: List of batched next states
        :type next_obs: torch.Tensor[float], dict[str, torch.Tensor[float]], tuple[torch.Tensor[float]]
        :param dones: List of batched dones
        :type dones: torch.Tensor[int]
        """
        with torch.no_grad():
            if self.double:  # Double Q-learning
                q_idx = self.actor_target(next_obs).argmax(dim=1).unsqueeze(1)
                q_target = self.actor(next_obs).gather(dim=1, index=q_idx).detach()
            else:
                q_target = self.actor_target(next_obs).max(axis=1)[0].unsqueeze(1)

            # target, if terminal then y_j = rewards
            y_j = rewards + self.gamma * q_target * (1 - dones)

        # Compute Q-values for actions taken and loss
        q_eval = self.actor(obs).gather(1, actions.long())
        loss: torch.Tensor = self.criterion(q_eval, y_j)

        # zero gradients, perform a backward pass, and update the weights
        self.optimizer.zero_grad()
        if self.accelerator is not None:
            self.accelerator.backward(loss)
        else:
            loss.backward()

        self.optimizer.step()
        return loss.detach()

    def learn(self, experiences: ExperiencesType) -> float:
        """Updates agent network parameters to learn from experiences.

        :param experiences: List of batched states, actions, rewards, next_states, dones in that order.
        :type state: list[torch.Tensor[float]]
        """
        states, actions, rewards, next_states, dones = experiences
        if self.accelerator is not None:
            actions = actions.to(self.accelerator.device)
            rewards = rewards.to(self.accelerator.device)
            dones = dones.to(self.accelerator.device)

        states = self.preprocess_observation(states)
        next_states = self.preprocess_observation(next_states)

        loss = self.update(states, actions, rewards, next_states, dones)

        # soft update target network
        self.soft_update()
        return loss.item()
    
    def soft_update(self) -> None:
        "Soft updates target network."
        self.target_params.lerp_(self.param_vals, self.tau)

    def test(
            self,
            env: GymEnvType,
            swap_channels: bool = False,
            max_steps: Optional[int] = None,
            loop: int = 1
            ) -> float:
        """Returns mean test score of agent in environment with epsilon-greedy policy.

        :param env: The environment to be tested in
        :type env: Gym-style environment
        :param swap_channels: Swap image channels dimension from last to first [H, W, C] -> [C, H, W], defaults to False
        :type swap_channels: bool, optional
        :param max_steps: Maximum number of testing steps, defaults to None
        :type max_steps: int, optional
        :param loop: Number of testing loops/episodes to complete. The returned score is the mean over these tests. Defaults to 1
        :type loop: int, optional
        """
        with torch.no_grad():
            rewards = []
            num_envs = env.num_envs if hasattr(env, "num_envs") else 1
            for _ in range(loop):
                state, info = env.reset()
                scores = np.zeros(num_envs)
                completed_episode_scores = np.zeros(num_envs)
                finished = np.zeros(num_envs)
                step = 0
                while not np.all(finished):
                    if swap_channels:
                        state = obs_channels_to_first(state)

                    action_mask = info.get("action_mask", None)
                    action = self.get_action(state, epsilon=0.0, action_mask=action_mask)
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

        if self.actor_detached is not None:
            actor_detached = self.actor_detached.clone()
        else:
            actor_detached = None

        optimizer = OptimizerWrapper(
            optim.Adam,
            networks=actor,
            optimizer_kwargs={"lr": self.lr, "capturable": self.cudagraphs},
            network_names=self.optimizer.network_names,
        )
        optimizer.load_state_dict(self.optimizer.state_dict())

        if self.accelerator is not None and wrap:
            (
                clone.actor,
                clone.optimizer,
            ) = self.accelerator.prepare(actor, optimizer.optimizer)
        else:
            clone.actor = actor
            clone.optimizer = optimizer
    
        clone.actor_detached = actor_detached
        clone.actor_target = actor_target
        
        TensorType = (torch.Tensor, TensorDict)
        for attribute in self.inspect_attributes().keys():
            if hasattr(self, attribute) and hasattr(clone, attribute):
                attr, clone_attr = getattr(self, attribute), getattr(clone, attribute)
                if isinstance(attr, TensorType) or isinstance(clone_attr, TensorType):
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
            self.optimizer = unwrap_optimizer(self.optimizer, self.actor, lr=self.lr)

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
        self.optimizer = OptimizerWrapper(
            optim.Adam,
            networks=self.actor,
            optimizer_kwargs={"lr": self.lr, "capturable": self.cudagraphs}
        )
        self.actor.load_state_dict(checkpoint["actor_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.init_hook()

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
            self = cls(**class_init_dict)
            self.arch = checkpoint["net_config"]["arch"]
            if self.arch == "mlp":
                self.actor = EvolvableMLP(**actor_init_dict)
                self.actor_target = EvolvableMLP(**actor_target_init_dict)
                self.actor_detached = EvolvableMLP(**actor_init_dict)
            elif self.arch == "cnn":
                self.actor = EvolvableCNN(**actor_init_dict)
                self.actor_target = EvolvableCNN(**actor_target_init_dict)
                self.actor_detached = EvolvableCNN(**actor_init_dict)
        else:
            class_init_dict["actor_network"] = MakeEvolvable(**actor_init_dict)
            self = cls(**class_init_dict)
            self.actor_target = MakeEvolvable(**actor_target_init_dict)
            self.actor_detached = MakeEvolvable(**actor_init_dict)

        self.optimizer = OptimizerWrapper(
            optim.Adam,
            networks=self.actor,
            optimizer_kwargs={"lr": self.lr, "capturable": self.cudagraphs}
        )
        self.actor.load_state_dict(actor_state_dict)
        self.optimizer.load_state_dict(optimizer_state_dict)
        self.init_hook()

        if accelerator is not None:
            self.wrap_models()

        for attribute in self.inspect_attributes().keys():
            setattr(self, attribute, checkpoint[attribute])

        return self
