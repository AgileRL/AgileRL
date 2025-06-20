from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from gymnasium import spaces

from agilerl.algorithms.core import OptimizerWrapper, RLAlgorithm
from agilerl.algorithms.core.registry import HyperparameterConfig, NetworkGroup
from agilerl.modules import EvolvableModule
from agilerl.networks.value_networks import ValueNetwork
from agilerl.typing import ArrayLike, ExperiencesType, GymEnvType, ObservationType
from agilerl.utils.algo_utils import make_safe_deepcopies, obs_channels_to_first
from agilerl.utils.evolvable_networks import get_default_encoder_config


class NeuralTS(RLAlgorithm):
    """Neural Thompson Sampling (NeuralTS) algorithm.

    Paper: https://arxiv.org/abs/2010.00827

    :param observation_space: Observation space of the environment
    :type observation_space: gym.spaces.Space
    :param action_space: Action space of the environment
    :type action_space: gym.spaces.Space
    :param index: Index to keep track of object instance during tournament selection and mutation, defaults to 0
    :type index: int, optional
    :param hp_config: RL hyperparameter mutation configuration, defaults to None, whereby algorithm mutations are disabled.
    :type hp_config: HyperparameterConfig, optional
    :param net_config: Network configuration, defaults to None
    :type net_config: dict, optional
    :param gamma: Positive scaling factor, defaults to 1.0
    :type gamma: float, optional
    :param lamb: Regularization parameter lambda, defaults to 1.0
    :type lamb: float, optional
    :param reg: Loss regularization parameter, defaults to 0.000625
    :type reg: float, optional
    :param batch_size: Size of batched sample from replay buffer for learning, defaults to 64
    :type batch_size: int, optional
    :param lr: Learning rate for optimizer, defaults to 3e-3
    :type lr: float, optional
    :param normalize_images: Normalize images flag, defaults to True
    :type normalize_images: bool, optional
    :param learn_step: Learning frequency, defaults to 2
    :type learn_step: int, optional
    :param mut: Most recent mutation to agent, defaults to None
    :type mut: str, optional
    :param actor_network: Custom actor network, defaults to None
    :type actor_network: EvolvableModule, optional
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
        hp_config: Optional[HyperparameterConfig] = None,
        net_config: Optional[Dict[str, Any]] = None,
        gamma: float = 1.0,
        lamb: float = 1.0,
        reg: float = 0.000625,
        batch_size: int = 64,
        lr: float = 3e-3,
        normalize_images: bool = True,
        learn_step: int = 2,
        mut: Optional[str] = None,
        actor_network: Optional[EvolvableModule] = None,
        device: str = "cpu",
        accelerator: Optional[Any] = None,
        wrap: bool = True,
    ):
        super().__init__(
            observation_space,
            action_space,
            index=index,
            hp_config=hp_config,
            device=device,
            accelerator=accelerator,
            normalize_images=normalize_images,
            name="NeuralTS",
        )

        assert learn_step >= 1, "Learn step must be greater than or equal to one."
        assert isinstance(learn_step, int), "Learn step rate must be an integer."
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
        assert isinstance(
            wrap, bool
        ), "Wrap models flag must be boolean value True or False."

        self.gamma = gamma
        self.learn_step = learn_step
        self.lamb = lamb
        self.reg = reg
        self.batch_size = batch_size
        self.lr = lr
        self.mut = mut
        self.net_config = net_config
        self.regret = [0]

        if actor_network is not None:
            if not isinstance(actor_network, EvolvableModule):
                raise TypeError(
                    f"'actor_network' argument is of type {type(actor_network)}, but must be of type EvolvableModule."
                )

            # Need to make deepcopies for target and detached networks
            self.actor = make_safe_deepcopies(actor_network)
        else:
            net_config = {} if net_config is None else net_config
            simba = net_config.get("simba", False)
            encoder_config = (
                get_default_encoder_config(observation_space, simba)
                if net_config.get("encoder_config") is None
                else net_config["encoder_config"]
            )

            if not simba and not isinstance(
                observation_space, (spaces.Dict, spaces.Tuple)
            ):
                # Layer norm is not used in the original implementation
                encoder_config["layer_norm"] = False

            net_config["encoder_config"] = encoder_config

            self.actor = ValueNetwork(
                observation_space=observation_space, device=self.device, **net_config
            )

        self.optimizer = OptimizerWrapper(optim.Adam, networks=self.actor, lr=self.lr)

        if self.accelerator is not None and wrap:
            self.wrap_models()

        # Initialize network layers
        self.actor.init_weights_gaussian(std_coeff=4, output_coeff=2)
        self.init_params()

        self.criterion = nn.MSELoss()

        # Register network groups for mutations
        self.register_mutation_hook(self.init_params)
        self.register_network_group(
            NetworkGroup(eval_network=self.actor, shared_networks=None, policy=True)
        )

    def init_params(self) -> None:
        """Initializes parameters for the agent network."""
        self.exp_layer = self.actor.get_output_dense()

        self.numel = sum(
            w.numel() for w in self.exp_layer.parameters() if w.requires_grad
        )
        self.sigma_inv = self.lamb * torch.eye(self.numel).to(self.device)
        self.theta_0 = torch.cat(
            [w.flatten() for w in self.exp_layer.parameters() if w.requires_grad]
        )

    def get_action(
        self, obs: ObservationType, action_mask: Optional[ArrayLike] = None
    ) -> int:
        """Returns the next action to take in the environment.

        :param obs: State observation, or multiple observations in a batch
        :type obs: numpy.ndarray[float]
        :param action_mask: Mask of legal actions 1=legal 0=illegal, defaults to None
        :type action_mask: numpy.ndarray, optional
        :return: Action to take in the environment
        :rtype: int
        """
        obs = self.preprocess_observation(obs)

        mu = self.actor(obs)
        g = torch.zeros((self.action_dim, self.numel)).to(self.device)
        for k, fx in enumerate(mu):
            self.optimizer.zero_grad()
            fx.backward(retain_graph=True)
            g[k] = torch.cat(
                [
                    w.grad.detach().flatten() / np.sqrt(self.exp_layer.weight.size(0))
                    for w in self.exp_layer.parameters()
                    if w.requires_grad
                ]
            )

        with torch.no_grad():
            action_values = torch.normal(
                mean=self.actor(obs),
                std=self.gamma
                * torch.sqrt(
                    torch.matmul(
                        torch.matmul(g[:, None, :], self.sigma_inv), g[:, :, None]
                    )[:, 0, :]
                ),
            )

        action_values = action_values.cpu().numpy()
        if action_mask is None:
            action = np.argmax(action_values)
        else:
            inv_mask = 1 - action_mask
            masked_action_values = np.ma.array(action_values, mask=inv_mask)
            action = np.argmax(masked_action_values)

        # Sherman-Morrison-Woodbury Update
        v = g[action].unsqueeze(-1)
        self.sigma_inv -= (self.sigma_inv @ v @ v.T @ self.sigma_inv) / (
            1 + v.T @ self.sigma_inv @ v
        )

        return action

    def learn(self, experiences: ExperiencesType) -> float:
        """Updates agent network parameters to learn from experiences.

        :param experiences: Batched states, rewards in that order.
        :type experiences: dict[str, torch.Tensor[float]]
        """
        states = experiences["obs"]
        rewards = experiences["reward"]

        pred_rewards = self.actor(states)

        # loss backprop
        loss = self.criterion(rewards, pred_rewards)
        loss += (
            self.reg
            * torch.norm(
                torch.cat(
                    [
                        w.flatten()
                        for w in self.exp_layer.parameters()
                        if w.requires_grad
                    ]
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

    def test(
        self,
        env: GymEnvType,
        swap_channels: bool = False,
        max_steps: int = 100,
        loop: int = 1,
    ) -> float:
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
        self.set_training_mode(False)
        with torch.no_grad():
            rewards = []
            for i in range(loop):
                obs = env.reset()
                score = 0
                for _ in range(max_steps):
                    if swap_channels:
                        obs = obs_channels_to_first(obs)
                    obs = torch.from_numpy(obs).float()
                    obs = obs.to(self.device)
                    action = np.argmax(self.actor(obs).cpu().numpy())
                    obs, reward = env.step(action)
                    score += reward
                rewards.append(score)
        mean_fit = np.mean(rewards)
        self.fitness.append(mean_fit)
        return mean_fit
