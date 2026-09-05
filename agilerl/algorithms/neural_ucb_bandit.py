# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import numpy as np
import torch
from gymnasium import spaces
from tensordict import TensorDict
from torch import nn, optim

from agilerl.algorithms.configs import (
    AlgorithmRuntime,
    BanditLearnConfig,
    BanditNetworkSetup,
    PopulationIndex,
)
from agilerl.algorithms.core import OptimizerWrapper, RLAlgorithm
from agilerl.algorithms.core.registry import (
    NetworkGroup,
    make_default_hp_config,
)
from agilerl.modules import EvolvableModule
from agilerl.networks.value_networks import ValueNetwork
from agilerl.protocols import BanditEnvProtocol
from agilerl.typing import (
    ActionMaskInput,
    BanditBatch,
    ObservationType,
    SupportedObservationSpace,
    numpy_action_mask,
)
from agilerl.utils.algo_utils import make_safe_deepcopies
from agilerl.utils.evolvable_networks import get_default_encoder_config


class NeuralUCB(RLAlgorithm[TensorDict]):
    """Neural Upper Confidence Bound (UCB).

    Paper: https://arxiv.org/abs/1911.04462

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
    :param normalize_images: Flag to normalize images, defaults to True
    :type normalize_images: bool, optional
    :param lr: Learning rate for optimizer, defaults to 1e-3
    :type lr: float, optional
    :param learn_step: Learning frequency, defaults to 2
    :type learn_step: int, optional
    :param mut: Most recent mutation to agent, defaults to None
    :type mut: str, optional
    :param actor_network: Custom actor network, defaults to None
    :type actor_network: EvolvableModule | None, optional
    :param device: Device for accelerated computing, 'cpu' or 'cuda', defaults to 'cpu'
    :type device: str, optional
    :param accelerator: Accelerator for distributed computing, defaults to None
    :type accelerator: accelerate.Accelerator(), optional
    :param wrap: Wrap models for distributed training upon creation, defaults to True
    :type wrap: bool, optional
    """

    # Bandit arms are discrete, so the network output size is a plain int
    action_dim: int

    def __init__(
        self,
        observation_space: SupportedObservationSpace,
        action_space: spaces.Space,
        member: PopulationIndex | None = None,
        learn: BanditLearnConfig | None = None,
        network: BanditNetworkSetup | None = None,
        runtime: AlgorithmRuntime | None = None,
    ) -> None:
        member = member or PopulationIndex()
        learn = learn or BanditLearnConfig()
        network = network or BanditNetworkSetup()
        runtime = runtime or AlgorithmRuntime()
        index = member.index
        hp_config = member.hp_config
        mut = member.mut
        gamma = learn.gamma
        lamb = learn.lamb
        reg = learn.reg
        batch_size = learn.batch_size
        lr = learn.lr
        learn_step = learn.learn_step
        net_config = network.net_config
        actor_network = network.actor_network
        normalize_images = network.normalize_images
        device = runtime.device
        accelerator = runtime.accelerator
        wrap = runtime.wrap

        super().__init__(
            observation_space,
            action_space,
            index=index,
            hp_config=hp_config,
            device=device,
            accelerator=accelerator,
            normalize_images=normalize_images,
            name=runtime.name or "NeuralUCB",
        )

        assert learn_step >= 1, "Learn step must be greater than or equal to one."
        assert isinstance(learn_step, int), "Learn step rate must be an integer."
        assert isinstance(
            gamma,
            (float, int),
        ), "Scaling factor must be a float or integer."
        assert gamma > 0, "Scaling factor must be positive."
        assert isinstance(
            lamb,
            (float, int),
        ), "Regularization parameter lambda must be a float or integer."
        assert lamb > 0, "Regularization parameter lambda must be greater than zero."
        assert isinstance(reg, float), "Loss regularization parameter must be a float."
        assert reg > 0, "Loss regularization parameter must be greater than zero."
        assert isinstance(batch_size, int), "Batch size must be an integer."
        assert batch_size >= 1, "Batch size must be greater than or equal to one."
        assert isinstance(lr, float), "Learning rate must be a float."
        assert lr > 0, "Learning rate must be greater than zero."
        assert isinstance(
            wrap,
            bool,
        ), "Wrap models flag must be boolean value True or False."

        self.gamma = gamma
        self.lamb = lamb
        self.reg = reg
        self.batch_size = batch_size
        self.learn_step = learn_step
        self.lr = lr
        self.net_config = net_config
        self.mut = mut
        self.regret: list[float] = [0.0]
        self.actor_network = None

        # Default RL hyperparameters to mutate when doing Evo-HPO
        self.hp_config = self.hp_config or make_default_hp_config(
            lr=self.lr,
            batch_size=self.batch_size,
            learn_step=self.learn_step,
        )

        if actor_network is not None:
            if not isinstance(actor_network, EvolvableModule):
                msg = f"'actor_network' argument is of type {type(actor_network)}, but must be of type EvolvableModule."
                raise TypeError(
                    msg,
                )

            # Need to make deepcopies for target and detached networks.
            self.actor = make_safe_deepcopies(actor_network)
        else:
            net_config = {} if net_config is None else net_config
            simba = net_config.get("simba", False)
            encoder_config = (
                get_default_encoder_config(self.observation_space, simba)
                if net_config.get("encoder_config") is None
                else net_config["encoder_config"]
            )

            if not simba and not isinstance(
                self.observation_space,
                (spaces.Dict, spaces.Tuple),
            ):
                # Layer norm is not used in the original implementation
                encoder_config["layer_norm"] = False

            net_config["encoder_config"] = encoder_config

            self.actor = ValueNetwork(
                observation_space=self.observation_space,
                device=self.device,
                **net_config,
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
            NetworkGroup(eval_network=self.actor, shared_networks=None, policy=True),
        )

        # Register metrics to keep track of during training
        self.metrics.register("loss")

    def init_params(self) -> None:
        """Initialize the parameters of the network."""
        exp_layer = self.actor.get_output_dense()
        # EvolvableMLP/MakeEvolvable networks build a final nn.Linear output layer
        assert isinstance(exp_layer, nn.Linear), (
            "Bandit actor network must expose an nn.Linear output dense layer."
        )
        self.exp_layer: nn.Linear = exp_layer

        self.numel = sum(
            w.numel() for w in self.exp_layer.parameters() if w.requires_grad
        )
        self.sigma_inv = self.lamb * torch.eye(self.numel).to(self.device)
        self.theta_0 = torch.cat(
            [w.flatten() for w in self.exp_layer.parameters() if w.requires_grad],
        )

    def get_action(
        self,
        obs: ObservationType,
        action_mask: ActionMaskInput = None,
        *args: Any,
        **kwargs: Any,
    ) -> int:
        """Return the next action to take in the environment.

        :param obs: State observation, or multiple observations in a batch
        :type obs: numpy.ndarray[float]
        :param action_mask: Mask of legal actions 1=legal 0=illegal, defaults to None
        :type action_mask: ActionMaskInput

        :return: Action to take in the environment
        :rtype: int
        """
        obs = self.preprocess_observation(obs)

        mu_raw = self.actor(obs).reshape(-1)
        mu = (
            mu_raw.repeat(self.action_dim)
            if (mu_raw.numel() == 1 and self.action_dim > 1)
            else mu_raw
        )
        g: torch.Tensor = torch.zeros((self.action_dim, self.numel)).to(self.device)
        if mu_raw.numel() == 1 and self.action_dim > 1:
            self.optimizer.zero_grad()
            mu_raw[0].backward(retain_graph=True)
            grad_vec = torch.cat(
                [
                    w.grad.detach().flatten()
                    for w in self.exp_layer.parameters()
                    if w.requires_grad and w.grad is not None
                ],
            ) / np.sqrt(self.exp_layer.weight.size(0))
            g[:] = grad_vec
        else:
            for k, fx in enumerate(mu):
                self.optimizer.zero_grad()
                fx.backward(retain_graph=True)
                g[k] = torch.cat(
                    [
                        w.grad.detach().flatten()
                        for w in self.exp_layer.parameters()
                        if w.requires_grad and w.grad is not None
                    ],
                ) / np.sqrt(self.exp_layer.weight.size(0))

        with torch.no_grad():
            action_values = mu + self.gamma * torch.sqrt(
                torch.matmul(
                    torch.matmul(g[:, None, :], self.sigma_inv),
                    g[:, :, None],
                )[:, 0, :].squeeze(-1),
            )

        action_values = action_values.cpu().numpy()
        if action_mask is None:
            action = np.argmax(action_values)
        else:
            inv_mask = 1 - numpy_action_mask(action_mask)
            masked_action_values = np.ma.array(action_values, mask=inv_mask)
            action = np.argmax(masked_action_values)

        # Sherman-Morrison-Woodbury Update
        v = g[action].unsqueeze(-1)
        self.sigma_inv -= (self.sigma_inv @ v @ v.T @ self.sigma_inv) / (
            1 + v.T @ self.sigma_inv @ v
        )

        return int(action)

    def _greedy_test_action(self, obs: ObservationType) -> int:
        """Greedy arm for evaluation: preprocess obs, no UCB bonus or posterior update."""
        with torch.no_grad():
            obs_tensor = self.preprocess_observation(obs)
            mu_raw = self.actor(obs_tensor).reshape(-1)
            if mu_raw.numel() == 1 and self.action_dim > 1:
                mu_raw = mu_raw.repeat(self.action_dim)
            return int(np.argmax(mu_raw.cpu().numpy()))

    def learn(self, experiences: TensorDict) -> float:
        """Update agent network parameters to learn from experiences.

        :param experiences: Batch of contexts (``obs``) and rewards sampled from
            the bandit replay buffer.
        :type experiences: TensorDict

        :return: Loss value from training step
        :rtype: float
        """
        batch: BanditBatch = BanditBatch.from_tensordict(experiences)
        states = batch.obs
        rewards = batch.reward

        pred_rewards = self.actor(states)

        # loss backprop
        loss = self.criterion(pred_rewards, rewards)
        loss += (
            self.reg
            * torch.norm(
                torch.cat(
                    [
                        w.flatten()
                        for w in self.exp_layer.parameters()
                        if w.requires_grad
                    ],
                )
                - self.theta_0,
            )
            ** 2
        )
        self.optimizer.zero_grad()
        if self.accelerator is not None:
            self.accelerator.backward(loss)
        else:
            loss.backward()

        self.optimizer.step()

        loss = loss.item()
        self.metrics.log("loss", loss)
        return loss

    def test(
        self,
        env: BanditEnvProtocol,
        max_steps: int = 100,
        loop: int = 1,
    ) -> float:
        """Return mean greedy test score in the environment.

        Uses :meth:`preprocess_observation` and a greedy forward pass only —
        unlike :meth:`get_action`, this does not apply the UCB bonus or update
        ``sigma_inv``.

        :param env: The bandit environment to be tested in
        :type env: BanditEnvProtocol
        :param max_steps: Maximum number of testing steps, defaults to 500
        :type max_steps: int, optional
        :param loop: Number of testing loops/episodes to complete. The returned score is the mean over these tests. Defaults to 3
        :type loop: int, optional

        :return: Mean test score of agent in environment
        :rtype: float
        """
        self.set_training_mode(False)
        with torch.no_grad():
            rewards = []
            for _i in range(loop):
                obs = env.reset()
                score = 0
                for _ in range(max_steps):
                    action = self._greedy_test_action(obs)
                    obs, reward = env.step(action)
                    score += reward
                rewards.append(score)
        mean_fit = float(np.mean(rewards))
        self.metrics.add_fitness(mean_fit)
        return mean_fit
