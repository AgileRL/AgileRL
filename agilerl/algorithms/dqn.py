# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

import warnings
from collections.abc import Callable
from typing import Any

import gymnasium as gym
import numpy as np
import numpy.typing as npt
import torch
from gymnasium import spaces
from tensordict import TensorDict
from tensordict.nn import CudaGraphModule
from torch import nn, optim

from agilerl.algorithms.configs import (
    AlgorithmRuntime,
    OffPolicyLearnConfig,
    PopulationIndex,
    QNetworkSetup,
)
from agilerl.algorithms.core import OptimizerWrapper, RLAlgorithm
from agilerl.algorithms.core.registry import (
    NetworkGroup,
    make_default_hp_config,
)
from agilerl.modules.base import EvolvableModule
from agilerl.networks.q_networks import QNetwork
from agilerl.typing import (
    ActionMaskInput,
    ObservationType,
    ReplayBatch,
    SupportedObservationSpace,
    TorchObsType,
    numpy_action_mask,
)
from agilerl.utils.algo_utils import (
    adam_kwargs,
    eval_mode,
    is_train_eval_invariant,
    make_safe_deepcopies,
    polyak_update,
)


class DQN(RLAlgorithm[TensorDict]):
    """Deep Q-Network (DQN).

    Paper: https://arxiv.org/abs/1312.5602

    :param observation_space: Observation space of the environment
    :type observation_space: SupportedObservationSpace
    :param action_space: Action space of the environment
    :type action_space: gymnasium.spaces.Discrete
    :param index: Index to keep track of object instance during tournament selection and mutation, defaults to 0
    :type index: int, optional
    :param hp_config: RL hyperparameter mutation configuration, defaults to None, whereby algorithm mutations are disabled.
    :type hp_config: HyperparameterConfig, optional
    :param net_config: Network configuration, defaults to None
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
    :param normalize_images: Flag to normalize images, defaults to True
    :type normalize_images: bool, optional
    :param actor_network: Custom actor network, defaults to None
    :type actor_network: nn.Module, optional
    :param device: Device for accelerated computing, 'cpu' or 'cuda', defaults to 'cpu'
    :type device: str, optional
    :param accelerator: Accelerator for distributed computing, defaults to None
    :type accelerator: accelerate.Accelerator(), optional
    :param cudagraphs: Use CUDA graphs for optimization, defaults to False
    :type cudagraphs: bool, optional
    :param wrap: Wrap models for distributed training upon creation, defaults to True
    :type wrap: bool, optional
    """

    # Narrowed from RLAlgorithm.action_space; enforced at construction.
    action_space: spaces.Discrete | spaces.MultiDiscrete

    # Discrete action space, so the network output size is a plain int
    action_dim: int

    # Hot-path callables: bound methods by default, CudaGraph wrappers when enabled.
    _get_action_impl: Callable[
        [TorchObsType, torch.Tensor | float, torch.Tensor],
        torch.Tensor,
    ]
    _update_impl: Callable[
        [TorchObsType, torch.Tensor, torch.Tensor, TorchObsType, torch.Tensor],
        torch.Tensor,
    ]

    def __init__(
        self,
        observation_space: SupportedObservationSpace,
        action_space: spaces.Discrete,
        member: PopulationIndex | None = None,
        learn: OffPolicyLearnConfig | None = None,
        network: QNetworkSetup | None = None,
        runtime: AlgorithmRuntime | None = None,
    ) -> None:
        member = member or PopulationIndex()
        learn = learn or OffPolicyLearnConfig()
        network = network or QNetworkSetup()
        runtime = runtime or AlgorithmRuntime()
        index = member.index
        hp_config = member.hp_config
        mut = member.mut
        batch_size = learn.batch_size
        lr = learn.lr
        learn_step = learn.learn_step
        gamma = learn.gamma
        tau = learn.tau
        net_config = network.net_config
        actor_network = network.actor_network
        double = network.double
        normalize_images = network.normalize_images
        cudagraphs = network.cudagraphs
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
            name=runtime.name or "DQN",
        )

        assert learn_step >= 1, "Learn step must be greater than or equal to one."
        assert isinstance(learn_step, int), "Learn step rate must be an integer."
        assert isinstance(batch_size, int), "Batch size must be an integer."
        assert batch_size >= 1, "Batch size must be greater than or equal to one."
        assert isinstance(lr, float), "Learning rate must be a float."
        assert lr > 0, "Learning rate must be greater than zero."
        assert isinstance(gamma, (float, int, torch.Tensor)), "Gamma must be a float."
        assert isinstance(tau, float), "Tau must be a float."
        assert tau > 0, "Tau must be greater than zero."
        assert isinstance(
            double,
            bool,
        ), "Double Q-learning flag must be boolean value True or False."
        assert isinstance(
            wrap,
            bool,
        ), "Wrap models flag must be boolean value True or False."

        self.batch_size = batch_size
        self.lr = lr
        self.learn_step = learn_step
        self.gamma = gamma
        self.tau = tau
        self.mut = mut
        self.double = double
        self.net_config = net_config
        self.cudagraphs = cudagraphs
        self.capturable = cudagraphs
        self.wrap = wrap

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
            self.actor, self.actor_target = make_safe_deepcopies(
                actor_network,
                actor_network,
            )
        else:
            net_config = {} if net_config is None else net_config

            def create_actor() -> QNetwork:
                return QNetwork(
                    observation_space=self.observation_space,
                    action_space=self.action_space,
                    device=self.device,
                    **net_config,
                )

            self.actor = create_actor()
            self.actor_target = create_actor()

        # Initialize target network (same pattern as DDPG; post-mutation sync via reinit_shared_networks)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self._actor_mode_invariant = is_train_eval_invariant(self.actor)

        # Initialize optimizer with OptimizerWrapper
        self.optimizer = OptimizerWrapper(
            optim.Adam,
            networks=self.actor,
            lr=self.lr,
            optimizer_kwargs=adam_kwargs(
                self.device, self.accelerator, capturable=self.capturable
            ),
        )

        if self.accelerator is not None and wrap:
            self.wrap_models()

        self.criterion = nn.MSELoss()

        # Hot-path dispatch: methods by default; CudaGraph wrappers when enabled.
        self._update_impl = self.update
        self._get_action_impl = self._get_action
        if self.cudagraphs:
            warnings.warn(
                "CUDA graphs for DQN are implemented experimentally and may not work as expected.",
                stacklevel=2,
            )
            compiled_update: Any = torch.compile(self.update, mode=None)
            compiled_get_action: Any = torch.compile(
                self._get_action,
                mode=None,
                fullgraph=True,
            )
            self._update_impl = CudaGraphModule(compiled_update)
            self._get_action_impl = CudaGraphModule(compiled_get_action)

        # Register DQN network groups
        self.register_network_group(
            NetworkGroup(
                eval_network=self.actor,
                shared_networks=self.actor_target,
                policy=True,
            ),
        )

        # Register metrics to keep track of during training
        self.metrics.register("loss")
        self.metrics.register_histogram("action_dist")

    def get_action(
        self,
        obs: ObservationType,
        epsilon: float = 0.0,
        action_mask: ActionMaskInput = None,
        *args: Any,
        **kwargs: Any,
    ) -> npt.NDArray:
        """Return the next action to take in the environment.

        :param obs: The current observation from the environment
        :type obs: npt.NDArray, dict[str, npt.NDArray], tuple[npt.NDArray]
        :param epsilon: Probability of taking a random action for exploration, defaults to 0
        :type epsilon: float, optional
        :param action_mask: Mask of legal actions 1=legal 0=illegal, defaults to None
        :type action_mask: ActionMaskInput
        :return: Selected action(s) for the given observation(s)
        :rtype: numpy.ndarray
        """
        # Preprocess observations and convert inputs to torch tensors
        torch_obs = self.preprocess_observation(obs)
        # Graph capture needs epsilon as a device tensor; eager compares the float.
        eps = torch.tensor(epsilon, device=self.device) if self.cudagraphs else epsilon
        if action_mask is not None:
            mask = torch.as_tensor(
                numpy_action_mask(action_mask),
                device=self.device,
            )
        else:
            if isinstance(torch_obs, torch.Tensor):
                batch_size = torch_obs.size(0)
            elif isinstance(torch_obs, TensorDict):
                batch_size = torch_obs.batch_size[0]
            elif isinstance(torch_obs, dict):
                sample = next(iter(torch_obs.values()))
                batch_size = sample.size(0)
            else:
                batch_size = torch_obs[0].size(0)

            mask = torch.ones((batch_size, self.action_dim), device=self.device)

        action = self._get_action_impl(torch_obs, eps, mask).cpu().numpy()

        if self.training:
            self.metrics.log_histogram("action_dist", action)

        return action

    def _get_action(
        self,
        obs: TorchObsType,
        epsilon: torch.Tensor | float,
        action_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return the next action to take in the environment.
        Epsilon is the probability of taking a random action, used for exploration.
        For greedy behaviour, set epsilon to 0.

        :param obs: The current observation from the environment
        :type obs: torch.Tensor, dict[str, torch.Tensor], tuple[torch.Tensor]
        :param epsilon: Probability of taking a random action for exploration, defaults to 0
        :type epsilon: float, optional
        :param action_mask: Mask of legal actions 1=legal 0=illegal
        :type action_mask: torch.Tensor
        :return: Selected action(s) as tensor
        :rtype: torch.Tensor
        """
        with eval_mode(self.actor, mode_invariant=self._actor_mode_invariant):
            with torch.no_grad():
                q_values = self.actor(obs)

        # Masked random actions
        masked_random_values = torch.rand_like(q_values) * action_mask
        masked_random_actions = torch.argmax(masked_random_values, dim=-1)

        # Masked policy actions
        masked_q_values = q_values.masked_fill((1 - action_mask).bool(), float("-inf"))
        masked_policy_actions = torch.argmax(masked_q_values, dim=-1)

        # actions_random = torch.randint_like(actions, n_act)
        use_policy = (
            torch.empty(masked_policy_actions.shape, device=q_values.device)
            .uniform_()
            .gt(epsilon)
        )

        # Recompute actions with masking
        return torch.where(use_policy, masked_policy_actions, masked_random_actions)

    def update(
        self,
        obs: TorchObsType,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_obs: TorchObsType,
        dones: torch.Tensor,
    ) -> torch.Tensor:
        """Update agent network parameters to learn from experiences.

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
        :return: Loss value from the update step
        :rtype: torch.Tensor
        """
        if actions.ndim == 1:
            actions = actions.unsqueeze(-1)
        actions = actions.long()

        # One online forward over cat(obs, next_obs); the next_obs half is detached
        # for greedy action selection, so gradients match two separate forwards.
        if (
            self.double
            and self._actor_mode_invariant
            and isinstance(obs, torch.Tensor)
            and isinstance(next_obs, torch.Tensor)
        ):
            batch_size = obs.shape[0]
            q_online = self.actor(torch.cat((obs, next_obs), dim=0))
            q_eval = q_online[:batch_size].gather(1, actions)
            with torch.no_grad():
                q_idx = q_online[batch_size:].detach().argmax(dim=1, keepdim=True)
                q_target = self.actor_target(next_obs).gather(dim=1, index=q_idx)
                y_j = rewards + self.gamma * q_target * (1 - dones)
        else:
            with torch.no_grad():
                if self.double:  # Double Q-learning
                    q_idx = self.actor(next_obs).argmax(dim=1).unsqueeze(1)
                    q_target = (
                        self.actor_target(next_obs).gather(dim=1, index=q_idx).detach()
                    )
                else:
                    q_target = self.actor_target(next_obs).max(dim=1)[0].unsqueeze(1)

                # target, if terminal then y_j = rewards
                y_j = rewards + self.gamma * q_target * (1 - dones)

            # Compute Q-values for actions taken
            q_eval = self.actor(obs).gather(1, actions)

        loss: torch.Tensor = self.criterion(q_eval, y_j)

        # zero gradients, perform a backward pass, and update the weights
        self.optimizer.zero_grad()
        if self.accelerator is not None:
            self.accelerator.backward(loss)
        else:
            loss.backward()

        self.optimizer.step()
        return loss.detach()

    def learn(self, experiences: TensorDict) -> float:
        """Update agent network parameters to learn from experiences.

        :param experiences: Batch of observations, actions, rewards, next
            observations and dones sampled from an off-policy replay buffer.
        :type experiences: TensorDict
        :return: Loss value from the learning step
        :rtype: float
        """
        batch: ReplayBatch = ReplayBatch.from_tensordict(experiences)
        actions = batch.action
        rewards = batch.reward
        dones = batch.done

        obs = self.preprocess_observation(batch.obs)
        next_obs = self.preprocess_observation(batch.next_obs)

        loss = self._update_impl(obs, actions, rewards, next_obs, dones)

        # soft update target network
        self.soft_update()

        loss_value = loss.item()
        self.metrics.log("loss", loss_value)
        return loss_value

    def soft_update(self) -> None:
        """Soft updates target network."""
        polyak_update(self.actor, self.actor_target, self.tau)

    def test(
        self,
        env: gym.vector.VectorEnv,
        max_steps: int | None = None,
        loop: int = 1,
    ) -> float:
        """Return mean test score of agent in environment with epsilon-greedy policy.

        :param env: The vectorized environment to be tested in
        :type env: gym.vector.VectorEnv
        :param max_steps: Maximum number of testing steps, defaults to None
        :type max_steps: int, optional
        :param loop: Number of testing loops/episodes to complete. The returned score is the mean over these tests. Defaults to 1
        :type loop: int, optional
        :return: Mean test score of agent in environment
        :rtype: float
        """
        self.set_training_mode(False)
        with torch.no_grad():
            rewards = []
            num_envs = env.num_envs if hasattr(env, "num_envs") else 1
            for _ in range(loop):
                obs, info = env.reset()
                scores = np.zeros(num_envs)
                completed_episode_scores = np.zeros(num_envs)
                finished = np.zeros(num_envs)
                step = 0
                while not np.all(finished):
                    action_mask = info.get("action_mask", None)
                    action = self.get_action(obs, epsilon=0.0, action_mask=action_mask)
                    obs, reward, done, trunc, info = env.step(action)
                    step += 1
                    scores += np.array(reward)
                    for idx, (d, t) in enumerate(zip(done, trunc, strict=False)):
                        if (
                            d or t or (max_steps is not None and step == max_steps)
                        ) and not finished[idx]:
                            completed_episode_scores[idx] = scores[idx]
                            finished[idx] = 1
                rewards.append(np.mean(completed_episode_scores))
        mean_fit = float(np.mean(rewards))
        self.metrics.add_fitness(mean_fit)
        return mean_fit
