import warnings
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch._dynamo
import torch.nn as nn
import torch.optim as optim
from gymnasium import spaces
from tensordict import TensorDict, from_module
from tensordict.nn import CudaGraphModule

from agilerl.algorithms.core import OptimizerWrapper, RLAlgorithm
from agilerl.algorithms.core.registry import HyperparameterConfig, NetworkGroup
from agilerl.modules.base import EvolvableModule
from agilerl.networks.q_networks import QNetwork
from agilerl.typing import ExperiencesType, GymEnvType, ObservationType, TorchObsType
from agilerl.utils.algo_utils import make_safe_deepcopies, obs_channels_to_first


class DQN(RLAlgorithm):
    """Deep Q-Network (DQN) algorithm.

    Paper: https://arxiv.org/abs/1312.5602

    :param observation_space: Observation space of the environment
    :type observation_space: gymnasium.spaces.Space
    :param action_space: Action space of the environment
    :type action_space: gymnasium.spaces.Space
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

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        index: int = 0,
        hp_config: Optional[HyperparameterConfig] = None,
        net_config: Optional[Dict[str, Any]] = None,
        batch_size: int = 64,
        lr: float = 1e-4,
        learn_step: int = 5,
        gamma: float = 0.99,
        tau: float = 1e-3,
        mut: Optional[str] = None,
        double: bool = False,
        normalize_images: bool = True,
        actor_network: Optional[EvolvableModule] = None,
        device: str = "cpu",
        accelerator: Optional[Any] = None,
        cudagraphs: bool = False,
        wrap: bool = True,
    ) -> None:
        super().__init__(
            observation_space,
            action_space,
            index=index,
            hp_config=hp_config,
            device=device,
            accelerator=accelerator,
            normalize_images=normalize_images,
            name="DQN",
        )

        assert learn_step >= 1, "Learn step must be greater than or equal to one."
        assert isinstance(learn_step, int), "Learn step rate must be an integer."
        assert isinstance(batch_size, int), "Batch size must be an integer."
        assert batch_size >= 1, "Batch size must be greater than or equal to one."
        assert isinstance(lr, float), "Learning rate must be a float."
        assert lr > 0, "Learning rate must be greater than zero."
        assert isinstance(learn_step, int), "Learn step rate must be an integer."
        assert learn_step >= 1, "Learn step must be greater than or equal to one."
        assert isinstance(gamma, (float, int, torch.Tensor)), "Gamma must be a float."
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
        self.learn_step = learn_step
        self.gamma = gamma
        self.tau = tau
        self.mut = mut
        self.double = double
        self.net_config = net_config
        self.cudagraphs = cudagraphs
        self.capturable = cudagraphs

        if actor_network is not None:
            if not isinstance(actor_network, EvolvableModule):
                raise TypeError(
                    f"'actor_network' argument is of type {type(actor_network)}, but must be of type EvolvableModule."
                )

            # Need to make deepcopies for target and detached networks
            self.actor, self.actor_target = make_safe_deepcopies(
                actor_network, actor_network
            )
        else:
            net_config = {} if net_config is None else net_config

            def create_actor():
                return QNetwork(
                    observation_space=observation_space,
                    action_space=action_space,
                    device=self.device,
                    **net_config,
                )

            self.actor = create_actor()
            self.actor_target = create_actor()

        # Copy over actor weights to target
        self.init_hook()

        # Initialize optimizer with OptimizerWrapper
        self.optimizer = OptimizerWrapper(
            optim.Adam,
            networks=self.actor,
            lr=self.lr,
            optimizer_kwargs={"capturable": self.capturable},
        )

        if self.accelerator is not None and wrap:
            self.wrap_models()

        self.criterion = nn.MSELoss()

        # torch.compile and cuda graph optimizations
        if self.cudagraphs:
            warnings.warn(
                "CUDA graphs for DQN are implemented experimentally and may not work as expected."
            )
            self.update = torch.compile(self.update, mode=None)
            self._get_action = torch.compile(
                self._get_action, mode=None, fullgraph=True
            )
            self.update = CudaGraphModule(self.update)
            self._get_action = CudaGraphModule(self._get_action)

        # Register DQN network groups and mutation hook
        self.register_network_group(
            NetworkGroup(
                eval_network=self.actor,
                shared_networks=self.actor_target,
                policy=True,
            )
        )
        self.register_mutation_hook(self.init_hook)

    def init_hook(self) -> None:
        """Resets module parameters for the detached and target networks."""
        param_vals: TensorDict = from_module(self.actor).detach()

        # NOTE: This removes the target params from the computation graph which
        # reduces memory overhead and speeds up training, however these won't
        # appear in the modules parameters
        target_params: TensorDict = param_vals.clone().lock_()

        # This hook is prompted after performing architecture mutations on policy / evaluation
        # networks, which will fail since the target network is a shared network that won't be
        # reintiialized until the end. We can bypass the error safely for this reason.
        try:
            target_params.to_module(self.actor_target)
        except KeyError:
            pass
        finally:
            self.param_vals = param_vals
            self.target_params = target_params

    def get_action(
        self,
        obs: ObservationType,
        epsilon: float = 0.0,
        action_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Returns the next action to take in the environment.

        :param obs: The current observation from the environment
        :type obs: np.ndarray, dict[str, np.ndarray], tuple[np.ndarray]
        :param epsilon: Probability of taking a random action for exploration, defaults to 0
        :type epsilon: float, optional
        :param action_mask: Mask of legal actions 1=legal 0=illegal, defaults to None
        :type action_mask: numpy.ndarray, optional
        :return: Selected action(s) for the given observation(s)
        :rtype: numpy.ndarray
        """
        # Preprocess observations and convert inputs to torch tensors
        torch_obs = self.preprocess_observation(obs)
        device = self.device if self.accelerator is None else self.accelerator.device
        epsilon = torch.tensor(epsilon, device=device)
        if action_mask is not None:
            # Need to stack if vectorized env
            action_mask = (
                np.stack(action_mask)
                if action_mask.dtype == np.object_ or isinstance(action_mask, list)
                else action_mask
            )
            action_mask = torch.as_tensor(action_mask, device=device)
        else:
            if isinstance(torch_obs, dict):
                sample = next(iter(torch_obs.values()))
                batch_size = sample.size(0)
            elif isinstance(torch_obs, tuple):
                batch_size = torch_obs[0].size(0)
            else:
                batch_size = torch_obs.size(0)

            action_mask = torch.ones((batch_size, self.action_dim), device=device)

        return self._get_action(torch_obs, epsilon, action_mask).cpu().numpy()

    def _get_action(
        self, obs: TorchObsType, epsilon: torch.Tensor, action_mask: torch.Tensor
    ) -> torch.Tensor:
        """Returns the next action to take in the environment.
        Epsilon is the probability of taking a random action, used for exploration.
        For greedy behaviour, set epsilon to 0.

        :param obs: The current observation from the environment
        :type obs: torch.Tensor, dict[str, torch.Tensor], tuple[torch.Tensor]
        :param epsilon: Probability of taking a random action for exploration, defaults to 0
        :type epsilon: float, optional
        :param action_mask: Mask of legal actions 1=legal 0=illegal, defaults to None
        :type action_mask: numpy.ndarray, optional
        :return: Selected action(s) as tensor
        :rtype: torch.Tensor
        """
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
        actions = torch.where(use_policy, masked_policy_actions, masked_random_actions)

        return actions

    def update(
        self,
        obs: TorchObsType,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_obs: TorchObsType,
        dones: torch.Tensor,
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
        :return: Loss value from the update step
        :rtype: torch.Tensor
        """
        with torch.no_grad():
            if self.double:  # Double Q-learning
                q_idx = self.actor(next_obs).argmax(dim=1).unsqueeze(1)
                q_target = (
                    self.actor_target(next_obs).gather(dim=1, index=q_idx).detach()
                )
            else:
                q_target = self.actor_target(next_obs).max(axis=1)[0].unsqueeze(1)

            # target, if terminal then y_j = rewards
            y_j = rewards + self.gamma * q_target * (1 - dones)

        if actions.ndim == 1:
            actions = actions.unsqueeze(-1)

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

        :param experiences: TensorDict of batched observations, actions, rewards, next_observations, dones in that order.
        :type experiences: tensordict.TensorDict
        :return: Loss value from the learning step
        :rtype: float
        """
        obs = experiences["obs"]
        actions = experiences["action"]
        rewards = experiences["reward"]
        next_obs = experiences["next_obs"]
        dones = experiences["done"]

        obs = self.preprocess_observation(obs)
        next_obs = self.preprocess_observation(next_obs)

        loss = self.update(obs, actions, rewards, next_obs, dones)

        # soft update target network
        self.soft_update()
        return loss.item()

    def soft_update(self) -> None:
        """Soft updates target network."""
        for eval_param, target_param in zip(
            self.actor.parameters(), self.actor_target.parameters()
        ):
            target_param.data.copy_(
                self.tau * eval_param.data + (1.0 - self.tau) * target_param.data
            )

    def test(
        self,
        env: GymEnvType,
        swap_channels: bool = False,
        max_steps: Optional[int] = None,
        loop: int = 1,
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
                    if swap_channels:
                        obs = obs_channels_to_first(obs)

                    action_mask = info.get("action_mask", None)
                    action = self.get_action(obs, epsilon=0.0, action_mask=action_mask)
                    obs, reward, done, trunc, info = env.step(action)
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
