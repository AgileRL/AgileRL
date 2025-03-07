import random
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from gymnasium import spaces
from numpy.typing import ArrayLike
from torch.nn.utils import clip_grad_norm_

from agilerl.algorithms.core import RLAlgorithm
from agilerl.algorithms.core.registry import HyperparameterConfig, NetworkGroup
from agilerl.algorithms.core.wrappers import OptimizerWrapper
from agilerl.modules.base import EvolvableModule
from agilerl.networks.q_networks import QNetwork
from agilerl.typing import GymEnvType, ObservationType
from agilerl.utils.algo_utils import make_safe_deepcopies, obs_channels_to_first


class CQN(RLAlgorithm):
    """The CQN algorithm class. CQN paper: https://arxiv.org/abs/2006.04779

    :param observation_space: The observation space of the environment.
    :type observation_space: spaces.Space
    :param action_space: The action space of the environment.
    :type action_space: spaces.Space
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
    :param double: Use double Q-learning, defaults to False
    :type double: bool, optional
    :param normalize_images: Normalize image observations, defaults to True
    :type normalize_images: bool, optional
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
        double: bool = False,
        normalize_images: bool = True,
        mut: Optional[str] = None,
        actor_network: Optional[EvolvableModule] = None,
        device: str = "cpu",
        accelerator: Optional[Any] = None,
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
            name="CQN",
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
        self.gamma = gamma
        self.learn_step = learn_step
        self.tau = tau
        self.mut = mut
        self.double = double
        self.net_config = net_config

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

        self.actor_target.load_state_dict(self.actor.state_dict())

        # Initialize optimizer
        self.optimizer = OptimizerWrapper(
            optim.Adam,
            networks=self.actor,
            lr=self.lr,
        )

        if self.accelerator is not None and wrap:
            self.wrap_models()

        self.criterion = nn.MSELoss()

        # Register policy for mutations
        self.register_network_group(
            NetworkGroup(eval=self.actor, shared=self.actor_target, policy=True)
        )

    def get_action(
        self,
        obs: ObservationType,
        epsilon: float = 0,
        action_mask: Optional[ArrayLike] = None,
    ) -> ArrayLike:
        """Returns the next action to take in the environment. Epsilon is the
        probability of taking a random action, used for exploration.
        For greedy behaviour, set epsilon to 0.

        :param obs: State observation, or multiple observations in a batch
        :type obs: numpy.ndarray[float]
        :param epsilon: Probablilty of taking a random action for exploration, defaults to 0
        :type epsilon: float, optional
        :param action_mask: Mask of legal actions 1=legal 0=illegal, defaults to None
        :type action_mask: numpy.ndarray, optional

        :return: Action to take in the environment
        :rtype: numpy.ndarray[int]
        """
        obs = self.preprocess_observation(obs)

        # epsilon-greedy
        if random.random() < epsilon:
            if action_mask is None:
                action = np.random.randint(0, self.action_dim, size=len(obs))
            else:
                action = np.argmax(
                    (
                        np.random.uniform(0, 1, (len(obs), self.action_dim))
                        * action_mask
                    ),
                    axis=1,
                )

        else:
            self.actor.eval()
            with torch.no_grad():
                action_values = self.actor(obs).cpu().data.numpy()
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
        :type obs: list[torch.Tensor[float]]

        :return: Loss from learning
        :rtype: float
        """
        states, actions, rewards, next_states, dones = experiences
        if self.accelerator is not None:
            actions = actions.to(self.accelerator.device)
            rewards = rewards.to(self.accelerator.device)
            dones = dones.to(self.accelerator.device)

        states = self.preprocess_observation(states)
        next_states = self.preprocess_observation(next_states)

        if self.double:  # Double Q-learning
            q_idx = self.actor(next_states).argmax(dim=1).unsqueeze(1)
            q_target_next = (
                self.actor_target(next_states).gather(dim=1, index=q_idx).detach()
            )
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

    def test(
        self,
        env: GymEnvType,
        swap_channels: bool = False,
        max_steps: Optional[int] = None,
        loop: int = 3,
    ):
        """Returns mean test score of agent in environment with epsilon-greedy policy.

        :param env: The environment to be tested in
        :type env: Gym-style environment
        :param swap_channels: Swap image channels dimension from last to first [H, W, C] -> [C, H, W], defaults to False
        :type swap_channels: bool, optional
        :param max_steps: Maximum number of testing steps, defaults to None.
        :type max_steps: int, optional
        :param loop: Number of testing loops/episodes to complete. The returned score is the mean. Defaults to 3
        :type loop: int, optional
        """
        self.set_training_mode(False)

        with torch.no_grad():
            rewards = []
            num_envs = env.num_envs if hasattr(env, "num_envs") else 1
            for i in range(loop):
                obs, info = env.reset()
                scores = np.zeros(num_envs)
                completed_episode_scores = np.zeros(num_envs)
                finished = np.zeros(num_envs)
                step = 0
                while not np.all(finished):
                    if swap_channels:
                        obs = obs_channels_to_first(obs)

                    action_mask = info.get("action_mask", None)
                    action = self.get_action(obs, epsilon=0, action_mask=action_mask)
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
