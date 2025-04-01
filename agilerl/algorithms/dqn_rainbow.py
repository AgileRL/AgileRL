from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.optim as optim
from gymnasium import spaces
from torch.nn.utils import clip_grad_norm_

from agilerl.algorithms.core import RLAlgorithm
from agilerl.algorithms.core.registry import HyperparameterConfig, NetworkGroup
from agilerl.algorithms.core.wrappers import OptimizerWrapper
from agilerl.modules.base import EvolvableModule
from agilerl.modules.configs import MlpNetConfig
from agilerl.networks.q_networks import RainbowQNetwork
from agilerl.typing import (
    ArrayLike,
    ExperiencesType,
    GymEnvType,
    ObservationType,
    TorchObsType,
)
from agilerl.utils.algo_utils import make_safe_deepcopies, obs_channels_to_first
from agilerl.wrappers.make_evolvable import MakeEvolvable


class RainbowDQN(RLAlgorithm):
    """Rainbow Deep Q-Network (DQN) algorithm.

    Paper: https://arxiv.org/abs/1710.02298

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
    :param noise_std: Noise standard deviation, defaults to 0.5
    :type noise_std: float, optional
    :param n_step: Step number to calculate n-step td error, defaults to 3
    :type n_step: int, optional
    :param mut: Most recent mutation to agent, defaults to None
    :type mut: str, optional
    :param combined_reward: Boolean flag indicating whether to use combined 1-step and n-step reward, defaults to False
    :type combined_reward: bool, optional
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
        beta: float = 0.4,
        prior_eps: float = 1e-6,
        num_atoms: int = 51,
        v_min: float = 0,
        v_max: float = 200,
        noise_std: float = 0.5,
        n_step: int = 3,
        mut: Optional[str] = None,
        normalize_images: bool = True,
        combined_reward: bool = False,
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
            name="Rainbow DQN",
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

        self.batch_size = batch_size
        self.learn_step = learn_step
        self.lr = lr
        self.gamma = gamma
        self.tau = tau
        self.beta = beta
        self.prior_eps = prior_eps
        self.num_atoms = num_atoms
        self.net_config = net_config
        self.v_min = v_min
        self.v_max = v_max
        self.n_step = n_step
        self.mut = mut
        self.combined_reward = combined_reward
        self.noise_std = noise_std

        self.support = torch.linspace(
            self.v_min, self.v_max, self.num_atoms, device=self.device
        )
        self.delta_z = (self.v_max - self.v_min) / (self.num_atoms - 1)

        if actor_network is not None:
            if isinstance(actor_network, MakeEvolvable):
                actor_network.rainbow = True
                actor_network = actor_network
                actor_network.support = self.support
                actor_network.num_atoms = self.num_atoms
                actor_network = MakeEvolvable(**actor_network.init_dict)
                actor_network.load_state_dict(actor_network.state_dict())
            elif not isinstance(actor_network, EvolvableModule):
                raise TypeError(
                    f"'actor_network' argument is of type {type(actor_network)}, but must be of type EvolvableModule."
                )

            self.actor, self.actor_target = make_safe_deepcopies(
                actor_network, actor_network
            )
        else:
            net_config = {} if net_config is None else net_config
            head_config: Optional[Dict[str, Any]] = net_config.get("head_config", None)

            head_config = MlpNetConfig(
                hidden_size=(
                    [64]
                    if head_config is None
                    else head_config.get("hidden_size", [64])
                ),
                noise_std=self.noise_std,
                output_activation="ReLU",
            )
            net_config["head_config"] = head_config

            def create_actor():
                return RainbowQNetwork(
                    observation_space=observation_space,
                    action_space=action_space,
                    support=self.support,
                    num_atoms=self.num_atoms,
                    noise_std=self.noise_std,
                    device=self.device,
                    **net_config,
                )

            self.actor = create_actor()
            self.actor_target = create_actor()

        # Create the target network by copying the actor network
        self.actor_target.load_state_dict(self.actor.state_dict())

        # Optimizer
        self.optimizer = OptimizerWrapper(optim.Adam, networks=self.actor, lr=self.lr)

        if self.accelerator is not None and wrap:
            self.wrap_models()

        # Put the nets into training mode
        self.actor.train()
        self.actor_target.train()

        # Register network groups for mutations
        self.register_network_group(
            NetworkGroup(eval=self.actor, shared=self.actor_target, policy=True)
        )

    def get_action(
        self,
        obs: ObservationType,
        action_mask: Optional[np.ndarray] = None,
        training: bool = True,
    ) -> np.ndarray:
        """Returns the next action to take in the environment.

        :param obs: State observation, or multiple observations in a batch
        :type obs: numpy.ndarray[float]
        :param action_mask: Mask of legal actions 1=legal 0=illegal, defaults to None
        :type action_mask: numpy.ndarray, optional
        :param training: Flag indicating whether the model is in training mode, defaults to True
        :type training: bool, optional
        :return: The action to take
        :rtype: numpy.ndarray
        """
        obs = self.preprocess_observation(obs)

        self.actor.train(mode=training)
        with torch.no_grad():
            action_values = self.actor(obs)

        if action_mask is None:
            action = np.argmax(action_values.cpu().data.numpy(), axis=-1)
        else:
            # Need to stack if vectorized env
            action_mask = (
                np.stack(action_mask)
                if action_mask.dtype == np.object_ or isinstance(action_mask, list)
                else action_mask
            )
            inv_mask = 1 - action_mask
            masked_action_values = np.ma.array(
                action_values.cpu().data.numpy(), mask=inv_mask
            )
            action = np.argmax(masked_action_values, axis=-1)

        self.actor.train()

        return action

    def _dqn_loss(
        self,
        states: TorchObsType,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
        gamma: float,
    ) -> torch.Tensor:
        """Calculates the DQN loss.

        :param states: Batch of current states
        :type states: torch.Tensor
        :param actions: Batch of actions taken
        :type actions: torch.Tensor
        :param rewards: Batch of rewards received
        :type rewards: torch.Tensor
        :param next_states: Batch of next states
        :type next_states: torch.Tensor
        :param dones: Batch of done flags indicating episode termination
        :type dones: torch.Tensor
        :param gamma: Discount factor
        :type gamma: float
        :return: Element-wise loss
        :rtype: torch.Tensor
        """
        states = self.preprocess_observation(states)
        next_states = self.preprocess_observation(next_states)

        with torch.no_grad():

            # Predict next actions from next_states
            next_actions = self.actor(next_states).argmax(1)

            # Predict the target q distribution for the same next states
            target_q_dist = self.actor_target(next_states, q=False)

            # Index the target q_dist to select the distributions corresponding to next_actions
            target_q_dist = target_q_dist[range(self.batch_size), next_actions]

            # Determine the target z values
            t_z = rewards + (1 - dones) * gamma * self.support
            t_z = t_z.clamp(min=self.v_min, max=self.v_max)

            # Finds closest support element index value
            b = (t_z - self.v_min) / self.delta_z

            # Find the neighbouring indices of b
            L = b.floor().long()
            u = b.ceil().long()

            # Shape of projected q distribution is (batch_size, num_atoms) as we have argmaxed over actions
            # Fix disappearing probability mass
            L[(u > 0) * (L == u)] -= 1
            u[(L < (self.num_atoms - 1)) * (L == u)] += 1
            offset = (
                torch.linspace(
                    0,
                    (self.batch_size - 1) * self.num_atoms,
                    self.batch_size,
                    device=self.device,
                )
                .long()
                .unsqueeze(1)
                .expand(self.batch_size, self.num_atoms)
            )
            proj_dist = torch.zeros(target_q_dist.size(), device=self.device)

            proj_dist.view(-1).index_add_(
                0, (L + offset).view(-1), (target_q_dist * (u.float() - b)).view(-1)
            )
            proj_dist.view(-1).index_add_(
                0, (u + offset).view(-1), (target_q_dist * (b - L.float())).view(-1)
            )

        # Calculate the current obs
        log_q_dist = self.actor(states, q=False, log=True)
        log_p = log_q_dist[range(self.batch_size), actions.squeeze().long()]

        # loss
        elementwise_loss = -(proj_dist * log_p).sum(1)
        return elementwise_loss

    def learn(
        self,
        experiences: ExperiencesType,
        n_experiences: Optional[ExperiencesType] = None,
        per: bool = False,
    ) -> Tuple[float, Optional[ArrayLike], Optional[ArrayLike]]:
        """Updates agent network parameters to learn from experiences.

        :param experiences: List of batched states, actions, rewards, next_states, dones in that order.
        :type experiences: TensorDict
        :param n_experiences: List of batched states, actions, rewards, next_states, dones in that order.
        :type n_experiences: TensorDict, optional
        :param per: Use prioritized experience replay buffer, defaults to True
        :type per: bool, optional

        :return: Tuple of loss, indices, and new priorities
        :rtype: Tuple[float, numpy.ndarray, numpy.ndarray]
        """
        n_step = n_experiences is not None
        states = experiences["obs"]
        actions = experiences["action"]
        rewards = experiences["reward"]
        next_states = experiences["next_obs"]
        dones = experiences["done"]
        if per:
            weights = experiences["weights"]
            idxs = experiences["idxs"]
            if n_step:
                n_states = n_experiences["obs"]
                n_actions = n_experiences["action"]
                n_rewards = n_experiences["reward"]
                n_next_states = n_experiences["next_obs"]
                n_dones = n_experiences["done"]

            if self.combined_reward or not n_step:
                elementwise_loss = self._dqn_loss(
                    states, actions, rewards, next_states, dones, self.gamma
                )
            if n_step:
                n_gamma = self.gamma**self.n_step
                n_step_elementwise_loss = self._dqn_loss(
                    n_states, n_actions, n_rewards, n_next_states, n_dones, n_gamma
                )
                if self.combined_reward:
                    elementwise_loss += n_step_elementwise_loss
                else:
                    elementwise_loss = n_step_elementwise_loss

            loss = torch.mean(elementwise_loss * weights)

        else:
            if n_step:
                idxs = experiences["idxs"]
                n_states = n_experiences["obs"]
                n_actions = n_experiences["action"]
                n_rewards = n_experiences["reward"]
                n_next_states = n_experiences["next_obs"]
                n_dones = n_experiences["done"]
            else:
                idxs = None

            new_priorities = None
            if self.combined_reward or not n_step:
                elementwise_loss = self._dqn_loss(
                    states, actions, rewards, next_states, dones, self.gamma
                )

            if n_step:
                n_gamma = self.gamma**self.n_step
                n_step_elementwise_loss = self._dqn_loss(
                    n_states, n_actions, n_rewards, n_next_states, n_dones, n_gamma
                )
                if self.combined_reward:
                    elementwise_loss += n_step_elementwise_loss
                else:
                    elementwise_loss = n_step_elementwise_loss

            loss = torch.mean(elementwise_loss)

        self.optimizer.zero_grad()
        if self.accelerator is not None:
            self.accelerator.backward(loss)
        else:
            loss.backward()

        clip_grad_norm_(self.actor.parameters(), 10.0)
        self.optimizer.step()

        # soft update target network
        self.soft_update()
        self.actor.reset_noise()
        self.actor_target.reset_noise()
        if per:
            loss_for_prior = elementwise_loss.detach().cpu().numpy()
            new_priorities = loss_for_prior + self.prior_eps

        return loss.item(), idxs, new_priorities

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
    ) -> float:
        """Returns mean test score of agent in environment with epsilon-greedy policy.

        :param env: The environment to be tested in
        :type env: Gym-style environment
        :param swap_channels: Swap image channels dimension from last to first [H, W, C] -> [C, H, W], defaults to False
        :type swap_channels: bool, optional
        :param max_steps: Maximum number of testing steps, defaults to None
        :type max_steps: int, optional
        :param loop: Number of testing loops/episodes to complete. The returned score is the mean over these tests. Defaults to 3
        :type loop: int, optional
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
                    action = self.get_action(
                        obs, training=False, action_mask=action_mask
                    )
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
