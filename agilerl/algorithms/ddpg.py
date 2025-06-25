import copy
import warnings
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from gymnasium import spaces

from agilerl.algorithms.core import OptimizerWrapper, RLAlgorithm
from agilerl.algorithms.core.registry import HyperparameterConfig, NetworkGroup
from agilerl.modules.base import EvolvableModule
from agilerl.modules.configs import MlpNetConfig
from agilerl.networks.actors import DeterministicActor
from agilerl.networks.base import EvolvableNetwork
from agilerl.networks.q_networks import ContinuousQNetwork
from agilerl.typing import (
    ArrayLike,
    ExperiencesType,
    GymEnvType,
    ObservationType,
)
from agilerl.utils.algo_utils import (
    make_safe_deepcopies,
    obs_channels_to_first,
    share_encoder_parameters,
)


class DDPG(RLAlgorithm):
    """Deep Deterministic Policy Gradient (DDPG) algorithm.

    Paper: https://arxiv.org/abs/1509.02971

    :param observation_space: Environment observation space
    :type observation_space: gym.spaces.Space
    :param action_space: Environment action space
    :type action_space: gym.spaces.Box
    :param O_U_noise: Use Ornstein Uhlenbeck action noise for exploration. If False, uses Gaussian noise. Defaults to True
    :type O_U_noise: bool, optional
    :param expl_noise: Scale for Ornstein Uhlenbeck action noise, or standard deviation for Gaussian exploration noise, defaults to 0.1
    :type expl_noise: Union[float, ArrayLike], optional
    :param vect_noise_dim: Vectorization dimension of environment for action noise, defaults to 1
    :type vect_noise_dim: int, optional
    :param mean_noise: Mean of exploration noise, defaults to 0.0
    :type mean_noise: float, optional
    :param theta: Rate of mean reversion in Ornstein Uhlenbeck action noise, defaults to 0.15
    :type theta: float, optional
    :param dt: Timestep for Ornstein Uhlenbeck action noise update, defaults to 1e-2
    :type dt: float, optional
    :param index: Index to keep track of object instance during tournament selection and mutation, defaults to 0
    :type index: int, optional
    :param hp_config: RL hyperparameter mutation configuration, defaults to None, whereby algorithm mutations are disabled.
    :type hp_config: HyperparameterConfig, optional
    :param net_config: Encoder configuration, defaults to None
    :type net_config: Optional[Dict[str, Any]], optional
    :param batch_size: Size of batched sample from replay buffer for learning, defaults to 64
    :type batch_size: int, optional
    :param lr_actor: Learning rate for actor optimizer, defaults to 1e-4
    :type lr_actor: float, optional
    :param lr_critic: Learning rate for critic optimizer, defaults to 1e-3
    :type lr_critic: float, optional
    :param learn_step: Learning frequency, defaults to 5
    :type learn_step: int, optional
    :param gamma: Discount factor, defaults to 0.99
    :type gamma: float, optional
    :param tau: For soft update of target network parameters, defaults to 1e-3
    :type tau: float, optional
    :param normalize_images: Normalize images flag, defaults to True
    :type normalize_images: bool, optional
    :param mut: Most recent mutation to agent, defaults to None
    :type mut: Optional[str], optional
    :param policy_freq: Frequency of critic network updates compared to policy network, defaults to 2
    :type policy_freq: int, optional
    :param actor_network: Custom actor network, defaults to None
    :type actor_network: Optional[nn.Module], optional
    :param critic_network: Custom critic network, defaults to None
    :type critic_network: Optional[nn.Module], optional
    :param share_encoders: Share encoders between actor and critic, defaults to False
    :type share_encoders: bool, optional
    :param device: Device for accelerated computing, 'cpu' or 'cuda', defaults to 'cpu'
    :type device: str, optional
    :param accelerator: Accelerator for distributed computing, defaults to None
    :type accelerator: accelerate.Accelerator(), optional
    :param wrap: Wrap models for distributed training upon creation, defaults to True
    :type wrap: bool, optional
    """

    action_space: spaces.Box

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        O_U_noise: bool = True,
        expl_noise: Union[float, ArrayLike] = 0.1,
        vect_noise_dim: int = 1,
        mean_noise: float = 0.0,
        theta: float = 0.15,
        dt: float = 1e-2,
        index: int = 0,
        hp_config: Optional[HyperparameterConfig] = None,
        net_config: Optional[Dict[str, Any]] = None,
        batch_size: int = 64,
        lr_actor: float = 1e-4,
        lr_critic: float = 1e-3,
        learn_step: int = 5,
        gamma: float = 0.99,
        tau: float = 1e-3,
        normalize_images: bool = True,
        mut: Optional[str] = None,
        policy_freq: int = 2,
        actor_network: Optional[EvolvableModule] = None,
        critic_network: Optional[EvolvableModule] = None,
        share_encoders: bool = False,
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
            name="DDPG",
        )

        assert learn_step >= 1, "Learn step must be greater than or equal to one."
        assert isinstance(learn_step, int), "Learn step rate must be an integer."
        assert isinstance(
            action_space, spaces.Box
        ), "DDPG only supports continuous action spaces."
        assert (isinstance(expl_noise, (float, int))) or (
            isinstance(expl_noise, np.ndarray)
            and expl_noise.shape == (vect_noise_dim, self.action_dim)
        ), f"Exploration action noise rate must be a float, or an array of size {self.action_dim}"
        if isinstance(expl_noise, (float, int)):
            assert (
                expl_noise >= 0
            ), "Exploration noise must be greater than or equal to zero."
        assert isinstance(batch_size, int), "Batch size must be an integer."
        assert batch_size >= 1, "Batch size must be greater than or equal to one."
        assert isinstance(lr_actor, float), "Actor learning rate must be a float."
        assert lr_actor > 0, "Actor learning rate must be greater than zero."
        assert isinstance(lr_critic, float), "Critic learning rate must be a float."
        assert lr_critic > 0, "Critic learning rate must be greater than zero."
        assert isinstance(learn_step, int), "Learn step rate must be an integer."
        assert learn_step >= 1, "Learn step must be greater than or equal to one."
        assert isinstance(gamma, (float, int, torch.Tensor)), "Gamma must be a float."
        assert isinstance(tau, float), "Tau must be a float."
        assert tau > 0, "Tau must be greater than zero."
        assert isinstance(policy_freq, int), "Policy frequency must be an integer."
        assert (
            policy_freq >= 1
        ), "Policy frequency must be greater than or equal to one."

        if (actor_network is not None) != (critic_network is not None):  # XOR operation
            warnings.warn(
                "Actor and critic networks must both be supplied to use custom networks. Defaulting to net config."
            )
        assert isinstance(
            wrap, bool
        ), "Wrap models flag must be boolean value True or False."

        self.batch_size = batch_size
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.learn_step = learn_step
        self.net_config = net_config
        self.gamma = gamma
        self.tau = tau
        self.wrap = wrap
        self.mut = mut
        self.policy_freq = policy_freq
        self.O_U_noise = O_U_noise
        self.vect_noise_dim = vect_noise_dim
        self.share_encoders = share_encoders
        self.expl_noise = (
            expl_noise
            if isinstance(expl_noise, np.ndarray)
            else expl_noise * np.ones((vect_noise_dim, self.action_dim))
        )
        self.mean_noise = (
            mean_noise
            if isinstance(mean_noise, np.ndarray)
            else mean_noise * np.ones((vect_noise_dim, self.action_dim))
        )
        self.current_noise = np.zeros((vect_noise_dim, self.action_dim))
        self.theta = theta
        self.dt = dt
        self.learn_counter = 0

        if actor_network is not None and critic_network is not None:
            if not isinstance(actor_network, EvolvableModule):
                raise TypeError(
                    f"'actor_network' is of type {type(actor_network)}, but must be of type EvolvableModule."
                )
            if not isinstance(critic_network, EvolvableModule):
                raise TypeError(
                    f"'critic_network' is of type {type(critic_network)}, but must be of type EvolvableModule."
                )

            self.actor, self.critic = make_safe_deepcopies(
                actor_network, critic_network
            )
            self.actor_target, self.critic_target = make_safe_deepcopies(
                actor_network, critic_network
            )
        else:
            net_config = {} if net_config is None else net_config
            head_config = net_config.get("head_config", None)
            if head_config is not None:
                critic_head_config = copy.deepcopy(head_config)
                critic_head_config["output_activation"] = None
            else:
                critic_head_config = MlpNetConfig(hidden_size=[64])

            critic_net_config = copy.deepcopy(net_config)
            critic_net_config["head_config"] = critic_head_config

            def create_actor():
                return DeterministicActor(
                    observation_space=observation_space,
                    action_space=action_space,
                    device=self.device,
                    **net_config,
                )

            def create_critic():
                return ContinuousQNetwork(
                    observation_space=observation_space,
                    action_space=action_space,
                    device=self.device,
                    **critic_net_config,
                )

            self.actor = create_actor()
            self.actor_target = create_actor()
            self.critic = create_critic()
            self.critic_target = create_critic()

        # Share encoders between actor and critic
        if self.share_encoders and all(
            isinstance(net, EvolvableNetwork) for net in [self.actor, self.critic]
        ):
            self.share_encoder_parameters()

            # Need to register a mutation hook that does this after every mutation
            self.register_mutation_hook(self.share_encoder_parameters)

        # Initialize target networks
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Optimizers
        self.actor_optimizer = OptimizerWrapper(
            optim.Adam, networks=self.actor, lr=lr_actor
        )
        self.critic_optimizer = OptimizerWrapper(
            optim.Adam, networks=self.critic, lr=lr_critic
        )

        if self.accelerator is not None and wrap:
            self.wrap_models()

        self.criterion = nn.MSELoss()

        # Register network groups for actor and critic
        self.register_network_group(
            NetworkGroup(
                eval_network=self.actor,
                shared_networks=self.actor_target,
                policy=True,
            )
        )
        self.register_network_group(
            NetworkGroup(
                eval_network=self.critic,
                shared_networks=self.critic_target,
                policy=False,
            )
        )

    def share_encoder_parameters(self) -> None:
        """Shares the encoder parameters between the actor and critic. Registered as a mutation hook
        when share_encoders=True."""
        if all(isinstance(net, EvolvableNetwork) for net in [self.actor, self.critic]):
            share_encoder_parameters(self.actor, self.critic, self.critic_target)
        else:
            warnings.warn(
                "Encoder sharing is disabled as actor or critic is not an EvolvableNetwork."
            )

    def get_action(self, obs: ObservationType, training: bool = True) -> np.ndarray:
        """Returns the next action to take in the environment. If training, random noise
        is added to the action to promote exploration.

        :param obs: Environment observation, or multiple observations in a batch
        :type obs: numpy.ndarray[float]
        :param training: Agent is training, use exploration noise, defaults to True
        :type training: bool, optional
        :return: Action
        :rtype: numpy.ndarray[float]
        """
        obs = self.preprocess_observation(obs)
        self.actor.eval()
        with torch.no_grad():
            action: torch.Tensor = self.actor(obs)

        action = action.cpu().data.numpy()

        self.actor.train()
        if training:

            action += self.action_noise()

        return action.clip(self.action_space.low, self.action_space.high)

    def action_noise(self) -> ArrayLike:
        """Create action noise for exploration, either Ornstein Uhlenbeck or
            from a normal distribution.

        :return: Action noise
        :rtype: np.ndArray
        """
        if self.O_U_noise:
            noise = (
                self.current_noise
                + self.theta * (self.mean_noise - self.current_noise) * self.dt
                + self.expl_noise
                * np.sqrt(self.dt)
                * np.random.normal(size=(self.vect_noise_dim, self.action_dim))
            )
            self.current_noise = noise
        else:
            noise = np.random.normal(
                self.mean_noise,
                self.expl_noise,
                size=(self.vect_noise_dim, self.action_dim),
            )
        return noise.astype(np.float32)

    def multi_dim_clamp(
        self,
        min: Union[float, np.ndarray],
        max: Union[float, np.ndarray],
        input: torch.Tensor,
    ) -> torch.Tensor:
        """Multi-dimensional clamp function

        :param min: Minimum value or array of minimum values
        :type min: Union[float, np.ndarray]
        :param max: Maximum value or array of maximum values
        :type max: Union[float, np.ndarray]
        :param input: Input tensor to be clamped
        :type input: torch.Tensor
        :return: Clamped tensor
        :rtype: torch.Tensor
        """
        if not isinstance(min, np.ndarray) and not isinstance(max, np.ndarray):
            return torch.clamp(input, min, max)

        device = self.device if self.accelerator is None else self.accelerator.device
        min = torch.from_numpy(min).to(device) if isinstance(min, np.ndarray) else min
        max = torch.from_numpy(max).to(device) if isinstance(max, np.ndarray) else max

        if isinstance(max, torch.Tensor) and isinstance(min, (int, float)):
            min = torch.full_like(max, min).to(device)
        if isinstance(min, torch.Tensor) and isinstance(max, (int, float)):
            max = torch.full_like(min, max).to(device)

        output = torch.max(torch.min(input, max), min).type(input.dtype)

        return output

    def reset_action_noise(self, indices: ArrayLike) -> None:
        """Reset action noise."""
        self.current_noise[indices] = self.mean_noise[indices]

    def learn(
        self,
        experiences: ExperiencesType,
        noise_clip: float = 0.5,
        policy_noise: float = 0.2,
    ) -> Tuple[float, float]:
        """Updates agent network parameters to learn from experiences.

        :param experiences: TensorDict of batched observations, actions, rewards, next_observations, dones.
        :type experiences: dict[str, torch.Tensor[float]]
        :param noise_clip: Maximum noise limit to apply to actions, defaults to 0.5
        :type noise_clip: float, optional
        :param policy_noise: Standard deviation of noise applied to policy, defaults to 0.2
        :type policy_noise: float, optional
        """
        obs = experiences["obs"]
        actions = experiences["action"]
        rewards = experiences["reward"]
        next_obs = experiences["next_obs"]
        dones = experiences["done"]

        obs = self.preprocess_observation(obs)
        next_obs = self.preprocess_observation(next_obs)

        q_value = self.critic(obs, actions)
        with torch.no_grad():
            next_actions = self.actor_target(next_obs)
            noise = actions.data.normal_(0, policy_noise)
            noise = self.multi_dim_clamp(-noise_clip, noise_clip, noise)
            next_actions = next_actions + noise
            next_actions = self.multi_dim_clamp(
                self.action_space.low, self.action_space.high, next_actions
            )

            q_value_next_state = self.critic_target(next_obs, next_actions)

        y_j = rewards + ((1 - dones) * self.gamma * q_value_next_state)

        critic_loss: torch.Tensor = self.criterion(q_value, y_j)

        # critic loss backprop
        self.critic_optimizer.zero_grad()
        if self.accelerator is not None:
            self.accelerator.backward(critic_loss)
        else:
            critic_loss.backward()

        self.critic_optimizer.step()

        # update actor and targets every policy_freq learn steps
        self.learn_counter += 1
        if self.learn_counter % self.policy_freq == 0:
            policy_actions = self.actor(obs)

            # actor loss
            actor_loss = -self.critic(obs, policy_actions).mean()

            # actor loss backprop
            self.actor_optimizer.zero_grad()
            if self.accelerator is not None:
                self.accelerator.backward(actor_loss)
            else:
                actor_loss.backward()
            self.actor_optimizer.step()

            self.soft_update(self.actor, self.actor_target)
            self.soft_update(self.critic, self.critic_target)

            actor_loss = actor_loss.item()
            critic_loss = critic_loss.item()

        else:
            actor_loss = None
            critic_loss = critic_loss.item()

        return actor_loss, critic_loss

    def soft_update(self, net: nn.Module, target: nn.Module) -> None:
        """Soft updates target network parameters.

        :param net: Network with parameters to be copied from
        :type net: nn.Module
        :param target: Target network with parameters to be updated
        :type target: nn.Module
        """
        for eval_param, target_param in zip(net.parameters(), target.parameters()):
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
        :param loop: Number of testing loops/episodes to complete. The returned score is the mean. Defaults to 3
        :type loop: int, optional

        :return: Mean test score
        :rtype: float
        """
        self.set_training_mode(False)
        with torch.no_grad():
            rewards = []
            num_envs = env.num_envs if hasattr(env, "num_envs") else 1
            for i in range(loop):
                obs, _ = env.reset()
                scores = np.zeros(num_envs)
                completed_episode_scores = np.zeros(num_envs)
                finished = np.zeros(num_envs)
                step = 0
                while not np.all(finished):
                    if swap_channels:
                        obs = obs_channels_to_first(obs)
                    action = self.get_action(obs, training=False)
                    obs, reward, done, trunc, _ = env.step(action)
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
