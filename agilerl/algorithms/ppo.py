import copy
import warnings
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
from agilerl.networks.actors import StochasticActor
from agilerl.networks.base import EvolvableNetwork
from agilerl.networks.value_networks import ValueNetwork
from agilerl.typing import ArrayOrTensor, ExperiencesType, GymEnvType
from agilerl.utils.algo_utils import (
    flatten_experiences,
    get_experiences_samples,
    is_vectorized_experiences,
    make_safe_deepcopies,
    obs_channels_to_first,
    share_encoder_parameters,
    stack_experiences,
)


class PPO(RLAlgorithm):
    """Proximal Policy Optimization (PPO) algorithm.

    Paper: https://arxiv.org/abs/1707.06347v2

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
    :param learn_step: Learning frequency, defaults to 2048
    :type learn_step: int, optional
    :param gamma: Discount factor, defaults to 0.99
    :type gamma: float, optional
    :param gae_lambda: Lambda for general advantage estimation, defaults to 0.95
    :type gae_lambda: float, optional
    :param mut: Most recent mutation to agent, defaults to None
    :type mut: str, optional
    :param action_std_init: Initial action standard deviation, defaults to 0.0
    :type action_std_init: float, optional
    :param clip_coef: Surrogate clipping coefficient, defaults to 0.2
    :type clip_coef: float, optional
    :param ent_coef: Entropy coefficient, defaults to 0.01
    :type ent_coef: float, optional
    :param vf_coef: Value function coefficient, defaults to 0.5
    :type vf_coef: float, optional
    :param max_grad_norm: Maximum norm for gradient clipping, defaults to 0.5
    :type max_grad_norm: float, optional
    :param target_kl: Target KL divergence threshold, defaults to None
    :type target_kl: float, optional
    :param normalize_images: Flag to normalize images, defaults to True
    :type normalize_images: bool, optional
    :param update_epochs: Number of policy update epochs, defaults to 4
    :type update_epochs: int, optional
    :param actor_network: Custom actor network, defaults to None
    :type actor_network: nn.Module, optional
    :param critic_network: Custom critic network, defaults to None
    :type critic_network: nn.Module, optional
    :param share_encoders: Flag to share encoder parameters between actor and critic, defaults to False
    :type share_encoders: bool, optional
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
        learn_step: int = 2048,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        mut: Optional[str] = None,
        action_std_init: float = 0.0,
        clip_coef: float = 0.2,
        ent_coef: float = 0.01,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        target_kl: Optional[float] = None,
        normalize_images: bool = True,
        update_epochs: int = 4,
        actor_network: Optional[EvolvableModule] = None,
        critic_network: Optional[EvolvableModule] = None,
        share_encoders: bool = True,
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
            name="PPO",
        )

        assert learn_step >= 1, "Learn step must be greater than or equal to one."
        assert isinstance(learn_step, int), "Learn step rate must be an integer."
        assert isinstance(batch_size, int), "Batch size must be an integer."
        assert batch_size >= 1, "Batch size must be greater than or equal to one."
        assert isinstance(lr, float), "Learning rate must be a float."
        assert lr > 0, "Learning rate must be greater than zero."
        assert isinstance(gamma, (float, int, torch.Tensor)), "Gamma must be a float."
        assert isinstance(gae_lambda, (float, int)), "Lambda must be a float."
        assert gae_lambda >= 0, "Lambda must be greater than or equal to zero."
        assert isinstance(
            action_std_init, (float, int)
        ), "Action standard deviation must be a float."
        assert (
            action_std_init >= 0
        ), "Action standard deviation must be greater than or equal to zero."
        assert isinstance(
            clip_coef, (float, int)
        ), "Clipping coefficient must be a float."
        assert (
            clip_coef >= 0
        ), "Clipping coefficient must be greater than or equal to zero."
        assert isinstance(
            ent_coef, (float, int)
        ), "Entropy coefficient must be a float."
        assert (
            ent_coef >= 0
        ), "Entropy coefficient must be greater than or equal to zero."
        assert isinstance(
            vf_coef, (float, int)
        ), "Value function coefficient must be a float."
        assert (
            vf_coef >= 0
        ), "Value function coefficient must be greater than or equal to zero."
        assert isinstance(
            max_grad_norm, (float, int)
        ), "Maximum norm for gradient clipping must be a float."
        assert (
            max_grad_norm >= 0
        ), "Maximum norm for gradient clipping must be greater than or equal to zero."
        assert (
            isinstance(target_kl, (float, int)) or target_kl is None
        ), "Target KL divergence threshold must be a float."
        if target_kl is not None:
            assert (
                target_kl >= 0
            ), "Target KL divergence threshold must be greater than or equal to zero."
        assert isinstance(
            update_epochs, int
        ), "Policy update epochs must be an integer."
        assert (
            update_epochs >= 1
        ), "Policy update epochs must be greater than or equal to one."
        assert isinstance(
            wrap, bool
        ), "Wrap models flag must be boolean value True or False."

        self.batch_size = batch_size
        self.lr = lr
        self.gamma = gamma
        self.learn_step = learn_step
        self.mut = mut
        self.gae_lambda = gae_lambda
        self.action_std_init = action_std_init
        self.net_config = net_config
        self.clip_coef = clip_coef
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.target_kl = target_kl
        self.update_epochs = update_epochs

        if actor_network is not None and critic_network is not None:
            if not isinstance(actor_network, EvolvableModule):
                raise TypeError(
                    f"Passed actor network is of type {type(actor_network)}, but must be of type EvolvableModule."
                )
            if not isinstance(critic_network, EvolvableModule):
                raise TypeError(
                    f"Passed critic network is of type {type(critic_network)}, but must be of type EvolvableModule."
                )

            self.actor, self.critic = make_safe_deepcopies(
                actor_network, critic_network
            )

        else:
            net_config = {} if net_config is None else net_config
            critic_net_config = copy.deepcopy(net_config)

            head_config = net_config.get("head_config", None)
            if head_config is not None:
                critic_head_config = copy.deepcopy(head_config)
                critic_head_config["output_activation"] = None
                critic_net_config.pop("squash_output", None)
            else:
                critic_head_config = MlpNetConfig(hidden_size=[16])

            critic_net_config["head_config"] = critic_head_config

            self.actor = StochasticActor(
                observation_space,
                action_space,
                action_std_init=self.action_std_init,
                device=self.device,
                **net_config,
            )

            self.critic = ValueNetwork(
                observation_space,
                device=self.device,
                **critic_net_config,
            )

        # Share encoders between actor and critic
        self.share_encoders = share_encoders
        if self.share_encoders and all(
            isinstance(net, EvolvableNetwork) for net in [self.actor, self.critic]
        ):
            self.share_encoder_parameters()

            # Need to register a mutation hook that does this after every mutation
            self.register_mutation_hook(self.share_encoder_parameters)

        self.optimizer = OptimizerWrapper(
            optim.Adam, networks=[self.actor, self.critic], lr=self.lr
        )

        if self.accelerator is not None and wrap:
            self.wrap_models()

        # Register network groups for mutations
        self.register_network_group(NetworkGroup(eval=self.actor, policy=True))
        self.register_network_group(NetworkGroup(eval=self.critic))

    def share_encoder_parameters(self) -> None:
        """Shares the encoder parameters between the actor and critic."""
        if all(isinstance(net, EvolvableNetwork) for net in [self.actor, self.critic]):
            share_encoder_parameters(self.actor, self.critic)
        else:
            warnings.warn(
                "Encoder sharing is disabled as actor or critic is not an EvolvableNetwork."
            )

    def _get_action_and_values(
        self,
        obs: ArrayOrTensor,
        action_mask: Optional[ArrayOrTensor] = None,
    ) -> Tuple[ArrayOrTensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns the next action to take in the environment and the values.

        :param obs: Environment observation, or multiple observations in a batch
        :type obs: numpy.ndarray[float]
        :param action_mask: Mask of legal actions 1=legal 0=illegal, defaults to None
        :type action_mask: numpy.ndarray, optional
        :return: Action, log probability, entropy, and state values
        :rtype: Tuple[ArrayOrTensor, torch.Tensor, torch.Tensor, torch.Tensor]
        """
        latent_pi = self.actor.extract_features(obs)
        action, log_prob, entropy = self.actor.forward_head(
            latent_pi, action_mask=action_mask
        )
        values = (
            self.critic.forward_head(latent_pi).squeeze(-1)
            if self.share_encoders
            else self.critic(obs).squeeze(-1)
        )
        return action, log_prob, entropy, values

    def evaluate_actions(
        self,
        obs: ArrayOrTensor,
        actions: ArrayOrTensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluates the actions.

        :param obs: Environment observation, or multiple observations in a batch
        :type obs: numpy.ndarray[float]
        :param actions: Actions to evaluate
        :type actions: torch.Tensor
        :return: Log probability, entropy, and state values
        :rtype: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        """
        obs = self.preprocess_observation(obs)
        _, _, entropy, values = self._get_action_and_values(obs)

        # log_prob of passed actions given the current policy
        log_prob = self.actor.action_log_prob(actions)

        # Use -log_prob as entropy when squashing output in continuous action spaces
        entropy = -log_prob.mean() if entropy is None else entropy

        return log_prob, entropy, values

    def get_action(
        self,
        obs: ArrayOrTensor,
        action_mask: Optional[ArrayOrTensor] = None,
    ) -> Tuple[ArrayOrTensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns the next action to take in the environment.

        :param obs: Environment observation, or multiple observations in a batch
        :type obs: numpy.ndarray[float]
        :param action_mask: Mask of legal actions 1=legal 0=illegal, defaults to None
        :type action_mask: numpy.ndarray, optional
        :return: Action, log probability, entropy, and state values
        :rtype: Tuple[ArrayOrTensor, torch.Tensor, torch.Tensor, torch.Tensor]
        """
        obs = self.preprocess_observation(obs)
        with torch.no_grad():
            action, log_prob, entropy, values = self._get_action_and_values(
                obs, action_mask
            )

        # Use -log_prob as entropy when squashing output in continuous action spaces
        entropy = -log_prob.mean() if entropy is None else entropy

        if isinstance(self.action_space, spaces.Box) and self.action_space.shape == (
            1,
        ):
            action = action.unsqueeze(1)

        # Clip to action space during inference
        action = action.cpu().data.numpy()
        if not self.training and isinstance(self.action_space, spaces.Box):
            if self.actor.squash_output:
                action = self.actor.scale_action(action)
            else:
                action = np.clip(action, self.action_space.low, self.action_space.high)

        return (
            action,
            log_prob.cpu().data.numpy(),
            entropy.cpu().data.numpy(),
            values.cpu().data.numpy(),
        )

    def learn(self, experiences: ExperiencesType) -> float:
        """Updates agent network parameters to learn from experiences.

        :param experience: List of batched states, actions, log_probs, rewards, dones, values, next_state, next_done in that order.
        :type experience: Tuple[Union[numpy.ndarray, Dict[str, numpy.ndarray]], ...]
        """
        (states, actions, log_probs, rewards, dones, values, next_state, next_done) = (
            stack_experiences(*experiences)
        )

        # Bootstrapping returns using GAE advantage estimation
        dones = dones.long()
        with torch.no_grad():
            num_steps = rewards.size(0)
            next_state = self.preprocess_observation(next_state)
            next_value = self.critic(next_state).reshape(1, -1).cpu()
            advantages = torch.zeros_like(rewards).float()
            last_gae_lambda = 0
            for t in reversed(range(num_steps)):
                if t == num_steps - 1:
                    next_non_terminal = 1.0 - next_done
                    nextvalue = next_value.squeeze()
                else:
                    next_non_terminal = 1.0 - dones[t + 1]
                    nextvalue = values[t + 1]

                # Calculate delta (TD error)
                delta = (
                    rewards[t] + self.gamma * nextvalue * next_non_terminal - values[t]
                )

                # Use recurrence relation to compute advantage
                advantages[t] = last_gae_lambda = (
                    delta
                    + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lambda
                )

            returns = advantages + values

        # Flatten experiences from (batch_size, num_envs, ...) to (batch_size*num_envs, ...)
        # after checking if experiences are vectorized
        experiences = (states, actions, log_probs, advantages, returns, values)
        if is_vectorized_experiences(*experiences):
            experiences = flatten_experiences(*experiences)

        # Move experiences to algo device
        experiences = self.to_device(*experiences)

        # Get number of samples from the returns tensor
        num_samples = experiences[4].size(0)
        batch_idxs = np.arange(num_samples)
        mean_loss = 0
        for epoch in range(self.update_epochs):
            np.random.shuffle(batch_idxs)
            for start in range(0, num_samples, self.batch_size):
                minibatch_idxs = batch_idxs[start : start + self.batch_size]
                (
                    batch_states,
                    batch_actions,
                    batch_log_probs,
                    batch_advantages,
                    batch_returns,
                    batch_values,
                ) = get_experiences_samples(minibatch_idxs, *experiences)

                batch_actions = batch_actions.squeeze()
                batch_returns = batch_returns.squeeze()
                batch_log_probs = batch_log_probs.squeeze()
                batch_advantages = batch_advantages.squeeze()
                batch_values = batch_values.squeeze()

                if len(minibatch_idxs) > 1:
                    log_prob, entropy, value = self.evaluate_actions(
                        obs=batch_states, actions=batch_actions
                    )

                    logratio = log_prob - batch_log_probs
                    ratio = logratio.exp()

                    with torch.no_grad():
                        approx_kl = ((ratio - 1) - logratio).mean()

                    minibatch_advs = batch_advantages
                    minibatch_advs = (minibatch_advs - minibatch_advs.mean()) / (
                        minibatch_advs.std() + 1e-8
                    )

                    # Policy loss
                    pg_loss1 = -minibatch_advs * ratio
                    pg_loss2 = -minibatch_advs * torch.clamp(
                        ratio, 1 - self.clip_coef, 1 + self.clip_coef
                    )

                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # Value loss
                    value = value.view(-1)
                    v_loss_unclipped = (value - batch_returns) ** 2
                    v_clipped = batch_values + torch.clamp(
                        value - batch_values, -self.clip_coef, self.clip_coef
                    )

                    v_loss_clipped = (v_clipped - batch_returns) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()

                    entropy_loss = entropy.mean()
                    loss = (
                        pg_loss - self.ent_coef * entropy_loss + v_loss * self.vf_coef
                    )

                    # actor + critic loss backprop
                    self.optimizer.zero_grad()
                    if self.accelerator is not None:
                        self.accelerator.backward(loss)
                    else:
                        loss.backward()

                    # Clip gradients
                    clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                    clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)

                    self.optimizer.step()

                    mean_loss += loss.item()

            if self.target_kl is not None:
                if approx_kl > self.target_kl:
                    break

        mean_loss /= num_samples * self.update_epochs
        return mean_loss

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

        :return: Mean test score of agent in environment
        :rtype: float
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
                    action, _, _, _ = self.get_action(obs, action_mask=action_mask)
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
