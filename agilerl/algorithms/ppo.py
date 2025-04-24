import copy
import warnings
from typing import Any, Dict, Optional, Tuple, Union, List

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
from agilerl.utils.rollout_buffer import RolloutBuffer


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
    :param num_envs: Number of parallel environments, defaults to 1
    :type num_envs: int, optional
    :param use_rollout_buffer: Flag to use the rollout buffer instead of tuple experiences, defaults to False
    :type use_rollout_buffer: bool, optional
    :param recurrent: Flag to use hidden states for recurrent policies, defaults to False
    :type recurrent: bool, optional
    :param hidden_state_size: Size of hidden states if used, defaults to None
    :type hidden_state_size: int, optional
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
        num_envs: int = 1,
        use_rollout_buffer: bool = False,
        rollout_buffer_config: Optional[Dict[str, Any]] = {},
        recurrent: bool = False,
        hidden_state_size: Optional[int] = None,
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
        assert isinstance(learn_step, int), "Learn step must be an integer."
        assert isinstance(batch_size, int), "Batch size must be an integer."
        assert batch_size >= 1, "Batch size must be greater than or equal to one."
        assert isinstance(lr, float), "Learning rate must be a float."
        assert lr > 0, "Learning rate must be greater than zero."
        assert isinstance(gamma, (float, int, torch.Tensor)), "Gamma must be a float."
        assert isinstance(gae_lambda, (float, int)), "Lambda must be a float."
        assert gae_lambda >= 0, "Lambda must be greater than or equal to zero."
        assert isinstance(action_std_init, (float, int)), (
            "Action standard deviation must be a float."
        )
        assert action_std_init >= 0, (
            "Action standard deviation must be greater than or equal to zero."
        )
        assert isinstance(clip_coef, (float, int)), (
            "Clipping coefficient must be a float."
        )
        assert clip_coef >= 0, (
            "Clipping coefficient must be greater than or equal to zero."
        )
        assert isinstance(ent_coef, (float, int)), (
            "Entropy coefficient must be a float."
        )
        assert ent_coef >= 0, (
            "Entropy coefficient must be greater than or equal to zero."
        )
        assert isinstance(vf_coef, (float, int)), (
            "Value function coefficient must be a float."
        )
        assert vf_coef >= 0, (
            "Value function coefficient must be greater than or equal to zero."
        )
        assert isinstance(max_grad_norm, (float, int)), (
            "Maximum norm for gradient clipping must be a float."
        )
        assert max_grad_norm >= 0, (
            "Maximum norm for gradient clipping must be greater than or equal to zero."
        )
        assert isinstance(target_kl, (float, int)) or target_kl is None, (
            "Target KL divergence threshold must be a float."
        )
        if target_kl is not None:
            assert target_kl >= 0, (
                "Target KL divergence threshold must be greater than or equal to zero."
            )
        assert isinstance(update_epochs, int), (
            "Policy update epochs must be an integer."
        )
        assert update_epochs >= 1, (
            "Policy update epochs must be greater than or equal to one."
        )
        assert isinstance(wrap, bool), (
            "Wrap models flag must be boolean value True or False."
        )

        # New parameters for using RolloutBuffer
        assert isinstance(use_rollout_buffer, bool), (
            "Use rollout buffer flag must be boolean value True or False."
        )
        assert isinstance(recurrent, bool), (
            "Has hidden states flag must be boolean value True or False."
        )
        if recurrent and hidden_state_size is None:
            warnings.warn(
                "Hidden states enabled but hidden_state_size not provided. Using default hidden_state_size."
            )

        if not use_rollout_buffer:
            warnings.warn(
                (
                    "DeprecationWarning: 'use_rollout_buffer=False' is deprecated and will be removed in a future release. "
                    "The PPO implementation now expects 'use_rollout_buffer=True' for improved performance, "
                    "cleaner support for recurrent policies, and easier integration with custom environments. "
                    "Please update your code to use 'use_rollout_buffer=True' and, if you require recurrent policies, set 'recurrent=True'.\n"
                    "Refer to the documentation for migration instructions and further details."
                ),
                DeprecationWarning,
                stacklevel=2,
            )

        if recurrent and not use_rollout_buffer:
            raise ValueError("use_rollout_buffer must be True if recurrent=True.")

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
        self.use_rollout_buffer = use_rollout_buffer
        self.num_envs = num_envs
        self.recurrent = recurrent
        self.hidden_state_size = hidden_state_size

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
            if (
                (
                    net_config.get("encoder_config", None) is None
                    or net_config.get("encoder_config", None).get(
                        "hidden_state_size", None
                    )
                    is None
                )
                and self.recurrent
                and self.hidden_state_size is not None
            ):
                # Set hidden state size for recurrent networks
                net_config["encoder_config"] = {
                    "hidden_state_size": self.hidden_state_size
                }

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
                recurrent=self.recurrent,
                **net_config,
            )

            self.critic = ValueNetwork(
                observation_space,
                device=self.device,
                recurrent=self.recurrent,
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

        # Initialize rollout buffer if enabled
        if self.use_rollout_buffer:
            self.rollout_buffer = RolloutBuffer(
                capacity=self.learn_step,
                observation_space=self.observation_space,
                action_space=self.action_space,
                device=self.device,
                num_envs=self.num_envs,
                gae_lambda=self.gae_lambda,
                gamma=self.gamma,
                recurrent=self.recurrent,
                hidden_state_size=self.hidden_state_size,
                **rollout_buffer_config,
            )

        if self.accelerator is not None and wrap:
            self.wrap_models()

        # Register network groups for mutations
        self.register_network_group(NetworkGroup(eval=self.actor, policy=True))
        self.register_network_group(NetworkGroup(eval=self.critic))

        self.hidden_state = None

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
        hidden_state: Optional[ArrayOrTensor] = None,
    ) -> Tuple[
        ArrayOrTensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[ArrayOrTensor]
    ]:
        """Returns the next action to take in the environment and the values.

        :param obs: Environment observation, or multiple observations in a batch
        :type obs: numpy.ndarray[float]
        :param action_mask: Mask of legal actions 1=legal 0=illegal, defaults to None
        :type action_mask: numpy.ndarray, optional
        :param hidden_state: Hidden state for recurrent policies, defaults to None
        :type hidden_state: numpy.ndarray, optional
        :return: Action, log probability, entropy, state values, and next hidden state
        :rtype: Tuple[ArrayOrTensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[ArrayOrTensor]]
        """
        if hidden_state is not None:
            latent_pi, next_hidden = self.actor.extract_features(
                obs, hidden_state=hidden_state
            )
            action, log_prob, entropy = self.actor.forward_head(
                latent_pi, action_mask=action_mask
            )
            values = (
                self.critic.forward_head(latent_pi).squeeze(-1)
                if self.share_encoders
                else self.critic(obs, hidden_state=next_hidden).squeeze(-1)
            )
            return action, log_prob, entropy, values, next_hidden
        else:
            latent_pi = self.actor.extract_features(obs)
            action, log_prob, entropy = self.actor.forward_head(
                latent_pi, action_mask=action_mask
            )
            values = (
                self.critic.forward_head(latent_pi).squeeze(-1)
                if self.share_encoders
                else self.critic(obs).squeeze(-1)
            )
            return action, log_prob, entropy, values, None

    def get_initial_hidden_state(self, num_envs: int) -> ArrayOrTensor:
        """
        Get the initial hidden state for the environment.
        """
        if not self.recurrent:
            raise ValueError(
                "Cannot get initial hidden state for non-recurrent networks."
            )

        # Return a batch of initial hidden states
        # Assuming self.actor.initialize_hidden_state() returns a single state (e.g., zeros)
        return self.actor.initialize_hidden_state(batch_size=num_envs)

    def evaluate_actions(
        self,
        obs: ArrayOrTensor,
        actions: ArrayOrTensor,
        hidden_state: Optional[ArrayOrTensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluates the actions.

        :param obs: Environment observation, or multiple observations in a batch
        :type obs: numpy.ndarray[float]
        :param actions: Actions to evaluate
        :type actions: torch.Tensor
        :param hidden_state: Hidden state for recurrent policies, defaults to None
        :type hidden_state: numpy.ndarray, optional
        :return: Log probability, entropy, and state values
        :rtype: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        """
        obs = self.preprocess_observation(obs)

        # Get values from actor-critic
        if self.recurrent and hidden_state is not None:
            # With recurrent state
            _, _, entropy, values, _ = self._get_action_and_values(
                obs, hidden_state=hidden_state
            )
        else:
            # Without recurrent state
            _, _, entropy, values, _ = self._get_action_and_values(obs)

        # Get log probability of the actions
        log_prob = self.actor.action_log_prob(actions)

        # Use -log_prob as entropy when squashing output in continuous action spaces
        if entropy is None:
            entropy = -log_prob.mean()

        return log_prob, entropy, values

    def get_action(
        self,
        obs: ArrayOrTensor,
        action_mask: Optional[ArrayOrTensor] = None,
        hidden_state: Optional[Dict[str, ArrayOrTensor]] = None,
    ) -> Union[
        Tuple[
            ArrayOrTensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            Optional[ArrayOrTensor],
        ],
        Tuple[ArrayOrTensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ]:
        """Returns the next action to take in the environment.

        :param obs: Environment observation, or multiple observations in a batch
        :type obs: numpy.ndarray[float]
        :param action_mask: Mask of legal actions 1=legal 0=illegal, defaults to None
        :type action_mask: numpy.ndarray, optional
        :param hidden_state: Hidden state for recurrent policies, defaults to None
        :type hidden_state: numpy.ndarray, optional
        :return: Action, log probability, entropy, state values, and next hidden state
        :rtype: Tuple[ArrayOrTensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[ArrayOrTensor]]
        """
        obs = self.preprocess_observation(obs)
        with torch.no_grad():
            if self.recurrent and hidden_state is not None:
                action, log_prob, entropy, values, next_hidden = (
                    self._get_action_and_values(obs, action_mask, hidden_state)
                )
            else:
                action, log_prob, entropy, values, next_hidden = (
                    self._get_action_and_values(obs, action_mask)
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

        if self.recurrent:
            return (
                action,
                log_prob.cpu().data.numpy(),
                entropy.cpu().data.numpy(),
                values.cpu().data.numpy(),
                next_hidden if next_hidden is not None else None,
            )
        else:
            return (
                action,
                log_prob.cpu().data.numpy(),
                entropy.cpu().data.numpy(),
                values.cpu().data.numpy(),
            )

    def collect_rollouts(
        self,
        env: GymEnvType,
        n_steps: int = None,
    ) -> None:
        """
        Collect rollouts from the environment and store them in the rollout buffer.

        :param env: The environment to collect rollouts from
        :type env: GymEnvType
        :param n_steps: Number of steps to collect, defaults to self.learn_step
        :type n_steps: int, optional
        """
        if not self.use_rollout_buffer:
            raise RuntimeError(
                "collect_rollouts can only be used when use_rollout_buffer=True"
            )

        n_steps = n_steps or self.learn_step
        self.rollout_buffer.reset()

        # Initial reset
        obs, info = env.reset()
        self.hidden_state = (
            self.get_initial_hidden_state(self.num_envs) if self.recurrent else None
        )

        for _ in range(n_steps):
            # Get action
            if self.recurrent:
                action, log_prob, _, value, next_hidden = self.get_action(
                    obs,
                    action_mask=info.get("action_mask", None),
                    hidden_state=self.hidden_state,
                )
                self.hidden_state = next_hidden
            else:
                # No need for next_hidden in non-recurrent networks, so we're not even returning it
                action, log_prob, _, value = self.get_action(
                    obs, action_mask=info.get("action_mask", None)
                )

            # Execute action
            next_obs, reward, done, truncated, next_info = env.step(action)

            # Add to buffer
            is_terminal = done or truncated

            # Ensure shapes are correct (num_envs, ...)
            reward = np.atleast_1d(reward)
            is_terminal = np.atleast_1d(is_terminal)
            value = np.atleast_1d(value)
            log_prob = np.atleast_1d(log_prob)

            self.rollout_buffer.add(
                obs=obs,
                action=action,
                reward=reward,
                done=is_terminal,
                value=value.reshape(-1),
                log_prob=log_prob,
                next_obs=next_obs,
                hidden_state=self.hidden_state,
            )

            # Reset hidden state for finished environments
            if self.recurrent and np.any(is_terminal):
                # Create a mask for finished environments
                finished_mask = is_terminal.astype(bool)
                # Get initial hidden states only for the finished environments
                num_finished = finished_mask.sum()
                # Need a way to get initial state for a subset of envs
                # For simplicity, re-initialize all and mask later, or handle dicts/tensors carefully
                initial_hidden_states_for_reset = self.get_initial_hidden_state(
                    self.num_envs
                )

                if isinstance(self.hidden_state, torch.Tensor):
                    reset_states = initial_hidden_states_for_reset[finished_mask]
                    if reset_states.shape[0] > 0:  # Only update if any finished
                        self.hidden_state[finished_mask] = reset_states
                elif isinstance(self.hidden_state, dict):
                    for key in self.hidden_state:
                        reset_states = initial_hidden_states_for_reset[key][
                            finished_mask
                        ]
                        if reset_states.shape[0] > 0:
                            self.hidden_state[key][finished_mask] = reset_states
                # Add handling for numpy if needed

            # Update for next step
            obs = next_obs
            info = next_info

        # Compute advantages and returns
        with torch.no_grad():
            # Get value for last observation
            if self.recurrent:
                _, _, _, last_value, _ = self._get_action_and_values(
                    obs, hidden_state=self.hidden_state
                )
            else:
                _, _, _, last_value, _ = self._get_action_and_values(obs)

            last_value = last_value.cpu().numpy()
            last_done = np.atleast_1d(done)  # Ensure last_done has shape (num_envs,)

        # Compute returns and advantages
        self.rollout_buffer.compute_returns_and_advantages(
            last_value=last_value, last_done=last_done
        )

    def learn(self, experiences: Union[ExperiencesType, None] = None) -> float:
        """Updates agent network parameters to learn from experiences.

        :param experience: List of batched states, actions, log_probs, rewards, dones, values, next_state, next_done in that order.
                        If use_rollout_buffer=True and experiences=None, uses data from rollout buffer.
        :type experience: Tuple[Union[numpy.ndarray, Dict[str, numpy.ndarray]], ...] or None
        """
        # Use rollout buffer if enabled and no experiences provided
        if self.use_rollout_buffer and experiences is None:
            return self._learn_from_rollout_buffer()
        elif self.use_rollout_buffer and experiences is not None:
            warnings.warn(
                "Both rollout buffer and experiences provided. Using provided experiences."
            )
        elif not self.use_rollout_buffer and experiences is None:
            raise ValueError(
                "Experiences cannot be None when use_rollout_buffer is False"
            )

        # Legacy learning from experiences tuple
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

    def _learn_from_rollout_buffer(self) -> float:
        """
        Learn from data in the rollout buffer.

        :return: Mean loss value
        :rtype: float
        """
        # Get data from buffer as tensors
        buffer_data = self.rollout_buffer.get_tensor_batch(device=self.device)

        # Extract tensors
        observations = buffer_data["observations"]
        actions = buffer_data["actions"]
        old_log_probs = buffer_data["log_probs"]
        advantages = buffer_data["advantages"]
        returns = buffer_data["returns"]
        values = buffer_data["values"]

        # Normalize advantages (helps with training stability)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Prepare for minibatch updates
        batch_size = self.batch_size
        num_samples = observations.size(0)
        assert num_samples == self.rollout_buffer.size(), (
            f"Expected {self.rollout_buffer.size()} samples, but got {num_samples}"
        )
        indices = np.arange(num_samples)

        mean_loss = 0
        approx_kl_divs = []

        for epoch in range(self.update_epochs):
            np.random.shuffle(indices)

            for start_idx in range(0, num_samples, batch_size):
                end_idx = min(start_idx + batch_size, num_samples)
                minibatch_indices = indices[start_idx:end_idx]

                # Get minibatch data
                mb_obs = observations[minibatch_indices]
                mb_actions = actions[minibatch_indices]
                mb_old_log_probs = old_log_probs[minibatch_indices]
                mb_advantages = advantages[minibatch_indices]
                mb_returns = returns[minibatch_indices]
                mb_old_values = values[minibatch_indices]

                mb_hidden_states = (
                    buffer_data["hidden_states"][minibatch_indices]
                    if self.recurrent
                    else None
                )

                # Evaluate actions and calculate policy loss
                new_log_probs, entropy, new_values = self.evaluate_actions(
                    mb_obs, mb_actions, hidden_state=mb_hidden_states
                )

                # Policy loss
                ratio = torch.exp(new_log_probs - mb_old_log_probs)
                policy_loss1 = -mb_advantages * ratio
                policy_loss2 = -mb_advantages * torch.clamp(
                    ratio, 1 - self.clip_coef, 1 + self.clip_coef
                )
                policy_loss = torch.max(policy_loss1, policy_loss2).mean()

                # Value loss
                value_loss = 0.5 * ((new_values - mb_returns) ** 2).mean()

                # Entropy loss (to encourage exploration)
                entropy_loss = -entropy.mean()

                # Total loss
                loss = (
                    policy_loss
                    + self.vf_coef * value_loss
                    + self.ent_coef * entropy_loss
                )

                # Calculate approximate KL divergence for early stopping
                with torch.no_grad():
                    log_ratio = new_log_probs - mb_old_log_probs
                    approx_kl = ((torch.exp(log_ratio) - 1) - log_ratio).mean().item()
                    approx_kl_divs.append(approx_kl)

                # Optimization step
                self.optimizer.zero_grad()
                loss.backward()
                # Clip gradients
                clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.optimizer.step()

                mean_loss += loss.item()

            # Early stopping based on KL divergence
            if self.target_kl is not None and np.mean(approx_kl_divs) > self.target_kl:
                break

        # Calculate mean loss over all batches and epochs
        num_updates = (
            num_samples // batch_size + int(num_samples % batch_size > 0)
        ) * len(approx_kl_divs)
        mean_loss = mean_loss / max(1, num_updates)

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
            for _ in range(loop):
                obs, info = env.reset()
                scores = np.zeros(num_envs)
                completed_episode_scores = np.zeros(num_envs)
                finished = np.zeros(num_envs, dtype=bool)
                step = 0
                test_hidden_state = (
                    self.get_initial_hidden_state(num_envs) if self.recurrent else None
                )

                while not np.all(finished):
                    if swap_channels:
                        obs = obs_channels_to_first(obs)

                    action_mask = info.get("action_mask", None)
                    if self.recurrent:
                        action, _, _, _, test_hidden_state = self.get_action(
                            obs, action_mask=action_mask, hidden_state=test_hidden_state
                        )
                    else:
                        action, _, _, _ = self.get_action(obs, action_mask=action_mask)

                    obs, reward, done, trunc, info = env.step(action)
                    step += 1

                    scores += np.array(reward)

                    # Check for episode termination
                    newly_finished = (
                        np.logical_or(
                            np.logical_or(done, trunc),
                            (max_steps is not None and step == max_steps),
                        )
                        & ~finished
                    )

                    if np.any(newly_finished):
                        completed_episode_scores[newly_finished] = scores[
                            newly_finished
                        ]
                        finished[newly_finished] = True

                rewards.append(np.mean(completed_episode_scores))

        mean_fit = np.mean(rewards)
        self.fitness.append(mean_fit)
        return mean_fit
