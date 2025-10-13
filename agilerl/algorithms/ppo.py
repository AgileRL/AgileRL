import copy
import warnings
from typing import Any, Callable, Optional, Union

import numpy as np
import torch
import torch.optim as optim
from gymnasium import spaces
from tensordict import TensorDict
from torch.nn.utils import clip_grad_norm_

from agilerl.algorithms.core import OptimizerWrapper, RLAlgorithm
from agilerl.algorithms.core.registry import HyperparameterConfig, NetworkGroup
from agilerl.components.rollout_buffer import RolloutBuffer
from agilerl.modules.base import EvolvableModule
from agilerl.modules.configs import MlpNetConfig
from agilerl.networks import EvolvableNetwork, StochasticActor
from agilerl.networks.value_networks import ValueNetwork
from agilerl.typing import ArrayOrTensor, BPTTSequenceType, ExperiencesType, GymEnvType
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
    :param num_envs: Number of parallel environments, defaults to 1
    :type num_envs: int, optional
    :param use_rollout_buffer: Flag to use the rollout buffer instead of tuple experiences, defaults to False
    :type use_rollout_buffer: bool, optional
    :param recurrent: Flag to use hidden states for recurrent policies, defaults to False
    :type recurrent: bool, optional
    :param device: Device for accelerated computing, 'cpu' or 'cuda', defaults to 'cpu'
    :type device: str, optional
    :param accelerator: Accelerator for distributed computing, defaults to None
    :type accelerator: accelerate.Accelerator(), optional
    :param wrap: Wrap models for distributed training upon creation, defaults to True
    :type wrap: bool, optional
    :param bptt_sequence_type: Type of sequence for BPTT learning, defaults to BPTTSequenceType.CHUNKED
    :type bptt_sequence_type: BPTTSequenceType, optional
    :param max_seq_len: Maximum sequence length for truncated BPTT, defaults to None, where complete episodes
    are used as sequences.
    :type max_seq_len: int, optional
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        index: int = 0,
        hp_config: Optional[HyperparameterConfig] = None,
        net_config: Optional[dict[str, Any]] = None,
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
        rollout_buffer_config: Optional[dict[str, Any]] = {},
        recurrent: bool = False,
        device: str = "cpu",
        accelerator: Optional[Any] = None,
        wrap: bool = True,
        bptt_sequence_type: BPTTSequenceType = BPTTSequenceType.CHUNKED,
        max_seq_len: Optional[int] = None,
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

        # New parameters for using RolloutBuffer
        assert isinstance(
            use_rollout_buffer, bool
        ), "Use rollout buffer flag must be boolean value True or False."
        assert isinstance(
            recurrent, bool
        ), "Has hidden states flag must be boolean value True or False."
        assert isinstance(
            bptt_sequence_type, BPTTSequenceType
        ), "bptt_sequence_type must be a BPTTSequenceType enum value."

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

        self.recurrent = recurrent
        self.use_rollout_buffer = use_rollout_buffer
        self.net_config = net_config

        if self.recurrent and not self.use_rollout_buffer:
            raise ValueError("use_rollout_buffer must be True if recurrent=True.")

        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        self.lr = lr
        self.gamma = gamma
        self.learn_step = learn_step
        self.mut = mut
        self.gae_lambda = gae_lambda
        self.action_std_init = action_std_init
        self.clip_coef = clip_coef
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.target_kl = target_kl
        self.update_epochs = update_epochs
        self.num_envs = num_envs
        self.rollout_buffer_config = rollout_buffer_config
        self.bptt_sequence_type = bptt_sequence_type

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
            net_config_dict = {} if self.net_config is None else self.net_config

            critic_net_config = copy.deepcopy(net_config_dict)

            head_config = net_config_dict.get("head_config", None)
            if head_config is not None:
                critic_head_config = copy.deepcopy(head_config)
                critic_head_config["output_activation"] = None
                critic_net_config.pop("squash_output", None)
            else:
                critic_head_config = MlpNetConfig(hidden_size=[16])

            critic_net_config["head_config"] = critic_head_config

            self.actor = StochasticActor(
                self.observation_space,
                self.action_space,
                action_std_init=self.action_std_init,
                device=self.device,
                recurrent=self.recurrent,
                encoder_name=("shared_encoder" if share_encoders else "actor_encoder"),
                **net_config_dict,
            )

            self.critic = ValueNetwork(
                self.observation_space,
                device=self.device,
                recurrent=self.recurrent,
                encoder_name=("shared_encoder" if share_encoders else "critic_encoder"),
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
            self.create_rollout_buffer()
            # Need to register a mutation hook that does this after every mutation (e.g. the batch size, sequence length, etc. have changed)
            self.register_mutation_hook(self.create_rollout_buffer)

        if self.accelerator is not None and wrap:
            self.wrap_models()

        # Register network groups for mutations
        self.register_network_group(NetworkGroup(eval_network=self.actor, policy=True))
        self.register_network_group(NetworkGroup(eval_network=self.critic))

        self.hidden_state = None

    def share_encoder_parameters(self) -> None:
        """Shares the encoder parameters between the actor and critic."""
        if all(isinstance(net, EvolvableNetwork) for net in [self.actor, self.critic]):
            share_encoder_parameters(self.actor, self.critic)
        else:
            warnings.warn(
                "Encoder sharing is disabled as actor or critic is not an EvolvableNetwork."
            )

    def create_rollout_buffer(self) -> None:
        """Creates a rollout buffer with the current configuration."""
        self.rollout_buffer = RolloutBuffer(
            capacity=-(self.learn_step // -self.num_envs),
            observation_space=self.observation_space,
            action_space=self.action_space,
            device=self.device,
            num_envs=self.num_envs,
            gae_lambda=self.gae_lambda,
            gamma=self.gamma,
            recurrent=self.recurrent,
            # recurrent specific parameters
            hidden_state_architecture=(
                self.get_hidden_state_architecture() if self.recurrent else None
            ),
            max_seq_len=self.max_seq_len if self.recurrent else None,
            bptt_sequence_type=self.bptt_sequence_type,
            **self.rollout_buffer_config,
        )

    def _extract_hidden_state(
        self, full_hidden_state: dict[str, ArrayOrTensor], encoder_name: str
    ) -> dict[str, ArrayOrTensor]:
        """Extract hidden state components for a specific network encoder.

        :param full_hidden_state: Complete hidden state dictionary
        :type full_hidden_state: dict[str, ArrayOrTensor]
        :param encoder_name: Name of the encoder to extract hidden states for
        :type encoder_name: str
        :return: Hidden state dictionary for the specific encoder
        :rtype: dict[str, ArrayOrTensor]
        """
        network_hidden_state = {}
        for key, value in full_hidden_state.items():
            if key.startswith(encoder_name):
                network_hidden_state[key] = value
        return network_hidden_state

    def _get_action_and_values(
        self,
        obs: ArrayOrTensor,
        action_mask: Optional[ArrayOrTensor] = None,
        hidden_state: Optional[
            dict[str, ArrayOrTensor]
        ] = None,  # Hidden state is a dict for recurrent policies
        *,
        sample: bool = True,
    ) -> tuple[
        ArrayOrTensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Optional[dict[str, ArrayOrTensor]],
    ]:
        """
        Returns the next action to take in the environment and the values.

        :param obs: Environment observation, or multiple observations in a batch
        :type obs: ArrayOrTensor
        :param action_mask: Mask of legal actions 1=legal 0=illegal, defaults to None
        :type action_mask: Optional[ArrayOrTensor]
        :param hidden_state: Hidden state for recurrent policies, defaults to None
        :type hidden_state: Optional[dict[str, ArrayOrTensor]]
        :param sample: Whether to sample an action, defaults to True
        :type sample: bool
        :return: Action, log probability, entropy, state values, and (if recurrent) next hidden state
        :rtype: tuple[ArrayOrTensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[dict[str, ArrayOrTensor]]]
        """
        if hidden_state is not None:
            if self.share_encoders:
                # When sharing encoders, both networks use the same hidden state
                latent_pi, next_hidden_actor = self.actor.extract_features(
                    obs, hidden_state=hidden_state
                )
                action, log_prob, entropy = self.actor.forward_head(
                    latent_pi, action_mask=action_mask, sample=sample
                )
                values = self.critic.forward_head(latent_pi).squeeze(-1)
                next_hidden_combined = next_hidden_actor
            else:
                # When not sharing encoders, extract separate hidden states for actor and critic
                actor_hidden_state = self._extract_hidden_state(
                    hidden_state, "actor_encoder"
                )
                critic_hidden_state = self._extract_hidden_state(
                    hidden_state, "critic_encoder"
                )

                # Forward pass through actor with its hidden state
                latent_pi, next_hidden_actor = self.actor.extract_features(
                    obs, hidden_state=actor_hidden_state
                )
                action, log_prob, entropy = self.actor.forward_head(
                    latent_pi, action_mask=action_mask, sample=sample
                )

                # Forward pass through critic with its hidden state
                values, next_hidden_critic = self.critic(
                    obs, hidden_state=critic_hidden_state
                )
                values = values.squeeze(-1)

                # Combine the next hidden states from both networks
                next_hidden_combined = {}
                if next_hidden_actor is not None:
                    next_hidden_combined.update(next_hidden_actor)
                if next_hidden_critic is not None:
                    next_hidden_combined.update(next_hidden_critic)

            return action, log_prob, entropy, values, next_hidden_combined
        else:
            latent_pi = self.actor.extract_features(obs)
            action, log_prob, entropy = self.actor.forward_head(
                latent_pi, action_mask=action_mask, sample=sample
            )
            values = (
                self.critic.forward_head(latent_pi).squeeze(-1)
                if self.share_encoders
                else self.critic(obs).squeeze(-1)
            )
            return action, log_prob, entropy, values, None

    def get_hidden_state_architecture(self) -> dict[str, tuple[int, ...]]:
        """Get the hidden state architecture for the environment.

        :return: Dictionary describing the hidden state architecture (name to shape)
        :rtype: dict[str, tuple[int, ...]]
        """
        return {
            k: v.shape for k, v in self.get_initial_hidden_state(self.num_envs).items()
        }

    def get_initial_hidden_state(self, num_envs: int = 1) -> dict[str, ArrayOrTensor]:
        """Get the initial hidden state for the environment.

        The hidden states are generally cached on a per Module basis.
        The reason the Cache is per Module is because the user might want to have a custom initialization for the hidden states.

        :param num_envs: Number of environments, defaults to 1
        :type num_envs: int, optional
        :return: Initial hidden state dictionary
        :rtype: dict[str, ArrayOrTensor]
        """
        # Return a batch of initial hidden states
        # Flat map them into "actor_*" and "critic_*" (if not sharing encoders)
        flat_hidden = {}

        actor_hidden = self.actor.initialize_hidden_state(batch_size=num_envs)
        flat_hidden.update(actor_hidden)

        # also add the critic hidden state if not sharing encoders
        if not self.share_encoders:
            critic_hidden = self.critic.initialize_hidden_state(batch_size=num_envs)
            flat_hidden.update(critic_hidden)

        return flat_hidden

    def evaluate_actions(
        self,
        obs: ArrayOrTensor,
        actions: ArrayOrTensor,
        hidden_state: Optional[dict[str, ArrayOrTensor]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluates the actions.

        :param obs: Environment observation, or multiple observations in a batch
        :type obs: ArrayOrTensor
        :param actions: Actions to evaluate
        :type actions: ArrayOrTensor
        :param hidden_state: Hidden state for recurrent policies, defaults to None. Expected shape: dict with tensors of shape (batch_size, 1, hidden_size).
        :type hidden_state: Optional[dict[str, ArrayOrTensor]]
        :return: Log probability, entropy, state values
        :rtype: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        """
        obs = self.preprocess_observation(obs)

        # Get values from actor-critic
        _, _, entropy, values, _ = self._get_action_and_values(
            obs, hidden_state=hidden_state, sample=False
        )

        log_prob = self.actor.action_log_prob(actions)

        # Use -log_prob as entropy when squashing output in continuous action spaces
        if entropy is None:
            entropy = -log_prob.mean()

        return log_prob, entropy, values

    def get_action(
        self,
        obs: ArrayOrTensor,
        action_mask: Optional[ArrayOrTensor] = None,
        hidden_state: Optional[dict[str, ArrayOrTensor]] = None,
    ) -> Union[
        tuple[
            np.ndarray,  # action
            np.ndarray,  # log_prob
            np.ndarray,  # entropy
            np.ndarray,  # values
            Optional[dict[str, ArrayOrTensor]],  # next_hidden_state
        ],
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],  # non-recurrent case
    ]:
        """Returns the next action to take in the environment.

        :param obs: Environment observation, or multiple observations in a batch
        :type obs: ArrayOrTensor
        :param action_mask: Mask of legal actions 1=legal 0=illegal, defaults to None
        :type action_mask: Optional[ArrayOrTensor]
        :param hidden_state: Hidden state for recurrent policies, defaults to None
        :type hidden_state: Optional[dict[str, ArrayOrTensor]]
        :return: Action, log probability, entropy, state values, and (if recurrent) next hidden state
        :rtype: Union[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[dict[str, ArrayOrTensor]]], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]
        """
        obs = self.preprocess_observation(obs)
        with torch.no_grad():
            (
                action,
                log_prob,
                entropy,
                values,
                next_hidden,
            ) = self._get_action_and_values(
                obs,
                action_mask,
                hidden_state,
                sample=True,  # Explicitly sample=True during get_action
            )

        # Use -log_prob as entropy when squashing output in continuous action spaces
        entropy = -log_prob.mean() if entropy is None else entropy

        # Clip to action space during inference
        action_np = action.cpu().data.numpy()
        if not self.training and isinstance(self.action_space, spaces.Box):
            if self.actor.squash_output:
                action_np = self.actor.scale_action(action_np)
            else:
                action_np = np.clip(
                    action_np, self.action_space.low, self.action_space.high
                )

        log_prob_np = log_prob.cpu().data.numpy()
        entropy_np = entropy.cpu().data.numpy()
        values_np = values.cpu().data.numpy()

        if self.recurrent:
            return (
                action_np,
                log_prob_np,
                entropy_np,
                values_np,
                next_hidden if next_hidden is not None else None,
            )
        else:
            return (
                action_np,
                log_prob_np,
                entropy_np,
                values_np,
            )

    def learn(self, experiences: Optional[ExperiencesType] = None) -> float:
        """Updates agent network parameters to learn from experiences.

        :param experiences: Tuple of batched states, actions, log_probs, rewards, dones, values, next_state, next_done.
                            If use_rollout_buffer=True and experiences=None, uses data from rollout buffer.
        :type experiences: Optional[ExperiencesType]
        :return: Mean loss value from training.
        :rtype: float
        """
        if self.use_rollout_buffer:
            # NOTE: we are still allowing experiences to be passed in for backwards compatibility
            # but we will remove this in a future releases.
            # i.e. it's possible to do one learn with rollouts, then another with experiences on the same agent
            if experiences is None:
                # Learn from the internal rollout buffer
                if self.recurrent:
                    return self._learn_from_rollout_buffer_bptt()
                else:
                    return self._learn_from_rollout_buffer_flat()

        return self._deprecated_learn_from_experiences(experiences)

    def _deprecated_learn_from_experiences(self, experiences: ExperiencesType) -> float:
        """Deprecated method for learning from experiences tuple format.

        This method is deprecated and will be removed in a future release. The PPO implementation
        now uses a rollout buffer for improved performance, cleaner support for recurrent policies,
        and easier integration with custom environments.

        To migrate:
        1. Set use_rollout_buffer=True when creating PPO agent
        2. If using recurrent policies, set recurrent=True
        3. Use collect_rollouts() to gather experiences instead of passing experiences tuple
        4. Call learn() without arguments to train on collected rollouts
        """
        if not experiences:
            raise ValueError(
                "Experiences must be provided when use_rollout_buffer is False"
            )

        # Not self.use_rollout_buffer
        (
            observations,
            actions,
            log_probs,
            rewards,
            dones,
            values,
            next_obs,
            next_done,
        ) = stack_experiences(*experiences)

        # Bootstrapping returns using GAE advantage estimation
        dones = dones.long()
        with torch.no_grad():
            num_steps = rewards.size(0)
            next_obs = self.preprocess_observation(next_obs)
            next_value = self.critic(next_obs).reshape(1, -1).cpu()
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
        experiences = (observations, actions, log_probs, advantages, returns, values)
        if is_vectorized_experiences(*experiences):
            experiences = flatten_experiences(*experiences)

        # Move experiences to algo device
        experiences = self.to_device(*experiences)

        # Get number of samples from the returns tensor
        num_samples = experiences[4].size(0)
        batch_idxs = np.arange(num_samples)
        mean_loss = 0
        for _ in range(self.update_epochs):
            np.random.shuffle(batch_idxs)
            for start in range(0, num_samples, self.batch_size):
                minibatch_idxs = batch_idxs[start : start + self.batch_size]
                (
                    batch_observations,
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
                        obs=batch_observations, actions=batch_actions, hidden_state=None
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

    def _learn_from_rollout_buffer_flat(
        self, buffer_td_external: Optional[TensorDict] = None
    ) -> float:
        """Learning procedure using flattened samples (no BPTT)."""
        if buffer_td_external is not None:
            buffer_td = buffer_td_external
        else:
            # .get_tensor_batch() returns a TensorDict on the specified device
            buffer_td = self.rollout_buffer.get_tensor_batch(device=self.device)

        if buffer_td.is_empty():
            warnings.warn("Buffer data is empty. Skipping learning step.")
            return 0.0

        # Normalize advantages globally
        valid_advantages: torch.Tensor = buffer_td.get("advantages")
        normalized_advantages = (valid_advantages - valid_advantages.mean()) / (
            valid_advantages.std() + 1e-8
        )
        buffer_td["advantages"] = normalized_advantages

        batch_size = self.batch_size
        num_samples = self.rollout_buffer.size()
        indices = np.arange(num_samples)
        mean_loss = 0.0
        approx_kl_divs = []
        for _ in range(self.update_epochs):
            np.random.shuffle(indices)
            for start_idx in range(0, num_samples, batch_size):
                end_idx = min(start_idx + batch_size, num_samples)
                minibatch_indices = indices[start_idx:end_idx]

                # Slice the TensorDict to get the minibatch
                minibatch_td = buffer_td[minibatch_indices]

                mb_obs = minibatch_td["observations"]
                mb_actions = minibatch_td["actions"]
                mb_log_probs = minibatch_td["log_probs"]
                mb_advantages = minibatch_td["advantages"]
                mb_returns = minibatch_td["returns"]
                mb_old_values = minibatch_td["values"]

                if isinstance(self.action_space, spaces.Discrete):
                    mb_actions = mb_actions.squeeze(-1)

                log_probs, entropy, values = self.evaluate_actions(
                    obs=mb_obs, actions=mb_actions, hidden_state=None
                )

                # Policy loss
                ratio = torch.exp(log_probs - mb_log_probs)
                policy_loss1 = -mb_advantages * ratio
                policy_loss2 = -mb_advantages * torch.clamp(
                    ratio, 1 - self.clip_coef, 1 + self.clip_coef
                )
                policy_loss = torch.max(policy_loss1, policy_loss2).mean()

                # Value loss
                value = values.view(-1)
                v_loss_unclipped = (value - mb_returns) ** 2
                v_clipped = mb_old_values + torch.clamp(
                    value - mb_old_values, -self.clip_coef, self.clip_coef
                )

                v_loss_clipped = (v_clipped - mb_returns) ** 2
                v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                v_loss = 0.5 * v_loss_max.mean()

                # Entropy loss
                entropy_loss = -entropy.mean()

                # Total loss
                loss = (
                    policy_loss + self.vf_coef * v_loss + self.ent_coef * entropy_loss
                )

                with torch.no_grad():
                    log_ratio = log_probs - mb_log_probs
                    approx_kl = ((ratio - 1) - log_ratio).mean().item()
                    approx_kl_divs.append(approx_kl)

                self.optimizer.zero_grad()
                if self.accelerator is not None:
                    self.accelerator.backward(loss)
                else:
                    loss.backward()

                clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)

                self.optimizer.step()

                mean_loss += loss.item()

            if self.target_kl is not None and np.mean(approx_kl_divs) > self.target_kl:
                break  # Early stopping for the epoch if KL divergence target is exceeded

        mean_loss /= num_samples * self.update_epochs
        return mean_loss

    def _learn_from_rollout_buffer_bptt(self) -> float:
        """Learning procedure using truncated BPTT for recurrent networks.

        :return: Mean loss over the epochs
        :rtype: float
        """
        buffer_size = (
            self.rollout_buffer.capacity
            if self.rollout_buffer.full
            else self.rollout_buffer.pos
        )

        # Normalize advantages globally
        valid_advantages: torch.Tensor = self.rollout_buffer.buffer.get("advantages")[
            :buffer_size
        ]
        original_shape = valid_advantages.shape
        flat_adv = valid_advantages.reshape(-1)
        normalized_flat_adv = (flat_adv - flat_adv.mean()) / (flat_adv.std() + 1e-8)
        self.rollout_buffer.buffer["advantages"][:buffer_size] = (
            normalized_flat_adv.reshape(original_shape)
        )

        # Form padded sequences to perform BPTT on
        self.rollout_buffer.prepare_sequence_tensors(device=self.device)

        # Here, batch_size means number of sequences per minibatch
        mean_loss = 0.0
        total_minibatch_updates_total = 0
        for epoch in range(self.update_epochs):
            approx_kl_divs_epoch = []  # KL divergences for this epoch's minibatches
            num_minibatches_this_epoch = 0

            # Itreate over minibatches of sequences
            minibatch_gen = self.rollout_buffer.get_minibatch_sequences(
                batch_size=self.batch_size,
            )
            for minibatch_padded, minibatch_unpadded in minibatch_gen:
                # Obs shape: (batch_seq * seq_len, *obs_dims) or nested TD
                # Actions shape: (batch_seq * seq_len, *act_dims)
                # Other tensors shape: (batch_seq * seq_len, )
                mb_obs_seq = minibatch_padded["observations"]
                mb_actions_seq = minibatch_padded["actions"]
                mb_pad_mask = minibatch_padded["pad_mask"]
                mb_old_log_probs = minibatch_unpadded["log_probs"]
                mb_advantages = minibatch_unpadded["advantages"]
                mb_values = minibatch_unpadded["values"]
                mb_returns = minibatch_unpadded["returns"]
                mb_initial_hidden_states_dict: dict[str, torch.Tensor] = (
                    minibatch_padded.get_non_tensor(
                        "initial_hidden_states", default=None
                    )
                )

                approx_kl_divs_minibatch_timesteps = []

                # For actor: {key: (layers, batch_seq_size, hidden_size)}
                if self.recurrent and mb_initial_hidden_states_dict is not None:
                    mb_initial_hidden_states_dict = {
                        # val is (batch_seq_size, layers, size), permute to (layers, batch_seq_size, size)
                        key: val.permute(1, 0, 2).contiguous().to(self.device)
                        for key, val in mb_initial_hidden_states_dict.items()
                    }

                # Need to flatten action dimension for Discrete action spaces
                if isinstance(self.action_space, spaces.Discrete):
                    mb_actions_seq = mb_actions_seq.squeeze(-1)

                # new_value: (batch_seq,),
                # entropy: (batch_seq,) or scalar,
                # log_prob: (batch_seq,)
                (
                    new_log_probs,
                    entropy,
                    new_values,
                ) = self.evaluate_actions(
                    obs=mb_obs_seq,
                    actions=mb_actions_seq,
                    hidden_state=mb_initial_hidden_states_dict,
                )

                # Mask out padded values
                new_values = new_values[mb_pad_mask]
                new_log_probs = new_log_probs[mb_pad_mask]
                entropy = entropy[mb_pad_mask]

                if isinstance(entropy, torch.Tensor):
                    entropy = entropy.mean()

                # Policy loss
                ratio = torch.exp(new_log_probs - mb_old_log_probs)
                policy_loss1 = -mb_advantages * ratio
                policy_loss2 = -mb_advantages * torch.clamp(
                    ratio, 1 - self.clip_coef, 1 + self.clip_coef
                )
                policy_loss = torch.max(policy_loss1, policy_loss2).mean()

                # Value loss
                v_loss_unclipped = (new_values - mb_returns) ** 2
                v_clipped = mb_values + torch.clamp(
                    new_values - mb_values, -self.clip_coef, self.clip_coef
                )

                v_loss_clipped = (v_clipped - mb_returns) ** 2
                v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                value_loss = v_loss_max.mean()

                # Entropy loss
                entropy_loss = -entropy

                with torch.no_grad():
                    log_ratio = new_log_probs - mb_old_log_probs
                    approx_kl_divs_minibatch_timesteps.append(
                        ((torch.exp(log_ratio) - 1) - log_ratio).mean().item()
                    )

                loss = (
                    policy_loss
                    + self.vf_coef * value_loss
                    + self.ent_coef * entropy_loss
                )

                self.optimizer.zero_grad()
                if self.accelerator is not None:
                    self.accelerator.backward(loss)
                else:
                    loss.backward()

                clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.optimizer.step()

                mean_loss += loss.item()
                num_minibatches_this_epoch += 1

                if (
                    self.target_kl is not None
                    and len(approx_kl_divs_minibatch_timesteps) > 0
                ):
                    # Average KL over all timesteps in this minibatch of sequences
                    kl_for_current_minibatch = np.mean(
                        approx_kl_divs_minibatch_timesteps
                    )
                    approx_kl_divs_epoch.append(
                        kl_for_current_minibatch
                    )  # Store minibatch average KL

                    if kl_for_current_minibatch > self.target_kl:
                        warnings.warn(
                            f"Epoch {epoch}, Minibatch: KL divergence {kl_for_current_minibatch:.4f} exceeded target {self.target_kl}. Stopping update for this epoch."
                        )
                        break  # Break from minibatch loop for this epoch

            total_minibatch_updates_total += num_minibatches_this_epoch
            # Check average KL for the epoch if target_kl is set and the inner loop wasn't broken by KL
            if self.target_kl is not None and len(approx_kl_divs_epoch) > 0:
                avg_kl_this_epoch = np.mean(approx_kl_divs_epoch)
                if (
                    avg_kl_this_epoch > self.target_kl
                    and not (  # Ensure this wasn't the break from inner loop
                        len(approx_kl_divs_minibatch_timesteps) > 0
                        and np.mean(approx_kl_divs_minibatch_timesteps) > self.target_kl
                    )
                ):
                    warnings.warn(
                        f"Epoch {epoch}: Average KL divergence {avg_kl_this_epoch:.4f} exceeded target {self.target_kl} after completing epoch. Consider adjusting learning rate or target_kl."
                    )
                    # This break is for the epoch loop if KL was exceeded on average for the epoch
                    # but not necessarily in the last minibatch that would have broken the inner loop.
                    break

            # If inner loop broke due to KL, this outer break also executes
            if (
                self.target_kl is not None
                and len(approx_kl_divs_minibatch_timesteps) > 0
                and np.mean(approx_kl_divs_minibatch_timesteps) > self.target_kl
            ):
                break

        mean_loss = mean_loss / max(1e-8, total_minibatch_updates_total)
        return mean_loss

    def test(
        self,
        env: GymEnvType,
        swap_channels: bool = False,
        max_steps: Optional[int] = None,
        loop: int = 3,
        vectorized: bool = True,
        callback: Optional[Callable[[float, dict[str, float]], None]] = None,
    ) -> float:
        """Returns mean test score of agent in environment with epsilon-greedy policy.

        :param env: The environment to be tested in
        :type env: GymEnvType
        :param swap_channels: Swap image channels dimension from last to first [H, W, C] -> [C, H, W], defaults to False
        :type swap_channels: bool, optional
        :param max_steps: Maximum number of testing steps, defaults to None
        :type max_steps: int, optional
        :param loop: Number of testing loops/episodes to complete. The returned score is the mean. Defaults to 3
        :type loop: int, optional
        :param vectorized: Whether the environment is vectorized, defaults to True
        :type vectorized: bool, optional
        :param callback: Optional callback function that takes the sum of rewards and the last info dictionary as input, defaults to None
        :type callback: Optional[Callable[[float, dict[str, float]], None]]

        :return: Mean test score of agent in environment
        :rtype: float
        """
        # set to evaluation mode. This is important for batch norm and dropout layers
        self.actor.eval()
        self.critic.eval()
        self.set_training_mode(False)

        with torch.no_grad():
            rewards = []
            num_envs = env.num_envs if hasattr(env, "num_envs") and vectorized else 1

            for _ in range(loop):
                obs, info = env.reset()
                scores = np.zeros(num_envs)
                completed_episode_scores = np.zeros(num_envs)
                finished = np.zeros(num_envs, dtype=bool)
                step = 0
                test_hidden_state = (
                    self.get_initial_hidden_state(num_envs) if self.recurrent else None
                )

                last_infos = (
                    [{}] * num_envs if vectorized else {}
                )  # Initialize last_info holder

                while not np.all(finished):
                    if swap_channels:
                        obs = obs_channels_to_first(obs)

                    # Process action mask
                    action_mask = None
                    if vectorized:
                        # Check if info is a list/array of dicts
                        if (
                            isinstance(info, (list, np.ndarray))
                            and len(info) == num_envs
                            and all(isinstance(i, dict) for i in info)
                        ):
                            masks = [env_info.get("action_mask") for env_info in info]
                            # If all environments returned a mask and they are not None
                            if all(m is not None for m in masks):
                                try:
                                    action_mask = np.stack(masks)
                                except Exception as e:
                                    warnings.warn(f"Could not stack action masks: {e}")
                                    action_mask = None
                            # If only some environments returned masks, we probably can't use them reliably
                            elif any(m is not None for m in masks):
                                warnings.warn(
                                    "Action masks not provided for all vectorized environments. Skipping mask."
                                )
                                action_mask = None
                        # Handle case where info might be a single dict even if vectorized (e.g. VecNormalize)
                        elif isinstance(info, dict):
                            action_mask = info.get("action_mask", None)

                    else:  # Not vectorized
                        if isinstance(info, dict):
                            action_mask = info.get("action_mask", None)

                    # Get action
                    if self.recurrent:
                        action, _, _, _, test_hidden_state = self.get_action(
                            obs, action_mask=action_mask, hidden_state=test_hidden_state
                        )
                    else:
                        action, _, _, _ = self.get_action(obs, action_mask=action_mask)

                    # Environment step
                    if vectorized:
                        obs, reward, done, trunc, info = env.step(action)
                        last_infos = info  # Store the array of infos
                    else:
                        obs, reward, done, trunc, info_single = env.step(action[0])
                        # Store info in a dictionary for consistency if not vectorized
                        info = {"final_info": info_single} if done or trunc else {}
                        last_infos = info  # Store the single info dict

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

                    # Reset hidden state for newly finished environments
                    if self.recurrent and np.any(newly_finished):
                        initial_hidden_states_for_reset = self.get_initial_hidden_state(
                            num_envs
                        )
                        if isinstance(test_hidden_state, dict):
                            for key in test_hidden_state:
                                reset_states = initial_hidden_states_for_reset[key][
                                    :, newly_finished, :
                                ]
                                if reset_states.shape[1] > 0:
                                    test_hidden_state[key][
                                        :, newly_finished, :
                                    ] = reset_states

                    if np.any(newly_finished):
                        completed_episode_scores[newly_finished] = scores[
                            newly_finished
                        ]
                        finished[newly_finished] = True

                # End of episode loop for one test run
                loop_reward_sum = np.sum(completed_episode_scores)

                # Prepare info for callback
                final_info_for_callback = {}
                if vectorized:
                    if (
                        isinstance(last_infos, (list, np.ndarray))
                        and len(last_infos) > 0
                    ):
                        final_info_for_callback = (
                            last_infos[0] if isinstance(last_infos[0], dict) else {}
                        )
                    elif isinstance(last_infos, dict):
                        final_info_for_callback = last_infos
                else:  # Not vectorized
                    if isinstance(last_infos, dict):
                        final_info_for_callback = last_infos

                if callback is not None:
                    callback(loop_reward_sum, final_info_for_callback)

                rewards.append(np.mean(completed_episode_scores))

        mean_fit = np.mean(rewards)
        self.fitness.append(mean_fit)

        # cleanup evaluation mode back into the default training mode (e.g. batch norm and dropout layers)
        self.set_training_mode(True)
        self.actor.train()
        self.critic.train()

        return mean_fit
