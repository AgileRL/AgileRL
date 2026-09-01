# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

import copy
import warnings
from collections.abc import Callable
from typing import Any, overload

import gymnasium as gym
import numpy as np
import numpy.typing as npt
import torch
from accelerate import Accelerator
from gymnasium import spaces
from tensordict import TensorDict
from torch import optim
from torch.nn.utils import clip_grad_norm_

from agilerl.algorithms.core import OptimizerWrapper, SingleAgentAlgorithm
from agilerl.algorithms.core.registry import (
    HyperparameterConfig,
    NetworkGroup,
    make_default_hp_config,
)
from agilerl.components.rollout_buffer import RolloutBuffer
from agilerl.modules.base import EvolvableModule
from agilerl.modules.configs import MlpNetConfig
from agilerl.networks import EvolvableNetwork, StochasticActor
from agilerl.networks.value_networks import ValueNetwork
from agilerl.typing import (
    ActionMaskInput,
    BPTTSequenceType,
    ObservationType,
    RolloutMinibatch,
    RolloutSequenceMinibatch,
    RolloutSequenceTargets,
    SupportedObservationSpace,
    TorchObsType,
)
from agilerl.utils.algo_utils import (
    get_num_envs,
    make_safe_deepcopies,
    share_encoder_parameters,
)

ActionReturnType = tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]
RecurrentActionReturnType = tuple[
    npt.NDArray,
    npt.NDArray,
    npt.NDArray,
    npt.NDArray,
    dict[str, torch.Tensor] | None,
]


class PPO(SingleAgentAlgorithm[TensorDict]):
    """Proximal Policy Optimization (PPO).

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
    :param rollout_buffer_config: Extra keyword arguments forwarded to the
        rollout buffer constructor, defaults to None (treated as an empty dict).
    :type rollout_buffer_config: dict[str, Any] | None, optional
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
    :param max_seq_len: Maximum sequence length for truncated BPTT, defaults to None, where complete episodes are used as sequences.
    :type max_seq_len: int, optional
    """

    # Custom networks must satisfy the StochasticActor/ValueNetwork interface
    # (extract_features/forward_head/action_log_prob/...), which PPO drives
    # unconditionally.
    actor: StochasticActor
    critic: ValueNetwork

    def __init__(
        self,
        observation_space: SupportedObservationSpace,
        action_space: spaces.Space,
        index: int = 0,
        hp_config: HyperparameterConfig | None = None,
        net_config: dict[str, Any] | None = None,
        batch_size: int = 64,
        lr: float = 1e-4,
        learn_step: int = 2048,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        mut: str | None = None,
        action_std_init: float = 0.0,
        clip_coef: float = 0.2,
        ent_coef: float = 0.01,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        target_kl: float | None = None,
        normalize_images: bool = True,
        update_epochs: int = 4,
        actor_network: EvolvableModule | None = None,
        critic_network: EvolvableModule | None = None,
        share_encoders: bool = True,
        num_envs: int = 1,
        rollout_buffer_config: dict[str, Any] | None = None,
        recurrent: bool = False,
        device: str = "cpu",
        accelerator: Accelerator | None = None,
        wrap: bool = True,
        bptt_sequence_type: str | BPTTSequenceType = BPTTSequenceType.CHUNKED,
        max_seq_len: int | None = None,
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
            action_std_init,
            (float, int),
        ), "Action standard deviation must be a float."
        assert action_std_init >= 0, (
            "Action standard deviation must be greater than or equal to zero."
        )
        assert isinstance(
            clip_coef,
            (float, int),
        ), "Clipping coefficient must be a float."
        assert clip_coef >= 0, (
            "Clipping coefficient must be greater than or equal to zero."
        )
        assert isinstance(
            ent_coef,
            (float, int),
        ), "Entropy coefficient must be a float."
        assert ent_coef >= 0, (
            "Entropy coefficient must be greater than or equal to zero."
        )
        assert isinstance(
            vf_coef,
            (float, int),
        ), "Value function coefficient must be a float."
        assert vf_coef >= 0, (
            "Value function coefficient must be greater than or equal to zero."
        )
        assert isinstance(
            max_grad_norm,
            (float, int),
        ), "Maximum norm for gradient clipping must be a float."
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
        assert isinstance(
            update_epochs,
            int,
        ), "Policy update epochs must be an integer."
        assert update_epochs >= 1, (
            "Policy update epochs must be greater than or equal to one."
        )
        assert isinstance(
            wrap,
            bool,
        ), "Wrap models flag must be boolean value True or False."

        assert isinstance(
            recurrent,
            bool,
        ), "Has hidden states flag must be boolean value True or False."
        if isinstance(bptt_sequence_type, str):
            bptt_sequence_type = BPTTSequenceType(bptt_sequence_type)

        self.recurrent = recurrent
        self.net_config = net_config
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
        self.rollout_buffer_config = rollout_buffer_config or {}
        self.bptt_sequence_type = bptt_sequence_type

        # Default RL hyperparameters to mutate when doing Evo-HPO
        self.hp_config = self.hp_config or make_default_hp_config(
            lr=self.lr,
            batch_size=self.batch_size,
            learn_step=self.learn_step,
        )

        if actor_network is not None and critic_network is not None:
            # Custom networks must satisfy the StochasticActor/ValueNetwork interface
            actor = self._as_stochastic_actor(actor_network)
            critic = self._as_value_network(critic_network)

            # Two independent user-supplied networks are distinct feature
            # extractors, so they cannot share an encoder.
            if not (
                isinstance(actor_network, StochasticActor)
                and isinstance(critic_network, ValueNetwork)
            ):
                share_encoders = False

            self.actor, self.critic = make_safe_deepcopies(actor, critic)
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
            optim.Adam,
            networks=[self.actor, self.critic],
            lr=self.lr,
        )

        # Initialize rollout buffer to store experiences for learning
        # NOTE: Need to register a mutation hook that does this after every mutation
        # (e.g. the batch size, sequence length, etc. have changed)
        # TODO: Try implementing a way to register mutation hooks that applies only after
        # certain attributes have been mutated!
        self.create_rollout_buffer()
        self.register_mutation_hook(self.create_rollout_buffer)

        if self.accelerator is not None and wrap:
            self.wrap_models()

        # Register network groups for mutations
        self.register_network_group(NetworkGroup(eval_network=self.actor, policy=True))
        self.register_network_group(NetworkGroup(eval_network=self.critic))

        self.hidden_state = None

        # Register metrics to keep track of during training
        for metric in ("loss", "policy_loss", "value_loss", "entropy_loss"):
            self.metrics.register(metric)

    def _as_stochastic_actor(self, network: EvolvableModule) -> StochasticActor:
        """Return *network* as a :class:`StochasticActor`.

        :param network: Custom actor network.
        :type network: EvolvableModule
        :return: A stochastic actor driving *network*.
        :rtype: StochasticActor
        """
        if isinstance(network, StochasticActor):
            return network
        return StochasticActor(
            self.observation_space,
            self.action_space,
            encoder=network,
            action_std_init=self.action_std_init,
            device=self.device,
            recurrent=self.recurrent,
        )

    def _as_value_network(self, network: EvolvableModule) -> ValueNetwork:
        """Return *network* as a :class:`ValueNetwork`.

        :param network: Custom critic network.
        :type network: EvolvableModule
        :return: A value network driving *network*.
        :rtype: ValueNetwork
        """
        if isinstance(network, ValueNetwork):
            return network
        return ValueNetwork(
            self.observation_space,
            encoder=network,
            device=self.device,
            recurrent=self.recurrent,
        )

    def share_encoder_parameters(self) -> None:
        """Shares the encoder parameters between the actor and critic."""
        if isinstance(self.actor, EvolvableNetwork) and isinstance(
            self.critic,
            EvolvableNetwork,
        ):
            share_encoder_parameters(self.actor, self.critic)
        else:
            warnings.warn(
                "Encoder sharing is disabled as actor or critic is not an EvolvableNetwork.",
                stacklevel=2,
            )

    def create_rollout_buffer(self) -> None:
        """Create a rollout buffer with the current configuration."""
        self.rollout_buffer = RolloutBuffer(
            capacity=-(self.learn_step // -self.num_envs),
            observation_space=self.env_observation_space,
            action_space=self.action_space,
            device=str(self.device),
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
        self,
        full_hidden_state: dict[str, torch.Tensor],
        encoder_name: str,
    ) -> dict[str, torch.Tensor]:
        """Extract hidden state components for a specific network encoder.

        :param full_hidden_state: Complete hidden state dictionary
        :type full_hidden_state: dict[str, torch.Tensor]
        :param encoder_name: Name of the encoder to extract hidden states for
        :type encoder_name: str
        :return: Hidden state dictionary for the specific encoder
        :rtype: dict[str, torch.Tensor]
        """
        return {
            key: value
            for key, value in full_hidden_state.items()
            if key.startswith(encoder_name)
        }

    def _get_action_and_values(
        self,
        obs: TorchObsType,
        action_mask: ActionMaskInput = None,
        hidden_state: (
            dict[str, torch.Tensor] | None
        ) = None,  # Hidden state is a dict for recurrent policies
        *,
        sample: bool = True,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        dict[str, torch.Tensor] | None,
    ]:
        """Return the next action to take in the environment and the values.

        :param obs: Environment observation, or multiple observations in a batch
        :type obs: TorchObsType
        :param action_mask: Mask of legal actions 1=legal 0=illegal, defaults to None
        :type action_mask: ActionMaskInput
        :param hidden_state: Hidden state for recurrent policies, defaults to None
        :type hidden_state: dict[str, torch.Tensor] | None
        :param sample: Whether to sample an action, defaults to True
        :type sample: bool
        :return: Action, log probability, entropy, state values, and (if recurrent) next hidden state
        :rtype: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor] | None]
        """
        if hidden_state is not None:
            if self.share_encoders:
                # When sharing encoders, both networks use the same hidden state.
                latent_pi, next_hidden_actor = self.actor.extract_features(
                    obs, hidden_state=hidden_state
                )
                action, log_prob, entropy = self.actor.forward_head(
                    latent_pi,
                    action_mask=action_mask,
                    sample=sample,
                )
                values = self.critic.forward_head(latent_pi).squeeze(-1)
                next_hidden_combined = next_hidden_actor
            else:
                # When not sharing encoders, extract separate hidden states for actor and critic
                actor_hidden_state = self._extract_hidden_state(
                    hidden_state,
                    "actor_encoder",
                )
                critic_hidden_state = self._extract_hidden_state(
                    hidden_state,
                    "critic_encoder",
                )

                # Forward pass through actor with its hidden state.
                latent_pi, next_hidden_actor = self.actor.extract_features(
                    obs, hidden_state=actor_hidden_state
                )
                action, log_prob, entropy = self.actor.forward_head(
                    latent_pi,
                    action_mask=action_mask,
                    sample=sample,
                )

                # Forward pass through critic with its hidden state.
                values, next_hidden_critic = self.critic(
                    obs, hidden_state=critic_hidden_state
                )
                values = values.squeeze(-1)

                # Combine the next hidden states from both networks
                next_hidden_combined: dict[str, torch.Tensor] = {}
                if next_hidden_actor is not None:
                    next_hidden_combined.update(next_hidden_actor)
                if next_hidden_critic is not None:
                    next_hidden_combined.update(next_hidden_critic)

            return action, log_prob, entropy, values, next_hidden_combined

        latent_pi = self.actor.extract_features(obs)
        action, log_prob, entropy = self.actor.forward_head(
            latent_pi,
            action_mask=action_mask,
            sample=sample,
        )
        values = (
            self.critic.forward_head(latent_pi).squeeze(-1)
            if self.share_encoders
            else self.critic(obs).squeeze(-1)
        )
        return action, log_prob, entropy, values, None

    def get_hidden_state_architecture(self) -> dict[str, tuple[int, ...]]:
        """Get the hidden state architecture for the environment.

        :return: Dictionary describing the hidden state architecture (name to
            ``(num_layers, num_envs, hidden_size)`` shape)
        :rtype: dict[str, tuple[int, ...]]
        """
        # Recurrent hidden states are always (num_layers, batch, hidden_size).
        return {
            k: tuple(v.shape)
            for k, v in self.get_initial_hidden_state(self.num_envs).items()
        }

    def get_initial_hidden_state(self, num_envs: int = 1) -> dict[str, torch.Tensor]:
        """Get the initial hidden state for the environment.

        The hidden states are generally cached on a per Module basis.
        The reason the Cache is per Module is because the user might want to have a custom initialization for the hidden states.

        :param num_envs: Number of environments, defaults to 1
        :type num_envs: int, optional
        :return: Initial hidden state dictionary
        :rtype: dict[str, torch.Tensor]
        """
        # Return a batch of initial hidden states
        # Flat map them into "actor_*" and "critic_*" (if not sharing encoders)
        flat_hidden: dict[str, torch.Tensor] = {}

        actor_hidden = self.actor.initialize_hidden_state(batch_size=num_envs)
        flat_hidden.update(actor_hidden)

        # also add the critic hidden state if not sharing encoders
        if not self.share_encoders:
            critic_hidden = self.critic.initialize_hidden_state(batch_size=num_envs)
            flat_hidden.update(critic_hidden)

        return flat_hidden

    def evaluate_actions(
        self,
        obs: ObservationType,
        actions: torch.Tensor,
        hidden_state: dict[str, torch.Tensor] | None = None,
        action_mask: ActionMaskInput = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate the actions.

        :param obs: Environment observation, or multiple observations in a batch
        :type obs: ObservationType
        :param actions: Actions to evaluate
        :type actions: torch.Tensor
        :param hidden_state: Hidden state for recurrent policies, defaults to None. Expected shape: dict with tensors of shape (batch_size, 1, hidden_size).
        :type hidden_state: dict[str, torch.Tensor] | None
        :param action_mask: Mask of legal actions 1=legal 0=illegal, defaults to None
        :type action_mask: ActionMaskInput
        :return: Log probability, entropy, state values
        :rtype: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        """
        preprocessed_obs = self.preprocess_observation(obs)

        # Get values from actor-critic
        _, _, entropy, values, _ = self._get_action_and_values(
            preprocessed_obs,
            action_mask=action_mask,
            hidden_state=hidden_state,
            sample=False,
        )

        log_prob = self.actor.action_log_prob(actions)

        # Use -log_prob as entropy when squashing output in continuous action spaces
        if entropy is None:
            entropy = -log_prob.mean()

        return log_prob, entropy, values

    @overload
    def get_action(
        self,
        obs: ObservationType,
        action_mask: ActionMaskInput = None,
        *,
        hidden_state: dict[str, torch.Tensor],
        **kwargs: Any,
    ) -> RecurrentActionReturnType: ...

    @overload
    def get_action(
        self,
        obs: ObservationType,
        action_mask: ActionMaskInput = None,
        hidden_state: None = None,
        *args: Any,
        **kwargs: Any,
    ) -> ActionReturnType: ...

    def get_action(
        self,
        obs: ObservationType,
        action_mask: ActionMaskInput = None,
        hidden_state: dict[str, torch.Tensor] | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> ActionReturnType | RecurrentActionReturnType:
        """Return the next action to take in the environment.

        :param obs: Environment observation, or multiple observations in a batch
        :type obs: ObservationType
        :param action_mask: Mask of legal actions 1=legal 0=illegal, defaults to None
        :type action_mask: ActionMaskInput
        :param hidden_state: Hidden state for recurrent policies, defaults to None
        :type hidden_state: dict[str, torch.Tensor] | None
        :return: Action, log probability, entropy, state values, and (if recurrent) next hidden state
        :rtype: tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray, dict[str, torch.Tensor] | None] | tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]
        """
        preprocessed_obs = self.preprocess_observation(obs)
        with torch.no_grad():
            (
                action,
                log_prob,
                entropy,
                values,
                next_hidden,
            ) = self._get_action_and_values(
                preprocessed_obs,
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
                # Scale on-device before converting: scale_action operates on
                # tensors, and mixing the numpy copy with on-device bound
                # tensors is undefined for CUDA.
                action_np = self.actor.scale_action(action).cpu().data.numpy()
            else:
                action_np = np.clip(
                    action_np,
                    self.action_space.low,
                    self.action_space.high,
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
        return (
            action_np,
            log_prob_np,
            entropy_np,
            values_np,
        )

    def learn(self, experiences: TensorDict | None = None) -> float:
        """Update agent network parameters to learn from experiences.

        :param experiences: Optional pre-collected rollout batch. When ``None``
            (the default), samples are drawn from the agent's internal rollout
            buffer.
        :type experiences: TensorDict | None
        :return: Mean loss value from training.
        :rtype: float
        """
        if self.recurrent:
            return self._learn_from_rollout_buffer_bptt()

        return self._learn_from_rollout_buffer_flat(experiences)

    def _learn_from_rollout_buffer_flat(
        self,
        buffer_td_external: TensorDict | None = None,
    ) -> float:
        """Learning procedure using flattened samples (no BPTT)."""
        if buffer_td_external is not None:
            buffer_td = buffer_td_external
        else:
            buffer_td = self.rollout_buffer.get_tensor_batch(device=str(self.device))

        if buffer_td.is_empty():
            warnings.warn("Buffer data is empty. Skipping learning step.", stacklevel=2)
            for metric_name in (
                "loss",
                "policy_loss",
                "value_loss",
                "entropy_loss",
            ):
                self.metrics.log(metric_name, 0.0)
            return 0.0

        batch_size = self.batch_size
        num_samples = (
            int(buffer_td.batch_size[0])
            if buffer_td_external is not None
            else self.rollout_buffer.size()
        )
        indices = np.arange(num_samples)

        # Wrap the buffer as a typed batch once, then index it for minibatches.
        # ``values`` is renamed to ``value_preds`` to avoid clashing with
        # ``TensorDict.values()``.
        minibatch_fields = [
            "observations",
            "actions",
            "log_probs",
            "advantages",
            "returns",
            "values",
            "action_masks",
        ]
        buffer_batch_td = buffer_td.select(*minibatch_fields, strict=False)
        buffer_batch_td.rename_key_("values", "value_preds")
        buffer_batch = RolloutMinibatch.from_tensordict(buffer_batch_td)

        # Normalize advantages globally
        valid_advantages = buffer_batch.advantages
        buffer_batch.advantages = (valid_advantages - valid_advantages.mean()) / (
            valid_advantages.std() + 1e-8
        )

        # Accumulated as tensors during the epoch loop, logged as floats.
        learn_metrics: dict[str, float | torch.Tensor] = {
            "loss": 0.0,
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy_loss": 0.0,
        }
        approx_kl_divs = []
        for _ in range(self.update_epochs):
            np.random.shuffle(indices)
            for start_idx in range(0, num_samples, batch_size):
                end_idx = min(start_idx + batch_size, num_samples)
                minibatch_indices = indices[start_idx:end_idx]

                minibatch = buffer_batch[torch.from_numpy(minibatch_indices)]
                mb_obs = minibatch.observations
                mb_actions = minibatch.actions
                mb_log_probs = minibatch.log_probs
                mb_advantages = minibatch.advantages
                mb_returns = minibatch.returns
                mb_old_values = minibatch.value_preds
                mb_action_masks = minibatch.action_masks

                if isinstance(self.action_space, spaces.Discrete):
                    mb_actions = mb_actions.squeeze(-1)

                log_probs, entropy, values = self.evaluate_actions(
                    obs=mb_obs,
                    actions=mb_actions,
                    hidden_state=None,
                    action_mask=mb_action_masks,
                )

                # Policy loss
                ratio = torch.exp(log_probs - mb_log_probs)
                policy_loss1 = -mb_advantages * ratio
                policy_loss2 = -mb_advantages * torch.clamp(
                    ratio,
                    1 - self.clip_coef,
                    1 + self.clip_coef,
                )
                policy_loss = torch.max(policy_loss1, policy_loss2).mean()

                # Value loss
                value = values.view(-1)
                v_loss_unclipped = (value - mb_returns) ** 2
                v_clipped = mb_old_values + torch.clamp(
                    value - mb_old_values,
                    -self.clip_coef,
                    self.clip_coef,
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

                if self.target_kl is not None:
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

                # Accumulate as tensors; a single device sync happens after
                # the epoch loop instead of four per minibatch
                learn_metrics["loss"] += loss.detach()
                learn_metrics["policy_loss"] += policy_loss.detach()
                learn_metrics["value_loss"] += v_loss.detach()
                learn_metrics["entropy_loss"] += entropy_loss.detach()

            # Early stopping for the epoch if KL divergence target is exceeded
            if self.target_kl is not None and np.mean(approx_kl_divs) > self.target_kl:
                break

        # Log metrics
        divisor = num_samples * self.update_epochs
        logged_metrics: dict[str, float] = {
            k: (v / divisor).item() if isinstance(v, torch.Tensor) else v / divisor
            for k, v in learn_metrics.items()
        }
        for key, value in logged_metrics.items():
            self.metrics.log(key, value)

        return logged_metrics["loss"]

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
        self.rollout_buffer.prepare_sequence_tensors(device=str(self.device))

        # Here, batch_size means number of sequences per minibatch.
        # Accumulated as tensors during the epoch loop, logged as floats.
        learn_metrics: dict[str, float | torch.Tensor] = {
            "loss": 0.0,
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy_loss": 0.0,
        }
        total_minibatch_updates_total = 0
        for epoch in range(self.update_epochs):
            approx_kl_divs_epoch = []  # KL divergences for this epoch's minibatches
            approx_kl_divs_minibatch_timesteps = []
            num_minibatches_this_epoch = 0

            # Itreate over minibatches of sequences
            minibatch_gen = self.rollout_buffer.get_minibatch_sequences(
                batch_size=self.batch_size,
            )
            for minibatch_padded, minibatch_unpadded in minibatch_gen:
                # Obs shape: (batch_seq * seq_len, *obs_dims) or nested TD
                # Actions shape: (batch_seq * seq_len, *act_dims)
                # Other tensors shape: (batch_seq * seq_len, )
                targets_td = minibatch_unpadded.select(
                    "log_probs", "advantages", "returns", "values", strict=False
                )
                targets_td.rename_key_("values", "value_preds")
                padded: RolloutSequenceMinibatch = (
                    RolloutSequenceMinibatch.from_tensordict(
                        minibatch_padded.select(
                            "observations",
                            "actions",
                            "pad_mask",
                            "action_masks",
                            strict=False,
                        )
                    )
                )
                targets: RolloutSequenceTargets = (
                    RolloutSequenceTargets.from_tensordict(targets_td)
                )
                mb_actions_seq = padded.actions
                mb_initial_hidden_states_dict: dict[str, torch.Tensor] | None = (
                    minibatch_padded.get_non_tensor(
                        "initial_hidden_states",
                        default=None,
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
                    obs=padded.observations,
                    actions=mb_actions_seq,
                    hidden_state=mb_initial_hidden_states_dict,
                    action_mask=padded.action_masks,
                )

                # Mask out padded values
                new_values = new_values[padded.pad_mask]
                new_log_probs = new_log_probs[padded.pad_mask]
                entropy = entropy[padded.pad_mask]

                if isinstance(entropy, torch.Tensor):
                    entropy = entropy.mean()

                # Policy loss
                ratio = torch.exp(new_log_probs - targets.log_probs)
                policy_loss1 = -targets.advantages * ratio
                policy_loss2 = -targets.advantages * torch.clamp(
                    ratio,
                    1 - self.clip_coef,
                    1 + self.clip_coef,
                )
                policy_loss = torch.max(policy_loss1, policy_loss2).mean()

                # Value loss
                v_loss_unclipped = (new_values - targets.returns) ** 2
                v_clipped = targets.value_preds + torch.clamp(
                    new_values - targets.value_preds,
                    -self.clip_coef,
                    self.clip_coef,
                )

                v_loss_clipped = (v_clipped - targets.returns) ** 2
                v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                value_loss = v_loss_max.mean()

                # Entropy loss
                entropy_loss = -entropy

                if self.target_kl is not None:
                    with torch.no_grad():
                        log_ratio = new_log_probs - targets.log_probs
                        approx_kl_divs_minibatch_timesteps.append(
                            ((torch.exp(log_ratio) - 1) - log_ratio).mean().item(),
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

                # Accumulate as tensors; a single device sync happens after
                # the epoch loop instead of four per minibatch
                learn_metrics["loss"] += loss.detach()
                learn_metrics["policy_loss"] += policy_loss.detach()
                learn_metrics["value_loss"] += value_loss.detach()
                learn_metrics["entropy_loss"] += entropy_loss.detach()
                num_minibatches_this_epoch += 1

                if (
                    self.target_kl is not None
                    and len(approx_kl_divs_minibatch_timesteps) > 0
                ):
                    # Average KL over all timesteps in this minibatch of sequences
                    kl_for_current_minibatch = np.mean(
                        approx_kl_divs_minibatch_timesteps,
                    )
                    approx_kl_divs_epoch.append(
                        kl_for_current_minibatch,
                    )  # Store minibatch average KL

                    if kl_for_current_minibatch > self.target_kl:
                        warnings.warn(
                            f"Epoch {epoch}, Minibatch: KL divergence {kl_for_current_minibatch:.4f} exceeded target {self.target_kl}. Stopping update for this epoch.",
                            stacklevel=2,
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
                        f"Epoch {epoch}: Average KL divergence {avg_kl_this_epoch:.4f} exceeded target {self.target_kl} after completing epoch. Consider adjusting learning rate or target_kl.",
                        stacklevel=2,
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

        # Log metrics
        divisor = max(1e-8, total_minibatch_updates_total)
        logged_metrics: dict[str, float] = {
            k: (v / divisor).item() if isinstance(v, torch.Tensor) else v / divisor
            for k, v in learn_metrics.items()
        }
        for key, value in logged_metrics.items():
            self.metrics.log(key, value)

        return logged_metrics["loss"]

    def test(
        self,
        env: gym.Env | gym.vector.VectorEnv,
        max_steps: int | None = None,
        loop: int = 3,
        vectorized: bool = True,
        callback: Callable[[float, dict[str, Any]], None] | None = None,
    ) -> float:
        """Return mean test score of agent in environment with epsilon-greedy policy.

        :param env: The environment to be tested in
        :type env: gym.Env | gym.vector.VectorEnv
        :param max_steps: Maximum number of testing steps, defaults to None
        :type max_steps: int, optional
        :param loop: Number of testing loops/episodes to complete. The returned score is the mean. Defaults to 3
        :type loop: int, optional
        :param vectorized: Whether the environment is vectorized, defaults to True
        :type vectorized: bool, optional
        :param callback: Optional callback function that takes the sum of rewards and the last info dictionary as input, defaults to None
        :type callback: Callable[[float, dict[str, Any]], None] | None

        :return: Mean test score of agent in environment
        :rtype: float
        """
        # set to evaluation mode. This is important for batch norm and dropout layers
        self.actor.eval()
        self.critic.eval()
        self.set_training_mode(False)

        with torch.no_grad():
            rewards = []
            num_envs = get_num_envs(env) if vectorized else 1

            for _ in range(loop):
                obs, info = env.reset()
                scores = np.zeros(num_envs)
                completed_episode_scores = np.zeros(num_envs)
                finished = np.zeros(num_envs, dtype=bool)
                step = 0
                test_hidden_state = (
                    self.get_initial_hidden_state(num_envs) if self.recurrent else None
                )

                # Initialize last_info holder
                last_infos = [{}] * num_envs if vectorized else {}
                while not np.all(finished):
                    # Process action mask
                    action_mask = None
                    if vectorized:
                        # Check if info is a list/array of dicts
                        if (
                            isinstance(info, (list, np.ndarray))
                            and len(info) == num_envs
                            and all(isinstance(i, dict) for i in info)
                        ):
                            # The guard established one info dict per
                            # sub-environment.
                            info_dicts = info
                            masks = [
                                env_info.get("action_mask")
                                for env_info in info_dicts
                                if isinstance(env_info, dict)
                            ]
                            present_masks = [m for m in masks if m is not None]
                            # If all environments returned a mask and they are not None
                            if len(present_masks) == len(masks):
                                try:
                                    action_mask = np.stack(present_masks)
                                except Exception as e:
                                    warnings.warn(
                                        f"Could not stack action masks: {e}",
                                        stacklevel=2,
                                    )
                                    action_mask = None
                            # If only some environments returned masks, we probably can't use them reliably
                            elif present_masks:
                                warnings.warn(
                                    "Action masks not provided for all vectorized environments. Skipping mask.",
                                    stacklevel=2,
                                )
                                action_mask = None
                        # Handle case where info might be a single dict even if vectorized (e.g. VecNormalize)
                        elif isinstance(info, dict):
                            action_mask = info.get("action_mask", None)

                    elif isinstance(info, dict):
                        action_mask = info.get("action_mask", None)

                    # Get action; the recurrent flag selects which arm of the
                    # get_action return union is produced.
                    if test_hidden_state is not None:
                        action, _, _, _, test_hidden_state = self.get_action(
                            obs,
                            action_mask=action_mask,
                            hidden_state=test_hidden_state,
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
                            num_envs,
                        )
                        if isinstance(test_hidden_state, dict):
                            for key in test_hidden_state:
                                reset_states = initial_hidden_states_for_reset[key][
                                    :,
                                    newly_finished,
                                    :,
                                ]
                                if reset_states.shape[1] > 0:
                                    test_hidden_state[key][
                                        :,
                                        newly_finished,
                                        :,
                                    ] = reset_states

                    if np.any(newly_finished):
                        completed_episode_scores[newly_finished] = scores[
                            newly_finished
                        ]
                        finished[newly_finished] = True

                # End of episode loop for one test run
                loop_reward_sum = np.sum(completed_episode_scores)

                # Prepare info for callback; check the dict leaf before the
                # sequence arms so the narrowing is exact.
                final_info_for_callback: dict[str, Any] = {}
                if isinstance(last_infos, dict):
                    final_info_for_callback = last_infos
                elif isinstance(last_infos, (list, np.ndarray)) and len(last_infos) > 0:
                    first_info = last_infos[0]
                    if isinstance(first_info, dict):
                        final_info_for_callback = first_info

                if callback is not None:
                    callback(float(loop_reward_sum), final_info_for_callback)

                eval_fitness = np.mean(completed_episode_scores)
                rewards.append(eval_fitness)

        mean_fit = float(np.mean(rewards))
        self.metrics.add_fitness(mean_fit)

        # cleanup evaluation mode back into the default training mode (e.g. batch norm and dropout layers)
        self.set_training_mode(True)
        self.actor.train()
        self.critic.train()

        return mean_fit
