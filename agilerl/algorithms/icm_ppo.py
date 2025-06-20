import copy
import warnings
from typing import Any, Callable, Dict, Optional, Tuple, Union

from agilerl.components.rollout_buffer import RolloutBuffer
from agilerl.algorithms.core.wrappers import OptimizerWrapper
from agilerl.modules.base import EvolvableModule
from agilerl.modules.configs import MlpNetConfig
from agilerl.networks.actors import StochasticActor
from agilerl.networks.base import EvolvableNetwork
from agilerl.networks.value_networks import ValueNetwork
from agilerl.utils.algo_utils import (
    flatten_experiences,
    get_experiences_samples,
    is_vectorized_experiences,
    make_safe_deepcopies,
    obs_channels_to_first,
    share_encoder_parameters,
    stack_experiences,
)

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from gymnasium import spaces
from torch.nn.utils import clip_grad_norm_
from tensordict import TensorDict

from agilerl.algorithms.core.base import RLAlgorithm
from agilerl.algorithms.ppo import PPO
from agilerl.algorithms.core.registry import HyperparameterConfig, NetworkGroup
from agilerl.typing import ArrayOrTensor, ExperiencesType, GymEnvType, BPTTSequenceType

# Imports managed by the broader environment, placeholders for linting/type-checking
from agilerl.components.icm import (
    ICM,
    get_evolvable_cnn_config,
    get_evolvable_mlp_config,
)
# BIG TODO: currently, when not using shared encoder & the ICM is recurrent,
# the ICM's Encoder hidden state is not stored in the rollout buffer, and
# the current logic does not work for this case. Update this to store the
# encoder hidden state in the rollout buffer, and to use this hidden state
# for the ICM in the learn() methods.
class ICMRolloutBuffer(RolloutBuffer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.encoder_out = np.zeros((self.capacity, self.num_envs), dtype=np.float32)

    def add(
        self,
        *, # force keyword arguments
        obs: ArrayOrTensor,
        action: ArrayOrTensor,
        reward: ArrayOrTensor,
        done: ArrayOrTensor,
        value: ArrayOrTensor,
        log_prob: ArrayOrTensor,
        encoder_out: ArrayOrTensor, # New argument for ICM
        next_obs: Optional[ArrayOrTensor] = None,
        hidden_state: Optional[Dict[str, ArrayOrTensor]] = None,
        next_hidden_state: Optional[Dict[str, ArrayOrTensor]] = None,
    ) -> None:
        """Add experience, including the ICM 'encoder_out' term, to the buffer."""
        # Call original add method first
        super().add(
            obs=obs,
            action=action,
            reward=reward,
            done=done,
            value=value,
            log_prob=log_prob,
            next_obs=next_obs,
            hidden_state=hidden_state,
            next_hidden_state=next_hidden_state,
        )
        # Store the encoder_output term (adjust index logic if super().add changes self.pos)
        if self.encoder_out is not None:
            pos_to_add = (self.pos - 1) % self.capacity # Position where last data was added
            encoder_out = np.asarray(encoder_out)
            self.encoder_out[pos_to_add] = encoder_out

class ICM_PPO(RLAlgorithm):
    """ICM-Augmented PPO algorithm.

    Combines PPO with an Intrinsic Curiosity Module (ICM) for exploration.
    The ICM provides an intrinsic reward based on the prediction error of the environment dynamics.
    This intrinsic reward is added to the extrinsic reward from the environment.
    The PPO agent then optimizes the policy based on this combined reward.
    The ICM module itself is trained concurrently.

    Paper: Pathak et al., 2017 - Curiosity-driven Exploration by Self-supervised Prediction
           (https://pathak22.github.io/noreward-rl/resources/icml17.pdf)

    :param observation_space: Observation space of the environment
    :type observation_space: gym.spaces.Space
    :param action_space: Action space of the environment
    :type action_space: gym.spaces.Space
    :param index: Index to keep track of object instance during tournament selection and mutation, defaults to 0
    :type index: int, optional
    :param hp_config: RL hyperparameter mutation configuration, defaults to None, whereby algorithm mutations are disabled.
    :type hp_config: HyperparameterConfig, optional
    :param net_config: Network configuration for PPO actor and critic, defaults to None
    :type net_config: dict, optional
    :param batch_size: Size of batched sample from replay buffer for learning, defaults to 64
    :type batch_size: int, optional
    :param lr: Learning rate for PPO optimizer, defaults to 1e-4
    :type lr: float, optional
    :param learn_step: Learning frequency (number of steps to collect per rollout), defaults to 2048
    :type learn_step: int, optional
    :param gamma: Discount factor for extrinsic rewards, defaults to 0.99
    :type gamma: float, optional
    :param gae_lambda: Lambda for general advantage estimation, defaults to 0.95
    :type gae_lambda: float, optional
    :param mut: Most recent mutation to agent, defaults to None
    :type mut: str, optional
    :param action_std_init: Initial action standard deviation for continuous actions, defaults to 0.0
    :type action_std_init: float, optional
    :param clip_coef: PPO clipping coefficient, defaults to 0.2
    :type clip_coef: float, optional
    :param ent_coef: Entropy coefficient for PPO loss, defaults to 0.01
    :type ent_coef: float, optional
    :param vf_coef: Value function coefficient for PPO loss, defaults to 0.5
    :type vf_coef: float, optional
    :param max_grad_norm: Maximum norm for gradient clipping, defaults to 0.5
    :type max_grad_norm: float, optional
    :param target_kl: Target KL divergence threshold for PPO, defaults to None
    :type target_kl: Optional[float], optional
    :param normalize_images: Normalize image observations (0-255 to 0-1) and permute channels, defaults to True
    :type normalize_images: bool, optional
    :param update_epochs: Number of epochs to update policy and value networks per learning step, defaults to 4
    :type update_epochs: int, optional
    :param actor_network: Custom PPO actor network, defaults to None
    :type actor_network: Optional[EvolvableModule], optional
    :param critic_network: Custom PPO critic network, defaults to None
    :type critic_network: Optional[EvolvableModule], optional
    :param share_encoders: Whether PPO actor and critic share encoders, defaults to True
    :type share_encoders: bool, optional
    :param num_envs: Number of parallel environments, defaults to 1
    :type num_envs: int, optional
    :param use_rollout_buffer: Whether to use RolloutBuffer (required for ICM-PPO), defaults to True
    :type use_rollout_buffer: bool, optional
    :param rollout_buffer_config: Configuration for RolloutBuffer, defaults to {}
    :type rollout_buffer_config: Optional[Dict[str, Any]], optional
    :param recurrent: Whether PPO policy is recurrent, defaults to False
    :type recurrent: bool, optional
    :param device: Device for PyTorch calculations (e.g., "cpu", "cuda"), defaults to "cpu"
    :type device: str, optional
    :param accelerator: Hugging Face Accelerate accelerator, defaults to None
    :type accelerator: Optional[Any], optional
    :param wrap: Wrap models for distributed training upon creation, defaults to True
    :type wrap: bool, optional
    :param load_bptt_full_buffer: Flag for BPTT loading strategy, defaults to False
    :type load_bptt_full_buffer: bool, optional

    ICM Specific Parameters:
    :param icm_lr: Learning rate for ICM optimizer, defaults to 1e-4
    :type icm_lr: float, optional
    :param icm_beta: Weight of the forward model loss vs inverse model loss in ICM (L_total = (1-beta)*L_I + beta*L_F), defaults to 0.2
    :type icm_beta: float, optional
    :param intrinsic_reward_weight: Weight (eta) for the intrinsic reward, defaults to 0.1
    :type intrinsic_reward_weight: float, optional
    :param use_shared_encoder_for_icm: Whether ICM shares the PPO actor's encoder, defaults to False
    :type use_shared_encoder_for_icm: bool, optional
    :param icm_encoder_net_config: Network configuration for ICM's own encoder (if not shared), defaults to None (uses default MLP/CNN)
    :type icm_encoder_net_config: Optional[Dict[str, Any]], optional
    :param icm_inverse_net_config: Network configuration for ICM's inverse model head, defaults to None (uses default MLP)
    :type icm_inverse_net_config: Optional[Dict[str, Any]], optional
    :param icm_forward_net_config: Network configuration for ICM's forward model head, defaults to None (uses default MLP)
    :type icm_forward_net_config: Optional[Dict[str, Any]], optional
    :param icm_loss_weight: Weight of the ICM loss in the total loss, defaults to 0.1
    :type icm_loss_weight: float, optional
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
        actor_network: Optional[Any] = None,  # EvolvableModule
        critic_network: Optional[Any] = None,  # EvolvableModule
        share_encoders: bool = True,
        num_envs: int = 1,
        use_rollout_buffer: bool = True,  # Must be True for ICM_PPO current design
        rollout_buffer_config: Optional[Dict[str, Any]] = {},
        recurrent: bool = False,
        device: str = "cpu",
        accelerator: Optional[Any] = None,
        wrap: bool = True,
        bptt_sequence_type: BPTTSequenceType = BPTTSequenceType.CHUNKED,
        # ICM specific parameters
        icm_lr: float = 1e-4,
        icm_beta: float = 0.2,
        intrinsic_reward_weight: float = 0.1,  # eta in the paper
        use_shared_encoder_for_icm: bool = False,
        icm_encoder_net_config: Optional[Dict[str, Any]] = None,
        icm_inverse_net_config: Optional[Dict[str, Any]] = None,
        icm_forward_net_config: Optional[Dict[str, Any]] = None,
        icm_loss_weight: float = 0.1,
    ):
        super().__init__(
            observation_space,
            action_space,
            index=index,
            hp_config=hp_config,
            device=device,
            accelerator=accelerator,
            normalize_images=normalize_images,
            name="ICM_PPO",
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
        assert isinstance(bptt_sequence_type, BPTTSequenceType), "bptt_sequence_type must be a BPTTSequenceType enum value."

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
        # Validate ICM parameters
        assert (
            isinstance(icm_lr, float) and icm_lr > 0
        ), "ICM learning rate must be a positive float."
        assert (
            isinstance(icm_beta, float) and 0 <= icm_beta <= 1
        ), "ICM beta must be a float between 0 and 1."
        assert (
            isinstance(icm_loss_weight, float) and 0 <= icm_loss_weight <= 1
        ), "ICM loss weight must be a float between 0 and 1."
        assert (
            isinstance(intrinsic_reward_weight, float) and intrinsic_reward_weight >= 0
        ), "Intrinsic reward weight (eta) must be non-negative."
        assert isinstance(
            use_shared_encoder_for_icm, bool
        ), "Flag for sharing ICM encoder must be boolean."

        if not isinstance(action_space, (spaces.Discrete, spaces.MultiDiscrete)):
            warnings.warn(
                "ICM is typically used with discrete action spaces. "
                "Ensure your ICM models (inverse net) are adapted for continuous actions if applicable.",
                UserWarning,
                stacklevel=2,
            )
        
        self.recurrent = recurrent
        self.use_rollout_buffer = use_rollout_buffer
        self.net_config = net_config

        if self.recurrent:
            if not self.use_rollout_buffer:
                raise ValueError("use_rollout_buffer must be True if recurrent=True.")
            net_config_dict = self.net_config if self.net_config is not None else {}
            self.max_seq_len = net_config_dict.get("encoder_config", {}).get(
                "max_seq_len", None
            )
            if self.max_seq_len is None:
                raise ValueError("max_seq_len must be provided in net_config['encoder_config'] if recurrent=True.")
        else:
            self.max_seq_len = None


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
        self.use_shared_encoder_for_icm = use_shared_encoder_for_icm
        self.intrinsic_reward_weight = intrinsic_reward_weight
        self.icm_loss_weight = icm_loss_weight
        self.icm_encoder_net_config = icm_encoder_net_config
        self.icm_inverse_net_config = icm_inverse_net_config
        self.icm_forward_net_config = icm_forward_net_config

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
        
        # Initialize ICM
        icm_init_params = {
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "use_internal_encoder": not self.use_shared_encoder_for_icm,
            "encoder_net_config": self.icm_encoder_net_config,
            "inverse_net_config": self.icm_inverse_net_config,
            "forward_net_config": self.icm_forward_net_config,
            "lr": icm_lr,
            "beta": icm_beta,
            "intrinsic_reward_weight": self.intrinsic_reward_weight,
            "device": self.device,
            "accelerator": self.accelerator,
        }
        if self.use_shared_encoder_for_icm:
            # If sharing encoder, ICM needs to know the feature dimension from the actor's encoder
            if hasattr(self.actor, 'latent_dim'):
                 icm_init_params["feature_dim"] = self.actor.latent_dim
            else:
                 warnings.warn("Shared encoder does not have 'latent_dim' attribute. ICM might not be configured correctly.")
        elif self.icm_encoder_net_config is not None:
            icm_init_params["encoder_net_config"] = self.icm_encoder_net_config
        else:
            print("WARNING:use_shared_encoder_for_icm is False and no encoder net config provided for ICM, using default")
            self.icm_net_encoder_config = {
                "recurrent": self.recurrent,
                "mlp_config": {
                    "hidden_size": [64, 128, 128, 256],
                    "activation": "ReLU",
                    "output_activation": None,
                    "layer_norm": False,
                    "dropout": 0.2,
                },
            }
            icm_init_params["encoder_net_config"] = self.icm_net_encoder_config

        self.icm = ICM(**icm_init_params)

        # Share encoders between actor and critic
        self.share_encoders = share_encoders
        if self.share_encoders and all(
            isinstance(net, EvolvableNetwork) for net in [self.actor, self.critic]
        ):
            self.share_encoder_parameters()
            # Need to register a mutation hook that does this after every mutation
            self.register_mutation_hook(self.share_encoder_parameters)

        # Create a single optimizer for all networks
        if self.use_shared_encoder_for_icm:
            # If sharing, ICM's inverse/forward models are trained by the main optimizer
            network_names = ['actor', 'critic', 'icm']#'icm_inverse_model', 'icm_forward_model']
            networks_to_optimize = [self.actor, self.critic, self.icm]#self.icm.inverse_model, self.icm.forward_model]
        else:
            # If not sharing, ICM has its own optimizer. Main optimizer only handles PPO networks.
            network_names = ['actor', 'critic']
            networks_to_optimize = [self.actor, self.critic]

            # Create a separate optimizer for ICM's models

            icm_nets_to_optimize = self.icm.params_to_optimize
            icm_names_to_optimize = ['icm_encoder', 'icm_inverse_model', 'icm_forward_model']
            self.icm_optimizer = OptimizerWrapper(
                optimizer_cls=optim.Adam,
                network_names=icm_names_to_optimize,
                networks=icm_nets_to_optimize,
                lr=self.icm_lr,
                lr_name='icm_lr',
            )

        self.optimizer = OptimizerWrapper(
            optimizer_cls=optim.Adam,
            network_names=network_names,
            networks=networks_to_optimize,
            lr=self.lr,
            lr_name='lr',
        )

        # Initialize rollout buffer if enabled
        if self.use_rollout_buffer:
            self.create_rollout_buffer()
            # Need to register a mutation hook that does this after every mutation (e.g. the batch size, sequence length, etc. have changed)
            self.register_mutation_hook(self.create_rollout_buffer)

        if self.accelerator is not None and wrap:
            self.wrap_models()

        # Register network groups for mutations
        self.register_network_group(NetworkGroup(eval=self.actor, policy=True))
        self.register_network_group(NetworkGroup(eval=self.critic))
        self.register_network_group(NetworkGroup(eval=self.icm))

        self.hidden_state = None


    def create_rollout_buffer(self) -> None:
        """Creates a rollout buffer with the current configuration and adds space for encoder_out."""
        self.rollout_buffer = RolloutBuffer(
            capacity=self.learn_step,
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
            **self.rollout_buffer_config,
        )

        if self.use_shared_encoder_for_icm:
            if not hasattr(self.actor, 'latent_dim') or self.actor.latent_dim is None:
                raise ValueError(
                    "PPO Actor network must have a valid 'latent_dim' attribute "
                    "(output dimension of its encoder) when use_shared_encoder_for_icm is True."
                )

            # The RolloutBuffer stores tensors on CPU by default.
            # self.rollout_buffer.buffer is a TensorDict with batch_size [capacity, num_envs].
            # New fields added to this TensorDict must also conform to this structure
            # and be on the CPU.

            encoder_output_dim = self.actor.latent_dim

            # Define the shape for the 'encoder_out' tensor: (capacity, num_envs, encoder_output_dim)
            encoder_out_tensor_shape = (
                self.rollout_buffer.capacity,
                self.rollout_buffer.num_envs,
                encoder_output_dim
            )

            # Initialize the 'encoder_out' field in the buffer's TensorDict.
            # This pre-allocates space for the encoder outputs.
            # It's crucial this tensor is on the CPU, like other buffer components.
            self.rollout_buffer.buffer["encoder_out"] = torch.zeros(
                encoder_out_tensor_shape,
                dtype=torch.float32,
                device="cpu"
            )

            return

        if self.recurrent:
            hidden_state_tensor_shape = (
                self.rollout_buffer.capacity,
                self.rollout_buffer.num_envs,
                self.icm.hidden_state_size
            )
            self.rollout_buffer.buffer["hidden_state"] = torch.zeros(
                hidden_state_tensor_shape,
                dtype=torch.float32,
                device="cpu"
            )
            self.rollout_buffer.buffer["next_hidden_state"] = torch.zeros(
                hidden_state_tensor_shape,
                dtype=torch.float32,
                device="cpu"
                )

    def add_to_rollout_buffer(self, *args, encoder_out: Optional[ArrayOrTensor] = None, icm_hidden_state: Optional[ArrayOrTensor] = None, icm_next_hidden_state: Optional[ArrayOrTensor] = None, **kwargs):
        """Extend the add method to add the encoder_out field to the rollout buffer."""
        if isinstance(self.rollout_buffer, RolloutBuffer):
            self.rollout_buffer.add(*args, **kwargs)
            pos = self.rollout_buffer.pos-1
            if self.use_shared_encoder_for_icm:
                self.rollout_buffer.buffer["encoder_out"][pos] = torch.as_tensor(encoder_out, dtype=torch.float32, device="cpu")
                # decrement pos by 1 to get the position before the call to super().add() incremented it
                return
            
            if self.recurrent:
                self.rollout_buffer.buffer["hidden_state"][pos] = torch.as_tensor(icm_hidden_state, dtype=torch.float32, device="cpu")
                if icm_next_hidden_state is not None:
                    self.rollout_buffer.buffer["next_hidden_state"][pos] = torch.as_tensor(icm_next_hidden_state, dtype=torch.float32, device="cpu")

    def _get_action_and_values(
        self,
        obs: ArrayOrTensor,
        action_mask: Optional[ArrayOrTensor] = None,
        hidden_state: Optional[ArrayOrTensor] = None,
        *,  # keyword-only
        sample: bool = True,  # NEW flag
    ) -> Tuple[
        ArrayOrTensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[ArrayOrTensor], Optional[ArrayOrTensor]
    ]:
        """Augmented PPO method to get the next action to take in the environment and the values,
        and store the hidden state and latent pi for use later.

        :param obs: Environment observation, or multiple observations in a batch
        :type obs: numpy.ndarray[float]
        :param action_mask: Mask of legal actions 1=legal 0=illegal, defaults to None
        :type action_mask: numpy.ndarray, optional
        :param hidden_state: Hidden state for recurrent policies, defaults to None
        :type hidden_state: numpy.ndarray, optional
        :param sample: Whether to sample an action or return the mode/mean. Defaults to True.
        :type sample: bool
        :return: Action, log probability, entropy, state values, and next hidden state
        :rtype: Tuple[ArrayOrTensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[ArrayOrTensor], Optional[ArrayOrTensor]]
        """
        if not self.use_shared_encoder_for_icm:
            return self._get_action_and_values_icm(obs, action_mask, hidden_state, sample=sample)

        if hidden_state is not None:
            latent_pi, next_hidden = self.actor.extract_features(
                obs, hidden_state=hidden_state
            )

            action, log_prob, entropy = self.actor.forward_head(
                latent_pi, action_mask=action_mask, sample=sample
            )

            if self.share_encoders:
                values = self.critic.forward_head(latent_pi).squeeze(-1)
            else:
                values, next_hidden = self.critic(obs, hidden_state=next_hidden)
                values = values.squeeze(-1)

            return action, log_prob, entropy, values, next_hidden, latent_pi
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

            return action, log_prob, entropy, values, None, latent_pi

    def evaluate_actions(
        self,
        obs: ArrayOrTensor,
        actions: ArrayOrTensor,
        hidden_state: Optional[Dict[str, ArrayOrTensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluates the actions.

        :param obs: Environment observation, or multiple observations in a batch
        :type obs: numpy.ndarray[float]
        :param actions: Actions to evaluate
        :type actions: torch.Tensor
        :param hidden_state: Hidden state for recurrent policies, defaults to None. Expected shape: dict with tensors of shape (batch_size, 1, hidden_size).
        :type hidden_state: Dict[str, ArrayOrTensor], optional
        :return: Log probability, entropy, and state values
        :rtype: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        """
        obs = self.preprocess_observation(obs)

        eval_hidden_state = None
        if self.recurrent and hidden_state is not None:
            # Reshape hidden state for RNN: (batch_size, 1, hidden_size) -> (1, batch_size, hidden_size)
            eval_hidden_state = {
                key: val.permute(1, 0, 2) for key, val in hidden_state.items()
            }

        # Get values from actor-critic
        if self.use_shared_encoder_for_icm:
            _, _, entropy, values, _, hidden_state, _ = self._get_action_and_values(
                obs, hidden_state=eval_hidden_state, sample=False  # No need to sample here
            )
        else:
            _, _, entropy, values, _, hidden_state = self._get_action_and_values(
                obs, hidden_state=eval_hidden_state, sample=False  # No need to sample here
            )

        # Get log probability of the actions using the *original* hidden state if needed by actor?
        # Let's assume actor.action_log_prob does not require hidden state for now, or handles it internally.
        # If it does require hidden state, we might need to pass eval_hidden_state or recalculate.
        log_prob = self.actor.action_log_prob(actions)

        # Use -log_prob as entropy when squashing output in continuous action spaces
        if entropy is None:
            entropy = -log_prob.mean()

        return log_prob, entropy, values, hidden_state

    def collect_rollouts(
        self,
        env: GymEnvType,
        n_steps: int = None,
        reset_on_collect: bool = True,
    ) -> None:
        """
        Collect rollouts from the environment and store them in the rollout buffer.

        :param env: The environment to collect rollouts from
        :type env: GymEnvType
        :param n_steps: Number of steps to collect, defaults to self.learn_step
        :type n_steps: int, optional
        :param reset_on_collect: Whether to reset the buffer before collecting (Note: current implementation always resets), defaults to True
        :type reset_on_collect: bool, optional
        :rtype: None
        """
        if not self.use_rollout_buffer:
            raise RuntimeError(
                "collect_rollouts can only be used when use_rollout_buffer=True"
            )

        n_steps = n_steps or self.learn_step
        if reset_on_collect:
            # Initial reset
            obs, info = env.reset()
        
            # self.hidden_state stores the current hidden state for the actor across steps
            self.hidden_state = self.get_initial_hidden_state(self.num_envs) if self.recurrent else None
        else: 
            obs, info = self.last_obs

        self.rollout_buffer.reset()

        current_hidden_state = self.hidden_state

        encoder_last_output = None
        encoder_output = None
        last_obs = None
        for _ in range(n_steps):
            # Get action
            if self.recurrent:
                returned_tuple = self.get_action(
                    obs,
                    action_mask=info.get("action_mask", None),
                    hidden_state=current_hidden_state,
                )
                if self.use_shared_encoder_for_icm:
                    action, log_prob, _, value, next_hidden, encoder_output = returned_tuple
                else:
                    action, log_prob, _, value, next_hidden = returned_tuple

                self.hidden_state = next_hidden
            else:
                # No need for next_hidden in non-recurrent networks, so we're not even returning it
                returned_tuple = self.get_action(
                    obs, action_mask=info.get("action_mask", None)
                )
                if self.use_shared_encoder_for_icm:
                    action, log_prob, _, value, _, encoder_output = returned_tuple
                else:
                    action, log_prob, _, value, _ = returned_tuple

            # Execute action
            next_obs, reward, done, truncated, next_info = env.step(action)

            if self.use_shared_encoder_for_icm:
                encoder_last_output = encoder_output
            else:
                last_obs = obs

            # Get intrinsic reward
            intrinsic_reward, _, _ = self.get_intrinsic_reward(
                action,
                last_obs,
                obs,
                encoder_last_output,
                encoder_output,
                current_hidden_state,
                next_hidden
            )
            combined_reward = (1 - self.intrinsic_reward_weight) * reward + intrinsic_reward.detach().cpu().numpy() # intrinsic reward is already weighted by self.intrinsic_reward_weight

            # Handle both single environment and vectorized environments terminal states
            if isinstance(done, list) or isinstance(done, np.ndarray):
                is_terminal = (
                    np.logical_or(done, truncated)
                    if isinstance(truncated, (list, np.ndarray))
                    else done
                )
            else:
                is_terminal = done or truncated

            # Ensure shapes are correct (num_envs, ...) for rollout buffer. This isn't necessary by itself, but it's good for debugging.
            reward = np.atleast_1d(combined_reward)
            is_terminal = np.atleast_1d(is_terminal)
            value = np.atleast_1d(value)
            log_prob = np.atleast_1d(log_prob)
            if self.use_shared_encoder_for_icm:
                encoder_output = np.atleast_1d(encoder_output)

            self.add_to_rollout_buffer(
                obs=obs,
                action=action,
                reward=reward,
                done=is_terminal,
                value=value.reshape(-1),
                log_prob=log_prob,
                next_obs=next_obs,
                hidden_state=current_hidden_state,
                encoder_out=encoder_output,
            )

            # Reset hidden state for finished environments
            if self.recurrent and np.any(is_terminal):
                # Create a mask for finished environments
                finished_mask = is_terminal.astype(bool)
                # Get initial hidden states only for the finished environments
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
                        # initial_hidden_states_for_reset[key] has shape (1, num_envs, hidden_size)
                        # finished_mask has shape (num_envs,)
                        # Index along the num_envs dimension (dim 1)
                        reset_states = initial_hidden_states_for_reset[key][
                            :, finished_mask, :
                        ]

                        if (
                            reset_states.shape[1] > 0
                        ):  # Check num_envs dimension if any env finished
                            # Assign to the correct slice along the num_envs dimension
                            self.hidden_state[key][:, finished_mask, :] = reset_states
                # Add handling for numpy if needed

            # Update the current hidden state for the next timestep
            if self.recurrent:
                current_hidden_state = self.hidden_state

            # Update for next step
            obs = next_obs
            info = next_info

        self.last_obs = (next_obs, info) # Store the last observation for the next step

        # Compute advantages and returns
        with torch.no_grad():
            # Get value for last observation
            if self.recurrent:
                returned_tuple = self._get_action_and_values(
                    obs, hidden_state=self.hidden_state
                )
            else:
                returned_tuple = self._get_action_and_values(obs)

            last_value = returned_tuple[3]

            last_value = last_value.cpu().numpy()
            last_done = np.atleast_1d(done)  # Ensure last_done has shape (num_envs,)

        # Compute returns and advantages
        self.rollout_buffer.compute_returns_and_advantages(
            last_value=last_value, last_done=last_done
        )

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
            Optional[Dict[str, ArrayOrTensor]],
            ArrayOrTensor,
        ],
        Tuple[ArrayOrTensor, torch.Tensor, torch.Tensor, torch.Tensor, ArrayOrTensor],
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
        if not self.use_shared_encoder_for_icm:
            #return super().get_action(obs, action_mask, hidden_state)
            return self._get_action_and_values_icm(obs, action_mask, hidden_state)

        obs = self.preprocess_observation(obs)
        with torch.no_grad():
            if self.recurrent and hidden_state is not None:
                action, log_prob, entropy, values, next_hidden, latent_pi = (
                    self._get_action_and_values(obs, action_mask, hidden_state)
                )
            else:
                action, log_prob, entropy, values, next_hidden, latent_pi = (
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
                latent_pi.cpu().data.numpy(),
            )
        else:
            return (
                action,
                log_prob.cpu().data.numpy(),
                entropy.cpu().data.numpy(),
                values.cpu().data.numpy(),
                latent_pi.cpu().data.numpy(),
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
        mean_icm_loss = 0
        mean_icm_i_loss = 0
        mean_icm_f_loss = 0
        for epoch in range(self.update_epochs):
            np.random.shuffle(batch_idxs)
            for start in range(0, num_samples, self.batch_size):
                minibatch_idxs = batch_idxs[start : start + self.batch_size]
                returned_samples = get_experiences_samples(minibatch_idxs, *experiences)
                batch_latent_pi = None
                if self.use_shared_encoder_for_icm:
                    (
                        batch_states,
                        batch_actions,
                        batch_log_probs,
                        batch_advantages,
                        batch_returns,
                        batch_values,
                        batch_latent_pi,
                    ) = returned_samples
                else:
                    (
                        batch_states,
                        batch_actions,
                        batch_log_probs,
                        batch_advantages,
                        batch_returns,
                        batch_values,
                    ) = returned_samples

                # batch_actions = batch_actions.squeeze()
                # batch_returns = batch_returns.squeeze()
                # batch_log_probs = batch_log_probs.squeeze()
                # batch_advantages = batch_advantages.squeeze()
                # batch_values = batch_values.squeeze()
                # batch_latent_pi = batch_latent_pi.squeeze() if batch_latent_pi is not None else None

                if len(minibatch_idxs) > 1:
                    log_prob, entropy, value, hidden_state = self.evaluate_actions(
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
                    icm_total_loss, icm_i_loss, icm_f_loss = torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0)

                    # add ICM loss if using shared encoder, or update independently if not (ppo update first, ICM)
                    final_loss = loss
                    if batch_latent_pi is not None: # i.e. not using shared encoder
                        # ICM loss
                        icm_total_loss, icm_i_loss, icm_f_loss, _, _ = self.icm.compute_loss(
                            action_batch_t=batch_actions,
                            obs_batch_t=batch_states[:-1],
                            next_obs_batch_t=batch_states[1:],
                            embedded_obs=batch_latent_pi[:-1],
                            embedded_next_obs=batch_latent_pi[1:],
                            hidden_state=hidden_state[:-1] if hidden_state else None,
                            hidden_state_next=hidden_state[1:] if hidden_state else None,
                        )

                        final_loss += icm_total_loss

                    # actor + critic loss backprop
                    self.optimizer.zero_grad()
                    if self.accelerator is not None:
                        self.accelerator.backward(final_loss)
                    else:
                        final_loss.backward()

                    # Clip gradients
                    clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                    clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                    if self.use_shared_encoder_for_icm:
                        clip_grad_norm_(self.icm.parameters(), self.max_grad_norm)

                    self.optimizer.step()

                    if batch_latent_pi is None: # i.e. not using shared encoder
                        icm_total_loss, icm_i_loss, icm_f_loss, _, _ = self.icm.update(
                            obs_batch=batch_states[:-1],
                            action_batch=batch_actions,
                            next_obs_batch=batch_states[1:],
                            hidden_state_obs=hidden_state[:-1] if hidden_state else None,
                            hidden_state_next_obs=hidden_state[1:] if hidden_state else None,
                        )

                    mean_loss += final_loss.item()
                    mean_icm_loss += icm_total_loss.item()
                    mean_icm_i_loss += icm_i_loss.item()
                    mean_icm_f_loss += icm_f_loss.item()

            if self.target_kl is not None:
                if 'approx_kl' in locals() and approx_kl > self.target_kl:
                    break

        mean_loss /= num_samples * self.update_epochs
        mean_icm_loss /= num_samples * self.update_epochs
        mean_icm_i_loss /= num_samples * self.update_epochs
        mean_icm_f_loss /= num_samples * self.update_epochs

        return mean_loss, mean_icm_loss, mean_icm_i_loss, mean_icm_f_loss

    def _learn_from_rollout_buffer(self) -> float:
        """
        Learn from data in the rollout buffer.

        :return: Mean loss value
        :rtype: float
        """
        # ------------------------------------------------------------------
        # Decide whether to use sequence-based BPTT training or flattened batch
        # ------------------------------------------------------------------

        if self.recurrent and self.max_seq_len is not None and self.max_seq_len > 0:
            return self._learn_from_rollout_buffer_bptt()
        else:
            return self._learn_from_rollout_buffer_flat()

    # ------------------------------------------------------------------
    # Original flattened learning logic moved to separate helper
    # ------------------------------------------------------------------

    def _learn_from_rollout_buffer_flat(self, buffer_td_external: Optional[TensorDict] = None) -> float:
        """Learning procedure using flattened samples (no BPTT)."""
        if buffer_td_external is not None:
            buffer_td = buffer_td_external
        else:
            # .get_tensor_batch() returns a TensorDict on the specified device
            buffer_td = self.rollout_buffer.get_tensor_batch(device=self.device)

        if buffer_td.is_empty():
            warnings.warn("Buffer data is empty. Skipping learning step.")
            return 0.0

        observations = buffer_td["observations"]
        advantages = buffer_td["advantages"]

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        batch_size = self.batch_size
        num_samples = observations.size(0) # Total number of samples in the buffer

        indices = np.arange(num_samples)
        mean_loss = 0.0
        approx_kl_divs = []
        total_minibatch_updates_this_run = 0

        for epoch in range(self.update_epochs):
            np.random.shuffle(indices)
            num_minibatches_this_epoch = 0

            for start_idx in range(0, num_samples, batch_size):
                end_idx = min(start_idx + batch_size, num_samples)
                minibatch_indices = indices[start_idx:end_idx]

                # Slice the TensorDict to get the minibatch
                minibatch_td = buffer_td[minibatch_indices]

                mb_obs = minibatch_td["observations"]
                mb_actions = minibatch_td["actions"]
                mb_old_log_probs = minibatch_td["log_probs"]
                mb_advantages = advantages[minibatch_indices] # Use globally normalized advantages
                mb_returns = minibatch_td["returns"]
                mb_latent_pi = minibatch_td.get("encoder_out", None)

                eval_hidden_state = None
                if self.recurrent:
                    # If recurrent, hidden_states should be in the buffer_td
                    # buffer_td["hidden_states"] is a TD: {key: tensor_shape_(total_samples, layers, size)}
                    # minibatch_td["hidden_states"] will be {key: tensor_shape_(len(minibatch_indices), layers, size)}
                    # _get_action_and_values expects dict {key: (layers, batch, size)}
                    if "hidden_states" in minibatch_td.keys(include_nested=True):
                        mb_hidden_states_td = minibatch_td.get("hidden_states")
                        eval_hidden_state = {
                            # v has shape (minibatch_size, layers, size), permute to (layers, minibatch_size, size)
                            k: v.permute(1, 0, 2).contiguous()
                            for k, v in mb_hidden_states_td.items()
                        }
                    else:
                        warnings.warn(
                            "Recurrent policy, but no hidden_states found in minibatch_td for flat learning."
                        )

                loss_dict = self.compute_loss(mb_obs, mb_actions, mb_old_log_probs, mb_advantages, mb_returns, mb_latent_pi, eval_hidden_state, learn_by_bptt=False)
                approx_kl_divs.append(loss_dict['approx_kl'])
                loss = loss_dict['loss']

                if self.use_shared_encoder_for_icm:
                    loss = self.icm_loss_weight * loss_dict['icm_total_loss'] + (1 - self.icm_loss_weight) * loss

                self.optimizer.zero_grad()
                loss.backward()  # Gradients accumulate over the sequence within this backward call
                clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                if self.use_shared_encoder_for_icm:
                    clip_grad_norm_(self.icm.parameters(), self.max_grad_norm)
                self.optimizer.step()

                if mb_latent_pi is None: # i.e. not using shared encoder
                    icm_total_loss, icm_i_loss, icm_f_loss, _, _ = self.icm.update(
                        obs_batch=mb_obs[:-1],
                        action_batch=mb_actions,
                        next_obs_batch=mb_obs[1:],
                        hidden_state_obs=eval_hidden_state[:-1] if eval_hidden_state else None,
                        hidden_state_next_obs=eval_hidden_state[1:] if eval_hidden_state else None,
                    )
                    mean_icm_loss += icm_total_loss.item()
                    mean_icm_i_loss += icm_i_loss.item()
                    mean_icm_f_loss += icm_f_loss.item()

                mean_loss += loss.item()
                num_minibatches_this_epoch +=1
            
            total_minibatch_updates_this_run += num_minibatches_this_epoch
            if self.target_kl is not None and np.mean(approx_kl_divs) > self.target_kl:
                break # Early stopping for the epoch if KL divergence target is exceeded

        mean_loss = mean_loss / max(1e-8, total_minibatch_updates_this_run)
        return mean_loss

    # ------------------------------------------------------------------
    # New BPTT learning logic
    # ------------------------------------------------------------------

    def _learn_from_rollout_buffer_bptt(self) -> float:
        """Learning procedure using truncated BPTT for recurrent networks."""
        seq_len = self.max_seq_len

        buffer_actual_size = (
            self.rollout_buffer.capacity
            if self.rollout_buffer.full
            else self.rollout_buffer.pos
        )
        if buffer_actual_size < seq_len:
            warnings.warn(
                f"Buffer size {buffer_actual_size} is less than seq_len {seq_len}. Skipping BPTT learning step."
            )
            return 0.0

        # Normalize advantages globally once before epochs
        valid_advantages_tensor = self.rollout_buffer.buffer["advantages"][:buffer_actual_size]
        if valid_advantages_tensor.numel() > 0:
            original_shape = valid_advantages_tensor.shape
            flat_adv = valid_advantages_tensor.reshape(-1)
            normalized_flat_adv = (flat_adv - flat_adv.mean()) / (flat_adv.std() + 1e-8)
            self.rollout_buffer.buffer["advantages"][:buffer_actual_size] = normalized_flat_adv.reshape(original_shape)
        else:
            warnings.warn("No advantages to normalize in BPTT pre-normalization step.")

        # Determine all possible start coordinates for sequences
        num_possible_starts_per_env = buffer_actual_size - seq_len + 1
        if num_possible_starts_per_env <= 0:
            warnings.warn(
                f"Not enough data in buffer ({buffer_actual_size} steps) to form sequences of length {seq_len}. Skipping BPTT."
            )
            return 0.0

        all_start_coords = []  # List of (env_idx, time_idx_in_env_rollout)
        if self.bptt_sequence_type == BPTTSequenceType.CHUNKED:
            num_chunks_per_env = buffer_actual_size // seq_len
            if num_chunks_per_env == 0:
                warnings.warn(f"Not enough data for any full chunks of length {seq_len}. Skipping BPTT.")
                return 0.0
            for env_idx in range(self.num_envs):
                for chunk_i in range(num_chunks_per_env):
                    all_start_coords.append((env_idx, chunk_i * seq_len))
        elif self.bptt_sequence_type == BPTTSequenceType.MAXIMUM:
            for env_idx in range(self.num_envs):
                for t_idx in range(num_possible_starts_per_env):
                    all_start_coords.append((env_idx, t_idx))
        elif self.bptt_sequence_type == BPTTSequenceType.FIFTY_PERCENT_OVERLAP:
            step_size = seq_len // 2
            if step_size == 0: # Fallback for seq_len=1
                warnings.warn(f"Sequence length {seq_len} too short for 50% overlap. Using CHUNKED behavior.")
                num_chunks_per_env = buffer_actual_size // seq_len
                if num_chunks_per_env == 0: return 0.0 # Not enough for even one chunk
                for env_idx in range(self.num_envs):
                    for chunk_i in range(num_chunks_per_env): all_start_coords.append((env_idx, chunk_i * seq_len))
            else:
                for env_idx in range(self.num_envs):
                    for time_idx in range(0, buffer_actual_size - seq_len + 1, step_size):
                        all_start_coords.append((env_idx, time_idx))
        else:
            raise ValueError(f"Unknown BPTTSequenceType: {self.bptt_sequence_type}")

        if not all_start_coords:
            warnings.warn("No BPTT sequences to sample. Skipping learning.")
            return 0.0
        
        sequences_per_minibatch = self.batch_size # Here, batch_size means number of sequences per minibatch
        mean_loss = 0.0
        total_minibatch_updates_total = 0

        for epoch in range(self.update_epochs):
            approx_kl_divs_epoch = [] # KL divergences for this epoch's minibatches
            np.random.shuffle(all_start_coords)
            num_minibatches_this_epoch = 0

            for i in range(0, len(all_start_coords), sequences_per_minibatch):
                current_coords_minibatch = all_start_coords[i : i + sequences_per_minibatch]
                if not current_coords_minibatch: continue

                # Fetch minibatch of sequences; returns TensorDict on self.device
                # Batch_size: [len(current_coords_minibatch), seq_len]
                # "initial_hidden_states" is a non-tensor entry in TD: Dict[str, Tensor(batch_seq_size, layers, size)]
                current_minibatch_td = self.rollout_buffer.get_specific_sequences_tensor_batch(
                    seq_len=seq_len, sequence_coords=current_coords_minibatch, device=self.device
                )

                if current_minibatch_td.is_empty() or "observations" not in current_minibatch_td.keys(include_nested=True, leaves_only=True):
                    warnings.warn("Skipping empty or invalid minibatch of sequences.")
                    continue

                mb_obs_seq = current_minibatch_td["observations"]      # Shape: (batch_seq, seq_len, *obs_dims) or nested TD
                mb_actions_seq = current_minibatch_td["actions"]    # Shape: (batch_seq, seq_len, *act_dims)
                mb_old_log_probs_seq = current_minibatch_td["log_probs"] # Shape: (batch_seq, seq_len)
                mb_advantages_seq = current_minibatch_td["advantages"]# Shape: (batch_seq, seq_len) (already normalized)
                mb_returns_seq = current_minibatch_td["returns"]      # Shape: (batch_seq, seq_len)
                mb_latent_pi = current_minibatch_td.get("encoder_out", None)

                mb_initial_hidden_states_dict = current_minibatch_td.get_non_tensor("initial_hidden_states", default=None)

                current_step_hidden_state_actor = None # For actor: {key: (layers, batch_seq_size, hidden_size)}
                approx_kl_divs_minibatch_timesteps = []


                if self.recurrent and mb_initial_hidden_states_dict is not None:
                    current_step_hidden_state_actor = {
                        # val is (batch_seq_size, layers, size), permute to (layers, batch_seq_size, size)
                        key: val.permute(1, 0, 2).contiguous().to(self.device)
                        for key, val in mb_initial_hidden_states_dict.items()
                    }

                loss_dict = self.compute_loss(mb_obs_seq, mb_actions_seq, mb_old_log_probs_seq, mb_advantages_seq, mb_returns_seq, mb_latent_pi, current_step_hidden_state_actor, seq_len=seq_len, learn_by_bptt=True)
                approx_kl_divs_minibatch_timesteps.append(loss_dict['approx_kl'])
                loss = loss_dict['loss']

                if self.use_shared_encoder_for_icm:
                    loss = self.icm_loss_weight * loss_dict['icm_total_loss'] + (1 - self.icm_loss_weight) * loss

                self.optimizer.zero_grad()
                loss.backward()  # Gradients accumulate over the sequence within this backward call
                clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                if self.use_shared_encoder_for_icm:
                    clip_grad_norm_(self.icm.parameters(), self.max_grad_norm)
                self.optimizer.step()


                if mb_latent_pi is None: # i.e. not using shared encoder
                    icm_total_loss, icm_i_loss, icm_f_loss, _, _ = self.icm.update(
                        obs_batch=mb_obs_seq[:, :-1],
                        action_batch=mb_actions_seq[:, :-1],
                        next_obs_batch=mb_obs_seq[:, 1:],
                        hidden_state_obs=mb_initial_hidden_states_dict,
                        hidden_state_next_obs=mb_initial_hidden_states_dict,
                    )
                    mean_icm_loss += icm_total_loss.item()
                    mean_icm_i_loss += icm_i_loss.item()
                    mean_icm_f_loss += icm_f_loss.item()

                mean_loss += loss.item()
                num_minibatches_this_epoch +=1
                
                if self.target_kl is not None and len(approx_kl_divs_minibatch_timesteps) > 0:
                    # Average KL over all timesteps in this minibatch of sequences
                    kl_for_current_minibatch = np.mean(approx_kl_divs_minibatch_timesteps)
                    approx_kl_divs_epoch.append(kl_for_current_minibatch) # Store minibatch average KL

                    if kl_for_current_minibatch > self.target_kl:
                        warnings.warn(
                            f"Epoch {epoch}, Minibatch: KL divergence {kl_for_current_minibatch:.4f} exceeded target {self.target_kl}. Stopping update for this epoch."
                        )
                        break # Break from minibatch loop for this epoch
            
            total_minibatch_updates_total += num_minibatches_this_epoch
            # Check average KL for the epoch if target_kl is set and the inner loop wasn't broken by KL
            if self.target_kl is not None and len(approx_kl_divs_epoch) > 0:
                avg_kl_this_epoch = np.mean(approx_kl_divs_epoch)
                if avg_kl_this_epoch > self.target_kl and not ( # Ensure this wasn't the break from inner loop
                    len(approx_kl_divs_minibatch_timesteps) > 0 and np.mean(approx_kl_divs_minibatch_timesteps) > self.target_kl
                ):
                    warnings.warn(
                        f"Epoch {epoch}: Average KL divergence {avg_kl_this_epoch:.4f} exceeded target {self.target_kl} after completing epoch. Consider adjusting learning rate or target_kl."
                    )
                    # This break is for the epoch loop if KL was exceeded on average for the epoch
                    # but not necessarily in the last minibatch that would have broken the inner loop.
                    break 
            
            # If inner loop broke due to KL, this outer break also executes
            if self.target_kl is not None and len(approx_kl_divs_minibatch_timesteps) > 0 and np.mean(approx_kl_divs_minibatch_timesteps) > self.target_kl:
                break


        mean_loss = mean_loss / max(1e-8, total_minibatch_updates_total)
        return mean_loss

    def compute_loss(self, obs, actions, old_log_probs, advantages, returns, latent_pi=None, hidden_state=None, **kwargs) -> Dict[str, Any]: 
        """Compute the loss for the actor, critic, and ICM networks.
        
        :param obs: The observation at time t
        :type obs: ArrayOrTensor
        :param actions: The action taken at time t
        :type actions: ArrayOrTensor
        """       
        learn_by_bptt = kwargs.get('learn_by_bptt', False)
        seq_len = kwargs.get('seq_len', None)

        if learn_by_bptt:
            if seq_len is None:
                seq_len = self.max_seq_len
            
            # Common BPTT processing logic for the current minibatch
            policy_loss_total = 0.0
            value_loss_total = 0.0
            entropy_loss_total = 0.0
            icm_total_loss = 0.0
            icm_i_loss = 0.0
            icm_f_loss = 0.0
            approx_kl = 0.0

            if self.use_shared_encoder_for_icm:
                latent_pi = latent_pi[:, 0] if latent_pi is not None else None

            for t in range(seq_len):
                if isinstance(obs, dict):
                    obs_t = {k_obs: v_obs[:, t] for k_obs, v_obs in obs.items()}
                else:
                    obs_t = obs[:, t]

                actions_t = actions[:, t]
                old_log_prob_t = old_log_probs[:, t]
                adv_t = advantages[:, t]
                return_t = returns[:, t]

                returned_tuple = (
                    self._get_action_and_values(
                        obs_t,
                        hidden_state=hidden_state,  # Pass dict (layers, batch, size)
                        sample=False,
                    )
                )

                if self.use_shared_encoder_for_icm:
                    _, _, entropy_t, new_value_t, next_hidden_state_for_step, next_latent_pi = returned_tuple
                else:
                    _, _, entropy_t, new_value_t, next_hidden_state_for_step = returned_tuple

                new_log_prob_t = self.actor.action_log_prob(actions_t)
                if entropy_t is None:
                    entropy_t = (
                        -new_log_prob_t.mean()
                    )  # Ensure entropy_t is scalar if new_log_prob_t is not
                else:
                    entropy_t = entropy_t.mean()  # Ensure scalar

                ratio = torch.exp(new_log_prob_t - old_log_prob_t)
                policy_loss1 = -adv_t * ratio
                policy_loss2 = -adv_t * torch.clamp(
                    ratio, 1 - self.clip_coef, 1 + self.clip_coef
                )
                policy_loss = torch.max(policy_loss1, policy_loss2).mean()
                value_loss = 0.5 * ((new_value_t - return_t) ** 2).mean()
                entropy_step_loss = -entropy_t  # entropy_t is already mean

                policy_loss_total += policy_loss
                value_loss_total += value_loss
                entropy_loss_total += entropy_step_loss

                if self.use_shared_encoder_for_icm:
                    icm_total_loss_step, icm_i_loss_step, icm_f_loss_step, _, _ = self.icm.compute_loss(
                        action_batch_t=actions_t,
                        obs_batch_t=obs_t,
                        next_obs_batch_t=obs_t,
                        embedded_obs=latent_pi,
                        embedded_next_obs=next_latent_pi,
                        hidden_state=hidden_state,
                        hidden_state_next=hidden_state,
                    )
                    icm_total_loss += icm_total_loss_step
                    icm_i_loss += icm_i_loss_step
                    icm_f_loss += icm_f_loss_step

                with torch.no_grad():
                    log_ratio = new_log_prob_t - old_log_prob_t
                    approx_kl = ((torch.exp(log_ratio) - 1) - log_ratio).mean().item()

                if self.recurrent and next_hidden_state_for_step is not None:
                    hidden_state = (
                        next_hidden_state_for_step  # Already (layers, batch, size)
                    )
                    if self.use_shared_encoder_for_icm:
                        latent_pi = next_latent_pi

            # Average losses over sequence length
            policy_loss_avg_over_seq = policy_loss_total / seq_len
            value_loss_avg_over_seq = value_loss_total / seq_len
            entropy_loss_avg_over_seq = entropy_loss_total / seq_len
            icm_total_loss_avg_over_seq = icm_total_loss / seq_len
            icm_i_loss_avg_over_seq = icm_i_loss / seq_len
            icm_f_loss_avg_over_seq = icm_f_loss / seq_len

            loss = (
                policy_loss_avg_over_seq
                + self.vf_coef * value_loss_avg_over_seq
                + self.ent_coef * entropy_loss_avg_over_seq
            )

            loss = self.icm_loss_weight * icm_total_loss_avg_over_seq + (1 - self.icm_loss_weight) * loss if self.use_shared_encoder_for_icm else loss

            return {'loss': loss, 'approx_kl': approx_kl, 'icm_total_loss': icm_total_loss_avg_over_seq, 'icm_i_loss': icm_i_loss_avg_over_seq, 'icm_f_loss': icm_f_loss_avg_over_seq}
        else:
            # Get values and next hidden state (which we ignore in flat learning)
            _, _, entropy_t, new_value_t, _, latent_pi = self._get_action_and_values(
                obs,
                hidden_state=hidden_state,
                sample=False,  # No need to sample here
            )

            # Compute log prob of taken actions
            new_log_prob_t = self.actor.action_log_prob(actions)

            # Use -log_prob as entropy when head returns None (continuous actions w/ squashing)
            if entropy_t is None:
                entropy_t = -new_log_prob_t

            ratio = torch.exp(new_log_prob_t - old_log_probs)
            policy_loss1 = -advantages * ratio
            policy_loss2 = -advantages * torch.clamp(
                ratio, 1 - self.clip_coef, 1 + self.clip_coef
            )
            policy_loss = torch.max(policy_loss1, policy_loss2).mean()

            value_loss = 0.5 * ((new_value_t - returns) ** 2).mean()
            entropy_loss = -entropy_t.mean()

            loss = (
                policy_loss
                + self.vf_coef * value_loss
                + self.ent_coef * entropy_loss
            )

            with torch.no_grad():
                log_ratio = new_log_prob_t - old_log_probs
                approx_kl = ((torch.exp(log_ratio) - 1) - log_ratio).mean().item()

            if self.use_shared_encoder_for_icm:
                icm_total_loss, icm_i_loss, icm_f_loss, _, _ = self.icm.compute_loss(
                    action_batch_t=actions,
                    obs_batch_t=obs,
                    next_obs_batch_t=obs,
                    embedded_obs=latent_pi[:, :-1],
                    embedded_next_obs=latent_pi[:, 1:],
                    hidden_state=hidden_state,
                    hidden_state_next=hidden_state,
                )

            loss = self.icm_loss_weight * icm_total_loss + (1 - self.icm_loss_weight) * loss if self.use_shared_encoder_for_icm else loss

            return {'loss': loss, 'approx_kl': approx_kl, 'icm_total_loss': icm_total_loss, 'icm_i_loss': icm_i_loss, 'icm_f_loss': icm_f_loss}

    def get_intrinsic_reward(self, 
                             action_batch: Any,
                             obs_batch: Optional[Any] = None,
                             next_obs_batch: Optional[Any] = None,
                             embedded_obs: Optional[torch.Tensor] = None,
                             embedded_next_obs: Optional[torch.Tensor] = None,
                             hidden_state_obs: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                             hidden_state_next_obs: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
                             ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]], Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """Function to get the intrinsic reward from the ICM model.
        
        :param obs: The observation at time t
        :type obs: ArrayOrTensor
        :param action: The action taken at time t
        :type action: ArrayOrTensor
        """
        return self.icm.get_intrinsic_reward(action_batch, obs_batch, next_obs_batch, embedded_obs, embedded_next_obs, hidden_state_obs, hidden_state_next_obs)
    
    def test(
        self,
        env: GymEnvType,
        swap_channels: bool = False,
        max_steps: Optional[int] = None,
        loop: int = 3,
        vectorized: bool = True,
        callback: Optional[Callable[[float, Dict[str, float]], None]] = None,
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
        :type callback: Optional[Callable[[float, Dict[str, float]], None]]

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
                        returned_tuple = self.get_action(
                            obs, action_mask=action_mask, hidden_state=test_hidden_state
                        )
                        if self.use_shared_encoder_for_icm:
                            action, _, _, _, test_hidden_state, _ = returned_tuple
                        else:
                            action, _, _, _, test_hidden_state = returned_tuple
                    else:
                        returned_tuple = self.get_action(obs, action_mask=action_mask)
                        if self.use_shared_encoder_for_icm:
                            action, _, _, _, _, _ = returned_tuple
                        else:
                            action, _, _, _ = returned_tuple

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
                        initial_hidden_states_for_reset = self.get_initial_hidden_state(num_envs)
                        if isinstance(test_hidden_state, dict):
                            for key in test_hidden_state:
                                reset_states = initial_hidden_states_for_reset[key][:, newly_finished, :]
                                if reset_states.shape[1] > 0:
                                    test_hidden_state[key][:, newly_finished, :] = reset_states

                    if np.any(newly_finished):
                        completed_episode_scores[newly_finished] = scores[newly_finished]
                        finished[newly_finished] = True

                # End of episode loop for one test run
                loop_reward_sum = np.sum(completed_episode_scores)

                # Prepare info for callback
                final_info_for_callback = {}
                if vectorized:
                    if isinstance(last_infos, (list, np.ndarray)) and len(last_infos) > 0:
                        final_info_for_callback = last_infos[0] if isinstance(last_infos[0], dict) else {}
                    elif isinstance(last_infos, dict):
                        final_info_for_callback = last_infos
                else: # Not vectorized
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
 
    def share_encoder_parameters(self) -> None:
        """Shares the encoder parameters between the actor and critic."""
        if all(isinstance(net, EvolvableNetwork) for net in [self.actor, self.critic]):
            share_encoder_parameters(self.actor, self.critic)
        else:
            warnings.warn(
                "Encoder sharing is disabled as actor or critic is not an EvolvableNetwork."
            )
    def get_hidden_state_architecture(self) -> Dict[str, Tuple[int, ...]]:
        """Get the hidden state architecture for the environment.

        :return: Dictionary describing the hidden state architecture (name to shape)
        :rtype: Dict[str, Tuple[int, ...]]
        """
        return {
            k: v.shape for k, v in self.get_initial_hidden_state(self.num_envs).items()
        }

    def get_initial_hidden_state(self, num_envs: int = 1) -> Dict[str, ArrayOrTensor]:
        """Get the initial hidden state for the environment. 
        
        The hidden states are generally cached on a per Module basis.
        The reason the Cache is per Module is because the user might want to have a custom initialization for the hidden states.

        :param num_envs: Number of environments, defaults to 1
        :type num_envs: int, optional
        :return: Initial hidden state dictionary
        :rtype: Dict[str, ArrayOrTensor]
        """
        if not self.recurrent:
            raise ValueError(
                "Cannot get initial hidden state for non-recurrent networks."
            )

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

    def _get_action_and_values_icm(
        self,
        obs: ArrayOrTensor,
        action_mask: Optional[ArrayOrTensor] = None,
        hidden_state: Optional[Dict[str, ArrayOrTensor]] = None, # Hidden state is a dict for recurrent policies
        *,
        sample: bool = True,
    ) -> Tuple[
        ArrayOrTensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[Dict[str, ArrayOrTensor]]
    ]:
        """
        Returns the next action to take in the environment and the values.
        
        :param obs: Environment observation, or multiple observations in a batch
        :type obs: ArrayOrTensor
        :param action_mask: Mask of legal actions 1=legal 0=illegal, defaults to None
        :type action_mask: Optional[ArrayOrTensor]
        :param hidden_state: Hidden state for recurrent policies, defaults to None
        :type hidden_state: Optional[Dict[str, ArrayOrTensor]]
        :param sample: Whether to sample an action, defaults to True
        :type sample: bool
        :return: Action, log probability, entropy, state values, and (if recurrent) next hidden state
        :rtype: Tuple[ArrayOrTensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[Dict[str, ArrayOrTensor]]]
        """
        if hidden_state is not None:
            latent_pi, next_hidden_actor = self.actor.extract_features(
                obs, hidden_state=hidden_state
            )
            action, log_prob, entropy = self.actor.forward_head(
                latent_pi, action_mask=action_mask, sample=sample
            )

            next_hidden_combined = next_hidden_actor # Start with actor's next hidden state

            if self.share_encoders:
                values = self.critic.forward_head(latent_pi).squeeze(-1)
            else:
                # If not sharing, critic might have its own hidden state components or update existing ones
                values, next_hidden_critic = self.critic(obs, hidden_state=hidden_state) # Pass original hidden_state
                values = values.squeeze(-1)
                if next_hidden_critic is not None: # Merge if critic returns its own next_hidden
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
            raise ValueError("Experiences must be provided when use_rollout_buffer is False")
        
        # Not self.use_rollout_buffer
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
