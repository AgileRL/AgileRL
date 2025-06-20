import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from gymnasium import spaces
from typing import Optional, Dict, Tuple, Union, Any
import torch.nn.functional as F

from agilerl.modules.mlp import EvolvableMLP
from agilerl.modules.cnn import EvolvableCNN
from agilerl.modules.lstm import EvolvableLSTM
from agilerl.modules.base import EvolvableModule

# Configuration helpers
def get_evolvable_mlp_config(
    user_config: Optional[Dict] = None, default_hidden_size: Optional[list[int]] = None, default_output_dim: Optional[int] = None
) -> Dict:
    if default_hidden_size is None:
        default_hidden_size = [256, 256]
    if default_output_dim is None:
        default_output_dim = default_hidden_size[-1] if default_hidden_size else 128 # Default to last hidden or a fixed value

    config = {
        "arch": "mlp",
        "hidden_size": default_hidden_size, # List of hidden layer sizes
        "output_dim": default_output_dim,   # Output dimension of the MLP itself
        "activation": "ReLU",
        "output_activation": None, # Usually None for features
        "layer_norm": False,
        "spectral_norm": False,
        "dropout": 0.0,
    }
    if user_config:
        config.update(user_config)
    return config

def get_evolvable_cnn_config(user_config: Optional[Dict] = None, default_fc_layer_dims: Optional[list[int]] = None) -> Dict:
    if default_fc_layer_dims is None:
        default_fc_layer_dims = [512]
    config = {
        "arch": "cnn",
        "channel_size": [32, 64, 64],
        "kernel_size": [8, 4, 3],
        "stride_size": [4, 2, 1],
        "num_outputs": default_fc_layer_dims, # Dense layer sizes after CNN layers
        "activation": "ReLU",
        "output_activation": None,
        "layer_norm": False,
        # input_shape will be set/inferred
    }
    if user_config:
        config.update(user_config)
    return config

def get_evolvable_lstm_config(user_config: Optional[Dict] = None, default_hidden_state_size: int = 256) -> Dict:
    config = {
        "arch": "lstm",
        "hidden_state_size": default_hidden_state_size, 
        "num_recurrent_layers": 1,
        "bidirectional": False # Conceptual, affects output_dim calculation
        # num_outputs for EvolvableLSTM will be hidden_state_size * (2 if bidirectional else 1)
    }
    if user_config:
        config.update(user_config)
    return config

class ICMFeatureEncoder(EvolvableModule):
    def __init__(self, 
                 observation_space: spaces.Space,
                 encoder_net_config: Dict, 
                 device: Union[torch.device, str] = 'cpu'):
        super().__init__(device=device) # Pass device to EvolvableModule
        self.observation_space = observation_space
        self.encoder_net_config = encoder_net_config
        self.is_recurrent = encoder_net_config.get('recurrent', False)
        self.arch = encoder_net_config.get('arch', 'mlp').lower()

        base_feature_dim_pre_lstm: int

        if self.arch == 'cnn':
            if not isinstance(observation_space, spaces.Box) or len(observation_space.shape) not in [3, 4]:
                raise ValueError("CNN encoder requires Box observation space with 3 (e.g. C,H,W or H,W,C) or 4 (e.g. N,C,H,W or N,H,W,C) dims.")
            
            obs_shape = observation_space.shape
            if len(obs_shape) == 3: # (C,H,W) or (H,W,C)
                # EvolvableCNN expects channels-first (C, H, W)
                if obs_shape[0] in [1, 3, 4]: input_channel, H, W = obs_shape[0], obs_shape[1], obs_shape[2]
                elif obs_shape[2] in [1, 3, 4]: H, W, input_channel = obs_shape[0], obs_shape[1], obs_shape[2]
                else: raise ValueError(f"Unsupported image shape: {obs_shape}. Can't infer C,H,W ordering.")
                self.input_shape = (input_channel, H, W)
            elif len(obs_shape) == 4: # Assuming (N,C,H,W) or (N,H,W,C)
                if obs_shape[1] in [1, 3, 4]: input_channel, H, W = obs_shape[1], obs_shape[2], obs_shape[3]
                elif obs_shape[3] in [1, 3, 4]: H, W, input_channel = obs_shape[1], obs_shape[2], obs_shape[3]
                else: raise ValueError(f"Unsupported batched image shape: {obs_shape}.")
                self.input_shape = (input_channel, H, W) # EvolvableCNN takes single image shape
            else:
                raise ValueError(f"Unsupported observation space shape for CNN: {obs_shape}")

            cnn_config_user = encoder_net_config.get('cnn_config', {})
            cnn_config = get_evolvable_cnn_config(cnn_config_user)
            
            # The num_outputs of EvolvableCNN is its feature dimension
            # If cnn_config includes 'num_outputs', use it. Otherwise, it's inferred by EvolvableCNN from fc_layer_dims.
            num_outputs_cnn = cnn_config.get('num_outputs') 

            self.base_encoder = EvolvableCNN(
                input_shape=self.input_shape,
                channel_size=cnn_config['channel_size'],
                kernel_size=cnn_config['kernel_size'],
                stride_size=cnn_config['stride_size'],
                num_outputs=num_outputs_cnn, # Can be None, EvolvableCNN infers from fc_layer_dims
                activation=cnn_config['activation'],
                output_activation=cnn_config['output_activation'], # Usually None for features
                layer_norm=cnn_config['layer_norm'],
                device=self.device
            )
            base_feature_dim_pre_lstm = self.base_encoder.output_dim
        
        elif self.arch == 'mlp':
            input_dim = int(np.prod(observation_space.shape))
            mlp_config_user = encoder_net_config.get('mlp_config', {})
            # Provide a default output_dim for the MLP part if not in user_config
            mlp_default_output_dim = mlp_config_user.get('hidden_size')[-1] if mlp_config_user.get('hidden_size') else 128
            mlp_config = get_evolvable_mlp_config(mlp_config_user, default_output_dim=mlp_default_output_dim)
            
            base_feature_dim_pre_lstm = mlp_config['output_dim']
            self.base_encoder = EvolvableMLP(
                num_inputs=input_dim,
                hidden_size=mlp_config['hidden_size'],
                num_outputs=base_feature_dim_pre_lstm, 
                activation=mlp_config['activation'],
                output_activation=mlp_config['output_activation'], # Usually None for features
                layer_norm=mlp_config['layer_norm'],
                device=self.device
            )
        else:
            raise ValueError(f"Unsupported encoder architecture: {self.arch}. Choose 'cnn' or 'mlp'.")

        self._output_feature_dim = base_feature_dim_pre_lstm
        self.lstm = None
        if self.is_recurrent:
            lstm_config_user = encoder_net_config.get('lstm_config', {})
            lstm_config = get_evolvable_lstm_config(lstm_config_user)
            
            # num_outputs for EvolvableLSTM is its actual output size per step
            lstm_num_outputs = lstm_config['hidden_state_size'] * (2 if lstm_config.get('bidirectional') else 1)

            self.lstm = EvolvableLSTM(
                input_size=base_feature_dim_pre_lstm,
                hidden_state_size=lstm_config['hidden_state_size'],
                num_layers=lstm_config['num_recurrent_layers'],
                num_outputs=lstm_num_outputs, 
                output_activation=None, # Usually None for features
                device=self.device
            )
            self._output_feature_dim = lstm_num_outputs

    @property
    def output_dim(self) -> int: # Match EvolvableModule property name
        return self._output_feature_dim

    def forward(self, obs: torch.Tensor, hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) \
        -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        
        if self.arch == 'mlp':
            obs = obs.reshape(obs.size(0), -1) 
        elif self.arch == 'cnn':
            # Ensure obs is channels-first for EvolvableCNN if not using normalize_inputs
            # If normalize_inputs=True in EvolvableCNN, it handles HWC internally.
            # Otherwise, input here should be BCHW.
            if not self.base_encoder.normalize_inputs: # type: ignore
                if obs.dim() == 3: # B,H,W (assuming C=1) or H,W,C - needs to be B,C,H,W
                    # This part is tricky without knowing exact input format vs. base_encoder expectations
                    pass # Assume preprocessor handles this to B,C,H,W if normalize_inputs=False
                if obs.dim() == 4 and obs.size(3) == self.input_shape[0]: # B,H,W,C -> B,C,H,W
                    obs = obs.permute(0, 3, 1, 2)
        
        features = self.base_encoder(obs)
        next_hidden_state = None

        if self.is_recurrent and self.lstm is not None:
            if hidden_state is None:
                batch_size = features.size(0)
                # EvolvableLSTM.init_hidden returns (h, c) for LSTM
                h_init, c_init = self.lstm.init_hidden(batch_size)
                hidden_state = (h_init.to(self.device), c_init.to(self.device))
            
            features = features.unsqueeze(1) 
            features, next_hidden_state = self.lstm(features, hidden_state)
            features = features.squeeze(1) 
        
        return features, next_hidden_state
    
    def get_init_dict(self):
        """Returns the initialization dictionary for the module."""
        init_dict = {
            "observation_space": self.observation_space,
            "encoder_net_config": self.encoder_net_config,
            "device": self.device
        }
        return init_dict

    def get_output_dim(self):
         return self.output_dim # Required by EvolvableModule if not using `num_outputs` directly in constructor

    def clone(self):
        """Returns a deep copy of the module."""
        clone = type(self)(**self.get_init_dict())
        clone.load_state_dict(self.state_dict())
        return clone

class ICMInverseModel(EvolvableMLP):
    def __init__(self, 
                 feature_dim: int, 
                 action_space: Union[spaces.Discrete, spaces.MultiDiscrete, spaces.Box], 
                 net_config: Optional[Dict] = None, 
                 device: Union[torch.device, str] = 'cpu'):
        input_dim = 2 * feature_dim
        action_dim = get_action_dim(action_space)
        
        # Store action space for loss computation
        self.action_space = action_space
        self.action_sizes = get_action_sizes(action_space)
        self.is_continuous = is_continuous_action_space(action_space)
        
        # Use feature_dim for default hidden layer sizes if not provided
        default_hidden = [feature_dim, feature_dim] if feature_dim > 0 else [256, 256]
        resolved_config = get_evolvable_mlp_config(net_config, default_hidden_size=default_hidden, default_output_dim=action_dim)
        
        # For continuous actions, we might want a different output activation
        output_activation = None if self.is_continuous else None  # Logits for discrete, raw for continuous
        
        super().__init__(
            num_inputs=input_dim,
            hidden_size=resolved_config['hidden_size'],
            num_outputs=action_dim, # Output is action logits (discrete) or action values (continuous)
            activation=resolved_config['activation'],
            output_activation=output_activation,
            layer_norm=resolved_config['layer_norm'],
            device=device
        )

    def forward(self, phi_state: torch.Tensor, phi_next_state: torch.Tensor) -> torch.Tensor:
        x = torch.cat([phi_state, phi_next_state], dim=1)
        return super().forward(x)

class ICMForwardModel(EvolvableMLP):
    def __init__(self, 
                 feature_dim: int, 
                 action_space: Union[spaces.Discrete, spaces.MultiDiscrete, spaces.Box], 
                 net_config: Optional[Dict] = None, 
                 device: Union[torch.device, str] = 'cpu'):
        input_dim = feature_dim + get_action_dim(action_space)
        
        default_hidden = [feature_dim, feature_dim] if feature_dim > 0 else [256, 256]
        resolved_config = get_evolvable_mlp_config(net_config, default_hidden_size=default_hidden, default_output_dim=feature_dim)
        
        super().__init__(
            num_inputs=input_dim,
            hidden_size=resolved_config['hidden_size'],
            num_outputs=feature_dim, # Output is predicted next state features
            activation=resolved_config['activation'],
            output_activation=None, # Raw features
            layer_norm=resolved_config['layer_norm'],
            device=device
        )

    def forward(self, phi_state: torch.Tensor, action_input: torch.Tensor) -> torch.Tensor:
        x = torch.cat([phi_state, action_input], dim=1)
        return super().forward(x)

# Helper functions for action space handling
def get_action_dim(action_space):
    """Get total action dimension for Discrete, MultiDiscrete, and Box spaces."""
    if isinstance(action_space, spaces.Discrete):
        return action_space.n
    elif isinstance(action_space, spaces.MultiDiscrete):
        return int(action_space.nvec.sum())
    elif isinstance(action_space, spaces.Box):
        return int(np.prod(action_space.shape))
    else:
        raise ValueError(f"Unsupported action space type: {type(action_space)}")

def get_action_sizes(action_space):
    """Get list of action sizes for each action dimension."""
    if isinstance(action_space, spaces.Discrete):
        return [action_space.n]
    elif isinstance(action_space, spaces.MultiDiscrete):
        return action_space.nvec.tolist()
    elif isinstance(action_space, spaces.Box):
        return list(action_space.shape)
    else:
        raise ValueError(f"Unsupported action space type: {type(action_space)}")

def actions_to_one_hot(actions, action_space):
    """Convert actions to one-hot encoding for discrete spaces, or normalize for continuous spaces."""
    if isinstance(action_space, spaces.Discrete):
        return F.one_hot(actions.long(), num_classes=action_space.n).float()
    elif isinstance(action_space, spaces.MultiDiscrete):
        # actions shape: (batch_size, num_action_dims)
        one_hots = []
        for i, n_actions in enumerate(action_space.nvec):
            one_hot = F.one_hot(actions[:, i].long(), num_classes=n_actions).float()
            one_hots.append(one_hot)
        return torch.cat(one_hots, dim=1)
    elif isinstance(action_space, spaces.Box):
        # For continuous actions, just return the actions as float tensors
        # Optionally normalize to [0, 1] range if action space has bounds
        actions_float = actions.float()
        if action_space.is_bounded():
            # Normalize to [0, 1] range
            low = torch.tensor(action_space.low, device=actions.device, dtype=torch.float32)
            high = torch.tensor(action_space.high, device=actions.device, dtype=torch.float32)
            actions_float = (actions_float - low) / (high - low)
        return actions_float
    else:
        raise ValueError(f"Unsupported action space type: {type(action_space)}")

def is_continuous_action_space(action_space):
    """Check if the action space is continuous."""
    return isinstance(action_space, spaces.Box)

class ICM(EvolvableModule):
    def __init__(self,
                 observation_space: spaces.Space,
                 action_space: spaces.Space,
                 use_internal_encoder: bool = False,
                 encoder_net_config: Optional[Dict] = None, 
                 feature_dim: int = None,
                 lr: float = 1e-4,
                 beta: float = 0.2,
                 intrinsic_reward_weight: float = 0.01,
                 inverse_net_config: Optional[Dict] = None,
                 forward_net_config: Optional[Dict] = None,
                 device: Union[torch.device, str] = 'cpu',
                 index: int = 0, 
                 hp_config: Optional[Dict] = None, 
                 accelerator: Optional[Any] = None 
                 ): 
        super().__init__(device=device)
        
        if use_internal_encoder and encoder_net_config is None:
            raise ValueError("encoder_net_config must be provided if use_internal_encoder is True.")

        if hp_config is not None:
            self.hps = hp_config 
        else:
            self.hps = {} 

        self.lr = self.hps.get('lr', lr) 
        self.beta = self.hps.get('beta', beta)
        self.intrinsic_reward_weight = self.hps.get('intrinsic_reward_weight', intrinsic_reward_weight)
        self.encoder_net_config = encoder_net_config
        self.feature_dim = feature_dim
        self.inverse_net_config = inverse_net_config
        self.forward_net_config = forward_net_config
        self.index = index
        self.accelerator = accelerator
        self.use_internal_encoder = use_internal_encoder

        if not isinstance(action_space, (spaces.Discrete, spaces.MultiDiscrete, spaces.Box)):
            raise ValueError("This ICM implementation currently supports Discrete, MultiDiscrete, and Box action spaces only.")

        self.observation_space = observation_space
        self.action_space = action_space
        self.is_continuous_action = is_continuous_action_space(action_space)
        
        self.encoder = None
        if self.use_internal_encoder:
            self.encoder = ICMFeatureEncoder(observation_space, encoder_net_config, self.device)
            self.is_recurrent = self.encoder.is_recurrent # Get from internally created encoder
        else: # No internal encoder, will rely on external embeddings.
            # Try to infer recurrence from config if provided, otherwise assume not recurrent.
            self.is_recurrent = encoder_net_config.get('recurrent', False) if encoder_net_config else False
        
        if feature_dim is not None:
            actual_feature_dim = feature_dim
        else:
            if self.use_internal_encoder:
                actual_feature_dim = self.encoder.output_dim
            else:
                # This is a critical path. If no feature_dim is provided and no internal encoder exists,
                # we cannot infer the dimension for the forward/inverse models.
                raise ValueError("`feature_dim` must be provided when `use_internal_encoder` is False.")

        self.inverse_model = ICMInverseModel(actual_feature_dim, action_space, inverse_net_config, self.device)
        self.forward_model = ICMForwardModel(actual_feature_dim, action_space, forward_net_config, self.device)

        params_to_optimize = []
        if self.use_internal_encoder and self.encoder is not None:
            params_to_optimize.append(self.encoder) # TODO: check if this is correct
        
        params_to_optimize.append(self.inverse_model)
        params_to_optimize.append(self.forward_model) # TODO: check if this is correct
        
        self.optimizer = None
        if params_to_optimize:
            # self.optimizer = optim.Adam(params_to_optimize, lr=self.lr)
            self.params_to_optimize = params_to_optimize
        else:
            # This can happen if use_internal_encoder is False and somehow inverse/forward models have no params.
            print("Warning: ICM has no parameters to optimize. Ensure this is intended.")

        self.mse_loss_fn = nn.MSELoss(reduction='none') 
        self.ce_loss_fn = nn.CrossEntropyLoss()

    def _to_tensor(self, data: Any, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        # Default to float32 if dtype is not specified, unless it's already a tensor
        if not isinstance(data, torch.Tensor):
            current_dtype = dtype if dtype is not None else torch.float32
            data = torch.tensor(data, device=self.device, dtype=current_dtype)
        elif data.device != self.device:
            data = data.to(self.device)
        # If already a tensor on the correct device, and dtype is specified, convert if necessary
        if isinstance(data, torch.Tensor) and dtype is not None and data.dtype != dtype:
            data = data.to(dtype)
        return data

    def embed_obs(self,
                obs_batch: torch.Tensor = None,
                next_obs_batch: torch.Tensor = None,
                embedded_obs: Optional[torch.Tensor] = None,
                embedded_next_obs: Optional[torch.Tensor] = None,
                hidden_state_obs: Optional[Tuple[torch.Tensor,torch.Tensor]] = None,
                hidden_state_next_obs: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[torch.Tensor, torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]], Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        hidden_state_t = None  # Initialize
        next_hidden_state_t = None  # Initialize
        if not self.use_internal_encoder:
            if embedded_obs is None or embedded_next_obs is None:
                raise ValueError("embedded_obs and embedded_next_obs must be provided when using an external encoder.")
            phi_obs = torch.as_tensor(embedded_obs).to(self.device)
            phi_next_obs = torch.as_tensor(embedded_next_obs).to(self.device)
        else:
            if obs_batch is None or next_obs_batch is None:
                raise ValueError("obs_batch and next_obs_batch must be provided if using internal encoder.")
            if self.encoder is None:
                 raise RuntimeError("ICM is configured to use an internal encoder, but it has not been initialized.")
                
            obs_batch_t = self._to_tensor(obs_batch) # Default to float32
            next_obs_batch_t = self._to_tensor(next_obs_batch) # Default to float32
    
            with torch.no_grad():
                phi_obs, hidden_state_t = self.encoder(obs_batch_t, hidden_state_obs if self.is_recurrent else None)
                phi_next_obs, next_hidden_state_t = self.encoder(next_obs_batch_t, hidden_state_next_obs if self.is_recurrent else None)      

        return phi_obs, phi_next_obs, hidden_state_t if self.is_recurrent else None, next_hidden_state_t if self.is_recurrent else None

    def get_intrinsic_reward(self, 
                             action_batch: torch.Tensor,
                             obs_batch: Optional[torch.Tensor] = None,
                             next_obs_batch: Optional[torch.Tensor] = None,
                             embedded_obs: Optional[torch.Tensor] = None,
                             embedded_next_obs: Optional[torch.Tensor] = None,
                             hidden_state_obs: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                             hidden_state_next_obs: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
                             ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]], Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        phi_obs, phi_next_obs, hidden_state, next_hidden_state = self.embed_obs(
            obs_batch,
            next_obs_batch,
            embedded_obs,
            embedded_next_obs,
            hidden_state_obs,
            hidden_state_next_obs
        )

        # Use appropriate dtype based on action space type
        dtype = torch.float32 if self.is_continuous_action else torch.long
        action_batch_t = self._to_tensor(action_batch, dtype=dtype)
        action_input = actions_to_one_hot(action_batch_t, self.action_space)

        with torch.no_grad():
            pred_phi_next_obs = self.forward_model(phi_obs, action_input)
        
        mse_per_feature = self.mse_loss_fn(pred_phi_next_obs, phi_next_obs)
        intrinsic_reward = 0.5 * mse_per_feature.sum(dim=1)
        intrinsic_reward *= self.intrinsic_reward_weight 
            
        returned_hidden_obs = hidden_state if not self.is_recurrent else None
        returned_hidden_next_obs = next_hidden_state if not self.is_recurrent else None
        return intrinsic_reward, returned_hidden_obs, returned_hidden_next_obs

    def update(self, 
               obs_batch: Any, 
               action_batch: Any, 
               next_obs_batch: Any,
               hidden_state_obs: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
               hidden_state_next_obs: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
               ) -> Tuple[float, float, float, Optional[Tuple[torch.Tensor, torch.Tensor]], Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        obs_batch_t = self._to_tensor(obs_batch)
        next_obs_batch_t = self._to_tensor(next_obs_batch)
        
        # Use appropriate dtype based on action space type
        dtype = torch.float32 if self.is_continuous_action else torch.long
        action_batch_t = self._to_tensor(action_batch, dtype=dtype)

        action_input = actions_to_one_hot(action_batch_t, self.action_space)

        total_loss, loss_I, loss_F, returned_hidden_obs, returned_hidden_next_obs = self.compute_loss(
            obs_batch_t,
            action_batch_t,
            next_obs_batch_t,
            hidden_state_obs,
            hidden_state_next_obs,
            action_input
        )
        
        if self.optimizer is None:
            return total_loss, loss_I, loss_F, returned_hidden_obs, returned_hidden_next_obs

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return total_loss, loss_I, loss_F, returned_hidden_obs, returned_hidden_next_obs

    def compute_loss(self,
                     action_batch_t: torch.Tensor,
                     obs_batch_t: torch.Tensor,
                     next_obs_batch_t: torch.Tensor,
                     embedded_obs: Optional[torch.Tensor] = None,
                     embedded_next_obs: Optional[torch.Tensor] = None,
                     hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                     hidden_state_next: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
        # === Encoder pass ===
        # Gradients need to flow through phi_obs and phi_next_obs to train the encoder
        phi_obs, phi_next_obs, hidden_state_after_encoder_obs, hidden_state_after_encoder_next_obs = self.embed_obs(
            obs_batch_t,
            next_obs_batch_t,
            embedded_obs,
            embedded_next_obs,
            hidden_state,
            hidden_state_next
        )

        # === Inverse Model Loss (L_I) ===
        # L_I trains the inverse model and the encoder (via phi_obs and phi_next_obs)
        pred_action_output = self.inverse_model(phi_obs, phi_next_obs) 
        
        if self.is_continuous_action:
            # For continuous actions, use MSE loss
            target_actions = actions_to_one_hot(action_batch_t, self.action_space)
            loss_I = self.mse_loss_fn(pred_action_output, target_actions).mean()
            targets = target_actions  # For forward model input
        else:
            # For discrete actions, use cross-entropy loss
            targets = torch.zeros_like(pred_action_output)
            if isinstance(self.action_space, spaces.Discrete):
                targets = F.one_hot(action_batch_t.long(), num_classes=self.action_space.n).float()
                loss_I = self.ce_loss_fn(pred_action_output, targets)
            elif isinstance(self.action_space, spaces.MultiDiscrete):
                # For MultiDiscrete, compute loss for each action dimension
                losses_I = []
                start_idx = 0
                for i, action_size in enumerate(self.inverse_model.action_sizes):
                    end_idx = start_idx + action_size
                    logits_i = pred_action_output[:, start_idx:end_idx]
                    targets_i = (
                        F.one_hot(action_batch_t[:, i].long(), num_classes=action_size).float()
                        if action_batch_t.dim() > 1
                        else F.one_hot(action_batch_t[i].long(), num_classes=action_size).float()
                    )
                    targets[:, start_idx:end_idx] = targets_i
                    loss_i = self.ce_loss_fn(logits_i, targets_i)
                    losses_I.append(loss_i)
                    start_idx = end_idx
                loss_I = torch.stack(losses_I).mean()
            else:
                raise ValueError(f"Unsupported action space type: {type(self.action_space)}")

        # === Forward Model Loss (L_F) ===
        # L_F trains the forward model and the encoder (via phi_next_obs as target).
        # phi_obs is detached as input to the forward model, as per the paper.
        pred_phi_next_obs = self.forward_model(phi_obs.detach(), targets)
        mse_per_feature_lf = self.mse_loss_fn(pred_phi_next_obs, phi_next_obs) 
        loss_F_per_sample = 0.5 * mse_per_feature_lf.sum(dim=1) 
        loss_F = loss_F_per_sample.mean()

        # === Total Loss and Optimization ===
        total_loss = (1 - self.beta) * loss_I + self.beta * loss_F
        
        return total_loss, loss_I, loss_F, hidden_state_after_encoder_obs, hidden_state_after_encoder_next_obs
    
    def forward(self, x: Dict[str, Any]):
        action_batch_t = x.get('action')
        obs_batch_t = x.get('obs')
        next_obs_batch_t = x.get('next_obs')
        embedded_obs = x.get('embedded_obs')
        embedded_next_obs = x.get('embedded_obs_next')
        hidden_state = x.get('hidden_state')
        hidden_state_next = x.get('hidden_state_next')
        if not self.use_internal_encoder:
            # When using shared encoder, the main algorithm should handle the update by calling compute_loss
            # and using the returned loss to update the shared components (encoder, inv/fwd models).
            # This forward pass is for when ICM is a standalone component whose loss needs to be calculated.
            return self.compute_loss(action_batch_t, obs_batch_t, next_obs_batch_t, embedded_obs, embedded_next_obs, hidden_state, hidden_state_next)
        else:
            # When using its own encoder, it can perform a full update cycle.
            return self.update(obs_batch_t, action_batch_t, next_obs_batch_t, hidden_state, hidden_state_next)

    def get_init_dict(self):
        """Returns the initialization dictionary for the module."""
        init_dict = {
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "encoder_net_config": self.encoder_net_config,
            "use_internal_encoder": self.use_internal_encoder,
            "feature_dim": self.feature_dim,
            "lr": self.lr,
            "beta": self.beta,
            "intrinsic_reward_weight": self.intrinsic_reward_weight,
            "inverse_net_config": self.inverse_net_config,
            "forward_net_config": self.forward_net_config,
            "device": self.device,
            "index": self.index,
            "hp_config": self.hps,
            "accelerator": self.accelerator,
        }
        return init_dict

    def clone(self):
        """Returns a deep copy of the module."""
        clone = type(self)(**self.get_init_dict())
        clone.load_state_dict(self.state_dict())
        return clone