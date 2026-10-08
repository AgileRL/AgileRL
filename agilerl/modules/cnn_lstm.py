# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any, Literal

import torch
from torch import nn

from agilerl.modules.base import EvolvableModule, MutationType, mutation
from agilerl.modules.configs import CnnLstmNetConfig
from agilerl.typing import ArrayOrTensor, BatchDimension, DeviceType
from agilerl.utils.evolvable_networks import create_cnn, get_activation


def _cnn_lstm_sample_input(
    input_shape: list[int],
    block_type: str,
    sample_input: torch.Tensor | None,
    device: DeviceType,
) -> torch.Tensor:
    """Build the CNN dummy input used to size the LSTM."""
    if block_type == "Conv2d":
        assert len(input_shape) == 3, (
            f"For Conv2d, input_shape should be (channels, height, width), got {input_shape}"
        )
        resolved = (
            torch.zeros(1, *input_shape, device=device)
            if sample_input is None
            else sample_input
        )
        assert len(resolved.shape) == 4, (
            f"Sample input for Conv2d must be (B, C, H, W), got shape {resolved.shape}"
        )
        return resolved.to(device)
    if block_type == "Conv3d":
        assert len(input_shape) == 3, (
            f"For Conv3d, input_shape should be (channels, height, width), got {input_shape}"
        )
        prefix = (
            "Sample input with shape format (B, C, D, H, W) must be provided for "
            "3D convolutional networks, got"
        )
        assert sample_input is not None, f"{prefix} {sample_input}."
        assert len(sample_input.shape) == 5, f"{prefix} {sample_input.shape}."
        return sample_input.to(device)
    msg = f"Invalid block type: {block_type}. Must be 'Conv2d' or 'Conv3d'."
    raise ValueError(msg)


class EvolvableCnnLstm(EvolvableModule):
    """Convolutional feature extractor followed by an LSTM for image POMDPs.

    Hidden-state keys follow :class:`~agilerl.modules.lstm.EvolvableLSTM`:
    ``{name}_h`` and ``{name}_c``. Architecture mutation is disabled until
    composite evolvable mutations exist.

    :param input_shape: Channel-first image shape ``(channels, height, width)``.
    :type input_shape: list[int]
    :param num_outputs: Encoder output dimension.
    :type num_outputs: int
    :param net_config: CNN trunk and LSTM head fields (``CnnLstmNetConfig``).
    :type net_config: CnnLstmNetConfig | dict[str, Any]
    """

    def __init__(
        self,
        input_shape: list[int],
        num_outputs: int,
        net_config: CnnLstmNetConfig | dict[str, Any],
        device: DeviceType = "cpu",
        name: str = "encoder",
        random_seed: int | None = None,
    ) -> None:
        super().__init__(device, random_seed)

        cfg = (
            net_config
            if isinstance(net_config, CnnLstmNetConfig)
            else CnnLstmNetConfig(**net_config)
        )

        assert len(cfg.kernel_size) == len(cfg.channel_size), (
            "Length of kernel size list must match channel size list."
        )
        assert len(cfg.stride_size) == len(cfg.channel_size), (
            "Length of stride size list must match channel size list."
        )
        assert num_outputs > 0, "'num_outputs' must be a positive integer."
        assert cfg.hidden_state_size > 0, (
            "'hidden_state_size' must be a positive integer."
        )
        assert cfg.num_layers > 0, "'num_layers' must be a positive integer."
        assert 0 <= cfg.dropout < 1, "'dropout' must be in [0, 1)."

        self.input_shape = input_shape
        self.name = name
        self.num_outputs = num_outputs
        self.channel_size = list(cfg.channel_size)
        self.stride_size = list(cfg.stride_size)
        self.kernel_size = list(cfg.kernel_size)
        self.hidden_state_size = cfg.hidden_state_size
        self.num_layers = cfg.num_layers
        self.block_type: Literal["Conv2d", "Conv3d"] = cfg.block_type
        self._activation = cfg.activation
        self.output_activation = cfg.output_activation
        self.min_hidden_layers = cfg.min_hidden_layers
        self.max_hidden_layers = cfg.max_hidden_layers
        self.min_channel_size = cfg.min_channel_size
        self.max_channel_size = cfg.max_channel_size
        self.layer_norm = cfg.layer_norm
        self.init_layers = cfg.init_layers
        self.min_hidden_state_size = cfg.min_hidden_state_size
        self.max_hidden_state_size = cfg.max_hidden_state_size
        self.min_layers = cfg.min_layers
        self.max_layers = cfg.max_layers
        self.dropout = cfg.dropout
        self.sample_input = _cnn_lstm_sample_input(
            input_shape,
            cfg.block_type,
            cfg.sample_input,
            device,
        )

        self.model = self.create_network()
        self.disable_mutations()

    @property
    def activation(self) -> str:
        return self._activation

    @activation.setter
    def activation(self, activation: str) -> None:
        self._activation = activation

    @property
    def net_config(self) -> dict[str, Any]:
        return {
            "channel_size": self.channel_size,
            "kernel_size": self.kernel_size,
            "stride_size": self.stride_size,
            "hidden_state_size": self.hidden_state_size,
            "sample_input": self.sample_input,
            "activation": self._activation,
            "output_activation": self.output_activation,
            "block_type": self.block_type,
            "num_layers": self.num_layers,
            "min_hidden_layers": self.min_hidden_layers,
            "max_hidden_layers": self.max_hidden_layers,
            "min_channel_size": self.min_channel_size,
            "max_channel_size": self.max_channel_size,
            "layer_norm": self.layer_norm,
            "init_layers": self.init_layers,
            "min_hidden_state_size": self.min_hidden_state_size,
            "max_hidden_state_size": self.max_hidden_state_size,
            "min_layers": self.min_layers,
            "max_layers": self.max_layers,
            "dropout": self.dropout,
        }

    @property
    def hidden_state_architecture(
        self,
    ) -> dict[str, tuple[int | type[BatchDimension], ...]]:
        return {
            "h": (self.num_layers, BatchDimension, self.hidden_state_size),
            "c": (self.num_layers, BatchDimension, self.hidden_state_size),
        }

    def create_network(self) -> nn.ModuleDict:
        """Build the CNN trunk, LSTM, and output projection."""
        net_dict = create_cnn(
            block_type=self.block_type,
            in_channels=self.input_shape[0],
            channel_size=self.channel_size,
            kernel_size=self.kernel_size,
            stride_size=self.stride_size,
            name=self.name,
            init_layers=self.init_layers,
            layer_norm=self.layer_norm,
            activation_fn=self.activation,
            device=self.device,
        )
        net_dict[f"{self.name}_flatten"] = nn.Flatten()
        cnn_trunk = nn.Sequential(net_dict)
        with torch.no_grad():
            cnn_feature_size = int(cnn_trunk(self.sample_input).shape[-1])

        model_dict = nn.ModuleDict()
        model_dict[f"{self.name}_cnn"] = cnn_trunk
        model_dict[f"{self.name}_lstm"] = nn.LSTM(
            input_size=cnn_feature_size,
            hidden_size=self.hidden_state_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout if self.num_layers > 1 else 0.0,
            device=self.device,
        )
        model_dict[f"{self.name}_lstm_output"] = nn.Linear(
            self.hidden_state_size,
            self.num_outputs,
            device=self.device,
        )
        model_dict[f"{self.name}_output_activation"] = get_activation(
            self.output_activation,
        )
        return model_dict

    def recreate_network(self) -> None:
        model = self.create_network()
        self.model = EvolvableModule.preserve_parameters(
            old_net=self.model,
            new_net=model,
        )

    @mutation(MutationType.ACTIVATION)
    def change_activation(self, activation: str, output: bool = False) -> None:
        if output:
            self.output_activation = activation
        else:
            self.activation = activation
        self.recreate_network()

    def init_weights_gaussian(
        self,
        std_coeff: float = 4,
        output_coeff: float = 4,
    ) -> None:
        """Initialise the LSTM output linear layer with a Gaussian distribution.

        :param std_coeff: Standard deviation coefficient, defaults to 4
        :param output_coeff: Unused; only the output linear layer is initialised.
        """
        output_layer = self.model[f"{self.name}_lstm_output"]
        EvolvableModule.apply_gaussian_init(output_layer, std_coeff=std_coeff)

    def _embed(self, x: torch.Tensor) -> torch.Tensor:
        cnn = self.model[f"{self.name}_cnn"]
        if x.dim() == 4:
            feat = cnn(x)
            return feat.unsqueeze(1)
        if x.dim() == 5:
            batch, steps = x.shape[:2]
            flat = x.reshape(batch * steps, *x.shape[2:])
            feat = cnn(flat)
            return feat.view(batch, steps, -1)
        msg = f"expected 4D or 5D image tensor, got {x.shape}"
        raise ValueError(msg)

    def forward(
        self,
        x: ArrayOrTensor,
        hidden_state: dict[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if hidden_state is None:
            msg = "Hidden state is required for CNN+LSTM forward pass."
            raise ValueError(msg)

        # self.device is not updated by module.to(...).
        device = next(self.parameters()).device
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32, device=device)
        else:
            x = x.to(device)

        seq = self._embed(x)
        h0 = hidden_state[f"{self.name}_h"].to(device)
        c0 = hidden_state[f"{self.name}_c"].to(device)

        sequence_input = False
        batch = seq.shape[0]
        if h0.shape[1] != batch:
            sequence_input = True
            seq = seq.view(h0.shape[1], -1, seq.shape[-1])

        lstm = self.model[f"{self.name}_lstm"]
        lstm_out, (h_n, c_n) = lstm(seq, (h0, c0))
        if sequence_input:
            out = lstm_out.reshape(-1, lstm_out.shape[-1])
        else:
            out = lstm_out.squeeze(1)

        out = self.model[f"{self.name}_lstm_output"](out)
        out = self.model[f"{self.name}_output_activation"](out)
        next_hidden = {
            **hidden_state,
            f"{self.name}_h": h_n,
            f"{self.name}_c": c_n,
        }
        return out, next_hidden
