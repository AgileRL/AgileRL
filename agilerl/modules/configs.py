from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import torch
import yaml


@dataclass
class NetConfig:
    """Dataclass for storing evolvable network configurations."""

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "NetConfig":
        return cls(**config)

    @classmethod
    def from_yaml(cls, path: str) -> "NetConfig":
        with open(path) as file:
            config: Dict[str, Any] = yaml.safe_load(file)
            assert "NET_CONFIG" in config, "NET_CONFIG not found in yaml file."
            net_config = config["NET_CONFIG"]

        return cls.from_dict(net_config)

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def __setitem__(self, key: str, value: Any) -> None:
        setattr(self, key, value)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def pop(self, key: str, default: Any = None) -> Any:
        attr = getattr(self, key, default)
        delattr(self, key)
        return attr

    def keys(self) -> List[str]:
        return list(self.__dataclass_fields__.keys())

    def values(self) -> List[Any]:
        return [getattr(self, key) for key in self.keys()]

    def items(self) -> Dict[str, Any]:
        return {key: getattr(self, key) for key in self.keys()}.items()


@dataclass
class MlpNetConfig(NetConfig):
    hidden_size: List[int]
    activation: str = field(default="ReLU")
    output_activation: Optional[str] = field(default=None)
    min_hidden_layers: int = field(default=1)
    max_hidden_layers: int = field(default=3)
    min_mlp_nodes: int = field(default=16)
    max_mlp_nodes: int = field(default=500)
    layer_norm: bool = field(default=True)
    output_vanish: bool = field(default=True)
    init_layers: bool = field(default=True)
    noisy: bool = field(default=False)
    noise_std: float = field(default=0.5)

    def __post_init__(self):
        assert (
            len(self.hidden_size) >= self.min_hidden_layers
        ), "Hidden layers must be greater than min_hidden_layers."

        assert (
            len(self.hidden_size) <= self.max_hidden_layers
        ), "Hidden layers must be less than max_hidden_layers."

        assert all(
            [
                self.min_mlp_nodes <= nodes and nodes <= self.max_mlp_nodes
                for nodes in self.hidden_size
            ]
        ), "Nodes must be within min_nodes and max_nodes."


@dataclass
class SimBaNetConfig(NetConfig):
    hidden_size: int
    num_blocks: int
    output_activation: Optional[str] = field(default=None)
    min_blocks: int = field(default=1)
    max_blocks: int = field(default=4)
    min_mlp_nodes: int = field(default=16)
    max_mlp_nodes: int = field(default=500)

    def __post_init__(self):
        assert (
            self.num_blocks >= self.min_blocks
        ), "Number of residual blocks must be greater than min_blocks."

        assert (
            self.num_blocks <= self.max_blocks
        ), "Number of residual blocks must be less than max_blocks."

        assert (
            self.min_mlp_nodes <= self.hidden_size
            and self.hidden_size <= self.max_mlp_nodes
        ), "Nodes must be within min_nodes and max_nodes."


@dataclass
class CnnNetConfig(NetConfig):
    channel_size: List[int]
    kernel_size: List[Union[int, Tuple[int, ...]]]
    stride_size: List[int]
    sample_input: Optional[torch.Tensor] = field(default=None)
    activation: str = field(default="ReLU")
    output_activation: Optional[str] = field(default=None)
    block_type: Literal["Conv2d", "Conv3d"] = field(default="Conv2d")
    min_hidden_layers: int = field(default=1)
    max_hidden_layers: int = field(default=6)
    min_channel_size: int = field(default=32)
    max_channel_size: int = field(default=256)
    layer_norm: bool = field(default=True)
    init_layers: bool = field(default=True)


@dataclass
class MultiInputNetConfig(NetConfig):
    channel_size: List[int]
    kernel_size: List[int]
    stride_size: List[int]
    latent_dim: int = 16
    cnn_block_type: Literal["Conv2d", "Conv3d"] = "Conv2d"
    sample_input: Optional[torch.Tensor] = field(default=None)
    vector_space_mlp: bool = False
    hidden_size: Optional[List[int]] = field(default=None)
    init_dicts: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    activation: str = "ReLU"
    output_activation: Optional[str] = field(default=None)
    min_hidden_layers: int = 1
    max_hidden_layers: int = 3
    min_mlp_nodes: int = 64
    max_mlp_nodes: int = 1024
    min_cnn_hidden_layers: int = 1
    max_cnn_hidden_layers: int = 6
    min_channel_size: int = 32
    max_channel_size: int = 256
    min_latent_dim: int = 8
    max_latent_dim: int = 128
    layer_norm: bool = False
    noisy: bool = False
    noise_std: float = 0.5
    init_layers: bool = True
