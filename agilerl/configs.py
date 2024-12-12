from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
import yaml



@dataclass(frozen=True)
class NetConfig:
    """Dataclass for storing evolvable network configurations."""
    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "NetConfig":
        return cls(**config)
    
    @classmethod
    def from_yaml(cls, path: str) -> "NetConfig":
        with open(path, "r") as file:
            config: Dict[str, Any] = yaml.safe_load(file)
            assert "NET_CONFIG" in config, "NET_CONFIG not found in yaml file."
            net_config = config["NET_CONFIG"]

        return cls.from_dict(net_config)

@dataclass(frozen=True)
class MlpNetConfig(NetConfig):
    hidden_size: List[int]
    activation: str = field(default="ReLU")
    output_activation: Optional[str] = field(default=None)
    min_hidden_layers: int = field(default=1)
    max_hidden_layers: int = field(default=3)
    min_mlp_nodes: int = field(default=64)
    max_mlp_nodes: int = field(default=500)
    layer_norm: bool = field(default=True)
    output_vanish: bool = field(default=True)
    init_layers: bool = field(default=True)
    noisy: bool = field(default=False)
    noise_std: float = field(default=0.5)
    device: str = field(default="cpu")

    def __post_init__(self):
        assert (
            len(self.hidden_size) >= self.min_hidden_layers, 
            "Hidden layers must be greater than min_hidden_layers."
        )
        assert (
            len(self.hidden_size) <= self.max_hidden_layers,
            "Hidden layers must be less than max_hidden_layers."
        )
        assert (
            all([self.min_mlp_nodes <= nodes 
                 and nodes <= self.max_mlp_nodes 
                 for nodes in self.hidden_size]), 
            "Nodes must be within min_nodes and max_nodes."
        )

@dataclass(frozen=True)
class CnnNetConfig(NetConfig):
    channel_size: List[int]
    kernel_size: List[int]
    stride_size: List[int]
    activation: str = field(default="ReLU")
    output_activation: Optional[str] = field(default=None)
    min_hidden_layers: int = field(default=1)
    max_hidden_layers: int = field(default=6)
    min_channel_size: int = field(default=32)
    max_channel_size: int = field(default=256)
    layer_norm: bool = field(default=True)
    output_vanish: bool = field(default=True)
    init_layers: bool = field(default=True)
    noise_std: float = field(default=0.5)
    n_agents: Optional[int] = field(default=None) # For multi-agent environments

    def __post_init__(self):
        assert (
            len(self.hidden_size) >= self.min_hidden_layers, 
            "Hidden layers must be greater than min_hidden_layers."
        )
        assert (
            len(self.hidden_size) <= self.max_hidden_layers,
            "Hidden layers must be less than max_hidden_layers."
        )
        assert (
            all([self.min_channel_size <= channels 
                 and channels <= self.max_channel_size 
                 for channels in self.channel_size]), 
            "Channels must be within min_channel_size and max_channel_size."
        )
        assert (
            all([self.min_mlp_nodes <= nodes 
                 and nodes <= self.max_mlp_nodes 
                 for nodes in self.hidden_size]), 
            "Nodes must be within min_nodes and max_nodes."
        )