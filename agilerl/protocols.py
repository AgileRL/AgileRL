"""Protocol definitions for AgileRL's evolvable algorithms and neural networks.

This module contains Protocol classes and type definitions that define the interfaces
for evolvable components in the AgileRL framework. These protocols ensure type safety
and provide clear contracts for implementing evolvable algorithms, neural networks,
and optimization components.

The key protocols include:
- EvolvableAlgorithmProtocol: Interface for algorithms that can evolve through mutations
- EvolvableModuleProtocol: Interface for neural network modules that support mutations
- EvolvableNetworkProtocol: Interface for neural networks with encoder-decoder structure
- MutationMethodProtocol: Interface for mutation operations on networks
- OptimizerWrapperProtocol: Interface for optimizer management

Type aliases are provided for common types used throughout the framework.
"""

from enum import Enum
from typing import (
    Any,
    Callable,
    Generator,
    Generic,
    Iterable,
    Protocol,
    TypeVar,
    runtime_checkable,
)

import numpy as np
import torch
from accelerate import Accelerator
from torch.optim.optimizer import Optimizer

NumpyObsType = np.ndarray | dict[str, np.ndarray] | tuple[np.ndarray, ...]
TorchObsType = torch.Tensor | dict[str, torch.Tensor] | tuple[torch.Tensor, ...]
ObservationType = NumpyObsType | TorchObsType
DeviceType = str | torch.device


class MutationType(Enum):
    """Enumeration of mutation types for evolvable neural networks.

    :param LAYER: Mutation that affects network layers (add/remove layers)
    :param NODE: Mutation that affects nodes within layers (add/remove nodes)
    :param ACTIVATION: Mutation that changes activation functions
    """

    LAYER = "layer"
    NODE = "node"
    ACTIVATION = "activation"


@runtime_checkable
class MutationMethodProtocol(Protocol):
    """Protocol for mutation methods that can be applied to evolvable modules.

    Mutation methods must have a mutation type and optional recreation kwargs
    to specify how the network should be rebuilt after mutation.
    """

    _mutation_type: MutationType
    _recreate_kwargs: dict[str, Any]

    def __call__(self, *args: Any, **kwargs: Any) -> Any: ...


@runtime_checkable
class OptimizerWrapperProtocol(Protocol):
    """Protocol for optimizer wrapper classes that manage optimization.

    Provides a consistent interface for optimizer management across different
    network configurations and training setups.
    """

    optimizer: Optimizer | dict[str, Optimizer]
    optimizer_cls: type[Optimizer] | dict[str, type[Optimizer]]
    lr: Callable[[], float]
    optimizer_kwargs: dict[str, Any]


@runtime_checkable
class EvolvableModuleProtocol(Protocol):
    """Protocol for neural network modules that support evolutionary mutations.

    Evolvable modules can undergo mutations to their architecture (layers, nodes,
    activations) and maintain state information about recent mutations for
    reconstruction and optimization purposes.
    """

    init_dict: dict[str, Any]
    device: DeviceType
    layer_mutation_methods: list[str]
    node_mutation_methods: list[str]
    mutation_methods: list[str]
    last_mutation_attr: str
    last_mutation: Callable[[Any], Any]
    rng: Generator | None

    @property
    def activation(self) -> str | None: ...
    def change_activation(self, activation: str, output: bool) -> None: ...
    def forward(self, x: Any) -> Any: ...
    def parameters(self) -> Generator: ...
    def to(self, device: DeviceType) -> None: ...
    def state_dict(self) -> dict[str, Any]: ...
    def disable_mutations(self) -> None: ...
    def get_mutation_methods(self) -> dict[str, MutationMethodProtocol]: ...
    def get_mutation_probs(self, new_layer_prob: float) -> list[float]: ...
    def sample_mutation_method(
        self, new_layer_prob: float, rng: Generator | None
    ) -> str: ...
    def clone(self) -> "EvolvableModuleProtocol": ...
    def load_state_dict(
        self, state_dict: dict[str, Any], strict: bool = True
    ) -> None: ...


@runtime_checkable
class EvolvableNetworkProtocol(EvolvableModuleProtocol, Protocol):
    """Protocol for evolvable neural networks with encoder-decoder architecture.

    Evolvable networks consist of an encoder for feature extraction and
    a head network for task-specific outputs. Both components can evolve
    independently through mutations.
    """

    def forward_head(self, latent: torch.Tensor, *args, **kwargs) -> torch.Tensor: ...
    def extract_features(self, x: TorchObsType) -> torch.Tensor: ...
    def build_network_head(self, *args, **kwargs) -> None: ...
    def add_latent_node(self, numb_new_nodes: int | None = None) -> dict[str, Any]: ...
    def remove_latent_node(
        self, numb_new_nodes: int | None = None
    ) -> dict[str, Any]: ...
    def recreate_encoder(self) -> None: ...
    def initialize_hidden_state(
        self, batch_size: int = 1
    ) -> dict[str, torch.Tensor]: ...
    def init_weights_gaussian(
        self, std_coeff: float = 4.0, output_coeff: float = 2.0
    ) -> None: ...
    def _build_encoder(self, *args, **kwargs) -> None: ...


T = TypeVar("T", bound=EvolvableModuleProtocol | EvolvableNetworkProtocol)


@runtime_checkable
class ModuleDictProtocol(Protocol, Generic[T]):
    """Protocol for dictionary-like containers of evolvable modules.

    Provides access to multiple evolvable modules through a dictionary interface
    and aggregates mutation methods across all contained modules.
    """

    device: DeviceType

    def __getitem__(self, key: str) -> T: ...
    def keys(self) -> Iterable[str]: ...
    def values(self) -> Iterable[T]: ...
    def items(self) -> Iterable[tuple[str, T]]: ...
    def modules(self) -> dict[str, T]: ...
    def get_mutation_methods(self) -> dict[str, MutationMethodProtocol]: ...
    def filter_mutation_methods(self, method: str) -> None: ...

    @property
    def mutation_methods(self) -> list[str]: ...
    @property
    def layer_mutation_methods(self) -> list[str]: ...
    @property
    def node_mutation_methods(self) -> list[str]: ...


EvolvableNetworkType = EvolvableModuleProtocol | ModuleDictProtocol
OptimizerType = Optimizer | dict[str, Optimizer] | OptimizerWrapperProtocol
EvolvableAttributeType = EvolvableNetworkType | OptimizerType
EvolvableNetworkDict = dict[str, EvolvableNetworkProtocol]
EvolvableAttributeDict = dict[str, EvolvableAttributeType]


@runtime_checkable
class NetworkConfigProtocol(Protocol):
    """Protocol for network configuration information.

    Stores metadata about networks including their name, evaluation status,
    and associated optimizer.
    """

    name: str
    eval: bool
    optimizer: str | None


@runtime_checkable
class NetworkGroupProtocol(Protocol):
    """Protocol for grouping related networks in an algorithm.

    Groups evaluation and shared networks together, indicating whether
    they represent policy networks and if they're used in multi-agent setups.
    """

    eval: EvolvableNetworkProtocol
    shared: EvolvableNetworkProtocol | list[EvolvableNetworkProtocol] | None
    policy: bool
    multiagent: bool


@runtime_checkable
class OptimizerConfig(Protocol):
    """Protocol for optimizer configuration and management.

    Defines the configuration for optimizers including which networks they
    optimize, learning rate, optimizer class, and additional parameters.
    """

    name: str
    networks: str | list[str]
    lr: str
    optimizer_cls: type[Optimizer] | list[type[Optimizer]]
    optimizer_kwargs: dict[str, Any] | list[dict[str, Any]]
    multiagent: bool

    def get_optimizer_cls(self) -> type[Optimizer] | list[type[Optimizer]]: ...


@runtime_checkable
class MutationRegistryProtocol(Protocol):
    """Protocol for registering and managing mutation-related components.

    Maintains collections of network groups, optimizers, and hooks that
    are used during the mutation and evolution process.
    """

    groups: list[NetworkGroupProtocol]
    optimizers: list[OptimizerConfig]
    hooks: list[Callable[[], None]]

    def networks(self) -> list[NetworkConfigProtocol]: ...


SelfEvolvableAlgorithm = TypeVar(
    "SelfEvolvableAlgorithm", bound="EvolvableAlgorithmProtocol"
)


@runtime_checkable
class EvolvableAlgorithmProtocol(Protocol):
    """Protocol for reinforcement learning algorithms that support evolution.

    Evolvable algorithms can undergo mutations to their network architectures
    and hyperparameters. They maintain state about fitness, scores, and steps
    for selection and mutation processes.
    """

    device: str | torch.device
    accelerator: Accelerator
    registry: MutationRegistryProtocol
    mut: str | None
    index: int
    scores: list[float]
    fitness: list[float]
    steps: list[int]
    torch_compiler: str | None

    def unwrap_models(self) -> None: ...
    def wrap_models(self) -> None: ...
    def load(
        cls: type[SelfEvolvableAlgorithm], path: str
    ) -> SelfEvolvableAlgorithm: ...
    def load_checkpoint(
        self, path: str, device: str, accelerator: Accelerator | None
    ) -> None: ...
    def save_checkpoint(self, path: str) -> None: ...
    def learn(
        self, experiences: tuple[Iterable[ObservationType], ...], **kwargs
    ) -> None: ...
    def get_action(self, obs: ObservationType, **kwargs) -> Any: ...
    def test(self, *args, **kwargs) -> np.ndarray: ...
    def evolvable_attributes(
        self, networks_only: bool = False
    ) -> EvolvableAttributeDict: ...
    def inspect_attributes(
        agent: SelfEvolvableAlgorithm, input_args_only: bool = False
    ) -> dict[str, Any]: ...
    def clone(
        self: SelfEvolvableAlgorithm, index: int | None, wrap: bool
    ) -> SelfEvolvableAlgorithm: ...
    def recompile(self) -> None: ...
    def mutation_hook(self) -> None: ...


# Define a TypeVar for EvolvableAlgorithm that can be used for generic typing
T_EvolvableAlgorithm = TypeVar("T_EvolvableAlgorithm", bound=EvolvableAlgorithmProtocol)


@runtime_checkable
class AgentWrapperProtocol(Protocol, Generic[T_EvolvableAlgorithm]):
    """Protocol for wrapper classes that encapsulate evolvable algorithms.

    Agent wrappers provide additional functionality around evolvable algorithms
    while maintaining the core interface for action selection and learning.
    """

    agent: T_EvolvableAlgorithm

    def get_action(self, obs: ObservationType, **kwargs) -> Any: ...
    def learn(
        self, experiences: tuple[Iterable[ObservationType], ...], **kwargs
    ) -> None: ...


@runtime_checkable
class LoraConfigProtocol(Protocol):
    """
    "Protocol for LoRA configuration.

    LoRA configuration is used to configure the LoRA module.
    """

    r: int
    lora_alpha: int
    target_modules: str
    task_type: str
    lora_dropout: float


@runtime_checkable
class PretrainedConfigProtocol(Protocol):
    """Protocol for HuggingFace pre-trained model configuration.

    Defines the interface for model configuration objects from HuggingFace transformers.
    These configs store model architecture parameters and can be converted to/from dictionaries.
    """

    # Common model architecture attributes (these are examples - actual configs may have more)
    vocab_size: int
    hidden_size: int
    num_attention_heads: int
    num_hidden_layers: int

    def to_dict(self) -> dict[str, Any]: ...
    def to_json_string(self) -> str: ...
    def save_pretrained(self, save_directory: str, **kwargs: Any) -> None: ...

    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path: str, **kwargs: Any
    ) -> "PretrainedConfigProtocol": ...

    @classmethod
    def from_dict(
        cls, config_dict: dict[str, Any], **kwargs: Any
    ) -> "PretrainedConfigProtocol": ...

    @classmethod
    def from_json_file(cls, json_file: str) -> "PretrainedConfigProtocol": ...


@runtime_checkable
class GenerationConfigProtocol(Protocol):
    """Protocol for text generation configuration.

    Used to configure parameters for text generation in language models.
    """

    do_sample: bool
    temperature: float
    max_length: int | None
    max_new_tokens: int | None
    min_new_tokens: int | None
    pad_token_id: int
    repetition_penalty: float
    top_p: float
    top_k: int
    min_p: float


@runtime_checkable
class PreTrainedModelProtocol(Protocol):
    """Protocol for HuggingFace pre-trained models.

    Defines the interface for pre-trained transformer models from HuggingFace.
    These models support text generation, state management, and device operations.
    """

    device: DeviceType
    config: Any

    def eval(self) -> "PreTrainedModelProtocol": ...
    def train(self, mode: bool = True) -> "PreTrainedModelProtocol": ...
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        generation_config: GenerationConfigProtocol | None = None,
        **kwargs: Any,
    ) -> torch.Tensor: ...
    def forward(self, *args: Any, **kwargs: Any) -> Any: ...
    def parameters(self) -> Generator: ...
    def state_dict(self) -> dict[str, Any]: ...
    def load_state_dict(
        self, state_dict: dict[str, Any], strict: bool = True
    ) -> None: ...
    def to(self, device: DeviceType) -> "PreTrainedModelProtocol": ...


@runtime_checkable
class PeftModelProtocol(Protocol):
    """Protocol for PEFT (Parameter-Efficient Fine-Tuning) models.

    PEFT models wrap pre-trained models with adapters for efficient fine-tuning.
    They extend PreTrainedModel functionality with adapter-specific operations.
    """

    peft_config: dict[str, Any]

    def eval(self) -> "PeftModelProtocol": ...
    def train(self, mode: bool = True) -> "PeftModelProtocol": ...
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        generation_config: GenerationConfigProtocol | None = None,
        **kwargs: Any,
    ) -> torch.Tensor: ...
    def forward(self, *args: Any, **kwargs: Any) -> Any: ...
    def parameters(self) -> Generator: ...
    def state_dict(self) -> dict[str, Any]: ...
    def load_state_dict(
        self, state_dict: dict[str, Any], strict: bool = True
    ) -> None: ...
    def to(self, device: DeviceType) -> "PeftModelProtocol": ...

    @classmethod
    def from_pretrained(
        cls, base_model: PreTrainedModelProtocol, adapter_path: str, **kwargs: Any
    ) -> "PeftModelProtocol": ...
