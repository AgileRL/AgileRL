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

from collections.abc import Callable, Iterable, Iterator
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    Protocol,
    TypeVar,
    runtime_checkable,
)

import numpy as np
import torch
from accelerate import Accelerator
from gymnasium import spaces
from torch._dynamo import OptimizedModule
from torch.nn import Module
from torch.optim.optimizer import Optimizer
from typing_extensions import Self

if TYPE_CHECKING:
    from agilerl.algorithms.core.registry import MutationRegistry
    from agilerl.typing import MutationApplyDict, ReasoningPrompts, TokenObsStepReturn

NumpyObsType = np.ndarray | dict[str, np.ndarray] | tuple[np.ndarray, ...]
TorchObsType = torch.Tensor | dict[str, torch.Tensor] | tuple[torch.Tensor, ...]
ObservationType = NumpyObsType | TorchObsType
DeviceType = str | torch.device


@runtime_checkable
class NamedCallable(Protocol):
    """A callable with a ``__name__``: a function, bound method, or class.

    ``__call__`` is typed gradually so ordinary classes (e.g.
    ``torch.optim.Optimizer`` subclasses, whose constructors don't accept
    arbitrary keyword arguments) still conform structurally.
    """

    __name__: str

    def __call__(self, *args: Any, **kwargs: Any) -> Any: ...  # noqa: ANN401 -- gradually typed so any callable conforms structurally


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

    def __call__(self, *args: Any, **kwargs: Any) -> Any:  # noqa: ANN401 -- mutation methods take and return arbitrary values
        pass


@runtime_checkable
class OptimizerLikeClass(Protocol):
    """Protocol for optimizer-like constructor callables/classes."""

    def __call__(
        self,
        params: Any,  # noqa: ANN401 -- accepts any optimizer's params/param-groups argument
        lr: float,
        **kwargs: Any,
    ) -> Optimizer | Any:  # noqa: ANN401 -- some optimizer-like classes return non-Optimizer handles
        pass


@runtime_checkable
class OptimizerWrapperProtocol(Protocol):
    """Protocol for optimizer wrapper classes that manage optimization.

    Provides a consistent interface for optimizer management across different
    network configurations and training setups.
    """

    optimizer: Optimizer | dict[str, Optimizer]
    optimizer_cls: type[Optimizer] | dict[str, type[Optimizer]] | OptimizerLikeClass
    lr: Callable[[], float]
    optimizer_kwargs: dict[str, Any]


@runtime_checkable
class EvolvableModuleProtocol(Protocol):
    """Protocol for neural network modules that support evolutionary mutations.

    Evolvable modules can undergo mutations to their architecture (layers, nodes,
    activations) and maintain state information about recent mutations for
    reconstruction and optimization purposes.
    """

    # Read-only: implementations expose these as properties, and a mutable
    # member's invariance would reject every one of them.
    @property
    def init_dict(self) -> dict[str, Any]:
        pass

    @property
    def layer_mutation_methods(self) -> list[str]:
        pass

    @property
    def node_mutation_methods(self) -> list[str]:
        pass

    @property
    def mutation_methods(self) -> list[str]:
        pass

    @property
    def last_mutation_attr(self) -> str | None:
        pass

    @property
    def last_mutation(self) -> "MutationMethodProtocol | None":
        pass

    @property
    def rng(self) -> np.random.Generator | None:
        pass

    # Read-only: implementations narrow this (EvolvableModule exposes `str`),
    # which a mutable member's invariance would reject.
    @property
    def device(self) -> DeviceType:
        pass

    @property
    def activation(self) -> str | None:
        pass

    def change_activation(self, activation: str, output: bool) -> None:
        pass

    def forward(self, x: Any, /) -> Any:  # noqa: ANN401 -- protocol forward must accept and return whatever conforming modules do
        pass

    def parameters(self) -> Iterator[torch.nn.Parameter]:
        pass

    def to(self, device: DeviceType) -> Self:
        pass

    def state_dict(self) -> dict[str, Any]:
        pass

    def disable_mutations(self) -> None:
        pass

    def get_mutation_methods(self) -> dict[str, MutationMethodProtocol]:
        pass

    def get_mutation_probs(self, new_layer_prob: float) -> list[float]:
        pass

    def sample_mutation_method(
        self,
        new_layer_prob: float,
        rng: np.random.Generator | None = None,
    ) -> str:
        pass

    def clone(self) -> Self:
        pass

    def load_state_dict(
        self,
        state_dict: dict[str, Any],
        strict: bool = True,
    ) -> None:
        pass


@runtime_checkable
class EvolvableNetworkProtocol(EvolvableModuleProtocol, Protocol):
    """Protocol for evolvable neural networks with encoder-decoder architecture.

    Evolvable networks consist of an encoder for feature extraction and
    a head network for task-specific outputs. Both components can evolve
    independently through mutations.
    """

    def forward_head(
        self,
        latent: torch.Tensor,
        /,
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        pass

    def extract_features(self, x: TorchObsType, /) -> torch.Tensor:
        pass

    def build_network_head(self, *args: Any, **kwargs: Any) -> None:
        pass

    def add_latent_node(self, numb_new_nodes: int | None = None) -> "MutationApplyDict":
        pass

    def remove_latent_node(
        self,
        numb_new_nodes: int | None = None,
    ) -> "MutationApplyDict":
        pass

    def recreate_encoder(self) -> None:
        pass

    def initialize_hidden_state(
        self,
        batch_size: int = 1,
    ) -> dict[str, torch.Tensor]:
        pass

    def init_weights_gaussian(
        self,
        std_coeff: float = 4.0,
        output_coeff: float = 2.0,
    ) -> None:
        pass

    def _build_encoder(self, *args: Any, **kwargs: Any) -> Module:
        pass


# Values may also be plain torch modules or torch.compile wrappers
# (OptimizedModule), which don't satisfy the evolvable protocols.
T = TypeVar(
    "T",
    bound=Module | OptimizedModule | EvolvableModuleProtocol | EvolvableNetworkProtocol,
)


@runtime_checkable
class ModuleDictProtocol(Protocol, Generic[T]):
    """Protocol for dictionary-like containers of evolvable modules.

    Provides access to multiple evolvable modules through a dictionary interface
    and aggregates mutation methods across all contained modules.
    """

    @property
    def device(self) -> DeviceType:
        pass

    def __getitem__(self, key: str) -> T:
        pass

    def keys(self) -> Iterable[str]:
        pass

    def values(self) -> Iterable[T]:
        pass

    def items(self) -> Iterable[tuple[str, T]]:
        pass

    def modules(self) -> dict[str, T]:
        pass

    def get_mutation_methods(self) -> dict[str, MutationMethodProtocol]:
        pass

    def filter_mutation_methods(self, method: str) -> None:
        pass

    @property
    def mutation_methods(self) -> list[str]:
        pass

    @property
    def layer_mutation_methods(self) -> list[str]:
        pass

    @property
    def node_mutation_methods(self) -> list[str]:
        pass


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

    def get_optimizer_cls(self) -> type[Optimizer] | list[type[Optimizer]]:
        pass


@runtime_checkable
class MutationRegistryProtocol(Protocol):
    """Protocol for registering and managing mutation-related components.

    Maintains collections of network groups, optimizers, and hooks that
    are used during the mutation and evolution process.
    """

    groups: list[NetworkGroupProtocol]
    optimizers: list[OptimizerConfig]
    hooks: list[Callable[[], None]]

    def networks(self) -> list[NetworkConfigProtocol]:
        pass


EvolvableAlgorithm = TypeVar(
    "EvolvableAlgorithm",
    bound="EvolvableAlgorithmProtocol",
)


@runtime_checkable
class EvolvableAlgorithmProtocol(Protocol):
    """Protocol for reinforcement learning algorithms that support evolution.

    Evolvable algorithms can undergo mutations to their network architectures
    and hyperparameters. They maintain state about fitness, scores, and steps
    for selection and mutation processes.
    """

    device: str | torch.device
    accelerator: Accelerator | None
    # Imported under TYPE_CHECKING only, to avoid a runtime import cycle.
    registry: "MutationRegistry"
    mut: str | None
    index: int
    # Scalars, or per-sub-agent rows for multi-agent metrics (sum_scores=False).
    fitness: list[float | np.ndarray]
    steps: int
    torch_compiler: str | None

    @property
    def scores(self) -> list[float | list[float]]:
        """Per-episode scores (per-group score rows for multi-agent metrics)."""
        ...

    @scores.setter
    def scores(self, value: list[float | list[float]]) -> None: ...

    def unwrap_models(self) -> None:
        pass

    def wrap_models(self) -> None:
        pass

    @classmethod
    def load(
        cls,
        path: str,
        device: str | torch.device = "cpu",
        accelerator: Accelerator | None = None,
    ) -> Self:
        pass

    def load_checkpoint(self, path: str) -> None:
        pass

    def save_checkpoint(self, path: str) -> None:
        pass

    def learn(self, experiences: Any) -> Any:  # noqa: ANN401 -- experience format and loss return vary per algorithm
        pass

    def get_action(self, obs: Any, **kwargs: Any) -> Any:  # noqa: ANN401 -- obs and action types vary across RL/LLM/multi-agent algorithms
        pass

    def test(self, env: Any) -> Any:  # noqa: ANN401 -- env type and fitness return vary per algorithm
        pass

    def evolvable_attributes(
        self,
        networks_only: bool = False,
    ) -> EvolvableAttributeDict:
        pass

    @staticmethod
    def inspect_attributes(
        agent: Any,  # noqa: ANN401 -- accepts any evolvable algorithm or wrapped agent
        input_args_only: bool = False,
    ) -> dict[str, Any]:
        pass

    def clone(
        self,
        index: int | None = None,
        wrap: bool = True,
    ) -> Self:
        pass

    def clean_up(self) -> None:
        pass

    def recompile(self) -> None:
        pass

    def mutation_hook(self) -> None:
        pass


# Define a TypeVar for EvolvableAlgorithm that can be used for generic typing
EvolvableAlgorithmT = TypeVar("EvolvableAlgorithmT", bound=EvolvableAlgorithmProtocol)


@runtime_checkable
class AgentWrapperProtocol(Protocol, Generic[EvolvableAlgorithmT]):
    """Protocol for wrapper classes that encapsulate evolvable algorithms.

    Agent wrappers provide additional functionality around evolvable algorithms
    while maintaining the core interface for action selection and learning.
    """

    agent: EvolvableAlgorithmT

    def get_action(self, obs: ObservationType, **kwargs: Any) -> Any:  # noqa: ANN401 -- action return type varies per wrapped algorithm
        pass

    def learn(
        self,
        experiences: tuple[Iterable[ObservationType], ...],
        **kwargs: Any,
    ) -> None:
        pass

    def evolvable_attributes(
        self,
        networks_only: bool = False,
    ) -> EvolvableAttributeDict:
        pass


@runtime_checkable
class LoraConfigProtocol(Protocol):
    """Protocol for LoRA configuration.

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

    def to_dict(self) -> dict[str, Any]:
        pass

    def to_json_string(self) -> str:
        pass

    def save_pretrained(self, save_directory: str, **kwargs: Any) -> None:
        pass

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        **kwargs: Any,
    ) -> "PretrainedConfigProtocol":
        pass

    @classmethod
    def from_dict(
        cls,
        config_dict: dict[str, Any],
        **kwargs: Any,
    ) -> "PretrainedConfigProtocol":
        pass

    @classmethod
    def from_json_file(cls, json_file: str) -> "PretrainedConfigProtocol":
        pass


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

    def eval(self) -> "PreTrainedModelProtocol":
        pass

    def train(self, mode: bool = True) -> "PreTrainedModelProtocol":
        pass

    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        generation_config: GenerationConfigProtocol | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        pass

    def forward(self, *args: Any, **kwargs: Any) -> Any:  # noqa: ANN401 -- model forward output type varies (logits, tuples, ModelOutput)
        pass

    def parameters(self) -> Iterator[torch.nn.Parameter]:
        pass

    def state_dict(self) -> dict[str, Any]:
        pass

    def load_state_dict(
        self,
        state_dict: dict[str, Any],
        strict: bool = True,
    ) -> None:
        pass

    def to(self, device: DeviceType) -> "PreTrainedModelProtocol":
        pass


@runtime_checkable
class PeftModelProtocol(Protocol):
    """Protocol for PEFT (Parameter-Efficient Fine-Tuning) models.

    PEFT models wrap pre-trained models with adapters for efficient fine-tuning.
    They extend PreTrainedModel functionality with adapter-specific operations.
    """

    peft_config: dict[str, Any]

    def eval(self) -> "PeftModelProtocol":
        pass

    def train(self, mode: bool = True) -> "PeftModelProtocol":
        pass

    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        generation_config: GenerationConfigProtocol | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        pass

    def forward(self, *args: Any, **kwargs: Any) -> Any:  # noqa: ANN401 -- model forward output type varies (logits, tuples, ModelOutput)
        pass

    def parameters(self) -> Iterator[torch.nn.Parameter]:
        pass

    def state_dict(self) -> dict[str, Any]:
        pass

    def load_state_dict(
        self,
        state_dict: dict[str, Any],
        strict: bool = True,
    ) -> None:
        pass

    def to(self, device: DeviceType) -> "PeftModelProtocol":
        pass

    @classmethod
    def from_pretrained(
        cls,
        base_model: PreTrainedModelProtocol,
        adapter_path: str,
        **kwargs: Any,
    ) -> "PeftModelProtocol":
        pass


@runtime_checkable
class BanditEnvProtocol(Protocol):
    """Protocol for contextual bandit environments.

    Any environment used with :func:`~agilerl.training.train_bandits.train_bandits`
    or :class:`~agilerl.algorithms.neural_ts_bandit.NeuralTS` /
    :class:`~agilerl.algorithms.neural_ucb_bandit.NeuralUCB` must satisfy this
    interface.  The built-in :class:`~agilerl.wrappers.learning.BanditEnv` is the
    reference implementation.

    :param arms: Number of arms (actions) in the bandit problem.
    :type arms: int
    :param num_envs: Number of parallel environments (typically 1).
    :type num_envs: int
    :param single_observation_space: Observation space for a single environment.
    :type single_observation_space: gymnasium.spaces.Space
    :param single_action_space: Action space for a single environment.
    :type single_action_space: gymnasium.spaces.Discrete
    """

    arms: int
    num_envs: int

    @property
    def single_observation_space(self) -> spaces.Space: ...

    @property
    def single_action_space(self) -> spaces.Discrete: ...

    def reset(self) -> np.ndarray:
        pass

    def step(self, k: int) -> tuple[np.ndarray, float]:
        pass


@runtime_checkable
class MultiTurnEnv(Protocol):
    """Protocol for text-level multi-turn LLM environments.

    Covers raw multi-turn envs (e.g. GEM environments) and text wrappers such
    as ``FormatRewardWrapper``: text observations and text actions. The
    token-level surface that ``TokenObservationWrapper`` exposes is described
    by :class:`TokenizedMultiTurnEnv`.
    """

    max_turns: int

    def reset(
        self, seed: int | None = None
    ) -> tuple[str | dict[str, Any], dict[str, Any]]:
        pass

    def step(
        self, action: str, **kwargs: Any
    ) -> tuple[str | dict[str, Any], float, bool, bool, dict[str, Any]]:
        pass

    def close(self) -> None:
        pass


@runtime_checkable
class TokenizedMultiTurnEnv(Protocol):
    """Protocol for token-level multi-turn LLM environments.

    The surface ``TokenObservationWrapper`` exposes to
    ``SyncMultiTurnVecEnv`` and the LLM algorithms' ``test`` loops:
    tokenized ``ReasoningPrompts`` observations, token-id tensor actions,
    and full-episode tensors for learning. ``step`` returns an empty
    observation mapping once the episode has ended.
    """

    max_turns: int

    def reset(
        self, seed: int | None = None
    ) -> "tuple[ReasoningPrompts, dict[str, Any]]":
        pass

    def step(self, full_completion_ids: torch.Tensor, /) -> "TokenObsStepReturn":
        pass

    def close(self) -> None:
        pass

    def get_episode_data(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        pass
