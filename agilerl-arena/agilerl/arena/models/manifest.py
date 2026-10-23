# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""The training manifest: the whole contract, in one place."""

from __future__ import annotations

from collections.abc import Callable
from difflib import get_close_matches
from pathlib import Path
from types import UnionType
from typing import (
    Annotated,
    Any,
    Literal,
    Union,
    get_args,
    get_origin,
    overload,
)

import yaml
from pydantic import (
    AliasChoices,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    PlainSerializer,
    field_serializer,
    model_validator,
)
from typing_extensions import Self

from agilerl.arena.models.algorithms.base import (
    AlgoSpec,
    LLMAlgorithmSpec,
    MultiAgentAlgorithmSpec,
    SingleAgentAlgorithmSpec,
)
from agilerl.arena.models.algorithms.ddpg import DDPGSpec
from agilerl.arena.models.algorithms.grpo import GRPOSpec
from agilerl.arena.models.algorithms.ippo import IPPOSpec
from agilerl.arena.models.algorithms.latent_ppo import LatentPPOSpec
from agilerl.arena.models.algorithms.maddpg import MADDPGSpec
from agilerl.arena.models.algorithms.matd3 import MATD3Spec
from agilerl.arena.models.algorithms.ppo import PPOSpec, RecurrentPPOSpec
from agilerl.arena.models.algorithms.rainbow_dqn import RainbowDQNSpec
from agilerl.arena.models.algorithms.rollout_llm import RolloutLLMSpec
from agilerl.arena.models.algorithms.sft import SFTSpec
from agilerl.arena.models.algorithms.supervised import SupervisedSpec
from agilerl.arena.models.algorithms.td3 import TD3Spec
from agilerl.arena.models.env import EnvSpec, LLMEnvSpec
from agilerl.arena.models.hpo import (
    MultiFrequencySelectionSpec,
    MutationProbabilities,
    MutationSpec,
    RLHyperparameter,
    SelectionStrategySpec,
    TournamentSelectionSpec,
    hp_only_mutation_probabilities,
    no_op_mutation_probabilities,
)
from agilerl.arena.models.model_info import (
    ARCH_HYBRID_MOE,
    ARCH_MOE,
    SUPPORTED_MODEL_INFO,
    default_lora_target_scope,
)
from agilerl.arena.models.networks import (
    FP8_QUANTIZATION_METHODS,
    LATENT_NETWORK_KEYS,
    NATIVE_FP8_MIN_COMPUTE_CAPABILITY,
    FinetuningNetworkSpec,
    LatentPPONetworkSpec,
    LoraConfigDict,
    MlpSpec,
    NetworkSpec,
    check_lora_config_supported,
    default_colocated_vllm_config,
    dump_network_section,
    network_arch_is_resolvable,
    normalize_manifest_network,
    resolve_lora_defaults,
    resolve_lora_moe_backend,
)
from agilerl.arena.models.registry import MANIFEST_REGISTRY
from agilerl.arena.models.training import (
    BufferSpec,
    CheckpointExportSpec,
    LLMRolloutBufferSpec,
    PerBufferArgs,
    ReplayBufferSpec,
    TrainingSpec,
)

API_VERSION = "agilerl.com/v1"


def _copy_when_unset(
    target: BaseModel, field: str, value: object, *, source_set: bool
) -> None:
    """Copy *value* onto *target.field* when the target omitted it.

    If both sides were set and they disagree, that is an error rather than a
    silent overwrite.
    """
    if field not in type(target).model_fields:
        return
    if field not in target.model_fields_set:
        if source_set:
            setattr(target, field, value)
        return
    if source_set and getattr(target, field) != value:
        msg = (
            f"algorithm.{field}={getattr(target, field)!r} disagrees with "
            f"{value!r} from another section."
        )
        raise ValueError(msg)


def _is_classic_hpo(algorithm: AlgoSpec) -> bool:
    """Whether the algorithm trains classic RL with evolutionary HPO by default."""
    return (
        isinstance(algorithm, (SingleAgentAlgorithmSpec, MultiAgentAlgorithmSpec))
        and not algorithm.bandit
        and not algorithm.offline
    )


def _network_arch_is_large(network: object) -> bool:
    """Whether the network section asks for CNN-scale horizons."""
    return isinstance(network, NetworkSpec) and network.encoder_config.arch in (
        "cnn",
        "multiinput",
    )


def _mutation_template(algorithm: AlgoSpec) -> MutationProbabilities | None:
    """HPO mutation probabilities for the paradigm, or None to leave alone."""
    if isinstance(algorithm, SFTSpec):
        return hp_only_mutation_probabilities(0.4, 0.6)
    if isinstance(algorithm, (LLMAlgorithmSpec, SupervisedSpec, LatentPPOSpec)):
        return hp_only_mutation_probabilities(0.2, 0.8)
    if _is_classic_hpo(algorithm):
        # PPO/DDPG-family algorithms never mutate the activation function.
        if isinstance(
            algorithm,
            (
                PPOSpec,
                RecurrentPPOSpec,
                IPPOSpec,
                DDPGSpec,
                TD3Spec,
                MADDPGSpec,
                MATD3Spec,
            ),
        ):
            return MutationProbabilities(act_mut=0.0)
        return MutationProbabilities()
    return None


def _fill_probabilities(
    mutation: MutationSpec, template: MutationProbabilities
) -> None:
    """Overlay the template's probabilities, keeping explicitly set fields."""
    explicit = mutation.probabilities.model_dump(
        include=set(mutation.probabilities.model_fields_set)
    )
    mutation.probabilities = template.model_copy(update=explicit)


def _copy_ranges(
    ranges: dict[str, RLHyperparameter],
) -> dict[str, RLHyperparameter]:
    """Copy HPO ranges off the shared class defaults."""
    return {name: bounds.model_copy() for name, bounds in ranges.items()}


def _algorithm_rl_hp(algorithm: AlgoSpec) -> dict[str, RLHyperparameter]:
    """HPO ranges for the fields the algorithm declares, copied."""
    return _copy_ranges(
        {
            name: bounds
            for name, bounds in type(algorithm).hpo_ranges.items()
            if name in type(algorithm).model_fields
            and _is_numeric(type(algorithm).model_fields[name].annotation)
        }
    )


def _supervised_rl_hp(
    algorithm: SupervisedSpec | LatentPPOSpec,
) -> dict[str, RLHyperparameter]:
    """HPO ranges for supervised or latent runs, minus inapplicable fields."""
    ranges = _algorithm_rl_hp(algorithm)
    if isinstance(algorithm, SupervisedSpec):
        if algorithm.task_type not in ("object_detection", "multiclass_classification"):
            ranges.pop("label_smoothing", None)
        if algorithm.entrypoint == "EvolvableMLP":
            ranges.pop("dropout", None)
    return ranges


def _default_classic_buffer(algorithm: AlgoSpec) -> ReplayBufferSpec:
    """The replay buffer an off-policy classic run gets when it names none."""
    if isinstance(algorithm, RainbowDQNSpec):
        return ReplayBufferSpec(
            standard_buffer=False,
            n_step_buffer=True,
            per_buffer=True,
            per_buffer_args=PerBufferArgs(alpha=0.6),
        )
    if isinstance(algorithm, MultiAgentAlgorithmSpec):
        return ReplayBufferSpec(max_size=100_000)
    return ReplayBufferSpec()


def _fill_mutation_defaults(
    algorithm: AlgoSpec, mutation: MutationSpec, hpo: bool
) -> None:
    """Overlay paradigm probabilities and ranges onto the mutation section."""
    if _is_classic_hpo(algorithm) and not hpo:
        return
    template = _mutation_template(algorithm)
    if template is None:
        return
    _fill_probabilities(mutation, template)
    if "rl_hp_selection" in mutation.model_fields_set:
        return
    if _is_classic_hpo(algorithm):
        mutation.rl_hp_selection = _algorithm_rl_hp(algorithm)
    elif isinstance(algorithm, (SupervisedSpec, LatentPPOSpec)) and hpo:
        mutation.rl_hp_selection = _supervised_rl_hp(algorithm)


def check_llm_engine_for_gpu(
    manifest: TrainingManifest, compute_capability: tuple[int, int]
) -> None:
    """Reject an FP8 MoE LoRA engine whose backend cannot run on this GPU.

    Only generation algorithms (rollout LLMs) build vLLM engines, and only
    catalog MoE models can be checked. An unset backend resolves at engine
    build, so only an explicit one fails here.

    :param manifest: The validated training manifest.
    :type manifest: TrainingManifest
    :param compute_capability: CUDA capability of the serving GPU.
    :type compute_capability: tuple[int, int]
    :raises ValueError: If an explicit backend cannot serve this run here.
    """
    algorithm = manifest.algorithm
    if not isinstance(algorithm, RolloutLLMSpec):
        return
    network = manifest.network
    model_id = (
        network.pretrained_model_name_or_path
        if isinstance(network, FinetuningNetworkSpec)
        else algorithm.pretrained_model_name_or_path
    )
    entry = SUPPORTED_MODEL_INFO.get(model_id) if model_id else None
    if entry is None or entry.architecture not in (ARCH_MOE, ARCH_HYBRID_MOE):
        return
    overrides = algorithm.vllm_engine_args or {}
    vllm_config = algorithm.vllm_config
    quantization = overrides.get(
        "quantization", vllm_config.quantization if vllm_config else None
    )
    backend = overrides.get("moe_backend")
    if backend is None:
        return
    resolve_lora_moe_backend(quantization, backend, compute_capability)
    if (
        str(backend).lower().replace("-", "_") == "triton"
        and isinstance(quantization, str)
        and quantization.lower() in FP8_QUANTIZATION_METHODS
        and compute_capability < NATIVE_FP8_MIN_COMPUTE_CAPABILITY
    ):
        msg = (
            f"moe_backend='triton' cannot serve FP8 MoE LoRA on this GPU "
            f"(compute capability {compute_capability[0]}.{compute_capability[1]}): "
            "Triton needs SM89+ native FP8. Use moe_backend='marlin'."
        )
        raise ValueError(msg)


class DeferredNetworkSpec(BaseModel):
    """A network section whose encoder arch is inferred from the observation space.

    Manifests for the classic RL algorithms may omit ``arch``; the trainer picks
    the encoder once it knows the observation space, so ``encoder_config`` cannot
    be validated against a concrete encoder spec here.
    """

    model_config = ConfigDict(extra="forbid")

    latent_dim: int = Field(
        default=32,
        gt=0,
        description="Width of the latent the encoder produces and the head consumes.",
    )
    min_latent_dim: int = Field(
        default=8, gt=0, description="Narrowest the latent may be mutated to."
    )
    max_latent_dim: int = Field(
        default=128, gt=1, description="Widest the latent may be mutated to."
    )
    encoder_config: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Encoder settings. Left unvalidated here because the architecture "
            "is only known once the observation space is."
        ),
    )
    head_config: MlpSpec | None = Field(
        default=None,
        description="Head mapping the latent onto the algorithm's outputs.",
    )
    random_seed: int | None = Field(
        default=None, description="Seed for weight initialization."
    )
    simba: bool = Field(default=False, description="Use a SimBa encoder.")
    squash_output: bool = Field(
        default=False,
        description="Squash actions through tanh into the action space's bounds.",
    )
    has_evolvable_encoder: bool = Field(
        default=False,
        description="Whether the encoder architecture can be mutated during HPO.",
    )


def _resolve_algorithm(data: object) -> object:
    """Dispatch the algorithm section to its registered spec class via ``name``."""
    if isinstance(data, BaseModel):
        return data
    if not isinstance(data, dict):
        # ValueError, not TypeError: pydantic only converts ValueError /
        # AssertionError into a ValidationError; a TypeError would escape the
        # validate CLI and submit paths as a crash.
        msg = f"The algorithm section must be a mapping, got {type(data).__name__}."
        raise ValueError(msg)

    payload = dict(data)
    name = payload.pop("name", None)
    if name is None:
        msg = (
            "The algorithm section must include a 'name' field naming a registered "
            f"algorithm. Registered: {', '.join(MANIFEST_REGISTRY.names())}."
        )
        raise ValueError(msg)
    try:
        return MANIFEST_REGISTRY.create(name, **payload)
    except KeyError as err:
        # Surface an unknown name as a validation error like any other bad
        # field, rather than a KeyError a caller cannot tell from a crash.
        msg = (
            f"'name' is not a registered algorithm: {name!r}. "
            f"Registered: {', '.join(MANIFEST_REGISTRY.names())}."
        )
        raise ValueError(msg) from err


def _serialize_algorithm(spec: AlgoSpec) -> dict[str, Any]:
    """Serialize the algorithm section, restoring the ``name`` discriminator.

    ``net_config`` is resolved onto the spec from the network section, which
    is what the payload carries, so it is left out rather than written twice.
    """
    dumped = spec.model_dump(mode="json", exclude_none=True, exclude={"net_config"})
    dumped["name"] = spec.name
    return dumped


def normalize_network_section(data: object) -> object:
    """Nest a top-level ``arch`` and pin a concrete NetworkSpec when it is set.

    Without this, an unknown encoder key fails ``NetworkSpec`` then matches
    ``DeferredNetworkSpec``, whose ``encoder_config`` is an untyped dict.
    """
    if isinstance(data, BaseModel) or not isinstance(data, dict):
        return data
    if "pretrained_model_name_or_path" in data:
        return data
    if LATENT_NETWORK_KEYS & data.keys():
        return data
    data = normalize_manifest_network(data)
    if isinstance(data, dict) and network_arch_is_resolvable(data):
        return NetworkSpec.model_validate(data)
    return data


def _default_selection_strategy(data: object) -> object:
    """Tag a selection section that omits ``strategy`` as the tournament branch."""
    if isinstance(data, dict) and "strategy" not in data:
        return {**data, "strategy": "tournament"}
    return data


def _is_numeric(annotation: object) -> bool:
    """Whether mutation can move this field between a numeric min and max.

    ``bool`` is deliberately excluded: it subclasses ``int``, but a grow factor
    applied to a flag means nothing. Every member of a union must be a scalar
    number, so ``float | list[float]`` (asymmetric clip bounds) is not mutable.
    """
    origin = get_origin(annotation)
    if origin in (Union, UnionType):
        members = [arg for arg in get_args(annotation) if arg is not type(None)]
        return bool(members) and all(_is_numeric(arg) for arg in members)
    return annotation in (int, float)


def _mutable_names(algorithm: type[AlgoSpec]) -> dict[str, str]:
    """Map every spelling of a mutable hyperparameter to its canonical field.

    Aliases have to be in here: ``lr`` is how most manifests spell LLMPPO's
    ``lr_actor``, and rejecting it would fail documents that run today.

    :param algorithm: The resolved algorithm's spec class.
    :type algorithm: type[AlgoSpec]
    :returns: Accepted spelling -> field name.
    :rtype: dict[str, str]
    """
    names: dict[str, str] = {}
    for field, info in algorithm.model_fields.items():
        names[field] = field
        alias = info.validation_alias
        if isinstance(alias, AliasChoices):
            names.update({c: field for c in alias.choices if isinstance(c, str)})
    return names


AlgorithmSection = Annotated[
    AlgoSpec,
    BeforeValidator(_resolve_algorithm),
    PlainSerializer(_serialize_algorithm, return_type=dict[str, Any]),
]
NetworkSection = Annotated[
    FinetuningNetworkSpec | NetworkSpec | DeferredNetworkSpec | LatentPPONetworkSpec,
    BeforeValidator(normalize_network_section),
]
SelectionSection = Annotated[
    SelectionStrategySpec,
    BeforeValidator(_default_selection_strategy),
]


class TrainingManifest(BaseModel):
    """A whole training run, described once."""

    model_config = ConfigDict(extra="forbid")

    api_version: str = Field(
        default=API_VERSION,
        validation_alias=AliasChoices("apiVersion", "api_version"),
        serialization_alias="apiVersion",
        description="Contract version this manifest is written against.",
    )
    algorithm: AlgorithmSection = Field(
        description="What to train, and the hyperparameters it trains with."
    )
    environment: EnvSpec = Field(
        description="What to train against: an environment, or a dataset."
    )
    training: TrainingSpec = Field(
        default_factory=TrainingSpec,
        description="How long to train for, and how the population evolves.",
    )
    network: NetworkSection | None = Field(
        default=None,
        description=(
            "The model. Either a network to build from scratch, or a "
            "pretrained model to fine-tune."
        ),
    )
    mutation: MutationSpec | None = Field(
        default=None,
        description="What population-based HPO may change between evolution rounds.",
    )
    replay_buffer: BufferSpec | None = Field(
        default=None,
        description="Where collected experience is held before it is learned from.",
    )
    selection_strategy: SelectionSection | None = Field(
        default=None,
        validation_alias=AliasChoices("selection_strategy", "tournament_selection"),
        serialization_alias="tournament_selection",
        description="How the population is ranked and replaced at each evolution round.",
    )

    @model_validator(mode="before")
    @classmethod
    def _tag_environment(cls, data: object) -> object:
        """Tag the environment with the branch the algorithm implies, when unset.

        The environment union discriminates on ``env_type``, but a manifest for
        an algorithm that only ever trains against one kind of environment does
        not have to repeat it.
        """
        if not isinstance(data, dict):
            return data
        environment = data.get("environment")
        algorithm = data.get("algorithm")
        if not isinstance(environment, dict) or "env_type" in environment:
            return data
        if not isinstance(algorithm, dict) or "name" not in algorithm:
            return data
        try:
            spec_cls = MANIFEST_REGISTRY.get(algorithm["name"])
        except KeyError:
            # Inferring env_type is all this does. Resolving the algorithm
            # section reports an unknown name as a validation error; raising
            # here would beat it to it with a traceback instead.
            return data
        tagged = {**environment, "env_type": str(spec_cls.env_type)}
        if spec_cls.objective is not None and "objective" not in tagged:
            tagged["objective"] = spec_cls.objective
        return {**data, "environment": tagged}

    @model_validator(mode="before")
    @classmethod
    def _merge_algorithm_lora_into_network(cls, data: object) -> object:
        """Fold an algorithm-level ``lora_config`` into the network's.

        LoRA is written in the network section and lifted onto the algorithm,
        but documents may spell it under the algorithm instead. Shared keys
        must agree; the merged network copy is the single value the contract
        validates, and the lift below copies it back.
        """
        if not isinstance(data, dict):
            return data
        algorithm = data.get("algorithm")
        network = data.get("network")
        if not isinstance(algorithm, dict) or not isinstance(network, dict):
            return data
        algo_lora = algorithm.get("lora_config")
        if algo_lora is None:
            return data
        if not isinstance(algo_lora, dict):
            return data
        net_lora = network.get("lora_config")
        if net_lora is not None and not isinstance(net_lora, dict):
            return data
        net_lora = net_lora or {}
        for key, value in algo_lora.items():
            if key in net_lora and net_lora[key] != value:
                msg = (
                    f"algorithm.lora_config={algo_lora!r} disagrees with "
                    f"network.lora_config={net_lora!r}."
                )
                raise ValueError(msg)
        merged = {**net_lora, **algo_lora}
        return {
            **data,
            "network": {**network, "lora_config": merged or None},
            "algorithm": {
                key: value for key, value in algorithm.items() if key != "lora_config"
            },
        }

    @model_validator(mode="before")
    @classmethod
    def _fold_network_mutation_ranges(cls, data: object) -> object:
        """Move ``mutation.network`` bounds onto the network they constrain.

        They bound the same encoder and head the network section configures, so
        keeping them apart lets the two disagree — a head sized outside its own
        mutation range validates only because nothing compares them.
        """
        if not isinstance(data, dict):
            return data
        # The section arrives as a spec when a caller builds the manifest from
        # objects rather than parsing a document -- from_trainer_specs does.
        mutation = data.get("mutation")
        if isinstance(mutation, BaseModel):
            mutation = mutation.model_dump(exclude_none=True)
        ranges = (mutation or {}).get("network")
        network = data.get("network")
        # Architecture-mutation bounds apply to encoder/head networks, not a
        # pretrained or latent model.
        if isinstance(network, BaseModel) and not isinstance(
            network, (FinetuningNetworkSpec, LatentPPONetworkSpec)
        ):
            network = dump_network_section(network, exclude_unset=True)
        if (
            not isinstance(ranges, dict)
            or not isinstance(network, dict)
            or not isinstance(mutation, dict)
        ):
            return data

        folded = dict(network)
        for key in ("min_latent_dim", "max_latent_dim"):
            if key in ranges:
                folded.setdefault(key, ranges[key])
        for section, bounds in (
            ("encoder_config", ranges.get("encoder")),
            ("head_config", ranges.get("head_net")),
        ):
            if not isinstance(bounds, dict):
                continue
            folded[section] = {**bounds, **(folded.get(section) or {})}
        return {**data, "network": folded, "mutation": {**mutation, "network": None}}

    @model_validator(mode="after")
    def _check_api_version(self) -> Self:
        if self.api_version != API_VERSION:
            msg = (
                f"Unsupported manifest apiVersion {self.api_version!r}; "
                f"this build of the manifest contract speaks {API_VERSION!r}."
            )
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def _check_network_matches_algorithm(self) -> Self:
        is_llm = isinstance(self.algorithm, LLMAlgorithmSpec)
        is_latent = isinstance(self.algorithm, LatentPPOSpec)
        if is_llm and isinstance(
            self.network, (NetworkSpec, DeferredNetworkSpec, LatentPPONetworkSpec)
        ):
            msg = (
                f"{self.algorithm.name} fine-tunes a pretrained model, so the "
                "network section must declare 'pretrained_model_name_or_path'."
            )
            raise ValueError(msg)
        if is_latent and isinstance(
            self.network, (NetworkSpec, DeferredNetworkSpec, FinetuningNetworkSpec)
        ):
            msg = (
                f"{self.algorithm.name} trains a latent segmentation model, so "
                "the network section must be a latent network (training_mode, "
                "num_conv_blocks, ...), not an encoder/head network."
            )
            raise ValueError(msg)
        if (
            not is_llm
            and not is_latent
            and isinstance(self.network, (FinetuningNetworkSpec, LatentPPONetworkSpec))
        ):
            msg = (
                f"{self.algorithm.name} builds its own networks, so the network "
                "section must not declare 'pretrained_model_name_or_path' or "
                "latent fields."
            )
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def _check_encoder_compatibility(self) -> Self:
        simba = (
            isinstance(self.network, (NetworkSpec, DeferredNetworkSpec))
            and self.network.simba
        )
        if isinstance(self.algorithm, PPOSpec) and self.algorithm.recurrent and simba:
            msg = (
                "`simba` and `recurrent` cannot both be set: a network cannot use "
                "both a SimBa and a recurrent (LSTM) encoder. Enable only one."
            )
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def _check_mutated_hyperparameters_exist(self) -> Self:
        """Reject HPO mutating a name the algorithm does not have, or a non-numeric field."""
        if self.mutation is None or not self.mutation.rl_hp_selection:
            return self

        spec = type(self.algorithm)
        accepted = _mutable_names(spec)
        for spelling in self.mutation.rl_hp_selection:
            field = accepted.get(spelling)
            if field is None:
                near = get_close_matches(spelling, sorted(accepted), n=1)
                hint = f" Did you mean {near[0]!r}?" if near else ""
                msg = (
                    f"mutation.rl_hp_selection cannot mutate {spelling!r}: "
                    f"{self.algorithm.name} has no such hyperparameter.{hint}"
                )
                raise ValueError(msg)
            if not _is_numeric(spec.model_fields[field].annotation):
                msg = (
                    f"mutation.rl_hp_selection cannot mutate {spelling!r}: "
                    f"{self.algorithm.name} declares it, but mutation moves a "
                    "value between a numeric min and max and this is not a number."
                )
                raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def _resolve_selection_brackets(self) -> Self:
        if isinstance(self.selection_strategy, MultiFrequencySelectionSpec):
            self.selection_strategy.resolve_brackets(self.training.pop_size)
        return self

    @model_validator(mode="after")
    def _default_classic_training(self) -> Self:
        """Fill the classic RL training loop: HPO on, arch-scaled horizons."""
        if not _is_classic_hpo(self.algorithm):
            return self
        training = self.training
        fields = training.model_fields_set
        if "max_steps" not in fields:
            training.max_steps = (
                12_000_000 if _network_arch_is_large(self.network) else 2_000_000
            )
        if "evo_steps" not in fields and "evo_epochs" not in fields:
            training.evo_steps = min(
                320_000 if _network_arch_is_large(self.network) else 160_000,
                training.max_steps,
            )
        if "learning_delay" not in fields and self.algorithm.off_policy:
            training.learning_delay = 20_000
        if "hpo" not in fields:
            training.hpo = True
        if "pop_size" not in fields:
            training.pop_size = 4
        if "experience_sharing" not in fields and self.algorithm.off_policy:
            training.experience_sharing = True
        if "sum_scores" not in fields and isinstance(
            self.algorithm, MultiAgentAlgorithmSpec
        ):
            training.sum_scores = True
        if (
            self.algorithm.off_policy
            and "expl_noise" not in type(self.algorithm).model_fields
            and not isinstance(self.algorithm, RainbowDQNSpec)
        ):
            if "eps_start" not in fields:
                training.eps_start = 1.0
            if "eps_end" not in fields:
                training.eps_end = 0.01
            if "eps_decay" not in fields:
                training.eps_decay = 0.99999
        return self

    @model_validator(mode="after")
    def _default_llm_training(self) -> Self:
        """Fill the LLM training loop from the environment regime."""
        if not isinstance(self.algorithm, LLMAlgorithmSpec):
            return self
        training = self.training
        fields = training.model_fields_set
        if "reporting_interval" not in fields:
            training.reporting_interval = 1
        if "evaluation_interval" not in fields:
            training.evaluation_interval = 10
        if "checkpoint_export" not in fields:
            training.checkpoint_export = CheckpointExportSpec()
        environment = self.environment
        if isinstance(environment, LLMEnvSpec):
            if environment.env_type == "rollout":
                if "max_steps" not in fields and isinstance(
                    self.algorithm, RolloutLLMSpec
                ):
                    training.max_steps = 200
                if (
                    "num_epochs" not in fields
                    and isinstance(self.algorithm, GRPOSpec)
                    and environment.dataset_backed_rollout
                ):
                    training.num_epochs = 1
            elif "num_epochs" not in fields:
                training.num_epochs = 1
        if training.async_rollout and "rollout_version_stamp" not in fields:
            training.rollout_version_stamp = "oldest_turn"
        return self

    @model_validator(mode="after")
    def _empty_mutation_is_a_noop(self) -> Self:
        """An empty ``mutation: {}`` is the same no-op as omitting the section.

        Except when HPO runs, where it keeps the field defaults for the
        paradigm fill below.
        """
        if self.mutation is not None and not self.mutation.model_fields_set:
            if self.training.hpo and _mutation_template(self.algorithm) is not None:
                return self
            self.mutation = MutationSpec(probabilities=no_op_mutation_probabilities())
        return self

    @model_validator(mode="after")
    def _default_hpo_sections(self) -> Self:
        """Fill the mutation section and classic replay buffer HPO runs use."""
        algorithm = self.algorithm
        classic_hpo = _is_classic_hpo(algorithm) and self.training.hpo
        if classic_hpo and self.mutation is None:
            self.mutation = MutationSpec()
        mutation = self.mutation
        if mutation is not None:
            _fill_mutation_defaults(algorithm, mutation, self.training.hpo)
        if (
            _is_classic_hpo(algorithm)
            and algorithm.off_policy
            and self.replay_buffer is None
        ):
            self.replay_buffer = _default_classic_buffer(algorithm)
        return self

    @model_validator(mode="after")
    def _default_supervised_latent_training(self) -> Self:
        """Fill the epoch-driven training loop supervised and latent runs use."""
        fields_set = self.training.model_fields_set
        if isinstance(self.algorithm, SupervisedSpec):
            detection = self.algorithm.task_type == "object_detection"
            if "num_epochs" not in fields_set:
                self.training.num_epochs = 8 if detection else 100
            if "evo_steps" not in fields_set and "evo_epochs" not in fields_set:
                self.training.evo_steps = 16 if detection else 128
            if "reporting_interval" not in fields_set:
                self.training.reporting_interval = 10
        elif isinstance(self.algorithm, LatentPPOSpec):
            if "max_steps" not in fields_set:
                self.training.max_steps = 2_000
            if "evo_steps" not in fields_set and "evo_epochs" not in fields_set:
                self.training.evo_steps = 8
            if "reporting_interval" not in fields_set:
                self.training.reporting_interval = 10
        return self

    @model_validator(mode="after")
    def _resolve_cross_section_defaults(self) -> Self:
        """Fill in the values one section implies for another."""
        if (
            self.training.evo_steps is None
            and self.training.evo_epochs is None
            and not isinstance(self.algorithm, (SupervisedSpec, LatentPPOSpec))
            and not _is_classic_hpo(self.algorithm)
        ):
            # An epoch-driven run states the interval as evo_epochs; the trainer
            # turns it into steps once it knows the dataset size, so defaulting
            # evo_steps here would put a second, contradictory answer in the payload.
            self.training.evo_steps = min(
                type(self.algorithm).default_evo_steps, self.training.max_steps
            )

        num_envs = self.environment.num_envs
        env_set_num_envs = "num_envs" in self.environment.model_fields_set
        for field_name in ("num_envs", "vect_noise_dim"):
            _copy_when_unset(
                self.algorithm,
                field_name,
                num_envs,
                source_set=env_set_num_envs,
            )

        if isinstance(self.network, FinetuningNetworkSpec):
            _copy_when_unset(
                self.algorithm,
                "pretrained_model_name_or_path",
                self.network.pretrained_model_name_or_path,
                source_set=True,
            )
            _copy_when_unset(
                self.algorithm,
                "max_model_len",
                self.network.max_context_length,
                source_set=True,
            )
            if self.algorithm.lora_config is None:
                # Explicit null is the same as omitted: LoRA on with defaults.
                self.algorithm.model_fields_set.discard("lora_config")
            _copy_when_unset(
                self.algorithm,
                "lora_config",
                self.network.lora_config,
                source_set=self.network.lora_config is not None,
            )

        if isinstance(self.algorithm, LLMAlgorithmSpec):
            algorithm = self.algorithm
            if not isinstance(self.network, FinetuningNetworkSpec):
                if algorithm.lora_config is None:
                    # The network copy above fills the usual case; this covers
                    # a manifest with no network section.
                    algorithm.lora_config = resolve_lora_defaults(
                        LoraConfigDict(), algorithm.pretrained_model_name_or_path
                    )
                else:
                    resolve_lora_defaults(
                        algorithm.lora_config,
                        algorithm.pretrained_model_name_or_path,
                    )
                check_lora_config_supported(
                    algorithm.lora_config, algorithm.pretrained_model_name_or_path
                )
            if (
                algorithm.lora_target_scope is None
                and "lora_target_scope" not in algorithm.model_fields_set
            ):
                algorithm.lora_target_scope = default_lora_target_scope(
                    algorithm.pretrained_model_name_or_path or ""
                )

        if (
            isinstance(self.algorithm, RolloutLLMSpec)
            and isinstance(self.environment, LLMEnvSpec)
            and self.environment.segment_prompt_tokens is not None
            and self.algorithm.max_output_tokens is not None
        ):
            # The harness keeps every prompt within segment_prompt_tokens, so a
            # segment's row is at most that plus one turn's generation. An
            # answer continuation generates up to max_output_tokens more.
            turn_tokens = self.algorithm.max_output_tokens * (
                2 if self.algorithm.answer_continuation else 1
            )
            _copy_when_unset(
                self.algorithm,
                "max_row_tokens",
                min(
                    self.environment.segment_prompt_tokens + turn_tokens,
                    self.algorithm.max_model_len,
                ),
                source_set=True,
            )

        if (
            isinstance(self.algorithm, RolloutLLMSpec)
            and self.training.rollout_mode == "colocated"
            and self.algorithm.vllm_config is None
        ):
            self.algorithm.vllm_config = default_colocated_vllm_config()
        return self

    @model_validator(mode="after")
    def _evo_steps_fit_the_horizon(self) -> Self:
        """Reject an evolution interval longer than the filled run length."""
        self.training.check_evo_steps_fit_horizon()
        return self

    @model_validator(mode="after")
    def _check_async_rollout_buffer(self) -> Self:
        """Async rollout queues generated groups, so it needs the LLM buffer."""
        if not self.training.async_rollout:
            return self
        if not isinstance(self.replay_buffer, LLMRolloutBufferSpec):
            got = (
                type(self.replay_buffer).__name__
                if self.replay_buffer is not None
                else None
            )
            msg = f"Async rollout requires an LLMRolloutBufferSpec, got {got}"
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def _check_prefix_cache_reuse(self) -> Self:
        """Prefix cache reuse spans ``max_rollout_version_lag`` versions, so it needs a bound."""
        if not self.training.reuse_prefix_cache_across_syncs:
            return self
        lag = (
            self.replay_buffer.max_rollout_version_lag
            if isinstance(self.replay_buffer, LLMRolloutBufferSpec)
            else None
        )
        if lag is None or lag < 1:
            msg = (
                "training.reuse_prefix_cache_across_syncs requires an LLM "
                f"replay_buffer with max_rollout_version_lag >= 1, got {lag}"
            )
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def _check_group_replay(self) -> Self:
        """Group replay swaps a member of a group whose returns tie, so it needs groups."""
        if not (
            isinstance(self.replay_buffer, LLMRolloutBufferSpec)
            and self.replay_buffer.group_replay_max_age > 0
        ):
            return self
        # A single-completion group always ties, so every completion would be replaced.
        if not (
            isinstance(self.algorithm, GRPOSpec) and self.algorithm.group_size >= 2
        ):
            msg = (
                "replay_buffer.group_replay_max_age > 0 requires a GRPO-family "
                f"algorithm with group_size >= 2, got {self.algorithm.name}"
            )
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def _check_pretrained_model_resolved(self) -> Self:
        """An LLM run has to name a model, from the algorithm or network section."""
        if (
            isinstance(self.algorithm, LLMAlgorithmSpec)
            and self.algorithm.pretrained_model_name_or_path is None
        ):
            msg = (
                "Required field 'pretrained_model_name_or_path' wasn't found in "
                "the manifest. This is required for LLM finetuning algorithms, "
                "and can be added under either the 'algorithm' or 'network' "
                "sections."
            )
            raise ValueError(msg)
        return self

    @field_serializer("network", mode="wrap")
    def _serialize_network(
        self, value: object, handler: Callable[[object], object], info: object
    ) -> object:
        """Keep ``arch``, the encoder's discriminator, in the dumped section.

        The field is ``exclude=True`` so it never reaches the network kwargs
        the algorithms are built with. A dump without ``arch`` cannot be
        re-validated against the encoder union.
        """
        dumped = handler(value)
        if isinstance(value, NetworkSpec) and isinstance(dumped, dict):
            encoder_config = dumped.get("encoder_config")
            if isinstance(encoder_config, dict):
                encoder_config["arch"] = value.encoder_config.arch
        return dumped

    @property
    def network_arch_is_resolvable(self) -> bool:
        """Whether the encoder architecture is declared rather than inferred."""
        return not isinstance(self.network, DeferredNetworkSpec)

    @staticmethod
    def load(source: str | Path | dict[str, Any]) -> dict[str, Any]:
        """Read a YAML/JSON manifest into a dict, or pass a dict through.

        :param source: Path to a YAML/JSON file, or an already-parsed manifest.
        :type source: str | Path | dict[str, Any]
        :returns: The raw manifest document.
        :rtype: dict[str, Any]
        """
        if isinstance(source, dict):
            return source
        with open(source) as fh:
            return yaml.safe_load(fh)

    @overload
    @classmethod
    def get_validated(
        cls,
        manifest: str | Path | dict[str, Any],
        *,
        mode: Literal["json"] = ...,
    ) -> dict[str, Any]: ...

    @overload
    @classmethod
    def get_validated(
        cls,
        manifest: str | Path | dict[str, Any],
        *,
        mode: Literal["python"],
    ) -> TrainingManifest: ...

    @classmethod
    def get_validated(
        cls,
        manifest: str | Path | dict[str, Any],
        *,
        mode: Literal["json", "python"] = "json",
    ) -> dict[str, Any] | TrainingManifest:
        """Validate a manifest and return it as a payload dict or a model.

        :param manifest: Path to a YAML/JSON file, or a raw manifest dict.
        :type manifest: str | Path | dict[str, Any]
        :param mode: ``"json"`` for the submission payload, ``"python"`` for the
            validated model.
        :type mode: Literal["json", "python"]
        :returns: The submission payload or the validated manifest.
        :rtype: dict[str, Any] | TrainingManifest
        """
        validated = cls.model_validate(cls.load(manifest))
        return validated if mode == "python" else validated.to_payload()

    def to_payload(self) -> dict[str, Any]:
        """Serialize to the JSON submission payload.

        :returns: A JSON-safe manifest document.
        :rtype: dict[str, Any]
        """
        payload = self.model_dump(mode="json", exclude_none=True, by_alias=True)
        # Optional sections dump away under exclude_none. Mutation is filled as
        # a no-op (architecture mutation off) plus rl_hp_selection, and
        # tournament_selection with tournament_size / elitism, so those keys
        # are present for later readers.
        payload.setdefault(
            "mutation",
            MutationSpec(probabilities=no_op_mutation_probabilities()).model_dump(
                mode="json", exclude_none=True
            ),
        )
        payload.setdefault(
            "tournament_selection",
            TournamentSelectionSpec().model_dump(mode="json", exclude_none=True),
        )
        # The network has no safe default: the trainer infers the encoder from
        # the observation space, and inventing one here would pick an
        # architecture the run never asked for.
        if (
            isinstance(self.network, DeferredNetworkSpec)
            and not self.network.model_fields_set
        ):
            payload.pop("network", None)
        # Omit dumped defaults that parsers treat as explicit choices.
        if (
            "rollout_mode" not in self.training.model_fields_set
            and "training" in payload
        ):
            payload["training"].pop("rollout_mode", None)
        if "training" in payload:
            training = payload["training"]
            if not isinstance(self.algorithm, LLMAlgorithmSpec) or (
                "training_gpus_per_agent" not in self.training.model_fields_set
            ):
                training.pop("training_gpus_per_agent", None)
        algorithm = payload.get("algorithm")
        if isinstance(algorithm, dict):
            if "num_envs" not in self.algorithm.model_fields_set:
                algorithm.pop("num_envs", None)
            if "vect_noise_dim" not in self.algorithm.model_fields_set:
                algorithm.pop("vect_noise_dim", None)
            if isinstance(self.algorithm, RolloutLLMSpec):
                vllm_config = self.algorithm.vllm_config
                vllm_payload = algorithm.get("vllm_config")
                if (
                    isinstance(vllm_payload, dict)
                    and vllm_config is not None
                    and "tensor_parallel_size" not in vllm_config.model_fields_set
                ):
                    vllm_payload.pop("tensor_parallel_size", None)
        environment = payload.get("environment")
        if isinstance(environment, dict):
            if environment.get("screenshot_field") is None:
                environment.pop("screenshot_field", None)
            if environment.get("goal_images_field") is None:
                environment.pop("goal_images_field", None)
        return payload
