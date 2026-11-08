# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Static info per supported pretrained model.

LoRA targets, sizing, and allowed ranks keyed by Hub id, for manifest
validation and the memory estimator. Arena runs without model weights, so
each id bundles one file under ``supported/``: ``config`` (its raw
``config.json``) and ``inspected`` (the parameter count and per-target LoRA
dims from a meta-device build of that config). Everything else is read from
those.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Mapping
from functools import cached_property
from importlib.resources import files
from typing import Any, Literal

# Python 3.10: Traversable is importlib.abc, not importlib.resources.abc.
if sys.version_info >= (3, 11):
    from importlib.resources.abc import Traversable
else:
    from importlib.abc import Traversable  # pragma: no cover

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    computed_field,
    field_serializer,
    field_validator,
    model_validator,
)
from typing_extensions import Self

ModelArchitecture = Literal["dense", "hybrid", "moe", "hybrid_moe"]
ARCH_DENSE: ModelArchitecture = "dense"
ARCH_HYBRID: ModelArchitecture = "hybrid"
ARCH_MOE: ModelArchitecture = "moe"
ARCH_HYBRID_MOE: ModelArchitecture = "hybrid_moe"

ModelStatus = Literal["live", "deprecated", "preview"]
STATUS_LIVE: ModelStatus = "live"
STATUS_DEPRECATED: ModelStatus = "deprecated"
STATUS_PREVIEW: ModelStatus = "preview"

ModelModality = Literal["text", "vision"]

# vLLM ``LoRAConfig.max_lora_rank`` values.
VLLM_LORA_RANKS = (1, 8, 16, 32, 64, 128, 256, 320, 512)
# vLLM fused MoE LoRA rejects ranks above this.
FUSED_MOE_LORA_MAX_RANK = 128

# vLLM 0.32 reasoning parser per Hugging Face model_type of a thinking model.
# vLLM enforces thinking_token_budget by forcing the parser's reasoning-end token.
REASONING_PARSERS: Mapping[str, str] = {
    "deepseek_v4": "deepseek_v4",
    "gemma4": "gemma4",
    "gemma4_text": "gemma4",
    "glm5_next": "glm47",
    "glm5_next_text": "glm47",
    "gpt_oss": "openai_gptoss",
    "mimo_v2": "mimo",
    "nemotron_h": "nemotron_v3",
    "nemotron_h_omni": "nemotron_v3",
    "qwen3": "qwen3",
    "qwen3_moe": "qwen3",
    "qwen3_5": "qwen3",
    "qwen3_5_moe": "qwen3",
    "qwen3_5_moe_text": "qwen3",
    "qwen3_5_text": "qwen3",
}

# The DSA indexer only picks top-k keys, so the loss sends it no gradient. vLLM
# replicates f_a_proj and g_a_proj across TP ranks but shards their LoRA, so
# only rank 0 applies it.
GLM5_NEXT_LORA_EXCLUDED_MODULES = frozenset(
    {"indexer.wk", "indexer.wq_b", "indexer.weights_proj", "f_a_proj", "g_a_proj"}
)

# PEFT exclude_modules per Hugging Face model_type: module-path suffixes that
# get no LoRA whatever target_modules names.
LORA_EXCLUDED_MODULES: Mapping[str, frozenset[str]] = {
    # vLLM runs the KV compressor and its indexer as raw matmuls that skip
    # LoRA. o_a_proj is a grouped linear; PEFT's delta mismatches its groups.
    "deepseek_v4": frozenset(
        {
            "o_a_proj",
            "compressor.kv_proj",
            "compressor.gate_proj",
            "compressor.indexer.kv_proj",
            "compressor.indexer.gate_proj",
            "compressor.indexer.q_b_proj",
            "compressor.indexer.scorer.weights_proj",
        }
    ),
    "glm5_next": GLM5_NEXT_LORA_EXCLUDED_MODULES,
    "glm5_next_text": GLM5_NEXT_LORA_EXCLUDED_MODULES,
    # vLLM 0.32 sizes the V slice of the fused qkv LoRA with the QK head size.
    "mimo_v2": frozenset({"q_proj", "k_proj", "v_proj"}),
}

# Hugging Face model_types of supported models that do not reason by default.
NON_REASONING_MODEL_TYPES = frozenset({"granitemoe", "granitemoehybrid", "qwen2"})

# vLLM 0.32 reasoning parsers with no reasoning_end_str, so vLLM cannot force
# the end of reasoning at thinking_token_budget.
PARSERS_WITHOUT_THINKING_BUDGET = frozenset({"openai_gptoss"})

# vLLM 0.32 reasoning parsers a run may set. The model's tool-call parser owns
# where reasoning ends in its tokens; the rollout engine checks at startup that
# the vLLM parser's reasoning_end_str spells the same ids.
KNOWN_REASONING_PARSERS = frozenset(
    {
        "deepseek_r1",
        "deepseek_v4",
        "gemma4",
        "glm47",
        "mimo",
        "nemotron_v3",
        "openai_gptoss",
        "qwen3",
    }
)


def enforces_thinking_budget(reasoning_parser: str | None) -> bool:
    """Return whether vLLM can enforce ``thinking_token_budget`` with this parser.

    :param reasoning_parser: The run's vLLM reasoning parser, or ``None``.
    :return: ``True`` when the parser has a reasoning-end string vLLM can force.
    """
    return (
        reasoning_parser is not None
        and reasoning_parser not in PARSERS_WITHOUT_THINKING_BUDGET
    )


def resolve_reasoning_parser(
    vllm_engine_args: Mapping[str, Any], model_type: str | None
) -> str | None:
    """Return the vLLM reasoning parser a run's rollout engine can enable.

    :param vllm_engine_args: The algorithm's ``vllm_engine_args``.
    :param model_type: Hugging Face ``model_type`` of the checkpoint, or ``None``
        when it is not known offline.
    :return: The manifest's ``reasoning_parser``, else the family's, else ``None``.
    """
    explicit = vllm_engine_args.get("reasoning_parser")
    if explicit or model_type is None:
        return explicit
    return REASONING_PARSERS.get(model_type)


def supported_path(hub_id: str) -> Traversable:
    """Bundled ``config`` + ``inspected`` JSON for ``hub_id``."""
    return files("agilerl.arena.models.supported").joinpath(f"{hub_id}.json")


def _bundled(hub_id: str) -> dict[str, Any]:
    """Parsed ``supported/`` file for ``hub_id``."""
    return json.loads(supported_path(hub_id).read_text(encoding="utf-8"))


class ModelInfo(BaseModel):
    """Static info for one supported Hub id."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    hub_id: str = Field(min_length=1, description="Hugging Face org/name.")
    architecture: ModelArchitecture = Field(
        default=ARCH_DENSE,
        description="Layer layout: dense, hybrid (Mamba), MoE, or hybrid MoE.",
    )
    status: ModelStatus = Field(
        default=STATUS_LIVE,
        description=(
            "Release status: live, deprecated (submits with a warning), or "
            "preview (submission rejected)."
        ),
    )
    modalities: tuple[ModelModality, ...] = Field(
        default=("text",),
        min_length=1,
        description="Input modalities the model handles: text, or text and vision.",
    )
    language_model_prefix: str | None = Field(
        default=None,
        min_length=1,
        description=(
            "Trainer module path of a VL checkpoint's language model, e.g. "
            "model.language_model. None for text-only checkpoints."
        ),
    )
    text_lora_modules: frozenset[str] | None = Field(
        default=None,
        description=(
            "Default target_modules of a VL checkpoint: language-model linears "
            "only. Text-only checkpoints default to every module."
        ),
    )
    rollout_gpu_memory_utilization: float | None = Field(
        default=None,
        gt=0,
        le=1,
        description=(
            "Rollout-engine gpu_memory_utilization when vllm_config omits it, for "
            "models whose vLLM allocations outside that budget overflow the card "
            "at vLLM's 0.9. None keeps 0.9."
        ),
    )

    @model_validator(mode="after")
    def _vl_entries_name_text_modules(self) -> Self:
        if (self.language_model_prefix is None) != (self.text_lora_modules is None):
            msg = "language_model_prefix and text_lora_modules are set together"
            raise ValueError(msg)
        if self.text_lora_modules is not None and (
            unknown := sorted(self.text_lora_modules - self.modules)
        ):
            msg = f"text_lora_modules {unknown} are not LoRA modules of {self.hub_id}"
            raise ValueError(msg)
        return self

    @field_validator("modalities")
    @classmethod
    def _require_text_or_text_and_vision(
        cls, value: tuple[ModelModality, ...]
    ) -> tuple[ModelModality, ...]:
        if value not in (("text",), ("text", "vision")):
            msg = "modalities must be ('text',) or ('text', 'vision')"
            raise ValueError(msg)
        return value

    @property
    def config(self) -> dict[str, Any]:
        """Raw ``config.json`` this id trains from, parsed anew on each access.

        Hugging Face config classes rewrite the sub-config dicts they are built
        from, so callers must not share one dict.
        """
        return _bundled(self.hub_id)["config"]

    @cached_property
    def inspected(self) -> dict[str, Any]:
        """``num_params`` and per-target ``lora_info`` from a meta-device build."""
        return _bundled(self.hub_id)["inspected"]

    @property
    def text_config(self) -> dict[str, Any]:
        """Language-model section of the config; multimodal ids nest it."""
        return (
            self.config.get("text_config")
            or self.config.get("llm_config")
            or self.config
        )

    @computed_field
    @property
    def modules(self) -> frozenset[str]:
        """Valid ``target_modules`` names: keys with a module not in ``LORA_EXCLUDED_MODULES``."""
        return frozenset(
            key
            for key in self._target_keys("module")
            if any(
                not self._is_excluded(key, entry)
                for entry in self.inspected["lora_info"][key]
            )
        )

    @computed_field
    @property
    def default_lora_modules(self) -> frozenset[str]:
        """Default ``target_modules``: the text model's modules only."""
        if self.text_lora_modules is None:
            return self.modules
        return self.text_lora_modules

    @property
    def biases_image_token(self) -> bool:
        """Rollout requests mask the tokenizer's image token with ``logit_bias``.

        Every catalogued vision checkpoint's tokenizer holds an image token,
        and no text-only one does.
        """
        return "vision" in self.modalities

    @property
    def tower_lora_modules(self) -> frozenset[str]:
        """``tower_target_modules`` names: modules outside the language model."""
        if self.language_model_prefix is None:
            return frozenset()
        return frozenset(
            key
            for key, group in self.inspected["lora_info"].items()
            if group[0]["kind"] == "module"
            and any(
                not entry.get("scope", "").startswith(self.language_model_prefix)
                for entry in group
            )
        )

    @computed_field
    @property
    def parameters(self) -> frozenset[str]:
        """Valid packed-expert ``target_parameters`` paths."""
        return self._target_keys("parameter")

    @computed_field
    @property
    def supports_expert_parallel(self) -> bool:
        """True when the model has packed experts the trainer can shard over EP."""
        return bool(self.parameters)

    @computed_field
    @property
    def num_routed_experts(self) -> int | None:
        """Routed expert count; None when the model has no packed experts."""
        if not self.parameters:
            return None
        text_config = self.text_config
        for key in ("n_routed_experts", "num_experts", "num_local_experts"):
            value = text_config.get(key)
            if isinstance(value, int):
                return value
        msg = (
            f"{self.hub_id} has packed experts but text_config has no routed "
            "expert count (n_routed_experts, num_experts, num_local_experts)"
        )
        raise ValueError(msg)

    @computed_field
    @property
    def status_notice(self) -> str | None:
        """What a user of this model should know about its release status."""
        if self.status == STATUS_PREVIEW:
            return f"Model {self.hub_id} is in preview and cannot be submitted yet."
        if self.status == STATUS_DEPRECATED:
            return f"Model {self.hub_id} is deprecated and will be removed."
        return None

    @computed_field
    @property
    def num_params(self) -> int | None:
        """Total parameters; None when the config cannot be built offline."""
        return self.inspected["num_params"]

    @computed_field
    @property
    def lora_ranks(self) -> tuple[int, ...] | None:
        """vLLM ``max_lora_rank`` values this model supports; None without dims.

        Capped by the smallest targetable dim, and by the fused MoE limit when
        the model has packed experts.
        """
        entries = [
            entry
            for key, group in self.inspected["lora_info"].items()
            for entry in group
            if not self._is_excluded(key, entry)
        ]
        dims = [
            entry[field]
            for entry in entries
            for field in ("in_features", "out_features")
            if field in entry
        ]
        if not dims:
            return None
        cap = min(dims)
        if self.parameters:
            cap = min(cap, FUSED_MOE_LORA_MAX_RANK)
        return tuple(rank for rank in VLLM_LORA_RANKS if rank <= cap)

    @computed_field
    @property
    def lora_info(self) -> dict[str, Any]:
        """Per-target LoRA dims plus the ``_gram_estimate`` sidecar."""
        return {
            **self.inspected["lora_info"],
            "_gram_estimate": {
                "hidden_dim": self.hidden_dim,
                "vocab_size": self.vocab_size,
                "num_hidden_layers": self.num_hidden_layers,
            },
        }

    @computed_field
    @property
    def model_type(self) -> str:
        """Hugging Face ``model_type``."""
        return self.config["model_type"]

    @computed_field
    @property
    def max_context_length(self) -> int:
        """Context window the checkpoint supports."""
        return self.text_config["max_position_embeddings"]

    @computed_field
    @property
    def hidden_dim(self) -> int:
        """GRAM estimator hidden size."""
        return self.text_config["hidden_size"]

    @computed_field
    @property
    def vocab_size(self) -> int:
        """GRAM estimator vocab size."""
        return self.text_config["vocab_size"]

    @computed_field
    @property
    def num_hidden_layers(self) -> int:
        """GRAM estimator layer count."""
        text_config = self.text_config
        # Nemotron-H omni configs list blocks instead of counting them.
        if "num_hidden_layers" in text_config:
            return text_config["num_hidden_layers"]
        return len(text_config["layers_block_type"])

    @field_serializer(
        "modules",
        "parameters",
        "text_lora_modules",
        "default_lora_modules",
    )
    def _serialize_sorted(self, value: frozenset[str] | None) -> list[str] | None:
        return None if value is None else sorted(value)

    def _is_excluded(self, key: str, entry: Mapping[str, Any]) -> bool:
        excluded = tuple(
            f".{name}" for name in LORA_EXCLUDED_MODULES.get(self.model_type, ())
        )
        leaf = key.rsplit(".", 1)[-1]
        return f"{entry.get('scope', '')}.{leaf}".endswith(excluded)

    def _target_keys(self, kind: str) -> frozenset[str]:
        return frozenset(
            key
            for key, group in self.inspected["lora_info"].items()
            if group[0]["kind"] == kind
        )


VL_LANGUAGE_MODEL_PREFIX = "model.language_model"
ATTENTION_LORA_MODULES = frozenset({"q_proj", "k_proj", "v_proj", "o_proj"})
GATED_MLP_LORA_MODULES = frozenset({"gate_proj", "up_proj", "down_proj"})
# Gated DeltaNet linear attention; the MoE shared expert uses the gated MLP names.
QWEN3_5_TEXT_LORA_MODULES = (
    ATTENTION_LORA_MODULES
    | GATED_MLP_LORA_MODULES
    | {"in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj"}
)
GEMMA4_TEXT_LORA_MODULES = (
    ATTENTION_LORA_MODULES
    | GATED_MLP_LORA_MODULES
    | {"per_layer_input_gate", "per_layer_projection", "per_layer_model_projection"}
)
NEMOTRON_H_TEXT_LORA_MODULES = ATTENTION_LORA_MODULES | {
    "in_proj",
    "out_proj",
    "up_proj",
    "down_proj",
}
# KDA linear attention plus MLA, less GLM5_NEXT_LORA_EXCLUDED_MODULES.
GLM5_NEXT_TEXT_LORA_MODULES = (
    ATTENTION_LORA_MODULES
    | GATED_MLP_LORA_MODULES
    | {"b_proj", "f_b_proj", "g_b_proj"}
    | {"q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "kv_b_proj"}
)
# vLLM keeps an unsharded expert-LoRA lora_B copy and the NIXL receive buffer
# outside the budget; on an H100-80 engine that leaves room for 0.85, not 0.9.
EXPERT_LORA_ROLLOUT_GPU_MEMORY_UTILIZATION = 0.85


def qwen3_5_vl(hub_id: str, architecture: ModelArchitecture) -> ModelInfo:
    """Catalog entry for a Qwen3.5-family VL checkpoint."""
    return ModelInfo(
        hub_id=hub_id,
        architecture=architecture,
        modalities=("text", "vision"),
        language_model_prefix=VL_LANGUAGE_MODEL_PREFIX,
        text_lora_modules=QWEN3_5_TEXT_LORA_MODULES,
    )


SUPPORTED_MODEL_INFO: Mapping[str, ModelInfo] = {
    info.hub_id: info
    for info in (
        ModelInfo(hub_id="Qwen/Qwen2.5-0.5B-Instruct", status=STATUS_DEPRECATED),
        ModelInfo(hub_id="Qwen/Qwen3-1.7B", status=STATUS_DEPRECATED),
        ModelInfo(hub_id="Qwen/Qwen3-4B", status=STATUS_DEPRECATED),
        qwen3_5_vl("Qwen/Qwen3.5-0.8B", ARCH_HYBRID),
        qwen3_5_vl("Qwen/Qwen3.5-2B", ARCH_HYBRID),
        qwen3_5_vl("Qwen/Qwen3.5-4B", ARCH_HYBRID),
        qwen3_5_vl("Qwen/Qwen3.5-9B", ARCH_HYBRID),
        qwen3_5_vl("Qwen/Qwen3.6-35B-A3B", ARCH_HYBRID_MOE),
        qwen3_5_vl("Qwen/Qwen3.8-27B", ARCH_HYBRID),
        ModelInfo(hub_id="ibm-granite/granite-4.0-micro", status=STATUS_DEPRECATED),
        ModelInfo(
            hub_id="ibm-granite/granite-4.0-micro-base", status=STATUS_DEPRECATED
        ),
        ModelInfo(
            hub_id="ibm-granite/granite-4.0-h-tiny",
            architecture=ARCH_HYBRID,
            status=STATUS_DEPRECATED,
        ),
        ModelInfo(
            hub_id="ibm-granite/granite-3.1-3b-a800m-instruct",
            architecture=ARCH_MOE,
            status=STATUS_DEPRECATED,
        ),
        ModelInfo(
            hub_id="google/gemma-4-E4B-it",
            modalities=("text", "vision"),
            language_model_prefix=VL_LANGUAGE_MODEL_PREFIX,
            text_lora_modules=GEMMA4_TEXT_LORA_MODULES,
        ),
        ModelInfo(
            hub_id="nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16", architecture=ARCH_HYBRID
        ),
        ModelInfo(
            hub_id="nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16",
            architecture=ARCH_HYBRID_MOE,
        ),
        ModelInfo(
            hub_id="nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16",
            architecture=ARCH_HYBRID_MOE,
            modalities=("text", "vision"),
            language_model_prefix="language_model",
            text_lora_modules=NEMOTRON_H_TEXT_LORA_MODULES,
        ),
        ModelInfo(hub_id="openai/gpt-oss-20b", architecture=ARCH_MOE),
        ModelInfo(
            hub_id="zai-org/GLM-5.3-Flash",
            architecture=ARCH_HYBRID_MOE,
            status=STATUS_PREVIEW,
            modalities=("text", "vision"),
            language_model_prefix=VL_LANGUAGE_MODEL_PREFIX,
            text_lora_modules=GLM5_NEXT_TEXT_LORA_MODULES,
            rollout_gpu_memory_utilization=EXPERT_LORA_ROLLOUT_GPU_MEMORY_UTILIZATION,
        ),
        ModelInfo(
            hub_id="deepseek-ai/DeepSeek-V4-Flash-0731",
            architecture=ARCH_MOE,
            status=STATUS_PREVIEW,
            rollout_gpu_memory_utilization=EXPERT_LORA_ROLLOUT_GPU_MEMORY_UTILIZATION,
        ),
        ModelInfo(
            hub_id="XiaomiMiMo/MiMo-V2.6-Flash-RL",
            architecture=ARCH_MOE,
            status=STATUS_PREVIEW,
            rollout_gpu_memory_utilization=EXPERT_LORA_ROLLOUT_GPU_MEMORY_UTILIZATION,
        ),
    )
}
