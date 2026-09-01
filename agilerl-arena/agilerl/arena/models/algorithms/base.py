# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Base classes for the manifest's ``algorithm`` section."""

from __future__ import annotations

import re
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from typing_extensions import Self

from agilerl.arena.models.descriptions import BETA, MICRO_BATCH, MINI_BATCH
from agilerl.arena.models.env import LLMEnvType
from agilerl.arena.models.hpo import RLHyperparameter
from agilerl.arena.models.networks import LoraConfigDict, NetworkSpec
from agilerl.arena.models.registry import AgentType

# Attention backends that build a padding-free (packed) forward. A dense backend
# builds an O(N^2) block-diagonal mask over the packed row instead, so packing
# under one is silently a padded forward.
PACKING_ATTN_IMPLEMENTATIONS = frozenset({"flash_attention_2", "flex_attention"})


def _range(
    minimum: float, maximum: float, grow: float, shrink: float
) -> RLHyperparameter:
    return RLHyperparameter(
        min=minimum, max=maximum, grow_factor=grow, shrink_factor=shrink
    )


RL_HPO_RANGES: dict[str, RLHyperparameter] = {
    "batch_size": _range(8, 512, 1.2, 0.8),
    "lr": _range(0.0000625, 0.01, 1.2, 0.8),
    "lr_actor": _range(0.0000625, 0.01, 1.2, 0.8),
    "lr_critic": _range(0.0000625, 0.01, 1.2, 0.8),
    "gamma": _range(0.9, 0.9999, 1.001, 0.999),
    "tau": _range(0.0001, 0.01, 1.2, 0.8),
    "theta": _range(0.05, 0.25, 1.2, 0.8),
    "dt": _range(0.005, 0.015, 1.2, 0.8),
    "gae_lambda": _range(0.9, 0.99, 1.01, 0.99),
    "clip_coef": _range(0.05, 0.35, 1.15, 0.85),
    "ent_coef": _range(0.001, 0.1, 1.2, 0.8),
    "vf_coef": _range(0.35, 0.65, 1.15, 0.85),
    "max_grad_norm": _range(0.35, 0.65, 1.15, 0.85),
    "beta": _range(0.2, 0.6, 1.2, 0.8),
    "policy_freq": _range(1, 10, 1.5, 0.75),
    "update_epochs": _range(1, 10, 1.5, 0.75),
    "max_seq_len": _range(8, 64, 1.2, 0.8),
    "learn_step": _range(1, 10, 1.5, 0.75),
}

# PPO / IPPO default learn_step is 2048. Mutation bounds match
# make_default_hp_config: [value // 4, value * 4] with grow 1.5 / shrink 0.75.
ON_POLICY_HPO_RANGES: dict[str, RLHyperparameter] = {
    **RL_HPO_RANGES,
    "learn_step": _range(512, 8192, 1.5, 0.75),
}

LLM_HPO_RANGES: dict[str, RLHyperparameter] = {
    "batch_size": _range(1, 32, 1.2, 0.8),
    "lr": _range(1e-12, 0.01, 1.2, 0.8),
    "lr_actor": _range(1e-7, 1e-5, 1.2, 0.8),
    "lr_critic": _range(1e-7, 1e-5, 1.2, 0.8),
    "clip_coef": _range(0.05, 0.35, 1.15, 0.85),
    "max_grad_norm": _range(0.35, 0.65, 1.15, 0.85),
    "beta": _range(0.0001, 0.01, 1.2, 0.8),
    "nll_alpha": _range(0.1, 2, 1.2, 0.8),
    "update_epochs": _range(1, 10, 1.5, 0.75),
    "temperature": _range(0.1, 1.0, 1.2, 0.8),
    "group_size": _range(4, 16, 1.5, 0.75),
}


class AlgorithmSpec(BaseModel):
    """Fields every algorithm's manifest section carries.

    Runtime-only state (torch modules, resolved hyperparameter configs, the
    algorithm class itself) is not declared here; the framework and the runtime
    add those on their own subclasses.
    """

    model_config = ConfigDict(extra="forbid")

    batch_size: int = Field(
        default=64,
        ge=1,
        description=(
            "Transitions sampled per learn step. For the LLM algorithms this is "
            "prompts per step, and the rollout batch is batch_size x group_size."
        ),
    )

    off_policy: ClassVar[bool] = False
    offline: ClassVar[bool] = False
    bandit: ClassVar[bool] = False
    default_evo_steps: ClassVar[int] = 10_000
    hpo_ranges: ClassVar[dict[str, RLHyperparameter]] = {}
    # Registry names that mean more than which class to build. "Recurrent PPO"
    # and "PPO" resolve to the same spec, so without this the recurrent variant
    # would validate into a plain PPO run.
    alias_implies: ClassVar[dict[str, dict[str, Any]]] = {}
    agent_type: ClassVar[AgentType]
    # Fills in environment.env_type when the manifest omits it.
    env_type: ClassVar[str | LLMEnvType] = "gym"
    # Fills in environment.objective the same way: the teacher-forced loss a
    # dataset env trains with. None everywhere except DPO and SFT.
    objective: ClassVar[str | None] = None

    @property
    def name(self) -> str:
        """The registry key this spec was resolved from."""
        return self.__class__.__name__.removesuffix("Spec")


class SingleAgentAlgorithmSpec(AlgorithmSpec):
    """Single-agent reinforcement learning algorithms."""

    learn_step: int = Field(
        default=5,
        ge=1,
        description="Environment steps collected between learn steps.",
    )
    gamma: float = Field(
        default=0.99,
        ge=0.0,
        le=1.0,
        description=(
            "Discount factor on future reward. Lower values make the agent "
            "greedier about near-term reward."
        ),
    )
    normalize_images: bool = Field(
        default=True,
        description="Scale image observations to [0, 1] before the encoder sees them.",
    )
    net_config: NetworkSpec | None = Field(
        default=None,
        description=(
            "Resolved from the manifest's network section; setting it here is "
            "not required."
        ),
    )

    agent_type: ClassVar[AgentType] = AgentType.SingleAgent
    hpo_ranges: ClassVar[dict[str, RLHyperparameter]] = RL_HPO_RANGES


class MultiAgentAlgorithmSpec(AlgorithmSpec):
    """Multi-agent reinforcement learning algorithms."""

    learn_step: int = Field(
        default=2048,
        ge=1,
        description="Environment steps collected between learn steps.",
    )
    gamma: float = Field(
        default=0.99,
        ge=0.0,
        le=1.0,
        description="Discount factor on future reward.",
    )
    normalize_images: bool = Field(
        default=True,
        description="Scale image observations to [0, 1] before the encoder sees them.",
    )
    torch_compiler: str | None = Field(
        default=None,
        description=(
            "torch.compile mode to build the sub-agent networks with, e.g. "
            "'default' or 'reduce-overhead'. Unset runs eager."
        ),
    )

    agent_type: ClassVar[AgentType] = AgentType.MultiAgent
    hpo_ranges: ClassVar[dict[str, RLHyperparameter]] = RL_HPO_RANGES


class LLMAlgorithmSpec(AlgorithmSpec):
    """LLM fine-tuning algorithms.

    ``pretrained_model_name_or_path``, ``max_model_len`` and ``lora_config`` are
    written under the manifest's ``network`` section and lifted onto the
    algorithm when the manifest is validated.
    """

    batch_size: int = Field(
        default=16,
        ge=1,
        description=(
            "Prompts sampled per step; the rollout batch is batch_size x group_size."
        ),
    )
    beta: float = Field(default=0.001, ge=0.0, le=1.0, description=BETA)
    max_grad_norm: float = Field(
        default=0.1,
        ge=0.0,
        description="Gradients are clipped to this global norm before each step.",
    )
    update_epochs: int = Field(
        default=1,
        ge=1,
        description="Optimizer passes over each batch of collected rollouts.",
    )
    use_separate_reference_adapter: bool = Field(
        default=True,
        description=(
            "Hold the reference policy in its own frozen LoRA adapter rather "
            "than the base weights. Incompatible with expert LoRA "
            "(lora_config.target_parameters), which needs a single adapter."
        ),
    )
    calc_position_embeddings: bool = Field(
        default=True,
        description=(
            "Compute position embeddings in the forward pass. Turn off only for "
            "models that build their own."
        ),
    )
    gradient_checkpointing: bool = Field(
        default=True,
        description=(
            "Recompute activations during the backward pass instead of storing "
            "them: much less memory, roughly 30% slower."
        ),
    )
    use_liger_loss: bool = Field(
        default=False,
        description=(
            "Use the fused Liger loss kernel. Cuts memory on long sequences; "
            "requires liger-kernel, which is Linux-only."
        ),
    )
    use_liger_kernel: bool | None = Field(
        default=None,
        description=(
            "Apply Liger's fused layer kernels to the model. Unset lets the "
            "trainer decide from the model architecture."
        ),
    )
    use_mamba_kernels: bool | None = Field(
        default=None,
        description=(
            "Use fused Mamba kernels for state-space models. Unset lets the "
            "trainer decide from the model architecture."
        ),
    )
    cast_logprobs_to_fp32: bool = Field(
        default=True,
        description=(
            "Accumulate log-probabilities in fp32. Guards against the ratio "
            "underflowing in bf16 training."
        ),
    )
    seed: int = Field(
        default=42,
        description="Seed for sampling, data order and initialization.",
    )

    quantization: str | dict[str, Any] | None = Field(
        default=None,
        description=(
            "Trainer-side bitsandbytes quantization: a preset name such as "
            "'nf4', or BitsAndBytesConfig keyword arguments."
        ),
    )
    activation_offload: bool = Field(
        default=False,
        description=(
            "Offload activations to host memory between forward and backward. "
            "Saves GPU memory at the cost of PCIe traffic."
        ),
    )
    use_sequence_packing: bool = Field(
        default=False,
        description=(
            "Pack several short sequences into one batch row so padding stops "
            "wasting compute."
        ),
    )
    lora_target_scope: str | None = Field(
        default=None,
        description=(
            "Restrict which blocks receive LoRA adapters, e.g. only the last N "
            "layers. Unset adapts every matched module."
        ),
    )
    chunk_rows: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Rows per chunk when the loss is computed in slices. Lower values "
            "cut peak memory on long sequences."
        ),
    )
    attn_implementation: str | None = Field(
        default=None,
        description=(
            "Attention backend passed to transformers, e.g. 'sdpa', 'eager' or "
            "'flash_attention_2'. Unset lets the trainer choose."
        ),
    )

    micro_batch_size_per_gpu: int | None = Field(
        default=None, ge=1, description=MICRO_BATCH
    )
    mini_batch_size: int | None = Field(default=None, ge=1, description=MINI_BATCH)
    hf_generate_chunk_size: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Sequences generated per call when generating through transformers "
            "rather than vLLM."
        ),
    )

    thinking_token_budget: int | None = Field(
        default=None,
        ge=0,
        description=(
            "Tokens the model may spend reasoning before it has to answer. "
            "Unset leaves reasoning length unconstrained."
        ),
    )
    answer_pattern: str | None = Field(
        default=None,
        description=(
            "Regex marking the answer span in a completion. Used both to detect "
            "an answer and to constrain generation."
        ),
    )
    answer_continuation: bool = Field(
        default=False,
        description=(
            "After the thinking budget runs out, continue into the answer "
            "instead of ending the episode."
        ),
    )

    zero_stage: int = Field(
        default=2,
        ge=2,
        le=3,
        description=(
            "DeepSpeed ZeRO stage. 2 shards optimizer state and gradients; 3 "
            "also shards parameters, which is slower but fits larger models."
        ),
    )
    deepspeed: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Overrides deep-merged onto the ZeRO stage preset. A nested "
            "zero_optimization.stage is ignored — zero_stage is authoritative."
        ),
    )
    vllm_engine_args: dict[str, Any] = Field(
        default_factory=dict,
        description="Engine arguments forwarded to vLLM verbatim.",
    )
    vllm_lora_export_dir: str | None = Field(
        default=None,
        description=(
            "Directory the trainer writes LoRA adapters to for vLLM to pick up. "
            "Unset uses the run's default adapter root."
        ),
    )

    pretrained_model_name_or_path: str | None = Field(
        default=None,
        min_length=1,
        description=(
            "HuggingFace model id or local path to fine-tune. Normally written "
            "in the network section and lifted onto the algorithm."
        ),
    )
    max_model_len: int = Field(
        default=1024,
        ge=1,
        description=(
            "Context window the run is sized for, prompt plus completion. "
            "Normally written as network.max_context_length."
        ),
    )
    lora_config: LoraConfigDict | None = Field(
        default=None,
        description=(
            "LoRA adapter configuration. Normally written in the network "
            "section and lifted onto the algorithm."
        ),
    )

    agent_type: ClassVar[AgentType] = AgentType.LLMAgent
    default_evo_steps: ClassVar[int] = 5
    hpo_ranges: ClassVar[dict[str, RLHyperparameter]] = LLM_HPO_RANGES
    env_type: ClassVar[LLMEnvType]

    @field_validator("answer_pattern")
    @classmethod
    def _validate_answer_pattern(cls, value: str | None) -> str | None:
        """An answer pattern has to compile, and it may not be anchored.

        Detection searches a turn's text for the pattern while a
        structured-output grammar admits only what the whole pattern matches,
        so an anchored pattern would mean two different things in the two
        places one pattern has to serve.
        """
        if value is None:
            return value
        if value.startswith("^") or (value.endswith("$") and not value.endswith("\\$")):
            msg = (
                f"answer_pattern {value!r} carries an anchor: a structured-output "
                "grammar matches the whole continuation and cannot honour one, so "
                "drop the leading ^ or trailing $"
            )
            raise ValueError(msg)
        try:
            re.compile(value)
        except re.error as err:
            msg = f"answer_pattern is not a valid regular expression: {err}"
            raise ValueError(msg) from err
        return value

    @model_validator(mode="after")
    def _validate_answer_continuation(self) -> Self:
        """The continuation's grammar is built from the answer pattern."""
        if self.answer_continuation and not self.answer_pattern:
            msg = (
                "answer_continuation requires answer_pattern to be set: the "
                "continuation constrains its output to that pattern, so there is "
                "no grammar to generate an answer under without one"
            )
            raise ValueError(msg)
        return self

    def _max_output_tokens(self) -> int | None:
        return None

    @model_validator(mode="after")
    def _validate_thinking_token_budget(self) -> Self:
        """A reasoning budget has to leave answer headroom inside the turn cap."""
        budget = self.thinking_token_budget
        if budget is None:
            return self
        max_output_tokens = self._max_output_tokens()
        if max_output_tokens is None:
            msg = (
                f"thinking_token_budget={budget} requires max_output_tokens to be set: "
                "the reasoning budget is only meaningful against a turn output cap"
            )
            raise ValueError(msg)
        if budget >= max_output_tokens:
            msg = (
                f"thinking_token_budget={budget} must be less than "
                f"max_output_tokens={max_output_tokens} so the turn can still emit an "
                "answer once the reasoning block is closed"
            )
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def _validate_use_sequence_packing(self) -> Self:
        """Packing needs an attention backend that builds the packed forward."""
        if (
            self.use_sequence_packing
            and self.attn_implementation not in PACKING_ATTN_IMPLEMENTATIONS
        ):
            msg = (
                "use_sequence_packing requires attn_implementation in "
                f"{sorted(PACKING_ATTN_IMPLEMENTATIONS)}, got "
                f"{self.attn_implementation!r}: a dense backend builds an O(N^2) "
                "block-diagonal mask over the packed row, so the packed forward "
                "is silently dropped in favour of the padded one"
            )
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def _validate_deepspeed_overrides(self) -> Self:
        """Reject DeepSpeed overrides whose settings the training path ignores."""
        if not self.deepspeed:
            return self
        if "activation_checkpointing" in self.deepspeed:
            msg = (
                "algorithm.deepspeed sets activation_checkpointing, which the "
                "training path never reads: activation checkpointing is configured "
                "through algorithm.gradient_checkpointing / activation_offload."
            )
            raise ValueError(msg)
        if "gradient_clipping" in self.deepspeed:
            msg = (
                "algorithm.deepspeed.gradient_clipping is ignored: the resolved "
                "config always takes gradient_clipping from "
                "algorithm.max_grad_norm. Set algorithm.max_grad_norm instead."
            )
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def _validate_mini_batch_size(self) -> Self:
        """A mini-batch has to be a whole number of micro-batches."""
        if (
            self.mini_batch_size is not None
            and self.micro_batch_size_per_gpu is not None
            and self.mini_batch_size % self.micro_batch_size_per_gpu != 0
        ):
            msg = (
                f"algorithm.mini_batch_size={self.mini_batch_size} is not a "
                "multiple of algorithm.micro_batch_size_per_gpu="
                f"{self.micro_batch_size_per_gpu}: gradient_accumulation_steps "
                "= mini_batch_size / micro_batch_size_per_gpu must be a whole "
                "number of backward passes."
            )
            raise ValueError(msg)
        return self


AlgoSpec = SingleAgentAlgorithmSpec | MultiAgentAlgorithmSpec | LLMAlgorithmSpec
