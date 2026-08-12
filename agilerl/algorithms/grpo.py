# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import functools
import gc
import inspect
import warnings
from collections.abc import Callable
from contextlib import nullcontext
from typing import TYPE_CHECKING, Any, Literal, Protocol

import numpy as np
import numpy.typing as npt
import torch

from agilerl import HAS_LIGER_KERNEL, HAS_LLM_DEPENDENCIES

if TYPE_CHECKING:
    from accelerate import Accelerator
    from peft import LoraConfig
    from transformers import BitsAndBytesConfig

if HAS_LIGER_KERNEL or TYPE_CHECKING:
    from liger_kernel.chunked_loss.grpo_loss import LigerFusedLinearGRPOFunction
else:
    # Keep the name resolvable when liger-kernel isn't installed so unit
    # tests can patch it. ``_liger_loss`` guards against actual use.
    LigerFusedLinearGRPOFunction = None  # type: ignore[assignment]

from agilerl.algorithms.core import ActionResult, LLMAlgorithm
from agilerl.algorithms.core.registry import HyperparameterConfig, NetworkGroup
from agilerl.llm_envs import ReasoningGym
from agilerl.protocols import (
    PeftModelProtocol,
    PreTrainedModelProtocol,
    TokenizedMultiTurnEnv,
)
from agilerl.typing import LLMObsType, LLMRolloutExperiences
from agilerl.utils.algo_utils import (
    CosineLRScheduleConfig,
    VLLMConfig,
    get_experiences_samples,
    stack_and_pad_experiences,
)
from agilerl.utils.llm_packing import (
    pack_padded_batch,
    unpack_hidden_states,
)
from agilerl.utils.llm_utils import (
    aggregate_metrics_dict,
    allreduce_minmax_int,
    attention_mask_from_padded_ids,
    baseline_free_turn_cells,
    build_completion_mask,
    calculate_k3_kl,
    fill_outside_mask,
    is_reasoning_prompts,
    masked_mean,
    masked_whiten,
    needs_cross_rank_seq_padding,
    normalize_reasoning_prompt_batch,
    pool_log_ratio_by_level,
    prepare_prompt_hf_generate,
    resolve_llm_device,
    stitch_completion_after_windowed_hf_generate,
    validate_importance_sampling_level,
    validate_llm_context_lengths,
)

if HAS_LLM_DEPENDENCIES or TYPE_CHECKING:
    from transformers import GenerationConfig

REFERENCE_KL_METRIC = "kl"
"""Metric name for the K3 KL between the actor policy and the reference policy."""

LIGER_CLIP_FRACTION_METRIC = "liger_clip_fraction"
"""Metric name for the clipped-token fraction the fused GRPO kernel reports."""

NUM_ITEMS_PARAM = "num_items_in_batch"

LIGER_TOKEN_NORMALIZED_LOSS_TYPE = {"grpo": "dapo", "cispo": "cispo"}
"""Liger loss type carrying each objective under a token-count normalizer.

Under ``loss_norm="accumulation_window"``, ``grpo`` maps to Liger's ``dapo``
reduction (same per-token objective and clip metric; divisor is
``num_items_in_batch``). ``cispo`` stays ``cispo`` because that Liger type
already divides by the token count.
"""


def _liger_normalizer_world_size() -> int:
    """Ranks the fused kernel divides its token-count normalizer by.

    :return: Default process-group size, ``1`` when distributed is inactive.
    :rtype: int
    """
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return int(torch.distributed.get_world_size())
    return 1


class _FusedKernelClass(Protocol):
    """Class-level surface of a fused-kernel ``autograd.Function``."""

    __name__: str
    forward: Callable[..., Any]


@functools.cache
def _liger_normalizer_slot(
    kernel: _FusedKernelClass,
) -> tuple[tuple[inspect.Parameter, ...], int]:
    """Post-``ctx`` parameters of ``kernel.forward`` and the token-count index, validated once per kernel class."""
    parameters = list(inspect.signature(kernel.forward).parameters.values())
    names = [parameter.name for parameter in parameters]
    if not parameters or names[0] != "ctx":
        msg = (
            f"{kernel.__name__}.forward must start with the autograd 'ctx' "
            f"parameter for its arguments to be filled positionally; got {names}."
        )
        raise RuntimeError(msg)
    parameters = parameters[1:]
    names = names[1:]
    if NUM_ITEMS_PARAM not in names:
        msg = (
            f"{kernel.__name__}.forward does not accept '{NUM_ITEMS_PARAM}', so "
            "the accumulation window's action-token count cannot reach the fused "
            f"normalizer. Signature: {names}."
        )
        raise RuntimeError(msg)
    return tuple(parameters), names.index(NUM_ITEMS_PARAM)


def _liger_args_with_normalizer(
    args: tuple[Any, ...],
    normalizer: float,
) -> tuple[Any, ...]:
    """Fused-kernel arguments extended positionally (``apply`` takes no keywords) to carry ``num_items_in_batch``."""
    parameters, index = _liger_normalizer_slot(LigerFusedLinearGRPOFunction)
    values = list(args)
    for parameter in parameters[len(values) : index + 1]:
        if parameter.default is inspect.Parameter.empty:
            msg = (
                f"Kernel parameter '{parameter.name}' precedes "
                f"'{NUM_ITEMS_PARAM}' and has no default, so the token count "
                "cannot be passed positionally."
            )
            raise RuntimeError(msg)
        values.append(parameter.default)
    values[index] = normalizer
    return tuple(values)


class StandardLossFn(Protocol):
    """Shared signature of the standard (non-Liger) minibatch loss functions."""

    def __call__(
        self,
        mask: torch.Tensor,
        log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        reference_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        turn_ids: torch.Tensor | None = None,
        sampling_log_probs: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]: ...


class GRPO(LLMAlgorithm[LLMRolloutExperiences]):
    """Group Relative Policy Optimization (GRPO).

    Paper: https://arxiv.org/pdf/2402.03300

    :param pad_token_id: Pad token id
    :type pad_token_id: int
    :param pad_token: Pad token
    :type pad_token: str
    :param model_name: Model name
    :type model_name: str, optional
    :param actor_network: HuggingFace LLM
    :type actor_network: PreTrainedModelProtocol
    :param model_config: Model configuration, to be used when creating the model from a name or path
    :type model_config: dict[str, Any], optional
    :param hp_config: RL hyperparameter mutation configuration, defaults to None, whereby algorithm mutations are disabled.
    :type hp_config: HyperparameterConfig, optional
    :param index: Index to keep track of object instance during tournament selection and mutation, defaults to 0
    :type index: int, optional
    :param batch_size: Mini-batch size for learning, defaults to 16
    :type batch_size: int, optional
    :param beta: Beta coefficient, controls the strength of the KL divergence penalty, defaults to 0.001
    :type beta: float, optional
    :param lr: Learning rate for optimizer, defaults to 5e-7
    :type lr: float, optional
    :param clip_coef: Surrogate clipping coefficient as either a symmetric scalar
        (mapped to ``[1-clip_coef, 1+clip_coef]``) or an explicit ratio tuple
        ``(clip_coef_min, clip_coef_max)``.
    :type clip_coef: float | tuple[float, float], optional
    :param max_grad_norm: Maximum norm for gradient clipping, defaults to 0.1
    :type max_grad_norm: float, optional
    :param update_epochs: Number of policy update epochs, defaults to 1
    :type update_epochs: int, optional
    :param group_size: Group size, defaults to 8
    :type group_size: int, optional
    :param temperature: Temperature, controls randomness of text generation
    :type temperature: float, optional
    :param repetition_penalty: Repetition penalty used during generation, defaults to 1.0
    :type repetition_penalty: float, optional
    :param top_p: Top-p nucleus sampling threshold, defaults to 0.95
    :type top_p: float, optional
    :param top_k: Top-k sampling threshold, defaults to 50
    :type top_k: int, optional
    :param min_p: Minimum probability cutoff for sampling, defaults to 0.0
    :type min_p: float, optional
    :param calc_position_embeddings: Flag indicating whether to calculate position embeddings, defaults to True
    :type calc_position_embeddings: bool, optional
    :param micro_batch_size_per_gpu: Trajectories per backward pass on one rank
        (the memory knob). Optimizer-step cadence comes from
        ``mini_batch_size``. If None, derived from the DeepSpeed config's
        gradient_accumulation_steps, defaults to None
    :type micro_batch_size_per_gpu: int, optional
    :param mini_batch_size: Per-rank trajectories covered by one optimizer
        step. DeepSpeed's gradient_accumulation_steps is set to
        ``mini_batch_size / micro_batch_size_per_gpu`` (validated for
        divisibility). Defaults to None, which resolves to
        ``micro_batch_size_per_gpu`` — one optimizer step per micro-batch. Set
        it to the per-rank rollout batch (``batch_size / num_processes x
        group_size``) to accumulate the whole batch into a single step.
    :type mini_batch_size: int, optional
    :param max_output_tokens: Max number of answer tokens, defaults to None
    :type max_output_tokens: int, optional
    :param min_output_tokens: Minimum output tokens, defaults to 0
    :type min_output_tokens: int, optional
    :param max_model_len: Maximum context window length, defaults to 1024
    :type max_model_len: int, optional
    :param hf_generate_chunk_size: Number of prompts per HuggingFace generation
        chunk. Ignored when ``use_vllm=True``.
    :type hf_generate_chunk_size: int | None, optional
    :param lora_config: Config for LoRA, defaults to None
    :type lora_config: LoraConfig, optional
    :param cosine_lr_schedule_config: Config for cosine lr scheduling, defaults to None
    :type cosine_lr_schedule_config: CosineLRScheduleConfig, optional
    :param use_memory_efficient_params: For colocated vLLM, offload the trainer's
        own base to CPU during rollout (and bring it back for the training step)
        so the rollout engine and the trainer never both hold a base on the GPU.
        Defaults to True; inert without colocated vLLM, and disabled under
        DeepSpeed ZeRO-3.
    :type use_memory_efficient_params: bool
    :param accelerator: Accelerator for distributed computing, defaults to None
    :type accelerator: accelerate.Accelerator(), optional
    :param device: Device to train on. Ignored when an accelerator is given (each rank
        owns its own GPU); ``None`` auto-detects CUDA/MPS/CPU.
    :type device: str, optional
    :param wrap: Wrap models for distributed training upon creation, defaults to True
    :type wrap: bool, optional
    :param clone: Flag to indicate if the instantiation is a cloning, defaults to False
    :type clone: bool, optional
    :param use_vllm: Flag to indicate if the model should use vllm for generation, defaults to False
    :type use_vllm: bool, optional
    :param vllm_config: Config for VLLM generation, defaults to None
    :type vllm_config: VLLMConfig, optional
    :param seed: Seed for the random number generator, defaults to 42
    :type seed: int, optional
    :param gradient_checkpointing: Flag to indicate if gradient checkpointing should be used, defaults to True
    :type gradient_checkpointing: bool, optional
    :param torch_compiler: Torch compile mode (e.g. ``'default'``), defaults to None
    :type torch_compiler: str | None, optional
    :param use_liger_loss: Use the Liger fused loss, defaults to ``False``
        (requires ``liger-kernel``; warns and falls back otherwise). **Not
        recommended for GRPO/CISPO/GSPO**: the upstream Liger GRPO kernel shows
        no speedup over AgileRL's already memory-bounded standard path and uses
        slightly more memory. PPO/REINFORCE route ``use_liger_loss`` through a
        different AgileRL liger-based kernel where it *does* help (see their
        docs). The Liger model patches (fused RMSNorm/RoPE/SwiGLU) apply whenever
        ``liger-kernel`` is installed and are independent of this flag.
    :type use_liger_loss: bool, optional
    :param use_kl_advantage_shaping: Apply KL-based shaping directly to token
        advantages before PPO clipping, defaults to False.
    :type use_kl_advantage_shaping: bool, optional
    :param adv_norm: Advantage normalization mode. ``"mean_std"`` divides by
        standard deviation, ``"mean_only"`` only centers, defaults to ``"mean_std"``.
    :type adv_norm: str, optional
    :param loss_type: PPO-style loss variant to optimize. One of ``"grpo"``,
        ``"gspo"``, or ``"cispo"``, defaults to ``"grpo"``. This selects the
        *objective*: ``"grpo"``/``"gspo"`` use the min-clip surrogate,
        ``"cispo"`` the clamped-weight x log-prob objective. ``"gspo"`` is
        sugar for ``"grpo"`` at trajectory level (it forces
        ``importance_sampling_level="trajectory"``).
    :type loss_type: Literal["grpo", "gspo", "cispo"], optional
    :param importance_sampling_level: Granularity at which the importance
        *ratio* is pooled before clipping/weighting, defaults to ``None``
        (resolves to ``"token"``; ``loss_type="gspo"`` forces ``"trajectory"``
        and warns if a different level was requested explicitly). This is
        independent of ``advantage_granularity`` (the advantage axis).

        * ``"token"`` — per-token ratio (standard GRPO / CISPO).
        * ``"turn"``  — pool the per-token log-ratio over each turn (length-
          normalized geometric mean) and clip/weight per turn. Requires
          ``turn_ids`` in :meth:`learn`.
        * ``"trajectory"`` — pool over the whole completion (GSPO).

        Turn/trajectory pooling couples a unit's tokens, so it has no fused Liger
        kernel and runs on the standard (always memory-bounded) path; only token
        level can use the Liger path when ``use_liger_loss=True``.
    :type importance_sampling_level: Literal["token", "turn", "trajectory"] | None, optional
    :param advantage_granularity: Unit at which the group-relative *advantage* is
        computed, independent of ``importance_sampling_level``. Defaults to
        ``"auto"``.

        * ``"trajectory"`` — one group-relative scalar per completion (standard
          GRPO), broadcast to all tokens.
        * ``"turn"`` — group-relative per turn (each turn's reward normalized
          within its group), broadcast to that turn's tokens. Requires
          ``turn_ids`` and per-turn rewards ``(batch, max_turns)`` in
          :meth:`learn`; falls back to trajectory if unavailable.
        * ``"auto"`` — follow the IS level (turn when it is ``"turn"``, else
          trajectory).

        There is no token-level advantage (group-relative needs a per-unit
        reward). Any advantage x IS combination is valid.
    :type advantage_granularity: Literal["auto", "trajectory", "turn"], optional
    :param action_granularity: Deprecated alias for ``advantage_granularity``;
        when set it overrides ``advantage_granularity`` and emits a
        ``DeprecationWarning``.
    :type action_granularity: str | None, optional
    :param use_separate_reference_adapter: Keep a dedicated ``reference`` LoRA
        adapter whose weights are frozen snapshots of the actor used for the
        KL-divergence baseline. When ``False`` the reference log-probs are
        obtained by disabling the actor adapter at inference time.
        Defaults to True.
    :type use_separate_reference_adapter: bool, optional
    :param whiten_advantages: If ``True``, whiten token-level advantages over
        valid action positions, defaults to False.
    :type whiten_advantages: bool, optional
    :param adv_clip_range: Optional symmetric clamp range applied to
        advantages before loss computation, defaults to None.
    :type adv_clip_range: float | None, optional
    :param filter_zero_adv: If ``True``, drop samples whose absolute
        advantage is below ``adv_filter_eps``, defaults to False.
    :type filter_zero_adv: bool, optional
    :param adv_filter_eps: Threshold used with
        ``filter_zero_adv``; samples with ``|advantage| <= eps`` are
        filtered out, defaults to 0.0.
    :type adv_filter_eps: float, optional
    :param turn_advantage_trajectory_fallback: With per-turn advantages, give a
        ``(sample, turn)`` cell whose group has fewer than two members that
        played the turn the sample's trajectory advantage instead of zero,
        defaults to True. Turns nobody played stay at zero either way.
    :type turn_advantage_trajectory_fallback: bool, optional
    :param reduce_memory_peak: Deprecated and ignored; previously hinted
        peak-memory batching. Configure ``micro_batch_size_per_gpu`` instead.
    :type reduce_memory_peak: bool, optional
    :param cast_logprobs_to_fp32: When ``True`` (default), run the per-token
        log-prob reduction (``gather`` / ``logsumexp``) in fp32 before casting
        back to the input dtype, for numerically stable log-probs. ``False`` runs
        it in the input dtype, saving a little memory at the cost of a per-token
        bf16 quantisation error that can bias importance-sampling ratios.
    :type cast_logprobs_to_fp32: bool, optional
    :param chunk_rows: Primary chunk-size knob for fused logit tiles. Applies to
        both standard and Liger paths.
    :type chunk_rows: int | None, optional
    :param quantization_config: Optional ``transformers.BitsAndBytesConfig`` for
        loading the base model in 4-/8-bit (QLoRA). ``lm_head`` is kept
        unquantized so the fused-linear-logprob path stays numerically exact.
    :type quantization_config: BitsAndBytesConfig | None, optional
    :param activation_offload: When ``True``, run the training forward inside
        ``torch.autograd.graph.save_on_cpu`` so tensors saved for backward live
        in pinned host RAM instead of GPU memory. Trades PCIe bandwidth for GPU
        memory (the win grows with sequence length); a no-op during rollout /
        reference forwards.
    :type activation_offload: bool, optional
    :param lora_target_scope: Optional PEFT LoRA path scope for multimodal models
        (e.g. ``"language_model"``). Passed to
        :func:`adapt_lora_config_for_model`.
    :type lora_target_scope: str | None, optional
    :param vllm_importance_sampling_correction: When ``True`` (default) and
        ``use_vllm=True``, correct the rollout/trainer log-prob mismatch by
        weighting each training token by ``clamp(exp(trainer - sampling),
        max=vllm_importance_sampling_cap)``. Active only for training rollouts;
        inert on the HuggingFace path and at eval.
    :type vllm_importance_sampling_correction: bool, optional
    :param vllm_importance_sampling_cap: Upper clamp on the vLLM
        importance-sampling ratio (default ``2.0``), bounding the correction
        weight to limit variance from outlier tokens. Must be > 0.
    :type vllm_importance_sampling_cap: float, optional
    :param use_sequence_packing: Opt in to padding-free sequence packing for the
        gradient forward (sequences pack into one varlen / blockmask pass). Only
        honoured under a FlashAttention-2 / FlexAttention backend, otherwise
        inert; the no-grad reference/old-logprob pass stays padded.
    :type use_sequence_packing: bool, optional
    :param loss_norm: Token population the policy loss is normalized over.
        ``"micro_batch"`` (default) normalizes each micro-batch on its own.
        ``"accumulation_window"`` normalizes by the action tokens of this rank's
        samples entering the optimizer step, so a token weighs the same wherever
        it lands in the rank's gradient-accumulation window; without it a short
        trajectory's tokens outweigh a long one's in proportion to the length
        ratio. Under data parallelism the gradient all-reduce then averages the
        per-rank means. Applies to the standard and the fused Liger path alike.
    :type loss_norm: Literal["micro_batch", "accumulation_window"], optional
    """

    _window_action_tokens: int | None = None
    """Action tokens of this rank's samples entering the optimizer step in progress."""

    _mini_batch_size_default = "micro_batch"

    def __init__(
        self,
        pad_token_id: int,
        pad_token: str,
        model_name: str | None = None,
        actor_network: PreTrainedModelProtocol | None = None,
        model_config: dict[str, Any] | None = None,
        hp_config: HyperparameterConfig | None = None,
        index: int = 0,
        batch_size: int = 16,
        beta: float = 0.001,
        lr: float = 5e-7,
        clip_coef: float | tuple[float, float] = 0.2,
        max_grad_norm: float = 0.1,
        update_epochs: int = 1,
        group_size: int = 8,
        temperature: float = 0.9,
        repetition_penalty: float = 1.0,
        top_p: float = 0.95,
        top_k: int = 50,
        min_p: float = 0.0,
        use_memory_efficient_params: bool = True,
        calc_position_embeddings: bool = True,
        micro_batch_size_per_gpu: int | None = None,
        mini_batch_size: int | None = None,
        max_output_tokens: int | None = None,
        min_output_tokens: int | None = None,
        max_model_len: int | None = 1024,
        hf_generate_chunk_size: int | None = None,
        lora_config: LoraConfig | None = None,
        cosine_lr_schedule_config: CosineLRScheduleConfig | None = None,
        accelerator: Accelerator | None = None,
        device: str | torch.device | None = None,
        wrap: bool = True,
        clone: bool = False,
        use_vllm: bool = False,
        vllm_config: VLLMConfig | None = None,
        seed: int = 42,
        gradient_checkpointing: bool = True,
        torch_compiler: str | None = None,
        use_liger_loss: bool = False,
        use_kl_advantage_shaping: bool = False,
        adv_norm: str = "mean_std",
        loss_type: Literal["grpo", "gspo", "cispo"] = "grpo",
        importance_sampling_level: Literal["token", "turn", "trajectory"] | None = None,
        advantage_granularity: Literal["auto", "trajectory", "turn"] = "auto",
        action_granularity: Literal["auto", "trajectory", "turn"] | None = None,
        use_separate_reference_adapter: bool = True,
        whiten_advantages: bool = False,
        adv_clip_range: float | None = None,
        filter_zero_adv: bool = False,
        adv_filter_eps: float = 0.0,
        turn_advantage_trajectory_fallback: bool = True,
        reduce_memory_peak: bool = False,
        cast_logprobs_to_fp32: bool = True,
        chunk_rows: int | None = None,
        quantization_config: BitsAndBytesConfig | None = None,
        activation_offload: bool = False,
        lora_target_scope: str | None = None,
        vllm_importance_sampling_correction: bool = True,
        vllm_importance_sampling_cap: float = 2.0,
        use_sequence_packing: bool = False,
        loss_norm: Literal["micro_batch", "accumulation_window"] = "micro_batch",
    ) -> None:
        resolved_device = resolve_llm_device(accelerator, device)
        super().__init__(
            index=index,
            batch_size=batch_size,
            lr=lr,
            max_grad_norm=max_grad_norm,
            clone=clone,
            calc_position_embeddings=calc_position_embeddings,
            seed=seed,
            pad_token_id=pad_token_id,
            pad_token=pad_token,
            use_memory_efficient_params=use_memory_efficient_params,
            use_liger_loss=use_liger_loss,
            lora_config=lora_config,
            use_separate_reference_adapter=use_separate_reference_adapter,
            use_vllm=use_vllm,
            vllm_config=vllm_config,
            model_name=model_name,
            actor_network=actor_network,
            model_config=model_config,
            micro_batch_size_per_gpu=micro_batch_size_per_gpu,
            mini_batch_size=mini_batch_size,
            cosine_lr_schedule_config=cosine_lr_schedule_config,
            wrap=wrap,
            hp_config=hp_config,
            device=resolved_device,
            accelerator=accelerator,
            name="GRPO",
            gradient_checkpointing=gradient_checkpointing,
            torch_compiler=torch_compiler,
            reduce_memory_peak=reduce_memory_peak,
            cast_logprobs_to_fp32=cast_logprobs_to_fp32,
            chunk_rows=chunk_rows,
            quantization_config=quantization_config,
            activation_offload=activation_offload,
            use_sequence_packing=use_sequence_packing,
            lora_target_scope=lora_target_scope,
            vllm_importance_sampling_correction=vllm_importance_sampling_correction,
            vllm_importance_sampling_cap=vllm_importance_sampling_cap,
        )
        self._validate_core_args(batch_size, lr, update_epochs, actor_network)
        self.clip_coef, self.clip_coef_min, self.clip_coef_max = (
            self._resolve_clip_coef(clip_coef)
        )
        self.update_epochs = update_epochs
        self.beta = beta
        self.temperature = temperature
        self.repetition_penalty = repetition_penalty
        self.top_p = top_p
        self.top_k = top_k
        self.min_p = min_p
        self.loss_norm = self._resolve_loss_norm(loss_norm)
        self._setup_advantage_options(
            adv_norm,
            group_size,
            advantage_granularity,
            action_granularity,
            whiten_advantages,
            adv_clip_range,
            filter_zero_adv,
            adv_filter_eps,
            turn_advantage_trajectory_fallback,
        )
        self._setup_objective(
            loss_type, importance_sampling_level, use_kl_advantage_shaping
        )
        self._setup_generation(
            max_output_tokens, min_output_tokens, max_model_len, hf_generate_chunk_size
        )

        self._setup_actors(actor_network, clone=clone)
        # Register network groups for mutations
        self.register_network_group(NetworkGroup(eval_network=self.actor, policy=True))
        if self.wrap:
            self.wrap_models()

        # Register metrics to keep track of during training
        self.metrics.register("loss")
        self.metrics.register(self.aux_metric_name)
        self.metrics.register("completion_length")

    def get_action(
        self,
        obs: LLMObsType,
        training: bool = True,
        repeat_prompts: bool = True,
        *args: Any,
        **kwargs: Any,
    ) -> ActionResult:
        """Return generated completions for each prompt (GRPO groups when training).

        :param obs: List of HF-style prompt dicts (this implementation mutates them).
        :type obs: LLMObsType
        :param training: If ``True``, generate with training sampling settings.
        :type training: bool
        :param repeat_prompts: If ``True`` and ``training=True``, duplicate each
            prompt ``self.group_size`` times (legacy GRPO grouped mode). If
            ``False``, treat the batch as already expanded trajectories.
        :type repeat_prompts: bool
        :return: An :class:`ActionResult` of completion token IDs, per-sequence
            action masks, and (when captured) per-completion vLLM sampling
            logprobs for the mismatch correction.
        :rtype: ActionResult
        """
        prompt_batch = normalize_reasoning_prompt_batch(obs)
        group_size = self.group_size if training and repeat_prompts else 1
        # Capture vLLM sampling logprobs only for training rollouts when the
        # mismatch correction is enabled; ``None`` on the HF path / eval.
        sampling_logps: list[torch.Tensor | None] | None = None
        capture_sampling_logps = (
            training and self.use_vllm and self.vllm_importance_sampling_correction
        )
        with self.select_adapter("actor"):
            self.actor.eval()
            if not self.use_vllm:
                actor_module = self._get_unwrapped_actor()
                try:
                    actor_device = next(actor_module.parameters()).device
                except StopIteration:
                    actor_device = torch.device(self.device)
                with torch.inference_mode(), self._amp_ctx():
                    completion_ids = []
                    completion_masks = []

                    for start in range(
                        0,
                        len(prompt_batch),
                        self.hf_generate_chunk_size,
                    ):
                        chunk = prompt_batch[
                            start : start + self.hf_generate_chunk_size
                        ]
                        for prompt_dict in chunk:
                            prompt = prepare_prompt_hf_generate(
                                prompt_dict, actor_device
                            )
                            input_ids = prompt["input_ids"]
                            attention_mask = prompt["attention_mask"]
                            stitch_ids = prompt["stitch_prefix_ids"]
                            initial_prompt_len = prompt["initial_prompt_len"]
                            if training and group_size > 1:
                                input_ids = input_ids.repeat(group_size, 1)
                                attention_mask = attention_mask.repeat(group_size, 1)
                            if (
                                stitch_ids is not None
                                and training
                                and group_size > 1
                                and stitch_ids.shape[0] == 1
                            ):
                                stitch_ids = stitch_ids.repeat(group_size, 1)
                            completion_id = self.actor.generate(
                                input_ids=input_ids,
                                attention_mask=attention_mask,
                                generation_config=self.generation_config,
                            )
                            completion_id, full_prompt_len = (
                                stitch_completion_after_windowed_hf_generate(
                                    completion_id,
                                    stitch_ids,
                                    initial_prompt_len,
                                )
                            )
                            completion_ids.append(completion_id)
                            completion_masks.append(
                                build_completion_mask(
                                    completion_id,
                                    full_prompt_len,
                                    self.pad_token_id,
                                )
                            )
            else:
                self._prepare_vllm_for_generation()
                (
                    completion_ids,
                    completion_masks,
                    sampling_logps,
                ) = self._generate_with_vllm_colocate(
                    prompt_batch,
                    group_size,
                    temperature=self.temperature
                    if training
                    else 0.01,  # Almost deterministic for evaluation
                    capture_sampling_logps=capture_sampling_logps,
                )

        return ActionResult(completion_ids, completion_masks, sampling_logps)

    @property
    def aux_metric_name(self) -> str:
        """Name of the scalar :meth:`learn` reports alongside ``loss``.

        The fused Liger kernel emits a divergence only when it is given
        reference log-probs, which this implementation withholds at
        ``beta == 0.0``; its first auxiliary slot then holds the clipped-token
        fraction. Configuration alone selects the loss path, so this name holds
        for every update of a run.

        :return: :data:`REFERENCE_KL_METRIC` or :data:`LIGER_CLIP_FRACTION_METRIC`.
        :rtype: str
        """
        if self.beta == 0.0 and self._liger_path_selected:
            return LIGER_CLIP_FRACTION_METRIC
        return REFERENCE_KL_METRIC

    def learn(
        self,
        experiences: LLMRolloutExperiences,
        turn_ids: torch.Tensor | None = None,
        sampling_logps: list[torch.Tensor | None] | None = None,
    ) -> dict[str, float]:
        """Update agent network parameters to learn from experiences.

        :param experiences: ``(completion_ids, action_masks, rewards)`` stacked
            batch. For ``importance_sampling_level="turn"`` with per-turn
            rewards, ``rewards`` is ``(batch, max_turns)``; otherwise it is one
            scalar per trajectory (per-turn rewards are summed to the episode
            return).
        :type experiences: LLMRolloutExperiences
        :param sampling_logps: Optional per-row flat vLLM sampling logprobs (one
            1-D tensor per trajectory, generated tokens only; concatenated
            across turns for multi-turn) for the sampling-mismatch correction.
            Parallel to the stacked ``completion_ids`` rows. ``None`` disables
            the correction for this update.
        :type sampling_logps: list[torch.Tensor | None] | None
        :param turn_ids: Optional ``(batch, seq_len-1)`` turn index per action
            token (``-1`` for non-action tokens), aligned with the action
            mask. Consumed independently by the two turn-level features:
            per-turn group-relative advantages (when ``advantage_granularity``
            resolves to ``"turn"``, which needs per-turn rewards) and turn-level
            importance-ratio pooling (when ``importance_sampling_level="turn"``).
            Ignored when neither applies.
        :type turn_ids: torch.Tensor | None
        :return: Dict with averaged ``loss``, :attr:`aux_metric_name` and
            ``completion_length`` (plus the ``vllm_is_*`` sampling-mismatch metrics
            when the correction is active).
        :rtype: dict[str, float]
        """
        gc.collect()
        torch.cuda.empty_cache()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

        self._prepare_vllm_for_training()
        with self.memory_efficient_params_context():
            completion_ids, action_masks, rewards, turn_ids = (
                self._prepare_experience_batch(experiences, turn_ids)
            )
            num_samples = completion_ids.shape[0]
            world_size = (
                self.accelerator.num_processes if self.accelerator is not None else 1
            )
            if (
                needs_cross_rank_seq_padding(self, world_size=world_size)
                and self.accelerator is not None
                and self.accelerator.num_processes > 1
            ):
                seq_len = completion_ids.shape[1]
                min_t, max_t = allreduce_minmax_int(seq_len, self.accelerator)
                if min_t != max_t:
                    msg = (
                        "Cross-rank completion sequence length mismatch before "
                        f"GRPO learn: min_t={min_t}, max_t={max_t}. Ranks must "
                        "pad completions to the same T before learn()."
                    )
                    raise RuntimeError(msg)

            advantages, batch_idxs = self._calculate_advantages(
                rewards, completion_ids, action_masks, turn_ids
            )
            aux_metric = self.aux_metric_name
            effective_num_samples = len(batch_idxs)
            if effective_num_samples == 0:
                warnings.warn(
                    "All samples were filtered by advantage threshold; skipping GRPO update.",
                    stacklevel=2,
                )
                if self.accelerator is not None and self.accelerator.num_processes > 1:
                    self.accelerator.wait_for_everyone()
                return {"loss": 0.0, aux_metric: 0.0}

            updates = 0
            batch_size = (
                min(num_samples, self.micro_batch_size_per_gpu)
                if hasattr(self, "micro_batch_size_per_gpu")
                else num_samples
            )
            with torch.no_grad():
                reference_log_probs, old_log_probs, _ = self._fused_forward_no_grad(
                    completion_ids,
                    batch_size,
                )

            is_turn_ids = turn_ids if self.importance_sampling_level == "turn" else None
            sampling_log_probs, is_metrics = (
                self._aligned_sampling_logprobs_and_metrics(
                    sampling_logps, action_masks, old_log_probs
                )
            )
            learn_metrics = {
                "loss": 0.0,
                aux_metric: 0.0,
            }

            # Ensure batch_size is not larger than the number of active samples
            batch_size = min(batch_size, effective_num_samples)
            self._warn_if_micro_batches_straddle_optimizer_steps(
                effective_num_samples, batch_size
            )
            if self.loss_norm == "accumulation_window":
                window_size = batch_size * self._accumulation_steps()
            else:
                window_size = effective_num_samples
            for _ in range(self.update_epochs):
                self.rng.shuffle(batch_idxs)
                for window_start in range(0, effective_num_samples, window_size):
                    window_idxs = batch_idxs[window_start : window_start + window_size]
                    if self.loss_norm == "accumulation_window":
                        self._record_window_action_tokens(action_masks, window_idxs)
                    for start in range(0, len(window_idxs), batch_size):
                        minibatch_idxs = window_idxs[start : start + batch_size]
                        loss, aux = self._loss(
                            batch_size,
                            minibatch_idxs,
                            completion_ids,
                            action_masks,
                            advantages,
                            old_log_probs,
                            reference_log_probs,
                            turn_ids=is_turn_ids,
                            sampling_log_probs=sampling_log_probs,
                        )
                        self._raise_if_loss_not_finite_on_any_rank(loss)

                        self._backward_pass(loss)
                        learn_metrics["loss"] += loss.item()
                        learn_metrics[aux_metric] += aux.item()
                        updates += 1
        result = {
            metric: value / max(updates, 1) for metric, value in learn_metrics.items()
        }
        completion_list = experiences[0]
        result["completion_length"] = float(
            np.mean([x.shape[-1] for x in completion_list])
        )

        # Aggregate across GPUs and report to the metrics tracker (new API).
        # (Fresh dict display so ty checks the values against the parameter's
        # wider, invariant dict value union.)
        agg = aggregate_metrics_dict(self.accelerator, {**result})
        agg["completion_length"] = int(agg["completion_length"])
        for key, value in agg.items():
            self.metrics.log(key, value)

        # Batch-level sampling-mismatch metrics bypass the per-update averaging.
        result.update(is_metrics)
        return result

    def test(
        self,
        env: ReasoningGym | TokenizedMultiTurnEnv,
        loop: int = 1,
        *args: Any,
        **kwargs: Any,
    ) -> npt.NDArray:
        """Return fitness (test) score of llm on test sub-set.

        :param env: Dataset-style ``ReasoningGym`` environment or tokenized
            multi-turn episode environment.
        :type env: ReasoningGym | TokenizedMultiTurnEnv
        :param loop: Number of outer test iterations over ``reset`` / ``step``.
        :type loop: int
        :return: Concatenated reward tensor from the test loop.
        :rtype: torch.Tensor
        """
        eval_context = getattr(env, "eval_mode", nullcontext)
        with eval_context():
            if isinstance(env, ReasoningGym):
                prompts = env.reset()
                rewards = []
                for _ in range(loop):
                    completion_ids = self.get_action(
                        prompts, training=False
                    ).completion_ids
                    next_prompts, reward = env.step(completion_ids)
                    prompts = next_prompts
                    rewards.append(reward)
                reward_tensor = torch.cat(rewards)
            elif isinstance(env, TokenizedMultiTurnEnv):
                all_rewards: list[torch.Tensor] = []
                for _ in range(loop):
                    prompt_dict, _info = env.reset()
                    terminated, truncated = False, False
                    while not terminated and not truncated:
                        completion_ids = self.get_action(
                            [prompt_dict],
                            training=False,
                        ).completion_ids
                        full = completion_ids[0]
                        obs, reward, terminated, truncated, _info = env.step(full)
                        # ``obs`` is the empty sentinel once the episode ends;
                        # only live prompts feed the next turn.
                        if is_reasoning_prompts(obs):
                            prompt_dict = obs
                        all_rewards.append(
                            torch.tensor(
                                [float(reward)],
                                dtype=torch.float32,
                                device=full.device,
                            )
                        )
                reward_tensor = torch.cat(all_rewards)
            else:
                msg = (
                    "env must be a ReasoningGym (or subclass) or "
                    f"TokenizedMultiTurnEnv; got {type(env).__name__}"
                )
                raise TypeError(msg)
        mean_fit = torch.mean(reward_tensor).item()
        self.metrics.add_fitness(mean_fit)
        if self.accelerator is not None:
            self.accelerator.wait_for_everyone()
        return np.array(mean_fit)

    def _validate_core_args(
        self,
        batch_size: int,
        lr: float,
        update_epochs: int,
        actor_network: PreTrainedModelProtocol | None,
    ) -> None:
        """Validate the core training arguments."""
        assert isinstance(batch_size, int), "Batch size must be an integer."
        assert batch_size >= 1, "Batch size must be greater than or equal to one."
        assert isinstance(lr, float), "Learning rate must be a float."
        assert lr > 0, "Learning rate must be greater than zero."
        assert isinstance(
            update_epochs,
            int,
        ), "Policy update epochs must be an integer."
        assert update_epochs >= 1, (
            "Policy update epochs must be greater than or equal to one."
        )
        if actor_network is not None:
            assert isinstance(
                actor_network,
                (PeftModelProtocol, PreTrainedModelProtocol),
            ), "Actor network must be a PeftModelProtocol or PreTrainedModelProtocol"

    @staticmethod
    def _resolve_clip_coef(
        clip_coef: float | tuple[float, float],
    ) -> tuple[float | tuple[float, float], float, float]:
        """Resolve a scalar or ``(min, max)`` clip_coef to explicit ratio bounds."""
        if isinstance(clip_coef, (tuple, list)):
            if len(clip_coef) != 2:
                msg = "clip_coef tuple must contain exactly two values."
                raise ValueError(msg)
            # min < max is intentionally not enforced for user-provided bounds.
            return clip_coef, float(clip_coef[0]), float(clip_coef[1])
        if isinstance(clip_coef, (float, int)):
            clip_coef = float(clip_coef)
            if clip_coef < 0:
                msg = "clip_coef must be greater than or equal to zero."
                raise ValueError(msg)
            return clip_coef, 1 - clip_coef, 1 + clip_coef
        msg = "clip_coef must be a float or a tuple or list of two floats."
        raise TypeError(msg)

    def _resolve_loss_norm(self, loss_norm: str) -> str:
        """Validate the token population the policy loss is normalized over.

        :param loss_norm: Requested normalization mode.
        :type loss_norm: str
        :return: The validated mode.
        :rtype: str
        :raises ValueError: If the mode is not a supported normalization, or a
            window normalizer is asked of a backend that cannot deliver it.
        """
        if loss_norm not in {"micro_batch", "accumulation_window"}:
            msg = (
                f"Invalid loss_norm '{loss_norm}'. Expected one of "
                "['micro_batch', 'accumulation_window']."
            )
            raise ValueError(msg)
        if loss_norm == "accumulation_window" and not self._uses_deepspeed:
            self._accumulation_steps_without_deepspeed()
        return loss_norm

    def _setup_advantage_options(
        self,
        adv_norm: str,
        group_size: int,
        advantage_granularity: str,
        action_granularity: str | None,
        whiten_advantages: bool,
        adv_clip_range: float | None,
        filter_zero_adv: bool,
        adv_filter_eps: float,
        turn_advantage_trajectory_fallback: bool,
    ) -> None:
        """Validate and store the advantage-computation options."""
        if adv_norm not in {"mean_std", "mean_only"}:
            msg = (
                f"Invalid adv_norm '{adv_norm}'. Expected one of "
                "['mean_std', 'mean_only']."
            )
            raise ValueError(msg)
        if group_size < 2:
            msg = (
                f"group_size must be >= 2 for GRPO-style group-relative "
                f"advantages; got {group_size}. A group of one yields a zero "
                "advantage for every sample (reward minus its own mean), so the "
                "policy receives no gradient signal."
            )
            raise ValueError(msg)
        if adv_clip_range is not None and adv_clip_range <= 0:
            msg = "adv_clip_range must be > 0 when provided."
            raise ValueError(msg)
        if adv_filter_eps < 0:
            msg = "adv_filter_eps must be >= 0."
            raise ValueError(msg)
        if action_granularity is not None:
            warnings.warn(
                "action_granularity is deprecated; use advantage_granularity.",
                DeprecationWarning,
                stacklevel=3,
            )
            advantage_granularity = action_granularity
        if advantage_granularity not in {"auto", "trajectory", "turn"}:
            msg = (
                f"Invalid advantage_granularity '{advantage_granularity}'. Expected "
                "one of ['auto', 'trajectory', 'turn']. The GRPO family has no "
                "token-level advantage (group-relative needs a reward per unit, "
                "and tokens have none) — use 'trajectory' or 'turn'."
            )
            raise ValueError(msg)
        self.adv_norm = adv_norm
        self.group_size = group_size
        self.advantage_granularity = advantage_granularity
        self.whiten_advantages = whiten_advantages
        self.adv_clip_range = adv_clip_range
        self.filter_zero_adv = filter_zero_adv
        self.adv_filter_eps = adv_filter_eps
        self.turn_advantage_trajectory_fallback = turn_advantage_trajectory_fallback

    def _setup_objective(
        self,
        loss_type: str,
        importance_sampling_level: str | None,
        use_kl_advantage_shaping: bool,
    ) -> None:
        """Validate and resolve the objective, IS level, and Liger routing."""
        if loss_type not in {"grpo", "gspo", "cispo"}:
            msg = (
                f"Invalid loss_type '{loss_type}'. "
                "Expected one of ['grpo', 'gspo', 'cispo']."
            )
            raise ValueError(msg)
        if importance_sampling_level is not None:
            validate_importance_sampling_level(
                importance_sampling_level, allow_auto=False
            )
        self.loss_type = loss_type
        if loss_type == "gspo":
            # GSPO is, by definition, the grpo objective at trajectory level.
            if importance_sampling_level not in (None, "trajectory"):
                warnings.warn(
                    "loss_type='gspo' implies trajectory-level importance "
                    "sampling; overriding importance_sampling_level="
                    f"'{importance_sampling_level}' with 'trajectory'.",
                    stacklevel=3,
                )
            self.importance_sampling_level = "trajectory"
        else:
            self.importance_sampling_level = importance_sampling_level or "token"
        if self.loss_type == "cispo" and self.beta != 0:
            warnings.warn(
                "CISPO is typically used with beta=0; nonzero beta adds KL "
                "regularization to the objective.",
                stacklevel=3,
            )
        # Turn-level pooling (and non-token CISPO) has no fused Liger kernel;
        # those combinations run the standard path, which is always
        # memory-bounded via the fused-linear-logprob path.
        self._liger_level_supported = self.importance_sampling_level != "turn" and not (
            loss_type == "cispo" and self.importance_sampling_level != "token"
        )
        if self.use_liger_loss and self.importance_sampling_level in {
            "turn",
            "trajectory",
        }:
            # Warn once, up front, about Liger + non-token IS memory behaviour;
            # suppresses the duplicate loss-time warning (warn-once in the base
            # ``_warn_liger_non_token_is`` helper).
            algo_name = (
                "GSPO" if self.importance_sampling_level == "trajectory" else "GRPO"
            )
            self._warn_liger_non_token_is(self.importance_sampling_level, algo_name)
        if self.use_liger_loss and use_kl_advantage_shaping:
            warnings.warn(
                "use_kl_advantage_shaping is not supported with use_liger_loss=True; "
                "disabling KL advantage shaping.",
                stacklevel=3,
            )
            use_kl_advantage_shaping = False
        self.use_kl_advantage_shaping = use_kl_advantage_shaping
        self._loss_fn = self._resolve_standard_loss_fn()

    def _setup_generation(
        self,
        max_output_tokens: int | None,
        min_output_tokens: int | None,
        max_model_len: int | None,
        hf_generate_chunk_size: int | None,
    ) -> None:
        """Validate context lengths and build the HF generation config."""
        if max_output_tokens is None and max_model_len is None:
            msg = "Either max_output_tokens or max_model_len must be specified"
            raise ValueError(
                msg,
            )
        self.max_output_tokens = (
            max_output_tokens if max_output_tokens is not None else max_model_len
        )
        self.min_output_tokens = min_output_tokens
        resolved_max_model_len = (
            max_model_len if max_model_len is not None else max_output_tokens
        )
        # One of the two is non-None (guarded above).
        assert resolved_max_model_len is not None
        self.max_model_len = resolved_max_model_len
        validate_llm_context_lengths(self.max_model_len, max_output_tokens)
        self.hf_generate_chunk_size = int(
            1 if hf_generate_chunk_size is None else max(1, hf_generate_chunk_size)
        )
        if self.use_vllm and hf_generate_chunk_size is not None:
            warnings.warn(
                "hf_generate_chunk_size is only used for HuggingFace generation "
                "(use_vllm=False) and will be ignored when use_vllm=True.",
                stacklevel=3,
            )
        self.generation_config = GenerationConfig(
            do_sample=True,
            temperature=self.temperature,
            max_length=self.max_model_len,
            max_new_tokens=max_output_tokens,
            min_new_tokens=min_output_tokens,
            pad_token_id=self.pad_token_id,
            repetition_penalty=self.repetition_penalty,
            top_p=self.top_p,
            top_k=self.top_k,
            min_p=self.min_p,
        )

    def _prepare_experience_batch(
        self,
        experiences: LLMRolloutExperiences,
        turn_ids: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Stack and pad the experience batch and move it to the device."""
        completion_ids, action_masks, rewards = stack_and_pad_experiences(
            *experiences,
            padding_values=[self.pad_token_id, False, None],
        )
        action_masks = action_masks.to(self.device)
        rewards = rewards.to(self.device).float()
        completion_ids = completion_ids.to(self.device)
        if turn_ids is not None:
            turn_ids = turn_ids.to(self.device)
            if turn_ids.shape[0] != completion_ids.shape[0]:
                msg = (
                    f"turn_ids batch ({turn_ids.shape[0]}) must match "
                    f"completion batch ({completion_ids.shape[0]})."
                )
                raise ValueError(msg)
        return completion_ids, action_masks, rewards, turn_ids

    def _calculate_advantages(
        self,
        rewards: torch.Tensor,
        completion_ids: torch.Tensor,
        action_masks: torch.Tensor,
        turn_ids: torch.Tensor | None,
    ) -> tuple[torch.Tensor, npt.NDArray]:
        """Group-relative advantages at the resolved granularity, post-processed.

        Post-processing (zero-filter / whiten / clip) is shape-agnostic across
        per-trajectory ``(B, 1)`` and per-turn-broadcast ``(B, T-1)``
        advantages. Returns the advantages and the indices of samples that
        survive the zero-advantage filter (all samples when it is disabled).
        """
        num_samples = completion_ids.shape[0]
        if self._resolve_advantage_granularity() == "turn" and turn_ids is not None:
            advantages = self._turn_broadcast_advantages(
                rewards, turn_ids, action_masks, num_samples
            )
        else:
            advantages = self._trajectory_advantages(
                rewards, num_samples, completion_ids
            )

        active_adv_mask = None
        if self.filter_zero_adv:
            per_sample_abs = (
                advantages.detach().reshape(num_samples, -1).abs().amax(dim=-1)
            )
            active_adv_mask = per_sample_abs > self.adv_filter_eps
        if self.whiten_advantages:
            advantages = self._whiten_advantages(
                advantages, action_masks, active_adv_mask
            )
        if self.adv_clip_range is not None:
            advantages = advantages.clamp(-self.adv_clip_range, self.adv_clip_range)

        if active_adv_mask is None:
            return advantages, np.arange(num_samples)
        return advantages, np.where(active_adv_mask.detach().cpu().numpy())[0]

    def _assert_batch_divisible_by_group(self, num_samples: int) -> None:
        """Require the trajectory batch to split evenly into GRPO groups.

        Called from :meth:`learn` *after* rewards-cardinality validation so a
        rewards/trajectory count mismatch surfaces its own error first.

        :param num_samples: Number of trajectories in the batch.
        :type num_samples: int
        :raises ValueError: If ``num_samples`` is not divisible by
            ``group_size``.
        """
        if num_samples % self.group_size != 0:
            msg = (
                f"Batch size ({num_samples}) must be divisible by "
                f"group_size ({self.group_size}) for GRPO."
            )
            raise ValueError(msg)

    def _calculate_advantage(
        self,
        rewards: torch.Tensor,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        """Calculate the group relative advantage for each groups reward.

        :param rewards: Tensor of rewards.
        :type rewards: torch.Tensor
        :param eps: Epsilon to prevent zero division error, defaults to 1e-8
        :type eps: float, optional
        :return: Tensor of group relative advantages.
        :rtype: torch.Tensor
        :raises ValueError: If the number of elements in ``rewards`` is not
            divisible by ``group_size``.
        """
        numel = rewards.numel()
        if numel % self.group_size != 0:
            msg = (
                f"Rewards must have a total element count divisible by "
                f"group_size ({self.group_size}); got {numel} elements."
            )
            raise ValueError(msg)
        rewards = rewards.view(-1, self.group_size)
        centered_rewards = rewards - rewards.mean(dim=1, keepdim=True)
        if self.adv_norm == "mean_only":
            advantage = centered_rewards
        else:
            advantage = centered_rewards / (rewards.std(dim=1, keepdim=True) + eps)
        return advantage.flatten().unsqueeze(1)

    def _calculate_turn_advantage(
        self,
        rewards: torch.Tensor,
        eps: float = 1e-8,
        turn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Group-relative advantage computed independently per turn.

        Treats each turn as a separate RL action: for turn ``k`` the reward is
        normalized within its group of ``group_size`` completions (the same
        group-relative scheme as :meth:`_calculate_advantage`, applied per
        turn). The caller then broadcasts each ``(sample, turn)`` advantage to
        every token of that turn via ``turn_ids``. ``turn_mask`` restricts each
        group statistic to the members that played the turn, so an episode that
        ended early contributes nothing to the turns it never reached, and cells
        left without a baseline come back as zero.

        :param rewards: Per-turn rewards ``(batch, max_turns)``; the batch dim
            is grouped in contiguous blocks of ``group_size``.
        :type rewards: torch.Tensor
        :param eps: Epsilon guarding the per-turn std division.
        :type eps: float, optional
        :param turn_mask: Boolean ``(batch, max_turns)`` mask of played turns;
            ``None`` lets every entry participate in the statistics.
        :type turn_mask: torch.Tensor | None, optional
        :return: Per-turn advantages ``(batch, max_turns)``.
        :rtype: torch.Tensor
        :raises ValueError: If the batch size is not divisible by ``group_size``.
        """
        batch = rewards.shape[0]
        if batch % self.group_size != 0:
            msg = (
                f"Per-turn rewards batch ({batch}) must be divisible by "
                f"group_size ({self.group_size})."
            )
            raise ValueError(msg)
        num_turns = rewards.shape[1]
        grouped = rewards.view(-1, self.group_size, num_turns)

        if turn_mask is None:
            centered = grouped - grouped.mean(dim=1, keepdim=True)
            if self.adv_norm == "mean_only":
                advantage = centered
            else:
                advantage = centered / (grouped.std(dim=1, keepdim=True) + eps)
            return advantage.reshape(batch, num_turns)

        valid = turn_mask.reshape(-1, self.group_size, num_turns).to(grouped.dtype)
        count = valid.sum(dim=1, keepdim=True)
        mean = (grouped * valid).sum(dim=1, keepdim=True) / count.clamp(min=1.0)
        centered = (grouped - mean) * valid
        if self.adv_norm == "mean_only":
            advantage = centered
        else:
            denom = (count - 1.0).clamp(min=1.0)
            std = (centered.pow(2).sum(dim=1, keepdim=True) / denom).sqrt()
            advantage = centered / (std + eps)
        advantage = torch.where(count > 1, advantage, torch.zeros_like(advantage))
        return advantage.reshape(batch, num_turns)

    def _turn_broadcast_advantages(
        self,
        rewards: torch.Tensor,
        turn_ids: torch.Tensor,
        action_masks: torch.Tensor,
        num_samples: int,
    ) -> torch.Tensor:
        """Per-turn group-relative advantages, broadcast to token positions.

        A turn is the action unit: each turn's reward is normalized within the
        group members that played it (:meth:`_calculate_turn_advantage`), then
        assigned to every token of that turn via ``turn_ids`` and masked to
        action positions. Under
        :attr:`turn_advantage_trajectory_fallback`, a played cell whose group
        has no second member at that turn takes the sample's trajectory
        advantage (:meth:`_calculate_advantage`) rather than zero.

        :param rewards: Per-turn rewards ``(B, max_turns)`` (or flat, reshaped).
        :type rewards: torch.Tensor
        :param turn_ids: ``(B, T-1)`` per-token turn indices (``-1`` = padding).
        :type turn_ids: torch.Tensor
        :param action_masks: ``(B, T-1)`` action-token mask.
        :type action_masks: torch.Tensor
        :param num_samples: Trajectory count in the batch.
        :type num_samples: int
        :return: ``(B, T-1)`` per-token advantages.
        :rtype: torch.Tensor
        :raises ValueError: If ``turn_ids`` reference more turns than rewards
            provide, or the batch is not divisible by ``group_size``.
        """
        self._assert_batch_divisible_by_group(num_samples)
        turn_rewards = (
            rewards if rewards.dim() > 1 else rewards.reshape(num_samples, -1)
        )
        safe_turn_ids = turn_ids.clamp(min=0).to(torch.int64)
        num_reward_turns = turn_rewards.shape[1]
        if int(safe_turn_ids.max().item()) >= num_reward_turns:
            msg = (
                "turn_ids reference a turn index beyond the number of "
                f"reward turns ({num_reward_turns}); rewards and "
                "turn_ids are misaligned."
            )
            raise ValueError(msg)
        turn_counts = torch.zeros(
            (turn_rewards.shape[0], num_reward_turns),
            dtype=torch.int64,
            device=safe_turn_ids.device,
        )
        # Clamping maps padding onto turn 0, so occupancy must accumulate: a
        # plain scatter_ has no defined winner among duplicate indices in a row.
        turn_counts.scatter_add_(1, safe_turn_ids, (turn_ids >= 0).to(torch.int64))
        turn_mask = (turn_counts > 0).to(turn_rewards.device)
        turn_advantages = self._calculate_turn_advantage(
            turn_rewards,
            turn_mask=turn_mask,
        ).to(self.device)
        if self.turn_advantage_trajectory_fallback:
            sparse = baseline_free_turn_cells(turn_mask, self.group_size).to(
                turn_advantages.device,
            )
            trajectory = (
                self._calculate_advantage(turn_rewards.sum(dim=1))
                .reshape(-1, 1)
                .to(turn_advantages.device)
                .expand_as(turn_advantages)
            )
            turn_advantages = torch.where(sparse, trajectory, turn_advantages)
        advantages = turn_advantages.gather(1, safe_turn_ids)  # (B, T-1)
        return advantages * action_masks.to(advantages.dtype)

    def _trajectory_advantages(
        self,
        rewards: torch.Tensor,
        num_samples: int,
        completion_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Per-trajectory group-relative advantage ``(B, 1)``.

        Per-turn reward matrices ``(B, max_turns)`` are collapsed to episode
        returns before the group-relative normalization
        (:meth:`_calculate_advantage`).

        :param rewards: Per-trajectory or per-turn rewards.
        :type rewards: torch.Tensor
        :param num_samples: Trajectory count in the batch.
        :type num_samples: int
        :param completion_ids: Completion ids, used only for error reporting.
        :type completion_ids: torch.Tensor
        :return: ``(B, 1)`` per-trajectory advantages.
        :rtype: torch.Tensor
        :raises ValueError: If rewards don't collapse to one scalar per
            trajectory, or the batch is not divisible by ``group_size``.
        """
        if rewards.dim() > 1 and rewards.shape[0] == num_samples:
            rewards = rewards.sum(dim=1)
        rewards = rewards.flatten()
        if rewards.shape[0] != num_samples:
            msg = (
                "Rewards must provide one scalar per trajectory after "
                f"collapse: got rewards={tuple(rewards.shape)} and "
                f"completion_ids={tuple(completion_ids.shape)}."
            )
            raise ValueError(msg)
        self._assert_batch_divisible_by_group(num_samples)
        return self._calculate_advantage(rewards).to(self.device)

    def _resolve_advantage_granularity(self) -> str:
        """Resolve the unit at which group-relative advantages are computed.

        Independent of :attr:`importance_sampling_level` (the IS / ratio-pooling
        axis). ``"auto"`` follows the IS level — turn IS implies per-turn
        advantages, otherwise per-trajectory — reproducing the original coupled
        default. Explicit ``"trajectory"`` / ``"turn"`` override, enabling any
        advantage x IS combination (e.g. per-turn advantages with token-level
        clipping, or one trajectory advantage with turn-level pooling).

        :return: ``"trajectory"`` or ``"turn"``.
        :rtype: str
        """
        if self.advantage_granularity == "auto":
            return "turn" if self.importance_sampling_level == "turn" else "trajectory"
        return self.advantage_granularity

    def _whiten_advantages(
        self,
        advantages: torch.Tensor,
        action_masks: torch.Tensor,
        active_adv_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Whiten advantages, handling per-trajectory and per-token shapes.

        * Per-trajectory ``(B, 1)``: whiten across (active) samples — the
          original GRPO behavior.
        * Per-token / per-turn ``(B, T-1)``: whiten over valid action
          positions (optionally restricted to active samples).

        :param advantages: ``(B, 1)`` or ``(B, T-1)`` advantages.
        :type advantages: torch.Tensor
        :param action_masks: ``(B, T-1)`` action-token mask.
        :type action_masks: torch.Tensor
        :param active_adv_mask: Optional ``(B,)`` per-sample keep mask.
        :type active_adv_mask: torch.Tensor | None
        :return: Whitened advantages with the same shape as ``advantages``.
        :rtype: torch.Tensor
        """
        if advantages.dim() <= 1 or advantages.shape[-1] == 1:
            adv = advantages.reshape(-1)
            mask = (
                active_adv_mask
                if active_adv_mask is not None and active_adv_mask.any()
                else torch.ones_like(adv, dtype=torch.bool)
            )
        else:
            adv = advantages
            mask = action_masks.bool()
            if active_adv_mask is not None:
                mask = mask & active_adv_mask.unsqueeze(-1)
        if mask.sum() <= 1:
            # Fewer than two whitenable values: variance is undefined, leave
            # the advantages untouched rather than dividing by ~0.
            return advantages
        whitened = masked_whiten(adv, mask.to(adv.dtype), shift_mean=True)
        result = torch.where(mask, whitened, adv)
        return result.reshape(advantages.shape)

    def _resolve_standard_loss_fn(
        self,
    ) -> StandardLossFn:
        """Resolve the active standard (non-Liger) loss function.

        Dispatch is on ``loss_type`` (``grpo``/``gspo`` min-clip vs ``cispo``
        clamped-weight); the importance-sampling level (token/turn/trajectory)
        is applied inside via ``self.importance_sampling_level``.
        """
        if self.loss_type == "cispo":
            return self._cispo_loss
        return self._grpo_loss_standard

    def _apply_kl_advantage_shaping(
        self,
        advantages: torch.Tensor,
        kl: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Apply ART-style zero-mean KL shaping to token advantages."""
        if not self.use_kl_advantage_shaping:
            return advantages
        mask_f = mask.float()
        masked_kl = kl * mask_f
        avg_kl = masked_kl.sum(dim=-1, keepdim=True) / mask_f.sum(
            dim=-1,
            keepdim=True,
        ).clamp(min=1.0)
        return advantages + self.beta * (avg_kl - masked_kl)

    def _record_window_action_tokens(
        self,
        action_masks: torch.Tensor,
        batch_idxs: npt.NDArray,
    ) -> None:
        """Record the action tokens of this rank's samples entering the optimizer step.

        :param action_masks: ``(B, T-1)`` action-token mask for the rank's batch.
        :type action_masks: torch.Tensor
        :param batch_idxs: Indices of the samples surviving the advantage filter.
        :type batch_idxs: npt.NDArray
        :return: None
        :rtype: None
        """
        self._window_action_tokens = int(action_masks[batch_idxs].sum().item())

    def _warn_if_micro_batches_straddle_optimizer_steps(
        self,
        effective_num_samples: int,
        micro_batch_size: int,
    ) -> None:
        """Warn when an epoch's micro-batches do not fill whole optimizer steps.

        Reads the engine accumulation width leniently so mocked or non-DeepSpeed
        actors never fail here; the strict accessor guards the loss path.

        :param effective_num_samples: Trajectories entering this update.
        :type effective_num_samples: int
        :param micro_batch_size: Trajectories per backward pass.
        :type micro_batch_size: int
        :return: None
        :rtype: None
        """
        if not self._uses_deepspeed:
            return
        accessor = getattr(self.actor, "gradient_accumulation_steps", None)
        if not callable(accessor):
            return
        steps = accessor()
        if not isinstance(steps, int) or isinstance(steps, bool) or steps <= 1:
            return
        micro_batches = -(-effective_num_samples // micro_batch_size)
        if micro_batches % steps == 0:
            return
        warnings.warn(
            f"The DeepSpeed engine folds {steps} micro-batches into one "
            f"optimizer step, but this update runs {micro_batches} "
            f"micro-batches per epoch, so the trailing {micro_batches % steps} "
            "micro-batch(es) only reach the optimizer during a later epoch or "
            "learn call. Choose mini_batch_size and micro_batch_size_per_gpu "
            "so the per-rank batch splits into whole optimizer steps.",
            stacklevel=3,
        )

    def _accumulation_steps_without_deepspeed(self) -> int:
        """Micro-batches one optimizer step spans with no DeepSpeed engine.

        :return: ``1``; :meth:`_backward_pass` steps and zeroes the optimizer on
            every micro-batch when no engine owns the accumulation.
        :rtype: int
        :raises ValueError: If the accelerator declares an accumulation width
            wider than one micro-batch, which no backward pass here applies.
        """
        width = (
            1
            if self.accelerator is None
            else self.accelerator.gradient_accumulation_steps
        )
        if width == 1:
            return 1
        msg = (
            f"The accelerator declares gradient_accumulation_steps={width!r}, "
            "but with no DeepSpeed engine each micro-batch takes its own "
            "optimizer step, so a window that wide is never accumulated and "
            "normalizing a loss over it would scale samples that never share a "
            "step. Run under DeepSpeed, which owns the accumulation, or leave "
            "the accelerator's accumulation width at 1."
        )
        raise ValueError(msg)

    def _accumulation_steps(self) -> int:
        """Micro-batches the live engine folds into one optimizer step.

        The DeepSpeed engine divides every micro-batch loss by this value before
        accumulating it, and ``set_train_batch_size`` can move it away from the
        plugin config, so the engine's own accessor is the value that matches
        the scaling actually applied.

        :return: Engine gradient-accumulation steps, ``1`` without DeepSpeed.
        :rtype: int
        :raises TypeError: If the actor exposes no accumulation-steps accessor.
        :raises RuntimeError: If the engine reports a non-positive step count.
        """
        if not self._uses_deepspeed:
            return self._accumulation_steps_without_deepspeed()
        accessor = getattr(self.actor, "gradient_accumulation_steps", None)
        if not callable(accessor):
            msg = (
                "Cannot read the DeepSpeed engine's accumulation steps: "
                f"{type(self.actor).__name__} has no callable "
                "gradient_accumulation_steps, which is the value the engine "
                "scales each micro-batch loss by."
            )
            raise TypeError(msg)
        steps = accessor()
        if not isinstance(steps, int) or isinstance(steps, bool) or steps < 1:
            msg = (
                f"DeepSpeed engine gradient_accumulation_steps() returned "
                f"{steps!r}; the loss cannot be scaled to a window without a "
                "positive step count."
            )
            raise RuntimeError(msg)
        return steps

    def _resolve_loss_window(self, mask: torch.Tensor) -> tuple[int, int] | None:
        """Accumulation steps and action tokens of the window a micro-batch joins.

        A single accumulation step means the optimizer sees exactly this
        micro-batch, so its own mask spans the window.

        :param mask: Action-token mask of the current micro-batch.
        :type mask: torch.Tensor
        :return: Accumulation steps and the window's action-token count, or
            ``None`` when the loss is normalized per micro-batch.
        :rtype: tuple[int, int] | None
        :raises RuntimeError: If the window's action-token count was never
            recorded or is not positive, or a single-step window holds no action
            tokens.
        """
        if self.loss_norm != "accumulation_window":
            return None
        steps = self._accumulation_steps()
        if steps == 1:
            tokens = int(mask.sum().item())
            if tokens <= 0:
                msg = (
                    "Micro-batch action-token count is zero, leaving the loss "
                    "normalizer undefined for an update that spans one "
                    "micro-batch."
                )
                raise RuntimeError(msg)
            return 1, tokens
        window_tokens = self._window_action_tokens
        if window_tokens is None:
            msg = (
                f"{type(self).__name__} has no recorded window action-token "
                "count: the loss ran before learn() counted the action tokens "
                "of the samples entering the update."
            )
            raise RuntimeError(msg)
        if window_tokens <= 0:
            msg = (
                f"The accumulation window holds {window_tokens} action tokens; "
                "the loss cannot be normalized by a non-positive count."
            )
            raise RuntimeError(msg)
        return steps, window_tokens

    def _reduce_masked_loss(
        self,
        loss: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Reduce per-token losses to the per-sequence shares the caller averages.

        Under ``loss_norm="micro_batch"`` a share is that sequence's mean over
        its own action tokens. Under ``loss_norm="accumulation_window"`` the
        caller's mean of the shares is ``steps * masked_sum / window_tokens``,
        which the engine's divide by ``steps`` turns into the window's
        per-token mean once the accumulated micro-batches are summed.

        :param loss: ``(B, T)`` per-token losses.
        :type loss: torch.Tensor
        :param mask: ``(B, T)`` action-token mask.
        :type mask: torch.Tensor
        :return: ``(B,)`` per-sequence contributions.
        :rtype: torch.Tensor
        """
        loss = fill_outside_mask(loss, mask)
        window = self._resolve_loss_window(mask)
        if window is not None:
            steps, window_tokens = window
            return (loss * mask).sum(dim=-1) * (mask.shape[0] * steps / window_tokens)
        denominator = mask.sum(dim=-1)
        denominator = torch.where(
            denominator > 0,
            denominator,
            torch.ones_like(denominator),
        )
        return (loss * mask).sum(dim=-1) / denominator

    def _loss(
        self,
        batch_size: int,
        minibatch_idxs: npt.NDArray,
        completion_ids: torch.Tensor,
        action_mask: torch.Tensor,
        advantages: torch.Tensor,
        old_log_probs: torch.Tensor,
        reference_log_probs: torch.Tensor,
        turn_ids: torch.Tensor | None = None,
        sampling_log_probs: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Slice out a minibatch and compute the active objective loss on it.

        :return: Mean loss and mean KL divergence.
        :rtype: tuple[torch.Tensor, torch.Tensor]
        """
        tensors = [
            completion_ids,
            action_mask,
            advantages,
            old_log_probs,
            reference_log_probs,
        ]
        if turn_ids is not None:
            tensors.append(turn_ids)
        # ``get_experiences_samples`` indexes each input positionally:
        # Tensor in -> Tensor out, so the tuple mirrors the all-Tensor inputs.
        (
            batch_ids,
            batch_action_mask,
            batch_advantages,
            batch_old_log_probs,
            batch_reference_log_probs,
            *rest,
        ) = get_experiences_samples(minibatch_idxs, *tensors)
        batch_turn_ids = rest[0] if rest else None
        batch_sampling_log_probs = (
            sampling_log_probs[minibatch_idxs]
            if sampling_log_probs is not None
            else None
        )
        return self._objective_loss(
            batch_size,
            batch_ids,
            batch_action_mask,
            batch_advantages,
            batch_old_log_probs,
            batch_reference_log_probs,
            batch_turn_ids,
            batch_sampling_log_probs,
        )

    @property
    def _liger_path_selected(self) -> bool:
        """Whether this run's configuration puts updates on the fused Liger kernel.

        The vLLM sampling-mismatch correction is fused into the kernel at
        token-level IS (via ``vllm_is_ratio``); at turn/trajectory level the
        per-token reweight cannot be pooled into the surrogate, so a run that
        enables the correction runs the standard path throughout — rather than
        alternating with the fused path as batches happen to carry sampling
        log-probs, which would change the meaning of the reported auxiliary
        scalar from update to update.

        :return: ``True`` when updates run the fused kernel.
        :rtype: bool
        """
        if not self.use_liger_loss or not self._liger_level_supported:
            return False
        return not (
            self.vllm_importance_sampling_correction
            and self.importance_sampling_level != "token"
        )

    def _use_liger_path(self) -> bool:
        """Whether to run the fused Liger kernel, warning once when it is bypassed.

        :return: ``True`` when updates run the fused kernel.
        :rtype: bool
        """
        if self.use_liger_loss and not self._liger_path_selected:
            self._warn_liger_path_bypassed()
        return self._liger_path_selected

    def _warn_liger_path_bypassed(self) -> None:
        """Warn once that the requested fused kernel cannot serve this run.

        :return: None
        :rtype: None
        """
        if not self._liger_level_supported:
            # Turn-level (and trajectory-level CISPO) pooling has no fused
            # kernel; warn-once in the base helper (already warned at init).
            algo_name = (
                "GSPO" if self.importance_sampling_level == "trajectory" else "GRPO"
            )
            self._warn_liger_non_token_is(self.importance_sampling_level, algo_name)
            return
        if not self._is_correction_liger_warned:
            warnings.warn(
                "use_liger_loss=True fuses the vLLM sampling-mismatch "
                "correction only at token-level importance sampling; "
                f"importance_sampling_level='{self.importance_sampling_level}' "
                "uses the standard PyTorch path. Set "
                "vllm_importance_sampling_correction=False to run the fused "
                "kernel without the correction.",
                stacklevel=2,
            )
            self._is_correction_liger_warned = True

    def _objective_loss(
        self,
        batch_size: int,
        batch_ids: torch.Tensor,
        action_mask: torch.Tensor,
        advantages: torch.Tensor,
        old_log_probs: torch.Tensor,
        reference_log_probs: torch.Tensor,
        turn_ids: torch.Tensor | None,
        sampling_log_probs: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the configured objective on one minibatch.

        Uses the fused Liger kernel when supported, otherwise the standard
        loss function at the configured importance-sampling level.
        """
        if self._use_liger_path():
            return self._liger_loss(
                batch_ids,
                action_mask,
                advantages,
                old_log_probs,
                reference_log_probs,
                sampling_log_probs=sampling_log_probs,
            )
        log_probs = self._get_logprobs(
            batch_ids,
            batch_size=batch_size,
            use_reference=False,
            eval_mode=False,
        )
        return self._loss_fn(
            action_mask,
            log_probs,
            old_log_probs,
            reference_log_probs,
            advantages,
            turn_ids,
            sampling_log_probs=sampling_log_probs,
        )

    def _log_importance_weights(
        self,
        token_log_ratio: torch.Tensor,
        mask: torch.Tensor,
        turn_ids: torch.Tensor | None,
        level: str,
    ) -> torch.Tensor:
        """Pool per-token log-ratios to the configured importance-sampling level.

        Returns a ``(B, T)``-broadcastable log importance-weight tensor:

        * ``"token"``    → the per-token log-ratio unchanged ``(B, T)``.
        * ``"trajectory"`` → length-normalized masked mean over the whole
          completion ``(B, 1)`` (GSPO).
        * ``"turn"``     → length-normalized masked mean within each turn,
          scattered back to every token of that turn ``(B, T)``. Falls back to
          the trajectory-level mean when ``turn_ids`` is ``None`` (a single
          turn is exactly trajectory-level).

        All three forms feed an identical downstream surrogate/clip; the only
        difference is the granularity at which the ratio is pooled. The
        per-turn pool is the same geometric-mean (length-normalized) form as
        the trajectory pool, just restricted to a turn's tokens.

        :param token_log_ratio: ``(B, T)`` per-token ``log pi_theta - log pi_old``.
        :type token_log_ratio: torch.Tensor
        :param mask: ``(B, T)`` action-token mask.
        :type mask: torch.Tensor
        :param turn_ids: ``(B, T)`` turn index per token (``-1`` non-action) or
            ``None``.
        :type turn_ids: torch.Tensor | None
        :param level: ``"token"`` / ``"turn"`` / ``"trajectory"``.
        :type level: str
        :return: ``(B, T)`` or ``(B, 1)`` log importance weights.
        :rtype: torch.Tensor
        """
        # Token level is the identity; turn level without turn_ids degenerates
        # to trajectory-wide pooling (both via pool_log_ratio_by_level).
        if level == "token":
            return token_log_ratio
        if level == "trajectory" or turn_ids is None:
            log_importance_weights, _ = pool_log_ratio_by_level(
                token_log_ratio, mask, None, "trajectory"
            )
            return log_importance_weights

        # turn-level: per-turn length-normalized mean, then scattered back to
        # tokens to preserve the ``(B, T)`` token-broadcast contract. Non-action
        # tokens map to turn 0 but are dropped by the masked reduction later.
        num_turns = max(int(turn_ids.max().item()) + 1, 1)
        turn_log_importance_weights, _ = pool_log_ratio_by_level(
            token_log_ratio, mask, turn_ids, "turn", num_turns
        )
        safe_turn_ids = turn_ids.clamp(min=0).to(torch.int64)
        return turn_log_importance_weights.gather(1, safe_turn_ids)

    def _compute_policy_loss(
        self,
        mask: torch.Tensor,
        log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        reference_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        turn_ids: torch.Tensor | None,
        level: str,
        objective: str,
        sampling_log_probs: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Shared GRPO-family surrogate over any importance-sampling level.

        The importance ratio is pooled to ``level`` (token/turn/trajectory) via
        :meth:`_log_importance_weights`; everything downstream — the clipped
        ``min`` surrogate (``objective="grpo"``) or the clamped-weight x
        log-prob objective (``objective="cispo"``), the optional KL term, and
        the masked reduction — is shape-agnostic and identical across levels.

        :param mask: ``(B, T)`` action-token mask.
        :type mask: torch.Tensor
        :param log_probs: ``(B, T)`` current-policy log-probs.
        :type log_probs: torch.Tensor
        :param old_log_probs: ``(B, T)`` old-policy log-probs.
        :type old_log_probs: torch.Tensor
        :param reference_log_probs: ``(B, T)`` reference-policy log-probs.
        :type reference_log_probs: torch.Tensor
        :param advantages: ``(B, 1)`` (per-trajectory) or ``(B, T)`` (per-turn,
            broadcast to tokens) advantages.
        :type advantages: torch.Tensor
        :param turn_ids: ``(B, T)`` turn index per token, or ``None``.
        :type turn_ids: torch.Tensor | None
        :param level: importance-sampling level.
        :type level: str
        :param objective: ``"grpo"`` or ``"cispo"``.
        :type objective: str
        :return: Mean loss and mean KL divergence.
        :rtype: tuple[torch.Tensor, torch.Tensor]
        """
        log_probs = fill_outside_mask(log_probs, mask)
        old_log_probs = fill_outside_mask(old_log_probs, mask)
        reference_log_probs = fill_outside_mask(reference_log_probs, mask)
        if sampling_log_probs is not None:
            sampling_log_probs = fill_outside_mask(sampling_log_probs, mask)
        kl = calculate_k3_kl(log_probs, reference_log_probs)
        advantages = self._apply_kl_advantage_shaping(advantages, kl, mask)
        token_log_ratio = log_probs - old_log_probs
        log_importance_weights = self._log_importance_weights(
            token_log_ratio, mask, turn_ids, level
        )
        ratio = torch.exp(log_importance_weights)
        if objective == "cispo":
            clamped_ratio = ratio.clamp(
                min=self.clip_coef_min,
                max=self.clip_coef_max,
            ).detach()
            loss = -(clamped_ratio * advantages * log_probs)
        else:
            clipped_ratio = ratio.clamp(self.clip_coef_min, self.clip_coef_max)
            loss = -torch.min(ratio * advantages, clipped_ratio * advantages)
        if sampling_log_probs is not None:
            # Truncated IS: reweight the policy term by the detached, clamped
            # trainer/vLLM probability ratio *before* the KL penalty, matching
            # the fused Liger kernel so use_liger_loss is a pure perf toggle.
            with torch.no_grad():
                mask_f = mask.to(loss.dtype)
                is_ratio = torch.exp(
                    (old_log_probs - sampling_log_probs) * mask_f
                ).clamp(max=self.vllm_importance_sampling_cap)
            loss = loss * is_ratio
        if not self.use_kl_advantage_shaping:
            loss = loss + self.beta * kl
        loss = self._reduce_masked_loss(loss, mask)
        # Average the KL metric over action tokens only — masked positions have
        # meaningless logprobs that explode the k3 estimator.
        return loss.mean(), masked_mean(kl, mask)

    def _grpo_loss_standard(
        self,
        mask: torch.Tensor,
        log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        reference_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        turn_ids: torch.Tensor | None = None,
        sampling_log_probs: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """GRPO min-clip surrogate at ``self.importance_sampling_level``.

        With the default token level this is standard GRPO; with
        ``importance_sampling_level="turn"`` / ``"trajectory"`` the importance
        ratio is pooled per turn / per trajectory (the latter is GSPO).

        :param mask: Action-token mask.
        :type mask: torch.Tensor
        :param log_probs: Current-policy log-probs.
        :type log_probs: torch.Tensor
        :param old_log_probs: Old-policy log-probs.
        :type old_log_probs: torch.Tensor
        :param reference_log_probs: Reference-policy log-probs.
        :type reference_log_probs: torch.Tensor
        :param advantages: ``(B, 1)`` or ``(B, T)`` advantages.
        :type advantages: torch.Tensor
        :param turn_ids: ``(B, T)`` turn indices (required for turn level).
        :type turn_ids: torch.Tensor | None
        :param sampling_log_probs: Optional ``(B, T-1)`` vLLM sampling logprobs
            for the sampling-mismatch correction.
        :type sampling_log_probs: torch.Tensor | None
        :return: Mean loss and mean KL divergence.
        :rtype: tuple[torch.Tensor, torch.Tensor]
        """
        return self._compute_policy_loss(
            mask,
            log_probs,
            old_log_probs,
            reference_log_probs,
            advantages,
            turn_ids,
            level=self.importance_sampling_level,
            objective="grpo",
            sampling_log_probs=sampling_log_probs,
        )

    def _gspo_loss(
        self,
        mask: torch.Tensor,
        log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        reference_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        turn_ids: torch.Tensor | None = None,
        sampling_log_probs: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate GSPO trajectory-level ratio clipped loss."""
        return self._compute_policy_loss(
            mask,
            log_probs,
            old_log_probs,
            reference_log_probs,
            advantages,
            turn_ids,
            level="trajectory",
            objective="grpo",
            sampling_log_probs=sampling_log_probs,
        )

    def _cispo_loss(
        self,
        mask: torch.Tensor,
        log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        reference_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        turn_ids: torch.Tensor | None = None,
        sampling_log_probs: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """CISPO clamped-ratio weighted log-prob objective at the configured level."""
        return self._compute_policy_loss(
            mask,
            log_probs,
            old_log_probs,
            reference_log_probs,
            advantages,
            turn_ids,
            level=self.importance_sampling_level,
            objective="cispo",
            sampling_log_probs=sampling_log_probs,
        )

    def _liger_loss(
        self,
        batch_ids: torch.Tensor,
        action_mask: torch.Tensor,
        advantages: torch.Tensor,
        old_log_probs: torch.Tensor,
        reference_log_probs: torch.Tensor,
        sampling_log_probs: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the fused Liger loss inside the activation-offload context.

        The fused path is the whole gradient-bearing forward when
        ``use_liger_loss=True``, so ``activation_offload`` reaches the training
        forward only here.

        :param batch_ids: Input token IDs.
        :type batch_ids: torch.Tensor
        :param action_mask: Boolean action mask (B, seq_len-1).
        :type action_mask: torch.Tensor
        :param advantages: Per-sample advantages (B,) or (B, 1).
        :type advantages: torch.Tensor
        :param old_log_probs: Log probs from the frozen old policy (B, seq_len-1).
        :type old_log_probs: torch.Tensor
        :param reference_log_probs: Log probs from the reference policy (B, seq_len-1).
        :type reference_log_probs: torch.Tensor
        :param sampling_log_probs: Optional ``(B, seq_len-1)`` vLLM sampling
            logprobs for the sampling-mismatch correction.
        :type sampling_log_probs: torch.Tensor | None
        :return: Mean loss and mean KL divergence (or clip-fraction when ``beta=0``).
        :rtype: tuple[torch.Tensor, torch.Tensor]
        """
        with self._activation_offload_ctx():
            return self._fused_kernel_loss(
                batch_ids,
                action_mask,
                advantages,
                old_log_probs,
                reference_log_probs,
                sampling_log_probs,
            )

    def _fused_kernel_loss(
        self,
        batch_ids: torch.Tensor,
        action_mask: torch.Tensor,
        advantages: torch.Tensor,
        old_log_probs: torch.Tensor,
        reference_log_probs: torch.Tensor,
        sampling_log_probs: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate the loss using the Liger Triton-fused kernel.

        Dispatches to the appropriate Liger ``loss_type`` /
        ``importance_sampling_level`` from ``self.loss_type`` and
        ``self.importance_sampling_level``:

        * grpo  @ token    → ``loss_type="grpo"``,  ``importance_sampling_level="token"``
        * grpo  @ trajectory → ``loss_type="grpo"``,  ``importance_sampling_level="trajectory"`` (GSPO)
        * cispo @ token    → ``loss_type="cispo"``, ``importance_sampling_level="token"``

        Under ``loss_norm="accumulation_window"`` the objective keeps its
        per-token form and clip metric but moves to the Liger loss type whose
        reduction divides by ``num_items_in_batch``
        (:data:`LIGER_TOKEN_NORMALIZED_LOSS_TYPE`), which is handed the
        window's action-token count; the returned scalar is then scaled by the
        accumulation steps the engine divides it by.

        Turn-level (and trajectory-level CISPO) never reach here — ``_loss``
        routes them to the standard PyTorch path because Liger's fused GRPO
        kernel has no turn mode.

        CISPO note: Liger's CISPO only clips importance weights from above
        (no lower bound), so ``epsilon_high`` is passed as the **absolute**
        upper bound ``self.clip_coef_max`` rather than the offset
        ``self.clip_coef_max - 1.0`` used by GRPO/GSPO.

        Sequence packing co-exists with the fused kernel: when a
        varlen/block-sparse backend is active (``_packing_mode``), the
        transformer forward runs on a single padding-free packed row and the
        resulting hidden states are scattered back onto the padded
        ``(B, T, H)`` frame (:func:`unpack_hidden_states`) before the kernel
        call, which is then identical to the unpacked path. This bounds the
        forward to real tokens; the kernel's own logit chunking is unchanged.

        :param batch_ids: Input token IDs.
        :type batch_ids: torch.Tensor
        :param action_mask: Boolean action mask (B, seq_len-1).
        :type action_mask: torch.Tensor
        :param advantages: Per-sample advantages (B,) or (B, 1).
        :type advantages: torch.Tensor
        :param old_log_probs: Log probs from the frozen old policy (B, seq_len-1).
        :type old_log_probs: torch.Tensor
        :param reference_log_probs: Log probs from the reference policy (B, seq_len-1).
        :type reference_log_probs: torch.Tensor
        :param sampling_log_probs: Optional ``(B, seq_len-1)`` vLLM sampling
            logprobs. When present (token-level IS only), the truncated
            importance-sampling ratio is fused into the kernel via
            ``vllm_is_ratio``; ``None`` disables the correction.
        :type sampling_log_probs: torch.Tensor | None
        :return: Mean loss and mean KL divergence (or clip-fraction when ``beta=0``).
        :rtype: tuple[torch.Tensor, torch.Tensor]
        """
        if not HAS_LIGER_KERNEL:
            msg = (
                "Liger loss was requested but `liger-kernel` is not available. "
                "Set use_liger_loss=False."
            )
            raise ImportError(msg)

        # Resolve Liger API parameters from the loss type + level.
        # ``_loss`` only routes here for Liger-supported combinations
        # (grpo @ token/trajectory, cispo @ token); turn-level never reaches this.
        importance_sampling_level = self.importance_sampling_level
        if self.loss_type == "cispo":
            liger_loss_type = "cispo"
            importance_sampling_level = "token"
            # Liger CISPO clamps importance weights against an *absolute* upper
            # bound (epsilon_high = clip_coef_max), not an offset from 1.0.
            epsilon_low = 1.0 - self.clip_coef_min  # unused by Liger CISPO
            epsilon_high = self.clip_coef_max
        else:  # "grpo" objective (token or trajectory/GSPO level)
            liger_loss_type = "grpo"
            epsilon_low = 1.0 - self.clip_coef_min
            epsilon_high = self.clip_coef_max - 1.0

        batch_ids = batch_ids.to(self.device)
        mask = action_mask.to(self.device).contiguous()  # (B, seq_len-1)
        window = self._resolve_loss_window(mask)
        if window is not None:
            liger_loss_type = LIGER_TOKEN_NORMALIZED_LOSS_TYPE[liger_loss_type]
        # Drop a trailing singleton dim only — squeezing a 1-D (1,) would
        # collapse it to a scalar.
        adv = advantages.to(self.device).contiguous()
        if adv.dim() > 1 and adv.shape[-1] == 1:
            adv = adv.squeeze(-1)  # (B, 1) -> (B,)
        old_log_probs = fill_outside_mask(
            old_log_probs.to(self.device).contiguous(),
            mask,
        )
        ref_log_probs: torch.Tensor | None = (
            fill_outside_mask(reference_log_probs.to(self.device).contiguous(), mask)
            if self.beta != 0.0
            else None
        )
        lm_head = self._get_lm_head()
        lm_head_weight = lm_head.weight
        lm_head_bias = lm_head.bias

        attention_mask = attention_mask_from_padded_ids(
            batch_ids, self.pad_token_id
        ).long()
        # Sequence packing (same gate as the standard path): on a varlen/block-
        # sparse backend, flatten real tokens into a padding-free forward and
        # scatter hidden states back onto the padded ``(B, T, H)`` frame so the
        # Liger call below is byte-for-byte the padded path. Dense backends
        # return ``None`` and fall back to the padded forward.
        packing_mode = self._packing_mode()
        packed = None
        if packing_mode is not None:
            packed = pack_padded_batch(batch_ids, attention_mask)
            # Per-sequence position_ids (no mask): transformers detects the
            # packed format and keeps sequences attention-isolated per layer.
            model_kwargs = {
                "input_ids": packed.input_ids,
                "position_ids": packed.position_ids,
                "use_cache": False,
            }
        else:
            model_kwargs = {
                "input_ids": batch_ids,
                "attention_mask": attention_mask,
                "use_cache": False,
            }
            if self.calc_position_embeddings:
                model_kwargs["position_ids"] = self._position_ids_from_mask(
                    attention_mask
                )
        # Identity-patch lm_head: the forward yields hidden states; the fused
        # kernel handles the lm_head matmul itself.
        with (
            self._patch_lm_head_to_identity(),
            self.select_adapter("actor"),
            self._amp_ctx(),
        ):
            self.actor.train()
            actor_output = self.actor(**model_kwargs)
        policy_hidden = (
            actor_output[0] if isinstance(actor_output, tuple) else actor_output.logits
        )  # packed (1, N, H) or padded (B, seq_len, H)
        if packed is not None:
            # Scatter packed hidden states back onto the padded (B, T, H) frame
            # so the kernel call below is identical to the padded path. Pad rows
            # are zeroed and masked out by ``action_mask`` downstream.
            policy_hidden = unpack_hidden_states(policy_hidden, packed)
        # The kernel weights its own per-token logprobs by ``mask``, so a
        # non-finite hidden row at an out-of-mask position would poison the fused
        # reduction (``nan * 0``). Zeroing those rows makes their logits the bias
        # alone; per-token independence leaves in-mask logprobs untouched.
        hidden_keep = torch.zeros(
            policy_hidden.shape[:2],
            dtype=torch.bool,
            device=policy_hidden.device,
        )
        hidden_keep[:, : mask.shape[1]] = mask.to(torch.bool)
        policy_hidden = fill_outside_mask(policy_hidden, hidden_keep.unsqueeze(-1))
        target_ids = batch_ids[:, 1:].contiguous()  # (B, seq_len-1)

        # vLLM sampling-mismatch correction (token-level IS only): the detached,
        # upper-clamped trainer/vLLM ratio is token-flattened to (n_tokens, 1)
        # below and fused into the kernel. None for trajectory (GSPO), which
        # routes the correction to the standard path via ``_use_liger_path``.
        vllm_is_ratio_arg = None

        # Token-level IS: flatten (B, T, H) -> (B*T, 1, H) so the fused kernel
        # chunks over tokens, bounding each chunk's logits to
        # (chunk_tokens, vocab) — exact for token-level IS. Trajectory-level
        # (GSPO) couples a sequence's tokens, so it keeps the padded layout
        # and chunks one sequence at a time.
        if importance_sampling_level == "token":
            batch, _seq_len, hidden_dim = policy_hidden.shape
            n_act = target_ids.shape[1]  # seq_len - 1
            n_tokens = batch * n_act
            # Flatten per-trajectory ((batch,) / (batch, 1)) or per-token
            # ((batch, n_act)) advantages to (n_tokens,).
            if adv.ndim == 1 and adv.shape[0] == batch:
                adv_arg = adv.unsqueeze(1).expand(batch, n_act).reshape(n_tokens)
            elif adv.ndim == 2 and adv.shape == (batch, n_act):
                adv_arg = adv.reshape(n_tokens)
            else:
                msg = (
                    f"Unexpected advantage shape {tuple(adv.shape)} for the "
                    f"Liger token-level loss; expected (batch={batch},) "
                    f"or (batch, n_act={n_act}) — got "
                    f"{tuple(adv.shape)}. Per-token shape comes from "
                    "advantage_granularity='turn'; trajectory shape from "
                    "advantage_granularity='trajectory'."
                )
                raise ValueError(msg)
            # Token-flatten the 5 layout-dependent tensors to ``(B*T, 1, ...)``.
            policy_arg = policy_hidden[:, :n_act, :].reshape(n_tokens, 1, hidden_dim)
            target_ids_arg = target_ids.reshape(n_tokens, 1)
            mask_arg = mask.reshape(n_tokens, 1)
            old_lp_arg = old_log_probs.reshape(n_tokens, 1)
            ref_lp_arg = (
                ref_log_probs.reshape(n_tokens, 1)
                if ref_log_probs is not None
                else None
            )
            if sampling_log_probs is not None:
                with torch.no_grad():
                    log_diff = fill_outside_mask(
                        old_log_probs - sampling_log_probs.to(self.device),
                        mask,
                    )
                    vllm_is_ratio_arg = (
                        torch.exp(log_diff)
                        .clamp(max=self.vllm_importance_sampling_cap)
                        .reshape(n_tokens, 1)
                    )
            chunk_size = self._resolve_fused_chunk_rows(
                getattr(lm_head_weight, "ds_shape", lm_head_weight.shape)[0],
                self.chunk_rows,
            )
        else:
            # Trajectory-level (GSPO): keep the padded layout and one-sequence-per-
            # chunk granularity (chunk_size=1 over the batch dim).
            policy_arg = policy_hidden
            target_ids_arg = target_ids
            mask_arg = mask
            old_lp_arg = old_log_probs
            ref_lp_arg = ref_log_probs
            adv_arg = adv
            chunk_size = 1

        kernel_args: tuple[Any, ...] = (
            policy_arg,
            lm_head_weight,
            target_ids_arg,
            mask_arg,
            adv_arg,
            lm_head_bias,
            ref_lp_arg,
            old_lp_arg,
            None,
            None,
            None,
            self.beta,
            epsilon_low,
            epsilon_high,
            liger_loss_type,
            self.max_output_tokens,
            importance_sampling_level,
            None,
            None,
            self.temperature,
            None,
            ref_log_probs is not None,  # use_ref_model
            chunk_size,
            vllm_is_ratio_arg,
        )
        if window is not None:
            # The kernel divides the count it is given by the world size, so the
            # rank-local window reaches the reduction as its own normalizer.
            kernel_args = _liger_args_with_normalizer(
                kernel_args,
                float(window[1] * _liger_normalizer_world_size()),
            )

        with self._liger_head_gather():
            loss, aux = LigerFusedLinearGRPOFunction.apply(*kernel_args)

        kl = aux[0]
        loss = loss.mean()
        if window is not None:
            loss = loss * window[0]
        return loss, kl

    # Backward-compatible alias kept for any external callers.
    _grpo_loss_liger = _liger_loss
