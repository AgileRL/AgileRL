# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import warnings
from collections.abc import Sequence
from dataclasses import replace
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import torch

from agilerl import HAS_LIGER_KERNEL, HAS_LLM_DEPENDENCIES
from agilerl.algorithms.core import ActionResult, LLMAlgorithm
from agilerl.algorithms.core.optimizer_wrapper import REPLICATED_GROUP_SUFFIX
from agilerl.algorithms.core.registry import HyperparameterConfig, NetworkGroup
from agilerl.lora.fused import set_fused_adapter_routing, unset_fused_adapter_routing

if TYPE_CHECKING:
    from peft import LoraConfig, PeftModel
    from transformers import PreTrainedModel

if HAS_LIGER_KERNEL or TYPE_CHECKING:
    from agilerl.algorithms.core.llm_ops.fused_loss import (
        LigerFusedLinearPolicyLossFunction,
        apply_fused_policy_loss,
    )
else:
    # Keep the name resolvable when liger-kernel isn't installed so unit
    # tests can patch it. ``_ppo_policy_loss_liger_from_hidden`` guards against
    # actual use.
    LigerFusedLinearPolicyLossFunction = None  # type: ignore[assignment]
    apply_fused_policy_loss = None  # type: ignore[assignment]
from agilerl.components.llm_rollout_data import EpisodeSegments
from agilerl.distributed import (
    FSDPConfig,
    aggregate_metrics_dict,
    resolve_device,
)
from agilerl.distributed.runtime import OptimizerStep
from agilerl.protocols import (
    PeftModelProtocol,
    PreTrainedModelProtocol,
)
from agilerl.typing import LLMObsType, LLMRolloutExperiences
from agilerl.utils.algo_utils import (
    CosineLRScheduleConfig,
    VLLMConfig,
    get_experiences_samples,
)
from agilerl.utils.llm_packing import unpack_hidden_states, unpack_values
from agilerl.utils.llm_utils import (
    LLM_RL_COMMON_METRIC_NAMES,
    PPO_METRIC_NAMES,
    VLLM_IS_METRIC_NAMES,
    BitsAndBytesConfig,
    calculate_k3_kl,
    clipped_is_surrogate,
    masked_mean,
    masked_whiten,
    normalize_prompt_batch,
    pool_by_turns,
    resolve_batch_advantage_granularity,
    validate_importance_sampling_level,
    value_fit,
)
from agilerl.utils.segment_rows import filler_stand_in
from agilerl.utils.vision_rows import VisionRows

if HAS_LLM_DEPENDENCIES:
    from transformers import GenerationConfig

logger = logging.getLogger(__name__)


class PPO(LLMAlgorithm[LLMRolloutExperiences]):
    """Turn-level PPO for LLM finetuning with actor/reference adapters.

    Each generation sequence (turn) is treated as a single RL action.
    GAE discounts between turns, not between tokens within a turn.
    Single-turn is the special case where all action tokens share turn 0.

    :param pad_token_id: Token id used for sequence padding.
    :type pad_token_id: int
    :param pad_token: Padding token string.
    :type pad_token: str
    :param model_name: HF model name or local path used when building internally.
    :type model_name: str | None, optional
    :param actor_network: Pre-built actor model. If omitted, ``model_name`` is used.
    :type actor_network: PreTrainedModel | PeftModel | None, optional
    :param model_config: Extra kwargs passed when constructing a model from ``model_name``.
    :type model_config: dict[str, Any] | None, optional
    :param hp_config: Hyperparameter mutation configuration.
    :type hp_config: HyperparameterConfig | None, optional
    :param index: Population index used by evolutionary workflows.
    :type index: int, optional
    :param batch_size: Batch size used for PPO updates.
    :type batch_size: int, optional
    :param beta: KL penalty coefficient against the reference policy.
    :type beta: float, optional
    :param vf_coef: Value loss coefficient.
    :type vf_coef: float, optional
    :param clip_coef: PPO clipping coefficient.
    :type clip_coef: float, optional
    :param gamma: Discount factor across turns.
    :type gamma: float, optional
    :param gae_lambda: GAE lambda used for turn-level advantage estimation.
    :type gae_lambda: float, optional
    :param critic_warmup_steps: First learn steps that update only the critic, on
        Monte Carlo returns (GAE ``lambda = 1``), leaving the policy fixed.
        ``critic_warmup_steps_done`` counts them and is checkpointed.
    :type critic_warmup_steps: int, optional
    :param lr_actor: Actor learning rate.
    :type lr_actor: float, optional
    :param lr_critic: Critic/value-head learning rate. If ``None``, ``lr_actor`` is used.
    :type lr_critic: float | None, optional
    :param max_grad_norm: Gradient clipping norm.
    :type max_grad_norm: float, optional
    :param share_grad_clip: Clip actor and critic grads by their combined norm.
    :type share_grad_clip: bool, optional
    :param update_epochs: Number of PPO epochs per update.
    :type update_epochs: int, optional
    :param temperature: Sampling temperature for generation.
    :type temperature: float, optional
    :param repetition_penalty: Repetition penalty used during generation.
    :type repetition_penalty: float, optional
    :param top_p: Nucleus sampling threshold.
    :type top_p: float, optional
    :param top_k: Top-k sampling threshold.
    :type top_k: int, optional
    :param min_p: Minimum probability cutoff for sampling.
    :type min_p: float, optional
    :param use_separate_reference_adapter: Whether to keep a separate reference adapter.
    :type use_separate_reference_adapter: bool, optional
    :param calc_position_embeddings: Whether to compute position embeddings.
    :type calc_position_embeddings: bool, optional
    :param micro_batch_size_per_gpu: Optional target micro-batch size per GPU.
    :type micro_batch_size_per_gpu: int | None, optional
    :param mini_batch_size: Per-rank trajectories covered by one optimizer
        step. ``None`` uses ``batch_size / world_size``.
        ``gradient_accumulation_steps`` is derived as
        ``mini_batch_size / micro_batch_size_per_gpu``.
    :type mini_batch_size: int | None, optional
    :param max_output_tokens: Maximum newly generated tokens per completion.
    :type max_output_tokens: int | None, optional
    :param min_output_tokens: Minimum newly generated tokens per completion.
    :type min_output_tokens: int | None, optional
    :param max_model_len: Maximum model context length.
    :type max_model_len: int, optional
    :param hf_generate_chunk_size: Number of prompts per HuggingFace generation chunk.
        Ignored when colocated.
    :type hf_generate_chunk_size: int | None, optional
    :param lora_config: LoRA configuration.
    :type lora_config: LoraConfig | None, optional
    :param cosine_lr_schedule_config: Warmup-cosine schedule stepped once per
        ``learn`` call; the actor's starts after ``critic_warmup_steps``.
    :type cosine_lr_schedule_config: CosineLRScheduleConfig | None, optional
    :param fsdp_config: FSDP2 sharding settings for distributed runs, defaults to None
    :type fsdp_config: FSDPConfig | None, optional
    :param device: Device for accelerated computing, 'cpu' or 'cuda', defaults to 'cpu'
    :type device: str, optional
    :param wrap: Whether to wrap models for distributed execution.
    :type wrap: bool, optional
    :param clone: Whether this instance is being created as a clone.
    :type clone: bool, optional
    :param offload_trainer_during_rollout: For colocated vLLM, offload the trainer's
        own base to CPU during rollout (and bring it back for the training step)
        so the rollout engine and the trainer never both hold a base on the GPU.
        Defaults to True; inert without colocated vLLM, and disabled under
        FSDP2 sharding.
    :type offload_trainer_during_rollout: bool, optional
    :param vllm_config: vLLM runtime configuration.
    :type vllm_config: VLLMConfig | None, optional
    :param seed: Random seed.
    :type seed: int, optional
    :param turn_level_clip: Legacy gate for per-turn ratio clipping, honored
        only when ``importance_sampling_level="auto"``. Superseded by
        ``importance_sampling_level``.
    :type turn_level_clip: bool, optional
    :param importance_sampling_level: IS / ratio-pooling level for the policy
        surrogate, orthogonal to ``advantage_granularity``. ``"token"`` clips per
        token; ``"turn"`` pools the ratio per turn;
        ``"trajectory"`` pools over the whole completion; the paired advantage is
        pooled to the same bucket. ``"auto"`` (default) uses the GAE granularity
        when ``turn_level_clip`` is set, else token. Turn/trajectory pooling
        couples a unit's tokens and cannot be token-chunked in the fused kernel,
        so set ``use_liger_loss=False`` there (the standard path is always
        memory-bounded).
    :type importance_sampling_level: Literal["auto", "token", "turn", "trajectory"], optional
    :param advantage_granularity: PPO action granularity. ``"turn"`` enforces
        turn-level updates, ``"token"`` enforces token-level updates, and
        ``"auto"`` uses token-level only when all samples are single-turn.
    :type advantage_granularity: Literal["turn", "token", "auto"], optional
    :param turn_ratio_pooling: Reduction used to pool per-token log-ratios into
        a per-turn ratio when the importance-sampling level is ``"turn"`` (the
        default ``"auto"`` resolves to turn for multi-turn batches); ignored at
        token/trajectory level. ``"sum"`` (default) yields the product ratio per
        turn — the standard, paper-aligned per-turn importance weight. ``"mean"``
        yields a length-normalized geometric-mean ratio (GSPO-style); reach for it
        on long or highly variable-length turns, where the product ratio is far
        outside the clip band on every turn and saturates the clipped surrogate —
        length-normalizing keeps the per-turn ratio in range so the surrogate stays
        informative.
    :type turn_ratio_pooling: Literal["sum", "mean"], optional
    :param turn_value_reduction: Aggregation used to map token critic values to
        turn values. ``"mean"`` reproduces existing behavior, ``"final_value"``
        uses the final action token value in each turn.
    :type turn_value_reduction: str, optional
    :param whiten_advantages: Whether to whiten computed advantages before PPO
        optimization.
    :type whiten_advantages: bool, optional
    :param gradient_checkpointing: Enable gradient checkpointing.
    :type gradient_checkpointing: bool, optional
    :param torch_compiler: Optional torch compile mode.
    :type torch_compiler: str | None, optional
    :param cast_logprobs_to_fp32: When ``True`` (default), run the per-token
        log-prob reduction (``gather`` / ``logsumexp``) in fp32 before casting
        back to the input dtype, for numerically stable log-probs. ``False`` runs
        it in the input dtype, saving a little memory at the cost of a per-token
        bf16 quantisation error that can bias importance-sampling ratios.
    :type cast_logprobs_to_fp32: bool, optional
    :param chunk_rows: Primary chunk-size setting for fused logit tiles. Applies
        to both standard and Liger paths.
    :type chunk_rows: int | None, optional
    :param use_liger_loss: Use the Liger fused policy loss, defaults to ``True``
        (requires ``liger-kernel``). **Recommended for PPO**: via AgileRL's
        ``LigerFusedLinearPolicyLossFunction`` (not the upstream Liger GRPO
        kernel), it is roughly memory-neutral with a mild speedup that grows with
        sequence length (~1.1x at long sequences) at token-level IS. Separate
        from the Liger *model* patches (fused RMSNorm/RoPE/SwiGLU), which apply
        whenever ``liger-kernel`` is installed.
    :type use_liger_loss: bool, optional
    :param fuse_actor_critic_pass: ``True`` runs the actor and critic rows of a
        micro-batch in one forward and backward; ``False`` runs them as two
        passes, so only one pass's activations are live. ``None`` (default)
        fuses when the memory estimate of the fused pass fits this GPU's total
        memory, and always fuses off CUDA.
    :type fuse_actor_critic_pass: bool | None, optional
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
    :param moe_lora_recompute: Recompute routed-expert LoRA activations in
        backward on frozen packed base weights. ``None`` (default) recomputes
        only outside activation-checkpointed blocks.
    :type moe_lora_recompute: bool | None, optional
    :param vllm_importance_sampling_correction: When ``True`` (default) and
        colocated, correct the rollout/trainer log-prob mismatch by
        weighting each training token by ``clamp(exp(trainer - sampling),
        max=vllm_importance_sampling_cap)``. Active only for training rollouts;
        inert on the HuggingFace path and at eval.
    :type vllm_importance_sampling_correction: bool, optional
    :param vllm_importance_sampling_cap: Upper clamp on the vLLM
        importance-sampling ratio (default ``2.0``), bounding the correction
        weight to limit variance from outlier tokens. Must be > 0.
    :type vllm_importance_sampling_cap: float, optional
    :param vllm_max_logprob_gap: Log a warning when a learn step's mean
        ``|trainer - vLLM|`` per-token log-prob gap exceeds this, in nats
        (default ``0.1``). bf16 engines usually sit near 0.01-0.03.
    :type vllm_max_logprob_gap: float, optional
    :param vllm_max_clip_fraction: Log a warning when the fraction of action
        tokens whose trainer/vLLM ratio reaches ``vllm_importance_sampling_cap``
        exceeds this (default ``0.02``). bf16 engines usually stay under 0.01.
    :type vllm_max_clip_fraction: float, optional
    :param use_sequence_packing: Opt in to padding-free sequence packing for the
        gradient forward. Only honoured under a FlashAttention-2 / FlexAttention
        backend, otherwise inert; the fused actor+critic pass packs only under
        FlexAttention, and the no-grad reference/old-value pass stays padded.
    :type use_sequence_packing: bool, optional
    :param lora_target_scope: Optional PEFT LoRA path scope for multimodal models
        (e.g. ``"language_model"``). Passed to
        :func:`adapt_lora_config_for_model`.
    :type lora_target_scope: str | None, optional
    """

    _mini_batch_size_default = "micro_batch"

    def __init__(
        self,
        pad_token_id: int,
        pad_token: str,
        model_name: str | None = None,
        actor_network: PreTrainedModel | PeftModel | None = None,
        model_config: dict[str, Any] | None = None,
        hp_config: HyperparameterConfig | None = None,
        index: int = 0,
        batch_size: int = 16,
        beta: float = 0.001,
        vf_coef: float = 0.5,
        clip_coef: float = 0.2,
        gamma: float = 1.0,
        gae_lambda: float = 0.95,
        critic_warmup_steps: int = 0,
        lr_actor: float = 5e-7,
        lr_critic: float | None = 5e-5,
        max_grad_norm: float = 1.0,
        share_grad_clip: bool = False,
        update_epochs: int = 1,
        temperature: float = 1.0,
        repetition_penalty: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 50,
        min_p: float = 0.0,
        use_separate_reference_adapter: bool = True,
        calc_position_embeddings: bool = True,
        micro_batch_size_per_gpu: int | None = None,
        mini_batch_size: int | None = None,
        max_output_tokens: int | None = None,
        min_output_tokens: int | None = None,
        max_model_len: int = 1024,
        hf_generate_chunk_size: int | None = None,
        lora_config: LoraConfig | None = None,
        cosine_lr_schedule_config: CosineLRScheduleConfig | None = None,
        fsdp_config: FSDPConfig | None = None,
        device: str | torch.device | None = None,
        wrap: bool = True,
        clone: bool = False,
        offload_trainer_during_rollout: bool = True,
        vllm_config: VLLMConfig | None = None,
        seed: int = 42,
        turn_level_clip: bool = True,
        importance_sampling_level: Literal[
            "auto", "token", "turn", "trajectory"
        ] = "auto",
        advantage_granularity: Literal["turn", "token", "auto"] = "auto",
        turn_ratio_pooling: Literal["sum", "mean"] = "sum",
        turn_value_reduction: Literal["mean", "final_value"] = "final_value",
        whiten_advantages: bool = True,
        gradient_checkpointing: bool = True,
        torch_compiler: str | None = None,
        cast_logprobs_to_fp32: bool = True,
        chunk_rows: int | None = None,
        use_liger_loss: bool = True,
        fuse_actor_critic_pass: bool | None = None,
        quantization_config: BitsAndBytesConfig | None = None,
        activation_offload: bool = False,
        moe_lora_recompute: bool | None = None,
        use_sequence_packing: bool = False,
        lora_target_scope: str | None = None,
        vllm_importance_sampling_correction: bool = True,
        vllm_importance_sampling_cap: float = 2.0,
        vllm_max_logprob_gap: float = 0.1,
        vllm_max_clip_fraction: float = 0.02,
    ) -> None:

        resolved_device = resolve_device(device)
        if cosine_lr_schedule_config is not None:
            cosine_lr_schedule_config = replace(
                cosine_lr_schedule_config, actor_start_step=critic_warmup_steps
            )
        super().__init__(
            index=index,
            batch_size=batch_size,
            lr=lr_actor,
            lr_critic=lr_critic,
            max_grad_norm=max_grad_norm,
            clone=clone,
            calc_position_embeddings=calc_position_embeddings,
            seed=seed,
            pad_token_id=pad_token_id,
            pad_token=pad_token,
            use_value_head=True,
            vllm_config=vllm_config,
            use_liger_loss=use_liger_loss,
            lora_config=lora_config,
            use_separate_reference_adapter=use_separate_reference_adapter,
            model_name=model_name,
            actor_network=actor_network,
            model_config=model_config,
            micro_batch_size_per_gpu=micro_batch_size_per_gpu,
            mini_batch_size=mini_batch_size,
            cosine_lr_schedule_config=cosine_lr_schedule_config,
            hp_config=hp_config,
            offload_trainer_during_rollout=offload_trainer_during_rollout,
            wrap=wrap,
            device=resolved_device,
            fsdp_config=fsdp_config,
            name="LLMPPO",
            gradient_checkpointing=gradient_checkpointing,
            torch_compiler=torch_compiler,
            cast_logprobs_to_fp32=cast_logprobs_to_fp32,
            chunk_rows=chunk_rows,
            quantization_config=quantization_config,
            activation_offload=activation_offload,
            moe_lora_recompute=moe_lora_recompute,
            use_sequence_packing=use_sequence_packing,
            lora_target_scope=lora_target_scope,
            vllm_importance_sampling_correction=vllm_importance_sampling_correction,
            vllm_importance_sampling_cap=vllm_importance_sampling_cap,
            vllm_max_logprob_gap=vllm_max_logprob_gap,
            vllm_max_clip_fraction=vllm_max_clip_fraction,
        )
        self._validate_core_args(
            batch_size, lr_actor, clip_coef, update_epochs, actor_network, clone
        )
        self.beta = beta
        self.vf_coef = vf_coef
        self.clip_coef = clip_coef
        # Expose lr_actor explicitly (base stores it as ``self.lr``): the split
        # LLM optimizer's lr_name is ``("lr_actor", "lr_critic")``, and the
        # clone/checkpoint init_dict captures attributes by constructor-param
        # name — both look up ``self.lr_actor``.
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic if lr_critic is not None else lr_actor
        self.update_epochs = update_epochs
        self.critic_warmup_steps = critic_warmup_steps
        self.critic_warmup_steps_done = 0
        self.share_grad_clip = share_grad_clip
        self.temperature = temperature
        self.repetition_penalty = repetition_penalty
        self.top_p = top_p
        self.top_k = top_k
        self.min_p = min_p
        self._setup_advantage_options(
            turn_level_clip,
            advantage_granularity,
            turn_value_reduction,
            whiten_advantages,
            gamma,
            gae_lambda,
        )
        self._setup_objective(importance_sampling_level, turn_ratio_pooling)
        self._setup_generation(
            max_output_tokens, min_output_tokens, max_model_len, hf_generate_chunk_size
        )
        self._setup_actors(actor_network, clone=clone)
        self.fuse_actor_critic_pass = fuse_actor_critic_pass
        self._fuses_actor_critic_pass = self._resolve_fuse_actor_critic_pass()

        # Register network groups for mutations
        self.register_network_group(NetworkGroup(eval_network=self.actor, policy=True))
        if not clone:
            self.wrap_models()

        # Register algorithm metrics
        for m in (
            *LLM_RL_COMMON_METRIC_NAMES,
            *PPO_METRIC_NAMES,
            *VLLM_IS_METRIC_NAMES,
        ):
            self.metrics.register(m)

    def get_action(
        self,
        obs: LLMObsType,
        training: bool = True,
        **kwargs: Any,
    ) -> ActionResult:
        """Generate completion tokens for each prompt in the batch.

        :param obs: A single prompt dict or a list of HF-style prompt dicts.
        :type obs: LLMObsType
        :param training: If ``False``, use near-deterministic decoding where applicable.
        :type training: bool
        :param kwargs: Additional keyword arguments accepted for base-class compatibility.
        :type kwargs: Any
        :return: An :class:`ActionResult` of per-prompt completion token IDs and
            masks. When the vLLM sampling-mismatch correction is enabled
            (training rollouts on the vLLM path), ``sampling_logps`` carries
            the captured per-row sampling logprobs; otherwise it is ``None``.
        :rtype: ActionResult
        """
        prompts = normalize_prompt_batch(obs)
        # Capture vLLM sampling logprobs only for training rollouts when the
        # mismatch correction is enabled; ``None`` on the HF path / eval.
        sampling_logps: list[torch.Tensor | None] | None = None
        capture_sampling_logps = (
            training and self.colocated and self.vllm_importance_sampling_correction
        )

        with self.select_adapter("actor"):
            self.actor.eval()
            if not self.colocated:
                token_ids_list, completion_masks = self._generate_with_hf(prompts)
            else:
                self._prepare_vllm_for_generation()
                (
                    token_ids_list,
                    completion_masks,
                    sampling_logps,
                ) = self._generate_with_vllm_colocate(
                    # RolloutPrompt is a TypedDict, i.e. a plain dict at
                    # runtime; the base helper takes untyped prompt dicts.
                    prompts,
                    1,
                    temperature=self.temperature
                    if training
                    else 0.01,  # Almost deterministic for evaluation
                    capture_sampling_logps=capture_sampling_logps,
                )

        return ActionResult(token_ids_list, completion_masks, sampling_logps)

    def learn(
        self,
        experiences: LLMRolloutExperiences,
        turn_ids: torch.Tensor | None = None,
        sampling_logps: list[torch.Tensor | None] | None = None,
        episode_segments: list[EpisodeSegments | None] | None = None,
        pixel_values: torch.Tensor | None = None,
        pixel_image_counts: Sequence[int] | None = None,
        image_token_id: int | None = None,
    ) -> dict[str, float]:
        """Update actor and critic adapters; a critic warmup step updates only the critic.

        :param experiences: ``(token_ids, action_masks, rewards)``. For
            single-turn, ``rewards`` is a flat tensor of scalars; for multi-turn,
            shape ``[batch, max_turns]`` per-turn rewards.
        :type experiences: LLMRolloutExperiences
        :param turn_ids: Optional ``[batch, seq_len - 1]`` tensor of turn indices;
            ``-1`` for non-action tokens. If ``None``, all action tokens are turn ``0``.
        :type turn_ids: torch.Tensor | None
        :param sampling_logps: Optional per-row flat vLLM sampling logprobs (one
            1-D tensor per trajectory, generated tokens only; concatenated across
            turns for multi-turn) for the vLLM sampling-mismatch correction.
            Parallel to the stacked ``token_ids`` rows. ``None`` disables
            the correction for this update.
        :type sampling_logps: list[torch.Tensor | None] | None
        :param episode_segments: Optional segment layout per trajectory,
            parallel to the stacked ``token_ids`` rows (``None`` for an
            unsegmented trajectory). Each segment runs its forwards as its own
            row; values, GAE and returns span the whole episode. The update
            keeps the optimizer steps of the unsegmented batch: each step
            accumulates a window of segment rows, padded so every rank runs the
            same micro-batches.
        :type episode_segments: list[EpisodeSegments | None] | None
        :param pixel_values: Optional vision rows of every trajectory, stacked
            in ``token_ids`` row order. The reference, actor and critic
            forwards each see the vision rows of the rows they run.
        :type pixel_values: torch.Tensor | None
        :param pixel_image_counts: Optional vision rows per trajectory, needed
            when trajectories hold different numbers of images.
        :type pixel_image_counts: Sequence[int] | None
        :param image_token_id: Token id the VL forward scatters one image
            feature row into. Required with ``pixel_values`` and
            ``episode_segments``, to cut the filler rows down to one image.
        :type image_token_id: int | None
        :return: Mean training metrics across PPO minibatch updates, actor and
            critic gradient norms averaged over optimizer steps, the pre-update
            value fit, ``critic_warmup``, row padding stats and
            ``learn_phase_<phase>_s`` wall seconds.
        :rtype: dict[str, float]
        """
        phase_timer = self._start_learn_phases()
        self._prepare_vllm_for_training()
        critic_warmup = self.critic_warmup_steps_done < self.critic_warmup_steps
        passes = 1 if self._fuses_actor_critic_pass or critic_warmup else 2
        with self.trainer_offload_context():
            token_ids, action_masks, turn_ids, rewards_2d = self._stack_rollout_batch(
                experiences, turn_ids
            )
            action_mask_bool = action_masks.bool()
            num_samples = token_ids.shape[0]
            ppo_granularity = self._resolve_advantage_granularity(turn_ids)

            batch_size = min(num_samples, self.micro_batch_size_per_gpu)
            updates = 0
            learn_metrics = dict.fromkeys(
                ("loss", "pg_loss", "vf_loss", "kl", "entropy", "clipfrac"), 0.0
            )
            grad_norm_totals = {
                "grad_norm_pre": 0.0,
                "grad_norm_post": 0.0,
                "critic_grad_norm_pre": 0.0,
                "critic_grad_norm_post": 0.0,
            }
            grad_updates = 0
            has_segments = self._has_episode_segments(episode_segments, num_samples)
            rows = None
            accumulation_steps = None
            if has_segments:
                self._check_segments_supported(self._resolve_is_level(ppo_granularity))
                rows, accumulation_steps = self._segment_rows(
                    token_ids,
                    action_masks,
                    episode_segments,
                    np.arange(num_samples),
                    pixel_values=pixel_values,
                    pixel_image_counts=pixel_image_counts,
                    image_token_id=image_token_id,
                )
                pixel_values = rows.pixel_values
                pixel_image_counts = rows.pixel_image_counts
                batch_size = self.micro_batch_size_per_gpu
            phase_timer.mark("prepare")
            reference_log_probs, old_log_probs, old_values = (
                self._fused_forward_no_grad(
                    token_ids if rows is None else rows.token_ids,
                    batch_size=batch_size,
                    pixel_values=pixel_values,
                    pixel_image_counts=pixel_image_counts,
                )
            )
            phase_timer.mark("no_grad_forward")
            # PPO always trains with a value head, so critic values are present.
            assert old_values is not None
            if rows is not None:
                # GAE and returns run over whole episodes.
                frame = (num_samples, int(action_masks.shape[1]))
                reference_log_probs = rows.merge_frame(reference_log_probs, *frame, 1.0)
                old_log_probs = rows.merge_frame(old_log_probs, *frame, 1.0)
                old_values = rows.merge_frame(old_values, *frame, 0.0)
            old_values = torch.masked_fill(old_values, ~action_mask_bool, 0.0)

            token_rewards = self._compute_token_rewards(
                action_masks, rewards_2d, turn_ids
            )

            old_log_probs = torch.masked_fill(old_log_probs, ~action_mask_bool, 1.0)
            reference_log_probs = torch.masked_fill(
                reference_log_probs, ~action_mask_bool, 1.0
            )
            # Warmup fits the critic to Monte Carlo returns of the fixed policy.
            gae_lambda = 1.0 if critic_warmup else self.gae_lambda
            if ppo_granularity == "token":
                returns, advantages = self._compute_gae_returns_token(
                    token_rewards, old_values, action_masks, gae_lambda
                )
            else:
                returns, advantages = self._compute_gae_returns(
                    token_rewards, old_values, action_masks, turn_ids, gae_lambda
                )
            del token_rewards
            # Pre-update values against the value targets, on the GAE axis.
            explained_variance, value_return_corr = value_fit(
                old_values,
                returns,
                action_mask_bool,
                turn_ids if ppo_granularity == "turn" else None,
                self.turn_value_reduction,
            )

            # The reweight applies only to the policy surrogate.
            sampling_log_probs, is_metrics = (
                self._aligned_sampling_logprobs_and_metrics(
                    sampling_logps, action_masks, old_log_probs
                )
            )

            row_episodes = np.arange(num_samples)
            if rows is not None:
                token_ids = rows.token_ids
                action_masks = rows.action_masks
                turn_ids = rows.split_frame(turn_ids, -1)
                old_log_probs = rows.split_frame(old_log_probs, 1.0)
                reference_log_probs = rows.split_frame(reference_log_probs, 1.0)
                returns = rows.split_frame(returns, 0.0)
                advantages = rows.split_frame(advantages, 0.0)
                old_values = rows.split_frame(old_values, 0.0)
                if sampling_log_probs is not None:
                    sampling_log_probs = rows.split_frame(sampling_log_probs, 0.0)
                row_episodes = rows.row_episodes
                num_samples = int(token_ids.shape[0])
            batch_idxs = np.arange(num_samples)
            filler_rows = row_episodes < 0
            padding_stats = self._row_padding_stats(token_ids, row_episodes, batch_idxs)
            vision_rows = None
            if pixel_values is not None:
                vision_rows = VisionRows(pixel_values, pixel_image_counts)
            phase_timer.mark("prepare")

            self.actor.train()
            for _epoch_idx in range(self.update_epochs):
                self.rng.shuffle(batch_idxs)
                loss_scales = None
                if accumulation_steps is not None:
                    loss_scales = self._segment_loss_scales(
                        np.array(
                            [
                                filler_rows[
                                    batch_idxs[start : start + batch_size]
                                ].all()
                                for start in range(0, num_samples, batch_size)
                            ]
                        ),
                        accumulation_steps,
                    )
                for start in range(0, num_samples, batch_size):
                    minibatch_idxs = batch_idxs[
                        start : min((start + batch_size), num_samples)
                    ]
                    phase_timer.mark("other")
                    loss_scale = (
                        1.0
                        if loss_scales is None
                        else float(loss_scales[start // batch_size])
                    )
                    # ``get_experiences_samples`` indexes each input
                    # positionally: Tensor in -> Tensor out, so the tuple
                    # mirrors the all-Tensor inputs.
                    (
                        batch_ids,
                        batch_action_mask,
                        batch_old_log_probs,
                        batch_reference_log_probs,
                        batch_returns,
                        batch_advantages,
                        batch_old_values,
                        batch_turn_ids,
                    ) = get_experiences_samples(
                        minibatch_idxs,
                        token_ids,
                        action_masks,
                        old_log_probs,
                        reference_log_probs,
                        returns,
                        advantages,
                        old_values,
                        turn_ids,
                    )
                    is_filler = bool(filler_rows[minibatch_idxs].all())
                    if is_filler:
                        batch_action_mask, batch_turn_ids = filler_stand_in(
                            batch_action_mask, batch_turn_ids
                        )

                    batch_pixel_values = (
                        vision_rows.for_minibatch(minibatch_idxs, num_samples).to(
                            self.device
                        )
                        if vision_rows is not None
                        else None
                    )

                    batch_sampling_log_probs = (
                        sampling_log_probs[minibatch_idxs]
                        if sampling_log_probs is not None
                        else None
                    )
                    # The correction is fused into the Liger kernel at token-level
                    # IS (via vllm_is_ratio); turn/trajectory pooling can't express
                    # the per-token reweight, so those fall back to the standard
                    # path (warn once, like GRPO).
                    liger_corr_fallback = (
                        batch_sampling_log_probs is not None
                        and self._resolve_is_level(ppo_granularity) != "token"
                    )
                    if (
                        self.use_liger_loss
                        and liger_corr_fallback
                        and not self._is_correction_liger_warned
                    ):
                        warnings.warn(
                            "use_liger_loss=True fuses the vLLM sampling-mismatch "
                            "correction only at token-level importance sampling; "
                            "turn/trajectory pooling uses the standard PyTorch path.",
                            stacklevel=2,
                        )
                        self._is_correction_liger_warned = True

                    use_liger = self.use_liger_loss and not liger_corr_fallback
                    policy_inputs = (
                        batch_action_mask,
                        batch_old_log_probs,
                        batch_reference_log_probs,
                        batch_advantages,
                        batch_turn_ids,
                        ppo_granularity,
                        batch_sampling_log_probs,
                    )
                    value_inputs = (
                        batch_old_values,
                        batch_returns,
                        batch_action_mask,
                        batch_turn_ids,
                        ppo_granularity,
                    )
                    if self._fuses_actor_critic_pass and not critic_warmup:
                        actor_hidden, values = self._actor_critic_hidden_states(
                            batch_ids, batch_pixel_values
                        )
                        if use_liger:
                            policy_loss, metrics = (
                                self._ppo_policy_loss_liger_from_hidden(
                                    actor_hidden, batch_ids, *policy_inputs
                                )
                            )
                        else:
                            policy_loss, metrics = self._ppo_policy_loss(
                                self._actor_log_probs_from_hidden(
                                    actor_hidden, batch_ids
                                ),
                                *policy_inputs,
                            )
                        vf_loss = self._ppo_value_loss(values, *value_inputs)
                        total_loss = policy_loss + vf_loss
                        self._raise_if_loss_not_finite_on_any_rank(total_loss)
                        phase_timer.mark("forward")
                        steps = [
                            self._backward_ppo_pass(
                                total_loss * loss_scale, accumulation_steps, passes
                            )
                        ]
                        unset_fused_adapter_routing(self.actor)
                    else:
                        policy_step = None
                        if critic_warmup:
                            policy_loss = torch.zeros(())
                            metrics = dict.fromkeys(learn_metrics, 0.0)
                        elif use_liger:
                            policy_loss, metrics = self._ppo_policy_loss_liger(
                                batch_ids, *policy_inputs, batch_pixel_values
                            )
                        else:
                            policy_loss, metrics = self._ppo_policy_loss(
                                self._fused_forward(
                                    batch_ids, pixel_values=batch_pixel_values
                                ),
                                *policy_inputs,
                            )
                        if not critic_warmup:
                            self._raise_if_loss_not_finite_on_any_rank(policy_loss)
                            phase_timer.mark("forward")
                            policy_step = self._backward_ppo_pass(
                                policy_loss * loss_scale, accumulation_steps, passes
                            )
                            unset_fused_adapter_routing(self.actor)

                        vf_loss = self._ppo_value_loss(
                            self._critic_values(batch_ids, batch_pixel_values),
                            *value_inputs,
                        )
                        self._raise_if_loss_not_finite_on_any_rank(vf_loss)
                        phase_timer.mark("forward")
                        value_step = self._backward_ppo_pass(
                            vf_loss * loss_scale, accumulation_steps, passes
                        )
                        unset_fused_adapter_routing(self.actor)
                        steps = [policy_step, value_step]
                    for step in steps:
                        if step is not None:
                            for key, norm in self._actor_critic_grad_norms(
                                step
                            ).items():
                                grad_norm_totals[key] += norm
                            grad_updates += 1
                    if is_filler:
                        continue

                    vf_loss_value = vf_loss.item()
                    for key in ("kl", "entropy", "clipfrac", "pg_loss"):
                        learn_metrics[key] += metrics[key]
                    learn_metrics["vf_loss"] += vf_loss_value
                    learn_metrics["loss"] += policy_loss.item() + vf_loss_value
                    updates += 1
        self.critic_warmup_steps_done += int(critic_warmup)
        averaged = {
            metric: value / max(updates, 1) for metric, value in learn_metrics.items()
        }
        grad_norms = (
            {key: total / grad_updates for key, total in grad_norm_totals.items()}
            if grad_updates > 0
            else {}
        )
        self._step_lr_scheduler()
        # Sampling-mismatch metrics and the value fit are computed once over the
        # full batch, so they bypass the per-update averaging above.
        result = {
            **averaged,
            **grad_norms,
            **is_metrics,
            "explained_variance": explained_variance,
            "value_return_corr": value_return_corr,
            "critic_warmup": float(critic_warmup),
            **padding_stats,
            **self._learn_phase_seconds(),
        }

        # Wire averaged metrics into the metrics tracker.
        token_ids_list = experiences[0]
        completion_length = np.mean([c.shape[-1] for c in token_ids_list])
        agg = aggregate_metrics_dict({**result, "completion_length": completion_length})
        agg["completion_length"] = int(agg["completion_length"])
        for key, value in agg.items():
            self.metrics.log(key, value)

        return result

    def _validate_core_args(
        self,
        batch_size: int,
        lr: float,
        clip_coef: float,
        update_epochs: int,
        actor_network: PreTrainedModel | PeftModel | None,
        clone: bool,
    ) -> None:
        """Validate the core training arguments."""
        assert isinstance(batch_size, int), "Batch size must be an integer."
        assert batch_size >= 1, "Batch size must be greater than or equal to one."
        assert isinstance(lr, float), "Actor learning rate must be a float."
        assert lr > 0, "Actor learning rate must be greater than zero."
        assert isinstance(clip_coef, (float, int)), "Clip coefficient must be a float."
        assert clip_coef >= 0, "Clipping coefficient must be non-negative."
        assert isinstance(update_epochs, int), "Update epochs must be an integer."
        assert update_epochs >= 1, "Update epochs must be at least one."
        if clone and actor_network is not None:
            assert isinstance(
                actor_network,
                (PeftModelProtocol, PreTrainedModelProtocol),
            ), "Actor network must be a PeftModelProtocol or PreTrainedModelProtocol"

    def _setup_advantage_options(
        self,
        turn_level_clip: bool,
        advantage_granularity: str,
        turn_value_reduction: str,
        whiten_advantages: bool,
        gamma: float,
        gae_lambda: float,
    ) -> None:
        """Validate and store the GAE advantage options."""
        granularities = ["auto", "token", "turn"]
        if advantage_granularity not in granularities:
            msg = f"advantage_granularity must be one of {granularities}."
            raise ValueError(msg)
        reductions = ["final_value", "mean"]
        if turn_value_reduction not in reductions:
            msg = f"turn_value_reduction must be one of {reductions}."
            raise ValueError(msg)
        if not isinstance(whiten_advantages, bool):
            msg = "whiten_advantages must be a boolean."
            raise TypeError(msg)
        self.turn_level_clip = turn_level_clip
        self.advantage_granularity = advantage_granularity
        self.turn_value_reduction = turn_value_reduction
        self.whiten_advantages = whiten_advantages
        self.gamma = gamma
        self.gae_lambda = gae_lambda

    def _setup_objective(
        self,
        importance_sampling_level: str,
        turn_ratio_pooling: str,
    ) -> None:
        """Validate and resolve the importance-sampling level and Liger routing."""
        validate_importance_sampling_level(importance_sampling_level, allow_auto=True)
        if turn_ratio_pooling not in {"sum", "mean"}:
            msg = "turn_ratio_pooling must be one of ['mean', 'sum']."
            raise ValueError(msg)
        # IS / ratio-pooling level for the policy surrogate, orthogonal to the
        # GAE advantage granularity. ``"auto"`` pools at GAE granularity when
        # ``turn_level_clip`` is set, else token-level. Explicit
        # ``token``/``turn``/``trajectory`` use length-normalized mean pooling.
        self.importance_sampling_level = importance_sampling_level
        # Turn-level ratio pooling reduction (sum=product ratio, mean=geometric
        # mean ratio) used by both the standard and Liger PPO policy losses.
        self.turn_ratio_pooling = turn_ratio_pooling
        # Warn once that Liger + an explicit non-token IS level is permitted
        # but not memory-bounded ("auto" is covered at loss time instead).
        if self.use_liger_loss and self.importance_sampling_level in {
            "turn",
            "trajectory",
        }:
            self._warn_liger_non_token_is(
                self.importance_sampling_level,
                "PPO",
                once_attr="_ppo_liger_mem_warned",
            )

    def _setup_generation(
        self,
        max_output_tokens: int | None,
        min_output_tokens: int | None,
        max_model_len: int,
        hf_generate_chunk_size: int | None,
    ) -> None:
        """Build the HF generation config."""
        self.max_output_tokens = (
            max_output_tokens if max_output_tokens is not None else max_model_len
        )
        self.min_output_tokens = min_output_tokens
        self.max_model_len = max_model_len
        self.hf_generate_chunk_size = int(
            1 if hf_generate_chunk_size is None else max(1, hf_generate_chunk_size)
        )
        if self.colocated and hf_generate_chunk_size is not None:
            warnings.warn(
                "hf_generate_chunk_size is only used for HuggingFace generation "
                "and is ignored when colocated.",
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

    def _resolve_advantage_granularity(self, turn_ids: torch.Tensor) -> str:
        """Resolve effective PPO granularity for the current batch.

        :param turn_ids: Turn index per token ``[batch, seq_len]``; ``-1`` for padding.
        :type turn_ids: torch.Tensor
        :return: Effective PPO granularity.
        :rtype: str
        """
        return resolve_batch_advantage_granularity(
            self.advantage_granularity, turn_ids, single_turn="token", multi_turn="turn"
        )

    def _resolve_is_level(self, ppo_granularity: str) -> str:
        """Resolve the effective importance-sampling (ratio-pooling) level.

        ``"auto"`` pools at the GAE granularity (``ppo_granularity``) when
        ``turn_level_clip`` is set, else token-level. Explicit
        ``token``/``turn``/``trajectory`` override.

        :param ppo_granularity: Resolved GAE advantage granularity (token/turn).
        :type ppo_granularity: str
        :return: One of ``"token"``, ``"turn"``, ``"trajectory"``.
        :rtype: str
        """
        if self.importance_sampling_level == "auto":
            return ppo_granularity if self.turn_level_clip else "token"
        return self.importance_sampling_level

    def _compute_gae_returns(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        action_mask: torch.Tensor,
        turn_ids: torch.Tensor,
        gae_lambda: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute turn-level GAE and broadcast advantages to all action tokens.

        Each generation turn is treated as a single RL action. Per-turn values
        are aggregated from token values according to ``self.turn_value_reduction``,
        and gamma discounts between turns (not between tokens within a turn).

        :param rewards: Per-token (penalised) rewards ``[batch, seq_len]``.
        :type rewards: torch.Tensor
        :param values: Per-token critic values ``[batch, seq_len]``.
        :type values: torch.Tensor
        :param action_mask: Bool mask of valid action positions ``[batch, seq_len]``.
        :type action_mask: torch.Tensor
        :param turn_ids: Turn index per token ``[batch, seq_len]``; ``-1`` for padding.
        :type turn_ids: torch.Tensor
        :param gae_lambda: GAE lambda; ``1.0`` gives Monte Carlo returns.
        :type gae_lambda: float
        :return: Tuple of ``(token_returns, token_advantages)``, each ``[batch, seq_len]``.
        :rtype: tuple[torch.Tensor, torch.Tensor]
        """
        batch_size = values.shape[0]
        num_turns = int(turn_ids.max().item()) + 1

        turn_values = pool_by_turns(
            values, turn_ids, num_turns, reduction=self.turn_value_reduction
        )
        turn_rewards = pool_by_turns(rewards, turn_ids, num_turns)

        turn_advantages = torch.zeros(batch_size, num_turns, device=values.device)
        last_gae = torch.zeros(batch_size, device=values.device)
        per_sample_num_turns = turn_ids.max(dim=1).values + 1

        for t in reversed(range(num_turns)):
            is_last_turn = t >= (per_sample_num_turns - 1)
            if t == num_turns - 1:
                next_turn_value = torch.zeros_like(turn_values[:, 0])
            else:
                next_turn_value = turn_values[:, t + 1]
            next_turn_value = torch.where(
                is_last_turn, torch.zeros_like(next_turn_value), next_turn_value
            )

            delta = (
                turn_rewards[:, t] + self.gamma * next_turn_value - turn_values[:, t]
            )
            has_turn = (per_sample_num_turns > t).float()
            last_gae = (delta + self.gamma * gae_lambda * last_gae) * has_turn
            turn_advantages[:, t] = last_gae

        del turn_rewards

        turn_index = torch.arange(num_turns, device=turn_ids.device).view(1, 1, -1)
        turn_mask = (turn_ids.unsqueeze(-1) == turn_index).any(dim=1).float()
        if self.whiten_advantages:
            turn_advantages_for_pg = masked_whiten(turn_advantages, turn_mask)
        else:
            turn_advantages_for_pg = turn_advantages
        valid_turn_mask = turn_ids >= 0
        safe_turn_ids = turn_ids.clamp(min=0)

        turn_returns = turn_advantages + turn_values
        token_returns = turn_returns.gather(dim=1, index=safe_turn_ids)
        token_advantages = turn_advantages_for_pg.gather(dim=1, index=safe_turn_ids)
        token_returns = token_returns * valid_turn_mask.float()
        token_advantages = token_advantages * valid_turn_mask.float()
        del turn_values, turn_advantages
        return token_returns, token_advantages

    def _compute_gae_returns_token(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        action_mask: torch.Tensor,
        gae_lambda: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute token-level GAE and returns.

        :param rewards: Per-token rewards ``[batch, seq_len]``.
        :type rewards: torch.Tensor
        :param values: Per-token critic values ``[batch, seq_len]``.
        :type values: torch.Tensor
        :param action_mask: Bool mask of valid action positions ``[batch, seq_len]``.
        :type action_mask: torch.Tensor
        :param gae_lambda: GAE lambda; ``1.0`` gives Monte Carlo returns.
        :type gae_lambda: float
        :return: Tuple of ``(token_returns, token_advantages)``, each ``[batch, seq_len]``.
        :rtype: tuple[torch.Tensor, torch.Tensor]
        """
        mask = action_mask.float()
        batch_size, seq_len = values.shape
        token_advantages = torch.zeros_like(values)
        last_gae = torch.zeros(batch_size, device=values.device)

        for t in reversed(range(seq_len)):
            if t == seq_len - 1:
                next_value = torch.zeros(batch_size, device=values.device)
                next_mask = torch.zeros(batch_size, device=values.device)
            else:
                next_value = values[:, t + 1]
                next_mask = mask[:, t + 1]

            delta = rewards[:, t] + self.gamma * next_value * next_mask - values[:, t]
            last_gae = delta + self.gamma * gae_lambda * last_gae * next_mask
            token_advantages[:, t] = last_gae * mask[:, t]

        token_returns = (token_advantages + values) * mask
        if self.whiten_advantages:
            token_advantages = masked_whiten(token_advantages, action_mask)
        return token_returns, token_advantages * mask

    def _backward_ppo_pass(
        self, loss: torch.Tensor, accumulation_steps: int | None, passes: int
    ) -> OptimizerStep | None:
        """Backward one pass of a micro-batch.

        With two passes per micro-batch, the accumulation window counts twice
        as many backward calls and each loss doubles to keep the micro-batch's
        weight. Actor and critic clip apart unless :attr:`share_grad_clip`.

        :param loss: Scaled loss of one pass.
        :type loss: torch.Tensor
        :param accumulation_steps: Micro-batches per optimizer step; ``None``
            uses :attr:`gradient_accumulation_steps`.
        :type accumulation_steps: int | None
        :param passes: Backward passes per micro-batch.
        :type passes: int
        :return: Gradient norms of the optimizer step, or ``None`` when the
            pass only accumulates.
        :rtype: OptimizerStep | None
        """
        steps = accumulation_steps or self.gradient_accumulation_steps
        roles = () if self.share_grad_clip else ("actor", "critic")
        clip_groups = tuple(
            frozenset({role, f"{role}{REPLICATED_GROUP_SUFFIX}"}) for role in roles
        )
        return self._backward_pass(passes * loss, passes * steps, clip_groups or None)

    def _restore_checkpoint_attributes(
        self,
        checkpoint: dict[str, Any],
        restore_config: bool,
        restore_hyperparameters: bool,
    ) -> None:
        """Restore attributes and the warmup count, then re-resolve the fused pass.

        The warmup count is training state, restored even without ``restore_config``.

        :param checkpoint: Loaded attribute payload.
        :type checkpoint: dict[str, Any]
        :param restore_config: See :meth:`load_checkpoint`.
        :type restore_config: bool
        :param restore_hyperparameters: See :meth:`load_checkpoint`.
        :type restore_hyperparameters: bool
        """
        super()._restore_checkpoint_attributes(
            checkpoint, restore_config, restore_hyperparameters
        )
        self.critic_warmup_steps_done = checkpoint.get("critic_warmup_steps_done", 0)
        self._fuses_actor_critic_pass = self._resolve_fuse_actor_critic_pass()

    def _resolve_fuse_actor_critic_pass(self) -> bool:
        """Resolve whether the actor and critic rows share one gradient pass.

        An explicit ``fuse_actor_critic_pass`` is kept. ``None`` fuses off
        CUDA, else fuses when the estimated fused training peak fits the
        budget of :meth:`_estimate_training`. Every input is fixed by the run
        config and device model, so all ranks pick the same path.

        :return: Whether to run the fused pass.
        :rtype: bool
        """
        if self.fuse_actor_critic_pass is not None:
            return self.fuse_actor_critic_pass
        if torch.device(self.device).type != "cuda":
            return True
        estimate = self._estimate_training(
            "ppo", self.beta, fuse_actor_critic_pass=True
        )
        gib = 1024**3
        logger.info(
            "LLMPPO %s the actor and critic passes: fused peak %.2f GiB, "
            "budget %.2f GiB",
            "fuses" if estimate.fits else "splits",
            estimate.total_bytes / gib,
            estimate.device_usable_bytes / gib,
        )
        return estimate.fits

    def _actor_critic_hidden_states(
        self, batch_ids: torch.Tensor, pixel_values: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Gradient-bearing forward of one micro-batch's actor and critic rows together.

        The batch runs once per adapter in a single forward with ``lm_head``
        identity-patched. The routing stays set until the caller unsets it
        after ``backward()``, so gradient checkpointing recomputes each row
        under its own adapter. FlexAttention packs the batch once and runs the
        packed row per adapter; FlashAttention-2 isolates packed documents only
        on a single row, so other backends run padded.

        :param batch_ids: ``(B, T)`` right-padded token ids.
        :type batch_ids: torch.Tensor
        :param pixel_values: Vision rows of the batch, or ``None`` for text.
        :type pixel_values: torch.Tensor | None
        :return: ``(B, T, H)`` actor hidden states and ``(B, T - 1)`` critic values.
        :rtype: tuple[torch.Tensor, torch.Tensor]
        """
        model_kwargs, packed = self._gradient_forward_inputs(
            batch_ids, pixel_values, copies=2
        )
        rows = 1 if packed is not None else batch_ids.shape[0]
        set_fused_adapter_routing(self.actor, ["actor"] * rows + ["critic"] * rows)
        with (
            self._patch_lm_head_to_identity(),
            self._amp_ctx(),
            self._activation_offload_ctx(),
        ):
            self.actor.train()
            hidden, _, values = self.actor(**model_kwargs)
        if packed is not None:
            return unpack_hidden_states(hidden[:1], packed), unpack_values(
                values[1], packed
            )
        return hidden[:rows], values[rows:, :-1]

    def _actor_log_probs_from_hidden(
        self, actor_hidden: torch.Tensor, batch_ids: torch.Tensor
    ) -> torch.Tensor:
        """Gradient-bearing actor log-probs scored from the actor rows' hidden states.

        :param actor_hidden: ``(B, T, H)`` actor last hidden states.
        :type actor_hidden: torch.Tensor
        :param batch_ids: ``(B, T)`` token ids the hidden states were run on.
        :type batch_ids: torch.Tensor
        :return: ``(B, T - 1)`` per-token log-probs.
        :rtype: torch.Tensor
        """
        fused_fn, _, _ = self._fused_logprob_fn_and_head()
        with self.shard_runtime.gather_layer(
            self._get_lm_head(), device=batch_ids.device
        ) as (head_w, head_b):
            return self._fused_chunk_logprobs(
                actor_hidden, batch_ids, fused_fn, head_w, head_b
            )

    def _actor_critic_grad_norms(self, step: OptimizerStep) -> dict[str, float]:
        """Split one optimizer step's gradient norms by actor and critic param groups.

        :param step: Optimizer step returned by :meth:`_backward_ppo_pass`.
        :type step: OptimizerStep
        :return: Pre- and post-clip norms of actor LoRA and critic LoRA + value head.
        :rtype: dict[str, float]
        """
        pre, post = {"actor": 0.0, "critic": 0.0}, {"actor": 0.0, "critic": 0.0}
        param_groups = self.optimizer._single_optimizer().param_groups
        rows = zip(param_groups, step.group_grad_norms, step.clip_coefs, strict=True)
        for group, norm, coef in rows:
            role = group["group"].removesuffix(REPLICATED_GROUP_SUFFIX)
            pre[role] += norm**2
            post[role] += (norm * coef) ** 2
        return {
            "grad_norm_pre": pre["actor"] ** 0.5,
            "grad_norm_post": post["actor"] ** 0.5,
            "critic_grad_norm_pre": pre["critic"] ** 0.5,
            "critic_grad_norm_post": post["critic"] ** 0.5,
        }

    def _critic_values(
        self, batch_ids: torch.Tensor, pixel_values: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Gradient-bearing critic forward, every row routed to the critic adapter.

        ``lm_head`` is identity-patched so no logits are computed. The routing
        stays set until the caller unsets it after ``backward()``, so gradient
        checkpointing recomputes under the critic adapter.

        :param batch_ids: ``(B, T)`` right-padded token ids.
        :type batch_ids: torch.Tensor
        :param pixel_values: Vision rows of the batch, or ``None`` for text.
        :type pixel_values: torch.Tensor | None
        :return: ``(B, T - 1)`` per-token values.
        :rtype: torch.Tensor
        """
        model_kwargs, packed = self._gradient_forward_inputs(batch_ids, pixel_values)
        set_fused_adapter_routing(self.actor, ["critic"])
        with (
            self._patch_lm_head_to_identity(),
            self._amp_ctx(),
            self._activation_offload_ctx(),
        ):
            self.actor.train()
            _, _, values = self.actor(**model_kwargs)
        if packed is not None:
            return unpack_values(values[0], packed)
        return values[:, :-1]

    def _ppo_policy_loss(
        self,
        log_probs: torch.Tensor,
        action_mask: torch.Tensor,
        old_log_probs: torch.Tensor,
        reference_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        turn_ids: torch.Tensor,
        ppo_granularity: str,
        sampling_log_probs: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Clipped policy surrogate plus ``beta``-weighted KL from actor log-probs.

        :param log_probs: ``(B, T - 1)`` actor log-probs with grad.
        :type log_probs: torch.Tensor
        :return: ``(policy_loss, metrics)`` with ``metrics`` keying scalar
            Python floats: ``kl``, ``pg_loss``, ``clipfrac``, ``entropy``.
        :rtype: tuple[torch.Tensor, dict[str, float]]
        """
        log_probs = torch.masked_fill(log_probs, ~action_mask.bool(), 1.0)
        kl = calculate_k3_kl(reference_log_probs, log_probs)
        masked_entropy = masked_mean(-log_probs.detach(), action_mask)
        token_log_ratio = log_probs - old_log_probs

        # Truncated importance sampling: reweight each token's surrogate by the
        # (detached, clamped) trainer/vLLM ratio to correct for the rollout
        # being drawn from vLLM rather than the trainer policy. Applies only to
        # the policy surrogate, not the value/KL terms.
        loss_weight = None
        if sampling_log_probs is not None:
            with torch.no_grad():
                mask_f = action_mask.to(token_log_ratio.dtype)
                loss_weight = torch.exp(
                    (old_log_probs - sampling_log_probs) * mask_f
                ).clamp(max=self.vllm_importance_sampling_cap)

        pg_loss, clipfrac = clipped_is_surrogate(
            token_log_ratio,
            advantages,
            action_mask,
            turn_ids,
            self._resolve_is_level(ppo_granularity),
            self.clip_coef,
            loss_weight=loss_weight,
            turn_reduction=self.turn_ratio_pooling,
        )
        kl_loss = masked_mean(kl, action_mask)
        metrics = {
            "kl": kl_loss.item(),
            "entropy": masked_entropy.mean().item(),
            "clipfrac": clipfrac.item(),
            "pg_loss": pg_loss.mean().item(),
        }
        return pg_loss + self.beta * kl_loss, metrics

    def _ppo_value_loss(
        self,
        values: torch.Tensor,
        old_values: torch.Tensor,
        returns: torch.Tensor,
        action_mask: torch.Tensor,
        turn_ids: torch.Tensor,
        ppo_granularity: str,
    ) -> torch.Tensor:
        """Clipped value loss on the GAE advantage axis (``ppo_granularity``).

        :param values: ``(B, T - 1)`` critic values with grad.
        :type values: torch.Tensor
        :return: Scalar value loss, weighted by ``vf_coef``.
        :rtype: torch.Tensor
        """
        values = torch.masked_fill(values, ~action_mask.bool(), 0.0)
        if ppo_granularity != "turn":
            return self._compute_vf_loss_token(values, old_values, returns, action_mask)
        num_turns = int(turn_ids.max().item()) + 1
        turn_pred = pool_by_turns(
            values, turn_ids, num_turns, reduction=self.turn_value_reduction
        )
        turn_old = pool_by_turns(
            old_values, turn_ids, num_turns, reduction=self.turn_value_reduction
        )
        turn_ret = pool_by_turns(returns, turn_ids, num_turns)
        turn_mask = torch.zeros_like(turn_pred)
        for t in range(num_turns):
            turn_mask[:, t] = (turn_ids == t).any(dim=1).float()
        vf_unclipped = (turn_ret - turn_pred).pow(2)
        clipped_turn_values = turn_old + torch.clamp(
            turn_pred - turn_old, -self.clip_coef, self.clip_coef
        )
        vf_clipped = (turn_ret - clipped_turn_values).pow(2)
        return (
            0.5
            * (torch.max(vf_unclipped, vf_clipped) * turn_mask).sum()
            / turn_mask.sum().clamp(min=1)
            * self.vf_coef
        )

    def _ppo_policy_loss_liger(
        self,
        batch_ids: torch.Tensor,
        batch_action_mask: torch.Tensor,
        batch_old_log_probs: torch.Tensor,
        batch_reference_log_probs: torch.Tensor,
        batch_advantages: torch.Tensor,
        batch_turn_ids: torch.Tensor,
        ppo_granularity: str,
        sampling_log_probs: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """PPO policy + KL loss of the split actor pass via the fused-linear PPO Function.

        Every row routes to the actor adapter and ``lm_head`` is
        identity-patched, so the forward returns the hidden states
        :meth:`_ppo_policy_loss_liger_from_hidden` scores. The routing stays
        set until the caller unsets it after ``backward()``, so gradient
        checkpointing recomputes under the actor adapter.

        :param pixel_values: Vision rows of the minibatch's token rows, or
            ``None`` for text.
        :type pixel_values: torch.Tensor | None
        :return: ``(policy_loss, metrics)`` with ``metrics`` keying scalar
            Python floats: ``kl``, ``pg_loss``, ``clipfrac``, ``entropy``.
        :rtype: tuple[torch.Tensor, dict[str, float]]
        """
        batch_ids = batch_ids.to(self.device)
        set_fused_adapter_routing(self.actor, ["actor"])
        with self._activation_offload_ctx():
            policy_hidden = self._actor_hidden_states(batch_ids, pixel_values)
        return self._ppo_policy_loss_liger_from_hidden(
            policy_hidden,
            batch_ids,
            batch_action_mask,
            batch_old_log_probs,
            batch_reference_log_probs,
            batch_advantages,
            batch_turn_ids,
            ppo_granularity,
            sampling_log_probs,
        )

    def _ppo_policy_loss_liger_from_hidden(
        self,
        policy_hidden: torch.Tensor,
        batch_ids: torch.Tensor,
        batch_action_mask: torch.Tensor,
        batch_old_log_probs: torch.Tensor,
        batch_reference_log_probs: torch.Tensor,
        batch_advantages: torch.Tensor,
        batch_turn_ids: torch.Tensor,
        ppo_granularity: str,
        sampling_log_probs: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """PPO policy + KL loss from actor hidden states via the fused-linear PPO Function.

        :class:`LigerFusedLinearPolicyLossFunction` computes the chunked policy
        + KL loss without ever materializing ``(B, T, V)`` for the autograd
        graph.

        Granularity dispatch:

        * ``ppo_granularity == "token"``: per-token policy loss inside the
          Liger Function.
        * ``ppo_granularity == "turn"`` and ``self.turn_level_clip``:
          token log-ratios are scatter-pooled into per-turn log-ratios
          inside the Liger Function, against turn-level advantages.
        * ``ppo_granularity == "turn"`` and not ``self.turn_level_clip``:
          per-token policy loss (broadcast turn advantages to tokens up
          front).

        :param policy_hidden: ``(B, T, H)`` actor last hidden states with grad.
        :type policy_hidden: torch.Tensor
        :param batch_ids: ``(B, T)`` token ids the hidden states were run on.
        :type batch_ids: torch.Tensor
        :return: ``(policy_loss, metrics)`` with ``metrics`` keying scalar
            Python floats: ``kl``, ``pg_loss``, ``clipfrac``, ``entropy``.
        :rtype: tuple[torch.Tensor, dict[str, float]]
        """
        if not HAS_LIGER_KERNEL:
            msg = (
                "Liger PPO loss was requested but `liger-kernel` is not "
                "available. Set use_liger_loss=False."
            )
            raise ImportError(msg)

        mask = batch_action_mask.to(self.device).contiguous()
        old_log_probs = batch_old_log_probs.to(self.device).contiguous()
        ref_log_probs = batch_reference_log_probs.to(self.device).contiguous()
        advantages = batch_advantages.to(self.device).contiguous()
        turn_ids = batch_turn_ids.to(self.device).contiguous()

        # Resolve the IS / ratio-pooling level and pool advantages to match.
        is_level = self._resolve_is_level(ppo_granularity)
        if is_level != "token":
            self._warn_liger_non_token_is(
                is_level, "PPO", once_attr="_ppo_liger_mem_warned"
            )
        # ``max_turns`` / ``full_turn_mask`` are derived from the global
        # ``turn_ids`` so chunks see consistent denominators for turn-level
        # IS pooling.
        max_turns = int(turn_ids.max().item()) + 1
        full_turn_mask = torch.zeros(turn_ids.shape[0], max_turns, device=self.device)
        for t in range(max_turns):
            full_turn_mask[:, t] = (turn_ids == t).any(dim=1).float()

        turn_ids_arg: torch.Tensor | None = None
        if is_level == "turn":
            # Liger fn expects per-turn advantages ``(B, max_turns)``; pool the
            # per-token advantages by turn-mean to match the pooled ratio.
            adv_for_liger = pool_by_turns(advantages, turn_ids, max_turns)
            turn_ids_arg = turn_ids
        elif is_level == "trajectory":
            mask_f = mask.to(advantages.dtype)
            adv_for_liger = (advantages * mask_f).sum(
                dim=-1, keepdim=True
            ) / mask_f.sum(dim=-1, keepdim=True).clamp(min=1.0)  # (B, 1)
        else:  # token
            adv_for_liger = advantages

        # Truncated importance sampling fused into the kernel (token level only):
        # reweight each token's policy loss by the detached, clamped trainer/vLLM
        # ratio. Non-token IS routes the correction to the standard path.
        vllm_is_ratio = None
        if sampling_log_probs is not None and is_level == "token":
            with torch.no_grad():
                mask_f = mask.to(old_log_probs.dtype)
                vllm_is_ratio = torch.exp(
                    (old_log_probs - sampling_log_probs.to(self.device)) * mask_f
                ).clamp(max=self.vllm_importance_sampling_cap)

        target_ids = batch_ids[:, 1:].contiguous()
        # Token level token-flattens the hidden states so the fused kernel
        # chunks tokens (bounded); turn/sequence keep the batch path. See
        # :func:`apply_fused_policy_loss`.
        with self._liger_head_gather() as (head_w, head_b):
            loss_pg_kl, aux = apply_fused_policy_loss(
                policy_hidden[:, :-1],
                head_w,
                head_b,
                target_ids,
                mask,
                adv_for_liger,
                ref_log_probs,
                old_log_probs,
                self.beta,
                self.clip_coef,  # epsilon_low
                self.clip_coef,  # epsilon_high
                self.temperature,
                is_level,
                turn_ids=turn_ids_arg,
                full_turn_mask=full_turn_mask,
                max_turns=max_turns,
                token_chunk_size=self._resolve_fused_chunk_rows(
                    head_w.shape[0],
                    self.chunk_rows,
                ),
                turn_log_ratio_reduction=self.turn_ratio_pooling,
                vllm_is_ratio=vllm_is_ratio,
            )
        metrics = {
            "kl": float(aux[0].item()),
            "clipfrac": float(aux[1].item()),
            "pg_loss": float(aux[2].item()),
            "entropy": float(aux[3].item()),
        }
        return loss_pg_kl, metrics

    def _compute_vf_loss_token(
        self,
        values: torch.Tensor,
        old_values: torch.Tensor,
        returns: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute token-level clipped value loss.

        :param values: Current value predictions ``[batch, seq_len]``.
        :type values: torch.Tensor
        :param old_values: Old value predictions ``[batch, seq_len]``.
        :type old_values: torch.Tensor
        :param returns: Token-level returns ``[batch, seq_len]``.
        :type returns: torch.Tensor
        :param action_mask: Bool mask of valid action positions ``[batch, seq_len]``.
        :type action_mask: torch.Tensor
        :return: Scalar token-level value loss.
        :rtype: torch.Tensor
        """
        vf_loss = (returns - values).pow(2)
        clipped_values = old_values + torch.clamp(
            values - old_values, -self.clip_coef, self.clip_coef
        )
        clipped_vf_loss = (returns - clipped_values).pow(2)
        return (
            0.5
            * masked_mean(torch.max(vf_loss, clipped_vf_loss), action_mask)
            * self.vf_coef
        )

    def _compute_token_rewards(
        self,
        action_mask: torch.Tensor,
        rewards: torch.Tensor,
        turn_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Assign per-turn rewards to each action token based on turn_ids.

        :param action_mask: Bool mask of action positions ``[batch, seq_len]``.
        :type action_mask: torch.Tensor
        :param rewards: Per-turn scalars ``[batch, max_turns]``.
        :type rewards: torch.Tensor
        :param turn_ids: Turn index per token ``[batch, seq_len]``; ``-1`` for non-action.
        :type turn_ids: torch.Tensor
        :return: Per-token rewards ``[batch, seq_len]``.
        :rtype: torch.Tensor
        """
        num_turns = rewards.shape[1]
        token_rewards = torch.zeros_like(action_mask, dtype=torch.float)
        for t in range(num_turns):
            mask_t = (turn_ids == t).float()
            token_rewards += mask_t * rewards[:, t : t + 1]
        return token_rewards
