# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import replace
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import torch

from agilerl import HAS_LLM_DEPENDENCIES
from agilerl.algorithms.core import ActionResult, LLMAlgorithm
from agilerl.algorithms.core.optimizer_wrapper import REPLICATED_GROUP_SUFFIX
from agilerl.algorithms.core.registry import HyperparameterConfig, NetworkGroup
from agilerl.arena.models.algorithms.llmppo import LLMPPOSpec
from agilerl.lora.fused import set_fused_adapter_routing, unset_fused_adapter_routing

if TYPE_CHECKING:
    from peft import LoraConfig, PeftModel
    from transformers import PreTrainedModel

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
from agilerl.utils.llm_utils import (
    LLM_RL_COMMON_METRIC_NAMES,
    PPO_METRIC_NAMES,
    VLLM_IS_METRIC_NAMES,
    BitsAndBytesConfig,
    MicroBatchMetrics,
    attention_mask_from_padded_ids,
    broadcast_log_ratio_by_level,
    calculate_k3_kl,
    clipped_token_surrogate,
    global_masked_whiten,
    k3_kl_penalty_reference,
    masked_mean,
    normalize_prompt_batch,
    off_policy_token_drops,
    pool_by_turns,
    resolve_batch_advantage_granularity,
    turn_state_values,
    validate_importance_sampling_level,
    validate_off_policy_token_mask_bounds,
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

    Every token of a turn shares the turn's geometric-mean ratio, the clip
    binds per token, and the policy loss is the mean over action tokens.

    The importance ratio divides by the rollout engine's sampling log-probs.
    Rows without them on every action token take the learn-start policy's
    from the no-grad forward.

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
    :param gae_lambda: GAE lambda; the default ``1.0`` uses Monte Carlo returns.
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
    :param reference_update_interval: Optimizer steps between copies of the
        actor adapter onto the reference. ``None`` (default) keeps the
        reference fixed.
    :type reference_update_interval: int | None, optional
    :param calc_position_embeddings: Whether to compute position embeddings.
    :type calc_position_embeddings: bool, optional
    :param micro_batch_size_per_gpu: Optional target micro-batch size per GPU.
    :type micro_batch_size_per_gpu: int | None, optional
    :param mini_batch_size: Per-rank trajectories covered by one optimizer
        step. ``None`` uses ``batch_size / world_size``.
        ``gradient_accumulation_steps`` is derived as
        ``mini_batch_size / micro_batch_size_per_gpu``.
    :type mini_batch_size: int | None, optional
    :param gradient_checkpointing: Recompute block activations in the backward
        pass. ``None`` lets the training-memory search pick it on CUDA for a
        catalog model; elsewhere it is on.
    :type gradient_checkpointing: bool | None, optional
    :param max_output_tokens: Maximum newly generated tokens per completion.
    :type max_output_tokens: int | None, optional
    :param min_output_tokens: Minimum newly generated tokens per completion.
    :type min_output_tokens: int | None, optional
    :param max_model_len: Maximum model context length.
    :type max_model_len: int, optional
    :param max_row_tokens: Longest training row (prompt plus completion) the
        memory estimate sizes for. ``None`` uses ``max_model_len``.
    :type max_row_tokens: int | None, optional
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
        token; ``"turn"`` gives every token of a turn the turn's
        length-normalized geometric-mean ratio; ``"trajectory"`` the
        completion's. Advantages and the clip stay per token. ``"auto"``
        (default) uses the GAE granularity when ``turn_level_clip`` is set,
        else token.
    :type importance_sampling_level: Literal["auto", "token", "turn", "trajectory"], optional
    :param advantage_granularity: PPO action granularity. ``"turn"`` enforces
        turn-level updates, ``"token"`` enforces token-level updates, and
        ``"auto"`` uses token-level only when all samples are single-turn.
    :type advantage_granularity: Literal["turn", "token", "auto"], optional
    :param whiten_advantages: Whether to whiten computed advantages before PPO
        optimization, with the mean and variance of the whole batch across ranks.
    :type whiten_advantages: bool, optional
    :param quantization_config: Optional ``transformers.BitsAndBytesConfig`` for
        loading the base model in 4-/8-bit (QLoRA). ``lm_head`` is kept
        unquantized so the fused-linear-logprob path stays numerically exact.
    :type quantization_config: BitsAndBytesConfig | None, optional
    :param vllm_importance_sampling_cap: Trainer/vLLM ratio at which the
        sampling-mismatch metrics count a token as clipped (default ``2.0``).
        Must be > 0.
    :type vllm_importance_sampling_cap: float, optional
    :param vllm_max_logprob_gap: Log a warning when a learn step's mean
        ``|trainer - vLLM|`` per-token log-prob gap exceeds this, in nats
        (default ``0.1``). bf16 engines usually sit near 0.01-0.03.
    :type vllm_max_logprob_gap: float, optional
    :param vllm_max_clip_fraction: Log a warning when the fraction of action
        tokens whose trainer/vLLM ratio reaches ``vllm_importance_sampling_cap``
        exceeds this (default ``0.02``). bf16 engines usually stay under 0.01.
    :type vllm_max_clip_fraction: float, optional
    :param lora_tower_target_modules: Tower and connector modules of a VL model
        to adapt as well. ``lora_config.target_modules`` always land on the
        language model only. Defaults to None.
    :type lora_tower_target_modules: list[str] | None, optional
    :param rollout_strip_multimodal_towers: Whether the vLLM engine serving the
        adapter skips the model's multimodal towers; tower LoRA skips the towers
        it leaves unbuilt. Builders derive it from the environment's modality.
        Defaults to False.
    :type rollout_strip_multimodal_towers: bool, optional
    :param kl_clamp: Per-token bound on the K3 KL penalty. Tokens where the
        policy sits below the reference with K3 above the bound (10 is about
        2.6 nats below) get no KL gradient and add nothing to the loss.
        The ``kl`` metric stays unbounded; ``kl_clamp_frac`` reports the
        share of action tokens past the bound. ``None`` disables the bound,
        defaults to 10.0. Must be > 0.
    :type kl_clamp: float | None, optional
    :param off_policy_token_mask_bounds: ``(low, high)`` band on each action
        token's ratio of the current policy to the policy that sampled it.
        Tokens outside get no policy or KL gradient (IcePop). Defaults to
        ``(0.5, 5.0)``; ``None`` masks nothing. Requires ``0 <= low < 1 < high``.
    :type off_policy_token_mask_bounds: tuple[float, float] | None, optional
    :param logit_bias: Token id to logit offset vLLM adds when it samples;
        the trainer's log-probs add the same. ``None`` adds none.
    :type logit_bias: dict[int, float] | None, optional
    :param micro_batch_tokens_per_gpu: Real tokens per learn micro-batch on
        one rank when the forward packs sequences; rows pack in order up to
        it, and a row longer than it runs alone. At least the longest row.
        ``None`` uses ``micro_batch_size_per_gpu`` rows of ``max_row_tokens``
        (else ``max_model_len``) tokens. Padded forwards count rows.
    :type micro_batch_tokens_per_gpu: int | None, optional
    :param routed_expert_chunk_mib: MiB of the widest activation of one
        routed-expert row chunk under DDP; FSDP reads
        ``fsdp_config.routed_expert_chunk_mib``. ``None`` lets the
        training-memory search pick it when it runs, else keeps 64.
    :type routed_expert_chunk_mib: int | None, optional
    :param stochastic_round_bf16: Stochastic-round fp32 master weights to bf16
        once per optimizer step, and reuse that tensor for the trainer forward
        and the rollout copy. Defaults to False (round-to-nearest).
    :type stochastic_round_bf16: bool, optional
    """

    gradient_row_copies = LLMPPOSpec.gradient_row_copies
    supports_sequence_packing = LLMPPOSpec.supports_sequence_packing

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
        gae_lambda: float = 1.0,
        critic_warmup_steps: int = 0,
        lr_actor: float = 1e-5,
        lr_critic: float | None = 1e-4,
        max_grad_norm: float = 1.0,
        share_grad_clip: bool = False,
        update_epochs: int = 1,
        temperature: float = 1.0,
        repetition_penalty: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 50,
        min_p: float = 0.0,
        use_separate_reference_adapter: bool = True,
        reference_update_interval: int | None = None,
        calc_position_embeddings: bool = True,
        micro_batch_size_per_gpu: int | None = None,
        mini_batch_size: int | None = None,
        gradient_checkpointing: bool | None = None,
        max_output_tokens: int | None = None,
        min_output_tokens: int | None = None,
        max_model_len: int = 1024,
        max_row_tokens: int | None = None,
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
        whiten_advantages: bool = True,
        quantization_config: BitsAndBytesConfig | None = None,
        lora_tower_target_modules: list[str] | None = None,
        rollout_strip_multimodal_towers: bool = False,
        vllm_importance_sampling_cap: float = 2.0,
        vllm_max_logprob_gap: float = 0.1,
        vllm_max_clip_fraction: float = 0.02,
        kl_clamp: float | None = 10.0,
        off_policy_token_mask_bounds: tuple[float, float] | None = (0.5, 5.0),
        logit_bias: dict[int, float] | None = None,
        micro_batch_tokens_per_gpu: int | None = None,
        routed_expert_chunk_mib: int | None = None,
        stochastic_round_bf16: bool = False,
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
            lora_config=lora_config,
            use_separate_reference_adapter=use_separate_reference_adapter,
            reference_update_interval=reference_update_interval,
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
            quantization_config=quantization_config,
            lora_tower_target_modules=lora_tower_target_modules,
            rollout_strip_multimodal_towers=rollout_strip_multimodal_towers,
            vllm_importance_sampling_cap=vllm_importance_sampling_cap,
            vllm_max_logprob_gap=vllm_max_logprob_gap,
            vllm_max_clip_fraction=vllm_max_clip_fraction,
            logit_bias=logit_bias,
            routed_expert_chunk_mib=routed_expert_chunk_mib,
            stochastic_round_bf16=stochastic_round_bf16,
        )
        self._validate_core_args(
            batch_size, lr_actor, clip_coef, update_epochs, actor_network, clone
        )
        if kl_clamp is not None and kl_clamp <= 0.0:
            msg = f"kl_clamp must be > 0 or None, got {kl_clamp}."
            raise ValueError(msg)
        validate_off_policy_token_mask_bounds(off_policy_token_mask_bounds)
        self.beta = beta
        self.kl_clamp = kl_clamp
        self.off_policy_token_mask_bounds = off_policy_token_mask_bounds
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
            whiten_advantages,
            gamma,
            gae_lambda,
        )
        self._setup_objective(importance_sampling_level)
        self._setup_generation(max_output_tokens, min_output_tokens, max_model_len)
        self.max_row_tokens = max_row_tokens
        self.micro_batch_tokens_per_gpu = micro_batch_tokens_per_gpu
        self._validate_micro_batch_tokens()
        self._setup_actors(actor_network, clone=clone)

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
            masks. On training rollouts ``sampling_logps`` carries the
            captured per-row sampling logprobs; otherwise it is ``None``.
        :rtype: ActionResult
        """
        prompts = normalize_prompt_batch(obs)
        self._ensure_vllm_for_generation()
        sampling_logps: list[torch.Tensor | None] | None = None
        sampling_supports: list[torch.Tensor | None] | None = None

        with self.select_adapter("actor"):
            self.actor.eval()
            self._prepare_vllm_for_generation()
            (
                token_ids_list,
                completion_masks,
                sampling_logps,
                sampling_supports,
            ) = self._generate_with_vllm_colocate(
                # RolloutPrompt is a TypedDict, i.e. a plain dict at
                # runtime; the base helper takes untyped prompt dicts.
                prompts,
                1,
                temperature=self.temperature
                if training
                else 0.01,  # Almost deterministic for evaluation
                capture_sampling_logps=training,
                capture_sampling_supports=training and self.records_sampling_supports,
            )

        return ActionResult(
            token_ids_list, completion_masks, sampling_logps, sampling_supports
        )

    def learn(
        self,
        experiences: LLMRolloutExperiences,
        turn_ids: torch.Tensor | None = None,
        sampling_logps: list[torch.Tensor | None] | None = None,
        episode_segments: list[EpisodeSegments | None] | None = None,
        pixel_values: torch.Tensor | None = None,
        pixel_image_counts: Sequence[int] | None = None,
        image_token_id: int | None = None,
        sampling_supports: list[torch.Tensor | None] | None = None,
    ) -> dict[str, float]:
        """Update actor and critic adapters; a critic warmup step updates only the critic.

        :param experiences: ``(token_ids, action_masks, rewards)``. For
            single-turn, ``rewards`` is a flat tensor of scalars; for multi-turn,
            shape ``[batch, max_turns]`` per-turn rewards.
        :type experiences: LLMRolloutExperiences
        :param turn_ids: Optional ``[batch, seq_len - 1]`` turn index per generated
            token, including tokens outside the action mask; ``-1`` elsewhere.
            If ``None``, all action tokens are turn ``0``.
        :type turn_ids: torch.Tensor | None
        :param sampling_logps: Optional per-row flat vLLM sampling logprobs (one
            1-D tensor per trajectory, generated tokens only; concatenated across
            turns for multi-turn), the old policy of the importance ratio.
            vLLM reports them under the penalised distribution it sampled
            from, and the policy carries no penalties, so the ratio is the
            exact policy-to-behaviour ratio. Parallel to the stacked
            ``token_ids`` rows. Rows without them, or ``None``, use the
            learn-start policy.
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
        :param sampling_supports: Optional per-row ``(n, K)`` token ids vLLM
            could sample at each generated token, ``-1`` padded, parallel to
            ``sampling_logps``. The policy, old-policy and reference log-probs
            of those tokens are normalized over their support, the
            distribution vLLM sampled from.
        :type sampling_supports: list[torch.Tensor | None] | None
        :return: Mean training metrics across PPO minibatch updates, actor and
            critic gradient norms averaged over optimizer steps, the pre-update
            value fit, ``critic_warmup``, row padding stats and
            ``learn_phase_<phase>_s`` wall seconds.
        :rtype: dict[str, float]
        """
        phase_timer = self._start_learn_phases()
        self._prepare_vllm_for_training()
        critic_warmup = self.critic_warmup_steps_done < self.critic_warmup_steps
        with self.trainer_offload_context(), self._vision_step(pixel_values):
            token_ids, action_masks, turn_ids, rewards_2d = self._stack_rollout_batch(
                experiences, turn_ids
            )
            action_mask_bool = action_masks.bool()
            num_samples = token_ids.shape[0]
            ppo_granularity = self._resolve_advantage_granularity(turn_ids)

            batch_size = min(num_samples, self.micro_batch_size_per_gpu)
            learn_metrics = MicroBatchMetrics(
                (
                    "vf_loss",
                    "policy_loss",
                    "pg_loss",
                    "kl",
                    "kl_clamp_frac",
                    "entropy",
                    "clipfrac",
                    "off_policy_token_mask_frac",
                )
            )
            grad_norm_totals = {
                "grad_norm_pre": 0.0,
                "grad_norm_post": 0.0,
                "critic_grad_norm_pre": 0.0,
                "critic_grad_norm_post": 0.0,
            }
            grad_updates = 0
            has_segments = self._has_episode_segments(episode_segments, num_samples)
            split = rows = None
            accumulation_steps = None
            if has_segments:
                self._check_segments_supported(self._resolve_is_level(ppo_granularity))
                split, rows = self._segment_rows(
                    token_ids,
                    action_masks,
                    episode_segments,
                    np.arange(num_samples),
                    pixel_values=pixel_values,
                    pixel_image_counts=pixel_image_counts,
                    image_token_id=image_token_id,
                    sampling_supports=sampling_supports,
                )
                pixel_values = rows.pixel_values
                pixel_image_counts = rows.pixel_image_counts
                batch_size = self.micro_batch_size_per_gpu
                support_ids = self._align_sampling_supports(
                    rows.sampling_supports, rows.token_ids, rows.action_masks
                )
            else:
                support_ids = self._align_sampling_supports(
                    sampling_supports, token_ids, action_masks
                )
            token_budget = None if has_segments else self._micro_batch_token_budget()
            phase_timer.mark("prepare")
            no_grad_images = 0 if pixel_values is None else int(pixel_values.shape[0])
            reference_log_probs, old_log_probs, old_values = (
                self._fused_forward_no_grad(
                    token_ids if rows is None else rows.token_ids,
                    batch_size=batch_size,
                    pixel_values=pixel_values,
                    pixel_image_counts=pixel_image_counts,
                    image_keys=(
                        None if pixel_values is None else torch.arange(no_grad_images)
                    ),
                    support_ids=support_ids,
                    token_budget=token_budget,
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
            # Turn GAE bootstraps from the value of every played turn, forced
            # tokens included, so values stay on every turn token.
            old_values = torch.masked_fill(old_values, turn_ids < 0, 0.0)

            old_log_probs = torch.masked_fill(old_log_probs, ~action_mask_bool, 1.0)
            reference_log_probs = torch.masked_fill(
                reference_log_probs, ~action_mask_bool, 1.0
            )
            # Warmup fits the critic to Monte Carlo returns of the fixed policy.
            gae_lambda = 1.0 if critic_warmup else self.gae_lambda
            if ppo_granularity == "token":
                returns, advantages = self._compute_gae_returns_token(
                    rewards_2d, old_values, action_masks, turn_ids, gae_lambda
                )
            else:
                token_rewards = self._compute_token_rewards(
                    action_masks, rewards_2d, turn_ids
                )
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
            )

            sampling_log_probs, is_metrics, is_excluded = (
                self._aligned_sampling_logprobs_and_metrics(
                    sampling_logps, action_masks, old_log_probs
                )
            )
            # Rows lacking sampling log-probs keep the learn-start policy's.
            if sampling_log_probs is not None:
                old_log_probs = sampling_log_probs

            row_episodes = np.arange(num_samples)
            image_keys = None if pixel_values is None else torch.arange(no_grad_images)
            if split is not None:
                frames = [
                    (turn_ids, -1),
                    (old_log_probs, 1.0),
                    (reference_log_probs, 1.0),
                    (returns, 0.0),
                    (advantages, 0.0),
                    (old_values, 0.0),
                ]
                rows, row_frames, accumulation_steps, image_keys = (
                    self._balanced_segment_frames(
                        split,
                        [(split.split_frame(frame, pad), pad) for frame, pad in frames],
                        num_samples,
                        image_token_id,
                    )
                )
                (
                    turn_ids,
                    old_log_probs,
                    reference_log_probs,
                    returns,
                    advantages,
                    old_values,
                ) = row_frames
                token_ids = rows.token_ids
                action_masks = rows.action_masks
                pixel_values = rows.pixel_values
                pixel_image_counts = rows.pixel_image_counts
                row_episodes = rows.row_episodes
                num_samples = int(token_ids.shape[0])
                support_ids = self._align_sampling_supports(
                    rows.sampling_supports, token_ids, action_masks
                )
            # At least 1: filler stand-in rows carry turn 0.
            num_turns = max(int(turn_ids.max().item()) + 1, 1)
            batch_idxs = np.arange(num_samples)
            filler_rows = row_episodes < 0
            padding_stats = self._row_padding_stats(token_ids, row_episodes, batch_idxs)
            phase_timer.mark("prepare")

            held_loss: torch.Tensor | None = None
            self.actor.train()
            epoch_orders = self._epoch_orders(
                batch_idxs, self.update_epochs, aligned=has_segments
            )
            windows = self._micro_batch_windows(
                epoch_orders,
                num_samples
                if token_budget is None
                else batch_size * self.gradient_accumulation_steps,
                batch_size,
                token_ids,
                token_budget,
            )
            micro_batches_per_step = self._micro_batches_per_step(
                windows,
                (accumulation_steps or self.gradient_accumulation_steps)
                if token_budget is None
                else None,
            )
            # Policy, KL and value units: each term averages over its window's units.
            term_units = (
                None
                if token_budget is None
                else (
                    self._loss_units_per_row(action_masks, None, "token"),
                    self._loss_units_per_row(action_masks, None, "token"),
                    self._loss_units_per_row(turn_ids >= 0, turn_ids, "turn")
                    if ppo_granularity == "turn"
                    else self._loss_units_per_row(action_masks, None, "token"),
                )
            )
            micro_batch_plan = []
            for epoch_windows in windows:
                for window in epoch_windows:
                    loss_scales = (
                        [1.0] * len(window)
                        if accumulation_steps is None
                        else self._segment_loss_scales(
                            np.array([filler_rows[idxs].all() for idxs in window]),
                            accumulation_steps,
                        ).tolist()
                    )
                    term_scales = (
                        [(None, None, None)] * len(window)
                        if term_units is None
                        else list(
                            zip(
                                *(
                                    self._window_loss_scales(units, window)
                                    for units in term_units
                                ),
                                strict=True,
                            )
                        )
                    )
                    window_steps = (
                        accumulation_steps if term_units is None else len(window)
                    )
                    micro_batch_plan += [
                        (idxs, scale, window_steps, scales)
                        for idxs, scale, scales in zip(
                            window, loss_scales, term_scales, strict=True
                        )
                    ]
            learn_vision = self._learn_vision_rows(
                (
                    VisionRows(pixel_values, pixel_image_counts, image_keys)
                    if pixel_values is not None
                    else None
                ),
                [entry[0] for entry in micro_batch_plan],
                num_samples,
            )
            for (
                minibatch_idxs,
                loss_scale,
                window_steps,
                (pg_scale, kl_scale, vf_scale),
            ) in micro_batch_plan:
                backward_steps = window_steps or self.gradient_accumulation_steps
                rows_index = self._device_index(minibatch_idxs, token_ids.device)
                phase_timer.mark("other")
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
                    batch_turn_ids,
                ) = get_experiences_samples(
                    rows_index,
                    token_ids,
                    action_masks,
                    old_log_probs,
                    reference_log_probs,
                    returns,
                    advantages,
                    turn_ids,
                )
                is_filler = bool(filler_rows[minibatch_idxs].all())
                if is_filler:
                    batch_action_mask, batch_turn_ids = filler_stand_in(
                        batch_action_mask, batch_turn_ids
                    )

                batch_pixel_values, batch_keys = next(learn_vision)

                batch_support_ids = (
                    support_ids[torch.as_tensor(minibatch_idxs)]
                    if support_ids is not None
                    else None
                )
                policy_inputs = (
                    batch_action_mask,
                    batch_old_log_probs,
                    batch_reference_log_probs,
                    batch_advantages,
                    batch_turn_ids,
                    num_turns,
                    ppo_granularity,
                )
                value_inputs = (
                    batch_returns,
                    batch_action_mask,
                    batch_turn_ids,
                    num_turns,
                    ppo_granularity,
                )
                if critic_warmup:
                    # Warmup fits the critic with the policy fixed, so
                    # actor params keep no grad and take no optimizer step.
                    values = self._critic_values(
                        batch_ids, batch_pixel_values, batch_keys
                    )
                    policy_loss = token_ids.new_zeros((), dtype=torch.float32)
                    metrics = dict.fromkeys(
                        (
                            "kl",
                            "kl_clamp_frac",
                            "entropy",
                            "clipfrac",
                            "off_policy_token_mask_frac",
                            "pg_loss",
                        ),
                        0.0,
                    )
                else:
                    log_probs, values = self._actor_critic_pass(
                        batch_ids,
                        batch_action_mask,
                        batch_support_ids,
                        batch_pixel_values,
                        batch_keys,
                    )
                    policy_loss, metrics = self._ppo_policy_loss(
                        log_probs,
                        *policy_inputs,
                        term_scales=(
                            None if pg_scale is None else (pg_scale, kl_scale)
                        ),
                    )
                vf_loss = self._ppo_value_loss(values, *value_inputs)
                if vf_scale is not None:
                    vf_loss = vf_loss * vf_scale
                total_loss = policy_loss + vf_loss
                held_loss = self._hold_loss_until_step(
                    total_loss, held_loss, backward_steps
                )
                phase_timer.mark("forward")
                # Autograd keeps any vision rows backward reads.
                del batch_pixel_values
                step = self._backward_ppo_pass(total_loss * loss_scale, window_steps)
                unset_fused_adapter_routing(self.actor)
                if step is not None:
                    for key, norm in self._actor_critic_grad_norms(step).items():
                        grad_norm_totals[key] += norm
                    grad_updates += 1
                if is_filler:
                    continue
                learn_metrics.add(
                    {"vf_loss": vf_loss, "policy_loss": policy_loss, **metrics},
                    counted=batch_action_mask.any(),
                )
            # Gradients still pending a step carry into the next learn call.
            if held_loss is not None:
                self._raise_if_loss_not_finite_on_any_rank(held_loss)
        self.critic_warmup_steps_done += int(critic_warmup)
        updates = learn_metrics.read()
        averaged = {
            metric: sum(
                update["policy_loss"] + update["vf_loss"]
                if metric == "loss"
                else update[metric]
                for update in updates
            )
            / max(len(updates), 1)
            for metric in (
                "loss",
                "pg_loss",
                "vf_loss",
                "kl",
                "entropy",
                "clipfrac",
                "off_policy_token_mask_frac",
            )
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
            "micro_batches_per_step": micro_batches_per_step,
            **padding_stats,
            **self._learn_phase_seconds(),
            **self._take_bf16_changed_frac(),
        }

        # Wire averaged metrics into the metrics tracker.
        token_ids_list = experiences[0]
        completion_length = np.mean([c.shape[-1] for c in token_ids_list])
        agg = aggregate_metrics_dict(
            {**result, "completion_length": completion_length}, excluded=is_excluded
        )
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
        whiten_advantages: bool,
        gamma: float,
        gae_lambda: float,
    ) -> None:
        """Validate and store the GAE advantage options."""
        granularities = ["auto", "token", "turn"]
        if advantage_granularity not in granularities:
            msg = f"advantage_granularity must be one of {granularities}."
            raise ValueError(msg)
        if not isinstance(whiten_advantages, bool):
            msg = "whiten_advantages must be a boolean."
            raise TypeError(msg)
        self.turn_level_clip = turn_level_clip
        self.advantage_granularity = advantage_granularity
        self.whiten_advantages = whiten_advantages
        self.gamma = gamma
        self.gae_lambda = gae_lambda

    def _setup_objective(self, importance_sampling_level: str) -> None:
        """Validate and store the importance-sampling level."""
        validate_importance_sampling_level(importance_sampling_level, allow_auto=True)
        # IS level for the policy surrogate, orthogonal to GAE granularity.
        # ``"auto"`` pools at GAE granularity when ``turn_level_clip`` is set,
        # else token-level. Turn and trajectory pool a geometric-mean ratio.
        self.importance_sampling_level = importance_sampling_level

    def _setup_generation(
        self,
        max_output_tokens: int | None,
        min_output_tokens: int | None,
        max_model_len: int,
    ) -> None:
        """Build the generation config."""
        self.max_output_tokens = (
            max_output_tokens if max_output_tokens is not None else max_model_len
        )
        self.min_output_tokens = min_output_tokens
        self.max_model_len = max_model_len
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
        are the critic at the last query token of that turn, and gamma discounts
        between turns (not between tokens within a turn).

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

        turn_values = turn_state_values(values, turn_ids, num_turns)
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

        valid_turn_mask = turn_ids >= 0
        safe_turn_ids = turn_ids.clamp(min=0)

        turn_returns = turn_advantages + turn_values
        token_returns = turn_returns.gather(dim=1, index=safe_turn_ids)
        token_advantages = turn_advantages.gather(dim=1, index=safe_turn_ids)
        # Whiten over the action tokens the policy loss averages over.
        if self.whiten_advantages:
            token_advantages = global_masked_whiten(token_advantages, action_mask)
        token_returns = token_returns * valid_turn_mask.float()
        token_advantages = token_advantages * valid_turn_mask.float()
        del turn_values, turn_advantages
        return token_returns, token_advantages

    def _compute_gae_returns_token(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        action_mask: torch.Tensor,
        turn_ids: torch.Tensor,
        gae_lambda: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute token-level GAE and returns over the action tokens.

        Each turn's reward is written on its last action token. GAE chains from one
        action token to the next, skipping the environment tokens between
        turns, and ends after the last action token of the row.

        :param rewards: Per-turn rewards ``[batch, max_turns]``.
        :type rewards: torch.Tensor
        :param values: Per-token critic values ``[batch, seq_len]``.
        :type values: torch.Tensor
        :param action_mask: Bool mask of valid action positions ``[batch, seq_len]``.
        :type action_mask: torch.Tensor
        :param turn_ids: Turn index per token ``[batch, seq_len]``; ``-1`` elsewhere.
        :type turn_ids: torch.Tensor
        :param gae_lambda: GAE lambda; ``1.0`` gives Monte Carlo returns.
        :type gae_lambda: float
        :return: Tuple of ``(token_returns, token_advantages)``, each ``[batch, seq_len]``.
        :rtype: tuple[torch.Tensor, torch.Tensor]
        """
        mask = action_mask.bool()
        batch_size, seq_len = values.shape
        positions = torch.arange(seq_len, device=values.device).expand(
            batch_size, seq_len
        )
        turn_end = torch.full(
            rewards.shape, -1, device=values.device, dtype=positions.dtype
        )
        turn_end.scatter_reduce_(
            1,
            turn_ids.clamp(min=0),
            positions.masked_fill(~mask | (turn_ids < 0), -1),
            reduce="amax",
            include_self=True,
        )
        token_rewards = torch.zeros_like(values)
        token_rewards.scatter_add_(
            1, turn_end.clamp(min=0), rewards.to(values.dtype) * (turn_end >= 0)
        )

        token_advantages = torch.zeros_like(values)
        next_value = torch.zeros(batch_size, device=values.device)
        last_gae = torch.zeros(batch_size, device=values.device)
        for t in reversed(range(seq_len)):
            delta = token_rewards[:, t] + self.gamma * next_value - values[:, t]
            gae = delta + self.gamma * gae_lambda * last_gae
            is_action = mask[:, t]
            token_advantages[:, t] = torch.where(is_action, gae, 0.0)
            next_value = torch.where(is_action, values[:, t], next_value)
            last_gae = torch.where(is_action, gae, last_gae)

        mask_f = mask.float()
        token_returns = (token_advantages + values) * mask_f
        if self.whiten_advantages:
            token_advantages = global_masked_whiten(token_advantages, action_mask)
        return token_returns, token_advantages * mask_f

    def _backward_ppo_pass(
        self, loss: torch.Tensor, accumulation_steps: int | None
    ) -> OptimizerStep | None:
        """Backward one micro-batch pass: fused actor and critic, or critic only in warmup.

        Actor and critic clip apart unless :attr:`share_grad_clip`.

        :param loss: Scaled loss of the pass.
        :type loss: torch.Tensor
        :param accumulation_steps: Micro-batches per optimizer step; ``None``
            uses :attr:`gradient_accumulation_steps`.
        :type accumulation_steps: int | None
        :return: Gradient norms of the optimizer step, or ``None`` when the
            pass only accumulates.
        :rtype: OptimizerStep | None
        """
        steps = accumulation_steps or self.gradient_accumulation_steps
        roles = () if self.share_grad_clip else ("actor", "critic")
        clip_groups = tuple(
            frozenset({role, f"{role}{REPLICATED_GROUP_SUFFIX}"}) for role in roles
        )
        return self._backward_pass(loss, steps, clip_groups or None)

    def _restore_checkpoint_attributes(
        self,
        checkpoint: dict[str, Any],
        restore_config: bool,
        restore_hyperparameters: bool,
    ) -> None:
        """Restore attributes and the warmup count.

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

    def _actor_critic_pass(
        self,
        batch_ids: torch.Tensor,
        score_mask: torch.Tensor,
        support_ids: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        image_keys: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Gradient-bearing actor log-probs and critic values of one micro-batch from one forward.

        The batch runs once per adapter in a single forward with ``lm_head``
        identity-patched, then the actor rows are scored where ``score_mask``
        is set. The routing stays set until the caller unsets it after
        ``backward()``, so gradient checkpointing recomputes each row under
        its own adapter. With packing on, the batch packs once and the packed
        row runs per adapter.

        :param batch_ids: ``(B, T)`` right-padded token ids.
        :type batch_ids: torch.Tensor
        :param score_mask: ``(B, T - 1)`` positions the lm_head scores.
        :type score_mask: torch.Tensor
        :param support_ids: ``(B, T - 1, K)`` sampling support per position, or ``None``.
        :type support_ids: torch.Tensor | None
        :param pixel_values: Vision rows of the batch, or ``None`` for text.
        :type pixel_values: torch.Tensor | None
        :param image_keys: Vision feature cache key of each vision row, or
            ``None`` to run the tower uncached.
        :type image_keys: torch.Tensor | None
        :return: ``(B, T - 1)`` actor log-probs, zero outside ``score_mask``,
            and ``(B, T - 1)`` critic values.
        :rtype: tuple[torch.Tensor, torch.Tensor]
        """
        mask = attention_mask_from_padded_ids(batch_ids, self.pad_token_id)
        fused_fn, _, _ = self._fused_logprob_fn_and_head()
        with self.sequence_layout.prepare_forward(
            batch_ids,
            mask,
            score_mask,
            support_ids,
            pixel_values=pixel_values,
            copies=2,
        ) as fwd:
            rows = fwd.kwargs["input_ids"].shape[0] // 2
            set_fused_adapter_routing(self.actor, ["actor"] * rows + ["critic"] * rows)
            with (
                self._vision_cache.images(
                    None if image_keys is None else torch.cat([image_keys, image_keys])
                ),
                self._patch_lm_head_to_identity(),
                self._amp_ctx(),
                self._activation_offload_ctx(),
            ):
                self.actor.train()
                hidden, _, values = self.actor(**fwd.kwargs)
            with self.shard_runtime.gather_layer(
                self._get_lm_head(), device=batch_ids.device
            ) as (head_w, head_b):
                log_probs = self._score_hidden(
                    fused_fn,
                    fwd.scoring_hidden(hidden[:rows]),
                    head_w,
                    head_b,
                    fwd.targets,
                    fwd.score_mask,
                    fwd.support_ids,
                    fwd.n_scored,
                )
            return fwd.restore(log_probs, "logprobs"), fwd.restore(
                values[rows:], "values"
            )

    def _critic_values(
        self,
        batch_ids: torch.Tensor,
        pixel_values: torch.Tensor | None = None,
        image_keys: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Gradient-bearing critic forward, every row routed to the critic adapter.

        ``lm_head`` is identity-patched so no logits are computed. The routing
        stays set until the caller unsets it after ``backward()``, so gradient
        checkpointing recomputes under the critic adapter.

        :param batch_ids: ``(B, T)`` right-padded token ids.
        :type batch_ids: torch.Tensor
        :param pixel_values: Vision rows of the batch, or ``None`` for text.
        :type pixel_values: torch.Tensor | None
        :param image_keys: Vision feature cache key of each vision row, or
            ``None`` to run the tower uncached.
        :type image_keys: torch.Tensor | None
        :return: ``(B, T - 1)`` per-token values.
        :rtype: torch.Tensor
        """
        mask = attention_mask_from_padded_ids(batch_ids, self.pad_token_id)
        with self.sequence_layout.prepare_forward(
            batch_ids, mask, pixel_values=pixel_values
        ) as fwd:
            set_fused_adapter_routing(self.actor, ["critic"])
            with (
                self._vision_cache.images(image_keys),
                self._patch_lm_head_to_identity(),
                self._amp_ctx(),
                self._activation_offload_ctx(),
            ):
                self.actor.train()
                _, _, values = self.actor(**fwd.kwargs)
            return fwd.restore(values, "values")

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

    def _ppo_policy_loss(
        self,
        log_probs: torch.Tensor,
        action_mask: torch.Tensor,
        old_log_probs: torch.Tensor,
        reference_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        turn_ids: torch.Tensor,
        num_turns: int,
        ppo_granularity: str,
        term_scales: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Clipped policy surrogate plus ``beta``-weighted KL from actor log-probs.

        :param log_probs: ``(B, T - 1)`` actor log-probs with grad.
        :type log_probs: torch.Tensor
        :param old_log_probs: ``(B, T - 1)`` log-probs of the policy that
            sampled each token.
        :type old_log_probs: torch.Tensor
        :param num_turns: Turn count of the whole learn batch.
        :type num_turns: int
        :param term_scales: Scales of the surrogate and the KL term
            (:meth:`_window_loss_scales`), or ``None`` for ``1``.
        :type term_scales: tuple[torch.Tensor, torch.Tensor] | None
        :return: ``(policy_loss, metrics)`` with ``metrics`` keying detached
            scalar tensors: ``kl``, ``kl_clamp_frac``, ``pg_loss``, ``clipfrac``,
            ``off_policy_token_mask_frac``, ``entropy``.
        :rtype: tuple[torch.Tensor, dict[str, torch.Tensor]]
        """
        log_probs = torch.masked_fill(log_probs, ~action_mask.bool(), 1.0)
        dropped = off_policy_token_drops(
            log_probs, old_log_probs, action_mask, self.off_policy_token_mask_bounds
        )
        kl = calculate_k3_kl(reference_log_probs, log_probs)
        penalty_reference, clamped = k3_kl_penalty_reference(
            reference_log_probs, log_probs, self.kl_clamp, dropped
        )
        kl_penalty = calculate_k3_kl(penalty_reference, log_probs)
        masked_entropy = masked_mean(-log_probs.detach(), action_mask)
        log_weights = broadcast_log_ratio_by_level(
            log_probs - old_log_probs,
            action_mask,
            turn_ids,
            self._resolve_is_level(ppo_granularity),
            num_turns,
        )
        surrogate, clipped = clipped_token_surrogate(
            log_weights,
            advantages,
            action_mask,
            1.0 - self.clip_coef,
            1.0 + self.clip_coef,
        )
        if dropped is not None:
            surrogate = surrogate.masked_fill(dropped, 0.0)
        pg_loss = masked_mean(surrogate, action_mask)
        metrics = {
            "kl": masked_mean(kl, action_mask).detach(),
            "kl_clamp_frac": masked_mean(clamped.float(), action_mask),
            "entropy": masked_entropy.mean(),
            "clipfrac": masked_mean(clipped.float(), action_mask).detach(),
            "off_policy_token_mask_frac": (
                torch.zeros((), device=log_probs.device)
                if dropped is None
                else masked_mean(dropped.float(), action_mask)
            ),
            "pg_loss": pg_loss.detach(),
        }
        kl_loss = masked_mean(kl_penalty, action_mask)
        if term_scales is not None:
            pg_loss = pg_loss * term_scales[0]
            kl_loss = kl_loss * term_scales[1]
        return pg_loss + self.beta * kl_loss, metrics

    def _ppo_value_loss(
        self,
        values: torch.Tensor,
        returns: torch.Tensor,
        action_mask: torch.Tensor,
        turn_ids: torch.Tensor,
        num_turns: int,
        ppo_granularity: str,
    ) -> torch.Tensor:
        """Value MSE on the GAE advantage axis (``ppo_granularity``).

        :param values: ``(B, T - 1)`` critic values with grad.
        :type values: torch.Tensor
        :param num_turns: Turn count of the whole learn batch.
        :type num_turns: int
        :return: Scalar value loss, weighted by ``vf_coef``.
        :rtype: torch.Tensor
        """
        values = torch.masked_fill(values, turn_ids < 0, 0.0)
        if ppo_granularity != "turn":
            return self._compute_vf_loss_token(values, returns, action_mask)
        turn_pred = turn_state_values(values, turn_ids, num_turns)
        turn_ret = turn_state_values(returns, turn_ids, num_turns)
        turn_mask = torch.zeros_like(turn_pred)
        for t in range(num_turns):
            turn_mask[:, t] = (turn_ids == t).any(dim=1).float()
        return (
            0.5
            * ((turn_ret - turn_pred).pow(2) * turn_mask).sum()
            / turn_mask.sum().clamp(min=1)
            * self.vf_coef
        )

    def _compute_vf_loss_token(
        self,
        values: torch.Tensor,
        returns: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Token-level value MSE.

        :param values: Current value predictions ``[batch, seq_len]``.
        :type values: torch.Tensor
        :param returns: Token-level returns ``[batch, seq_len]``.
        :type returns: torch.Tensor
        :param action_mask: Bool mask of valid action positions ``[batch, seq_len]``.
        :type action_mask: torch.Tensor
        :return: Scalar token-level value loss.
        :rtype: torch.Tensor
        """
        return 0.5 * masked_mean((returns - values).pow(2), action_mask) * self.vf_coef

    def _compute_token_rewards(
        self,
        action_mask: torch.Tensor,
        rewards: torch.Tensor,
        turn_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Assign each turn's reward to every token of that turn.

        :param action_mask: Bool mask of action positions ``[batch, seq_len]``.
        :type action_mask: torch.Tensor
        :param rewards: Per-turn scalars ``[batch, max_turns]``.
        :type rewards: torch.Tensor
        :param turn_ids: Turn index per generated token ``[batch, seq_len]``,
            forced tokens included; ``-1`` elsewhere.
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
