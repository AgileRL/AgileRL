# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from agilerl import HAS_LIGER_KERNEL, HAS_LLM_DEPENDENCIES
from agilerl.algorithms.configs import (
    PopulationIndex,
    REINFORCELLMObjective,
    REINFORCELLMSetup,
)
from agilerl.algorithms.core import ActionResult, LLMAlgorithm
from agilerl.algorithms.core.advantage_granularity import (
    resolve_batch_advantage_granularity,
)
from agilerl.algorithms.core.llm_init import named_llm_setup
from agilerl.algorithms.core.registry import NetworkGroup

if HAS_LIGER_KERNEL or TYPE_CHECKING:
    from agilerl.algorithms.core.llm_ops.fused_loss import (
        LigerFusedLinearPolicyLossFunction,
        apply_fused_policy_loss,
    )
else:
    # Keep the name resolvable when liger-kernel isn't installed so unit
    # tests can patch it. ``_reinforce_loss_liger`` guards against actual use.
    LigerFusedLinearPolicyLossFunction = None  # type: ignore[assignment]
    apply_fused_policy_loss = None  # type: ignore[assignment]
from agilerl.protocols import (
    PeftModelProtocol,
    PreTrainedModelProtocol,
)
from agilerl.typing import LLMObsType, LLMRolloutExperiences
from agilerl.utils.algo_utils import (
    get_experiences_samples,
    stack_and_pad_experiences,
)
from agilerl.utils.llm_utils import (
    aggregate_metrics_dict,
    attention_mask_from_padded_ids,
    build_completion_mask,
    clipped_is_surrogate,
    masked_mean,
    normalize_prompt_batch,
    pool_by_turns,
    prepare_prompt_hf_generate,
    validate_importance_sampling_level,
    validate_llm_context_lengths,
)

if HAS_LLM_DEPENDENCIES:
    from transformers import GenerationConfig


class REINFORCE(LLMAlgorithm[LLMRolloutExperiences]):
    """Turn-level REINFORCE with Return Batch Normalization (ReBN) for LLM
    finetuning.

    ReBN normalizes per-turn Monte Carlo returns across the entire batch of
    transitions. This gives per-turn credit assignment with arbitrary discount
    factors.

    Optionally uses PPO-style clipped surrogate objectives for safe multi-epoch
    updates (controlled by ``clip_coef`` and ``update_epochs``).

    :param pad_token_id: Pad token id.
    :type pad_token_id: int
    :param pad_token: Pad token string.
    :type pad_token: str
    :param model_name: Model name or path.
    :type model_name: str | None
    :param actor_network: Pre-instantiated HuggingFace model.
    :type actor_network: PreTrainedModelProtocol | None
    :param model_config: Model configuration dict.
    :type model_config: dict[str, Any] | None
    :param hp_config: RL hyperparameter mutation configuration.
    :type hp_config: HyperparameterConfig | None
    :param index: Instance index for tournament selection.
    :type index: int
    :param batch_size: Mini-batch size for learning.
    :type batch_size: int
    :param beta: KL penalty coefficient against the reference policy.
    :type beta: float
    :param clip_coef: PPO-style surrogate clipping coefficient.
    :type clip_coef: float
    :param gamma: Discount factor for multi-turn returns.
    :type gamma: float
    :param lr: Learning rate for the actor optimizer.
    :type lr: float
    :param max_grad_norm: Maximum gradient norm for clipping.
    :type max_grad_norm: float
    :param update_epochs: Number of policy update epochs per batch.
    :type update_epochs: int
    :param temperature: Sampling temperature for generation.
    :type temperature: float
    :param repetition_penalty: Repetition penalty for generation.
    :type repetition_penalty: float
    :param top_p: Top-p (nucleus) sampling parameter.
    :type top_p: float
    :param top_k: Top-k sampling parameter.
    :type top_k: int
    :param min_p: Min-p sampling parameter.
    :type min_p: float
    :param use_separate_reference_adapter: Use a dedicated LoRA adapter for
        the frozen reference policy.
    :type use_separate_reference_adapter: bool
    :param calc_position_embeddings: Calculate position embeddings explicitly.
    :type calc_position_embeddings: bool
    :param micro_batch_size_per_gpu: Micro-batch size for gradient accumulation.
    :type micro_batch_size_per_gpu: int | None
    :param mini_batch_size: Per-rank trajectories covered by one optimizer
        step; DeepSpeed's gradient_accumulation_steps is set to
        ``mini_batch_size / micro_batch_size_per_gpu``. Defaults to None,
        which resolves to ``micro_batch_size_per_gpu`` (one optimizer step
        per micro-batch).
    :type mini_batch_size: int | None, optional
    :param max_output_tokens: Maximum new tokens per generation.
    :type max_output_tokens: int | None
    :param min_output_tokens: Minimum new tokens per generation.
    :type min_output_tokens: int | None
    :param max_model_len: Maximum context window length.
    :type max_model_len: int
    :param hf_generate_chunk_size: Number of prompts per HuggingFace generation
        chunk. Ignored when ``use_vllm=True``.
    :type hf_generate_chunk_size: int | None, optional
    :param use_memory_efficient_params: For colocated vLLM, offload the trainer's
        own base to CPU during rollout (and bring it back for the training step)
        so the rollout engine and the trainer never both hold a base on the GPU.
        Defaults to True; inert without colocated vLLM, and disabled under
        DeepSpeed ZeRO-3.
    :type use_memory_efficient_params: bool
    :param lora_config: LoRA adapter configuration.
    :type lora_config: LoraConfig | None
    :param cosine_lr_schedule_config: Cosine LR schedule configuration.
    :type cosine_lr_schedule_config: CosineLRScheduleConfig | None
    :param accelerator: HuggingFace Accelerator for distributed training.
    :type accelerator: Accelerator | None
    :param device: Device to train on. Ignored when an accelerator is given (each rank
        owns its own GPU); ``None`` auto-detects CUDA/MPS/CPU.
    :type device: str
    :param wrap: Wrap models for distributed training upon creation.
    :type wrap: bool
    :param clone: Whether this is a clone instantiation.
    :type clone: bool
    :param use_vllm: Use vLLM for generation.
    :type use_vllm: bool
    :param vllm_config: vLLM configuration.
    :type vllm_config: VLLMConfig | None
    :param seed: Random seed.
    :type seed: int
    :param advantage_granularity: Policy-action granularity (ReBN advantage axis).
        ``"turn"`` enforces turn-level advantages, ``"token"`` enforces
        token-level advantages, and ``"auto"`` uses token-level only when all
        samples are single-turn.
    :type advantage_granularity: Literal["turn", "token", "auto"]
    :param action_granularity: Deprecated alias for ``advantage_granularity``;
        when set it overrides ``advantage_granularity`` and emits a
        ``DeprecationWarning``.
    :type action_granularity: str | None, optional
    :param importance_sampling_level: IS / ratio-pooling level for the clipped
        surrogate, orthogonal to ``advantage_granularity``. ``"token"`` (default)
        clips per token; ``"turn"`` pools the ratio per turn (requires
        ``turn_ids`` in :meth:`learn`); ``"trajectory"`` pools over
        the whole completion; the advantage is pooled to the same bucket.
        Turn/trajectory pooling cannot be token-chunked in the fused kernel, so
        set ``use_liger_loss=False`` there (the standard path is always
        memory-bounded).
    :type importance_sampling_level: Literal["token", "turn", "trajectory"], optional
    :param turn_ratio_pooling: Reduction used to pool per-token log-ratios into a
        per-turn ratio when ``importance_sampling_level="turn"``; ignored at
        token/trajectory level. ``"sum"`` (default) yields the product ratio per
        turn - the standard, paper-aligned per-turn importance weight. ``"mean"``
        yields a length-normalized geometric-mean ratio (GSPO-style); reach for it
        on long or highly variable-length turns, where the product ratio is far
        outside the clip band on every turn and saturates the clipped surrogate -
        length-normalizing keeps the per-turn ratio in range so the surrogate stays
        informative.
    :type turn_ratio_pooling: Literal["sum", "mean"], optional
    :param gradient_checkpointing: Enable gradient checkpointing.
    :type gradient_checkpointing: bool
    :param torch_compiler: Torch compiler mode.
    :type torch_compiler: str | None
    :param cast_logprobs_to_fp32: When ``True`` (default), run the per-token
        log-prob reduction (``gather`` / ``logsumexp``) in fp32 before casting
        back to the input dtype, for numerically stable log-probs. ``False`` runs
        it in the input dtype, saving a little memory at the cost of a per-token
        bf16 quantisation error that can bias importance-sampling ratios.
    :type cast_logprobs_to_fp32: bool, optional
    :param chunk_rows: Primary chunk-size setting for fused logit tiles. Applies to
        both standard and Liger paths.
    :type chunk_rows: int | None, optional
    :param use_liger_loss: Use the Liger fused policy loss, defaults to ``False``
        (requires ``liger-kernel``). **Recommended for REINFORCE**: via AgileRL's
        ``LigerFusedLinearPolicyLossFunction`` (the same liger-based path as PPO,
        not the upstream Liger GRPO kernel), it is roughly memory-neutral with a
        mild speedup that grows with sequence length at token-level IS. Separate
        from the Liger *model* patches (fused RMSNorm/RoPE/SwiGLU), which apply
        whenever ``liger-kernel`` is installed.
    :type use_liger_loss: bool, optional
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
        gradient forward pass. Only honoured under a FlashAttention-2 backend;
        otherwise inert.
    :type use_sequence_packing: bool, optional
    :param lora_target_scope: Optional PEFT LoRA path scope for multimodal models
        (e.g. ``"language_model"``). Passed to
        :func:`adapt_lora_config_for_model`.
    :type lora_target_scope: str | None, optional
    """

    _mini_batch_size_default = "micro_batch"

    def __init__(
        self,
        llm: REINFORCELLMSetup,
        objective: REINFORCELLMObjective | None = None,
        member: PopulationIndex | None = None,
    ) -> None:
        objective = objective or REINFORCELLMObjective()
        member = member or PopulationIndex()
        super().__init__(named_llm_setup(llm, "LLMREINFORCE"), member)
        self._bind_reinforce_llm(llm, objective)

    def _bind_reinforce_llm(
        self, llm: REINFORCELLMSetup, objective: REINFORCELLMObjective
    ) -> None:
        """Bind REINFORCE_LLM objective, generation, and actor networks."""
        train = llm.train
        gen = llm.generation
        model = llm.model
        self._validate_core_args(
            train.batch_size,
            train.lr,
            objective.clip_coef,
            objective.update_epochs,
            model.actor_network,
            train.clone,
        )
        self.beta = objective.beta
        self.clip_coef = objective.clip_coef
        self.update_epochs = objective.update_epochs
        self.temperature = gen.temperature
        self.repetition_penalty = gen.repetition_penalty
        self.top_p = gen.top_p
        self.top_k = gen.top_k
        self.min_p = gen.min_p
        self._setup_advantage_options(
            objective.advantage_granularity,
            objective.action_granularity,
            objective.gamma,
        )
        self._setup_objective(
            objective.importance_sampling_level, objective.turn_ratio_pooling
        )
        self._setup_generation(
            gen.max_output_tokens,
            gen.min_output_tokens,
            gen.max_model_len,
            gen.hf_generate_chunk_size,
        )
        self._setup_actors(model.actor_network, clone=train.clone)
        self.register_network_group(NetworkGroup(eval_network=self.actor, policy=True))
        if self.wrap:
            self.wrap_models()
        for m in ("loss", "kl", "entropy", "completion_length"):
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
        :param kwargs: Additional keyword arguments accepted for base-class
            signature compatibility. Unused in this implementation.
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
                    token_ids_list = []
                    completion_masks = []

                    for start in range(
                        0,
                        len(prompts),
                        self.hf_generate_chunk_size,
                    ):
                        chunk = prompts[start : start + self.hf_generate_chunk_size]
                        for prompt in chunk:
                            prompt = prepare_prompt_hf_generate(prompt, actor_device)
                            input_ids = prompt["input_ids"]
                            attention_mask = prompt["attention_mask"]
                            token_ids = self.actor.generate(
                                input_ids=input_ids,
                                attention_mask=attention_mask,
                                generation_config=self.generation_config,
                            )
                            token_ids_list.append(token_ids)
                            completion_masks.append(
                                build_completion_mask(
                                    token_ids,
                                    int(input_ids.shape[-1]),
                                    self.pad_token_id,
                                )
                            )
            else:
                self._prepare_vllm_for_generation()
                (
                    token_ids_list,
                    completion_masks,
                    sampling_logps,
                ) = self._generate_with_vllm_colocate(
                    # ReasoningPrompts is a TypedDict, i.e. a plain dict at
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
    ) -> dict[str, float]:
        """Update actor using REINFORCE with Return Batch Normalization.

        :param experiences: ``(token_ids, action_masks, rewards)``. For
            single-turn, ``rewards`` is a flat tensor of scalars; for multi-turn,
            shape ``[batch, max_turns]`` per-turn rewards.
        :type experiences: LLMRolloutExperiences
        :param turn_ids: Optional ``[batch, seq_len - 1]`` tensor of turn indices per
            token; ``-1`` for non-action tokens. If ``None``, all action tokens are
            treated as turn ``0``.
        :type turn_ids: torch.Tensor | None
        :param sampling_logps: Optional per-row flat vLLM sampling logprobs (one
            1-D tensor per trajectory, generated tokens only; concatenated across
            turns for multi-turn) for the vLLM sampling-mismatch correction.
            Parallel to the stacked ``token_ids`` rows. ``None`` disables
            the correction for this update.
        :type sampling_logps: list[torch.Tensor | None] | None
        :return: Dict with keys ``loss``, ``kl``, ``pg_loss``,
            ``entropy``, averaged over all minibatch updates.
        :rtype: dict[str, float]
        """
        self._prepare_vllm_for_training()

        with self.memory_efficient_params_context():
            token_ids, action_masks, rewards = stack_and_pad_experiences(
                *experiences,
                padding_values=[self.pad_token_id, False, None],
            )
            token_ids = token_ids.to(self.device)
            action_masks = action_masks.to(self.device)
            action_mask_bool = action_masks.bool()
            num_samples = token_ids.shape[0]

            if turn_ids is None:
                turn_ids = torch.where(
                    action_mask_bool,
                    torch.zeros_like(action_masks, dtype=torch.long),
                    torch.full_like(action_masks, -1, dtype=torch.long),
                )
                rewards_2d = rewards.flatten().to(self.device).float().unsqueeze(-1)
            else:
                turn_ids = turn_ids.to(self.device)
                rewards_2d = rewards.to(self.device).float()
                if rewards_2d.dim() == 1:
                    rewards_2d = rewards_2d.unsqueeze(-1)
            policy_granularity = self._resolve_advantage_granularity(turn_ids)

            del rewards

            batch_idxs = np.arange(num_samples)
            batch_size = (
                min(num_samples, self.micro_batch_size_per_gpu)
                if hasattr(self, "micro_batch_size_per_gpu")
                else num_samples
            )
            learn_metrics = {
                "loss": 0.0,
                "kl": 0.0,
                "pg_loss": 0.0,
                "entropy": 0.0,
            }
            updates = 0

            reference_log_probs, old_log_probs, _ = self._fused_forward_no_grad(
                token_ids,
                batch_size,
            )
            token_rewards = self._compute_token_rewards(
                action_masks, rewards_2d, turn_ids
            )
            old_log_probs = torch.masked_fill(old_log_probs, ~action_mask_bool, 1.0)
            reference_log_probs = torch.masked_fill(
                reference_log_probs, ~action_mask_bool, 1.0
            )
            token_penalised_rewards = token_rewards - self.beta * (
                old_log_probs - reference_log_probs
            )

            if policy_granularity == "token":
                advantages = self._compute_rebn_advantages_token(
                    token_penalised_rewards,
                    action_masks,
                )
            else:
                advantages = self._compute_rebn_advantages(
                    token_penalised_rewards, action_masks, turn_ids
                )
            del token_rewards, token_penalised_rewards

            sampling_log_probs, is_metrics = (
                self._aligned_sampling_logprobs_and_metrics(
                    sampling_logps, action_masks, old_log_probs
                )
            )

            self.actor.train()
            for _epoch_idx in range(self.update_epochs):
                self.rng.shuffle(batch_idxs)
                for start in range(0, num_samples, batch_size):
                    minibatch_idxs = batch_idxs[
                        start : min((start + batch_size), num_samples)
                    ]
                    # ``get_experiences_samples`` indexes each input
                    # positionally: Tensor in -> Tensor out, so the tuple
                    # mirrors the all-Tensor inputs.
                    (
                        batch_ids,
                        batch_action_mask,
                        batch_old_log_probs,
                        batch_reference_log_probs,
                        batch_advantages,
                        batch_turn_ids,
                    ) = get_experiences_samples(
                        minibatch_idxs,
                        token_ids,
                        action_masks,
                        old_log_probs,
                        reference_log_probs,
                        advantages,
                        turn_ids,
                    )

                    batch_mask_bool = batch_action_mask.bool()

                    # Slice the aligned vLLM sampling logprobs for this
                    # minibatch; ``None`` when the correction is off / no
                    # logprobs were captured.
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
                        and self.importance_sampling_level != "token"
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

                    if self.use_liger_loss and not liger_corr_fallback:
                        # Liger fused clipped policy loss (no (B, T, V) logits
                        # saved for backward). KL stays a logging metric;
                        # REINFORCE folds it into the advantage upstream.
                        # Works for both granularities since per-turn ReBN
                        # is already broadcast to per-token by the caller. The
                        # token-level vLLM correction is fused in via vllm_is_ratio.
                        pg_loss, metrics = self._reinforce_loss_liger(
                            batch_ids,
                            batch_action_mask,
                            batch_old_log_probs,
                            batch_reference_log_probs,
                            batch_advantages,
                            batch_turn_ids,
                            batch_sampling_log_probs,
                        )
                        self._raise_if_loss_not_finite_on_any_rank(pg_loss)
                        self._backward_pass(pg_loss)
                        learn_metrics["kl"] += metrics["kl"]
                        learn_metrics["entropy"] += metrics["entropy"]
                        learn_metrics["pg_loss"] += metrics["pg_loss"]
                        learn_metrics["loss"] += pg_loss.item()
                        updates += 1
                        continue

                    with self.select_adapter("actor"):
                        batch_log_probs = self._get_logprobs(
                            batch_ids,
                            batch_size=batch_size,
                            use_reference=False,
                            eval_mode=False,
                        )
                    batch_log_probs = torch.masked_fill(
                        batch_log_probs, ~batch_mask_bool, 1.0
                    )

                    kl = batch_log_probs - batch_reference_log_probs
                    masked_entropy = masked_mean(
                        -batch_log_probs.detach(), batch_action_mask
                    )

                    # Clipped surrogate at the configured IS / ratio-pooling
                    # level (token / turn / sequence). Pools ratio + advantage
                    # to the bucket; token reduces to the original behavior.
                    token_log_ratio = batch_log_probs - batch_old_log_probs
                    # Truncated importance sampling: reweight each token's
                    # surrogate by the (detached, clamped) trainer/vLLM ratio to
                    # correct for the rollout being drawn from vLLM rather than
                    # the trainer policy.
                    loss_weight = None
                    if batch_sampling_log_probs is not None:
                        with torch.no_grad():
                            mask_f = batch_action_mask.to(token_log_ratio.dtype)
                            loss_weight = torch.exp(
                                (batch_old_log_probs - batch_sampling_log_probs)
                                * mask_f
                            ).clamp(max=self.vllm_importance_sampling_cap)
                    pg_loss, _clipfrac = clipped_is_surrogate(
                        token_log_ratio,
                        batch_advantages,
                        batch_action_mask,
                        batch_turn_ids,
                        self.importance_sampling_level,
                        self.clip_coef,
                        loss_weight=loss_weight,
                        turn_reduction=self.turn_ratio_pooling,
                    )

                    self._raise_if_loss_not_finite_on_any_rank(pg_loss)
                    self._backward_pass(pg_loss)

                    learn_metrics["kl"] += masked_mean(kl, batch_action_mask).item()
                    learn_metrics["entropy"] += masked_entropy.item()
                    learn_metrics["pg_loss"] += pg_loss.item()
                    learn_metrics["loss"] += pg_loss.item()
                    updates += 1

        averaged = {
            metric: value / max(updates, 1) for metric, value in learn_metrics.items()
        }
        result = dict(averaged)
        # Sampling-mismatch metrics are computed once over the full batch, so
        # they bypass the per-update averaging above.
        result.update(is_metrics)

        # Wire averaged metrics into the metrics tracker; position 0 is the
        # per-trajectory completion-id batch.
        token_ids_list = experiences[0]
        completion_length = float(np.mean([c.shape[-1] for c in token_ids_list]))
        agg = aggregate_metrics_dict(
            self.accelerator,
            {
                "loss": averaged["loss"],
                "kl": averaged["kl"],
                "entropy": averaged["entropy"],
                "completion_length": completion_length,
            },
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
        actor_network: PreTrainedModelProtocol | None,
        clone: bool,
    ) -> None:
        """Validate the core training arguments."""
        assert isinstance(batch_size, int), "Batch size must be an integer."
        assert batch_size >= 1, "Batch size must be greater than or equal to one."
        assert isinstance(lr, float), "Learning rate must be a float."
        assert lr > 0, "Learning rate must be greater than zero."
        assert isinstance(clip_coef, (float, int)), (
            "Clipping coefficient must be a float."
        )
        assert clip_coef >= 0, (
            "Clipping coefficient must be greater than or equal to zero."
        )
        assert isinstance(update_epochs, int), (
            "Policy update epochs must be an integer."
        )
        assert update_epochs >= 1, (
            "Policy update epochs must be greater than or equal to one."
        )
        if clone and actor_network is not None:
            assert isinstance(
                actor_network,
                (PeftModelProtocol, PreTrainedModelProtocol),
            ), "Actor network must be a PeftModelProtocol or PreTrainedModelProtocol"

    def _setup_advantage_options(
        self,
        advantage_granularity: str,
        action_granularity: str | None,
        gamma: float,
    ) -> None:
        """Validate and store the ReBN advantage options."""
        valid_action_granularities = {"turn", "token", "auto"}
        if action_granularity is not None:
            warnings.warn(
                "action_granularity is deprecated; use advantage_granularity.",
                DeprecationWarning,
                stacklevel=3,
            )
            advantage_granularity = action_granularity
        if advantage_granularity not in valid_action_granularities:
            msg = (
                "advantage_granularity must be one of "
                f"{sorted(valid_action_granularities)}."
            )
            raise ValueError(msg)
        self.advantage_granularity = advantage_granularity
        self.gamma = gamma

    def _setup_objective(
        self,
        importance_sampling_level: str,
        turn_ratio_pooling: str,
    ) -> None:
        """Validate and resolve the importance-sampling level and Liger routing."""
        validate_importance_sampling_level(importance_sampling_level, allow_auto=False)
        if turn_ratio_pooling not in {"sum", "mean"}:
            msg = "turn_ratio_pooling must be one of ['mean', 'sum']."
            raise ValueError(msg)
        # IS / ratio-pooling level for the clipped surrogate, orthogonal to the
        # ReBN advantage granularity (``advantage_granularity``). ``"token"`` (the
        # default) preserves the original token-level clip; ``"turn"`` /
        # ``"trajectory"`` pool the ratio (length-normalized mean) per turn /
        # whole completion. Turn level requires ``turn_ids`` in ``learn``.
        self.importance_sampling_level = importance_sampling_level
        # Turn-level ratio pooling reduction (sum=product ratio, mean=geometric
        # mean ratio) used by both the standard and Liger REINFORCE losses.
        self.turn_ratio_pooling = turn_ratio_pooling
        # Warn once, up front, when Liger is paired with a non-token IS level.
        # It is permitted but not memory-bounded: turn-/trajectory-level pooling
        # couples a unit's tokens, so the fused kernel processes one whole
        # sequence per chunk and materializes a (seq_len, vocab) logits tensor
        # per trajectory. The shared once-flag suppresses the loss-time dup.
        if self.use_liger_loss and self.importance_sampling_level in {
            "turn",
            "trajectory",
        }:
            self._warn_liger_non_token_is(
                self.importance_sampling_level,
                "REINFORCE",
                once_attr="_reinforce_liger_mem_warned",
            )

    def _setup_generation(
        self,
        max_output_tokens: int | None,
        min_output_tokens: int | None,
        max_model_len: int,
        hf_generate_chunk_size: int | None,
    ) -> None:
        """Validate context lengths and build the HF generation config."""
        self.max_output_tokens = (
            max_output_tokens if max_output_tokens is not None else max_model_len
        )
        self.min_output_tokens = min_output_tokens
        self.max_model_len = max_model_len
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

    def _reinforce_loss_liger(
        self,
        batch_ids: torch.Tensor,
        batch_action_mask: torch.Tensor,
        batch_old_log_probs: torch.Tensor,
        batch_reference_log_probs: torch.Tensor,
        batch_advantages: torch.Tensor,
        turn_ids: torch.Tensor | None = None,
        sampling_log_probs: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """REINFORCE clipped policy loss via the fused-linear PPO Function.

        Captures the last hidden state via an ``lm_head`` forward-pre-hook
        (so the full ``(B, T, V)`` logits never need to be saved for
        backward) and invokes :class:`LigerFusedLinearPolicyLossFunction` with
        ``beta=0`` - REINFORCE folds the KL penalty into the advantage
        during the no-grad pass, so the gradient-time loss is pure clipped
        policy gradient. KL is still computed inside the kernel and
        returned as a logging metric.

        :param batch_ids: ``(B, seq_len)`` token IDs for this minibatch.
        :type batch_ids: torch.Tensor
        :param batch_action_mask: ``(B, seq_len-1)`` bool mask of valid
            action positions.
        :type batch_action_mask: torch.Tensor
        :param batch_old_log_probs: ``(B, seq_len-1)`` old-policy logprobs.
        :type batch_old_log_probs: torch.Tensor
        :param batch_reference_log_probs: ``(B, seq_len-1)`` reference
            logprobs (used for the KL metric only).
        :type batch_reference_log_probs: torch.Tensor
        :param batch_advantages: ``(B, seq_len-1)`` per-token advantages.
        :type batch_advantages: torch.Tensor
        :return: ``(pg_loss, metrics)`` where ``metrics`` carries
            ``kl``, ``pg_loss``, ``entropy``, ``clipfrac`` Python floats.
        :rtype: tuple[torch.Tensor, dict[str, float]]
        """
        if not HAS_LIGER_KERNEL:
            msg = (
                "Liger REINFORCE loss was requested but `liger-kernel` is not "
                "available. Set use_liger_loss=False."
            )
            raise ImportError(msg)

        is_level = self.importance_sampling_level
        if is_level != "token":
            self._warn_liger_non_token_is(
                is_level, "REINFORCE", once_attr="_reinforce_liger_mem_warned"
            )

        batch_ids = batch_ids.to(self.device)
        mask = batch_action_mask.to(self.device).contiguous()
        old_log_probs = batch_old_log_probs.to(self.device).contiguous()
        ref_log_probs = batch_reference_log_probs.to(self.device).contiguous()
        advantages = batch_advantages.to(self.device).contiguous()

        # Pool advantages to match the ratio bucket the fused kernel produces
        # (``is_level`` resolved above).
        turn_ids_arg: torch.Tensor | None = None
        full_turn_mask: torch.Tensor | None = None
        max_turns: int | None = None
        if is_level == "turn":
            if turn_ids is None:
                msg = "importance_sampling_level='turn' requires turn_ids."
                raise ValueError(msg)
            turn_ids = turn_ids.to(self.device)
            max_turns = int(turn_ids.max().item()) + 1
            full_turn_mask = torch.zeros(
                turn_ids.shape[0], max_turns, device=self.device
            )
            for t in range(max_turns):
                full_turn_mask[:, t] = (turn_ids == t).any(dim=1).float()
            advantages = pool_by_turns(advantages, turn_ids, max_turns).contiguous()
            turn_ids_arg = turn_ids
        elif is_level == "trajectory":
            mask_f = mask.to(advantages.dtype)
            advantages = (
                (advantages * mask_f).sum(dim=-1, keepdim=True)
                / mask_f.sum(dim=-1, keepdim=True).clamp(min=1.0)
            ).contiguous()  # (B, 1)

        # Truncated importance sampling fused into the kernel (token level only):
        # reweight each token's policy loss by the detached, clamped trainer/vLLM
        # ratio. Non-token IS routes the correction to the standard path.
        vllm_is_ratio = None
        if sampling_log_probs is not None and is_level == "token":
            with torch.no_grad():
                ratio_mask = mask.to(old_log_probs.dtype)
                vllm_is_ratio = torch.exp(
                    (old_log_probs - sampling_log_probs.to(self.device)) * ratio_mask
                ).clamp(max=self.vllm_importance_sampling_cap)

        # Identity-patch lm_head so the actor forward outputs the last hidden
        # state (B, T, H) directly instead of computing the full (B, T, V)
        # logits only to discard them. lm_head_weight is passed separately to
        # LigerFusedLinearPolicyLossFunction which handles the matmul and its grad.
        lm_head = self._get_lm_head()
        lm_head_weight = lm_head.weight
        lm_head_bias = lm_head.bias

        attention_mask = attention_mask_from_padded_ids(
            batch_ids, self.pad_token_id
        ).long()
        kwargs: dict[str, Any] = {
            "input_ids": batch_ids,
            "attention_mask": attention_mask,
            "use_cache": False,
        }
        if self.calc_position_embeddings:
            kwargs["position_ids"] = self._position_ids_from_mask(attention_mask)

        with (
            self._patch_lm_head_to_identity(),
            self.select_adapter("actor"),
            self._amp_ctx(),
        ):
            self.actor.train()
            actor_output = self.actor(**kwargs)
        policy_hidden = (
            actor_output[0] if isinstance(actor_output, tuple) else actor_output.logits
        )  # (B, T, H)
        target_ids = batch_ids[:, 1:].contiguous()  # (B, T-1)
        # Hidden states are aligned with target ids: predict ids[:, 1:] from
        # hidden[:, :-1]. Token level token-flattens the hidden states so the
        # fused kernel chunks tokens (bounded); turn/sequence keep the batch
        # path. beta=0: KL handled upstream via the ReBN advantage.
        with self._liger_head_gather():
            loss, aux = apply_fused_policy_loss(
                policy_hidden[:, :-1],
                lm_head_weight,
                lm_head_bias,
                target_ids,
                mask,
                advantages,
                ref_log_probs,
                old_log_probs,
                0.0,  # beta
                self.clip_coef,  # epsilon_low
                self.clip_coef,  # epsilon_high
                self.temperature,
                is_level,
                turn_ids=turn_ids_arg,
                full_turn_mask=full_turn_mask,
                max_turns=max_turns,
                token_chunk_size=self._resolve_fused_chunk_rows(
                    getattr(lm_head_weight, "ds_shape", lm_head_weight.shape)[0],
                    self.chunk_rows,
                ),
                turn_log_ratio_reduction=self.turn_ratio_pooling,
                vllm_is_ratio=vllm_is_ratio,
            )
        # aux = [kl, clipfrac, pg_loss, entropy] scalars in fp32.
        metrics = {
            "kl": float(aux[0].item()),
            "clipfrac": float(aux[1].item()),
            "pg_loss": float(aux[2].item()),
            "entropy": float(aux[3].item()),
        }
        return loss, metrics

    def _resolve_advantage_granularity(self, turn_ids: torch.Tensor) -> str:
        """Resolve effective policy granularity for the current batch.

        :param turn_ids: Turn index per token ``[batch, seq_len]``; ``-1`` for padding.
        :type turn_ids: torch.Tensor
        :return: Effective policy granularity.
        :rtype: str
        """
        return resolve_batch_advantage_granularity(
            self.advantage_granularity,
            turn_ids,
            single_turn="token",
            multi_turn="turn",
        )

    def _compute_rebn_advantages(
        self,
        rewards: torch.Tensor,
        action_mask: torch.Tensor,
        turn_ids: torch.Tensor,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        """Compute Return Batch Normalization (ReBN) advantages.

        For each turn, computes the discounted Monte Carlo return G_t, then
        z-scores all per-turn returns across the batch to produce advantages.
        Advantages are broadcast back to token level for the policy gradient.

        :param rewards: Per-token rewards ``[batch, seq_len]`` (from
            :meth:`_compute_token_rewards`).
        :type rewards: torch.Tensor
        :param action_mask: Mask of action positions ``[batch, seq_len]``.
        :type action_mask: torch.Tensor
        :param turn_ids: Turn index per token ``[batch, seq_len]``; ``-1`` for padding.
        :type turn_ids: torch.Tensor
        :param eps: Small constant added to the standard deviation when z-scoring.
        :type eps: float
        :return: Token-level advantages ``[batch, seq_len]``.
        :rtype: torch.Tensor
        """
        batch_size = rewards.shape[0]
        num_turns = int(turn_ids.max().item()) + 1

        turn_rewards = pool_by_turns(rewards, turn_ids, num_turns)

        per_sample_num_turns = turn_ids.max(dim=1).values + 1

        # Compute Monte Carlo returns: G_t = r_t + gamma * G_{t+1}
        turn_returns = torch.zeros_like(turn_rewards)
        for t in reversed(range(num_turns)):
            is_last_turn = t >= (per_sample_num_turns - 1)
            if t == num_turns - 1:
                next_return = torch.zeros(batch_size, device=rewards.device)
            else:
                next_return = turn_returns[:, t + 1]
            next_return = torch.where(
                is_last_turn, torch.zeros_like(next_return), next_return
            )
            turn_returns[:, t] = turn_rewards[:, t] + self.gamma * next_return

        # ReBN: z-score returns across all valid (sample, turn) pairs
        valid_mask = torch.zeros_like(turn_returns, dtype=torch.bool)
        for t in range(num_turns):
            valid_mask[:, t] = per_sample_num_turns > t

        valid_returns = turn_returns[valid_mask]
        if valid_returns.numel() > 1:
            mean_g = valid_returns.mean()
            std_g = valid_returns.std() + eps
            normalized_returns = (turn_returns - mean_g) / std_g
        else:
            normalized_returns = torch.zeros_like(turn_returns)

        # Broadcast turn-level advantages to token level
        token_advantages = torch.zeros_like(rewards)
        for t in range(num_turns):
            mask_t = (turn_ids == t).float()
            token_advantages += mask_t * normalized_returns[:, t : t + 1]

        return token_advantages * action_mask

    def _compute_rebn_advantages_token(
        self,
        rewards: torch.Tensor,
        action_mask: torch.Tensor,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        """Compute token-level ReBN advantages.

        This computes discounted Monte Carlo returns over token positions, then
        z-scores all valid token returns across the batch.

        :param rewards: Per-token rewards ``[batch, seq_len]``.
        :type rewards: torch.Tensor
        :param action_mask: Mask of action positions ``[batch, seq_len]``.
        :type action_mask: torch.Tensor
        :param eps: Small constant added to standard deviation when z-scoring.
        :type eps: float
        :return: Token-level advantages ``[batch, seq_len]``.
        :rtype: torch.Tensor
        """
        mask = action_mask.float()
        batch_size, seq_len = rewards.shape
        token_returns = torch.zeros_like(rewards)
        next_return = torch.zeros(batch_size, device=rewards.device)

        for t in reversed(range(seq_len)):
            if t == seq_len - 1:
                next_mask = torch.zeros(batch_size, device=rewards.device)
            else:
                next_mask = mask[:, t + 1]
            next_return = rewards[:, t] + self.gamma * next_return * next_mask
            token_returns[:, t] = next_return * mask[:, t]

        valid_returns = token_returns[action_mask.bool()]
        if valid_returns.numel() > 1:
            mean_g = valid_returns.mean()
            std_g = valid_returns.std() + eps
            normalized_returns = (token_returns - mean_g) / std_g
        else:
            normalized_returns = torch.zeros_like(token_returns)

        return normalized_returns * mask

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
