# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import warnings
from contextlib import nullcontext
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import numpy.typing as npt
import torch
from accelerate import Accelerator

from agilerl import HAS_LIGER_KERNEL, HAS_LLM_DEPENDENCIES
from agilerl.algorithms.core import ActionResult, LLMAlgorithm
from agilerl.algorithms.core.llm_ops.fused_lora import unset_fused_adapter_routing
from agilerl.algorithms.core.registry import HyperparameterConfig, NetworkGroup
from agilerl.llm_envs import ReasoningGym

if TYPE_CHECKING:
    from peft import LoraConfig

if HAS_LIGER_KERNEL or TYPE_CHECKING:
    from agilerl.algorithms.core.llm_ops.fused_loss import (
        LigerFusedLinearPolicyLossFunction,
        apply_fused_policy_loss,
    )
else:
    # Keep the name resolvable when liger-kernel isn't installed so unit
    # tests can patch it. ``_ppo_loss_liger`` guards against actual use.
    LigerFusedLinearPolicyLossFunction = None  # type: ignore[assignment]
    apply_fused_policy_loss = None  # type: ignore[assignment]
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
from agilerl.utils.llm_utils import (
    BitsAndBytesConfig,
    aggregate_metrics_dict,
    attention_mask_from_padded_ids,
    build_completion_mask,
    calculate_k3_kl,
    clipped_is_surrogate,
    is_reasoning_prompts,
    masked_mean,
    masked_whiten,
    normalize_reasoning_prompt_batch,
    pool_by_turns,
    prepare_prompt_hf_generate,
    resolve_llm_device,
    stitch_completion_after_windowed_hf_generate,
    validate_importance_sampling_level,
    validate_llm_context_lengths,
)

if HAS_LLM_DEPENDENCIES:
    from transformers import GenerationConfig


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
    :type actor_network: Any | None, optional
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
    :param lr_actor: Actor learning rate.
    :type lr_actor: float, optional
    :param lr_critic: Critic/value-head learning rate. If ``None``, ``lr_actor`` is used.
    :type lr_critic: float | None, optional
    :param max_grad_norm: Gradient clipping norm.
    :type max_grad_norm: float, optional
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
    :param max_output_tokens: Maximum newly generated tokens per completion.
    :type max_output_tokens: int | None, optional
    :param min_output_tokens: Minimum newly generated tokens per completion.
    :type min_output_tokens: int | None, optional
    :param max_model_len: Maximum model context length.
    :type max_model_len: int, optional
    :param hf_generate_chunk_size: Number of prompts per HuggingFace generation chunk.
        Ignored when ``use_vllm=True``.
    :type hf_generate_chunk_size: int | None, optional
    :param lora_config: LoRA configuration.
    :type lora_config: LoraConfig | None, optional
    :param cosine_lr_schedule_config: Cosine LR scheduler configuration.
    :type cosine_lr_schedule_config: CosineLRScheduleConfig | None, optional
    :param accelerator: Optional HuggingFace ``Accelerator`` instance.
    :type accelerator: Accelerator | None, optional
    :param device: Device to train on. Ignored when an accelerator is given (each rank
        owns its own GPU); ``None`` auto-detects CUDA/MPS/CPU.
    :type device: str, optional
    :param wrap: Whether to wrap models for distributed execution.
    :type wrap: bool, optional
    :param clone: Whether this instance is being created as a clone.
    :type clone: bool, optional
    :param use_vllm: Whether to route generation through vLLM.
    :type use_vllm: bool, optional
    :param use_memory_efficient_params: For colocated vLLM, offload the trainer's
        own base to CPU during rollout (and bring it back for the training step)
        so the rollout engine and the trainer never both hold a base on the GPU.
        Defaults to True; inert without colocated vLLM, and disabled under
        DeepSpeed ZeRO-3.
    :type use_memory_efficient_params: bool, optional
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
        on long or highly variable-length turns, where the product ratio lands far
        outside the clip band on every turn and saturates the clipped surrogate —
        length-normalizing keeps the per-turn ratio in range so the surrogate stays
        informative.
    :type turn_ratio_pooling: Literal["sum", "mean"], optional
    :param action_granularity: Deprecated alias for ``advantage_granularity``;
        when set it overrides ``advantage_granularity`` and emits a
        ``DeprecationWarning``.
    :type action_granularity: str | None, optional
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
    :param reduce_memory_peak: Deprecated and ignored; previously hinted
        peak-memory batching. Configure ``micro_batch_size_per_gpu`` instead.
    :type reduce_memory_peak: bool, optional
    :param cast_logprobs_to_fp32: When ``True`` (default), run the per-token
        log-prob reduction (``gather`` / ``logsumexp``) in fp32 before casting
        back to the input dtype, for numerically stable log-probs. ``False`` runs
        it in the input dtype, saving a little memory at the cost of a per-token
        bf16 quantisation error that can bias importance-sampling ratios.
    :type cast_logprobs_to_fp32: bool, optional
    :param chunk_rows: Primary chunk-size knob for fused logit tiles. Applies
        to both standard and Liger paths.
    :type chunk_rows: int | None, optional
    :param use_liger_loss: Use the Liger fused policy loss, defaults to ``False``
        (requires ``liger-kernel``). **Recommended for PPO**: via AgileRL's
        ``LigerFusedLinearPolicyLossFunction`` (not the upstream Liger GRPO
        kernel), it is roughly memory-neutral with a mild speedup that grows with
        sequence length (~1.1x at long sequences) at token-level IS. Separate
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
        gradient forward (actor and critic share ids and pack into one varlen /
        blockmask pass). Only honoured under a FlashAttention-2 / FlexAttention
        backend, otherwise inert; the no-grad reference/old-value pass stays
        padded.
    :type use_sequence_packing: bool, optional
    :param lora_target_scope: Optional PEFT LoRA path scope for multimodal models
        (e.g. ``"language_model"``). Passed to
        :func:`adapt_lora_config_for_model`.
    :type lora_target_scope: str | None, optional
    """

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
        beta: float = 0.01,
        vf_coef: float = 0.5,
        clip_coef: float = 0.2,
        gamma: float = 1.0,
        gae_lambda: float = 1.0,
        lr_actor: float = 5e-7,
        lr_critic: float | None = 5e-5,
        max_grad_norm: float = 1.0,
        update_epochs: int = 1,
        temperature: float = 1.0,
        repetition_penalty: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 50,
        min_p: float = 0.0,
        use_separate_reference_adapter: bool = True,
        calc_position_embeddings: bool = True,
        micro_batch_size_per_gpu: int | None = None,
        max_output_tokens: int | None = None,
        min_output_tokens: int | None = None,
        max_model_len: int = 1024,
        hf_generate_chunk_size: int | None = None,
        lora_config: LoraConfig | None = None,
        cosine_lr_schedule_config: CosineLRScheduleConfig | None = None,
        accelerator: Accelerator | None = None,
        device: str | torch.device | None = None,
        wrap: bool = True,
        clone: bool = False,
        use_vllm: bool = False,
        use_memory_efficient_params: bool = True,
        vllm_config: VLLMConfig | None = None,
        seed: int = 42,
        turn_level_clip: bool = True,
        importance_sampling_level: Literal[
            "auto", "token", "turn", "trajectory"
        ] = "auto",
        advantage_granularity: Literal["turn", "token", "auto"] = "auto",
        turn_ratio_pooling: Literal["sum", "mean"] = "sum",
        action_granularity: Literal["turn", "token", "auto"] | None = None,
        turn_value_reduction: Literal["mean", "final_value"] = "final_value",
        whiten_advantages: bool = True,
        gradient_checkpointing: bool = True,
        torch_compiler: str | None = None,
        reduce_memory_peak: bool = False,
        cast_logprobs_to_fp32: bool = True,
        chunk_rows: int | None = None,
        use_liger_loss: bool = False,
        quantization_config: BitsAndBytesConfig | None = None,
        activation_offload: bool = False,
        use_sequence_packing: bool = False,
        lora_target_scope: str | None = None,
        vllm_importance_sampling_correction: bool = True,
        vllm_importance_sampling_cap: float = 2.0,
    ) -> None:

        resolved_device = resolve_llm_device(accelerator, device)
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
            use_vllm=use_vllm,
            vllm_config=vllm_config,
            use_liger_loss=use_liger_loss,
            lora_config=lora_config,
            use_separate_reference_adapter=use_separate_reference_adapter,
            model_name=model_name,
            actor_network=actor_network,
            model_config=model_config,
            micro_batch_size_per_gpu=micro_batch_size_per_gpu,
            cosine_lr_schedule_config=cosine_lr_schedule_config,
            hp_config=hp_config,
            use_memory_efficient_params=use_memory_efficient_params,
            wrap=wrap,
            device=resolved_device,
            accelerator=accelerator,
            name="LLMPPO",
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
        self.temperature = temperature
        self.repetition_penalty = repetition_penalty
        self.top_p = top_p
        self.top_k = top_k
        self.min_p = min_p
        self._setup_advantage_options(
            turn_level_clip,
            advantage_granularity,
            action_granularity,
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

        # Register network groups for mutations
        self.register_network_group(NetworkGroup(eval_network=self.actor, policy=True))
        if self.wrap:
            self.wrap_models()

        # Register algorithm metrics
        for m in (
            "loss",
            "pg_loss",
            "vf_loss",
            "kl",
            "entropy",
            "clipfrac",
            "completion_length",
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
        prompt_batch = normalize_reasoning_prompt_batch(obs)
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
                    # ReasoningPrompts is a TypedDict, i.e. a plain dict at
                    # runtime; the base helper takes untyped prompt dicts.
                    prompt_batch,
                    1,
                    temperature=self.temperature
                    if training
                    else 0.01,  # Almost deterministic for evaluation
                    capture_sampling_logps=capture_sampling_logps,
                )

        return ActionResult(completion_ids, completion_masks, sampling_logps)

    if TYPE_CHECKING:
        # PPO always trains with a value head, so the fused forward always
        # returns critic values; narrow the base ``Tensor | None`` here.
        def _fused_forward(
            self,
            ids: torch.Tensor,
            batch_size: int,
            attention_mask: torch.Tensor | None = None,
        ) -> tuple[torch.Tensor, torch.Tensor]: ...

    def learn(
        self,
        experiences: LLMRolloutExperiences,
        turn_ids: torch.Tensor | None = None,
        sampling_logps: list[torch.Tensor | None] | None = None,
    ) -> dict[str, float]:
        """Update actor and critic adapters using configured PPO granularity.

        :param experiences: ``(completion_ids, action_masks, rewards)``. For
            single-turn, ``rewards`` is a flat tensor of scalars; for multi-turn,
            shape ``[batch, max_turns]`` per-turn rewards.
        :type experiences: LLMRolloutExperiences
        :param turn_ids: Optional ``[batch, seq_len - 1]`` tensor of turn indices;
            ``-1`` for non-action tokens. If ``None``, all action tokens are turn ``0``.
        :type turn_ids: torch.Tensor | None
        :param sampling_logps: Optional per-row flat vLLM sampling logprobs (one
            1-D tensor per trajectory, generated tokens only; concatenated across
            turns for multi-turn) for the vLLM sampling-mismatch correction.
            Parallel to the stacked ``completion_ids`` rows. ``None`` disables
            the correction for this update.
        :type sampling_logps: list[torch.Tensor | None] | None
        :return: Mean training metrics across PPO minibatch updates.
        :rtype: dict[str, float]
        """
        self._prepare_vllm_for_training()
        with self.memory_efficient_params_context():
            completion_ids, action_masks, rewards = stack_and_pad_experiences(
                *experiences,
                padding_values=[self.pad_token_id, False, None],
            )
            completion_ids = completion_ids.to(self.device)
            action_masks = action_masks.to(self.device)
            action_mask_bool = action_masks.bool()
            num_samples = completion_ids.shape[0]

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
            ppo_granularity = self._resolve_advantage_granularity(turn_ids)

            del rewards

            batch_idxs = np.arange(num_samples)
            batch_size = (
                min(num_samples, self.micro_batch_size_per_gpu)
                if hasattr(self, "micro_batch_size_per_gpu")
                else num_samples
            )
            updates = 0
            learn_metrics = {
                "loss": 0.0,
                "pg_loss": 0.0,
                "vf_loss": 0.0,
                "kl": 0.0,
                "entropy": 0.0,
                "clipfrac": 0.0,
            }
            reference_log_probs, old_log_probs, old_values = (
                self._fused_forward_no_grad(
                    completion_ids,
                    batch_size=batch_size,
                )
            )
            # PPO always trains with a value head, so critic values are present.
            assert old_values is not None
            old_values = torch.masked_fill(old_values, ~action_mask_bool, 0.0)

            token_rewards = self._compute_token_rewards(
                action_masks, rewards_2d, turn_ids
            )

            old_log_probs = torch.masked_fill(old_log_probs, ~action_mask_bool, 1.0)
            reference_log_probs = torch.masked_fill(
                reference_log_probs, ~action_mask_bool, 1.0
            )
            if ppo_granularity == "token":
                returns, advantages = self._compute_gae_returns_token(
                    token_rewards,
                    old_values,
                    action_masks,
                )
            else:
                returns, advantages = self._compute_gae_returns(
                    token_rewards, old_values, action_masks, turn_ids
                )
            del token_rewards

            # The reweight applies only to the policy surrogate.
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
                        batch_returns,
                        batch_advantages,
                        batch_old_values,
                        batch_turn_ids,
                    ) = get_experiences_samples(
                        minibatch_idxs,
                        completion_ids,
                        action_masks,
                        old_log_probs,
                        reference_log_probs,
                        returns,
                        advantages,
                        old_values,
                        turn_ids,
                    )

                    batch_mask_bool = batch_action_mask.bool()

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

                    if self.use_liger_loss and not liger_corr_fallback:
                        # Liger fused policy + KL (no (B, T, V) logits saved
                        # for backward) plus an unfused critic pass for the
                        # value loss. The token-level vLLM correction is fused
                        # in via ``vllm_is_ratio``. See :meth:`_ppo_loss_liger`.
                        total_loss, metrics = self._ppo_loss_liger(
                            batch_ids,
                            batch_action_mask,
                            batch_old_log_probs,
                            batch_reference_log_probs,
                            batch_returns,
                            batch_advantages,
                            batch_old_values,
                            batch_turn_ids,
                            ppo_granularity,
                            batch_sampling_log_probs,
                        )
                        self._raise_if_loss_not_finite_on_any_rank(total_loss)
                        self._backward_pass(total_loss)
                        unset_fused_adapter_routing(self._get_unwrapped_actor())
                        learn_metrics["kl"] += metrics["kl"]
                        learn_metrics["entropy"] += metrics["entropy"]
                        learn_metrics["clipfrac"] += metrics["clipfrac"]
                        learn_metrics["pg_loss"] += metrics["pg_loss"]
                        learn_metrics["vf_loss"] += metrics["vf_loss"]
                        learn_metrics["loss"] += total_loss.item()
                        updates += 1
                        continue

                    # Fused forward: actor logprobs + critic values in one pass.
                    batch_log_probs, batch_values = self._fused_forward(
                        batch_ids,
                        batch_size=batch_size,
                    )
                    batch_log_probs = torch.masked_fill(
                        batch_log_probs, ~batch_mask_bool, 1.0
                    )
                    kl = calculate_k3_kl(batch_log_probs, batch_reference_log_probs)
                    masked_entropy = masked_mean(
                        -batch_log_probs.detach(), batch_action_mask
                    )

                    batch_values = torch.masked_fill(
                        batch_values, ~batch_mask_bool, 0.0
                    )
                    token_log_ratio = batch_log_probs - batch_old_log_probs
                    is_level = self._resolve_is_level(ppo_granularity)

                    # Truncated importance sampling: reweight each token's
                    # surrogate by the (detached, clamped) trainer/vLLM ratio to
                    # correct for the rollout being drawn from vLLM rather than
                    # the trainer policy. Applies only to the policy surrogate,
                    # not the value/KL terms.
                    loss_weight = None
                    if batch_sampling_log_probs is not None:
                        with torch.no_grad():
                            mask_f = batch_action_mask.to(token_log_ratio.dtype)
                            loss_weight = torch.exp(
                                (batch_old_log_probs - batch_sampling_log_probs)
                                * mask_f
                            ).clamp(max=self.vllm_importance_sampling_cap)

                    pg_loss, clipfrac = self._ppo_policy_surrogate(
                        token_log_ratio,
                        batch_advantages,
                        batch_action_mask,
                        batch_turn_ids,
                        is_level,
                        loss_weight=loss_weight,
                    )

                    # Value loss granularity follows the GAE advantage axis
                    # (``ppo_granularity``), independent of the IS level.
                    if ppo_granularity == "turn":
                        mb_num_turns = int(batch_turn_ids.max().item()) + 1
                        turn_pred = pool_by_turns(
                            batch_values,
                            batch_turn_ids,
                            mb_num_turns,
                            reduction=self.turn_value_reduction,
                        )
                        turn_old = pool_by_turns(
                            batch_old_values,
                            batch_turn_ids,
                            mb_num_turns,
                            reduction=self.turn_value_reduction,
                        )
                        turn_ret = pool_by_turns(
                            batch_returns, batch_turn_ids, mb_num_turns
                        )
                        turn_mask = torch.zeros_like(turn_pred)
                        for t in range(mb_num_turns):
                            turn_mask[:, t] = (batch_turn_ids == t).any(dim=1).float()

                        vf_loss = (turn_ret - turn_pred).pow(2)
                        clipped_turn_values = turn_old + torch.clamp(
                            turn_pred - turn_old, -self.clip_coef, self.clip_coef
                        )
                        clipped_vf_loss = (turn_ret - clipped_turn_values).pow(2)
                        vf_loss = (
                            0.5
                            * (torch.max(vf_loss, clipped_vf_loss) * turn_mask).sum()
                            / turn_mask.sum().clamp(min=1)
                            * self.vf_coef
                        )
                    else:
                        vf_loss = self._compute_vf_loss_token(
                            batch_values,
                            batch_old_values,
                            batch_returns,
                            batch_action_mask,
                        )

                    kl_loss = masked_mean(kl, batch_action_mask)
                    total_loss = pg_loss + vf_loss + self.beta * kl_loss

                    self._raise_if_loss_not_finite_on_any_rank(total_loss)
                    self._backward_pass(total_loss)
                    unset_fused_adapter_routing(self._get_unwrapped_actor())

                    learn_metrics["kl"] += kl_loss.item()
                    learn_metrics["entropy"] += masked_entropy.mean().item()
                    learn_metrics["clipfrac"] += clipfrac.item()
                    learn_metrics["pg_loss"] += pg_loss.mean().item()
                    learn_metrics["vf_loss"] += vf_loss.mean().item()
                    learn_metrics["loss"] += total_loss.item()
                    updates += 1

        averaged = {
            metric: value / max(updates, 1) for metric, value in learn_metrics.items()
        }
        result = dict(averaged)
        # Sampling-mismatch metrics are computed once over the full batch, so
        # they bypass the per-update averaging above.
        result.update(is_metrics)

        # Wire averaged metrics into the metrics tracker.
        completion_list = experiences[0]
        completion_length = np.mean([c.shape[-1] for c in completion_list])
        agg = aggregate_metrics_dict(
            self.accelerator,
            {
                "loss": averaged["loss"],
                "pg_loss": averaged["pg_loss"],
                "vf_loss": averaged["vf_loss"],
                "kl": averaged["kl"],
                "entropy": averaged["entropy"],
                "clipfrac": averaged["clipfrac"],
                "completion_length": completion_length,
            },
        )
        agg["completion_length"] = int(agg["completion_length"])
        for key, value in agg.items():
            self.metrics.log(key, value)

        return result

    def test(
        self,
        env: ReasoningGym | TokenizedMultiTurnEnv,
        loop: int = 1,
        *args: Any,
        **kwargs: Any,
    ) -> npt.NDArray:
        """Return fitness (test) score of llm on test sub-set.

        ``ReasoningGym`` (and compatible dataset envs): ``reset`` returns a batch
        of prompt dicts; each ``step`` accepts completion id tensors and returns
        the next batch plus rewards. ``loop`` iterations advance the test
        dataloader that many times.

        :param env: A :class:`~agilerl.utils.llm_utils.ReasoningGym` or
            :class:`~agilerl.llm_envs.TokenObservationWrapper`.
        :type env: ReasoningGym | TokenizedMultiTurnEnv
        :param loop: Number of outer test iterations (dataloader passes or episodes).
        :type loop: int
        :return: Mean reward from the test loop (scalar numpy array).
        :rtype: npt.NDArray
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
        mean_fit = torch.mean(reward_tensor.float()).item()
        self.metrics.add_fitness(mean_fit)
        if self.accelerator is not None:
            self.accelerator.wait_for_everyone()
        return np.array(mean_fit)

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
        assert isinstance(lr, float), "Actor learning rate must be a float."
        assert lr > 0, "Actor learning rate must be greater than zero."
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
        turn_level_clip: bool,
        advantage_granularity: str,
        action_granularity: str | None,
        turn_value_reduction: str,
        whiten_advantages: bool,
        gamma: float,
        gae_lambda: float,
    ) -> None:
        """Validate and store the GAE advantage options."""
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
        valid_turn_value_reductions = {"mean", "final_value"}
        if turn_value_reduction not in valid_turn_value_reductions:
            msg = (
                "turn_value_reduction must be one of "
                f"{sorted(valid_turn_value_reductions)}."
            )
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
        # GAE advantage granularity. ``"auto"`` preserves legacy behavior:
        # turn-level when the batch is multi-turn and ``turn_level_clip`` is set,
        # else token-level. Explicit ``token``/``turn``/``trajectory`` override
        # (length-normalized mean pooling, consistent with GRPO/GSPO).
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

    def _resolve_advantage_granularity(self, turn_ids: torch.Tensor) -> str:
        """Resolve effective PPO granularity for the current batch.

        :param turn_ids: Turn index per token ``[batch, seq_len]``; ``-1`` for padding.
        :type turn_ids: torch.Tensor
        :return: Effective PPO granularity.
        :rtype: str
        """
        if self.advantage_granularity in {"turn", "token"}:
            return self.advantage_granularity

        per_sample_num_turns = turn_ids.max(dim=1).values + 1
        all_single_turn = bool((per_sample_num_turns <= 1).all())
        return "token" if all_single_turn else "turn"

    def _resolve_is_level(self, ppo_granularity: str) -> str:
        """Resolve the effective importance-sampling (ratio-pooling) level.

        ``"auto"`` reproduces the legacy ``turn_level_clip`` behavior: pool at
        the GAE granularity (``ppo_granularity``) when ``turn_level_clip`` is
        set, else token-level. Explicit ``token``/``turn``/``trajectory`` override.

        :param ppo_granularity: Resolved GAE advantage granularity (token/turn).
        :type ppo_granularity: str
        :return: One of ``"token"``, ``"turn"``, ``"trajectory"``.
        :rtype: str
        """
        if self.importance_sampling_level == "auto":
            return ppo_granularity if self.turn_level_clip else "token"
        return self.importance_sampling_level

    def _ppo_policy_surrogate(
        self,
        token_log_ratio: torch.Tensor,
        advantages: torch.Tensor,
        action_mask: torch.Tensor,
        turn_ids: torch.Tensor,
        is_level: str,
        loss_weight: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Clipped PPO policy surrogate at the given IS / ratio-pooling level.

        The importance ratio (and the advantage paired with it) is pooled to
        ``is_level`` — token (no pooling), turn (configured by
        ``self.turn_ratio_pooling``), or trajectory (mean over the whole
        completion). Returns
        ``(pg_loss, clipfrac)``; used by the non-Liger PPO path. Operates only
        on ``(B, T)`` tensors, so it is memory-bounded.

        :param token_log_ratio: ``(B, T)`` per-token ``log pi - log pi_old``.
        :type token_log_ratio: torch.Tensor
        :param advantages: ``(B, T)`` per-token advantages.
        :type advantages: torch.Tensor
        :param action_mask: ``(B, T)`` action-token mask.
        :type action_mask: torch.Tensor
        :param turn_ids: ``(B, T)`` turn index per token (``-1`` non-action).
        :type turn_ids: torch.Tensor
        :param is_level: ``"token"`` / ``"turn"`` / ``"trajectory"``.
        :type is_level: str
        :param loss_weight: Optional ``(B, T)`` detached per-token weight (the
            truncated vLLM importance ratio) multiplied into the surrogate
            before the masked mean; ``None`` leaves the loss unchanged.
        :type loss_weight: torch.Tensor | None, optional
        :return: ``(pg_loss, clipfrac)`` scalars.
        :rtype: tuple[torch.Tensor, torch.Tensor]
        """
        return clipped_is_surrogate(
            token_log_ratio,
            advantages,
            action_mask,
            turn_ids,
            is_level,
            self.clip_coef,
            loss_weight=loss_weight,
            turn_reduction=self.turn_ratio_pooling,
        )

    def _compute_gae_returns(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        action_mask: torch.Tensor,
        turn_ids: torch.Tensor,
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
        :return: Tuple of ``(token_returns, token_advantages)``, each ``[batch, seq_len]``.
        :rtype: tuple[torch.Tensor, torch.Tensor]
        """
        batch_size = values.shape[0]
        num_turns = int(turn_ids.max().item()) + 1

        turn_values = pool_by_turns(
            values,
            turn_ids,
            num_turns,
            reduction=self.turn_value_reduction,
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
            last_gae = (delta + self.gamma * self.gae_lambda * last_gae) * has_turn
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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute token-level GAE and returns.

        :param rewards: Per-token rewards ``[batch, seq_len]``.
        :type rewards: torch.Tensor
        :param values: Per-token critic values ``[batch, seq_len]``.
        :type values: torch.Tensor
        :param action_mask: Bool mask of valid action positions ``[batch, seq_len]``.
        :type action_mask: torch.Tensor
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
            last_gae = delta + self.gamma * self.gae_lambda * last_gae * next_mask
            token_advantages[:, t] = last_gae * mask[:, t]

        token_returns = (token_advantages + values) * mask
        if self.whiten_advantages:
            token_advantages = masked_whiten(token_advantages, action_mask)
        return token_returns, token_advantages * mask

    def _ppo_loss_liger(
        self,
        batch_ids: torch.Tensor,
        batch_action_mask: torch.Tensor,
        batch_old_log_probs: torch.Tensor,
        batch_reference_log_probs: torch.Tensor,
        batch_returns: torch.Tensor,
        batch_advantages: torch.Tensor,
        batch_old_values: torch.Tensor,
        batch_turn_ids: torch.Tensor,
        ppo_granularity: str,
        sampling_log_probs: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """PPO loss via the fused-linear PPO Function (token + turn modes).

        Two body forwards (matching the doubled-forward shape of
        :meth:`_fused_forward`):

        1. **Actor pass** under ``select_adapter("actor")``. ``lm_head``
           pre-hook captures hidden states; :class:`LigerFusedLinearPolicyLossFunction`
           computes the chunked policy + KL loss without ever materializing
           ``(B, T, V)`` for the autograd graph.
        2. **Critic pass** under ``select_adapter("critic")``. Standard
           forward through the value head; value loss computed with the
           token-level clipped formulation in token mode and the turn-level
           clipped formulation in turn mode. The ``(B, T, 1)`` value tensor
           is small — fusion buys nothing here.

        Granularity dispatch:

        * ``ppo_granularity == "token"``: per-token policy loss inside the
          Liger Function; per-token clipped value loss outside.
        * ``ppo_granularity == "turn"`` and ``self.turn_level_clip``:
          token log-ratios are scatter-pooled into per-turn log-ratios
          inside the Liger Function; turn-level advantages and clipping
          + per-turn value loss outside.
        * ``ppo_granularity == "turn"`` and not ``self.turn_level_clip``:
          per-token policy loss (broadcast turn advantages to tokens up
          front); per-turn value loss outside.

        :return: ``(total_loss, metrics)`` with ``metrics`` keying scalar
            Python floats: ``kl``, ``pg_loss``, ``vf_loss``, ``clipfrac``,
            ``entropy``.
        :rtype: tuple[torch.Tensor, dict[str, float]]
        """
        if not HAS_LIGER_KERNEL:
            msg = (
                "Liger PPO loss was requested but `liger-kernel` is not "
                "available. Set use_liger_loss=False."
            )
            raise ImportError(msg)

        batch_ids = batch_ids.to(self.device)
        mask = batch_action_mask.to(self.device).contiguous()
        old_log_probs = batch_old_log_probs.to(self.device).contiguous()
        ref_log_probs = batch_reference_log_probs.to(self.device).contiguous()
        advantages = batch_advantages.to(self.device).contiguous()
        returns = batch_returns.to(self.device).contiguous()
        old_values = batch_old_values.to(self.device).contiguous()
        turn_ids = batch_turn_ids.to(self.device).contiguous()
        mask_bool = mask.bool()

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

        # Resolve the IS / ratio-pooling level and pool advantages to match.
        is_level = self._resolve_is_level(ppo_granularity)
        if is_level != "token":
            self._warn_liger_non_token_is(
                is_level, "PPO", once_attr="_ppo_liger_mem_warned"
            )
        # ``max_turns`` / ``full_turn_mask`` are derived from the global
        # ``turn_ids`` so chunks see consistent denominators; needed by both
        # turn-level value loss and turn-level IS pooling.
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

        # Identity-patch lm_head so the actor forward outputs the last hidden
        # state (B, T, H) directly instead of computing the full (B, T, V)
        # logits only to discard them. lm_head_weight is passed separately to
        # LigerFusedLinearPolicyLossFunction which handles the matmul and its grad.
        lm_head = self._get_lm_head()
        lm_head_weight = lm_head.weight
        lm_head_bias = lm_head.bias

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

        target_ids = batch_ids[:, 1:].contiguous()
        # Token level token-flattens the hidden states so the fused kernel
        # chunks tokens (bounded); turn/sequence keep the batch path. See
        # :func:`apply_fused_policy_loss`.
        with self._liger_head_gather():
            loss_pg_kl, aux = apply_fused_policy_loss(
                policy_hidden[:, :-1],
                lm_head_weight,
                lm_head_bias,
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
                    getattr(lm_head_weight, "ds_shape", lm_head_weight.shape)[0],
                    self.chunk_rows,
                ),
                turn_log_ratio_reduction=self.turn_ratio_pooling,
                vllm_is_ratio=vllm_is_ratio,
            )
        kl_metric = float(aux[0].item())
        clipfrac_metric = float(aux[1].item())
        pg_loss_metric = float(aux[2].item())
        entropy_metric = float(aux[3].item())

        # ---- Critic pass (unfused — value tensor is small) ----
        # Identity-patch lm_head here too: only critic_output[2] (the value
        # head) is needed; the (B, T, V) logits would otherwise be materialised
        # and immediately discarded.
        with (
            self._patch_lm_head_to_identity(),
            self.select_adapter("critic"),
            self._amp_ctx(),
        ):
            self.actor.train()
            critic_output = self.actor(**kwargs)
        # Value head wrappers return ``(hidden, loss, value)`` when lm_head is
        # identity-patched; index [2] is the same value tensor either way.
        critic_value = (
            critic_output[2]
            if isinstance(critic_output, tuple)
            else critic_output.value
        )
        # Align with the unfused path's ``[:, :-1]`` shift.
        batch_values = critic_value[:, :-1]
        batch_values = torch.masked_fill(batch_values, ~mask_bool, 0.0)
        if ppo_granularity == "turn":
            # Per-turn clipped value loss — same formula as the unfused
            # turn branch in :meth:`learn`.
            turn_pred = pool_by_turns(
                batch_values,
                turn_ids,
                max_turns,
                reduction=self.turn_value_reduction,
            )
            turn_old = pool_by_turns(
                old_values,
                turn_ids,
                max_turns,
                reduction=self.turn_value_reduction,
            )
            turn_ret = pool_by_turns(returns, turn_ids, max_turns)
            vf_unclipped = (turn_ret - turn_pred).pow(2)
            clipped_turn_values = turn_old + torch.clamp(
                turn_pred - turn_old, -self.clip_coef, self.clip_coef
            )
            vf_clipped = (turn_ret - clipped_turn_values).pow(2)
            vf_loss = (
                0.5
                * (torch.max(vf_unclipped, vf_clipped) * full_turn_mask).sum()
                / full_turn_mask.sum().clamp(min=1)
                * self.vf_coef
            )
        else:
            vf_loss = self._compute_vf_loss_token(
                batch_values, old_values, returns, mask
            )

        total_loss = loss_pg_kl + vf_loss
        metrics = {
            "kl": kl_metric,
            "clipfrac": clipfrac_metric,
            "pg_loss": pg_loss_metric,
            "vf_loss": float(vf_loss.item()),
            "entropy": entropy_metric,
        }
        return total_loss, metrics

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
