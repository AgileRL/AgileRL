from __future__ import annotations

import gc
import warnings
from collections.abc import Callable
from contextlib import nullcontext
from inspect import Signature, signature
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import torch

from agilerl import HAS_LIGER_KERNEL, HAS_LLM_DEPENDENCIES
from agilerl.utils.llm_utils import calculate_k3_kl

if TYPE_CHECKING:
    from accelerate import Accelerator
    from peft import LoraConfig

    from agilerl.llm_envs import ReasoningGym

if HAS_LIGER_KERNEL or TYPE_CHECKING:
    from liger_kernel.chunked_loss.grpo_loss import LigerFusedLinearGRPOFunction
else:
    # Keep the name resolvable when liger-kernel isn't installed so unit
    # tests can patch it. ``_liger_loss`` guards against actual use.
    LigerFusedLinearGRPOFunction = None  # type: ignore[assignment]

from agilerl.algorithms.core import ActionResult, LLMAlgorithm
from agilerl.algorithms.core.registry import HyperparameterConfig, NetworkGroup
from agilerl.protocols import (
    MultiTurnEnv,
    PeftModelProtocol,
    PreTrainedModelProtocol,
)
from agilerl.typing import ExperiencesType, LLMObsType
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
    ReasoningGym,
    build_completion_mask,
    masked_mean,
    normalize_reasoning_prompt_batch,
    pool_log_ratio_by_level,
    prepare_prompt_hf_generate,
    stitch_completion_after_windowed_hf_generate,
    validate_importance_sampling_level,
    validate_llm_context_lengths,
)

if HAS_LLM_DEPENDENCIES or TYPE_CHECKING:
    from transformers import GenerationConfig


class GRPO(LLMAlgorithm):
    """The GRPO algorithm class. GRPO paper: https://arxiv.org/pdf/2402.03300.

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
    :param micro_batch_size_per_gpu: If specified, gradient_accumulation_steps will be
        calculated to achieve the target batch_size. If None, uses existing
        gradient_accumulation_steps from DeepSpeed config, defaults to None
    :type micro_batch_size_per_gpu: int, optional
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
    :param use_memory_efficient_params: Deprecated and ignored. Colocated vLLM
        shares its base with the trainer (one resident copy), so there is no
        separate copy to shuttle CPU<->GPU. Kept for API compatibility.
    :type use_memory_efficient_params: bool
    :param accelerator: Accelerator for distributed computing, defaults to None
    :type accelerator: accelerate.Accelerator(), optional
    :param device: Device for accelerated computing, 'cpu' or 'cuda', defaults to 'cpu'
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
    :param use_liger_loss: Use Liger kernel for memory-efficient loss
        computation. Defaults to ``False``. Pass ``True`` to opt in
        (requires ``liger-kernel`` to be installed; warns and falls back
        to ``False`` otherwise). Supported for ``loss_type`` values
        ``'grpo'``, ``'cispo'``, and ``'gspo'``. Note that the Liger path
        uses DAPO-style batch normalisation for ``'cispo'`` rather than
        the per-sequence-then-batch normalisation of the standard path;
        numerical values will differ slightly but gradient direction is
        equivalent.
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
        sugar for ``"grpo"`` at sequence level (it forces
        ``importance_sampling_level="sequence"``).
    :type loss_type: Literal["grpo", "gspo", "cispo"], optional
    :param importance_sampling_level: Granularity at which the importance
        *ratio* is pooled before clipping/weighting, defaults to ``"token"``.
        This is independent of ``advantage_granularity`` (the advantage axis).

        * ``"token"`` — per-token ratio (standard GRPO / CISPO).
        * ``"turn"``  — pool the per-token log-ratio over each turn (length-
          normalized geometric mean) and clip/weight per turn. Requires
          ``turn_ids`` in :meth:`learn`.
        * ``"sequence"`` — pool over the whole completion (GSPO).

        Turn- and sequence-level pooling couple a turn/sequence's tokens, so
        they have no fused Liger kernel: they run on the standard PyTorch
        path, which is always memory-bounded (the fused-linear-logprob path is
        unconditional). Token level keeps the (faster, already-bounded) Liger
        path when ``use_liger_loss=True``.
    :type importance_sampling_level: Literal["token", "turn", "sequence"], optional
    :param advantage_granularity: Unit at which the group-relative *advantage* is
        computed, independent of ``importance_sampling_level``. Defaults to
        ``"auto"``.

        * ``"trajectory"`` — one group-relative scalar per completion (standard
          GRPO), broadcast to all tokens.
        * ``"turn"`` — group-relative per turn (each turn's reward normalized
          within its group), broadcast to that turn's tokens. Requires
          ``turn_ids`` and per-turn rewards ``(batch, max_turns)`` in
          :meth:`learn`; falls back to trajectory if unavailable.
        * ``"auto"`` — follow the IS level (turn when
          ``importance_sampling_level="turn"``, else trajectory), preserving the
          original coupled default.

        There is no token-level advantage (group-relative needs a reward per
        unit; tokens have none). Any advantage x IS combination is valid, e.g.
        per-turn advantages with token-level clipping.
    :type advantage_granularity: Literal["auto", "trajectory", "turn"], optional
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
        max_output_tokens: int | None = None,
        min_output_tokens: int | None = None,
        max_model_len: int | None = 1024,
        hf_generate_chunk_size: int | None = None,
        lora_config: LoraConfig | None = None,
        cosine_lr_schedule_config: CosineLRScheduleConfig | None = None,
        accelerator: Accelerator | None = None,
        device: str = "cpu",
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
        importance_sampling_level: Literal["token", "turn", "sequence"] = "token",
        advantage_granularity: Literal["auto", "trajectory", "turn"] = "auto",
        action_granularity: Literal["auto", "trajectory", "turn"] | None = None,
        use_separate_reference_adapter: bool = True,
        whiten_advantages: bool = False,
        adv_clip_range: float | None = None,
        filter_zero_adv: bool = False,
        adv_filter_eps: float = 0.0,
        reduce_memory_peak: bool = False,
        cast_logprobs_to_fp32: bool = True,
        fused_logprobs_chunk_rows: int | None = None,
        quantization_config: Any | None = None,
        activation_offload: bool = False,
        lora_target_scope: str | None = None,
        liger_token_chunk_size: int | None = None,
        vllm_importance_sampling_correction: bool = False,
        vllm_importance_sampling_apply: bool = True,
        vllm_importance_sampling_cap: float = 2.0,
        use_sequence_packing: bool = False,
    ) -> None:
        resolved_device = (
            f"cuda:{accelerator.process_index}"
            if accelerator is not None
            else (
                "cuda"
                if torch.cuda.is_available()
                else "mps"
                if torch.backends.mps.is_available()
                else "cpu"
            )
        )
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
            fused_logprobs_chunk_rows=fused_logprobs_chunk_rows,
            quantization_config=quantization_config,
            activation_offload=activation_offload,
            use_sequence_packing=use_sequence_packing,
            lora_target_scope=lora_target_scope,
            liger_token_chunk_size=liger_token_chunk_size,
            vllm_importance_sampling_correction=vllm_importance_sampling_correction,
            vllm_importance_sampling_apply=vllm_importance_sampling_apply,
            vllm_importance_sampling_cap=vllm_importance_sampling_cap,
        )
        assert isinstance(batch_size, int), "Batch size must be an integer."
        assert batch_size >= 1, "Batch size must be greater than or equal to one."
        assert isinstance(lr, float), "Learning rate must be a float."
        assert lr > 0, "Learning rate must be greater than zero."
        if isinstance(clip_coef, (tuple, list)):
            if len(clip_coef) != 2:
                msg = "clip_coef tuple must contain exactly two values."
                raise ValueError(msg)
            clip_coef_min = float(clip_coef[0])
            clip_coef_max = float(clip_coef[1])
            # Intentionally do not enforce clip_coef_min < clip_coef_max here to
            # preserve existing behavior for user-provided tuple/list bounds.
        elif isinstance(clip_coef, (float, int)):
            clip_coef = float(clip_coef)
            if clip_coef < 0:
                msg = "clip_coef must be greater than or equal to zero."
                raise ValueError(msg)
            clip_coef_min = 1 - clip_coef
            clip_coef_max = 1 + clip_coef
        else:
            msg = "clip_coef must be a float or a tuple or list of two floats."
            raise TypeError(msg)
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
        self.clip_coef = clip_coef
        self.clip_coef_min = clip_coef_min
        self.clip_coef_max = clip_coef_max
        self.update_epochs = update_epochs
        self.group_size = group_size
        self.beta = beta
        self.temperature = temperature
        self.repetition_penalty = repetition_penalty
        self.top_p = top_p
        self.top_k = top_k
        self.min_p = min_p
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
        self.adv_norm = adv_norm
        # Padding-free sequence packing for the gradient forward is opt-in and
        # only honoured under a FlashAttention-2 backend (see
        # LLMAlgorithm._sequence_packing_active); otherwise inert. The flag is
        # stored on the base class (forwarded via super().__init__ above).
        if loss_type not in {"grpo", "gspo", "cispo"}:
            msg = (
                f"Invalid loss_type '{loss_type}'. "
                "Expected one of ['grpo', 'gspo', 'cispo']."
            )
            raise ValueError(msg)
        if adv_clip_range is not None and adv_clip_range <= 0:
            msg = "adv_clip_range must be > 0 when provided."
            raise ValueError(msg)
        if adv_filter_eps < 0:
            msg = "adv_filter_eps must be >= 0."
            raise ValueError(msg)
        validate_importance_sampling_level(importance_sampling_level, allow_auto=False)
        if action_granularity is not None:
            warnings.warn(
                "action_granularity is deprecated; use advantage_granularity.",
                DeprecationWarning,
                stacklevel=2,
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
        self.loss_type = loss_type
        # Three orthogonal axes back the GRPO family:
        #   objective        : "grpo" (min-clip surrogate) or "cispo" (clamped-
        #                      weight x logp)
        #   importance_sampling_level : where the IS *ratio* is pooled before the
        #                      clip/weight -- "token" (none), "turn" (per turn),
        #                      "sequence" (whole completion, i.e. GSPO).
        #   advantage_granularity : the unit the *advantage* is computed at --
        #                      "trajectory" (one group-relative scalar per
        #                      completion) or "turn" (group-relative per turn).
        #                      Independent of the IS level. "auto" follows the IS
        #                      level (turn IS -> turn advantage, else trajectory)
        #                      to preserve the original coupled default.
        # ``loss_type`` is the back-compat user knob: "gspo" is sugar for the
        # grpo objective at sequence level.
        self.objective = "cispo" if loss_type == "cispo" else "grpo"
        if loss_type == "gspo":
            # GSPO is, by definition, grpo-objective at the sequence level.
            self.importance_sampling_level = "sequence"
        else:
            self.importance_sampling_level = importance_sampling_level
        self.advantage_granularity = advantage_granularity
        self.whiten_advantages = whiten_advantages
        self.adv_clip_range = adv_clip_range
        self.filter_zero_adv = filter_zero_adv
        self.adv_filter_eps = adv_filter_eps
        # ``liger_token_chunk_size`` (per-chunk token count for the Liger
        # fused-loss path) is validated and stored by ``super().__init__`` above;
        # ``None`` falls back to the legacy ``AGILERL_LIGER_TOKEN_CHUNK`` env var
        # (default 2048) via ``self._resolve_liger_token_chunk()``.
        if self.loss_type == "cispo" and self.beta != 0:
            warnings.warn(
                "CISPO is typically used with beta=0; nonzero beta adds KL "
                "regularization to the objective.",
                stacklevel=2,
            )
        # Upstream Liger's GRPO kernel only knows token- and sequence-level
        # importance sampling. Turn-level pooling couples a turn's tokens, so
        # it has no fused kernel -- it runs on the decoupled standard path
        # (always memory-bounded via the fused-linear-logprob path).
        self._liger_unsupported_level = self.importance_sampling_level == "turn" or (
            self.objective == "cispo" and self.importance_sampling_level != "token"
        )
        # Warn once, up front, about the memory behaviour of Liger + non-token
        # IS. The combination is permitted; only token-level IS gets the
        # token-flattened (memory-bounded) Liger path. The canonical message
        # lives in the base ``_warn_liger_non_token_is`` helper (warn-once via
        # ``_liger_non_token_warned``); warning here suppresses the duplicate
        # loss-time warning.
        self._liger_non_token_warned = False
        if self.use_liger_loss and self.importance_sampling_level in {
            "turn",
            "sequence",
        }:
            algo_name = (
                "GSPO" if self.importance_sampling_level == "sequence" else "GRPO"
            )
            self._warn_liger_non_token_is(self.importance_sampling_level, algo_name)
        if self.use_liger_loss and use_kl_advantage_shaping:
            warnings.warn(
                "use_kl_advantage_shaping is not supported with use_liger_loss=True; "
                "disabling KL advantage shaping.",
                stacklevel=2,
            )
            use_kl_advantage_shaping = False
        self.use_kl_advantage_shaping = use_kl_advantage_shaping
        self._loss_fn = self._resolve_standard_loss_fn()
        if max_output_tokens is None and max_model_len is None:
            msg = "Either max_output_tokens or max_model_len must be specified"
            raise ValueError(
                msg,
            )
        self.max_output_tokens = (
            max_output_tokens if max_output_tokens is not None else max_model_len
        )
        self.min_output_tokens = min_output_tokens
        self.max_model_len = (
            max_model_len if max_model_len is not None else max_output_tokens
        )
        validate_llm_context_lengths(self.max_model_len, max_output_tokens)
        self.hf_generate_chunk_size = int(
            1 if hf_generate_chunk_size is None else max(1, hf_generate_chunk_size)
        )
        if self.use_vllm and hf_generate_chunk_size is not None:
            warnings.warn(
                "hf_generate_chunk_size is only used for HuggingFace generation "
                "(use_vllm=False) and will be ignored when use_vllm=True.",
                stacklevel=2,
            )
        self.generation_config = GenerationConfig(
            do_sample=True,
            temperature=temperature,
            max_length=self.max_model_len,
            max_new_tokens=max_output_tokens,
            min_new_tokens=min_output_tokens,
            pad_token_id=pad_token_id,
            repetition_penalty=repetition_penalty,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
        )

        self._setup_actors(actor_network, clone=clone)
        # Register network groups for mutations
        self.register_network_group(NetworkGroup(eval_network=self.actor, policy=True))
        if self.wrap:
            self.wrap_models()

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
                            if training and group_size > 1:
                                prompt["input_ids"] = prompt["input_ids"].repeat(
                                    group_size,
                                    1,
                                )
                                prompt["attention_mask"] = prompt[
                                    "attention_mask"
                                ].repeat(
                                    group_size,
                                    1,
                                )
                            stitch_ids = prompt.pop("stitch_prefix_ids", None)
                            if (
                                stitch_ids is not None
                                and training
                                and group_size > 1
                                and stitch_ids.shape[0] == 1
                            ):
                                stitch_ids = stitch_ids.repeat(group_size, 1)
                            initial_prompt_len = prompt.pop("initial_prompt_len", None)
                            completion_id = self.actor.generate(
                                **prompt,
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

    def learn(
        self,
        experiences: ExperiencesType,
        turn_ids: torch.Tensor | None = None,
        sampling_logps: list[torch.Tensor | None] | None = None,
    ) -> dict[str, float]:
        """Update agent network parameters to learn from experiences.

        :param experiences: ``(completion_ids, action_masks, rewards)`` stacked
            batch. For ``importance_sampling_level="turn"`` with per-turn
            rewards, ``rewards`` is ``(batch, max_turns)``; otherwise it is one
            scalar per trajectory (per-turn rewards are summed to the episode
            return).
        :type experiences: ExperiencesType
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
        :return: Dict with keys ``mean_loss`` and ``mean_kl``, averaged over the update.
        :rtype: dict[str, float]
        """
        gc.collect()
        torch.cuda.empty_cache()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

        self._prepare_vllm_for_training()

        with self.memory_efficient_params_context():
            completion_ids, action_masks, rewards = stack_and_pad_experiences(
                *experiences,
                padding_values=[self.pad_token_id, False, None],
            )
            action_masks = action_masks.to(self.device)
            rewards = rewards.to(self.device).float()
            completion_ids = completion_ids.to(self.device)
            num_samples = completion_ids.shape[0]
            # NOTE: the group-size divisibility check is deferred into the
            # advantage branches below, so it runs *after* rewards-cardinality
            # validation. That ordering makes a rewards/trajectory count
            # mismatch report its own (more specific) error instead of the
            # generic divisibility message.

            if turn_ids is not None:
                turn_ids = turn_ids.to(self.device)
                if turn_ids.shape[0] != num_samples:
                    msg = (
                        f"turn_ids batch ({turn_ids.shape[0]}) must match "
                        f"completion batch ({num_samples})."
                    )
                    raise ValueError(msg)

            # Advantage granularity (advantage_granularity) and IS / ratio-pooling
            # level (importance_sampling_level) are independent. Resolve the
            # advantage unit first; the IS level is applied later in the loss.
            adv_granularity = self._resolve_advantage_granularity()
            use_turn_advantage = adv_granularity == "turn" and turn_ids is not None
            if use_turn_advantage:
                self._assert_batch_divisible_by_group(num_samples)
                # Per-turn group-relative advantages, broadcast to tokens. A
                # turn is the action unit: each turn's reward is normalized
                # within its group, then assigned to every token of that turn.
                turn_rewards = (
                    rewards if rewards.dim() > 1 else rewards.reshape(num_samples, -1)
                )
                turn_advantages = self._calculate_turn_advantage(turn_rewards).to(
                    self.device
                )
                safe_turn_ids = turn_ids.clamp(min=0).to(torch.int64)
                num_reward_turns = turn_advantages.shape[1]
                if int(safe_turn_ids.max().item()) >= num_reward_turns:
                    msg = (
                        "turn_ids reference a turn index beyond the number of "
                        f"reward turns ({num_reward_turns}); rewards and "
                        "turn_ids are misaligned."
                    )
                    raise ValueError(msg)
                advantages = turn_advantages.gather(1, safe_turn_ids)  # (B, T-1)
                advantages = advantages * action_masks.to(advantages.dtype)
            else:
                # Per-trajectory group-relative advantage. If callers pass
                # per-turn rewards [batch, max_turns], collapse to episode returns.
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
                advantages = self._calculate_advantage(rewards).to(self.device)

            # The IS ratio is pooled per turn only at turn level; token/sequence
            # ignore turn_ids. This is decoupled from the advantage granularity
            # above — e.g. per-turn advantages may pair with token-level IS.
            is_turn_ids = turn_ids if self.importance_sampling_level == "turn" else None

            # Advantage post-processing (filter / whiten / clip) is
            # shape-agnostic across per-trajectory (B, 1) and per-turn-broadcast
            # (B, T-1) advantages.
            per_sample_abs = (
                advantages.detach().reshape(num_samples, -1).abs().amax(dim=-1)
            )
            active_adv_mask = None
            if self.filter_zero_adv:
                active_adv_mask = per_sample_abs > self.adv_filter_eps
            if self.whiten_advantages:
                advantages = self._whiten_advantages(
                    advantages, action_masks, active_adv_mask
                )
            if self.adv_clip_range is not None:
                advantages = advantages.clamp(-self.adv_clip_range, self.adv_clip_range)

            if active_adv_mask is not None:
                batch_idxs = np.where(active_adv_mask.detach().cpu().numpy())[0]
                if batch_idxs.size == 0:
                    warnings.warn(
                        "All samples were filtered by advantage threshold; skipping GRPO update.",
                        stacklevel=2,
                    )
                    return {"mean_loss": 0.0, "mean_kl": 0.0}
            else:
                batch_idxs = np.arange(num_samples)
            learn_metrics = {
                "mean_loss": 0.0,
                "mean_kl": 0.0,
            }
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

            # Align the captured per-row vLLM sampling logprobs onto the
            # (B, T-1) action frame (single-turn and multi-turn alike), then
            # measure (and optionally correct) the vLLM-vs-trainer mismatch.
            # Metrics are logged whenever logprobs were captured, independently
            # of whether the correction is applied to the loss.
            is_metrics: dict[str, float] = {}
            sampling_log_probs, n_skipped = self._align_sampling_logprobs(
                sampling_logps, action_masks, old_log_probs
            )
            if sampling_log_probs is not None:
                is_metrics = self._sampling_mismatch_metrics(
                    old_log_probs, sampling_log_probs, action_masks
                )
                if n_skipped:
                    is_metrics["vllm_is_rows_skipped"] = float(n_skipped)
                    warnings.warn(
                        f"{n_skipped}/{num_samples} rows had a token-count "
                        "mismatch between captured vLLM logprobs and the action "
                        "mask; their importance ratio defaults to 1 (no "
                        "correction). Check rollout/trainer tokenisation if this "
                        "is large.",
                        stacklevel=2,
                    )

            effective_num_samples = len(batch_idxs)
            if effective_num_samples == 0:
                warnings.warn(
                    "No active samples after filtering; skipping GRPO update.",
                    stacklevel=2,
                )
                return {"mean_loss": 0.0, "mean_kl": 0.0}

            # Ensure batch_size is not larger than the number of active samples
            batch_size = min(batch_size, effective_num_samples)

            for _ in range(self.update_epochs):
                self.rng.shuffle(batch_idxs)
                for start in range(0, effective_num_samples, batch_size):
                    minibatch_idxs = batch_idxs[
                        start : min((start + batch_size), effective_num_samples)
                    ]
                    if len(minibatch_idxs) == 0:
                        continue
                    loss, kl = self._loss(
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
                    if not loss.isfinite():
                        msg = f"Loss is not finite: {loss}"
                        raise ValueError(msg)
                    self._backward_pass(loss)
                    learn_metrics["mean_loss"] += loss.item()
                    learn_metrics["mean_kl"] += kl.item()
                    updates += 1
        result = {
            metric: value / max(updates, 1) for metric, value in learn_metrics.items()
        }
        # Sampling-mismatch metrics are computed once over the full batch, so
        # they bypass the per-update averaging above.
        result.update(is_metrics)
        return result

    def test(
        self,
        env: ReasoningGym | MultiTurnEnv,
        loop: int = 1,
        *args: Any,
        **kwargs: Any,
    ) -> np.ndarray:
        """Return fitness (test) score of llm on test sub-set.

        :param env: Dataset-style ``ReasoningGym`` environment or tokenized
            multi-turn episode environment.
        :type env: ReasoningGym | MultiTurnEnv
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
            elif isinstance(env, MultiTurnEnv):
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
                        prompt_dict, reward, terminated, truncated, _info = env.step(
                            full,
                        )
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
                    f"MultiTurnEnv; got {type(env).__name__}"
                )
                raise TypeError(msg)
        mean_fit = torch.mean(reward_tensor).item()
        self.fitness.append(mean_fit)
        return np.array(mean_fit)

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
        if self.group_size == 1:
            # A singleton group has no group-relative signal: the centered
            # reward is identically zero and the (unbiased) std is undefined,
            # so the advantage is zero by definition. Returning zeros avoids a
            # 0/0 NaN without perturbing the group_size > 1 numerics.
            return torch.zeros(numel, 1, dtype=rewards.dtype, device=rewards.device)
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
    ) -> torch.Tensor:
        """Group-relative advantage computed independently per turn.

        Treats each turn as a separate RL action: for turn ``k`` the reward is
        normalized within its group of ``group_size`` completions (the same
        group-relative scheme as :meth:`_calculate_advantage`, applied per
        turn). The caller then broadcasts each ``(sample, turn)`` advantage to
        every token of that turn via ``turn_ids``.

        :param rewards: Per-turn rewards ``(batch, max_turns)``; the batch dim
            is grouped in contiguous blocks of ``group_size``.
        :type rewards: torch.Tensor
        :param eps: Epsilon guarding the per-turn std division.
        :type eps: float, optional
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
        if self.group_size == 1:
            # Singleton group: no per-turn group-relative signal (see
            # _calculate_advantage); the advantage is zero by definition.
            return torch.zeros(
                batch, num_turns, dtype=rewards.dtype, device=rewards.device
            )
        grouped = rewards.view(-1, self.group_size, num_turns)
        centered = grouped - grouped.mean(dim=1, keepdim=True)
        if self.adv_norm == "mean_only":
            advantage = centered
        else:
            advantage = centered / (grouped.std(dim=1, keepdim=True) + eps)
        return advantage.reshape(batch, num_turns)

    def _resolve_advantage_granularity(self) -> str:
        """Resolve the unit at which group-relative advantages are computed.

        Independent of :attr:`importance_sampling_level` (the IS / ratio-pooling
        axis). ``"auto"`` follows the IS level — turn IS implies per-turn
        advantages, otherwise per-trajectory — reproducing the original coupled
        default. Explicit ``"trajectory"`` / ``"turn"`` override, enabling any
        advantage x IS combination (e.g. per-turn advantages with token-level
        clipping, or one trajectory advantage with turn-level pooling).

        :return: ``"trajectory"`` or ``"turn"``.
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
        :param action_masks: ``(B, T-1)`` action-token mask.
        :param active_adv_mask: Optional ``(B,)`` per-sample keep mask.
        :return: Whitened advantages with the same shape as ``advantages``.
        """
        if advantages.dim() <= 1 or advantages.shape[-1] == 1:
            adv = advantages.squeeze(-1).clone()
            if active_adv_mask is not None and active_adv_mask.any():
                active = adv[active_adv_mask]
                adv[active_adv_mask] = (active - active.mean()) / (
                    active.std(unbiased=False) + 1e-8
                )
            else:
                adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)
            return adv.unsqueeze(-1)
        mask_f = action_masks.to(advantages.dtype)
        if active_adv_mask is not None:
            mask_f = mask_f * active_adv_mask.unsqueeze(-1).to(advantages.dtype)
        denom = mask_f.sum().clamp(min=1.0)
        mean = (advantages * mask_f).sum() / denom
        var = (((advantages - mean) ** 2) * mask_f).sum() / denom
        whitened = (advantages - mean) / (var.sqrt() + 1e-8)
        return torch.where(mask_f.bool(), whitened, advantages)

    def _resolve_standard_loss_fn(
        self,
    ) -> Callable[
        [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        tuple[torch.Tensor, torch.Tensor],
    ]:
        """Resolve the active standard (non-Liger) loss function.

        Dispatch is on the *objective* (``grpo`` min-clip vs ``cispo``
        clamped-weight); the importance-sampling level (token/turn/sequence)
        is applied inside via ``self.importance_sampling_level``.
        """
        if self.objective == "cispo":
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

    def _reduce_masked_loss(
        self,
        loss: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Reduce per-token losses to a per-sequence mean over valid action tokens."""
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
        minibatch_idxs: np.ndarray,
        completion_ids: torch.Tensor,
        action_mask: torch.Tensor,
        advantages: torch.Tensor,
        old_log_probs: torch.Tensor,
        reference_log_probs: torch.Tensor,
        turn_ids: torch.Tensor | None = None,
        sampling_log_probs: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample a minibatch and compute the active objective loss.

        :param batch_size: Micro-batch size used for log-prob computation.
        :type batch_size: int
        :param minibatch_idxs: Indices selecting the current minibatch.
        :type minibatch_idxs: np.ndarray
        :param completion_ids: Full completion token IDs.
        :type completion_ids: torch.Tensor
        :param action_mask: Full action mask.
        :type action_mask: torch.Tensor
        :param advantages: Full advantages tensor.
        :type advantages: torch.Tensor
        :param old_log_probs: Full old policy log probabilities.
        :type old_log_probs: torch.Tensor
        :param reference_log_probs: Full reference policy log probabilities.
        :type reference_log_probs: torch.Tensor
        :param turn_ids: Optional full ``(B, seq_len-1)`` turn indices used by
            turn-level importance sampling; ``None`` for token/sequence levels.
        :type turn_ids: torch.Tensor | None
        :return: Mean loss and mean KL divergence.
        :rtype: tuple[torch.Tensor, torch.Tensor]
        """
        if turn_ids is not None:
            (
                batch_ids,
                batch_action_mask,
                batch_advantages,
                batch_old_log_probs,
                batch_reference_log_probs,
                batch_turn_ids,
            ) = get_experiences_samples(
                minibatch_idxs,
                completion_ids,
                action_mask,
                advantages,
                old_log_probs,
                reference_log_probs,
                turn_ids,
            )
        else:
            (
                batch_ids,
                batch_action_mask,
                batch_advantages,
                batch_old_log_probs,
                batch_reference_log_probs,
            ) = get_experiences_samples(
                minibatch_idxs,
                completion_ids,
                action_mask,
                advantages,
                old_log_probs,
                reference_log_probs,
            )
            batch_turn_ids = None

        batch_sampling_log_probs = (
            sampling_log_probs[minibatch_idxs]
            if sampling_log_probs is not None
            else None
        )

        # Upstream Liger's GRPO kernel can't express turn-level (or
        # sequence-level CISPO) pooling, nor the per-token vLLM sampling-mismatch
        # correction, so those cases run on the decoupled standard path, which is
        # always memory-bounded via the fused-linear-logprob path.
        if (
            self.use_liger_loss
            and not self._liger_unsupported_level
            and batch_sampling_log_probs is None
        ):
            return self._liger_loss(
                batch_ids,
                batch_action_mask,
                batch_advantages,
                batch_old_log_probs,
                batch_reference_log_probs,
            )
        if (
            self.use_liger_loss
            and batch_sampling_log_probs is not None
            and not self._is_correction_liger_warned
        ):
            warnings.warn(
                "use_liger_loss=True is incompatible with the vLLM "
                "sampling-mismatch correction (the fused kernel cannot apply a "
                "per-token importance weight); using the standard PyTorch path.",
                stacklevel=2,
            )
            self._is_correction_liger_warned = True
        if self.use_liger_loss and self._liger_unsupported_level:
            # Non-token IS (turn-level, or sequence-level CISPO) has no fused
            # kernel and runs the standard path; emit the canonical
            # not-memory-bounded warning (warn-once, shared with the
            # constructor). Note this guards on the real ``_liger_unsupported_level``
            # condition, not the routing fall-through above: token-level GRPO that
            # only falls through for the vLLM correction is fully Liger-supported
            # and must not be warned here.
            algo_name = (
                "GSPO" if self.importance_sampling_level == "sequence" else "GRPO"
            )
            self._warn_liger_non_token_is(self.importance_sampling_level, algo_name)
        batch_log_probs = self._get_logprobs(
            batch_ids,
            batch_size=batch_size,
            use_reference=False,
            eval_mode=False,
        )
        return self._loss_fn(
            batch_action_mask,
            batch_log_probs,
            batch_old_log_probs,
            batch_reference_log_probs,
            batch_advantages,
            batch_turn_ids,
            sampling_log_probs=batch_sampling_log_probs,
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
        * ``"sequence"`` → length-normalized masked mean over the whole
          completion ``(B, 1)`` (GSPO).
        * ``"turn"``     → length-normalized masked mean within each turn,
          scattered back to every token of that turn ``(B, T)``. Falls back to
          the sequence-level mean when ``turn_ids`` is ``None`` (a single
          turn is exactly sequence-level).

        All three forms feed an identical downstream surrogate/clip; the only
        difference is the granularity at which the ratio is pooled. The
        per-turn pool is the same geometric-mean (length-normalized) form as
        the sequence pool, just restricted to a turn's tokens.

        :param token_log_ratio: ``(B, T)`` per-token ``log pi_theta - log pi_old``.
        :param mask: ``(B, T)`` action-token mask.
        :param turn_ids: ``(B, T)`` turn index per token (``-1`` non-action) or
            ``None``.
        :param level: ``"token"`` / ``"turn"`` / ``"sequence"``.
        :return: ``(B, T)`` or ``(B, 1)`` log importance weights.
        """
        # Token level is the identity; ``turn_ids=None`` at turn level degenerates
        # to one sequence-wide turn, so route it through the sequence pooling.
        # Both reuse :func:`pool_log_ratio_by_level` (the same length-normalized
        # geometric-mean pooling shared by the non-Liger surrogate helpers).
        if level == "token":
            return token_log_ratio
        if level == "sequence" or turn_ids is None:
            log_importance_weights, _ = pool_log_ratio_by_level(
                token_log_ratio, mask, None, "sequence"
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

        The importance ratio is pooled to ``level`` (token/turn/sequence) via
        :meth:`_log_importance_weights`; everything downstream — the clipped
        ``min`` surrogate (``objective="grpo"``) or the clamped-weight x
        log-prob objective (``objective="cispo"``), the optional KL term, and
        the masked reduction — is shape-agnostic and identical across levels.

        :param mask: ``(B, T)`` action-token mask.
        :param log_probs: ``(B, T)`` current-policy log-probs.
        :param old_log_probs: ``(B, T)`` old-policy log-probs.
        :param reference_log_probs: ``(B, T)`` reference-policy log-probs.
        :param advantages: ``(B, 1)`` (per-trajectory) or ``(B, T)`` (per-turn,
            broadcast to tokens) advantages.
        :param turn_ids: ``(B, T)`` turn index per token, or ``None``.
        :param level: importance-sampling level.
        :param objective: ``"grpo"`` or ``"cispo"``.
        :return: Mean loss and mean KL divergence.
        """
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
        if not self.use_kl_advantage_shaping:
            loss = loss + self.beta * kl
        if sampling_log_probs is not None and self.vllm_importance_sampling_apply:
            # Truncated importance sampling: reweight each token by the
            # (detached, clamped) trainer/vLLM probability ratio to correct for
            # the rollout being drawn from vLLM rather than the trainer policy.
            # Broadcasts cleanly over (B, 1) sequence-level losses.
            with torch.no_grad():
                mask_f = mask.to(loss.dtype)
                is_ratio = torch.exp(
                    (old_log_probs - sampling_log_probs) * mask_f
                ).clamp(max=self.vllm_importance_sampling_cap)
            loss = loss * is_ratio
        loss = self._reduce_masked_loss(loss, mask)
        # Report the KL averaged over ACTION tokens only. ``kl`` is the k3
        # estimator over the full (B, T) frame; at masked (pad/prompt/non-action)
        # positions the policy and reference logprobs are meaningless and can
        # diverge by tens of nats, and ``exp(diff)`` in k3 then explodes the
        # naive ``kl.mean()`` to ~1e29. (The loss is already masked above; the
        # advantage-shaping path masks too — this keeps the *metric* consistent.)
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
        ``importance_sampling_level="turn"`` / ``"sequence"`` the importance
        ratio is pooled per turn / per sequence (the latter is GSPO).

        :param mask: Action-token mask.
        :param log_probs: Current-policy log-probs.
        :param old_log_probs: Old-policy log-probs.
        :param reference_log_probs: Reference-policy log-probs.
        :param advantages: ``(B, 1)`` or ``(B, T)`` advantages.
        :param turn_ids: ``(B, T)`` turn indices (required for turn level).
        :param sampling_log_probs: Optional ``(B, T-1)`` vLLM sampling logprobs
            for the sampling-mismatch correction.
        :return: Mean loss and mean KL divergence.
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
        """Calculate GSPO sequence-level ratio clipped loss."""
        return self._compute_policy_loss(
            mask,
            log_probs,
            old_log_probs,
            reference_log_probs,
            advantages,
            turn_ids,
            level="sequence",
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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate the loss using the Liger Triton-fused kernel.

        Dispatches to the appropriate Liger ``loss_type`` /
        ``importance_sampling_level`` from the resolved ``self.objective`` and
        ``self.importance_sampling_level``:

        * grpo  @ token    → ``loss_type="grpo"``,  ``importance_sampling_level="token"``
        * grpo  @ sequence → ``loss_type="grpo"``,  ``importance_sampling_level="sequence"`` (GSPO)
        * cispo @ token    → ``loss_type="cispo"``, ``importance_sampling_level="token"``

        Turn-level (and sequence-level CISPO) never reach here — ``_loss``
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
        :return: Mean loss and mean KL divergence (or clip-fraction when ``beta=0``).
        :rtype: tuple[torch.Tensor, torch.Tensor]
        """
        if not HAS_LIGER_KERNEL:
            msg = (
                "Liger loss was requested but `liger-kernel` is not available. "
                "Set use_liger_loss=False."
            )
            raise ImportError(msg)

        # Resolve Liger API parameters from the resolved objective + level.
        # ``_loss`` only routes here for Liger-supported combinations
        # (grpo @ token/sequence, cispo @ token); turn-level never reaches this.
        importance_sampling_level = self.importance_sampling_level
        if self.objective == "cispo":
            liger_loss_type = "cispo"
            importance_sampling_level = "token"
            # Liger CISPO clamps importance weights against an *absolute* upper
            # bound (epsilon_high = clip_coef_max), not an offset from 1.0.
            epsilon_low = 1.0 - self.clip_coef_min  # unused by Liger CISPO
            epsilon_high = self.clip_coef_max
        else:  # "grpo" objective (token or sequence/GSPO level)
            liger_loss_type = "grpo"
            epsilon_low = 1.0 - self.clip_coef_min
            epsilon_high = self.clip_coef_max - 1.0
            # The GSPO (sequence-level) not-memory-bounded warning is emitted
            # once up front in ``__init__`` via ``_warn_liger_non_token_is``; no
            # duplicate is needed here.

        batch_ids = batch_ids.to(self.device)
        mask = action_mask.to(self.device).contiguous()  # (B, seq_len-1)
        # Normalise a trailing singleton ``(B, 1) -> (B,)`` for the kernel, but
        # never squeeze a 1-D ``(B,)``: a single-sample minibatch arrives as
        # ``(1,)`` and ``squeeze(-1)`` would collapse it to a 0-dim scalar,
        # which the token-level shape detection below then rejects.
        adv = advantages.to(self.device).contiguous()
        if adv.dim() > 1 and adv.shape[-1] == 1:
            adv = adv.squeeze(-1)  # (B, 1) -> (B,)
        old_log_probs = old_log_probs.to(self.device).contiguous()
        reference_log_probs = (
            reference_log_probs.to(self.device).contiguous()
            if self.beta != 0.0
            else None
        )
        lm_head = self._get_lm_head()
        lm_head_weight = lm_head.weight
        lm_head_bias = lm_head.bias

        attention_mask = (batch_ids != self.pad_token_id).long()
        # Sequence packing (the same gate as the standard path): when a
        # varlen/block-sparse backend is active, flatten the minibatch's real
        # tokens into a single padding-free row for the transformer forward,
        # then scatter the resulting hidden states back onto the padded
        # ``(B, T, H)`` frame. The Liger kernel call below is then byte-for-byte
        # the padded path — only how ``policy_hidden`` is produced changes, so
        # both token-level (grpo/cispo) and sequence-level (GSPO) benefit from
        # the cheaper forward. Dense backends return ``None`` here and fall back
        # to the padded forward, exactly as ``_get_logprobs`` does.
        packing_mode = self._packing_mode()
        packed = None
        if packing_mode is not None:
            packed = pack_padded_batch(batch_ids, attention_mask)
            # Hand the model only the per-sequence ``position_ids`` (reset per
            # segment) with ``attention_mask=None``: transformers detects the
            # packed format and AND-composes a block-diagonal constraint onto
            # each layer's native mask (FA2 → cu_seqlens + per-layer window_size;
            # flex → sparse block-diagonal BlockMask + window on sliding layers).
            # No token attends across sequences and sliding-window attention is
            # preserved per layer, so packing is correct for SWA models too.
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
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)
                model_kwargs["position_ids"] = position_ids
        # Identity-patch lm_head so the actor forward outputs the last hidden
        # state (B, T, H) directly instead of materializing (B, T, V) logits
        # only to discard them. lm_head_weight is passed separately to
        # LigerFusedLinearGRPOFunction which handles the matmul and its grad.
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
        target_ids = batch_ids[:, 1:].contiguous()  # (B, seq_len-1)

        # Token-chunk the fused-linear loss for token-level importance sampling
        # (CISPO / GRPO). The upstream Liger function chunks dim 0 (the batch),
        # so a single long trajectory (B=1) materialises the whole
        # ``(seq_len, vocab)`` logits matrix in one chunk and OOMs at long
        # context (gemma's vocab is 262k). Flattening ``(B, T, H) -> (B*T, 1, H)``
        # makes it chunk *tokens* instead, bounding each chunk's logits to
        # ``(chunk_tokens, vocab)``. This is exact for token-level IS with the
        # global cispo/dapo normaliser (validated: bit-identical to the
        # non-chunked loss). Sequence-level IS (GSPO) is not token-independent,
        # so it keeps the batch path.
        # Both importance-sampling levels feed the *same* fused kernel; only the
        # input layout and the chunk granularity differ. The token level flattens
        # ``(B, T, H) -> (B*T, 1, H)`` (and the matching ``(B, T) -> (B*T, 1)``
        # tensors) so the kernel chunks over *tokens* — bounding each chunk's
        # logits to ``(token_chunk_size, vocab)``; the non-token (sequence/GSPO)
        # path keeps the padded ``(B, T, ...)`` layout and chunks one whole
        # sequence at a time. ``policy_arg`` / ``target_ids_arg`` / ``mask_arg`` /
        # ``old_lp_arg`` / ``ref_lp_arg`` / ``adv_arg`` and ``chunk_size`` are the
        # only positional args that vary; everything else is written once below.
        if importance_sampling_level == "token":
            batch, _seq_len, hidden_dim = policy_hidden.shape
            n_act = target_ids.shape[1]  # seq_len - 1
            n_tokens = batch * n_act
            # ``adv`` arrives in one of three shapes depending on
            # ``advantage_granularity`` upstream:
            #   * ``(batch,)``           — trajectory-level (one scalar per
            #     completion); broadcast to every action token.
            #   * ``(batch, 1)``         — same, already a column vector.
            #   * ``(batch, n_act)``     — already per-token (turn-level
            #     ``advantage_granularity`` broadcasts the per-turn advantage to
            #     tokens via ``turn_advantages.gather(1, turn_ids)``).
            # Detect and flatten to ``(n_tokens,)`` for the token-flatten Liger
            # call below.
            if adv.ndim == 1 and adv.shape[0] == batch:
                adv_arg = adv.unsqueeze(1).expand(batch, n_act).reshape(n_tokens)
            elif adv.ndim == 2 and adv.shape == (batch, 1):
                adv_arg = adv.expand(batch, n_act).reshape(n_tokens)
            elif adv.ndim == 2 and adv.shape == (batch, n_act):
                adv_arg = adv.reshape(n_tokens)
            else:
                msg = (
                    f"Unexpected advantage shape {tuple(adv.shape)} for the "
                    f"Liger token-level loss; expected (batch={batch},), "
                    f"(batch, 1), or (batch, n_act={n_act}) — got "
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
                reference_log_probs.reshape(n_tokens, 1)
                if reference_log_probs is not None
                else None
            )
            # Tokens per chunk: bounds the transient (chunk_tokens, vocab) logits.
            # Prefers the constructor / INIT_HP value, falling back to the legacy
            # ``AGILERL_LIGER_TOKEN_CHUNK`` env var (default 2048).
            chunk_size = self._resolve_liger_token_chunk()
        else:
            # Sequence-level (GSPO): keep the padded layout and one-sequence-per-
            # chunk granularity (chunk_size=1 over the batch dim).
            policy_arg = policy_hidden
            target_ids_arg = target_ids
            mask_arg = mask
            old_lp_arg = old_log_probs
            ref_lp_arg = reference_log_probs
            adv_arg = adv
            chunk_size = 1

        loss, aux = LigerFusedLinearGRPOFunction.apply(
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
            reference_log_probs is not None,  # use_ref_model
            chunk_size,
            None,
        )

        kl = aux[0]
        return loss.mean(), kl

    # Backward-compatible alias kept for any external callers.
    _grpo_loss_liger = _liger_loss


def _signatures_without_loss_type() -> tuple[Signature, Signature]:
    """Build class and ``__init__`` signatures without ``loss_type``."""
    grpo_sig = signature(GRPO.__init__)
    class_params = [
        param
        for param in grpo_sig.parameters.values()
        if param.name not in {"self", "loss_type"}
    ]
    init_params = [
        param for param in grpo_sig.parameters.values() if param.name != "loss_type"
    ]
    return (
        grpo_sig.replace(parameters=class_params),
        grpo_sig.replace(parameters=init_params),
    )
