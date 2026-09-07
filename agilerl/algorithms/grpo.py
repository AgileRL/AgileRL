# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import functools
import inspect
import warnings
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Protocol

import numpy.typing as npt
import torch

from agilerl import HAS_LIGER_KERNEL, HAS_LLM_DEPENDENCIES

if HAS_LIGER_KERNEL or TYPE_CHECKING:
    from liger_kernel.chunked_loss.grpo_loss import LigerFusedLinearGRPOFunction
else:
    # Keep the name resolvable when liger-kernel isn't installed so unit
    # tests can patch it. ``_liger_loss`` guards against actual use.
    LigerFusedLinearGRPOFunction = None  # type: ignore[assignment]

from agilerl.algorithms.configs import GRPOObjective, GRPOSetup, PopulationIndex
from agilerl.algorithms.core import ActionResult, LLMAlgorithm
from agilerl.algorithms.core.llm_init import named_llm_setup
from agilerl.algorithms.core.registry import NetworkGroup
from agilerl.algorithms.grpo_advantage import GRPOAdvantageMixin
from agilerl.algorithms.grpo_learn import GRPOLearnMixin
from agilerl.protocols import (
    PeftModelProtocol,
    PreTrainedModelProtocol,
)
from agilerl.typing import LLMObsType, LLMRolloutExperiences
from agilerl.utils.algo_utils import (
    get_experiences_samples,
)
from agilerl.utils.llm_packing import (
    pack_padded_batch,
    unpack_hidden_states,
)
from agilerl.utils.llm_utils import (
    attention_mask_from_padded_ids,
    build_completion_mask,
    calculate_k3_kl,
    fill_outside_mask,
    masked_mean,
    normalize_prompt_batch,
    pool_log_ratio_by_level,
    prepare_prompt_hf_generate,
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


class GRPO(
    GRPOLearnMixin,
    GRPOAdvantageMixin,
    LLMAlgorithm[LLMRolloutExperiences],
):
    """Group Relative Policy Optimization (GRPO).

    Paper: https://arxiv.org/pdf/2402.03300

    :param llm: Model, generation, and training setup.
    :type llm: GRPOSetup
    :param objective: Group-relative objective. Defaults to :class:`GRPOObjective`.
    :type objective: GRPOObjective, optional
    :param member: Population index and mutation bookkeeping.
    :type member: PopulationIndex, optional
    """

    _window_action_tokens: int | None = None
    """Action tokens of this rank's samples entering the optimizer step in progress."""

    _mini_batch_size_default = "micro_batch"

    def __init__(
        self,
        llm: GRPOSetup,
        objective: GRPOObjective | None = None,
        member: PopulationIndex | None = None,
    ) -> None:
        objective = objective or GRPOObjective()
        member = member or PopulationIndex()
        super().__init__(named_llm_setup(llm, "GRPO"), member)
        self._bind_grpo(llm, objective)

    def _bind_grpo(self, llm: GRPOSetup, objective: GRPOObjective) -> None:
        """Bind GRPO objective, generation, and actor networks."""
        train = llm.train
        gen = llm.generation
        model = llm.model
        self._validate_core_args(
            train.batch_size, train.lr, objective.update_epochs, model.actor_network
        )
        self.clip_coef, self.clip_coef_min, self.clip_coef_max = (
            self._resolve_clip_coef(objective.clip_coef)
        )
        self.update_epochs = objective.update_epochs
        self.beta = objective.beta
        self.temperature = gen.temperature
        self.repetition_penalty = gen.repetition_penalty
        self.top_p = gen.top_p
        self.top_k = gen.top_k
        self.min_p = gen.min_p
        self.loss_norm = self._resolve_loss_norm(objective.loss_norm)
        self._setup_advantage_options(
            objective.adv_norm,
            objective.group_size,
            objective.advantage_granularity,
            objective.action_granularity,
            objective.whiten_advantages,
            objective.adv_clip_range,
            objective.filter_zero_adv,
            objective.adv_filter_eps,
            objective.turn_advantage_trajectory_fallback,
        )
        self._setup_objective(
            objective.loss_type,
            objective.importance_sampling_level,
            objective.use_kl_advantage_shaping,
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
        prompts = normalize_prompt_batch(obs)
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
                            if training and group_size > 1:
                                input_ids = input_ids.repeat(group_size, 1)
                                attention_mask = attention_mask.repeat(group_size, 1)
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
                    prompts,
                    group_size,
                    temperature=self.temperature
                    if training
                    else 0.01,  # Almost deterministic for evaluation
                    capture_sampling_logps=capture_sampling_logps,
                )

        return ActionResult(token_ids_list, completion_masks, sampling_logps)

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
        if (
            self.advantage_granularity == "turn"
            and self.importance_sampling_level == "trajectory"
        ):
            warnings.warn(
                "advantage_granularity='turn' with "
                "importance_sampling_level='trajectory' applies one "
                "completion-level ratio to per-turn advantages. Set "
                "advantage_granularity='trajectory' to match, or use token/turn "
                "importance sampling to keep turn advantages.",
                stacklevel=3,
            )
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
        token_ids: torch.Tensor,
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
            token_ids,
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
