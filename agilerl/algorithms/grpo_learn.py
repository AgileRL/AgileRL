# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""GRPO learn loop, batch prep, and gradient-accumulation windows."""

from __future__ import annotations

import gc
import warnings

import numpy as np
import numpy.typing as npt
import torch

from agilerl.typing import LLMRolloutExperiences
from agilerl.utils.algo_utils import stack_and_pad_experiences
from agilerl.utils.llm_utils import (
    aggregate_metrics_dict,
    allreduce_minmax_int,
    needs_cross_rank_seq_padding,
)


class GRPOLearnMixin:
    """Optimizer-step loop for GRPO / GSPO / CISPO."""

    def learn(
        self,
        experiences: LLMRolloutExperiences,
        turn_ids: torch.Tensor | None = None,
        sampling_logps: list[torch.Tensor | None] | None = None,
    ) -> dict[str, float]:
        """Update agent network parameters to learn from experiences.

        :param experiences: ``(token_ids, action_masks, rewards)`` stacked
            batch. For ``importance_sampling_level="turn"`` with per-turn
            rewards, ``rewards`` is ``(batch, max_turns)``; otherwise it is one
            scalar per trajectory (per-turn rewards are summed to the episode
            return).
        :type experiences: LLMRolloutExperiences
        :param sampling_logps: Optional per-row flat vLLM sampling logprobs (one
            1-D tensor per trajectory, generated tokens only; concatenated
            across turns for multi-turn) for the sampling-mismatch correction.
            Parallel to the stacked ``token_ids`` rows. ``None`` disables
            the correction for this update.
        :type sampling_logps: list[torch.Tensor | None] | None
        :param turn_ids: ``(batch, seq_len-1)`` turn index per action token
            (``-1`` for non-action tokens), aligned with the action mask.
            Required when the resolved advantage granularity is ``"turn"``
            (per-turn group-relative advantages need per-turn rewards). Also consumed by
            turn-level importance-ratio pooling when
            ``importance_sampling_level="turn"``. Ignored when neither applies.
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
            token_ids, action_masks, rewards, turn_ids = self._prepare_experience_batch(
                experiences, turn_ids
            )
            num_samples = token_ids.shape[0]
            world_size = (
                self.accelerator.num_processes if self.accelerator is not None else 1
            )
            if (
                needs_cross_rank_seq_padding(self, world_size=world_size)
                and self.accelerator is not None
                and self.accelerator.num_processes > 1
            ):
                seq_len = token_ids.shape[1]
                min_t, max_t = allreduce_minmax_int(seq_len, self.accelerator)
                if min_t != max_t:
                    msg = (
                        "Cross-rank completion sequence length mismatch before "
                        f"GRPO learn: min_t={min_t}, max_t={max_t}. Ranks must "
                        "pad completions to the same T before learn()."
                    )
                    raise RuntimeError(msg)

            advantages, batch_idxs = self._calculate_advantages(
                rewards, token_ids, action_masks, turn_ids
            )
            aux_metric = self.aux_metric_name
            effective_num_samples = len(batch_idxs)
            if effective_num_samples == 0:
                # Single-process only: multi-process filtering masks advantages
                # instead of dropping samples, so every rank always enters the
                # update loop with the same number of micro-batches.
                warnings.warn(
                    "All samples were filtered by advantage threshold; skipping GRPO update.",
                    stacklevel=2,
                )
                return {"loss": 0.0, aux_metric: 0.0}

            updates = 0
            batch_size = (
                min(num_samples, self.micro_batch_size_per_gpu)
                if hasattr(self, "micro_batch_size_per_gpu")
                else num_samples
            )
            with torch.no_grad():
                reference_log_probs, old_log_probs, _ = self._fused_forward_no_grad(
                    token_ids,
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
                            token_ids,
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
        token_ids_list = experiences[0]
        result["completion_length"] = float(
            np.mean([x.shape[-1] for x in token_ids_list])
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
    def _prepare_experience_batch(
        self,
        experiences: LLMRolloutExperiences,
        turn_ids: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Stack and pad the experience batch and move it to the device."""
        token_ids, action_masks, rewards = stack_and_pad_experiences(
            *experiences,
            padding_values=[self.pad_token_id, False, None],
        )
        action_masks = action_masks.to(self.device)
        rewards = rewards.to(self.device).float()
        token_ids = token_ids.to(self.device)
        if turn_ids is not None:
            turn_ids = turn_ids.to(self.device)
            if turn_ids.shape[0] != token_ids.shape[0]:
                msg = (
                    f"turn_ids batch ({turn_ids.shape[0]}) must match "
                    f"completion batch ({token_ids.shape[0]})."
                )
                raise ValueError(msg)
        return token_ids, action_masks, rewards, turn_ids
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
