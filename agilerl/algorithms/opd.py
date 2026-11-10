# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""On-policy distillation built on GRPO's importance-sampled policy loss."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, ClassVar

import torch

from agilerl.algorithms.checkpoint_pool import kept_rows, locality_token_mask
from agilerl.algorithms.grpo import GRPO
from agilerl.utils.algo_utils import inherit_init_signature

if TYPE_CHECKING:
    from collections.abc import Sequence

    from agilerl.components.llm_rollout_data import EpisodeSegments
    from agilerl.typing import LLMRolloutExperiences

DEFAULTS: dict[str, Any] = {
    "temperature": 1.0,
    "top_k": 0,
    "beta": 0.0,
    "group_size": 4,
}
"""Pure-distillation defaults: the unmodified policy distribution and no reference."""


@inherit_init_signature(
    GRPO,
    defaults=DEFAULTS,
    extra={"teacher_coef": 1.0, "teacher_gap_clip": 10.0, "locality_filter": False},
)
class OPD(GRPO):
    """On-policy distillation: the student samples, a teacher scores its tokens.

    Each action token's advantage gains ``teacher_coef * (teacher - sampling)``
    log-probs, bounded to ``teacher_gap_clip`` nats: the negative per-token
    reverse KL to the teacher that scored the rollout. Group-relative task
    advantages still apply; with no task reward they are zero and the update is
    pure distillation. The teacher log-probs arrive with each rollout, as
    ``teacher_logps`` to :meth:`learn`; the teacher may be the live policy given
    a reference trace (self-distillation) or a separate model.

    Sampling must use the unmodified policy (temperature 1, no top-k / top-p /
    min-p / repetition penalty), since the teacher scores that distribution.
    Group advantages are not whitened or filtered, and a group of one is allowed;
    with no task reward a group adds samples per prompt but no contrast.

    Each learn logs the mean response length and the share of repeated 4-grams
    in each response, and warns once when over half of them repeat: looping,
    lengthening responses are how a student collapses under a teacher it cannot
    follow.

    :param teacher_coef: Weight of the teacher term on the advantage, > 0.
    :type teacher_coef: float
    :param teacher_gap_clip: Symmetric bound, in nats, on each token's
        teacher-minus-sampling gap; ``None`` leaves it unbounded.
    :type teacher_gap_clip: float | None
    :param locality_filter: Keep a rollout only when the teacher solves its
        problem more often and the student failed it. Task rewards are then
        zeroed, so the update is the teacher term on the kept rows.
    :type locality_filter: bool
    """

    requires_group_contrast: ClassVar[bool] = False

    def __init__(
        self,
        *args: Any,
        teacher_coef: float = 1.0,
        teacher_gap_clip: float | None = 10.0,
        **kwargs: Any,
    ) -> None:
        """Validate the distillation settings, then build the GRPO agent."""
        for name, value in DEFAULTS.items():
            kwargs.setdefault(name, value)
        locality_filter = bool(kwargs.pop("locality_filter", False))
        if teacher_coef <= 0.0:
            msg = f"teacher_coef must be > 0, got {teacher_coef}."
            raise ValueError(msg)
        if teacher_gap_clip is not None and teacher_gap_clip <= 0.0:
            msg = f"teacher_gap_clip must be > 0 or None, got {teacher_gap_clip}."
            raise ValueError(msg)
        if kwargs.get("whiten_advantages") or kwargs.get("filter_zero_adv"):
            msg = (
                "OPD adds a per-token teacher term after the group advantages; "
                "whiten_advantages and filter_zero_adv act on the group "
                "advantages alone, so leave both False."
            )
            raise ValueError(msg)
        plain_sampling = {
            "temperature": 1.0,
            "top_k": 0,
            "top_p": 1.0,
            "min_p": 0.0,
            "repetition_penalty": 1.0,
        }
        sampling = {
            name: kwargs.get(name, plain_sampling[name]) for name in plain_sampling
        }
        if sampling != plain_sampling:
            msg = (
                "OPD compares the student's sampling log-probs with the "
                "teacher's, so sampling must use the unmodified policy: "
                "temperature 1, top_k 0, top_p 1, min_p 0, repetition_penalty 1. "
                f"Got {sampling}."
            )
            raise ValueError(msg)
        super().__init__(*args, **kwargs)
        self.teacher_coef = teacher_coef
        self.teacher_gap_clip = teacher_gap_clip
        self.locality_filter = locality_filter
        self.metrics.register("teacher_reverse_kl")
        if locality_filter:
            for name in (
                "rollout_kept_frac",
                "pool_miss_coverage",
                "active_teacher_step",
                "student_held_out_score",
            ):
                self.metrics.register(name)
        self.metrics.register("teacher_gap_clip_frac")
        self.metrics.register("response_length")
        self.metrics.register("response_repeat_4gram_frac")
        self._warned_repetition = False

    def learn(
        self,
        experiences: LLMRolloutExperiences,
        turn_ids: torch.Tensor | None = None,
        sampling_logps: list[torch.Tensor | None] | None = None,
        pixel_values: torch.Tensor | None = None,
        pixel_image_counts: Sequence[int] | None = None,
        episode_segments: list[EpisodeSegments | None] | None = None,
        image_token_id: int | None = None,
        sampling_supports: list[torch.Tensor | None] | None = None,
        token_advantages: list[torch.Tensor | None] | None = None,
        teacher_logps: list[torch.Tensor | None] | None = None,
        teacher_rates: Sequence[float] | None = None,
        pool_metrics: dict[str, float] | None = None,
    ) -> dict[str, float | list[int]]:
        """Update the policy toward the teacher; see :meth:`GRPO.learn`.

        :param token_advantages: Optional per-row flat advantage terms added
            on top of the teacher term, as in :meth:`GRPO.learn`.
        :type token_advantages: list[torch.Tensor | None] | None
        :param teacher_logps: Per-row flat teacher log-probs of the generated
            tokens, parallel to ``sampling_logps``, on every action token.
        :type teacher_logps: list[torch.Tensor | None] | None
        :param teacher_rates: One teacher success rate per row. Required when
            ``locality_filter`` is set.
        :type teacher_rates: Sequence[float] | None
        :param pool_metrics: Checkpoint-pool counters to log with this step.
        :type pool_metrics: dict[str, float] | None
        :return: :meth:`GRPO.learn`'s metrics plus ``teacher_reverse_kl`` (the
            unbounded mean per-token reverse KL), ``teacher_gap_clip_frac``,
            ``response_length`` (mean action tokens per row) and
            ``response_repeat_4gram_frac``.
        :rtype: dict[str, float | list[int]]
        :raises ValueError: If a row lacks teacher or sampling log-probs, or
            either is not finite. With ``locality_filter``, also when the
            teacher rates are missing or no failed rollout is kept.
        """
        gaps = self._teacher_gaps(teacher_logps, sampling_logps)
        kept: list[bool] | None = None
        if self.locality_filter:
            kept = self._kept_rows(experiences, teacher_rates)
            gaps = [
                gap if keep else torch.zeros_like(gap)
                for gap, keep in zip(gaps, kept, strict=True)
            ]
            rewards = experiences[2]
            experiences = (
                experiences[0],
                experiences[1],
                torch.zeros_like(rewards),
                *experiences[3:],
            )
        flat = torch.cat(gaps)
        bounded = [
            gap
            if self.teacher_gap_clip is None
            else gap.clamp(-self.teacher_gap_clip, self.teacher_gap_clip)
            for gap in gaps
        ]
        result = super().learn(
            experiences,
            turn_ids=turn_ids,
            sampling_logps=sampling_logps,
            pixel_values=pixel_values,
            pixel_image_counts=pixel_image_counts,
            episode_segments=episode_segments,
            image_token_id=image_token_id,
            sampling_supports=sampling_supports,
            token_advantages=[
                self.teacher_coef * gap
                if extra is None
                else self.teacher_coef * gap + extra
                for gap, extra in zip(
                    bounded, token_advantages or [None] * len(bounded), strict=True
                )
            ],
        )
        repeat_frac = _repeated_ngram_frac(experiences[0], experiences[1])
        kl_rows = flat
        if kept is not None:
            kl_rows = torch.cat(
                [gap for gap, keep in zip(gaps, kept, strict=True) if keep]
            )
        stats: dict[str, float] = {
            "teacher_reverse_kl": -float(kl_rows.mean()) if kl_rows.numel() else 0.0,
            "teacher_gap_clip_frac": (
                0.0
                if self.teacher_gap_clip is None or not flat.numel()
                else float((flat.abs() > self.teacher_gap_clip).float().mean())
            ),
            "response_length": float(sum(gap.numel() for gap in gaps)) / len(gaps),
            "response_repeat_4gram_frac": repeat_frac,
        }
        if kept is not None:
            stats["rollout_kept_frac"] = sum(kept) / len(kept)
            stats.update(pool_metrics or {})
        self._check_repetition(repeat_frac)
        for key, value in stats.items():
            self.metrics.log(key, value)
        return {**result, **stats}

    def _kept_rows(
        self,
        experiences: LLMRolloutExperiences,
        teacher_rates: Sequence[float] | None,
    ) -> list[bool]:
        """Rows the problem filter and the failed-rollout filter both keep.

        :param experiences: The batch; rewards are ``experiences[2]``.
        :type experiences: LLMRolloutExperiences
        :param teacher_rates: One teacher success rate per row.
        :type teacher_rates: Sequence[float] | None
        :return: ``True`` for each row the teacher term should train.
        :rtype: list[bool]
        :raises ValueError: If the rates are missing, or no failed rollout
            with an action token is kept.
        """
        if teacher_rates is None:
            msg = (
                "locality_filter needs teacher_rates: the teacher's empirical "
                "success rate on each rollout's problem."
            )
            raise ValueError(msg)
        rewards = experiences[2].detach().float()
        row_rewards = [
            float(value) for value in rewards.reshape(rewards.shape[0], -1).sum(dim=1)
        ]
        kept = kept_rows(row_rewards, list(teacher_rates), int(self.group_size))
        masks = experiences[1]
        if not any(
            locality_token_mask(masks[index], keep).any()
            for index, keep in enumerate(kept)
        ):
            msg = (
                "Locality filters kept no rollouts. The teacher success rate "
                "was not higher than the student's on any problem with a "
                "failed rollout."
            )
            raise ValueError(msg)
        return kept

    def _check_repetition(self, repeat_frac: float) -> None:
        """Warn once when over half of the responses' 4-grams repeat.

        :param repeat_frac: This learn's mean share of repeated 4-grams.
        :type repeat_frac: float
        """
        if self._warned_repetition or repeat_frac <= 0.5:
            return
        self._warned_repetition = True
        warnings.warn(
            f"{repeat_frac:.0%} of OPD responses' 4-grams repeat. Looping, "
            "lengthening responses are how a student collapses under a teacher "
            "it cannot follow; check samples and response_length, or use a "
            "teacher closer to the student.",
            stacklevel=3,
        )

    @staticmethod
    def _teacher_gaps(
        teacher_logps: list[torch.Tensor | None] | None,
        sampling_logps: list[torch.Tensor | None] | None,
    ) -> list[torch.Tensor]:
        """Per-row teacher minus sampling log-probs.

        :param teacher_logps: Per-row flat teacher log-probs.
        :type teacher_logps: list[torch.Tensor | None] | None
        :param sampling_logps: Per-row flat sampling log-probs.
        :type sampling_logps: list[torch.Tensor | None] | None
        :return: One float32 gap per row.
        :rtype: list[torch.Tensor]
        :raises ValueError: If a row lacks either log-prob, their lengths
            differ, or either is not finite.
        """
        teacher_rows = teacher_logps or []
        sampling_rows = sampling_logps or []
        if len(teacher_rows) != len(sampling_rows) or not teacher_rows:
            msg = (
                "OPD needs teacher_logps and sampling_logps for every row; got "
                f"{len(teacher_rows)} and {len(sampling_rows)} rows."
            )
            raise ValueError(msg)
        gaps = []
        for row, (teacher, sampling) in enumerate(
            zip(teacher_rows, sampling_rows, strict=True)
        ):
            if teacher is None or sampling is None:
                missing = "teacher_logps" if teacher is None else "sampling_logps"
                msg = f"OPD needs {missing} on every action token; row {row} has none."
                raise ValueError(msg)
            if teacher.numel() != sampling.numel():
                msg = (
                    f"Row {row} has {teacher.numel()} teacher_logps and "
                    f"{sampling.numel()} sampling_logps."
                )
                raise ValueError(msg)
            gap = teacher.reshape(-1).float() - sampling.reshape(-1).float()
            if not torch.isfinite(gap).all():
                msg = f"teacher_logps or sampling_logps are not finite on row {row}."
                raise ValueError(msg)
            gaps.append(gap)
        return gaps


def _repeated_ngram_frac(
    token_ids: list[torch.Tensor] | torch.Tensor,
    action_masks: list[torch.Tensor] | torch.Tensor,
) -> float:
    """Mean share, over rows, of each response's 4-grams that already appeared in it.

    :param token_ids: Per-row token ids.
    :type token_ids: list[torch.Tensor] | torch.Tensor
    :param action_masks: Per-row masks; position ``t`` marks token ``t + 1``.
    :type action_masks: list[torch.Tensor] | torch.Tensor
    :return: The mean over rows with at least one 4-gram; 0 when none has one.
    :rtype: float
    """
    fracs = []
    for ids, mask in zip(token_ids, action_masks, strict=True):
        flat_mask = mask.reshape(-1).to(torch.bool)
        response = ids.reshape(-1)[1 : flat_mask.numel() + 1][flat_mask]
        if response.numel() < 4:
            continue
        grams = response.unfold(0, 4, 1)
        fracs.append(1.0 - torch.unique(grams, dim=0).shape[0] / grams.shape[0])
    return sum(fracs) / len(fracs) if fracs else 0.0
