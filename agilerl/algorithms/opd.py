# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""On-policy distillation built on GRPO's importance-sampled policy loss."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar

import torch

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


@dataclass(frozen=True)
class TeacherRow:
    """Which teacher scored one rollout, and the signals logged beside the loss."""

    name: str | None = None
    domain: str | None = None
    score_s: float | None = None
    response_tokens: int | None = None


@inherit_init_signature(
    GRPO,
    defaults=DEFAULTS,
    extra={"teacher_coef": 1.0, "teacher_gap_clip": 10.0},
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
    Group advantages are not whitened or filtered, and a group of one is allowed.

    :param teacher_coef: Weight of the teacher term on the advantage, > 0.
    :type teacher_coef: float
    :param teacher_gap_clip: Symmetric bound, in nats, on each token's
        teacher-minus-sampling gap; ``None`` leaves it unbounded.
    :type teacher_gap_clip: float | None
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
        self.metrics.register("teacher_reverse_kl")
        self.metrics.register("teacher_gap_clip_frac")

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
        teacher_rows: list[TeacherRow] | None = None,
    ) -> dict[str, float | list[int]]:
        """Update the policy toward the teacher; see :meth:`GRPO.learn`.

        :param token_advantages: Optional per-row flat advantage terms added
            on top of the teacher term, as in :meth:`GRPO.learn`.
        :type token_advantages: list[torch.Tensor | None] | None
        :param teacher_logps: Per-row flat teacher log-probs of the generated
            tokens, parallel to ``sampling_logps``, on every action token.
        :type teacher_logps: list[torch.Tensor | None] | None
        :param teacher_rows: Per-row teacher name, domain, scoring time and
            response length. ``None`` leaves the global teacher metrics only.
        :type teacher_rows: list[TeacherRow] | None
        :return: :meth:`GRPO.learn`'s metrics plus ``teacher_reverse_kl`` (the
            unbounded mean per-token reverse KL) and ``teacher_gap_clip_frac``.
            A labelled row also logs those means per teacher and per domain.
        :rtype: dict[str, float | list[int]]
        :raises ValueError: If a row lacks teacher or sampling log-probs,
            either is not finite, or ``teacher_rows`` does not match the rows.
        """
        gaps = self._teacher_gaps(teacher_logps, sampling_logps)
        if teacher_rows is not None and len(teacher_rows) != len(gaps):
            msg = (
                "teacher_rows must match the rollout rows; got "
                f"{len(teacher_rows)} labels and {len(gaps)} rows."
            )
            raise ValueError(msg)
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
        stats: dict[str, float] = {
            "teacher_reverse_kl": -float(flat.mean()) if flat.numel() else 0.0,
            "teacher_gap_clip_frac": (
                0.0
                if self.teacher_gap_clip is None or not flat.numel()
                else float((flat.abs() > self.teacher_gap_clip).float().mean())
            ),
        }
        stats.update(self._labelled_teacher_metrics(gaps, sampling_logps, teacher_rows))
        for key, value in stats.items():
            self._log_teacher_metric(key, value)
        return {**result, **stats}

    def _log_teacher_metric(self, name: str, value: float) -> None:
        """Log a scalar, registering a per-teacher or per-domain name on first use."""
        if name not in self.metrics.additional_metrics:
            self.metrics.register(name)
        self.metrics.log(name, value)

    @staticmethod
    def _labelled_teacher_metrics(
        gaps: list[torch.Tensor],
        sampling_logps: list[torch.Tensor | None] | None,
        teacher_rows: list[TeacherRow] | None,
    ) -> dict[str, float]:
        """Token-mean reverse KL, scoring time, response length and token NLL by label.

        :param gaps: Per-row teacher-minus-sampling log-probs.
        :type gaps: list[torch.Tensor]
        :param sampling_logps: Per-row sampling log-probs, parallel to ``gaps``.
        :type sampling_logps: list[torch.Tensor | None] | None
        :param teacher_rows: One label per row, or ``None``.
        :type teacher_rows: list[TeacherRow] | None
        :return: Metric names to values. Empty when no row is labelled.
        :rtype: dict[str, float]
        :raises ValueError: If the label count differs from the row count.
        """
        if teacher_rows is None:
            return {}
        if len(teacher_rows) != len(gaps):
            msg = (
                "teacher_rows must match the rollout rows; got "
                f"{len(teacher_rows)} labels and {len(gaps)} rows."
            )
            raise ValueError(msg)
        sampling_rows = sampling_logps or []
        kl_sum: dict[str, float] = {}
        kl_count: dict[str, int] = {}
        nll_sum: dict[str, float] = {}
        nll_count: dict[str, int] = {}
        score_s: dict[str, list[float]] = {}
        response_tokens: dict[str, list[float]] = {}
        for gap, row, sampling in zip(gaps, teacher_rows, sampling_rows, strict=True):
            labels = (
                (f"teacher/{_metric_label(row.name)}", row.name),
                (f"domain/{_metric_label(row.domain)}", row.domain),
            )
            for key, label in labels:
                if label is None:
                    continue
                kl_sum[key] = kl_sum.get(key, 0.0) + float(gap.sum())
                kl_count[key] = kl_count.get(key, 0) + int(gap.numel())
                if sampling is not None:
                    nll_sum[key] = nll_sum.get(key, 0.0) - float(sampling.sum())
                    nll_count[key] = nll_count.get(key, 0) + int(sampling.numel())
            if row.name is not None and row.score_s is not None:
                score_s.setdefault(f"teacher/{_metric_label(row.name)}", []).append(
                    row.score_s
                )
            if row.domain is not None and row.response_tokens is not None:
                response_tokens.setdefault(
                    f"domain/{_metric_label(row.domain)}", []
                ).append(float(row.response_tokens))
        metrics = {
            f"{key}/reverse_kl": -kl_sum[key] / kl_count[key]
            for key in kl_sum
            if kl_count[key]
        }
        metrics.update(
            {
                f"{key}/token_nll": nll_sum[key] / nll_count[key]
                for key in nll_sum
                if nll_count[key]
            }
        )
        metrics.update(
            {
                f"{key}/score_s": sum(values) / len(values)
                for key, values in score_s.items()
            }
        )
        metrics.update(
            {
                f"{key}/response_tokens": sum(values) / len(values)
                for key, values in response_tokens.items()
            }
        )
        return metrics

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


def _metric_label(label: str | None) -> str:
    """A metric-path segment. Slashes would add another level under the prefix."""
    return "" if label is None else label.replace("/", "_")
