# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for GRPO loss normalization and fused-path activation offload.

Covers ``loss_norm="accumulation_window"`` on the standard and the fused Liger
path, the ``num_items_in_batch`` plumbing the fused reduction needs, and
``activation_offload`` reaching the fused training forward. Pure CPU: the fused
kernel is a stand-in mirroring liger's signature and both of its reductions.
"""

from __future__ import annotations

import inspect
from contextlib import nullcontext
from types import SimpleNamespace
from typing import Any

import pytest
import torch

pytest.importorskip("transformers", reason="LLM tests require transformers.")
pytest.importorskip("peft", reason="LLM tests require peft.")

import numpy as np

from agilerl.algorithms import grpo as grpo_module
from agilerl.algorithms.core.base import LLMAlgorithm
from agilerl.algorithms.grpo import GRPO

CLIP_MIN = 0.8
CLIP_MAX = 1.2
VOCAB = 5
HIDDEN = 3
KERNEL_NAME = "LigerFusedLinearGRPOFunction"
TOKEN_COUNT_LOSS_TYPES = ("dapo", "cispo", "vespo")


def _world_size() -> int:
    """Ranks the fused kernel divides its token-count normalizer by."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return int(torch.distributed.get_world_size())
    return 1


class _FakeFusedKernel:
    """Kernel stand-in mirroring liger's signature and both its normalizers."""

    last_num_items: float | None = None
    last_loss_type: str | None = None
    last_args: tuple = ()

    @classmethod
    def forward(
        cls,
        ctx,
        _input,
        weight,
        selected_token_ids,
        attention_mask,
        advantages,
        bias=None,
        ref_per_token_logps=None,
        old_per_token_logps=None,
        ref_input=None,
        ref_weight=None,
        ref_bias=None,
        beta=0.04,
        epsilon_low=0.2,
        epsilon_high=0.2,
        loss_type="dapo",
        max_completion_length=None,
        importance_sampling_level="token",
        sapo_temperature_pos=1.0,
        sapo_temperature_neg=1.05,
        temperature=1.0,
        compiled=True,
        use_ref_model=True,
        chunk_size=1,
        vllm_is_ratio=None,
        delta=None,
        use_bias_correction_kl=False,
        vespo_k_pos=2.0,
        vespo_lambda_pos=3.0,
        vespo_k_neg=3.0,
        vespo_lambda_neg=2.0,
        num_items_in_batch=None,
    ):
        """Per-token loss for the requested objective under its own reduction.

        ``dapo``/``cispo``/``vespo`` divide by ``num_items_in_batch`` when it is
        given and by their own micro-batch otherwise; ``grpo`` averages
        per-sequence means and never reads the count.
        """
        cls.last_num_items = num_items_in_batch
        cls.last_loss_type = loss_type
        logits = (_input.squeeze(1) @ weight.t()) / temperature
        logps = torch.log_softmax(logits, dim=-1).gather(
            1,
            selected_token_ids.reshape(-1, 1),
        )
        ratio = torch.exp(logps - old_per_token_logps)
        if loss_type == "cispo":
            clamped = torch.clamp(ratio, None, epsilon_high).detach()
            per_token_loss = -clamped * advantages.unsqueeze(1) * logps
        else:
            clamped = torch.clamp(ratio, 1 - epsilon_low, 1 + epsilon_high)
            per_token_loss = -torch.min(
                ratio * advantages.unsqueeze(1),
                clamped * advantages.unsqueeze(1),
            )
        mask = attention_mask.to(logps.dtype)
        masked = per_token_loss * mask
        if loss_type in TOKEN_COUNT_LOSS_TYPES:
            count = (
                torch.as_tensor(float(num_items_in_batch))
                if num_items_in_batch is not None
                else mask.sum()
            )
            loss = masked.sum() / torch.clamp(count / _world_size(), min=1.0)
        else:
            per_row = masked.sum(-1) / mask.sum(-1).clamp(min=1.0)
            loss = per_row.sum() / mask.shape[0]
        return loss, (torch.zeros(()),)

    @classmethod
    def apply(cls, *args):
        """Invoke ``forward`` the way ``torch.autograd.Function.apply`` does."""
        cls.last_args = args
        return cls.forward(None, *args)


class _KernelWithoutTokenCount:
    """Kernel stand-in whose signature cannot carry the token count."""

    @classmethod
    def forward(cls, ctx, _input):
        """Accept the autograd context and nothing else."""
        return _input


class _KernelWithoutCtx:
    """Kernel stand-in whose forward drops the autograd context."""

    @classmethod
    def forward(cls, _input, num_items_in_batch=None):
        """Accept the input tensor without an autograd context."""
        return _input, num_items_in_batch


class _CallableActor:
    """DeepSpeed engine stand-in returning the owner's fixed hidden states."""

    def __init__(self, owner: _Stub, accumulation_steps: int) -> None:
        self._owner = owner
        self._accumulation_steps = accumulation_steps

    def gradient_accumulation_steps(self) -> int:
        """Report the engine's live accumulation steps."""
        return self._accumulation_steps

    def train(self) -> None:
        """Match the training-mode call the fused path makes."""
        return

    def __call__(self, **_kwargs: Any):
        """Return the hidden states in place of logits."""
        return SimpleNamespace(logits=self._owner.hidden)


class _Stub:
    """Stand-in carrying the state the loss paths read, bound to GRPO's methods."""

    def __init__(
        self,
        *,
        loss_norm: str = "accumulation_window",
        loss_type: str = "cispo",
        accumulation_steps: int = 4,
        window_tokens: int | None = None,
        uses_deepspeed: bool = True,
        activation_offload: bool = False,
        lm_head: torch.nn.Linear | None = None,
        accelerator: Any = None,
    ) -> None:
        self.device = torch.device("cpu")
        self.accelerator = accelerator
        self.loss_norm = loss_norm
        self.loss_type = loss_type
        self.importance_sampling_level = "token"
        self.clip_coef_min = CLIP_MIN
        self.clip_coef_max = CLIP_MAX
        self.beta = 0.0
        self.temperature = 1.0
        self.max_output_tokens = 32
        self.chunk_rows = 4
        self.pad_token_id = 0
        self.calc_position_embeddings = False
        self.use_kl_advantage_shaping = False
        self.vllm_importance_sampling_cap = 2.0
        self.activation_offload = activation_offload
        self._uses_deepspeed = uses_deepspeed
        self._window_action_tokens = window_tokens
        self.lm_head = lm_head or torch.nn.Linear(HIDDEN, VOCAB, bias=False)
        self.hidden: torch.Tensor | None = None
        self.actor: Any = _CallableActor(self, accumulation_steps)

    _accumulation_steps = GRPO._accumulation_steps
    _accumulation_steps_without_deepspeed = GRPO._accumulation_steps_without_deepspeed
    _resolve_loss_norm = GRPO._resolve_loss_norm
    _activation_offload_ctx = GRPO._activation_offload_ctx
    _apply_kl_advantage_shaping = GRPO._apply_kl_advantage_shaping
    _compute_policy_loss = GRPO._compute_policy_loss
    _fused_kernel_loss = GRPO._fused_kernel_loss
    _liger_loss = GRPO._liger_loss
    _log_importance_weights = GRPO._log_importance_weights
    _record_window_action_tokens = GRPO._record_window_action_tokens
    _reduce_masked_loss = GRPO._reduce_masked_loss
    _resolve_loss_window = GRPO._resolve_loss_window

    def _get_lm_head(self) -> torch.nn.Linear:
        return self.lm_head

    def _packing_mode(self) -> None:
        return None

    def _resolve_fused_chunk_rows(self, _vocab: int, _chunk_rows: int) -> int:
        return 2

    def _patch_lm_head_to_identity(self):
        return nullcontext()

    def select_adapter(self, _name: str):
        return nullcontext()

    def _amp_ctx(self):
        return nullcontext()

    def _liger_head_gather(self):
        return nullcontext()


class _SaveOnCpuSpy:
    """Stand-in for ``torch.autograd.graph.save_on_cpu`` recording its use."""

    def __init__(self, real: Any) -> None:
        self._real = real
        self.pin_memory_flags: list[bool] = []
        self.depth = 0
        self.max_depth = 0

    def __call__(self, **kwargs: Any):
        """Build a depth-recording wrapper around the real offload context."""
        self.pin_memory_flags.append(bool(kwargs.get("pin_memory", False)))
        return _SpyContext(self, self._real(**kwargs))


class _SpyContext:
    """Offload context recording how deep the wrapped work sits inside it."""

    def __init__(self, spy: _SaveOnCpuSpy, inner: Any) -> None:
        self._spy = spy
        self._inner = inner

    def __enter__(self):
        """Enter the real context and record the depth."""
        self._spy.depth += 1
        self._spy.max_depth = max(self._spy.max_depth, self._spy.depth)
        return self._inner.__enter__()

    def __exit__(self, *exc_info: object):
        """Leave the real context and record the depth."""
        self._spy.depth -= 1
        return self._inner.__exit__(*exc_info)


def _mask_of_lengths(lengths: list[int], width: int) -> torch.Tensor:
    """Action-token mask with one row per requested length."""
    mask = torch.zeros(len(lengths), width)
    for row, length in enumerate(lengths):
        mask[row, :length] = 1.0
    return mask


def _micro_batch_reduce(loss: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Per-sequence mean over that sequence's own action tokens."""
    return (loss * mask).sum(dim=-1) / mask.sum(dim=-1).clamp(min=1.0)


def _per_token_weights(
    algo: _Stub,
    masks: list[torch.Tensor],
    accumulation_steps: int,
    reduce_fn: Any,
) -> list[torch.Tensor]:
    """Per-token gradient weights the optimizer sees for each micro-batch.

    Each micro-batch loss is divided by ``accumulation_steps`` the way DeepSpeed
    scales it before accumulating.
    """
    weights = []
    for mask in masks:
        per_token = torch.zeros_like(mask, requires_grad=True)
        reduced = reduce_fn(algo, per_token, mask)
        (reduced.mean() / accumulation_steps).backward()
        assert per_token.grad is not None
        weights.append(per_token.grad)
    return weights


def _token_log_probs(
    algo: _Stub,
    hidden: torch.Tensor,
    batch_ids: torch.Tensor,
    width: int,
) -> torch.Tensor:
    """Per-token log-probs of the selected ids under the stub's head."""
    logits = algo.lm_head(hidden[:, :width, :]) / algo.temperature
    return torch.log_softmax(logits, dim=-1).gather(
        2,
        batch_ids[:, 1:].unsqueeze(-1),
    )[..., 0]


def _fused_inputs(lengths: list[int], width: int, seed: int):
    """Hidden states, token ids and an action mask for a fused-path call."""
    generator = torch.Generator().manual_seed(seed)
    hidden = torch.randn(len(lengths), width + 1, HIDDEN, generator=generator)
    batch_ids = torch.randint(1, VOCAB, (len(lengths), width + 1), generator=generator)
    return hidden, batch_ids, _mask_of_lengths(lengths, width)


@pytest.fixture(autouse=True)
def _reset_kernel_spy(monkeypatch: pytest.MonkeyPatch) -> None:
    """Hand the shared kernel spy and AgileRL's module globals back untouched."""
    monkeypatch.setattr(_FakeFusedKernel, "last_num_items", None)
    monkeypatch.setattr(_FakeFusedKernel, "last_loss_type", None)
    monkeypatch.setattr(_FakeFusedKernel, "last_args", ())
    monkeypatch.setattr(grpo_module, "HAS_LIGER_KERNEL", grpo_module.HAS_LIGER_KERNEL)
    monkeypatch.setattr(
        grpo_module,
        KERNEL_NAME,
        getattr(grpo_module, KERNEL_NAME, None),
        raising=False,
    )


@pytest.fixture
def fused_kernel(monkeypatch: pytest.MonkeyPatch) -> type[_FakeFusedKernel]:
    """Install the fused-kernel stand-in behind AgileRL's kernel name."""
    monkeypatch.setattr(grpo_module, "HAS_LIGER_KERNEL", True)
    monkeypatch.setattr(grpo_module, KERNEL_NAME, _FakeFusedKernel, raising=False)
    return _FakeFusedKernel


class TestGRPOLossNormConfig:
    """The mode is validated at construction and defaults to the micro-batch."""

    @pytest.mark.parametrize("loss_norm", ["micro_batch", "accumulation_window"])
    def test_supported_modes_are_accepted(self, loss_norm: str) -> None:
        assert _Stub()._resolve_loss_norm(loss_norm) == loss_norm

    def test_unknown_mode_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="Invalid loss_norm 'per_token'"):
            _Stub()._resolve_loss_norm("per_token")

    def test_default_is_the_micro_batch(self) -> None:
        default = inspect.signature(GRPO.__init__).parameters["loss_norm"].default
        assert default == "micro_batch"

    def test_the_window_is_accepted_when_each_micro_batch_is_a_step(self) -> None:
        algo = _Stub(uses_deepspeed=False, accelerator=None)
        assert algo._resolve_loss_norm("accumulation_window") == "accumulation_window"

    def test_a_declared_accelerate_window_without_deepspeed_is_rejected(self) -> None:
        algo = _Stub(
            uses_deepspeed=False,
            accelerator=SimpleNamespace(gradient_accumulation_steps=4),
        )
        with pytest.raises(ValueError, match="never accumulated"):
            algo._resolve_loss_norm("accumulation_window")

    def test_the_micro_batch_mode_ignores_the_declared_accelerate_window(self) -> None:
        algo = _Stub(
            uses_deepspeed=False,
            accelerator=SimpleNamespace(gradient_accumulation_steps=4),
        )
        assert algo._resolve_loss_norm("micro_batch") == "micro_batch"


class TestWindowNormalizedReduction:
    """The standard path's reduction under ``loss_norm``."""

    def test_per_token_weights_are_equal_across_sequence_lengths(self) -> None:
        lengths = [4, 100]
        width = 128
        masks = [_mask_of_lengths([length], width) for length in lengths]
        window_tokens = sum(lengths)
        algo = _Stub(accumulation_steps=len(lengths), window_tokens=window_tokens)

        weights = _per_token_weights(
            algo,
            masks,
            len(lengths),
            GRPO._reduce_masked_loss,
        )
        short_weight = float(weights[0][0, 0])
        long_weight = float(weights[1][0, 0])
        assert short_weight == pytest.approx(1.0 / window_tokens)
        assert long_weight == pytest.approx(1.0 / window_tokens)

        baseline = _per_token_weights(
            algo,
            masks,
            len(lengths),
            lambda _algo, loss, mask: _micro_batch_reduce(loss, mask),
        )
        ratio = float(baseline[0][0, 0]) / float(baseline[1][0, 0])
        assert ratio == pytest.approx(lengths[1] / lengths[0])

    def test_out_of_mask_tokens_carry_no_weight(self) -> None:
        mask = _mask_of_lengths([3], 8)
        algo = _Stub(accumulation_steps=2, window_tokens=10)
        weights = _per_token_weights(algo, [mask], 2, GRPO._reduce_masked_loss)[0]
        assert torch.all(weights[0, 3:] == 0.0)
        assert torch.all(weights[0, :3] > 0.0)

    def test_accumulated_loss_equals_the_window_token_mean(self) -> None:
        steps = 4
        lengths = [2, 7, 11, 40]
        width = 64
        masks = [_mask_of_lengths([length], width) for length in lengths]
        generator = torch.Generator().manual_seed(0)
        losses = [
            torch.rand(mask.shape, generator=generator) * mask * 3.0 for mask in masks
        ]
        window_tokens = sum(lengths)
        algo = _Stub(accumulation_steps=steps, window_tokens=window_tokens)

        returned = [
            algo._reduce_masked_loss(loss, mask).mean()
            for loss, mask in zip(losses, masks, strict=True)
        ]
        accumulated = float(sum(returned) / steps)
        total = float(
            sum((loss * mask).sum() for loss, mask in zip(losses, masks, strict=True)),
        )
        assert accumulated == pytest.approx(total / window_tokens, rel=1e-6)

    def test_multi_row_micro_batch_normalizes_over_the_window(self) -> None:
        mask = _mask_of_lengths([3, 9], 16)
        loss = torch.full(mask.shape, 0.5) * mask
        window_tokens = 40
        steps = 3
        algo = _Stub(accumulation_steps=steps, window_tokens=window_tokens)
        reduced = algo._reduce_masked_loss(loss, mask).mean()
        expected = steps * float((loss * mask).sum()) / window_tokens
        assert float(reduced) == pytest.approx(expected, rel=1e-6)

    def test_single_accumulation_step_uses_the_micro_batch(self) -> None:
        mask = _mask_of_lengths([4, 6], 8)
        loss = torch.full(mask.shape, 2.0) * mask
        algo = _Stub(accumulation_steps=1, uses_deepspeed=False)
        reduced = algo._reduce_masked_loss(loss, mask).mean()
        assert float(reduced) == pytest.approx(2.0)

    def test_micro_batch_mode_keeps_the_per_sequence_mean(self) -> None:
        mask = _mask_of_lengths([3, 9], 16)
        generator = torch.Generator().manual_seed(4)
        loss = torch.rand(mask.shape, generator=generator) * mask
        algo = _Stub(loss_norm="micro_batch", accumulation_steps=4, window_tokens=40)
        reduced = algo._reduce_masked_loss(loss, mask)
        assert torch.allclose(reduced, _micro_batch_reduce(loss, mask))

    def test_non_finite_padding_stays_out_of_the_window_reduction(self) -> None:
        mask = torch.tensor([[1.0, 1.0, 0.0]])
        loss = torch.tensor([[2.0, 4.0, float("nan")]])
        algo = _Stub(accumulation_steps=1, uses_deepspeed=False)
        reduced = algo._reduce_masked_loss(loss, mask)
        assert reduced.tolist() == pytest.approx([3.0])

    def test_missing_window_token_count_raises(self) -> None:
        mask = _mask_of_lengths([4], 8)
        algo = _Stub(accumulation_steps=4)
        with pytest.raises(RuntimeError, match="no recorded window action-token"):
            algo._reduce_masked_loss(torch.zeros(mask.shape), mask)

    def test_zero_window_token_count_raises(self) -> None:
        mask = _mask_of_lengths([4], 8)
        algo = _Stub(accumulation_steps=4, window_tokens=0)
        with pytest.raises(RuntimeError, match="non-positive count"):
            algo._reduce_masked_loss(torch.zeros(mask.shape), mask)

    def test_empty_micro_batch_without_accumulation_raises(self) -> None:
        mask = torch.zeros(1, 8)
        algo = _Stub(accumulation_steps=1, uses_deepspeed=False)
        with pytest.raises(RuntimeError, match="action-token count is zero"):
            algo._reduce_masked_loss(torch.zeros(mask.shape), mask)


class TestAccumulationSteps:
    """The step count comes from the engine that applies the scaling."""

    def test_without_deepspeed_the_update_spans_one_micro_batch(self) -> None:
        algo = _Stub(accumulation_steps=8, uses_deepspeed=False)
        assert algo._accumulation_steps() == 1

    def test_a_declared_accelerate_window_without_deepspeed_raises(self) -> None:
        algo = _Stub(
            uses_deepspeed=False,
            accelerator=SimpleNamespace(gradient_accumulation_steps=2),
        )
        with pytest.raises(ValueError, match="never accumulated"):
            algo._accumulation_steps()

    def test_the_engine_outranks_what_the_accelerator_declares(self) -> None:
        algo = _Stub(
            accumulation_steps=6,
            accelerator=SimpleNamespace(gradient_accumulation_steps=2),
        )
        assert algo._accumulation_steps() == 6

    def test_the_engine_accessor_supplies_the_count(self) -> None:
        algo = _Stub(accumulation_steps=6)
        assert algo._accumulation_steps() == 6

    def test_missing_engine_accessor_raises(self) -> None:
        algo = _Stub(window_tokens=10)
        algo.actor = SimpleNamespace()
        with pytest.raises(TypeError, match="no callable gradient_accumulation_steps"):
            algo._accumulation_steps()

    def test_non_positive_engine_count_raises(self) -> None:
        algo = _Stub(window_tokens=10)
        algo.actor = SimpleNamespace(gradient_accumulation_steps=lambda: 0)
        with pytest.raises(RuntimeError, match="returned 0"):
            algo._accumulation_steps()


class TestWindowActionTokenRecording:
    """The window counts only the samples that survive the advantage filter."""

    def test_only_surviving_samples_are_counted(self) -> None:
        algo = _Stub()
        action_masks = _mask_of_lengths([5, 9, 2], 16)
        algo._record_window_action_tokens(action_masks, np.array([0, 2]))
        assert algo._window_action_tokens == 5 + 2

    def test_the_whole_batch_is_counted_without_filtering(self) -> None:
        algo = _Stub()
        action_masks = _mask_of_lengths([5, 9, 2], 16)
        algo._record_window_action_tokens(action_masks, np.arange(3))
        assert algo._window_action_tokens == 5 + 9 + 2


class TestFusedKernelNormalizer:
    """``num_items_in_batch`` plumbing into the fused kernel's reduction."""

    def test_defaults_bridge_the_gap_to_the_token_count(
        self,
        fused_kernel: type[_FakeFusedKernel],
    ) -> None:
        required = (torch.zeros(1), torch.zeros(1), torch.zeros(1), torch.ones(1), None)
        args = grpo_module._liger_args_with_normalizer(required, 512.0)
        names = [
            parameter.name
            for parameter in list(
                inspect.signature(fused_kernel.forward).parameters.values(),
            )[1:]
        ]
        assert len(args) == names.index("num_items_in_batch") + 1
        assert args[-1] == 512.0
        assert args[names.index("use_bias_correction_kl")] is False
        assert args[names.index("loss_type")] == "dapo"

    def test_a_kernel_without_the_token_count_is_rejected(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(
            grpo_module,
            KERNEL_NAME,
            _KernelWithoutTokenCount,
            raising=False,
        )
        with pytest.raises(RuntimeError, match="does not accept 'num_items_in_batch'"):
            grpo_module._liger_args_with_normalizer((torch.zeros(1),), 8.0)

    def test_a_kernel_without_the_autograd_context_is_rejected(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(grpo_module, KERNEL_NAME, _KernelWithoutCtx, raising=False)
        with pytest.raises(RuntimeError, match="must start with the autograd"):
            grpo_module._liger_args_with_normalizer((torch.zeros(1),), 8.0)

    def test_a_missing_required_argument_is_rejected(
        self,
        fused_kernel: type[_FakeFusedKernel],
    ) -> None:
        with pytest.raises(RuntimeError, match="'weight' precedes"):
            grpo_module._liger_args_with_normalizer((torch.zeros(1),), 8.0)


class TestFusedWindowNormalization:
    """The fused path under ``loss_norm="accumulation_window"``."""

    @staticmethod
    def _run(
        loss_type: str,
        objective: str,
        seed: int,
        steps: int = 4,
        lengths: tuple[int, ...] = (5, 9),
        width: int = 16,
        loss_norm: str = "accumulation_window",
    ):
        hidden, batch_ids, mask = _fused_inputs(list(lengths), width, seed)
        window_tokens = sum(lengths) * 2
        advantages = torch.tensor([[0.4], [-0.9]])
        algo = _Stub(
            loss_norm=loss_norm,
            loss_type=loss_type,
            accumulation_steps=steps,
            window_tokens=window_tokens,
        )
        algo.hidden = hidden
        log_probs = _token_log_probs(algo, hidden, batch_ids, width)
        spread = torch.linspace(0.0, 0.3, steps=log_probs.numel()).reshape(
            log_probs.shape,
        )
        old_log_probs = (log_probs - spread).detach()

        fused_loss, _metric = algo._liger_loss(
            batch_ids,
            mask,
            advantages,
            old_log_probs,
            None,
        )
        eager_loss, _kl = algo._compute_policy_loss(
            mask,
            _token_log_probs(algo, hidden, batch_ids, width),
            old_log_probs,
            log_probs.detach(),
            advantages,
            None,
            level="token",
            objective=objective,
        )
        ratio = torch.exp(log_probs.detach() - old_log_probs)
        if objective == "cispo":
            per_token = -(ratio.clamp(max=CLIP_MAX) * advantages * log_probs.detach())
        else:
            clipped = ratio.clamp(CLIP_MIN, CLIP_MAX)
            per_token = -torch.min(ratio * advantages, clipped * advantages)
        return SimpleNamespace(
            fused=fused_loss,
            eager=eager_loss,
            masked_sum=float((per_token * mask).sum()),
            window_tokens=window_tokens,
            steps=steps,
            min_ratio=float(ratio.min()),
        )

    def test_the_min_clip_objective_reaches_the_token_count_reduction(
        self,
        fused_kernel: type[_FakeFusedKernel],
    ) -> None:
        result = self._run("grpo", "grpo", seed=11)
        expected = result.steps * result.masked_sum / result.window_tokens
        assert fused_kernel.last_loss_type == "dapo"
        assert fused_kernel.last_num_items == pytest.approx(
            result.window_tokens * _world_size(),
        )
        assert result.fused.item() == pytest.approx(expected, rel=1e-5)
        assert result.eager.item() == pytest.approx(expected, rel=1e-5)

    def test_cispo_keeps_its_objective_and_gains_the_window(
        self,
        fused_kernel: type[_FakeFusedKernel],
    ) -> None:
        result = self._run("cispo", "cispo", seed=3)
        # Ratios stay above the lower clamp the fused CISPO does not apply, so
        # the two objectives coincide.
        assert result.min_ratio >= CLIP_MIN
        expected = result.steps * result.masked_sum / result.window_tokens
        assert fused_kernel.last_loss_type == "cispo"
        assert fused_kernel.last_num_items == pytest.approx(
            result.window_tokens * _world_size(),
        )
        assert result.fused.item() == pytest.approx(expected, rel=1e-5)
        assert result.eager.item() == pytest.approx(expected, rel=1e-5)

    @pytest.mark.parametrize(
        ("loss_type", "objective"),
        [("grpo", "grpo"), ("cispo", "cispo")],
    )
    def test_a_single_step_window_is_the_micro_batch(
        self,
        fused_kernel: type[_FakeFusedKernel],
        loss_type: str,
        objective: str,
    ) -> None:
        result = self._run(loss_type, objective, seed=13, steps=1, lengths=(5, 9))
        # The window record is ignored; the micro-batch's own mask spans it.
        assert fused_kernel.last_num_items == pytest.approx(14.0 * _world_size())
        assert result.fused.item() == pytest.approx(
            result.masked_sum / 14.0,
            rel=1e-5,
        )

    @pytest.mark.parametrize("loss_type", ["grpo", "cispo"])
    def test_micro_batch_mode_leaves_the_kernel_call_untouched(
        self,
        fused_kernel: type[_FakeFusedKernel],
        loss_type: str,
    ) -> None:
        self._run(loss_type, loss_type, seed=7, loss_norm="micro_batch")
        assert fused_kernel.last_loss_type == loss_type
        assert fused_kernel.last_num_items is None
        assert len(fused_kernel.last_args) == 24


class TestFusedActivationOffload:
    """``activation_offload`` reaches the fused training forward."""

    @staticmethod
    def _call(algo: _Stub) -> tuple[torch.Tensor, torch.Tensor]:
        hidden, batch_ids, mask = _fused_inputs([5, 9], 16, seed=17)
        algo.hidden = hidden
        return algo._liger_loss(
            batch_ids,
            mask,
            torch.tensor([[0.4], [-0.9]]),
            torch.zeros(2, 16),
            None,
        )

    @pytest.fixture
    def spy(self, monkeypatch: pytest.MonkeyPatch) -> _SaveOnCpuSpy:
        """Record every entry into ``torch.autograd.graph.save_on_cpu``."""
        spy = _SaveOnCpuSpy(torch.autograd.graph.save_on_cpu)
        monkeypatch.setattr(torch.autograd.graph, "save_on_cpu", spy)
        return spy

    def test_the_fused_loss_runs_inside_the_offload_context(
        self,
        fused_kernel: type[_FakeFusedKernel],
        spy: _SaveOnCpuSpy,
    ) -> None:
        algo = _Stub(activation_offload=True, window_tokens=28)
        self._call(algo)
        assert spy.pin_memory_flags == [True]
        assert spy.max_depth == 1
        assert spy.depth == 0
        assert fused_kernel.last_num_items == pytest.approx(28.0 * _world_size())

    def test_the_context_is_inert_when_the_flag_is_off(
        self,
        fused_kernel: type[_FakeFusedKernel],
        spy: _SaveOnCpuSpy,
    ) -> None:
        algo = _Stub(activation_offload=False, window_tokens=28)
        self._call(algo)
        assert spy.pin_memory_flags == []

    def test_the_context_is_inert_without_grad(
        self,
        fused_kernel: type[_FakeFusedKernel],
        spy: _SaveOnCpuSpy,
    ) -> None:
        algo = _Stub(activation_offload=True, window_tokens=28)
        with torch.no_grad():
            self._call(algo)
        assert spy.pin_memory_flags == []

    def test_offload_composes_with_the_window_scaling(
        self,
        fused_kernel: type[_FakeFusedKernel],
        spy: _SaveOnCpuSpy,
    ) -> None:
        offloaded = _Stub(activation_offload=True, window_tokens=28)
        plain = _Stub(
            activation_offload=False,
            window_tokens=28,
            lm_head=offloaded.lm_head,
        )
        offloaded_loss, _ = self._call(offloaded)
        plain_loss, _ = self._call(plain)
        assert spy.pin_memory_flags == [True]
        assert offloaded_loss.item() == plain_loss.item()

    def test_the_context_comes_from_the_algorithm_hierarchy(self) -> None:
        assert GRPO._activation_offload_ctx is LLMAlgorithm._activation_offload_ctx
        assert (
            inspect.signature(GRPO.__init__).parameters["activation_offload"].default
            is False
        )
