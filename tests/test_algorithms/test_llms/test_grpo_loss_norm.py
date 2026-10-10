# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for GRPO loss normalization and fused-path activation offload.

Covers ``loss_norm="accumulation_window"`` on the standard and the fused Liger
path, the ``num_items_in_batch`` plumbing the fused reduction needs, and
``activation_offload`` reaching the fused training forward. Pure CPU: the fused
kernel is a stand-in mirroring liger's signature and both of its reductions.
"""

from __future__ import annotations

import datetime
import inspect
import multiprocessing
import sys
import traceback
import warnings
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Shard, distribute_tensor

pytest.importorskip("transformers", reason="LLM tests require transformers.")
pytest.importorskip("peft", reason="LLM tests require peft.")

import numpy as np

from agilerl.algorithms import grpo as grpo_module
from agilerl.algorithms.core.base import LLMAlgorithm
from agilerl.algorithms.grpo import GRPO
from agilerl.distributed import FSDPConfig
from agilerl.distributed.runtime import FSDPRuntime

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
        old = logps.detach() if old_per_token_logps is None else old_per_token_logps
        ratio = torch.exp(logps - old)
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
    """Actor stand-in returning the owner's fixed hidden states."""

    def __init__(self, owner: _Stub) -> None:
        self._owner = owner

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
        activation_offload: bool = False,
        lm_head: torch.nn.Linear | None = None,
    ) -> None:
        self.device = torch.device("cpu")
        self.loss_norm = loss_norm
        self.loss_type = loss_type
        self.importance_sampling_level = "token"
        self.clip_coef_min = CLIP_MIN
        self.clip_coef_max = CLIP_MAX
        self.beta = 0.0
        self.temperature = 1.0
        self.cast_logprobs_to_fp32 = True
        self.max_output_tokens = 32
        self.chunk_rows = 4
        self.pad_token_id = 0
        self.calc_position_embeddings = False
        self.use_kl_advantage_shaping = False
        self.vllm_importance_sampling_cap = 2.0
        self.off_policy_token_mask_bounds = None
        self.off_policy_sequence_mask_threshold = None
        self.use_bias_correction_kl = False
        self.kl_clamp = 10.0
        self.activation_offload = activation_offload
        self.gradient_accumulation_steps = accumulation_steps
        self._window_action_tokens = window_tokens
        self._segment_accumulation_steps = None
        self.lm_head = lm_head or torch.nn.Linear(HIDDEN, VOCAB, bias=False)
        self.hidden: torch.Tensor | None = None
        self.actor: Any = _CallableActor(self)

    _resolve_loss_norm = GRPO._resolve_loss_norm
    _activation_offload_ctx = GRPO._activation_offload_ctx
    _actor_hidden_states = GRPO._actor_hidden_states
    _apply_kl_advantage_shaping = GRPO._apply_kl_advantage_shaping
    _clipped_units = GRPO._clipped_units
    _compute_policy_loss = GRPO._compute_policy_loss
    _fused_kernel_loss = GRPO._fused_kernel_loss
    _gradient_forward_inputs = GRPO._gradient_forward_inputs
    _liger_loss = GRPO._liger_loss
    _log_importance_weights = GRPO._log_importance_weights
    _masks_off_policy_tokens = GRPO._masks_off_policy_tokens
    _logprobs_from_hidden_fused = staticmethod(GRPO._logprobs_from_hidden_fused)
    _record_global_window_action_tokens = GRPO._record_global_window_action_tokens
    _reduce_masked_loss = GRPO._reduce_masked_loss
    _resolve_loss_window = GRPO._resolve_loss_window
    _token_loss_weights = GRPO._token_loss_weights
    _warn_if_micro_batches_straddle_optimizer_steps = (
        GRPO._warn_if_micro_batches_straddle_optimizer_steps
    )

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
        return nullcontext((self.lm_head.weight, self.lm_head.bias))


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

    Each micro-batch loss is divided by ``accumulation_steps`` so the
    accumulated gradient matches a mini-batch mean.
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
    """The mode is validated at construction and defaults to the window."""

    @pytest.mark.parametrize(
        "loss_norm", ["micro_batch", "accumulation_window", "episode"]
    )
    def test_supported_modes_are_accepted(self, loss_norm: str) -> None:
        assert _Stub()._resolve_loss_norm(loss_norm) == loss_norm

    def test_unknown_mode_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="Invalid loss_norm 'per_token'"):
            _Stub()._resolve_loss_norm("per_token")

    def test_default_is_the_accumulation_window(self) -> None:
        default = inspect.signature(GRPO.__init__).parameters["loss_norm"].default
        assert default == "accumulation_window"

    def test_the_window_is_accepted_when_each_micro_batch_is_a_step(self) -> None:
        algo = _Stub(accumulation_steps=1)
        assert algo._resolve_loss_norm("accumulation_window") == "accumulation_window"

    def test_the_micro_batch_mode_is_accepted_alongside_accumulation(self) -> None:
        algo = _Stub(accumulation_steps=4)
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

    def test_single_accumulation_step_normalizes_by_the_recorded_window(self) -> None:
        mask = _mask_of_lengths([4, 6], 8)
        loss = torch.full(mask.shape, 2.0) * mask
        algo = _Stub(accumulation_steps=1)
        algo._record_global_window_action_tokens(mask, np.arange(2))

        reduced = algo._reduce_masked_loss(loss, mask).mean()

        assert float(reduced) == pytest.approx(2.0)

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="sync debug mode needs CUDA"
    )
    def test_single_accumulation_step_reduces_without_a_host_sync(self) -> None:
        # Arrange
        device = torch.device("cuda")
        mask = _mask_of_lengths([4, 6], 8).to(device)
        loss = torch.full(mask.shape, 2.0, device=device) * mask
        algo = _Stub(accumulation_steps=1)
        algo.device = device
        algo._record_global_window_action_tokens(mask, np.arange(2))

        # Act
        torch.cuda.set_sync_debug_mode("error")
        try:
            reduced = algo._reduce_masked_loss(loss, mask).mean()
        finally:
            torch.cuda.set_sync_debug_mode("default")

        # Assert
        assert float(reduced) == pytest.approx(2.0)

    def test_micro_batch_mode_keeps_the_per_sequence_mean(self) -> None:
        mask = _mask_of_lengths([3, 9], 16)
        generator = torch.Generator().manual_seed(4)
        loss = torch.rand(mask.shape, generator=generator) * mask
        algo = _Stub(loss_norm="micro_batch", accumulation_steps=4, window_tokens=40)
        reduced = algo._reduce_masked_loss(loss, mask)
        assert torch.allclose(reduced, _micro_batch_reduce(loss, mask))

    def test_a_per_sequence_loss_spreads_over_its_action_tokens(self) -> None:
        mask = _mask_of_lengths([3, 9], 16)
        loss = torch.tensor([[0.5], [-2.0]])
        algo = _Stub(loss_norm="micro_batch")

        shares = algo._reduce_masked_loss(loss, mask)

        assert shares.tolist() == pytest.approx([0.5, -2.0])

    def test_non_finite_padding_stays_out_of_the_window_reduction(self) -> None:
        mask = torch.tensor([[1.0, 1.0, 0.0]])
        loss = torch.tensor([[2.0, 4.0, float("nan")]])
        algo = _Stub(accumulation_steps=1, window_tokens=2)
        reduced = algo._reduce_masked_loss(loss, mask)
        assert reduced.tolist() == pytest.approx([3.0])

    def test_missing_window_token_count_raises(self) -> None:
        mask = _mask_of_lengths([4], 8)
        algo = _Stub(accumulation_steps=4)
        with pytest.raises(RuntimeError, match="no recorded window action-token"):
            algo._reduce_masked_loss(torch.zeros(mask.shape), mask)

    def test_window_without_action_tokens_gives_zero_shares(self) -> None:
        mask = torch.zeros(2, 8)
        algo = _Stub(accumulation_steps=4)
        algo._record_global_window_action_tokens(mask, np.arange(2))

        shares = algo._reduce_masked_loss(torch.ones(mask.shape), mask)

        assert shares.tolist() == [0.0, 0.0]

    def test_empty_micro_batch_without_accumulation_gives_zero_shares(self) -> None:
        mask = torch.zeros(1, 8)
        algo = _Stub(accumulation_steps=1)
        algo._record_global_window_action_tokens(mask, np.arange(1))

        shares = algo._reduce_masked_loss(torch.ones(mask.shape), mask)

        assert shares.tolist() == [0.0]


class TestGRPOTokenLossWeights:
    """The per-token weights both loss paths reduce with."""

    def test_micro_batch_rows_each_spread_one_over_the_batch(self) -> None:
        mask = _mask_of_lengths([2, 5], 8)
        algo = _Stub(loss_norm="micro_batch")

        weights = algo._token_loss_weights(mask, None)

        assert weights[0, :2].tolist() == pytest.approx([1 / 4, 1 / 4])
        assert weights[1, :5].tolist() == pytest.approx([1 / 10] * 5)
        assert weights.sum(dim=-1).tolist() == pytest.approx([0.5, 0.5])
        assert torch.all(weights[mask == 0] == 0.0)

    def test_window_tokens_weigh_steps_over_window_tokens(self) -> None:
        mask = _mask_of_lengths([2, 5], 8)
        algo = _Stub(accumulation_steps=3, window_tokens=30)

        weights = algo._token_loss_weights(mask, algo._resolve_loss_window())

        assert torch.allclose(weights, mask * (3 / 30))

    def test_rows_without_action_tokens_weigh_nothing(self) -> None:
        mask = _mask_of_lengths([0, 4], 8)
        algo = _Stub(loss_norm="micro_batch")

        weights = algo._token_loss_weights(mask, None)

        assert weights[0].tolist() == [0.0] * 8
        assert weights[1, :4].tolist() == pytest.approx([1 / 8] * 4)


class TestAccumulationSteps:
    """The step count comes from the trainer loop that applies the scaling."""

    def test_the_configured_width_is_reported(self) -> None:
        algo = _Stub(accumulation_steps=6)
        assert algo.gradient_accumulation_steps == 6

    def test_a_single_step_window_reports_one(self) -> None:
        algo = _Stub(accumulation_steps=1)
        assert algo.gradient_accumulation_steps == 1


class TestStraddleWarning:
    """A trailing micro-batch is worth warning about; a full window is not."""

    @staticmethod
    def _emitted(algo: _Stub, samples: int, micro_batch_size: int) -> list:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            algo._warn_if_micro_batches_straddle_optimizer_steps(
                samples, micro_batch_size
            )
        return caught

    def test_a_trailing_micro_batch_warns(self) -> None:
        algo = _Stub(accumulation_steps=2)
        with pytest.warns(UserWarning, match="whole optimizer steps"):
            algo._warn_if_micro_batches_straddle_optimizer_steps(3, 1)

    def test_micro_batches_filling_whole_steps_stay_silent(self) -> None:
        assert self._emitted(_Stub(accumulation_steps=2), 4, 1) == []

    def test_a_single_step_window_stays_silent(self) -> None:
        """Every micro-batch takes its own step, so none can straddle one."""
        assert self._emitted(_Stub(accumulation_steps=1), 3, 2) == []


class TestGRPORecordGlobalWindowActionTokens:
    """The window counts the action tokens of its own rows only."""

    def test_only_the_window_rows_are_counted(self) -> None:
        algo = _Stub()
        action_masks = _mask_of_lengths([5, 9, 2], 16)

        algo._record_global_window_action_tokens(action_masks, np.array([0, 2]))

        assert algo._window_action_tokens.item() == 5 + 2

    def test_the_count_stays_a_device_tensor(self) -> None:
        algo = _Stub()
        action_masks = _mask_of_lengths([5, 9, 2], 16)

        algo._record_global_window_action_tokens(action_masks, np.arange(3))

        assert isinstance(algo._window_action_tokens, torch.Tensor)
        assert algo._window_action_tokens.device == action_masks.device
        assert algo._window_action_tokens.item() == 5 + 9 + 2

    def test_a_window_without_action_tokens_counts_one(self) -> None:
        algo = _Stub()

        algo._record_global_window_action_tokens(torch.zeros(2, 16), np.arange(2))

        assert algo._window_action_tokens.item() == 1.0


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
        window_tokens: int | None = None,
    ):
        hidden, batch_ids, mask = _fused_inputs(list(lengths), width, seed)
        if window_tokens is None:
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

        fused_loss, _, _, _ = algo._liger_loss(
            batch_ids,
            mask,
            advantages,
            old_log_probs,
            None,
        )
        eager_loss, _, _ = algo._compute_policy_loss(
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
            per_sequence_mean=float(_micro_batch_reduce(per_token, mask).mean()),
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
        result = self._run(
            loss_type, objective, seed=13, steps=1, lengths=(5, 9), window_tokens=14
        )
        assert result.fused.item() == pytest.approx(
            result.masked_sum / 14.0,
            rel=1e-5,
        )

    @pytest.mark.parametrize(
        ("loss_type", "objective"),
        [("grpo", "grpo"), ("cispo", "cispo")],
    )
    def test_micro_batch_is_the_mean_of_per_sequence_means(
        self,
        fused_kernel: type[_FakeFusedKernel],
        loss_type: str,
        objective: str,
    ) -> None:
        result = self._run(loss_type, objective, seed=7, loss_norm="micro_batch")

        assert result.min_ratio >= CLIP_MIN
        assert result.fused.item() == pytest.approx(result.per_sequence_mean, rel=1e-5)
        assert result.eager.item() == pytest.approx(result.per_sequence_mean, rel=1e-5)

    def test_window_without_action_tokens_gives_zero_loss(
        self,
        fused_kernel: type[_FakeFusedKernel],
    ) -> None:
        # Arrange: a window of padding rows only.
        hidden, batch_ids, mask = _fused_inputs([0, 0], 16, seed=5)
        algo = _Stub(loss_type="cispo", accumulation_steps=4)
        algo._record_global_window_action_tokens(mask, np.arange(2))
        algo.hidden = hidden

        # Act
        loss, _, _, _ = algo._liger_loss(
            batch_ids, mask, torch.tensor([[0.4], [-0.9]]), torch.zeros(2, 16), None
        )

        # Assert
        assert loss.item() == 0.0


class TestFusedActivationOffload:
    """``activation_offload`` reaches the fused training forward."""

    @staticmethod
    def _call(
        algo: _Stub,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
        offloaded_loss, _, _, _ = self._call(offloaded)
        plain_loss, _, _, _ = self._call(plain)
        assert spy.pin_memory_flags == [True]
        assert offloaded_loss.item() == plain_loss.item()

    def test_the_context_comes_from_the_algorithm_hierarchy(self) -> None:
        assert GRPO._activation_offload_ctx is LLMAlgorithm._activation_offload_ctx
        assert (
            inspect.signature(GRPO.__init__).parameters["activation_offload"].default
            is False
        )


class TestFusedKernelPolicyLogProbs:
    def test_matches_eager_log_softmax_on_action_tokens(
        self,
        fused_kernel: type[_FakeFusedKernel],
    ) -> None:
        # Arrange
        torch.manual_seed(0)
        algo = _Stub(window_tokens=14)
        hidden, batch_ids, mask = _fused_inputs([5, 9], 16, seed=23)
        algo.hidden = hidden

        # Act
        _, _, _, log_probs = algo._liger_loss(
            batch_ids,
            mask,
            torch.tensor([[0.4], [-0.9]]),
            torch.zeros(2, 16),
            None,
        )

        # Assert
        logits = hidden[:, :-1] @ algo.lm_head.weight.T
        expected = torch.log_softmax(logits.float(), dim=-1).gather(
            -1, batch_ids[:, 1:].unsqueeze(-1)
        )
        keep = mask.bool()
        assert log_probs.shape == mask.shape
        assert not log_probs.requires_grad
        assert torch.allclose(log_probs[keep], expected.squeeze(-1)[keep], atol=1e-5)
        assert torch.equal(log_probs[~keep], torch.zeros_like(log_probs[~keep]))


class TestFusedKernelScoredPositions:
    """The fused kernel scores only action positions and keeps the full-frame loss."""

    @pytest.mark.parametrize("loss_type", ["grpo", "cispo"])
    def test_micro_batch_loss_matches_the_full_frame_reduction(
        self,
        fused_kernel: type[_FakeFusedKernel],
        loss_type: str,
    ) -> None:
        # Arrange
        width = 16
        hidden, batch_ids, mask = _fused_inputs([5, 9], width, seed=29)
        advantages = torch.tensor([[0.4], [-0.9]])
        algo = _Stub(loss_norm="micro_batch", loss_type=loss_type)
        algo.hidden = hidden
        log_probs = _token_log_probs(algo, hidden, batch_ids, width).detach()
        spread = torch.linspace(0.0, 0.1, steps=log_probs.numel()).reshape(
            log_probs.shape
        )
        old_log_probs = log_probs - spread

        # Act
        loss, _, _, _ = algo._liger_loss(
            batch_ids, mask, advantages, old_log_probs, None
        )

        # Assert: the mean over rows of each row's mean over its own tokens.
        ratio = torch.exp(log_probs - old_log_probs)
        if loss_type == "cispo":
            per_token = -(ratio.clamp(max=CLIP_MAX) * advantages * log_probs)
        else:
            clipped = ratio.clamp(CLIP_MIN, CLIP_MAX)
            per_token = -torch.min(ratio * advantages, clipped * advantages)
        expected = _micro_batch_reduce(per_token, mask).mean()
        assert torch.allclose(loss, expected, atol=1e-6)

    @pytest.mark.parametrize("loss_type", ["grpo", "cispo"])
    def test_missing_old_log_probs_match_the_detached_policy(
        self,
        fused_kernel: type[_FakeFusedKernel],
        loss_type: str,
    ) -> None:
        # Arrange
        width = 16
        hidden, batch_ids, mask = _fused_inputs([5, 9], width, seed=31)
        advantages = torch.tensor([[0.4], [-0.9]])
        explicit = _Stub(loss_norm="micro_batch", loss_type=loss_type)
        implicit = _Stub(
            loss_norm="micro_batch",
            loss_type=loss_type,
            lm_head=torch.nn.Linear(HIDDEN, VOCAB, bias=False),
        )
        implicit.lm_head.load_state_dict(explicit.lm_head.state_dict())
        explicit.hidden = hidden
        implicit.hidden = hidden
        old_log_probs = _token_log_probs(explicit, hidden, batch_ids, width).detach()

        # Act
        explicit_loss, _, _, explicit_log_probs = explicit._liger_loss(
            batch_ids, mask, advantages, old_log_probs, None
        )
        explicit_loss.backward()
        implicit_loss, _, _, implicit_log_probs = implicit._liger_loss(
            batch_ids, mask, advantages, None, None
        )
        implicit_loss.backward()

        # Assert
        assert torch.allclose(implicit_loss, explicit_loss, atol=1e-6)
        assert torch.allclose(
            implicit.lm_head.weight.grad, explicit.lm_head.weight.grad, atol=1e-6
        )
        assert torch.allclose(implicit_log_probs, explicit_log_probs, atol=1e-6)


class TestLigerNormalizerWorldSize:
    def test_returns_one_when_distributed_inactive(self, monkeypatch) -> None:
        monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
        monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)

        assert grpo_module._liger_normalizer_world_size() == 1

    def test_uses_process_group_world_size(self, monkeypatch) -> None:
        monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
        monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
        monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 4)

        assert grpo_module._liger_normalizer_world_size() == 4


RANK_ACTION_LENGTHS = ([0], [3], [9, 4])
"""Per-rank action lengths: no action rows, one row chunk, nine row chunks."""
RANK_WIDTH = 16
RANK_TIMEOUT_S = 120


class _ChunkedCispoKernel(_FakeFusedKernel):
    """CISPO kernel stand-in that chunks rows and normalizes like liger 0.8.1.

    Each row chunk slices the head weight per vocab block, and without
    ``num_items_in_batch`` each row chunk all-reduces the action-token count.
    """

    @classmethod
    def apply(cls, *args):
        """Run the chunked forward from positional kernel arguments."""
        bound = inspect.signature(cls.forward).bind(None, *args).arguments
        weight = bound["weight"]
        mask = bound["attention_mask"].to(weight.dtype)
        num_items = bound.get("num_items_in_batch")
        n_chunks = max(1, bound["_input"].shape[0] // bound["chunk_size"])
        chunks = zip(
            torch.chunk(bound["_input"], n_chunks),
            torch.chunk(bound["selected_token_ids"], n_chunks),
            torch.chunk(mask, n_chunks),
            torch.chunk(bound["advantages"], n_chunks),
            torch.chunk(bound["old_per_token_logps"], n_chunks),
            strict=True,
        )
        loss = torch.zeros(())
        for hidden, targets, mask_chunk, adv, old in chunks:
            logits = torch.cat(
                [
                    hidden.squeeze(1) @ weight[start : start + 2].t()
                    for start in range(0, weight.shape[0], 2)
                ],
                dim=-1,
            )
            logps = torch.log_softmax(logits, dim=-1).gather(1, targets)
            ratio = torch.exp(logps - old).clamp(max=bound["epsilon_high"]).detach()
            per_token = -ratio * adv.unsqueeze(1) * logps
            if num_items is None:
                count = mask.sum()
                dist.all_reduce(count)
            else:
                count = torch.as_tensor(float(num_items))
            normalizer = torch.clamp(count / dist.get_world_size(), min=1.0)
            loss = loss + (per_token * mask_chunk).sum() / normalizer
        return loss, (torch.zeros(()),)


class _ShardedHeadStub(_Stub):
    """Stub whose head is gathered through the FSDP2 runtime."""

    def _liger_head_gather(self):
        return FSDPRuntime(FSDPConfig()).gather_layer(self.lm_head, device=self.device)


def _sharded_head_cispo_rank(rank: int, world_size: int, store_path: str) -> None:
    """One rank of the fused CISPO loss with a vocab-sharded frozen head."""
    dist.init_process_group(
        "gloo",
        store=dist.FileStore(store_path, world_size),
        rank=rank,
        world_size=world_size,
        timeout=datetime.timedelta(seconds=30),
    )
    try:
        grpo_module.HAS_LIGER_KERNEL = True
        grpo_module.LigerFusedLinearGRPOFunction = _ChunkedCispoKernel
        torch.manual_seed(0)
        dense_head = torch.nn.Linear(HIDDEN, VOCAB, bias=False).requires_grad_(False)
        sharded_head = torch.nn.Linear(HIDDEN, VOCAB, bias=False)
        sharded_head.weight = torch.nn.Parameter(
            distribute_tensor(
                dense_head.weight.clone(),
                init_device_mesh("cpu", (world_size,)),
                [Shard(0)],
            ),
            requires_grad=False,
        )
        lengths = RANK_ACTION_LENGTHS[rank]
        hidden, batch_ids, mask = _fused_inputs(lengths, RANK_WIDTH, seed=rank)
        hidden.requires_grad_(True)
        advantages = torch.linspace(-0.9, 0.7, len(lengths)).unsqueeze(-1)
        reference = _Stub(loss_norm="micro_batch", lm_head=dense_head)
        log_probs = _token_log_probs(reference, hidden, batch_ids, RANK_WIDTH)
        spread = torch.linspace(0.0, 0.1, log_probs.numel()).reshape(log_probs.shape)
        old_log_probs = (log_probs - spread).detach()
        algo = _ShardedHeadStub(
            loss_norm="micro_batch", loss_type="cispo", lm_head=sharded_head
        )
        algo.hidden = hidden

        loss, _, _, _ = algo._liger_loss(
            batch_ids, mask, advantages, old_log_probs, None
        )
        loss.backward()

        ratio = torch.exp(log_probs.detach() - old_log_probs).clamp(max=CLIP_MAX)
        per_token = -(ratio * advantages * log_probs)
        expected = _micro_batch_reduce(per_token, mask).mean()
        fused_hidden_grad = hidden.grad.clone()
        hidden.grad = None
        expected.backward()
        assert torch.allclose(loss, expected, atol=1e-6), (rank, loss, expected)
        assert torch.allclose(fused_hidden_grad, hidden.grad, atol=1e-6), rank
    except BaseException:
        traceback.print_exc()
        sys.exit(1)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(sys.platform == "win32", reason="gloo TCP transport is missing")
class TestFusedKernelCollectivesAcrossRanks:
    """Ranks with different action-row counts complete the sharded-head gather."""

    def test_uneven_row_chunks_complete_and_match_the_dense_reference(
        self, tmp_path: Path
    ) -> None:
        # Arrange
        world_size = len(RANK_ACTION_LENGTHS)
        context = multiprocessing.get_context("spawn")
        store_path = str(tmp_path / "store")
        processes = [
            context.Process(
                target=_sharded_head_cispo_rank,
                args=(rank, world_size, store_path),
            )
            for rank in range(world_size)
        ]

        # Act
        for process in processes:
            process.start()
        for process in processes:
            process.join(timeout=RANK_TIMEOUT_S)
        hung = [process.pid for process in processes if process.is_alive()]
        for process in processes:
            if process.is_alive():
                process.kill()

        # Assert
        assert hung == []
        assert [process.exitcode for process in processes] == [0] * world_size


WINDOW_RANK_LENGTHS = {
    "uneven": ([[0], [3]], [[9, 4], [7, 1]]),
    "empty": ([[0], [0]], [[0, 0], [0]]),
}
"""Per-scenario, per-rank micro-batch action lengths of one accumulation window."""


def _window_micro_batch(rank: int, step: int, lengths: list[int]):
    """Hidden states, token ids, action mask and advantages of one micro-batch."""
    hidden, batch_ids, mask = _fused_inputs(lengths, RANK_WIDTH, seed=10 * rank + step)
    advantages = torch.linspace(-0.8, 0.6, len(lengths)).unsqueeze(-1) + rank
    return hidden, batch_ids, mask, advantages


def _window_token_mean_rank(
    rank: int, world_size: int, store_path: str, scenario: str
) -> None:
    """One rank of an accumulation window on the standard and the fused path."""
    dist.init_process_group(
        "gloo",
        store=dist.FileStore(store_path, world_size),
        rank=rank,
        world_size=world_size,
        timeout=datetime.timedelta(seconds=30),
    )
    try:
        grpo_module.HAS_LIGER_KERNEL = True
        grpo_module.LigerFusedLinearGRPOFunction = _FakeFusedKernel
        windows = WINDOW_RANK_LENGTHS[scenario]
        union = [
            _window_micro_batch(other, step, lengths)
            for other, rank_window in enumerate(windows)
            for step, lengths in enumerate(rank_window)
        ]
        own = [
            _window_micro_batch(rank, step, lengths)
            for step, lengths in enumerate(windows[rank])
        ]
        steps = len(own)
        torch.manual_seed(0)
        initial_head = torch.nn.Linear(HIDDEN, VOCAB, bias=False)
        heads = {}
        for path in ("standard", "fused", "union"):
            heads[path] = torch.nn.Linear(HIDDEN, VOCAB, bias=False)
            heads[path].load_state_dict(initial_head.state_dict())
        standard = _Stub(loss_type="cispo", accumulation_steps=steps)
        standard.lm_head = heads["standard"]
        fused = _Stub(loss_type="cispo", accumulation_steps=steps)
        fused.lm_head = heads["fused"]
        window_mask = torch.cat([mask for _, _, mask, _ in own])
        window_rows = np.arange(window_mask.shape[0])
        standard._record_global_window_action_tokens(window_mask, window_rows)
        fused._record_global_window_action_tokens(window_mask, window_rows)

        losses = []
        for hidden, batch_ids, mask, advantages in own:
            log_probs = _token_log_probs(standard, hidden, batch_ids, RANK_WIDTH)
            standard_loss = standard._reduce_masked_loss(
                -advantages * log_probs, mask
            ).mean()
            (standard_loss / steps).backward()
            fused.hidden = hidden
            fused_loss, _, _, _ = fused._liger_loss(
                batch_ids, mask, advantages, None, None
            )
            (fused_loss / steps).backward()
            losses += [standard_loss.detach(), fused_loss.detach()]
        for path in ("standard", "fused"):
            grad = heads[path].weight.grad
            dist.all_reduce(grad, op=dist.ReduceOp.SUM)
            grad /= world_size

        reference = _Stub(loss_norm="micro_batch", lm_head=heads["union"])
        union_sum = torch.zeros(())
        union_tokens = 0.0
        for hidden, batch_ids, mask, advantages in union:
            log_probs = _token_log_probs(reference, hidden, batch_ids, RANK_WIDTH)
            union_sum = union_sum + (-advantages * log_probs * mask).sum()
            union_tokens += float(mask.sum())
        (union_sum / max(union_tokens, 1.0)).backward()
        expected = heads["union"].weight.grad

        assert all(torch.isfinite(loss) for loss in losses), (rank, losses)
        for path in ("standard", "fused"):
            assert torch.allclose(heads[path].weight.grad, expected, atol=1e-6), (
                rank,
                path,
                heads[path].weight.grad,
                expected,
            )
        if union_tokens == 0.0:
            assert all(loss.item() == 0.0 for loss in losses), (rank, losses)
    except BaseException:
        traceback.print_exc()
        sys.exit(1)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(sys.platform == "win32", reason="gloo TCP transport is missing")
class TestGRPOAccumulationWindowAcrossRanks:
    """The rank-averaged window gradient is the token mean over every rank's tokens."""

    @pytest.mark.parametrize("scenario", sorted(WINDOW_RANK_LENGTHS))
    def test_averaged_gradient_is_the_union_token_mean(
        self, tmp_path: Path, scenario: str
    ) -> None:
        # Arrange
        world_size = len(WINDOW_RANK_LENGTHS[scenario])
        context = multiprocessing.get_context("spawn")
        store_path = str(tmp_path / "store")
        processes = [
            context.Process(
                target=_window_token_mean_rank,
                args=(rank, world_size, store_path, scenario),
            )
            for rank in range(world_size)
        ]

        # Act
        for process in processes:
            process.start()
        for process in processes:
            process.join(timeout=RANK_TIMEOUT_S)
        hung = [process.pid for process in processes if process.is_alive()]
        for process in processes:
            if process.is_alive():
                process.kill()

        # Assert
        assert hung == []
        assert [process.exitcode for process in processes] == [0] * world_size
