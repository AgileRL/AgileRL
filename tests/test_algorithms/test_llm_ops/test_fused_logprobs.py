# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the chunked fused-logprob compile dispatch."""

from contextlib import contextmanager
from unittest.mock import patch

import pytest
import torch

from agilerl.algorithms.core.llm_ops.fused_logprobs import (
    _FUSED_LOGPROB_COMPILE_STATE,
    FusedLinearLogProbsFunction,
    _fused_logprob_chunk,
    _fused_logprob_chunk_dispatch,
    fp32_lm_head_operands,
    fused_linear_logprobs_chunked,
)


@pytest.fixture(autouse=True)
def _fresh_compile_state():
    """Isolate the process-global compile cache between tests."""
    saved = dict(_FUSED_LOGPROB_COMPILE_STATE)
    _FUSED_LOGPROB_COMPILE_STATE.update({"fn": None, "disabled": False})
    yield
    _FUSED_LOGPROB_COMPILE_STATE.update(saved)


def _args():
    torch.manual_seed(0)
    h = torch.randn(4, 8)
    w = torch.randn(16, 8)
    targets = torch.randint(0, 16, (4,))
    return h, w, None, targets, 1.0, True


def test_fused_logprob_chunk_applies_bias_and_temperature():
    """Exercise the optional lm_head bias and temperature!=1 branches and
    confirm the result matches a plain log-softmax of the scaled, biased logits.
    """
    torch.manual_seed(0)
    h = torch.randn(4, 8)
    w = torch.randn(16, 8)
    bias = torch.randn(16)
    targets = torch.randint(0, 16, (4,))
    out = _fused_logprob_chunk(h, w, bias, targets, 2.0, True)
    expected = (
        ((h @ w.t() + bias) / 2.0)
        .log_softmax(dim=-1)
        .gather(dim=-1, index=targets.unsqueeze(-1))
        .squeeze(-1)
    )
    assert out.shape == (4,)
    assert torch.allclose(out, expected, atol=1e-5)


class TestFusedLogprobChunkDispatch:
    def test_non_cuda_device_uses_eager(self):
        expected = _fused_logprob_chunk(*_args())
        got = _fused_logprob_chunk_dispatch(torch.device("cpu"), *_args())
        assert torch.allclose(got, expected)
        assert _FUSED_LOGPROB_COMPILE_STATE["fn"] is None

    def test_cuda_device_compiles_and_caches(self):
        # torch.compile is stubbed so the dispatch logic runs without a GPU;
        # the "compiled" callable is just the eager fn.
        with patch("torch.compile", side_effect=lambda fn, **kw: fn) as mock_compile:
            first = _fused_logprob_chunk_dispatch(torch.device("cuda"), *_args())
            second = _fused_logprob_chunk_dispatch(torch.device("cuda"), *_args())
        mock_compile.assert_called_once()  # cached after the first call
        assert torch.allclose(first, second)
        assert _FUSED_LOGPROB_COMPILE_STATE["fn"] is not None

    def test_compiled_failure_latches_eager_fallback(self):
        def _raise_compile_error(*args, **kwargs):
            msg = "triton backend exploded"
            raise RuntimeError(msg)

        expected = _fused_logprob_chunk(*_args())
        with patch("torch.compile", side_effect=lambda fn, **kw: _raise_compile_error):
            got = _fused_logprob_chunk_dispatch(torch.device("cuda"), *_args())
        assert torch.allclose(got, expected)
        assert _FUSED_LOGPROB_COMPILE_STATE["disabled"] is True
        # Latched: subsequent calls short-circuit to eager without recompiling.
        with patch("torch.compile") as mock_compile:
            again = _fused_logprob_chunk_dispatch(torch.device("cuda"), *_args())
        mock_compile.assert_not_called()
        assert torch.allclose(again, expected)


class TestFp32CastBeforeMatmul:
    """``cast_to_fp32`` must upcast the matmul operands, not its product: a
    reduced-precision product that saturated to ``inf`` makes ``selected -
    log_z`` an ``inf - inf`` NaN no matter what it is cast to afterwards.
    """

    def test_overflowing_fp16_logits_stay_finite(self):
        # 8 * 300 * 300 = 720_000, well past the fp16 max of 65_504.
        h = torch.full((1, 2, 8), 300.0, dtype=torch.float16)
        w = torch.full((16, 8), 300.0, dtype=torch.float16)
        targets = torch.zeros(1, 2, dtype=torch.long)
        assert not torch.isfinite(h.reshape(2, 8) @ w.t()).any()

        out = fused_linear_logprobs_chunked(h, w, None, targets, chunk_rows=1)

        assert torch.isfinite(out).all()
        # Uniform logits: every token's logprob is -log(V), to the fp32
        # resolution available at a logit magnitude of 720_000.
        expected = torch.full((1, 2), -torch.tensor(16.0).log().item())
        assert torch.allclose(out.float(), expected, atol=0.1)

    def test_overflowing_fp16_bias_stays_finite(self):
        h = torch.full((1, 1, 4), 100.0, dtype=torch.float16)
        w = torch.full((3, 4), 75.0, dtype=torch.float16)
        bias = torch.full((3,), 60000.0, dtype=torch.float16)
        targets = torch.zeros(1, 1, dtype=torch.long)
        assert not torch.isfinite(h.reshape(1, 4) @ w.t() + bias).any()

        out = fused_linear_logprobs_chunked(h, w, bias, targets, chunk_rows=1)

        assert torch.isfinite(out).all()

    def test_cast_disabled_keeps_operand_dtype(self):
        torch.manual_seed(0)
        h = (torch.randn(4, 8) * 0.02).to(torch.float16)
        w = (torch.randn(16, 8) * 0.02).to(torch.float16)
        targets = torch.randint(0, 16, (4,))

        out = _fused_logprob_chunk(h, w, None, targets, 1.0, False)

        assert out.dtype == torch.float16


class TestFp32LmHeadOperands:
    """A ``(V, H)`` fp32 copy is gigabytes at production vocab sizes, so a chunk
    loop takes at most one, shared by every chunk.
    """

    def test_returns_the_same_objects_when_no_cast_is_needed(self):
        w = torch.randn(16, 8, dtype=torch.bfloat16)
        bias = torch.randn(16, dtype=torch.bfloat16)

        assert fp32_lm_head_operands(w, bias, False) == (w, bias)

        w32 = w.float()
        bias32 = bias.float()
        assert fp32_lm_head_operands(w32, bias32, True) == (w32, bias32)

    def test_upcasts_weight_and_bias_together(self):
        w = torch.randn(16, 8, dtype=torch.bfloat16)
        bias = torch.randn(16, dtype=torch.bfloat16)

        head_weight, head_bias = fp32_lm_head_operands(w, bias, True)

        assert head_weight.dtype == torch.float32
        assert head_bias.dtype == torch.float32
        assert torch.equal(head_weight, w.float())

    def test_missing_bias_stays_none(self):
        w = torch.randn(16, 8, dtype=torch.bfloat16)

        assert fp32_lm_head_operands(w, None, True)[1] is None


class TestLmHeadUpcastIsHoistedOutOfTheChunkLoop:
    """Every chunk of a loop must receive one and the same fp32 head."""

    @staticmethod
    def _recording_chunk(seen):
        def chunk(h_chunk, lm_head_weight, lm_head_bias, target_chunk, temp, cast):
            seen.append((lm_head_weight.data_ptr(), lm_head_weight.dtype))
            return _fused_logprob_chunk(
                h_chunk, lm_head_weight, lm_head_bias, target_chunk, temp, cast
            )

        return chunk

    def test_forward_upcasts_the_head_once(self):
        torch.manual_seed(0)
        hidden = torch.randn(2, 6, 8, dtype=torch.bfloat16)
        weight = (torch.randn(16, 8) * 0.02).to(torch.bfloat16)
        targets = torch.randint(0, 16, (2, 6))
        seen = []

        with patch(
            "agilerl.algorithms.core.llm_ops.fused_logprobs._fused_logprob_chunk",
            new=self._recording_chunk(seen),
        ):
            fused_linear_logprobs_chunked(hidden, weight, None, targets, chunk_rows=3)

        assert len(seen) == 4  # 12 rows / chunk_rows=3
        assert {dtype for _ptr, dtype in seen} == {torch.float32}
        assert len({ptr for ptr, _dtype in seen}) == 1
        assert seen[0][0] != weight.data_ptr()

    def test_backward_recompute_upcasts_the_head_once(self):
        torch.manual_seed(0)
        hidden = torch.randn(2, 6, 8, dtype=torch.bfloat16, requires_grad=True)
        weight = (torch.randn(16, 8) * 0.02).to(torch.bfloat16)
        targets = torch.randint(0, 16, (2, 6))
        logps = FusedLinearLogProbsFunction.apply(
            hidden, weight, None, targets, 1.0, True, 3
        )
        seen = []

        with patch(
            "agilerl.algorithms.core.llm_ops.fused_logprobs._fused_logprob_chunk",
            new=self._recording_chunk(seen),
        ):
            logps.sum().backward()

        assert len(seen) == 4
        assert {dtype for _ptr, dtype in seen} == {torch.float32}
        assert len({ptr for ptr, _dtype in seen}) == 1
        assert hidden.grad is not None
        assert torch.isfinite(hidden.grad).all()


class TestMixedDtypeOperands:
    """An fp16 checkpoint under the bf16 autocast reaches the fused matmul
    with fp32 hidden states and an fp16 lm_head weight; the kernel
    promotes to a common dtype rather than crashing on the operand mismatch.
    """

    def test_chunk_promotes_mismatched_dtypes_to_common_dtype(self):
        torch.manual_seed(0)
        h = torch.randn(4, 8)  # fp32 hidden (autocast-promoted final norm)
        w = (torch.randn(16, 8) * 0.02).to(torch.float16)  # checkpoint-dtype head
        bias = torch.randn(16).to(torch.float16)
        targets = torch.randint(0, 16, (4,))

        out = _fused_logprob_chunk(h, w, bias, targets, 1.0, True)
        # fp16 -> fp32 casts are exact, so promoting inside the kernel must
        # bit-match computing with pre-upcast operands.
        expected = _fused_logprob_chunk(h, w.float(), bias.float(), targets, 1.0, True)
        assert out.dtype == torch.float32
        assert torch.equal(out, expected)

    def test_autograd_function_backward_with_mismatched_dtypes(self):
        torch.manual_seed(0)
        hidden = torch.randn(2, 4, 8, requires_grad=True)
        weight = (torch.randn(16, 8) * 0.02).to(torch.float16).requires_grad_(True)
        targets = torch.randint(0, 16, (2, 4))

        logps = FusedLinearLogProbsFunction.apply(
            hidden, weight, None, targets, 1.0, True, 3
        )
        logps.sum().backward()
        # Gradients go to the leaves in their own dtypes.
        assert hidden.grad is not None
        assert hidden.grad.dtype == torch.float32
        assert weight.grad is not None
        assert weight.grad.dtype == torch.float16
        assert hidden.grad.abs().sum() > 0
        assert weight.grad.abs().sum() > 0


def test_fused_logprob_backward_skips_when_no_inputs_require_grad():
    hidden = torch.randn(1, 4, 8)
    weight = torch.randn(16, 8)
    targets = torch.randint(0, 16, (1, 4))

    class Ctx:
        needs_hidden_grad = False
        needs_weight_grad = False
        needs_bias_grad = False
        chunk_rows = 2
        temperature = 1.0
        cast_to_fp32 = True
        saved_tensors = (hidden, weight, None, targets)

    grad_output = torch.ones(1, 4)
    grads = FusedLinearLogProbsFunction.backward(Ctx(), grad_output)
    assert grads[0] is None
    assert grads[1] is None


def test_fused_logprob_backward_rejects_zero3_trainable_lm_head():
    """ZeRO-3 partitioned lm_head cannot be trained through the fused path."""
    hidden = torch.randn(1, 4, 8)
    weight = torch.randn(16, 8)
    weight.ds_id = 0
    targets = torch.randint(0, 16, (1, 4))

    class Ctx:
        needs_hidden_grad = False
        needs_weight_grad = True
        needs_bias_grad = False
        chunk_rows = 2
        temperature = 1.0
        cast_to_fp32 = True
        saved_tensors = (hidden, weight, None, targets)

    with pytest.raises(RuntimeError, match="ZeRO-3 full-finetune of lm_head"):
        FusedLinearLogProbsFunction.backward(Ctx(), torch.ones(1, 4))


def test_fused_linear_logprobs_chunked_gathers_empty_zero3_shard():
    """ZeRO-3 empty local shard must be materialized before the fused matmul."""
    pytest.importorskip("deepspeed", reason="ZeRO-3 gather mock requires deepspeed.")
    torch.manual_seed(0)
    V, H = 16, 8
    B, T = 2, 4
    dtype = torch.float32
    device = torch.device("cpu")

    # Local placeholder matches a non-owner ZeRO-3 partition (numel == 0).
    weight = torch.nn.Parameter(torch.empty(0, dtype=dtype, device=device))
    weight.ds_id = 0
    weight.ds_shape = (V, H)
    full_weight = torch.randn(V, H, dtype=dtype, device=device)

    gather_entered = 0
    shapes_inside: list[tuple[int, ...]] = []

    @contextmanager
    def materialize_gather(params=None, modifier_rank=None):
        nonlocal gather_entered
        gather_entered += 1
        p = next(iter(params))
        saved = p.data
        p.data = full_weight.clone()
        shapes_inside.append(tuple(p.shape))
        try:
            yield
        finally:
            p.data = saved

    hidden = torch.randn(B, T, H, dtype=dtype, device=device)
    targets = torch.randint(0, V, (B, T), device=device)

    with patch(
        "deepspeed.zero.GatheredParameters",
        side_effect=materialize_gather,
    ):
        out = fused_linear_logprobs_chunked(hidden, weight, None, targets, chunk_rows=4)

    assert gather_entered == 1
    assert shapes_inside[0][0] != 0
    assert shapes_inside[0] == (V, H)
    assert out.shape == (B, T)
    assert torch.isfinite(out).all()
