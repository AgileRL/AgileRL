# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the chunked fused-logprob compile dispatch."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from agilerl.algorithms.core.base import LLMAlgorithm
from agilerl.algorithms.core.llm_ops.fused_logprobs import (
    _FUSED_LOGPROB_COMPILE_STATE,
    FusedLinearLogProbsFunction,
    _fused_logprob_chunk,
    _fused_logprob_chunk_dispatch,
    fused_linear_logprobs_chunked,
    scored_position_index,
    support_logprob_chunk,
    support_scores,
)


class TestScoredPositionIndex:
    def test_left_aligns_each_rows_scored_columns_in_order(self) -> None:
        mask = torch.tensor(
            [
                [0, 1, 1, 0, 1],
                [0, 0, 1, 0, 0],
            ],
            dtype=torch.bool,
        )

        index = scored_position_index(mask)

        assert torch.equal(index, torch.tensor([[1, 2, 4], [2, 0, 1]]))

    def test_pads_with_distinct_unscored_columns(self) -> None:
        mask = torch.tensor([[1, 1, 1, 1], [0, 1, 0, 0]], dtype=torch.bool)

        index = scored_position_index(mask)

        assert index.shape == (2, 4)
        assert torch.equal(index[1].sort().values, torch.arange(4))
        assert mask.gather(1, index).sum(dim=-1).tolist() == [4, 1]

    def test_an_empty_mask_keeps_one_column(self) -> None:
        mask = torch.zeros(2, 3, dtype=torch.bool)

        index = scored_position_index(mask)

        assert torch.equal(index, torch.zeros(2, 1, dtype=torch.long))


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
    return h, w, None, targets, 1.0, True, None


def test_fused_logprob_chunk_applies_bias_and_temperature():
    """Exercise the optional lm_head bias and temperature!=1 branches and
    confirm the result matches a plain log-softmax of the scaled, biased logits.
    """
    torch.manual_seed(0)
    h = torch.randn(4, 8)
    w = torch.randn(16, 8)
    bias = torch.randn(16)
    targets = torch.randint(0, 16, (4,))
    out = _fused_logprob_chunk(h, w, bias, targets, 2.0, True, None)
    expected = (
        ((h @ w.t() + bias) / 2.0)
        .log_softmax(dim=-1)
        .gather(dim=-1, index=targets.unsqueeze(-1))
        .squeeze(-1)
    )
    assert out.shape == (4,)
    assert torch.allclose(out, expected, atol=1e-5)


def test_fused_logprob_chunk_promotes_mismatched_dtypes():
    """fp32 hidden with an fp16 head promotes both to fp32 for the matmul."""
    torch.manual_seed(0)
    h = torch.randn(4, 8)
    w = torch.randn(16, 8, dtype=torch.float16)
    targets = torch.randint(0, 16, (4,))

    out = _fused_logprob_chunk(h, w, None, targets, 1.0, False, None)

    expected = (
        (h @ w.float().t())
        .log_softmax(dim=-1)
        .gather(dim=-1, index=targets.unsqueeze(-1))
        .squeeze(-1)
    )
    assert torch.allclose(out, expected, atol=1e-3)


def _softcapped_reference(
    h: torch.Tensor,
    w: torch.Tensor,
    targets: torch.Tensor,
    softcap: float,
    temperature: float,
) -> torch.Tensor:
    logits = softcap * torch.tanh((h @ w.t()) / softcap) / temperature
    return logits.log_softmax(dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)


class TestFusedLinearLogprobsSoftcap:
    """A softcap squashes the logits before the temperature, as the model head does."""

    def test_chunked_logprobs_match_softcapped_logits(self) -> None:
        torch.manual_seed(0)
        h = torch.randn(2, 5, 8)
        w = torch.randn(16, 8) * 4.0
        targets = torch.randint(0, 16, (2, 5))

        out = fused_linear_logprobs_chunked(
            h, w, None, targets, temperature=0.7, chunk_rows=3, softcap=3.0
        )

        expected = _softcapped_reference(h, w, targets, 3.0, 0.7)
        assert torch.allclose(out, expected, atol=1e-5)
        uncapped = fused_linear_logprobs_chunked(
            h, w, None, targets, temperature=0.7, chunk_rows=3
        )
        assert not torch.allclose(uncapped, expected, atol=1e-2)

    def test_gradient_matches_autograd_through_softcap(self) -> None:
        # Arrange
        torch.manual_seed(1)
        h = torch.randn(2, 5, 8, dtype=torch.float64, requires_grad=True)
        w = (torch.randn(16, 8, dtype=torch.float64) * 4.0).requires_grad_(True)
        targets = torch.randint(0, 16, (2, 5))

        # Act
        fused = FusedLinearLogProbsFunction.apply(
            h, w, None, targets, 0.7, False, 3, 3.0
        )
        fused_grads = torch.autograd.grad(fused.sum(), (h, w))
        reference = _softcapped_reference(h, w, targets, 3.0, 0.7)
        reference_grads = torch.autograd.grad(reference.sum(), (h, w))

        # Assert
        assert torch.allclose(fused_grads[0], reference_grads[0], atol=1e-10)
        assert torch.allclose(fused_grads[1], reference_grads[1], atol=1e-10)


def _biased_reference(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    targets: torch.Tensor,
    supports: torch.Tensor,
    logit_bias: torch.Tensor,
) -> torch.Tensor:
    """Masked log-softmax of ``(3 tanh(hW / 3) + logit_bias) / 0.7`` over each support."""
    logits = (3.0 * torch.tanh((hidden @ weight.t()) / 3.0) + logit_bias) / 0.7
    allowed = torch.ones_like(logits, dtype=torch.bool)
    for position, row in enumerate(supports[0]):
        ids = row[row >= 0].long()
        if ids.numel() > 0:
            allowed[0, position] = False
            allowed[0, position, ids] = True
    masked = logits.masked_fill(~allowed, float("-inf"))
    return masked.log_softmax(-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)


class TestFusedLinearLogprobsLogitBias:
    """A logit bias adds after the softcap and before the temperature, as vLLM applies it."""

    def test_chunked_logprobs_score_the_biased_logits(self) -> None:
        # Arrange
        hidden, weight, targets, supports = _support_case()
        logit_bias = torch.zeros(16, dtype=torch.float64)
        logit_bias[[7, 9]] = torch.tensor([-100.0, 1.5], dtype=torch.float64)
        expected = _biased_reference(hidden, weight, targets, supports, logit_bias)

        # Act
        logps = fused_linear_logprobs_chunked(
            hidden,
            weight,
            None,
            targets,
            temperature=0.7,
            cast_to_fp32=False,
            chunk_rows=2,
            softcap=3.0,
            support_ids=supports,
            logit_bias=logit_bias,
        )
        unbiased = fused_linear_logprobs_chunked(
            hidden,
            weight,
            None,
            targets,
            temperature=0.7,
            cast_to_fp32=False,
            chunk_rows=2,
            softcap=3.0,
            support_ids=supports,
        )

        # Assert
        assert torch.allclose(logps, expected, atol=1e-12)
        assert not torch.allclose(unbiased[0, [0, 2]], expected[0, [0, 2]], atol=1e-3)
        assert logps[0, 1].item() == 0.0

    def test_gradient_matches_autograd_through_the_bias(self) -> None:
        # Arrange
        hidden, weight, targets, supports = _support_case()
        hidden.requires_grad_(True)
        weight.requires_grad_(True)
        logit_bias = torch.zeros(16, dtype=torch.float64)
        logit_bias[[7, 9]] = torch.tensor([-100.0, 1.5], dtype=torch.float64)

        # Act
        fused = FusedLinearLogProbsFunction.apply(
            hidden, weight, None, targets, 0.7, False, 2, 3.0, supports, logit_bias
        )
        fused_grads = torch.autograd.grad(fused.sum(), (hidden, weight))
        reference = _biased_reference(hidden, weight, targets, supports, logit_bias)
        reference_grads = torch.autograd.grad(reference.sum(), (hidden, weight))

        # Assert
        assert torch.allclose(fused_grads[0], reference_grads[0], atol=1e-10)
        assert torch.allclose(fused_grads[1], reference_grads[1], atol=1e-10)


class TestLLMAlgorithmLogitBiasVector:
    def test_spreads_the_bias_over_the_vocabulary(self) -> None:
        algo = SimpleNamespace(logit_bias={2: -100.0, 5: 1.5}, device="cpu")

        vector = LLMAlgorithm._logit_bias_vector(algo, 6)

        assert vector is not None
        assert vector.dtype == torch.float32
        assert vector.tolist() == [0.0, 0.0, -100.0, 0.0, 0.0, 1.5]

    def test_without_a_bias_returns_none(self) -> None:
        algo = SimpleNamespace(logit_bias=None, device="cpu")

        assert LLMAlgorithm._logit_bias_vector(algo, 6) is None


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
        def boom(*args, **kwargs):
            msg = "triton backend exploded"
            raise RuntimeError(msg)

        expected = _fused_logprob_chunk(*_args())
        with patch("torch.compile", side_effect=lambda fn, **kw: boom):
            got = _fused_logprob_chunk_dispatch(torch.device("cuda"), *_args())
        assert torch.allclose(got, expected)
        assert _FUSED_LOGPROB_COMPILE_STATE["disabled"] is True
        # Latched: subsequent calls short-circuit to eager without recompiling.
        with patch("torch.compile") as mock_compile:
            again = _fused_logprob_chunk_dispatch(torch.device("cuda"), *_args())
        mock_compile.assert_not_called()
        assert torch.allclose(again, expected)


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
        softcap = None
        saved_tensors = (hidden, weight, None, targets, None, None)

    grad_output = torch.ones(1, 4)
    grads = FusedLinearLogProbsFunction.backward(Ctx(), grad_output)
    assert grads[0] is None
    assert grads[1] is None


def _support_case() -> tuple[torch.Tensor, ...]:
    """Hidden states, head, targets and supports for one restricted and one free row."""
    torch.manual_seed(0)
    hidden = torch.randn(1, 3, 8, dtype=torch.float64)
    weight = torch.randn(16, 8, dtype=torch.float64)
    targets = torch.tensor([[5, 2, 9]])
    # Position 0: support {5, 7}; position 1: support {2} (forced); position 2: none.
    supports = torch.tensor([[[5, 7], [2, -1], [-1, -1]]], dtype=torch.int32)
    return hidden, weight, targets, supports


def _reference_logprobs(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    targets: torch.Tensor,
    supports: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """Per-token logprobs from a masked log-softmax over each position's support."""
    logits = hidden @ weight.t() / temperature
    allowed = torch.zeros_like(logits, dtype=torch.bool)
    for position, row in enumerate(supports[0]):
        ids = row[row >= 0].long()
        if ids.numel() == 0:
            allowed[0, position] = True
        else:
            allowed[0, position, ids] = True
    masked = logits.masked_fill(~allowed, float("-inf"))
    return masked.log_softmax(-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)


class TestSupportScores:
    def test_sums_the_scores_of_the_support_only(self) -> None:
        scores = torch.tensor([[0.0, 1.0, 2.0, 3.0]])
        support = torch.tensor([[1, 3, -1]], dtype=torch.int32)

        selected, log_z, restricted = support_scores(scores, support, torch.tensor([1]))

        assert selected.tolist() == [1.0]
        assert torch.allclose(log_z, torch.logsumexp(torch.tensor([[1.0, 3.0]]), -1))
        assert restricted.tolist() == [True]

    def test_a_row_without_support_stays_finite_and_unrestricted(self) -> None:
        scores = torch.tensor([[0.0, 1.0, 2.0]], requires_grad=True)
        support = torch.tensor([[-1, -1]], dtype=torch.int32)

        selected, log_z, restricted = support_scores(scores, support, torch.tensor([2]))
        log_z.sum().backward()

        assert selected.tolist() == [2.0]
        assert restricted.tolist() == [False]
        assert torch.isfinite(log_z).all()
        assert torch.isfinite(scores.grad).all()


class TestFusedLinearLogprobsChunked:
    def test_normalizes_each_position_over_its_support(self) -> None:
        # Arrange
        hidden, weight, targets, supports = _support_case()
        expected = _reference_logprobs(hidden, weight, targets, supports, 0.7)

        # Act
        logps = fused_linear_logprobs_chunked(
            hidden,
            weight,
            None,
            targets,
            temperature=0.7,
            cast_to_fp32=False,
            chunk_rows=2,
            support_ids=supports,
        )

        # Assert
        assert torch.allclose(logps, expected, atol=1e-12)
        assert logps[0, 1].item() == 0.0

    def test_without_supports_normalizes_over_the_vocabulary(self) -> None:
        hidden, weight, targets, _supports = _support_case()
        full = torch.full_like(_supports, -1)
        expected = _reference_logprobs(hidden, weight, targets, full, 1.0)

        logps = fused_linear_logprobs_chunked(
            hidden, weight, None, targets, cast_to_fp32=False, chunk_rows=2
        )

        assert torch.allclose(logps, expected, atol=1e-12)

    def test_rows_with_a_support_skip_the_vocabulary_matmul(self) -> None:
        # Arrange
        hidden, weight, targets, supports = _support_case()
        supports = supports[:, :2]
        expected = _reference_logprobs(
            hidden[:, :2], weight, targets[:, :2], supports, 0.7
        )

        # Act
        with patch(
            "agilerl.algorithms.core.llm_ops.fused_logprobs."
            "_fused_logprob_chunk_dispatch",
            side_effect=AssertionError("full-vocab chunk ran"),
        ):
            logps = fused_linear_logprobs_chunked(
                hidden[:, :2],
                weight,
                None,
                targets[:, :2],
                temperature=0.7,
                cast_to_fp32=False,
                chunk_rows=1,
                support_ids=supports,
            )

        # Assert
        assert torch.allclose(logps, expected, atol=1e-12)


class TestLLMAlgorithmScoreHidden:
    def test_padding_of_a_supports_batch_stays_off_the_vocabulary_matmul(
        self,
    ) -> None:
        # Arrange
        torch.manual_seed(4)
        hidden = torch.randn(2, 3, 8, dtype=torch.float64)
        weight = torch.randn(16, 8, dtype=torch.float64)
        targets = torch.tensor([[5, 2, 9], [1, 3, 4]])
        supports = torch.tensor(
            [[[5, 7], [2, 6], [-1, -1]], [[-1, -1], [3, 8], [-1, -1]]],
            dtype=torch.int32,
        )
        score_mask = supports[..., 0] >= 0
        algo = SimpleNamespace(
            temperature=0.7, cast_logprobs_to_fp32=False, chunk_rows=2
        )
        expected = _reference_logprobs(
            hidden[:1, :2], weight, targets[:1, :2], supports[:1, :2], 0.7
        )

        # Act
        with patch(
            "agilerl.algorithms.core.llm_ops.fused_logprobs."
            "_fused_logprob_chunk_dispatch",
            side_effect=AssertionError("full-vocab chunk ran"),
        ):
            logps = LLMAlgorithm._score_hidden(
                algo,
                fused_linear_logprobs_chunked,
                hidden,
                weight,
                None,
                targets,
                score_mask,
                supports,
            )

        # Assert
        assert torch.allclose(logps[0, :2], expected[0], atol=1e-12)
        assert logps[0, 2].item() == 0.0
        assert logps[1, 0].item() == 0.0
        assert logps[1, 2].item() == 0.0
        assert logps[1, 1].item() < 0.0


class TestSupportLogprobChunk:
    def test_matches_the_masked_log_softmax_with_bias_and_softcap(self) -> None:
        # Arrange
        torch.manual_seed(2)
        hidden = torch.randn(3, 8, dtype=torch.float64)
        weight = torch.randn(16, 8, dtype=torch.float64) * 4.0
        bias = torch.randn(16, dtype=torch.float64)
        targets = torch.tensor([5, 2, 11])
        supports = torch.tensor([[7, 5, 1], [2, -1, -1], [11, 3, -1]])
        logits = 3.0 * torch.tanh((hidden @ weight.t() + bias) / 3.0) / 0.7
        allowed = torch.zeros_like(logits, dtype=torch.bool)
        for row, ids in enumerate(supports):
            allowed[row, ids[ids >= 0]] = True
        expected = (
            logits.masked_fill(~allowed, float("-inf"))
            .log_softmax(-1)
            .gather(-1, targets.unsqueeze(-1))
            .squeeze(-1)
        )

        # Act
        logps = support_logprob_chunk(
            hidden, weight, bias, targets, 0.7, False, 3.0, supports
        )

        # Assert
        assert torch.allclose(logps, expected, atol=1e-12)

    def test_upcasts_a_bf16_head_to_the_fp32_result(self) -> None:
        torch.manual_seed(3)
        hidden = torch.randn(2, 8, dtype=torch.bfloat16)
        weight = torch.randn(16, 8, dtype=torch.bfloat16)
        targets = torch.tensor([4, 9])
        supports = torch.tensor([[4, 9], [9, 4]])
        logits = hidden.float() @ weight.float().t()
        expected = logits[:, [4, 9]].log_softmax(-1)[[0, 1], [0, 1]]

        logps = support_logprob_chunk(
            hidden, weight, None, targets, 1.0, True, None, supports
        )

        assert logps.dtype == torch.float32
        assert torch.allclose(logps, expected, atol=1e-6)

    def test_a_target_outside_its_support_has_probability_zero(self) -> None:
        hidden = torch.randn(1, 8)
        weight = torch.randn(16, 8)

        logps = support_logprob_chunk(
            hidden,
            weight,
            None,
            torch.tensor([3]),
            1.0,
            False,
            None,
            torch.tensor([[1, 2]]),
        )

        assert logps.item() == float("-inf")


class TestFusedLinearLogProbsFunctionSupport:
    def test_gradient_matches_the_masked_log_softmax(self) -> None:
        # Arrange
        hidden, weight, targets, supports = _support_case()
        hidden.requires_grad_(True)
        weight.requires_grad_(True)
        ref_hidden = hidden.detach().clone().requires_grad_(True)
        ref_weight = weight.detach().clone().requires_grad_(True)
        _reference_logprobs(
            ref_hidden, ref_weight, targets, supports, 0.7
        ).sum().backward()

        # Act
        FusedLinearLogProbsFunction.apply(
            hidden, weight, None, targets, 0.7, False, 2, None, supports
        ).sum().backward()

        # Assert
        assert torch.allclose(hidden.grad, ref_hidden.grad, atol=1e-12)
        assert torch.allclose(weight.grad, ref_weight.grad, atol=1e-12)
        # A forced token carries no gradient into its hidden state.
        assert torch.count_nonzero(hidden.grad[0, 1]) == 0
