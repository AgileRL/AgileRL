# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the chunked fused-logprob compile dispatch."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from torch.utils.flop_counter import FlopCounterMode

from agilerl.algorithms.core.base import LLMAlgorithm
from agilerl.algorithms.core.llm_ops.fused_logprobs import (
    _FUSED_LOGPROB_COMPILE_STATE,
    FusedLinearLogProbsFunction,
    HeadLogitsFunction,
    _fused_logprob_chunk,
    _fused_logprob_chunk_dispatch,
    _support_hidden_grad_dispatch,
    _support_logprob_chunk_dispatch,
    fused_linear_logprobs_chunked,
    split_support_rows,
    support_chunk_rows,
    support_logprob_chunk,
    support_logprob_hidden_grad,
    support_scores,
)


@pytest.fixture(autouse=True)
def _fresh_compile_state():
    """Isolate the process-global compile cache between tests."""
    saved = dict(_FUSED_LOGPROB_COMPILE_STATE)
    _FUSED_LOGPROB_COMPILE_STATE.update(
        {"fn": None, "support_fn": None, "support_grad_fn": None}
    )
    yield
    _FUSED_LOGPROB_COMPILE_STATE.update(saved)


def _args():
    torch.manual_seed(0)
    h = torch.randn(4, 8)
    w = torch.randn(16, 8)
    targets = torch.randint(0, 16, (4,))
    return h, w, None, targets, 1.0, None


def test_fused_logprob_chunk_applies_bias_and_temperature():
    """Exercise the optional lm_head bias and temperature!=1 branches and
    confirm the result matches a plain log-softmax of the scaled, biased logits.
    """
    torch.manual_seed(0)
    h = torch.randn(4, 8)
    w = torch.randn(16, 8)
    bias = torch.randn(16)
    targets = torch.randint(0, 16, (4,))
    out = _fused_logprob_chunk(h, w, bias, targets, 2.0, None)
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

    out = _fused_logprob_chunk(h, w, None, targets, 1.0, None)

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
        h = torch.randn(2, 5, 8, dtype=torch.float32, requires_grad=True)
        w = (torch.randn(16, 8, dtype=torch.float32) * 4.0).requires_grad_(True)
        targets = torch.randint(0, 16, (2, 5))

        # Act
        fused = FusedLinearLogProbsFunction.apply(h, w, None, targets, 0.7, 3, 3.0)
        fused_grads = torch.autograd.grad(fused.sum(), (h, w))
        reference = _softcapped_reference(h, w, targets, 3.0, 0.7)
        reference_grads = torch.autograd.grad(reference.sum(), (h, w))

        # Assert
        assert torch.allclose(fused_grads[0], reference_grads[0], atol=1e-5)
        assert torch.allclose(fused_grads[1], reference_grads[1], atol=1e-5)


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
        logit_bias = torch.zeros(16, dtype=torch.float32)
        logit_bias[[7, 9]] = torch.tensor([-100.0, 1.5], dtype=torch.float32)
        expected = _biased_reference(hidden, weight, targets, supports, logit_bias)

        # Act
        logps = fused_linear_logprobs_chunked(
            hidden,
            weight,
            None,
            targets,
            temperature=0.7,
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
            chunk_rows=2,
            softcap=3.0,
            support_ids=supports,
        )

        # Assert
        assert torch.allclose(logps, expected, atol=1e-6)
        assert not torch.allclose(unbiased[0, [0, 2]], expected[0, [0, 2]], atol=1e-3)
        assert logps[0, 1].item() == 0.0

    def test_gradient_matches_autograd_through_the_bias(self) -> None:
        # Arrange
        hidden, weight, targets, supports = _support_case()
        hidden.requires_grad_(True)
        weight.requires_grad_(True)
        logit_bias = torch.zeros(16, dtype=torch.float32)
        logit_bias[[7, 9]] = torch.tensor([-100.0, 1.5], dtype=torch.float32)

        # Act
        fused = FusedLinearLogProbsFunction.apply(
            hidden, weight, None, targets, 0.7, 2, 3.0, supports, logit_bias
        )
        fused_grads = torch.autograd.grad(fused.sum(), (hidden, weight))
        reference = _biased_reference(hidden, weight, targets, supports, logit_bias)
        reference_grads = torch.autograd.grad(reference.sum(), (hidden, weight))

        # Assert
        assert torch.allclose(fused_grads[0], reference_grads[0], atol=1e-5)
        assert torch.allclose(fused_grads[1], reference_grads[1], atol=1e-5)


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

    def test_compiled_failure_raises_without_fallback(self):
        def boom(*args, **kwargs):
            msg = "triton backend exploded"
            raise RuntimeError(msg)

        with patch("torch.compile", side_effect=lambda fn, **kw: boom):
            with pytest.raises(RuntimeError, match="triton backend exploded"):
                _fused_logprob_chunk_dispatch(torch.device("cuda"), *_args())
            # No eager fallback is latched: the cached compiled fn keeps raising.
            with pytest.raises(RuntimeError, match="triton backend exploded"):
                _fused_logprob_chunk_dispatch(torch.device("cuda"), *_args())


def _support_args(device: str = "cpu") -> tuple:
    torch.manual_seed(0)
    hidden = torch.randn(6, 8, device=device)
    weight = torch.randn(16, 8, device=device)
    bias = torch.randn(16, device=device)
    supports = torch.randint(0, 16, (6, 4), device=device, dtype=torch.int32)
    supports[1, 2:] = -1
    targets = supports[:, 0].long()
    return hidden, weight, bias, targets, 0.7, 3.0, supports


class TestSupportLogprobChunkDispatch:
    def test_non_cuda_device_uses_eager(self) -> None:
        expected = support_logprob_chunk(*_support_args())

        got = _support_logprob_chunk_dispatch(torch.device("cpu"), *_support_args())

        assert torch.equal(got, expected)
        assert _FUSED_LOGPROB_COMPILE_STATE["support_fn"] is None

    def test_cuda_device_compiles_once(self) -> None:
        # torch.compile is stubbed so the dispatch logic runs without a GPU.
        with patch("torch.compile", side_effect=lambda fn, **kw: fn) as mock_compile:
            first = _support_logprob_chunk_dispatch(
                torch.device("cuda"), *_support_args()
            )
            second = _support_logprob_chunk_dispatch(
                torch.device("cuda"), *_support_args()
            )

        mock_compile.assert_called_once_with(support_logprob_chunk, dynamic=True)
        assert torch.equal(first, second)
        assert torch.equal(first, support_logprob_chunk(*_support_args()))

    @pytest.mark.gpu
    @pytest.mark.parametrize("rows", [6, 5, 3])
    def test_compiled_matches_eager_logprobs_and_hidden_grad(self, rows) -> None:
        # Arrange: fp32 throughout; compiled kernels reorder reductions only.
        hidden, weight, bias, targets, temperature, softcap, supports = _support_args(
            "cuda"
        )
        upstream = torch.linspace(-1.0, 1.0, rows, device="cuda")

        def logps_and_grad(fn) -> tuple[torch.Tensor, torch.Tensor]:
            h = hidden[:rows].clone().requires_grad_(True)
            logps = fn(
                h,
                weight,
                bias,
                targets[:rows],
                temperature,
                softcap,
                supports[:rows],
            )
            (grad,) = torch.autograd.grad(logps, h, grad_outputs=upstream)
            return logps.detach(), grad

        # Act
        eager = logps_and_grad(support_logprob_chunk)
        compiled = logps_and_grad(
            lambda *args: _support_logprob_chunk_dispatch(torch.device("cuda"), *args)
        )

        # Assert
        assert torch.allclose(compiled[0], eager[0], rtol=1e-5, atol=1e-5)
        assert torch.allclose(compiled[1], eager[1], rtol=1e-5, atol=1e-5)


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
        softcap = None
        saved_tensors = (hidden, weight, None, targets, None, None, None, None)

    grad_output = torch.ones(1, 4)
    grads = FusedLinearLogProbsFunction.backward(Ctx(), grad_output)
    assert grads[0] is None
    assert grads[1] is None


def _support_case() -> tuple[torch.Tensor, ...]:
    """Hidden states, head, targets and supports for one restricted and one free row."""
    torch.manual_seed(0)
    hidden = torch.randn(1, 3, 8, dtype=torch.float32)
    weight = torch.randn(16, 8, dtype=torch.float32)
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
            chunk_rows=2,
            support_ids=supports,
        )

        # Assert
        assert torch.allclose(logps, expected, atol=1e-6)
        assert logps[0, 1].item() == 0.0

    def test_without_supports_normalizes_over_the_vocabulary(self) -> None:
        hidden, weight, targets, _supports = _support_case()
        full = torch.full_like(_supports, -1)
        expected = _reference_logprobs(hidden, weight, targets, full, 1.0)

        logps = fused_linear_logprobs_chunked(
            hidden, weight, None, targets, chunk_rows=2
        )

        assert torch.allclose(logps, expected, atol=1e-6)

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
                chunk_rows=1,
                support_ids=supports,
            )

        # Assert
        assert torch.allclose(logps, expected, atol=1e-6)

    def test_a_vocab_wide_support_gathers_one_row_at_a_time(self) -> None:
        # Arrange
        hidden, weight, targets, _supports = _support_case()
        supports = torch.arange(16, dtype=torch.int32).expand(1, 3, 16)
        expected = _reference_logprobs(hidden, weight, targets, supports, 1.0)
        gathered_rows = []

        def record_rows(h_chunk, *args):
            gathered_rows.append(h_chunk.shape[0])
            return support_logprob_chunk(h_chunk, *args)

        # Act
        with patch(
            "agilerl.algorithms.core.llm_ops.fused_logprobs.support_logprob_chunk",
            side_effect=record_rows,
        ):
            logps = fused_linear_logprobs_chunked(
                hidden,
                weight,
                None,
                targets,
                chunk_rows=2,
                support_ids=supports,
            )

        # Assert
        assert torch.allclose(logps, expected, atol=1e-12)
        assert gathered_rows == [1, 1, 1]


def _plain_logprobs(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    targets: torch.Tensor,
) -> torch.Tensor:
    """Full-vocabulary log-softmax of ``hidden @ weight.T + bias`` in the inputs' dtype."""
    logits = hidden @ weight.t()
    if bias is not None:
        logits = logits + bias
    return logits.log_softmax(-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)


def _leaves(*tensors: torch.Tensor) -> list[torch.Tensor]:
    return [tensor.detach().clone().requires_grad_(True) for tensor in tensors]


class TestFusedLinearLogProbsFunctionReference:
    def test_fp32_values_and_gradients_match_a_plain_log_softmax(self) -> None:
        # Arrange
        torch.manual_seed(5)
        hidden, weight, bias = _leaves(
            torch.randn(2, 7, 16), torch.randn(40, 16), torch.randn(40)
        )
        targets = torch.randint(0, 40, (2, 7))
        ref_hidden, ref_weight, ref_bias = _leaves(hidden, weight, bias)
        expected = _plain_logprobs(ref_hidden, ref_weight, ref_bias, targets)
        expected_grads = torch.autograd.grad(
            expected.sum(), (ref_hidden, ref_weight, ref_bias)
        )

        # Act
        logps = FusedLinearLogProbsFunction.apply(
            hidden, weight, bias, targets, 1.0, 3, None
        )
        grads = torch.autograd.grad(logps.sum(), (hidden, weight, bias))

        # Assert: both sides run fp32; 3-row chunks only reorder the sums.
        assert logps.dtype == torch.float32
        assert torch.allclose(logps, expected, atol=1e-5)
        for got, want in zip(grads, expected_grads, strict=True):
            assert torch.allclose(got, want, atol=1e-5)

    def test_bf16_cpu_head_matches_the_fp32_reference_of_its_operands(self) -> None:
        # Arrange
        torch.manual_seed(6)
        hidden, weight = _leaves(
            torch.randn(2, 7, 16, dtype=torch.bfloat16),
            torch.randn(40, 16, dtype=torch.bfloat16),
        )
        targets = torch.randint(0, 40, (2, 7))
        ref_hidden, ref_weight = _leaves(hidden.float(), weight.float())
        expected = _plain_logprobs(ref_hidden, ref_weight, None, targets)
        expected_grads = torch.autograd.grad(expected.sum(), (ref_hidden, ref_weight))

        # Act
        logps = FusedLinearLogProbsFunction.apply(
            hidden, weight, None, targets, 1.0, 3, None
        )
        grad_hidden, grad_weight = torch.autograd.grad(logps.sum(), (hidden, weight))

        # Assert: the math and the log-probs are fp32; the hidden gradient
        # rounds once to bf16 (relative 2**-8).
        assert logps.dtype == torch.float32
        assert torch.allclose(logps, expected, atol=1e-5)
        assert torch.allclose(
            grad_hidden.float(), expected_grads[0], rtol=2**-8, atol=1e-6
        )
        # The bf16 weight-gradient buffer rounds once per chunk (5 chunks).
        assert torch.allclose(
            grad_weight.float(),
            expected_grads[1],
            rtol=0.0,
            atol=5 * 2**-8 * float(expected_grads[1].abs().max()),
        )


@pytest.mark.gpu
@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="torch.mm(out_dtype=) is CUDA-only."
)
class TestHeadLogitsFunction:
    def test_bf16_operands_give_fp32_logits_and_close_gradients(self) -> None:
        # Arrange
        torch.manual_seed(7)
        hidden, weight = _leaves(
            torch.randn(64, 256, device="cuda", dtype=torch.bfloat16),
            torch.randn(1024, 256, device="cuda", dtype=torch.bfloat16) * 0.05,
        )
        grad = torch.randn(64, 1024, device="cuda")
        reference = hidden.detach().float() @ weight.detach().float().t()

        # Act
        logits = HeadLogitsFunction.apply(hidden, weight)
        grad_hidden, grad_weight = torch.autograd.grad(logits, (hidden, weight), grad)

        # Assert: the same bf16 inputs with fp32 accumulation on both sides.
        assert logits.dtype == torch.float32
        assert torch.allclose(logits, reference, rtol=1e-4, atol=1e-4)
        # The backward rounds the gradient to bf16 and its outputs are bf16:
        # two roundings of relative 2**-8 each.
        for got, want in (
            (grad_hidden, grad @ weight.detach().float()),
            (grad_weight, grad.t() @ hidden.detach().float()),
        ):
            assert got.dtype == torch.bfloat16
            assert (got.float() - want).norm() / want.norm() < 2**-7

    def test_chunked_bf16_logprobs_skip_the_fp32_head_copy(self) -> None:
        # Arrange
        torch.manual_seed(8)
        vocab, width = 32768, 1024
        hidden = torch.randn(4, 64, width, device="cuda", dtype=torch.bfloat16)
        weight = torch.randn(vocab, width, device="cuda", dtype=torch.bfloat16) * 0.02
        targets = torch.randint(0, vocab, (4, 64), device="cuda")
        expected = _plain_logprobs(hidden.float(), weight.float(), None, targets)
        # The first call compiles and autotunes the chunk kernels; autotuning
        # allocates an L2-cache-sized flush buffer outside the chunk's tiles.
        with torch.no_grad():
            fused_linear_logprobs_chunked(hidden, weight, None, targets, chunk_rows=64)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        baseline = torch.cuda.memory_allocated()

        # Act
        with torch.no_grad():
            logps = fused_linear_logprobs_chunked(
                hidden, weight, None, targets, chunk_rows=64
            )
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated() - baseline

        # Assert: fp32 log-probs match the fp32 reference up to accumulation
        # order; the peak holds the (64, V) fp32 tiles, far below one fp32
        # (V, H) head copy.
        assert logps.dtype == torch.float32
        assert torch.allclose(logps, expected, atol=1e-3)
        assert peak < vocab * width * 4 // 4

    def test_eager_bf16_chunk_gives_fp32_logprobs_of_the_bf16_operands(
        self,
    ) -> None:
        # Arrange
        torch.manual_seed(9)
        hidden = torch.randn(32, 128, device="cuda", dtype=torch.bfloat16)
        weight = torch.randn(512, 128, device="cuda", dtype=torch.bfloat16) * 0.05
        targets = torch.randint(0, 512, (32,), device="cuda")
        expected = (
            (hidden.float() @ weight.float().t())
            .log_softmax(dim=-1)
            .gather(dim=-1, index=targets.unsqueeze(-1))
            .squeeze(-1)
        )

        # Act
        logps = _fused_logprob_chunk(hidden, weight, None, targets, 1.0, None)

        # Assert: fp32 accumulation of the same bf16 inputs on both sides.
        assert logps.dtype == torch.float32
        assert torch.allclose(logps, expected, rtol=1e-4, atol=1e-4)


class TestHeadLogitsFunctionBackward:
    def test_frozen_weight_gets_no_gradient(self) -> None:
        # Arrange
        torch.manual_seed(10)
        hidden = torch.randn(6, 8, dtype=torch.bfloat16)
        weight = torch.randn(24, 8, dtype=torch.bfloat16)
        grad_logits = torch.randn(6, 24)
        ctx = SimpleNamespace(
            saved_tensors=(hidden, weight), needs_input_grad=(True, False)
        )

        # Act
        grad_hidden, grad_weight = HeadLogitsFunction.backward(ctx, grad_logits)

        # Assert
        assert grad_weight is None
        assert torch.equal(grad_hidden, grad_logits.to(torch.bfloat16) @ weight)

    def test_trainable_weight_gets_its_gradient(self) -> None:
        # Arrange
        torch.manual_seed(11)
        hidden = torch.randn(6, 8, dtype=torch.bfloat16)
        weight = torch.randn(24, 8, dtype=torch.bfloat16)
        grad_logits = torch.randn(6, 24)
        ctx = SimpleNamespace(
            saved_tensors=(hidden, weight), needs_input_grad=(True, True)
        )

        # Act
        _, grad_weight = HeadLogitsFunction.backward(ctx, grad_logits)

        # Assert
        assert torch.equal(grad_weight, grad_logits.to(torch.bfloat16).t() @ hidden)


class TestFusedLinearLogProbsFunctionFrozenHead:
    @staticmethod
    def _backward(
        weight_requires_grad: bool,
    ) -> tuple[int, torch.Tensor, torch.Tensor | None]:
        """Backward FLOPs and the hidden and weight gradients of one fused call."""
        torch.manual_seed(12)
        hidden = torch.randn(2, 9, 16, requires_grad=True)
        weight = torch.randn(40, 16, requires_grad=weight_requires_grad)
        targets = torch.randint(0, 40, (2, 9))
        logps = FusedLinearLogProbsFunction.apply(
            hidden, weight, None, targets, 1.0, 4, None
        )
        with FlopCounterMode(display=False) as counter:
            logps.sum().backward()
        return counter.get_total_flops(), hidden.grad, weight.grad

    def test_frozen_head_skips_the_weight_gradient_gemm(self) -> None:
        # Arrange
        gemm = 2 * (2 * 9) * 40 * 16

        # Act
        frozen_flops, frozen_hidden_grad, frozen_weight_grad = self._backward(False)
        trained_flops, trained_hidden_grad, trained_weight_grad = self._backward(True)

        # Assert: the recompute and the hidden gradient are one GEMM each; a
        # trainable head adds the (V, H) weight-gradient GEMM.
        assert frozen_flops == 2 * gemm
        assert trained_flops == 3 * gemm
        assert frozen_weight_grad is None
        assert trained_weight_grad is not None
        assert torch.equal(frozen_hidden_grad, trained_hidden_grad)


class TestLLMAlgorithmScoreHidden:
    @pytest.mark.parametrize(
        "score_mask",
        [
            pytest.param(
                torch.tensor(
                    [[1, 1, 1, 0, 0], [0, 1, 0, 1, 0], [0, 0, 0, 0, 0]],
                    dtype=torch.bool,
                ),
                id="padded-with-an-empty-row",
            ),
            pytest.param(
                torch.tensor([[0, 1, 1, 0, 0, 0, 1, 1, 1, 0]], dtype=torch.bool),
                id="packed",
            ),
        ],
    )
    def test_runs_the_head_on_the_scored_tokens_only(
        self, score_mask: torch.Tensor
    ) -> None:
        # Arrange
        torch.manual_seed(9)
        rows, width = score_mask.shape
        hidden, weight = _leaves(
            torch.randn(rows, width, 8),
            torch.randn(16, 8),
        )
        targets = torch.randint(0, 16, (rows, width))
        ref_hidden, ref_weight = _leaves(hidden, weight)
        expected = _plain_logprobs(ref_hidden, ref_weight, None, targets) * score_mask
        expected_grads = torch.autograd.grad(expected.sum(), (ref_hidden, ref_weight))
        algo = SimpleNamespace(temperature=1.0)
        head_rows = []

        def fused(head_hidden: torch.Tensor, *args, **kwargs) -> torch.Tensor:
            head_rows.append(tuple(head_hidden.shape))
            return LLMAlgorithm._logprobs_from_hidden_fused_grad(
                head_hidden, *args, **kwargs
            )

        # Act
        logps = LLMAlgorithm._score_hidden(
            algo, fused, hidden, weight, None, targets, score_mask
        )
        grads = torch.autograd.grad(logps.sum(), (hidden, weight))

        # Assert: both sides run fp32; the head reorders the logit sums.
        assert head_rows == [(1, int(score_mask.sum()), 8)]
        assert torch.allclose(logps, expected, atol=1e-5)
        assert torch.allclose(grads[0], expected_grads[0], atol=1e-5)
        assert torch.allclose(grads[1], expected_grads[1], atol=1e-5)
        assert torch.count_nonzero(grads[0][~score_mask]) == 0

    def test_a_mask_without_scored_tokens_scores_nothing(self) -> None:
        # Arrange
        hidden, weight = _leaves(torch.randn(2, 3, 8), torch.randn(16, 8))
        targets = torch.randint(0, 16, (2, 3))
        algo = SimpleNamespace(temperature=1.0)

        # Act
        logps = LLMAlgorithm._score_hidden(
            algo,
            LLMAlgorithm._logprobs_from_hidden_fused_grad,
            hidden,
            weight,
            None,
            targets,
            torch.zeros(2, 3, dtype=torch.bool),
        )
        grads = torch.autograd.grad(logps.sum(), (hidden, weight))

        # Assert
        assert torch.equal(logps, torch.zeros(2, 3))
        assert torch.count_nonzero(grads[0]) == 0
        assert torch.count_nonzero(grads[1]) == 0

    def test_padding_of_a_supports_batch_stays_off_the_vocabulary_matmul(
        self,
    ) -> None:
        # Arrange
        torch.manual_seed(4)
        hidden = torch.randn(2, 3, 8, dtype=torch.float32)
        weight = torch.randn(16, 8, dtype=torch.float32)
        targets = torch.tensor([[5, 2, 9], [1, 3, 4]])
        supports = torch.tensor(
            [[[5, 7], [2, 6], [-1, -1]], [[-1, -1], [3, 8], [-1, -1]]],
            dtype=torch.int32,
        )
        score_mask = supports[..., 0] >= 0
        algo = SimpleNamespace(temperature=0.7)
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
                LLMAlgorithm._logprobs_from_hidden_fused,
                hidden,
                weight,
                None,
                targets,
                score_mask,
                supports,
            )

        # Assert
        assert torch.allclose(logps[0, :2], expected[0], atol=1e-6)
        assert logps[0, 2].item() == 0.0
        assert logps[1, 0].item() == 0.0
        assert logps[1, 2].item() == 0.0
        assert logps[1, 1].item() < 0.0


class TestSupportLogprobChunk:
    def test_matches_the_masked_log_softmax_with_bias_and_softcap(self) -> None:
        # Arrange
        torch.manual_seed(2)
        hidden = torch.randn(3, 8, dtype=torch.float32)
        weight = torch.randn(16, 8, dtype=torch.float32) * 4.0
        bias = torch.randn(16, dtype=torch.float32)
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
        logps = support_logprob_chunk(hidden, weight, bias, targets, 0.7, 3.0, supports)

        # Assert
        assert torch.allclose(logps, expected, atol=1e-6)

    def test_upcasts_a_bf16_head_to_the_fp32_result(self) -> None:
        torch.manual_seed(3)
        hidden = torch.randn(2, 8, dtype=torch.bfloat16)
        weight = torch.randn(16, 8, dtype=torch.bfloat16)
        targets = torch.tensor([4, 9])
        supports = torch.tensor([[4, 9], [9, 4]])
        logits = hidden.float() @ weight.float().t()
        expected = logits[:, [4, 9]].log_softmax(-1)[[0, 1], [0, 1]]

        logps = support_logprob_chunk(
            hidden, weight, None, targets, 1.0, None, supports
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
            None,
            torch.tensor([[1, 2]]),
        )

        assert logps.item() == float("-inf")


def _autograd_support_hidden_grad(
    hidden: torch.Tensor, upstream: torch.Tensor, *args: torch.Tensor | float | None
) -> torch.Tensor:
    """Hidden gradient of :func:`support_logprob_chunk` by autograd."""
    h = hidden.detach().clone().requires_grad_(True)
    logps = support_logprob_chunk(h, *args)
    (grad,) = torch.autograd.grad(logps, h, grad_outputs=upstream)
    return grad


class TestSupportLogprobHiddenGrad:
    def test_matches_autograd_with_bias_softcap_and_logit_bias(self) -> None:
        # Arrange
        torch.manual_seed(4)
        hidden = torch.randn(4, 8, dtype=torch.float32)
        weight = torch.randn(16, 8, dtype=torch.float32) * 4.0
        bias = torch.randn(16, dtype=torch.float32)
        logit_bias = torch.randn(16, dtype=torch.float32)
        targets = torch.tensor([5, 2, 11, 9])
        supports = torch.tensor([[7, 5, 1], [2, -1, -1], [11, 3, -1], [4, 9, 0]])
        upstream = torch.tensor([0.5, -1.0, 2.0, 1.5])
        args = (weight, bias, targets, 0.7, 3.0, supports)

        # Act
        grad = support_logprob_hidden_grad(
            hidden, *args, upstream, logit_bias=logit_bias
        )

        # Assert
        expected = _autograd_support_hidden_grad(hidden, upstream, *args, logit_bias)
        assert grad.dtype == torch.float32
        assert torch.allclose(grad, expected, atol=1e-6)

    def test_matches_autograd_without_bias_softcap_or_temperature(self) -> None:
        torch.manual_seed(5)
        hidden = torch.randn(3, 8, dtype=torch.float32)
        weight = torch.randn(16, 8, dtype=torch.float32)
        targets = torch.tensor([1, 6, 6])
        supports = torch.tensor([[1, 6, 3], [6, 1, -1], [6, -1, -1]])
        upstream = torch.ones(3)
        args = (weight, None, targets, 1.0, None, supports)

        grad = support_logprob_hidden_grad(hidden, *args, upstream)

        expected = _autograd_support_hidden_grad(hidden, upstream, *args)
        assert torch.allclose(grad, expected, atol=1e-6)
        # A one-id support is a forced token and carries no gradient.
        assert torch.count_nonzero(grad[2]) == 0

    def test_a_target_outside_its_support_gets_zero_gradient(self) -> None:
        torch.manual_seed(6)
        hidden = torch.randn(2, 8)
        weight = torch.randn(16, 8)
        targets = torch.tensor([3, 2])
        supports = torch.tensor([[1, 2], [1, 2]])

        grad = support_logprob_hidden_grad(
            hidden, weight, None, targets, 1.0, None, supports, torch.ones(2)
        )

        assert torch.count_nonzero(grad[0]) == 0
        assert torch.count_nonzero(grad[1]) > 0


class TestSupportHiddenGradDispatch:
    def test_cuda_device_compiles_once(self) -> None:
        # torch.compile is stubbed so the dispatch logic runs without a GPU.
        hidden, weight, bias, targets, temperature, softcap, supports = _support_args()
        upstream = torch.ones(hidden.shape[0])
        args = (hidden, weight, bias, targets, temperature, softcap, supports, upstream)
        with patch("torch.compile", side_effect=lambda fn, **kw: fn) as mock_compile:
            first = _support_hidden_grad_dispatch(torch.device("cuda"), *args)
            second = _support_hidden_grad_dispatch(torch.device("cuda"), *args)

        mock_compile.assert_called_once_with(support_logprob_hidden_grad, dynamic=True)
        assert torch.equal(first, second)
        assert torch.equal(first, support_logprob_hidden_grad(*args))


class TestSupportChunkRows:
    # Nemotron 3.5 Lightning lm_head: vocab 131072, hidden 2688.
    head_w = torch.empty(131072, 2688, dtype=torch.bfloat16, device="meta")

    def test_a_top_k_support_fills_one_vocab_tile(self) -> None:
        assert support_chunk_rows(512, self.head_w, 50) == 499

    def test_a_narrow_support_takes_more_rows_than_the_tile(self) -> None:
        assert support_chunk_rows(512, self.head_w, 20) == 1248

    def test_a_vocab_wide_support_takes_one_row(self) -> None:
        assert support_chunk_rows(512, self.head_w, 131072) == 1


class TestSplitSupportRows:
    def test_mixed_rows_split_in_row_order(self) -> None:
        flat_support = torch.tensor([[-1, -1], [3, -1], [-1, -1], [0, 2], [5, 1]])

        support_rows, vocab_rows = split_support_rows(flat_support)

        assert support_rows.tolist() == [1, 3, 4]
        assert vocab_rows.tolist() == [0, 2]

    def test_every_row_with_a_support_leaves_no_vocab_rows(self) -> None:
        support_rows, vocab_rows = split_support_rows(torch.tensor([[2], [7], [1]]))

        assert support_rows.tolist() == [0, 1, 2]
        assert vocab_rows.numel() == 0

    def test_no_support_frame_scores_every_row_over_the_vocab(self) -> None:
        assert split_support_rows(None) == (None, None)


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
            hidden, weight, None, targets, 0.7, 2, None, supports
        ).sum().backward()

        # Assert
        assert torch.allclose(hidden.grad, ref_hidden.grad, atol=1e-6)
        assert torch.allclose(weight.grad, ref_weight.grad, atol=1e-6)
        # A forced token carries no gradient into its hidden state.
        assert torch.count_nonzero(hidden.grad[0, 1]) == 0

    def test_frozen_head_gradient_matches_the_masked_log_softmax(self) -> None:
        # Arrange
        hidden, weight, targets, supports = _support_case()
        hidden.requires_grad_(True)
        ref_hidden = hidden.detach().clone().requires_grad_(True)
        upstream = torch.tensor([[0.5, -2.0, 1.5]])
        (
            _reference_logprobs(ref_hidden, weight, targets, supports, 0.7) * upstream
        ).sum().backward()

        # Act
        (
            FusedLinearLogProbsFunction.apply(
                hidden, weight, None, targets, 0.7, 2, None, supports
            )
            * upstream
        ).sum().backward()

        # Assert
        assert weight.grad is None
        assert torch.allclose(hidden.grad, ref_hidden.grad, atol=1e-6)
        assert torch.count_nonzero(hidden.grad[0, 1]) == 0
        assert torch.count_nonzero(hidden.grad[0, 2]) > 0
