"""Unit tests for the chunked fused-logprob compile dispatch."""

from unittest.mock import patch

import pytest
import torch

from agilerl.algorithms.core.llm_ops.fused_logprobs import (
    _FUSED_LOGPROB_COMPILE_STATE,
    FusedLinearLogProbsFunction,
    _fused_logprob_chunk,
    _fused_logprob_chunk_dispatch,
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
        saved_tensors = (hidden, weight, None, targets)

    grad_output = torch.ones(1, 4)
    grads = FusedLinearLogProbsFunction.backward(Ctx(), grad_output)
    assert grads[0] is None
    assert grads[1] is None
