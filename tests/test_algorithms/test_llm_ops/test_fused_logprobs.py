"""Unit tests for the chunked fused-logprob compile dispatch."""

from unittest.mock import patch

import pytest
import torch

from agilerl.algorithms.core.llm_ops.fused_logprobs import (
    _FUSED_LOGPROB_COMPILE_STATE,
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
            raise RuntimeError("triton backend exploded")

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
