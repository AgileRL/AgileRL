"""Signature-level checks for the GRPO-family ``liger_chunk_size`` knob.

Asserts the constructor plumbing without building a model, so it runs without
vLLM/DeepSpeed (the heavier LLM suites ``importorskip`` those).
"""

from __future__ import annotations

import inspect

import pytest

from agilerl.algorithms.cispo import CISPO
from agilerl.algorithms.grpo import GRPO
from agilerl.algorithms.gspo import GSPO


@pytest.mark.parametrize("algo", [GRPO, GSPO, CISPO])
def test_liger_chunk_size_defaults_to_one(algo):
    """GRPO and its variants expose ``liger_chunk_size`` defaulting to 1."""
    params = inspect.signature(algo.__init__).parameters
    assert "liger_chunk_size" in params
    assert params["liger_chunk_size"].default == 1
