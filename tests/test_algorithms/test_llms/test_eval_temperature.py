"""Signature-level checks for the ``eval_temperature`` knob.

These assert the constructor plumbing without building a model, so they run
without vLLM/DeepSpeed (the heavier LLM suites ``importorskip`` those).
"""

from __future__ import annotations

import inspect

import pytest

from agilerl.algorithms.cispo import CISPO
from agilerl.algorithms.grpo import GRPO
from agilerl.algorithms.gspo import GSPO
from agilerl.algorithms.ppo_llm import PPO
from agilerl.algorithms.reinforce_llm import REINFORCE

LLM_ALGOS = [GRPO, GSPO, CISPO, PPO, REINFORCE]


@pytest.mark.parametrize("algo", LLM_ALGOS)
def test_eval_temperature_param_defaults_to_eval_value(algo):
    """Every LLM RL algorithm exposes ``eval_temperature`` defaulting to 0.01."""
    params = inspect.signature(algo.__init__).parameters
    assert "eval_temperature" in params
    assert params["eval_temperature"].default == 0.01


@pytest.mark.parametrize("algo", [GSPO, CISPO])
def test_eval_temperature_survives_signature_rebuild(algo):
    """GSPO/CISPO rebuild their signature to hide ``loss_type``; the new knob
    must come through that rebuild and ``loss_type`` must stay hidden."""
    params = inspect.signature(algo.__init__).parameters
    assert "eval_temperature" in params
    assert "loss_type" not in params
