# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Tests for ReplayBufferSpec, NStepBufferArgs, and PerBufferArgs.

Covers Pydantic validation, field defaults, aliases, and ``init_buffer``
dispatch to the correct buffer class for single-agent, multi-agent, and
specialised (n-step, PER) scenarios.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from agilerl.arena.models.algorithms import (
    DDPGSpec,
    DQNSpec,
    MADDPGSpec,
    RainbowDQNSpec,
)
from agilerl.components.replay_buffer import (
    MultiStepReplayBuffer,
    PrioritizedReplayBuffer,
    ReplayBuffer,
)
from agilerl.models.training import (
    NStepBufferArgs,
    PerBufferArgs,
    ReplayBufferSpec,
    init_buffer,
)

# ============================================================================
# NStepBufferArgs
# ============================================================================


class TestNStepBufferArgs:
    """Validation of the n-step sub-model."""

    def test_defaults(self):
        args = NStepBufferArgs()
        assert args.n_step == 3

    def test_custom_n_step(self):
        args = NStepBufferArgs(n_step=5)
        assert args.n_step == 5

    def test_n_step_ge_1(self):
        with pytest.raises(ValidationError):
            NStepBufferArgs(n_step=0)

    def test_negative_n_step_rejected(self):
        with pytest.raises(ValidationError):
            NStepBufferArgs(n_step=-1)


# ============================================================================
# PerBufferArgs
# ============================================================================


class TestPerBufferArgs:
    """Validation of the PER sub-model."""

    def test_defaults(self):
        args = PerBufferArgs()
        assert args.alpha == 0.5

    def test_custom_alpha(self):
        args = PerBufferArgs(alpha=0.8)
        assert args.alpha == 0.8

    def test_alpha_lower_bound(self):
        args = PerBufferArgs(alpha=0.0)
        assert args.alpha == 0.0

    def test_alpha_upper_bound(self):
        args = PerBufferArgs(alpha=1.0)
        assert args.alpha == 1.0

    def test_alpha_below_lower_bound_rejected(self):
        with pytest.raises(ValidationError):
            PerBufferArgs(alpha=-0.1)

    def test_alpha_above_upper_bound_rejected(self):
        with pytest.raises(ValidationError):
            PerBufferArgs(alpha=1.1)


# ============================================================================
# ReplayBufferSpec - field defaults and Pydantic validation
# ============================================================================


class TestReplayBufferSpecValidation:
    """Pydantic field defaults, aliases, and constraint checks."""

    def test_defaults(self):
        spec = ReplayBufferSpec()
        assert spec.max_size == 100_000
        assert spec.standard_buffer is True
        assert spec.n_step_buffer is False
        assert isinstance(spec.n_step_buffer_args, NStepBufferArgs)
        assert spec.per_buffer is False
        assert isinstance(spec.per_buffer_args, PerBufferArgs)

    def test_custom_max_size(self):
        spec = ReplayBufferSpec(max_size=50_000)
        assert spec.max_size == 50_000

    def test_memory_size_alias(self):
        spec = ReplayBufferSpec(memory_size=25_000)
        assert spec.max_size == 25_000

    def test_max_size_alias_takes_precedence(self):
        spec = ReplayBufferSpec(max_size=10_000)
        assert spec.max_size == 10_000

    def test_max_size_ge_1(self):
        spec = ReplayBufferSpec(max_size=1)
        assert spec.max_size == 1

    def test_max_size_zero_rejected(self):
        with pytest.raises(ValidationError):
            ReplayBufferSpec(max_size=0)

    def test_max_size_negative_rejected(self):
        with pytest.raises(ValidationError):
            ReplayBufferSpec(max_size=-5)

    def test_enable_n_step_buffer(self):
        spec = ReplayBufferSpec(n_step_buffer=True)
        assert spec.n_step_buffer is True

    def test_custom_n_step_buffer_args(self):
        spec = ReplayBufferSpec(
            n_step_buffer=True,
            n_step_buffer_args=NStepBufferArgs(n_step=7),
        )
        assert spec.n_step_buffer_args.n_step == 7

    def test_enable_per_buffer(self):
        spec = ReplayBufferSpec(per_buffer=True)
        assert spec.per_buffer is True

    def test_custom_per_buffer_args(self):
        spec = ReplayBufferSpec(
            per_buffer=True,
            per_buffer_args=PerBufferArgs(alpha=0.9),
        )
        assert spec.per_buffer_args.alpha == 0.9

    def test_from_dict(self):
        spec = ReplayBufferSpec.model_validate(
            {"memory_size": 75_000, "per_buffer": True}
        )
        assert spec.max_size == 75_000
        assert spec.per_buffer is True


# ============================================================================
# init_buffer - standard (single-agent) path
# ============================================================================


class TestInitBufferStandard:
    """Standard single-agent replay buffer initialization."""

    def test_standard_buffer_default(self):
        spec = ReplayBufferSpec()
        buf = init_buffer(spec, DQNSpec())
        assert isinstance(buf, ReplayBuffer)
        assert buf.max_size == 100_000

    def test_standard_buffer_custom_size(self):
        spec = ReplayBufferSpec(max_size=2048)
        buf = init_buffer(spec, DQNSpec())
        assert isinstance(buf, ReplayBuffer)
        assert buf.max_size == 2048

    def test_standard_buffer_device_forwarded(self):
        spec = ReplayBufferSpec()
        buf = init_buffer(spec, DQNSpec(), device="cpu")
        assert isinstance(buf, ReplayBuffer)

    @pytest.mark.parametrize("algo_cls", [DQNSpec, DDPGSpec])
    def test_standard_buffer_for_off_policy_algos(self, algo_cls):
        spec = ReplayBufferSpec()
        buf = init_buffer(spec, algo_cls())
        assert isinstance(buf, ReplayBuffer)
        assert not isinstance(buf, (MultiStepReplayBuffer, PrioritizedReplayBuffer))


# ============================================================================
# init_buffer - n-step path
# ============================================================================


class TestInitBufferNStep:
    """N-step replay buffer initialization."""

    def test_n_step_buffer_created(self):
        spec = ReplayBufferSpec(n_step_buffer=True)
        buf = init_buffer(spec, DQNSpec())
        assert isinstance(buf, MultiStepReplayBuffer)

    def test_n_step_uses_algo_gamma(self):
        algo = DQNSpec(gamma=0.95)
        spec = ReplayBufferSpec(n_step_buffer=True)
        buf = init_buffer(spec, algo)
        assert isinstance(buf, MultiStepReplayBuffer)
        assert buf.gamma == 0.95

    def test_n_step_uses_custom_n(self):
        spec = ReplayBufferSpec(
            n_step_buffer=True,
            n_step_buffer_args=NStepBufferArgs(n_step=5),
        )
        buf = init_buffer(spec, DQNSpec())
        assert isinstance(buf, MultiStepReplayBuffer)
        assert buf.n_step == 5

    def test_n_step_default_n(self):
        spec = ReplayBufferSpec(n_step_buffer=True)
        buf = init_buffer(spec, DQNSpec())
        assert buf.n_step == 3

    def test_n_step_respects_max_size(self):
        spec = ReplayBufferSpec(n_step_buffer=True, max_size=512)
        buf = init_buffer(spec, DQNSpec())
        assert buf.max_size == 512

    def test_n_step_missing_gamma_raises(self):
        """An algo spec without ``gamma`` should raise when n-step is requested."""
        from unittest.mock import MagicMock

        from agilerl import AgentType

        algo = MagicMock(spec=["agent_type"])
        algo.agent_type = AgentType.SingleAgent
        spec = ReplayBufferSpec(n_step_buffer=True)
        with pytest.raises(ValueError, match=r"[Gg]amma"):
            init_buffer(spec, algo)


# ============================================================================
# init_buffer - PER path
# ============================================================================


class TestInitBufferPER:
    """Prioritized experience replay buffer initialization."""

    def test_per_buffer_with_rainbow_dqn(self):
        spec = ReplayBufferSpec(per_buffer=True)
        buf = init_buffer(spec, RainbowDQNSpec())
        assert isinstance(buf, PrioritizedReplayBuffer)

    def test_per_buffer_uses_custom_alpha(self):
        spec = ReplayBufferSpec(
            per_buffer=True,
            per_buffer_args=PerBufferArgs(alpha=0.7),
        )
        buf = init_buffer(spec, RainbowDQNSpec())
        assert isinstance(buf, PrioritizedReplayBuffer)
        assert buf.alpha == 0.7

    def test_per_buffer_default_alpha(self):
        spec = ReplayBufferSpec(per_buffer=True)
        buf = init_buffer(spec, RainbowDQNSpec())
        assert buf.alpha == 0.5

    def test_per_buffer_respects_max_size(self):
        spec = ReplayBufferSpec(per_buffer=True, max_size=4096)
        buf = init_buffer(spec, RainbowDQNSpec())
        assert buf.max_size == 4096

    def test_per_buffer_non_rainbow_raises(self):
        spec = ReplayBufferSpec(per_buffer=True)
        with pytest.raises(ValueError, match="Rainbow DQN"):
            init_buffer(spec, DQNSpec())

    @pytest.mark.parametrize("algo_cls", [DDPGSpec])
    def test_per_buffer_rejects_non_rainbow_algos(self, algo_cls):
        spec = ReplayBufferSpec(per_buffer=True)
        with pytest.raises(ValueError, match="Rainbow DQN"):
            init_buffer(spec, algo_cls())


# ============================================================================
# init_buffer - multi-agent path
# ============================================================================


class TestInitBufferMultiAgent:
    """Multi-agent replay buffer initialization."""

    def test_multi_agent_buffer_created(self):
        spec = ReplayBufferSpec()
        buf = init_buffer(spec, MADDPGSpec())
        assert isinstance(buf, ReplayBuffer)

    def test_multi_agent_ignores_n_step_flag(self):
        """Even with ``n_step_buffer=True``, multi-agent gets a plain
        ``ReplayBuffer``.
        """
        spec = ReplayBufferSpec(n_step_buffer=True)
        buf = init_buffer(spec, MADDPGSpec())
        assert isinstance(buf, ReplayBuffer)
        assert not isinstance(buf, MultiStepReplayBuffer)

    def test_multi_agent_ignores_per_flag(self):
        """Even with ``per_buffer=True``, multi-agent gets a plain
        ``ReplayBuffer``.
        """
        spec = ReplayBufferSpec(per_buffer=True)
        buf = init_buffer(spec, MADDPGSpec())
        assert isinstance(buf, ReplayBuffer)
        assert not isinstance(buf, PrioritizedReplayBuffer)

    def test_multi_agent_respects_max_size(self):
        spec = ReplayBufferSpec(max_size=10_000)
        buf = init_buffer(spec, MADDPGSpec())
        assert isinstance(buf, ReplayBuffer)
        assert buf.max_size == 10_000

    def test_multi_agent_device_forwarded(self):
        spec = ReplayBufferSpec()
        buf = init_buffer(spec, MADDPGSpec(), device="cpu")
        assert isinstance(buf, ReplayBuffer)


# ============================================================================
# init_buffer - priority between flags
# ============================================================================


class TestInitBufferFlagPriority:
    """When multiple flags are set, per takes precedence for the main memory
    (the n-step buffer is built separately by ``init_n_step_buffer``).
    Standard is the fallback when neither is set.
    """

    def test_per_takes_precedence_over_n_step(self):
        spec = ReplayBufferSpec(n_step_buffer=True, per_buffer=True)
        buf = init_buffer(spec, RainbowDQNSpec(net_config=None))
        assert isinstance(buf, PrioritizedReplayBuffer)

    def test_per_with_unsupported_algorithm_raises(self):
        spec = ReplayBufferSpec(n_step_buffer=True, per_buffer=True)
        with pytest.raises(ValueError, match="only supported for Rainbow"):
            init_buffer(spec, DQNSpec())

    def test_standard_when_both_flags_false(self):
        spec = ReplayBufferSpec(n_step_buffer=False, per_buffer=False)
        buf = init_buffer(spec, DQNSpec())
        assert isinstance(buf, ReplayBuffer)
        assert not isinstance(buf, (MultiStepReplayBuffer, PrioritizedReplayBuffer))

    def test_multi_agent_overrides_all_flags(self):
        spec = ReplayBufferSpec(n_step_buffer=True, per_buffer=True)
        buf = init_buffer(spec, MADDPGSpec())
        assert isinstance(buf, ReplayBuffer)
        assert not isinstance(buf, (MultiStepReplayBuffer, PrioritizedReplayBuffer))
