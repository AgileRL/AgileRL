# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Fused forward passes vision tensors when the batch carries them."""

from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from agilerl.algorithms import GRPO
from agilerl.algorithms.core.base import LLMAlgorithm
from tests.test_algorithms.test_core_base import _LLM_DEPS_SKIP, _make_llm_agent
from tests.test_algorithms.test_llms.llm_helpers import create_module


def _make_cpu_grpo_for_kernel_tests(**kwargs: object) -> GRPO:
    defaults: dict[str, object] = {
        "actor_network": create_module(
            input_size=6,
            max_tokens=4,
            vocab_size=64,
            device="cpu",
        ),
        "pad_token_id": 63,
        "pad_token": "<pad>",
        "batch_size": 4,
        "group_size": 2,
        "max_output_tokens": 4,
        "max_model_len": 12,
        "wrap": False,
        "gradient_checkpointing": False,
        "device": "cpu",
        "use_liger_loss": False,
        "advantage_granularity": "trajectory",
    }
    defaults.update(kwargs)
    return GRPO(**defaults)


@_LLM_DEPS_SKIP
class TestFusedForwardPixelValues:
    def test_fused_forward_passes_pixel_values_to_model_pass(self) -> None:
        # Arrange
        agent = _make_llm_agent()
        agent.use_value_head = False
        batch_size = 2
        seq_len = 5
        ids = torch.randint(1, 32, (batch_size, seq_len))
        pixel_values = torch.randn(batch_size, 3, 4, 4)
        agent._packing_mode = MagicMock(return_value=None)
        agent._fused_model_pass = MagicMock(
            return_value=(torch.zeros(batch_size, seq_len - 1), None),
        )

        # Act
        agent._fused_forward(ids, batch_size, pixel_values=pixel_values)

        # Assert
        kwargs = agent._fused_model_pass.call_args.kwargs
        assert torch.equal(kwargs["pixel_values"], pixel_values)

    def test_fused_forward_omits_pixel_values_when_absent(self) -> None:
        # Arrange
        agent = _make_llm_agent()
        agent.use_value_head = False
        batch_size = 2
        seq_len = 5
        ids = torch.randint(1, 32, (batch_size, seq_len))
        agent._packing_mode = MagicMock(return_value=None)
        agent._fused_model_pass = MagicMock(
            return_value=(torch.zeros(batch_size, seq_len - 1), None),
        )

        # Act
        agent._fused_forward(ids, batch_size)

        # Assert
        assert agent._fused_model_pass.call_args.kwargs.get("pixel_values") is None

    def test_fused_packed_forward_keeps_pixel_batch_size(self) -> None:
        # Arrange
        agent = _make_llm_agent()
        agent.use_value_head = False
        batch_size = 2
        seq_len = 6
        ids = torch.randint(1, 32, (batch_size, seq_len))
        mask = torch.ones_like(ids)
        pixel_values = torch.randn(batch_size, 3, 4, 4)
        hidden = torch.randn(1, seq_len, 8)
        actor = MagicMock()
        actor.forward = MagicMock(return_value=SimpleNamespace(logits=hidden))
        agent.actor = actor
        agent._get_unwrapped_actor = MagicMock(return_value=actor)
        agent._fused_logprob_fn_and_head = MagicMock(
            return_value=(
                MagicMock(return_value=torch.zeros(1, seq_len - 1)),
                torch.randn(32, 8),
                None,
            )
        )
        agent._patch_lm_head_to_identity = MagicMock(return_value=nullcontext())
        agent._amp_ctx = MagicMock(return_value=nullcontext())
        agent._activation_offload_ctx = MagicMock(return_value=nullcontext())

        # Act
        with patch(
            "agilerl.algorithms.core.base.unpack_logprobs",
            return_value=torch.zeros(batch_size, seq_len - 1),
        ):
            agent._fused_packed_forward(ids, mask, pixel_values=pixel_values)

        # Assert
        forwarded = actor.call_args.kwargs["pixel_values"]
        assert forwarded.shape[0] == batch_size
        assert torch.equal(forwarded, pixel_values)

    def test_fused_packed_forward_value_head_repeats_pixel_values(self) -> None:
        # Arrange
        agent = _make_llm_agent()
        agent.use_value_head = True
        batch_size = 3
        seq_len = 6
        ids = torch.randint(1, 32, (batch_size, seq_len))
        mask = torch.ones_like(ids)
        pixel_values = torch.randn(batch_size, 3, 4, 4)
        hidden = torch.randn(2, seq_len, 8)
        actor = MagicMock()
        actor.forward = MagicMock(return_value=SimpleNamespace(logits=hidden))
        agent.actor = actor
        agent._get_unwrapped_actor = MagicMock(return_value=actor)
        agent._fused_logprob_fn_and_head = MagicMock(
            return_value=(
                MagicMock(return_value=torch.zeros(1, seq_len - 1)),
                torch.randn(32, 8),
                None,
            )
        )
        agent._patch_lm_head_to_identity = MagicMock(return_value=nullcontext())
        agent._amp_ctx = MagicMock(return_value=nullcontext())
        agent._activation_offload_ctx = MagicMock(return_value=nullcontext())

        # Act
        with patch(
            "agilerl.algorithms.core.base.unpack_logprobs",
            return_value=torch.zeros(batch_size, seq_len - 1),
        ):
            agent._fused_packed_forward(ids, mask, pixel_values=pixel_values)

        # Assert
        forwarded = actor.call_args.kwargs["pixel_values"]
        expected = pixel_values.repeat(2, 1, 1, 1)
        assert forwarded.shape[0] == 6
        assert torch.equal(forwarded, expected)

    def test_fused_forward_value_head_repeats_pixel_values(self) -> None:
        # Arrange
        agent = _make_llm_agent()
        agent.use_value_head = True
        batch_size = 2
        seq_len = 5
        ids = torch.randint(1, 32, (batch_size, seq_len))
        pixel_values = torch.randn(batch_size, 3, 4, 4)
        agent._packing_mode = MagicMock(return_value=None)
        agent._fused_model_pass = MagicMock(
            return_value=(
                torch.zeros(batch_size * 2, seq_len - 1),
                torch.zeros(batch_size * 2, seq_len - 1),
            ),
        )

        # Act
        agent._fused_forward(ids, batch_size, pixel_values=pixel_values)

        # Assert
        kwargs = agent._fused_model_pass.call_args.kwargs
        expected = pixel_values.repeat(2, 1, 1, 1)
        assert torch.equal(kwargs["pixel_values"], expected)

    def test_get_logprobs_passes_pixel_values_to_actor(self) -> None:
        # Arrange
        agent = _make_llm_agent()
        captured: list[dict[str, torch.Tensor]] = []
        original_forward = agent.actor.forward

        def recording_forward(**kwargs: torch.Tensor) -> MagicMock:
            captured.append(dict(kwargs))
            return original_forward(**kwargs)

        agent.actor.forward = recording_forward
        batch_size = 2
        seq_len = 5
        ids = torch.randint(1, 32, (batch_size, seq_len))
        pixel_values = torch.randn(batch_size, 3, 4, 4)

        # Act
        with patch.object(
            LLMAlgorithm,
            "_logprobs_from_hidden_fused_grad",
            return_value=torch.randn(batch_size, seq_len - 1),
        ):
            agent._get_logprobs(ids, batch_size=batch_size, pixel_values=pixel_values)

        # Assert
        assert torch.equal(captured[0]["pixel_values"], pixel_values)

    def test_fused_model_pass_moves_pixel_values_to_actor_device(self) -> None:
        # Arrange
        agent = _make_llm_agent()
        agent.device = torch.device("meta")
        agent.calc_position_embeddings = False
        captured: dict[str, torch.Tensor] = {}

        def recording_forward(**kwargs: torch.Tensor) -> SimpleNamespace:
            captured["pixel_values"] = kwargs["pixel_values"]
            batch, seq_len = kwargs["input_ids"].shape
            return SimpleNamespace(logits=torch.zeros(batch, seq_len, 4))

        agent.actor.forward = recording_forward
        agent._get_unwrapped_actor = lambda: agent.actor
        agent._patch_lm_head_to_identity = lambda: nullcontext()
        agent._amp_ctx = lambda: nullcontext()
        agent._activation_offload_ctx = lambda: nullcontext()
        agent._fused_logprob_fn_and_head = lambda: (
            lambda *args, **kwargs: torch.zeros(2, 4),
            torch.zeros(4, 4),
            None,
        )
        ids = torch.ones(2, 5, dtype=torch.long)
        mask = torch.ones_like(ids)
        pixel_values = torch.randn(2, 3, 2, 2)

        # Act
        agent._fused_model_pass(
            ids,
            mask,
            ["actor", "actor"],
            batch_size=2,
            pixel_values=pixel_values,
        )

        # Assert
        assert captured["pixel_values"].device.type == "meta"

    def test_get_logprobs_omits_pixel_values_when_absent(self) -> None:
        # Arrange
        agent = _make_llm_agent()
        captured: list[dict[str, torch.Tensor]] = []
        original_forward = agent.actor.forward

        def recording_forward(**kwargs: torch.Tensor) -> MagicMock:
            captured.append(dict(kwargs))
            return original_forward(**kwargs)

        agent.actor.forward = recording_forward
        batch_size = 2
        seq_len = 5
        ids = torch.randint(1, 32, (batch_size, seq_len))

        # Act
        with patch.object(
            LLMAlgorithm,
            "_logprobs_from_hidden_fused_grad",
            return_value=torch.randn(batch_size, seq_len - 1),
        ):
            agent._get_logprobs(ids, batch_size=batch_size)

        # Assert
        assert "pixel_values" not in captured[0]

    def test_get_logprobs_packed_keeps_pixel_batch_size(self) -> None:
        # Arrange
        agent = _make_llm_agent()
        captured: list[dict[str, torch.Tensor]] = []
        original_forward = agent.actor.forward

        def recording_forward(**kwargs: torch.Tensor) -> MagicMock:
            captured.append(dict(kwargs))
            return original_forward(**kwargs)

        agent.actor.forward = recording_forward
        agent._packing_mode = MagicMock(return_value="varlen")
        batch_size = 2
        seq_len = 5
        ids = torch.randint(1, 32, (batch_size, seq_len))
        pixel_values = torch.randn(batch_size, 3, 4, 4)

        agent._fused_logprob_fn_and_head = MagicMock(
            return_value=(
                MagicMock(return_value=torch.zeros(1, seq_len - 1)),
                torch.randn(32, 8),
                None,
            )
        )
        agent._patch_lm_head_to_identity = MagicMock(return_value=nullcontext())
        agent._amp_ctx = MagicMock(return_value=nullcontext())
        agent._activation_offload_ctx = MagicMock(return_value=nullcontext())

        # Act
        with (
            patch(
                "agilerl.algorithms.core.base.unpack_logprobs",
                return_value=torch.zeros(batch_size, seq_len - 1),
            ),
            torch.enable_grad(),
        ):
            agent._get_logprobs(ids, batch_size=batch_size, pixel_values=pixel_values)

        # Assert
        forwarded = captured[0]["pixel_values"]
        assert forwarded.shape[0] == batch_size
        assert torch.equal(forwarded, pixel_values)


@_LLM_DEPS_SKIP
class TestFusedKernelLossPixelValues:
    def test_fused_kernel_loss_moves_pixel_values_to_device(self) -> None:
        # Arrange
        grpo = _make_cpu_grpo_for_kernel_tests(use_liger_loss=True)
        grpo.device = torch.device("cpu")
        batch_size = 2
        seq_len = 5
        batch_ids = torch.randint(1, 32, (batch_size, seq_len))
        action_mask = torch.ones(batch_size, seq_len - 1, dtype=torch.bool)
        advantages = torch.zeros(batch_size)
        old_log_probs = torch.zeros(batch_size, seq_len - 1)
        reference_log_probs = torch.zeros(batch_size, seq_len - 1)
        pixel_values = torch.randn(batch_size, 3, 4, 4)
        hidden = torch.randn(batch_size, seq_len, 6)
        captured: list[dict[str, torch.Tensor]] = []

        def recording_forward(**kwargs: torch.Tensor) -> tuple[torch.Tensor]:
            captured.append(dict(kwargs))
            return (hidden,)

        grpo.actor.forward = recording_forward
        grpo._packing_mode = MagicMock(return_value=None)
        grpo._get_lm_head = MagicMock(
            return_value=torch.nn.Linear(6, 64, bias=False),
        )
        grpo._patch_lm_head_to_identity = MagicMock(return_value=nullcontext())
        grpo._amp_ctx = MagicMock(return_value=nullcontext())
        head = grpo._get_lm_head.return_value
        grpo._liger_head_gather = MagicMock(
            return_value=nullcontext((head.weight, head.bias))
        )
        grpo._resolve_fused_chunk_rows = MagicMock(return_value=2)

        fake_loss = torch.tensor(1.0, requires_grad=True)
        fake_aux = (torch.tensor(0.0), torch.tensor(0.0))
        mock_liger = MagicMock()
        mock_liger.apply = MagicMock(return_value=(fake_loss, fake_aux))

        # Act
        with (
            patch("agilerl.algorithms.grpo.HAS_LIGER_KERNEL", True),
            patch("agilerl.algorithms.grpo.LigerFusedLinearGRPOFunction", mock_liger),
        ):
            grpo._fused_kernel_loss(
                batch_ids,
                action_mask,
                advantages,
                old_log_probs,
                reference_log_probs,
                pixel_values=pixel_values,
            )

        # Assert
        forwarded = captured[0]["pixel_values"]
        assert forwarded.device == grpo.device
        assert torch.equal(forwarded, pixel_values)

    def test_fused_kernel_loss_packed_keeps_pixel_batch_size(self) -> None:
        # Arrange
        grpo = _make_cpu_grpo_for_kernel_tests(use_liger_loss=True)
        grpo.device = torch.device("cpu")
        batch_size = 2
        seq_len = 6
        batch_ids = torch.randint(1, 32, (batch_size, seq_len))
        action_mask = torch.ones(batch_size, seq_len - 1, dtype=torch.bool)
        advantages = torch.zeros(batch_size)
        old_log_probs = torch.zeros(batch_size, seq_len - 1)
        reference_log_probs = torch.zeros(batch_size, seq_len - 1)
        pixel_values = torch.randn(batch_size, 3, 4, 4)
        packed_hidden = torch.randn(1, seq_len, 6)
        padded_hidden = torch.randn(batch_size, seq_len, 6)
        captured: list[dict[str, torch.Tensor]] = []

        def recording_forward(**kwargs: torch.Tensor) -> tuple[torch.Tensor]:
            captured.append(dict(kwargs))
            return (packed_hidden,)

        grpo.actor.forward = recording_forward
        grpo._packing_mode = MagicMock(return_value="varlen")
        grpo._get_lm_head = MagicMock(
            return_value=torch.nn.Linear(6, 64, bias=False),
        )
        grpo._patch_lm_head_to_identity = MagicMock(return_value=nullcontext())
        grpo._amp_ctx = MagicMock(return_value=nullcontext())
        head = grpo._get_lm_head.return_value
        grpo._liger_head_gather = MagicMock(
            return_value=nullcontext((head.weight, head.bias))
        )
        grpo._resolve_fused_chunk_rows = MagicMock(return_value=2)

        fake_loss = torch.tensor(1.0, requires_grad=True)
        fake_aux = (torch.tensor(0.0), torch.tensor(0.0))
        mock_liger = MagicMock()
        mock_liger.apply = MagicMock(return_value=(fake_loss, fake_aux))

        # Act
        with (
            patch("agilerl.algorithms.grpo.HAS_LIGER_KERNEL", True),
            patch(
                "agilerl.algorithms.grpo.unpack_hidden_states",
                return_value=padded_hidden,
            ),
            patch("agilerl.algorithms.grpo.LigerFusedLinearGRPOFunction", mock_liger),
        ):
            grpo._fused_kernel_loss(
                batch_ids,
                action_mask,
                advantages,
                old_log_probs,
                reference_log_probs,
                pixel_values=pixel_values,
            )

        # Assert
        forwarded = captured[0]["pixel_values"]
        assert forwarded.shape[0] == batch_size
        assert forwarded.device == grpo.device
        assert torch.equal(forwarded, pixel_values)
