# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Fused forward passes vision tensors when the batch carries them."""

from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from peft import LoraConfig

from agilerl.algorithms import GRPO
from agilerl.algorithms.core.base import LLMAlgorithm
from agilerl.components.llm_rollout_data import EpisodeSegments
from agilerl.utils.segment_rows import SegmentRows
from tests.test_algorithms.test_core_base import _LLM_DEPS_SKIP, _make_llm_agent
from tests.test_algorithms.test_llms.llm_helpers import create_module
from tests.test_algorithms.test_llms.test_grpo_episode_loss_norm import (
    _ProbeFusedKernel,
)
from tests.test_algorithms.test_llms.vision_helpers import (
    IMAGE_TOKEN_ID,
    PAD_TOKEN_ID,
    PIXEL_DIM,
    VisionHiddenStatesModel,
    vision_config,
    vision_model,
)


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
        agent._fused_forward(ids, pixel_values=pixel_values)

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
        agent._fused_forward(ids)

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
        grpo = _make_cpu_grpo_for_kernel_tests(
            use_liger_loss=True, loss_norm="micro_batch"
        )
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

        # Act
        with (
            patch("agilerl.algorithms.grpo.HAS_LIGER_KERNEL", True),
            patch(
                "agilerl.algorithms.grpo.LigerFusedLinearGRPOFunction",
                _ProbeFusedKernel,
            ),
            patch.object(_ProbeFusedKernel, "inputs", []) as kernel_inputs,
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
        [inputs] = kernel_inputs
        assert inputs["num_items_in_batch"] == 1.0
        assert torch.equal(inputs["_input"], hidden[:, : seq_len - 1].reshape(-1, 1, 6))

    def test_fused_kernel_loss_packed_keeps_pixel_batch_size(self) -> None:
        # Arrange
        grpo = _make_cpu_grpo_for_kernel_tests(
            use_liger_loss=True, loss_norm="micro_batch"
        )
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

        # Act
        with (
            patch("agilerl.algorithms.grpo.HAS_LIGER_KERNEL", True),
            patch(
                "agilerl.algorithms.core.base.unpack_hidden_states",
                return_value=padded_hidden,
            ),
            patch(
                "agilerl.algorithms.grpo.LigerFusedLinearGRPOFunction",
                _ProbeFusedKernel,
            ),
            patch.object(_ProbeFusedKernel, "inputs", []) as kernel_inputs,
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
        [inputs] = kernel_inputs
        assert inputs["num_items_in_batch"] == 1.0
        assert torch.equal(
            inputs["_input"], padded_hidden[:, : seq_len - 1].reshape(-1, 1, 6)
        )


@_LLM_DEPS_SKIP
class TestPixelValuesForFusedSlice:
    def test_keeps_every_image_in_one_fused_row(self) -> None:
        agent = _make_llm_agent()
        pixels = torch.arange(8).reshape(8, 1, 1, 1)

        first = agent._pixel_values_for_fused_slice(pixels, 0, 1, 2)
        second = agent._pixel_values_for_fused_slice(pixels, 1, 2, 2)

        assert torch.equal(first, pixels[:4])
        assert torch.equal(second, pixels[4:])

    def test_slices_one_image_per_fused_row(self) -> None:
        agent = _make_llm_agent()
        pixels = torch.arange(2).reshape(2, 1, 1, 1)

        selected = agent._pixel_values_for_fused_slice(pixels, 0, 1, 2)

        assert torch.equal(selected, pixels[:1])

    def test_keeps_every_image_when_the_slice_spans_every_fused_row(self) -> None:
        agent = _make_llm_agent()
        pixels = torch.arange(6).reshape(6, 1, 1, 1)

        selected = agent._pixel_values_for_fused_slice(pixels, 0, 4, 4)

        assert torch.equal(selected, pixels)

    def test_rejects_a_leading_dim_that_does_not_divide_the_rows(self) -> None:
        agent = _make_llm_agent()
        pixels = torch.ones(3, 1, 1, 1)

        with pytest.raises(ValueError, match="does not divide"):
            agent._pixel_values_for_fused_slice(pixels, 0, 1, 2)

    def test_slices_unequal_image_counts(self) -> None:
        agent = _make_llm_agent()
        pixels = torch.arange(7).reshape(7, 1, 1, 1)

        first = agent._pixel_values_for_fused_slice(
            pixels, 0, 1, 2, image_counts=[3, 4]
        )
        second = agent._pixel_values_for_fused_slice(
            pixels, 1, 2, 2, image_counts=[3, 4]
        )

        assert torch.equal(first, pixels[:3])
        assert torch.equal(second, pixels[3:])

    def test_rejects_image_counts_that_do_not_sum_to_the_pixel_rows(self) -> None:
        agent = _make_llm_agent()
        pixels = torch.ones(5, 1, 1, 1)

        with pytest.raises(
            ValueError,
            match="pixel_values leading dim 5 does not match the fused image counts",
        ):
            agent._pixel_values_for_fused_slice(pixels, 0, 1, 2, image_counts=[3, 4])


@_LLM_DEPS_SKIP
class TestLLMAlgorithmFusedForwardNoGrad:
    def test_each_fused_row_gets_its_own_sample_images(self) -> None:
        # Arrange: sample 0 has one image, sample 1 has two; the reference and
        # actor rows of each sample run one row per forward.
        agent = _make_cpu_grpo_for_kernel_tests()
        ids = torch.randint(1, 32, (2, 5))
        pixel_values = torch.arange(3.0).reshape(3, 1)
        seen: list[list[float]] = []

        def record_pixel_values(
            _module: torch.nn.Module,
            _args: tuple[Any, ...],
            kwargs: dict[str, Any],
        ) -> None:
            seen.append(kwargs["pixel_values"].flatten().tolist())

        handle = agent.actor.get_base_model().register_forward_pre_hook(
            record_pixel_values, with_kwargs=True
        )

        # Act
        try:
            agent._fused_forward_no_grad(
                ids, 1, pixel_values=pixel_values, pixel_image_counts=[1, 2]
            )
        finally:
            handle.remove()

        # Assert
        assert seen == [[0.0], [1.0, 2.0], [0.0], [1.0, 2.0]]


@_LLM_DEPS_SKIP
class TestLLMAlgorithmSegmentRows:
    @staticmethod
    def _text_segment_rows() -> tuple[GRPO, SegmentRows, torch.Tensor, torch.Tensor]:
        """Segment rows of a batch whose episode 0 restarts into a segment with no image.

        Episode 0 splits ``[5, 5]`` with one image in segment 0 only; episode 1
        is unsegmented with one image. One row per micro-batch puts the
        text-only segment in a forward of its own.

        :return: The agent, its padded segment rows, and the text-only
            segment's token ids and action mask.
        """
        torch.manual_seed(0)
        agent = _make_cpu_grpo_for_kernel_tests(
            actor_network=VisionHiddenStatesModel(vision_config()),
            micro_batch_size_per_gpu=1,
            calc_position_embeddings=False,
            lora_config=LoraConfig(
                r=4,
                lora_alpha=8,
                target_modules=["linear_1"],
                task_type="CAUSAL_LM",
                lora_dropout=0.0,
            ),
        )
        ids = torch.randint(0, IMAGE_TOKEN_ID, (2, 10))
        ids[:, 0] = IMAGE_TOKEN_ID
        ids[1, 8:] = PAD_TOKEN_ID
        mask = torch.zeros(2, 9, dtype=torch.bool)
        mask[0, 1:4] = True
        mask[0, 6:9] = True
        mask[1, 1:7] = True
        segments = [
            EpisodeSegments(
                token_lengths=torch.tensor([5, 5]), pixel_rows=torch.tensor([1, 0])
            ),
            None,
        ]
        pixel_values = torch.arange(2.0).unsqueeze(-1).expand(2, PIXEL_DIM).contiguous()
        _split, rows = agent._segment_rows(
            ids,
            mask,
            segments,
            np.arange(2),
            pixel_values=pixel_values,
            pixel_image_counts=[1, 1],
            image_token_id=IMAGE_TOKEN_ID,
        )
        return agent, rows, ids[0:1, 5:], mask[0:1, 5:]

    def test_every_one_row_forward_runs_the_vision_tower(self) -> None:
        # Arrange
        agent, rows, _text_ids, _text_mask = self._text_segment_rows()
        model = vision_model(agent)

        # Act
        agent._fused_forward_no_grad(
            rows.token_ids,
            1,
            pixel_values=rows.pixel_values,
            pixel_image_counts=rows.pixel_image_counts,
            score_mask=rows.action_masks,
        )

        # Assert: the model records only forwards that carry vision rows, and
        # the reference and actor pass each forward every row on its own.
        assert len(model.forwards) == 2 * int(rows.token_ids.shape[0])
        assert all(forward["pixel_values"].shape[0] == 1 for forward in model.forwards)

    def test_text_segment_keeps_its_text_only_log_probs(self) -> None:
        # Arrange: row 1 is episode 0's text-only segment.
        agent, rows, text_ids, text_mask = self._text_segment_rows()
        text_positions = int(text_mask.shape[1])
        expected_ref, expected_actor, _ = agent._fused_forward_no_grad(
            text_ids, 1, score_mask=text_mask
        )

        # Act
        ref, actor, _ = agent._fused_forward_no_grad(
            rows.token_ids,
            1,
            pixel_values=rows.pixel_values,
            pixel_image_counts=rows.pixel_image_counts,
            score_mask=rows.action_masks,
        )

        # Assert: fp32 GEMMs over a wider row may round differently in the last bits.
        assert torch.equal(rows.token_ids[1, :5], text_ids[0])
        assert torch.allclose(
            ref[1:2, :text_positions], expected_ref, rtol=0.0, atol=1e-6
        )
        assert torch.allclose(
            actor[1:2, :text_positions], expected_actor, rtol=0.0, atol=1e-6
        )
        assert not actor[1, text_positions:].any()
