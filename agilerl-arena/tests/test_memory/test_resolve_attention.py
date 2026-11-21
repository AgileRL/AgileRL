# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0
"""The trainer's attention backend follows the model family, then the device."""

import pytest

from agilerl.arena.memory.resolve.attention import (
    FAMILY_ATTN_IMPLEMENTATIONS,
    KERNEL_RULE_MODEL_TYPES,
    context_parallel_blocker,
    resolve_attn_implementation,
)
from agilerl.arena.models.model_info import SUPPORTED_MODEL_INFO


class TestResolveAttnImplementation:
    @pytest.mark.parametrize(
        ("model_type", "expected"),
        [
            ("nemotron_h", "flash_attention_2"),
            ("nemotron_h_omni", "flash_attention_2"),
            ("gemma4", "flex_attention"),
            ("gemma3_text", "flex_attention"),
            ("gpt_oss", "flex_attention"),
            ("mimo_v2", "flex_attention"),
            ("deepseek_v4", "flex_attention"),
            ("glm5_next", "sdpa"),
            ("glm5_next_text", "sdpa"),
        ],
    )
    @pytest.mark.parametrize("cuda", [True, False])
    def test_family_backend_wins_on_any_device(
        self, model_type: str, expected: str, cuda: bool
    ) -> None:
        assert resolve_attn_implementation(model_type, cuda) == expected

    def test_flash_attention_2_on_cuda(self) -> None:
        assert resolve_attn_implementation("qwen3", True) == "flash_attention_2"

    def test_sdpa_off_cuda(self) -> None:
        assert resolve_attn_implementation("qwen3", False) == "sdpa"

    def test_unknown_model_type_takes_the_kernel_rule(self) -> None:
        assert resolve_attn_implementation(None, True) == "flash_attention_2"


class TestContextParallelBlocker:
    @pytest.mark.parametrize("backend", ["flash_attention_2", "flex_attention"])
    def test_ulysses_backends_run_context_parallel(self, backend: str) -> None:
        assert context_parallel_blocker(backend, 2) is None

    def test_cp_of_one_runs_on_any_backend(self) -> None:
        assert context_parallel_blocker("eager", 1) is None

    @pytest.mark.parametrize("model_type", ["glm5_next", "glm5_next_text"])
    def test_glm5_next_runs_context_parallel_on_its_pinned_backend(
        self, model_type: str
    ) -> None:
        backend = resolve_attn_implementation(model_type, True)

        assert backend == "sdpa"
        assert context_parallel_blocker(backend, 2, model_type) is None

    def test_dense_backend_refuses_context_parallel(self) -> None:
        reason = context_parallel_blocker("sdpa", 2, "qwen3")

        assert reason == (
            "fsdp.cp=2 needs flash_attention_2 or flex_attention attention; "
            "this model trains with sdpa"
        )

    def test_unknown_model_on_dense_backend_refuses_context_parallel(self) -> None:
        reason = context_parallel_blocker("eager", 4)

        assert reason == (
            "fsdp.cp=4 needs flash_attention_2 or flex_attention attention; "
            "this model trains with eager"
        )

    def test_deepseek_v4_runs_context_parallel_on_its_backend(self) -> None:
        backend = resolve_attn_implementation("deepseek_v4", True)

        assert context_parallel_blocker(backend, 2) is None


class TestAttnImplementationRegistry:
    def test_every_catalog_model_type_is_registered(self) -> None:
        catalog_types = {
            info.config["model_type"] for info in SUPPORTED_MODEL_INFO.values()
        }

        unregistered = (
            catalog_types - FAMILY_ATTN_IMPLEMENTATIONS.keys() - KERNEL_RULE_MODEL_TYPES
        )

        assert unregistered == set()

    def test_no_model_type_is_registered_twice(self) -> None:
        assert FAMILY_ATTN_IMPLEMENTATIONS.keys() & KERNEL_RULE_MODEL_TYPES == set()
