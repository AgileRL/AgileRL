# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Run settings Arena gives a manifest on a bundled environment."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import ValidationError

from agilerl.models import SUPPORTED_MODEL_INFO
from agilerl.models.bundled_envs import (
    BUNDLED_RUN_SETTINGS,
    BundledRunSettings,
    ContextLength,
    apply_bundled_run_settings,
    apply_context_length,
    apply_run_settings,
    context_minimum,
    deep_merge,
    fill_and_expand,
    is_already_expanded,
    manifest_model_type,
    trains_on_groups,
)
from agilerl.models.defaults import default_verdict
from agilerl.models.env import BUNDLED_ENVIRONMENTS, LLMEnvSpec
from agilerl.models.manifest import TrainingManifest, fill_default_context_length
from agilerl.models.verdict import context_length_errors, verdict

GEMMA4 = "google/gemma-4-E4B-it"
NEMOTRON_NANO = "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16"
QWEN = "Qwen/Qwen3.5-0.8B"

VWA_SETTINGS = BundledRunSettings(context_length=ContextLength(default=32768))


def _context_manifest(
    algorithm: dict[str, Any], network: dict[str, Any] | None = None
) -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "algorithm": algorithm,
        "environment": {"segment_prompt_tokens": 24000},
    }
    if network is not None:
        manifest["network"] = network
    return manifest


def _grpo_on(environment: dict[str, Any], network: dict[str, Any]) -> dict[str, Any]:
    return {
        "algorithm": {"name": "GRPO"},
        "environment": environment,
        "network": network,
    }


class TestApplyRunSettings:
    def test_fills_unset_fields_and_keeps_manifest_values(self) -> None:
        settings = BundledRunSettings(
            defaults={"algorithm": {"max_output_tokens": 2048, "temperature": 1.0}},
        )
        manifest: dict[str, Any] = {"algorithm": {"temperature": 0.7}}

        apply_run_settings(manifest, "Env", settings)

        assert manifest == {
            "algorithm": {"max_output_tokens": 2048, "temperature": 0.7}
        }

    def test_applies_group_defaults_to_a_grouped_grpo_family_algorithm(self) -> None:
        settings = BundledRunSettings(
            defaults={"replay_buffer": {"kind": "llm", "max_size": 2}},
            group_defaults={"replay_buffer": {"group_replay_max_age": 10}},
        )
        manifest: dict[str, Any] = {"algorithm": {"name": "CISPO", "group_size": 4}}

        apply_run_settings(manifest, "Env", settings)

        assert manifest["replay_buffer"] == {
            "kind": "llm",
            "max_size": 2,
            "group_replay_max_age": 10,
        }

    @pytest.mark.parametrize(
        "algorithm",
        [{"name": "LLMPPO"}, {"name": "GRPO", "group_size": 1}],
        ids=["llmppo", "grpo-group-size-1"],
    )
    def test_skips_group_defaults_without_groups(
        self, algorithm: dict[str, Any]
    ) -> None:
        settings = BundledRunSettings(
            defaults={"replay_buffer": {"kind": "llm"}},
            group_defaults={"replay_buffer": {"group_replay_max_age": 10}},
        )
        manifest: dict[str, Any] = {"algorithm": algorithm}

        apply_run_settings(manifest, "Env", settings)

        assert manifest["replay_buffer"] == {"kind": "llm"}

    def test_keeps_a_manifest_group_replay_age(self) -> None:
        settings = BundledRunSettings(
            group_defaults={"replay_buffer": {"group_replay_max_age": 10}},
        )
        manifest: dict[str, Any] = {
            "algorithm": {"name": "GRPO"},
            "replay_buffer": {"group_replay_max_age": 0},
        }

        apply_run_settings(manifest, "Env", settings)

        assert manifest["replay_buffer"] == {"group_replay_max_age": 0}

    def test_checks_the_context_after_filling_defaults(self) -> None:
        settings = BundledRunSettings(
            defaults={"algorithm": {"max_output_tokens": 5120}},
            context_length=VWA_SETTINGS.context_length,
        )
        manifest = _context_manifest({}, {"max_context_length": 16384})

        with pytest.raises(
            ValueError,
            match=(
                r"Env needs network\.max_context_length >= 29120 "
                r"\(segment_prompt_tokens 24000 \+ max_output_tokens 5120\); "
                r"the manifest sets 16384"
            ),
        ):
            apply_run_settings(manifest, "Env", settings)

    @pytest.mark.parametrize(
        ("max_output_tokens", "budget"),
        [(5120, 4096), (2048, 1024), (1025, 1)],
        ids=["profile-default", "2048", "1025"],
    )
    def test_derives_the_thinking_budget_from_max_output_tokens(
        self, max_output_tokens: int, budget: int
    ) -> None:
        settings = BundledRunSettings(
            defaults={"algorithm": {"max_output_tokens": 5120}},
            thinking_budget_reserve=1024,
        )
        manifest: dict[str, Any] = {
            "algorithm": {"max_output_tokens": max_output_tokens},
            "network": {"pretrained_model_name_or_path": GEMMA4},
        }

        apply_run_settings(manifest, "Env", settings)

        assert manifest["algorithm"]["thinking_token_budget"] == budget

    def test_derives_the_thinking_budget_for_a_model_named_on_the_algorithm(
        self,
    ) -> None:
        settings = BundledRunSettings(thinking_budget_reserve=1024)
        manifest: dict[str, Any] = {
            "algorithm": {
                "max_output_tokens": 2048,
                "pretrained_model_name_or_path": "Qwen/Qwen3-1.7B",
            }
        }

        apply_run_settings(manifest, "Env", settings)

        assert manifest["algorithm"]["thinking_token_budget"] == 1024

    @pytest.mark.parametrize(
        "model",
        ["Qwen/Qwen2.5-0.5B-Instruct", "tiny-model", None],
        ids=["qwen2", "unsupported", "unset"],
    )
    def test_omits_the_thinking_budget_for_a_model_without_a_reasoning_parser(
        self, model: str | None
    ) -> None:
        settings = BundledRunSettings(thinking_budget_reserve=1024)
        manifest: dict[str, Any] = {
            "algorithm": {"max_output_tokens": 5120},
            "network": {"pretrained_model_name_or_path": model},
        }

        apply_run_settings(manifest, "Env", settings)

        assert "thinking_token_budget" not in manifest["algorithm"]

    @pytest.mark.parametrize(
        "algorithm",
        [
            {"max_output_tokens": 5120},
            {
                "max_output_tokens": 5120,
                "vllm_engine_args": {"reasoning_parser": "openai_gptoss"},
            },
        ],
        ids=["family-parser", "manifest-parser"],
    )
    def test_omits_the_thinking_budget_for_gpt_oss(
        self, algorithm: dict[str, Any]
    ) -> None:
        settings = BundledRunSettings(thinking_budget_reserve=1024)
        manifest: dict[str, Any] = {
            "algorithm": algorithm,
            "network": {"pretrained_model_name_or_path": "openai/gpt-oss-20b"},
        }

        apply_run_settings(manifest, "Env", settings)

        assert "thinking_token_budget" not in manifest["algorithm"]

    def test_derives_the_thinking_budget_from_a_manifest_reasoning_parser(
        self,
    ) -> None:
        settings = BundledRunSettings(thinking_budget_reserve=1024)
        manifest: dict[str, Any] = {
            "algorithm": {
                "max_output_tokens": 5120,
                "vllm_engine_args": {"reasoning_parser": "qwen3"},
            },
            "network": {"pretrained_model_name_or_path": "tiny-model"},
        }

        apply_run_settings(manifest, "Env", settings)

        assert manifest["algorithm"]["thinking_token_budget"] == 4096

    def test_reads_the_model_from_the_algorithm_when_network_is_null(self) -> None:
        settings = BundledRunSettings(thinking_budget_reserve=1024)
        manifest: dict[str, Any] = {
            "algorithm": {
                "max_output_tokens": 5120,
                "pretrained_model_name_or_path": "Qwen/Qwen3-1.7B",
            },
            "network": None,
        }

        apply_run_settings(manifest, "Env", settings)

        assert manifest["algorithm"]["thinking_token_budget"] == 4096

    @pytest.mark.parametrize(
        "algorithm",
        [{"max_output_tokens": 1024}, {"max_output_tokens": None}],
        ids=["1024", "unset"],
    )
    def test_omits_the_thinking_budget_when_it_cannot_be_valid(
        self, algorithm: dict[str, Any]
    ) -> None:
        settings = BundledRunSettings(thinking_budget_reserve=1024)
        manifest: dict[str, Any] = {
            "algorithm": algorithm,
            "network": {"pretrained_model_name_or_path": GEMMA4},
        }

        apply_run_settings(manifest, "Env", settings)

        assert "thinking_token_budget" not in manifest["algorithm"]

    @pytest.mark.parametrize("budget", [512, None], ids=["512", "null"])
    def test_keeps_a_manifest_thinking_budget(self, budget: int | None) -> None:
        settings = BundledRunSettings(
            defaults={"algorithm": {"max_output_tokens": 5120}},
            thinking_budget_reserve=1024,
        )
        manifest: dict[str, Any] = {"algorithm": {"thinking_token_budget": budget}}

        apply_run_settings(manifest, "Env", settings)

        assert manifest["algorithm"]["thinking_token_budget"] == budget

    def test_merges_a_vllm_config_default_onto_the_colocated_default(self) -> None:
        settings = BundledRunSettings(
            defaults={
                "algorithm": {"vllm_config": {"limit_mm_per_prompt": {"image": 4}}}
            },
        )
        manifest: dict[str, Any] = {"algorithm": {"name": "CISPO"}}

        apply_run_settings(manifest, "Env", settings)

        assert manifest["algorithm"]["vllm_config"] == {
            "gpu_memory_utilization": 0.9,
            "limit_mm_per_prompt": {"image": 4},
        }

    def test_merges_a_vllm_config_default_onto_the_manifest_vllm_config(self) -> None:
        settings = BundledRunSettings(
            defaults={
                "algorithm": {"vllm_config": {"limit_mm_per_prompt": {"image": 4}}}
            },
        )
        manifest: dict[str, Any] = {
            "algorithm": {"name": "CISPO", "vllm_config": {"max_num_seqs": 64}}
        }

        apply_run_settings(manifest, "Env", settings)

        assert manifest["algorithm"]["vllm_config"] == {
            "max_num_seqs": 64,
            "limit_mm_per_prompt": {"image": 4},
        }

    def test_skips_the_colocated_default_for_a_non_rollout_algorithm(self) -> None:
        settings = BundledRunSettings(
            defaults={
                "algorithm": {"vllm_config": {"limit_mm_per_prompt": {"image": 4}}}
            },
        )
        manifest: dict[str, Any] = {"algorithm": {"name": "SFT"}}

        apply_run_settings(manifest, "Env", settings)

        assert manifest["algorithm"]["vllm_config"] == {
            "limit_mm_per_prompt": {"image": 4}
        }


class TestTrainsOnGroups:
    @pytest.mark.parametrize(
        "algorithm",
        [
            {"name": "GRPO"},
            {"name": "CISPO", "group_size": 2},
            {"name": "GSPO", "group_size": 8},
        ],
        ids=["grpo-default-group-size", "cispo", "gspo"],
    )
    def test_accepts_a_grpo_family_algorithm_with_groups(
        self, algorithm: dict[str, Any]
    ) -> None:
        assert trains_on_groups(algorithm) is True

    @pytest.mark.parametrize(
        "algorithm",
        [
            {"name": "GRPO", "group_size": 1},
            {"name": "LLMPPO"},
            {"name": "SFT"},
            {},
        ],
        ids=["group-size-1", "llmppo", "unregistered", "unnamed"],
    )
    def test_rejects_an_algorithm_without_groups(
        self, algorithm: dict[str, Any]
    ) -> None:
        assert trains_on_groups(algorithm) is False


class TestMalformedSections:
    @pytest.mark.parametrize("name", ["AgentWorldModel", "VisualWebArena"])
    def test_a_non_mapping_algorithm_is_a_verdict_error(self, name: str) -> None:
        # Arrange
        document = _grpo_on(
            {"name": name, "env_type": "rollout"},
            {"pretrained_model_name_or_path": QWEN},
        )
        document["algorithm"] = "GRPO"

        # Act
        record = verdict(document)

        # Assert
        assert record["ok"] is False
        assert any(
            "algorithm section must be a mapping, got str" in error["message"]
            for error in record["errors"]
        )

    def test_a_non_mapping_replay_buffer_is_a_verdict_error(self) -> None:
        # Arrange
        document = _grpo_on(
            {"name": "VisualWebArena", "env_type": "rollout"},
            {"pretrained_model_name_or_path": QWEN},
        )
        document["replay_buffer"] = "llm"

        # Act
        record = verdict(document)

        # Assert
        assert record["ok"] is False
        assert any(
            "replay_buffer section must be a mapping, got str" in error["message"]
            for error in record["errors"]
        )

    def test_a_non_mapping_network_is_a_verdict_error(self) -> None:
        # Arrange
        document = _grpo_on(
            {"name": "AgentWorldModel", "env_type": "rollout"},
            {"pretrained_model_name_or_path": QWEN},
        )
        document["network"] = QWEN

        # Act
        record = verdict(document)

        # Assert
        assert record["ok"] is False
        assert any(
            "network section must be a mapping, got str" in error["message"]
            for error in record["errors"]
        )

    def test_fills_defaults_when_the_algorithm_section_is_absent(self) -> None:
        # Arrange
        settings = BundledRunSettings(
            defaults={"algorithm": {"max_output_tokens": 2048}}
        )
        manifest: dict[str, Any] = {}

        # Act
        apply_run_settings(manifest, "Env", settings)

        # Assert
        assert manifest["algorithm"] == {"max_output_tokens": 2048}


class TestApplyContextLength:
    def test_fills_context_when_the_algorithm_section_is_absent(self) -> None:
        # Arrange
        manifest = {"environment": {"segment_prompt_tokens": 24000}}

        # Act
        apply_context_length(manifest, "Env", VWA_SETTINGS)

        # Assert
        assert manifest["algorithm"] == {}
        assert manifest["network"] == {"max_context_length": 32768}

    def test_fills_the_default_when_no_context_is_set(self) -> None:
        manifest = _context_manifest({"max_output_tokens": 5120})

        apply_context_length(manifest, "Env", VWA_SETTINGS)

        assert manifest["network"] == {"max_context_length": 32768}
        assert "max_model_len" not in manifest["algorithm"]

    def test_network_context_takes_a_manifest_max_model_len(self) -> None:
        manifest = _context_manifest(
            {"max_output_tokens": 5120, "max_model_len": 65536}, {}
        )

        apply_context_length(manifest, "Env", VWA_SETTINGS)

        assert manifest["network"] == {"max_context_length": 65536}

    def test_keeps_a_manifest_network_context(self) -> None:
        manifest = _context_manifest(
            {"max_output_tokens": 5120}, {"max_context_length": 29120}
        )

        apply_context_length(manifest, "Env", VWA_SETTINGS)

        assert manifest["network"] == {"max_context_length": 29120}

    @pytest.mark.parametrize(
        ("algorithm", "network", "field"),
        [
            ({}, {"max_context_length": 4096}, "network.max_context_length"),
            ({"max_model_len": 4096}, {}, "algorithm.max_model_len"),
            (
                {"max_model_len": 4096},
                {"max_context_length": 32768},
                "algorithm.max_model_len",
            ),
        ],
        ids=["network", "algorithm", "algorithm-beside-network"],
    )
    def test_rejects_a_manifest_field_below_a_fixed_minimum(
        self, algorithm: dict[str, Any], network: dict[str, Any], field: str
    ) -> None:
        manifest: dict[str, Any] = {"algorithm": algorithm, "network": network}
        context = ContextLength(default=32768, minimum=32768)

        with pytest.raises(
            ValueError,
            match=rf"Env needs {field} >= 32768; the manifest sets 4096",
        ):
            apply_context_length(
                manifest, "Env", BundledRunSettings(context_length=context)
            )

    def test_minimum_follows_max_output_tokens(self) -> None:
        manifest = _context_manifest(
            {"max_output_tokens": 10000}, {"max_context_length": 32768}
        )

        with pytest.raises(
            ValueError,
            match=(
                r"Env needs network\.max_context_length >= 34000 "
                r"\(segment_prompt_tokens 24000 \+ max_output_tokens 10000\); "
                r"the manifest sets 32768"
            ),
        ):
            apply_context_length(manifest, "Env", VWA_SETTINGS)

    def test_fills_the_minimum_when_the_default_is_below_it(self) -> None:
        manifest = _context_manifest({"max_output_tokens": 10000})

        apply_context_length(manifest, "Env", VWA_SETTINGS)

        assert manifest["network"] == {"max_context_length": 65536}
        assert "max_model_len" not in manifest["algorithm"]

    def test_rounds_a_short_default_up_to_the_smallest_option(self) -> None:
        manifest: dict[str, Any] = {"algorithm": {}, "environment": {}}
        settings = BundledRunSettings(
            context_length=ContextLength(default=512, minimum=512)
        )

        apply_context_length(manifest, "Env", settings)

        assert manifest["network"] == {"max_context_length": 1024}

    def test_answer_continuation_doubles_the_turn(self) -> None:
        manifest = _context_manifest(
            {"max_output_tokens": 5120, "answer_continuation": True},
            {"max_context_length": 32768},
        )

        with pytest.raises(
            ValueError,
            match=(
                r"Env needs network\.max_context_length >= 34240 "
                r"\(segment_prompt_tokens 24000 \+ max_output_tokens 5120 "
                r"x 2 for answer_continuation\); the manifest sets 32768"
            ),
        ):
            apply_context_length(manifest, "Env", VWA_SETTINGS)

    def test_accepts_a_context_that_fits_the_answer_continuation(self) -> None:
        manifest = _context_manifest(
            {"max_output_tokens": 5120, "answer_continuation": True},
            {"max_context_length": 34240},
        )

        apply_context_length(manifest, "Env", VWA_SETTINGS)

        assert manifest["network"] == {"max_context_length": 34240}

    def test_unset_segment_prompt_tokens_has_no_minimum(self) -> None:
        manifest = _context_manifest(
            {"max_output_tokens": 5120}, {"max_context_length": 4096}
        )
        manifest["environment"]["segment_prompt_tokens"] = None

        apply_context_length(manifest, "Env", VWA_SETTINGS)

        assert manifest["network"] == {"max_context_length": 4096}

    def test_unset_max_output_tokens_has_no_minimum(self) -> None:
        manifest = _context_manifest({}, {"max_context_length": 4096})

        apply_context_length(manifest, "Env", VWA_SETTINGS)

        assert manifest["network"] == {"max_context_length": 4096}


class TestApplyContextLengthCatalogCap:
    def test_caps_the_fill_at_the_catalog_window(self) -> None:
        # Arrange
        window = SUPPORTED_MODEL_INFO[QWEN].max_context_length
        settings = BundledRunSettings(
            context_length=ContextLength(default=window * 2, minimum=1024)
        )
        manifest: dict[str, Any] = {
            "algorithm": {},
            "environment": {},
            "network": {"pretrained_model_name_or_path": QWEN},
        }

        # Act
        apply_context_length(manifest, "Env", settings)

        # Assert
        assert manifest["network"]["max_context_length"] == window

    def test_rejects_a_model_whose_window_is_below_the_minimum(self) -> None:
        # Arrange
        window = SUPPORTED_MODEL_INFO[QWEN].max_context_length
        settings = BundledRunSettings(
            context_length=ContextLength(default=window * 2, minimum=window * 2)
        )
        manifest: dict[str, Any] = {
            "algorithm": {},
            "environment": {},
            "network": {"pretrained_model_name_or_path": QWEN},
        }

        # Act / Assert
        with pytest.raises(
            ValueError,
            match=rf"{QWEN} is not supported on Env: it has {window} tokens",
        ):
            apply_context_length(manifest, "Env", settings)

    def test_uses_the_catalog_window_when_the_next_option_exceeds_it(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange
        monkeypatch.setattr(
            "agilerl.models.bundled_envs.SUPPORTED_MODEL_INFO",
            {"fake/model": SimpleNamespace(max_context_length=40000)},
        )
        manifest = _context_manifest(
            {"max_output_tokens": 10000},
            {"pretrained_model_name_or_path": "fake/model"},
        )

        # Act
        apply_context_length(manifest, "Env", VWA_SETTINGS)

        # Assert
        assert manifest["network"]["max_context_length"] == 40000


class TestFillDefaultContextLength:
    def test_fills_2048_for_an_unset_context(self) -> None:
        # Arrange
        manifest = _grpo_on({"name": "Mine"}, {"pretrained_model_name_or_path": QWEN})

        # Act
        filled = fill_default_context_length(manifest)

        # Assert
        assert filled is True
        assert manifest["network"]["max_context_length"] == 2048

    def test_leaves_room_for_the_output_cap(self) -> None:
        # Arrange
        manifest = _grpo_on({"name": "Mine"}, {"pretrained_model_name_or_path": QWEN})
        manifest["algorithm"]["max_output_tokens"] = 4096

        # Act
        fill_default_context_length(manifest)

        # Assert
        assert manifest["network"]["max_context_length"] == 8192

    def test_reads_a_quoted_output_cap(self) -> None:
        # Arrange
        manifest = _grpo_on({"name": "Mine"}, {"pretrained_model_name_or_path": QWEN})
        manifest["algorithm"]["max_output_tokens"] = "4096"

        # Act
        fill_default_context_length(manifest)

        # Assert
        assert manifest["network"]["max_context_length"] == 8192

    def test_keeps_2048_when_the_output_cap_is_not_an_int(self) -> None:
        # Arrange
        manifest = _grpo_on({"name": "Mine"}, {"pretrained_model_name_or_path": QWEN})
        manifest["algorithm"]["max_output_tokens"] = "lots"

        # Act
        filled = fill_default_context_length(manifest)

        # Assert
        assert filled is True
        assert manifest["network"]["max_context_length"] == 2048

    def test_leaves_room_for_an_answer_continuation(self) -> None:
        # Arrange
        manifest = _grpo_on({"name": "Mine"}, {"pretrained_model_name_or_path": QWEN})
        manifest["algorithm"]["max_output_tokens"] = 2048
        manifest["algorithm"]["answer_continuation"] = True

        # Act
        fill_default_context_length(manifest)

        # Assert
        assert manifest["network"]["max_context_length"] == 8192

    def test_reads_a_quoted_answer_continuation(self) -> None:
        # Arrange
        manifest = _grpo_on({"name": "Mine"}, {"pretrained_model_name_or_path": QWEN})
        manifest["algorithm"]["max_output_tokens"] = "2048"
        manifest["algorithm"]["answer_continuation"] = "true"

        # Act
        fill_default_context_length(manifest)

        # Assert
        assert manifest["network"]["max_context_length"] == 8192

    def test_answer_continuation_without_a_cap_stays_at_2048(self) -> None:
        # Arrange
        manifest = _grpo_on({"name": "Mine"}, {"pretrained_model_name_or_path": QWEN})
        manifest["algorithm"]["answer_continuation"] = True

        # Act
        fill_default_context_length(manifest)

        # Assert
        assert manifest["network"]["max_context_length"] == 2048

    def test_caps_at_the_catalog_window(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Arrange
        monkeypatch.setattr(
            "agilerl.models.manifest.SUPPORTED_MODEL_INFO",
            {"fake/model": SimpleNamespace(max_context_length=1500)},
        )
        manifest = _grpo_on(
            {"name": "Mine"}, {"pretrained_model_name_or_path": "fake/model"}
        )

        # Act
        fill_default_context_length(manifest)

        # Assert
        assert manifest["network"]["max_context_length"] == 1500

    def test_keeps_a_set_context(self) -> None:
        # Arrange
        manifest = _grpo_on(
            {"name": "Mine"},
            {"pretrained_model_name_or_path": QWEN, "max_context_length": 4096},
        )

        # Act
        filled = fill_default_context_length(manifest)

        # Assert
        assert filled is False
        assert manifest["network"]["max_context_length"] == 4096


class TestApplyBundledRunSettings:
    def test_fills_a_name_only_visual_web_arena_from_its_segment(self) -> None:
        # Arrange
        manifest = {
            "algorithm": {"name": "CISPO", "max_output_tokens": 10000},
            "environment": {"name": "VisualWebArena", "env_type": "rollout"},
        }

        # Act
        apply_bundled_run_settings(manifest)

        # Assert
        assert manifest["network"] == {"max_context_length": 65536}

    def test_rejects_a_text_only_model_on_visual_web_arena(self) -> None:
        # Arrange
        manifest = {
            "algorithm": {"name": "CISPO"},
            "environment": {"name": "VisualWebArena", "env_type": "rollout"},
            "network": {"pretrained_model_name_or_path": NEMOTRON_NANO},
        }

        # Act / Assert
        with pytest.raises(
            ValueError,
            match=rf"{NEMOTRON_NANO} cannot train on VisualWebArena: it takes text only",
        ):
            apply_bundled_run_settings(manifest)

    def test_accepts_a_text_only_model_on_agent_world_model(self) -> None:
        # Arrange
        manifest = {
            "algorithm": {"name": "GRPO"},
            "environment": {"name": "AgentWorldModel", "env_type": "rollout"},
            "network": {"pretrained_model_name_or_path": NEMOTRON_NANO},
        }

        # Act
        apply_bundled_run_settings(manifest)

        # Assert
        assert manifest["network"]["max_context_length"] == 32768

    def test_fills_an_environment_that_names_its_image(self) -> None:
        # Arrange
        manifest: dict[str, Any] = {
            "environment": {"name": "AgentWorldModel", "env_image": "awm:custom"},
        }

        # Act
        apply_bundled_run_settings(manifest)

        # Assert
        assert manifest["environment"] == {
            "name": "AgentWorldModel",
            "env_image": "awm:custom",
        }
        assert manifest["algorithm"]["max_output_tokens"] == 2048
        assert manifest["network"]["max_context_length"] == 32768

    def test_a_second_pass_changes_nothing(self) -> None:
        # Arrange
        manifest: dict[str, Any] = {
            "algorithm": {"name": "GRPO", "group_size": 4},
            "environment": {"name": "VisualWebArena", "env_type": "rollout"},
            "network": {"pretrained_model_name_or_path": QWEN},
        }
        apply_bundled_run_settings(manifest)
        once = copy.deepcopy(manifest)

        # Act
        apply_bundled_run_settings(manifest)

        # Assert
        assert manifest == once

    def test_leaves_a_non_bundled_environment(self) -> None:
        # Arrange
        manifest: dict[str, Any] = {"environment": {"name": "Mine"}}

        # Act
        apply_bundled_run_settings(manifest)

        # Assert
        assert manifest == {"environment": {"name": "Mine"}}

    def test_covers_every_bundled_environment(self) -> None:
        assert set(BUNDLED_RUN_SETTINGS) == BUNDLED_ENVIRONMENTS


class TestTrainingManifestRunSettings:
    def test_fills_a_bundled_environment_without_touching_the_document(self) -> None:
        # Arrange
        document = _grpo_on(
            {"name": "AgentWorldModel", "env_type": "rollout", "num_envs": 1},
            {"pretrained_model_name_or_path": QWEN},
        )

        # Act
        manifest = TrainingManifest.model_validate(document)

        # Assert
        assert manifest.network.max_context_length == 32768
        assert manifest.algorithm.max_output_tokens == 2048
        assert manifest.context_defaulted is False
        assert "max_context_length" not in document["network"]

    def test_rounds_a_visual_web_arena_segment_fill_to_an_allowed_length(
        self,
    ) -> None:
        # Arrange
        document = _grpo_on(
            {"name": "VisualWebArena", "env_type": "rollout"},
            {"pretrained_model_name_or_path": QWEN},
        )
        document["algorithm"]["max_output_tokens"] = 10000

        # Act
        manifest = TrainingManifest.model_validate(document)

        # Assert
        assert manifest.network.max_context_length == 65536
        assert context_length_errors(manifest) == []

    def test_rounds_a_visual_web_arena_continuation_fill_to_an_allowed_length(
        self,
    ) -> None:
        # Arrange
        document = _grpo_on(
            {"name": "VisualWebArena", "env_type": "rollout"},
            {"pretrained_model_name_or_path": QWEN},
        )
        document["algorithm"]["answer_continuation"] = True
        document["algorithm"]["constrain_answer_pattern"] = r"stop \[.*\]"

        # Act
        manifest = TrainingManifest.model_validate(document)

        # Assert
        assert manifest.network.max_context_length == 65536
        assert context_length_errors(manifest) == []

    def test_fills_the_default_context_for_an_unset_context(self) -> None:
        # Arrange
        document = _grpo_on(
            {
                "name": "Mine",
                "env_type": "rollout",
                "env_url": "http://env",
                "max_turns": 1,
            },
            {"pretrained_model_name_or_path": QWEN},
        )

        # Act
        manifest = TrainingManifest.model_validate(document)

        # Assert
        assert manifest.network.max_context_length == 2048
        assert manifest.context_defaulted is True

    def test_keeps_a_set_context(self) -> None:
        # Arrange
        document = _grpo_on(
            {
                "name": "Mine",
                "env_type": "rollout",
                "env_url": "http://env",
                "max_turns": 1,
            },
            {"pretrained_model_name_or_path": QWEN, "max_context_length": 4096},
        )

        # Act
        manifest = TrainingManifest.model_validate(document)

        # Assert
        assert manifest.network.max_context_length == 4096
        assert manifest.context_defaulted is False

    def test_rejects_a_bundled_context_below_the_minimum(self) -> None:
        # Arrange
        document = _grpo_on(
            {"name": "AgentWorldModel", "env_type": "rollout", "num_envs": 1},
            {"pretrained_model_name_or_path": QWEN, "max_context_length": 4096},
        )

        # Act / Assert
        with pytest.raises(
            ValidationError,
            match=r"AgentWorldModel needs network\.max_context_length >= 32768",
        ):
            TrainingManifest.model_validate(document)


class TestVerdictMalformedBundledManifest:
    @pytest.mark.parametrize(
        ("section", "value"),
        [
            ("algorithm", {"name": "GRPO", "group_size": "many"}),
            ("algorithm", {"name": "GRPO", "max_output_tokens": "lots"}),
            ("network", "Qwen"),
            ("replay_buffer", "llm"),
        ],
        ids=["group-size", "max-output-tokens", "network", "replay-buffer"],
    )
    def test_reports_a_wrongly_typed_field_as_an_error(
        self, section: str, value: object
    ) -> None:
        # Arrange
        document = _grpo_on(
            {"name": "VisualWebArena"}, {"pretrained_model_name_or_path": QWEN}
        )
        document[section] = value

        # Act
        record = verdict(document)

        # Assert
        assert record["ok"] is False
        assert record["errors"]

    def test_reads_a_quoted_group_size_as_pydantic_does(self) -> None:
        # Arrange
        document = _grpo_on(
            {"name": "VisualWebArena"}, {"pretrained_model_name_or_path": QWEN}
        )
        document["algorithm"]["group_size"] = "4"

        # Act
        manifest = TrainingManifest.model_validate(document)

        # Assert
        assert manifest.replay_buffer.group_replay_max_age == 10

    def test_rejects_a_quoted_context_below_the_minimum(self) -> None:
        # Arrange
        document = _grpo_on(
            {"name": "AgentWorldModel"},
            {"pretrained_model_name_or_path": QWEN, "max_context_length": "4096"},
        )

        # Act
        record = verdict(document)

        # Assert
        assert record["ok"] is False
        assert (
            "AgentWorldModel needs network.max_context_length >= 32768"
            in (record["errors"][0]["message"])
        )


class TestLLMEnvSpecBundledSource:
    def test_accepts_a_name_only_bundled_environment(self) -> None:
        # Act
        env = LLMEnvSpec.model_validate(
            {"name": "VisualWebArena", "env_type": "rollout"}
        )

        # Assert
        assert env.env_image is None

    def test_rejects_a_name_only_environment_that_is_not_bundled(self) -> None:
        with pytest.raises(ValidationError, match="exactly one source"):
            LLMEnvSpec.model_validate({"name": "Mine", "env_type": "rollout"})


class TestNameOnlyBundledSubmit:
    @pytest.mark.parametrize("name", sorted(BUNDLED_ENVIRONMENTS))
    def test_a_name_only_bundled_run_passes_the_submit_check(self, name: str) -> None:
        # Arrange
        document = _grpo_on(
            {"name": name, "env_type": "rollout"},
            {"pretrained_model_name_or_path": QWEN},
        )
        document["training"] = {"max_steps": 10}

        # Act
        record = verdict(document, submit=True)

        # Assert
        assert record["ok"], record

    @pytest.mark.parametrize("name", sorted(BUNDLED_ENVIRONMENTS))
    def test_the_user_view_default_passes_the_submit_check(self, name: str) -> None:
        # Arrange
        template = default_verdict(
            _grpo_on(
                {"name": name, "env_type": "rollout"},
                {"pretrained_model_name_or_path": QWEN},
            ),
            user_view=True,
        )
        assert template["ok"], template

        # Act
        record = verdict(template["payload"], submit=True)

        # Assert
        assert record["ok"], record


class TestDeepMerge:
    def test_nested_dicts_merge_and_other_values_replace(self) -> None:
        merged = deep_merge(
            {"env_config": {"action_field": "action_str", "keep": 1}, "num_envs": 8},
            {"env_config": {"system_prompt": "go"}, "num_envs": 4},
        )

        assert merged == {
            "env_config": {
                "action_field": "action_str",
                "keep": 1,
                "system_prompt": "go",
            },
            "num_envs": 4,
        }


class TestIsAlreadyExpanded:
    @pytest.mark.parametrize(
        "env",
        [
            {"env_image": "awm:1"},
            {"env_url": "http://env"},
            {"entrypoint": "mod:Env"},
            {"env_services": [{"name": "site"}]},
        ],
    )
    def test_a_named_source_counts(self, env: dict[str, Any]) -> None:
        assert is_already_expanded(env) is True

    @pytest.mark.parametrize("env", [{"name": "AgentWorldModel"}, {"env_services": []}])
    def test_no_source_does_not_count(self, env: dict[str, Any]) -> None:
        assert is_already_expanded(env) is False


class TestManifestModelType:
    def test_a_local_checkpoint_reads_its_config(self, tmp_path: Path) -> None:
        # Arrange
        (tmp_path / "config.json").write_text(json.dumps({"model_type": "qwen2"}))

        # Act
        model_type = manifest_model_type(
            {"network": {"pretrained_model_name_or_path": str(tmp_path)}}
        )

        # Assert
        assert model_type == "qwen2"

    def test_an_unknown_hub_id_without_a_local_config_has_none(self) -> None:
        assert (
            manifest_model_type(
                {"network": {"pretrained_model_name_or_path": "acme/not-a-model"}}
            )
            is None
        )


class TestContextMinimum:
    def test_settings_without_a_context_length_have_no_minimum(self) -> None:
        assert context_minimum({"environment": {}}, BundledRunSettings()) == (None, "")


class TestFillAndExpand:
    def test_fills_settings_then_expands_and_sets_segment_tokens(self) -> None:
        # Arrange
        manifest: dict[str, Any] = {
            "algorithm": {"name": "GRPO"},
            "environment": {"name": "VisualWebArena", "env_type": "rollout"},
            "network": {"pretrained_model_name_or_path": QWEN},
        }

        # Act
        fill_and_expand(manifest, lambda env: {**env, "env_image": "vwa:1"})

        # Assert
        assert manifest["environment"]["env_image"] == "vwa:1"
        assert manifest["environment"]["segment_prompt_tokens"] == (
            BUNDLED_RUN_SETTINGS["VisualWebArena"].segment_prompt_tokens
        )
        assert manifest["algorithm"]["max_output_tokens"] == 5120

    def test_leaves_a_manifest_without_an_environment_section(self) -> None:
        manifest: dict[str, Any] = {"algorithm": {"name": "GRPO"}}

        fill_and_expand(manifest, lambda env: {**env, "env_image": "x"})

        assert manifest == {"algorithm": {"name": "GRPO"}}

    def test_leaves_an_unchanged_environment_in_place(self) -> None:
        environment = {"name": "Mine", "env_type": "rollout", "env_url": "http://env"}
        manifest: dict[str, Any] = {"environment": environment}

        fill_and_expand(manifest, dict)

        assert manifest["environment"] is environment
