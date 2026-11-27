# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Tests for the manifest field visibility marker."""

from __future__ import annotations

import copy
from typing import Any

import pytest
from pydantic import BaseModel, Field, model_validator
from typing_extensions import Self

from agilerl.models.manifest import TrainingManifest
from agilerl.models.visibility import (
    HIDDEN,
    MISSING,
    arena_chosen_paths,
    marked_fields,
    path_value,
    user_view,
)

GRPO: dict[str, Any] = {
    "algorithm": {"name": "GRPO", "group_size": 4, "batch_size": 2},
    "environment": {
        "env_type": "rollout",
        "dataset": "openai/gsm8k",
        "reward_file_path": "reward.py",
        "prompt_template": {"user_0": "{question}"},
    },
    "network": {
        "pretrained_model_name_or_path": "Qwen/Qwen3.5-0.8B",
        "max_context_length": 1024,
    },
    "training": {"max_steps": 100},
}


def grpo(**sections: dict[str, Any]) -> dict[str, Any]:
    """The GRPO document with each section in *sections* merged in."""
    document = copy.deepcopy(GRPO)
    for name, values in sections.items():
        document.setdefault(name, {}).update(values)
    return document


def chosen(document: dict[str, Any]) -> list[str]:
    manifest = TrainingManifest.model_validate(document)
    return arena_chosen_paths(TrainingManifest, document, manifest)


class TestMarkedFields:
    def test_tiers_by_payload_path(self) -> None:
        # Act
        tiers = {field.path: field.tier for field in marked_fields(TrainingManifest)}

        # Assert
        assert tiers["algorithm.batch_size"] == "auto"
        assert tiers["algorithm.vllm_config.gpu_memory_utilization"] == "auto"
        assert tiers["network.lora_config.task_type"] == "hidden"
        assert tiers["environment.cpus_per_env_host"] == "advanced"
        assert tiers["environment.env_host_resource"] == "advanced"
        assert tiers["training.pop_size"] == "hidden"
        assert tiers["mutation"] == "hidden"
        assert tiers["tournament_selection"] == "hidden"
        assert "environment.env_vars" not in tiers
        assert "network.lora_config.lora_r" not in tiers

    def test_platform_written_fields_are_marked_on_the_field(self) -> None:
        # Act
        written = {
            field.path: field.tier
            for field in marked_fields(TrainingManifest)
            if field.platform_written
        }

        # Assert
        assert written == {
            "environment.env_hosts": "advanced",
            "environment.env_port": "advanced",
            "environment.cpus_per_env_host": "advanced",
            "environment.env_host_memory_bytes": "advanced",
            "network.lora_config.target_parameters": "hidden",
            "algorithm.lora_config.target_parameters": "hidden",
            "mutation": "hidden",
            "tournament_selection": "hidden",
        }

    def test_keys_carry_every_spelling(self) -> None:
        # Act
        fields = {field.path: field for field in marked_fields(TrainingManifest)}

        # Assert
        assert fields["training.pop_size"].keys == (
            ("training",),
            ("pop_size", "population_size"),
        )
        assert fields["tournament_selection"].keys == (
            ("tournament_selection", "selection_strategy"),
        )


class TestUserView:
    def test_drops_marked_fields_and_the_blocks_they_empty(self) -> None:
        # Arrange
        payload = {
            "algorithm": {
                "name": "GRPO",
                "batch_size": 16,
                "vllm_config": {"gpu_memory_utilization": 0.9},
            },
            "training": {"pop_size": 1, "max_steps": 10},
            "mutation": {"rl_hp_selection": {}},
        }

        # Act
        view = user_view(payload, marked_fields(TrainingManifest))

        # Assert
        assert view == {"algorithm": {"name": "GRPO"}, "training": {"max_steps": 10}}
        assert payload["training"]["pop_size"] == 1

    def test_one_section_under_its_key(self) -> None:
        # Arrange
        section = {"lora_config": {"lora_r": 8, "task_type": "CAUSAL_LM"}}

        # Act
        view = user_view({"network": section}, marked_fields(TrainingManifest))

        # Assert
        assert view == {"network": {"lora_config": {"lora_r": 8}}}


class TestArenaChosenPaths:
    def test_payload_round_trips(self) -> None:
        # Arrange
        payload = TrainingManifest.model_validate(GRPO).to_payload()

        # Act
        paths = chosen(payload)

        # Assert
        assert paths == []

    def test_reports_a_user_set_env_host_field_as_advanced(self) -> None:
        # Arrange
        environment = {
            "env_type": "rollout",
            "entrypoint": "tasks:Env",
            "env_host_resource": "pool=cpu",
        }
        document = grpo()
        document["environment"] = environment

        manifest = TrainingManifest.model_validate(document)

        # Act
        hidden = chosen(document)
        advanced = arena_chosen_paths(
            TrainingManifest, document, manifest, tier="advanced"
        )

        # Assert
        assert hidden == []
        assert advanced == ["environment.env_host_resource"]

    @pytest.mark.parametrize(
        ("population_size", "expected"),
        [(1, []), (2, ["training.pop_size"])],
    )
    def test_reads_the_alias_spelling(
        self, population_size: int, expected: list[str]
    ) -> None:
        # Act
        paths = chosen(grpo(training={"population_size": population_size}))

        # Assert
        assert paths == expected

    def test_accepts_arenas_own_value_beside_other_keys_in_the_block(self) -> None:
        # Arrange
        lora_config = {"task_type": "CAUSAL_LM", "lora_r": 8}
        network = {**GRPO["network"], "lora_config": lora_config}

        # Act
        paths = chosen(grpo(network=network))

        # Assert
        assert paths == []

    def test_accepts_a_user_set_auto_field(self) -> None:
        # Arrange
        algorithm = {"batch_size": 4, "vllm_config": {"gpu_memory_utilization": 0.5}}

        # Act
        paths = chosen(grpo(algorithm=algorithm))

        # Assert
        assert paths == []

    def test_reports_a_nested_lora_field(self) -> None:
        # Arrange
        network = {**GRPO["network"], "lora_config": {"task_type": "SEQ_CLS"}}

        # Act
        paths = chosen(grpo(network=network))

        # Assert
        assert paths == ["network.lora_config.task_type"]

    def test_skips_platform_stamped_fields(self) -> None:
        # Arrange
        document = grpo(
            environment={"cpus_per_env_host": 2, "env_host_memory_bytes": 1024},
            mutation={"name": "Mutations", "rl_hp_selection": {}},
        )

        # Act
        paths = chosen(document)

        # Assert
        assert paths == []

    def test_skips_an_env_host_count_a_stored_run_carries(self) -> None:
        # Arrange
        document = grpo()
        document["environment"] = {
            "env_type": "rollout",
            "entrypoint": "tasks:Env",
            "env_hosts": 1,
        }

        # Act
        paths = chosen(document)

        # Assert
        assert paths == []


class Derived(BaseModel):
    """Two hidden fields; y follows x when y is unset."""

    x: int = Field(default=1, json_schema_extra=HIDDEN)
    y: int | None = Field(default=None, json_schema_extra=HIDDEN)

    @model_validator(mode="after")
    def _y_follows_x(self) -> Self:
        if self.y is None:
            self.y = self.x
        return self


class Inner(BaseModel):
    a: int = Field(default=1, json_schema_extra=HIDDEN)
    b: int = 1


class Outer(BaseModel):
    """A block Arena fills with a = 7 when the document leaves it out."""

    block: Inner | None = None

    @model_validator(mode="after")
    def _default_block(self) -> Self:
        if self.block is None:
            self.block = Inner(a=7)
        return self


class TestPathValue:
    def test_a_missing_key_is_missing(self) -> None:
        assert path_value({"a": {"b": 1}}, "a.c") is MISSING

    def test_a_present_key_is_its_value(self) -> None:
        assert path_value({"a": {"b": 1}}, "a.b") == 1


class TestArenaChosenPathsResolution:
    def test_a_value_arena_resolves_from_another_set_field_is_not_reported(
        self,
    ) -> None:
        # Arrange
        document = {"x": 5, "y": 5}

        # Act
        paths = arena_chosen_paths(Derived, document, Derived.model_validate(document))

        # Assert
        assert paths == ["x"]

    def test_a_value_arena_resolves_for_the_whole_block_is_not_reported(
        self,
    ) -> None:
        # Arrange
        document = {"block": {"a": 7, "b": 3}}

        # Act
        paths = arena_chosen_paths(Outer, document, Outer.model_validate(document))

        # Assert
        assert paths == []
