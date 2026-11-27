# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Plain refusals for Snowflake app manifests that fail validation."""

from __future__ import annotations

import copy
from collections.abc import Callable
from pathlib import Path

import pytest
import yaml

from agilerl.snowflake.manifest_refusal import (
    app_algorithms,
    field_path,
    load_app_manifest,
)

GRPO: dict[str, object] = {
    "algorithm": {"name": "GRPO", "group_size": 4, "batch_size": 2},
    "environment": {
        "env_type": "rollout",
        "dataset": "rows",
        "prompt_template": {"user_0": "{question}"},
        "reward": {"type": "exact_match"},
    },
    "network": {
        "pretrained_model_name_or_path": "Qwen/Qwen3.5-0.8B",
        "max_context_length": 128,
    },
    "training": {"max_steps": 2},
}
ALGORITHMS = "CISPO, DPO, GRPO, GSPO, LLMPPO, LLMREINFORCE, OPD, SFT"
PYDANTIC_TEXT = ("validation error", "Input should be", "pydantic", "[type=")


def refusal(tmp_path: Path, change: Callable[[dict], object]) -> str:
    document = copy.deepcopy(GRPO)
    change(document)
    path = tmp_path / "manifest.yaml"
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    with pytest.raises(ValueError, match="Nothing was started") as exc:
        load_app_manifest(path)
    # Snowflake prints a chained cause, which would carry the pydantic text.
    assert exc.value.__cause__ is None
    assert exc.value.__context__ is None or exc.value.__suppress_context__
    return str(exc.value)


class TestLoadAppManifest:
    def test_accepts_a_valid_manifest(self, tmp_path: Path) -> None:
        path = tmp_path / "manifest.yaml"
        path.write_text(yaml.safe_dump(GRPO), encoding="utf-8")

        manifest = load_app_manifest(path)

        assert manifest.algorithm.name == "GRPO"

    @pytest.mark.parametrize(
        ("change", "expected"),
        [
            (
                lambda d: d["algorithm"].update(name="GRPOO"),
                (
                    "algorithm.name is 'GRPOO', which the app does not train. Did you "
                    f"mean GRPO? Set it to one of: {ALGORITHMS}.\nNothing was started."
                ),
            ),
            (
                lambda d: d["algorithm"].pop("name"),
                (
                    f"algorithm.name is missing. Set it to one of: {ALGORITHMS}.\n"
                    "Nothing was started."
                ),
            ),
            (
                lambda d: d["training"].update(max_steps="lots"),
                (
                    "The manifest is not valid.\n"
                    "training.max_steps is 'lots'. Set it to a whole number.\n"
                    "Nothing was started."
                ),
            ),
            (
                lambda d: d["training"].update(max_steps=-1),
                (
                    "The manifest is not valid.\n"
                    "training.max_steps is -1. It should be greater than or equal to 1.\n"
                    "Nothing was started."
                ),
            ),
            (
                lambda d: d["network"].pop("pretrained_model_name_or_path"),
                (
                    "The manifest is not valid.\n"
                    "network.pretrained_model_name_or_path is missing. Add it to the "
                    "manifest.\nNothing was started."
                ),
            ),
            (
                lambda d: d["environment"].update(reward={"type": "llm_judge"}),
                (
                    "The manifest is not valid.\n"
                    "environment.reward.type is 'llm_judge', which is not one of the "
                    "options. Set it to one of: 'exact_match', 'numeric', 'regex', "
                    "'json_schema', 'tool_call', 'snowflake_function'.\n"
                    "Nothing was started."
                ),
            ),
            (
                lambda d: d["environment"].update(reward={"type": "regex"}),
                (
                    "The manifest is not valid.\n"
                    "environment.reward.pattern is missing. Add it to the manifest.\n"
                    "Nothing was started."
                ),
            ),
            (
                lambda d: d["environment"].update(reward="exact_match"),
                (
                    "The manifest is not valid.\n"
                    "environment.reward is 'exact_match'. Set it to a mapping of fields.\n"
                    "Nothing was started."
                ),
            ),
            (
                lambda d: d["training"].update(max_stepz=3),
                (
                    "The manifest is not valid.\n"
                    "training.max_stepz is not a manifest field. Remove it or fix its "
                    "spelling.\nNothing was started."
                ),
            ),
        ],
        ids=[
            "unknown-algorithm",
            "missing-algorithm-name",
            "wrong-type",
            "out-of-range",
            "missing-required-field",
            "bad-reward-type",
            "reward-missing-field",
            "reward-not-a-mapping",
            "unknown-field",
        ],
    )
    def test_names_the_field_the_problem_and_the_fix(
        self, tmp_path: Path, change: Callable[[dict], object], expected: str
    ) -> None:
        message = refusal(tmp_path, change)

        assert message == expected
        assert not any(text in message for text in PYDANTIC_TEXT)

    def test_a_classic_algorithm_lists_only_what_the_app_trains(
        self, tmp_path: Path
    ) -> None:
        message = refusal(tmp_path, lambda d: d["algorithm"].update(name="DQN"))

        assert message.startswith("algorithm.name is 'DQN', which the app does not")
        assert f"Set it to one of: {ALGORITHMS}." in message
        assert "PPO," not in message.replace("LLMPPO", "")

    def test_a_manifest_wide_check_keeps_its_own_sentence(self, tmp_path: Path) -> None:
        message = refusal(tmp_path, lambda d: d.pop("network"))

        assert message.splitlines()[1].startswith(
            "Required field 'pretrained_model_name_or_path' wasn't found"
        )
        assert "Value error" not in message

    @pytest.mark.parametrize(
        ("value", "shown_as"), [([1], "a list"), ({"n": 1}, "a mapping")]
    )
    def test_names_the_kind_of_a_list_or_mapping_value(
        self, tmp_path: Path, value: object, shown_as: str
    ) -> None:
        message = refusal(tmp_path, lambda d: d["training"].update(max_steps=value))

        assert message.splitlines()[1] == (
            f"training.max_steps is {shown_as}. Set it to a whole number."
        )

    def test_an_algorithm_that_is_not_a_section_is_left_to_validation(
        self, tmp_path: Path
    ) -> None:
        message = refusal(tmp_path, lambda d: d.update(algorithm="GRPO"))

        assert message.startswith("The manifest is not valid.\n")
        assert message.endswith("Nothing was started.")
        assert not any(text in message for text in PYDANTIC_TEXT)


class TestFieldPath:
    def test_drops_union_branch_names_that_are_not_manifest_keys(self) -> None:
        document = {"environment": {"reward": {"type": "regex"}}}

        path = field_path(
            ("environment", "rollout", "reward", "regex", "pattern"), document
        )

        assert path == ("environment.reward.pattern", "environment.rollout")

    def test_indexes_list_entries(self) -> None:
        document = {"training": {"stages": [{"a": 1}, {"b": 2}]}}

        assert field_path(("training", "stages", 1, "b"), document) == (
            "training.stages[1].b",
            None,
        )


def test_app_algorithms_are_the_llm_algorithms() -> None:
    assert ", ".join(app_algorithms()) == ALGORITHMS
