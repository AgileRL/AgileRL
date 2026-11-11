# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""What the OPD spec and its self-distillation teacher accept and reject."""

from __future__ import annotations

import copy

import pytest
from pydantic import ValidationError

from agilerl.arena.models import GRPOSpec, OPDSpec, SFTSpec, TrainingManifest
from agilerl.arena.models.algorithms.opd import (
    distillation_teacher,
    teacher_engine_gpus,
)
from agilerl.arena.models.distillation import (
    DistillationTeacherConfig,
    DistillationTeacherModel,
)
from agilerl.arena.models.networks import VLLMConfig

TEACHER_OPD = {
    "algorithm": {
        "name": "OPD",
        "batch_size": 2,
        "teacher": {"context_key": "solution"},
    },
    "environment": {
        "env_type": "rollout",
        "dataset": "openai/gsm8k",
        "reward_file_path": "reward.py",
        "prompt_template": {"user_0": "{question}"},
    },
    "network": {
        "pretrained_model_name_or_path": "Qwen/Qwen2.5-0.5B-Instruct",
        "max_context_length": 512,
    },
    "training": {"max_steps": 100},
}


def teacher_manifest(**sections: dict) -> dict:
    doc = copy.deepcopy(TEACHER_OPD)
    for section, values in sections.items():
        doc[section] = values
    return doc


class TestDistillationTeacherConfig:
    def test_defaults_read_the_reference_trace_key(self) -> None:
        config = DistillationTeacherConfig()

        assert config.context_key == "reference_trace"
        assert "{reference}" in config.template
        assert config.coef == 1.0

    def test_self_distillation_reserves_the_reference_in_the_rollout_engine(
        self,
    ) -> None:
        config = DistillationTeacherConfig(max_context_tokens=256)

        assert config.context_tokens == 256
        assert config.rollout_engine_context_tokens == 256

    def test_a_teacher_model_takes_no_reference_unless_named(self) -> None:
        model = DistillationTeacherModel(pretrained_model_name_or_path="big/model")

        config = DistillationTeacherConfig(model=model)

        assert config.context_key is None
        assert config.context_tokens == 0
        assert model.gpus == 1

    def test_a_teacher_model_with_a_reference_reserves_it_on_its_own_engine(
        self,
    ) -> None:
        model = DistillationTeacherModel(
            pretrained_model_name_or_path="big/model",
            vllm_config=VLLMConfig(tensor_parallel_size=4),
        )

        config = DistillationTeacherConfig(
            model=model, context_key="trace", max_context_tokens=256
        )

        assert config.context_tokens == 256
        assert config.rollout_engine_context_tokens == 0
        assert model.gpus == 4

    def test_a_teacher_adapter_scores_in_the_rollout_engine(self) -> None:
        config = DistillationTeacherConfig(
            adapter="s3://bucket/grpo/checkpoint", context_key="trace"
        )

        assert config.rollout_engine_context_tokens == config.max_context_tokens

    def test_a_teacher_adapter_takes_no_reference_unless_named(self) -> None:
        config = DistillationTeacherConfig(adapter="/ckpt")

        assert config.context_key is None
        assert config.rollout_engine_context_tokens == 0

    def test_a_base_teacher_scores_on_base_weights_without_a_reference(self) -> None:
        config = DistillationTeacherConfig(base=True)

        assert config.context_key is None
        assert config.scores_base_weights is True
        assert config.is_live_policy is False

    def test_self_distillation_scores_under_the_live_adapter(self) -> None:
        config = DistillationTeacherConfig()

        assert config.is_live_policy is True
        assert config.scores_base_weights is False

    def test_rejects_both_a_base_teacher_and_an_adapter(self) -> None:
        with pytest.raises(ValidationError, match=r"got adapter, base"):
            DistillationTeacherConfig(adapter="/ckpt", base=True)

    def test_rejects_both_a_teacher_model_and_adapter(self) -> None:
        with pytest.raises(ValidationError, match=r"got model, adapter"):
            DistillationTeacherConfig(
                adapter="/ckpt",
                model=DistillationTeacherModel(
                    pretrained_model_name_or_path="big/model"
                ),
            )

    def test_rejects_a_template_without_the_reference(self) -> None:
        with pytest.raises(ValidationError, match=r"must contain '\{reference\}'"):
            DistillationTeacherConfig(template="Solve it.")

    def test_rejects_a_non_positive_weight(self) -> None:
        with pytest.raises(ValidationError, match="coef"):
            DistillationTeacherConfig(coef=0.0)


class TestOPDSpec:
    def test_defaults_sample_the_unmodified_policy(self) -> None:
        spec = OPDSpec()

        assert (spec.temperature, spec.top_k, spec.beta, spec.group_size) == (
            1.0,
            0,
            0.0,
            4,
        )
        assert distillation_teacher(spec) == DistillationTeacherConfig()

    @pytest.mark.parametrize(
        ("setting", "reported"),
        [
            ({"temperature": 0.9}, "temperature=0.9"),
            ({"top_k": 50}, "top_k=50"),
            ({"top_p": 0.95, "top_k": 20}, "top_p=0.95"),
            ({"repetition_penalty": 1.1}, "repetition_penalty=1.1"),
            ({"whiten_advantages": True}, "whiten_advantages=True"),
            ({"filter_zero_adv": True}, "filter_zero_adv=True"),
            (
                {"constrain_answer_pattern": r"\d+", "top_k": 20},
                "constrain_answer_pattern is set",
            ),
        ],
    )
    def test_rejects_settings_that_change_the_scored_distribution(
        self, setting: dict, reported: str
    ) -> None:
        with pytest.raises(ValidationError, match=reported):
            OPDSpec(**setting)

    def test_other_algorithms_have_no_teacher(self) -> None:
        assert distillation_teacher(GRPOSpec()) is None
        assert distillation_teacher(SFTSpec()) is None
        assert "teacher" not in GRPOSpec.model_fields


class TestTrainingManifestTeacher:
    def test_dataset_backed_rollout_validates(self) -> None:
        validated = TrainingManifest.model_validate(TEACHER_OPD)

        assert isinstance(validated.algorithm, OPDSpec)
        assert validated.algorithm.teacher.context_key == "solution"

    def test_env_of_unknown_modality_is_rejected(self) -> None:
        doc = teacher_manifest(
            environment={
                "env_type": "rollout",
                "entrypoint": "tasks:SupportEnv",
                "max_turns": 4,
            }
        )

        with pytest.raises(ValidationError, match="text-only LLM rollout env"):
            TrainingManifest.model_validate(doc)

    def test_text_env_validates(self) -> None:
        doc = teacher_manifest(
            environment={
                "env_type": "rollout",
                "entrypoint": "tasks:SupportEnv",
                "max_turns": 4,
                "modality": "text",
            }
        )

        assert TrainingManifest.model_validate(doc).environment.text_only

    @staticmethod
    def _teacher_model_doc(teacher_id: str = "Qwen/Qwen3-4B", **training: int) -> dict:
        return teacher_manifest(
            algorithm={
                "name": "OPD",
                "batch_size": 2,
                "vllm_config": {"tensor_parallel_size": 1},
                "teacher": {"model": {"pretrained_model_name_or_path": teacher_id}},
            },
            training={
                "max_steps": 100,
                **(training or {"rollout_engines_per_agent": 1}),
            },
        )

    def test_a_teacher_model_with_a_written_layout_validates(self) -> None:
        validated = TrainingManifest.model_validate(self._teacher_model_doc())

        assert teacher_engine_gpus(validated.algorithm) == 1
        assert validated.algorithm.teacher.context_key is None

    def test_a_teacher_model_survives_the_run_manifest_round_trip(self) -> None:
        validated = TrainingManifest.model_validate(self._teacher_model_doc())

        again = TrainingManifest.model_validate(validated.to_payload())

        assert again.algorithm.teacher.model == validated.algorithm.teacher.model

    def test_a_teacher_outside_the_catalog_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="not in the model catalog"):
            TrainingManifest.model_validate(self._teacher_model_doc("acme/unknown"))

    def test_a_teacher_with_another_vocabulary_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="need one vocabulary"):
            TrainingManifest.model_validate(self._teacher_model_doc("Qwen/Qwen3.8-27B"))

    def test_a_teacher_model_needs_its_gpu_layout_written_out(self) -> None:
        doc = self._teacher_model_doc()
        doc["training"].pop("rollout_engines_per_agent")

        with pytest.raises(ValidationError, match="GPU layout written out"):
            TrainingManifest.model_validate(doc)

    def test_a_teacher_model_needs_the_student_width_written_out(self) -> None:
        doc = self._teacher_model_doc()
        del doc["algorithm"]["vllm_config"]

        with pytest.raises(ValidationError, match="GPU layout written out"):
            TrainingManifest.model_validate(doc)

    def test_async_rollout_with_the_default_buffer_validates(self) -> None:
        doc = teacher_manifest(
            training={
                "max_steps": 100,
                "rollout_mode": "async",
                "rollout_engines_per_agent": 1,
            },
            replay_buffer={"kind": "llm"},
        )

        validated = TrainingManifest.model_validate(doc)

        assert validated.replay_buffer.group_replay_max_age > 0
