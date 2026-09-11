# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Contract tests: what a manifest accepts, rejects, and resolves to."""

from __future__ import annotations

import copy
import json
import subprocess
import sys

import pytest
from jsonschema import Draft202012Validator
from pydantic import ValidationError

from agilerl.arena.models import (
    MANIFEST_REGISTRY,
    GRPOSpec,
    GymEnvSpec,
    LLMEnvSpec,
    LLMRolloutBufferSpec,
    MultiFrequencySelectionSpec,
    MutationSpec,
    ReplayBufferSpec,
    TrainingManifest,
    __version__,
    manifest_schema,
)

DQN = {
    "algorithm": {"name": "DQN", "lr": 0.001},
    "environment": {"name": "LunarLander-v3", "num_envs": 8},
    "network": {
        "latent_dim": 64,
        "encoder_config": {"hidden_size": [64]},
        "head_config": {"hidden_size": [64]},
    },
    "training": {"max_steps": 1000, "pop_size": 2},
}

GRPO = {
    "algorithm": {"name": "GRPO", "group_size": 4, "batch_size": 2},
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


def manifest(base: dict, **overrides: dict) -> dict:
    """Deep-copy *base* and merge a per-section override into it."""
    out = copy.deepcopy(base)
    for section, values in overrides.items():
        out.setdefault(section, {}).update(values)
    return out


class TestAcceptance:
    def test_rl_manifest_validates(self) -> None:
        assert TrainingManifest.model_validate(DQN).algorithm.name == "DQN"

    def test_llm_manifest_validates(self) -> None:
        validated = TrainingManifest.model_validate(GRPO)
        assert isinstance(validated.algorithm, GRPOSpec)
        assert isinstance(validated.environment, LLMEnvSpec)

    def test_every_registered_algorithm_is_reachable_by_name(self) -> None:
        for name in MANIFEST_REGISTRY.names():
            assert MANIFEST_REGISTRY.get(name).__name__.endswith("Spec")

    def test_registry_aliases_algorithm_names(self) -> None:
        assert MANIFEST_REGISTRY.get("Rainbow DQN") is MANIFEST_REGISTRY.get(
            "RainbowDQN"
        )
        assert MANIFEST_REGISTRY.get("Recurrent PPO") is MANIFEST_REGISTRY.get("PPO")
        assert MANIFEST_REGISTRY.get("RecurrentPPO") is MANIFEST_REGISTRY.get("PPO")

    @pytest.mark.parametrize("name", ["LLMPPO", "LLMREINFORCE"])
    def test_llm_policy_gradient_manifests_take_advantage_granularity(
        self, name: str
    ) -> None:
        # The constructors accept the field; the manifest must be able to set it.
        doc = manifest(GRPO, algorithm={"advantage_granularity": "token"})
        doc["algorithm"] = {
            k: v for k, v in doc["algorithm"].items() if k != "group_size"
        }
        doc["algorithm"]["name"] = name
        validated = TrainingManifest.model_validate(doc)
        assert validated.algorithm.advantage_granularity == "token"


class TestUnknownKeys:
    def test_unknown_algorithm_key_is_an_error(self) -> None:
        with pytest.raises(ValidationError, match="not permitted"):
            TrainingManifest.model_validate(
                manifest(DQN, algorithm={"definitely_not_a_field": 1})
            )

    def test_unknown_training_key_is_an_error(self) -> None:
        with pytest.raises(ValidationError, match="not permitted"):
            TrainingManifest.model_validate(manifest(DQN, training={"nope": 1}))

    def test_unknown_top_level_section_is_an_error(self) -> None:
        with pytest.raises(ValidationError, match="not permitted"):
            TrainingManifest.model_validate({**DQN, "nope": {}})

    def test_unknown_algorithm_without_env_type_is_a_validation_error(self) -> None:
        # env_type is inferred from the algorithm, so a misspelled name reaches
        # the registry from the inference path first.
        doc = copy.deepcopy(GRPO)
        doc["algorithm"] = {"name": "GRPOO"}
        doc["environment"] = {"dataset": "openai/gsm8k"}
        with pytest.raises(ValidationError, match="not a registered algorithm"):
            TrainingManifest.model_validate(doc)

    def test_unknown_environment_key_is_an_error(self) -> None:
        with pytest.raises(ValidationError, match="not permitted"):
            TrainingManifest.model_validate(
                manifest(DQN, environment={"use_vectorized_env": True})
            )

    def test_unknown_encoder_config_key_is_an_error(self) -> None:
        with pytest.raises(ValidationError, match="not permitted"):
            TrainingManifest.model_validate(
                manifest(
                    DQN,
                    network={
                        "arch": "mlp",
                        "encoder_config": {
                            "hidden_size": [64],
                            "not_a_field": 1,
                        },
                        "head_config": {"hidden_size": [64]},
                    },
                )
            )

    def test_peft_lora_r_is_an_error(self) -> None:
        with pytest.raises(ValidationError, match="not permitted"):
            TrainingManifest.model_validate(
                manifest(GRPO, network={"lora_config": {"r": 8}})
            )


class TestMalformedSections:
    # Shape errors must surface as ValidationError like any other bad input.
    # A raw TypeError would escape verdict() and crash the CLI / submit path.

    def test_non_mapping_algorithm_section_is_a_validation_error(self) -> None:
        with pytest.raises(ValidationError, match="must be a mapping"):
            TrainingManifest.model_validate({**copy.deepcopy(DQN), "algorithm": "DQN"})

    def test_wrong_typed_clip_coef_is_a_validation_error(self) -> None:
        with pytest.raises(ValidationError, match="float or a list/tuple"):
            TrainingManifest.model_validate(
                manifest(GRPO, algorithm={"clip_coef": "abc"})
            )

    def test_non_numeric_clip_coef_pair_is_a_validation_error(self) -> None:
        with pytest.raises(ValidationError, match="entries must be floats"):
            TrainingManifest.model_validate(
                manifest(GRPO, algorithm={"clip_coef": [None, 0.2]})
            )

    def test_a_field_named_self_is_a_validation_error(self) -> None:
        # `self` must not collide with the registry's create() parameters.
        with pytest.raises(ValidationError, match="not permitted"):
            TrainingManifest.model_validate(manifest(DQN, algorithm={"self": 1}))

    def test_hpo_range_with_min_above_max_is_a_validation_error(self) -> None:
        with pytest.raises(ValidationError, match="less than or equal to max"):
            TrainingManifest.model_validate(
                manifest(
                    DQN,
                    mutation={"rl_hp_selection": {"lr": {"min": 1.0, "max": 0.0001}}},
                )
            )


class TestResolution:
    def test_evo_steps_defaults_to_the_algorithms_cadence(self) -> None:
        validated = TrainingManifest.model_validate(GRPO)
        assert validated.training.evo_steps == GRPOSpec.default_evo_steps

    def test_evo_steps_is_never_longer_than_the_run(self) -> None:
        validated = TrainingManifest.model_validate(
            manifest(DQN, training={"max_steps": 100})
        )
        assert validated.training.evo_steps == 100

    def test_explicit_evo_steps_wins(self) -> None:
        validated = TrainingManifest.model_validate(
            manifest(DQN, training={"evo_steps": 250})
        )
        assert validated.training.evo_steps == 250

    def test_env_width_reaches_the_algorithm(self) -> None:
        ddpg = copy.deepcopy(DQN)
        ddpg["algorithm"] = {"name": "DDPG", "lr_actor": 0.001}
        ddpg["environment"]["num_envs"] = 12
        validated = TrainingManifest.model_validate(ddpg)
        assert validated.algorithm.vect_noise_dim == 12

    def test_network_is_lifted_onto_the_llm_algorithm(self) -> None:
        validated = TrainingManifest.model_validate(GRPO)
        assert (
            validated.algorithm.pretrained_model_name_or_path
            == "Qwen/Qwen2.5-0.5B-Instruct"
        )
        assert validated.algorithm.max_model_len == 512

    def test_colocated_rollout_turns_the_trainers_vllm_on(self) -> None:
        validated = TrainingManifest.model_validate(GRPO)
        assert validated.algorithm.use_vllm is True
        assert validated.algorithm.vllm_config is not None
        assert validated.algorithm.vllm_config.sleep_mode is True
        assert validated.algorithm.vllm_config.max_num_seqs == 16
        assert validated.algorithm.vllm_config.gpu_memory_utilization == 0.9

    def test_async_rollout_turns_the_trainers_vllm_off(self) -> None:
        validated = TrainingManifest.model_validate(
            manifest(
                GRPO,
                environment={
                    "env_type": "rollout",
                    "entrypoint": "gem:make",
                    "env_config": {"env_id": "game:Sudoku-v0"},
                    "dataset": None,
                    "reward_file_path": None,
                    "prompt_template": None,
                },
                training={
                    "rollout_mode": "async",
                    "rollout_engines_per_agent": 1,
                },
                replay_buffer={"kind": "llm"},
            )
        )
        assert validated.algorithm.use_vllm is False

    def test_explicit_use_vllm_false_with_colocated_is_an_error(self) -> None:
        with pytest.raises(ValidationError, match="use_vllm"):
            TrainingManifest.model_validate(
                manifest(
                    GRPO,
                    algorithm={"use_vllm": False},
                    training={"rollout_mode": "colocated"},
                )
            )

    def test_explicit_use_vllm_false_without_rollout_mode_stays_false(self) -> None:
        validated = TrainingManifest.model_validate(
            manifest(GRPO, algorithm={"use_vllm": False})
        )
        assert validated.algorithm.use_vllm is False

    def test_explicit_use_vllm_false_without_rollout_mode_round_trips(self) -> None:
        payload = TrainingManifest.model_validate(
            manifest(GRPO, algorithm={"use_vllm": False})
        ).to_payload()
        assert payload["algorithm"]["use_vllm"] is False
        assert "rollout_mode" not in payload["training"]
        reloaded = TrainingManifest.model_validate(payload)
        assert reloaded.algorithm.use_vllm is False

    def test_explicit_use_vllm_true_with_async_is_an_error(self) -> None:
        with pytest.raises(ValidationError, match="use_vllm"):
            TrainingManifest.model_validate(
                manifest(
                    GRPO,
                    algorithm={
                        "use_vllm": True,
                        "vllm_config": {"max_num_seqs": 4},
                    },
                    environment={
                        "env_type": "rollout",
                        "entrypoint": "gem:make",
                        "env_config": {"env_id": "game:Sudoku-v0"},
                        "dataset": None,
                        "reward_file_path": None,
                        "prompt_template": None,
                    },
                    training={
                        "rollout_mode": "async",
                        "rollout_engines_per_agent": 1,
                    },
                    replay_buffer={"kind": "llm"},
                )
            )

    def test_gym_num_envs_default_does_not_overwrite_ppo(self) -> None:
        validated = TrainingManifest.model_validate(
            {
                "algorithm": {"name": "PPO"},
                "environment": {"name": "CartPole-v1"},
                "training": {"max_steps": 100, "pop_size": 2},
            }
        )
        assert validated.algorithm.num_envs == 1

    def test_explicit_env_num_envs_fills_unset_ppo_num_envs(self) -> None:
        validated = TrainingManifest.model_validate(
            {
                "algorithm": {"name": "PPO"},
                "environment": {"name": "CartPole-v1", "num_envs": 8},
                "training": {"max_steps": 100, "pop_size": 2},
            }
        )
        assert validated.algorithm.num_envs == 8

    def test_disagreeing_num_envs_is_an_error(self) -> None:
        with pytest.raises(ValidationError, match="num_envs"):
            TrainingManifest.model_validate(
                {
                    "algorithm": {"name": "PPO", "num_envs": 4},
                    "environment": {"name": "CartPole-v1", "num_envs": 16},
                    "training": {"max_steps": 100, "pop_size": 2},
                }
            )

    def test_unset_ppo_num_envs_is_omitted_from_the_payload(self) -> None:
        payload = TrainingManifest.model_validate(
            {
                "algorithm": {"name": "PPO"},
                "environment": {"name": "CartPole-v1"},
                "training": {"max_steps": 100, "pop_size": 2},
            }
        ).to_payload()
        assert "num_envs" not in payload["algorithm"]
        assert payload["environment"]["num_envs"] == 16
        reloaded = TrainingManifest.model_validate(payload)
        assert reloaded.algorithm.num_envs == 16

    def test_unset_vect_noise_dim_is_omitted_from_the_payload(self) -> None:
        payload = TrainingManifest.model_validate(
            {
                "algorithm": {"name": "DDPG"},
                "environment": {"name": "Pendulum-v1"},
                "training": {"max_steps": 100, "pop_size": 2},
            }
        ).to_payload()
        assert "vect_noise_dim" not in payload["algorithm"]
        assert payload["environment"]["num_envs"] == 16
        reloaded = TrainingManifest.model_validate(payload)
        assert reloaded.algorithm.vect_noise_dim == 16

    def test_empty_mutation_is_a_noop(self) -> None:
        payload = TrainingManifest.model_validate(
            manifest(DQN, mutation={})
        ).to_payload()
        assert payload["mutation"]["probabilities"]["arch_mut"] == 0.0
        assert payload["mutation"]["probabilities"]["no_mut"] == 1.0

    def test_dormant_threshold_is_accepted_and_omitted_from_the_payload(self) -> None:
        payload = TrainingManifest.get_validated(
            manifest(
                DQN,
                mutation={"mutation_sd": 0.2, "dormant_threshold": 0.05},
            )
        )
        assert payload["mutation"]["mutation_sd"] == 0.2
        assert "dormant_threshold" not in payload["mutation"]
        spec = TrainingManifest.model_validate(
            manifest(DQN, mutation={"dormant_threshold": 0.05})
        )
        assert spec.mutation.dormant_threshold == 0.05

    def test_omitted_mutation_is_a_noop(self) -> None:
        payload = TrainingManifest.model_validate(DQN).to_payload()
        assert payload["mutation"]["probabilities"]["arch_mut"] == 0.0

    def test_dqn_eval_steps_below_the_llm_interval_is_valid(self) -> None:
        validated = TrainingManifest.model_validate(
            manifest(DQN, training={"eval_steps": 8})
        )
        assert validated.training.eval_steps == 8

    def test_llm_model_id_on_algorithm_only_round_trips(self) -> None:
        raw = {
            "algorithm": {
                "name": "GRPO",
                "pretrained_model_name_or_path": "Qwen/Qwen2.5-0.5B-Instruct",
                "group_size": 4,
                "batch_size": 2,
            },
            "environment": {
                "env_type": "rollout",
                "dataset": "openai/gsm8k",
                "reward_file_path": "reward.py",
                "prompt_template": {"user_0": "{question}"},
            },
            "training": {"max_steps": 100},
        }
        payload = TrainingManifest.model_validate(raw).to_payload()
        assert "network" not in payload
        reloaded = TrainingManifest.model_validate(payload)
        assert (
            reloaded.algorithm.pretrained_model_name_or_path
            == "Qwen/Qwen2.5-0.5B-Instruct"
        )

    def test_mf_pbt_brackets_resolve_against_the_population(self) -> None:
        validated = TrainingManifest.model_validate(
            manifest(
                DQN,
                training={"pop_size": 8},
                selection_strategy={"strategy": "multi_frequency"},
            )
        )
        strategy = validated.selection_strategy
        assert isinstance(strategy, MultiFrequencySelectionSpec)
        assert (
            strategy.n_winners
            + strategy.n_survivors
            + strategy.n_open_for_migration
            + strategy.n_losers
            == 4
        )


class TestNothingIsStrippedInTransit:
    @pytest.mark.parametrize(
        "field",
        [
            "use_liger_loss",
            "use_separate_reference_adapter",
            "micro_batch_size_per_gpu",
        ],
    )
    def test_llm_fields_survive_a_round_trip(self, field: str) -> None:
        value = 4 if field == "micro_batch_size_per_gpu" else True
        payload = TrainingManifest.model_validate(
            manifest(GRPO, algorithm={field: value})
        ).to_payload()
        assert payload["algorithm"][field] == value

    def test_max_output_tokens_survives(self) -> None:
        payload = TrainingManifest.model_validate(
            manifest(GRPO, algorithm={"max_output_tokens": 4096})
        ).to_payload()
        assert payload["algorithm"]["max_output_tokens"] == 4096

    def test_reward_fn_name_survives(self) -> None:
        payload = TrainingManifest.model_validate(
            manifest(GRPO, environment={"reward_fn_name": "combined_rewards"})
        ).to_payload()
        assert payload["environment"]["rubric_name"] == "combined_rewards"

    def test_unknown_vllm_config_keys_are_rejected(self) -> None:
        with pytest.raises(ValidationError):
            TrainingManifest.model_validate(
                manifest(
                    GRPO,
                    algorithm={
                        "vllm_config": {
                            "max_num_seqs": 4,
                            "reasoning_parser": "x",
                        }
                    },
                )
            )

    def test_explicit_vllm_engine_args_are_kept(self) -> None:
        validated = TrainingManifest.model_validate(
            manifest(
                GRPO,
                algorithm={"vllm_engine_args": {"reasoning_parser": "x"}},
            )
        )
        assert validated.algorithm.vllm_engine_args == {"reasoning_parser": "x"}

    def test_a_payload_revalidates(self) -> None:
        payload = TrainingManifest.model_validate(GRPO).to_payload()
        assert TrainingManifest.model_validate(payload).algorithm.name == "GRPO"

    def test_explicit_false_reference_adapter_survives(self) -> None:
        payload = TrainingManifest.get_validated(
            manifest(GRPO, algorithm={"use_separate_reference_adapter": False}),
            mode="json",
        )
        assert payload["algorithm"]["use_separate_reference_adapter"] is False


class TestEnvironment:
    def test_every_environment_branch_can_be_versioned(self) -> None:
        # A custom environment's version selects which uploaded archive runs, and
        # it is not a gym-only idea.
        for section in (
            {"env_type": "rollout", "dataset": "x", "version": 3},
            {"name": "LunarLander-v3", "version": 3},
        ):
            base = GRPO if "env_type" in section else DQN
            validated = TrainingManifest.model_validate(
                manifest(base, environment=section)
            )
            assert validated.environment.version == 3


class TestCrossSectionConsistency:
    def test_llm_algorithm_requires_a_pretrained_network(self) -> None:
        broken = copy.deepcopy(GRPO)
        broken["network"] = {"latent_dim": 32}
        with pytest.raises(ValidationError, match="pretrained_model_name_or_path"):
            TrainingManifest.model_validate(broken)

    def test_rl_algorithm_rejects_a_pretrained_network(self) -> None:
        broken = copy.deepcopy(DQN)
        broken["network"] = {
            "pretrained_model_name_or_path": "Qwen/Qwen2.5-0.5B-Instruct",
            "max_context_length": 128,
        }
        with pytest.raises(ValidationError, match="builds its own networks"):
            TrainingManifest.model_validate(broken)

    def test_mutation_ranges_bound_the_network_they_describe(self) -> None:
        validated = TrainingManifest.model_validate(
            {
                **DQN,
                "mutation": {"network": {"head_net": {"max_mlp_nodes": 1024}}},
                "network": {
                    "latent_dim": 64,
                    "encoder_config": {"hidden_size": [64]},
                    "head_config": {"hidden_size": [512]},
                },
            }
        )
        assert validated.network.head_config.max_mlp_nodes == 1024
        assert validated.mutation.network is None
        assert "network" not in validated.to_payload()["mutation"]

    def test_mutation_network_is_kept_when_it_cannot_fold(self) -> None:
        payload = {key: value for key, value in DQN.items() if key != "network"}
        payload["mutation"] = {"network": {"head_net": {"max_mlp_nodes": 1024}}}

        validated = TrainingManifest.model_validate(payload)

        assert validated.network is None
        assert validated.mutation.network is not None
        assert validated.mutation.network.head_net["max_mlp_nodes"] == 1024

    def test_network_section_wins_when_mutation_ranges_disagree(self) -> None:
        validated = TrainingManifest.model_validate(
            {
                **DQN,
                "mutation": {"network": {"head_net": {"max_mlp_nodes": 500}}},
                "network": {
                    "latent_dim": 64,
                    "encoder_config": {"hidden_size": [64]},
                    "head_config": {"hidden_size": [64], "max_mlp_nodes": 128},
                },
            }
        )
        assert validated.network.head_config.max_mlp_nodes == 128
        assert "network" not in validated.to_payload()["mutation"]

    def test_an_unsupported_api_version_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="apiVersion"):
            TrainingManifest.model_validate({**DQN, "apiVersion": "agilerl.com/v99"})

    @pytest.mark.parametrize("replay_buffer", [None, {"max_size": 64}])
    def test_async_rollout_requires_the_llm_rollout_buffer(self, replay_buffer) -> None:
        # The rollout engines queue their groups in the LLM deque; the
        # RL buffer (or none) has nowhere to put them.
        broken = manifest(
            GRPO,
            environment={
                "env_type": "rollout",
                "entrypoint": "gem:make",
                "env_config": {"env_id": "game:Sudoku-v0"},
                "dataset": None,
                "reward_file_path": None,
                "prompt_template": None,
            },
            training={
                "rollout_mode": "async",
                "rollout_engines_per_agent": 1,
            },
        )
        broken["replay_buffer"] = replay_buffer
        with pytest.raises(ValidationError, match="LLMRolloutBufferSpec"):
            TrainingManifest.model_validate(broken)

    def test_an_anchored_answer_pattern_is_rejected(self) -> None:
        # Detection searches for the pattern; a grammar matches the whole
        # continuation against it. An anchor would mean two different things.
        broken = manifest(
            GRPO, algorithm={"answer_pattern": r"^<answer>[A-D]</answer>$"}
        )
        with pytest.raises(ValidationError, match="carries an anchor"):
            TrainingManifest.model_validate(broken)

    @pytest.mark.parametrize("name", ["SFT", "DPO"])
    def test_padded_forward_algorithms_reject_sequence_packing(self, name) -> None:
        # Neither algorithm builds the packed forward, so the request would be
        # dropped without a word.
        broken = manifest(
            GRPO,
            algorithm={
                "name": name,
                "use_sequence_packing": True,
                "attn_implementation": "flash_attention_2",
            },
        )
        broken["algorithm"].pop("group_size", None)
        with pytest.raises(
            ValidationError, match="does not accept use_sequence_packing"
        ):
            TrainingManifest.model_validate(broken)


class TestEpochDrivenTraining:
    """A run may state its schedule in epochs; the trainer converts to steps."""

    def test_accepts_num_epochs_with_evo_epochs(self) -> None:
        validated = TrainingManifest.model_validate(
            manifest(DQN, training={"num_epochs": 4, "evo_epochs": 1})
        )
        assert validated.training.num_epochs == 4
        assert validated.training.evo_epochs == 1

    def test_evo_epochs_alone_is_rejected(self) -> None:
        # The trainer only converts epochs for an epoch-driven loop, so the
        # interval would otherwise be silently dropped.
        with pytest.raises(ValidationError, match="evo_epochs needs"):
            TrainingManifest.model_validate(manifest(DQN, training={"evo_epochs": 1}))

    def test_evo_steps_is_not_defaulted_alongside_evo_epochs(self) -> None:
        # Two intervals in one payload would contradict each other; the trainer
        # computes evo_steps from evo_epochs once it knows the dataset size.
        validated = TrainingManifest.model_validate(
            manifest(DQN, training={"num_epochs": 4, "evo_epochs": 1})
        )
        assert validated.training.evo_steps is None

    def test_a_step_driven_run_still_gets_its_evo_steps_default(self) -> None:
        validated = TrainingManifest.model_validate(DQN)
        assert validated.training.evo_steps is not None

    def test_zero_evo_steps_is_rejected(self) -> None:
        # The trainer's next-event maths is `steps % evo_steps`, so a zero
        # interval is a ZeroDivisionError.
        with pytest.raises(ValidationError, match="evo_steps"):
            TrainingManifest.model_validate(manifest(DQN, training={"evo_steps": 0}))


class TestReplayBufferKind:
    """``kind`` is ``classic`` or ``llm``."""

    def test_an_omitted_kind_is_classic(self) -> None:
        validated = TrainingManifest.model_validate(
            {**DQN, "replay_buffer": {"max_size": 64}}
        )
        assert isinstance(validated.replay_buffer, ReplayBufferSpec)
        assert validated.replay_buffer.kind == "classic"

    def test_classic_kind_selects_the_experience_buffer(self) -> None:
        validated = TrainingManifest.model_validate(
            {**DQN, "replay_buffer": {"kind": "classic", "max_size": 64}}
        )
        assert isinstance(validated.replay_buffer, ReplayBufferSpec)

    def test_llm_kind_selects_the_rollout_deque(self) -> None:
        validated = TrainingManifest.model_validate(
            {**DQN, "replay_buffer": {"kind": "llm", "max_size": 64}}
        )
        assert isinstance(validated.replay_buffer, LLMRolloutBufferSpec)
        assert "kind" not in LLMRolloutBufferSpec.model_fields
        assert validated.replay_buffer.model_dump()["kind"] == "llm"

    def test_llm_rollout_buffer_spec_rejects_kind_as_a_constructor_arg(self) -> None:
        with pytest.raises(ValidationError):
            LLMRolloutBufferSpec(kind="llm")

    def test_an_unknown_kind_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            TrainingManifest.model_validate(
                {**DQN, "replay_buffer": {"kind": "nope", "max_size": 64}}
            )

    def test_rl_is_not_a_kind(self) -> None:
        with pytest.raises(ValidationError):
            TrainingManifest.model_validate(
                {**DQN, "replay_buffer": {"kind": "rl", "max_size": 64}}
            )


class TestCli:
    def _run(self, *args: str, stdin: str | None = None) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, "-m", "agilerl.arena.cli", "manifest", *args],
            capture_output=True,
            text=True,
            input=stdin,
            check=False,
        )

    def test_reads_a_manifest_from_stdin(self, tmp_path) -> None:
        result = self._run("validate", "-", "--json", stdin=json.dumps(GRPO))
        assert result.returncode == 0
        assert json.loads(result.stdout)["ok"] is True

    def test_json_reports_an_invalid_manifest_without_failing(self) -> None:
        # The verdict is data. A non-zero exit has to mean the tool broke, or a
        # caller cannot tell a bad manifest from a broken validator.
        broken = manifest(GRPO, algorithm={"name": "GPRO"})
        result = self._run("validate", "-", "--json", stdin=json.dumps(broken))
        assert result.returncode == 0
        report = json.loads(result.stdout)
        assert report["ok"] is False
        assert report["errors"][0]["path"] == "algorithm"
        assert "not a registered algorithm" in report["errors"][0]["message"]

    def test_json_reports_a_non_mapping_algorithm_section(self) -> None:
        # A malformed section shape must surface as a verdict, never as an
        # escaped TypeError traceback with exit 1 and empty stdout.
        broken = {**copy.deepcopy(GRPO), "algorithm": "just-a-string"}
        result = self._run("validate", "-", "--json", stdin=json.dumps(broken))
        assert result.returncode == 0, result.stderr
        report = json.loads(result.stdout)
        assert report["ok"] is False
        assert any("must be a mapping" in e["message"] for e in report["errors"])
        assert "Traceback" not in result.stderr

    def test_json_reports_the_package_version(self) -> None:
        result = self._run("validate", "-", "--json", stdin=json.dumps(GRPO))
        assert json.loads(result.stdout)["manifest_version"] == __version__

    def test_an_unreadable_manifest_exits_differently(self) -> None:
        result = self._run("validate", "/does/not/exist.yaml", "--json")
        assert result.returncode == 2
        assert result.stdout == ""

    def test_error_paths_name_the_field_a_submitter_wrote(self) -> None:
        broken = manifest(GRPO, algorithm={"vllm_config": {"max_num_seqs": 0}})
        result = self._run("validate", "-", "--json", stdin=json.dumps(broken))
        paths = [e["path"] for e in json.loads(result.stdout)["errors"]]
        assert any(p.endswith("vllm_config.max_num_seqs") for p in paths), paths
        assert not any("function-after" in p for p in paths), paths


class TestSchema:
    def test_schema_states_the_version_that_produced_it(self) -> None:
        assert manifest_schema()["x-manifest-version"] == __version__

    def test_schema_is_valid_json_schema(self) -> None:
        Draft202012Validator.check_schema(manifest_schema())

    def test_an_alias_spelling_is_accepted_but_not_drawn_twice(self) -> None:
        variant = next(
            v
            for v in manifest_schema()["properties"]["algorithm"]["oneOf"]
            if v["title"] == "LLMPPO"
        )
        assert variant["properties"]["lr"]["x-ui-alias-of"] == "lr_actor"
        assert "x-ui-alias-of" not in variant["properties"]["lr_actor"]

    @pytest.mark.parametrize("source", [DQN, GRPO], ids=["rl", "llm"])
    def test_the_payload_we_submit_satisfies_the_schema_we_publish(
        self, source: dict
    ) -> None:
        # The schema is what the UI form validates against; a payload it rejects
        # means the form and the models disagree.
        payload = TrainingManifest.model_validate(source).to_payload()
        errors = list(Draft202012Validator(manifest_schema()).iter_errors(payload))
        assert not errors, [f"{list(e.path)}: {e.message}" for e in errors]

    def test_optional_scalars_are_one_input_not_a_type_choice(self) -> None:
        variant = next(
            v
            for v in manifest_schema()["properties"]["algorithm"]["oneOf"]
            if v["title"] == "GRPO"
        )
        field = variant["properties"]["mini_batch_size"]
        assert "anyOf" not in field
        assert field["type"] == ["integer", "null"]
        assert field["minimum"] == 1

    def test_free_form_objects_ask_for_a_json_editor(self) -> None:
        variant = next(
            v
            for v in manifest_schema()["properties"]["algorithm"]["oneOf"]
            if v["title"] == "GRPO"
        )
        assert variant["properties"]["vllm_engine_args"]["x-ui-widget"] == "json"

    def test_a_model_with_fields_is_never_a_json_blob(self) -> None:
        schema = manifest_schema()
        blobs: list[str] = []

        def check(node: object, path: str) -> None:
            if isinstance(node, list):
                for i, item in enumerate(node):
                    check(item, f"{path}[{i}]")
                return
            if not isinstance(node, dict):
                return
            if node.get("x-ui-widget") == "json" and node.get("properties"):
                blobs.append(path)
            for key, value in node.items():
                check(value, f"{path}.{key}")

        check(schema, "$")
        assert not blobs, blobs

    def test_aliased_spellings_are_declared(self) -> None:
        schema = manifest_schema()
        assert "tournament_selection" in schema["properties"]
        assert "memory_size" in schema["$defs"]["ReplayBufferSpec"]["properties"]

    def test_schema_offers_each_algorithm_once(self) -> None:
        variants = manifest_schema()["properties"]["algorithm"]["oneOf"]
        titles = [v["title"] for v in variants]
        assert len(titles) == len(set(titles))
        accepted = {n for v in variants for n in v["properties"]["name"]["enum"]}
        assert accepted == set(MANIFEST_REGISTRY.names())

    def test_schema_defaults_an_aliased_algorithm_to_its_canonical_name(self) -> None:
        variants = manifest_schema()["properties"]["algorithm"]["oneOf"]
        rainbow = next(v for v in variants if v["title"] == "RainbowDQN")
        assert rainbow["properties"]["name"]["default"] == "RainbowDQN"
        assert "Rainbow DQN" in rainbow["properties"]["name"]["enum"]

    def test_the_payload_fills_the_sections_the_runtime_indexes_into(self) -> None:
        # Omitted mutation and tournament_selection dump with the keys later
        # readers index (rl_hp_selection, tournament_size, elitism).
        payload = TrainingManifest.model_validate(DQN).to_payload()
        assert payload["mutation"]["rl_hp_selection"] == {}
        assert payload["mutation"]["probabilities"] == {
            "no_mut": 1.0,
            "arch_mut": 0.0,
            "new_layer": 0.0,
            "params_mut": 0.0,
            "act_mut": 0.0,
            "rl_hp_mut": 0.0,
        }
        assert payload["tournament_selection"]["tournament_size"] == 2
        assert payload["tournament_selection"]["elitism"] is True

    def test_an_omitted_mutation_section_does_not_architecture_mutate(self) -> None:
        # An omitted mutation section dumps as a no-op so LoRA policies are
        # not architecture-mutated after arena submit.
        for source in (DQN, GRPO):
            payload = TrainingManifest.model_validate(source).to_payload()
            probs = payload["mutation"]["probabilities"]
            assert probs["no_mut"] == 1.0
            assert probs["arch_mut"] == 0.0
            assert probs["new_layer"] == 0.0
            assert probs["params_mut"] == 0.0

    def test_an_llm_run_keeps_its_reference_adapter(self) -> None:
        # The algorithm classes default this on for every LLM paradigm but SFT.
        # to_payload() emits it either way, and the runtime only setdefaults, so
        # a wrong default here silently disables the reference adapter.
        for name in ("GRPO", "GSPO", "CISPO"):
            payload = TrainingManifest.model_validate(
                manifest(GRPO, algorithm={"name": name})
            ).to_payload()
            assert payload["algorithm"]["use_separate_reference_adapter"] is True

    def test_sft_has_no_reference_adapter(self) -> None:
        doc = copy.deepcopy(GRPO)
        doc["algorithm"] = {"name": "SFT"}
        doc["environment"] = {"env_type": "dataset", "objective": "sft", "dataset": "d"}
        payload = TrainingManifest.model_validate(doc).to_payload()
        assert payload["algorithm"]["use_separate_reference_adapter"] is False

    def test_recurrent_ppo_stays_recurrent(self) -> None:
        # "Recurrent PPO" and "PPO" resolve to one spec, so the alias has to
        # carry the flag or the run silently trains as plain PPO.
        doc = copy.deepcopy(DQN)
        doc["algorithm"] = {"name": "Recurrent PPO"}
        payload = TrainingManifest.model_validate(doc).to_payload()
        assert payload["algorithm"]["recurrent"] is True
        assert payload["algorithm"]["name"] == "Recurrent PPO"

        doc["algorithm"] = {"name": "RecurrentPPO"}
        unspaced = TrainingManifest.model_validate(doc).to_payload()
        assert unspaced["algorithm"]["recurrent"] is True
        assert unspaced["algorithm"]["name"] == "Recurrent PPO"

        doc["algorithm"] = {"name": "PPO"}
        plain = TrainingManifest.model_validate(doc).to_payload()
        assert plain["algorithm"]["recurrent"] is False
        assert plain["algorithm"]["name"] == "PPO"

    def test_mutating_a_field_the_algorithm_lacks_is_rejected(self) -> None:
        doc = manifest(
            DQN, mutation={"rl_hp_selection": {"theta": {"min": 0.1, "max": 1.0}}}
        )
        with pytest.raises(ValidationError, match="no such hyperparameter"):
            TrainingManifest.model_validate(doc)

    def test_a_near_miss_names_the_field_it_meant(self) -> None:
        doc = manifest(
            DQN, mutation={"rl_hp_selection": {"taus": {"min": 0.1, "max": 1.0}}}
        )
        with pytest.raises(ValidationError, match="Did you mean 'tau'"):
            TrainingManifest.model_validate(doc)

    def test_mutating_a_non_numeric_field_is_rejected(self) -> None:
        # The field exists, so a name check alone would let this through and
        # then apply a grow factor to a flag.
        doc = manifest(
            GRPO,
            mutation={
                "rl_hp_selection": {"gradient_checkpointing": {"min": 0, "max": 1}}
            },
        )
        with pytest.raises(ValidationError, match="not a number"):
            TrainingManifest.model_validate(doc)

    def test_mutating_a_list_or_scalar_union_is_rejected(self) -> None:
        # GRPO clip_coef is float | list[float]. A grow factor cannot move a
        # [low, high] pair between min and max.
        doc = manifest(
            GRPO,
            mutation={"rl_hp_selection": {"clip_coef": {"min": 0.05, "max": 0.35}}},
        )
        with pytest.raises(ValidationError, match="not a number"):
            TrainingManifest.model_validate(doc)

    def test_an_aliased_spelling_may_be_mutated(self) -> None:
        # Most manifests spell LLMPPO's lr_actor as lr; rejecting the alias
        # would fail documents that run today.
        doc = copy.deepcopy(GRPO)
        doc["algorithm"] = {"name": "LLMPPO"}
        doc["mutation"] = {"rl_hp_selection": {"lr": {"min": 1e-7, "max": 1e-5}}}
        assert TrainingManifest.model_validate(doc)

    def test_hpo_ranges_cover_only_fields_the_algorithm_has(self) -> None:
        variants = manifest_schema()["properties"]["algorithm"]["oneOf"]
        for variant in variants:
            for field in variant.get("x-hpo-ranges", {}):
                assert field in variant["properties"], (
                    f"{variant['title']} offers a range for {field}, "
                    "which it does not accept"
                )
        ppo = next(v for v in variants if v["title"] == "PPO")
        ddpg = next(v for v in variants if v["title"] == "DDPG")
        assert "theta" not in ppo["x-hpo-ranges"]
        assert "theta" in ddpg["x-hpo-ranges"]

    def test_hpo_ranges_are_family_specific(self) -> None:
        # RL and LLM share field names and disagree on bounds. Collapsing them
        # would let HPO mutate an LLM learning rate into the RL band mid-run.
        variants = manifest_schema()["properties"]["algorithm"]["oneOf"]
        rl = next(v for v in variants if v["title"] == "RainbowDQN")
        llm = next(v for v in variants if v["title"] == "DPO")
        assert rl["x-hpo-ranges"]["beta"] == {
            "min": 0.2,
            "max": 0.6,
            "grow_factor": 1.2,
            "shrink_factor": 0.8,
        }
        assert llm["x-hpo-ranges"]["beta"] == {
            "min": 0.0001,
            "max": 0.01,
            "grow_factor": 1.2,
            "shrink_factor": 0.8,
        }

    def test_learn_step_ranges_differ_by_algorithm(self) -> None:
        variants = manifest_schema()["properties"]["algorithm"]["oneOf"]
        dqn = next(v for v in variants if v["title"] == "DQN")
        ppo = next(v for v in variants if v["title"] == "PPO")
        assert dqn["x-hpo-ranges"]["learn_step"] == {
            "min": 1,
            "max": 10,
            "grow_factor": 1.5,
            "shrink_factor": 0.75,
        }
        assert ppo["x-hpo-ranges"]["learn_step"] == {
            "min": 512,
            "max": 8192,
            "grow_factor": 1.5,
            "shrink_factor": 0.75,
        }

    def test_a_list_or_scalar_union_is_not_offered_for_mutation(self) -> None:
        variants = manifest_schema()["properties"]["algorithm"]["oneOf"]
        grpo = next(v for v in variants if v["title"] == "GRPO")
        ppo = next(v for v in variants if v["title"] == "PPO")
        assert "clip_coef" not in grpo["x-hpo-ranges"]
        assert "clip_coef" in ppo["x-hpo-ranges"]

    def test_hpo_ranges_are_valid_rl_hyperparameters(self) -> None:
        # They are emitted as plain JSON, but rl_hp_selection has to accept them
        # back verbatim — that round trip is the whole point of the shape.
        variants = manifest_schema()["properties"]["algorithm"]["oneOf"]
        emitted = {
            field: bounds
            for v in variants
            for field, bounds in v.get("x-hpo-ranges", {}).items()
        }
        assert emitted
        MutationSpec(rl_hp_selection=emitted)

    def test_every_field_carries_help_text(self) -> None:
        # The schema is what renders the UI form, so an undescribed field ships
        # as a labelled box with no explanation.
        schema = manifest_schema()
        undescribed: list[str] = []

        def check(node: dict, path: str) -> None:
            for name, field in (node.get("properties") or {}).items():
                if "description" not in field:
                    undescribed.append(f"{path}.{name}")

        check(schema, "")
        for variant in schema["properties"]["algorithm"]["oneOf"]:
            check(variant, f"algorithm[{variant['title']}]")
        for name, definition in schema["$defs"].items():
            check(definition, name)

        assert not undescribed

    def test_gym_sync_describes_behaviour_not_a_runtime(self) -> None:
        description = GymEnvSpec.model_fields["sync"].description
        assert description is not None
        assert "Ray" not in description

    def test_schema_declares_the_manifest_sections(self) -> None:
        properties = manifest_schema()["properties"]
        assert {
            "algorithm",
            "environment",
            "training",
            "network",
            "mutation",
            "replay_buffer",
        } <= set(properties)


class TestClusterSupported:
    def test_cqn_and_bandits_are_local_only(self) -> None:
        from agilerl.arena.models.algorithms import (
            CQNSpec,
            DQNSpec,
            NeuralTSSpec,
            NeuralUCBSpec,
        )

        assert CQNSpec.cluster_supported is False
        assert NeuralTSSpec.cluster_supported is False
        assert NeuralUCBSpec.cluster_supported is False
        assert DQNSpec.cluster_supported is True


class TestRolloutSamplingFields:
    def test_grpo_defaults_match_the_algorithm_ctor(self) -> None:
        spec = GRPOSpec.model_construct(group_size=4)
        assert spec.top_p == 0.95
        assert spec.top_k == 50
        assert spec.min_p == 0.0
        assert spec.repetition_penalty == 1.0
        assert spec.loss_type == "grpo"

    def test_llm_ppo_and_reinforce_default_top_p_to_one(self) -> None:
        from agilerl.arena.models.algorithms import LLMPPOSpec, LLMREINFORCESpec

        assert LLMPPOSpec.model_construct().top_p == 1.0
        assert LLMREINFORCESpec.model_construct().top_p == 1.0

    def test_gspo_and_cispo_pin_loss_type(self) -> None:
        from agilerl.arena.models.algorithms import CISPOSpec, GSPOSpec

        assert GSPOSpec.model_construct(group_size=4).loss_type == "gspo"
        assert CISPOSpec.model_construct(group_size=4).loss_type == "cispo"
        with pytest.raises(ValidationError):
            GSPOSpec(group_size=4, loss_type="grpo")
