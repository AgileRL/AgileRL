# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Tests for ``agilerl.arena.models`` manifest validation."""

from __future__ import annotations

import warnings
from unittest.mock import patch

import pytest
import yaml
from generate_arena_manifests import (
    arena_algorithm_names,
    generate_arena_manifests,
    write_arena_manifest,
)
from pydantic import ValidationError

from agilerl.arena.models import (
    ARENA_REGISTRY,
    ReplayBufferSpec,
    TrainingManifest,
    TrainingSpec,
)
from agilerl.arena.models.algorithms.dqn import DQNSpec
from agilerl.arena.models.env import EnvSpec, LLMEnvType
from agilerl.arena.models.manifest import (
    _coerce_environment,
    _resolve_algorithm,
    _resolve_network,
)
from agilerl.arena.models.networks import (
    FinetuningNetworkSpec,
    MlpSpec,
    QNetworkSpec,
)


def _manifest(**sections) -> dict:
    """Build a manifest dict with sensible defaults."""
    data = {
        "algorithm": sections.pop("algorithm", {"name": "DQN"}),
        "environment": sections.pop("environment", {"name": "CartPole-v1"}),
    }
    data.update(sections)
    return data


@pytest.mark.parametrize("name", ["DQN", "PPO", "GRPO"])
def test_arena_registry_contains_algorithm(name: str) -> None:
    assert ARENA_REGISTRY.get(name).spec_cls.__name__ == f"{name}Spec"


def test_env_spec_defaults() -> None:
    spec = EnvSpec(name="CartPole-v1")
    assert spec.num_envs == 16
    assert spec.version is None


def test_training_and_replay_buffer_aliases() -> None:
    training = TrainingSpec.model_validate(
        {"population_size": 4, "metrics_interval": 123}
    )
    assert training.pop_size == 4
    assert training.evo_steps == 123

    replay = ReplayBufferSpec.model_validate({"memory_size": 4096})
    assert replay.max_size == 4096


def test_get_validated_inserts_empty_platform_sections() -> None:
    payload = TrainingManifest.get_validated(
        _manifest(algorithm={"name": "DQN", "lr": 3e-4})
    )
    assert payload["mutation"] == {}
    assert payload["tournament_selection"] == {}
    assert payload["network"] == {}
    assert payload["algorithm"]["name"] == "DQN"


def test_get_validated_serializes_the_selection_section_for_the_platform() -> None:
    payload = TrainingManifest.get_validated(
        _manifest(selection_strategy={"tournament_size": 3, "elitism": False})
    )
    assert payload["tournament_selection"] == {"tournament_size": 3, "elitism": False}
    assert "selection_strategy" not in payload


@pytest.mark.parametrize("key", ["selection_strategy", "tournament_selection"])
def test_get_validated_python_mode_populates_selection_strategy(key: str) -> None:
    validated = TrainingManifest.get_validated(
        _manifest(**{key: {"tournament_size": 3}}), mode="python"
    )
    assert validated.selection_strategy.tournament_size == 3


@pytest.mark.parametrize("key", ["selection_strategy", "tournament_selection"])
def test_deprecated_tournament_selection_property_returns_the_spec(key: str) -> None:
    validated = TrainingManifest.get_validated(
        _manifest(**{key: {"tournament_size": 3}}), mode="python"
    )
    with pytest.warns(DeprecationWarning, match="tournament_selection is deprecated"):
        assert validated.tournament_selection is validated.selection_strategy


def test_deprecated_tournament_selection_property_is_none_when_unset() -> None:
    validated = TrainingManifest.get_validated(_manifest(), mode="python")
    with pytest.warns(DeprecationWarning, match="tournament_selection is deprecated"):
        assert validated.tournament_selection is None


def test_deprecated_tournament_selection_property_is_not_serialized() -> None:
    validated = TrainingManifest.get_validated(
        _manifest(selection_strategy={"tournament_size": 3}), mode="python"
    )
    assert "tournament_selection" not in TrainingManifest.model_computed_fields
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        dumped = validated.model_dump(exclude_none=True, by_alias=True)
    assert dumped["tournament_selection"] == {"tournament_size": 3, "elitism": True}


def test_get_validated_payload_is_unchanged_by_the_selection_key_used() -> None:
    legacy = TrainingManifest.get_validated(
        _manifest(tournament_selection={"tournament_size": 3})
    )
    current = TrainingManifest.get_validated(
        _manifest(selection_strategy={"tournament_size": 3})
    )
    assert legacy == current


def test_get_validated_accepts_the_core_discriminator() -> None:
    # The (selection) strategy discriminator is accepted but never forwarded to Arena
    with patch("agilerl.arena.models.manifest.logger") as mock_logger:
        payload = TrainingManifest.get_validated(
            _manifest(
                selection_strategy={"strategy": "tournament", "tournament_size": 3}
            )
        )
    mock_logger.warning.assert_not_called()
    assert payload["tournament_selection"] == {"tournament_size": 3, "elitism": True}


def test_get_validated_rejects_a_non_tournament_strategy() -> None:
    # Arena runs tournament selection only
    with pytest.raises(ValidationError):
        TrainingManifest.get_validated(
            _manifest(
                selection_strategy={"strategy": "multi_frequency", "n_winners": 1}
            )
        )


def test_get_validated_accepts_the_core_parameter_mutation_fields() -> None:
    # The ReGraMa band fields are accepted at their defaults but never forwarded to Arena
    with patch("agilerl.arena.models.manifest.logger") as mock_logger:
        payload = TrainingManifest.get_validated(
            _manifest(
                mutation={
                    "mutation_sd": 0.2,
                    "amplified_gauss_param_mut": True,
                    "random_reset_param_mut": True,
                    "dormant_threshold": 0.05,
                }
            )
        )
    mock_logger.warning.assert_not_called()
    assert payload["mutation"]["mutation_sd"] == 0.2
    assert not {
        "amplified_gauss_param_mut",
        "random_reset_param_mut",
        "dormant_threshold",
    } & set(payload["mutation"])


@pytest.mark.parametrize(
    "mutation",
    [
        {"amplified_gauss_param_mut": False},
        {"random_reset_param_mut": False},
    ],
)
def test_get_validated_rejects_unsupported_parameter_mutations(mutation: dict) -> None:
    # Arena runs the default Gaussian parameter mutation bands only, so a request
    # the platform cannot support must fail
    with pytest.raises(ValidationError, match="Gaussian parameter mutation bands"):
        TrainingManifest.get_validated(_manifest(mutation=mutation))


def test_get_validated_reports_unknown_keys_under_the_current_selection_key() -> None:
    with patch("agilerl.arena.models.manifest.logger") as mock_logger:
        TrainingManifest.get_validated(
            _manifest(selection_strategy={"tournament_size": 3, "bogus_key": 1})
        )
    mock_logger.warning.assert_called_once()
    assert "selection_strategy.bogus_key" in mock_logger.warning.call_args.args[1]


def test_get_validated_reports_unknown_keys_under_the_legacy_selection_key() -> None:
    with patch("agilerl.arena.models.manifest.logger") as mock_logger:
        TrainingManifest.get_validated(
            _manifest(tournament_selection={"tournament_size": 3, "bogus_key": 1})
        )
    mock_logger.warning.assert_called_once()
    assert "tournament_selection.bogus_key" in mock_logger.warning.call_args.args[1]


def test_get_validated_warns_on_unknown_algorithm_field() -> None:
    with patch("agilerl.arena.models.manifest.logger") as mock_logger:
        TrainingManifest.get_validated(
            _manifest(algorithm={"name": "DQN", "lr": 3e-4, "bogus_algo_field": 1})
        )
    mock_logger.warning.assert_called_once()
    template, formatted = mock_logger.warning.call_args.args
    assert "unrecognized manifest field" in template
    assert "algorithm.bogus_algo_field" in formatted


def test_get_validated_warns_on_unknown_top_level_field() -> None:
    with patch("agilerl.arena.models.manifest.logger") as mock_logger:
        TrainingManifest.get_validated(_manifest(bogus_top_level=123))
    mock_logger.warning.assert_called_once()
    assert "bogus_top_level" in mock_logger.warning.call_args.args[1]


def test_get_validated_no_warning_for_aliased_fields() -> None:
    # population_size / metrics_interval / memory_size are validation aliases,
    # not unknown fields.
    with patch("agilerl.arena.models.manifest.logger") as mock_logger:
        TrainingManifest.get_validated(
            _manifest(
                algorithm={"name": "DQN", "lr": 3e-4},
                training={"population_size": 4, "metrics_interval": 123},
                replay_buffer={"memory_size": 4096},
            )
        )
    mock_logger.warning.assert_not_called()


def test_collect_unknown_fields_ignores_non_dict_raw() -> None:
    from agilerl.arena.models.manifest import _collect_unknown_fields

    validated = TrainingManifest.get_validated(_manifest(), mode="python")
    assert _collect_unknown_fields("not-a-dict", validated) == []
    assert _collect_unknown_fields(None, validated) == []


def test_known_field_names_includes_all_alias_forms() -> None:
    from pydantic import AliasChoices, BaseModel, Field

    from agilerl.arena.models.manifest import _known_field_names

    class _M(BaseModel):
        plain: int = Field(default=0)
        aliased: int = Field(default=0, alias="aliased_in")
        val_str: int = Field(default=0, validation_alias="val_str_in")
        val_choices: int = Field(
            default=0, validation_alias=AliasChoices("choice_a", "choice_b")
        )

    names = _known_field_names(_M())
    assert {
        "plain",
        "aliased",
        "aliased_in",
        "val_str",
        "val_str_in",
        "val_choices",
        "choice_a",
        "choice_b",
    } <= names


def test_deferred_simba_recurrent_conflict_raises() -> None:
    # arch omitted -> net_config stays None; the simba flag is read from the raw
    # network dict, and simba + recurrent must be rejected.
    with pytest.raises(ValidationError, match="simba"):
        TrainingManifest.get_validated(
            _manifest(
                algorithm={"name": "PPO", "recurrent": True},
                network={"latent_dim": 64, "simba": True},
            )
        )


def test_get_validated_accepts_env_spec_objects() -> None:
    payload = TrainingManifest.get_validated(
        _manifest(environment=EnvSpec(name="LunarLander-v3", num_envs=8, version="v9"))
    )
    assert payload["environment"] == {
        "name": "LunarLander-v3",
        "num_envs": 8,
        "version": "v9",
    }


def test_get_validated_passes_network_raw_when_arch_missing() -> None:
    """When `arch` is absent, the client defers to the server."""
    network = {
        "encoder_config": {"hidden_size": [64], "activation": "ReLU"},
        "head_config": {"hidden_size": [64], "activation": "ReLU"},
    }
    payload = TrainingManifest.get_validated(_manifest(network=network))
    assert payload["network"] == network


def test_net_config_never_set_when_arch_present() -> None:
    """`arch` is resolved server-side, so `net_config` must never be populated on
    the algorithm spec (a set `net_config` is rejected server-side).
    """
    network = {
        "arch": "mlp",
        "encoder_config": {"hidden_size": [64], "activation": "ReLU"},
        "head_config": {"hidden_size": [64], "activation": "ReLU"},
    }
    manifest = TrainingManifest.get_validated(
        _manifest(algorithm={"name": "DQN"}, network=network), mode="python"
    )
    assert manifest.algorithm.net_config is None

    payload = TrainingManifest.get_validated(
        _manifest(algorithm={"name": "DQN"}, network=network)
    )
    assert "net_config" not in payload["algorithm"]


def test_get_validated_raises_for_unknown_algorithm() -> None:
    with pytest.raises(KeyError, match="No registry entry for algorithm"):
        TrainingManifest.get_validated(_manifest(algorithm={"name": "NOT_REAL"}))


def test_llm_requires_pretrained_model_name() -> None:
    with pytest.raises(ValidationError, match="pretrained_model_name_or_path"):
        TrainingManifest.get_validated(
            _manifest(algorithm={"name": "DPO"}, environment={"name": "my-llm-env"})
        )


def test_llm_pretrained_model_can_come_from_network_section() -> None:
    payload = TrainingManifest.get_validated(
        _manifest(
            algorithm={"name": "DPO"},
            environment={"name": "my-llm-env"},
            network={
                "pretrained_model_name_or_path": "Qwen/Qwen2.5-0.5B-Instruct",
                "max_context_length": 2048,
            },
        )
    )
    assert payload["algorithm"]["pretrained_model_name_or_path"] == (
        "Qwen/Qwen2.5-0.5B-Instruct"
    )
    assert payload["algorithm"]["max_model_len"] == 2048


def test_llm_lora_and_max_output_tokens_excluded_from_algorithm() -> None:
    payload = TrainingManifest.get_validated(
        _manifest(
            algorithm={"name": "GRPO", "group_size": 8},
            environment={"name": "my-llm-env"},
            network={
                "pretrained_model_name_or_path": "Qwen/Qwen2.5-0.5B-Instruct",
                "max_context_length": 512,
                "lora_config": {"lora_r": 16},
            },
        )
    )
    algorithm = payload["algorithm"]
    # Model fields the platform expects under algorithm are still there.
    assert algorithm["pretrained_model_name_or_path"] == "Qwen/Qwen2.5-0.5B-Instruct"
    assert algorithm["max_model_len"] == 512
    # lora_config and max_output_tokens must not be emitted under algorithm.
    assert "lora_config" not in algorithm
    assert "max_output_tokens" not in algorithm
    # lora_config still travels under the network section.
    assert payload["network"]["lora_config"]["lora_r"] == 16


def test_get_validated_loads_yaml_file(tmp_path) -> None:
    path = tmp_path / "manifest.yaml"
    path.write_text(yaml.safe_dump(_manifest()))

    payload = TrainingManifest.get_validated(path)
    assert payload["algorithm"]["name"] == "DQN"


def test_get_validated_python_mode_returns_model() -> None:
    manifest = TrainingManifest.get_validated(_manifest(), mode="python")
    assert manifest.algorithm.name == "DQN"
    assert manifest.environment["name"] == "CartPole-v1"


def test_generates_one_manifest_per_arena_algorithm(tmp_path) -> None:
    names = arena_algorithm_names()
    manifests = generate_arena_manifests(tmp_path)
    assert len(manifests) == len(names)
    assert {path.stem for path, _ in manifests.values()} == {
        name.lower() for name in names
    }


@pytest.mark.parametrize("algo_name", arena_algorithm_names())
def test_generated_manifest_validates(algo_name: str, tmp_path) -> None:
    path, validated = write_arena_manifest(algo_name, tmp_path)
    assert path.exists()
    assert validated["algorithm"]["name"] == algo_name
    assert validated["environment"]["name"]
    assert "mutation" in validated
    assert "tournament_selection" in validated


def test_llm_env_type_str() -> None:
    assert str(LLMEnvType.REASONING) == "reasoning"


def test_training_spec_rejects_invalid_eval_steps() -> None:
    with pytest.raises(
        ValueError, match="eval_steps must be greater than evaluation_interval"
    ):
        TrainingSpec(eval_steps=10, evaluation_interval=10)


def test_training_spec_rejects_evo_steps_above_max_steps() -> None:
    with pytest.raises(
        ValueError, match=r"evo_steps .* must be less than or equal to max_steps"
    ):
        TrainingSpec(max_steps=100, evo_steps=200)


def test_training_spec_rejects_eps_start_below_eps_end() -> None:
    with pytest.raises(
        ValueError, match=r"eps_start .* must be greater than or equal to eps_end"
    ):
        TrainingSpec(eps_start=0.1, eps_end=0.9)


def test_registry_override_logs_warning() -> None:
    with patch("agilerl.arena.models.algo.logger") as mock_logger:
        ARENA_REGISTRY.add("DQN", DQNSpec)
    mock_logger.warning.assert_called_once_with(
        "Overriding existing registration for algorithm %r",
        "DQN",
    )


def test_resolve_algorithm_requires_name() -> None:
    with pytest.raises(ValueError, match="must include a 'name' field"):
        _resolve_algorithm({"lr": 1e-3})


def test_resolve_algorithm_rejects_non_mapping() -> None:
    with pytest.raises(TypeError, match="Expected a dict or AlgorithmSpec"):
        _resolve_algorithm(42)


def test_coerce_environment_rejects_invalid_type() -> None:
    with pytest.raises(TypeError, match="Expected a dict or environment spec"):
        _coerce_environment("not-an-env")


def test_resolve_network_accepts_finetuning_spec() -> None:
    spec = FinetuningNetworkSpec(
        pretrained_model_name_or_path="gpt2",
        max_context_length=512,
    )
    resolved = _resolve_network(spec)
    assert resolved["pretrained_model_name_or_path"] == "gpt2"


def test_python_manifest_accepts_network_spec_object() -> None:
    manifest = TrainingManifest.model_validate(
        {
            "algorithm": {"name": "DQN"},
            "environment": {"name": "CartPole-v1"},
            "network": QNetworkSpec(
                encoder_config=MlpSpec(hidden_size=[64]),
                head_config=MlpSpec(hidden_size=[64]),
            ),
        }
    )
    assert manifest.network is not None
    assert manifest.network["encoder_config"]["hidden_size"] == [64]
    assert "arch" not in manifest.network["encoder_config"]
