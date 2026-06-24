"""Tests for ``agilerl.arena.models`` manifest validation."""

from __future__ import annotations

from unittest.mock import patch

import pytest
import yaml
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
    _normalize_network_arch,
    _resolve_algorithm,
    _resolve_network,
)
from agilerl.arena.models.networks import (
    FinetuningNetworkSpec,
    MlpSpec,
    QNetworkSpec,
)
from generate_arena_manifests import (
    arena_algorithm_names,
    generate_arena_manifests,
    write_arena_manifest,
)
from pydantic import ValidationError


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
        _manifest(algorithm={"name": "DQN", "lr": 3e-4, "cudagraphs": True})
    )
    assert payload["mutation"] == {}
    assert payload["tournament_selection"] == {}
    assert payload["network"] == {}
    assert payload["algorithm"]["name"] == "DQN"
    assert "cudagraphs" not in payload["algorithm"]


def test_get_validated_accepts_env_spec_objects() -> None:
    payload = TrainingManifest.get_validated(
        _manifest(environment=EnvSpec(name="LunarLander-v3", num_envs=8, version="v9"))
    )
    assert payload["environment"] == {
        "name": "LunarLander-v3",
        "num_envs": 8,
        "version": "v9",
    }


def test_get_validated_normalizes_network_for_platform() -> None:
    payload = TrainingManifest.get_validated(
        _manifest(
            network={
                "arch": "mlp",
                "encoder_config": {"hidden_size": [64], "activation": "ReLU"},
                "head_config": {"hidden_size": [64], "activation": "ReLU"},
            }
        )
    )
    network = payload["network"]
    assert network["arch"] == "mlp"
    assert network["name"] == "EvolvableMLP"
    assert "arch" not in network["encoder_config"]


def test_get_validated_raises_for_missing_network_arch() -> None:
    with pytest.raises(ValidationError, match="Missing encoder architecture"):
        TrainingManifest.get_validated(
            _manifest(
                network={
                    "encoder_config": {"hidden_size": [64], "activation": "ReLU"},
                    "head_config": {"hidden_size": [64], "activation": "ReLU"},
                }
            )
        )


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


def test_dqn_generated_manifest_omits_cudagraphs(tmp_path) -> None:
    _path, validated = write_arena_manifest("DQN", tmp_path)
    assert validated["algorithm"]["name"] == "DQN"
    assert "cudagraphs" not in validated["algorithm"]


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


def test_normalize_network_arch_passthrough_for_spec() -> None:
    spec = MlpSpec(hidden_size=[64])
    assert _normalize_network_arch(spec) is spec


def test_normalize_network_arch_creates_encoder_from_top_level_arch() -> None:
    normalized = _normalize_network_arch(
        {"arch": "mlp", "head_config": {"hidden_size": [64]}}
    )
    assert normalized["encoder_config"]["arch"] == "mlp"


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
    assert manifest.network["encoder_config"]["arch"] == "mlp"
