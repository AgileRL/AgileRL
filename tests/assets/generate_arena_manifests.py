"""Build full Arena training manifests for every arena-eligible algorithm.

Used by ``tests/test_models/test_manifest.py`` (ephemeral tmp dirs) and as a
manual helper to inspect manifests locally.

Usage::

    uv run python tests/assets/generate_arena_manifests.py
    uv run python tests/assets/generate_arena_manifests.py /tmp/arena_manifests
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import get_args
from unittest.mock import MagicMock

import yaml

from agilerl import HAS_LLM_DEPENDENCIES, AgentType
from agilerl.models.algo import ALGO_REGISTRY, LLMAlgorithmSpec
from agilerl.models.env import ArenaEnvSpec
from agilerl.models.hpo import MutationSpec, TournamentSelectionSpec
from agilerl.models.manifest import ArenaManifest
from agilerl.models.networks import (
    DeterministicActorSpec,
    MlpSpec,
    NetworkSpec,
    QNetworkSpec,
    RainbowQNetworkSpec,
    StochasticActorSpec,
)
from agilerl.models.training import ReplayBufferSpec, TrainingSpec
from agilerl.training.trainer import ArenaTrainer

_ASSETS_DIR = Path(__file__).resolve().parent
_TINY_LLM = str(_ASSETS_DIR / "tiny_llm")
_DEFAULT_ENV = {
    AgentType.SingleAgent: "CartPole-v1",
    AgentType.MultiAgent: "pettingzoo.mpe.simple_speaker_listener_v4",
    AgentType.LLMAgent: "CartPole-v1",
}
_GRPO_FAMILY = frozenset({"GRPO", "CISPO", "GSPO"})
_DEFAULT_MLP_NETWORK: dict[type[NetworkSpec], NetworkSpec] = {
    QNetworkSpec: QNetworkSpec(
        encoder_config=MlpSpec(hidden_size=[64]),
        head_config=MlpSpec(hidden_size=[64]),
    ),
    RainbowQNetworkSpec: RainbowQNetworkSpec(
        encoder_config=MlpSpec(hidden_size=[64]),
        head_config=MlpSpec(hidden_size=[64]),
    ),
    StochasticActorSpec: StochasticActorSpec(
        encoder_config=MlpSpec(hidden_size=[64]),
        head_config=MlpSpec(hidden_size=[64]),
    ),
    DeterministicActorSpec: DeterministicActorSpec(
        encoder_config=MlpSpec(hidden_size=[64]),
        head_config=MlpSpec(hidden_size=[64]),
    ),
}


def arena_algorithm_names() -> list[str]:
    """Return sorted arena algorithm names, skipping LLM algos when deps are absent."""
    names = sorted(ALGO_REGISTRY.arena_algorithms())
    if HAS_LLM_DEPENDENCIES:
        return names
    return [
        name
        for name in names
        if not issubclass(ALGO_REGISTRY.get(name).spec_cls, LLMAlgorithmSpec)
    ]


def _attach_default_network(algorithm) -> None:
    if getattr(algorithm, "net_config", None) is not None:
        return

    net_config_field = type(algorithm).model_fields.get("net_config")
    if net_config_field is None:
        return

    for candidate in get_args(net_config_field.annotation):
        if candidate is type(None):
            continue
        default_net = _DEFAULT_MLP_NETWORK.get(candidate)
        if default_net is not None:
            algorithm.net_config = default_net.model_copy(deep=True)
            return


def _default_algorithm_spec(name: str):
    spec_cls = ALGO_REGISTRY.get(name).spec_cls
    if not HAS_LLM_DEPENDENCIES and issubclass(spec_cls, LLMAlgorithmSpec):
        msg = "LLM dependencies are required to build manifests for LLM algorithms."
        raise RuntimeError(msg)

    if issubclass(spec_cls, LLMAlgorithmSpec):
        kwargs: dict = {"pretrained_model_name_or_path": _TINY_LLM}
        if name in _GRPO_FAMILY:
            kwargs["group_size"] = 2
        return spec_cls(**kwargs)

    return spec_cls()


def write_arena_manifest(algo_name: str, output_dir: Path) -> Path:
    """Write one Arena manifest YAML for *algo_name* and return its path."""
    algorithm = _default_algorithm_spec(algo_name)
    _attach_default_network(algorithm)

    trainer_kwargs: dict = {
        "algorithm": algorithm,
        "environment": ArenaEnvSpec(name=_DEFAULT_ENV[algorithm.agent_type]),
        "training": TrainingSpec(),
        "mutation": MutationSpec(),
        "tournament": TournamentSelectionSpec(),
        "client": MagicMock(),
    }
    if algorithm.off_policy:
        trainer_kwargs["replay_buffer"] = ReplayBufferSpec()

    trainer = ArenaTrainer(**trainer_kwargs)
    manifest = ArenaManifest.get_validated(trainer.to_manifest(), mode="json")

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{algo_name.lower()}.yaml"
    with path.open("w") as fh:
        yaml.safe_dump(manifest, fh, sort_keys=False, default_flow_style=False)
    return path


def generate_arena_manifests(output_dir: Path) -> dict[str, Path]:
    """Write manifests for all arena algorithms and return ``{name: path}``."""
    paths: dict[str, Path] = {}
    for name in arena_algorithm_names():
        paths[name] = write_arena_manifest(name, output_dir)
    return paths


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    output_dir = Path(args[0]) if args else Path.cwd() / "arena_manifests"
    for path in generate_arena_manifests(output_dir).values():
        print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
