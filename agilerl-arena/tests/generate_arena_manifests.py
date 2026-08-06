# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Build full Arena training manifests for every registered algorithm.

Used by ``test_arena_models.py`` and as a manual helper to inspect manifests::

    uv run python agilerl-arena/tests/generate_arena_manifests.py
    uv run python agilerl-arena/tests/generate_arena_manifests.py /tmp/arena_manifests
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, get_args

import yaml

from agilerl.arena import AgentType
from agilerl.arena.models import TrainingManifest as ArenaManifest
from agilerl.arena.models.algo import ARENA_REGISTRY, LLMAlgorithmSpec
from agilerl.arena.models.env import EnvSpec as ArenaEnvSpec
from agilerl.arena.models.hpo import MutationSpec, TournamentSelectionSpec
from agilerl.arena.models.networks import (
    DeterministicActorSpec,
    MlpSpec,
    NetworkSpec,
    QNetworkSpec,
    RainbowQNetworkSpec,
    StochasticActorSpec,
)
from agilerl.arena.models.training import ReplayBufferSpec, TrainingSpec

_REPO_ROOT = Path(__file__).resolve().parents[2]
_TINY_LLM = str(_REPO_ROOT / "tests" / "assets" / "tiny_llm")
_DEFAULT_ENV = {
    AgentType.SingleAgent: "CartPole-v1",
    AgentType.MultiAgent: "mpe2.simple_speaker_listener_v4",
    AgentType.LLMAgent: "CartPole-v1",
}
_GRPO_FAMILY = frozenset({"GRPO", "CISPO", "GSPO"})
_OFF_POLICY_ALGOS = frozenset({"DQN", "DDPG", "TD3", "RainbowDQN", "MADDPG", "MATD3"})
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
    """Return sorted names of all algorithms registered for Arena."""
    return sorted(ARENA_REGISTRY._entries)


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
    spec_cls = ARENA_REGISTRY.get(name).spec_cls
    if issubclass(spec_cls, LLMAlgorithmSpec):
        kwargs: dict = {"pretrained_model_name_or_path": _TINY_LLM}
        if name in _GRPO_FAMILY:
            kwargs["group_size"] = 2
        return spec_cls(**kwargs)

    return spec_cls()


def _build_manifest_dict(algorithm, algo_name: str) -> dict[str, Any]:
    """Assemble a manifest dict from arena model instances."""
    data: dict[str, Any] = {
        "algorithm": algorithm,
        "environment": ArenaEnvSpec(name=_DEFAULT_ENV[algorithm.agent_type]),
        "training": TrainingSpec(),
        "mutation": MutationSpec(),
        "tournament_selection": TournamentSelectionSpec(),
    }
    if algo_name in _OFF_POLICY_ALGOS:
        data["replay_buffer"] = ReplayBufferSpec()
    net_config = getattr(algorithm, "net_config", None)
    if net_config is not None:
        data["network"] = net_config
    return data


def write_arena_manifest(
    algo_name: str, output_dir: Path
) -> tuple[Path, dict[str, Any]]:
    """Write one Arena manifest YAML for *algo_name*.

    Returns the output path and the validated JSON payload.
    """
    algorithm = _default_algorithm_spec(algo_name)
    _attach_default_network(algorithm)
    manifest = ArenaManifest.get_validated(
        _build_manifest_dict(algorithm, algo_name), mode="json"
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{algo_name.lower()}.yaml"
    with path.open("w") as fh:
        yaml.safe_dump(manifest, fh, sort_keys=False, default_flow_style=False)
    return path, manifest


def generate_arena_manifests(
    output_dir: Path,
) -> dict[str, tuple[Path, dict[str, Any]]]:
    """Write manifests for all arena algorithms and return ``{name: (path, payload)}``."""
    manifests: dict[str, tuple[Path, dict[str, Any]]] = {}
    for name in arena_algorithm_names():
        manifests[name] = write_arena_manifest(name, output_dir)
    return manifests


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    output_dir = Path(args[0]) if args else Path.cwd() / "arena_manifests"
    for path, _manifest in generate_arena_manifests(output_dir).values():
        print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
