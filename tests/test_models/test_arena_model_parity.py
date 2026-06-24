"""Parity tests between core ``agilerl.models`` and ``agilerl.arena.models``.

The check is directional: every field on an Arena spec must also exist on the
corresponding core spec (Arena fields ⊆ core fields). The reverse is *not*
required — ``agilerl.arena.models`` is a deliberately decoupled, torch-free
fork, so core may carry trainer-side fields the Arena manifest does not expose.
Arena-only fields remain a violation, and shared fields must agree on defaults.
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel

from agilerl import HAS_ARENA_DEPENDENCIES, HAS_LLM_DEPENDENCIES
from agilerl.models import ALGO_REGISTRY
from agilerl.models.hpo import (
    MutationProbabilities,
    MutationSpec,
    TournamentSelectionSpec,
)
from agilerl.models.training import (
    NStepBufferArgs,
    PerBufferArgs,
    ReplayBufferSpec,
    TrainingSpec,
)

requires_arena = pytest.mark.skipif(
    not HAS_ARENA_DEPENDENCIES, reason="agilerl-arena is not installed"
)

# Fields present on core specs for local training but omitted from Arena manifests.
_CORE_ONLY_ALGO_FIELDS = frozenset(
    {
        "actor_network",
        "critic_network",
        "critic_networks",
        "actor_networks",
        "cudagraphs",
    }
)


def _arena_registry():
    from agilerl.arena.models import ARENA_REGISTRY

    return ARENA_REGISTRY


def shared_algorithm_names() -> list[str]:
    """Algorithms registered in both core and arena model registries."""
    if not HAS_ARENA_DEPENDENCIES:
        return []
    arena_registry = _arena_registry()
    return sorted(set(ALGO_REGISTRY._entries) & set(arena_registry._entries))


def arena_only_algorithm_names() -> list[str]:
    """Algorithms registered for Arena but not in the core registry."""
    if not HAS_ARENA_DEPENDENCIES:
        return []
    arena_registry = _arena_registry()
    return sorted(set(arena_registry._entries) - set(ALGO_REGISTRY._entries))


def core_only_algorithm_names() -> list[str]:
    """Algorithms registered for local/core training but not on Arena."""
    if not HAS_ARENA_DEPENDENCIES:
        return []
    arena_registry = _arena_registry()
    return sorted(set(ALGO_REGISTRY._entries) - set(arena_registry._entries))


def _serializable_field_names(model_cls: type[BaseModel]) -> set[str]:
    return {name for name, field in model_cls.model_fields.items() if not field.exclude}


def _default_json_dump(
    model_cls: type[BaseModel], *, exclude: frozenset[str] = frozenset()
) -> dict[str, Any]:
    """Return per-field defaults for *model_cls* in JSON-compatible form.

    Reads ``model_fields`` directly so specs with required fields (no default)
    can still be compared on the fields that *do* have defaults.
    """
    from pydantic_core import PydanticUndefined

    defaults: dict[str, Any] = {}
    for name, field in model_cls.model_fields.items():
        if name in exclude or field.exclude:
            continue
        if field.default is not PydanticUndefined:
            value = field.default
        elif field.default_factory is not None:
            try:
                value = field.default_factory()
            except TypeError:
                # ``default_factory`` may take a ``data`` arg in pydantic v2.10+;
                # skip rather than fail the parity check on such fields.
                continue
        else:
            # Required field with no default — skipped; defaults can't differ.
            continue
        if value is None:
            continue
        if isinstance(value, BaseModel):
            value = value.model_dump(mode="json", exclude_none=True)
        defaults[name] = value
    return defaults


def _assert_field_name_parity(
    core_cls: type[BaseModel],
    arena_cls: type[BaseModel],
    *,
    core_exclude: frozenset[str] = frozenset(),
    label: str,
) -> None:
    core_fields = _serializable_field_names(core_cls) - core_exclude
    arena_fields = _serializable_field_names(arena_cls)
    arena_only = arena_fields - core_fields
    assert not arena_only, (
        f"{label}: Arena spec defines fields absent from core: "
        f"{sorted(arena_only)}. Arena fields must be a subset of core fields."
    )


def _assert_model_parity(
    core_cls: type[BaseModel],
    arena_cls: type[BaseModel],
    *,
    core_exclude: frozenset[str] = frozenset(),
    label: str,
) -> None:
    core_fields = _serializable_field_names(core_cls) - core_exclude
    arena_fields = _serializable_field_names(arena_cls)
    arena_only = arena_fields - core_fields
    assert not arena_only, (
        f"{label}: Arena spec defines fields absent from core: "
        f"{sorted(arena_only)}. Arena fields must be a subset of core fields."
    )

    core_dump = _default_json_dump(core_cls, exclude=core_exclude)
    arena_dump = _default_json_dump(arena_cls)
    arena_only_defaults = set(arena_dump) - set(core_dump)
    assert not arena_only_defaults, (
        f"{label}: Arena defines default-bearing fields absent from core: "
        f"{sorted(arena_only_defaults)}"
    )
    for key in arena_dump:
        assert core_dump[key] == arena_dump[key], (
            f"{label}.{key}: default values differ "
            f"(core={core_dump[key]!r}, arena={arena_dump[key]!r})"
        )


@requires_arena
class TestArenaModelRegistryParity:
    def test_shared_algorithms_are_non_empty(self) -> None:
        assert shared_algorithm_names(), "Expected at least one shared algorithm"

    def test_arena_algorithms_exist_in_core_registry(self) -> None:
        missing = arena_only_algorithm_names()
        if missing and not HAS_LLM_DEPENDENCIES:
            from agilerl.arena import AgentType

            arena_registry = _arena_registry()
            non_llm_missing = [
                name
                for name in missing
                if arena_registry.get(name).spec_cls.agent_type != AgentType.LLMAgent
            ]
            assert non_llm_missing == [], (
                "Non-LLM Arena algorithms must be registered in core: "
                f"{non_llm_missing}"
            )
            pytest.skip(
                "LLM algorithms are not registered in core without agilerl[llm]"
            )

        assert missing == [], (
            f"Arena algorithms missing from core ALGO_REGISTRY: {missing}"
        )


@requires_arena
class TestArenaSharedAlgorithmSpecParity:
    @pytest.mark.parametrize("name", shared_algorithm_names())
    def test_shared_algorithm_manifest_fields_match(self, name: str) -> None:
        arena_registry = _arena_registry()
        core_cls = ALGO_REGISTRY.get(name).spec_cls
        arena_cls = arena_registry.get(name).spec_cls
        _assert_model_parity(
            core_cls,
            arena_cls,
            core_exclude=_CORE_ONLY_ALGO_FIELDS,
            label=name,
        )


@requires_arena
class TestArenaSharedSupportModelParity:
    @pytest.mark.parametrize(
        ("label", "core_cls", "arena_import", "arena_attr"),
        [
            ("TrainingSpec", TrainingSpec, "agilerl.arena.models", "TrainingSpec"),
            (
                "ReplayBufferSpec",
                ReplayBufferSpec,
                "agilerl.arena.models",
                "ReplayBufferSpec",
            ),
            (
                "NStepBufferArgs",
                NStepBufferArgs,
                "agilerl.arena.models.training",
                "NStepBufferArgs",
            ),
            (
                "PerBufferArgs",
                PerBufferArgs,
                "agilerl.arena.models.training",
                "PerBufferArgs",
            ),
            ("MutationSpec", MutationSpec, "agilerl.arena.models", "MutationSpec"),
            (
                "MutationProbabilities",
                MutationProbabilities,
                "agilerl.arena.models.hpo",
                "MutationProbabilities",
            ),
            (
                "TournamentSelectionSpec",
                TournamentSelectionSpec,
                "agilerl.arena.models",
                "TournamentSelectionSpec",
            ),
        ],
    )
    def test_support_model_defaults_match(
        self,
        label: str,
        core_cls: type[BaseModel],
        arena_import: str,
        arena_attr: str,
    ) -> None:
        import importlib

        arena_cls = getattr(importlib.import_module(arena_import), arena_attr)
        _assert_model_parity(core_cls, arena_cls, label=label)

    @pytest.mark.parametrize(
        "label",
        [
            "MlpSpec",
            "CnnSpec",
            "LstmSpec",
            "SimbaSpec",
            "MultiInputSpec",
            "QNetworkSpec",
            "StochasticActorSpec",
            "DeterministicActorSpec",
        ],
    )
    def test_network_model_field_names_match(self, label: str) -> None:
        import importlib

        core_cls = getattr(importlib.import_module("agilerl.models.networks"), label)
        arena_cls = getattr(
            importlib.import_module("agilerl.arena.models.networks"), label
        )
        _assert_field_name_parity(core_cls, arena_cls, label=label)
