# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""The 3.0 CLI, ArenaClient and manifest schema name no classic RL, simulation or HPO concept."""

from __future__ import annotations

import inspect
import re
from typing import TYPE_CHECKING

import click
import pytest

from agilerl.cli import main
from agilerl.client import ArenaClient
from agilerl.models.schema import manifest_schema

if TYPE_CHECKING:
    from collections.abc import Iterator

# Matched at a word start (after splitting camelCase), so ``supported`` and
# ``LLMPPO`` pass while ``gymnasium`` and ``PPOConfig`` do not.
BANNED_TERMS = (
    "simulation",
    "gym",
    "pettingzoo",
    "dqn",
    "rainbow",
    "ppo",
    "td3",
    "ddpg",
    "maddpg",
    "matd3",
    "ippo",
    "cqn",
    "neural_ucb",
    "neuralucb",
    "neural_ts",
    "neuralts",
    "bandit",
    "population",
    "pop_size",
    "mutation",
    "tournament",
    "evo_steps",
    "evo_epochs",
    "llm_based",
    "classic",
)
BANNED = re.compile(r"(?<![a-z0-9])(?:" + "|".join(map(re.escape, BANNED_TERMS)) + ")")

# (term, location) -> why the hit is legitimate.
ALLOWED = {
    (
        "classic",
        "ArenaClient.get_default_run_spec doc",
    ): "documents the error that tells users classic RL is gone",
    (
        "evo_steps",
        "schema property evo_steps",
    ): "the LLM eval cadence; renaming the field would break every manifest",
    ("llm_based", "arena projects create --llm-based name"): "1.x compat; hidden",
    (
        "simulation",
        "arena projects create --simulation-based name",
    ): "1.x compat; hidden, fails with the reason",
    (
        "simulation",
        "arena projects create --simulation-based opts",
    ): "1.x compat; hidden, fails with the reason",
    ("llm_based", "ArenaClient.create_project params"): "1.x compat",
    ("llm_based", "ArenaClient.create_project doc"): "1.x compat",
    **{
        (term, f"schema property {name}"): (
            "accepted for 1.x manifests; deprecated in the schema, no description"
        )
        for term, name in (
            ("pop_size", "pop_size"),
            ("population", "population_size"),
            ("evo_epochs", "evo_epochs"),
            ("mutation", "mutation"),
            ("tournament", "tournament_selection"),
        )
    },
}


def normalise(text: str) -> str:
    """Split camelCase into words and lowercase.

    :param text: Text to normalise.
    :return: Lowercase text with camelCase word breaks as spaces.
    """
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)
    return re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", text).lower()


def cli_fields(command: click.Command, path: str) -> Iterator[tuple[str, str]]:
    """Yield ``(location, text)`` for a command, its parameters and every subcommand.

    :param command: Click command or group to walk.
    :param path: Command path so far, e.g. ``arena projects``.
    :return: Iterator of location and text pairs.
    """
    yield f"{path} name", command.name or ""
    yield f"{path} help", command.help or ""
    yield f"{path} short_help", command.short_help or ""
    yield f"{path} epilog", command.epilog or ""
    for param in command.params:
        where = f"{path} {param.opts[0] if param.opts else param.name}"
        yield f"{where} name", param.name or ""
        yield f"{where} opts", " ".join(param.opts + param.secondary_opts)
        yield f"{where} help", getattr(param, "help", None) or ""
        if not callable(param.default):
            yield f"{where} default", str(param.default)
        if isinstance(param.type, click.Choice):
            yield f"{where} choices", " ".join(map(str, param.type.choices))
    # The registered tree, gated groups included; list_commands hides them by capability.
    if isinstance(command, click.Group):
        for name, sub in command.commands.items():
            yield from cli_fields(sub, f"{path} {name}")


def client_fields() -> Iterator[tuple[str, str]]:
    """Yield ``(location, text)`` for each public ArenaClient method.

    :return: Iterator of location and text pairs.
    """
    for name, method in inspect.getmembers(ArenaClient, callable):
        if name.startswith("_"):
            continue
        yield f"ArenaClient.{name} name", name
        yield (
            f"ArenaClient.{name} params",
            " ".join(inspect.signature(method).parameters),
        )
        yield f"ArenaClient.{name} doc", inspect.getdoc(method) or ""


def schema_fields(node: object, path: str) -> Iterator[tuple[str, str]]:
    """Yield ``(location, text)`` for every property name, extension and description in a schema.

    A property name's location is the name alone, so one allow-list entry
    covers it wherever the schema repeats it.

    :param node: JSON Schema node to walk.
    :param path: JSON path so far, e.g. ``schema.properties``.
    :return: Iterator of location and text pairs.
    """
    if isinstance(node, list):
        for index, item in enumerate(node):
            yield from schema_fields(item, f"{path}[{index}]")
        return
    if not isinstance(node, dict):
        return
    for key, value in node.items():
        where = f"{path}.{key}"
        if key == "description" and isinstance(value, str):
            yield f"{path} description", value
        elif key == "properties" and isinstance(value, dict):
            for name in value:
                yield f"schema property {name}", name
        elif key.startswith("x-"):
            # Extension keys are names too, e.g. x-ui-widget.
            yield f"{where} name", key
        yield from schema_fields(value, where)


def banned_hits() -> set[tuple[str, str]]:
    """Return every ``(term, location)`` where a banned term appears.

    :return: Set of term and location pairs.
    """
    fields = [
        *cli_fields(main, "arena"),
        *client_fields(),
        *schema_fields(manifest_schema(), "schema"),
    ]
    return {
        (match.group(0), location)
        for location, text in fields
        for match in BANNED.finditer(normalise(text))
    }


class TestPublicSurface:
    def test_names_no_classic_simulation_or_hpo_term(self) -> None:
        hits = banned_hits()

        unexpected = sorted(hits - ALLOWED.keys())

        assert not unexpected, "\n".join(
            f"{term!r} in {location}" for term, location in unexpected
        )

    def test_every_allow_list_entry_still_matches(self) -> None:
        hits = banned_hits()

        stale = sorted(ALLOWED.keys() - hits)

        assert not stale, stale


class TestBannedPattern:
    @pytest.mark.parametrize(
        "text", ["PPO", "PPOConfig", "gymnasium", "pop_size", "--simulation-based"]
    )
    def test_flags_a_banned_word(self, text: str) -> None:
        assert BANNED.search(normalise(text))

    @pytest.mark.parametrize("text", ["supported", "LLMPPO", "llmppo", "GRPO"])
    def test_passes_a_word_that_only_contains_a_term(self, text: str) -> None:
        assert not BANNED.search(normalise(text))
