# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""The manifest's ``mutation`` and ``selection_strategy`` sections."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing_extensions import Self

from agilerl.arena.models.networks import min_max_validator


class RLHyperparameter(BaseModel):
    """Min/max range and mutation factors for a single RL hyperparameter."""

    model_config = ConfigDict(extra="forbid")

    min: float = Field(description="Lower bound mutation may not go below.")
    max: float = Field(description="Upper bound mutation may not go above.")
    grow_factor: float = Field(
        default=1.2,
        ge=1.0,
        description="Multiplier applied when mutation grows the value.",
    )
    shrink_factor: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Multiplier applied when mutation shrinks the value.",
    )

    @model_validator(mode="after")
    def _validate_range(self) -> Self:
        return min_max_validator("min", "max")(self)


class MutationProbabilities(BaseModel):
    """Mutation probability distribution."""

    model_config = ConfigDict(extra="forbid")

    no_mut: float = Field(
        default=0.4, ge=0.0, le=1.0, description="Weight on leaving an agent unchanged."
    )
    arch_mut: float = Field(
        default=0.2,
        ge=0.0,
        le=1.0,
        description="Weight on resizing a layer of the network.",
    )
    new_layer: float = Field(
        default=0.2, ge=0.0, le=1.0, description="Weight on adding or removing a layer."
    )
    params_mut: float = Field(
        default=0.2,
        ge=0.0,
        le=1.0,
        description="Weight on perturbing the network weights directly.",
    )
    act_mut: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Weight on swapping an activation function.",
    )
    rl_hp_mut: float = Field(
        default=0.2,
        ge=0.0,
        le=1.0,
        description="Weight on mutating one of the rl_hp_selection hyperparameters.",
    )


def no_op_mutation_probabilities() -> MutationProbabilities:
    """Probabilities that leave every agent unchanged."""
    return MutationProbabilities(
        no_mut=1.0,
        arch_mut=0.0,
        new_layer=0.0,
        params_mut=0.0,
        act_mut=0.0,
        rl_hp_mut=0.0,
    )


class NetworkMutationRanges(BaseModel):
    """Architecture-mutation bounds, written alongside the mutation probabilities.

    They constrain the same encoder and head the ``network`` section configures,
    so :class:`~agilerl.arena.models.manifest.TrainingManifest` folds them into it.

    :param min_latent_dim: Smallest latent width architecture mutation may pick.
    :type min_latent_dim: int | None
    :param max_latent_dim: Largest latent width architecture mutation may pick.
    :type max_latent_dim: int | None
    :param encoder: Bounds folded into the network's ``encoder_config``.
    :type encoder: dict[str, Any]
    :param head_net: Bounds folded into the network's ``head_config``.
    :type head_net: dict[str, Any]
    """

    model_config = ConfigDict(extra="forbid")

    min_latent_dim: int | None = Field(
        default=None,
        gt=0,
        description="Smallest latent width architecture mutation may pick.",
    )
    max_latent_dim: int | None = Field(
        default=None,
        gt=1,
        description="Largest latent width architecture mutation may pick.",
    )
    encoder: dict[str, Any] = Field(
        default_factory=dict,
        description="Bounds folded into the network's encoder_config.",
    )
    head_net: dict[str, Any] = Field(
        default_factory=dict,
        description="Bounds folded into the network's head_config.",
    )


class MutationSpec(BaseModel):
    """The manifest's ``mutation`` section."""

    model_config = ConfigDict(extra="forbid")

    name: str | None = Field(
        default=None, description="Label for the mutation operator in logs."
    )
    probabilities: MutationProbabilities = Field(
        default_factory=MutationProbabilities,
        description="Relative weights over the kinds of mutation. Normalized, so they need not sum to 1.",
    )
    rl_hp_selection: dict[str, RLHyperparameter] = Field(
        default_factory=dict,
        description=(
            "Hyperparameters HPO may mutate, keyed by the algorithm field name "
            "(lr, batch_size, learn_step, ...). Empty leaves them all fixed."
        ),
    )
    mutation_sd: float = Field(
        default=0.1,
        ge=0.0,
        description="Standard deviation of the noise added when weights are perturbed.",
    )
    rand_seed: int = Field(
        default=42, ge=0, description="Seed for mutation, for repeatable runs."
    )
    network: NetworkMutationRanges | None = Field(
        default=None,
        description=(
            "Bounds architecture mutation must stay inside. Folded onto the "
            "network section they constrain when the manifest is validated."
        ),
    )
    dormant_threshold: float = Field(
        default=0.01,
        ge=0.0,
        exclude=True,
        description=(
            "Normalised GraMa score at or below which a neuron counts as dormant. "
            "Not forwarded in the platform payload."
        ),
    )


class TournamentSelectionSpec(BaseModel):
    """Tournament branch of the ``selection_strategy`` section."""

    model_config = ConfigDict(extra="forbid")

    strategy: Literal["tournament"] = Field(
        default="tournament", description="Selects tournament selection."
    )
    name: str | None = Field(
        default=None, description="Label for the operator in logs."
    )
    tournament_size: int = Field(
        default=2,
        ge=1,
        description=(
            "Agents sampled per tournament, of which the fittest survives. "
            "Larger tournaments apply more selection pressure."
        ),
    )
    elitism: bool = Field(
        default=True, description="Always carry the fittest agent through unchanged."
    )


def resolve_frequency_ratios(
    evolution_frequency_ratios: list[int] | None,
    n_subpopulations: int,
) -> list[int]:
    """Default and validate the MF-PBT per-subpopulation evolution-frequency ratios.

    :param evolution_frequency_ratios: The configured ratios, or None/[]
        to default to [1, 5, 10, ...].
    :type evolution_frequency_ratios: list[int] | None
    :param n_subpopulations: Number of subpopulations the ratios must cover.
    :type n_subpopulations: int
    :return: A new list of n_subpopulations strictly-increasing ratios >= 1.
    :rtype: list[int]
    :raises ValueError: If the ratios are not n_subpopulations strictly-increasing
        integers >= 1.
    """
    ratios = (
        list(evolution_frequency_ratios)
        if evolution_frequency_ratios
        else [1] + [5 * i for i in range(1, n_subpopulations)]
    )
    if len(ratios) != n_subpopulations:
        msg = (
            f"evolution_frequency_ratios must have length n_subpopulations "
            f"({n_subpopulations}), got {len(ratios)}."
        )
        raise ValueError(msg)
    if any(r < 1 for r in ratios):
        msg = "Each evolution_frequency_ratio must be >= 1."
        raise ValueError(msg)
    if any(ratios[i] >= ratios[i + 1] for i in range(len(ratios) - 1)):
        msg = "evolution_frequency_ratios must be strictly increasing."
        raise ValueError(msg)
    return ratios


def resolve_selection_brackets(
    population_size: int,
    n_subpopulations: int,
    evolution_frequency_ratios: list[int] | None,
    n_winners: int | None,
    n_survivors: int | None,
    n_open_for_migration: int | None,
    n_losers: int | None,
) -> tuple[int, int, list[int], int, int, int, int]:
    """Resolve the MF-PBT bracket defaults, then hard-check the operator's invariants.

    :param population_size: Total population size (>= 6, a multiple of
        n_subpopulations, with population_size // n_subpopulations >= 3).
    :type population_size: int
    :param n_subpopulations: Number of subpopulations (>= 2).
    :type n_subpopulations: int
    :param evolution_frequency_ratios: Per-subpopulation frequency ratios, or
        None/[] for [1, 5, 10, ...].
    :type evolution_frequency_ratios: list[int] | None
    :param n_winners: Winners bracket size, or None for round(0.25 * subpopulation_size).
    :type n_winners: int | None
    :param n_survivors: Survivors bracket size (>= 0; None -> 0).
    :type n_survivors: int | None
    :param n_open_for_migration: Open-for-migration bracket size, or None for
        round(0.25 * subpopulation_size).
    :type n_open_for_migration: int | None
    :param n_losers: Losers bracket size, or None for the remainder.
    :type n_losers: int | None
    :return: The resolved (population_size, n_subpopulations,
        evolution_frequency_ratios, n_winners, n_survivors, n_open_for_migration,
        n_losers).
    :rtype: tuple[int, int, list[int], int, int, int, int]
    :raises ValueError: On any violated invariant.
    """
    if population_size < 6:
        msg = (
            "population_size must be >= 6 (the smallest MF-PBT layout is 2 "
            f"subpopulations of 3 agents), got {population_size}."
        )
        raise ValueError(msg)
    if n_subpopulations < 2:
        msg = f"n_subpopulations must be >= 2, got {n_subpopulations}."
        raise ValueError(msg)
    if population_size % n_subpopulations != 0:
        msg = (
            f"population_size ({population_size}) must be divisible by "
            f"n_subpopulations ({n_subpopulations})."
        )
        raise ValueError(msg)

    subpop_size = population_size // n_subpopulations
    if subpop_size < 3:
        msg = (
            "population_size // n_subpopulations must be >= 3 "
            "so each subpopulation can host at least one "
            "winner, one open-for-migration agent and one loser; got "
            f"population_size={population_size}, "
            f"n_subpopulations={n_subpopulations} -> subpopulation_size={subpop_size}."
        )
        raise ValueError(msg)

    if n_winners is None:
        n_winners = round(0.25 * subpop_size)
    if n_survivors is None:
        n_survivors = 0
    if n_open_for_migration is None:
        n_open_for_migration = round(0.25 * subpop_size)
    if n_losers is None:
        n_losers = subpop_size - n_winners - n_survivors - n_open_for_migration

    if n_winners < 1:
        msg = f"n_winners must be >= 1, got {n_winners}."
        raise ValueError(msg)
    if n_survivors < 0:
        msg = f"n_survivors must be >= 0, got {n_survivors}."
        raise ValueError(msg)
    if n_open_for_migration < 1:
        msg = f"n_open_for_migration must be >= 1, got {n_open_for_migration}."
        raise ValueError(msg)
    if n_losers < 1:
        msg = f"n_losers must be >= 1, got {n_losers}."
        raise ValueError(msg)
    bracket_sum = n_winners + n_survivors + n_open_for_migration + n_losers
    if bracket_sum != subpop_size:
        msg = (
            f"n_winners + n_survivors + n_open_for_migration + n_losers "
            f"({bracket_sum}) must equal population_size // n_subpopulations "
            f"({subpop_size})."
        )
        raise ValueError(msg)

    evolution_frequency_ratios = resolve_frequency_ratios(
        evolution_frequency_ratios, n_subpopulations
    )

    return (
        population_size,
        n_subpopulations,
        evolution_frequency_ratios,
        n_winners,
        n_survivors,
        n_open_for_migration,
        n_losers,
    )


class MultiFrequencySelectionSpec(BaseModel):
    """MF-PBT branch of the ``selection_strategy`` section.

    The total population size is configured in the manifest's training block,
    not on this spec. This spec only validates the population-size-independent
    constraints; :meth:`resolve_brackets` fills in the rest.
    """

    model_config = ConfigDict(extra="forbid")

    strategy: Literal["multi_frequency"] = Field(
        default="multi_frequency", description="Selects multi-frequency PBT."
    )
    name: str | None = Field(
        default=None, description="Label for the operator in logs."
    )
    n_subpopulations: int = Field(
        default=2,
        ge=2,
        description=(
            "Subpopulations the population is split into, each evolving at its "
            "own frequency. At least 2, since migration draws from the others."
        ),
    )
    n_winners: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Top agents per subpopulation whose weights are copied to the "
            "losers. Unset resolves to a quarter of the subpopulation."
        ),
    )
    n_survivors: int | None = Field(
        default=None,
        ge=0,
        description="Agents carried through untouched. Unset resolves to 0.",
    )
    n_open_for_migration: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Agents eligible to be replaced by a winner from another "
            "subpopulation. Unset resolves to a quarter of the subpopulation."
        ),
    )
    n_losers: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Agents replaced from within their own subpopulation. Unset "
            "resolves to whatever the other brackets leave over."
        ),
    )
    evolution_frequency_ratios: list[int] | None = Field(
        default=None,
        description=(
            "How often each subpopulation evolves, relative to the first. "
            "Strictly increasing, one per subpopulation; unset gives "
            "[1, 5, 10, ...] so some subpopulations explore fast and others slow."
        ),
    )

    @model_validator(mode="after")
    def _resolve_ratios(self) -> Self:
        self.evolution_frequency_ratios = resolve_frequency_ratios(
            self.evolution_frequency_ratios, self.n_subpopulations
        )
        return self

    def resolve_brackets(self, population_size: int) -> Self:
        """Fill in the bracket sizes this spec left unset, for *population_size*.

        :param population_size: Total population size from the training section.
        :type population_size: int
        :return: This spec, with every bracket size resolved.
        :rtype: Self
        :raises ValueError: If the resulting layout violates an MF-PBT invariant.
        """
        (
            _,
            _,
            _,
            self.n_winners,
            self.n_survivors,
            self.n_open_for_migration,
            self.n_losers,
        ) = resolve_selection_brackets(
            population_size,
            self.n_subpopulations,
            self.evolution_frequency_ratios,
            self.n_winners,
            self.n_survivors,
            self.n_open_for_migration,
            self.n_losers,
        )
        return self


SelectionStrategySpec = Annotated[
    TournamentSelectionSpec | MultiFrequencySelectionSpec,
    Field(discriminator="strategy"),
]


def mutation_ceiling(
    mutation_spec: MutationSpec | None, name: str, current: int
) -> int:
    """Highest value HPO can mutate ``name`` to, or ``current`` when it cannot.

    Env-host session budgets and collector slot pools are sized once and never
    resized, so anything they depend on is budgeted at this ceiling rather than
    the value the manifest starts at.

    :param mutation_spec: The manifest's mutation section, if any.
    :type mutation_spec: MutationSpec | None
    :param name: The hyperparameter whose ceiling is asked for.
    :type name: str
    :param current: The value the run starts at.
    :type current: int
    :returns: The ceiling.
    :rtype: int
    """
    selection = getattr(mutation_spec, "rl_hp_selection", None) or {}
    if not isinstance(selection, dict):
        return current
    hp_range = selection.get(name)
    if hp_range is None:
        return current
    return max(current, int(getattr(hp_range, "max", current)))
