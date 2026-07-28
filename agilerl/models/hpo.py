from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing_extensions import Self


class RLHyperparameter(BaseModel):
    """Min/max range and mutation factors for a single RL hyperparameter.

    :param min: Minimum value of the hyperparameter.
    :type min: float
    :param max: Maximum value of the hyperparameter.
    :type max: float
    :param grow_factor: Factor by which the hyperparameter will be grown during mutation.
    :type grow_factor: float
    :param shrink_factor: Factor by which the hyperparameter will be shrunk during mutation.
    :type shrink_factor: float
    """

    min: float
    max: float
    grow_factor: float = Field(default=1.2, ge=1.0)
    shrink_factor: float = Field(default=0.8, ge=0.0, le=1.0)


class MutationProbabilities(BaseModel):
    """Mutation probability distribution.

    :param no_mut: Probability of no mutation.
    :type no_mut: float
    :param arch_mut: Probability of architecture mutation.
    :type arch_mut: float
    :param new_layer: Probability of new layer mutation.
    :type new_layer: float
    :param params_mut: Probability of parameters mutation.
    :type params_mut: float
    :param act_mut: Probability of activation mutation.
    :type act_mut: float
    :param rl_hp_mut: Probability of RL hyperparameter mutation.
    :type rl_hp_mut: float
    """

    no_mut: float = Field(default=0.4, ge=0.0, le=1.0)
    arch_mut: float = Field(default=0.2, ge=0.0, le=1.0)
    new_layer: float = Field(default=0.2, ge=0.0, le=1.0)
    params_mut: float = Field(default=0.2, ge=0.0, le=1.0)
    act_mut: float = Field(default=0.0, ge=0.0, le=1.0)
    rl_hp_mut: float = Field(default=0.2, ge=0.0, le=1.0)


class MutationSpec(BaseModel):
    """Pydantic model for Mutations object.

    :param probabilities: Probability distribution for the mutations.
    :type probabilities: MutationProbabilities
    :param rl_hp_selection: RL hyperparameters to mutate.
    :type rl_hp_selection: dict[str, RLHyperparameter]
    :param mutation_sd: Standard deviation of the mutation.
    :type mutation_sd: float
    :param rand_seed: Random seed for repeatability.
    :type rand_seed: int
    """

    model_config = ConfigDict(extra="forbid")

    probabilities: MutationProbabilities = Field(default_factory=MutationProbabilities)
    rl_hp_selection: dict[str, RLHyperparameter] = Field(default_factory=dict)
    mutation_sd: float = Field(default=0.1, ge=0.0)
    rand_seed: int = Field(default=42, ge=0)


class TournamentSelectionSpec(BaseModel):
    """Pydantic model for TournamentSelection object.

    :param selection_strategy: Discriminator selecting this (tournament) branch of the
        manifest's tournament_selection union. Fixed to "tournament".
    :type selection_strategy: Literal["tournament"]
    :param tournament_size: Size of the tournament.
    :type tournament_size: int
    :param elitism: Whether elitism is enabled.
    :type elitism: bool
    """

    selection_strategy: Literal["tournament"] = "tournament"
    tournament_size: int = Field(default=2, ge=2)
    elitism: bool = True


def resolve_and_validate_frequency_ratios(
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


class MultiFrequencySelectionSpec(BaseModel):
    """Pydantic model for the MultiFrequencySelection object.

    The total population size is configured in the manifest's training block,
    not on this spec. This spec only validates the population-size-independent
    constraints.

    :param selection_strategy: Discriminator selecting this (MF-PBT) branch of the
        manifest's tournament_selection union. Fixed to "multi_frequency".
    :type selection_strategy: Literal["multi_frequency"]
    :param n_subpopulations: Number of subpopulations (>= 2, since MF-PBT migration
        draws from *other* subpopulations).
    :type n_subpopulations: int
    :param n_winners: Agents in the winners bracket (>= 1; default round(0.25 *
        population_size // n_subpopulations)).
    :type n_winners: int | None
    :param n_survivors: Agents in the survivors bracket (>= 0).
    :type n_survivors: int
    :param n_open_for_migration: Agents in the open-for-migration bracket (>= 1;
        default round(0.25 * population_size // n_subpopulations)).
    :type n_open_for_migration: int | None
    :param n_losers: Agents in the losers bracket (>= 1; default the remainder
        population_size // n_subpopulations - n_winners - n_survivors -
        n_open_for_migration).
    :type n_losers: int | None
    :param evolution_frequency_ratios: Per-subpopulation evolution-frequency ratios
        (strictly increasing integers, each >= 1; one per subpopulation; default
        [1, 5, 10, ...]).
    :type evolution_frequency_ratios: list[int] | None
    """

    model_config = ConfigDict(extra="forbid")

    selection_strategy: Literal["multi_frequency"] = "multi_frequency"
    n_subpopulations: int = Field(default=2, ge=2)
    n_winners: int | None = Field(default=None, ge=1)
    n_survivors: int = Field(default=0, ge=0)
    n_open_for_migration: int | None = Field(default=None, ge=1)
    n_losers: int | None = Field(default=None, ge=1)
    evolution_frequency_ratios: list[int] | None = Field(default=None)

    @model_validator(mode="after")
    def _resolve_and_validate_ratios(self) -> Self:
        """Default and validate the frequency ratios (population-size independent)."""
        self.evolution_frequency_ratios = resolve_and_validate_frequency_ratios(
            self.evolution_frequency_ratios, self.n_subpopulations
        )
        return self


def default_selection_strategy(value: Any) -> Any:
    """Inject the default selection_strategy for config dicts that omit it.

    :param value: Raw tournament_selection value from the manifest.
    :type value: Any
    :returns: The value with selection_strategy defaulted to "tournament" when it
        was a discriminator-less dict; otherwise the value unchanged.
    :rtype: Any
    """
    if isinstance(value, dict) and "selection_strategy" not in value:
        return {**value, "selection_strategy": "tournament"}
    return value


SelectionStrategySpec = Annotated[
    TournamentSelectionSpec | MultiFrequencySelectionSpec,
    Field(discriminator="selection_strategy"),
]
"""Discriminated union for the manifest's tournament_selection block."""


def split_selection_spec(
    selection: TournamentSelectionSpec | MultiFrequencySelectionSpec | None,
) -> tuple[TournamentSelectionSpec | None, MultiFrequencySelectionSpec | None]:
    """Split a unified tournament_selection value into the two trainer kwargs.

    :param selection: The resolved tournament_selection spec, or None when unset.
    :type selection: TournamentSelectionSpec | MultiFrequencySelectionSpec | None
    :returns: (tournament_spec, multi_frequency_selection_spec) with at most one set.
    :rtype: tuple[TournamentSelectionSpec | None, MultiFrequencySelectionSpec | None]
    """
    if isinstance(selection, MultiFrequencySelectionSpec):
        return None, selection
    return selection, None


def check_selection_strategy_exclusive(
    tournament_selection: TournamentSelectionSpec | None,
    multi_frequency_selection_spec: MultiFrequencySelectionSpec | None,
) -> None:
    """Reject configuring MF-PBT and tournament selection together in the trainer.

    :param tournament_selection: Tournament-selection spec, or None when unset.
    :type tournament_selection: TournamentSelectionSpec | None
    :param multi_frequency_selection_spec: MF-PBT spec, or None when unset.
    :type multi_frequency_selection_spec: MultiFrequencySelectionSpec | None
    :raises ValueError: If both strategies are configured simultaneously.
    """
    if multi_frequency_selection_spec is not None and tournament_selection is not None:
        msg = "Cannot set both 'tournament_selection' and 'multi_frequency_selection'."
        raise ValueError(msg)
