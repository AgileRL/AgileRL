from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing_extensions import Self

if TYPE_CHECKING:
    from agilerl.models.training import TrainingSpec


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


class MultiFrequencySelectionSpec(BaseModel):
    """Pydantic model for the MultiFrequencySelection object.

    :param selection_strategy: Discriminator selecting this (MF-PBT) branch of the
        manifest's tournament_selection union. Fixed to "multi_frequency".
    :type selection_strategy: Literal["multi_frequency"]
    :param n_subpopulations: Number of subpopulations (>= 2, since MF-PBT migration
        draws from *other* subpopulations).
    :type n_subpopulations: int
    :param n_individuals_per_subpopulation: Agents in each subpopulation (>= 3,
        since a valid four-bracket partition needs at least one winner, one
        open-for-migration and one loser slot).
    :type n_individuals_per_subpopulation: int
    :param n_winners: Agents in the winners bracket (default round(0.25 *
        n_individuals_per_subpopulation)).
    :type n_winners: int | None
    :param n_survivors: Agents in the survivors bracket.
    :type n_survivors: int
    :param n_open_for_migration: Agents in the open-for-migration bracket (default
        round(0.25 * n_individuals_per_subpopulation)).
    :type n_open_for_migration: int | None
    :param n_losers: Agents in the losers bracket (default n_individuals -
        n_winners - n_survivors - n_open_for_migration).
    :type n_losers: int | None
    :param evolution_frequency_ratios: Per-subpopulation evolution-frequency ratios
        (strictly increasing integers, each >= 1; one per subpopulation; default
        [1, 5, 10, ...]).
    :type evolution_frequency_ratios: list[int] | None
    """

    model_config = ConfigDict(extra="forbid")

    selection_strategy: Literal["multi_frequency"] = "multi_frequency"
    n_subpopulations: int = Field(default=2, ge=2)
    n_individuals_per_subpopulation: int = Field(default=8, ge=3)
    n_winners: int | None = Field(default=None, ge=1)
    n_survivors: int = Field(default=0, ge=0)
    n_open_for_migration: int | None = Field(default=None, ge=1)
    n_losers: int | None = Field(default=None, ge=1)
    evolution_frequency_ratios: list[int] | None = Field(default=None)

    @model_validator(mode="after")
    def _resolve_and_validate(self) -> Self:
        """Resolve the None defaults, then hard-check the operator's invariants."""
        n_ind = self.n_individuals_per_subpopulation

        # Resolve the None defaults
        if self.n_winners is None:
            self.n_winners = round(0.25 * n_ind)
        if self.n_open_for_migration is None:
            self.n_open_for_migration = round(0.25 * n_ind)
        if self.n_losers is None:
            self.n_losers = (
                n_ind - self.n_winners - self.n_survivors - self.n_open_for_migration
            )
        if self.evolution_frequency_ratios is None:
            self.evolution_frequency_ratios = [1] + [
                5 * i for i in range(1, self.n_subpopulations)
            ]

        # The derived default value of n_losers can fall to <= 0, so a guard is kept
        if self.n_losers < 1:
            msg = f"n_losers must be >= 1, got {self.n_losers}."
            raise ValueError(msg)
        bracket_sum = (
            self.n_winners
            + self.n_survivors
            + self.n_open_for_migration
            + self.n_losers
        )
        if bracket_sum != n_ind:
            msg = (
                f"n_winners + n_survivors + n_open_for_migration + n_losers "
                f"({bracket_sum}) must equal n_individuals_per_subpopulation "
                f"({n_ind})."
            )
            raise ValueError(msg)
        ratios = self.evolution_frequency_ratios
        if len(ratios) != self.n_subpopulations:
            msg = (
                f"evolution_frequency_ratios must have length n_subpopulations "
                f"({self.n_subpopulations}), got {len(ratios)}."
            )
            raise ValueError(msg)
        if any(r < 1 for r in ratios):
            msg = "Each evolution_frequency_ratio must be >= 1."
            raise ValueError(msg)
        if any(ratios[i] >= ratios[i + 1] for i in range(len(ratios) - 1)):
            msg = "evolution_frequency_ratios must be strictly increasing."
            raise ValueError(msg)
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


def resolve_multi_frequency_selection_pop_size(
    multi_frequency_selection_spec: MultiFrequencySelectionSpec | None,
    training: TrainingSpec,
) -> None:
    """Derive and enforce the MF-PBT population size on a training spec, in place.

    Under MF-PBT the population size is not configured directly: it is fully determined
    by the subpopulation layout as n_subpopulations *
    n_individuals_per_subpopulation. This writes that derived value onto *training*
    and rejects an explicit pop_size (or its population_size alias) that contradicts it.

    :param multi_frequency_selection_spec: MF-PBT spec, or None when unset.
    :type multi_frequency_selection_spec: MultiFrequencySelectionSpec | None
    :param training: Training spec updated in place; its pop_size is set to the
        derived value.
    :type training: ~agilerl.models.training.TrainingSpec
    :raises ValueError: If an explicitly-set pop_size conflicts with the derived
        value.
    """
    if multi_frequency_selection_spec is None:
        return

    derived = (
        multi_frequency_selection_spec.n_subpopulations
        * multi_frequency_selection_spec.n_individuals_per_subpopulation
    )
    pop_size_set = (
        "pop_size" in training.model_fields_set
        or "population_size" in training.model_fields_set
    )
    if pop_size_set and training.pop_size != derived:
        msg = (
            f"'pop_size' ({training.pop_size}) conflicts with the MF-PBT "
            "derived value n_subpopulations * n_individuals_per_subpopulation "
            f"= {derived}. Omit 'pop_size' when 'multi_frequency_selection' is configured; it is "
            "derived automatically."
        )
        raise ValueError(msg)
    training.pop_size = derived
