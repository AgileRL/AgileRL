# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Annotated, Literal

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
    :param amplified_gauss_param_mut: Whether a parameter mutation applies its
        amplified ("super") Gaussian band, which perturbs weights at ten times their
        own magnitude. Defaults to False, since that noise can destabilise training;
        switch it on to include it.
    :type amplified_gauss_param_mut: bool
    :param random_reset_param_mut: Whether a parameter mutation applies its
        random-reset Gaussian band, which redraws a weight from N(0, 1) and so discards
        its trained value outright. Defaults to True; switch it off when that proves
        too aggressive.
    :type random_reset_param_mut: bool
    :param dormant_threshold: Normalised GraMa score at or below which a neuron counts as
        dormant. The score is a neuron's mean absolute pre-activation gradient divided by
        its layer's mean. Raising it resets more neurons per mutation.
    :type dormant_threshold: float
    """

    model_config = ConfigDict(extra="forbid")

    probabilities: MutationProbabilities = Field(default_factory=MutationProbabilities)
    rl_hp_selection: dict[str, RLHyperparameter] = Field(default_factory=dict)
    mutation_sd: float = Field(default=0.1, ge=0.0)
    rand_seed: int = Field(default=42, ge=0)
    amplified_gauss_param_mut: bool = False
    random_reset_param_mut: bool = True
    dormant_threshold: float = Field(default=0.01, ge=0.0)


class TournamentSelectionSpec(BaseModel):
    """Pydantic model for TournamentSelection object.

    :param strategy: Discriminator selecting this (tournament) branch of the
        manifest's selection_strategy union. Fixed to "tournament".
    :type strategy: Literal["tournament"]
    :param tournament_size: Size of the tournament.
    :type tournament_size: int
    :param elitism: Whether elitism is enabled.
    :type elitism: bool
    """

    strategy: Literal["tournament"] = "tournament"
    tournament_size: int = Field(default=2, ge=1)
    elitism: bool = True


class MultiFrequencySelectionSpec(BaseModel):
    """Pydantic model for the MultiFrequencySelection object.

    The total population size is configured in the manifest's training block,
    not on this spec. This spec only validates the population-size-independent
    constraints.

    :param strategy: Discriminator selecting this (MF-PBT) branch of the
        manifest's selection_strategy union. Fixed to "multi_frequency".
    :type strategy: Literal["multi_frequency"]
    :param n_subpopulations: Number of subpopulations (>= 2, since MF-PBT migration
        draws from *other* subpopulations).
    :type n_subpopulations: int
    :param n_winners: Agents in the winners bracket (>= 1; default round(0.25 *
        population_size // n_subpopulations)).
    :type n_winners: int | None
    :param n_survivors: Agents in the survivors bracket (>= 0; default 0).
    :type n_survivors: int | None
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

    strategy: Literal["multi_frequency"] = "multi_frequency"
    n_subpopulations: int = Field(default=2, ge=2)
    n_winners: int | None = Field(default=None, ge=1)
    n_survivors: int | None = Field(default=None, ge=0)
    n_open_for_migration: int | None = Field(default=None, ge=1)
    n_losers: int | None = Field(default=None, ge=1)
    evolution_frequency_ratios: list[int] | None = Field(default=None)

    @model_validator(mode="after")
    def _resolve_and_validate_ratios(self) -> Self:
        """Default and validate the frequency ratios (population-size independent)."""
        from agilerl.hpo.multi_frequency import resolve_and_validate_frequency_ratios

        self.evolution_frequency_ratios = resolve_and_validate_frequency_ratios(
            self.evolution_frequency_ratios, self.n_subpopulations
        )
        return self


# Discriminated union for the manifest's selection_strategy block.
SelectionStrategySpec = Annotated[
    TournamentSelectionSpec | MultiFrequencySelectionSpec,
    Field(discriminator="strategy"),
]
