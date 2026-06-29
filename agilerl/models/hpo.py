from __future__ import annotations

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
    :param mutate_elite: Whether the elite member of the population is itself mutated.
    :type mutate_elite: bool
    """

    model_config = ConfigDict(extra="forbid")

    probabilities: MutationProbabilities = Field(default_factory=MutationProbabilities)
    rl_hp_selection: dict[str, RLHyperparameter] = Field(default_factory=dict)
    mutation_sd: float = Field(default=0.1, ge=0.0)
    rand_seed: int = Field(default=42, ge=0)
    mutate_elite: bool = False


class TournamentSelectionSpec(BaseModel):
    """Pydantic model for TournamentSelection object.

    :param tournament_size: Size of the tournament.
    :type tournament_size: int
    :param elitism: Whether elitism is enabled.
    :type elitism: bool
    """

    tournament_size: int = Field(default=2, ge=2)
    elitism: bool = True


class MFPBTSpec(BaseModel):
    """Pydantic model for the MF-PBT (Multiple-Frequencies PBT) evolution regime.

    MF-PBT replaces tournament-selection + mutation. The population is split into
    ``n_subpopulations`` subpopulations of ``n_individuals_per_subpopulation``
    agents each (so ``pop_size`` is derived, not configured). Each subpopulation
    ``i`` evolves every ``evolution_frequency_ratios[i]`` cycles, and is partitioned
    by fitness rank into four brackets whose sizes must sum to the per-subpopulation
    individual count.

    :param n_subpopulations: Number of subpopulations.
    :type n_subpopulations: int
    :param n_individuals_per_subpopulation: Agents in each subpopulation.
    :type n_individuals_per_subpopulation: int
    :param evolution_frequency_ratios: Per-subpopulation evolution-frequency ratios
        (strictly increasing integers, each ``>= 1``; one per subpopulation).
    :type evolution_frequency_ratios: list[int]
    :param n_winners: Agents in the winners bracket.
    :type n_winners: int
    :param n_survivors: Agents in the survivors bracket.
    :type n_survivors: int
    :param n_open_for_migration: Agents in the open-for-migration bracket.
    :type n_open_for_migration: int
    :param n_losers: Agents in the losers bracket.
    :type n_losers: int
    :param rand_seed: Random seed for reproducible winner-clone selection.
    :type rand_seed: int
    """

    model_config = ConfigDict(extra="forbid")

    n_subpopulations: int = Field(default=4, ge=1)
    n_individuals_per_subpopulation: int = Field(default=4, ge=1)
    evolution_frequency_ratios: list[int] = Field(default_factory=lambda: [1, 2, 4, 8])
    n_winners: int = Field(default=1, ge=0)
    n_survivors: int = Field(default=1, ge=0)
    n_open_for_migration: int = Field(default=1, ge=0)
    n_losers: int = Field(default=1, ge=0)
    rand_seed: int = Field(default=42, ge=0)

    @model_validator(mode="after")
    def _validate_mf_pbt(self) -> Self:
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
        bracket_sum = (
            self.n_winners
            + self.n_survivors
            + self.n_open_for_migration
            + self.n_losers
        )
        if bracket_sum != self.n_individuals_per_subpopulation:
            msg = (
                f"n_winners + n_survivors + n_open_for_migration + n_losers "
                f"({bracket_sum}) must equal n_individuals_per_subpopulation "
                f"({self.n_individuals_per_subpopulation})."
            )
            raise ValueError(msg)
        # The winners bracket sources the replacement clones (evolution) and supplies
        # the subpopulation elite (migration), so it cannot be empty when either fires.
        if self.n_winners < 1 and (self.n_losers > 0 or self.n_open_for_migration > 0):
            msg = (
                "n_winners must be >= 1 when n_losers > 0 (winner-clones replace "
                "losers) or n_open_for_migration > 0 (the top winner is the "
                "subpopulation elite used for migration)."
            )
            raise ValueError(msg)
        return self
