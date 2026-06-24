from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


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


class CrossoverSpec(BaseModel):
    """Pydantic model for the :class:`Crossover` (recombination) operator, an
    alternative to tournament selection in the evolutionary HPO loop.

    :param num_parents: Number of top agents (by fitness) that form the
        recombination pool. The paper uses ~80% of the population (e.g. 13 for a
        population of 16). Must not exceed the population size (validated when the
        :class:`Crossover` object is built, where the population size is known).
    :type num_parents: int
    :param swap_prob: Per-section probability of exchanging hyperparameter genes
        between the two parent chromosomes during the two-point crossover.
    :type swap_prob: float
    :param elitism: Whether the top ``number_of_elites`` agents are cloned
        unchanged into the next generation.
    :type elitism: bool
    :param number_of_elites: Number of highest-fitness agents preserved exactly
        when ``elitism`` is True. The elite returned for checkpointing is always
        the single best agent regardless of this value. Defaults to 1.
    :type number_of_elites: int
    :param rand_seed: Random seed for reproducible recombination.
    :type rand_seed: int
    """

    model_config = ConfigDict(extra="forbid")

    num_parents: int = Field(default=2, ge=2)
    swap_prob: float = Field(default=0.7, ge=0.0, le=1.0)
    elitism: bool = True
    number_of_elites: int = Field(default=1, ge=1)
    rand_seed: int = Field(default=42, ge=0)
