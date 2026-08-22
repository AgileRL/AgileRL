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
    :param random_reset_param_mut: Whether the Gaussian parameter mutation keeps its
        random-reset band, which *replaces* a selected weight with a fresh
        ``N(0, 1)`` draw rather than perturbing it. Disabling it leaves that band's
        weights at their trained values; the band's probability mass is **not**
        redistributed to the other bands.
    :type random_reset_param_mut: bool
    :param amplified_gauss_param_mut: Whether the Gaussian parameter mutation keeps
        its amplified ("super") noise band, whose standard deviation is 10x the
        weight's magnitude. Disabling it leaves that band's weights at their trained
        values; the band's probability mass is **not** redistributed.
    :type amplified_gauss_param_mut: bool
    :param regrama_param_mut: Whether the parameter mutation resets dormant neurons
        (ReGraMa, "Measure gradients, not activations!") before the Gaussian pass,
        instead of relying on Gaussian noise alone. ``mutation_sd`` still scales the
        Gaussian pass's ordinary noise either way.
    :type regrama_param_mut: bool
    :param dormant_tau: ReGraMa dormancy threshold (a neuron with normalised score
        ``<= dormant_tau`` is dormant and is reset). Independent of the diagnostic
        ``training.dormant_tau``; only used when ``regrama_param_mut`` is set.
    :type dormant_tau: float
    :param regrama_out_scale: ReGraMa revival strength. A Xavier-reset neuron's
        outgoing weights are re-seeded at this fraction of the consumer layer's
        live column scale instead of being zeroed, so the revived neuron has a
        non-zero gradient (both for its own score and for its incoming weights).
        A scale below ``dormant_tau`` risks the revived neuron being re-flagged as
        dormant before it learns anything; the default trades that against the size
        of the perturbation, and was picked from a PPO/Hopper-v4 sweep (one seed) in
        which 0.02 beat the whole 0.05--0.25 range by a wide margin.
        ``0.0`` restores the zeroed-outgoing behaviour.
        Only used when ``regrama_param_mut`` is set.
    :type regrama_out_scale: float
    """

    model_config = ConfigDict(extra="forbid")

    probabilities: MutationProbabilities = Field(default_factory=MutationProbabilities)
    rl_hp_selection: dict[str, RLHyperparameter] = Field(default_factory=dict)
    mutation_sd: float = Field(default=0.1, ge=0.0)
    rand_seed: int = Field(default=42, ge=0)
    mutate_elite: bool = False
    random_reset_param_mut: bool = True
    amplified_gauss_param_mut: bool = True
    regrama_param_mut: bool = False
    dormant_tau: float = Field(default=0.1, gt=0.0)
    regrama_out_scale: float = Field(default=0.02, ge=0.0)


class TournamentSelectionSpec(BaseModel):
    """Pydantic model for TournamentSelection object.

    :param tournament_size: Size of the tournament.
    :type tournament_size: int
    :param elitism: Whether elitism is enabled.
    :type elitism: bool
    """

    tournament_size: int = Field(default=2, ge=2)
    elitism: bool = True
