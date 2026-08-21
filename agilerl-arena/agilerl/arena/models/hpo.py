# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Literal

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
    :param dormant_reset_param_mut: Whether a parameter mutation first resets the
        neurons that have gone dormant (ReGraMa) before adding Gaussian noise.
        Declared so a core manifest carrying it is accepted rather than rejected as
        an unknown field, and excluded so the platform payload is unaffected. Arena
        runs the Gaussian parameter operator only, so requesting ReGraMa fails
        validation here instead of silently degrading.
    :type dormant_reset_param_mut: bool
    :param amplified_gauss_param_mut: Whether a parameter mutation applies its
        amplified ("super") Gaussian band. Declared and excluded like
        dormant_reset_param_mut; disabling it is rejected for the same reason.
    :type amplified_gauss_param_mut: bool
    :param random_reset_param_mut: Whether a parameter mutation applies its
        random-reset Gaussian band. Declared and excluded like
        dormant_reset_param_mut; disabling it is rejected for the same reason.
    :type random_reset_param_mut: bool
    :param dormant_threshold: Normalised GraMa score at or below which a neuron counts as
        dormant. Declared and excluded like dormant_reset_param_mut, and needs no
        rejection of its own since it is inert once ReGraMa is refused.
    :type dormant_threshold: float
    """

    model_config = ConfigDict(extra="forbid")

    probabilities: MutationProbabilities = Field(default_factory=MutationProbabilities)
    rl_hp_selection: dict[str, RLHyperparameter] = Field(default_factory=dict)
    mutation_sd: float = Field(default=0.1, ge=0.0)
    rand_seed: int = Field(default=42, ge=0)
    dormant_reset_param_mut: bool = Field(default=False, exclude=True)
    amplified_gauss_param_mut: bool = Field(default=True, exclude=True)
    random_reset_param_mut: bool = Field(default=True, exclude=True)
    dormant_threshold: float = Field(default=0.01, ge=0.0, exclude=True)

    @model_validator(mode="after")
    def _reject_unsupported_param_mutation(self) -> Self:
        """Reject parameter-mutation settings the platform does not support.

        ReGraMa is not implemented in Arena, so a manifest requesting it or
        requesting that either Gaussian band be dropped fails here rather than
        training with the default Gaussian operator without saying so.

        :returns: The validated spec.
        :rtype: MutationSpec
        :raises ValueError: If ReGraMa is enabled or either the amplified or the
            random-reset Gaussian band is disabled.
        """
        if (
            self.dormant_reset_param_mut
            or not self.amplified_gauss_param_mut
            or not self.random_reset_param_mut
        ):
            msg = (
                "The Arena platform only supports the default Gaussian parameter "
                "mutations: enabling ReGraMa or disabling the amplified or "
                "random-reset Gaussian mutations is not available."
            )
            raise ValueError(msg)
        return self


class TournamentSelectionSpec(BaseModel):
    """Pydantic model for TournamentSelection object.

    :param strategy: Discriminator carried by core manifests, whose
        ``selection_strategy`` block is a union over selection strategies. Declared
        so it is accepted rather than reported as an unknown field, and excluded so
        the platform payload is unaffected. Arena runs tournament selection only, so
        a non-tournament block fails validation here instead of silently degrading.
    :type strategy: Literal["tournament"]
    :param tournament_size: Size of the tournament.
    :type tournament_size: int
    :param elitism: Whether elitism is enabled.
    :type elitism: bool
    """

    strategy: Literal["tournament"] = Field(default="tournament", exclude=True)
    tournament_size: int = Field(default=2, ge=2)
    elitism: bool = True
