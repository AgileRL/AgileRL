from __future__ import annotations

from typing import Literal

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
    :param arch_mut_type: Architecture-mutation strategy: ``"original"`` (AgileRL's
        default add/remove node/channel/layer) or ``"func_preserving"``
        (function-preserving Net2Net-style *additions* -- new units are added with
        zero outgoing weights and new head layers are identity-initialised).
        Removals are the original random-count positional operator under both
        settings, so the two differ only in how capacity is added.
    :type arch_mut_type: Literal["original", "func_preserving"]
    :param arch_fp_noise: Symmetry-breaking noise scale for function-preserving
        additions (read *only* when ``arch_mut_type == "func_preserving"`` and an
        architecture add fires; completely inert otherwise, so its value is
        irrelevant for ``"original"`` runs). A relative factor ``alpha``: the new
        units' outgoing weights are seeded with ``randn * (alpha * sigma)`` where
        ``sigma`` is the std of the existing outgoing weights in that consuming
        layer. The default ``0.1`` breaks the new units' symmetry so they receive
        incoming-weight gradient and the added capacity is recruitable, at a
        negligible (~1%) function-preservation cost; set ``0.0`` for exact-zero,
        byte-identical preservation.
    :type arch_fp_noise: float
    :param arch_encoder_layer_mut: Whether ``add_layer`` / ``remove_layer`` are
        enabled on the *encoder* as well as the head. AgileRL disables encoder
        layer mutations by default because restructuring the encoder resets the
        representation feeding every head, which adds a lot of variance; a
        function-preserving deepening injects no such shock, so the default here
        is ``None`` -> ``arch_mut_type == "func_preserving"``. Set it explicitly
        to compare arms on an equal search space (an ``"original"`` baseline
        needs ``true`` to match a ``"func_preserving"`` arm). Only takes effect
        for **MLP** encoders; see :class:`EvolvableNetwork
        <agilerl.networks.base.EvolvableNetwork>`.
    :type arch_encoder_layer_mut: bool | None
    """

    model_config = ConfigDict(extra="forbid")

    probabilities: MutationProbabilities = Field(default_factory=MutationProbabilities)
    rl_hp_selection: dict[str, RLHyperparameter] = Field(default_factory=dict)
    mutation_sd: float = Field(default=0.1, ge=0.0)
    rand_seed: int = Field(default=42, ge=0)
    mutate_elite: bool = False
    arch_mut_type: Literal["original", "func_preserving"] = "original"
    arch_fp_noise: float = Field(default=0.1, ge=0.0)
    arch_encoder_layer_mut: bool | None = None

    def encoder_layer_mutations_enabled(self) -> bool:
        """Resolve whether encoder layer mutations should be enabled.

        ``arch_encoder_layer_mut`` is tri-state: an explicit ``True``/``False``
        wins, while ``None`` derives the value from the mutation strategy so that
        function-preserving runs get encoder deepening without a second knob.

        :return: Whether to enable encoder ``add_layer`` / ``remove_layer``.
        :rtype: bool
        """
        if self.arch_encoder_layer_mut is not None:
            return self.arch_encoder_layer_mut

        return self.arch_mut_type == "func_preserving"


class TournamentSelectionSpec(BaseModel):
    """Pydantic model for TournamentSelection object.

    :param tournament_size: Size of the tournament.
    :type tournament_size: int
    :param elitism: Whether elitism is enabled.
    :type elitism: bool
    """

    tournament_size: int = Field(default=2, ge=2)
    elitism: bool = True
