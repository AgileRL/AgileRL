# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Network specs: the manifest contract, plus what the trainer needs at runtime."""

from __future__ import annotations

from typing import Literal

from gymnasium import spaces

from agilerl.arena.models.networks import (
    CnnLstmSpec,
    CnnSpec,
    ContinuousQNetworkSpec,
    DeterministicActorSpec,
    EncoderType,
    FinetuningNetworkSpec,
    LoraConfigDict,
    LstmSpec,
    MlpActivation,
    MlpSpec,
    MultiInputSpec,
    NetworkSpec,
    QNetworkSpec,
    RainbowQNetworkSpec,
    SimbaSpec,
    StochasticActorSpec,
    ValueNetworkSpec,
    dump_network_section,
    encoder_spec_for_arch,
    min_max_validator,
    network_arch_is_resolvable,
    normalize_manifest_network,
)

__all__ = [
    "CnnLstmSpec",
    "CnnSpec",
    "ContinuousQNetworkSpec",
    "DeterministicActorSpec",
    "EncoderType",
    "FinetuningNetworkSpec",
    "LoraConfigDict",
    "LstmSpec",
    "MlpActivation",
    "MlpSpec",
    "MultiInputSpec",
    "NetworkSpec",
    "QNetworkSpec",
    "RainbowQNetworkSpec",
    "SimbaSpec",
    "StochasticActorSpec",
    "ValueNetworkSpec",
    "dump_network_section",
    "encoder_spec_for_arch",
    "infer_encoder_arch",
    "min_max_validator",
    "network_arch_is_resolvable",
    "normalize_manifest_network",
]


def infer_encoder_arch(
    observation_space: spaces.Space,
    *,
    recurrent: bool = False,
    simba: bool = False,
) -> Literal["mlp", "cnn", "cnn_lstm", "lstm", "simba", "multiinput"]:
    """Infer the encoder architecture from an observation space.

    Mirrors the branch order in
    :func:`agilerl.utils.evolvable_networks.get_default_encoder_config` and
    :meth:`agilerl.networks.base.EvolvableNetwork._build_encoder` so the schema
    used to validate ``encoder_config`` always matches the encoder that will be
    built. ``simba`` takes precedence over ``recurrent`` on vector spaces.
    On image spaces, ``recurrent`` selects ``cnn_lstm``; ``simba`` does not apply.

    :param observation_space: The (single-agent or per-agent) observation space.
    :param recurrent: Whether the algorithm requests a recurrent encoder.
    :param simba: Whether the network requests a SimBa encoder.
    :returns: One of ``"mlp"``, ``"cnn"``, ``"cnn_lstm"``, ``"lstm"``, ``"simba"``, ``"multiinput"``.
    """
    if isinstance(observation_space, (spaces.Dict, spaces.Tuple)):
        return "multiinput"
    if isinstance(observation_space, spaces.Box) and len(observation_space.shape) == 3:
        return "cnn_lstm" if recurrent else "cnn"
    if simba:
        return "simba"
    if recurrent:
        return "lstm"
    return "mlp"
