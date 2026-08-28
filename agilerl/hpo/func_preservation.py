# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from agilerl.modules.custom_components import (
    GumbelSoftmax,
    NoisyLinear,
    ResidualBlock,
    SimbaResidualBlock,
)
from agilerl.networks.base import EvolvableNetwork

# Noise applied to a new unit's outgoing weights, as a
# fraction of the consumer layer's existing column scale.
FP_NOISE_SCALE = 0.02

CONV_LAYER_TYPES = (nn.Conv1d, nn.Conv2d, nn.Conv3d)

# The layers a stream is made of: the ones owning a weight row per unit.
WeightLayer = nn.Conv1d | nn.Conv2d | nn.Conv3d | nn.Linear | NoisyLinear
WEIGHT_LAYER_TYPES = (*CONV_LAYER_TYPES, nn.Linear, NoisyLinear)

# Normalisation between a widened layer and the layer that reads it defeats the
# fade, as most normalisation techniques pool their statistics across units, so
# adding a unit moves every existing one.
NORM_LAYER_TYPES: tuple[type[nn.Module], ...] = (
    nn.LayerNorm,
    nn.GroupNorm,
    nn.BatchNorm1d,
    nn.BatchNorm2d,
    nn.BatchNorm3d,
    nn.InstanceNorm1d,
    nn.InstanceNorm2d,
    nn.InstanceNorm3d,
    nn.RMSNorm,
)

# Activations that mix units together, so a new unit changes its neighbours'
# outputs even when it contributes nothing downstream.
CROSS_UNIT_ACTIVATION_TYPES: tuple[type[nn.Module], ...] = (
    nn.Softmax,
    nn.LogSoftmax,
    nn.Softmin,
    GumbelSoftmax,
)

# Activations under which an identity-initialised layer is itself the identity.
IDENTITY_SAFE_ACTIVATION_TYPES: tuple[type[nn.Module], ...] = (nn.ReLU, nn.Identity)

# Containers hold no units of their own, so they never stand a mutation down.
CONTAINER_TYPES: tuple[type[nn.Module], ...] = (
    nn.Sequential,
    nn.ModuleDict,
    nn.ModuleList,
)

NODE_ADDITIONS = frozenset({"add_node", "add_channel"})
LAYER_ADDITIONS = frozenset({"add_layer"})
LATENT_ADDITIONS = frozenset({"add_latent_node"})
PRESERVED_MUTATIONS = NODE_ADDITIONS | LAYER_ADDITIONS | LATENT_ADDITIONS


def _primary_weight(layer: WeightLayer) -> torch.Tensor:
    """Return the weight tensor that defines a layer's shape.

    :param layer: A weight layer, as reported by :func:`weight_stacks`.
    :type layer: WeightLayer

    :return: The layer's weight, or its mean weight when it is noisy.
    :rtype: torch.Tensor
    """
    return layer.weight_mu if isinstance(layer, NoisyLinear) else layer.weight


def _weight_layers(container: nn.Module) -> list[WeightLayer]:
    """Return the weight layers a container holds, in forward order.

    :param container: Module to scan.
    :type container: nn.Module

    :return: The weight layers, in registration order.
    :rtype: list[WeightLayer]
    """
    return [
        module
        for _, module in container.named_modules()
        if isinstance(module, WEIGHT_LAYER_TYPES)
    ]


def _inner_module(submodule: nn.Module) -> nn.Module:
    """Unwrap an evolvable wrapper down to the module that owns the layers.

    :param submodule: Possibly wrapped module.
    :type submodule: nn.Module

    :return: The module that owns the weight layers.
    :rtype: nn.Module
    """
    wrapped = getattr(submodule, "wrapped", None)
    return submodule if wrapped is None else wrapped


def _container(submodule: nn.Module) -> nn.Module:
    """Return the container that holds a sub-module's weight layers.

    :param submodule: Possibly wrapped module.
    :type submodule: nn.Module

    :return: The container to scan in registration order.
    :rtype: nn.Module
    """
    inner = _inner_module(submodule)
    return getattr(inner, "model", inner)


def _stack_signature(layers: list[WeightLayer]) -> tuple[int, tuple[int, ...]]:
    """Return a stream's shape fingerprint: input width plus hidden widths.

    :param layers: A stream's weight layers, in forward order.
    :type layers: list[WeightLayer]

    :return: The stream's input width and its hidden widths.
    :rtype: tuple[int, tuple[int, ...]]
    """
    return _primary_weight(layers[0]).shape[1], tuple(
        _primary_weight(layer).shape[0] for layer in layers[:-1]
    )


def weight_stacks(submodule: nn.Module) -> list[list[WeightLayer]]:
    """Return the parallel weight-layer streams of a sub-module.

    Most modules have one stream. A duelling head has two (the inherited value
    stream and its sibling advantage stream) and one architecture mutation
    widens or deepens both, so both need the same fixup.

    :param submodule: Module that was mutated.
    :type submodule: nn.Module

    :return: One list of weight layers per stream, each in forward order.
    :rtype: list[list[WeightLayer]]
    """
    inner = _inner_module(submodule)
    primary_container = getattr(inner, "model", inner)
    primary = _weight_layers(primary_container)
    if not primary:
        return []

    stacks: list[list[WeightLayer]] = [primary]
    signature = _stack_signature(primary)
    for child in inner.children():
        if child is primary_container:
            continue
        layers = _weight_layers(child)
        if layers and _stack_signature(layers) == signature:
            stacks.append(layers)

    return stacks


def hidden_widths(submodule: nn.Module) -> list[int]:
    """Snapshot the output width of every non-output weight layer.

    :param submodule: Module about to be mutated.
    :type submodule: nn.Module

    :return: The hidden widths of the primary stream, in forward order.
    :rtype: list[int]
    """
    stacks = weight_stacks(submodule)
    if not stacks:
        return []
    return [_primary_weight(layer).shape[0] for layer in stacks[0][:-1]]


def _modules_between(
    container: nn.Module,
    producer: nn.Module,
    consumer: nn.Module | None = None,
) -> list[nn.Module]:
    """Return the modules a widened output passes through after its producer.

    :param container: Container holding the layers.
    :type container: nn.Module
    :param producer: Layer whose output is being widened.
    :type producer: nn.Module
    :param consumer: Layer that reads the producer's output. When omitted,
        every module registered after the producer is returned.
    :type consumer: nn.Module | None

    :return: The intervening modules, in forward order.
    :rtype: list[nn.Module]
    """
    ordered = [module for _, module in container.named_modules()]
    start = ordered.index(producer)
    stop = len(ordered) if consumer is None else ordered.index(consumer)
    return ordered[start + 1 : stop]


def _interposed_blocker(modules: list[nn.Module]) -> str | None:
    """Return why the modules between producer and consumer prevent preservation.

    :param modules: Modules a value passes through after the producer.
    :type modules: list[nn.Module]

    :return: A reason key, or None when nothing blocks.
    :rtype: str | None
    """
    for module in modules:
        if isinstance(module, NORM_LAYER_TYPES):
            return "norm"
        if isinstance(module, CROSS_UNIT_ACTIVATION_TYPES):
            return "cross_unit_activation"
    return None


def _is_multi_input(module: nn.Module) -> bool:
    """Return whether a module fuses several sub-encoders into one vector.

    :param module: Module to inspect.
    :type module: nn.Module

    :return: Whether the module is a multi-input encoder.
    :rtype: bool
    """
    return hasattr(module, "feature_net") and hasattr(module, "final_dense")


def _structural_blocker(module: nn.Module) -> str | None:
    """Return why widening a layer inside a module is out of scope, if it is.

    Each of these defeats the fade at the point where a hidden layer grows. A
    recurrent core fuses its gate non-linearities, so no single weight matrix
    holds one unit's incoming weights. A residual skip carries a new unit's
    coordinate straight past the layer that would fade it. A multi-input
    encoder's fusion layer reads an interleaved concatenation, so widening a
    sub-encoder does not append columns at the tail.

    :param module: Module about to be mutated.
    :type module: nn.Module

    :return: A reason key, or None when the architecture is supported.
    :rtype: str | None
    """
    for _, child in module.named_modules():
        if isinstance(child, nn.RNNBase):
            return "recurrent"
        if isinstance(child, SimbaResidualBlock):
            return "simba"
        if isinstance(child, ResidualBlock):
            return "residual"
        if _is_multi_input(child):
            return "multi_input"
    return None


def node_addition_blocker(submodule: nn.Module, hidden_layer: int) -> str | None:
    """Return why widening a layer cannot preserve the function, if it cannot.

    :param submodule: Module that was mutated.
    :type submodule: nn.Module
    :param hidden_layer: Index of the widened layer, as the operator reported it.
    :type hidden_layer: int

    :return: A reason key, or None when preservation applies.
    :rtype: str | None
    """
    structural = _structural_blocker(submodule)
    if structural is not None:
        return structural

    stacks = weight_stacks(submodule)
    if not stacks or not 0 <= hidden_layer < len(stacks[0]) - 1:
        return "no_consumer"

    layers = stacks[0]
    between = _modules_between(
        _container(submodule),
        layers[hidden_layer],
        layers[hidden_layer + 1],
    )
    return _interposed_blocker(between)


def layer_addition_blocker(submodule: nn.Module) -> str | None:
    """Return why an inserted layer cannot be an identity, if it cannot.

    Net2DeeperNet is exact only while the activation is positively homogeneous
    and idempotent on its own output, which holds for ReLU and Identity.

    :param submodule: Module that was mutated.
    :type submodule: nn.Module

    :return: A reason key, or None when preservation applies.
    :rtype: str | None
    """
    structural = _structural_blocker(submodule)
    if structural is not None:
        return structural

    stacks = weight_stacks(submodule)
    if not stacks or len(stacks[0]) < 2:
        return "not_mlp"

    layers = stacks[0]
    if any(isinstance(layer, CONV_LAYER_TYPES) for layer in layers):
        return "not_mlp"

    new_layer = layers[-2]
    weight = _primary_weight(new_layer)
    if weight.shape[0] != weight.shape[1]:
        return "non_square"

    between = _modules_between(_container(submodule), new_layer, layers[-1])
    interposed = _interposed_blocker(between)
    if interposed is not None:
        return interposed

    if any(
        not isinstance(
            module,
            (*IDENTITY_SAFE_ACTIVATION_TYPES, *CONTAINER_TYPES),
        )
        for module in between
    ):
        return "non_relu"
    return None


def latent_addition_blocker(network: EvolvableNetwork) -> str | None:
    """Return why widening the latent cannot preserve the function, if it cannot.

    :param network: Network whose latent was widened.
    :type network: EvolvableNetwork

    :return: A reason key, or None when preservation applies.
    :rtype: str | None
    """
    if _is_multi_input(network):
        return "multi_input"

    encoder = network.encoder
    stacks = weight_stacks(encoder)
    if not stacks:
        return "no_latent"

    return _interposed_blocker(_modules_between(_container(encoder), stacks[0][-1]))


def _fade_new_columns(
    consumer: WeightLayer,
    columns: slice,
    rng: np.random.Generator,
    noise_scale: float,
) -> None:
    """Fade the columns through which new units reach a consuming layer.

    A small positive scale lets the new units learn more rapidly since
    gradients flow faster at a small cost in exactness. A noisy layer's
    stochastic scales are always zeroed, so the new units add no exploration
    noise either.

    :param consumer: Layer that reads the widened output.
    :type consumer: WeightLayer
    :param columns: Columns the new units occupy.
    :type columns: slice
    :param rng: Generator the noise is drawn from.
    :type rng: np.random.Generator
    :param noise_scale: Noise size, relative to the existing column scale.
    :type noise_scale: float

    :return: None.
    :rtype: None
    """
    weight = _primary_weight(consumer)
    with torch.no_grad():
        # Measure the columns the new block sits beside, so the noise stays
        # small relative to the signal the consumer already carries.
        keep = torch.ones(weight.shape[1], dtype=torch.bool, device=weight.device)
        keep[columns] = False
        scale = float(weight[:, keep].std())

        noise = torch.as_tensor(
            rng.standard_normal(tuple(weight[:, columns].shape)),
            dtype=weight.dtype,
            device=weight.device,
        )
        weight[:, columns] = noise * noise_scale * (scale if scale > 0.0 else 1e-3)

        if isinstance(consumer, NoisyLinear):
            consumer.weight_sigma[:, columns] = 0.0
            consumer.weight_epsilon[:, columns] = 0.0


def preserve_added_nodes(
    submodule: nn.Module,
    hidden_layer: int,
    old_width: int,
    rng: np.random.Generator,
    noise_scale: float,
) -> None:
    """Fade a widened layer's new units so the module's output is unchanged.

    The new units keep the incoming weights the stock operator gave them,
    and only their *outgoing* weights are rewritten.

    :param submodule: Module that was mutated, cleared by
        :func:`node_addition_blocker`.
    :type submodule: nn.Module
    :param hidden_layer: Index of the widened layer within its stream.
    :type hidden_layer: int
    :param old_width: The layer's width before the mutation.
    :type old_width: int
    :param rng: Generator the fan-out noise is drawn from.
    :type rng: np.random.Generator
    :param noise_scale: Noise size, relative to the existing column scale.
    :type noise_scale: float

    :return: None.
    :rtype: None
    """
    for layers in weight_stacks(submodule):
        producer, consumer = layers[hidden_layer], layers[hidden_layer + 1]
        new_width = _primary_weight(producer).shape[0]
        # A layer held at its maximum width grows by nothing.
        if new_width <= old_width:
            continue

        # A dense layer reads the flattened feature map, so each widened
        # channel owns a whole H*W block of its columns.
        block = 1
        if isinstance(producer, CONV_LAYER_TYPES) and not isinstance(
            consumer,
            CONV_LAYER_TYPES,
        ):
            block = _primary_weight(consumer).shape[1] // new_width

        _fade_new_columns(
            consumer,
            slice(old_width * block, new_width * block),
            rng,
            noise_scale,
        )


def preserve_added_layer(submodule: nn.Module) -> None:
    """Initialise an inserted layer to the identity so the output is unchanged.

    A noisy layer's stochastic scale is zeroed as well, so the identity holds
    while training.

    :param submodule: Module that was mutated, cleared by
        :func:`layer_addition_blocker`.
    :type submodule: nn.Module

    :return: None.
    :rtype: None
    """
    for layers in weight_stacks(submodule):
        new_layer = layers[-2]
        with torch.no_grad():
            _primary_weight(new_layer).zero_().fill_diagonal_(1.0)
            if isinstance(new_layer, NoisyLinear):
                new_layer.weight_sigma.zero_()
                new_layer.bias_mu.zero_()
                new_layer.bias_sigma.zero_()
            elif new_layer.bias is not None:
                new_layer.bias.zero_()


def preserve_added_latent(
    network: EvolvableNetwork,
    old_latent: int,
    rng: np.random.Generator,
    noise_scale: float,
) -> None:
    """Fade the head's new latent columns so the network's output is unchanged.

    :param network: Network whose latent was widened, cleared by
        :func:`latent_addition_blocker`.
    :type network: EvolvableNetwork
    :param old_latent: Latent width before the mutation.
    :type old_latent: int
    :param rng: Generator the fan-out noise is drawn from.
    :type rng: np.random.Generator
    :param noise_scale: Noise size, relative to the existing column scale.
    :type noise_scale: float

    :return: None.
    :rtype: None
    """
    new_latent = network.latent_dim
    # A latent held at its maximum grows by nothing.
    if new_latent <= old_latent:
        return

    for layers in weight_stacks(network.head_net):
        consumer = layers[0]

        # A continuous critic reads the actions from the columns past the
        # latent, so they have to slide out to their new offset before the new
        # latent columns are faded over the ones they used to occupy.
        extra = _primary_weight(consumer).shape[1] - new_latent
        if extra > 0:
            action_weights = (
                [consumer.weight_mu, consumer.weight_sigma, consumer.weight_epsilon]
                if isinstance(consumer, NoisyLinear)
                else [_primary_weight(consumer)]
            )
            with torch.no_grad():
                for weight in action_weights:
                    weight[:, new_latent : new_latent + extra] = weight[
                        :,
                        old_latent : old_latent + extra,
                    ].clone()

        _fade_new_columns(consumer, slice(old_latent, new_latent), rng, noise_scale)


def base_mutation(mut_method: str) -> str:
    """Reduce a possibly prefixed mutation name to its trailing method.

    :param mut_method: Mutation method name.
    :type mut_method: str

    :return: The trailing method name.
    :rtype: str
    """
    return mut_method.split(".")[-1]


def resolve_target(network: nn.Module, mut_method: str) -> nn.Module | None:
    """Walk a dotted mutation name to the module it acts on.

    A latent mutation names no sub-module, so it resolves to the network that
    owns the latent.

    :param network: Module the mutation was applied to.
    :type network: nn.Module
    :param mut_method: Mutation method name.
    :type mut_method: str

    :return: The module the mutation acts on, or None when it cannot be found.
    :rtype: nn.Module | None
    """
    target = network
    for segment in mut_method.split(".")[:-1]:
        # Sub-modules of a ModuleDict are registered under their key, so a
        # per-agent segment resolves by attribute lookup like any other.
        child = getattr(target, segment, None)
        if child is None:
            return None
        target = child

    return target
