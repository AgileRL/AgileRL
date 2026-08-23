# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Function-preserving architecture mutations.

AgileRL grows a network by appending units or a layer and initialising the new
capacity randomly, which changes the network's output immediately. The mutated
agent is then evaluated as a different policy than the one that earned its place
in the population, and selection usually culls it before the added capacity can
be trained into anything useful. Initialising the new capacity so the network is
unchanged removes that penalty: the agent keeps its fitness and the extra
capacity is exploited by subsequent training.

Thus, the capacity addition operations are made, when possible, function-preserving:

* add_node, add_channel, add_latent_node: the new units keep their freshly
  initialised incoming weights, while the outgoing weights that carry them into
  the consuming layer are faded to near zero (to avoid adding dormant neurons).
  The consumer's output is therefore unchanged whatever the activation does,
  and the new units still receive meaningful gradients fast.
* add_layer: the newly inserted layer is initialised to the identity, which is exact
  for ReLU and Identity activations.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
from torch import nn

from agilerl.modules.custom_components import (
    GumbelSoftmax,
    ResidualBlock,
    SimbaResidualBlock,
)

# Noise applied to a new unit's outgoing weights, as a
# fraction of the consumer layer's existing column scale.
FP_NOISE_SCALE = 0.01

CONV_LAYER_TYPES: tuple[type[nn.Module], ...] = (nn.Conv1d, nn.Conv2d, nn.Conv3d)

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

NODE_ADDITIONS = frozenset({"add_node", "add_channel"})
LAYER_ADDITIONS = frozenset({"add_layer"})
LATENT_ADDITIONS = frozenset({"add_latent_node"})
PRESERVED_MUTATIONS = NODE_ADDITIONS | LAYER_ADDITIONS | LATENT_ADDITIONS

# Why a mutation could not be made function-preserving, phrased for a warning.
DECLINE_REASONS = {
    "norm": "a normalisation layer re-scales the widened units together",
    "cross_unit_activation": "the activation mixes units together",
    "recurrent": "a recurrent core owns no per-unit weight rows",
    "multi_input": "a multi-input encoder interleaves its features",
    "residual": "a residual skip bypasses the faded weights",
    "simba": "a SimBa block's residual skip bypasses the faded weights",
    "not_mlp": "the layer stack is not an MLP",
    "non_relu": "the activation is neither ReLU nor Identity",
    "non_square": "the inserted layer is not square",
    "no_consumer": "no consuming layer could be resolved",
    "no_latent": "the network exposes no latent dimension",
    "not_written": "the fixup found no layer it could initialise",
}


def _is_conv(layer: nn.Module) -> bool:
    """Return whether a layer is a convolution of any dimensionality.

    :param layer: Candidate layer.
    :type layer: nn.Module

    :return: Whether the layer is a convolution.
    :rtype: bool
    """
    return isinstance(layer, CONV_LAYER_TYPES)


def _is_linear(layer: nn.Module) -> bool:
    """Return whether a layer is dense.

    :param layer: Candidate layer.
    :type layer: nn.Module

    :return: Whether the layer is dense.
    :rtype: bool
    """
    if isinstance(layer, nn.Linear):
        return True
    weight_mu = getattr(layer, "weight_mu", None)
    return isinstance(weight_mu, torch.Tensor) and weight_mu.dim() == 2


def _is_weight_layer(layer: nn.Module) -> bool:
    """Return whether a layer owns a weight matrix the surgery can rewrite.

    :param layer: Candidate layer.
    :type layer: nn.Module

    :return: Whether the layer is a convolution or a dense layer.
    :rtype: bool
    """
    return _is_conv(layer) or _is_linear(layer)


def _primary_weight(layer: nn.Module) -> torch.Tensor:
    """Return the weight tensor that defines a layer's shape.

    :param layer: Weight layer.
    :type layer: nn.Module

    :return: The layer's weight, or weight_mu when it is noisy.
    :rtype: torch.Tensor

    :raises TypeError: If the layer owns neither tensor.
    """
    for name in ("weight", "weight_mu"):
        weight = getattr(layer, name, None)
        if isinstance(weight, torch.Tensor):
            return weight

    msg = f"{type(layer).__name__} owns no weight tensor."
    raise TypeError(msg)


def _out_dim(layer: nn.Module) -> int:
    """Return how many outputs a weight layer produces.

    :param layer: Weight layer.
    :type layer: nn.Module

    :return: The layer's output width.
    :rtype: int
    """
    return int(_primary_weight(layer).shape[0])


def _in_dim(layer: nn.Module) -> int:
    """Return how many inputs a weight layer consumes.

    :param layer: Weight layer.
    :type layer: nn.Module

    :return: The layer's input width.
    :rtype: int
    """
    return int(_primary_weight(layer).shape[1])


def _inner_module(submodule: nn.Module) -> nn.Module:
    """Unwrap an evolvable wrapper down to the module that owns the layers.

    :param submodule: Possibly wrapped module.
    :type submodule: nn.Module

    :return: The module that owns the weight layers.
    :rtype: nn.Module
    """
    if hasattr(submodule, "model"):
        return submodule
    wrapped = getattr(submodule, "wrapped", None)
    if wrapped is not None and hasattr(wrapped, "model"):
        return wrapped
    return submodule


def _ordered_weight_layers(container: nn.Module) -> list[nn.Module]:
    """Return a container's weight layers in forward order.

    :param container: Module to scan.
    :type container: nn.Module

    :return: The weight layers, in forward order.
    :rtype: list[nn.Module]
    """
    return [layer for _, layer in container.named_modules() if _is_weight_layer(layer)]


def _stack_signature(layers: list[nn.Module]) -> tuple[int, tuple[int, ...]]:
    """Return a stream's shape fingerprint: input width plus hidden widths.

    :param layers: A stream's weight layers, in forward order.
    :type layers: list[nn.Module]

    :return: The stream's input width and its hidden widths.
    :rtype: tuple[int, tuple[int, ...]]
    """
    return _in_dim(layers[0]), tuple(_out_dim(layer) for layer in layers[:-1])


def weight_stacks(submodule: nn.Module) -> list[list[nn.Module]]:
    """Return the parallel weight-layer streams of a sub-module.

    Most modules have one stream. A duelling head has two (the inherited value
    stream and its sibling advantage stream) and one architecture mutation
    widens or deepens both, so both need the same fixup.

    :param submodule: Module that was mutated.
    :type submodule: nn.Module

    :return: One list of weight layers per stream, each in forward order.
    :rtype: list[list[nn.Module]]
    """
    inner = _inner_module(submodule)
    primary_container = getattr(inner, "model", inner)
    primary = _ordered_weight_layers(primary_container)
    if not primary:
        return []

    stacks = [primary]
    signature = _stack_signature(primary)
    for name, child in inner.named_children():
        if child is primary_container or name == "model":
            continue
        layers = _ordered_weight_layers(child)
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
    return [_out_dim(layer) for layer in stacks[0][:-1]]


def _container(submodule: nn.Module) -> nn.Module:
    """Return the container that holds a sub-module's weight layers.

    :param submodule: Possibly wrapped module.
    :type submodule: nn.Module

    :return: The container to scan in registration order.
    :rtype: nn.Module
    """
    inner = _inner_module(submodule)
    return getattr(inner, "model", inner)


def _modules_between(
    container: nn.Module,
    producer: nn.Module,
    consumer: nn.Module,
) -> list[nn.Module]:
    """Return the modules registered strictly between two weight layers.

    :param container: Container holding both layers.
    :type container: nn.Module
    :param producer: Layer whose output is being widened.
    :type producer: nn.Module
    :param consumer: Layer that reads the producer's output.
    :type consumer: nn.Module

    :return: The intervening modules, in forward order.
    :rtype: list[nn.Module]
    """
    ordered = [module for _, module in container.named_modules()]
    try:
        start = ordered.index(producer)
        stop = ordered.index(consumer)
    except ValueError:
        return []
    return ordered[start + 1 : stop]


def _modules_after(container: nn.Module, producer: nn.Module) -> list[nn.Module]:
    """Return the modules registered after a weight layer.

    :param container: Container holding the layer.
    :type container: nn.Module
    :param producer: Layer whose output is being widened.
    :type producer: nn.Module

    :return: The trailing modules, in forward order.
    :rtype: list[nn.Module]
    """
    ordered = [module for _, module in container.named_modules()]
    try:
        start = ordered.index(producer)
    except ValueError:
        return []
    return ordered[start + 1 :]


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
    """Return whether a module fuses several sub-encoders into one output.

    :param module: Candidate module.
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


def node_addition_blocker(
    submodule: nn.Module,
    hidden_layer: int | None,
) -> str | None:
    """Return why widening a layer cannot preserve the function, if it cannot.

    :param submodule: Module that was mutated.
    :type submodule: nn.Module
    :param hidden_layer: Index of the widened layer, or None when the mutation
        did not report one.
    :type hidden_layer: int | None

    :return: A reason key, or None when preservation applies.
    :rtype: str | None
    """
    structural = _structural_blocker(submodule)
    if structural is not None:
        return structural

    stacks = weight_stacks(submodule)
    if hidden_layer is None or not stacks:
        return "no_consumer"
    if not 0 <= hidden_layer < len(stacks[0]) - 1:
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
    if any(_is_conv(layer) for layer in layers):
        return "not_mlp"

    new_layer = layers[-2]
    weight = _primary_weight(new_layer)
    if weight.shape[0] != weight.shape[1]:
        return "non_square"

    between = _modules_between(_container(submodule), new_layer, layers[-1])
    interposed = _interposed_blocker(between)
    if interposed is not None:
        return interposed

    activations = [
        module
        for module in between
        if not isinstance(module, (nn.Sequential, nn.ModuleDict, nn.ModuleList))
    ]
    if any(
        not isinstance(module, IDENTITY_SAFE_ACTIVATION_TYPES) for module in activations
    ):
        return "non_relu"
    return None


def latent_addition_blocker(network: nn.Module) -> str | None:
    """Return why widening the latent cannot preserve the function, if it cannot.

    :param network: Network whose latent was widened.
    :type network: nn.Module

    :return: A reason key, or None when preservation applies.
    :rtype: str | None
    """
    if _is_multi_input(network):
        return "multi_input"

    encoder = getattr(network, "encoder", None)
    if encoder is None or not hasattr(network, "latent_dim"):
        return "no_latent"

    stacks = weight_stacks(encoder)
    if not stacks:
        return "no_latent"

    return _interposed_blocker(_modules_after(_container(encoder), stacks[0][-1]))


def _spatial_block(submodule: nn.Module) -> int:
    """Return how many columns one convolution channel spans after flattening.

    :param submodule: Convolutional module that was widened.
    :type submodule: nn.Module

    :return: Columns per channel, or 1 when the module is not convolutional.
    :rtype: int
    """
    output_size = getattr(_inner_module(submodule), "cnn_output_size", None)
    if output_size is None:
        return 1
    return max(int(math.prod(tuple(output_size)[2:])), 1)


def _existing_column_scale(weight: torch.Tensor, columns: slice) -> float:
    """Return the scale of the columns a new block sits beside.

    :param weight: Consumer weight tensor.
    :type weight: torch.Tensor
    :param columns: Columns the new units occupy.
    :type columns: slice

    :return: Standard deviation of the surrounding columns, never zero.
    :rtype: float
    """
    keep = torch.ones(weight.shape[1], dtype=torch.bool, device=weight.device)
    keep[columns] = False
    scale = weight.data[:, keep].std() if bool(keep.any()) else weight.data.std()
    value = float(scale)
    return value if value > 0.0 else 1e-3


def _fade_new_columns(
    consumer: nn.Module,
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
    :type consumer: nn.Module
    :param columns: Columns the new units occupy.
    :type columns: slice
    :param rng: Generator the noise is drawn from.
    :type rng: np.random.Generator
    :param noise_scale: Noise size, relative to the existing column scale.
    :type noise_scale: float

    :return: None.
    :rtype: None
    """
    with torch.no_grad():
        for name in ("weight", "weight_mu"):
            weight = getattr(consumer, name, None)
            if not isinstance(weight, torch.Tensor):
                continue
            if noise_scale <= 0.0:
                weight.data[:, columns] = 0.0
                continue
            block = weight.data[:, columns]
            noise = torch.as_tensor(
                rng.standard_normal(tuple(block.shape)),
                dtype=weight.dtype,
                device=weight.device,
            )
            weight.data[:, columns] = noise * (
                noise_scale * _existing_column_scale(weight, columns)
            )

        for name in ("weight_sigma", "weight_epsilon"):
            noisy = getattr(consumer, name, None)
            if isinstance(noisy, torch.Tensor):
                noisy.data[:, columns] = 0.0


def _shift_trailing_columns(
    consumer: nn.Module,
    old_width: int,
    new_width: int,
    extra: int,
) -> None:
    """Slide a head's non-latent input columns to their widened offset.

    A continuous critic's head reads cat([latent, actions]), so its action
    columns start at the latent width.

    :param consumer: The head's entry layer.
    :type consumer: nn.Module
    :param old_width: Latent width before the mutation.
    :type old_width: int
    :param new_width: Latent width after the mutation.
    :type new_width: int
    :param extra: Width of the trailing non-latent block.
    :type extra: int

    :return: None.
    :rtype: None
    """
    with torch.no_grad():
        for name in ("weight", "weight_mu", "weight_sigma", "weight_epsilon"):
            weight = getattr(consumer, name, None)
            if not isinstance(weight, torch.Tensor):
                continue
            block = weight.data[:, old_width : old_width + extra].clone()
            weight.data[:, new_width : new_width + extra] = block


def preserve_added_nodes(
    submodule: nn.Module,
    hidden_layer: int,
    old_width: int,
    rng: np.random.Generator,
    noise_scale: float = FP_NOISE_SCALE,
) -> bool:
    """Fade a widened layer's new units so the module's output is unchanged.

    The new units keep the incoming weights the stock operator gave them,
    and only their *outgoing* weights are rewritten.

    :param submodule: Module that was mutated.
    :type submodule: nn.Module
    :param hidden_layer: Index of the widened layer within its stream.
    :type hidden_layer: int
    :param old_width: The layer's width before the mutation.
    :type old_width: int
    :param rng: Generator the fan-out noise is drawn from.
    :type rng: np.random.Generator
    :param noise_scale: Noise size, relative to the existing column scale.
    :type noise_scale: float

    :return: Whether every stream that grew was faded. A layer held at its
        maximum width grows by nothing, so it needs no fixup and reports True.
    :rtype: bool
    """
    covered = True
    for layers in weight_stacks(submodule):
        if not 0 <= hidden_layer < len(layers) - 1:
            covered = False
            continue

        producer, consumer = layers[hidden_layer], layers[hidden_layer + 1]
        new_width = _out_dim(producer)
        if new_width - old_width <= 0:
            continue

        block = (
            _spatial_block(submodule)
            if _is_conv(producer) and _is_linear(consumer)
            else 1
        )
        _fade_new_columns(
            consumer,
            slice(old_width * block, new_width * block),
            rng,
            noise_scale,
        )

    return covered


def preserve_added_layer(submodule: nn.Module) -> bool:
    """Initialise an inserted layer to the identity so the output is unchanged.

    A noisy layer's stochastic scale is zeroed as well, so the identity holds
    while training.

    :param submodule: Module that was mutated.
    :type submodule: nn.Module

    :return: Whether any layer was initialised.
    :rtype: bool
    """
    written = False
    for layers in weight_stacks(submodule):
        if len(layers) < 2:
            continue

        new_layer = layers[-2]
        weight = _primary_weight(new_layer)
        if weight.dim() != 2 or weight.shape[0] != weight.shape[1]:
            continue

        with torch.no_grad():
            for name in ("weight", "weight_mu"):
                tensor = getattr(new_layer, name, None)
                if isinstance(tensor, torch.Tensor):
                    tensor.data.zero_()
                    tensor.data.fill_diagonal_(1.0)
            for name in ("weight_sigma", "bias", "bias_mu", "bias_sigma"):
                tensor = getattr(new_layer, name, None)
                if isinstance(tensor, torch.Tensor):
                    tensor.data.zero_()
        written = True

    return written


def preserve_added_latent(
    network: nn.Module,
    old_latent: int,
    rng: np.random.Generator,
    noise_scale: float = FP_NOISE_SCALE,
) -> bool:
    """Fade the head's new latent columns so the network's output is unchanged.

    :param network: Network whose latent was widened.
    :type network: nn.Module
    :param old_latent: Latent width before the mutation.
    :type old_latent: int
    :param rng: Generator the fan-out noise is drawn from.
    :type rng: np.random.Generator
    :param noise_scale: Noise size, relative to the existing column scale.
    :type noise_scale: float

    :return: Whether every head stream's new columns were faded. A latent held
        at its maximum grows by nothing, so it needs no fixup and reports True.
    :rtype: bool
    """
    new_latent = int(getattr(network, "latent_dim", 0))
    if new_latent - old_latent <= 0:
        return True

    head = getattr(network, "head_net", None)
    if head is None:
        return False

    stacks = weight_stacks(head)
    covered = bool(stacks)
    for layers in stacks:
        consumer = layers[0]
        extra = _in_dim(consumer) - new_latent
        # A head narrower than the latent it reads cannot be the consumer the
        # fade was meant for, so dismiss it and report the miss.
        if extra < 0:
            covered = False
            continue
        if extra > 0:
            _shift_trailing_columns(consumer, old_latent, new_latent, extra)
        _fade_new_columns(consumer, slice(old_latent, new_latent), rng, noise_scale)

    return covered


def base_mutation(mut_method: str | None) -> str:
    """Reduce a possibly prefixed mutation name to its trailing method.

    :param mut_method: Mutation method name, or None.
    :type mut_method: str | None

    :return: The trailing method name, empty when there is none.
    :rtype: str
    """
    return mut_method.split(".")[-1] if mut_method else ""


def is_latent_mutation(base: str) -> bool:
    """Return whether a mutation resizes a network's latent.

    :param base: Trailing method name.
    :type base: str

    :return: Whether the mutation resizes the latent.
    :rtype: bool
    """
    return base.endswith("latent_node")


def _child_module(container: nn.Module, name: str) -> nn.Module | None:
    """Return a container's child by dictionary key or attribute name.

    :param container: Module or module mapping to look inside.
    :type container: nn.Module
    :param name: Key or attribute naming the child.
    :type name: str

    :return: The child module, or None when the name does not resolve.
    :rtype: nn.Module | None
    """
    mapping: Any = container
    try:
        return mapping[name]
    except (KeyError, IndexError, TypeError):
        return getattr(container, name, None)


def resolve_target(network: nn.Module, mut_method: str | None) -> nn.Module | None:
    """Walk a dotted mutation name to the module it acts on.

    A latent mutation names no sub-module, so it resolves to the network that
    owns the latent.

    :param network: Module the mutation was applied to.
    :type network: nn.Module
    :param mut_method: Mutation method name, or None.
    :type mut_method: str | None

    :return: The module the mutation acts on, or None when it cannot be found.
    :rtype: nn.Module | None
    """
    if not mut_method:
        return None

    target: nn.Module = network
    for segment in mut_method.split(".")[:-1]:
        child = _child_module(target, segment)
        if child is None:
            return None
        target = child
    return target
