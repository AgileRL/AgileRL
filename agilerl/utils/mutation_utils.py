# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from functools import wraps
from typing import TYPE_CHECKING, NamedTuple, TypeGuard, TypeVar, cast

import fastrand  # ty: ignore[unresolved-import] — C extension without type stubs
import numpy as np
import torch
from torch import nn

from agilerl.modules import (
    EvolvableCNN,
    EvolvableModule,
    EvolvableResNet,
    EvolvableWrapper,
    ModuleDict,
    NoisyLinear,
)
from agilerl.modules.custom_components import (
    GumbelSoftmax,
    ResidualBlock,
    SimbaResidualBlock,
)
from agilerl.utils.evolvable_networks import (
    ACTIVATION_FUNCTIONS,
    CONV_LAYER_FUNCTIONS,
    NORMALIZATION_FUNCTIONS,
    compile_model,
)

if TYPE_CHECKING:
    from agilerl.algorithms.core import EvolvableAlgorithm
    from agilerl.hpo.mutation import Mutations
    from agilerl.networks.base import EvolvableNetwork
    from agilerl.typing import MutationReturn

IndividualT = TypeVar("IndividualT", bound="EvolvableAlgorithm")

# Outgoing-weight scale a revived neuron is re-seeded at, as a fraction of the
# consumer layer's live column scale.
REGRAMA_OUT_SCALE = 0.02

# Ceiling applied to every rewritten weight, matching the Gaussian operator's.
MAGNITUDE_LIMIT = 1e6

# Every block type EvolvableCNN can build.
CONV_LAYER_TYPES: tuple[type[nn.Module], ...] = tuple(CONV_LAYER_FUNCTIONS.values())

WeightLayer = nn.Linear | NoisyLinear | nn.Conv2d | nn.Conv3d
WEIGHT_LAYER_TYPES: tuple[type[WeightLayer], ...] = (
    nn.Linear,
    NoisyLinear,
    nn.Conv2d,
    nn.Conv3d,
)

ACTIVATION_TYPES: tuple[type[nn.Module], ...] = tuple(ACTIVATION_FUNCTIONS.values())

# Normalised layers are not compatible with function-preserving architecture mutations as
# they pool their statistics across old and new units.
NORM_LAYER_TYPES: tuple[type[nn.Module], ...] = tuple(NORMALIZATION_FUNCTIONS.values())

# Noise applied to a new unit's outgoing weights, as a fraction of the consumer
# layer's existing column scale.
FP_NOISE_SCALE = 0.02

# Activations that mix units together, so a new unit changes its neighbours'
# outputs even when it contributes nothing downstream.
CROSS_UNIT_ACTIVATION_TYPES: tuple[type[nn.Module], ...] = (
    nn.Softmax,
    nn.LogSoftmax,
    nn.Softmin,
    GumbelSoftmax,
)

# Only idempotent activations are compatible with the identity initialisation used during
# function-preserving layer additions.
IDENTITY_SAFE_ACTIVATION_TYPES: tuple[type[nn.Module], ...] = (nn.ReLU, nn.Identity)

# Network wrapped containing layers affected by mutations
CONTAINER_TYPES: tuple[type[nn.Module], ...] = (
    nn.Sequential,
    nn.ModuleDict,
    nn.ModuleList,
)

NODE_ADDITIONS = frozenset({"add_node", "add_channel"})
LAYER_ADDITIONS = frozenset({"add_layer"})
LATENT_ADDITIONS = frozenset({"add_latent_node"})
PRESERVED_MUTATIONS = NODE_ADDITIONS | LAYER_ADDITIONS | LATENT_ADDITIONS


class ProducerContext(NamedTuple):
    """The layers a measured activation's neurons are wired between.

    producer's weight rows are the neurons' incoming weights and consumers'
    weight columns their outgoing ones; norm is any normalisation applied to
    those neurons between the producer and the activation.
    """

    producer: WeightLayer | None
    norm: nn.Module | None
    consumers: list[WeightLayer]


class ConsumerTarget(NamedTuple):
    """One consumer tensor whose columns a reset neuron owns.

    weight is rewritten in place, stride columns of it per producer neuron.
    is_noise_scale marks a tensor of noise scales rather than weights, revived
    as a constant non-negative fill rather than a random direction.
    """

    weight: torch.Tensor
    stride: int
    is_noise_scale: bool = False


def weight_param(module: WeightLayer) -> torch.Tensor:
    """The weight tensor of a weight layer, or weight_mu for a noisy one."""
    return module.weight_mu if isinstance(module, NoisyLinear) else module.weight


def owns_trainable_weight(module: WeightLayer) -> bool:
    """Check if a critic's encoder is shared with and trained by the actor."""
    weight = weight_param(module)
    return isinstance(weight, nn.Parameter) and weight.requires_grad


def unwrap_module(module: nn.Module) -> nn.Module:
    """Strip wrapper layers that hide the real module."""
    while isinstance(module, EvolvableWrapper):
        module = module.wrapped
    return module


def weight_layers(container: nn.Module) -> list[WeightLayer]:
    """The weight layers container holds, in registration (forward) order."""
    return [
        module
        for _name, module in container.named_modules()
        if isinstance(module, WEIGHT_LAYER_TYPES)
    ]


def head_entry_layers(head: nn.Module | None) -> list[WeightLayer]:
    """The first weight layer of every parallel stream in head."""
    if head is None:
        return []
    children = list(unwrap_module(head).children())
    entries = [next(iter(weight_layers(child)), None) for child in children]
    found = [entry for entry in entries if entry is not None]
    is_flat_stream = any(
        child is entry for child, entry in zip(children, entries, strict=True)
    )
    return found[:1] if is_flat_stream else found


def activation_modules(root: nn.Module, *, include_output: bool) -> list[nn.Module]:
    """The activation sub-modules of root to measure, in forward order."""
    ordered = list(root.named_modules())
    if include_output:
        return [
            module for _name, module in ordered if isinstance(module, ACTIVATION_TYPES)
        ]

    targets: list[nn.Module] = []
    for index, (name, module) in enumerate(ordered):
        if not isinstance(module, ACTIVATION_TYPES):
            continue
        parent = name.rpartition(".")[0]
        prefix = f"{parent}." if parent else ""
        # An activation terminates its stream when no later layer of the same
        # container takes its neurons as input.
        has_later_consumer = any(
            other_name.startswith(prefix) and isinstance(other, WEIGHT_LAYER_TYPES)
            for other_name, other in ordered[index + 1 :]
        )
        if has_later_consumer:
            targets.append(module)
    return targets


def target_activations(network: nn.Module) -> list[nn.Module]:
    """The activation sub-modules measured for one network, in forward order."""
    halves = (
        (getattr(network, "encoder", None), True),
        (getattr(network, "head_net", None), False),
    )
    return [
        module
        for root, is_encoder in halves
        if root is not None
        for module in activation_modules(root, include_output=is_encoder)
    ]


def shared_encoder_heads(
    networks: Sequence[tuple[str | None, nn.Module]],
    network_id: str | None,
    policy_network: nn.Module,
) -> list[WeightLayer]:
    """Return the head entry layers of the networks sharing policy_network's encoder."""
    entries: list[WeightLayer] = []
    for other_id, other in networks:
        if other is policy_network or other_id != network_id:
            continue
        encoder = getattr(other, "encoder", None)
        if encoder is None:
            continue
        # share_encoder_parameters writes the policy encoder's values into the
        # other encoders as plain detached tensors, so a borrowed encoder is
        # exactly one whose weight layers are not nn.Parameter.
        layers = weight_layers(encoder)
        if not layers or any(owns_trainable_weight(layer) for layer in layers):
            continue
        entries.extend(head_entry_layers(getattr(other, "head_net", None)))
    return entries


def dormant_indices(per_neuron: torch.Tensor, dormant_threshold: float) -> list[int]:
    """The indices of the layer's dormant neurons, ascending.

    Scores are normalised by the layer mean. A layer whose mean is zero has no
    live unit left and is reported entirely dormant. Non-finite scores are
    coerced to zero, i.e. treated as dormant: a diverged unit is precisely one
    worth re-initialising.
    """
    scores = torch.nan_to_num(per_neuron, nan=0.0, posinf=0.0, neginf=0.0)
    mean = float(scores.mean())
    if mean <= 0.0:
        return list(range(scores.numel()))
    return torch.nonzero(scores / mean <= dormant_threshold).flatten().tolist()


def resolve_producer_and_next(
    act_module: nn.Module,
    encoder: nn.Module | None,
    head: nn.Module | None,
) -> ProducerContext:
    """Find the layer that produced act_module's neurons and its consumers.

    A normalisation applied to those neurons between the producer and the
    activation is returned alongside them; one applied before the producer is not.
    """
    for root, is_encoder in ((encoder, True), (head, False)):
        if root is None:
            continue

        ordered = list(unwrap_module(root).named_modules())
        name = next((n for n, m in ordered if m is act_module), None)
        if name is None:
            continue

        parent = name.rpartition(".")[0]
        prefix = f"{parent}." if parent else ""
        container = name.split(".")[0]

        producer: WeightLayer | None = None
        norm: nn.Module | None = None
        consumers: list[WeightLayer] = []
        enclosing: WeightLayer | None = None
        passed = False
        for other_name, other in ordered:
            if other is act_module:
                passed = True
                continue
            in_stream = other_name.startswith(prefix)
            if not passed and in_stream and isinstance(other, NORM_LAYER_TYPES):
                norm = other
                continue
            if not isinstance(other, WEIGHT_LAYER_TYPES):
                continue
            if not passed:
                if in_stream:
                    producer = other
                    norm = None
            elif in_stream:
                if not consumers:
                    consumers = [other]
            elif enclosing is None and not other_name.startswith(f"{container}."):
                enclosing = other

        if not consumers and is_encoder:
            consumers = (
                [enclosing] if enclosing is not None else head_entry_layers(head)
            )

        return ProducerContext(producer, norm, consumers)

    return ProducerContext(None, None, [])


def live_column_scale(weight: torch.Tensor, stride: int, keep: list[int]) -> float:
    """The median outgoing-column norm of a consumer over keep, strictly positive."""
    if weight.dim() > 2:  # conv consumer: one filter per producer neuron
        blocks = weight.reshape(weight.shape[0], weight.shape[1], -1)
        fan_out = blocks.shape[0] * blocks.shape[2]  # conv fans count the kernel
    else:
        blocks = weight.reshape(weight.shape[0], -1, stride)
        fan_out = blocks.shape[0]

    for selected in (blocks[:, keep, :], blocks):
        norms = selected.pow(2).sum(dim=(0, 2)).sqrt()
        norms = norms[norms.isfinite()]
        if norms.numel() and (median := float(norms.median())) > 0.0:
            return median

    fan_in = blocks.shape[1] * blocks.shape[2]
    bound = math.sqrt(6.0 / (fan_in + fan_out))
    return bound / math.sqrt(3.0) * math.sqrt(blocks.shape[0] * blocks.shape[2])


def revived_block(
    template: torch.Tensor,
    scale: float,
    rng: np.random.Generator,
    *,
    is_noise_scale: bool = False,
) -> torch.Tensor:
    """Return the new random weights and noise scales for dormant neurons.

    An outgoing weight is a random direction rescaled to scale; a noise-scale value
    is a constant non-negative fill of the same norm.
    """
    if is_noise_scale:
        return torch.full_like(template, scale / math.sqrt(template.numel()))

    sampled = rng.standard_normal(size=tuple(template.shape))
    block = torch.as_tensor(sampled, dtype=template.dtype, device=template.device)
    return block * (scale / float(block.norm()))


def resolve_consumers(
    producer: WeightLayer,
    next_layers: list[WeightLayer],
    cnn_spatial: int | None,
) -> list[ConsumerTarget]:
    """Return the consumer neurons associated with a producer neuron."""
    producer_is_conv = isinstance(producer, CONV_LAYER_TYPES)
    producer_neurons = weight_param(producer).shape[0]
    consumers: list[ConsumerTarget] = []
    for next_layer in next_layers:
        # A dense layer feeding a convolution is a pairing the ReGraMa cannot index.
        next_is_conv = isinstance(next_layer, CONV_LAYER_TYPES)
        if next_is_conv and not producer_is_conv:
            continue

        next_weight = weight_param(next_layer).data
        stride = 1
        if producer_is_conv and not next_is_conv:
            if cnn_spatial is None:
                continue
            stride = cnn_spatial

        if next_weight.shape[1] != producer_neurons * stride:
            continue

        consumers.append(ConsumerTarget(next_weight, stride))

        # The consumer's own noise columns are scales rather than weights and are
        # revived as such.
        if isinstance(next_layer, NoisyLinear):
            consumers.append(
                ConsumerTarget(
                    next_layer.weight_sigma.data,
                    stride,
                    is_noise_scale=True,
                ),
            )

    return consumers


def shared_latent_blocks(
    producer: WeightLayer,
    entry_layers: Sequence[WeightLayer],
) -> list[ConsumerTarget]:
    """Each sharing head's latent columns as a writable view, noise scales included."""
    if isinstance(producer, CONV_LAYER_TYPES):
        return []

    span = weight_param(producer).shape[0]
    blocks: list[ConsumerTarget] = []
    for entry in entry_layers:
        weight = weight_param(entry).data
        if weight.dim() != 2 or weight.shape[1] < span:
            continue
        blocks.append(ConsumerTarget(weight[:, :span], 1))
        if isinstance(entry, NoisyLinear):
            blocks.append(
                ConsumerTarget(
                    entry.weight_sigma.data[:, :span],
                    1,
                    is_noise_scale=True,
                ),
            )
    return blocks


def reset_layer_neurons(
    producer: WeightLayer,
    consumers: list[ConsumerTarget],
    norm: nn.Module | None,
    indices: list[int],
    rng: np.random.Generator,
) -> None:
    """Re-initialise the given neurons of producer and their outgoing weights.

    Each reset neuron gets Xavier-uniform incoming weights, a zero bias, the
    layer's initial noise scale if it is a noisy layer, an outgoing block of norm
    REGRAMA_OUT_SCALE times the consumer's live column scale (a signed random
    direction for a weight, a uniform non-negative fill for a noise scale) and a
    neutral normalisation entry.
    """
    noisy = producer if isinstance(producer, NoisyLinear) else None
    producer_weight = weight_param(producer).data
    neurons = producer_weight.shape[0]

    # Xavier-uniform bound of the producing layer.
    if producer_weight.dim() == 2:  # Linear: (out_features, in_features)
        fan_in, fan_out = producer_weight.shape[1], neurons
    else:  # Conv: (out_channels, in_channels, *kernel)
        receptive = math.prod(producer_weight.shape[2:])
        fan_in = producer_weight.shape[1] * receptive
        fan_out = neurons * receptive
    bound = math.sqrt(6.0 / (fan_in + fan_out))

    # Measure the outgoing scale against the neurons this pass leaves alone.
    reset = set(indices)
    keep = [neuron for neuron in range(neurons) if neuron not in reset]
    out_scales = [
        REGRAMA_OUT_SCALE * live_column_scale(target.weight, target.stride, keep)
        for target in consumers
    ]

    bias_param = (
        producer.bias_mu if isinstance(producer, NoisyLinear) else producer.bias
    )
    bias = None if bias_param is None else bias_param.data
    if bias is not None:
        bias[indices] = 0.0

    if noisy is not None:
        # A revived unit wants the layer's initial noise scale, not the collapsed
        # or inflated one it inherits.
        noisy.weight_sigma.data[indices] = noisy.std_init / math.sqrt(noisy.in_features)
        noisy.bias_sigma.data[indices] = noisy.std_init / math.sqrt(noisy.out_features)

    entries: tuple[tuple[torch.Tensor | None, float], ...] = ()
    if isinstance(norm, nn.LayerNorm):
        entries = ((norm.weight, 1.0), (norm.bias, 0.0))
    elif isinstance(
        norm,
        (nn.BatchNorm2d, nn.BatchNorm3d, nn.InstanceNorm2d, nn.InstanceNorm3d),
    ):
        entries = (
            (norm.weight, 1.0),
            (norm.bias, 0.0),
            (norm.running_mean, 0.0),
            (norm.running_var, 1.0),
        )
    for tensor, identity in entries:
        if tensor is not None and tuple(tensor.shape) == (neurons,):
            tensor.data[indices] = identity

    for index in indices:
        sampled = rng.uniform(-bound, bound, size=tuple(producer_weight[index].shape))
        producer_weight[index] = torch.as_tensor(
            sampled,
            dtype=producer_weight.dtype,
            device=producer_weight.device,
        )
        for target, scale in zip(consumers, out_scales, strict=True):
            columns = slice(index * target.stride, (index + 1) * target.stride)
            target.weight[:, columns] = revived_block(
                target.weight[:, columns],
                scale,
                rng,
                is_noise_scale=target.is_noise_scale,
            )

    # A diverged agent reaches the operator carrying non-finite weights, which a
    # reset must not leave in place.
    rewritten = [producer_weight, *(target.weight for target in consumers)]
    if bias is not None:
        rewritten.append(bias)
    if noisy is not None:
        rewritten += [noisy.weight_sigma.data, noisy.bias_sigma.data]
    for tensor in rewritten:
        tensor.clamp_(-MAGNITUDE_LIMIT, MAGNITUDE_LIMIT).nan_to_num_()


def reset_dormant_neurons(
    network: nn.Module,
    per_neuron_list: list[torch.Tensor | None],
    dormant_threshold: float,
    rng: np.random.Generator,
    shared_latent_heads: Sequence[WeightLayer] = (),
) -> int:
    """Reset every dormant neuron of one evaluation network.

    Walks the network's measured activations, resolves each one's producer,
    normalisation and consumers, and re-initialises the neurons whose normalised
    GraMa score is at or below dormant_threshold.

    :param network: An evaluation network to operate on, in place.
    :type network: torch.nn.Module
    :param per_neuron_list: That network's entry of the captured snapshot, one
        score tensor per measured activation, or None where no gradient flowed.
    :type per_neuron_list: list[torch.Tensor | None]
    :param dormant_threshold: Normalised score at or below which a neuron is
        dormant.
    :type dormant_threshold: float
    :param rng: Seeded generator owned by the caller.
    :type rng: numpy.random.Generator
    :param shared_latent_heads: Head entry layers of the networks sharing this
        network's encoder.
    :type shared_latent_heads: Sequence[WeightLayer]
    :return: How many neurons were reset.
    :rtype: int
    """
    scores = [
        (act_module, per_neuron)
        for act_module, per_neuron in zip(
            target_activations(network),
            per_neuron_list,
            strict=True,
        )
        if per_neuron is not None
    ]

    encoder = getattr(network, "encoder", None)
    head = getattr(network, "head_net", None)

    # Each conv neuron owns a whole flattened H*W column block of the dense layer
    # the convolutional stack flattens into.
    spatial_of: dict[int, int] = {}
    for _name, sub in network.named_modules():
        if not isinstance(sub, (EvolvableCNN, EvolvableResNet)):
            continue
        spatial = math.prod(int(dim) for dim in sub.cnn_output_size[2:])
        for _child_name, child in sub.named_modules():
            spatial_of[id(child)] = spatial

    head_entries = head_entry_layers(head) if shared_latent_heads else []

    neurons_reset = 0
    for act_module, per_neuron in scores:
        producer, norm, next_layers = resolve_producer_and_next(
            act_module,
            encoder,
            head,
        )
        if producer is None or not next_layers or not owns_trainable_weight(producer):
            continue
        # The producer is resolved structurally, so a width it cannot have
        # produced means the wrong layer was picked out.
        if per_neuron.numel() != weight_param(producer).shape[0]:
            continue

        consumers = resolve_consumers(
            producer,
            next_layers,
            spatial_of.get(id(producer)),
        )
        if not consumers:
            continue

        if any(layer is entry for layer in next_layers for entry in head_entries):
            consumers += shared_latent_blocks(producer, shared_latent_heads)

        indices = dormant_indices(per_neuron, dormant_threshold)
        if not indices:
            continue

        reset_layer_neurons(producer, consumers, norm, indices, rng)
        neurons_reset += len(indices)

    return neurons_reset


def layer_container(submodule: nn.Module) -> nn.Module:
    """The container holding a sub-module's weight layers, in forward order."""
    inner = unwrap_module(submodule)
    return getattr(inner, "model", inner)


def weight_stacks(submodule: nn.Module) -> list[list[WeightLayer]]:
    """The parallel weight-layer streams of a sub-module, each in forward order.

    Most modules have one stream. A duelling head has two (the inherited value
    stream and its sibling advantage stream) and one architecture mutation widens
    or deepens both, so both need the same fixup.
    """
    inner = unwrap_module(submodule)
    primary_container = layer_container(submodule)
    primary = weight_layers(primary_container)
    if not primary:
        return []

    # A sibling forms a parallel stream when its layers read the same widths as
    # the primary one's, i.e. the stream's input width followed by its hidden
    # widths. Output widths are left out: a duelling head's value stream ends in
    # a single node and its advantage stream in one per action.
    signature = [weight_param(layer).shape[1] for layer in primary]
    stacks: list[list[WeightLayer]] = [primary]
    for child in inner.children():
        if child is primary_container:
            continue
        layers = weight_layers(child)
        if layers and [weight_param(layer).shape[1] for layer in layers] == signature:
            stacks.append(layers)

    return stacks


def hidden_widths(submodule: nn.Module) -> list[int]:
    """The output width of every non-output weight layer of the primary stream."""
    stacks = weight_stacks(submodule)
    if not stacks:
        return []
    return [weight_param(layer).shape[0] for layer in stacks[0][:-1]]


def modules_between(
    container: nn.Module,
    producer: nn.Module,
    consumer: nn.Module | None = None,
) -> list[nn.Module]:
    """The modules a widened output passes through after producer, in forward order.

    With no consumer, every module registered after producer is returned.
    """
    ordered = [module for _name, module in container.named_modules()]
    start = ordered.index(producer)
    stop = len(ordered) if consumer is None else ordered.index(consumer)
    return ordered[start + 1 : stop]


def interposed_blocker(modules: Sequence[nn.Module]) -> bool:
    """Whether the modules between a producer and its consumer defeat the fade."""
    return any(
        isinstance(module, (*NORM_LAYER_TYPES, *CROSS_UNIT_ACTIVATION_TYPES))
        for module in modules
    )


def structural_blocker(module: nn.Module) -> bool:
    """Whether widening a layer inside module is out of scope.

    Each of these defeats the fade at the point where a hidden layer grows. A
    recurrent core fuses its gate non-linearities, so no single weight matrix
    holds one unit's incoming weights. A residual skip carries a new unit's
    coordinate straight past the layer that would fade it. A multi-input
    encoder's fusion layer reads an interleaved concatenation, so widening a
    sub-encoder does not append columns at the tail.
    """
    for _name, child in module.named_modules():
        if isinstance(child, (nn.RNNBase, SimbaResidualBlock, ResidualBlock)):
            return True
        if hasattr(child, "feature_net") and hasattr(child, "final_dense"):
            return True
    return False


def node_addition_blocker(submodule: nn.Module, hidden_layer: int) -> bool:
    """Whether widening a layer cannot preserve the function."""
    if structural_blocker(submodule):
        return True

    stacks = weight_stacks(submodule)
    if not stacks or not 0 <= hidden_layer < len(stacks[0]) - 1:
        return True

    layers = stacks[0]
    return interposed_blocker(
        modules_between(
            layer_container(submodule),
            layers[hidden_layer],
            layers[hidden_layer + 1],
        ),
    )


def layer_addition_blocker(submodule: nn.Module) -> bool:
    """Whether an inserted layer cannot be an identity.

    Net2DeeperNet is exact only while the activation is positively homogeneous
    and idempotent on its own output, which holds for ReLU and Identity.
    """
    if structural_blocker(submodule):
        return True

    stacks = weight_stacks(submodule)
    if not stacks or len(stacks[0]) < 2:
        return True

    layers = stacks[0]
    if any(isinstance(layer, CONV_LAYER_TYPES) for layer in layers):
        return True

    new_layer = layers[-2]
    weight = weight_param(new_layer)
    if weight.shape[0] != weight.shape[1]:
        return True

    between = modules_between(layer_container(submodule), new_layer, layers[-1])
    if interposed_blocker(between):
        return True

    return any(
        not isinstance(module, (*IDENTITY_SAFE_ACTIVATION_TYPES, *CONTAINER_TYPES))
        for module in between
    )


def latent_addition_blocker(network: EvolvableNetwork) -> bool:
    """Whether widening the latent cannot preserve the function."""
    # Detected structurally, not by isinstance: EvolvableMultiInput fuses
    # several sub-encoders into one vector via these two attributes.
    if hasattr(network, "feature_net") and hasattr(network, "final_dense"):
        return True

    encoder = network.encoder
    stacks = weight_stacks(encoder)
    if not stacks:
        return True

    return interposed_blocker(
        modules_between(layer_container(encoder), stacks[0][-1]),
    )


def fade_new_columns(
    consumer: WeightLayer,
    columns: slice,
    rng: np.random.Generator,
    noise_scale: float,
) -> None:
    """Fade the columns through which new units reach a consuming layer.

    A small positive scale lets the new units learn more rapidly since gradients
    flow faster at a small cost in exactness. A noisy layer's stochastic scales
    are always zeroed, so the new units add no exploration noise either.
    """
    weight = weight_param(consumer)
    with torch.no_grad():
        # Measure the columns the new block sits beside, so the noise stays small
        # relative to the signal the consumer already carries.
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

    The new units keep the incoming weights the stock operator gave them, and
    only their outgoing weights are rewritten. submodule must have been cleared
    by node_addition_blocker.
    """
    for layers in weight_stacks(submodule):
        producer, consumer = layers[hidden_layer], layers[hidden_layer + 1]
        new_width = weight_param(producer).shape[0]
        # A layer held at its maximum width grows by nothing.
        if new_width <= old_width:
            continue

        # A dense layer reads the flattened feature map, so each widened channel
        # owns a whole H*W block of its columns.
        block = 1
        if isinstance(producer, CONV_LAYER_TYPES) and not isinstance(
            consumer,
            CONV_LAYER_TYPES,
        ):
            block = weight_param(consumer).shape[1] // new_width

        fade_new_columns(
            consumer,
            slice(old_width * block, new_width * block),
            rng,
            noise_scale,
        )


def preserve_added_layer(submodule: nn.Module) -> None:
    """Initialise an inserted layer to the identity so the output is unchanged.

    A noisy layer's stochastic scale is zeroed as well, so the identity holds
    while training. submodule must have been cleared by layer_addition_blocker.
    """
    for layers in weight_stacks(submodule):
        new_layer = layers[-2]
        with torch.no_grad():
            weight_param(new_layer).zero_().fill_diagonal_(1.0)
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

    network must have been cleared by latent_addition_blocker.
    """
    new_latent = network.latent_dim
    # A latent held at its maximum grows by nothing.
    if new_latent <= old_latent:
        return

    for layers in weight_stacks(network.head_net):
        consumer = layers[0]

        # A continuous critic reads the actions from the columns past the latent,
        # so they have to slide out to their new offset before the new latent
        # columns are faded over the ones they used to occupy.
        extra = weight_param(consumer).shape[1] - new_latent
        if extra > 0:
            action_weights = (
                [consumer.weight_mu, consumer.weight_sigma, consumer.weight_epsilon]
                if isinstance(consumer, NoisyLinear)
                else [weight_param(consumer)]
            )
            with torch.no_grad():
                for weight in action_weights:
                    weight[:, new_latent : new_latent + extra] = weight[
                        :,
                        old_latent : old_latent + extra,
                    ].clone()

        fade_new_columns(consumer, slice(old_latent, new_latent), rng, noise_scale)


def resolve_target(network: nn.Module, mut_method: str) -> nn.Module | None:
    """Walk a dotted mutation name to the module it acts on.

    A latent mutation names no sub-module, so it resolves to the network that
    owns the latent.
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


def pre_mutation_widths(network: EvolvableModule, mut_method: str) -> list[int] | None:
    """Record the widths an architecture mutation is about to change.

    The stock operator appends new units at the tail, so the fixup needs to know
    where the old ones ended.

    :param network: Network about to be mutated.
    :type network: EvolvableModule
    :param mut_method: The mutation method about to be applied.
    :type mut_method: str

    :return: The affected widths, or None when there is nothing to record.
    :rtype: list[int] | None
    """
    target = resolve_target(network, mut_method)
    if target is None:
        return None

    # mut_method may be dotted/per-agent-prefixed (see resolve_target); only
    # the trailing method identifies which mutation this is.
    if mut_method.split(".")[-1].endswith("latent_node"):
        return [cast("EvolvableNetwork", target).latent_dim]

    return hidden_widths(target)


def preserve_architecture_mutation(
    network: EvolvableModule,
    applied_mut: str | None,
    mut_dict: MutationReturn,
    before: list[int],
    rng: np.random.Generator,
) -> None:
    """Initialise an addition's new capacity so the network's function is unchanged.

    Dispatches on the mutation that was actually applied rather than the one that
    was sampled, since add_layer falls back to widening once a stream has reached
    its depth limit. An addition the architecture cannot support keeps the stock
    operator's random initialisation.

    :param network: Network that was mutated, operated on in place.
    :type network: EvolvableModule
    :param applied_mut: The mutation the network reports having applied.
    :type applied_mut: str | None
    :param mut_dict: The mutation's own report of what it changed.
    :type mut_dict: MutationReturn
    :param before: The widths recorded by :func:`pre_mutation_widths`.
    :type before: list[int]
    :param rng: Seeded generator the symmetry-breaking noise is drawn from.
    :type rng: numpy.random.Generator

    :return: None.
    :rtype: None
    """
    if applied_mut is None:
        return

    # applied_mut may be dotted/per-agent-prefixed (see resolve_target); only
    # the trailing method identifies the mutation that ran.
    base = applied_mut.split(".")[-1]
    if base not in PRESERVED_MUTATIONS:
        return

    target = resolve_target(network, applied_mut)
    if target is None:
        return

    if base in LATENT_ADDITIONS:
        latent_network = cast("EvolvableNetwork", target)
        if not latent_addition_blocker(latent_network):
            preserve_added_latent(latent_network, before[0], rng, FP_NOISE_SCALE)
        return

    if base in LAYER_ADDITIONS:
        if not layer_addition_blocker(target):
            preserve_added_layer(target)
        return

    reported = mut_dict.get("hidden_layer")
    hidden_layer = reported if isinstance(reported, int) else -1
    if not node_addition_blocker(target, hidden_layer):
        preserve_added_nodes(
            target,
            hidden_layer,
            before[hidden_layer],
            rng,
            FP_NOISE_SCALE,
        )


def set_global_seed(seed: int | None) -> None:
    """Set the global seed for random number generators.

    :param seed: Random seed for repeatability
    :type seed: int
    """
    if seed is None:
        return

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    fastrand.pcg32_seed(seed)


def is_module_dict(
    module: EvolvableModule,
) -> TypeGuard[ModuleDict[EvolvableModule]]:
    """Narrow an evaluation module to its per-agent ``ModuleDict`` mapping.

    :param module: The evaluation module to check
    :type module: EvolvableModule
    :return: Whether the module is a per-agent ``ModuleDict``
    :rtype: TypeGuard[ModuleDict[EvolvableModule]]
    """
    return isinstance(module, ModuleDict)


def as_module_dict(module: EvolvableModule) -> ModuleDict[EvolvableModule]:
    """Narrow a multi-agent evaluation module to its per-agent mapping.

    :param module: The evaluation module to reinterpret
    :type module: EvolvableModule
    :return: The module as a mapping of per-agent modules
    :rtype: ModuleDict[EvolvableModule]
    :raises TypeError: If the module is not a per-agent ``ModuleDict``
    """
    if not is_module_dict(module):
        msg = "Multi-agent mutation requires a per-agent ModuleDict container."
        raise TypeError(msg)

    return module


def get_exp_layer(offspring: EvolvableModule) -> nn.Linear:
    """Get the output layer of different types of offsprings for bandit algorithms.

    :param offspring: The offspring to inspect
    :type offspring: EvolvableModule

    :return: The output layer of the offspring
    :rtype: nn.Linear
    """
    if not isinstance(offspring, EvolvableModule):
        msg = f"Bandit algorithm architecture {type(offspring)} not supported."
        raise TypeError(msg)

    exp_layer = offspring.get_output_dense()
    if not isinstance(exp_layer, nn.Linear):
        msg = (
            f"Bandit algorithm architecture {type(offspring)} not supported: expected "
            f"a linear output layer, found {type(exp_layer)}."
        )
        raise TypeError(msg)

    return exp_layer


def reinit_shared_networks(
    mutation_func: Callable[[Mutations, IndividualT], IndividualT],
) -> Callable[[Mutations, IndividualT], IndividualT]:
    """Reinitialize shared networks after architecture and parameter mutations (decorator).

    :param mutation_func: The mutation function to decorate
    :type mutation_func: Callable[[Mutations, IndividualT], IndividualT]
    :return: The decorated mutation function
    :rtype: Callable[[Mutations, IndividualT], IndividualT]
    """

    @wraps(mutation_func)
    def wrapper(self: Mutations, individual: IndividualT) -> IndividualT:
        individual = mutation_func(self, individual)

        # Drop dynamo's compiled graphs and guards so a mutated architecture is
        # not served a stale one.
        torch._dynamo.reset()

        if individual.mut == "None":
            return individual

        compiled_model = individual.torch_compiler is not None
        if compiled_model:
            # Static parameter shapes would fail dynamo's guards on a mutated
            # architecture. Suppressed here and below: dynamo's config module types
            # each attribute from its default value, so the flag reads as a literal.
            torch._dynamo.config.force_parameter_static_shapes = False  # ty: ignore[invalid-assignment]
            individual.recompile()

        # A shared network mirrors its group's evaluation network, whose
        # architecture the mutation may have changed.
        for net_group in individual.registry.groups:
            for shared_name in net_group.shared_network_names():
                eval_offspring: EvolvableModule = getattr(
                    individual,
                    net_group.eval_network_name(),
                )
                ind_shared: nn.Module = self._reinit_from_mutated(
                    eval_offspring,
                    remove_prefix=compiled_model,
                )
                if self.accelerator is None:
                    ind_shared = ind_shared.to(self.device)

                if compiled_model:
                    torch._dynamo.config.force_parameter_static_shapes = False  # ty: ignore[invalid-assignment]
                    ind_shared = compile_model(
                        ind_shared,
                        individual.torch_compiler,
                    )

                setattr(individual, shared_name, ind_shared)

        return individual

    return wrapper
