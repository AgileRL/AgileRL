# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from functools import wraps
from typing import TYPE_CHECKING, NamedTuple, TypeGuard, TypeVar

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
from agilerl.utils.evolvable_networks import (
    ACTIVATION_FUNCTIONS,
    CONV_LAYER_FUNCTIONS,
    NORMALIZATION_FUNCTIONS,
    compile_model,
)

if TYPE_CHECKING:
    from agilerl.algorithms.core import EvolvableAlgorithm
    from agilerl.hpo.mutation import Mutations

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

# Normalisations hold per-neuron state of their own, so a revived neuron's entry
# has to be reset with it.
NORM_LAYER_TYPES: tuple[type[nn.Module], ...] = tuple(NORMALIZATION_FUNCTIONS.values())


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
    """Whether module owns the weights the surgery would rewrite.

    :func:`~agilerl.utils.algo_utils.share_encoder_parameters` pins a non-policy
    network's encoder to detached, non-leaf clones of the policy encoder's
    parameters.
    """
    weight = weight_param(module)
    return isinstance(weight, nn.Parameter) and weight.requires_grad


def unwrap_module(module: nn.Module) -> nn.Module:
    """Strip wrapper layers that hide the real module."""
    while isinstance(module, EvolvableWrapper):
        module = module.wrapped
    return module


def first_weight_layer(module: nn.Module) -> WeightLayer | None:
    """The first weight-bearing layer inside module, in forward order."""
    for _name, child in module.named_modules():
        if isinstance(child, WEIGHT_LAYER_TYPES):
            return child
    return None


def head_entry_layers(head: nn.Module | None) -> list[WeightLayer]:
    """The first weight layer of every parallel stream in head."""
    if head is None:
        return []
    children = list(unwrap_module(head).children())
    entries = [first_weight_layer(child) for child in children]
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
        layers = [
            module
            for _name, module in encoder.named_modules()
            if isinstance(module, WEIGHT_LAYER_TYPES)
        ]
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
    """Draw the outgoing block a reset neuron is revived with, of norm scale.

    A weight block is a random direction rescaled to scale; a noise-scale block
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
    """Pair each usable consumer weight tensor with its per-neuron column stride.

    A consumer must spend its columns on exactly these neurons: one block each,
    none interleaved. Anything failing that is not this producer's consumer and
    rewriting its columns would corrupt weights belonging to other neurons, so it
    is skipped instead.
    """
    producer_is_conv = isinstance(producer, CONV_LAYER_TYPES)
    producer_neurons = weight_param(producer).shape[0]
    consumers: list[ConsumerTarget] = []
    for next_layer in next_layers:
        # A dense layer feeding a convolution is a pairing the surgery cannot index.
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
