# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import NamedTuple

import numpy as np
import torch
from torch import nn

from agilerl.modules.custom_components import NewGELU
from agilerl.utils.evolvable_networks import ACTIVATION_FUNCTIONS

logger = logging.getLogger(__name__)

# Outgoing-weight scale a revived neuron is re-seeded at, as a fraction of the
# consumer layer's live column scale.
REGRAMA_OUT_SCALE = 0.02

# Ceiling applied to every rewritten weight, matching the Gaussian operator's.
MAGNITUDE_LIMIT = 1e6

# Every block type EvolvableCNN can build.
CONV_LAYER_TYPES: tuple[type[nn.Module], ...] = (nn.Conv1d, nn.Conv2d, nn.Conv3d)

# Activation sub-modules are recognised by type.
ACTIVATION_TYPES: tuple[type[nn.Module], ...] = (
    *dict.fromkeys(ACTIVATION_FUNCTIONS.values()),
    NewGELU,
)

# Normalisations hold per-neuron state of their own, so a revived neuron's entry
# has to be reset with it.
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

# Attributes identifying a layer that remaps its input onto fresh neurons.
OUTPUT_WIDTH_ATTRS = ("out_features", "out_channels", "hidden_size")


class _ProducerContext(NamedTuple):
    """The layers a measured activation's neurons are wired between.

    producer's weight rows are the neurons' incoming weights and consumers'
    weight columns their outgoing ones; norm is any normalisation applied to
    those neurons between the producer and the activation.
    """

    producer: nn.Module | None
    norm: nn.Module | None
    consumers: list[nn.Module]


class _ConsumerTarget(NamedTuple):
    """One consumer tensor whose columns a reset neuron owns.

    weight is rewritten in place, stride columns of it per producer neuron.
    is_noise_scale marks a tensor holding noise scales rather than weights,
    which are revived differently.
    """

    weight: torch.Tensor
    stride: int
    is_noise_scale: bool = False


def _is_weight_layer(module: nn.Module) -> bool:
    """Whether module carries resettable weights (Linear/Conv/Noisy)."""
    if isinstance(module, (nn.Linear, *CONV_LAYER_TYPES)):
        return True
    return hasattr(module, "weight_mu") and hasattr(module, "bias_mu")


def _weight_param(module: nn.Module) -> torch.Tensor:
    """The weight tensor of a weight layer, or weight_mu for a noisy one.

    Typed as a plain tensor rather than a parameter on purpose: a network whose
    encoder is pinned by share_encoder_parameters holds detached, non-leaf
    clones there. Raises TypeError if the module carries no weight tensor.
    """
    weight = getattr(module, "weight_mu", None)
    if weight is None:
        weight = getattr(module, "weight", None)
    if not isinstance(weight, torch.Tensor):
        msg = f"{type(module).__name__} exposes no weight tensor to reset."
        raise TypeError(msg)
    return weight


def _owns_trainable_weight(module: nn.Module) -> bool:
    """Whether module owns the weights the surgery would rewrite.

    :func:`~agilerl.utils.algo_utils.share_encoder_parameters` pins a non-policy
    network's encoder to detached, non-leaf clones of the policy encoder's
    parameters, and the algorithm re-runs that pinning from its mutation hook. So
    resetting a neuron's incoming weights there is discarded moments later, while
    the matching outgoing rewrite in that network's head survives, leaving the
    head compensating a reset that no longer exists.
    """
    weight = _weight_param(module)
    return isinstance(weight, nn.Parameter) and weight.requires_grad


def _unwrap_module(module: nn.Module | None) -> nn.Module | None:
    """Strip wrapper layers that hide the real module."""
    seen: set[int] = set()
    while module is not None and id(module) not in seen:
        seen.add(id(module))
        wrapped = getattr(module, "wrapped", None)
        if wrapped is None:
            break
        module = wrapped
    return module


def _first_weight_layer(module: nn.Module | None) -> nn.Module | None:
    """The first weight-bearing layer inside module, in forward order."""
    if module is None:
        return None
    for _name, child in module.named_modules():
        if _is_weight_layer(child):
            return child
    return None


def _head_entry_layers(head: nn.Module | None) -> list[nn.Module]:
    """The first weight layer of every parallel stream in head.

    A latent neuron's outgoing weights live in whatever the head feeds it to:
    one layer for a plain MLP head, but two for a duelling Q-network, whose
    value and advantage streams are sibling sub-networks that both consume the
    full latent.
    """
    head = _unwrap_module(head)
    if head is None:
        return []
    children = list(head.children())
    # A head whose own children are layers is a single flat stream.
    if any(_is_weight_layer(child) for child in children):
        first = _first_weight_layer(head)
        return [first] if first is not None else []
    entries = [_first_weight_layer(child) for child in children]
    return [entry for entry in entries if entry is not None]


def _activation_modules(root: nn.Module, *, include_output: bool) -> list[nn.Module]:
    """The activation sub-modules of root to measure, in forward order.

    include_output also admits the stream-terminating activations. Whether an
    excluded activation terminates its stream is decided structurally rather
    than positionally, which is what keeps parallel streams correct: a
    duelling Q-network's head holds two independent sub-networks.
    """
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
        has_later_consumer = any(
            other_name.startswith(prefix)
            and any(hasattr(other, attr) for attr in OUTPUT_WIDTH_ATTRS)
            for other_name, other in ordered[index + 1 :]
        )
        if has_later_consumer:
            targets.append(module)
    return targets


def target_activations(network: nn.Module) -> list[nn.Module]:
    """The activation sub-modules measured for one network, in forward order.

    :param network: One evaluation network of an agent.
    :type network: torch.nn.Module
    :return: The activation sub-modules to hook, in forward order.
    :rtype: list[torch.nn.Module]
    """
    targets: list[nn.Module] = []
    encoder = getattr(network, "encoder", None)
    if encoder is not None:
        targets += _activation_modules(encoder, include_output=True)
    head = getattr(network, "head_net", None)
    if head is not None:
        targets += _activation_modules(head, include_output=False)
    return targets


def shared_encoder_heads(
    networks: Sequence[tuple[str | None, nn.Module]],
    network_id: str | None,
    policy_network: nn.Module,
) -> list[nn.Module]:
    """Return the head entry layers of the networks sharing policy_network's encoder.

    :param networks: The measured evaluation networks, as
        :meth:`~agilerl.algorithms.core.base.EvolvableAlgorithm.eval_networks`
        returns them.
    :type networks: Sequence[tuple[str | None, torch.nn.Module]]
    :param network_id: Sub-policy key of policy_network, or None.
    :type network_id: str | None
    :param policy_network: The policy evaluation network whose encoder is shared.
    :type policy_network: torch.nn.Module
    :return: One entry layer per stream of every sharing network's head.
    :rtype: list[torch.nn.Module]
    """
    entries: list[nn.Module] = []
    for other_id, other in networks:
        if other is policy_network or other_id != network_id:
            continue
        encoder = getattr(other, "encoder", None)
        if encoder is None:
            continue
        # share_encoder_parameters writes the policy encoder's values into the
        # other encoders as plain detached tensors, so a borrowed encoder is
        # exactly one whose weight layers are no longer nn.Parameter.
        layers = [
            module
            for _name, module in encoder.named_modules()
            if _is_weight_layer(module)
        ]
        if not layers or any(_owns_trainable_weight(layer) for layer in layers):
            continue
        entries.extend(_head_entry_layers(getattr(other, "head_net", None)))
    return entries


def _dormant_indices(per_neuron: torch.Tensor, dormant_threshold: float) -> list[int]:
    """The indices of the layer's dormant neurons, ascending.

    Scores are normalised by the layer mean. A layer whose mean is zero has no
    live unit left and is reported entirely dormant. Non-finite scores are
    coerced to zero, i.e. treated as dormant: a diverged unit is precisely one
    worth re-initialising.
    """
    scores = torch.nan_to_num(
        per_neuron.detach(),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    if scores.numel() == 0:
        return []
    mean = float(scores.mean())
    if mean <= 0.0:
        return list(range(scores.numel()))
    normalised = scores / mean
    return torch.nonzero(normalised <= dormant_threshold).flatten().tolist()


def _resolve_producer_and_next(
    act_module: nn.Module,
    encoder: nn.Module | None,
    head: nn.Module | None,
) -> _ProducerContext:
    """Find the layer that produced act_module's neurons and its consumers.

    Any normalisation applied between the producer and the activation is returned
    alongside them, tracked as "the last norm seen since the running producer" and
    cleared whenever a later weight layer takes over. That is what distinguishes
    an evolvable MLP's linear -> layer_norm -> activation from a SimBa block's
    layer_norm -> linear -> activation, where the norm applies to the block's
    input and leaves these neurons alone.

    Every field is empty when the activation cannot be located, so the caller
    skips it.
    """
    for root, is_encoder in ((encoder, True), (head, False)):
        search_root = _unwrap_module(root)
        if search_root is None:
            continue

        ordered = list(search_root.named_modules())
        name = next((n for n, m in ordered if m is act_module), None)
        if name is None:
            continue

        parent = name.rpartition(".")[0]
        prefix = f"{parent}." if parent else ""
        container = name.split(".")[0]

        producer: nn.Module | None = None
        norm: nn.Module | None = None
        consumers: list[nn.Module] = []
        enclosing: nn.Module | None = None
        passed = False
        for other_name, other in ordered:
            if other is act_module:
                passed = True
                continue
            in_stream = other_name.startswith(prefix)
            if not passed and in_stream and isinstance(other, NORM_LAYER_TYPES):
                norm = other
                continue
            if not _is_weight_layer(other):
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
                [enclosing] if enclosing is not None else _head_entry_layers(head)
            )

        return _ProducerContext(producer, norm, consumers)

    return _ProducerContext(None, None, [])


def _live_column_scale(weight: torch.Tensor, stride: int, keep: list[int]) -> float:
    """The median outgoing-column norm of a consumer over keep, strictly positive."""
    if weight.dim() > 2:  # conv consumer: one filter per producer neuron
        blocks = weight.reshape(weight.shape[0], weight.shape[1], -1)
        fan_out = blocks.shape[0] * blocks.shape[2]  # conv fans count the kernel
    else:
        blocks = weight.reshape(weight.shape[0], -1, stride)
        fan_out = blocks.shape[0]

    def median_norm(indices: list[int] | None) -> float:
        selected = blocks if indices is None else blocks[:, indices, :]
        if selected.shape[1] == 0:
            return 0.0
        norms = selected.pow(2).sum(dim=(0, 2)).sqrt()
        norms = norms[norms.isfinite()]
        return float(norms.median()) if norms.numel() else 0.0

    for candidate in (median_norm(keep) if keep else 0.0, median_norm(None)):
        if candidate > 0.0:
            return candidate

    fan_in = blocks.shape[1] * blocks.shape[2]
    block_entries = blocks.shape[0] * blocks.shape[2]
    bound = float(np.sqrt(6.0 / (fan_in + fan_out)))
    # RMS of U(-bound, bound) is bound / sqrt(3).
    return bound / np.sqrt(3.0) * float(np.sqrt(block_entries))


def _noise_params(
    module: nn.Module,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """A noisy layer's (weight_sigma, bias_sigma) data tensors."""
    weight_sigma = getattr(module, "weight_sigma", None)
    bias_sigma = getattr(module, "bias_sigma", None)
    return (
        weight_sigma.data if weight_sigma is not None else None,
        bias_sigma.data if bias_sigma is not None else None,
    )


def _revived_block(
    template: torch.Tensor,
    scale: float,
    rng: np.random.Generator,
    *,
    is_noise_scale: bool = False,
) -> torch.Tensor:
    """Draw the outgoing block a reset neuron is revived with, of norm scale.

    A weight block is a random direction rescaled to scale. A noise-scale block
    is instead a single non-negative value, sized the way the weight block is but
    against the noise scales' own live columns rather than the weights'.
    """
    if scale <= 0.0 or template.numel() == 0:
        return torch.zeros_like(template)

    if is_noise_scale:
        return torch.full_like(template, scale / float(np.sqrt(template.numel())))

    sampled = rng.standard_normal(size=tuple(template.shape))
    block = torch.as_tensor(sampled, dtype=template.dtype, device=template.device)
    norm = float(block.norm())
    if norm <= 0.0:
        return torch.zeros_like(template)
    return block * (scale / norm)


def _resolve_consumers(
    producer: nn.Module,
    next_layers: list[nn.Module],
    cnn_channels: int | None,
    cnn_spatial: int | None,
) -> list[_ConsumerTarget]:
    """Pair each usable consumer weight tensor with its per-neuron column stride.

    A consumer must spend its columns on exactly these neurons: one block each,
    none interleaved. Anything failing that is not this producer's consumer and
    rewriting its columns would corrupt weights belonging to other neurons, so it
    is skipped instead. Each noisy consumer's parallel noise-scale tensor is
    returned as a target of its own.
    """
    producer_is_conv = isinstance(producer, CONV_LAYER_TYPES)
    producer_neurons = _weight_param(producer).data.shape[0]
    consumers: list[_ConsumerTarget] = []
    for next_layer in next_layers:
        # A dense layer feeding a convolution is a pairing the surgery cannot index.
        next_is_conv = isinstance(next_layer, CONV_LAYER_TYPES)
        if next_is_conv and not producer_is_conv:
            continue

        next_weight = _weight_param(next_layer).data
        stride = 1
        if producer_is_conv and not next_is_conv:
            if cnn_spatial is None or cnn_channels is None:
                logger.debug(
                    "ReGraMa: no flattened column layout for %s -> %s; "
                    "leaving the layer unreset.",
                    type(producer).__name__,
                    type(next_layer).__name__,
                )
                continue
            stride = cnn_spatial

        if next_weight.shape[1] != producer_neurons * stride:
            logger.debug(
                "ReGraMa: %s spends %d columns where %s's neurons need %d; "
                "leaving the layer unreset.",
                type(next_layer).__name__,
                next_weight.shape[1],
                type(producer).__name__,
                producer_neurons * stride,
            )
            continue

        consumers.append(_ConsumerTarget(next_weight, stride))

        # The consumer's own noise columns are scales rather than weights and are
        # revived as such.
        next_sigma, _next_bias_sigma = _noise_params(next_layer)
        if next_sigma is not None and next_sigma.shape == next_weight.shape:
            consumers.append(_ConsumerTarget(next_sigma, stride, is_noise_scale=True))

    return consumers


def _shared_latent_blocks(
    producer: nn.Module,
    entry_layers: Sequence[nn.Module],
) -> list[_ConsumerTarget]:
    """Each sharing head's latent columns as a writable view, noise scales included."""
    if isinstance(producer, CONV_LAYER_TYPES):
        return []

    span = _weight_param(producer).data.shape[0]
    blocks: list[_ConsumerTarget] = []
    for entry in entry_layers:
        weight = _weight_param(entry).data
        if weight.dim() != 2 or weight.shape[1] < span:
            continue
        blocks.append(_ConsumerTarget(weight[:, :span], 1))
        sigma, _bias_sigma = _noise_params(entry)
        if sigma is not None and sigma.shape == weight.shape:
            blocks.append(_ConsumerTarget(sigma[:, :span], 1, is_noise_scale=True))
    return blocks


def _reset_layer_neurons(
    producer: nn.Module,
    consumers: list[_ConsumerTarget],
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
    producer_weight = _weight_param(producer).data
    producer_bias = getattr(producer, "bias_mu", None)
    if producer_bias is None:
        producer_bias = getattr(producer, "bias", None)
    producer_bias = (
        producer_bias.data if isinstance(producer_bias, torch.Tensor) else None
    )
    sigma_weight, sigma_bias = _noise_params(producer)
    neurons = producer_weight.shape[0]

    # Xavier-uniform bound of the producing layer.
    fan_out = producer_weight.shape[0]
    if producer_weight.dim() == 2:  # Linear: (out_features, in_features)
        fan_in = producer_weight.shape[1]
    else:  # Conv: (out_channels, in_channels, *kernel)
        receptive = 1
        for dim in producer_weight.shape[2:]:
            receptive *= int(dim)
        fan_in = int(producer_weight.shape[1]) * receptive
        fan_out = fan_out * receptive
    bound = float(np.sqrt(6.0 / (fan_in + fan_out)))

    # A revived unit wants the layer's initial noise scale, not the collapsed or
    # inflated one it inherits.
    noise_fills: tuple[float, float] | None = None
    if sigma_weight is not None and sigma_weight.dim() >= 2:
        std_init = float(getattr(producer, "std_init", 0.5))
        noise_fills = (
            std_init / float(np.sqrt(sigma_weight.shape[1])),
            std_init / float(np.sqrt(sigma_weight.shape[0])),
        )

    # Measure against the neurons this pass leaves alone.
    keep = [n for n in range(neurons) if n not in set(indices)]
    out_scales = [
        REGRAMA_OUT_SCALE * _live_column_scale(target.weight, target.stride, keep)
        for target in consumers
    ]

    for index in indices:
        sampled = rng.uniform(-bound, bound, size=tuple(producer_weight[index].shape))
        producer_weight[index] = torch.as_tensor(
            sampled,
            dtype=producer_weight.dtype,
            device=producer_weight.device,
        )
        if producer_bias is not None:
            producer_bias[index] = 0.0
        if noise_fills is not None:
            weight_fill, bias_fill = noise_fills
            if sigma_weight is not None:
                sigma_weight[index] = weight_fill
            if sigma_bias is not None:
                sigma_bias[index] = bias_fill
        for target, scale in zip(consumers, out_scales, strict=True):
            weight, stride = target.weight, target.stride
            block = weight[:, index * stride : (index + 1) * stride]
            weight[:, index * stride : (index + 1) * stride] = _revived_block(
                block,
                scale,
                rng,
                is_noise_scale=target.is_noise_scale,
            )
        if norm is not None:
            for tensor, identity in (
                (getattr(norm, "weight", None), 1.0),
                (getattr(norm, "bias", None), 0.0),
                (getattr(norm, "running_mean", None), 0.0),
                (getattr(norm, "running_var", None), 1.0),
            ):
                if (
                    tensor is not None
                    and tensor.dim() == 1
                    and tensor.shape[0] == neurons
                ):
                    tensor.data[index] = identity

    # Defensive clamp and NaN scrub so a reset never propagates a broken value.
    for tensor in (producer_weight, producer_bias, sigma_weight, sigma_bias):
        if tensor is not None:
            tensor.clamp_(-MAGNITUDE_LIMIT, MAGNITUDE_LIMIT).nan_to_num_()
    for target in consumers:
        target.weight.clamp_(-MAGNITUDE_LIMIT, MAGNITUDE_LIMIT).nan_to_num_()


def reset_dormant_neurons(
    network: nn.Module,
    per_neuron_list: list[torch.Tensor | None] | None,
    dormant_threshold: float,
    rng: np.random.Generator,
    shared_latent_heads: Sequence[nn.Module] = (),
) -> int:
    """Reset every dormant neuron of one evaluation network.

    Walks the network's measured activations, resolves each one's producer,
    normalisation and consumers, and re-initialises the neurons whose normalised
    GraMa score is at or below dormant_threshold. Layers whose producer is
    unresolvable, has no usable consumer, or does not own its weights are skipped
    untouched.

    :param network: An evaluation network to operate on, in place.
    :type network: torch.nn.Module
    :param per_neuron_list: That network's entry of the captured snapshot.
    :type per_neuron_list: list[torch.Tensor | None] | None
    :param dormant_threshold: Normalised score at or below which a neuron is
        dormant.
    :type dormant_threshold: float
    :param rng: Seeded generator owned by the caller.
    :type rng: numpy.random.Generator
    :param shared_latent_heads: Head entry layers of the networks sharing this
        network's encoder.
    :type shared_latent_heads: Sequence[torch.nn.Module]
    :return: How many neurons were reset.
    :rtype: int
    """
    # A network with a length mismatch is skipped so scores are not mis-paired
    # with layers.
    targets = target_activations(network)
    if not per_neuron_list or len(per_neuron_list) != len(targets):
        return 0
    scores = [
        (act_module, per_neuron)
        for act_module, per_neuron in zip(targets, per_neuron_list, strict=False)
        if per_neuron is not None
    ]
    if not scores:
        return 0

    encoder = getattr(network, "encoder", None)
    head = getattr(network, "head_net", None)

    cnn_dims: dict[int, tuple[int, int]] = {}
    if encoder is not None:
        for _name, sub in encoder.named_modules():
            shape = getattr(sub, "cnn_output_size", None)
            if shape is None or len(shape) < 3:
                continue
            spatial = 1
            for dim in shape[2:]:
                spatial *= int(dim)
            for _child_name, child in sub.named_modules():
                cnn_dims[id(child)] = (int(shape[1]), spatial)

    head_entries = _head_entry_layers(head)

    neurons_reset = 0
    for act_module, per_neuron in scores:
        producer, norm, next_layers = _resolve_producer_and_next(
            act_module,
            encoder,
            head,
        )
        if producer is None or not next_layers or not _owns_trainable_weight(producer):
            continue
        if per_neuron.numel() != _weight_param(producer).data.shape[0]:
            continue

        cnn_channels, cnn_spatial = cnn_dims.get(id(producer), (None, None))
        consumers = _resolve_consumers(producer, next_layers, cnn_channels, cnn_spatial)
        if not consumers:
            continue

        if shared_latent_heads and any(
            layer is entry for layer in next_layers for entry in head_entries
        ):
            consumers = consumers + _shared_latent_blocks(producer, shared_latent_heads)

        indices = _dormant_indices(per_neuron, dormant_threshold)
        if not indices:
            continue

        _reset_layer_neurons(producer, consumers, norm, indices, rng)
        neurons_reset += len(indices)

    return neurons_reset
