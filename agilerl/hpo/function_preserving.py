"""Function-preserving architecture-mutation surgery (Net2Net-style).

AgileRL's default architecture mutations rebuild a network from scratch and copy
back only the overlapping top-left block of every weight tensor
(:func:`agilerl.modules.base.EvolvableModule.preserve_parameters`). New units are
therefore left at fresh random init (their outgoing weights immediately change the
function) and removals drop the *first* ``N`` units positionally. This module
implements the function-preserving variants selected by
``arch_mut_type == "func_preserving"`` on the ``mutation:`` manifest block:

* **add_node / add_channel** -- the new units keep their native orthogonal-gain-√2
  incoming weights (what ``create_mlp`` / ``create_cnn`` already produce) but their
  *outgoing* weights are set to zero, so they contribute nothing initially and the
  function is preserved (Net2WiderNet with zeroed fan-out).
* **remove_node / remove_channel** -- before the standard positional removal (which
  keeps the first ``N`` units), every hidden layer of the mutated sub-network is
  reordered by a *function-preserving permutation* so its highest-activation units
  come first; the standard removal then drops the lowest-activation units.
* **add_layer** (head MLP only -- the encoder has LAYER mutations disabled) -- the
  newly inserted layer is initialised to the identity (Net2DeeperNet). Exact when
  the activation is ReLU / Identity; a warning is emitted otherwise.

All surgery operates purely on the weight tensors of the mutated module (both the
producing layer and the single consuming layer live inside the same
``EvolvableMLP`` / ``EvolvableCNN`` ``self.model``), so no cross-module plumbing is
needed. The convolutional last-layer -> flatten -> ``_linear_output`` boundary is
handled by treating each channel's flattened features as a contiguous ``H*W`` block
(``nn.Flatten`` is channel-outermost).

The functions here are pure tensor operations (given per-neuron activation scores);
they import no plotting code and are unit-tested standalone.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn

from agilerl.utils.dormant_neurons import (
    _activation_modules,
    capture_per_neuron_scores,
)

# Base mutation names (with any ``<agent_id>.`` / ``<submodule>.`` prefix stripped).
ADD_NODE_MUTATIONS = frozenset({"add_node", "add_channel"})
ADD_LAYER_MUTATIONS = frozenset({"add_layer"})
REMOVE_MUTATIONS = frozenset({"remove_node", "remove_channel"})
# Function-preserving activations for Net2DeeperNet identity layers.
_PRESERVING_ACTIVATIONS = frozenset({"ReLU", "Identity", None})


def parse_mut_target(mut_method: str) -> tuple[str | None, str | None, str]:
    """Split a mutation-method name into ``(agent_id, submodule, base_name)``.

    Single-agent methods look like ``"head_net.add_node"`` (``agent_id`` is
    ``None``); multi-agent ``ModuleDict`` methods look like
    ``"agent_0.head_net.add_node"``.

    :param mut_method: The mutation-method attribute name.
    :return: ``(agent_id, submodule, base_name)`` where ``submodule`` is
        ``"encoder"`` / ``"head_net"`` (or ``None`` if the name is unprefixed).
    """
    parts = mut_method.split(".")
    base = parts[-1]
    submodule = parts[-2] if len(parts) >= 2 else None
    agent_id = ".".join(parts[:-2]) if len(parts) >= 3 else None
    return agent_id, submodule, base


def resolve_target(
    network: Any, agent_id: str | None, submodule_name: str
) -> tuple[nn.Module, nn.Module]:
    """Resolve the forward-network and the target evolvable sub-module.

    :param network: Either an ``EvolvableNetwork`` (single sub-network) or a
        ``ModuleDict`` of per-agent sub-networks.
    :param agent_id: The ``ModuleDict`` key, or ``None`` when *network* is already
        a single sub-network.
    :param submodule_name: ``"encoder"`` or ``"head_net"``.
    :return: ``(forward_network, submodule)`` -- the network that accepts an
        observation batch and the evolvable module being mutated.
    """
    fwd_net = network[agent_id] if agent_id is not None else network
    submodule = getattr(fwd_net, submodule_name)
    return fwd_net, submodule


# --------------------------------------------------------------------------- #
# Low-level tensor helpers
# --------------------------------------------------------------------------- #
def _is_conv(layer: nn.Module) -> bool:
    return isinstance(layer, (nn.Conv2d, nn.Conv3d))


def _is_linear(layer: nn.Module) -> bool:
    if isinstance(layer, nn.Linear):
        return True
    weight_mu = getattr(layer, "weight_mu", None)
    return weight_mu is not None and weight_mu.dim() == 2


def _is_weight_layer(layer: nn.Module) -> bool:
    """Whether *layer* is a channel/feature transform (conv or (noisy) linear)."""
    return _is_conv(layer) or _is_linear(layer)


def _inner_module(submodule: nn.Module) -> nn.Module:
    """Return the module that actually owns the ``model`` sequential.

    Stochastic-policy heads wrap their evolvable MLP in an
    ``EvolvableDistribution`` (exposed as ``wrapped``); the encoder and value/Q
    heads own ``model`` directly.
    """
    if getattr(submodule, "model", None) is not None:
        return submodule
    wrapped = getattr(submodule, "wrapped", None)
    if wrapped is not None and getattr(wrapped, "model", None) is not None:
        return wrapped
    return submodule


def _ordered_weight_layers(submodule: nn.Module) -> list[nn.Module]:
    """Return the conv/linear layers of ``submodule.model`` in forward order.

    Norm, activation and flatten layers are skipped, so the returned list is
    ``[hidden_0, ..., hidden_{N-1}, output]`` for both MLP and CNN modules.
    ``EvolvableDistribution`` heads are unwrapped to their inner MLP first.
    """
    model = getattr(_inner_module(submodule), "model", None)
    if model is None:
        return []
    return [m for m in model if _is_weight_layer(m)]


def _weight_tensors(layer: nn.Module) -> list[torch.Tensor]:
    """All 2-D/4-D weight tensors of *layer* (handles ``NoisyLinear``)."""
    names = ("weight", "weight_mu", "weight_sigma", "weight_epsilon")
    return [getattr(layer, n) for n in names if getattr(layer, n, None) is not None]


def _bias_tensors(layer: nn.Module) -> list[torch.Tensor]:
    """All 1-D bias tensors of *layer* (handles ``NoisyLinear``)."""
    names = ("bias", "bias_mu", "bias_sigma", "bias_epsilon")
    return [getattr(layer, n) for n in names if getattr(layer, n, None) is not None]


def _primary_weight(layer: nn.Module) -> torch.Tensor:
    """The layer's main weight tensor (``weight`` or ``weight_mu``)."""
    weight = getattr(layer, "weight", None)
    return weight if weight is not None else layer.weight_mu


def _out_dim(layer: nn.Module) -> int:
    return int(_primary_weight(layer).shape[0])


def _spatial_size(submodule: nn.Module) -> int:
    """Flattened spatial size ``H*W`` of a CNN encoder's last conv output."""
    size = getattr(_inner_module(submodule), "cnn_output_size", None)
    if size is None or len(size) <= 2:
        return 1
    return int(math.prod(int(d) for d in size[2:]))


def _permute_out(layer: nn.Module, perm: torch.Tensor) -> None:
    """Reorder a layer's output units (dim 0 of weights, plus biases)."""
    for w in _weight_tensors(layer):
        w.data = w.data[perm].contiguous()
    for b in _bias_tensors(layer):
        b.data = b.data[perm].contiguous()


def _permute_in(layer: nn.Module, perm: torch.Tensor, block: int = 1) -> None:
    """Reorder a layer's input units (dim 1 of weights) by *perm*.

    When *block* > 1 the input features are grouped into contiguous blocks of that
    size (the conv-last-layer -> flatten -> linear boundary, one ``H*W`` block per
    channel) and whole blocks are permuted.
    """
    for w in _weight_tensors(layer):
        if block == 1:
            w.data = w.data[:, perm].contiguous()
        else:
            out_features = w.shape[0]
            grouped = w.data.view(out_features, -1, block)
            grouped = grouped[:, perm, :]
            w.data = grouped.reshape(out_features, -1).contiguous()


# --------------------------------------------------------------------------- #
# The three function-preserving operations
# --------------------------------------------------------------------------- #
def zero_new_outgoing(
    submodule: nn.Module, hidden_layer: int, old_width: int | None
) -> int:
    """Zero the outgoing weights of newly added units after add_node/add_channel.

    The new units are the trailing ``new_width - old_width`` rows of the producing
    layer; their outgoing weights are the matching input slice of the consuming
    layer. Zeroing them makes the addition function-preserving.

    :param submodule: The mutated ``EvolvableMLP`` / ``EvolvableCNN``.
    :param hidden_layer: Index of the widened hidden layer.
    :param old_width: The layer's output width *before* the mutation (``None`` or a
        value ``>= new_width`` means nothing was actually added -- a no-op).
    :return: The number of units whose outgoing weights were zeroed.
    """
    layers = _ordered_weight_layers(submodule)
    if old_width is None or not 0 <= hidden_layer < len(layers) - 1:
        return 0

    producer = layers[hidden_layer]
    consumer = layers[hidden_layer + 1]
    new_width = _out_dim(producer)
    num_added = new_width - old_width
    if num_added <= 0:
        return 0

    conv_to_linear = _is_conv(producer) and _is_linear(consumer)
    block = _spatial_size(submodule) if conv_to_linear else 1
    with torch.no_grad():
        for w in _weight_tensors(consumer):
            w.data[:, old_width * block : new_width * block] = 0.0
    return num_added


def identity_new_layer(submodule: nn.Module) -> bool:
    """Set a freshly inserted head layer to the identity (Net2DeeperNet).

    ``add_layer`` appends a hidden layer of the *same* width as the one below, so
    the new layer is the penultimate weight layer and is square; the output layer
    is preserved exactly by ``preserve_parameters``. Overwriting the new layer with
    an identity weight (and zero bias) makes the deepening function-preserving when
    the activation is ReLU / Identity.

    :param submodule: The mutated head ``EvolvableMLP``.
    :return: ``True`` if an identity layer was written, ``False`` otherwise.
    """
    layers = _ordered_weight_layers(submodule)
    if len(layers) < 2:
        return False
    new_layer = layers[-2]
    weight = _primary_weight(new_layer)
    if weight.dim() != 2 or weight.shape[0] != weight.shape[1]:
        return False

    with torch.no_grad():
        if getattr(new_layer, "weight", None) is not None:
            new_layer.weight.data.zero_()
            new_layer.weight.data.fill_diagonal_(1.0)
        if getattr(new_layer, "weight_mu", None) is not None:
            new_layer.weight_mu.data.zero_()
            new_layer.weight_mu.data.fill_diagonal_(1.0)
        if getattr(new_layer, "weight_sigma", None) is not None:
            new_layer.weight_sigma.data.zero_()
        for b in _bias_tensors(new_layer):
            b.data.zero_()
    return True


def permute_submodule_by_activation(
    fwd_net: nn.Module, submodule_name: str, obs: Any
) -> None:
    """Function-preservingly reorder every hidden layer by descending activation.

    A single forward pass on *obs* scores each measured unit by its mean absolute
    activation. Each hidden layer's units are then permuted so the most active come
    first, moving the producing layer's output rows/bias *and* the consuming
    layer's input columns together (a consistent relabelling that leaves the
    function unchanged). The subsequent standard removal -- which keeps the first
    ``N`` units -- therefore drops the least-active ones.

    :param fwd_net: The (sub-)network that accepts *obs* (encoder + ``head_net``).
    :param submodule_name: ``"encoder"`` or ``"head_net"`` -- the module to reorder.
    :param obs: A preprocessed observation batch accepted by *fwd_net*.
    """
    submodule = getattr(fwd_net, submodule_name)
    layers = _ordered_weight_layers(submodule)
    num_hidden = len(layers) - 1
    if num_hidden < 1:
        return

    acts = _activation_modules(submodule, include_output=(submodule_name == "encoder"))
    captured = dict(capture_per_neuron_scores(fwd_net, obs))
    block = _spatial_size(submodule)

    for i in range(num_hidden):
        if i >= len(acts):
            break
        scores = captured.get(acts[i])
        if scores is None:
            continue
        producer = layers[i]
        consumer = layers[i + 1]
        if scores.numel() != _out_dim(producer):
            continue
        perm = torch.sort(scores, descending=True, stable=True).indices
        perm = perm.to(_primary_weight(producer).device)
        _permute_out(producer, perm)
        if _is_conv(producer) and _is_linear(consumer):
            _permute_in(consumer, perm, block=block)
        else:
            _permute_in(consumer, perm, block=1)


def hidden_widths(submodule: nn.Module) -> list[int]:
    """Output widths of a module's hidden (non-output) weight layers."""
    layers = _ordered_weight_layers(submodule)
    return [_out_dim(layer) for layer in layers[:-1]]


def has_norm_layer(network: nn.Module) -> bool:
    """Whether *network* contains a normalisation layer.

    A norm layer (LayerNorm/BatchNorm/GroupNorm) re-normalises over the changed
    unit set, so it breaks function preservation for *every* add mutation
    (add_node/add_channel/add_layer) regardless of the activation.
    """
    # NOTE: ``EvolvableModule`` overrides ``.modules()`` to return group names, so
    # iterate ``named_modules()`` (which recurses over the real sub-modules).
    for _name, module in network.named_modules():
        if isinstance(
            module,
            (
                nn.LayerNorm,
                nn.BatchNorm1d,
                nn.BatchNorm2d,
                nn.BatchNorm3d,
                nn.GroupNorm,
            ),
        ):
            return True
    return False


def has_nonpreserving_activation(network: nn.Module) -> bool:
    """Whether any sub-network uses a non-ReLU/Identity base activation.

    Only add_layer (Net2DeeperNet identity init) relies on the base activation
    being ReLU/Identity. add_node/add_channel zero the new units' fan-out and so
    stay function-preserving under *any* activation, so this check must not gate
    their warning.
    """
    for submodule_name in ("encoder", "head_net"):
        submodule = getattr(network, submodule_name, None)
        if submodule is None:
            continue
        activation = getattr(_inner_module(submodule), "activation", None)
        if activation not in _PRESERVING_ACTIVATIONS:
            return True
    return False


def has_layernorm_or_nonrelu(network: nn.Module) -> bool:
    """Whether *network* uses a norm layer or a non-ReLU/Identity activation.

    Retained for the add_layer caveat, whose Net2DeeperNet identity init requires
    both no norm layer and a ReLU/Identity activation.
    """
    return has_norm_layer(network) or has_nonpreserving_activation(network)
