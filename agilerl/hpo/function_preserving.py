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
* **remove_node / remove_channel** -- the units removed are exactly the
  **τ-dormant** ones of the sampled layer (Sokar et al. 2023, Definition 3.1:
  normalised mean absolute activation ``s_i <= tau``), which by definition
  contribute ~nothing to the layer's output. Mechanically, every hidden layer of
  the mutated sub-network is first reordered by a *function-preserving
  permutation* so its highest-activation units come first, and the removal count
  handed to the standard positional removal is the dormant count of the sampled
  layer -- so the standard removal drops precisely those units. A layer with no
  dormant unit yields a no-op removal (there is nothing removable without
  changing the function); a dormant count larger than the module's
  ``min_mlp_nodes`` / ``min_channel_size`` / ``min_latent_dim`` budget is clamped
  to the most-dormant units that fit.
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
they import no plotting code and are unit-tested standalone. The scores are measured
on a fresh observation batch collected at mutation time (see
:func:`~agilerl.utils.dormant_neurons.collect_observation_batch`), so a removal costs
one env rollout plus one forward pass per mutated network.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn

from agilerl.utils.dormant_neurons import (
    _activation_modules,
    _per_neuron_score,
    capture_per_neuron_scores,
    normalised_scores,
)

# Base mutation names (with any ``<agent_id>.`` / ``<submodule>.`` prefix stripped).
ADD_NODE_MUTATIONS = frozenset({"add_node", "add_channel"})
ADD_LAYER_MUTATIONS = frozenset({"add_layer"})
REMOVE_MUTATIONS = frozenset({"remove_node", "remove_channel"})
# Latent-dimension mutations cross the encoder->head boundary (the producing layer
# is the encoder's output, the consuming layer is the head's first layer), so they
# are handled separately from the within-sub-module node/channel operators above.
LATENT_ADD_MUTATIONS = frozenset({"add_latent_node", "add_latent_channel"})
LATENT_REMOVE_MUTATIONS = frozenset({"remove_latent_node", "remove_latent_channel"})
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


def module_rng(submodule: nn.Module) -> Any | None:
    """The RNG the evolvable sub-module's own mutation methods draw from.

    A removal's target layer has to be chosen *before* the mutation method runs
    (the dormant count is layer-specific), so the caller draws it here instead --
    from the same generator the method would have used, keeping the layer choice
    identically distributed.

    :param submodule: The evolvable sub-module being mutated.
    :return: Its ``numpy`` generator, or ``None`` if it exposes none.
    """
    return getattr(_inner_module(submodule), "rng", None)


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
def _existing_columns_std(weight: torch.Tensor, col_slice: slice) -> float:
    """Std of the weight's columns *outside* ``col_slice`` (the pre-existing fan-out).

    Used to scale symmetry-breaking noise to the layer's own weight magnitude, so
    the ``arch_fp_noise`` factor is architecture/scale-invariant. Falls back to the
    full-tensor std (then a tiny constant) when there are no existing columns or the
    existing block is (near-)constant, so the new units still break symmetry.
    """
    ncols = int(weight.shape[1])
    mask = torch.ones(ncols, dtype=torch.bool, device=weight.device)
    mask[col_slice] = False
    existing = weight.data[:, mask]
    std = float(existing.std()) if existing.numel() else 0.0
    if std <= 0.0:
        std = float(weight.data.std())
    if std <= 0.0:
        std = 1e-3
    return std


def _fill_new_block(weight: torch.Tensor, col_slice: slice, noise_scale: float) -> None:
    """Fill a block of new input columns of *weight* (in place, no grad).

    ``noise_scale <= 0`` sets them to exactly zero (the original function-preserving
    behaviour -- no RNG is drawn, so seeded runs stay byte-identical). A positive
    value seeds them with ``randn * (noise_scale * sigma)`` where ``sigma`` is the
    std of the existing columns, breaking the new units' symmetry so their incoming
    weights receive gradient (see ``arch_fp_noise``). The draw uses the global torch
    RNG (seeded in ``Mutations.__init__``), matching ``_gaussian_parameter_mutation``.
    """
    if noise_scale <= 0.0:
        weight.data[:, col_slice] = 0.0
        return
    sigma = _existing_columns_std(weight, col_slice)
    block = weight.data[:, col_slice]
    weight.data[:, col_slice] = torch.randn_like(block) * (noise_scale * sigma)


def _init_new_input_columns(
    consumer: nn.Module, col_slice: slice, noise_scale: float
) -> None:
    """Zero/noise the new input columns of a consuming layer's weight tensors.

    Only the primary weight (``weight`` / ``weight_mu``) is noised; any auxiliary
    ``NoisyLinear`` tensors (``weight_sigma`` / ``weight_epsilon``) are zeroed so the
    new units add no extra stochasticity, matching the original zeroing behaviour.
    """
    primary = _primary_weight(consumer)
    with torch.no_grad():
        for w in _weight_tensors(consumer):
            if w is primary:
                _fill_new_block(w, col_slice, noise_scale)
            else:
                w.data[:, col_slice] = 0.0


def init_new_outgoing(
    submodule: nn.Module,
    hidden_layer: int,
    old_width: int | None,
    noise_scale: float = 0.0,
) -> int:
    """Initialise the outgoing weights of newly added units after add_node/add_channel.

    The new units are the trailing ``new_width - old_width`` rows of the producing
    layer; their outgoing weights are the matching input slice of the consuming
    layer. Setting them to zero (``noise_scale == 0``) makes the addition exactly
    function-preserving; a small ``noise_scale`` breaks their symmetry instead (see
    :func:`_fill_new_block`).

    :param submodule: The mutated ``EvolvableMLP`` / ``EvolvableCNN``.
    :param hidden_layer: Index of the widened hidden layer.
    :param old_width: The layer's output width *before* the mutation (``None`` or a
        value ``>= new_width`` means nothing was actually added -- a no-op).
    :param noise_scale: Symmetry-breaking noise factor (``arch_fp_noise``); ``0``
        keeps the exact-zero, function-preserving behaviour.
    :return: The number of units whose outgoing weights were (re)initialised.
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
    col_slice = slice(old_width * block, new_width * block)
    _init_new_input_columns(consumer, col_slice, noise_scale)
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


def dormant_removal_count(
    scores: torch.Tensor | None, tau: float, budget: int
) -> tuple[int, int]:
    """Number of τ-dormant units of a layer, and how many of them may be removed.

    Implements the removal rule of Definition 3.1 (Sokar et al. 2023): a unit is
    dormant when its activation, normalised by the layer mean, is ``<= tau``. Only
    dormant units are removable without changing the layer's function, so a layer
    with none of them yields ``0`` -- a deliberate no-op removal rather than an
    arbitrary one.

    :param scores: Mean absolute activation of each unit of the layer, or ``None``
        when no activation was captured for it.
    :type scores: torch.Tensor | None
    :param tau: Dormancy threshold.
    :type tau: float
    :param budget: The largest count the module's hard limit will actually apply
        (see :func:`removal_budget`); the dormant count is clamped to it, which
        drops the *most* dormant units since the layer has been reordered by
        descending activation.
    :type budget: int
    :return: ``(dormant_count, removal_count)`` -- the raw count of τ-dormant units
        and that count clamped to *budget*. Both are ``0`` when *scores* is
        unavailable or holds no finite entry.
    :rtype: tuple[int, int]
    """
    if scores is None:
        return 0, 0
    normalised = normalised_scores(scores)
    if normalised is None:
        return 0, 0
    dormant = int((normalised <= tau).sum().item())
    return dormant, max(0, min(dormant, budget))


def removal_budget(current_width: int, min_width: int, *, strict: bool = True) -> int:
    """Largest removal count a module's hard limit will actually apply.

    The evolvable modules guard their removals with a minimum width and silently
    skip the shrink when the requested count does not fit, so the count must be
    clamped here or a large dormant set would remove *nothing*. The guard is
    ``width - n > min_width`` for :class:`~agilerl.modules.mlp.EvolvableMLP` and
    the latent dimension (strict) but ``width - n >= min_width`` for
    :class:`~agilerl.modules.cnn.EvolvableCNN` (inclusive), hence *strict*.

    :param current_width: The layer's current width.
    :type current_width: int
    :param min_width: The module's minimum width for that layer.
    :type min_width: int
    :param strict: Whether the module's guard is strict (``>``) or inclusive
        (``>=``), defaults to True.
    :type strict: bool, optional
    :return: The maximum number of units that may be removed (``0`` if the layer
        already sits at its floor).
    :rtype: int
    """
    return max(0, current_width - min_width - (1 if strict else 0))


def layer_removal_budget(submodule: nn.Module, hidden_layer: int) -> int:
    """Removal budget of one hidden layer of an evolvable MLP / CNN sub-module.

    :param submodule: The ``EvolvableMLP`` / ``EvolvableCNN`` being mutated.
    :param hidden_layer: Index of the hidden layer the removal targets.
    :return: The maximum number of units removable from that layer, ``0`` when the
        layer index is out of range or no minimum-width attribute is exposed.
    """
    widths = hidden_widths(submodule)
    if not 0 <= hidden_layer < len(widths):
        return 0
    inner = _inner_module(submodule)
    min_nodes = getattr(inner, "min_mlp_nodes", None)
    if min_nodes is not None:
        return removal_budget(widths[hidden_layer], int(min_nodes), strict=True)
    min_channels = getattr(inner, "min_channel_size", None)
    if min_channels is not None:
        # EvolvableCNN's guard is inclusive (``>= min_channel_size``).
        return removal_budget(widths[hidden_layer], int(min_channels), strict=False)
    return 0


def latent_removal_budget(fwd_net: Any) -> int:
    """Removal budget of the encoder->head latent dimension.

    :param fwd_net: The ``EvolvableNetwork`` being mutated.
    :return: The maximum number of latent units removable, ``0`` when the network
        does not expose the latent bounds.
    """
    latent_dim = getattr(fwd_net, "latent_dim", None)
    min_latent = getattr(fwd_net, "min_latent_dim", None)
    if latent_dim is None or min_latent is None:
        return 0
    return removal_budget(int(latent_dim), int(min_latent), strict=True)


def permute_submodule_by_activation(
    fwd_net: nn.Module, submodule_name: str, obs: Any
) -> list[torch.Tensor | None]:
    """Function-preservingly reorder every hidden layer by descending activation.

    A single forward pass on *obs* scores each measured unit by its mean absolute
    activation. Each hidden layer's units are then permuted so the most active come
    first, moving the producing layer's output rows/bias *and* the consuming layer's
    input columns together (a consistent relabelling that leaves the function
    unchanged). The subsequent standard removal -- which keeps the first ``N`` units
    -- therefore drops the least-active ones, and with ``N`` set from
    :func:`dormant_removal_count` those are exactly the τ-dormant units.

    :param fwd_net: The (sub-)network that accepts *obs* (encoder + ``head_net``).
    :param submodule_name: ``"encoder"`` or ``"head_net"`` -- the module to reorder.
    :param obs: A preprocessed observation batch accepted by *fwd_net*.
    :return: The scores used, one entry per hidden layer of the sub-module (in
        forward order), with ``None`` where no usable activation was measured.
        Returned so the caller can size the removal without a second forward pass;
        the values are the *pre*-permutation ones, which the count does not depend on.
    :rtype: list[torch.Tensor | None]
    """
    submodule = getattr(fwd_net, submodule_name)
    layers = _ordered_weight_layers(submodule)
    num_hidden = len(layers) - 1
    if num_hidden < 1:
        return []

    acts = _activation_modules(submodule, include_output=(submodule_name == "encoder"))
    captured = dict(capture_per_neuron_scores(fwd_net, obs))
    block = _spatial_size(submodule)

    layer_scores: list[torch.Tensor | None] = [None] * num_hidden
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
        layer_scores[i] = scores
        perm = torch.sort(scores, descending=True, stable=True).indices
        perm = perm.to(_primary_weight(producer).device)
        _permute_out(producer, perm)
        if _is_conv(producer) and _is_linear(consumer):
            _permute_in(consumer, perm, block=block)
        else:
            _permute_in(consumer, perm, block=1)
    return layer_scores


# --------------------------------------------------------------------------- #
# Latent-dimension mutations (encoder output <-> head input boundary)
# --------------------------------------------------------------------------- #
def is_latent_mutation(base: str) -> bool:
    """Whether *base* is an add/remove latent-dimension mutation."""
    return base in LATENT_ADD_MUTATIONS or base in LATENT_REMOVE_MUTATIONS


def parse_latent_target(mut_method: str) -> tuple[str | None, str]:
    """Split a latent mutation-method name into ``(agent_id, base_name)``.

    Latent mutations are defined on the ``EvolvableNetwork`` itself, so -- unlike
    node/channel/layer mutations -- they carry no ``encoder``/``head_net`` segment:
    single-agent names are bare (``"add_latent_node"``) and multi-agent
    ``ModuleDict`` names are ``"<agent_id>.add_latent_node"``.

    :param mut_method: The mutation-method attribute name.
    :return: ``(agent_id, base_name)`` where ``agent_id`` is the ``ModuleDict`` key
        (or ``None`` for a single-agent network).
    """
    parts = mut_method.split(".")
    base = parts[-1]
    agent_id = ".".join(parts[:-1]) or None
    return agent_id, base


def resolve_latent_network(network: Any, agent_id: str | None) -> Any:
    """Return the ``EvolvableNetwork`` (encoder + ``head_net``) a latent mutation targets."""
    return network[agent_id] if agent_id is not None else network


def head_first_layer(fwd_net: Any) -> nn.Module | None:
    """The head network's first weight layer (unwraps ``EvolvableDistribution``)."""
    head = getattr(fwd_net, "head_net", None)
    if head is None:
        return None
    layers = _ordered_weight_layers(head)
    return layers[0] if layers else None


def encoder_output_layer(fwd_net: Any) -> nn.Module | None:
    """The encoder's output (latent-producing) weight layer."""
    encoder = getattr(fwd_net, "encoder", None)
    if encoder is None:
        return None
    layers = _ordered_weight_layers(encoder)
    return layers[-1] if layers else None


def init_new_latent_outgoing(
    fwd_net: Any, old_latent: int | None, noise_scale: float = 0.0
) -> int:
    """Zero/noise the head's new input columns after a latent-dim widening.

    New latent units are the trailing input columns of the head's first weight
    layer (latent is flat, so ``block == 1``); filling them zero (or with small
    symmetry-breaking noise) makes ``add_latent_node`` / ``add_latent_channel``
    function-preserving across the encoder->head boundary.

    :param fwd_net: The mutated ``EvolvableNetwork`` (encoder + ``head_net``).
    :param old_latent: The latent dim *before* the mutation (``None`` or ``>=`` the
        new latent dim means nothing was added -- a no-op, e.g. capped at
        ``max_latent_dim``).
    :param noise_scale: Symmetry-breaking noise factor (``arch_fp_noise``).
    :return: The number of new latent units whose fan-out was (re)initialised.
    """
    consumer = head_first_layer(fwd_net)
    if consumer is None or old_latent is None:
        return 0
    new_latent = int(_primary_weight(consumer).shape[1])
    num_added = new_latent - old_latent
    if num_added <= 0:
        return 0
    _init_new_input_columns(consumer, slice(old_latent, new_latent), noise_scale)
    return num_added


def latent_scores(fwd_net: Any, obs: Any) -> torch.Tensor | None:
    """Mean absolute activation of each latent unit (the encoder's output).

    Scores the *latent* the head actually consumes -- i.e. ``encoder(obs)`` -- so it
    works whether or not the encoder exposes a latent output-activation sub-module
    (an MLP encoder has an ``Identity`` output activation, a CNN encoder's is named
    differently), unlike :func:`capture_per_neuron_scores` which only measures
    name-matched activation modules.

    Returns ``None`` rather than raising when the encoder cannot be run on *obs*
    alone (a recurrent encoder needs a hidden state), so the caller degrades to the
    original positional removal instead of failing the whole mutation.

    :param fwd_net: The ``EvolvableNetwork`` being mutated (encoder + ``head_net``).
    :param obs: A preprocessed observation batch accepted by *fwd_net*.
    :return: A 1-D tensor of per-latent-unit scores, or ``None`` if unavailable.
    """
    encoder = getattr(fwd_net, "encoder", None)
    if encoder is None:
        return None
    was_training = encoder.training
    try:
        encoder.eval()
        with torch.no_grad():
            latent = encoder(obs)
    except Exception:  # e.g. a recurrent encoder demanding a hidden state
        return None
    finally:
        encoder.train(was_training)
    if not isinstance(latent, torch.Tensor):
        return None
    return _per_neuron_score(latent)


def permute_latent_by_activation(fwd_net: Any, obs: Any) -> torch.Tensor | None:
    """Function-preservingly reorder the latent units by descending activation.

    Scores the encoder's latent output on *obs*, then relabels the latent units so
    the most active come first -- moving the encoder output layer's rows/bias *and*
    the head's first-layer input columns together (a consistent relabelling that
    leaves the function unchanged). The subsequent positional latent removal
    therefore drops the lowest-activation latent units, which with a
    :func:`dormant_removal_count`-sized removal are the τ-dormant ones.

    :param fwd_net: The ``EvolvableNetwork`` being mutated (encoder + ``head_net``).
    :param obs: A preprocessed observation batch accepted by *fwd_net*.
    :return: The scores used, or ``None`` when the latent could not be scored or
        relabelled (no resolvable encoder-output/head-input pair, an encoder that
        cannot be run on *obs*, or a width mismatch).
    :rtype: torch.Tensor | None
    """
    producer = encoder_output_layer(fwd_net)
    consumer = head_first_layer(fwd_net)
    if producer is None or consumer is None:
        return None
    latent = latent_scores(fwd_net, obs)
    if latent is None or latent.numel() != _out_dim(producer):
        return None
    perm = torch.sort(latent, descending=True, stable=True).indices
    perm = perm.to(_primary_weight(producer).device)
    _permute_out(producer, perm)
    _permute_in(consumer, perm, block=1)
    return latent


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
