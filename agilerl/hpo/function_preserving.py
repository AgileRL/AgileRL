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
* **add_layer** -- the newly inserted layer is initialised to the identity
  (Net2DeeperNet). Exact when the activation is ReLU / Identity; a warning is
  emitted otherwise. Applies to the head, and to the encoder when it is an
  ``EvolvableMLP`` and ``arch_encoder_layer_mut`` opted its LAYER mutations back
  in (they are disabled by default; see
  :class:`agilerl.networks.base.EvolvableNetwork`).

Only *additions* are modified. ``remove_node`` / ``remove_channel`` /
``remove_latent_node`` deliberately keep AgileRL's original random-count positional
behaviour, so the ``func_preserving`` and ``original`` regimes differ purely in how
capacity is added.

All surgery operates purely on the weight tensors of the mutated module (both the
producing layer and the single consuming layer live inside the same
``EvolvableMLP`` / ``EvolvableCNN`` ``self.model``), so no cross-module plumbing is
needed. The convolutional last-layer -> flatten -> ``_linear_output`` boundary is
handled by treating each channel's flattened features as a contiguous ``H*W`` block
(``nn.Flatten`` is channel-outermost).

The functions here are pure tensor operations; they need no observation batch, no
forward pass and no plotting code, and are unit-tested standalone.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn

# Base mutation names (with any ``<agent_id>.`` / ``<submodule>.`` prefix stripped).
ADD_NODE_MUTATIONS = frozenset({"add_node", "add_channel"})
ADD_LAYER_MUTATIONS = frozenset({"add_layer"})
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


def _stack_signature(layers: list[nn.Module]) -> tuple[int, tuple[int, ...]]:
    """The ``(input dim, hidden widths)`` that a parallel stream must share.

    Output widths are deliberately excluded: a duelling head's value and advantage
    streams end in ``num_atoms`` and ``num_actions * num_atoms`` respectively, and
    are parallel precisely because everything *before* the output layer matches.
    """
    return (
        int(_primary_weight(layers[0]).shape[1]),
        tuple(_out_dim(layer) for layer in layers[:-1]),
    )


def _weight_stacks(submodule: nn.Module) -> list[list[nn.Module]]:
    """Return every parallel weight stack of *submodule*, the primary one first.

    Most modules own a single stack, ``self.model``. A *branched* head owns more:
    :class:`~agilerl.networks.custom_modules.DuelingDistributionalMLP` keeps its
    value stream in the inherited ``model`` and its advantage stream in a sibling
    ``nn.Sequential`` (``advantage_net``), and its ``recreate_network`` grows both
    from the same ``hidden_size``. An addition therefore lands on every stream and
    the fixup must too -- applying it to ``model`` alone leaves the network neither
    preserved nor equal to the stock operator's output.

    Siblings are matched structurally rather than by name: a candidate qualifies
    only if its :func:`_stack_signature` matches the primary stack's, so an
    unrelated ``nn.Sequential`` is skipped untouched rather than mis-initialised.

    :param submodule: The evolvable module being mutated.
    :return: One list of conv/(noisy) linear layers per stack, in forward order.
    """
    inner = _inner_module(submodule)
    model = getattr(inner, "model", None)
    if model is None:
        return []
    primary = [m for m in model if _is_weight_layer(m)]
    if not primary:
        return []

    stacks = [primary]
    signature = _stack_signature(primary)
    # NOTE: ``EvolvableModule`` overrides ``.modules()`` to return group names, so
    # iterate ``named_children()`` to reach the real sibling modules.
    for _name, child in inner.named_children():
        if child is model or not isinstance(child, nn.Sequential):
            continue
        candidate = [m for m in child if _is_weight_layer(m)]
        if candidate and _stack_signature(candidate) == signature:
            stacks.append(candidate)
    return stacks


def _ordered_weight_layers(submodule: nn.Module) -> list[nn.Module]:
    """Return the conv/linear layers of ``submodule.model`` in forward order.

    Norm, activation and flatten layers are skipped, so the returned list is
    ``[hidden_0, ..., hidden_{N-1}, output]`` for both MLP and CNN modules.
    ``EvolvableDistribution`` heads are unwrapped to their inner MLP first. This is
    the *primary* stack only -- see :func:`_weight_stacks` for branched heads.
    """
    stacks = _weight_stacks(submodule)
    return stacks[0] if stacks else []


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
    :return: The number of units whose outgoing weights were (re)initialised in
        each stack (every parallel stream of a branched head is fixed up).
    """
    if old_width is None:
        return 0

    num_added = 0
    for layers in _weight_stacks(submodule):
        if not 0 <= hidden_layer < len(layers) - 1:
            continue

        producer = layers[hidden_layer]
        consumer = layers[hidden_layer + 1]
        new_width = _out_dim(producer)
        added = new_width - old_width
        if added <= 0:
            continue

        conv_to_linear = _is_conv(producer) and _is_linear(consumer)
        block = _spatial_size(submodule) if conv_to_linear else 1
        col_slice = slice(old_width * block, new_width * block)
        _init_new_input_columns(consumer, col_slice, noise_scale)
        num_added = added

    return num_added


def identity_new_layer(submodule: nn.Module) -> bool:
    """Set a freshly inserted head layer to the identity (Net2DeeperNet).

    ``add_layer`` appends a hidden layer of the *same* width as the one below, so
    the new layer is the penultimate weight layer and is square; the output layer
    is preserved exactly by ``preserve_parameters``. Overwriting the new layer with
    an identity weight (and zero bias) makes the deepening function-preserving when
    the activation is ReLU / Identity.

    :param submodule: The mutated head ``EvolvableMLP``.
    :return: ``True`` if an identity layer was written in at least one stack.
    """
    wrote = False
    for layers in _weight_stacks(submodule):
        if len(layers) < 2:
            continue
        new_layer = layers[-2]
        weight = _primary_weight(new_layer)
        if weight.dim() != 2 or weight.shape[0] != weight.shape[1]:
            continue

        with torch.no_grad():
            if getattr(new_layer, "weight", None) is not None:
                new_layer.weight.data.zero_()
                new_layer.weight.data.fill_diagonal_(1.0)
            if getattr(new_layer, "weight_mu", None) is not None:
                new_layer.weight_mu.data.zero_()
                new_layer.weight_mu.data.fill_diagonal_(1.0)
            # A zeroed sigma makes the *training*-mode forward the identity too;
            # ``weight_epsilon`` needs no reset since it is multiplied by it.
            if getattr(new_layer, "weight_sigma", None) is not None:
                new_layer.weight_sigma.data.zero_()
            for b in _bias_tensors(new_layer):
                b.data.zero_()
        wrote = True
    return wrote


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


def head_first_layers(fwd_net: Any) -> list[nn.Module]:
    """The first weight layer of each of the head's parallel streams.

    A branched head consumes the latent once per stream, so a latent widening has
    to be compensated in every one of them.

    :param fwd_net: The ``EvolvableNetwork`` (encoder + ``head_net``).
    :return: One layer per stack, primary stream first; empty if there is no head.
    """
    head = getattr(fwd_net, "head_net", None)
    if head is None:
        return []
    return [layers[0] for layers in _weight_stacks(head) if layers]


def head_first_layer(fwd_net: Any) -> nn.Module | None:
    """The head network's first weight layer (unwraps ``EvolvableDistribution``).

    The *primary* stream only -- see :func:`head_first_layers`.
    """
    layers = head_first_layers(fwd_net)
    return layers[0] if layers else None


def _shift_trailing_head_inputs(
    consumer: nn.Module, old_latent: int, new_latent: int, extra: int
) -> None:
    """Move a head's non-latent trailing input columns to their new offset.

    ``preserve_parameters`` copies the overlapping top-left block, so after a latent
    widening the trailing block still sits at the *old* offset while the head now
    reads it from the new one. Sliding it across restores the correspondence; the
    source is cloned first because the two ranges overlap whenever
    ``new_latent < old_latent + extra``.
    """
    with torch.no_grad():
        for w in _weight_tensors(consumer):
            w.data[:, new_latent : new_latent + extra] = w.data[
                :, old_latent : old_latent + extra
            ].clone()


def init_new_latent_outgoing(
    fwd_net: Any, old_latent: int | None, noise_scale: float = 0.0
) -> int:
    """Zero/noise the head's new input columns after a latent-dim widening.

    New latent units occupy columns ``[old_latent, new_latent)`` of the head's first
    weight layer (latent is flat, so ``block == 1``); filling them zero (or with
    small symmetry-breaking noise) makes ``add_latent_node`` /
    ``add_latent_channel`` function-preserving across the encoder->head boundary.

    The new width is read from the network's own ``latent_dim``, **not** from the
    consumer's tensor width: a head may take more than the latent.
    :class:`~agilerl.networks.q_networks.ContinuousQNetwork` (the DDPG/TD3/MADDPG/
    MATD3 critic) builds its head with ``num_inputs=latent_dim + num_actions`` and
    forwards ``torch.cat([latent, actions])``, so sizing off the tensor would treat
    that action block as new latent units and overwrite it -- zeroing ``dQ/da`` and
    leaving the deterministic actor with no policy gradient. Those trailing inputs
    are instead slid to the offset the widened head reads them from, which keeps the
    addition function-preserving for such critics too.

    :param fwd_net: The mutated ``EvolvableNetwork`` (encoder + ``head_net``).
    :param old_latent: The latent dim *before* the mutation (``None`` or ``>=`` the
        new latent dim means nothing was added -- a no-op, e.g. capped at
        ``max_latent_dim``).
    :param noise_scale: Symmetry-breaking noise factor (``arch_fp_noise``).
    :return: The number of new latent units whose fan-out was (re)initialised.
    """
    if old_latent is None:
        return 0

    latent_dim = getattr(fwd_net, "latent_dim", None)
    if latent_dim is None:
        return 0

    new_latent = int(latent_dim)
    num_added = new_latent - old_latent
    if num_added <= 0:
        return 0

    for consumer in head_first_layers(fwd_net):
        # Anything past the latent is a non-latent head input (e.g. the critic's
        # action block). A head narrower than the latent consumes something this
        # surgery cannot describe, so leave it to the original operator.
        extra = int(_primary_weight(consumer).shape[1]) - new_latent
        if extra < 0:
            continue
        if extra > 0:
            _shift_trailing_head_inputs(consumer, old_latent, new_latent, extra)
        _init_new_input_columns(consumer, slice(old_latent, new_latent), noise_scale)

    return num_added


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
