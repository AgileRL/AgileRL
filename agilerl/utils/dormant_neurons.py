"""Gradient-based dormant-neuron diagnostics for AgileRL agents (GraMa).

Implements the gradient-magnitude neural-activity metric (**GraMa**) of equation 2
of *"Measure gradients, not activations! Enhancing neuronal activity in deep
reinforcement learning"* -- the gradient analogue of the τ-dormant neuron metric
of Sokar et al. (2023). For a batch of inputs ``D`` a neuron ``i`` in layer ``l``
has score

    ``G_i = E_x|∇_{z_i} L(x)| / ( (1 / H_l) * Σ_k E_x|∇_{z_k} L(x)| )``

where ``∇_{z_i}L`` is the gradient of the training loss w.r.t. neuron ``i``'s
*pre-activation*, and the neuron is **τ-dormant** when ``G_i <= τ``. ``G_i``
measures a neuron's *learning capacity* (how much it can still be updated by a
gradient step) rather than its expressivity.

The paper writes the numerator as ``∇_{h_i}L`` over the post-activation output
``h_i = act(z_i)``, but its released implementation (``GradientReDo`` in
``utils/ReDo.py``) scores neurons by ``|∂L/∂W|`` row-averaged over the producing
layer's incoming weights -- and ``∂L/∂W_ij = Σ_b ∂L/∂z_bi · x_bj`` is driven by
the *pre-activation* gradient, not the post-activation one. We follow the
released method rather than the notation, and take the gradient one step earlier,
at the activation's input. The difference is the derivative factor
(``∇_{z_i}L = ∇_{h_i}L · act'(z_i)``), and it is the whole metric: ``act'`` is
the only activation-function-dependent term, so dropping it makes a permanently
inactive ReLU unit -- or a saturated Tanh unit -- indistinguishable from a live
one, since ``∇_{h_i}L`` is just the downstream weight projection and stays
healthy regardless. Reading the input rather than the weight gradient keeps the
absolute value *inside* the batch expectation, as eq. 2 specifies, and scores
each neuron without the input-magnitude confound ``x_bj`` introduces.

Dense units count as one neuron each (the expectation is taken over the batch
dimension); convolutional feature maps count as one neuron each (the expectation
is taken over the batch *and* spatial dimensions). The reported fraction is
``dormant_count / total_count`` aggregated across every measured layer of every
measured network of the agent.

The per-neuron gradient is captured **cheaply, during the real training backward
pass**: an activation sub-module's ``grad_input`` (the tuple a backward hook
receives) is exactly ``∇_{z_i}L``, so no extra forward/backward pass and no
observation batch are needed. :class:`GraMaCapture` wraps an agent's per-cycle
training block, registers the hooks, keeps the per-neuron mean ``|∇_{z_i}L|`` of
that cycle's **last** training minibatch, and stores the result on the agent as
``_grama_scores`` -- a list aligned to :func:`_eval_networks` order, each entry a
list aligned to :func:`_target_activations` order (``None`` for a layer whose
gradient was never seen). :func:`dormant_neuron_fraction` (the diagnostic) and the
ReBorn parameter mutation both read this stored snapshot.

Which networks/layers are measured (matching the thesis design decisions):

* **Networks** -- all *evaluation* networks of the agent, with target/shared
  networks excluded. These are read off the agent's mutation registry
  (``agent.registry.groups``): each group's ``eval_network`` is measured and its
  ``shared_networks`` (e.g. DQN's frozen ``actor_target``) are skipped so the
  frozen copy does not double-count. For PPO this is the actor and critic, for
  DQN the online Q-network, for IPPO every per-agent actor and critic.
* **Layers** -- for each network the encoder's output activation *is* counted
  (its latent output is a hidden representation), while the head network's output
  activations are *not* counted (those units have fixed semantics such as action
  logits or a state value). Both halves of that rule are resolved structurally,
  never by sub-module name, since the evolvable modules disagree on naming and a
  name marker silently skips whole encoders (``EvolvableMultiInput``'s activation
  is called just ``output``). An activation is identified by **type** -- the shared
  :data:`~agilerl.utils.evolvable_networks.ACTIVATION_FUNCTIONS` registry -- and
  counts as an *output* activation when no projecting layer follows it within its
  own stream (:func:`_is_output_activation`). Per-stream rather than per-network,
  because a duelling Q-network's ``head_net`` terminates in two parallel streams.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np
from typing_extensions import Self

from agilerl.modules import ModuleDict
from agilerl.modules.custom_components import NewGELU
from agilerl.utils.evolvable_networks import ACTIVATION_FUNCTIONS

if TYPE_CHECKING:
    import torch
    from torch import nn

logger = logging.getLogger(__name__)

# Activation sub-modules are recognised by *type*, never by name: the evolvable
# encoders disagree on naming (``EvolvableMLP`` emits ``*_activation_output``,
# ``EvolvableCNN``/``EvolvableLSTM``/``EvolvableResNet`` emit ``*_output_activation``
# and ``EvolvableMultiInput`` emits a bare ``output``), so a name marker silently
# skips whole encoders. ``NewGELU`` is added explicitly because it substitutes for
# ``GELU`` in the registry only when ``new_gelu`` is set.
_ACTIVATION_TYPES: tuple[type[nn.Module], ...] = (
    *dict.fromkeys(ACTIVATION_FUNCTIONS.values()),
    NewGELU,
)

# Attribute under which the captured per-neuron gradient snapshot is stored on an
# agent by :class:`GraMaCapture`.
GRAMA_SCORES_ATTR = "_grama_scores"


def _remaps_neurons(module: nn.Module) -> bool:
    """Whether *module* maps its input onto a fresh set of neurons.

    Identified by the presence of an output-width attribute, which every
    projecting layer carries (``out_features`` on ``Linear``/``NoisyLinear``,
    ``out_channels`` on the convolutions, ``hidden_size`` on the recurrent
    modules). Deliberately excludes the elementwise layers -- normalisations,
    dropout, flattening -- which preserve neuron identity, and cannot key on a
    ``weight`` attribute instead: ``NoisyLinear`` stores ``weight_mu``/
    ``weight_sigma`` and would be missed, while ``LayerNorm`` has ``weight`` and
    would be matched wrongly.

    :param module: The sub-module to classify.
    :return: ``True`` if the module projects onto new neurons.
    """
    return any(
        hasattr(module, attr)
        for attr in ("out_features", "out_channels", "hidden_size")
    )


def _is_output_activation(name: str, ordered: list[tuple[str, nn.Module]]) -> bool:
    """Whether the activation at *name* terminates its stream.

    An activation is a *hidden* one when its output is consumed by a further
    projecting layer, and an *output* one when nothing follows it. Deciding this
    structurally -- rather than positionally -- is what keeps parallel streams
    correct: a duelling Q-network's ``head_net`` holds two independent
    sub-networks, so "the last activation of ``head_net``" would leave the value
    stream's output activation misclassified as hidden.

    :param name: Qualified name of the activation within its root module.
    :param ordered: ``named_modules()`` of that root, in registration order.
    :return: ``True`` if no projecting layer follows it in the same stream.
    """
    parent = name.rpartition(".")[0]
    prefix = f"{parent}." if parent else ""
    seen = False
    for other_name, other in ordered:
        if other_name == name:
            seen = True
            continue
        # Restrict the lookahead to the activation's own parent container so a
        # sibling stream registered later cannot mask the end of this one.
        if seen and other_name.startswith(prefix) and _remaps_neurons(other):
            return False
    return True


def _activation_modules(root: nn.Module, *, include_output: bool) -> list[nn.Module]:
    """Return the post-activation sub-modules of *root* to hook, in forward order.

    Sub-modules are recognised by type (:data:`_ACTIVATION_TYPES`), so this is
    independent of each encoder's naming convention. ``named_modules`` yields
    registration order, which is forward order for the ordered-dict-built
    ``nn.Sequential`` networks the evolvable modules produce.

    :param root: The module to search (an encoder or a head network).
    :param include_output: Whether to also include the stream-terminating output
        activations (see :func:`_is_output_activation`).
    :return: The activation sub-modules whose gradients should be measured.
    """
    ordered = list(root.named_modules())
    return [
        m
        for name, m in ordered
        if isinstance(m, _ACTIVATION_TYPES)
        and (include_output or not _is_output_activation(name, ordered))
    ]


def _per_neuron_grad(grad_input: Any) -> torch.Tensor | None:
    """Reduce an activation module's ``grad_input`` to one ``|∇_{z_i}L|`` per neuron.

    ``grad_input`` is the tuple a full backward hook receives; its first element
    is the gradient of the loss w.r.t. the module's *input*, i.e. the
    pre-activation gradient ``∇_{z_i}L`` (see the module docstring for why the
    metric is defined there and not on the activation's output). Dense gradients
    have shape ``(batch, H)`` and are averaged over the batch dimension;
    convolutional gradients have shape ``(batch, C, *spatial)`` and are averaged
    over the batch *and* spatial dimensions, so each feature map counts as a
    single neuron (dimension 1). The absolute value is taken *before* the batch
    reduction, as eq. 2's ``E_x|∇L|`` requires.

    :param grad_input: The gradient tuple from ``register_full_backward_hook``.
    :return: A 1-D tensor of mean absolute gradients, one entry per neuron, or
        ``None`` if no gradient flowed through the module.
    """
    if isinstance(grad_input, (tuple, list)):
        grad = grad_input[0] if len(grad_input) > 0 else None
    else:
        grad = grad_input
    if grad is None:
        return None
    g = grad.detach().abs()
    if g.dim() <= 1:
        return g
    reduce_dims = [d for d in range(g.dim()) if d != 1]
    return g.mean(dim=reduce_dims)


def _count_dormant(per_neuron: torch.Tensor, tau: float) -> tuple[int, int]:
    """Count τ-dormant neurons in one layer.

    Neurons whose gradient is not finite are **excluded** from both counts, not
    coerced to a magnitude. A diverged agent reaches here with NaN/inf gradients
    yet a finite fitness, so it survives the tournament's fitness guard and can be
    the agent selected for measurement, and either coercion misreports it: NaN
    never satisfies ``<= tau``, so it would silently mask genuinely dormant
    neurons in the same layer, while one inf drives the layer mean to infinity and
    crushes every other neuron's normalised score to zero. Mapping both to zero
    instead (as the ReBorn operator does, where recycling a broken neuron is the
    right *action*) would label an exploding neuron dormant -- the opposite of
    what it is -- and make a diverged layer read identically to a genuinely dead
    one. A layer with nothing finite left contributes ``(0, 0)``, which propagates
    to ``nan`` in :func:`dormant_neuron_fraction` rather than a fabricated value.

    :param per_neuron: Mean absolute gradient of each neuron in the layer.
    :param tau: Dormancy threshold.
    :return: ``(dormant_count, total_count)`` over the layer's finite neurons.
    """
    # Tensor method, not ``torch.isfinite``: this module keeps ``torch`` behind
    # ``TYPE_CHECKING`` so it stays import-light.
    per_neuron = per_neuron[per_neuron.isfinite()]
    total = per_neuron.numel()
    if total == 0:
        return 0, 0
    layer_mean = per_neuron.mean()
    if float(layer_mean) <= 0.0:
        # Every neuron has exactly zero gradient -> the whole layer is dormant.
        return total, total
    scores = per_neuron / layer_mean
    dormant = int((scores <= tau).sum().item())
    return dormant, total


def _target_activations(network: nn.Module) -> list[nn.Module]:
    """Return the ordered post-activation sub-modules measured for a network.

    The encoder's output activation *is* included (its latent is a hidden
    representation), while the head network's final output activation is *not*
    (those units have fixed semantics such as action logits or a state value).

    :param network: An ``EvolvableNetwork`` (encoder + ``head_net``).
    :return: The activation sub-modules to hook, in forward order.
    """
    targets: list[nn.Module] = []
    if hasattr(network, "encoder"):
        targets += _activation_modules(network.encoder, include_output=True)
    if hasattr(network, "head_net"):
        targets += _activation_modules(network.head_net, include_output=False)
    return targets


def _eval_networks(agent: Any) -> list[tuple[str | None, nn.Module]]:
    """Return the agent's evaluation networks as ``(network_id, network)`` pairs.

    Target / shared networks are excluded (only ``group.eval_network`` of each
    registry group is returned). For multi-agent algorithms each per-network
    module is yielded with its network/group id (the ``ModuleDict`` key); for
    single-agent algorithms the id is ``None``.

    :param agent: An AgileRL algorithm instance.
    :return: List of ``(network_id, network)`` pairs to measure.
    """
    pairs: list[tuple[str | None, nn.Module]] = []
    for group in agent.registry.groups:
        eval_net = getattr(agent, group.eval_network)
        if isinstance(eval_net, ModuleDict):
            for network_id, sub_net in eval_net.items():
                pairs.append((network_id, sub_net))
        else:
            pairs.append((None, eval_net))
    return pairs


class GraMaCapture:
    """Capture per-neuron pre-activation gradient magnitudes during training.

    Registers a full backward hook on every measured activation sub-module of
    every evaluation network of *agent* (see :func:`_eval_networks` /
    :func:`_target_activations`). Each hook reduces the module's ``grad_input`` --
    exactly ``∇_{z_i}L`` -- to one mean-absolute value per neuron and keeps **only
    the most recent** minibatch's value, overwriting any earlier one. On exit those
    last-minibatch values are written to ``agent._grama_scores`` (a list per
    evaluation network -- in :func:`_eval_networks` order -- each a list per
    measured layer aligned to :func:`_target_activations` order, with ``None`` for a
    layer whose gradient never flowed) and every hook is removed, even if the
    wrapped training block raises.

    Keeping one minibatch rather than a running mean over the cycle is deliberate,
    on two counts. Eq. 2's ``E_{x∈D}`` is an expectation at *fixed* parameters,
    while a cycle mean spans every optimizer step of that cycle (for the benchmark
    PPO config, five ``learn()`` calls and ~1600 updates) and so averages gradients
    taken w.r.t. parameter vectors that no longer exist; the released reference
    implementation likewise thresholds whatever single minibatch's gradient is
    currently populated. And the consumer decides surgery on the network *as it
    stands at the end of the cycle* -- ReBorn resets neurons in exactly that
    network -- so it must be scored in that state, not in a stale average. The
    trade-off is single-minibatch noise, bounded by the training batch size (1024
    under the benchmark configs).

    Because the hooks ride on the *real* training backward pass, no extra forward
    or backward pass is performed. Wrap an agent's whole per-cycle training block::

        with GraMaCapture(agent):
            ...  # agent.learn() calls
        scores = agent._grama_scores

    A measured activation whose gradient never flows during the training loss (its
    hook never fires) is stored as ``None`` and skipped downstream -- a defensive
    fallback for layers that are not part of what is being *trained*. Under the
    thesis benchmark configs (PPO with ``share_encoders=False``, DQN, IPPO) every
    measured layer is in the training-loss graph, so this rarely triggers; it would
    fire for genuinely disconnected sub-networks, e.g. the 0-parameter placeholder
    critic encoder PPO builds under its ``share_encoders=True`` *default* (which the
    configs override). (A layer that *is* in the graph but receives an all-zero
    gradient still fires, and is correctly counted as fully dormant.)
    """

    def __init__(self, agent: Any) -> None:
        self.agent = agent
        self._handles: list[Any] = []
        # Parallel to _eval_networks(agent): for each network a list (aligned to
        # _target_activations order) holding that layer's most recent per-neuron
        # gradient, or ``None`` for a layer that has not fired yet.
        self._latest: list[list[Any]] = []

    def __enter__(self) -> Self:
        # Registration is best-effort: an agent that does not expose the diagnostic
        # surface (e.g. a mock in tests) must never break training. On any failure
        # we drop any partial hooks and capture nothing.
        try:
            for net_idx, (_network_id, network) in enumerate(
                _eval_networks(self.agent)
            ):
                targets = _target_activations(network)
                self._latest.append([None] * len(targets))
                for mod_idx, module in enumerate(targets):
                    handle = module.register_full_backward_hook(
                        self._make_hook(net_idx, mod_idx)
                    )
                    self._handles.append(handle)
        except Exception as exc:  # capture must never break training
            logger.warning("GraMa capture could not register hooks: %s", exc)
            self._remove_handles()
            self._latest = []
        return self

    def _make_hook(self, net_idx: int, mod_idx: int):
        latest = self._latest

        def _hook(_module: nn.Module, grad_input: Any, _grad_output: Any) -> None:
            # A raising backward hook would abort the training backward pass, so
            # swallow anything unexpected here.
            try:
                per_neuron = _per_neuron_grad(grad_input)
                if per_neuron is None:
                    return
                # Overwrite: only the last minibatch survives (see class docstring).
                latest[net_idx][mod_idx] = per_neuron
            except Exception:  # never break the training backward pass
                return

        return _hook

    def _remove_handles(self) -> None:
        try:
            for handle in self._handles:
                handle.remove()
        except Exception:  # teardown must never break training
            pass
        self._handles = []

    def __exit__(self, *_exc: object) -> bool:
        try:
            scores: list[list[torch.Tensor | None]] = [
                list(net_latest) for net_latest in self._latest
            ]
            setattr(self.agent, GRAMA_SCORES_ATTR, scores)
        except Exception as exc:  # capture must never break training
            logger.warning("GraMa capture could not store scores: %s", exc)
            setattr(self.agent, GRAMA_SCORES_ATTR, None)
        finally:
            self._remove_handles()
        return False


def capture_per_neuron_scores(
    network: nn.Module, per_neuron_list: list[torch.Tensor | None] | None
) -> list[tuple[nn.Module, torch.Tensor]]:
    """Zip a stored per-neuron gradient snapshot with a network's activations.

    *per_neuron_list* is this network's entry of ``agent._grama_scores`` (see
    :class:`GraMaCapture`): the per-neuron mean ``|∇_{h_i}L|`` captured during
    training, aligned to :func:`_target_activations` order. Returns
    ``(activation_module, per_neuron)`` pairs -- skipping layers whose gradient was
    never captured -- preserving the contract the ReBorn parameter mutation
    consumes. Returns ``[]`` if no scores are available or their count no longer
    matches the network's measured layers (e.g. after an architecture mutation), so
    callers degrade gracefully instead of misaligning.

    :param network: An ``EvolvableNetwork`` (encoder + ``head_net``).
    :param per_neuron_list: The captured per-layer gradient scores for *network*.
    :return: ``(module, per_neuron_tensor)`` pairs in forward order.
    """
    targets = _target_activations(network)
    if not per_neuron_list or len(per_neuron_list) != len(targets):
        return []
    return [
        (module, per_neuron)
        for module, per_neuron in zip(targets, per_neuron_list, strict=False)
        if per_neuron is not None
    ]


def _measure_network(
    network: nn.Module,
    per_neuron_list: list[torch.Tensor | None] | None,
    tau: float,
) -> tuple[int, int]:
    """Compute ``(dormant, total)`` neuron counts for a single network.

    :param network: An ``EvolvableNetwork`` (encoder + ``head_net``).
    :param per_neuron_list: The captured per-layer gradient scores for *network*.
    :param tau: Dormancy threshold.
    :return: ``(dormant_count, total_count)`` summed over the measured layers.
    """
    dormant = total = 0
    for _module, per_neuron in capture_per_neuron_scores(network, per_neuron_list):
        d, t = _count_dormant(per_neuron, tau)
        dormant += d
        total += t
    return dormant, total


def dormant_neuron_fraction(agent: Any, tau: float = 0.1) -> float:
    """Fraction of τ-dormant (GraMa) neurons across all of *agent*'s eval networks.

    Reads the per-neuron pre-activation gradient snapshot captured during the
    agent's last training block (``agent._grama_scores``, populated by
    :class:`GraMaCapture`) and aggregates the τ-dormant count over every measured
    layer of every evaluation network (targets excluded); see the module docstring
    for the exact networks and layers measured.

    :param agent: An AgileRL algorithm instance (PPO, DQN, IPPO, ...).
    :param tau: Dormancy threshold (``G_i <= tau`` is dormant). Defaults to 0.1
        (the value used throughout the benchmarking harness); ``tau=0.0`` counts
        only exactly-inactive neurons, which is degenerate for e.g. Tanh networks
        whose gradients are almost never exactly zero.
    :return: ``dormant_count / total_count`` in ``[0, 1]``, or ``nan`` if no
        gradient snapshot is available or none could be measured (also ``nan`` --
        with a logged warning -- if the measurement fails, so the diagnostic never
        breaks training).
    """
    try:
        scores = getattr(agent, GRAMA_SCORES_ATTR, None)
        if not scores:
            return float("nan")

        dormant = total = 0
        for idx, (_network_id, network) in enumerate(_eval_networks(agent)):
            per_neuron_list = scores[idx] if idx < len(scores) else None
            d, t = _measure_network(network, per_neuron_list, tau)
            dormant += d
            total += t
    except Exception as exc:  # the diagnostic must never break training
        logger.warning("Dormant-neuron fraction could not be measured: %s", exc)
        return float("nan")

    if total == 0:
        return float("nan")
    return dormant / total


def best_agent_dormant_fraction(agents: list[Any], tau: float = 0.1) -> float:
    """Dormant-neuron fraction of the best agent (highest last fitness).

    The best agent is the one with the greatest final fitness, mirroring the
    elite selection used elsewhere in the codebase. Multi-agent dict fitnesses
    are ranked by their summed value.

    :param agents: The population of agents.
    :param tau: Dormancy threshold.
    :return: The best agent's dormant-neuron fraction, or ``nan`` if no agent has
        a recorded fitness.
    """

    def _last_fitness(agent: Any) -> float:
        fitness = getattr(agent, "fitness", None)
        if not fitness:
            return float("-inf")
        last = fitness[-1]
        if isinstance(last, dict):
            return float(np.sum(list(last.values())))
        if isinstance(last, (list, tuple, np.ndarray)):
            return float(np.sum(last))
        return float(last)

    if not agents:
        return float("nan")
    best = max(agents, key=_last_fitness)
    return dormant_neuron_fraction(best, tau)
