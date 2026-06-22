"""Population-diversity diagnostics for AgileRL evolutionary HPO.

Measures how diverse a *population* of agents is, each cycle, along three
deliberately separate axes -- all normalised to ``[0, 1]`` against the **fixed
mutation search space** (never population-observed extrema, so the value is
comparable across generations and runs):

* **Hyperparameter diversity** -- for every mutable RL hyperparameter
  (``agent.registry.hp_config``) each agent's value is rescaled to ``[0, 1]`` via
  the hyperparameter's ``min``/``max`` (in *log space* for strictly-positive
  ranges spanning >= 1 order of magnitude, e.g. learning rates), and the per-HP
  diversity is ``2 * std`` of those normalised values (the ``2x`` rescales the
  ``[0, 0.5]`` maximum std of bounded data onto ``[0, 1]``). Averaged over HPs.
* **Architecture diversity** -- each evolvable network is *scalarized* to a
  descriptor vector capturing shape and capacity: **depth** (number of hidden
  layers), **total width** (sum of layer sizes), **mean width**, **max width**
  and **total parameter count**. Each descriptor is normalised by its own
  search-space bounds (read off the module: ``min/max_hidden_layers``,
  ``min/max_mlp_nodes`` for MLPs, ``min/max_channel_size`` for CNNs; parameter
  count by an analytic max-architecture reference), reduced to ``2 * std`` across
  the population, and averaged over descriptors, network roles and (for
  multi-agent algorithms) per-sub-policy slots.
* **Activation diversity** -- the normalised Shannon entropy
  ``-sum p_i log p_i / log K`` of the activation-function distribution across the
  population, where ``K`` is the number of *selectable* activations
  (``Mutations.activation_selection``), not the number currently present.
  Averaged over network roles.

Which networks are measured matches the dormant-neuron diagnostic: only each
registry group's ``eval_network`` (target/shared copies excluded), with
multi-agent ``ModuleDict`` networks unrolled into their per-sub-policy slots.
Every block is wrapped defensively so the diagnostic returns ``nan`` for an axis
rather than ever breaking training.
"""

from __future__ import annotations

import itertools
import math
from collections import Counter
from typing import Any

import numpy as np

from agilerl.modules import ModuleDict


def _normalized_spread(
    values: list[float], lo: float, hi: float, *, log_scale: bool = False
) -> float:
    """Population spread of *values* rescaled onto ``[0, 1]`` by fixed bounds.

    :param values: One value per agent.
    :param lo: Lower search-space bound for the quantity.
    :param hi: Upper search-space bound for the quantity.
    :param log_scale: Whether to rescale in log space (for ratio-scale quantities).
    :return: ``2 * std`` of the bound-normalised values, clipped to ``[0, 1]``.
    """
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return 0.0
    if log_scale:
        if lo <= 0.0 or hi <= 0.0 or np.any(arr <= 0.0):
            return 0.0
        arr, lo, hi = np.log(arr), math.log(lo), math.log(hi)
    if hi <= lo:
        return 0.0
    norm = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
    return float(min(1.0, 2.0 * float(np.std(norm))))


def _normalized_entropy(labels: list[str], k: int) -> float:
    """Normalised Shannon entropy of categorical *labels* over ``K`` options.

    :param labels: One activation label per agent.
    :param k: Number of selectable categories (the normalisation denominator).
    :return: ``-sum p_i log p_i / log K`` in ``[0, 1]``.
    """
    n = len(labels)
    if n == 0 or k <= 1:
        return 0.0
    probs = [count / n for count in Counter(labels).values()]
    entropy = -sum(p * math.log(p) for p in probs if p > 0.0)
    return float(min(1.0, entropy / math.log(k)))


def _unroll_network(network: Any) -> list[tuple[str | None, Any]]:
    """Return ``(sub_id, network)`` pairs, unrolling multi-agent ``ModuleDict``s.

    Single-agent networks yield one ``(None, network)`` pair; multi-agent
    ``ModuleDict`` networks (IPPO) yield one pair per sub-policy, keyed by the
    ``ModuleDict`` key so slots align across the population.
    """
    if isinstance(network, ModuleDict):
        return list(network.items())
    return [(None, network)]


def _leaf_arch_modules(network: Any) -> list[Any]:
    """Return the evolvable MLP/CNN leaf modules of *network*, in a stable order.

    Detected by duck-typing (``min_hidden_layers`` plus a ``hidden_size`` or
    ``channel_size`` layer-size list) rather than ``isinstance`` to avoid import
    cycles. The order is the deterministic torch sub-module registration order,
    identical across agents, so descriptor slots align across the population.
    """
    try:
        candidates = list(network.torch_modules())
    except Exception:  # diagnostics must never break training
        candidates = list(network.modules()) if hasattr(network, "modules") else []
    leaves = []
    for module in candidates:
        has_sizes = hasattr(module, "hidden_size") or hasattr(module, "channel_size")
        if hasattr(module, "min_hidden_layers") and has_sizes:
            leaves.append(module)
    return leaves


def _mlp_param_count(sizes: list[int]) -> float:
    """Approximate dense parameter count for a layer-width sequence (weights+biases)."""
    return float(sum(a * b + b for a, b in itertools.pairwise(sizes)))


def _max_param_count(module: Any) -> float:
    """Analytic dense parameter count of an MLP *module* at its maximal architecture.

    Used as the fixed upper bound for the total-parameter descriptor (MLPs only;
    see :func:`_arch_descriptors`).
    """
    max_layers = int(module.max_hidden_layers)
    max_nodes = int(module.max_mlp_nodes)
    seq = [int(module.num_inputs), *([max_nodes] * max_layers), int(module.num_outputs)]
    return _mlp_param_count(seq)


def _arch_descriptors(module: Any) -> dict[str, tuple[float, tuple[float, float]]]:
    """Map descriptor name -> ``(value, (lo, hi) bounds)`` for one leaf module."""
    if hasattr(module, "hidden_size"):
        sizes = list(module.hidden_size)
        min_nodes, max_nodes = int(module.min_mlp_nodes), int(module.max_mlp_nodes)
    else:
        sizes = list(module.channel_size)
        min_nodes, max_nodes = (
            int(module.min_channel_size),
            int(module.max_channel_size),
        )
    if not sizes:
        return {}

    min_layers, max_layers = (
        int(module.min_hidden_layers),
        int(module.max_hidden_layers),
    )
    descriptors: dict[str, tuple[float, tuple[float, float]]] = {
        "depth": (float(len(sizes)), (float(min_layers), float(max_layers))),
        "total_width": (
            float(sum(sizes)),
            (float(min_layers * min_nodes), float(max_layers * max_nodes)),
        ),
        "mean_width": (float(np.mean(sizes)), (float(min_nodes), float(max_nodes))),
        "max_width": (float(max(sizes)), (float(min_nodes), float(max_nodes))),
    }
    # Total-parameter descriptor only for MLPs: the analytic max-architecture
    # bound is exact for dense stacks, whereas a CNN leaf's parameters are
    # dominated by the (spatially-dependent) conv->latent projection, for which
    # a clean fixed bound is not available -- including it there would saturate
    # to 1.0 for every agent and add no diversity signal.
    if hasattr(module, "hidden_size"):
        try:
            n_params = float(sum(p.numel() for p in module.parameters()))
            max_params = _max_param_count(module)
            if max_params > 0.0:
                descriptors["total_params"] = (n_params, (0.0, max_params))
        except Exception:  # the param descriptor is best-effort
            pass
    return descriptors


def _hp_diversity(agents: list[Any]) -> float:
    """Mean per-hyperparameter normalised spread across the population."""
    config = getattr(agents[0].registry, "hp_config", None)
    if not config:
        return float("nan")
    spreads: list[float] = []
    for name in config.names():
        try:
            param = config[name]
            lo, hi = float(param.min), float(param.max)
            values = [float(getattr(agent, name)) for agent in agents]
        except (TypeError, ValueError):
            continue  # non-scalar hyperparameter (e.g. array-valued noise)
        log_scale = lo > 0.0 and hi > 0.0 and (hi / lo) >= 10.0
        spreads.append(_normalized_spread(values, lo, hi, log_scale=log_scale))
    return float(np.mean(spreads)) if spreads else float("nan")


def _arch_diversity(agents: list[Any]) -> float:
    """Mean per-descriptor normalised spread across the population."""
    per_agent: list[dict[tuple, float]] = []
    bounds: dict[tuple, tuple[float, float]] = {}
    for agent in agents:
        features: dict[tuple, float] = {}
        for g_idx, group in enumerate(agent.registry.groups):
            network = getattr(agent, group.eval_network)
            for sub_id, sub_net in _unroll_network(network):
                for leaf_idx, module in enumerate(_leaf_arch_modules(sub_net)):
                    for name, (value, bound) in _arch_descriptors(module).items():
                        key = (g_idx, sub_id, leaf_idx, name)
                        features[key] = value
                        bounds.setdefault(key, bound)
        per_agent.append(features)

    if not bounds:
        return float("nan")
    spreads = []
    for key, (lo, hi) in bounds.items():
        values = [feats[key] for feats in per_agent if key in feats]
        spreads.append(_normalized_spread(values, lo, hi))
    return float(np.mean(spreads)) if spreads else float("nan")


def _activation_diversity(
    agents: list[Any], activation_options: list[str] | None
) -> float:
    """Mean per-network normalised activation entropy across the population."""
    per_agent: list[dict[tuple, str]] = []
    keys: set[tuple] = set()
    for agent in agents:
        features: dict[tuple, str] = {}
        for g_idx, group in enumerate(agent.registry.groups):
            network = getattr(agent, group.eval_network)
            for sub_id, sub_net in _unroll_network(network):
                for leaf_idx, module in enumerate(_leaf_arch_modules(sub_net)):
                    activation = getattr(module, "activation", None)
                    if activation is None:
                        continue
                    key = (g_idx, sub_id, leaf_idx)
                    features[key] = str(activation)
                    keys.add(key)
        per_agent.append(features)

    if not keys:
        return float("nan")
    if activation_options:
        k = len(activation_options)
    else:  # fall back to the number of distinct activations observed
        observed = {label for feats in per_agent for label in feats.values()}
        k = max(len(observed), 1)
    spreads = [
        _normalized_entropy([feats[key] for feats in per_agent if key in feats], k)
        for key in keys
    ]
    return float(np.mean(spreads)) if spreads else float("nan")


def population_diversity(
    agents: list[Any], activation_options: list[str] | None = None
) -> dict[str, float]:
    """Normalised hyperparameter / architecture / activation diversity of a population.

    :param agents: The current population of AgileRL agents.
    :param activation_options: The selectable activation functions
        (``Mutations.activation_selection``) used as the entropy normaliser;
        defaults to ``["ReLU", "ELU", "GELU"]`` (the ``Mutations`` default) when
        ``None``.
    :return: ``{"hp": float, "arch": float, "activation": float}``, each in
        ``[0, 1]`` (or ``nan`` for an axis that could not be measured).
    """
    if not agents:
        return {"hp": float("nan"), "arch": float("nan"), "activation": float("nan")}

    options = activation_options or ["ReLU", "ELU", "GELU"]

    def _safe(fn, *args) -> float:
        try:
            return fn(*args)
        except Exception:  # diagnostics must never break training
            return float("nan")

    return {
        "hp": _safe(_hp_diversity, agents),
        "arch": _safe(_arch_diversity, agents),
        "activation": _safe(_activation_diversity, agents, options),
    }
