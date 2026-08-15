"""Dormant-neuron diagnostics for AgileRL agents.

Implements the τ-dormant neuron metric of Sokar et al. (2023), "The Dormant
Neuron Phenomenon in Deep Reinforcement Learning" (Definition 3.1): for a batch
of inputs ``D``, a neuron ``i`` in layer ``l`` has score

    ``s_i = E_x|h_i(x)| / ( (1 / H_l) * Σ_k E_x|h_k(x)| )``

evaluated on the *post-activation* outputs, and is **τ-dormant** when
``s_i <= τ``. Dense units count as one neuron each (the expectation is taken
over the batch dimension); convolutional feature maps count as one neuron each
(the expectation is taken over the batch *and* spatial dimensions). The reported
fraction is ``dormant_count / total_count`` aggregated across every measured
layer of every measured network of the agent.

Which networks/layers are measured (matching the thesis design decisions):

* **Networks** -- all *evaluation* networks of the agent, with target/shared
  networks excluded. These are read off the agent's mutation registry
  (``agent.registry.groups``): each group's ``eval_network`` is measured and its
  ``shared_networks`` (e.g. DQN's frozen ``actor_target``) are skipped so the
  frozen copy does not double-count. For PPO this is the actor and critic, for
  DQN the online Q-network, for IPPO every per-agent actor and critic.
* **Layers** -- for each network the encoder's output activation *is* counted
  (its latent output is a hidden representation), while the head network's final
  output activation is *not* counted (those units have fixed semantics such as
  action logits or a state value). Concretely we hook every ``*_activation_*``
  sub-module of ``network.encoder`` and every ``*_activation_*`` sub-module of
  ``network.head_net`` except the one named ``*_activation_output``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from agilerl.modules import ModuleDict

if TYPE_CHECKING:
    from torch import nn

logger = logging.getLogger(__name__)

# Sub-module name marker for an activation layer (see
# ``agilerl.utils.evolvable_networks.create_mlp`` / ``create_cnn``).
_ACTIVATION_MARKER = "_activation_"
_OUTPUT_ACTIVATION_SUFFIX = "_activation_output"


def _activation_modules(root: nn.Module, *, include_output: bool) -> list[nn.Module]:
    """Return the post-activation sub-modules of *root* to hook.

    :param root: The module to search (an encoder or a head network).
    :param include_output: Whether to also include the final
        ``*_activation_output`` activation.
    :return: The activation sub-modules whose outputs should be measured.
    """
    modules: list[nn.Module] = []
    for name, module in root.named_modules():
        if _ACTIVATION_MARKER not in name:
            continue
        if not include_output and name.endswith(_OUTPUT_ACTIVATION_SUFFIX):
            continue
        modules.append(module)
    return modules


def _per_neuron_score(activation: torch.Tensor) -> torch.Tensor:
    """Reduce a post-activation tensor to one ``E_x|h_i|`` value per neuron.

    Dense outputs have shape ``(batch, H)`` and are averaged over the batch
    dimension. Convolutional outputs have shape ``(batch, C, *spatial)`` and are
    averaged over the batch *and* spatial dimensions, so each feature map counts
    as a single neuron (dimension 1).

    :param activation: The post-activation output captured by a forward hook.
    :return: A 1-D tensor of mean absolute activations, one entry per neuron.
    """
    act = activation.detach().abs()
    if act.dim() <= 1:
        return act
    reduce_dims = [d for d in range(act.dim()) if d != 1]
    return act.mean(dim=reduce_dims)


def normalised_scores(per_neuron: torch.Tensor) -> torch.Tensor | None:
    """Normalise per-neuron activations by their layer mean (Definition 3.1).

    Non-finite entries are dropped before the mean is taken -- a diverged agent can
    carry NaN/inf activations yet a finite fitness that slips past the tournament's
    guard, and a single ``inf`` would drive the layer mean to infinity and crush
    every other unit's score to zero.

    Shared with the function-preserving architecture mutation, whose removals are
    sized by the same τ-dormancy test the diagnostic below counts with.

    :param per_neuron: Mean absolute activation of each neuron in the layer.
    :type per_neuron: torch.Tensor
    :return: ``s_i`` for each *finite* neuron, or ``None`` when the layer has no
        finite entries. A layer whose finite mean is non-positive (every unit is
        exactly zero) yields an all-zero score tensor, i.e. fully dormant.
    :rtype: torch.Tensor | None
    """
    finite = per_neuron[per_neuron.isfinite()]
    if finite.numel() == 0:
        return None
    layer_mean = finite.mean()
    if float(layer_mean) <= 0.0:
        return torch.zeros_like(finite)
    return finite / layer_mean


def _count_dormant(per_neuron: torch.Tensor, tau: float) -> tuple[int, int]:
    """Count τ-dormant neurons in one layer per Definition 3.1.

    :param per_neuron: Mean absolute activation of each neuron in the layer.
    :param tau: Dormancy threshold.
    :return: ``(dormant_count, total_count)`` for the layer. Non-finite neurons are
        excluded from *both* counts rather than coerced, so a diverged agent reads
        as ``nan`` instead of a fabricated fraction.
    """
    scores = normalised_scores(per_neuron)
    if scores is None:
        return 0, 0
    return int((scores <= tau).sum().item()), scores.numel()


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


def capture_per_neuron_scores(
    network: nn.Module, obs: Any
) -> list[tuple[nn.Module, torch.Tensor]]:
    """Run one forward pass and capture ``E_x|h_i|`` per neuron per measured layer.

    Registers forward hooks on exactly the activation sub-modules the dormant
    diagnostic measures (see :func:`_target_activations`), runs a single
    ``no_grad`` evaluation forward pass, and returns the per-neuron mean absolute
    activation of each measured layer. Sharing this with the function-preserving
    architecture mutation (which ranks neurons by activation before a removal)
    guarantees both operate on the identical set of neurons.

    :param network: An ``EvolvableNetwork`` (encoder + ``head_net``).
    :param obs: A preprocessed observation batch accepted by ``network``.
    :return: A list of ``(activation_module, per_neuron_tensor)`` pairs in forward
        order (one entry per measured layer that fired).
    """
    targets = _target_activations(network)
    if not targets:
        return []

    captured: dict[nn.Module, torch.Tensor] = {}

    def _make_hook(module: nn.Module):
        def _hook(_module: nn.Module, _inputs: Any, output: Any) -> None:
            if isinstance(output, torch.Tensor):
                captured[module] = _per_neuron_score(output)

        return _hook

    handles = [module.register_forward_hook(_make_hook(module)) for module in targets]
    was_training = network.training
    try:
        network.eval()
        with torch.no_grad():
            network(obs)
    finally:
        for handle in handles:
            handle.remove()
        network.train(was_training)

    return [(module, captured[module]) for module in targets if module in captured]


def _measure_network(network: nn.Module, obs: Any, tau: float) -> tuple[int, int]:
    """Compute ``(dormant, total)`` neuron counts for a single network.

    :param network: An ``EvolvableNetwork`` (encoder + ``head_net``).
    :param obs: A preprocessed observation batch accepted by ``network``.
    :param tau: Dormancy threshold.
    :return: ``(dormant_count, total_count)`` summed over the measured layers.
    """
    dormant = total = 0
    for _module, per_neuron in capture_per_neuron_scores(network, obs):
        d, t = _count_dormant(per_neuron, tau)
        dormant += d
        total += t
    return dormant, total


def _eval_networks(agent: Any) -> list[tuple[str | None, nn.Module]]:
    """Return the agent's evaluation networks as ``(network_id, network)`` pairs.

    Target / shared networks are excluded (only ``group.eval_network`` of each
    registry group is returned). For multi-agent algorithms each per-network
    module is yielded with its network/group id (the ``ModuleDict`` key) so the
    matching observations can be routed to it; for single-agent algorithms the id
    is ``None``.

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


def _concat_obs(obs_list: list[Any]) -> Any:
    """Concatenate a list of (possibly nested) observation batches over axis 0."""
    sample = obs_list[0]
    if isinstance(sample, dict):
        return {k: _concat_obs([o[k] for o in obs_list]) for k in sample}
    if isinstance(sample, tuple):
        return tuple(_concat_obs([o[i] for o in obs_list]) for i in range(len(sample)))
    return np.concatenate([np.asarray(o) for o in obs_list], axis=0)


def _obs_count(obs: Any) -> int:
    """Return the number of rows (batch dimension) in a nested observation."""
    if isinstance(obs, dict):
        return _obs_count(next(iter(obs.values())))
    if isinstance(obs, tuple):
        return _obs_count(obs[0])
    return int(np.asarray(obs).shape[0])


def _slice_obs(obs: Any, n: int) -> Any:
    """Keep the first *n* rows of a nested observation batch."""
    if isinstance(obs, dict):
        return {k: _slice_obs(v, n) for k, v in obs.items()}
    if isinstance(obs, tuple):
        return tuple(_slice_obs(o, n) for o in obs)
    return obs[:n]


def collect_observation_batch(
    env: Any,
    agent: Any,
    batch_size: int = 256,
    *,
    multi_agent: bool = False,
) -> Any:
    """Collect a fresh batch of observations by stepping *env* with *agent*.

    Resets the (vectorized) environment and rolls the agent forward, accumulating
    the observations it visits until at least *batch_size* rows have been
    gathered. This provides the input distribution ``D`` over which the dormant
    scores are averaged. Stepping is wrapped defensively: if the agent's action
    cannot be produced or applied, collection stops with whatever has been
    gathered so far (at least the reset observations), so the diagnostic never
    crashes training.

    :param env: A vectorized environment (single- or multi-agent).
    :param agent: The agent used to choose actions while stepping.
    :param batch_size: Target number of observation rows to collect.
    :param multi_agent: Whether *env*/*agent* use the multi-agent (dict) API.
    :return: A stacked observation batch in the same structure the environment
        emits (array / dict / tuple, or ``{agent_id: observations}``).
    """
    obs, info = env.reset()
    collected = [obs]
    count = _obs_count(obs)

    while count < batch_size:
        try:
            if multi_agent:
                action = agent.get_action(obs, info)[0]
            else:
                out = agent.get_action(obs, action_mask=info.get("action_mask"))
                action = out[0] if isinstance(out, tuple) else out
            obs, _, _, _, info = env.step(action)
        except Exception:
            break
        collected.append(obs)
        count += _obs_count(obs)

    batch = _concat_obs(collected)
    return _slice_obs(batch, batch_size)


def dormant_neuron_fraction(agent: Any, obs_batch: Any, tau: float = 0.0) -> float:
    """Fraction of τ-dormant neurons across all of *agent*'s eval networks.

    Implements Definition 3.1 of Sokar et al. (2023) aggregated over every
    measured layer of every evaluation network (targets excluded); see the
    module docstring for the exact networks and layers measured.

    :param agent: An AgileRL algorithm instance (PPO, DQN, IPPO, ...).
    :param obs_batch: A batch of raw observations as returned by the environment
        -- an array / dict / tuple for single-agent algorithms, or a mapping
        ``{agent_id: observations}`` for multi-agent algorithms.
    :param tau: Dormancy threshold (``s_i <= tau`` is dormant). Defaults to 0.0,
        i.e. only exactly-dead neurons count.
    :return: ``dormant_count / total_count`` in ``[0, 1]``, or ``nan`` if no
        neurons could be measured (also ``nan`` -- with a logged warning -- if the
        measurement fails, so the diagnostic never breaks training).
    """
    try:
        pairs = _eval_networks(agent)
        multi_agent = any(network_id is not None for network_id, _ in pairs)
        if multi_agent:
            # Networks are keyed by network/group id (AgileRL's IPPO
            # parameter-shares across same-prefix agents). Preprocess with those
            # ids so observations are grouped onto the matching network exactly
            # as ``get_action`` does.
            network_ids = list({network_id for network_id, _ in pairs})
            preprocessed = agent.preprocess_observation(obs_batch, network_ids)
        else:
            preprocessed = agent.preprocess_observation(obs_batch)

        dormant = total = 0
        for network_id, network in pairs:
            obs = preprocessed if network_id is None else preprocessed[network_id]
            d, t = _measure_network(network, obs, tau)
            dormant += d
            total += t
    except Exception as exc:  # the diagnostic must never break training
        logger.warning("Dormant-neuron fraction could not be measured: %s", exc)
        return float("nan")

    if total == 0:
        return float("nan")
    return dormant / total


def best_agent_dormant_fraction(
    agents: list[Any], obs_batch: Any, tau: float = 0.0
) -> float:
    """Dormant-neuron fraction of the best agent (highest last fitness).

    The best agent is the one with the greatest final fitness, mirroring the
    elite selection used elsewhere in the codebase. Multi-agent dict fitnesses
    are ranked by their summed value.

    :param agents: The population of agents.
    :param obs_batch: Observation batch passed to :func:`dormant_neuron_fraction`.
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
    return dormant_neuron_fraction(best, obs_batch, tau)
