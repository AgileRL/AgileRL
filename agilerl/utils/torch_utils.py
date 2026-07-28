import math
from collections.abc import Callable, Sequence
from functools import singledispatch
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
import torch.nn.functional as F
from gymnasium import spaces
from jaxtyping import Float, Int, Num, Shaped

from agilerl.typing import ActionLogits, LogProbs

# Per-sub-action category counts of a ``MultiDiscrete`` space (``spaces.MultiDiscrete.nvec``).
NvecType = Sequence[int] | Int[npt.NDArray[np.integer], " num_sub_actions"]


def map_pytree(
    f: Callable[[np.ndarray | torch.Tensor], Any],
    item: object,
) -> object:
    """Apply a function to all tensors/arrays in a nested data structure.

    Recursively traverses nested dictionaries, lists, tuples, and sets,
    applying the given function to any numpy arrays or PyTorch tensors found.

    :param f: Function to apply to arrays/tensors
    :type f: Callable[[npt.NDArray | torch.Tensor], Any]
    :param item: Nested data structure to traverse
    :type item: Any
    :return: Data structure with function applied to all arrays/tensors
    :rtype: Any
    """
    if not callable(f):
        msg = "f must be callable."
        raise TypeError(msg)

    if isinstance(item, dict):
        return {k: map_pytree(f, v) for k, v in item.items()}
    if isinstance(item, (list, set, tuple)):
        return [map_pytree(f, v) for v in item]
    if isinstance(item, (np.ndarray, torch.Tensor)):
        return f(item)
    return item


def to(item: object, device: torch.device | str) -> object:
    """Move all tensors/arrays in a nested data structure to specified device.

    :param item: Nested data structure containing tensors/arrays
    :type item: Any
    :param device: Target device to move tensors to
    :type device: torch.device
    :return: Data structure with tensors moved to device
    :rtype: Any
    """
    return map_pytree(lambda x: torch.tensor(x).to(device), item)


def to_decorator(
    f: Callable[..., Any],
    device: torch.device | str,
) -> Callable[..., Any]:
    """Move the output of a function to a specified device (decorator).

    :param f: Function whose output should be moved to device
    :type f: Callable
    :param device: Target device
    :type device: torch.device
    :return: Decorated function
    :rtype: Callable
    """

    def new_f(*args: Any, **kwargs: Any) -> object:
        return to(f(*args, **kwargs), device)

    return new_f


def parameter_norm(model: nn.Module) -> float:
    """Calculate the L2 norm of all parameters in a model.

    :param model: PyTorch model
    :type model: nn.Module
    :return: L2 norm of all model parameters
    :rtype: float
    """
    norm = 0.0
    for param in model.parameters():
        norm += (param.norm() ** 2).item()
    return math.sqrt(norm)


def get_transformer_logs(
    attentions: list[Float[torch.Tensor, "..."]],
    model: nn.Module,
    attn_mask: Shaped[torch.Tensor, "..."],
) -> dict[str, tuple[float | Float[torch.Tensor, ""], int | Num[torch.Tensor, ""]]]:
    """Extract logging information from transformer attention weights.

    Computes attention entropy and parameter norm for transformer models,
    which can be useful for monitoring training dynamics.

    :param attentions: List of attention weight tensors from transformer layers
    :type attentions: list[Float[torch.Tensor, "..."]]
    :param model: Transformer model
    :type model: nn.Module
    :param attn_mask: Attention mask tensor
    :type attn_mask: Shaped[torch.Tensor, "..."]
    :return: Dictionary containing attention entropy and parameter norm; the
        attention-entropy pair stays as 0-dim tensors derived from ``attn_mask``.
    :rtype: dict[str, tuple[float | Float[torch.Tensor, ""], int | Num[torch.Tensor, ""]]]
    """
    logs: dict[
        str, tuple[float | Float[torch.Tensor, ""], int | Num[torch.Tensor, ""]]
    ] = {}
    n = attn_mask.sum()
    model_attention_entropy = -sum(
        (
            ((x * torch.log(x + 1e-7)).sum(dim=-1) * attn_mask.unsqueeze(1))
            .sum()
            .item()
            for x in attentions
        ),
    ) / (len(attentions) * n)
    model_parameter_norm = parameter_norm(model)
    logs["attention_entropy"] = (model_attention_entropy, n * len(attentions))
    logs["parameter_norm"] = (model_parameter_norm, 1)
    return logs


# --------------------------------------------------------------------------- #
# Distribution helpers (Discrete, Box, MultiDiscrete, MultiBinary)            #
# Used by TorchDistribution in networks/distributions.py         #
# --------------------------------------------------------------------------- #


def sample_discrete(
    logits: Float[torch.Tensor, "batch num_actions"],
) -> Int[torch.Tensor, " batch"]:
    """Sample from a categorical distribution over a discrete action space.

    :param logits: Logits of the distribution.
    :type logits: Float[torch.Tensor, "batch num_actions"]
    :return: Sampled action.
    :rtype: Int[torch.Tensor, " batch"]
    """
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, 1).squeeze(-1)


def log_prob_discrete(
    logits: Float[torch.Tensor, "batch num_actions"],
    action: Num[torch.Tensor, "..."],
    n_actions: int | None = None,
) -> LogProbs:
    """Log probability of actions under a categorical distribution.

    :param logits: Logits of the distribution.
    :type logits: Float[torch.Tensor, "batch num_actions"]
    :param action: Action, as ``(batch,)``, ``(batch, 1)`` or one-hot ``(batch, num_actions)``;
        any other rank or leading size raises ``ValueError``.
    :type action: Num[torch.Tensor, "..."]
    :param n_actions: Number of actions.
    :type n_actions: int | None
    :return: Log probability of the action.
    :rtype: LogProbs
    :raises ValueError: If the action shape is not compatible with the logits shape.
    """
    log_p_all = torch.log_softmax(logits, dim=-1)
    action_long = action.long()

    if action_long.ndim == log_p_all.ndim - 1:
        action_indices_for_gather = action_long.unsqueeze(-1)
    elif action_long.ndim == log_p_all.ndim:
        if action_long.shape[-1] == 1:
            action_indices_for_gather = action_long
        elif (
            n_actions is not None
            and action_long.shape == log_p_all.shape
            and action_long.shape[-1] == n_actions
        ):
            action_indices_for_gather = torch.argmax(action_long, dim=-1, keepdim=True)
        else:
            msg = (
                f"Action shape {action.shape} is not compatible with Discrete space. "
                f"Expected (batch_size,), (batch_size, 1), or (batch_size, num_actions). "
                f"Logits shape: {log_p_all.shape}."
            )
            raise ValueError(msg)
    else:
        msg = (
            f"Action tensor ndim {action.ndim} is not compatible with logits ndim {log_p_all.ndim}. "
            f"Expected action ndim to be {log_p_all.ndim - 1} or {log_p_all.ndim}."
        )
        raise ValueError(msg)

    return log_p_all.gather(-1, action_indices_for_gather).squeeze(-1)


def entropy_discrete(
    logits: Float[torch.Tensor, "batch num_actions"],
) -> Float[torch.Tensor, " batch"]:
    """Entropy of a categorical distribution.

    :param logits: Logits of the distribution.
    :type logits: Float[torch.Tensor, "batch num_actions"]
    :return: Entropy of the distribution.
    :rtype: Float[torch.Tensor, " batch"]
    """
    p = torch.softmax(logits, dim=-1)
    return -(p * torch.log(p + 1e-8)).sum(-1)


def sample_continuous(
    mu: Float[torch.Tensor, "batch action_dim"],
    log_std: Float[torch.Tensor, "batch action_dim"],
    squash_output: bool = False,
) -> Float[torch.Tensor, "batch action_dim"]:
    """Sample from a diagonal Gaussian; optionally squash with tanh.

    :param mu: Mean of the distribution.
    :type mu: Float[torch.Tensor, "batch action_dim"]
    :param log_std: Log standard deviation of the distribution.
    :type log_std: Float[torch.Tensor, "batch action_dim"]
    :param squash_output: Whether to squash the output with tanh.
    :type squash_output: bool
    :return: Sampled action.
    :rtype: Float[torch.Tensor, "batch action_dim"]
    """
    eps = torch.randn_like(mu)
    out = mu + torch.exp(log_std) * eps
    if squash_output:
        out = torch.tanh(out)
    return out


def log_prob_continuous(
    mu: Float[torch.Tensor, "batch action_dim"],
    log_std: Float[torch.Tensor, "batch action_dim"],
    action: Float[torch.Tensor, "batch ..."],
) -> LogProbs:
    """Log probability of actions under a diagonal Gaussian.

    :param mu: Mean of the distribution.
    :type mu: Float[torch.Tensor, "batch action_dim"]
    :param log_std: Log standard deviation of the distribution.
    :type log_std: Float[torch.Tensor, "batch action_dim"]
    :param action: Action, shaped like ``mu``. The rank is left unconstrained
        because callers reach here having dropped the trailing axis of a width-1
        action space, which broadcasts to ``(batch, batch)`` and collapses to one
        log probability repeated across the batch.
    :type action: Float[torch.Tensor, "batch ..."]
    :return: Log probability of the action.
    :rtype: LogProbs
    """
    var = torch.exp(2 * log_std)
    return (
        -0.5 * (((action - mu) ** 2) / var + 2 * log_std + math.log(2 * math.pi))
    ).sum(-1)


def entropy_continuous(
    mu: Float[torch.Tensor, "batch action_dim"],
    log_std: Float[torch.Tensor, "batch action_dim"],
) -> Float[torch.Tensor, " batch"]:
    """Entropy of a diagonal Gaussian.

    :param mu: Mean of the distribution.
    :type mu: Float[torch.Tensor, "batch action_dim"]
    :param log_std: Log standard deviation of the distribution.
    :type log_std: Float[torch.Tensor, "batch action_dim"]
    :return: Entropy of the distribution.
    :rtype: Float[torch.Tensor, " batch"]
    """
    return 0.5 * (1 + math.log(2 * math.pi)) * mu.size(-1) + log_std.sum(-1)


def sample_multi_discrete(
    logits: Float[torch.Tensor, "batch sum_nvec"],
    nvec: NvecType,
) -> Int[torch.Tensor, "batch num_sub_actions"]:
    """Sample from independent categoricals for a MultiDiscrete action space.

    :param logits: Logits of the distribution.
    :type logits: Float[torch.Tensor, "batch sum_nvec"]
    :param nvec: Number of actions for each discrete action space.
    :type nvec: NvecType
    :return: Sampled action.
    :rtype: Int[torch.Tensor, "batch num_sub_actions"]
    """
    actions: list[Int[torch.Tensor, " batch"]] = []
    offset = 0
    for size in nvec:
        logits_i = logits[:, offset : offset + size]
        probs_i = torch.softmax(logits_i, dim=-1)
        act_i = torch.multinomial(probs_i, 1).squeeze(-1)
        actions.append(act_i)
        offset += size
    return torch.stack(actions, dim=-1)


def log_prob_multi_discrete(
    logits: Float[torch.Tensor, "batch sum_nvec"],
    nvec: NvecType,
    action: Num[torch.Tensor, "batch num_sub_actions"],
) -> LogProbs:
    """Log probability of actions under independent categoricals.

    :param logits: Logits of the distribution.
    :type logits: Float[torch.Tensor, "batch sum_nvec"]
    :param nvec: Number of actions for each discrete action space.
    :type nvec: NvecType
    :param action: Action.
    :type action: Num[torch.Tensor, "batch num_sub_actions"]
    :return: Log probability of the action.
    :rtype: LogProbs
    """
    logps: list[LogProbs] = []
    offset = 0
    for idx, size in enumerate(nvec):
        logits_i = logits[:, offset : offset + size]
        logp_all = torch.log_softmax(logits_i, dim=-1)
        act_i = action[:, idx].long()
        logp_i = logp_all.gather(-1, act_i.unsqueeze(-1)).squeeze(-1)
        logps.append(logp_i)
        offset += size
    return torch.stack(logps, dim=-1).sum(-1)


def entropy_multi_discrete(
    logits: Float[torch.Tensor, "batch sum_nvec"],
    nvec: NvecType,
) -> Float[torch.Tensor, " batch"]:
    """Entropy of independent categoricals for MultiDiscrete.

    :param logits: Logits of the distribution.
    :type logits: Float[torch.Tensor, "batch sum_nvec"]
    :param nvec: Number of actions for each discrete action space.
    :type nvec: NvecType
    :return: Entropy of the distribution.
    :rtype: Float[torch.Tensor, " batch"]
    """
    entropies: list[Float[torch.Tensor, " batch"]] = []
    offset = 0
    for size in nvec:
        logits_i = logits[:, offset : offset + size]
        p_i = torch.softmax(logits_i, dim=-1)
        ent_i = -(p_i * torch.log(p_i + 1e-8)).sum(-1)
        entropies.append(ent_i)
        offset += size
    return torch.stack(entropies, dim=-1).sum(-1)


def sample_multi_binary(
    logits: Float[torch.Tensor, "batch num_binary"],
) -> Float[torch.Tensor, "batch num_binary"]:
    """Sample from independent Bernoullis for a MultiBinary action space.

    :param logits: Logits of the distribution.
    :type logits: Float[torch.Tensor, "batch num_binary"]
    :return: Sampled action, 0.0/1.0 valued in the dtype of ``logits``.
    :rtype: Float[torch.Tensor, "batch num_binary"]
    """
    probs = torch.sigmoid(logits)
    return torch.bernoulli(probs)


def log_prob_multi_binary(
    logits: Float[torch.Tensor, "batch num_binary"],
    action: Num[torch.Tensor, "batch num_binary"],
) -> LogProbs:
    """Log probability of actions under independent Bernoullis.

    :param logits: Logits of the distribution.
    :type logits: Float[torch.Tensor, "batch num_binary"]
    :param action: Action.
    :type action: Num[torch.Tensor, "batch num_binary"]
    :return: Log probability of the action.
    :rtype: LogProbs
    """
    log_p1 = -F.softplus(-logits)
    log_p0 = -logits + log_p1
    a = action.float()
    return (a * log_p1 + (1.0 - a) * log_p0).sum(-1)


def entropy_multi_binary(
    logits: Float[torch.Tensor, "batch num_binary"],
) -> Float[torch.Tensor, " batch"]:
    """Entropy of independent Bernoullis for MultiBinary.

    :param logits: Logits of the distribution.
    :type logits: Float[torch.Tensor, "batch num_binary"]
    :return: Entropy of the distribution.
    :rtype: Float[torch.Tensor, " batch"]
    """
    p = torch.sigmoid(logits)
    return -(p * torch.log(p + 1e-8) + (1 - p) * torch.log(1 - p + 1e-8)).sum(-1)


# --------------------------------------------------------------------------- #
# Single-dispatch API: pass action_space as first argument                    #
# --------------------------------------------------------------------------- #


@singledispatch
def sample_from_space(
    action_space: spaces.Space,
    *,
    logits: ActionLogits | None = None,
    mu: Float[torch.Tensor, "batch action_dim"] | None = None,
    log_std: Float[torch.Tensor, "batch action_dim"] | None = None,
    squash_output: bool = False,
) -> Num[torch.Tensor, "batch ..."]:
    """Sample from the distribution for the given action space. Dispatches on action_space type.

    :param action_space: Action space.
    :type action_space: spaces.Space
    :param logits: Logits of the distribution.
    :type logits: ActionLogits | None
    :param mu: Mean of the distribution.
    :type mu: Float[torch.Tensor, "batch action_dim"] | None
    :param log_std: Log standard deviation of the distribution.
    :type log_std: Float[torch.Tensor, "batch action_dim"] | None
    :param squash_output: Whether to squash the output.
    :type squash_output: bool
    :return: Sampled action; integer for Discrete/MultiDiscrete, floating for
        Box/MultiBinary.
    :rtype: Num[torch.Tensor, "batch ..."]
    :raises NotImplementedError: If the action space is not supported.
    """
    msg = f"Unsupported action space for sampling: {type(action_space).__name__}"
    raise NotImplementedError(msg)


@sample_from_space.register(spaces.Discrete)
def _sample_discrete(
    action_space: spaces.Discrete,
    *,
    logits: ActionLogits | None = None,
    mu: Float[torch.Tensor, "batch action_dim"] | None = None,
    log_std: Float[torch.Tensor, "batch action_dim"] | None = None,
    squash_output: bool = False,
) -> Num[torch.Tensor, "batch ..."]:
    assert logits is not None
    return sample_discrete(logits)


@sample_from_space.register(spaces.Box)
def _sample_box(
    action_space: spaces.Box,
    *,
    logits: ActionLogits | None = None,
    mu: Float[torch.Tensor, "batch action_dim"] | None = None,
    log_std: Float[torch.Tensor, "batch action_dim"] | None = None,
    squash_output: bool = False,
) -> Num[torch.Tensor, "batch ..."]:
    assert mu is not None
    assert log_std is not None
    return sample_continuous(mu, log_std, squash_output)


@sample_from_space.register(spaces.MultiDiscrete)
def _sample_multi_discrete(
    action_space: spaces.MultiDiscrete,
    *,
    logits: ActionLogits | None = None,
    mu: Float[torch.Tensor, "batch action_dim"] | None = None,
    log_std: Float[torch.Tensor, "batch action_dim"] | None = None,
    squash_output: bool = False,
) -> Num[torch.Tensor, "batch ..."]:
    assert logits is not None
    return sample_multi_discrete(logits, action_space.nvec)


@sample_from_space.register(spaces.MultiBinary)
def _sample_multi_binary(
    action_space: spaces.MultiBinary,
    *,
    logits: ActionLogits | None = None,
    mu: Float[torch.Tensor, "batch action_dim"] | None = None,
    log_std: Float[torch.Tensor, "batch action_dim"] | None = None,
    squash_output: bool = False,
) -> Num[torch.Tensor, "batch ..."]:
    assert logits is not None
    return sample_multi_binary(logits)


@singledispatch
def log_prob_from_space(
    action_space: spaces.Space,
    action: Num[torch.Tensor, "batch ..."],
    *,
    logits: ActionLogits | None = None,
    mu: Float[torch.Tensor, "batch action_dim"] | None = None,
    log_std: Float[torch.Tensor, "batch action_dim"] | None = None,
) -> LogProbs:
    """Log probability of action under the distribution. Dispatches on action_space type.

    :param action_space: Action space.
    :type action_space: spaces.Space
    :param action: Action.
    :type action: Num[torch.Tensor, "batch ..."]
    :param logits: Logits of the distribution.
    :type logits: ActionLogits | None
    :param mu: Mean of the distribution.
    :type mu: Float[torch.Tensor, "batch action_dim"] | None
    :param log_std: Log standard deviation of the distribution.
    :type log_std: Float[torch.Tensor, "batch action_dim"] | None
    :return: Log probability of the action.
    :rtype: LogProbs
    :raises NotImplementedError: If the action space is not supported.
    """
    msg = f"Unsupported action space for log_prob: {type(action_space).__name__}"
    raise NotImplementedError(msg)


@log_prob_from_space.register(spaces.Discrete)
def _log_prob_discrete(
    action_space: spaces.Discrete,
    action: Num[torch.Tensor, "batch ..."],
    *,
    logits: ActionLogits | None = None,
    mu: Float[torch.Tensor, "batch action_dim"] | None = None,
    log_std: Float[torch.Tensor, "batch action_dim"] | None = None,
) -> LogProbs:
    assert logits is not None
    n_actions = getattr(action_space, "n", None)
    return log_prob_discrete(logits, action, n_actions)


@log_prob_from_space.register(spaces.Box)
def _log_prob_box(
    action_space: spaces.Box,
    action: Num[torch.Tensor, "batch ..."],
    *,
    logits: ActionLogits | None = None,
    mu: Float[torch.Tensor, "batch action_dim"] | None = None,
    log_std: Float[torch.Tensor, "batch action_dim"] | None = None,
) -> LogProbs:
    assert mu is not None
    assert log_std is not None
    return log_prob_continuous(mu, log_std, action)


@log_prob_from_space.register(spaces.MultiDiscrete)
def _log_prob_multi_discrete(
    action_space: spaces.MultiDiscrete,
    action: Num[torch.Tensor, "batch ..."],
    *,
    logits: ActionLogits | None = None,
    mu: Float[torch.Tensor, "batch action_dim"] | None = None,
    log_std: Float[torch.Tensor, "batch action_dim"] | None = None,
) -> LogProbs:
    assert logits is not None
    return log_prob_multi_discrete(logits, action_space.nvec, action)


@log_prob_from_space.register(spaces.MultiBinary)
def _log_prob_multi_binary(
    action_space: spaces.MultiBinary,
    action: Num[torch.Tensor, "batch ..."],
    *,
    logits: ActionLogits | None = None,
    mu: Float[torch.Tensor, "batch action_dim"] | None = None,
    log_std: Float[torch.Tensor, "batch action_dim"] | None = None,
) -> LogProbs:
    assert logits is not None
    return log_prob_multi_binary(logits, action)


@singledispatch
def entropy_from_space(
    action_space: spaces.Space,
    *,
    logits: ActionLogits | None = None,
    mu: Float[torch.Tensor, "batch action_dim"] | None = None,
    log_std: Float[torch.Tensor, "batch action_dim"] | None = None,
) -> Float[torch.Tensor, " batch"]:
    """Entropy of the distribution. Dispatches on action_space type.

    :param action_space: Action space.
    :type action_space: spaces.Space
    :param logits: Logits of the distribution.
    :type logits: ActionLogits | None
    :param mu: Mean of the distribution.
    :type mu: Float[torch.Tensor, "batch action_dim"] | None
    :param log_std: Log standard deviation of the distribution.
    :type log_std: Float[torch.Tensor, "batch action_dim"] | None
    :return: Entropy of the distribution.
    :rtype: Float[torch.Tensor, " batch"]
    :raises NotImplementedError: If the action space is not supported.
    """
    msg = f"Unsupported action space for entropy: {type(action_space).__name__}"
    raise NotImplementedError(msg)


@entropy_from_space.register(spaces.Discrete)
def _entropy_discrete(
    action_space: spaces.Discrete,
    *,
    logits: ActionLogits | None = None,
    mu: Float[torch.Tensor, "batch action_dim"] | None = None,
    log_std: Float[torch.Tensor, "batch action_dim"] | None = None,
) -> Float[torch.Tensor, " batch"]:
    assert logits is not None
    return entropy_discrete(logits)


@entropy_from_space.register(spaces.Box)
def _entropy_box(
    action_space: spaces.Box,
    *,
    logits: ActionLogits | None = None,
    mu: Float[torch.Tensor, "batch action_dim"] | None = None,
    log_std: Float[torch.Tensor, "batch action_dim"] | None = None,
) -> Float[torch.Tensor, " batch"]:
    assert mu is not None
    assert log_std is not None
    return entropy_continuous(mu, log_std)


@entropy_from_space.register(spaces.MultiDiscrete)
def _entropy_multi_discrete(
    action_space: spaces.MultiDiscrete,
    *,
    logits: ActionLogits | None = None,
    mu: Float[torch.Tensor, "batch action_dim"] | None = None,
    log_std: Float[torch.Tensor, "batch action_dim"] | None = None,
) -> Float[torch.Tensor, " batch"]:
    assert logits is not None
    return entropy_multi_discrete(logits, action_space.nvec)


@entropy_from_space.register(spaces.MultiBinary)
def _entropy_multi_binary(
    action_space: spaces.MultiBinary,
    *,
    logits: ActionLogits | None = None,
    mu: Float[torch.Tensor, "batch action_dim"] | None = None,
    log_std: Float[torch.Tensor, "batch action_dim"] | None = None,
) -> Float[torch.Tensor, " batch"]:
    assert logits is not None
    return entropy_multi_binary(logits)
