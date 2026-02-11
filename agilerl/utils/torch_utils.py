import math
from functools import singledispatch
from typing import Any, Callable, Sequence, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gymnasium import spaces


def map_pytree(f: Callable[[Union[np.ndarray, torch.Tensor]], Any], item: Any):
    """Apply a function to all tensors/arrays in a nested data structure.

    Recursively traverses nested dictionaries, lists, tuples, and sets,
    applying the given function to any numpy arrays or PyTorch tensors found.

    :param f: Function to apply to arrays/tensors
    :type f: Callable[[Union[np.ndarray, torch.Tensor]], Any]
    :param item: Nested data structure to traverse
    :type item: Any
    :return: Data structure with function applied to all arrays/tensors
    :rtype: Any
    """
    if isinstance(item, dict):
        return {k: map_pytree(f, v) for k, v in item.items()}
    elif isinstance(item, list) or isinstance(item, set) or isinstance(item, tuple):
        return [map_pytree(f, v) for v in item]
    elif isinstance(item, np.ndarray) or isinstance(item, torch.Tensor):
        return f(item)
    else:
        return item


def to(item: Any, device: torch.device):
    """Move all tensors/arrays in a nested data structure to specified device.

    :param item: Nested data structure containing tensors/arrays
    :type item: Any
    :param device: Target device to move tensors to
    :type device: torch.device
    :return: Data structure with tensors moved to device
    :rtype: Any
    """
    return map_pytree(lambda x: torch.tensor(x).to(device), item)


def to_decorator(f, device):
    """Decorator that moves the output of a function to a specified device.

    :param f: Function whose output should be moved to device
    :type f: Callable
    :param device: Target device
    :type device: torch.device
    :return: Decorated function
    :rtype: Callable
    """

    def new_f(*args, **kwargs):
        return to(f(*args, **kwargs), device)

    return new_f


def parameter_norm(model: nn.Module):
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
    attentions: list[torch.Tensor], model: nn.Module, attn_mask: torch.Tensor
):
    """Extract logging information from transformer attention weights.

    Computes attention entropy and parameter norm for transformer models,
    which can be useful for monitoring training dynamics.

    :param attentions: List of attention weight tensors from transformer layers
    :type attentions: list[torch.Tensor]
    :param model: Transformer model
    :type model: nn.Module
    :param attn_mask: Attention mask tensor
    :type attn_mask: torch.Tensor
    :return: Dictionary containing attention entropy and parameter norm
    :rtype: dict[str, tuple[float, int]]
    """
    logs = {}
    n = attn_mask.sum()
    model_attention_entropy = -sum(
        map(
            lambda x: ((x * torch.log(x + 1e-7)).sum(dim=-1) * attn_mask.unsqueeze(1))
            .sum()
            .item(),
            attentions,
        )
    ) / (len(attentions) * n)
    model_parameter_norm = parameter_norm(model)
    logs["attention_entropy"] = (model_attention_entropy, n * len(attentions))
    logs["parameter_norm"] = (model_parameter_norm, 1)
    return logs


# --------------------------------------------------------------------------- #
# Distribution helpers (Discrete, Box, MultiDiscrete, MultiBinary)            #
# Used by TorchDistribution in networks/distributions_experimental.py         #
# --------------------------------------------------------------------------- #


def sample_discrete(logits: torch.Tensor) -> torch.Tensor:
    """Sample from a categorical distribution over a discrete action space.

    :param logits: Logits of the distribution.
    :type logits: torch.Tensor
    :return: Sampled action.
    :rtype: torch.Tensor
    """
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, 1).squeeze(-1)


def log_prob_discrete(
    logits: torch.Tensor,
    action: torch.Tensor,
    n_actions: int | None = None,
) -> torch.Tensor:
    """Log probability of actions under a categorical distribution.

    :param logits: Logits of the distribution.
    :type logits: torch.Tensor
    :param action: Action.
    :type action: torch.Tensor
    :param n_actions: Number of actions.
    :type n_actions: int | None
    :return: Log probability of the action.
    :rtype: torch.Tensor
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
            raise ValueError(
                f"Action shape {action.shape} is not compatible with Discrete space. "
                f"Expected (batch_size,), (batch_size, 1), or (batch_size, num_actions). "
                f"Logits shape: {log_p_all.shape}."
            )
    else:
        raise ValueError(
            f"Action tensor ndim {action.ndim} is not compatible with logits ndim {log_p_all.ndim}. "
            f"Expected action ndim to be {log_p_all.ndim - 1} or {log_p_all.ndim}."
        )

    return log_p_all.gather(-1, action_indices_for_gather).squeeze(-1)


def entropy_discrete(logits: torch.Tensor) -> torch.Tensor:
    """Entropy of a categorical distribution.

    :param logits: Logits of the distribution.
    :type logits: torch.Tensor
    :return: Entropy of the distribution.
    :rtype: torch.Tensor
    """
    p = torch.softmax(logits, dim=-1)
    return -(p * torch.log(p + 1e-8)).sum(-1)


def sample_continuous(
    mu: torch.Tensor,
    log_std: torch.Tensor,
    squash_output: bool = False,
) -> torch.Tensor:
    """Sample from a diagonal Gaussian; optionally squash with tanh.

    :param mu: Mean of the distribution.
    :type mu: torch.Tensor
    :param log_std: Log standard deviation of the distribution.
    :type log_std: torch.Tensor
    :param squash_output: Whether to squash the output with tanh.
    :type squash_output: bool
    :return: Sampled action.
    :rtype: torch.Tensor
    """
    eps = torch.randn_like(mu)
    out = mu + torch.exp(log_std) * eps
    if squash_output:
        out = torch.tanh(out)
    return out


def log_prob_continuous(
    mu: torch.Tensor,
    log_std: torch.Tensor,
    action: torch.Tensor,
) -> torch.Tensor:
    """Log probability of actions under a diagonal Gaussian.

    :param mu: Mean of the distribution.
    :type mu: torch.Tensor
    :param log_std: Log standard deviation of the distribution.
    :type log_std: torch.Tensor
    :param action: Action.
    :type action: torch.Tensor
    :return: Log probability of the action.
    :rtype: torch.Tensor
    """
    var = torch.exp(2 * log_std)
    return (
        -0.5 * (((action - mu) ** 2) / var + 2 * log_std + math.log(2 * math.pi))
    ).sum(-1)


def entropy_continuous(mu: torch.Tensor, log_std: torch.Tensor) -> torch.Tensor:
    """Entropy of a diagonal Gaussian.

    :param mu: Mean of the distribution.
    :type mu: torch.Tensor
    :param log_std: Log standard deviation of the distribution.
    :type log_std: torch.Tensor
    :return: Entropy of the distribution.
    :rtype: torch.Tensor
    """
    return 0.5 * (1 + math.log(2 * math.pi)) * mu.size(-1) + log_std.sum(-1)


def sample_multi_discrete(
    logits: torch.Tensor,
    nvec: Sequence[int],
) -> torch.Tensor:
    """Sample from independent categoricals for a MultiDiscrete action space.

    :param logits: Logits of the distribution.
    :type logits: torch.Tensor
    :param nvec: Number of actions for each discrete action space.
    :type nvec: Sequence[int]
    :return: Sampled action.
    :rtype: torch.Tensor
    """
    actions: list[torch.Tensor] = []
    offset = 0
    for size in nvec:
        logits_i = logits[:, offset : offset + size]
        probs_i = torch.softmax(logits_i, dim=-1)
        act_i = torch.multinomial(probs_i, 1).squeeze(-1)
        actions.append(act_i)
        offset += size
    return torch.stack(actions, dim=-1)


def log_prob_multi_discrete(
    logits: torch.Tensor,
    nvec: Sequence[int],
    action: torch.Tensor,
) -> torch.Tensor:
    """Log probability of actions under independent categoricals.

    :param logits: Logits of the distribution.
    :type logits: torch.Tensor
    :param nvec: Number of actions for each discrete action space.
    :type nvec: Sequence[int]
    :param action: Action.
    :type action: torch.Tensor
    :return: Log probability of the action.
    :rtype: torch.Tensor
    """
    logps = []
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
    logits: torch.Tensor,
    nvec: Sequence[int],
) -> torch.Tensor:
    """Entropy of independent categoricals for MultiDiscrete.

    :param logits: Logits of the distribution.
    :type logits: torch.Tensor
    :param nvec: Number of actions for each discrete action space.
    :type nvec: Sequence[int]
    :return: Entropy of the distribution.
    :rtype: torch.Tensor
    """
    entropies = []
    offset = 0
    for size in nvec:
        logits_i = logits[:, offset : offset + size]
        p_i = torch.softmax(logits_i, dim=-1)
        ent_i = -(p_i * torch.log(p_i + 1e-8)).sum(-1)
        entropies.append(ent_i)
        offset += size
    return torch.stack(entropies, dim=-1).sum(-1)


def sample_multi_binary(logits: torch.Tensor) -> torch.Tensor:
    """Sample from independent Bernoullis for a MultiBinary action space.

    :param logits: Logits of the distribution.
    :type logits: torch.Tensor
    :return: Sampled action.
    :rtype: torch.Tensor
    """
    probs = torch.sigmoid(logits)
    return torch.bernoulli(probs)


def log_prob_multi_binary(
    logits: torch.Tensor,
    action: torch.Tensor,
) -> torch.Tensor:
    """Log probability of actions under independent Bernoullis.

    :param logits: Logits of the distribution.
    :type logits: torch.Tensor
    :param action: Action.
    :type action: torch.Tensor
    :return: Log probability of the action.
    :rtype: torch.Tensor
    """
    log_p1 = -F.softplus(-logits)
    log_p0 = -logits + log_p1
    a = action.float()
    return (a * log_p1 + (1.0 - a) * log_p0).sum(-1)


def entropy_multi_binary(logits: torch.Tensor) -> torch.Tensor:
    """Entropy of independent Bernoullis for MultiBinary.

    :param logits: Logits of the distribution.
    :type logits: torch.Tensor
    :return: Entropy of the distribution.
    :rtype: torch.Tensor
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
    logits: torch.Tensor | None = None,
    mu: torch.Tensor | None = None,
    log_std: torch.Tensor | None = None,
    squash_output: bool = False,
) -> torch.Tensor:
    """Sample from the distribution for the given action space. Dispatches on action_space type.

    :param action_space: Action space.
    :type action_space: spaces.Space
    :param logits: Logits of the distribution.
    :type logits: torch.Tensor | None
    :param mu: Mean of the distribution.
    :type mu: torch.Tensor | None
    :param log_std: Log standard deviation of the distribution.
    :type log_std: torch.Tensor | None
    :param squash_output: Whether to squash the output.
    :type squash_output: bool
    :return: Sampled action.
    :rtype: torch.Tensor
    :raises NotImplementedError: If the action space is not supported.
    """
    raise NotImplementedError(
        f"Unsupported action space for sampling: {type(action_space).__name__}"
    )


@sample_from_space.register(spaces.Discrete)
def _sample_discrete(
    action_space: spaces.Discrete,
    *,
    logits: torch.Tensor | None = None,
    mu: torch.Tensor | None = None,
    log_std: torch.Tensor | None = None,
    squash_output: bool = False,
) -> torch.Tensor:
    assert logits is not None
    return sample_discrete(logits)


@sample_from_space.register(spaces.Box)
def _sample_box(
    action_space: spaces.Box,
    *,
    logits: torch.Tensor | None = None,
    mu: torch.Tensor | None = None,
    log_std: torch.Tensor | None = None,
    squash_output: bool = False,
) -> torch.Tensor:
    assert mu is not None and log_std is not None
    return sample_continuous(mu, log_std, squash_output)


@sample_from_space.register(spaces.MultiDiscrete)
def _sample_multi_discrete(
    action_space: spaces.MultiDiscrete,
    *,
    logits: torch.Tensor | None = None,
    mu: torch.Tensor | None = None,
    log_std: torch.Tensor | None = None,
    squash_output: bool = False,
) -> torch.Tensor:
    assert logits is not None
    return sample_multi_discrete(logits, action_space.nvec)


@sample_from_space.register(spaces.MultiBinary)
def _sample_multi_binary(
    action_space: spaces.MultiBinary,
    *,
    logits: torch.Tensor | None = None,
    mu: torch.Tensor | None = None,
    log_std: torch.Tensor | None = None,
    squash_output: bool = False,
) -> torch.Tensor:
    assert logits is not None
    return sample_multi_binary(logits)


@singledispatch
def log_prob_from_space(
    action_space: spaces.Space,
    action: torch.Tensor,
    *,
    logits: torch.Tensor | None = None,
    mu: torch.Tensor | None = None,
    log_std: torch.Tensor | None = None,
) -> torch.Tensor:
    """Log probability of action under the distribution. Dispatches on action_space type.

    :param action_space: Action space.
    :type action_space: spaces.Space
    :param action: Action.
    :type action: torch.Tensor
    :param logits: Logits of the distribution.
    :type logits: torch.Tensor | None
    :param mu: Mean of the distribution.
    :type mu: torch.Tensor | None
    :param log_std: Log standard deviation of the distribution.
    :type log_std: torch.Tensor | None
    :return: Log probability of the action.
    :rtype: torch.Tensor
    :raises NotImplementedError: If the action space is not supported.
    """
    raise NotImplementedError(
        f"Unsupported action space for log_prob: {type(action_space).__name__}"
    )


@log_prob_from_space.register(spaces.Discrete)
def _log_prob_discrete(
    action_space: spaces.Discrete,
    action: torch.Tensor,
    *,
    logits: torch.Tensor | None = None,
    mu: torch.Tensor | None = None,
    log_std: torch.Tensor | None = None,
) -> torch.Tensor:
    assert logits is not None
    n_actions = getattr(action_space, "n", None)
    return log_prob_discrete(logits, action, n_actions)


@log_prob_from_space.register(spaces.Box)
def _log_prob_box(
    action_space: spaces.Box,
    action: torch.Tensor,
    *,
    logits: torch.Tensor | None = None,
    mu: torch.Tensor | None = None,
    log_std: torch.Tensor | None = None,
) -> torch.Tensor:
    assert mu is not None and log_std is not None
    return log_prob_continuous(mu, log_std, action)


@log_prob_from_space.register(spaces.MultiDiscrete)
def _log_prob_multi_discrete(
    action_space: spaces.MultiDiscrete,
    action: torch.Tensor,
    *,
    logits: torch.Tensor | None = None,
    mu: torch.Tensor | None = None,
    log_std: torch.Tensor | None = None,
) -> torch.Tensor:
    assert logits is not None
    return log_prob_multi_discrete(logits, action_space.nvec, action)


@log_prob_from_space.register(spaces.MultiBinary)
def _log_prob_multi_binary(
    action_space: spaces.MultiBinary,
    action: torch.Tensor,
    *,
    logits: torch.Tensor | None = None,
    mu: torch.Tensor | None = None,
    log_std: torch.Tensor | None = None,
) -> torch.Tensor:
    assert logits is not None
    return log_prob_multi_binary(logits, action)


@singledispatch
def entropy_from_space(
    action_space: spaces.Space,
    *,
    logits: torch.Tensor | None = None,
    mu: torch.Tensor | None = None,
    log_std: torch.Tensor | None = None,
) -> torch.Tensor:
    """Entropy of the distribution. Dispatches on action_space type.

    :param action_space: Action space.
    :type action_space: spaces.Space
    :param logits: Logits of the distribution.
    :type logits: torch.Tensor | None
    :param mu: Mean of the distribution.
    :type mu: torch.Tensor | None
    :param log_std: Log standard deviation of the distribution.
    :type log_std: torch.Tensor | None
    :return: Entropy of the distribution.
    :rtype: torch.Tensor
    :raises NotImplementedError: If the action space is not supported.
    """
    raise NotImplementedError(
        f"Unsupported action space for entropy: {type(action_space).__name__}"
    )


@entropy_from_space.register(spaces.Discrete)
def _entropy_discrete(
    action_space: spaces.Discrete,
    *,
    logits: torch.Tensor | None = None,
    mu: torch.Tensor | None = None,
    log_std: torch.Tensor | None = None,
) -> torch.Tensor:
    assert logits is not None
    return entropy_discrete(logits)


@entropy_from_space.register(spaces.Box)
def _entropy_box(
    action_space: spaces.Box,
    *,
    logits: torch.Tensor | None = None,
    mu: torch.Tensor | None = None,
    log_std: torch.Tensor | None = None,
) -> torch.Tensor:
    assert mu is not None and log_std is not None
    return entropy_continuous(mu, log_std)


@entropy_from_space.register(spaces.MultiDiscrete)
def _entropy_multi_discrete(
    action_space: spaces.MultiDiscrete,
    *,
    logits: torch.Tensor | None = None,
    mu: torch.Tensor | None = None,
    log_std: torch.Tensor | None = None,
) -> torch.Tensor:
    assert logits is not None
    return entropy_multi_discrete(logits, action_space.nvec)


@entropy_from_space.register(spaces.MultiBinary)
def _entropy_multi_binary(
    action_space: spaces.MultiBinary,
    *,
    logits: torch.Tensor | None = None,
    mu: torch.Tensor | None = None,
    log_std: torch.Tensor | None = None,
) -> torch.Tensor:
    assert logits is not None
    return entropy_multi_binary(logits)
