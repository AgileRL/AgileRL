# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Render env observation payloads to prompt text and chat roles."""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from typing import Any

from agilerl.utils.algo_utils import is_str_keyed_dict

__all__ = [
    "DEFAULT_OBSERVATION_ROLE",
    "OBSERVATION_ROLES",
    "observation_role",
    "process_observation",
]

#: Chat role an env observation speaks as when it does not say.
DEFAULT_OBSERVATION_ROLE = "user"

#: Roles an env may claim for an observation. ``assistant`` is excluded: those
#: turns are the policy's own generations, and letting an env inject them would
#: put untrained tokens inside a trained span.
OBSERVATION_ROLES = frozenset({"user", "tool", "system"})


def process_observation(obs: object, observation_field: str | None = None) -> str:
    """Render an OpenEnv observation payload to prompt text.

    The default observation processor. A named ``observation_field`` wins: an
    env that carries its text somewhere only its author knows should be told,
    not guessed at. Otherwise the specified shapes are tried — an MCP tool
    result, our own ``TextObservation`` ``prompt``, then ``error``. Warns
    rather than returning silently empty text, since an empty prompt otherwise
    surfaces as an inexplicably broken episode much later.
    """
    if isinstance(obs, str):
        return obs
    if not is_str_keyed_dict(obs):
        warnings.warn(
            f"OpenEnv observation is a {type(obs).__name__}, not text or a "
            "mapping; the episode gets an empty prompt.",
            stacklevel=2,
        )
        return ""
    if observation_field is not None:
        named = obs.get(observation_field)
        if isinstance(named, str):
            return named
    result = obs.get("result")
    if isinstance(result, dict):
        raw_blocks = result.get("content")
        blocks = raw_blocks if isinstance(raw_blocks, list) else []
        texts: list[str] = []
        for block in blocks:
            if not is_str_keyed_dict(block):
                continue
            block_text = block.get("text")
            if isinstance(block_text, str):
                texts.append(block_text)
        if texts:
            return "\n".join(texts)
        data = result.get("data")
        if isinstance(data, str):
            return data
    prompt = obs.get("prompt")
    if isinstance(prompt, str):
        return prompt
    error = obs.get("error")
    if error:
        return f"Error: {error}"
    asked = (
        f" No '{observation_field}' either, which the manifest named."
        if observation_field is not None
        else ""
    )
    warnings.warn(
        "OpenEnv observation carried no text the episode could use, so it gets "
        f"an empty prompt. Keys present: {sorted(obs)}. Expected a 'prompt' "
        "(TextObservation), a 'result' (MCP tool call), or a named "
        "observation_field. An observation with no text fields at all (board "
        f"states, screens) needs an observation_processor to render it.{asked}",
        stacklevel=2,
    )
    return ""


def observation_role(payload: object, info: Mapping[str, Any] | None = None) -> str:
    """The chat role an env claims for one observation.

    An env labels a turn by putting ``role`` on the step metadata (``info``) or on
    the observation payload itself; ``info`` wins. Anything outside
    :data:`OBSERVATION_ROLES` is rejected rather than quietly rendered as a user
    message — a tool result presented as "the user said X" is exactly the
    confusion the label exists to prevent.

    :param payload: The raw observation payload from the env.
    :param info: Step metadata, where an env that separates the two puts the role.
    :return: One of :data:`OBSERVATION_ROLES`.
    :rtype: str
    """
    raw: object = None
    if info is not None:
        raw = info.get("role")
    if raw is None and isinstance(payload, Mapping):
        raw = payload.get("role")
    if raw is None:
        return DEFAULT_OBSERVATION_ROLE
    role = str(raw)
    if role not in OBSERVATION_ROLES:
        msg = (
            f"Env labelled an observation with role {role!r}; expected one of "
            f"{sorted(OBSERVATION_ROLES)}."
        )
        raise ValueError(msg)
    return role
