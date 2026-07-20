"""Base64 ``.npy`` serialization for Arena RL and supervised inference payloads."""

from __future__ import annotations

import base64
import io
from typing import TypeAlias

import numpy as np

# The wire format is a pytree: arrays at the leaves (or None where a payload
# such as a per-agent action mask is absent), nested through dicts and tuples.
# In practice AgileRL bounds the depth — observation/action spaces reject
# nesting beyond one dict/tuple level (see check_supported_space) and
# multi-agent payloads add one level on top — but the codec is
# structure-agnostic.
#
# NOTE for these functions' bodies: the leaf isinstance checks must run before
# the container checks. `ndarray` is not a final class, so while it remains in
# the narrowed union, `isinstance(x, dict)` leaves an `ndarray & dict`
# intersection open and ty degrades the container's values to `object`.
RLData: TypeAlias = np.ndarray | dict[str, "RLData"] | tuple["RLData", ...] | None
SerializedRLData: TypeAlias = (
    str | dict[str, "SerializedRLData"] | tuple["SerializedRLData", ...] | None
)


def _encode_array(data: np.ndarray, batched: bool) -> str:
    """Encode a single array as a base64-encoded ``.npy`` payload."""
    if not batched:
        data = np.expand_dims(data, axis=0)

    buffer = io.BytesIO()
    np.save(buffer, data, allow_pickle=False)
    buffer.seek(0)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _decode_array(data: str, batched: bool) -> np.ndarray:
    """Decode a base64-encoded ``.npy`` payload back to an array."""
    decoded = base64.b64decode(data, validate=True)
    arr: np.ndarray = np.load(io.BytesIO(decoded), allow_pickle=False)
    return arr if batched else arr.squeeze(axis=0)


def serialize(data: RLData, batched: bool = False) -> SerializedRLData:
    """Serialize RL data to a base64-encoded ``.npy`` representation.

    :param data: The RL data to serialize.
    :type data: RLData
    :param batched: Whether the data is batched.
    :type batched: bool
    :return: The serialized RL data.
    :rtype: SerializedRLData
    """
    if data is None:
        return None
    if isinstance(data, np.ndarray):
        return _encode_array(data, batched)
    if isinstance(data, dict):
        return {k: serialize(v, batched) for k, v in data.items()}
    if isinstance(data, (tuple, list)):
        return tuple(serialize(v, batched) for v in data)
    return _encode_array(data, batched)


def deserialize(data: SerializedRLData, batched: bool = False) -> RLData:
    """Deserialize a base64-encoded representation back to RL data.

    :param data: The serialized RL data to deserialize.
    :type data: SerializedRLData
    :param batched: Whether the data is batched.
    :type batched: bool
    :return: The deserialized RL data.
    :rtype: RLData
    """
    if data is None:
        return None
    if isinstance(data, str):
        return _decode_array(data, batched)
    if isinstance(data, dict):
        return {k: deserialize(v, batched) for k, v in data.items()}
    if isinstance(data, (tuple, list)):
        return tuple(deserialize(v, batched) for v in data)
    return _decode_array(data, batched)


def get_batch_size(observation: RLData) -> int:
    """Extract batch size from the first leaf array in an observation.

    :param observation: The observation to get the batch size from.
    :type observation: RLData
    :return: The batch size.
    :rtype: int
    :raises ValueError: If the first leaf is ``None`` rather than an array.
    """
    while not isinstance(observation, np.ndarray):
        if observation is None:
            msg = "Cannot infer a batch size from a None observation."
            raise ValueError(msg)
        if isinstance(observation, dict):
            observation = next(iter(observation.values()))
        else:
            observation = observation[0]
    return observation.shape[0]
