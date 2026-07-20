"""Base64 ``.npy`` serialization for Arena RL and supervised inference payloads."""

from __future__ import annotations

import base64
import io
from typing import TypeAlias

import numpy as np

RLData: TypeAlias = np.ndarray | dict[str, "RLData"] | tuple["RLData", ...]
SerializedRLData: TypeAlias = (
    str | dict[str, "SerializedRLData"] | tuple["SerializedRLData", ...] | None
)


def serialize(data: RLData | None, batched: bool = False) -> SerializedRLData:
    """Serialize RL data to a base64-encoded ``.npy`` representation.

    :param data: The RL data to serialize.
    :type data: RLData
    :param batched: Whether the data is batched.
    :type batched: bool
    :return: The serialized RL data.
    :rtype: SerializedRLData
    """
    if isinstance(data, dict):
        # Recursive alias narrowing diverges in ty; the dict arm is dict[str, RLData].
        return {k: serialize(v, batched) for k, v in data.items()}  # ty: ignore[invalid-return-type, invalid-argument-type]
    if isinstance(data, (tuple, list)):
        return tuple(serialize(v, batched) for v in data)
    if data is None:
        return None

    if not batched:
        data = np.expand_dims(data, axis=0)

    buffer = io.BytesIO()
    np.save(buffer, data, allow_pickle=False)
    buffer.seek(0)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def deserialize(data: SerializedRLData, batched: bool = False) -> RLData | None:
    """Deserialize a base64-encoded representation back to RL data.

    :param data: The serialized RL data to deserialize.
    :type data: SerializedRLData
    :param batched: Whether the data is batched.
    :type batched: bool
    :return: The deserialized RL data.
    :rtype: RLData
    """
    if isinstance(data, dict):
        return {k: deserialize(v, batched) for k, v in data.items()}
    if isinstance(data, (tuple, list)):
        return tuple(deserialize(v, batched) for v in data)
    if data is None:
        return None

    decoded = base64.b64decode(data, validate=True)
    arr: np.ndarray = np.load(io.BytesIO(decoded), allow_pickle=False)
    return arr if batched else arr.squeeze(axis=0)


def get_batch_size(observation: RLData) -> int:
    """Extract batch size from the first leaf array in an observation.

    :param observation: The observation to get the batch size from.
    :type observation: RLData
    :return: The batch size.
    :rtype: int
    """
    while isinstance(observation, (dict, tuple)):
        if isinstance(observation, dict):
            # Recursive alias narrowing diverges in ty; values are RLData.
            observation = next(iter(observation.values()))  # ty: ignore[invalid-assignment]
        else:
            observation = observation[0]
    return observation.shape[0]
