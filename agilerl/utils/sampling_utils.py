from collections.abc import Callable
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


def select_batch_idxs(x: torch.Tensor, idxs: torch.Tensor) -> torch.Tensor:
    return torch.gather(
        x,
        dim=0,
        index=idxs.repeat(*x.shape[1:], 1).permute(
            len(x.shape) - 1,
            *list(range(len(x.shape) - 1)),
        ),
    )


def map_all_kvs(
    f: Callable[..., Any],
    kvs: tuple[tuple[Any, ...], ...],
) -> tuple[tuple[Any, ...], ...]:
    return tuple([tuple(map(f, items)) for items in kvs])


def map_decoder_kvs(
    f: Callable[..., Any],
    kvs: tuple[tuple[Any, ...], ...],
) -> tuple[tuple[Any, ...], ...]:
    return tuple([(tuple(map(f, items[:2])) + tuple(items[2:])) for items in kvs])


def pad_sequence(
    seq: torch.Tensor,
    to_len: int,
    val: float,
    device: torch.device | str,
    dim: int,
) -> torch.Tensor:
    return torch.cat(
        (
            seq,
            torch.full(
                (*seq.shape[:dim], to_len - seq.shape[dim], *seq.shape[dim + 1 :]),
                val,
            ).to(device),
        ),
        dim=dim,
    )


def update_kvs(
    kvs: Any,
    updated_kvs: Any,
    lens_chosen: torch.Tensor,
    idx: int,
) -> Any:
    for i, layer in enumerate(kvs):
        for x, item in enumerate(layer):
            item[lens_chosen, :, idx, :] = updated_kvs[i][x][:, :, idx, :]
    return kvs


def update_decoder_kvs(
    kvs: Any,
    updated_kvs: Any,
    lens_chosen: torch.Tensor,
    idx: int,
) -> Any:
    for i, layer in enumerate(kvs):
        for x, item in enumerate(layer[:2]):
            item[lens_chosen, :, idx, :] = updated_kvs[i][x][:, :, idx, :]
    return kvs


def get_relevant_kvs(
    kvs: Any,
    lens_chosen: torch.Tensor,
    idx: int,
) -> tuple[tuple[Any, ...], ...]:
    kvs = map_all_kvs(lambda x: select_batch_idxs(x, lens_chosen), kvs)
    return map_all_kvs(lambda x: x[:, :, :idx, :], kvs)


def top_k_logits(logits: torch.Tensor, k: int) -> torch.Tensor:
    # logits = (batch, time, dim)
    _, bottom_k_idx = torch.topk(-logits, logits.shape[2] - k, dim=2)
    return torch.scatter(logits, dim=2, index=bottom_k_idx, value=float("-inf"))


def top_p_logits(logits: torch.Tensor, p: float) -> torch.Tensor:
    # logits = (batch, time, dim)
    sorted_logits, _ = torch.sort(logits, dim=2, descending=True)
    num_to_take = torch.sum(
        torch.cumsum(F.softmax(sorted_logits, dim=2), dim=2) <= p,
        dim=2,
    ).unsqueeze(2)
    mask = logits < torch.gather(
        sorted_logits,
        dim=2,
        index=torch.clamp(num_to_take, max=logits.shape[2] - 1),
    )
    return logits.masked_fill(mask, float("-inf"))


def process_logits(
    logits: torch.Tensor,
    temp: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
) -> torch.Tensor:
    logits /= temp
    if top_k is not None:
        logits = top_k_logits(logits, top_k)
    if top_p is not None:
        logits = top_p_logits(logits, top_p)
    return logits


def always_terminate(s: np.ndarray) -> bool:
    return True
