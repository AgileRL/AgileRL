# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Re-tie embeddings and fill RoPE buffers after FSDP ``to_empty``."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Protocol, cast, runtime_checkable

import torch
from torch import nn
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS


@runtime_checkable
class ModelWithTiedWeightKeys(Protocol):
    all_tied_weights_keys: Mapping[str, str]


def restore_after_to_empty(model: nn.Module) -> None:
    """Re-tie input/output embeddings, which ``to_empty`` unties.

    HuggingFace ``tie_weights`` on the head's owner also ties its submodules.

    :raises RuntimeError: If a tied parameter is still separate from its
        source, which the safetensors load would leave empty.
    """
    owner = None
    for module in nn.Module.modules(model):
        # PEFT __getattr__ forwards lm_head; ownership is the registered child.
        children = dict(module.named_children())
        if isinstance(children.get("lm_head"), nn.Module) or isinstance(
            children.get("embed_out"), nn.Module
        ):
            owner = module
            break
    if owner is None:
        return
    tie = getattr(owner, "tie_weights", None)
    if callable(tie):
        tie()
    if not isinstance(owner, ModelWithTiedWeightKeys):
        return
    for target, source in owner.all_tied_weights_keys.items():
        if owner.get_parameter(target) is not owner.get_parameter(source):
            msg = (
                f"{type(owner).__name__}.{target} is not tied to {source} "
                "after tie_weights()"
            )
            raise RuntimeError(msg)


def init_rope_buffers(model: nn.Module) -> None:
    """Fill RoPE frequency buffers after a meta-device safetensors load.

    ``inv_freq`` and ``original_inv_freq`` are non-persistent, so they are
    not in the checkpoint. ``to_empty`` allocates them uninitialised. Needed
    when the model was built on meta and loaded from safetensors (FSDP ranks
    sharing one checkpoint). Same computation as HuggingFace ``_init_weights``.
    """
    with torch.no_grad():
        for module in nn.Module.modules(model):
            if "RotaryEmbedding" not in type(module).__name__ or not hasattr(
                module, "original_inv_freq"
            ):
                continue
            rope_type = cast("str", module.rope_type)
            rope_fn = cast(
                "Callable[..., tuple[torch.Tensor, object]]",
                module.compute_default_rope_parameters
                if rope_type == "default"
                else ROPE_INIT_FUNCTIONS[rope_type],
            )
            inv_freq_buf = cast("torch.Tensor", module.inv_freq)
            original_buf = cast("torch.Tensor", module.original_inv_freq)
            inv_freq, _ = rope_fn(module.config, inv_freq_buf.device)
            inv_freq_buf.copy_(inv_freq)
            original_buf.copy_(inv_freq)
