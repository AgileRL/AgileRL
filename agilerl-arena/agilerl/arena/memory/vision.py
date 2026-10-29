# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0
"""Closed-form vision-tower memory terms: image patch grids, pixel rows, feature cache and encoder profile.

Every function is a pure function of the specs in
:mod:`agilerl.arena.memory.specs`. No torch, no I/O.
"""

from __future__ import annotations

import math

from agilerl.arena.memory.specs import ModelArch


def encoder_profile_items(
    arch: ModelArch,
    max_num_batched_tokens: int,
    max_num_seqs: int,
    max_model_len: int,
    max_images_per_prompt: int | None,
) -> int:
    """Maximum-size images vLLM encodes in its profile pass (``MultiModalBudget``).

    :param arch: Model architecture.
    :param max_num_batched_tokens: The engine's scheduler-step token cap.
    :param max_num_seqs: The engine's concurrent sequences.
    :param max_model_len: The engine's context length.
    :param max_images_per_prompt: ``limit_mm_per_prompt["image"]``; ``None``
        takes vLLM's default for an unset modality, 0 disables images.
    :return: Images in the profile batch; 0 without a profile image.
    """
    tokens = arch.vision_profile_patches // arch.vision_spatial_merge_size**2
    if not tokens or max_images_per_prompt == 0:
        return 0
    # vLLM allows 999 items of a modality the config leaves unset.
    limit = 999 if max_images_per_prompt is None else max_images_per_prompt
    # The encoder budget is max_num_batched_tokens, raised to fit one image.
    by_encoder = max(max_num_batched_tokens, tokens) // tokens
    per_prompt = max(1, min(limit, max_model_len // tokens))
    return max(1, min(by_encoder, max_num_seqs * per_prompt))


def encoder_profile_bytes(
    arch: ModelArch, act_bytes: float, tensor_parallel: int, items: int
) -> tuple[int, int]:
    """Vision-encoder transient and merged-output bytes of vLLM's profile batch.

    :param arch: Model architecture.
    :param act_bytes: Bytes per activation element.
    :param tensor_parallel: Engine tensor-parallel width.
    :param items: Images in the batch, from :func:`encoder_profile_items`.
    :return: (encoder transient, merged outputs held through the decoder pass).
    """
    patches = arch.vision_profile_patches * items
    if not patches:
        return 0, 0
    vh = arch.vision_hidden_size
    pixels = arch.vision_in_channels * arch.vision_temporal_patch_size
    pixels *= arch.vision_patch_size**2
    # Pixel input plus one encoder layer's live activations per patch; q/k/v
    # and the MLP intermediate shard over TP. Fit to Qwen3.6-35B-A3B on A100
    # 80GB under vLLM 0.32, one image: 1.67 GiB at TP=1, 1.24 GiB at TP=2.
    per_patch = (
        pixels + 4 * vh + (3 * vh + arch.vision_intermediate_size) // tensor_parallel
    )
    merged = patches // arch.vision_spatial_merge_size**2
    outputs = merged * (arch.vision_out_hidden_size or arch.hidden_size)
    return int(patches * per_patch * act_bytes), int(outputs * act_bytes)


def _merge_grid(arch: ModelArch, width: int, height: int) -> tuple[int, int]:
    """Qwen2-VL ``smart_resize``: sides on a patch x merge grid within its pixel bounds."""
    # The Qwen3.5/3.6 processors' size.shortest_edge and size.longest_edge.
    min_pixels, max_pixels = 65536, 16777216
    factor = arch.vision_patch_size * arch.vision_spatial_merge_size
    h = round(height / factor) * factor
    w = round(width / factor) * factor
    if h * w > max_pixels:
        beta = math.sqrt(height * width / max_pixels)
        h = max(factor, math.floor(height / beta / factor) * factor)
        w = max(factor, math.floor(width / beta / factor) * factor)
    elif h * w < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h = math.ceil(height * beta / factor) * factor
        w = math.ceil(width * beta / factor) * factor
    return h // arch.vision_patch_size, w // arch.vision_patch_size


def _pooled_budget(arch: ModelArch, width: int, height: int) -> tuple[int, int]:
    """Gemma4: the largest aspect-preserving pool-aligned grid under its patch budget."""
    patch = arch.vision_patch_size
    pool = arch.vision_spatial_merge_size
    side = patch * pool
    factor = math.sqrt(arch.vision_max_patches * patch**2 / (height * width))
    h = math.floor(factor * height / side) * side
    w = math.floor(factor * width / side) * side
    longest = (arch.vision_max_patches // pool**2) * side
    if h == 0:
        h, w = side, min(math.floor(width / height) * side, longest)
    elif w == 0:
        h, w = min(math.floor(height / width) * side, longest), side
    return h // patch, w // patch


def _dynamic_tiles(arch: ModelArch, width: int, height: int) -> tuple[int, int]:
    """Nemotron RADIO dynamic resolution, one tile per image.

    The processor's token budget (its max_model_len x pixel-shuffle factor^2)
    exceeds ``max_num_patches``, so the per-image budget is ``max_num_patches``.
    """
    patch = arch.vision_patch_size
    budget = arch.vision_max_patches
    floor_patches = arch.vision_min_patches
    divisor = arch.vision_spatial_merge_size
    # Mirrors the processor's round(x + 0.5), which rounds half to even.
    closest_h = round(height / patch + 0.5)
    closest_w = round(width / patch + 0.5)
    factor = min(math.sqrt(budget / (closest_h * closest_w)), 1.0)
    h = math.floor(factor * closest_h)
    w = math.floor(factor * closest_w)
    if budget > floor_patches and h * w < floor_patches:
        up = math.sqrt(floor_patches / (h * w))
        h, w = math.ceil(up * h), math.ceil(up * w)
    if h % divisor:
        up_h = h + divisor - h % divisor
        h = up_h if up_h * w <= budget else max(divisor, h - h % divisor)
    if w % divisor:
        up_w = w + divisor - w % divisor
        w = up_w if h * up_w <= budget else max(divisor, w - w % divisor)
    return h, w


def vision_patch_grid(arch: ModelArch, width: int, height: int) -> tuple[int, int]:
    """Patch rows and columns the family's image processor cuts from one image.

    :param arch: Model architecture.
    :param width: Image width in pixels.
    :param height: Image height in pixels.
    :return: ``(rows, cols)``; ``(0, 0)`` without an image or a known rule.
    """
    if not (width and height and arch.vision_patch_size):
        return 0, 0
    if arch.vision_resize == "merge_grid":
        return _merge_grid(arch, width, height)
    if arch.vision_resize == "pooled_budget":
        return _pooled_budget(arch, width, height)
    if arch.vision_resize == "dynamic_tiles":
        return _dynamic_tiles(arch, width, height)
    return 0, 0


def vision_pixel_bytes(arch: ModelArch, patches: int) -> int:
    """Host bytes of one image's fp32 pixel row.

    :param arch: Model architecture.
    :param patches: Patches in the image's grid.
    :return: ``patches x channels x temporal x patch^2 x 4``.
    """
    patch_dim = (
        arch.vision_in_channels
        * arch.vision_temporal_patch_size
        * arch.vision_patch_size**2
    )
    return patches * patch_dim * 4


def vision_feature_cache_bytes(arch: ModelArch, patches: int, act_bytes: float) -> int:
    """Pinned host bytes VisionFeatureCache keeps for one image's tower outputs.

    :param arch: Model architecture.
    :param patches: Patches in the image's grid.
    :param act_bytes: Bytes per output element.
    :return: Bytes of every output's pinned block.
    """
    tokens = patches // arch.vision_spatial_merge_size**2
    # A patch-merger tower returns its patch sequence beside the merged tokens;
    # a pooling tower returns only its pooled tokens at the tower width.
    if arch.vision_out_hidden_size:
        outputs = [
            patches * arch.vision_hidden_size,
            tokens * arch.vision_out_hidden_size,
        ]
    else:
        outputs = [tokens * arch.vision_hidden_size]
    return sum(pinned_block_bytes(int(n * act_bytes)) for n in outputs)


def pinned_block_bytes(n_bytes: int) -> int:
    """Host bytes torch's pinned allocator reserves for an ``n_bytes`` tensor: the next power of two.

    :param n_bytes: Tensor bytes.
    :return: Reserved bytes.
    """
    return 1 << (n_bytes - 1).bit_length()


def vision_tower_lora_bytes(
    arch: ModelArch,
    patches: int,
    act_bytes: float,
    adapter_bytes: float,
    lora_dropout: bool,
) -> int:
    """Tower tensors a LoRA-trained vision tower keeps for one micro-batch's backward.

    :param arch: Model architecture.
    :param patches: Patches in the micro-batch's images.
    :param act_bytes: Bytes per activation element.
    :param adapter_bytes: Bytes per adapter element.
    :param lora_dropout: ``lora_dropout > 0``.
    :return: Bytes held for the backward.
    """
    patch_dim = (
        arch.vision_in_channels
        * arch.vision_temporal_patch_size
        * arch.vision_patch_size**2
    )
    # The checkpointed encoder keeps each block's input and the cast pixels.
    held = patches * (arch.vision_n_layers * arch.vision_hidden_size + patch_dim)
    held *= act_bytes
    # A patch merger also keeps the tower output and its LoRA output, plus,
    # with dropout, its two linears' dropped-out inputs and masks.
    if arch.vision_out_hidden_size:
        held += patches * 2 * arch.vision_hidden_size * act_bytes
        if lora_dropout:
            held += patches * 2 * arch.vision_hidden_size * (adapter_bytes + 1)
    return int(held)
