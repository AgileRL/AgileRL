# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""MiMo-V2 vision-language model: the vision tower's tokens replace ``<|image_pad|>`` embeddings of the native language model.

The language model keeps 1-D positions, as vLLM's ``MiMoV2OmniForCausalLM``
does. Pixels arrive as ``(images, h, w, patch_dim)`` patch-grid rows, one grid
per call.

The vision tower is computed as vLLM 0.32's ``MiMoVisionTransformer``.
Patches embed with a Conv3d, then pre-norm blocks of grouped-query attention
with 2-D rotary positions and a gated SiLU MLP. Blocks in
``fullatt_block_indexes`` attend over each image. The others attend to the
patches within ``visual_token_window_size`` of the query, with each head's
sink added to the logit of the image's first patch. A run of column-window
blocks sees the patches in column-major merge-window order. The merger
RMS-normalizes, folds each merge window, and projects to the language model.
"""

from __future__ import annotations

from typing import Any, ClassVar, cast

import torch
import torch.nn.functional as F
from torch import nn
from transformers import GenerationMixin, PreTrainedModel
from transformers import initialization as init
from transformers.cache_utils import Cache
from transformers.modeling_outputs import CausalLMOutputWithPast, MoeModelOutputWithPast
from transformers.models.mimo_v2_flash.modeling_mimo_v2_flash import (
    MiMoV2FlashModel,
)

from agilerl.architectures.mimo_v2.configuration import (
    MiMoV2Config,
    MiMoV2TextConfig,
    MiMoV2VisionConfig,
)
from agilerl.architectures.vision_rows import packed_patches

__all__ = [
    "MiMoV2ForConditionalGeneration",
    "MiMoV2Model",
    "MiMoV2VisionBlock",
    "MiMoV2VisionModel",
]

# Query rows per window-attention call; keeps the mask at heads x 512 x 640.
WINDOW_QUERY_CHUNK = 512


class MiMoV2VisionRMSNorm(nn.Module):
    """RMSNorm in fp32, cast back before the weight multiply, as vLLM's kernel."""

    def __init__(self, hidden_size: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Normalize the last dim.

        :param hidden_states: ``(..., hidden)`` inputs.
        :return: Normalized ``hidden_states`` in its dtype.
        """
        dtype = hidden_states.dtype
        states = hidden_states.float()
        states = states * torch.rsqrt(states.pow(2).mean(-1, keepdim=True) + self.eps)
        return states.to(dtype) * self.weight


class MiMoV2VisionPatchEmbed(nn.Module):
    """Conv3d patch embedding; kernel equals stride, so one matmul per patch."""

    def __init__(self, config: MiMoV2VisionConfig) -> None:
        super().__init__()
        kernel = (config.temporal_patch_size, config.patch_size, config.patch_size)
        self.proj = nn.Conv3d(
            config.in_chans, config.hidden_size, kernel, stride=kernel, bias=False
        )

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        """Embed flattened ``(channels, frames, patch, patch)`` patches.

        :param patches: ``(patches, patch_dim)`` pixels.
        :return: ``(patches, hidden)`` embeddings.
        """
        weight = self.proj.weight
        return F.linear(patches.to(weight.dtype), weight.flatten(1))


class MiMoV2VisionAttention(nn.Module):
    """Grouped-query attention over each image, whole or within a window."""

    def __init__(self, config: MiMoV2VisionConfig, use_sink: bool) -> None:
        super().__init__()
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_key_value_heads
        self.qk_channels = config.qk_channels
        self.kv_channels = config.kv_channels
        self.window = config.visual_token_window_size
        self.qkv = nn.Linear(
            config.hidden_size,
            config.num_heads * config.qk_channels
            + config.num_key_value_heads * (config.qk_channels + config.kv_channels),
        )
        self.proj = nn.Linear(config.num_heads * config.kv_channels, config.hidden_size)
        self.sinks = nn.Parameter(torch.zeros(config.num_heads)) if use_sink else None

    def forward(
        self,
        hidden_states: torch.Tensor,
        seq_lens: list[int],
        cos: torch.Tensor,
        sin: torch.Tensor,
        full_attention: bool,
    ) -> torch.Tensor:
        """Attend within each image.

        :param hidden_states: ``(patches, hidden)`` of every image, concatenated.
        :param seq_lens: Patches of each image.
        :param cos: ``(patches, qk_channels / 2)`` rotary cosines.
        :param sin: ``(patches, qk_channels / 2)`` rotary sines.
        :param full_attention: Whether each patch attends over its whole image.
        :return: ``(patches, hidden)`` outputs.
        """
        # Head counts from the weights, so a tensor-parallel shard runs only its heads.
        num_heads = self.proj.weight.shape[1] // self.kv_channels
        groups = self.num_heads // self.num_kv_heads
        num_kv_heads = num_heads // groups
        q, k, v = self.qkv(hidden_states).split(
            [
                num_heads * self.qk_channels,
                num_kv_heads * self.qk_channels,
                num_kv_heads * self.kv_channels,
            ],
            dim=-1,
        )
        q = rotate(q.unflatten(-1, (num_heads, -1)), cos, sin).transpose(0, 1)
        k = rotate(k.unflatten(-1, (num_kv_heads, -1)), cos, sin).transpose(0, 1)
        v = v.unflatten(-1, (num_kv_heads, -1)).transpose(0, 1)
        k = k.repeat_interleave(groups, dim=0)
        v = v.repeat_interleave(groups, dim=0)
        scale = self.qk_channels**-0.5
        outputs = []
        for image_q, image_k, image_v in zip(
            q.split(seq_lens, dim=1),
            k.split(seq_lens, dim=1),
            v.split(seq_lens, dim=1),
            strict=True,
        ):
            if full_attention:
                outputs.append(
                    F.scaled_dot_product_attention(
                        image_q, image_k, image_v, scale=scale
                    )
                )
            else:
                outputs.append(
                    window_attention(
                        image_q, image_k, image_v, self.window, self.sinks, scale
                    )
                )
        attended = torch.cat(outputs, dim=1).transpose(0, 1).flatten(1)
        return self.proj(attended)


def rotate(states: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Neox rotation of whole heads by per-patch angles.

    :param states: ``(patches, heads, head_dim)``.
    :param cos: ``(patches, head_dim / 2)`` cosines.
    :param sin: ``(patches, head_dim / 2)`` sines.
    :return: Rotated ``states``.
    """
    cos = cos.unsqueeze(-2).to(states.dtype)
    sin = sin.unsqueeze(-2).to(states.dtype)
    first, second = states.chunk(2, dim=-1)
    return torch.cat([first * cos - second * sin, second * cos + first * sin], dim=-1)


def window_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    window: int,
    sinks: torch.Tensor | None,
    scale: float,
) -> torch.Tensor:
    """Bidirectional attention to keys at most ``window`` patches away; ``sinks`` bias key 0.

    :param q: ``(heads, patches, head_dim)`` queries of one image.
    :param k: ``(heads, patches, head_dim)`` keys.
    :param v: ``(heads, patches, head_dim)`` values.
    :param window: Largest query-key distance attended.
    :param sinks: ``(heads,)`` logit bias on key 0, or None.
    :param scale: Softmax scale.
    :return: ``(heads, patches, head_dim)`` outputs.
    """
    patches = q.shape[1]
    positions = torch.arange(patches, device=q.device)
    outputs = []
    for start in range(0, patches, WINDOW_QUERY_CHUNK):
        stop = min(patches, start + WINDOW_QUERY_CHUNK)
        key_start = max(0, start - window)
        key_stop = min(patches, stop + window)
        distance = positions[start:stop, None] - positions[None, key_start:key_stop]
        allowed = distance.abs() <= window
        bias = torch.zeros(
            q.shape[0], *allowed.shape, dtype=q.dtype, device=q.device
        ).masked_fill(~allowed, float("-inf"))
        if sinks is not None and key_start == 0:
            bias[:, :, 0] += sinks.to(q.dtype)[:, None] * allowed[:, 0]
        outputs.append(
            F.scaled_dot_product_attention(
                q[:, start:stop],
                k[:, key_start:key_stop],
                v[:, key_start:key_stop],
                attn_mask=bias,
                scale=scale,
            )
        )
    return torch.cat(outputs, dim=1)


class MiMoV2VisionMLP(nn.Module):
    """``down(silu(gate(x)) * up(x))`` with biases."""

    def __init__(self, config: MiMoV2VisionConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Apply the gated MLP.

        :param hidden_states: ``(patches, hidden)`` inputs.
        :return: ``(patches, hidden)`` outputs.
        """
        gated = F.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states)
        return self.down_proj(gated)


class MiMoV2VisionBlock(nn.Module):
    """Pre-norm attention and MLP with residuals."""

    def __init__(self, config: MiMoV2VisionConfig, index: int) -> None:
        super().__init__()
        self.full_attention = index in cast("list[int]", config.fullatt_block_indexes)
        self.norm1 = MiMoV2VisionRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.norm2 = MiMoV2VisionRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.attn = MiMoV2VisionAttention(
            config, use_sink=config.use_sink and not self.full_attention
        )
        self.mlp = MiMoV2VisionMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        seq_lens: list[int],
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        """Run one block.

        :param hidden_states: ``(patches, hidden)`` inputs.
        :param seq_lens: Patches of each image.
        :param cos: Rotary cosines in the block's patch order.
        :param sin: Rotary sines in the block's patch order.
        :return: ``(patches, hidden)`` outputs.
        """
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states), seq_lens, cos, sin, self.full_attention
        )
        return hidden_states + self.mlp(self.norm2(hidden_states))


class MiMoV2VisionMerger(nn.Module):
    """RMSNorm, fold each merge window into one row, then a GELU MLP."""

    def __init__(self, config: MiMoV2VisionConfig) -> None:
        super().__init__()
        folded = config.hidden_size * config.spatial_merge_size**2
        self.ln_q = MiMoV2VisionRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = nn.Sequential(
            nn.Linear(folded, folded, bias=False),
            nn.GELU(),
            nn.Linear(folded, config.out_hidden_size, bias=False),
        )
        self.folded = folded

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Merge patches into image tokens.

        :param hidden_states: ``(patches, hidden)`` in merge-window order.
        :return: ``(patches / merge**2, out_hidden_size)`` tokens.
        """
        return self.mlp(self.ln_q(hidden_states).view(-1, self.folded))


class MiMoV2VisionModel(PreTrainedModel):
    """MiMo-V2 vision tower from patch rows to language-model image tokens."""

    config: MiMoV2VisionConfig
    config_class = MiMoV2VisionConfig
    base_model_prefix = "visual"
    main_input_name = "pixel_values"
    _no_split_modules: ClassVar[list[str]] = ["MiMoV2VisionBlock"]
    # The tower runs its own attention; any backend the language model picks is fine.
    _supports_sdpa = True
    _supports_flash_attn = True
    _supports_flex_attn = True
    _supports_attention_backend = True

    def __init__(self, config: MiMoV2VisionConfig) -> None:
        super().__init__(config)
        self.patch_embed = MiMoV2VisionPatchEmbed(config)
        self.blocks = nn.ModuleList(
            [MiMoV2VisionBlock(config, index) for index in range(config.depth)]
        )
        self.merger = MiMoV2VisionMerger(config)
        self.post_init()

    @torch.no_grad()
    def _init_weights(self, module: nn.Module) -> None:
        super()._init_weights(module)
        if isinstance(module, MiMoV2VisionAttention) and module.sinks is not None:
            init.zeros_(module.sinks)

    def rotary_angles(
        self, grid_thw: torch.Tensor, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """2-D rotary cosines and sines of each patch, in the processor's merge-window order.

        :param grid_thw: ``(images, 3)`` patch grids.
        :param device: Device of the angles.
        :return: ``(patches, qk_channels / 2)`` cosines and sines.
        """
        merge = self.config.spatial_merge_size
        rotary_dim = self.config.qk_channels // 2
        inv_freq = 1.0 / (
            10000.0
            ** (
                torch.arange(0, rotary_dim, 2, device=device, dtype=torch.float32)
                / rotary_dim
            )
        )
        positions = []
        for frames, height, width in grid_thw.tolist():
            rows = torch.arange(height, device=device)[:, None].expand(height, width)
            cols = torch.arange(width, device=device)[None, :].expand(height, width)
            pair = torch.stack([rows, cols], dim=-1)
            pair = pair.reshape(height // merge, merge, width // merge, merge, 2)
            positions.append(
                pair.permute(0, 2, 1, 3, 4).reshape(-1, 2).repeat(frames, 1)
            )
        angles = (torch.cat(positions)[..., None] * inv_freq).flatten(1)
        return angles.cos(), angles.sin()

    def column_order(
        self, grid_thw: torch.Tensor, device: torch.device
    ) -> torch.Tensor:
        """Merge-window index of each position in column-major window order.

        :param grid_thw: ``(images, 3)`` patch grids.
        :param device: Device of the index.
        :return: ``(patches / merge**2,)`` window indices.
        """
        merge = self.config.spatial_merge_size
        orders = []
        offset = 0
        for frames, height, width in grid_thw.tolist():
            windows = frames * (height // merge) * (width // merge)
            index = torch.arange(windows, device=device).view(
                frames, height // merge, width // merge
            )
            orders.append(index.transpose(1, 2).reshape(-1) + offset)
            offset += windows
        return torch.cat(orders)

    def reorder(self, states: torch.Tensor, window_order: torch.Tensor) -> torch.Tensor:
        """Permute ``states`` by whole merge windows.

        :param states: ``(patches, ...)`` per-patch tensor.
        :param window_order: Merge-window permutation.
        :return: Permuted ``states``.
        """
        unit = self.config.spatial_merge_size**2
        return states.unflatten(0, (-1, unit))[window_order].flatten(0, 1)

    def forward(
        self, pixel_values: torch.Tensor, grid_thw: torch.Tensor
    ) -> torch.Tensor:
        """Encode packed patches into image tokens.

        :param pixel_values: ``(patches, patch_dim)`` patches of every image, in
            the processor's merge-window order.
        :param grid_thw: ``(images, 3)`` frame, row and column patch counts.
        :return: ``(patches / merge**2, out_hidden_size)`` image tokens.
        """
        device = pixel_values.device
        hidden_states = self.patch_embed(pixel_values)
        cos, sin = self.rotary_angles(grid_thw, device)
        column = self.column_order(grid_thw, device)
        restore = torch.argsort(column)
        column_cos = self.reorder(cos, column)
        column_sin = self.reorder(sin, column)
        seq_lens = (
            (grid_thw[:, 1] * grid_thw[:, 2]).repeat_interleave(grid_thw[:, 0]).tolist()
        )
        window_types = cast("list[int]", self.config.vit_window_attn_types)
        for index, block in enumerate(self.blocks):
            is_column = window_types[index] == 1
            was_column = index > 0 and window_types[index - 1] == 1
            if is_column and not was_column:
                hidden_states = self.reorder(hidden_states, column)
            elif was_column and not is_column:
                hidden_states = self.reorder(hidden_states, restore)
            hidden_states = block(
                hidden_states,
                seq_lens,
                column_cos if is_column else cos,
                column_sin if is_column else sin,
            )
        if window_types[-1] == 1:
            hidden_states = self.reorder(hidden_states, restore)
        return self.merger(hidden_states)


class MiMoV2PreTrainedModel(PreTrainedModel):
    """Shared loading settings of the MiMo-V2 vision-language classes."""

    config: MiMoV2Config
    config_class = MiMoV2Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules: ClassVar[list[str]] = [
        "MiMoV2FlashDecoderLayer",
        "MiMoV2VisionBlock",
    ]
    _skip_keys_device_placement: ClassVar[list[str]] = ["past_key_values"]
    # The language model's backends; the vision tower runs its own SDPA.
    _supports_flash_attn = True
    _supports_sdpa = False
    _supports_flex_attn = True
    _supports_attention_backend = True
    _compatible_flash_implementations: ClassVar[list[str]] = ["flash_attention_4"]
    _keep_in_fp32_modules_strict: ClassVar[list[str]] = ["e_score_correction_bias"]
    _keys_to_ignore_on_load_unexpected: ClassVar[list[str]] = [
        r"^model\.mtp\.",
        r"^audio_encoder\.",
        r"^speech_embeddings\.",
    ]


class MiMoV2Model(MiMoV2PreTrainedModel):
    """Vision tower and language model; image tokens fill the ``image_token_id`` positions."""

    def __init__(self, config: MiMoV2Config) -> None:
        super().__init__(config)
        self.visual = MiMoV2VisionModel._from_config(config.vision_config)
        self.language_model = MiMoV2FlashModel._from_config(config.text_config)
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        """Token embeddings of the language model."""
        return self.language_model.embed_tokens

    def set_input_embeddings(self, value: nn.Module) -> None:
        """Replace the language model's token embeddings."""
        self.language_model.embed_tokens = value

    def get_image_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Image tokens of ``(images, h, w, patch_dim)`` patch-grid rows.

        :param pixel_values: One patch grid per image, all of one shape.
        :return: ``(images * h * w / merge**2, hidden)`` tokens in image order.
        """
        patches, image_grid_hw = packed_patches(pixel_values)
        return self.visual(patches, F.pad(image_grid_hw, (1, 0), value=1))

    def vision_filler_features(self, device: torch.device) -> torch.Tensor:
        """Image tokens of one zero image on the smallest grid the merger takes.

        :param device: Device of the zero image.
        :return: The tower's tokens.
        """
        vision = cast("MiMoV2VisionConfig", self.config.vision_config)
        side = vision.spatial_merge_size
        patch_dim = vision.in_chans * vision.temporal_patch_size * vision.patch_size**2
        return self.get_image_features(
            torch.zeros(1, side, side, patch_dim, device=device)
        )

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        pixel_values: torch.Tensor | None = None,
        use_cache: bool | None = None,
        **kwargs: Any,
    ) -> MoeModelOutputWithPast:
        """Embed tokens, place image tokens, and run the language model.

        :param input_ids: ``(batch, seq)`` token ids.
        :param attention_mask: Attention mask for the language model.
        :param position_ids: 1-D positions.
        :param past_key_values: Decoding cache.
        :param inputs_embeds: Embeddings instead of ``input_ids``.
        :param pixel_values: ``(images, h, w, patch_dim)`` patch-grid rows.
        :param use_cache: Whether to return the cache.
        :return: The language model's output.
        :raises ValueError: If ``pixel_values`` come without ``input_ids``, or
            the image tokens do not fill the image positions.
        """
        embeds = (
            self.get_input_embeddings()(input_ids)
            if inputs_embeds is None
            else inputs_embeds
        )
        if pixel_values is not None:
            if input_ids is None:
                msg = "pixel_values need input_ids to place the images"
                raise ValueError(msg)
            image_tokens = self.get_image_features(pixel_values).to(
                embeds.device, embeds.dtype
            )
            is_image = input_ids == self.config.image_token_id
            n_positions = int(is_image.sum())
            if n_positions != image_tokens.shape[0]:
                msg = (
                    f"{n_positions} image positions for "
                    f"{image_tokens.shape[0]} image tokens"
                )
                raise ValueError(msg)
            embeds = embeds.masked_scatter(
                is_image.unsqueeze(-1).expand_as(embeds), image_tokens
            )
        return self.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=embeds,
            use_cache=use_cache,
            **kwargs,
        )


class MiMoV2ForConditionalGeneration(MiMoV2PreTrainedModel, GenerationMixin):
    """MiMo-V2 vision-language model and LM head; ``tie_word_embeddings`` ties the head to the token embeddings."""

    _tied_weights_keys: ClassVar[dict[str, str]] = {
        "lm_head.weight": "model.language_model.embed_tokens.weight"
    }

    def __init__(self, config: MiMoV2Config) -> None:
        super().__init__(config)
        self.model = MiMoV2Model(config)
        text = cast("MiMoV2TextConfig", config.text_config)
        self.lm_head = nn.Linear(text.hidden_size, text.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        """Token embeddings of the language model."""
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value: nn.Module) -> None:
        """Replace the language model's token embeddings."""
        self.model.set_input_embeddings(value)

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        pixel_values: torch.Tensor | None = None,
        use_cache: bool | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs: Any,
    ) -> CausalLMOutputWithPast:
        """Logits of the last ``logits_to_keep`` positions, or all when 0.

        :param input_ids: ``(batch, seq)`` token ids.
        :param attention_mask: Attention mask for the language model.
        :param position_ids: 1-D positions.
        :param past_key_values: Decoding cache.
        :param inputs_embeds: Embeddings instead of ``input_ids``.
        :param pixel_values: ``(images, h, w, patch_dim)`` patch-grid rows.
        :param use_cache: Whether to return the cache.
        :param logits_to_keep: Trailing positions to score, or an index tensor.
        :return: Logits and the cache.
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            pixel_values=pixel_values,
            use_cache=use_cache,
            **kwargs,
        )
        hidden_states = outputs.last_hidden_state
        positions = (
            slice(-logits_to_keep, None)
            if isinstance(logits_to_keep, int)
            else logits_to_keep
        )
        return CausalLMOutputWithPast(
            logits=self.lm_head(hidden_states[:, positions, :]),
            past_key_values=outputs.past_key_values,
        )
