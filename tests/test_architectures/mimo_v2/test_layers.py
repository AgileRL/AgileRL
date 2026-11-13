# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""fp32 log-sum-exp sink renormalisation for ``MiMoV2Flash`` flex attention."""

from __future__ import annotations

import pytest
import torch
from torch import nn
from torch.nn.attention.flex_attention import create_block_mask, flex_attention
from transformers.integrations import flex_attention as flex_integration
from transformers.modeling_utils import AttentionInterface
from transformers.models.mimo_v2_flash.modeling_mimo_v2_flash import (
    MiMoV2FlashAttention,
    eager_attention_forward,
)

from agilerl.architectures.mimo_v2 import layers as attention_module
from agilerl.architectures.mimo_v2.configuration import MiMoV2Config
from agilerl.architectures.mimo_v2.layers import (
    apply_attention_sinks,
    install_mimo_v2_fp32_lse_flex_attention,
    mimo_v2_flex_attention_forward,
)
from tests.test_architectures.mimo_v2.checkpoint_helpers import tiny_hub_config

SEQ_LEN = 32
SWA_LAYER = 1
# fp32 flex and eager differ only in summation order.
FP32_TOLERANCE = {"rtol": 1e-5, "atol": 1e-5}


def swa_attention() -> MiMoV2FlashAttention:
    config = MiMoV2Config.from_dict(tiny_hub_config()).text_config
    module = MiMoV2FlashAttention(config, layer_idx=SWA_LAYER).train()
    with torch.no_grad():
        module.sinks.copy_(torch.linspace(-1.0, 2.0, config.num_attention_heads))
    return module


def attention_inputs(
    module: MiMoV2FlashAttention,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    heads = module.config.num_attention_heads
    kv_heads = heads // module.num_key_value_groups
    query = torch.randn(1, heads, SEQ_LEN, module.head_dim)
    key = torch.randn(1, kv_heads, SEQ_LEN, module.head_dim)
    value = torch.randn(1, kv_heads, SEQ_LEN, module.v_head_dim)
    return query, key, value


def in_window(window: int):
    def mask_mod(_batch, _head, q_idx, kv_idx):
        return (kv_idx <= q_idx) & (kv_idx > q_idx - window)

    return mask_mod


def additive_window_mask(window: int) -> torch.Tensor:
    q_idx = torch.arange(SEQ_LEN)[:, None]
    kv_idx = torch.arange(SEQ_LEN)[None, :]
    allowed = in_window(window)(None, None, q_idx, kv_idx)
    mask = torch.zeros(SEQ_LEN, SEQ_LEN).masked_fill(~allowed, float("-inf"))
    return mask[None, None]


def gradients(output: torch.Tensor, *tensors: torch.Tensor) -> tuple[torch.Tensor, ...]:
    torch.manual_seed(1)
    upstream = torch.randn_like(output)
    return torch.autograd.grad((output * upstream).sum(), tensors)


@pytest.fixture
def uncompiled_flex(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run flex eagerly: compiled flex on CPU cannot return the log-sum-exp."""

    def flex(query, key, value, training, **kwargs):
        return flex_attention(query, key, value, **kwargs)

    monkeypatch.setattr(attention_module, "compile_friendly_flex_attention", flex)


def sink_inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    scores = torch.randn(2, 3, 5, 7, requires_grad=True)
    value = torch.randn(2, 3, 7, 4, requires_grad=True)
    sinks = torch.tensor([-1.0, 0.5, 3.0], requires_grad=True)
    return scores, value, sinks


def softmax_with_sink(
    scores: torch.Tensor, value: torch.Tensor, sinks: torch.Tensor
) -> torch.Tensor:
    sink_column = sinks.view(1, -1, 1, 1).expand(*scores.shape[:-1], 1)
    probs = torch.softmax(torch.cat([scores, sink_column], dim=-1), dim=-1)
    return probs[..., :-1] @ value


def renormalised(
    scores: torch.Tensor, value: torch.Tensor, sinks: torch.Tensor
) -> torch.Tensor:
    return apply_attention_sinks(
        torch.softmax(scores, dim=-1) @ value, scores.logsumexp(dim=-1), sinks
    )


class TestApplyAttentionSinks:
    def test_matches_softmax_over_keys_and_sink(self) -> None:
        scores, value, sinks = sink_inputs()

        output = renormalised(scores, value, sinks)

        assert torch.allclose(
            output, softmax_with_sink(scores, value, sinks), **FP32_TOLERANCE
        )

    def test_gradients_match_softmax_over_keys_and_sink(self) -> None:
        # Arrange
        scores, value, sinks = sink_inputs()

        # Act
        grads = gradients(renormalised(scores, value, sinks), scores, value, sinks)
        expected = gradients(
            softmax_with_sink(scores, value, sinks), scores, value, sinks
        )

        # Assert
        for grad, expected_grad in zip(grads, expected, strict=True):
            assert torch.allclose(grad, expected_grad, **FP32_TOLERANCE)

    def test_keeps_the_output_dtype(self) -> None:
        output = apply_attention_sinks(
            torch.ones(1, 2, 3, 4, dtype=torch.bfloat16),
            torch.zeros(1, 2, 3),
            torch.zeros(2, dtype=torch.bfloat16),
        )

        assert output.dtype == torch.bfloat16
        assert torch.equal(output, torch.full_like(output, 0.5))


class TestApplyAttentionSinksBf16:
    def test_matches_the_fp32_renormalisation_within_bf16_rounding(self) -> None:
        # Arrange
        torch.manual_seed(0)
        attention_output = torch.randn(2, 3, 5, 4).to(torch.bfloat16)
        lse = torch.randn(2, 3, 5) * 4
        sinks = torch.tensor([-1.0, 0.5, 3.0], dtype=torch.bfloat16)
        sink_lse = sinks.float().view(1, -1, 1, 1).expand(2, 3, 5, 1)
        renorm = torch.exp(
            lse.unsqueeze(-1)
            - torch.logsumexp(torch.cat([lse.unsqueeze(-1), sink_lse], -1), -1, True)
        )
        expected = (attention_output.float() * renorm).to(torch.bfloat16)

        # Act
        output = apply_attention_sinks(attention_output, lse, sinks)

        # Assert
        assert output.dtype == torch.bfloat16
        # The bf16 factor and the bf16 product each round once: two bf16 ulps.
        assert torch.allclose(output.float(), expected.float(), rtol=2**-7, atol=0)


@pytest.mark.usefixtures("uncompiled_flex")
class TestMimoV2FlexAttentionForward:
    # CPU flex has no backward and rejects inputs that require grad.
    @torch.no_grad()
    def test_sink_layer_matches_eager_attention(self) -> None:
        # Arrange
        module = swa_attention()
        query, key, value = attention_inputs(module)
        window = module.sliding_window
        block_mask = create_block_mask(
            in_window(window), None, None, SEQ_LEN, SEQ_LEN, device="cpu"
        )

        # Act
        output, _ = mimo_v2_flex_attention_forward(
            module,
            query,
            key,
            value,
            block_mask,
            scaling=module.scaling,
            s_aux=module.sinks,
        )
        expected, _ = eager_attention_forward(
            module, query, key, value, additive_window_mask(window), module.scaling
        )

        # Assert
        assert output.shape == (1, SEQ_LEN, query.shape[1], module.v_head_dim)
        assert torch.allclose(output, expected, **FP32_TOLERANCE)

    def test_other_modules_with_sinks_run_the_stock_forward(self) -> None:
        module = nn.Module()
        query = torch.randn(1, 4, SEQ_LEN, 16)

        with pytest.raises(ValueError, match="Attention sinks cannot be run on CPU"):
            mimo_v2_flex_attention_forward(
                module, query, query, query, None, s_aux=torch.zeros(4)
            )


class TestInstallMimoV2Fp32LseFlexAttention:
    def test_flex_backend_resolves_to_the_fp32_lse_forward(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange
        monkeypatch.setattr(
            flex_integration,
            "flex_attention_forward",
            flex_integration.flex_attention_forward,
        )
        monkeypatch.setitem(
            AttentionInterface._global_mapping,
            "flex_attention",
            AttentionInterface._global_mapping["flex_attention"],
        )

        # Act
        install_mimo_v2_fp32_lse_flex_attention()

        # Assert
        assert flex_integration.flex_attention_forward is mimo_v2_flex_attention_forward
        assert AttentionInterface()["flex_attention"] is mimo_v2_flex_attention_forward
