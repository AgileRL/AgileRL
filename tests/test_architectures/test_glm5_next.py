# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.distributed as dist
from transformers.models.glm5_next.configuration_glm5_next import (
    Glm5NextTextConfig,
)
from transformers.models.glm5_next.modeling_glm5_next import (
    Glm5NextTextAttention,
    Glm5NextTextHyperConnection,
    Glm5NextTextIndexer,
    Glm5NextTextModel,
)

import agilerl.architectures.glm5_next.patches as glm5_next
from agilerl.architectures.catalog import FAMILY_RUNTIME_CONFIGS
from agilerl.architectures.glm5_next import (
    CAUSAL_SDPA_WHEN_TOPK_COVERS,
    causal_sdpa_attention_forward,
    install_glm5_next_patches,
    topk_covers_causal_prefix,
    topk_skipping_model_forward,
)
from agilerl.architectures.glm5_next.patches import (
    SequenceShard,
    expand_latent_kv,
    sequence_shard_sparse_mask,
    topk_attention_mask,
)
from agilerl.architectures.hyper_connections import (
    COMPILED_HYPER_CONNECTION_MIX,
    hyper_connection_forward,
)
from tests.helper_functions import load_without_llm_dependencies

HIDDEN = 64
KPOOL = 4
SEQ = 16
BATCH = 2
SEED = 0
# Default init (std 0.02) leaves gradients near the comparison tolerance.
INIT_STD = 0.1
# Masked and is_causal SDPA both run the CPU math kernel over the same scores.
ATOL = 1e-6
RTOL = 1e-5
# Four k-pools of four tokens: 16 keeps all four, 8 keeps two plus the tail.
COVERING_TOPK = SEQ
PARTIAL_TOPK = SEQ // 2
# The second DSA layer reuses the first layer's top-k.
SHARED_DSA_LAYERS = (("indexed_attention", "full"), ("indexed_attention", "shared"))
HYBRID_LAYERS = (("linear_attention", "full"), ("indexed_attention", "full"))

STOCK_ATTENTION_FORWARD = Glm5NextTextAttention.forward
STOCK_MODEL_FORWARD = Glm5NextTextModel.__dict__["forward"]
STOCK_HYPER_CONNECTION_FORWARD = Glm5NextTextHyperConnection.__dict__["forward"]


def tiny_config(
    index_topk: int = COVERING_TOPK,
    attn_implementation: str = "sdpa",
    always_select_tail: bool = True,
    layers: tuple[tuple[str, str], ...] = SHARED_DSA_LAYERS,
) -> Glm5NextTextConfig:
    config = Glm5NextTextConfig(
        vocab_size=32,
        hidden_size=HIDDEN,
        intermediate_size=64,
        moe_intermediate_size=32,
        num_hidden_layers=len(layers),
        layer_types=[layer_type for layer_type, _ in layers],
        mlp_layer_types=["dense"] * len(layers),
        indexer_types=[indexer_type for _, indexer_type in layers],
        linear_head_dim=16,
        linear_num_heads=4,
        num_attention_heads=4,
        num_key_value_heads=4,
        q_lora_rank=32,
        kv_lora_rank=32,
        qk_rope_head_dim=0,
        qk_nope_head_dim=16,
        v_head_dim=16,
        index_n_heads=4,
        index_head_dim=16,
        index_topk=index_topk,
        index_kpool=KPOOL,
        index_kpool_always_select_tail=always_select_tail,
        n_routed_experts=4,
        n_shared_experts=1,
        num_experts_per_tok=2,
        hc_mult=4,
        pad_token_id=0,
    )
    config._attn_implementation = attn_implementation
    return config


def tiny_model(
    index_topk: int, layers: tuple[tuple[str, str], ...] = SHARED_DSA_LAYERS
) -> Glm5NextTextModel:
    torch.manual_seed(SEED)
    model = Glm5NextTextModel(tiny_config(index_topk, layers=layers))
    for parameter in model.parameters():
        torch.nn.init.normal_(parameter, std=INIT_STD)
    return model


def run_model(
    model: Glm5NextTextModel, inputs: torch.Tensor, weights: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor | None]]:
    model.zero_grad(set_to_none=True)
    inputs = inputs.clone().requires_grad_(True)
    output = model(inputs_embeds=inputs, use_cache=False).last_hidden_state
    (output * weights).sum().backward()
    assert inputs.grad is not None
    grads = {
        name: None if parameter.grad is None else parameter.grad.clone()
        for name, parameter in model.named_parameters()
    }
    return output.detach(), inputs.grad, grads


def use_stock_forwards(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(Glm5NextTextModel, "forward", STOCK_MODEL_FORWARD)
    monkeypatch.setattr(Glm5NextTextAttention, "forward", STOCK_ATTENTION_FORWARD)


def use_patched_forwards(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        Glm5NextTextModel, "forward", topk_skipping_model_forward(STOCK_MODEL_FORWARD)
    )
    monkeypatch.setattr(Glm5NextTextAttention, "forward", causal_sdpa_attention_forward)


def record_sdpa_masks(monkeypatch: pytest.MonkeyPatch) -> list[tuple[bool, bool]]:
    """Record ``(attn_mask is None, is_causal)`` for every SDPA call."""
    calls: list[tuple[bool, bool]] = []
    sdpa: Callable[..., torch.Tensor] = torch.nn.functional.scaled_dot_product_attention

    def recording_sdpa(*args: object, **kwargs: object) -> torch.Tensor:
        calls.append((kwargs.get("attn_mask") is None, bool(kwargs.get("is_causal"))))
        return sdpa(*args, **kwargs)

    monkeypatch.setattr(
        torch.nn.functional, "scaled_dot_product_attention", recording_sdpa
    )
    return calls


def record_indexer_layers(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Record the layer index of every DSA indexer call."""
    layers: list[int] = []
    indexer_forward = Glm5NextTextIndexer.forward

    def recording_forward(
        self: Glm5NextTextIndexer, *args: object, **kwargs: object
    ) -> torch.Tensor:
        layers.append(self.layer_idx)
        return indexer_forward(self, *args, **kwargs)

    monkeypatch.setattr(Glm5NextTextIndexer, "forward", recording_forward)
    return layers


def causal_topk_indices() -> torch.Tensor:
    """Top-k indices selecting every key at or before each query."""
    keys = torch.arange(SEQ).expand(BATCH, SEQ, SEQ)
    queries = torch.arange(SEQ)[None, :, None]
    return keys.masked_fill(keys > queries, -1).to(torch.int32)


class TestTopkCoversCausalPrefix:
    @pytest.mark.parametrize(
        ("seq_length", "expected"),
        [(SEQ, True), (SEQ + KPOOL - 1, True), (SEQ + KPOOL, False)],
    )
    def test_holds_while_every_complete_pool_fits_topk(
        self, seq_length: int, expected: bool
    ) -> None:
        attention_mask = torch.ones(BATCH, seq_length, dtype=torch.bool)

        covers = topk_covers_causal_prefix(tiny_config(), seq_length, attention_mask)

        assert covers is expected

    def test_holds_without_a_mask(self) -> None:
        assert topk_covers_causal_prefix(tiny_config(), SEQ, None)

    def test_padding_breaks_coverage(self) -> None:
        attention_mask = torch.ones(BATCH, SEQ, dtype=torch.bool)
        attention_mask[0, 0] = False

        assert not topk_covers_causal_prefix(tiny_config(), SEQ, attention_mask)

    def test_eager_backend_breaks_coverage(self) -> None:
        config = tiny_config(attn_implementation="eager")

        assert not topk_covers_causal_prefix(config, SEQ, None)

    def test_no_tail_breaks_coverage(self) -> None:
        config = tiny_config(always_select_tail=False)

        assert not topk_covers_causal_prefix(config, SEQ, None)


class TestTopkSkippingModelForward:
    @pytest.mark.parametrize(
        ("index_topk", "layers", "expected_sdpa_calls", "expected_indexer_layers"),
        [
            pytest.param(
                COVERING_TOPK,
                SHARED_DSA_LAYERS,
                [(True, True)] * 2,
                [],
                id="covering-shared",
            ),
            pytest.param(
                PARTIAL_TOPK,
                SHARED_DSA_LAYERS,
                [(False, False)] * 2,
                [0],
                id="partial-shared",
            ),
            pytest.param(
                COVERING_TOPK, HYBRID_LAYERS, [(True, True)], [], id="covering-hybrid"
            ),
            pytest.param(
                PARTIAL_TOPK, HYBRID_LAYERS, [(False, False)], [1], id="partial-hybrid"
            ),
        ],
    )
    def test_outputs_and_gradients_match_stock(
        self,
        monkeypatch: pytest.MonkeyPatch,
        index_topk: int,
        layers: tuple[tuple[str, str], ...],
        expected_sdpa_calls: list[tuple[bool, bool]],
        expected_indexer_layers: list[int],
    ) -> None:
        # Arrange
        model = tiny_model(index_topk, layers)
        inputs = torch.randn(BATCH, SEQ, HIDDEN)
        weights = torch.randn(BATCH, SEQ, HIDDEN)
        use_stock_forwards(monkeypatch)
        stock_output, stock_input_grad, stock_grads = run_model(model, inputs, weights)
        use_patched_forwards(monkeypatch)
        sdpa_calls = record_sdpa_masks(monkeypatch)
        indexer_layers = record_indexer_layers(monkeypatch)

        # Act
        output, input_grad, grads = run_model(model, inputs, weights)

        # Assert
        assert sdpa_calls == expected_sdpa_calls
        assert indexer_layers == expected_indexer_layers
        assert torch.allclose(output, stock_output, atol=ATOL, rtol=RTOL)
        assert torch.allclose(input_grad, stock_input_grad, atol=ATOL, rtol=RTOL)
        assert stock_input_grad.abs().max() > 1e-2
        for name, stock_grad in stock_grads.items():
            grad = grads[name]
            assert (grad is None) == (stock_grad is None), name
            if grad is not None and stock_grad is not None:
                assert torch.allclose(grad, stock_grad, atol=ATOL, rtol=RTOL), name

    def test_stock_indexer_parameters_get_no_gradient(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange
        model = tiny_model(PARTIAL_TOPK)
        inputs = torch.randn(BATCH, SEQ, HIDDEN)
        weights = torch.randn(BATCH, SEQ, HIDDEN)
        use_stock_forwards(monkeypatch)

        # Act
        _, _, grads = run_model(model, inputs, weights)

        # Assert
        indexer_grads = {
            name: grad for name, grad in grads.items() if ".indexer." in name
        }
        assert indexer_grads
        assert all(grad is None for grad in indexer_grads.values())

    def test_cache_keeps_the_indexer(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Arrange
        model = tiny_model(COVERING_TOPK)
        inputs = torch.randn(BATCH, SEQ, HIDDEN)
        use_stock_forwards(monkeypatch)
        with torch.no_grad():
            stock_output = model(inputs_embeds=inputs, use_cache=True).last_hidden_state
        use_patched_forwards(monkeypatch)
        sdpa_calls = record_sdpa_masks(monkeypatch)
        indexer_layers = record_indexer_layers(monkeypatch)

        # Act
        with torch.no_grad():
            output = model(inputs_embeds=inputs, use_cache=True).last_hidden_state

        # Assert
        assert sdpa_calls == [(False, False)] * 2
        assert indexer_layers == [0]
        assert torch.allclose(output, stock_output, atol=ATOL, rtol=RTOL)

    def test_padding_keeps_the_indexer(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Arrange
        model = tiny_model(COVERING_TOPK)
        inputs = torch.randn(BATCH, SEQ, HIDDEN)
        attention_mask = torch.ones(BATCH, SEQ, dtype=torch.long)
        attention_mask[0, 0] = 0
        use_stock_forwards(monkeypatch)
        with torch.no_grad():
            stock_output = model(
                inputs_embeds=inputs, attention_mask=attention_mask, use_cache=False
            ).last_hidden_state
        use_patched_forwards(monkeypatch)
        sdpa_calls = record_sdpa_masks(monkeypatch)
        indexer_layers = record_indexer_layers(monkeypatch)

        # Act
        with torch.no_grad():
            output = model(
                inputs_embeds=inputs, attention_mask=attention_mask, use_cache=False
            ).last_hidden_state

        # Assert: the padded query row has no visible key, so compare the unpadded row.
        assert sdpa_calls == [(False, False)] * 2
        assert indexer_layers == [0]
        assert torch.allclose(output[1], stock_output[1], atol=ATOL, rtol=RTOL)


class TestCausalSdpaAttentionForward:
    def test_shared_layer_without_a_mask_matches_causal_topk(self) -> None:
        # Arrange
        torch.manual_seed(SEED)
        attention = Glm5NextTextAttention(tiny_config(), layer_idx=1)
        hidden_states = torch.randn(BATCH, SEQ, HIDDEN)
        attention_mask = torch.ones(BATCH, SEQ, dtype=torch.bool)
        expected, _, _ = causal_sdpa_attention_forward(
            attention,
            hidden_states,
            attention_mask,
            prev_topk_indices=causal_topk_indices(),
        )

        # Act
        output, _, topk_indices = causal_sdpa_attention_forward(
            attention, hidden_states, None
        )

        # Assert
        assert topk_indices is None
        assert torch.allclose(output, expected, atol=ATOL, rtol=RTOL)

    def test_shared_layer_without_prior_topk_raises(self) -> None:
        attention = Glm5NextTextAttention(tiny_config(), layer_idx=1)
        hidden_states = torch.randn(BATCH, SEQ, HIDDEN)
        attention_mask = torch.ones(BATCH, SEQ, dtype=torch.bool)

        with pytest.raises(ValueError, match="previous full indexer layer"):
            causal_sdpa_attention_forward(attention, hidden_states, attention_mask)


def latent_kv(attention: Glm5NextTextAttention) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(SEED)
    kv_nope = torch.randn(BATCH, 1, SEQ, attention.kv_lora_rank)
    k_rot = torch.randn(BATCH, 1, SEQ, attention.qk_rope_head_dim)
    return kv_nope, k_rot


class TestExpandLatentKv:
    def test_without_rope_matches_stock_and_keeps_the_key_a_view(self) -> None:
        # Arrange
        torch.manual_seed(SEED)
        attention = Glm5NextTextAttention(tiny_config(), layer_idx=0)
        kv_nope, k_rot = latent_kv(attention)
        expected_key, expected_value = attention.expand_kv(kv_nope, k_rot)

        # Act
        key, value = expand_latent_kv(attention, kv_nope, k_rot)

        # Assert
        assert torch.equal(key, expected_key)
        assert torch.equal(value, expected_value)
        assert key.untyped_storage().data_ptr() == value.untyped_storage().data_ptr()

    def test_with_rope_matches_stock(self) -> None:
        # Arrange
        config = tiny_config()
        config.qk_rope_head_dim = 8
        torch.manual_seed(SEED)
        attention = Glm5NextTextAttention(config, layer_idx=0)
        kv_nope, k_rot = latent_kv(attention)
        expected_key, expected_value = attention.expand_kv(kv_nope, k_rot)

        # Act
        key, value = expand_latent_kv(attention, kv_nope, k_rot)

        # Assert
        assert key.shape[-1] == attention.qk_head_dim
        assert torch.equal(key, expected_key)
        assert torch.equal(value, expected_value)


class TestTopkAttentionMask:
    @pytest.mark.parametrize("attn_implementation", ["sdpa", "eager"])
    def test_matches_stock_mask(self, attn_implementation: str) -> None:
        # Arrange
        attention = Glm5NextTextAttention(
            tiny_config(attn_implementation=attn_implementation), layer_idx=0
        )
        generator = torch.Generator().manual_seed(SEED)
        # -1 and SEQ select nothing; repeats select one key.
        topk_indices = torch.randint(-1, SEQ + 1, (BATCH, SEQ, 6), generator=generator)
        topk_indices[0, 0] = -1
        query_states = torch.zeros(BATCH, 4, SEQ, 16, dtype=torch.bfloat16)
        expected = attention.build_attention_mask_from_topk(
            topk_indices=topk_indices, query_states=query_states, kv_length=SEQ
        )

        # Act
        mask = topk_attention_mask(attention, topk_indices, query_states, SEQ)

        # Assert
        assert expected is not None
        assert mask.shape == (BATCH, 1, SEQ, SEQ)
        assert mask.dtype == expected.dtype
        assert torch.equal(mask, expected)


class TestSequenceShardShardCausalMask:
    def test_masks_keys_after_each_global_query_position(self) -> None:
        # Arrange
        shard = SequenceShard(MagicMock())
        shard.begin_forward(torch.tensor([0, 8]))

        # Act
        with patch.object(glm5_next.dist, "get_rank", return_value=1):
            mask = shard.shard_causal_mask(2, 4, torch.device("cpu"))

        # Assert
        expected = torch.tensor([[True, True, True, False], [True, True, True, True]])
        assert torch.equal(mask, expected[None, None])

    def test_reuses_the_mask_within_a_forward_and_rebuilds_for_a_new_one(
        self,
    ) -> None:
        # Arrange
        shard = SequenceShard(MagicMock())
        shard.begin_forward(torch.tensor([0, 8]))

        # Act
        with patch.object(glm5_next.dist, "get_rank", return_value=0):
            first = shard.shard_causal_mask(2, 4, torch.device("cpu"))
            same_forward = shard.shard_causal_mask(2, 4, torch.device("cpu"))
            shard.begin_forward(torch.tensor([0, 8]))
            next_forward = shard.shard_causal_mask(2, 4, torch.device("cpu"))

        # Assert
        assert same_forward is first
        assert next_forward is not first
        assert torch.equal(next_forward, first)


@pytest.mark.usefixtures("gloo_process_group")
class TestSequenceShardSparseMask:
    def test_no_padding_mask_gives_the_causal_mask_when_topk_covers(self) -> None:
        # Arrange
        torch.manual_seed(SEED)
        attention = Glm5NextTextAttention(tiny_config(), layer_idx=0)
        shard = SequenceShard(dist.group.WORLD)
        shard.begin_forward(torch.tensor([0, SEQ]))
        hidden_states = torch.randn(BATCH, SEQ, HIDDEN)
        q_resid = attention.q_a_layernorm(attention.q_a_proj(hidden_states))
        query_states = torch.zeros(BATCH, 4, SEQ, 16)

        # Act
        mask, topk_indices = sequence_shard_sparse_mask(
            attention, shard, hidden_states, q_resid, query_states, None, None
        )

        # Assert
        expected = torch.ones(SEQ, SEQ, dtype=torch.bool).tril()
        assert torch.equal(mask, expected[None, None])
        assert topk_indices is None

    def test_shared_layer_without_prior_topk_raises(self) -> None:
        # Arrange
        torch.manual_seed(SEED)
        attention = Glm5NextTextAttention(tiny_config(PARTIAL_TOPK), layer_idx=1)
        shard = SequenceShard(dist.group.WORLD)
        shard.begin_forward(torch.tensor([0, SEQ]))
        hidden_states = torch.randn(BATCH, SEQ, HIDDEN)
        q_resid = attention.q_a_layernorm(attention.q_a_proj(hidden_states))
        query_states = torch.zeros(BATCH, 4, SEQ, 16)

        # Act / Assert
        with pytest.raises(ValueError, match="previous full indexer layer"):
            sequence_shard_sparse_mask(
                attention, shard, hidden_states, q_resid, query_states, None, None
            )


class TestInstallGlm5NextPatches:
    def test_patches_model_and_attention_forward_idempotently(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange
        use_stock_forwards(monkeypatch)
        monkeypatch.setattr(
            Glm5NextTextAttention, CAUSAL_SDPA_WHEN_TOPK_COVERS, False, raising=False
        )
        monkeypatch.setattr(
            Glm5NextTextHyperConnection, "forward", STOCK_HYPER_CONNECTION_FORWARD
        )
        monkeypatch.setattr(
            Glm5NextTextHyperConnection,
            COMPILED_HYPER_CONNECTION_MIX,
            False,
            raising=False,
        )
        monkeypatch.setattr(glm5_next, "HAS_LIGER_KERNEL", False)
        patch = FAMILY_RUNTIME_CONFIGS["glm5_next_text"].patch

        # Act
        install_glm5_next_patches(patch)
        install_glm5_next_patches(patch)

        # Assert
        assert Glm5NextTextAttention.forward is causal_sdpa_attention_forward
        assert Glm5NextTextModel.forward.__wrapped__ is STOCK_MODEL_FORWARD
        assert Glm5NextTextHyperConnection.forward is hyper_connection_forward

    @pytest.mark.parametrize("model_type", ["glm5_next", "glm5_next_text"])
    def test_catalog_installs_the_attention_patch(self, model_type: str) -> None:
        patch = FAMILY_RUNTIME_CONFIGS[model_type].patch

        assert patch.install is install_glm5_next_patches


class TestImportFallbacks:
    def test_without_llm_dependencies_leaves_modeling_names_unset(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        module = load_without_llm_dependencies(glm5_next, monkeypatch)

        for name in (
            "ALL_ATTENTION_FUNCTIONS",
            "Glm5NextTextAttention",
            "Glm5NextTextHyperConnection",
            "Glm5NextTextModel",
            "eager_attention_forward",
        ):
            assert getattr(module, name) is None, name
