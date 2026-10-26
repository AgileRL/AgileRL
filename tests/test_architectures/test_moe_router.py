# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Tests for the out-of-place MoE router forwards."""

from __future__ import annotations

from collections.abc import Callable
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.utils.checkpoint import (
    CheckpointPolicy,
    checkpoint,
    create_selective_checkpoint_contexts,
)
from transformers.models.flex_olmo.modeling_flex_olmo import FlexOlmoTopKRouter
from transformers.models.gemma4.modeling_gemma4 import Gemma4TextRouter
from transformers.models.gpt_oss.modeling_gpt_oss import GptOssTopKRouter
from transformers.models.mellum.modeling_mellum import MellumTopKRouter
from transformers.models.minimax.modeling_minimax import MiniMaxTopKRouter
from transformers.models.mixtral.modeling_mixtral import MixtralTopKRouter
from transformers.models.olmoe.modeling_olmoe import OlmoeTopKRouter
from transformers.models.qwen2_moe.modeling_qwen2_moe import Qwen2MoeTopKRouter
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeTopKRouter
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeTopKRouter
from transformers.models.qwen3_next.modeling_qwen3_next import Qwen3NextTopKRouter
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeTalkerTextTopKRouter,
    Qwen3OmniMoeTextTopKRouter,
    Qwen3OmniMoeThinkerTextTopKRouter,
)
from transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe import (
    Qwen3VLMoeTextTopKRouter,
)
from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextTopKRouter

from agilerl.architectures.moe_router import (
    ROUTER_FORWARDS,
    ROUTER_RENORM_OUT_OF_PLACE,
    patch_moe_routers,
)
from tests.helper_functions import restore_moe_routers

# Routers that renormalize only when ``config.norm_topk_prob`` is set.
OPTIONAL_RENORM_CLASSES = (
    Qwen2MoeTopKRouter,
    OlmoeTopKRouter,
    FlexOlmoTopKRouter,
    MellumTopKRouter,
    Qwen4ExpTextTopKRouter,
    Qwen3MoeTopKRouter,
    Qwen3NextTopKRouter,
    Qwen3OmniMoeThinkerTextTopKRouter,
    Qwen3OmniMoeTalkerTextTopKRouter,
)

ROUTER_CLASSES = (
    *OPTIONAL_RENORM_CLASSES,
    MixtralTopKRouter,
    MiniMaxTopKRouter,
    Qwen3OmniMoeTextTopKRouter,
    Qwen3_5MoeTopKRouter,
    Qwen3VLMoeTextTopKRouter,
    Gemma4TextRouter,
)

# Captured at collection, before any test patches a class.
TRANSFORMERS_FORWARDS = {cls: vars(cls)["forward"] for cls in ROUTER_CLASSES}

ROUTER_CASES = [
    pytest.param(cls, norm, id=f"{cls.__name__}-norm{norm}")
    for cls in ROUTER_CLASSES
    for norm in (True, False)
    if norm or cls in OPTIONAL_RENORM_CLASSES
]


def qualname(cls: type) -> str:
    return f"{cls.__module__}.{cls.__qualname__}"


def make_router(cls: type, norm_topk_prob: bool, dtype: torch.dtype) -> nn.Module:
    """Router with random weights, so softmax ties and zero scores are unlikely."""
    config = SimpleNamespace(
        hidden_size=16,
        num_experts=8,
        num_local_experts=8,
        num_experts_per_tok=3,
        top_k_experts=3,
        norm_topk_prob=norm_topk_prob,
        rms_norm_eps=1e-6,
    )
    router = cls(config)
    torch.manual_seed(0)
    for param in router.parameters():
        nn.init.normal_(param, mean=1.0 if param.ndim == 1 else 0.0, std=0.5)
    return router.to(dtype)


def hidden_states(dtype: torch.dtype) -> torch.Tensor:
    return torch.randn(2, 5, 16, generator=torch.Generator().manual_seed(1)).to(dtype)


def save_topk(
    _context: object, op: torch._ops.OpOverload, *_args: object, **_kwargs: object
) -> CheckpointPolicy:
    if op.overloadpacket is torch.ops.aten.topk:
        return CheckpointPolicy.MUST_SAVE
    return CheckpointPolicy.PREFER_RECOMPUTE


def checkpoint_saving_topk(
    forward: Callable[[nn.Module, torch.Tensor], tuple[torch.Tensor, ...]],
    router: nn.Module,
    inputs: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    """Run *forward* under selective checkpointing that saves ``topk``."""
    return checkpoint(
        forward,
        router,
        inputs,
        use_reentrant=False,
        context_fn=lambda: create_selective_checkpoint_contexts(save_topk),
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
@pytest.mark.parametrize(("cls", "norm_topk_prob"), ROUTER_CASES)
class TestRouterForwards:
    def test_outputs_bitwise_equal_transformers_forward(
        self, cls, norm_topk_prob, dtype
    ):
        # Arrange
        router = make_router(cls, norm_topk_prob, dtype)
        inputs = hidden_states(dtype)

        # Act
        expected = TRANSFORMERS_FORWARDS[cls](router, inputs)
        outputs = ROUTER_FORWARDS[qualname(cls)](router, inputs)

        # Assert
        assert len(outputs) == len(expected) == 3
        for output, reference in zip(outputs, expected, strict=True):
            assert output.dtype == reference.dtype
            assert torch.equal(output, reference)

    def test_grads_match_uncheckpointed_when_topk_is_saved(
        self, cls, norm_topk_prob, dtype
    ):
        # Arrange
        router = make_router(cls, norm_topk_prob, dtype)
        inputs = hidden_states(dtype).requires_grad_()
        wrt = [inputs, *router.parameters()]
        _, scores, _ = TRANSFORMERS_FORWARDS[cls](router, inputs)
        expected = torch.autograd.grad(scores.float().square().sum(), wrt)

        # Act
        _, scores, _ = checkpoint_saving_topk(
            ROUTER_FORWARDS[qualname(cls)], router, inputs
        )
        grads = torch.autograd.grad(scores.float().square().sum(), wrt)

        # Assert
        for grad, reference in zip(grads, expected, strict=True):
            assert torch.equal(grad, reference)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
@pytest.mark.parametrize("cls", ROUTER_CLASSES, ids=lambda cls: cls.__name__)
class TestTransformersRouterForwards:
    def test_in_place_renormalization_fails_recompute(self, cls, dtype):
        router = make_router(cls, True, dtype)
        inputs = hidden_states(dtype).requires_grad_()
        _, scores, _ = checkpoint_saving_topk(
            TRANSFORMERS_FORWARDS[cls], router, inputs
        )

        with pytest.raises(RuntimeError, match="has been mutated"):
            scores.float().square().sum().backward()


class TestRouterForwardsTable:
    def test_lists_every_router_class(self):
        assert set(ROUTER_FORWARDS) == {qualname(cls) for cls in ROUTER_CLASSES}


@pytest.mark.usefixtures("pristine_moe_routers")
class TestPatchMoeRouters:
    def test_installs_forward_on_listed_routers_in_model(self):
        # Arrange
        router = make_router(Qwen3_5MoeTopKRouter, True, torch.float32)
        model = nn.ModuleDict({"router": router})
        inputs = hidden_states(torch.float32)
        expected = router(inputs)

        # Act
        patch_moe_routers(model)

        # Assert
        assert (
            Qwen3_5MoeTopKRouter.forward
            is ROUTER_FORWARDS[qualname(Qwen3_5MoeTopKRouter)]
        )
        assert vars(Qwen3_5MoeTopKRouter)[ROUTER_RENORM_OUT_OF_PLACE] is True
        for output, reference in zip(router(inputs), expected, strict=True):
            assert torch.equal(output, reference)

    def test_is_idempotent(self):
        model = nn.ModuleDict(
            {"router": make_router(Gemma4TextRouter, True, torch.float32)}
        )
        patch_moe_routers(model)

        patch_moe_routers(model)

        assert Gemma4TextRouter.forward is ROUTER_FORWARDS[qualname(Gemma4TextRouter)]

    def test_leaves_absent_and_unlisted_routers_untouched(self):
        # Arrange
        gpt_oss_forward = vars(GptOssTopKRouter)["forward"]
        config = SimpleNamespace(
            num_experts_per_tok=2, num_local_experts=4, hidden_size=16
        )
        model = nn.ModuleDict(
            {
                "qwen3_5": make_router(Qwen3_5MoeTopKRouter, True, torch.float32),
                "gpt_oss": GptOssTopKRouter(config),
            }
        )

        # Act
        patch_moe_routers(model)

        # Assert
        assert vars(GptOssTopKRouter)["forward"] is gpt_oss_forward
        assert Qwen3MoeTopKRouter.forward is TRANSFORMERS_FORWARDS[Qwen3MoeTopKRouter]
        assert ROUTER_RENORM_OUT_OF_PLACE not in vars(Qwen3MoeTopKRouter)


class TestRestoreMoeRouters:
    def test_drops_the_flag_and_restores_forward_on_exit(self):
        # Arrange
        model = nn.ModuleDict(
            {"router": make_router(Qwen3_5MoeTopKRouter, True, torch.float32)}
        )

        # Act
        with restore_moe_routers():
            patch_moe_routers(model)
            patched = vars(Qwen3_5MoeTopKRouter)[ROUTER_RENORM_OUT_OF_PLACE]

        # Assert
        assert patched is True
        assert ROUTER_RENORM_OUT_OF_PLACE not in vars(Qwen3_5MoeTopKRouter)
        assert (
            vars(Qwen3_5MoeTopKRouter)["forward"]
            is TRANSFORMERS_FORWARDS[Qwen3_5MoeTopKRouter]
        )
