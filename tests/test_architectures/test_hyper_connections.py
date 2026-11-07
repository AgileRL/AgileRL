# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable

import pytest
import torch
from torch import nn
from transformers.models.deepseek_v4.configuration_deepseek_v4 import (
    DeepseekV4Config,
)
from transformers.models.deepseek_v4.modeling_deepseek_v4 import (
    DeepseekV4HyperConnection,
)
from transformers.models.glm5_next.configuration_glm5_next import (
    Glm5NextTextConfig,
)
from transformers.models.glm5_next.modeling_glm5_next import (
    Glm5NextTextHyperConnection,
)

from agilerl.architectures.hyper_connections import (
    COMPILED_HYPER_CONNECTION_MIX,
    compile_hyper_connection_mix,
    hyper_connection_forward,
    hyper_connection_mix,
    sinkhorn_normalize,
)
from tests.helper_functions import skip_torch_compile_on_windows_cpu

HIDDEN = 32
HC_MULT = 4
BATCH = 2
SEQ = 8
SEED = 0
# Default init (std 0.02) gives near-uniform mixing matrices that hide errors.
INIT_STD = 0.5
# fp32 CPU; inductor may fuse the 4-element row and column sums, and the
# pre-weighted collapse sum, in another order.
ATOL = 1e-6
RTOL = 1e-5

STOCK_FORWARDS = {
    DeepseekV4HyperConnection: DeepseekV4HyperConnection.__dict__["forward"],
    Glm5NextTextHyperConnection: Glm5NextTextHyperConnection.__dict__["forward"],
}


def deepseek_v4_hyper_connection() -> nn.Module:
    config = DeepseekV4Config(hidden_size=HIDDEN, hc_mult=HC_MULT)
    return DeepseekV4HyperConnection(config)


def glm5_next_hyper_connection() -> nn.Module:
    config = Glm5NextTextConfig(hidden_size=HIDDEN, hc_mult=HC_MULT)
    return Glm5NextTextHyperConnection(config)


HYPER_CONNECTIONS = [
    pytest.param(deepseek_v4_hyper_connection, id="deepseek_v4"),
    pytest.param(glm5_next_hyper_connection, id="glm5_next"),
]


def seeded(build: Callable[[], nn.Module]) -> nn.Module:
    torch.manual_seed(SEED)
    module = build()
    for parameter in module.parameters():
        nn.init.normal_(parameter, std=INIT_STD)
    return module


def streams(seq_len: int = SEQ) -> torch.Tensor:
    torch.manual_seed(SEED + 1)
    return torch.randn(BATCH, seq_len, HC_MULT, HIDDEN, requires_grad=True)


def run(
    forward: Callable[..., tuple[torch.Tensor, ...]],
    module: nn.Module,
    hidden_streams: torch.Tensor,
) -> tuple[tuple[torch.Tensor, ...], dict[str, torch.Tensor]]:
    module.zero_grad(set_to_none=True)
    hidden_streams.grad = None
    outputs = forward(module, hidden_streams)
    torch.manual_seed(SEED + 2)
    sum((output * torch.randn_like(output)).sum() for output in outputs).backward()
    assert hidden_streams.grad is not None
    grads = {"hidden_streams": hidden_streams.grad.clone()}
    for name, parameter in module.named_parameters():
        assert parameter.grad is not None, name
        grads[name] = parameter.grad.clone()
    return tuple(output.detach() for output in outputs), grads


class TestSinkhornNormalize:
    def test_rows_and_columns_sum_to_one(self):
        # Arrange
        torch.manual_seed(SEED)
        comb = torch.rand(BATCH, SEQ, HC_MULT, HC_MULT) + 0.1

        # Act
        normalized = sinkhorn_normalize(comb, 20, 1e-6)

        # Assert
        ones = torch.ones(BATCH, SEQ, HC_MULT)
        assert torch.allclose(normalized.sum(dim=-2), ones, atol=1e-5)
        assert torch.allclose(normalized.sum(dim=-1), ones, atol=1e-3)

    def test_one_iteration_only_normalizes_columns(self):
        comb = torch.tensor([[1.0, 3.0], [1.0, 1.0]])

        normalized = sinkhorn_normalize(comb, 1, 0.0)

        assert torch.equal(normalized, torch.tensor([[0.5, 0.75], [0.5, 0.25]]))


def eager_mix(
    module: nn.Module, hidden_streams: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return hyper_connection_mix(
        hidden_streams,
        module.fn,
        module.base,
        module.scale,
        module.hc_mult,
        module.hc_sinkhorn_iters,
        module.hc_eps,
        module.input_norm.eps,
    )


@pytest.mark.parametrize("build", HYPER_CONNECTIONS)
class TestHyperConnectionMix:
    def test_eager_outputs_and_gradients_match_stock(self, build):
        # Arrange
        module = seeded(build)
        stock_forward = STOCK_FORWARDS[type(module)]
        hidden_streams = streams()

        # Act
        stock_outputs, stock_grads = run(stock_forward, module, hidden_streams)
        outputs, grads = run(eager_mix, module, hidden_streams)

        # Assert
        for output, stock_output in zip(outputs, stock_outputs, strict=True):
            assert torch.allclose(output, stock_output, atol=ATOL, rtol=RTOL)
        assert grads.keys() == stock_grads.keys()
        for name, grad in grads.items():
            assert torch.allclose(grad, stock_grads[name], atol=ATOL, rtol=RTOL), name


@skip_torch_compile_on_windows_cpu
@pytest.mark.parametrize("build", HYPER_CONNECTIONS)
class TestHyperConnectionForward:
    def test_outputs_and_gradients_match_stock(self, build):
        # Arrange
        module = seeded(build)
        stock_forward = STOCK_FORWARDS[type(module)]
        hidden_streams = streams()

        # Act
        stock_outputs, stock_grads = run(stock_forward, module, hidden_streams)
        outputs, grads = run(hyper_connection_forward, module, hidden_streams)

        # Assert
        for output, stock_output in zip(outputs, stock_outputs, strict=True):
            assert torch.allclose(output, stock_output, atol=ATOL, rtol=RTOL)
        assert grads.keys() == stock_grads.keys()
        for name, grad in grads.items():
            assert torch.allclose(grad, stock_grads[name], atol=ATOL, rtol=RTOL), name

    def test_saves_only_the_streams_and_parameters(self, build):
        # Arrange
        module = seeded(build).to(torch.bfloat16)
        hidden_streams = streams().detach().to(torch.bfloat16).requires_grad_(True)
        saved = []

        def record(tensor):
            saved.append(tensor)
            return tensor

        # Act
        with torch.autograd.graph.saved_tensors_hooks(record, lambda t: t):
            hyper_connection_forward(module, hidden_streams)

        # Assert
        inputs = [hidden_streams, module.fn, module.base, module.scale]
        assert len(saved) == len(inputs)
        for tensor, expected in zip(saved, inputs, strict=True):
            assert tensor is expected

    def test_frozen_parameters_get_no_gradient(self, build):
        # Arrange
        module = seeded(build)
        module.requires_grad_(False)
        hidden_streams = streams()
        stock_streams = streams()
        torch.manual_seed(SEED + 2)
        weights = [
            torch.randn(BATCH, SEQ, HC_MULT),
            torch.randn(BATCH, SEQ, HC_MULT, HC_MULT),
            torch.randn(BATCH, SEQ, HIDDEN),
        ]

        # Act
        outputs = hyper_connection_forward(module, hidden_streams)
        sum((o * w).sum() for o, w in zip(outputs, weights, strict=True)).backward()
        stock_outputs = STOCK_FORWARDS[type(module)](module, stock_streams)
        sum(
            (o * w).sum() for o, w in zip(stock_outputs, weights, strict=True)
        ).backward()

        # Assert
        assert all(parameter.grad is None for parameter in module.parameters())
        assert torch.allclose(
            hidden_streams.grad, stock_streams.grad, atol=ATOL, rtol=RTOL
        )

    def test_new_sequence_length_reuses_the_compiled_graph(self, build):
        # Arrange
        module = seeded(build)
        run(hyper_connection_forward, module, streams(SEQ))

        # Act
        with torch.compiler.set_stance("fail_on_recompile"):
            outputs, _ = run(hyper_connection_forward, module, streams(SEQ + 5))

        # Assert
        stock_outputs, _ = run(STOCK_FORWARDS[type(module)], module, streams(SEQ + 5))
        assert torch.allclose(outputs[1], stock_outputs[1], atol=ATOL, rtol=RTOL)


@pytest.mark.parametrize("hyper_connection_cls", list(STOCK_FORWARDS))
class TestCompileHyperConnectionMix:
    def test_installs_the_compiled_forward_once(
        self, hyper_connection_cls, monkeypatch
    ):
        # Arrange
        monkeypatch.setattr(
            hyper_connection_cls, "forward", STOCK_FORWARDS[hyper_connection_cls]
        )
        monkeypatch.setattr(
            hyper_connection_cls, COMPILED_HYPER_CONNECTION_MIX, False, raising=False
        )

        # Act
        compile_hyper_connection_mix(hyper_connection_cls)
        compile_hyper_connection_mix(hyper_connection_cls)

        # Assert
        assert hyper_connection_cls.forward is hyper_connection_forward
        assert vars(hyper_connection_cls)[COMPILED_HYPER_CONNECTION_MIX] is True
