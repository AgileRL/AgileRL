# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""CPU-only LSTM module tests (no GPU mark)."""

import pytest
import torch

from agilerl.modules.lstm import EvolvableLSTM


class TestEvolvableLSTMForward:
    def test_forward_converts_non_tensor_input(self, device):
        lstm = EvolvableLSTM(
            input_size=3,
            hidden_state_size=16,
            num_outputs=2,
            num_layers=1,
            device=device,
        )
        h0 = torch.zeros(1, 1, 16, device=device)
        c0 = torch.zeros(1, 1, 16, device=device)

        output, next_hidden = lstm.forward(
            [[0.1, 0.2, 0.3]],
            hidden_state={f"{lstm.name}_h": h0, f"{lstm.name}_c": c0},
        )

        assert output.shape == (1, 2)
        assert f"{lstm.name}_h" in next_hidden
        assert f"{lstm.name}_c" in next_hidden

    def test_forward_raises_when_hidden_state_missing(self, device):
        lstm = EvolvableLSTM(
            input_size=10,
            hidden_state_size=32,
            num_outputs=4,
            num_layers=1,
            device=device,
        )

        with pytest.raises(ValueError, match="Hidden state is required"):
            lstm.forward(torch.randn(1, 10, device=device), hidden_state=None)
