"""CPU-only CNN module tests (no GPU mark)."""

from unittest.mock import MagicMock, patch

import numpy as np
import torch

from agilerl.modules.cnn import EvolvableCNN, MutableKernelSizes


class TestMutableKernelSizesAddLayerConv2d:
    def test_mutable_kernel_sizes_add_layer_conv2d(self, device):
        mut = MutableKernelSizes(
            sizes=[(3, 3), (3, 3)],
            cnn_block_type="Conv2d",
            sample_input=torch.zeros(1, 1, 16, 16, device=device),
            rng=np.random.default_rng(42),
        )
        mut.add_layer(5)
        assert mut.sizes[-1] == (5, 5)


class TestMutableKernelSizesEmptyCandidates:
    def test_change_kernel_size_empty_candidates_fallback(self, device):
        mut = MutableKernelSizes(
            sizes=[(3, 3)],
            cnn_block_type="Conv2d",
            sample_input=torch.zeros(1, 1, 8, 8, device=device),
            rng=np.random.default_rng(42),
        )
        mut.rng = MagicMock()
        mut.rng.integers.return_value = np.int64(1)
        with patch.object(mut, "calc_max_kernel_sizes", return_value=[0]):
            mut.change_kernel_size(
                hidden_layer=0,
                channel_size=[32],
                stride_size=[1],
                input_shape=(1, 8, 8),
            )
        assert mut.sizes[0] == (1, 1)
        mut.rng.integers.assert_called_once()


class TestEvolvableCNNRngSetter:
    def test_rng_setter_propagates_to_submodules(self, device):
        evolvable_cnn = EvolvableCNN(
            input_shape=[1, 16, 16],
            channel_size=[32],
            kernel_size=[3],
            stride_size=[1],
            num_outputs=4,
            device=device,
            random_seed=0,
        )
        new_rng = np.random.default_rng(99)
        evolvable_cnn.rng = new_rng
        assert evolvable_cnn.rng is new_rng
        assert evolvable_cnn.mut_kernel_size.rng is new_rng
        for module in evolvable_cnn.modules():
            if module is evolvable_cnn:
                continue
            assert module.rng is new_rng


class TestEvolvableCNNAddLayerRevertBranches:
    def test_add_layer_revert_l_in_lt_2(self, device):
        evolvable_cnn = EvolvableCNN(
            input_shape=[1, 32, 32],
            channel_size=[32],
            kernel_size=[3],
            stride_size=[2],
            num_outputs=4,
            max_hidden_layers=3,
            device=device,
            random_seed=0,
        )
        evolvable_cnn.cnn_output_size = (1, 32, 16, 1)
        with (
            patch("agilerl.modules.cnn.any", return_value=False),
            patch.object(
                evolvable_cnn, "add_channel", return_value=None
            ) as mock_add_channel,
        ):
            evolvable_cnn.add_layer()
        mock_add_channel.assert_called_once()

    def test_add_layer_revert_max_s_new_lt_1(self, device):
        evolvable_cnn = EvolvableCNN(
            input_shape=[1, 32, 32],
            channel_size=[32],
            kernel_size=[3],
            stride_size=[2],
            num_outputs=4,
            max_hidden_layers=3,
            device=device,
            random_seed=0,
        )
        evolvable_cnn.cnn_output_size = (1, 32, 16, 16)
        initial_channels = len(evolvable_cnn.channel_size)
        initial_kernels = len(evolvable_cnn.mut_kernel_size)
        evolvable_cnn.rng = MagicMock()
        evolvable_cnn.rng.integers.return_value = np.int64(20)
        with (
            patch("agilerl.modules.cnn.any", return_value=False),
            patch.object(
                evolvable_cnn, "add_channel", return_value=None
            ) as mock_add_channel,
        ):
            evolvable_cnn.add_layer()
        assert len(evolvable_cnn.channel_size) == initial_channels
        assert len(evolvable_cnn.mut_kernel_size) == initial_kernels
        mock_add_channel.assert_called_once()
