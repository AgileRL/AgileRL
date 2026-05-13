import pytest
import torch

from agilerl.modules.resnet import EvolvableResNet
from tests.helper_functions import assert_state_dicts_equal

pytestmark = pytest.mark.gpu


######### Test instantiation #########
class TestEvolvableResNetInit:
    @pytest.mark.parametrize(
        "input_shape, channel_size, kernel_size, stride_size, num_outputs, num_blocks",
        [
            ([3, 32, 32], 64, 3, 1, 10, 2),
            ([3, 64, 64], 128, 3, 2, 10, 3),
        ],
    )
    def test_instantiation_without_errors(
        self,
        input_shape,
        channel_size,
        kernel_size,
        stride_size,
        num_outputs,
        num_blocks,
        device,
    ):
        evolvable_resnet = EvolvableResNet(
            input_shape=input_shape,
            channel_size=channel_size,
            kernel_size=kernel_size,
            stride_size=stride_size,
            num_outputs=num_outputs,
            num_blocks=num_blocks,
            device=device,
        )
        assert isinstance(evolvable_resnet, EvolvableResNet)

    @pytest.mark.parametrize(
        "input_shape, channel_size, kernel_size, stride_size, num_outputs, num_blocks",
        [
            ([3, 32, 32], 64, 3, 1, 10, "two"),  # Incorrect type for num_blocks
            ([3, 32, 32], 64, 3, 1, 0, 2),  # num_outputs cannot be zero
        ],
    )
    def test_incorrect_instantiation(
        self,
        input_shape,
        channel_size,
        kernel_size,
        stride_size,
        num_outputs,
        num_blocks,
        device,
    ):
        with pytest.raises(AssertionError):
            EvolvableResNet(
                input_shape=input_shape,
                channel_size=channel_size,
                kernel_size=kernel_size,
                stride_size=stride_size,
                num_outputs=num_outputs,
                num_blocks=num_blocks,
                device=device,
            )


class TestEvolvableResNetChangeActivation:
    def test_resnet_change_activation_noop(self, device):
        evolvable_resnet = EvolvableResNet(
            input_shape=[3, 32, 32],
            channel_size=64,
            kernel_size=3,
            stride_size=1,
            num_outputs=10,
            num_blocks=2,
            device=device,
        )
        evolvable_resnet.change_activation("ReLU", output=True)


######### Test forward #########
class TestEvolvableResNetForward:
    @pytest.mark.parametrize(
        "input_shape, channel_size, kernel_size, stride_size, num_outputs, num_blocks, output_shape",
        [
            ([3, 32, 32], 64, 3, 1, 10, 2, (1, 10)),
            ([3, 64, 64], 128, 3, 2, 10, 3, (1, 10)),
        ],
    )
    def test_forward(
        self,
        input_shape,
        channel_size,
        kernel_size,
        stride_size,
        num_outputs,
        num_blocks,
        output_shape,
        device,
    ):
        evolvable_resnet = EvolvableResNet(
            input_shape=input_shape,
            channel_size=channel_size,
            kernel_size=kernel_size,
            stride_size=stride_size,
            num_outputs=num_outputs,
            num_blocks=num_blocks,
            device=device,
        )
        input_tensor = torch.randn(1, *input_shape).to(device)
        output = evolvable_resnet.forward(input_tensor)
        assert output.shape == output_shape

    ######### Test forward with numpy input #########
    def test_forward_numpy_input(self, device):
        """Covers forward when x is not torch.Tensor."""
        import numpy as np

        evolvable_resnet = EvolvableResNet(
            input_shape=[3, 32, 32],
            channel_size=64,
            kernel_size=3,
            stride_size=1,
            num_outputs=10,
            num_blocks=2,
            device=device,
        )
        x = np.random.randn(3, 32, 32).astype(np.float32)
        output = evolvable_resnet.forward(x)
        assert output.shape == (1, 10)


######### Test add_block #########
class TestEvolvableResNetAddBlock:
    @pytest.mark.parametrize(
        "input_shape, channel_size, kernel_size, stride_size, num_outputs, num_blocks",
        [
            ([3, 32, 32], 64, 3, 1, 10, 2),
            ([3, 64, 64], 128, 3, 2, 10, 3),
        ],
    )
    def test_add_block(
        self,
        input_shape,
        channel_size,
        kernel_size,
        stride_size,
        num_outputs,
        num_blocks,
        device,
    ):
        evolvable_resnet = EvolvableResNet(
            input_shape=input_shape,
            channel_size=channel_size,
            kernel_size=kernel_size,
            stride_size=stride_size,
            num_outputs=num_outputs,
            num_blocks=num_blocks,
            device=device,
        )
        initial_blocks = evolvable_resnet.num_blocks
        evolvable_resnet.add_block()
        assert evolvable_resnet.num_blocks == initial_blocks + 1

    def test_add_block_fallback_add_channel_when_max_blocks(self, device):
        """Covers add_block when num_blocks >= max_blocks, falls back to add_channel."""
        evolvable_resnet = EvolvableResNet(
            input_shape=[3, 32, 32],
            channel_size=64,
            kernel_size=3,
            stride_size=1,
            num_outputs=10,
            num_blocks=4,
            max_blocks=4,
            device=device,
        )
        initial_channels = evolvable_resnet.channel_size
        evolvable_resnet.add_block()
        assert evolvable_resnet.channel_size >= initial_channels


######### Test remove_block #########
class TestEvolvableResNetRemoveBlock:
    @pytest.mark.parametrize(
        "input_shape, channel_size, kernel_size, stride_size, num_outputs, num_blocks",
        [
            ([3, 32, 32], 64, 3, 1, 10, 2),
            ([3, 64, 64], 128, 3, 2, 10, 3),
        ],
    )
    def test_remove_block(
        self,
        input_shape,
        channel_size,
        kernel_size,
        stride_size,
        num_outputs,
        num_blocks,
        device,
    ):
        evolvable_resnet = EvolvableResNet(
            input_shape=input_shape,
            channel_size=channel_size,
            kernel_size=kernel_size,
            stride_size=stride_size,
            num_outputs=num_outputs,
            num_blocks=num_blocks,
            device=device,
        )
        initial_blocks = evolvable_resnet.num_blocks
        evolvable_resnet.remove_block()
        assert evolvable_resnet.num_blocks == initial_blocks - 1

    def test_remove_block_fallback_add_channel_when_min_blocks(self, device):
        """Covers remove_block when num_blocks <= min_blocks, falls back to add_channel."""
        evolvable_resnet = EvolvableResNet(
            input_shape=[3, 32, 32],
            channel_size=64,
            kernel_size=3,
            stride_size=1,
            num_outputs=10,
            num_blocks=1,
            min_blocks=1,
            device=device,
        )
        initial_channels = evolvable_resnet.channel_size
        evolvable_resnet.remove_block()
        assert evolvable_resnet.channel_size >= initial_channels


######### Test add_channel #########
class TestEvolvableResNetAddChannel:
    @pytest.mark.parametrize(
        "input_shape, channel_size, kernel_size, stride_size, num_outputs, num_blocks",
        [
            ([3, 32, 32], 64, 3, 1, 10, 2),
            ([3, 64, 64], 128, 3, 2, 10, 3),
        ],
    )
    def test_add_channel(
        self,
        input_shape,
        channel_size,
        kernel_size,
        stride_size,
        num_outputs,
        num_blocks,
        device,
    ):
        evolvable_resnet = EvolvableResNet(
            input_shape=input_shape,
            channel_size=channel_size,
            kernel_size=kernel_size,
            stride_size=stride_size,
            num_outputs=num_outputs,
            num_blocks=num_blocks,
            device=device,
        )
        initial_channel_size = evolvable_resnet.channel_size
        result = evolvable_resnet.add_channel()
        numb_new_channels = result["numb_new_channels"]
        assert evolvable_resnet.channel_size == initial_channel_size + numb_new_channels

    def test_add_channel_hard_limit_not_exceeded(self, device):
        """Covers add_channel when channel_size + numb_new_channels >= max_channel_size."""
        evolvable_resnet = EvolvableResNet(
            input_shape=[3, 32, 32],
            channel_size=248,
            kernel_size=3,
            stride_size=1,
            num_outputs=10,
            num_blocks=2,
            max_channel_size=256,
            device=device,
        )
        evolvable_resnet.add_channel(numb_new_channels=32)
        assert evolvable_resnet.channel_size <= 256


######### Test remove_channel #########
class TestEvolvableResNetRemoveChannel:
    @pytest.mark.parametrize(
        "input_shape, channel_size, kernel_size, stride_size, num_outputs, num_blocks",
        [
            ([3, 32, 32], 128, 3, 1, 10, 2),
            ([3, 64, 64], 128, 3, 2, 10, 3),
        ],
    )
    def test_remove_channel(
        self,
        input_shape,
        channel_size,
        kernel_size,
        stride_size,
        num_outputs,
        num_blocks,
        device,
    ):
        evolvable_resnet = EvolvableResNet(
            input_shape=input_shape,
            channel_size=channel_size,
            kernel_size=kernel_size,
            stride_size=stride_size,
            num_outputs=num_outputs,
            num_blocks=num_blocks,
            device=device,
        )
        initial_channel_size = evolvable_resnet.channel_size
        result = evolvable_resnet.remove_channel()
        numb_new_channels = result["numb_new_channels"]
        assert evolvable_resnet.channel_size == initial_channel_size - numb_new_channels

    def test_remove_channel_hard_limit_not_below_min(self, device):
        """Covers remove_channel when channel_size - numb_new_channels <= min_channel_size."""
        evolvable_resnet = EvolvableResNet(
            input_shape=[3, 32, 32],
            channel_size=40,
            kernel_size=3,
            stride_size=1,
            num_outputs=10,
            num_blocks=2,
            min_channel_size=32,
            device=device,
        )
        initial = evolvable_resnet.channel_size
        evolvable_resnet.remove_channel(numb_new_channels=16)
        assert evolvable_resnet.channel_size == initial


######### Test net_config #########
class TestEvolvableResNetNetConfig:
    def test_net_config_excludes_attrs(self, device):
        """Covers net_config property popping num_inputs, num_outputs, device, name."""
        evolvable_resnet = EvolvableResNet(
            input_shape=[3, 32, 32],
            channel_size=64,
            kernel_size=3,
            stride_size=1,
            num_outputs=10,
            num_blocks=2,
            device=device,
        )
        net_config = evolvable_resnet.net_config
        assert "num_outputs" not in net_config
        assert "device" not in net_config
        assert "name" not in net_config
        assert "input_shape" in net_config or "channel_size" in net_config


######### Test clone #########
class TestEvolvableResNetClone:
    @pytest.mark.parametrize(
        "input_shape, channel_size, kernel_size, stride_size, num_outputs, num_blocks",
        [
            ([3, 32, 32], 64, 3, 1, 10, 2),
            ([3, 64, 64], 128, 3, 2, 10, 3),
        ],
    )
    def test_clone_instance(
        self,
        input_shape,
        channel_size,
        kernel_size,
        stride_size,
        num_outputs,
        num_blocks,
        device,
    ):
        evolvable_resnet = EvolvableResNet(
            input_shape=input_shape,
            channel_size=channel_size,
            kernel_size=kernel_size,
            stride_size=stride_size,
            num_outputs=num_outputs,
            num_blocks=num_blocks,
            device=device,
        )
        original_net_dict = dict(evolvable_resnet.model.named_parameters())
        clone = evolvable_resnet.clone()
        clone_net = clone.model
        assert isinstance(clone, EvolvableResNet)
        assert_state_dicts_equal(clone.state_dict(), evolvable_resnet.state_dict())
        for key, param in clone_net.named_parameters():
            torch.testing.assert_close(param, original_net_dict[key])
