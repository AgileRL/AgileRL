import pytest
import torch
import torch.nn as nn
from accelerate import Accelerator

from agilerl.utils.algo_utils import stack_and_pad_experiences, unwrap_optimizer


@pytest.mark.parametrize("distributed", [(True), (False)])
def test_algo_utils_single_net(distributed):
    simple_net = nn.Sequential(nn.Linear(2, 3), nn.ReLU())
    lr = 0.01
    optimizer = torch.optim.Adam(simple_net.parameters(), lr=lr)
    if distributed:
        accelerator = Accelerator(device_placement=False)
        optimizer = accelerator.prepare(optimizer)
    else:
        accelerator = None

    unwrapped_optimizer = unwrap_optimizer(optimizer, simple_net, lr)
    assert isinstance(unwrapped_optimizer, torch.optim.Adam)


def test_algo_utils_multi_nets():
    simple_net = nn.Sequential(nn.Linear(2, 3), nn.ReLU())
    simple_net_two = nn.Sequential(nn.Linear(4, 3), nn.ReLU())
    lr = 0.01
    optimizer = torch.optim.Adam(
        [
            {"params": simple_net.parameters(), "lr": lr},
            {"params": simple_net_two.parameters(), "lr": lr},
        ]
    )
    accelerator = Accelerator(device_placement=False)
    optimizer = accelerator.prepare(optimizer)
    unwrapped_optimizer = unwrap_optimizer(optimizer, [simple_net, simple_net_two], lr)
    assert isinstance(unwrapped_optimizer, torch.optim.Adam)


def test_stack_and_pad_experiences():
    tensor1 = torch.tensor([[1, 2, 3], [4, 5, 6]])
    tensor2 = torch.tensor([8])
    tensor3 = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    tensor4 = torch.tensor([1, 3, 4])  # This tensor should be returned without change
    tensor_list = [[tensor1, tensor2, tensor3], tensor4]
    stacked_tensor, unchanged_tensor = stack_and_pad_experiences(
        *tensor_list, padding_values=[0, 0]
    )
    assert torch.equal(unchanged_tensor, tensor4)
    assert stacked_tensor == torch.tensor(
        [
            [
                1,
                2,
                3,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            ],
            [
                4,
                5,
                6,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            ],
            [
                8,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            ],
            [
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
            ],
        ]
    )
