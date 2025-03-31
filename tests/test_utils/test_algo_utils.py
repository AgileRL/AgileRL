import glob
import os

import pytest
import torch
import torch.nn as nn
from accelerate import Accelerator
from torch.optim.lr_scheduler import SequentialLR

from agilerl.utils.algo_utils import (
    CosineLRScheduleConfig,
    create_warmup_cosine_scheduler,
    remove_nested_files,
    stack_and_pad_experiences,
    unwrap_optimizer,
)


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


def test_stack_and_pad_experiences_with_padding():
    tensor1 = torch.tensor([[1, 2, 3], [4, 5, 6]])
    tensor2 = torch.tensor([[8]])
    tensor3 = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    tensor4 = torch.tensor([1, 3, 4])  # This tensor should be returned without change
    tensor5 = torch.tensor([[10, 11, 12]])
    tensor6 = torch.tensor([[13, 14, 15, 16, 17]])
    tensor_list = [[tensor1, tensor2, tensor3], tensor4, [tensor5, tensor6]]
    stacked_tensor, unchanged_tensor, stacked_tensor_2 = stack_and_pad_experiences(
        *tensor_list, padding_values=[0, 0, 99]
    )
    assert torch.equal(unchanged_tensor, tensor4)
    assert torch.equal(
        stacked_tensor,
        torch.tensor(
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
        ),
    )
    assert torch.equal(
        stacked_tensor_2, torch.tensor([[10, 11, 12, 99, 99], [13, 14, 15, 16, 17]])
    )


def test_stack_and_pad_experiences_without_padding():
    tensor1 = torch.tensor([[1, 2, 3]])
    tensor2 = torch.tensor([[2, 3, 4]])
    tensor3 = torch.tensor([[5, 6, 7]])  # This tensor should be returned without change
    tensor_list = [[tensor1, tensor2, tensor3]]
    stacked_tensor = stack_and_pad_experiences(*tensor_list, padding_values=[0, 0])[0]
    assert stacked_tensor.shape == (3, 3)
    assert torch.equal(stacked_tensor, torch.tensor([[1, 2, 3], [2, 3, 4], [5, 6, 7]]))


def test_create_warmup_cosine_scheduler():
    basic_net = nn.Sequential(nn.Linear(1, 1))
    optimizer = torch.optim.Adam(basic_net.parameters(), lr=0.01)

    lr_scheduler = create_warmup_cosine_scheduler(
        optimizer,
        CosineLRScheduleConfig(num_epochs=10, warmup_proportion=0.05),
        0.01,
        0.1,
    )
    assert isinstance(lr_scheduler, SequentialLR)


def test_remove_nested_files():
    """Test the remove_nested_files function"""
    # Create test directory structure
    os.makedirs("test_dir/nested_dir", exist_ok=True)

    # Create some test files
    with open("test_dir/file1.txt", "w") as f:
        f.write("test1")
    with open("test_dir/nested_dir/file2.txt", "w") as f:
        f.write("test2")

    # Test removing the directory structure
    files = glob.glob("test_dir/*")
    remove_nested_files(files)

    # Verify files and directories were removed
    assert not os.path.exists("test_dir/file1.txt")
    assert not os.path.exists("test_dir/nested_dir/file2.txt")
    assert not os.path.exists("test_dir/nested_dir")
