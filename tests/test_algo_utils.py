import pytest
import torch
import torch.nn as nn
from accelerate import Accelerator

from agilerl.utils.algo_utils import unwrap_optimizer


@pytest.mark.parametrize(
        "distributed",
        [
            (True),
            (False)
        ]
)
def test_algo_utils_single_net(distributed):
    simple_net = nn.Sequential(nn.Linear(2,3), nn.ReLU()) 
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
    simple_net = nn.Sequential(nn.Linear(2,3), nn.ReLU()) 
    simple_net_two = nn.Sequential(nn.Linear(4,3), nn.ReLU()) 
    lr = 0.01
    optimizer = torch.optim.Adam([{"params": simple_net.parameters(), "lr": lr},
                                 {"params": simple_net_two.parameters(), "lr": lr}])
    accelerator = Accelerator(device_placement=False)
    optimizer = accelerator.prepare(optimizer)
    unwrapped_optimizer = unwrap_optimizer(optimizer, [simple_net, simple_net_two], lr)
    assert isinstance(unwrapped_optimizer, torch.optim.Adam)
