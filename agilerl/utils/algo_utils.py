from collections import OrderedDict

import torch
from accelerate import Accelerator
from accelerate.optimizer import AcceleratedOptimizer


def unwrap_optimizer(optimizer, network, lr):
    if isinstance(optimizer, AcceleratedOptimizer):
        if isinstance(network, (list, tuple)):
            optim_arg = [{"params": net.parameters(), "lr": lr} for net in network]
            unwrapped_optimizer = type(optimizer.optimizer)(optim_arg)
        else:
            unwrapped_optimizer = type(optimizer.optimizer)(network.parameters(), lr=lr)
        unwrapped_optimizer.load_state_dict(optimizer.state_dict())
        return unwrapped_optimizer
    else:
        return optimizer


def chkpt_attribute_to_device(chkpt_dict, device):
    """Place checkpoint attributes on device. Used when loading saved agents.

    :param chkpt_dict: Checkpoint dictionary
    :type chkpt_dict: dict
    :param device: Device for accelerated computing, 'cpu' or 'cuda'
    :type device: str
    """
    for key, value in chkpt_dict.items():
        if hasattr(value, "device") and not isinstance(value, Accelerator):
            chkpt_dict[key] = value.to(device)
    return chkpt_dict


def compile_model(model, mode: str | None = "default"):
    """Compiles torch model if not already compiled

    :param model: torch model
    :type model: nn.Module
    :param mode: torch compile mode, defaults to "default"
    :type mode: str, optional
    :return: compiled model
    :rtype: OptimizedModule
    """
    return (
        torch.compile(model, mode=mode)
        if not isinstance(model, torch._dynamo.eval_frame.OptimizedModule)
        and mode is not None
        else model
    )


def remove_compile_prefix(state_dict):
    """Removes _orig_mod prefix on state dict created by torch compile

    :param state_dict: model state dict
    :type state_dict: dict
    :return: state dict with prefix removed
    :rtype: dict
    """
    return OrderedDict(
        [
            (k.split(".", 1)[1], v) if k.startswith("_orig_mod") else (k, v)
            for k, v in state_dict.items()
        ]
    )
