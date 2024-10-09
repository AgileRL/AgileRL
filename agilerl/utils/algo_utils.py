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


def key_in_nested_dict(nested_dict, target):
    """Helper function to determine if key is in nested dictionary

    :param nested_dict: Nested dictionary
    :type nested_dict: Dict[str, Dict[str, ...]]
    :param target: Target string
    :type target: str
    """
    for k, v in nested_dict.items():
        if k == target:
            return True
        if isinstance(v, dict):
            return key_in_nested_dict(v, target)
    return False
