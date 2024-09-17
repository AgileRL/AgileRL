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


def clip_mod(key_words_dict):
    """clip torch dynamo (eval_frame) Optimized Module prefix
    The prefix is added when the corresponding nn.Module (MARL algos)
    is "modified" by torch.compile
    :param key_words_dict: the compiled module's state_dict
    :type key_words_dict: str
    """
    return {k.split(".", 1)[1]: v for k, v in key_words_dict.items()}
