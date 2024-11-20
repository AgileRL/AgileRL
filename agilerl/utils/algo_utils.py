from collections import OrderedDict
from typing import Union, Dict, Any
import torch
import numpy as np
from accelerate import Accelerator
from accelerate.optimizer import AcceleratedOptimizer
from torch.optim import Optimizer
from torch.nn import Module
from torch._dynamo import OptimizedModule

from agilerl.networks.base import EvolvableModule
from agilerl.typing import NumpyObsType, TorchObsType, NetworkType, OptimizerType

def unwrap_optimizer(
        optimizer: OptimizerType,
        network: NetworkType,
        lr: float
        ) -> Optimizer:
    """Unwraps AcceleratedOptimizer to get the underlying optimizer.
    
    :param optimizer: AcceleratedOptimizer
    :type optimizer: AcceleratedOptimizer
    :param network: Network or list of networks
    :type network: Union[Module, List[Module], Tuple[Module, ...]]
    :param lr: Learning rate
    :type lr: float
    :return: Unwrapped optimizer
    rtype: Optimizer
    """
    if isinstance(optimizer, AcceleratedOptimizer):
        if isinstance(network, (list, tuple)):
            optim_arg = [{"params": net.parameters(), "lr": lr} for net in network]
            unwrapped_optimizer: Optimizer = type(optimizer.optimizer)(optim_arg)
        else:
            unwrapped_optimizer: Optimizer = type(optimizer.optimizer)(network.parameters(), lr=lr)
        unwrapped_optimizer.load_state_dict(optimizer.state_dict())
        return unwrapped_optimizer
    else:
        return optimizer
    
def recursive_check_module_attrs(obj: Any) -> bool:
    """Recursively check if the object has any attributes that are EvolvableModule's or Optimizer's.

    :param obj: The object to check for EvolvableModule's or Optimizer's.
    :type obj: Any
    :return: True if the object has any attributes that are EvolvableModule's or Optimizer's, False otherwise.
    :rtype: bool
    """
    if isinstance(obj, (OptimizedModule, EvolvableModule, Optimizer)):
        return True
    if isinstance(obj, dict):
        return any(recursive_check_module_attrs(v) for v in obj.values())
    if isinstance(obj, list):
        return any(recursive_check_module_attrs(v) for v in obj)
    return False

def chkpt_attribute_to_device(chkpt_dict: Dict[str, torch.Tensor], device: str) -> Dict[str, Any]:
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

def key_in_nested_dict(nested_dict: Dict[str, Any], target: str) -> bool:
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

def compile_model(model: Module, mode: Union[str, None] = "default") -> Module:
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

def remove_compile_prefix(state_dict: Dict[str, Any]) -> Dict[str, Any]:
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

def obs_to_tensor(obs: NumpyObsType, device: Union[str, torch.device]) -> TorchObsType:
    """
    Moves the observation to the given device.

    :param obs:
    :type obs: NumpyObsType
    :param device: PyTorch device
    :type device: Union[str, torch.device]
    :return: PyTorch tensor of the observation on a desired device.
    :rtype: TorchObsType
    """
    if isinstance(obs, torch.Tensor):
        return obs.to(device)
    if isinstance(obs, np.ndarray):
        return torch.as_tensor(obs, device=device)
    elif isinstance(obs, dict):
        return {key: torch.as_tensor(_obs, device=device) for (key, _obs) in obs.items()}
    else:
        raise Exception(f"Unrecognized type of observation {type(obs)}")