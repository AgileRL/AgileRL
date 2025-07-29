import os
from typing import Any, Dict, List, Optional

from accelerate import Accelerator


def convert_path(path: Optional[str]) -> Optional[str]:
    """Converts a path to an absolute path.

    :param path: Path to convert.
    :type path: Optional[str]

    :return: Absolute path.
    :rtype: Optional[str]
    """
    if path is None:
        return None
    return os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../", path)


def add_system_configs(cfg: Dict[str, Any], accelerator: Accelerator) -> Dict[str, Any]:
    """Adds system configurations to the configuration dictionary.

    :param cfg: Configuration dictionary.
    :type cfg: Dict[str, Any]
    :param accelerator: Accelerator object.
    :type accelerator: Accelerator

    :return: Configuration dictionary with system configurations.
    :rtype: Dict[str, Any]
    """
    cfg["system"] = {}
    cfg["system"]["device"] = str(accelerator.device)
    cfg["system"]["num_processes"] = accelerator.num_processes
    cfg["system"]["use_fp16"] = accelerator.mixed_precision != "no"
    return cfg["system"]


def to_bin(n: int, pad_to_size: Optional[int] = None) -> List[int]:
    """Converts a number to a binary list.

    :param n: Number to convert.
    :type n: int
    :param pad_to_size: Size to pad the binary list to.
    :type pad_to_size: Optional[int]

    :return: Binary list.
    :rtype: List[int]
    """
    bins = to_bin(n // 2) + [n % 2] if n > 1 else [n]
    if pad_to_size is None:
        return bins
    return ([0] * (pad_to_size - len(bins))) + bins


def strip_from_end(str_item: str, strip_key: str) -> str:
    """Strips a string from the end.

    :param str_item: String to strip.
    :type str_item: str
    :param strip_key: Key to strip.
    :type strip_key: str

    :return: Stripped string.
    :rtype: str
    """
    return strip_from_beginning(str_item[::-1], strip_key[::-1])[::-1]


def strip_from_beginning(str_item: str, strip_key: str) -> str:
    """Strips a string from the beginning.

    :param str_item: String to strip.
    :type str_item: str
    :param strip_key: Key to strip.
    :type strip_key: str

    :return: Stripped string.
    :rtype: str
    """
    if str_item[: len(strip_key)] == strip_key:
        return str_item[len(strip_key) :]
    return str_item
