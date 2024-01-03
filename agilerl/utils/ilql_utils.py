import os


def convert_path(path):
    if path is None:
        return None
    return os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../", path)


def add_system_configs(cfg, accelerator):
    cfg["system"] = {}
    cfg["system"]["device"] = str(accelerator.device)
    cfg["system"]["num_processes"] = accelerator.num_processes
    cfg["system"]["use_fp16"] = accelerator.use_fp16
    return cfg["system"]


def to_bin(n, pad_to_size=None):
    bins = to_bin(n // 2) + [n % 2] if n > 1 else [n]
    if pad_to_size is None:
        return bins
    return ([0] * (pad_to_size - len(bins))) + bins


def strip_from_end(str_item, strip_key):
    return strip_from_beginning(str_item[::-1], strip_key[::-1])[::-1]


def strip_from_beginning(str_item, strip_key):
    if str_item[: len(strip_key)] == strip_key:
        return str_item[len(strip_key) :]
    return str_item
