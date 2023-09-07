import os
import pickle as pkl
import sys
from collections import defaultdict
from typing import Any, Dict, List

import torch


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


def migrate_pkl(pkl_path):
    from wordle import wordle_game

    sys.modules["wordle.wordle_mdp"] = wordle_game
    with open(pkl_path, "rb") as f:
        d = pkl.load(f)
    del sys.modules["wordle.wordle_mdp"]
    with open(pkl_path, "wb") as f:
        pkl.dump(d, f)


def convert_old_ad_model(model_path, dataset, multiple_transformers=False):
    w = torch.load(model_path, map_location="cpu")
    embs = w["model.transformer.wte.weight"]
    if embs.shape[0] < dataset.tokenizer.num_tokens():
        w["model.transformer.wte.weight"] = torch.cat(
            (
                embs,
                torch.zeros(
                    dataset.tokenizer.num_tokens() - embs.shape[0], embs.shape[1]
                ),
            ),
            dim=0,
        )
    head = w["model.lm_head.weight"]
    if head.shape[0] < dataset.tokenizer.num_tokens():
        w["model.lm_head.weight"] = torch.cat(
            (
                head,
                torch.zeros(
                    dataset.tokenizer.num_tokens() - head.shape[0], head.shape[1]
                ),
            ),
            dim=0,
        )
    if multiple_transformers:
        for k, v in list(w.items()):
            if k.startswith("model."):
                w["lm_policy." + k[len("model.") :]] = v.clone()
                w["lm_target." + k[len("model.") :]] = v.clone()
    torch.save(w, model_path)


def stack_dicts(dicts: List[Dict[str, Any]]):
    stacked = defaultdict(list)
    for item in dicts:
        for k, v in item.items():
            stacked[k].append(v)
    # rough quick check for dicts have the same set of keys, could be more rigorous
    assert len(set(map(len, stacked.values()))) == 1
    return stacked


def unstack_dicts(stacked_dict: Dict[str, List[Any]]):
    dict_len = set(map(lambda x: len(list(x)), stacked_dict.values()))
    assert len(dict_len) == 1
    dicts = [{} for _ in range(dict_len.pop())]
    for k, v in stacked_dict.items():
        for i, item in enumerate(v):
            dicts[i][k] = item
    return dicts


class PrecisionRecallAcc:
    def __init__(self, classes) -> None:
        self.precisions = {class_: [0, 0] for class_ in classes}
        self.recalls = {class_: [0, 0] for class_ in classes}
        self.correct = 0
        self.total = 0

    def add_item(self, prediction_type: Any, actual_type: Any, correct: bool):
        correct = int(correct)
        if prediction_type in self.precisions:
            self.precisions[prediction_type][0] += correct
            self.precisions[prediction_type][1] += 1
        if actual_type in self.recalls:
            self.recalls[actual_type][0] += correct
            self.recalls[actual_type][1] += 1
        self.correct += correct
        self.total += 1

    def return_summary(self):
        logs = {}
        logs["accuracy"] = (self.correct / self.total, self.total)
        for k, (a, b) in self.precisions.items():
            logs[str(k) + "_precision"] = (a / b if b != 0 else -1, b)
        for k, (a, b) in self.recalls.items():
            logs[str(k) + "_recall"] = (a / b if b != 0 else -1, b)
        return logs
