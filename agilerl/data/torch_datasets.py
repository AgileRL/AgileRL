from typing import Union

import torch
from torch.utils.data import Dataset, IterableDataset

from agilerl.data.rl_data import Iterable_RL_Dataset, List_RL_Dataset


class GeneralIterDataset(IterableDataset):
    def __init__(
        self, rl_dataset: Iterable_RL_Dataset, device: Union[torch.device, str]
    ):
        self.rl_dataset = rl_dataset
        self.device = device

    def __iter__(self):
        return self

    def __next__(self):
        return self.rl_dataset.sample_item()

    def collate(self, items):
        return self.rl_dataset.collate(items, self.device)

    def collate_simple(self, items):
        return items


class GeneralDataset(Dataset):
    def __init__(self, rl_dataset: List_RL_Dataset, device: Union[torch.device, str]):
        self.rl_dataset = rl_dataset
        self.device = device

    def __len__(self):
        return self.rl_dataset.size()

    def __getitem__(self, i):
        return self.rl_dataset.get_item(i)

    def collate(self, items):
        return self.rl_dataset.collate(items, self.device)

    def collate_simple(self, items):
        return items
