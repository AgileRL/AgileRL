# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterator
from typing import Any

import torch
from torch.utils.data import Dataset, IterableDataset

from agilerl.data.rl_data import DataPoint, Iterable_RL_Dataset, List_RL_Dataset


class GeneralIterDataset(IterableDataset):
    def __init__(
        self, rl_dataset: Iterable_RL_Dataset, device: torch.device | str
    ) -> None:
        self.rl_dataset = rl_dataset
        self.device = device

    def __iter__(self) -> Iterator[DataPoint]:
        return self

    def __next__(self) -> DataPoint:
        return self.rl_dataset.sample_item()

    def collate(self, items: list[Any]) -> dict[str, torch.Tensor]:
        return self.rl_dataset.collate(items, self.device)

    def collate_simple(self, items: list[Any]) -> list[Any]:
        return items


class GeneralDataset(Dataset):
    def __init__(self, rl_dataset: List_RL_Dataset, device: torch.device | str) -> None:
        self.rl_dataset = rl_dataset
        self.device = device

    def __len__(self) -> int:
        return self.rl_dataset.size()

    def __getitem__(self, index: int) -> DataPoint:
        return self.rl_dataset.get_item(index)

    def collate(self, items: list[Any]) -> dict[str, torch.Tensor]:
        return self.rl_dataset.collate(items, self.device)

    def collate_simple(self, items: list[Any]) -> list[Any]:
        return items
