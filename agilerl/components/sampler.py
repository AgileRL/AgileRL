import warnings
from typing import Any, List, Optional, Union

import torch
from tensordict import TensorDict
from torch.utils.data import DataLoader

from agilerl.components import (
    MultiAgentReplayBuffer,
    MultiStepReplayBuffer,
    PrioritizedReplayBuffer,
    ReplayBuffer,
)
from agilerl.components.data import ReplayDataset
from agilerl.typing import ExperiencesType

BufferType = Union[
    ReplayBuffer, MultiAgentReplayBuffer, PrioritizedReplayBuffer, MultiStepReplayBuffer
]


class Sampler:
    """Sampler class to handle both standard and distributed training.

    :param memory: Replay buffer memory, defaults to None
    :type memory: Optional[Union[ReplayBuffer, MultiAgentReplayBuffer, PrioritizedReplayBuffer, MultiStepReplayBuffer]], optional
    :param dataset: Dataset for distributed sampling, defaults to None
    :type dataset: Optional[ReplayDataset], optional
    :param dataloader: DataLoader for distributed sampling, defaults to None
    :type dataloader: Optional[DataLoader], optional
    :raises AssertionError: If neither memory nor (dataset and dataloader) are provided
    """

    @staticmethod
    def tensordict_collate_fn(
        batch: List[TensorDict],
    ) -> Union[TensorDict, List[TensorDict]]:
        """Custom collate function that properly handles TensorDict objects.

        :param batch: List of TensorDict objects to collate
        :type batch: List[TensorDict]
        :return: Either a single TensorDict or a list of TensorDicts
        :rtype: Union[TensorDict, List[TensorDict]]
        """
        return TensorDict(
            {key: torch.stack([b[key] for b in batch]) for key in batch[0].keys()},
            batch_size=len(batch),
        )

    def __init__(
        self,
        memory: Optional[BufferType] = None,
        dataset: Optional[ReplayDataset] = None,
        dataloader: Optional[DataLoader] = None,
    ) -> None:

        assert (memory is not None) or (
            (dataset is not None) and (dataloader is not None)
        ), "Sampler needs to be initialized with either 'memory' or ('dataset' AND 'dataloader')."

        self.distributed = (
            dataloader is not None
            and dataset is not None
            and isinstance(dataloader, DataLoader)
        )
        self.per = isinstance(memory, PrioritizedReplayBuffer)
        self.n_step = isinstance(memory, MultiStepReplayBuffer)
        self.memory = memory
        self.dataset = dataset

        # Process the dataloader
        if self.distributed:
            # Need to use a custom collate function for TensorDict buffers
            if isinstance(dataset.buffer, ReplayBuffer):
                self.dataloader = self._replace_dataloader_collate_fn(dataloader)
            else:
                self.dataloader = dataloader

            if not isinstance(self.dataset, ReplayDataset):
                warnings.warn("Dataset is not an agilerl ReplayDataset.")
            if not isinstance(self.dataloader, DataLoader):
                warnings.warn("Dataset is not a torch DataLoader object.")

            self.sample = self.sample_distributed
        else:
            self.dataloader = dataloader
            if self.per:
                if not isinstance(self.memory, PrioritizedReplayBuffer):
                    warnings.warn("Memory is not an agilerl PrioritizedReplayBuffer.")
                self.sample = self.sample_per

            elif self.n_step:
                if not isinstance(self.memory, MultiStepReplayBuffer):
                    warnings.warn("Memory is not an agilerl MultiStepReplayBuffer.")
                self.sample = self.sample_n_step

            else:
                if not isinstance(self.memory, (ReplayBuffer, MultiAgentReplayBuffer)):
                    warnings.warn(
                        "Memory is not an agilerl ReplayBuffer or MultiAgentReplayBuffer."
                    )
                self.sample = self.sample_standard

    def _replace_dataloader_collate_fn(self, dataloader: DataLoader) -> DataLoader:
        """Create a new DataLoader with the tensordict_collate_fn while preserving all other parameters.

        :param dataloader: Original DataLoader
        :type dataloader: DataLoader
        :return: New DataLoader with tensordict_collate_fn
        :rtype: DataLoader
        """
        # Create a simplified set of parameters for the new dataloader
        params = {
            "dataset": dataloader.dataset,
            "batch_size": dataloader.batch_size,
            "collate_fn": self.tensordict_collate_fn,
        }

        # Optional parameters - only include if they are not default values
        if dataloader.num_workers != 0:
            params["num_workers"] = dataloader.num_workers

        if dataloader.pin_memory:
            params["pin_memory"] = True

        if dataloader.drop_last:
            params["drop_last"] = True

        if hasattr(dataloader, "prefetch_factor") and dataloader.prefetch_factor != 2:
            params["prefetch_factor"] = dataloader.prefetch_factor

        if hasattr(dataloader, "persistent_workers") and dataloader.persistent_workers:
            params["persistent_workers"] = True

        # Create a new DataLoader
        return DataLoader(**params)

    def sample_standard(
        self, batch_size: int, return_idx: bool = False
    ) -> ExperiencesType:
        """Sample a batch of experiences from the standard replay buffer.

        :param batch_size: Size of the batch to sample
        :type batch_size: int
        :param return_idx: Whether to return indices, defaults to False
        :type return_idx: bool, optional
        :return: Sampled batch of experiences
        :rtype: TensorDict
        """
        return self.memory.sample(batch_size, return_idx)

    def sample_distributed(
        self, batch_size: int, return_idx: Optional[bool] = None
    ) -> TensorDict:
        """Sample a batch of experiences from the distributed dataset.

        :param batch_size: Size of the batch to sample
        :type batch_size: int
        :param return_idx: Not used in distributed sampling, defaults to None
        :type return_idx: Optional[bool], optional
        :return: Sampled batch of experiences
        :rtype: TensorDict
        """
        self.dataset.batch_size = batch_size
        return next(iter(self.dataloader))

    def sample_per(self, batch_size: int, beta: float) -> ExperiencesType:
        """Sample a batch of experiences from the Prioritized Experience Replay buffer.

        :param batch_size: Size of the batch to sample
        :type batch_size: int
        :param beta: Importance-sampling weight
        :type beta: float
        :return: Sampled batch of experiences, indices, and importance-sampling weights
        :rtype: TensorDict
        """
        return self.memory.sample(batch_size, beta)

    def sample_n_step(self, idxs: Any) -> ExperiencesType:
        """Sample a batch of experiences from the n-step replay buffer.

        :param idxs: Indices to sample from
        :type idxs: Any
        :return: Sampled batch of experiences
        :rtype: TensorDict
        """
        return self.memory.sample_from_indices(idxs)

    @classmethod
    def create_dataloader(
        cls, dataset: ReplayDataset, batch_size: Optional[int] = None, **kwargs
    ) -> DataLoader:
        """Helper method to create a DataLoader with the appropriate collate function.

        :param dataset: Dataset to create a DataLoader for
        :type dataset: ReplayDataset
        :param batch_size: Batch size for the DataLoader, defaults to None
        :type batch_size: Optional[int], optional
        :param kwargs: Additional arguments to pass to the DataLoader
        :return: DataLoader with tensordict_collate_fn
        :rtype: DataLoader
        """
        # Ensure we don't override collate_fn if explicitly provided
        if "collate_fn" not in kwargs:
            kwargs["collate_fn"] = cls.tensordict_collate_fn

        return DataLoader(dataset, batch_size=batch_size, **kwargs)
