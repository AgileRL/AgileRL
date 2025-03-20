import warnings
from typing import Any, Optional, Union

from tensordict import TensorDict
from torch.utils.data import DataLoader

from agilerl.components import (
    MultiAgentReplayBuffer,
    NStepReplayBuffer,
    PrioritizedReplayBuffer,
    ReplayBuffer,
)
from agilerl.components.data import ReplayDataset

BufferType = Union[
    ReplayBuffer, MultiAgentReplayBuffer, PrioritizedReplayBuffer, NStepReplayBuffer
]


class Sampler:
    """Sampler class to handle both standard and distributed training.

    :param memory: Replay buffer memory, defaults to None
    :type memory: Optional[Union[ReplayBuffer, MultiAgentReplayBuffer, PrioritizedReplayBuffer, NStepReplayBuffer]], optional
    :param dataset: Dataset for distributed sampling, defaults to None
    :type dataset: Optional[ReplayDataset], optional
    :param dataloader: DataLoader for distributed sampling, defaults to None
    :type dataloader: Optional[DataLoader], optional
    :raises AssertionError: If neither memory nor (dataset and dataloader) are provided
    """

    def __init__(
        self,
        memory: Optional[BufferType] = None,
        dataset: Optional[ReplayDataset] = None,
        dataloader: Optional[DataLoader] = None,
    ) -> None:

        assert (memory is not None) or (
            (dataset is not None) and (dataloader is not None)
        ), "Sampler needs to be initialized with either 'memory' or ('dataset' AND 'dataloader')."

        self.distributed = dataloader is not None and dataset is not None
        self.per = isinstance(memory, PrioritizedReplayBuffer)
        self.n_step = isinstance(memory, NStepReplayBuffer)
        self.memory = memory
        self.dataset = dataset
        self.dataloader = dataloader

        if self.distributed:
            if not isinstance(self.dataset, ReplayDataset):
                warnings.warn("Dataset is not an agilerl ReplayDataset.")
            if not isinstance(self.dataloader, DataLoader):
                warnings.warn("Dataset is not a torch DataLoader object.")
            self.sample = self.sample_distributed

        elif self.per:
            if not isinstance(self.memory, PrioritizedReplayBuffer):
                warnings.warn("Memory is not an agilerl PrioritizedReplayBuffer.")
            self.sample = self.sample_per

        elif self.n_step:
            if not isinstance(self.memory, NStepReplayBuffer):
                warnings.warn("Memory is not an agilerl NStepReplayBuffer.")
            self.sample = self.sample_n_step

        else:
            if not isinstance(self.memory, (ReplayBuffer, MultiAgentReplayBuffer)):
                warnings.warn(
                    "Memory is not an agilerl ReplayBuffer or MultiAgentReplayBuffer."
                )
            self.sample = self.sample_standard

    def sample_standard(self, batch_size: int, return_idx: bool = False) -> TensorDict:
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

    def sample_per(self, batch_size: int, beta: float) -> TensorDict:
        """Sample a batch of experiences from the Prioritized Experience Replay buffer.

        :param batch_size: Size of the batch to sample
        :type batch_size: int
        :param beta: Importance-sampling weight
        :type beta: float
        :return: Sampled batch of experiences, indices, and importance-sampling weights
        :rtype: TensorDict
        """
        return self.memory.sample(batch_size, beta)

    def sample_n_step(self, idxs: Any) -> TensorDict:
        """Sample a batch of experiences from the n-step replay buffer.

        :param idxs: Indices to sample from
        :type idxs: Any
        :return: Sampled batch of experiences
        :rtype: TensorDict
        """
        return self.memory.sample_from_indices(idxs)
