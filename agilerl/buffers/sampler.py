import warnings
from typing import Any, Optional, Tuple, Union

from torch.utils.data import DataLoader

from agilerl.components.multi_agent_replay_buffer import MultiAgentReplayBuffer
from agilerl.components.replay_buffer import (
    MultiStepReplayBuffer,
    PrioritizedReplayBuffer,
    ReplayBuffer,
)
from agilerl.components.replay_data import ReplayDataset

BufferType = Union[
    ReplayBuffer, MultiAgentReplayBuffer, PrioritizedReplayBuffer, MultiStepReplayBuffer
]


class Sampler:
    """Sampler class to handle both standard and distributed training.

    :param distributed: Whether to use distributed sampling, defaults to False
    :type distributed: bool, optional
    :param per: Whether to use Prioritized Experience Replay (PER), defaults to False
    :type per: bool, optional
    :param n_step: Whether to use n-step returns, defaults to False
    :type n_step: bool, optional
    :param memory: Replay buffer memory, defaults to None
    :type memory: Optional[Union[ReplayBuffer, MultiAgentReplayBuffer, PrioritizedReplayBuffer, MultiStepReplayBuffer]], optional
    :param dataset: Dataset for distributed sampling, defaults to None
    :type dataset: Optional[ReplayDataset], optional
    :param dataloader: DataLoader for distributed sampling, defaults to None
    :type dataloader: Optional[DataLoader], optional
    :raises AssertionError: If neither memory nor (dataset and dataloader) are provided
    """

    def __init__(
        self,
        distributed: bool = False,
        per: bool = False,
        n_step: bool = False,
        memory: Optional[BufferType] = None,
        dataset: Optional[ReplayDataset] = None,
        dataloader: Optional[DataLoader] = None,
    ) -> None:

        assert (memory is not None) or (
            (dataset is not None) and (dataloader is not None)
        ), "Sampler needs to be initialized with either 'memory' or ('dataset' AND 'dataloader')."

        self.distributed = distributed
        self.per = per
        self.n_step = n_step
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
            if not isinstance(self.memory, MultiStepReplayBuffer):
                warnings.warn("Memory is not an agilerl MultiStepReplayBuffer.")
            self.sample = self.sample_n_step

        else:
            if not isinstance(self.memory, (ReplayBuffer, MultiAgentReplayBuffer)):
                warnings.warn(
                    "Memory is not an agilerl ReplayBuffer or MultiAgentReplayBuffer."
                )
            self.sample = self.sample_standard

    def sample_standard(self, batch_size: int, return_idx: bool = False) -> Any:
        """Sample a batch of experiences from the standard replay buffer.

        :param batch_size: Size of the batch to sample
        :type batch_size: int
        :param return_idx: Whether to return indices, defaults to False
        :type return_idx: bool, optional
        :return: Sampled batch of experiences
        :rtype: Any
        """
        return self.memory.sample(batch_size, return_idx)

    def sample_distributed(
        self, batch_size: int, return_idx: Optional[bool] = None
    ) -> Any:
        """Sample a batch of experiences from the distributed dataset.

        :param batch_size: Size of the batch to sample
        :type batch_size: int
        :param return_idx: Not used in distributed sampling, defaults to None
        :type return_idx: Optional[bool], optional
        :return: Sampled batch of experiences
        :rtype: Any
        """
        self.dataset.batch_size = batch_size
        return next(iter(self.dataloader))

    def sample_per(self, batch_size: int, beta: float) -> Tuple[Any, Any, Any]:
        """Sample a batch of experiences from the Prioritized Experience Replay buffer.

        :param batch_size: Size of the batch to sample
        :type batch_size: int
        :param beta: Importance-sampling weight
        :type beta: float
        :return: Sampled batch of experiences, indices, and importance-sampling weights
        :rtype: Tuple[Any, Any, Any]
        """
        return self.memory.sample(batch_size, beta)

    def sample_n_step(self, idxs: Any) -> Any:
        """Sample a batch of experiences from the n-step replay buffer.

        :param idxs: Indices to sample from
        :type idxs: Any
        :return: Sampled batch of experiences
        :rtype: Any
        """
        return self.memory.sample_from_indices(idxs)
