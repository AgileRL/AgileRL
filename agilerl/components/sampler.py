import warnings

from torch.utils.data import DataLoader

from agilerl.components.multi_agent_replay_buffer import MultiAgentReplayBuffer
from agilerl.components.replay_buffer import (
    MultiStepReplayBuffer,
    PrioritizedReplayBuffer,
    ReplayBuffer,
)
from agilerl.components.replay_data import ReplayDataset


class Sampler:
    """Sampler class to handle both standard and distributed training."""

    def __init__(
        self,
        distributed=False,
        per=False,
        n_step=False,
        memory=None,
        dataset=None,
        dataloader=None,
    ):
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

    def sample_standard(self, batch_size, return_idx=False):
        return self.memory.sample(batch_size, return_idx)

    def sample_distributed(self, batch_size, return_idx=None):
        self.dataset.batch_size = batch_size
        return next(iter(self.dataloader))

    def sample_per(self, batch_size, beta):
        return self.memory.sample(batch_size, beta)

    def sample_n_step(self, idxs):
        return self.memory.sample_from_indices(idxs)
