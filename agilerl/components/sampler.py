from torch.utils.data import DataLoader

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
            assert isinstance(
                self.dataset, ReplayDataset
            ), "Dataset must be agilerl ReplayDataset."
            assert isinstance(
                self.dataloader, DataLoader
            ), "Dataset must be torch DataLoader."
            self.sample = self.sample_distributed
        elif self.per:
            assert isinstance(
                self.memory, PrioritizedReplayBuffer
            ), "Memory must be agilerl PrioritizedReplayBuffer."
            self.sample = self.sample_per
        elif self.n_step:
            assert isinstance(
                self.memory, MultiStepReplayBuffer
            ), "Memory must be agilerl MultiStepReplayBuffer."
            self.sample = self.sample_n_step
        else:
            assert isinstance(
                self.memory, ReplayBuffer
            ), "Memory must be agilerl ReplayBuffer."
            self.sample = self.sample_standard

    def sample_standard(self, batch_size):
        return self.memory.sample(batch_size)

    def sample_distributed(self, batch_size):
        self.dataset.batch_size = batch_size
        return next(iter(self.dataloader))

    def sample_per(self, batch_size, beta):
        return self.memory.sample(batch_size, beta)

    def sample_n_step(self, idxs):
        return self.memory.sample_from_indices(idxs)
