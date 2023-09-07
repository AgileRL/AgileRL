class Sampler:
    """Sampler class to handle both standard and distributed training."""

    def __init__(self, distributed=False, memory=None, dataset=None, dataloader=None):
        self.distributed = distributed
        self.memory = memory
        self.dataset = dataset
        self.dataloader = dataloader

        if self.distributed:
            self.sample = self.sample_distributed
        else:
            self.sample = self.sample_standard

    def sample_standard(self, batch_size):
        return self.memory.sample(batch_size)

    def sample_distributed(self, batch_size):
        self.dataset.batch_size = batch_size
        return next(iter(self.dataloader))
