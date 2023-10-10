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
        self.distributed = distributed
        self.per = per
        self.n_step = n_step
        self.memory = memory
        self.dataset = dataset
        self.dataloader = dataloader

        if self.distributed:
            self.sample = self.sample_distributed
        elif self.per:
            self.sample = self.sample_per
        elif self.n_step:
            self.sample = self.sample_n_step
        else:
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
