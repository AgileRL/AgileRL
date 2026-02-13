import torch

from agilerl.typing import DeviceType


class RunningMeanStd:
    """Tracks mean, variance, and count of values using torch tensors.

    :param epsilon: Small value to avoid division by zero, defaults to 1e-4
    :type epsilon: float, optional
    :param shape: Shape of the tensor, defaults to ()
    :type shape: tuple[int, ...], optional
    :param device: Device to store the tensors, defaults to "cpu"
    :type device: DeviceType, optional
    :param dtype: Data type of the tensor, defaults to torch.float32
    :type dtype: torch.dtype, optional
    """

    def __init__(
        self,
        epsilon: float = 1e-4,
        shape: tuple[int, ...] = (),
        device: DeviceType = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:

        self.epsilon = epsilon
        self.device = device
        self.mean = torch.zeros(shape, dtype=dtype, device=device)
        self.var = torch.ones(shape, dtype=dtype, device=device)
        self.count = torch.tensor(epsilon, dtype=dtype, device=device)

    def update(self, x: torch.Tensor) -> None:
        """Update mean, variance, and count using a batch of samples.

        :param x: Batch of samples.
        :type x: torch.Tensor
        """
        batch_mean = torch.mean(x, dim=0)
        batch_var = torch.var(x, dim=0, unbiased=False)  # Matches NumPy's default
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(
        self,
        batch_mean: torch.Tensor,
        batch_var: torch.Tensor,
        batch_count: int,
    ) -> None:
        """Update mean and variance using batch moments.

        :param batch_mean: Mean of the batch
        :type batch_mean: torch.Tensor
        :param batch_var: Variance of the batch
        :type batch_var: torch.Tensor
        :param batch_count: Number of samples in the batch
        :type batch_count: int
        """
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        self.mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + (delta**2) * (self.count * batch_count / tot_count)
        self.var = m2 / tot_count
        self.count = tot_count
