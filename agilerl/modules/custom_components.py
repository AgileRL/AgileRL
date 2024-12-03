import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from agilerl.typing import DeviceType

class GumbelSoftmax(nn.Module):
    """Applies gumbel softmax function element-wise"""

    @staticmethod
    def gumbel_softmax(logits: torch.Tensor, tau: float = 1.0, eps: float = 1e-20) -> torch.Tensor:
        """Implementation of the gumbel softmax activation function

        :param logits: Tensor containing unnormalized log probabilities for each class.
        :type logits: torch.Tensor
        :param tau: Tau, defaults to 1.0
        :type tau: float, optional
        :param eps: Epsilon, defaults to 1e-20
        :type eps: float, optional
        """
        torch.compiler.cudagraph_mark_step_begin()
        epsilon = torch.rand_like(logits)  # epsilon = U
        gumbel_noise = -torch.log(-torch.log(epsilon + eps) + eps)
        y = logits + gumbel_noise
        return F.softmax(y / tau, dim=-1)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.gumbel_softmax(input)

class NoisyLinear(nn.Module):
    """The Noisy Linear Neural Network class.

    :param in_features: Input features size
    :type in_features: int
    :param out_features: Output features size
    :type out_features: int
    :param std_init: Standard deviation, defaults to 0.5
    :type std_init: float, optional
    :param device: Device, defaults to "cpu"
    :type device: str, optional
    """
    weight_epsilon: torch.Tensor
    bias_epsilon: torch.Tensor

    def __init__(
            self,
            in_features: int,
            out_features: int,
            std_init: float = 0.5,
            device: DeviceType = "cpu"
            ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        self.device = device

        self.weight_mu = nn.Parameter(torch.FloatTensor(out_features, in_features, device=device))
        self.weight_sigma = nn.Parameter(torch.FloatTensor(out_features, in_features, device=device))
        self.register_buffer(
            "weight_epsilon", torch.FloatTensor(out_features, in_features, device=device)
        )

        self.bias_mu = nn.Parameter(torch.FloatTensor(out_features, device=device))
        self.bias_sigma = nn.Parameter(torch.FloatTensor(out_features, device=device))
        self.register_buffer("bias_epsilon", torch.FloatTensor(out_features, device=device))

        self.reset_parameters()
        self.reset_noise()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns output of neural network.

        :param x: Neural network input
        :type x: torch.Tensor
        :return: Neural network output
        :rtype: torch.Tensor
        """
        if self.training:
            weight = self.weight_mu + self.weight_sigma.mul(self.weight_epsilon)
            bias = self.bias_mu + self.bias_sigma.mul(self.bias_epsilon)
        else:
            weight = self.weight_mu
            bias = self.bias_mu

        return F.linear(x, weight, bias)

    def reset_parameters(self) -> None:
        """Resets neural network parameters."""
        mu_range = 1 / math.sqrt(self.in_features)

        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))

        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

    def reset_noise(self) -> None:
        """Resets neural network noise."""
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)

        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def _scale_noise(self, size: int) -> torch.Tensor:
        """Returns noisy tensor.

        :param size: Tensor of same size as noisy output
        :type size: torch.Tensor()
        """
        x = torch.randn(size, device=self.device)
        return x.sign().mul_(x.abs().sqrt_())

class NewGELU(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """

    def forward(self, x):
        return (
            0.5
            * x
            * (
                1.0
                + torch.tanh(
                    math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))
                )
            )
        )
