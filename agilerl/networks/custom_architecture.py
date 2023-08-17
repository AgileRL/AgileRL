import torch
import torch.nn.functional as F
import torch.nn as nn

class GumbelSoftmax(nn.Module):
    """Applies gumbel softmax function element-wise
    """
    @staticmethod
    def gumbel_softmax(logits, tau=1.0, eps=1e-20):
        """Implementation of the gumbel softmax activation function
        
        :param logits: Tensor containing unnormalized log probabilities for each class.
        :type logits: torch.Tensor
        """
        epsilon = torch.rand_like(logits)
        logits += -torch.log(-torch.log(epsilon + eps) + eps)
        return F.softmax(logits / tau, dim=-1)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.gumbel_softmax(input)