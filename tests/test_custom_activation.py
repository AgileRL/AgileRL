import torch

from agilerl.networks.custom_activation import GumbelSoftmax


def test_apply_gumbel_softmax():
    # Initialize the GumbelSoftmax class
    gumbel_softmax = GumbelSoftmax()

    # Create input tensor
    logits = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    # Apply gumbel softmax function
    output = gumbel_softmax(logits)

    # Check if output tensor has the same shape as input tensor
    assert output.shape == logits.shape

    # Check if output tensor is within the range [0, 1]
    assert torch.all(output >= 0) and torch.all(output <= 1)

    # Check if output tensor sums up to 1 along the last dimension
    assert torch.all(torch.isclose(torch.sum(output, dim=-1), torch.tensor([1.0, 1.0])))
