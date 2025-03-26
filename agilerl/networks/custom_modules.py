from typing import Any, Dict, List

import torch
import torch.nn.functional as F

from agilerl.modules.base import EvolvableModule
from agilerl.modules.mlp import EvolvableMLP
from agilerl.utils.evolvable_networks import create_mlp


class DuelingDistributionalMLP(EvolvableMLP):
    """A multi-layer perceptron network that calculates state-action values through
    the use of separate advantage and value networks. It outputs a distribution of values
    for both of these networks. Used in the Rainbow DQN algorithm.

    :param num_inputs: Number of input features.
    :type num_inputs: int
    :param num_outputs: Number of output features.
    :type num_outputs: int
    :param hidden_size: List of hidden layer sizes.
    :type hidden_size: List[int]
    :param num_atoms: Number of atoms in the distribution.
    :type num_atoms: int
    :param support: Support of the distribution.
    :type support: torch.Tensor
    :param layer_norm: Normalization between layers, defaults to True
    :type layer_norm: bool, optional
    :param output_vanish: Vanish output by multiplying by 0.1, defaults to True
    :type output_vanish: bool, optional
    :param init_layers: Initialise network layers, defaults to True
    :type init_layers: bool, optional
    :param noisy: Use noisy layers, defaults to True
    :type noisy: bool, optional
    :param noise_std: Standard deviation of the noise. Defaults to 0.5.
    :type noise_std: float, optional
    :param activation: Activation layer, defaults to 'ReLU'
    :type activation: str, optional
    :param output_activation: Output activation layer, defaults to None
    :type output_activation: str, optional
    :param min_hidden_layers: Minimum number of hidden layers the network will shrink down to, defaults to 1
    :type min_hidden_layers: int, optional
    :param max_hidden_layers: Maximum number of hidden layers the network will expand to, defaults to 3
    :type max_hidden_layers: int, optional
    :param min_mlp_nodes: Minimum number of nodes a layer can have within the network, defaults to 64
    :type min_mlp_nodes: int, optional
    :param max_mlp_nodes: Maximum number of nodes a layer can have within the network, defaults to 500
    :type max_mlp_nodes: int, optional
    :param new_gelu: Use new GELU activation function, defaults to False
    :type new_gelu: bool, optional
    :param device: Device for accelerated computing, 'cpu' or 'cuda', defaults to 'cpu'
    :type device: str, optional
    """

    def __init__(
        self,
        num_inputs: int,
        num_outputs: int,
        hidden_size: List[int],
        num_atoms: int,
        support: torch.Tensor,
        layer_norm: bool = True,
        output_vanish: bool = True,
        init_layers: bool = False,
        noisy: bool = True,
        noise_std: float = 0.5,
        activation: str = "ReLU",
        output_activation: str = None,
        min_hidden_layers: int = 1,
        max_hidden_layers: int = 3,
        min_mlp_nodes: int = 64,
        max_mlp_nodes: int = 500,
        new_gelu: bool = False,
        device: str = "cpu",
    ) -> None:

        super().__init__(
            num_inputs,
            num_atoms,
            hidden_size,
            activation,
            output_activation,
            min_hidden_layers,
            max_hidden_layers,
            min_mlp_nodes,
            max_mlp_nodes,
            layer_norm=layer_norm,
            output_vanish=output_vanish,
            init_layers=init_layers,
            noisy=noisy,
            noise_std=noise_std,
            new_gelu=new_gelu,
            device=device,
            name="value",
        )

        self.num_atoms = num_atoms
        self.num_actions = num_outputs
        self.support = support

        self.advantage_net = create_mlp(
            input_size=num_inputs,
            output_size=num_outputs * num_atoms,
            hidden_size=self.hidden_size,
            output_vanish=self.output_vanish,
            output_activation=self.output_activation,
            noisy=self.noisy,
            init_layers=self.init_layers,
            layer_norm=self.layer_norm,
            activation=self.activation,
            noise_std=self.noise_std,
            device=self.device,
            new_gelu=self.new_gelu,
            name="advantage",
        )

    @property
    def net_config(self) -> Dict[str, Any]:
        net_config = super().net_config.copy()
        net_config.pop("num_atoms")
        net_config.pop("support")
        return net_config

    def forward(
        self, x: torch.Tensor, q: bool = True, log: bool = False
    ) -> torch.Tensor:
        """Forward pass of the network.

        :param obs: Input to the network.
        :type obs: torch.Tensor, dict[str, torch.Tensor], or list[torch.Tensor]
        :param q: Whether to return Q values. Defaults to True.
        :type q: bool
        :param log: Whether to return log probabilities. Defaults to False.
        :type log: bool

        :return: Output of the network.
        :rtype: torch.Tensor
        """
        value: torch.Tensor = self.model(x)
        advantage: torch.Tensor = self.advantage_net(x)

        batch_size = value.size(0)
        value = value.view(batch_size, 1, self.num_atoms)
        advantage = advantage.view(batch_size, self.num_actions, self.num_atoms)

        x = value + advantage - advantage.mean(1, keepdim=True)
        if log:
            x = F.log_softmax(x.view(-1, self.num_atoms), dim=-1)
            return x.view(-1, self.num_actions, self.num_atoms)

        x = F.softmax(x.view(-1, self.num_atoms), dim=-1)
        x = x.view(-1, self.num_actions, self.num_atoms).clamp(min=1e-3)
        if q:
            x = torch.sum(x * self.support, dim=2)

        return x

    def recreate_network(self) -> None:
        """Recreates the network with the same parameters."""
        # Recreate value net with the same parameters
        super().recreate_network()

        advantage_net = create_mlp(
            input_size=self.num_inputs,
            output_size=self.num_actions * self.num_atoms,
            hidden_size=self.hidden_size,
            output_activation=self.output_activation,
            output_vanish=self.output_vanish,
            noisy=self.noisy,
            init_layers=self.init_layers,
            layer_norm=self.layer_norm,
            activation=self.activation,
            noise_std=self.noise_std,
            device=self.device,
            new_gelu=self.new_gelu,
            name="advantage",
        )

        self.advantage_net = EvolvableModule.preserve_parameters(
            self.advantage_net, advantage_net
        )
