from typing import List, Dict, Any
import torch
import torch.nn.functional as F

from agilerl.modules.base import EvolvableModule
from agilerl.modules.mlp import EvolvableMLP
from agilerl.utils.evolvable_networks import create_mlp

class RainbowMLP(EvolvableMLP):
    def __init__(
            self,
            num_inputs: int,
            num_outputs: int,
            hidden_size: List[int],
            num_atoms: int,
            support: torch.Tensor,
            noise_std: float = 0.5,
            **kwargs) -> None:

        super().__init__(
            num_inputs, num_atoms, hidden_size, 
            noisy=True, noise_std=noise_std, name="value", **kwargs
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
            name="advantage"
        )
    
    @property
    def init_dict(self) -> Dict[str, Any]:
        mlp_dict = super().init_dict
        mlp_dict["num_atoms"] = self.num_atoms
        mlp_dict['num_outputs'] = self.num_actions
        mlp_dict["support"] = self.support
        mlp_dict.pop("noisy")
        mlp_dict.pop("name")
        return mlp_dict

    def forward(self, x: torch.Tensor, q: bool = True, log: bool = False) -> torch.Tensor:
        """Forward pass of the RainbowMLP.

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
            output_vanish=self.output_vanish,
            output_activation=self.output_activation,
            noisy=self.noisy,
            init_layers=self.init_layers,
            layer_norm=self.layer_norm,
            activation=self.activation,
            noise_std=self.noise_std,
            device=self.device,
            new_gelu=self.new_gelu,
            name="advantage"
        )

        self.advantage_net = EvolvableModule.preserve_parameters(self.advantage_net, advantage_net)