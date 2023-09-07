import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from collections import OrderedDict

class GumbelSoftmax(nn.Module):
    """Applies gumbel softmax function element-wise"""

    @staticmethod
    def gumbel_softmax(logits, tau=1.0, eps=1e-20):
        """Implementation of the gumbel softmax activation function

        :param logits: Tensor containing unnormalized log probabilities for each class.
        :type logits: torch.Tensor
        :param tau: Tau, defaults to 1.0
        :type tau: float, optional
        :param eps: Epsilon, defaults to 1e-20
        :type eps: float, optional
        """
        epsilon = torch.rand_like(logits)
        logits += -torch.log(-torch.log(epsilon + eps) + eps)
        return F.softmax(logits / tau, dim=-1)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.gumbel_softmax(input)


class CustomModel(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(CustomModel, self).__init__()
        layers = []
        
        # Add input layer
        layers.append(nn.Linear(input_size, hidden_sizes[0]))
        layers.append(nn.ReLU())  # Activation function
        
        # Add hidden layers
        for i in range(len(hidden_sizes) - 1):
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
            layers.append(nn.ReLU())  # Activation function
        
        # Add output layer
        layers.append(nn.Linear(hidden_sizes[-1], output_size))
        
        # Combine all layers into a sequential model
        self.model = nn.Sequential(*layers)

        self.softmax = nn.Softmax()

    def forward(self, x):
        return self.softmax(self.model(x))

class MakeEvolvable(nn.Module):
    def __init__(self, network, input_tensor, device):
        super().__init__()
        self.network = network
        self.input_tensor = input_tensor
        self.device = device 
        self.network_information, self.in_features, self.out_features, self.hidden_layers, self.activation, self.output_activation = self.detect_architecture(self.input_tensor)

    def get_activation(self, activation_names):
        """Returns activation function for corresponding activation name.

        :param activation_names: Activation function name
        :type activation_names: str
        """
        activation_functions = {
            'tanh': nn.Tanh,
            'linear': nn.Identity,
            'relu': nn.ReLU,
            'elu': nn.ELU,
            'softsign': nn.Softsign,
            'sigmoid': nn.Sigmoid,
            'gumbel_softmax': GumbelSoftmax,
            'softplus': nn.Softplus,
            'softmax': nn.Softmax,
            'lrelu': nn.LeakyReLU,
            'prelu': nn.PReLU,
            'gelu': nn.GELU}

        return activation_functions[activation_names](dim=1) if activation_names == 'softmax' else activation_functions[activation_names]()

    def detect_architecture(self, input_tensor):
        """
        Determine the architecture of a neural network.

        Args:
            model (nn.Module): The PyTorch model to analyze.
            input_tensor (torch.Tensor): A sample input tensor to the model for shape inference.

        Returns:
            list: A list of dictionaries, each containing information about a layer's name, input shape, and output shape.
        """
        network_information= {}
        class_names = {}
        in_features_list = []
        out_features_list = []
        activation_list = []

        def register_hooks(module):
            def forward_hook(module, input, output):
                class_name = str(module.__class__.__name__)
                layer_dict = {}
                if class_name not in class_names.keys():
                    class_names[class_name] = 0
                else:
                    class_names[class_name] += 1
                if isinstance(module, nn.Linear):
                    layer_dict["layer_info"] = str(module)
                    layer_dict['in_features'] = module.in_features
                    layer_dict['out_features'] = module.out_features
                    network_information[f"{class_name}_{class_names[class_name]}"] = layer_dict
                    in_features_list.append(module.in_features)
                    out_features_list.append(module.out_features)
                else:
                    # Catch activation functions
                    network_information[f"{class_name}".lower()] = str(module.__class__)
                    activation_list.append(self.get_activation(f"{class_name}".lower()))

            if not isinstance(module, nn.Sequential) and not isinstance(module, nn.ModuleList) and not isinstance(module, type(self.network)):
                hooks.append(module.register_forward_hook(forward_hook))
            
        hooks = []
        self.network.apply(register_hooks)

        # Forward pass to collect input and output shapes
        with torch.no_grad():
            self.network(input_tensor)

        # Remove hooks
        for hook in hooks:
            hook.remove()

        in_features = in_features_list[0]
        out_features = out_features_list[-1]
        hidden_size = in_features_list[1:]
        activation = activation_list[0]
        output_activation = activation_list[-1]

        return network_information, in_features, out_features, hidden_size, activation, output_activation
    
    #### This function needs modifying for new evolvable wrapper
    def create_net(self):
        """Creates and returns neural network.
        """
        net_dict = OrderedDict()

        net_dict["linear_layer_0"] = nn.Linear(
            self.num_inputs, self.hidden_size[0])
        if self.layer_norm:
            net_dict["layer_norm_0"] = nn.LayerNorm(self.hidden_size[0])
        net_dict["activation_0"] = self.get_activation(self.activation)

        if len(self.hidden_size) > 1:
            for l_no in range(1, len(self.hidden_size)):
                net_dict[f"linear_layer_{str(l_no)}"] = nn.Linear(
                    self.hidden_size[l_no - 1], self.hidden_size[l_no])
                if self.layer_norm:
                    net_dict[f"layer_norm_{str(l_no)}"] = nn.LayerNorm(
                        self.hidden_size[l_no])
                net_dict[f"activation_{str(l_no)}"] = self.get_activation(
                    self.activation)

        output_layer = nn.Linear(self.hidden_size[-1], self.num_outputs)

        if self.output_vanish:
            output_layer.weight.data.mul_(0.1)
            output_layer.bias.data.mul_(0.1)

        net_dict["linear_layer_output"] = output_layer
        if self.output_activation is not None:
            net_dict["activation_output"] = self.get_activation(
                self.output_activation)
            
        net = nn.Sequential(net_dict)
            
        if self.accelerator is None:
            net = net.to(self.device)

        return net



# Example input tensor
input_size = 10
input_tensor = torch.rand(32, input_size)  # Batch size of 32, input size of 10

# Instantiate the CustomModel
hidden_sizes = [64, 64]  # You can adjust these hidden layer sizes
output_size = 1 
custom_model = CustomModel(input_size, hidden_sizes, output_size)

evolvable_model = MakeEvolvable(custom_model, input_tensor, device)


print(evolvable_model.in_features)
print(evolvable_model.out_features)
print(evolvable_model.hidden_layers)
print(evolvable_model.activation)
print(evolvable_model.output_activation)
print(evolvable_model.network_information)