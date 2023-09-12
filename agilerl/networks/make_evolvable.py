import copy
from collections import OrderedDict
from typing import List

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

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.model(x))

class MakeEvolvable(nn.Module):
    def __init__(self, 
                 network, 
                 input_tensor, 
                 device="cpu", 
                 accelerator=None, 
                 init_layers=True,
                 output_vanish=True,
                 activation=None):
        super().__init__()

        self.device = device 
        self.input_tensor = input_tensor.to(self.device)
        self.detect_architecture(network.to(self.device), self.input_tensor)
        self.accelerator = accelerator
        self.layer_norm = False
        self.init_layers = init_layers
        self.output_vanish = output_vanish
        self.net = self.create_net() # Reset the names of the layers to allow for cloning
        if activation is not None:
            self.activation = activation
        
    def forward(self, x): #
        """Returns output of neural network.

        :param x: Neural network input
        :type x: torch.Tensor() or np.array
        """
        if not isinstance(x, torch.Tensor):
            x = torch.FloatTensor(np.array(x))
            if self.accelerator is None:
                x = x.to(self.device)

        # print("Tensor device", x.device)
        # for name, param in self.net.named_parameters():
        #     print(f"Device {name}: ", param.device)
        x = self.net(x)
        return x

    def get_activation(self, activation_names):#
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

    def detect_architecture(self, network, input_tensor):
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
                elif isinstance(module, nn.LayerNorm):
                    self.layer_norm = True
                else:
                    # Catch activation functions
                    network_information[f"{class_name}".lower()] = str(module.__class__)
                    activation_list.append(f"{class_name}".lower())

            if not isinstance(module, nn.Sequential) and not isinstance(module, nn.ModuleList) and not isinstance(module, type(network)):
                hooks.append(module.register_forward_hook(forward_hook))
            
        hooks = []
        network.apply(register_hooks)

        # Forward pass to collect input and output shapes
        with torch.no_grad():
            network(input_tensor)

        # Remove hooks
        for hook in hooks:
            hook.remove()

        self.num_inputs = in_features_list[0]
        self.num_outputs = out_features_list[-1]
        self.hidden_size = in_features_list[1:]
        self.activation = activation_list[0]
        self.output_activation = activation_list[-1]

    def layer_init(self, layer, std=np.sqrt(2), bias_const=0.0):

        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_const)
        return layer

    #### Do we still need this when we already have the nn passed in
    #### For now lets agree to use original network as a blueprint to then create new evolved nets 
    #### We will just reset the self.network parameter and therefore keep the create nets function
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
                    self.hidden_size[l_no - 1], self.hidden_size[l_no]
                )
                if self.init_layers:
                    net_dict[f"linear_layer_{str(l_no)}"] = self.layer_init(
                        net_dict[f"linear_layer_{str(l_no)}"]
                    )
                if self.layer_norm:
                    net_dict[f"layer_norm_{str(l_no)}"] = nn.LayerNorm(
                        self.hidden_size[l_no]
                    )
                net_dict[f"activation_{str(l_no)}"] = self.get_activation(
                    self.activation
                )

        output_layer = nn.Linear(self.hidden_size[-1], self.num_outputs)
        if self.init_layers:
            output_layer = self.layer_init(output_layer)

        #### Do we need this with this class??
        if self.output_vanish:
            output_layer.weight.data.mul_(0.1)
            output_layer.bias.data.mul_(0.1)

        net_dict["linear_layer_output"] = output_layer
        if self.output_activation is not None:
            net_dict["activation_output"] = self.get_activation(self.output_activation)

        net = nn.Sequential(net_dict)

        if self.accelerator is None:
            net = net.to(self.device)

        return net
    
    def get_model_dict(self):
        """Returns dictionary with model information and weights.
        """
        model_dict = self.init_dict
        model_dict.update(
            {'stored_values': self.extract_parameters(without_layer_norm=False)})
        return model_dict
    
    def count_parameters(self, without_layer_norm=False): #
        """Returns number of parameters in neural network.

        :param without_layer_norm: Exclude normalization layers, defaults to False
        :type without_layer_norm: bool, optional
        """
        count = 0
        for name, param in self.named_parameters():
            if not without_layer_norm or 'layer_norm' not in name:
                count += param.data.cpu().numpy().flatten().shape[0]
        return count
    
    def extract_grad(self, without_layer_norm=False): #
        """Returns current pytorch gradient in same order as genome's flattened
        parameter vector.

        :param without_layer_norm: Exclude normalization layers, defaults to False
        :type without_layer_norm: bool, optional
        """
        tot_size = self.count_parameters(without_layer_norm)
        pvec = np.zeros(tot_size, np.float32)
        count = 0
        try:
            for name, param in self.named_parameters():
                if not without_layer_norm or "layer_norm" not in name:
                    sz = param.grad.data.cpu().numpy().flatten().shape[0]
                    pvec[count : count + sz] = param.grad.data.cpu().numpy().flatten()
                    count += sz
        except:
            return np.zeros(tot_size, np.float32)
        return pvec.copy()
    
    def extract_parameters(self, without_layer_norm=False): #
        """Returns current flattened neural network weights.

        :param without_layer_norm: Exclude normalization layers, defaults to False
        :type without_layer_norm: bool, optional
        """
        tot_size = self.count_parameters(without_layer_norm)
        pvec = np.zeros(tot_size, np.float32)
        count = 0
        for name, param in self.named_parameters():
            if not without_layer_norm or "layer_norm" not in name:
                sz = param.data.cpu().detach().numpy().flatten().shape[0]
                pvec[count : count + sz] = param.data.cpu().detach().numpy().flatten()
                count += sz
        return copy.deepcopy(pvec)
    
    #### When is this ever used?
    def inject_parameters(self, pvec, without_layer_norm=False): #
        """Injects a flat vector of neural network parameters into the model's current
        neural network weights.

        :param pvec: Network weights
        :type pvec: np.array()
        :param without_layer_norm: Exclude normalization layers, defaults to False
        :type without_layer_norm: bool, optional
        """
        count = 0

        for name, param in self.named_parameters():
            if not without_layer_norm or "layer_norm" not in name:
                sz = param.data.cpu().numpy().flatten().shape[0]
                raw = pvec[count : count + sz]
                reshaped = raw.reshape(param.data.cpu().numpy().shape)
                param.data = torch.from_numpy(copy.deepcopy(reshaped)).type(
                    torch.FloatTensor
                )
                count += sz
        return pvec
    
    #### Need to think about the best way to define the init dict when we're dealing with 
    #### architectures other than an mlp
    #### For just mlp functionality it can be kept as is but I don't like it
    @property
    def init_dict(self): #
        """Returns model information in dictionary."""
        # init_dict = {
        #     "num_inputs": self.num_inputs,
        #     "num_outputs": self.num_outputs,
        #     "hidden_size": self.hidden_size,
        #     "activation": self.activation,
        #     "output_activation": self.output_activation,
        #     "layer_norm": self.layer_norm,
        #     "device": self.device,
        #     "accelerator": self.accelerator,
        # }
        init_dict = {
            "device": self.device,
            "network": self.net,
            "input_tensor": self.input_tensor
        }
        return init_dict
    
    #### Leave the same for now but again same as init_dict function
    @property
    def short_dict(self):
        """Returns shortened version of model information in dictionary."""
        short_dict = {
            "hidden_size": self.hidden_size,
            "activation": self.activation,
            "output_activation": self.output_activation,
            "layer_norm": self.layer_norm,
        }
        return short_dict
        
    #### 
    def add_layer(self): #
        """Adds a hidden layer to neural network."""
        # add layer to hyper params
        if len(self.hidden_size) < 3:  # HARD LIMIT
            self.hidden_size += [self.hidden_size[-1]]

            # copy old params to new net
            new_net = self.create_net()
            new_net = self.preserve_parameters(old_net=self.net, new_net=new_net)
            self.net = new_net
        else:
            self.add_node()
    
    ####
    def remove_layer(self):
        """Removes a hidden layer from neural network."""
        if len(self.hidden_size) > 1:  # HARD LIMIT
            self.hidden_size = self.hidden_size[:1]
            new_net = self.create_net()
            new_net = self.shrink_preserve_parameters(old_net=self.net, new_net=new_net)
            self.net = new_net
        else:
            self.add_node()
    
    ####
    def add_node(self, hidden_layer=None, numb_new_nodes=None):
        """Adds nodes to hidden layer of neural network.

        :param hidden_layer: Depth of hidden layer to add nodes to, defaults to None
        :type hidden_layer: int, optional
        :param numb_new_nodes: Number of nodes to add to hidden layer, defaults to None
        :type numb_new_nodes: int, optional
        """
        if hidden_layer is None:
            hidden_layer = np.random.randint(0, len(self.hidden_size), 1)[0]
        else:
            hidden_layer = min(hidden_layer, len(self.hidden_size) - 1)
        if numb_new_nodes is None:
            numb_new_nodes = np.random.choice([16, 32, 64], 1)[0]

        if self.hidden_size[hidden_layer] + numb_new_nodes <= 500:  # HARD LIMIT
            self.hidden_size[hidden_layer] += numb_new_nodes
            new_net = self.create_net()
            new_net = self.preserve_parameters(old_net=self.net, new_net=new_net)

            self.net = new_net

        return {"hidden_layer": hidden_layer, "numb_new_nodes": numb_new_nodes}
    
    ####
    def remove_node(self, hidden_layer=None, numb_new_nodes=None): #
        """Removes nodes from hidden layer of neural network.

        :param hidden_layer: Depth of hidden layer to remove nodes from, defaults to None
        :type hidden_layer: int, optional
        :param numb_new_nodes: Number of nodes to remove from hidden layer, defaults to None
        :type numb_new_nodes: int, optional
        """
        if hidden_layer is None:
            hidden_layer = np.random.randint(0, len(self.hidden_size), 1)[0]
        else:
            hidden_layer = min(hidden_layer, len(self.hidden_size) - 1)
        if numb_new_nodes is None:
            numb_new_nodes = np.random.choice([16, 32, 64], 1)[0]

        if self.hidden_size[hidden_layer] - numb_new_nodes > 64:  # HARD LIMIT
            self.hidden_size[hidden_layer] = (
                self.hidden_size[hidden_layer] - numb_new_nodes
            )
            new_net = self.create_net()
            new_net = self.shrink_preserve_parameters(old_net=self.net, new_net=new_net)

            self.net = new_net

        return {"hidden_layer": hidden_layer, "numb_new_nodes": numb_new_nodes}
    
    ####
    def clone(self): #
        """Returns clone of neural net with identical parameters."""
        clone = MakeEvolvable(**copy.deepcopy(self.init_dict))
        clone.load_state_dict(copy.deepcopy(self.state_dict()))
        return clone

    ####
    def preserve_parameters(self, old_net, new_net):
        """Returns new neural network with copied parameters from old network.

        :param old_net: Old neural network
        :type old_net: nn.Module()
        :param new_net: New neural network
        :type new_net: nn.Module()
        """
        old_net_dict = dict(old_net.named_parameters())

        for key, param in new_net.named_parameters():
            if key in old_net_dict.keys():
                if old_net_dict[key].data.size() == param.data.size():
                    param.data = old_net_dict[key].data
                else:
                    if "norm" not in key:
                        old_size = old_net_dict[key].data.size()
                        new_size = param.data.size()
                        if len(param.data.size()) == 1:
                            param.data[: min(old_size[0], new_size[0])] = old_net_dict[
                                key
                            ].data[: min(old_size[0], new_size[0])]
                        else:
                            param.data[
                                : min(old_size[0], new_size[0]),
                                : min(old_size[1], new_size[1]),
                            ] = old_net_dict[key].data[
                                : min(old_size[0], new_size[0]),
                                : min(old_size[1], new_size[1]),
                            ]

        return new_net

    ####
    def shrink_preserve_parameters(self, old_net, new_net):
        """Returns shrunk new neural network with copied parameters from old network.

        :param old_net: Old neural network
        :type old_net: nn.Module()
        :param new_net: New neural network
        :type new_net: nn.Module()
        """
        old_net_dict = dict(old_net.named_parameters())

        for key, param in new_net.named_parameters():
            if key in old_net_dict.keys():
                if old_net_dict[key].data.size() == param.data.size():
                    param.data = old_net_dict[key].data
                else:
                    if "norm" not in key:
                        old_size = old_net_dict[key].data.size()
                        new_size = param.data.size()
                        min_0 = min(old_size[0], new_size[0])
                        if len(param.data.size()) == 1:
                            param.data[:min_0] = old_net_dict[key].data[:min_0]
                        else:
                            min_1 = min(old_size[1], new_size[1])
                            param.data[:min_0, :min_1] = old_net_dict[key].data[
                                :min_0, :min_1
                            ]
        return new_net





# # Example input tensor
# input_size = 10
# input_tensor = torch.rand(32, input_size)  # Batch size of 32, input size of 10
# # Instantiate the CustomModel
# hidden_sizes = [64, 64]  # You can adjust these hidden layer sizes
# output_size = 1 
# custom_model = CustomModel(input_size, hidden_sizes, output_size)
# evolvable_model = MakeEvolvable(custom_model, input_tensor, device)
# print(evolvable_model.num_inputs)
# print(evolvable_model.num_outputs)
# print(evolvable_model.hidden_size)
# print(evolvable_model.activation)
# print(evolvable_model.output_activation)
# print(evolvable_model.layer_norm)