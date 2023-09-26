import torch.nn as nn
import torch
import torch.nn.functional as F
import torch
import torch.nn as nn
from collections import deque
from collections import OrderedDict
import numpy as np
import copy

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

class MakeEvolvable(nn.Module):
    def __init__(self, 
                 network, 
                 input_tensor, 
                 init_layers=True,
                 output_vanish=True,
                 rand_seed=42,
                 device="cpu", 
                 accelerator=None,
                 **kwargs):
        super().__init__()

        self.init_layers = init_layers
        self.output_vanish = output_vanish
        self.rng = np.random.RandomState(rand_seed)
        self.device = device
        self.accelerator = accelerator
        
        # Set the layer counters
        self.conv_counter = -1
        self.lin_counter = -1
        self.mlp_norm = None
        self.cnn_norm = None

        # Placeholder convolutional variables (needed for init_dict function to work)
        self.has_conv_layers = False
        self.pooling = None
        self.pooling_kernel = None
        self.cnn_activation = None
        self.input_tensor = input_tensor.to(self.device)
        self.in_channels, self.channel_size, self.kernel_size \
        ,self.stride_size, self.padding = None, None, None, None, None

        # If first instance, network used to instantiate, upon cloning, init_dict used instead
        if not kwargs:
            self.detect_architecture(network.to(self.device), self.input_tensor)
        else:
            for key, value in kwargs.items():
                setattr(self, key, value)
        
        self.feature_net, self.value_net, self.net = self.create_nets()

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

        #### Not sure about this line        
        x = x.type(torch.float32)
        x = self.net(x)
        return x
    
    def layer_init(self, layer, std=np.sqrt(2), bias_const=0.0):
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_const)
        return layer
    
    def get_pooling(self, pooling_names, kernel_size):
        """Returns pooling layer for corresponding activation name.

        :param activation_names: Activation function name
        :type activation_names: str
        """
        pooling_functions = {
            "MaxPool1d": nn.MaxPool1d,
            "MaxPool2d": nn.MaxPool2d,
            "MaxPool3d": nn.MaxPool3d,
            "MaxUnpool1d": nn.MaxUnpool1d,
            "MaxUnpool2d": nn.MaxUnpool2d,
            "MaxUnpool3d": nn.MaxUnpool3d,
            "AvgPool1d": nn.AvgPool1d,
            "AvgPool2d": nn.AvgPool2d,
            "AvgPool3d": nn.AvgPool3d,
            "FractionalMaxPool2d": nn.FractionalMaxPool2d,
            "FractionalMaxPool3d": nn.FractionalMaxPool3d,
            "LPPool1d": nn.LPPool1d,
            "LPPool2d": nn.LPPool2d,
            "AdaptiveMaxPool1d": nn.AdaptiveMaxPool1d,
            "AdaptiveMaxPool2d": nn.AdaptiveMaxPool2d,
            "AdaptiveMaxPool3d": nn.AdaptiveMaxPool3d,
            "AdaptiveAvgPool1d": nn.AdaptiveAvgPool1d,
            "AdaptiveAvgPool2d": nn.AdaptiveAvgPool2d,
            "AdaptiveAvgPool3d": nn.AdaptiveAvgPool3d
        }

        return pooling_functions[pooling_names](kernel_size)

    def get_activation(self, activation_names):#
        """Returns activation function for corresponding activation name.

        :param activation_names: Activation function name
        :type activation_names: str
        """
        activation_functions = {
            'Tanh': nn.Tanh,
            'Linear': nn.Identity,
            'ReLU': nn.ReLU,
            'ELU': nn.ELU,
            'Softsign': nn.Softsign,
            'Sigmoid': nn.Sigmoid,
            'GumbelSoftmax': GumbelSoftmax,
            'Softplus': nn.Softplus,
            'Softmax': nn.Softmax,
            'LeakyReLU': nn.LeakyReLU,
            'PReLU': nn.PReLU,
            'GELU': nn.GELU}

        return activation_functions[activation_names](dim=1) if activation_names == 'softmax' else activation_functions[activation_names]()

    def get_normalization(self, normalization_name, layer_size):
        """Returns normalization layer for corresponding normalization name.

        :param normalization_names: Normalization layer name
        :type normalization_names: str
        :param layer_size: The layer after which the normalization layer will be applied
        :param layer_size: int
        """
        normalization_functions = {
            "BatchNorm1d": nn.BatchNorm1d, 
            "BatchNorm2d": nn.BatchNorm2d, 
            "BatchNorm3d": nn.BatchNorm3d,
            "InstanceNorm1d": nn.InstanceNorm1d, 
            "InstanceNorm2d": nn.InstanceNorm2d, 
            "InstanceNorm3d": nn.InstanceNorm3d,
            "LayerNorm": nn.LayerNorm}

        return normalization_functions[normalization_name](layer_size)

    def detect_architecture(self, network, input_tensor):
        """Determine the architecture of a neural network.

        :param network: Neural network whose architecture is being detected
        :type network: nn.Module
        :param input_tensor: Tensor that will be passed into the network
        :type input_tensor: torch.Tensor
        """
        in_features_list = []
        out_features_list = []
        mlp_activations = []
        in_channel_list = []
        out_channel_list = []
        kernel_size_list = []
        stride_size_list = []
        padding_list = []
        layer_indices = {"cnn":{},
                         "mlp":{}}

        ## STORE THE LAYER INDEX OF THE DIFFERENT LAYERS!!!
        def register_hooks(module):
            def forward_hook(module, input, output):
                if isinstance(module, nn.modules.conv._ConvNd):
                    self.has_conv_layers = True
                    self.conv_counter += 1
                    in_channel_list.append(module.in_channels)
                    out_channel_list.append(module.out_channels)
                    kernel_size_list.append(module.kernel_size)
                    stride_size_list.append(module.stride)
                    padding_list.append(module.padding)
                elif isinstance(module, nn.Linear):
                    self.lin_counter += 1
                    in_features_list.append(module.in_features)
                    out_features_list.append(module.out_features)
                elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
                                         nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d,
                                         nn.LayerNorm)):
                    if len(output.shape) <= 2:
                        self.mlp_norm = str(module.__class__.__name__)
                        if "mlp_norm" not in layer_indices["mlp"].keys():
                            layer_indices["mlp"]["mlp_norm"] = []
                        layer_indices["mlp"]["mlp_norm"].append(self.lin_counter)
                    else:
                        self.cnn_norm = str(module.__class__.__name__)
                        if "cnn_norm" not in layer_indices["cnn"].keys():
                            layer_indices["cnn"]["cnn_norm"] = []
                        layer_indices["cnn"]["cnn_norm"].append(self.conv_counter)
                elif isinstance(module, (nn.MaxPool1d, nn.MaxPool2d, nn.MaxPool3d
                                        ,nn.MaxUnpool1d, nn.MaxUnpool2d, nn.MaxUnpool3d
                                        ,nn.AvgPool1d, nn.AvgPool2d, nn.AvgPool3d
                                        ,nn.FractionalMaxPool2d, nn.FractionalMaxPool3d
                                        ,nn.LPPool1d, nn.LPPool2d, nn.AdaptiveMaxPool1d
                                        ,nn.AdaptiveMaxPool2d, nn.AdaptiveMaxPool3d
                                        ,nn.AdaptiveAvgPool1d, nn.AdaptiveAvgPool2d
                                        ,nn.AdaptiveAvgPool3d)):
                    self.pooling = str(module.__class__.__name__)
                    self.pooling_kernel = module.kernel_size
                    if "cnn_pool" not in layer_indices["cnn"].keys():
                        layer_indices["cnn"]["cnn_pool"] = []
                    layer_indices["cnn"]["cnn_pool"].append(self.conv_counter)
                elif isinstance(module, nn.Flatten):
                    pass
                else:
                    # Catch activation functions
                    if len(output.shape) <= 2:
                        mlp_activations.append(str(module.__class__.__name__))
                    else:
                        self.cnn_activation = str(module.__class__.__name__)
                        if "cnn_act" not in layer_indices["cnn"].keys():
                            layer_indices["cnn"]["cnn_act"] = []
                        layer_indices["cnn"]["cnn_act"].append(self.conv_counter)

            if not isinstance(module, nn.Sequential) and not isinstance(module, nn.ModuleList)  \
                and not isinstance(module, type(network)):
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
        self.mlp_output_activation = mlp_activations.pop() if len(mlp_activations) == len(out_features_list) else None
        self.mlp_activation = mlp_activations[-1]
        self.layer_indices = layer_indices
        if self.has_conv_layers == True:
            self.arch = "cnn"
            self.in_channels = in_channel_list[0]
            self.channel_size = out_channel_list
            self.kernel_size = kernel_size_list
            self.stride_size = stride_size_list
            self.padding = padding_list
        else:
            self.arch = "mlp"

        # print(network_information)

        self.conv_counter = -1
        self.lin_counter = -1

    def create_mlp(self, input_size, output_size, hidden_size, name):
        """Creates and returns multi-layer perceptron."""
        net_dict = OrderedDict()
        net_dict[f"{name}_linear_layer_0"] = nn.Linear(input_size, hidden_size[0])
        if self.mlp_norm is not None:
            net_dict[f"{name}_layer_norm_0"] = self.get_normalization(self.mlp_norm, hidden_size[0])
        net_dict[f"{name}_activation_0"] = self.get_activation(self.mlp_activation)

        if len(hidden_size) > 1:
            for l_no in range(1, len(hidden_size)):
                net_dict[f"{name}linear_layer_{str(l_no)}"] = nn.Linear(
                    hidden_size[l_no - 1], hidden_size[l_no]
                )
                if self.init_layers:
                    net_dict[f"{name}linear_layer_{str(l_no)}"] = self.layer_init(
                        net_dict[f"{name}linear_layer_{str(l_no)}"]
                )
                net_dict[f"activation_{str(l_no)}"] = self.get_activation(
                    self.mlp_activation
                )
                if (self.mlp_norm is not None) and (l_no in self.layer_indices["mlp"]["mlp_norm"]):
                    net_dict[f"layer_norm_{str(l_no)}"] = self.get_normalization(
                        self.mlp_norm,
                        hidden_size[l_no]
                )
        output_layer = nn.Linear(hidden_size[-1], output_size)
        if self.init_layers:
            output_layer = self.layer_init(output_layer)

        #### Do we need this with this class??
        if self.output_vanish:
            output_layer.weight.data.mul_(0.1)
            output_layer.bias.data.mul_(0.1)
        
        net_dict[f"{name}_linear_layer_output"] = output_layer
        if self.mlp_output_activation is not None:
            net_dict["activation_output"] = self.get_activation(self.mlp_output_activation)

        return nn.Sequential(net_dict)
    
    
    def create_cnn(self, input_size, channel_size, kernel_size, stride_size, padding, name):
        """Creates and returns convolutional neural network."""
        net_dict = OrderedDict()
        net_dict[f"{name}_conv_layer_0"] = nn.Conv2d(
            in_channels=input_size,
            out_channels=channel_size[0],
            kernel_size=kernel_size[0],
            stride=stride_size[0],
            padding=padding[0]
        )
        if (self.cnn_norm is not None) and (0 in self.layer_indices["cnn"]["cnn_norm"]):
            net_dict[f"{name}_layer_norm_0"] = self.get_normalization(
                self.cnn_norm,
                channel_size[0])
            net_dict[f"{name}_activation_0"] = self.get_activation(self.cnn_activation)

        if len(channel_size) > 1:
            for l_no in range(1, len(channel_size)):
                net_dict[f"{name}_conv_layer_{str(l_no)}"] = nn.Conv2d(
                    in_channels=channel_size[l_no - 1],
                    out_channels=channel_size[l_no],
                    kernel_size=kernel_size[l_no],
                    stride=stride_size[l_no],
                    padding=padding[l_no]
                )
                if (self.cnn_norm is not None) and (l_no in self.layer_indices["cnn"]["cnn_norm"]):
                    net_dict[f"{name}_layer_norm_{str(l_no)}"] = self.get_normalization(
                        self.cnn_norm,
                        channel_size[l_no])
                
                if l_no in self.layer_indices["cnn"]["cnn_act"]:
                    net_dict[f"{name}_activation_{str(l_no)}"] = self.get_activation(
                        self.cnn_activation
                    )
                if (self.pooling is not None) and (l_no in self.layer_indices["cnn"]["cnn_pool"]):
                    net_dict[f"{name}_pooling_{str(l_no)}"] = self.get_pooling(
                        self.pooling,
                        self.pooling_kernel)

        return nn.Sequential(net_dict)
    
    def create_nets(self):
        """Creates and returns neural networks."""

        # Check if any CNN layers otherwise return just a mlp
        if len(self.layer_indices["cnn"]) != 0:

            feature_net = self.create_cnn(
                self.in_channels,
                self.channel_size,
                self.kernel_size,
                self.stride_size,
                self.padding,
                name="feature",
            )
            # print("FEATURE NET")
            # feature_net = feature_net.to(self.device)
            # print(feature_net(self.input_tensor).shape)
            
            input_size = (feature_net(torch.zeros(*self.input_tensor.shape))
                          ).to(self.device).view(1,-1).size(1)
        else:
            feature_net = None
            input_size = self.num_inputs

        value_net = self.create_mlp(
            input_size,
            self.num_outputs,
            self.hidden_size,
            name="value",
        )
        
        if self.accelerator is None:
                self.feature_net = feature_net.to(self.device) if feature_net is not None else feature_net
                self.value_net = value_net.to(self.device)
        
        if len(self.layer_indices["cnn"]) != 0:
            net = nn.Sequential(
                feature_net,
                nn.Flatten(),
                value_net
            )
        else:
            net = value_net

        return feature_net, value_net, net 
    
    def get_model_dict(self):
        """Returns dictionary with model information and weights."""
        model_dict = self.init_dict
        model_dict.update(
            {"stored_values": self.extract_parameters(without_layer_norm=False)}
        )
        return model_dict
    
    def count_parameters(self, without_layer_norm=False):
        """Returns number of parameters in neural network.

        :param without_layer_norm: Exclude normalization layers, defaults to False
        :type without_layer_norm: bool, optional
        """
        count = 0
        for name, param in self.named_parameters():
            if not without_layer_norm or "layer_norm" not in name:
                count += param.data.cpu().numpy().flatten().shape[0]
        return count
    
    def extract_grad(self, without_layer_norm=False):
        """Returns current pytorch gradient in same order as genome's flattened
        parameter vector.

        :param without_layer_norm: Exclude normalization layers, defaults to False
        :type without_layer_norm: bool, optional
        """
        tot_size = self.count_parameters(without_layer_norm)
        pvec = np.zeros(tot_size, np.float32)
        count = 0
        for name, param in self.named_parameters():
            if not without_layer_norm or "layer_norm" not in name:
                sz = param.grad.data.cpu().numpy().flatten().shape[0]
                pvec[count : count + sz] = param.grad.data.cpu().numpy().flatten()
                count += sz
        return pvec.copy()
    
    def extract_parameters(self, without_layer_norm=False):
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
    
    def inject_parameters(self, pvec, without_layer_norm=False):
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
    
    # #### Is this function even relevant?
    # # @property
    # # def short_dict(self):
    # #     """Returns shortened version of model information in dictionary."""
    # #     short_dict = {
    # #         "channel_size": self.channel_size,
    # #         "kernel_size": self.kernel_size,
    # #         "stride_size": self.stride_size,
    # #         "hidden_size": self.hidden_size,
    # #         "num_atoms": self.num_atoms,
    # #         "mlp_activation": self.mlp_activation,
    # #         "cnn_activation": self.cnn_activation,
    # #         "layer_norm": self.layer_norm,
    # #     }
    # #     return short_dict

    # @property
    # def init_dict(self):
    #     """Returns model information in dictionary."""
    #     init_dict = {
    #         "device": self.device,
    #         "network": self.net,
    #         "input_tensor": self.input_tensor
    #     }
    #     return init_dict

    @property
    def make_conv_dict(self):
        conv_dict = {
                    "in_channels": self.in_channels,
                    "channel_size": self.channel_size,
                    "kernel_size": self.kernel_size,
                    "stride_size": self.stride_size, 
                    "padding": self.padding, 
                    "pooling": self.pooling,
                    "pooling_kernel": self.pooling_kernel,
                    "cnn_activation": self.cnn_activation
                }
        return conv_dict
    
    @property
    def init_dict(self):
        init_dict = {
                "network": None,
                "input_tensor": self.input_tensor,
                "num_inputs": self.num_inputs,
                "num_outputs": self.num_outputs,
                "hidden_size": self.hidden_size,
                "mlp_activation": self.mlp_activation,
                "mlp_output_activation": self.mlp_output_activation,
                "layer_indices": self.layer_indices,
                "mlp_norm": self.mlp_norm,
                "cnn_norm": self.cnn_norm,
                "device": self.device,
                "accelerator":self.accelerator,
                "in_channels": self.in_channels,
                "channel_size": self.channel_size,
                "kernel_size": self.kernel_size,
                "stride_size": self.stride_size, 
                "padding": self.padding, 
                "pooling": self.pooling,
                "pooling_kernel": self.pooling_kernel,
                "cnn_activation": self.cnn_activation
            }  
        
        return init_dict
     
    def add_mlp_layer(self):
        """Adds a hidden layer to Multi-layer Perceptron."""
        if len(self.hidden_size) < 3:  # HARD LIMIT
            self.hidden_size += [self.hidden_size[-1]]

            self.recreate_nets()
        else:
            self.add_mlp_node()

    def remove_mlp_layer(self):
        if len(self.hidden_size) > 1:  # HARD LIMIT
            self.hidden_size = self.hidden_size[:1]
            self.recreate_nets(shrink_params=True)
        else:
            self.add_mlp_node()

    def add_mlp_node(self, hidden_layer=None, numb_new_nodes=None):
        """Adds nodes to hidden layer of Multi-layer Perceptron.

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
            numb_new_nodes = np.random.choice([32, 64, 128], 1)[0]

        if self.hidden_size[hidden_layer] + numb_new_nodes <= 1024:  # HARD LIMIT
            self.hidden_size[hidden_layer] += numb_new_nodes

            self.recreate_nets()
        return {"hidden_layer": hidden_layer, "numb_new_nodes": numb_new_nodes}
    
    def remove_mlp_node(self, hidden_layer=None, numb_new_nodes=None): #
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
            self.hidden_size[hidden_layer] -= numb_new_nodes
            new_net = self.recreate_nets(shrink_params=True)

        return {"hidden_layer": hidden_layer, "numb_new_nodes": numb_new_nodes}

    def add_cnn_layer(self):
        """Adds a hidden layer to Convolutional Neural Network."""
        # if self.multi:
        #     if len(self.channel_size) < 6:  # HARD LIMIT
        #         self.channel_size += [self.channel_size[-1]]
        #         self.kernel_size += [(1, 3, 3)]
        #         stride_size_list = [
        #             [4],
        #             [4, 2],
        #             [4, 2, 1],
        #             [2, 2, 2, 1],
        #             [2, 1, 2, 1, 2],
        #             [2, 1, 2, 1, 2, 1],
        #         ]
        #         self.stride_size = stride_size_list[len(self.channel_size) - 1]

        #         self.recreate_nets()
        #     else:
        #         self.add_cnn_channel()

        # else:
        if len(self.channel_size) < 6:  # HARD LIMIT
            self.channel_size += [self.channel_size[-1]]
            self.kernel_size += [3]
            self.padding += [self.padding[-1]]
            stride_size_list = [
                [4],
                [4, 2],
                [4, 2, 1],
                [2, 2, 2, 1],
                [2, 1, 2, 1, 2],
                [2, 1, 2, 1, 2, 1],
            ]
            self.stride_size = stride_size_list[len(self.channel_size) - 1]

            # Add in an activation layer after the cnn layer
            if "cnn_act" not in self.layer_indices["cnn"].keys():
                self.layer_indices["cnn"]["cnn_act"] = []
            self.layer_indices["cnn"]["cnn_act"].append(len(self.channel_size) - 1)

            # Generate a random number to decide on 
            rand_num = self.rng.uniform(0, 1)

            if rand_num < 0.33:
                if "cnn_pool" not in self.layer_indices["cnn"].keys():
                    self.layer_indices["cnn"]["cnn_pool"] = []
                self.layer_indices["cnn"]["cnn_pool"].append(len(self.channel_size) - 1)
            elif 0.33 <= rand_num <= 0.66:
                if "cnn_norm" not in self.layer_indices["cnn"].keys():
                    self.layer_indices["cnn"]["cnn_norm"] = []
                self.layer_indices["cnn"]["cnn_norm"].append(len(self.channel_size) - 1)
            else:
                pass

            self.recreate_nets()
        else:
            self.add_cnn_channel()

    def change_cnn_kernel(self):
        """Randomly alters convolution kernel of random CNN layer."""

        # if self.multi:
        #     if len(self.channel_size) > 1:
        #         hidden_layer = np.random.randint(1, min(4, len(self.channel_size)), 1)[
        #             0
        #         ]
        #         kernel_size_value = np.random.choice([3, 4, 5, 7])
        #         if self.critic:
        #             self.kernel_size[hidden_layer] = tuple(
        #                 min(kernel_size_value, self.n_agents - 1)
        #                 if idx == 0
        #                 else kernel_size_value
        #                 for idx in range(3)
        #             )
        #         else:
        #             self.kernel_size[hidden_layer] = tuple(
        #                 1 if idx == 0 else kernel_size_value for idx in range(3)
        #             )
        #         self.recreate_nets()
        #     else:
        #         self.add_cnn_layer()
        # else:
        if len(self.channel_size) > 1:
            hidden_layer = np.random.randint(1, min(4, len(self.channel_size)), 1)[0]
            self.kernel_size[hidden_layer] = np.random.choice([3, 4, 5, 7])

            self.recreate_nets()
        else:
            self.add_cnn_layer()

    def add_cnn_channel(self, hidden_layer=None, numb_new_channels=None):
        """Adds channel to hidden layer of Convolutional Neural Network.

        :param hidden_layer: Depth of hidden layer to add channel to, defaults to None
        :type hidden_layer: int, optional
        :param numb_new_channels: Number of channels to add to hidden layer, defaults to None
        :type numb_new_channels: int, optional
        """
        if hidden_layer is None:
            hidden_layer = np.random.randint(0, len(self.channel_size), 1)[0]
        else:
            hidden_layer = min(hidden_layer, len(self.channel_size) - 1)
        if numb_new_channels is None:
            numb_new_channels = np.random.choice([8, 16, 32], 1)[0]

        if self.channel_size[hidden_layer] + numb_new_channels <= 256:  # HARD LIMIT
            self.channel_size[hidden_layer] += numb_new_channels

            self.recreate_nets()

    #     return {"hidden_layer": hidden_layer, "numb_new_channels": numb_new_channels}

    def recreate_nets(self, shrink_params=False):
        """Recreates neural networks."""
        new_feature_net, new_value_net, new_net = self.create_nets()
        if self.feature_net is not None:
            new_feature_net = self.preserve_parameters(
                old_net=self.feature_net, new_net=new_feature_net
            )
        if shrink_params:
            new_value_net = self.shrink_preserve_parameters(
            old_net=self.value_net, new_net=new_value_net
        )
        else:
            new_value_net = self.preserve_parameters(
                old_net=self.value_net, new_net=new_value_net
            )
        new_net = self.preserve_parameters(
            old_net=self.net, new_net=new_net
        )
        self.feature_net, self.value_net, self.net = (
            new_feature_net,
            new_value_net,
            new_net,
        )

    # def clone(self):
    #     """Returns clone of neural net with identical parameters."""
    #     clone = EvolvableCNN(**copy.deepcopy(self.init_dict))
    #     clone.load_state_dict(self.state_dict())
    #     clone.rainbow = self.rainbow
    #     clone.critic = self.critic
    #     return clone

    def clone(self): #
        """Returns clone of neural net with identical parameters."""
        clone = MakeEvolvable(**copy.deepcopy(self.init_dict))
        clone.load_state_dict(copy.deepcopy(self.state_dict()))
        return clone
    

    # def clone(self):
    #     clone = MakeEvolvable(**copy.deepcopy(self.init_dict))

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
                        elif len(param.data.size()) == 2:
                            param.data[
                                : min(old_size[0], new_size[0]),
                                : min(old_size[1], new_size[1]),
                            ] = old_net_dict[key].data[
                                : min(old_size[0], new_size[0]),
                                : min(old_size[1], new_size[1]),
                            ]
                        else:
                            param.data[
                                : min(old_size[0], new_size[0]),
                                : min(old_size[1], new_size[1]),
                                : min(old_size[2], new_size[2]),
                                : min(old_size[3], new_size[3]),
                            ] = old_net_dict[key].data[
                                : min(old_size[0], new_size[0]),
                                : min(old_size[1], new_size[1]),
                                : min(old_size[2], new_size[2]),
                                : min(old_size[3], new_size[3]),
                            ]

        return new_net

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




