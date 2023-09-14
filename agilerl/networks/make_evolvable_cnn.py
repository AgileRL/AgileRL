import torch.nn as nn
import torch
import torch.nn.functional as F
import torch
import torch.nn as nn
from collections import deque
from collections import OrderedDict
import numpy as np

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


import torch.nn as nn

# Create an example neural network model
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()

        # Define the convolutional layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.conv = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)  # Batch Normalization for CNN layer 1
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)  # Batch Normalization for CNN layer 2

        # Define the max-pooling layers
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Define fully connected layers
        self.fc1 = nn.Linear(32 * 32 * 32, 128)  # Assuming input images are 128x128
        self.ln1 = nn.LayerNorm(128)  # Layer Normalization for FC layer 1
        self.fc2 = nn.Linear(128, num_classes)

        # Define activation function
        self.relu = nn.ReLU()

        self.softmax = nn.Softmax()

    def forward(self, x):
        # Forward pass through convolutional layers with Batch Normalization
        x = self.relu(self.bn1(self.conv(self.conv1(x))))
        x = self.pool(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)

        # Flatten the output for the fully connected layers
        x = x.view(x.size(0), -1)

        # Forward pass through fully connected layers with Layer Normalization
        x = self.relu(self.ln1(self.fc1(x)))
        x = self.softmax(self.fc2(x))

        return x

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
        # Set the layer counters
        self.conv_counter = -1
        self.lin_counter = -1
        self.mlp_norm = None
        self.cnn_norm = None
        self.has_conv_layers = False
        self.device = device 
        self.input_tensor = input_tensor.to(self.device)
        self.detect_architecture(network.to(self.device), self.input_tensor)
        print(self.create_cnn(self.in_channels, self.channel_size, self.kernel_size, self.stride_size, "feature"))
        self.accelerator = accelerator
        self.layer_norm = False
        self.init_layers = init_layers
        self.output_vanish = output_vanish
        self.init_layers = init_layers
        #self.net = self.create_net() # Reset the names of the layers to allow for cloning
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
    
    def layer_init(self, layer, std=np.sqrt(2), bias_const=0.0):
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_const)
        return layer
    
    def get_pooling(self, pooling_names, kernel_size):
        """Returns pooling layer for corresponding activation name.

        :param activation_names: Activation function name
        :type activation_names: str
        """
        pooling_dict = {
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

        return pooling_dict[pooling_names](kernel_size)

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
        normalization_layers = {
            "BatchNorm1d": nn.BatchNorm1d, 
            "BatchNorm2d": nn.BatchNorm2d, 
            "BatchNorm3d": nn.BatchNorm3d,
            "InstanceNorm1d": nn.InstanceNorm1d, 
            "InstanceNorm2d": nn.InstanceNorm2d, 
            "InstanceNorm3d": nn.InstanceNorm3d,
            "LayerNorm": nn.LayerNorm}

        return normalization_layers[normalization_name](layer_size)

    def detect_architecture(self, network, input_tensor):
        """Determine the architecture of a neural network.

        :param network: Neural network whose architecture is being detected
        :type network: nn.Module
        :param input_tensor: Tensor that will be passed into the network
        :type input_tensor: torch.Tensor
        """
        network_information= {}
        in_features_list = []
        out_features_list = []
        mlp_activations = []
        in_channel_list = []
        out_channel_list = []
        kernel_size_list = []
        stride_size_list = []
        layer_indices = {"cnn":{},
                         "mlp":{}}

        ## STORE THE LAYER INDEX OF THE DIFFERENT LAYERS!!!
        def register_hooks(module):
            def forward_hook(module, input, output):
                class_name = str(module.__class__.__name__)
                if isinstance(module, nn.modules.conv._ConvNd):
                    self.has_conv_layers = True
                    self.conv_counter += 1
                    in_channel_list.append(module.in_channels)
                    out_channel_list.append(module.out_channels)
                    kernel_size_list.append(module.kernel_size)
                    stride_size_list.append(module.stride)
                elif isinstance(module, nn.Linear):
                    self.lin_counter += 1
                    in_features_list.append(module.in_features)
                    out_features_list.append(module.out_features)
                elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
                                         nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d,
                                         nn.LayerNorm)):
                    self.normalization = True
                    print("hello from within the norm detection layer")
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

                else:
                    # Catch activation functions
                    network_information[f"{class_name}".lower()] = str(module.__class__)
                    if len(output.shape) <= 2:
                        mlp_activations.append(str(module.__class__.__name__))
                    else:
                        self.cnn_activation = str(module.__class__.__name__)

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
            self.in_channels = in_channel_list[0]
            self.channel_size = out_channel_list
            self.kernel_size = kernel_size_list
            self.stride_size = stride_size_list

        # print(network_information)

        self.cnn_counter = -1
        self.mlp_counter = -1

    def create_mlp(self, input_size, output_size, hidden_size, name):
        """Creates and returns multi-layer perceptron."""
        net_dict = OrderedDict()
        net_dict[f"{name}_linear_layer_0"] = nn.Linear(
            input_size, hidden_size[0])
        if self.mlp_norm is not None:
            net_dict[f"{name}_layer_norm_0"] = self.get_normalization(hidden_size[0])
        net_dict[f"{name}_activation_0"] = self.get_activation(self.mlp_activation)

        if len(hidden_size) > 1:
            for l_no in range(1, len(hidden_size)):
                net_dict[f"linear_layer_{str(l_no)}"] = nn.Linear(
                    hidden_size[l_no - 1], hidden_size[l_no]
                )
                if self.init_layers:
                    net_dict[f"linear_layer_{str(l_no)}"] = self.layer_init(
                        net_dict[f"linear_layer_{str(l_no)}"]
                    )
                if (self.mlp_norm is not None) and (l_no in self.layer_indices["mlp"]["mlp_norm"]):
                    net_dict[f"layer_norm_{str(l_no)}"] = self.get_normalization(
                        self.mlp_norm,
                        hidden_size[l_no]
                    )
                net_dict[f"activation_{str(l_no)}"] = self.get_activation(
                    self.mlp_activation
                )

        output_layer = nn.Linear(hidden_size[-1], output_size)
        if self.init_layers:
            output_layer = self.layer_init(output_layer)

        #### Do we need this with this class??
        if self.output_vanish:
            output_layer.weight.data.mul_(0.1)
            output_layer.bias.data.mul_(0.1)
        
        net_dict[f"{name}_linear_layer_output"] = output_layer
        if self.mlp_outout_activation is not None:
            net_dict["activation_output"] = self.get_activation(self.mlp_output_activation)

        return nn.Sequential(net_dict)
    
    
    def create_cnn(self, input_size, channel_size, kernel_size, stride_size, name):
        """Creates and returns convolutional neural network."""
        net_dict = OrderedDict()
        net_dict[f"{name}_conv_layer_0"] = nn.Conv2d(
            in_channels=input_size,
            out_channels=channel_size[0],
            kernel_size=kernel_size[0],
            stride=stride_size[0],
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
                )
                if (self.cnn_norm is not None) and (l_no in self.layer_indices["cnn"]["cnn_norm"]):
                    net_dict[f"{name}_layer_norm_{str(l_no)}"] = self.get_normalization(
                        self.cnn_norm,
                        channel_size[l_no])
                    
                if (self.pooling is not None) and (l_no in self.layer_indices["cnn"]["cnn_pool"]):
                    net_dict[f"{name}_layer_norm_{str(l_no)}"] = self.get_pooling(
                        self.pooling,
                        self.pooling_kernel)
                    
                net_dict[f"{name}_activation_{str(l_no)}"] = self.get_activation(
                    self.cnn_activation
                )

        return nn.Sequential(net_dict)


# Example usage:
# Instantiate the CNN model
device = "cuda" if torch.cuda.is_available() else "cpu"
num_classes = 10  # Change this to the number of classes in your dataset
input_tensor = torch.randn(1, 3, 128, 128)
model = SimpleCNN(num_classes)
evo = MakeEvolvable(model, input_tensor, device)
# Print the model architecture
print(evo.in_channels)
print(evo.channel_size)
print(evo.kernel_size)
print(evo.stride_size)
print("MLP Norm", evo.mlp_norm)
print("CNN Norm", evo.cnn_norm)
print(evo.normalization)
print(evo.has_conv_layers)
print(evo.mlp_activation)
print("MLP outputactivation:", evo.mlp_output_activation)
print(evo.cnn_activation)
print(evo.pooling)
print(evo.layer_indices)


