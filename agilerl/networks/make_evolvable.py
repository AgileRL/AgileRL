import torch
import torch.nn as nn
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    def forward(self, x):
        return self.model(x)

class MakeEvolvable(nn.Module):
    def __init__(self, network, input_tensor, device):
        super().__init__()
        self.network = network
        self.input_tensor = input_tensor
        self.device = device 
        self.network_information, self.in_features, self.out_features, self.hidden_layers = self.detect_architecture(self.input_tensor)

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

        return network_information, in_features, out_features, hidden_size



# Example input tensor
input_size = 10
input_tensor = torch.rand(32, input_size)  # Batch size of 32, input size of 10

# Instantiate the CustomModel
hidden_sizes = [64, 128, 64]  # You can adjust these hidden layer sizes
output_size = 1  # Change this based on your task (e.g., regression, binary classification)
custom_model = CustomModel(input_size, hidden_sizes, output_size)

evolvable_model = MakeEvolvable(custom_model, input_tensor, device)


print(evolvable_model.in_features, evolvable_model.out_features, evolvable_model.hidden_layers, evolvable_model.network_information)