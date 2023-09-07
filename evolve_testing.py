import torch
import torch.nn as nn
import json


# def analyze_network(model, input_tensor):
#     """
#     Analyze a PyTorch neural network by constructing a dictionary representing the forward graph.

#     Args:
#         model (nn.Module): The PyTorch model to analyze.
#         input_tensor (torch.Tensor): A sample input tensor to the model for shape inference.

#     Returns:
#         dict: A dictionary representing the forward graph of the network.
#     """
#     forward_graph = {}

#     def register_hooks(module):
#         def hook(module, input, output):
#             input_shape = input[0].shape if isinstance(input, tuple) else input.shape
#             output_shape = output.shape
#             forward_graph[str(module.__class__.__name__)] = {
#                 "Input Shape": input_shape,
#                 "Output Shape": output_shape
#             }

#         if (not isinstance(module, nn.Sequential) and not isinstance(module, nn.ModuleList)):# and not isinstance(module, CustomModel)):
#             hooks.append(module.register_forward_hook(hook))

#     hooks = []
#     model.apply(register_hooks)

#     # Forward pass to collect input and output shapes
#     with torch.no_grad():
#         model(input_tensor)

#     # Remove hooks
#     for hook in hooks:
#         hook.remove()

#     return forward_graph

# # Example usage:
# if __name__ == "__main__":
#     class CustomModel(nn.Module):
#         def __init__(self):
#             super(CustomModel, self).__init__()
#             self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
#             self.relu1 = nn.ReLU()
#             self.fc1 = nn.Linear(64 * 32 * 32, 128)
#             self.relu2 = nn.ReLU()
#             self.fc2 = nn.Linear(128, 10)

#         def forward(self, x):
#             x = self.conv1(x)
#             x = self.relu1(x)
#             x = x.view(x.size(0), -1)
#             x = self.fc1(x)
#             x = self.relu2(x)
#             x = self.fc2(x)
#             return x

#     custom_model = CustomModel()
#     input_tensor = torch.randn(1, 3, 32, 32)  # Sample input tensor

    
#     #Analyze the network and obtain the forward graph
#     forward_graph = analyze_network(custom_model, input_tensor)
#     print(forward_graph)

#     #Print the forward graph
#     for layer, info in forward_graph.items():
#         print(f"Layer: {layer}")
#         print(f"Input Shape: {info['Input Shape']}")
#         print(f"Output Shape: {info['Output Shape']}\n")

#     # for module in custom_model.modules():
#     #     print(module)

import torch
import torch.nn as nn

def analyze_network(model, input_tensor):
    """
    Analyze a PyTorch neural network by inspecting the input and output shapes of each layer.

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
            # input_shape = input[0].shape if isinstance(input, tuple) else input.shape
            # output_shape = output.shape
            # forward_graph[str(module.__class__.__name__)] = {
            #     "Input Shape": input_shape,
            #     "Output Shape": output_shape
            # }
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
                        
        if not isinstance(module, nn.Sequential) and not isinstance(module, nn.ModuleList) and not isinstance(module, type(custom_model)):
            hooks.append(module.register_forward_hook(forward_hook))
        
    hooks = []
    model.apply(register_hooks)

    # Forward pass to collect input and output shapes
    with torch.no_grad():
        model(input_tensor)

    # Remove hooks
    for hook in hooks:
        hook.remove()

    in_features = in_features_list[0]
    out_features = out_features_list[-1]
    hidden_size = in_features_list[1:]

    return network_information, in_features, out_features, hidden_size



# Example usage:
if __name__ == "__main__":
    import torch
import torch.nn as nn

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

# Example input tensor
input_size = 10
input_tensor = torch.rand(32, input_size)  # Batch size of 32, input size of 10

# Instantiate the CustomModel
hidden_sizes = [64, 128, 64]  # You can adjust these hidden layer sizes
output_size = 1  # Change this based on your task (e.g., regression, binary classification)
custom_model = CustomModel(input_size, hidden_sizes, output_size)


# Analyze the network and print layer information
layer_info, in_features, out_features, hidden_size = analyze_network(custom_model, input_tensor)
#print(json.dumps(layer_info, indent=4))
# for info in layer_info:
#     print(f"Layer: {info['Layer']}")
#     print(f"Input Shape: {info['Input Shape']}")
#     print(f"Output Shape: {info['Output Shape']}\n")
print("*"*50)
print(json.dumps(layer_info, indent=4))
print("In features", in_features)
print("Out features", out_features)
print("Hidden layer sizes", hidden_size)

