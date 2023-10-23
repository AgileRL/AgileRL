import torch.nn as nn

def unpack_network(model):
    """Unpacks an nn.Sequential type model"""
    layer_list = []
    for layer in model.children():

        if isinstance(layer, nn.Sequential):
            # If it's an nn.Sequential, recursively unpack its children
            layer_list.extend(unpack_network(layer))
        else:
            if isinstance(layer, nn.Flatten):
                pass
            else:
                layer_list.append(layer)

    return layer_list