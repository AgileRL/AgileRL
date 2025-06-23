#!/usr/bin/env python3

import torch
import sys
import os

# Add the agilerl directory to Python path so we can import our fixed function
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'agilerl'))

from agilerl.utils.algo_utils import chkpt_attribute_to_device

def test_chkpt_attribute_to_device_fix():
    """Test our fix for the chkpt_attribute_to_device function."""
    
    print("Testing chkpt_attribute_to_device fix...")
    
    # Test basic functionality first
    print("1. Testing basic tensor transfer...")
    basic_dict = {
        "tensor1": torch.ones((2, 2)),
        "tensor2": torch.zeros((3, 3)),
        "not_tensor": "string",
    }
    
    result = chkpt_attribute_to_device(basic_dict, "cpu")
    assert result["tensor1"].device.type == "cpu"
    assert result["tensor2"].device.type == "cpu"
    assert result["not_tensor"] == "string"
    print("✓ Basic tensor transfer works")
    
    # Test deeply nested structures (like optimizer states)
    print("2. Testing deeply nested optimizer-like structures...")
    nested_dict = {
        "state": {
            0: {
                "step": torch.tensor(100),
                "exp_avg": torch.ones((5, 5)),
                "exp_avg_sq": torch.zeros((5, 5)),
            },
            1: {
                "step": torch.tensor(50),
                "exp_avg": torch.ones((3, 3)),
                "exp_avg_sq": torch.zeros((3, 3)),
            }
        },
        "param_groups": [
            {
                "lr": 0.001,
                "betas": (0.9, 0.999),
                "eps": 1e-08,
                "weight_decay": 0,
                "amsgrad": False,
                "params": [0, 1]
            }
        ]
    }
    
    result_nested = chkpt_attribute_to_device(nested_dict, "cpu")
    
    # Check that deeply nested tensors were moved to device
    assert result_nested["state"][0]["exp_avg"].device.type == "cpu"
    assert result_nested["state"][0]["exp_avg_sq"].device.type == "cpu"
    assert result_nested["state"][0]["step"].device.type == "cpu"
    assert result_nested["state"][1]["exp_avg"].device.type == "cpu"
    assert result_nested["state"][1]["exp_avg_sq"].device.type == "cpu"
    assert result_nested["state"][1]["step"].device.type == "cpu"
    
    # Check that non-tensor values in nested structure are preserved
    assert result_nested["param_groups"][0]["lr"] == 0.001
    assert result_nested["param_groups"][0]["betas"] == (0.9, 0.999)
    assert result_nested["param_groups"][0]["params"] == [0, 1]
    print("✓ Deeply nested optimizer-like structures work")
    
    # Test mixed nested structures (lists in dicts, dicts in lists, etc.)
    print("3. Testing mixed nested structures...")
    mixed_structure = {
        "list_of_tensors": [torch.ones((2, 2)), torch.zeros((3, 3))],
        "dict_in_list": [
            {"nested_tensor": torch.ones((4, 4))},
            {"another_tensor": torch.zeros((2, 2))}
        ],
        "primitive_values": {
            "string": "test",
            "number": 42,
            "boolean": True,
            "none_value": None
        }
    }
    
    result_mixed = chkpt_attribute_to_device(mixed_structure, "cpu")
    
    # Check list of tensors
    assert result_mixed["list_of_tensors"][0].device.type == "cpu"
    assert result_mixed["list_of_tensors"][1].device.type == "cpu"
    
    # Check dict in list
    assert result_mixed["dict_in_list"][0]["nested_tensor"].device.type == "cpu"
    assert result_mixed["dict_in_list"][1]["another_tensor"].device.type == "cpu"
    
    # Check primitive values are preserved
    assert result_mixed["primitive_values"]["string"] == "test"
    assert result_mixed["primitive_values"]["number"] == 42
    assert result_mixed["primitive_values"]["boolean"] == True
    assert result_mixed["primitive_values"]["none_value"] is None
    print("✓ Mixed nested structures work")
    
    # Test with direct tensor (not in dict or list)
    print("4. Testing direct tensor...")
    direct_tensor = torch.ones((5, 5))
    result_direct = chkpt_attribute_to_device(direct_tensor, "cpu")
    assert isinstance(result_direct, torch.Tensor)
    assert result_direct.device.type == "cpu"
    print("✓ Direct tensor works")
    
    # Test with empty structures
    print("5. Testing empty structures...")
    empty_dict = {}
    empty_list = []
    empty_tuple = ()
    
    result_empty_dict = chkpt_attribute_to_device(empty_dict, "cpu")
    result_empty_list = chkpt_attribute_to_device(empty_list, "cpu")
    result_empty_tuple = chkpt_attribute_to_device(empty_tuple, "cpu")
    
    assert result_empty_dict == {}
    assert result_empty_list == []
    assert result_empty_tuple == ()
    print("✓ Empty structures work")
    
    print("\n🎉 All tests passed! The fix correctly handles deeply nested structures.")
    print("This should resolve the CUDA deserialization error when loading GPU-trained")
    print("models on CPU-only machines.")

if __name__ == "__main__":
    test_chkpt_attribute_to_device_fix()