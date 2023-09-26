import os
os.chdir("C:/Users/jako/data/custom_cough/utils/")

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from utils.helper_functions import training_grounds, initialize_weights, cut_when_needed, stretch_when_needed, create_3d_tensor_from_pca


class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc = nn.Linear(10, 1)

    def forward(self, x):
        return self.fc(x)

def test_training_grounds():
    # Create a simple toy model
    model = SimpleNN()

    # Generate random data
    data = torch.randn(100, 10)
    labels = torch.randint(0, 2, (100,))

    # Put them in a dataloader
    dataset = TensorDataset(data, labels)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters())
    
    train_loss, train_acc, precision, recall, f1 = training_grounds(model.to("cpu"), dataloader, loss_fn, optimizer, device="cpu") # ENFORCE CPU FOR THE TEST
    
    # Assertions to ensure the function is working correctly
    assert 0 <= train_loss <= np.inf, "Training loss should be positive"
    assert 0 <= train_acc <= 1, "Training accuracy should be between 0 and 1"
    assert 0 <= precision <= 1, "Precision should be between 0 and 1"
    assert 0 <= recall <= 1, "Recall should be between 0 and 1"
    assert 0 <= f1 <= 1, "F1 score should be between 0 and 1"
    
    

def test_initialize_weights():
    # Create a sample model

    model = SimpleNN()

    # Apply initialize_weights
    initialize_weights(model)

    # Check weights initialization
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            assert (module.weight.abs() > 0).all(), "Conv weights not initialized"
            if module.bias is not None:
                assert (module.bias == 0).all(), "Conv bias not initialized to 0"
        elif isinstance(module, nn.BatchNorm2d):
            assert (module.weight == 1).all(), "BatchNorm weight not initialized to 1"
            assert (module.bias == 0).all(), "BatchNorm bias not initialized to 0"
        elif isinstance(module, nn.Linear):
            assert (module.weight.abs() > 0).all(), "Linear weights not initialized"
            assert (module.bias == 0).all(), "Linear bias not initialized to 0"
            
            
def test_cut_when_needed():
    # Test longer signal
    signal = np.random.rand(7*16000 + 100)
    result = cut_when_needed(signal)
    assert len(result) == 7*16000, f"Expected length: {7*16000}, but got {len(result)}"

    # Test shorter signal
    signal = np.random.rand(7*16000 - 100)
    result = cut_when_needed(signal)
    assert len(result) == len(signal), f"Expected length: {len(signal)}, but got {len(result)}"

    # Test exact length
    signal = np.random.rand(7*16000)
    result = cut_when_needed(signal)
    assert len(result) == len(signal), f"Expected length: {len(signal)}, but got {len(result)}"

def test_stretch_when_needed():
    # Test shorter signal
    signal = np.random.rand(7*16000 - 100)
    result = stretch_when_needed(signal)
    assert len(result) == 7*16000, f"Expected length: {7*16000}, but got {len(result)}"

    # Test longer signal
    signal = np.random.rand(7*16000 + 100)
    result = stretch_when_needed(signal)
    assert len(result) == len(signal), f"Expected length: {len(signal)}, but got {len(result)}"

    # Test exact length
    signal = np.random.rand(7*16000)
    result = stretch_when_needed(signal)
    assert len(result) == len(signal), f"Expected length: {len(signal)}, but got {len(result)}"
    
    
    
def test_create_3d_tensor_from_pca():
    # Generate random PCA data
    pca_data = np.random.rand(500, 3) * 10  # 500 points in 3D space, values between 0 and 10
    
    # Generate 3D tensor
    tensor_3d = create_3d_tensor_from_pca(pca_data, shape=(22, 4, 4))
    
    # Test tensor shape
    assert tensor_3d.shape == (22, 4, 4), f"Expected shape (22, 4, 4), but got {tensor_3d.shape}"
    
    # Determine non-zero values in the tensor
    non_zero_values = tensor_3d > 0
    assert non_zero_values.sum() <= len(pca_data), "More non-zero values in the tensor than PCA data points"
    
    # Test that tensor contains zeros where no PCA points were located
    zero_values = tensor_3d == 0
    assert zero_values.sum() >= np.prod(tensor_3d.shape) - len(pca_data), "Not enough zeros in the tensor"