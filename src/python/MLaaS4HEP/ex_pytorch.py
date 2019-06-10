"""
Basic example of ML model implemented via pytorch framework
"""
from MLaaS4HEP.jarray.pytorch import JaggedArrayLinear
import torch

def model(idim):
    "Simple PyTorch model for testing purposes"
    model = torch.nn.Sequential(
        JaggedArrayLinear(idim, 5),
        torch.nn.ReLU(),
        torch.nn.Linear(5, 1),
    )
    return model


