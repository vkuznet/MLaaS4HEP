"""
Basic example of ML model implemented via pytorch framework
"""
from MLaaS4HEP.jarray.pytorch import JaggedArrayLinear
try:
    import torch
except ImportError:
    torch = None

def model(idim):
    "Simple PyTorch model for testing purposes"
    if not torch:
        raise Exception("Please install pytorch package")
    ml_model = torch.nn.Sequential(
        JaggedArrayLinear(idim, 5),
        torch.nn.ReLU(),
        torch.nn.Linear(5, 1),
    )
    return ml_model
