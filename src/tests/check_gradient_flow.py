import torch
import sys
import os

# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.model import SineLayer

layer = SineLayer(3, 256, is_first=False, omega_0=30.0)

x = torch.randn(1, 3, requires_grad=True)
y = layer(x).sum()

y.backward()

print("Gradient magnitude:", x.grad.norm().item())
