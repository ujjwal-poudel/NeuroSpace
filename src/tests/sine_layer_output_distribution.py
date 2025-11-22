import torch
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.model import SineLayer

# Test a single SineLayer (hidden layer)
layer = SineLayer(3, 256, is_first=False, omega_0=30.0)

# Random inputs
x = torch.randn(10000, 3)

# Forward pass
y = layer(x).detach().numpy()

plt.hist(y.flatten(), bins=200, density=True)
plt.title("Distribution of SineLayer Output After Initialization")
plt.xlabel("Value")
plt.ylabel("Density")
plt.show()