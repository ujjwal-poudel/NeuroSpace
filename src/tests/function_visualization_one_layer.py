import torch
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from src.model import SineLayer

layer = SineLayer(1, 1, is_first=True, omega_0=30.0)

x = torch.linspace(-1, 1, 1000).unsqueeze(1)
y = layer(x).detach().numpy()

plt.plot(x.numpy(), y)
plt.title("SineLayer Function After Initialization")
plt.xlabel("x")
plt.ylabel("Output")
plt.show()
