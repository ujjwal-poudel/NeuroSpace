import torch
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

# Bad init layer: random init
bad_layer = torch.nn.Linear(3, 256)

# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.model import SineLayer
good_layer = SineLayer(3, 256, is_first=False, omega_0=30.0)

x = torch.randn(10000, 3)

bad_out = torch.sin(bad_layer(x)).detach().numpy().flatten()
good_out = good_layer(x).detach().numpy().flatten()

plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.hist(bad_out, bins=200, density=True, color='red')
plt.title("Bad Init: Output Distribution")

plt.subplot(1,2,2)
plt.hist(good_out, bins=200, density=True, color='green')
plt.title("Good Init (SIREN): Output Distribution")

plt.show()