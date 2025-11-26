"""
This module defines the neural network components used in the NeuroSpace 3D-aware GAN.

Contents:
    1. SineLayer:
        A fully-connected layer with a sinusoidal activation function, used
        to implement SIREN (Sinusoidal Representation Networks). These layers
        allow the model to learn high-frequency signals such as textures,
        lighting variations, and geometric details.

    2. Generator:
        A SIREN-based implicit neural representation. The generator receives
        3D coordinates (sampled along camera rays) and predicts:
            - RGB color for each sample point
            - Density (sigma), representing how solid/opaque the point is
        This forms the core of an implicit 3D scene representation similar
        to NeRF. The generator on its own does not render images; it only
        maps 3D points to radiance and density.

    3. Discriminator:
        A convolutional neural network that receives a 2D image (real or
        generated) and outputs a scalar probability indicating whether the
        image is real. This is used for adversarial training within a GAN
        framework to guide the generator toward creating realistic outputs.

    4. Test Block:
        A simple sanity test that verifies the shapes and forward passes of
        the generator and discriminator. This ensures that the module runs
        without errors when executed directly.

This module does not handle:
    - Ray sampling
    - Volume rendering
    - Camera pose sampling
    - Training loops
    - Loss functions

These responsibilities belong to other components such as the renderer
and training script. This module strictly defines the neural architectures.
"""

import torch
import torch.nn as nn
import numpy as np


class SineLayer(nn.Module):
    """A fully-connected layer with a sinusoidal activation function.

    This layer implements the core building block of SIREN networks.
    The activation function is:
        sin(omega_0 * (W x + b))

    Using sine activations allows the network to model high-frequency
    details in signals, which is critical for implicit neural
    representations such as 3D scenes, textures, and fine geometry.

    Args:
        in_features (int): Number of input channels.
        out_features (int): Number of output channels.
        bias (bool): Whether to include a bias parameter. Defaults to True.
        is_first (bool): If True, apply special initialization suitable for
            the first SIREN layer. This is required for training stability.
        omega_0 (float): Frequency scaling factor for the sine activation.
            Larger values enable the layer to represent higher-frequency
            features.
    """

    def __init__(self, in_features, out_features, bias=True, is_first=False, omega_0=30.0):
        super().__init__()

        self.omega_0 = omega_0
        self.is_first = is_first
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        self.init_weights()

    def init_weights(self):
        """Initializes weights according to SIREN initialization rules.

        The first SIREN layer uses a uniform initialization scaled by the
        inverse of the number of input features. Subsequent layers use a
        scaled version of Xavier uniform initialization, divided by the
        frequency factor omega_0.
        """
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(
                    -1 / self.linear.in_features,
                    1 / self.linear.in_features
                )
            else:
                bound = np.sqrt(6 / self.linear.in_features) / self.omega_0
                self.linear.weight.uniform_(-bound, bound)

    def forward(self, x):
        """Applies a linear transform followed by sinusoidal activation.

        Args:
            x (Tensor): Input tensor of shape (..., in_features).

        Returns:
            Tensor: Activated output tensor of shape (..., out_features).
        """
        return torch.sin(self.omega_0 * self.linear(x))


class Generator(nn.Module):
    """Implicit 3D scene representation based on SIREN.

    The generator receives 3D coordinates sampled along camera rays and
    returns:
        - RGB color for each coordinate
        - Density (sigma), representing opacity at that point

    This is the core of the implicit neural field. A separate volumetric
    renderer is required to convert these point-wise predictions into a
    2D image.

    Args:
        input_dim (int): Dimensionality of input coordinates. Typically 3.
        hidden_dim (int): Width of hidden SineLayers.
        output_dim (int): Number of output channels (RGB + density = 4).
    """

    def __init__(self, input_dim=3, hidden_dim=256, output_dim=4):
        super().__init__()

        self.net = nn.Sequential(
            SineLayer(input_dim, hidden_dim, is_first=True, omega_0=30.0),
            SineLayer(hidden_dim, hidden_dim, omega_0=30.0),
            SineLayer(hidden_dim, hidden_dim, omega_0=30.0),
            SineLayer(hidden_dim, hidden_dim, omega_0=30.0),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, coords):
        """
        Maps 3D coordinates to color and density values.

        Accepts:
            coords of shape (N, 3)
            or
            coords of shape (B, N, 3)

        Returns:
            rgb:   same shape as coords but last dim is 3
            sigma: same shape but last dim is 1
        """
        orig_shape = coords.shape

        # If coords are (B, N, 3), flatten them
        if coords.dim() == 3:
            B, N, _ = coords.shape
            coords_flat = coords.reshape(B * N, 3)
        else:
            coords_flat = coords

        # Pass flattened points through network
        out = self.net(coords_flat)

        rgb = torch.sigmoid(out[:, :3])
        sigma = torch.abs(out[:, 3:4])

        # Restore shape if batched input was used
        if coords.dim() == 3:
            rgb = rgb.reshape(B, N, 3)
            sigma = sigma.reshape(B, N, 1)

        return rgb, sigma


class Discriminator(nn.Module):
    """Convolutional discriminator for adversarial training.

    This network receives a 2D image (real or generated) and outputs a
    probability indicating whether the image is real. It is a standard
    convolutional architecture with progressive downsampling.

    Args:
        image_size (int): Height/width of the input image. Defaults to 64.
    """

    def __init__(self, image_size=64):
        super().__init__()

        def block(in_channels, out_channels, batch_norm=True):
            layers = [
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout2d(0.25)
            ]
            if batch_norm:
                layers.append(nn.BatchNorm2d(out_channels, eps=1e-5))
            return layers

        self.model = nn.Sequential(
            *block(3, 16, batch_norm=False),
            *block(16, 32),
            *block(32, 64),
            *block(64, 128)
        )

        ds_size = image_size // (2 ** 4)

        self.adv_layer = nn.Sequential(
            nn.Linear(128 * ds_size * ds_size, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        """Evaluates image authenticity.

        Args:
            img (Tensor):
                Tensor of shape (batch_size, 3, H, W).

        Returns:
            Tensor:
                Scalar probability per image, shape (batch_size, 1).
        """
        features = self.model(img)
        features = features.view(features.size(0), -1)
        return self.adv_layer(features)


if __name__ == "__main__":
    print("Testing model components...")

    dummy_coords = torch.randn(8, 1000, 3)
    generator = Generator()
    rgb, sigma = generator(dummy_coords)
    print(f"Generator output shapes: RGB={rgb.shape}, Sigma={sigma.shape}")

    dummy_img = torch.randn(8, 3, 64, 64)
    discriminator = Discriminator()
    validity = discriminator(dummy_img)
    print(f"Discriminator output shape: {validity.shape}")
