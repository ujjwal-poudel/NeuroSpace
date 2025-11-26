"""
This module defines a simple pinhole camera model used to generate camera poses
and per-pixel camera rays for a NeRF-style renderer. The camera is placed at a
random position on a sphere around the scene. For each pixel in the output image,
the camera model computes an origin point and a direction vector, which together
define a ray that will be sampled during volumetric rendering.
"""

import torch
import torch.nn.functional as F
import math


class Camera:
    """
    A basic pinhole camera model for NeRF style ray generation.
    The camera supports:
    1. Creating random camera poses around the target scene.
    2. Generating ray origins and directions for every pixel.
    """

    def __init__(self, fov=60, device="cpu"):
        """
        Initialize the camera.

        Parameters:
            fov: Field of view in degrees.
            device: Torch device used for storing all tensors.
        """
        self.fov = fov
        self.device = device

    def get_random_pose(self, radius=4.0):
        """
        Generate a random camera pose on a sphere of the given radius.

        The camera position is sampled in spherical coordinates.
        The function then constructs a world-to-camera transformation matrix
        that contains right, up, forward axes and camera origin.

        Returns:
            pose: A 4x4 tensor representing the camera extrinsic matrix.
        """

        # Sample random spherical coordinates
        theta = torch.rand(1) * 2 * math.pi
        phi = torch.rand(1) * math.pi

        # Convert spherical coordinates to Cartesian (x, y, z)
        x = radius * torch.sin(phi) * torch.cos(theta)
        y = radius * torch.sin(phi) * torch.sin(theta)
        z = radius * torch.cos(phi)

        # Camera origin in world coordinates
        origin = torch.tensor([x, y, z], device=self.device).reshape(3)

        # Forward vector points toward the scene center (0,0,0)
        forward = F.normalize(-origin, dim=0)

        # Temporary up vector before orthogonalization
        up = torch.tensor([0.0, 1.0, 0.0], device=self.device)

        # Right vector is perpendicular to both up and forward
        right = F.normalize(torch.cross(up, forward), dim=0)

        # Now recompute a true orthogonal up vector
        up = F.normalize(torch.cross(forward, right), dim=0)

        # Build the 4x4 pose matrix
        pose = torch.eye(4, device=self.device)
        pose[0, :3] = right
        pose[1, :3] = up
        pose[2, :3] = forward
        pose[:3, 3] = origin

        return pose

    def generate_rays(self, H, W, pose):
        """
        Generate ray origins and directions for an image of size HxW.

        Every pixel corresponds to one ray. The ray origin is the camera position.
        The direction is obtained by projecting the pixel through the pinhole model
        and transforming the vector into world space using the camera pose.

        Parameters:
            H: Image height.
            W: Image width.
            pose: Camera pose matrix returned by get_random_pose.

        Returns:
            rays_o: Tensor of shape (H*W, 3) containing ray origins.
            rays_d: Tensor of shape (H*W, 3) containing ray directions.
        """

        # Create a grid of pixel coordinates
        i, j = torch.meshgrid(
            torch.linspace(0, W - 1, W, device=self.device),
            torch.linspace(0, H - 1, H, device=self.device),
            indexing="xy"
        )

        # Compute focal length from field of view
        focal = 0.5 * W / math.tan(0.5 * self.fov * math.pi / 180)

        # Create normalized device direction vectors in camera space
        dirs = torch.stack([
            (i - W * 0.5) / focal,
            -(j - H * 0.5) / focal,
            -torch.ones_like(i)
        ], dim=-1)

        # Extract rotation matrix from pose
        R = pose[:3, :3]

        # Rotate directions from camera space into world space
        rays_d = dirs.reshape(-1, 3) @ R.T
        rays_d = F.normalize(rays_d, dim=-1)

        # The origin of each ray is always the camera position
        rays_o = pose[:3, 3].expand(rays_d.shape)

        return rays_o, rays_d
