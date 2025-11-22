import os
import torch
from torch.utils.data import Dataset
from PIL import Image, UnidentifiedImageError
import torchvision.transforms as transforms


class LivingRoomDataset(Dataset):
    """Dataset for loading and preprocessing living-room images.

    This class scans a directory for image files, applies preprocessing
    transformations, and returns normalized image tensors. It serves as
    a reusable component that can be plugged into a DataLoader within
    training or inference scripts.

    Attributes:
        root_dir (str): Path to the dataset directory.
        image_paths (List[str]): List of valid image file paths.
        transform (torchvision.transforms.Compose): Transform pipeline applied to each image.
        image_size (int): Target resolution for output images.
    """

    def __init__(self, root_dir, image_size=64):
        """Initializes the dataset and prepares preprocessing steps.

        Args:
            root_dir (str): Directory containing image files.
            image_size (int, optional): Final image resolution (height and width).
                Defaults to 64.

        Raises:
            FileNotFoundError: If `root_dir` does not exist or is not a directory.
            RuntimeError: If no supported images are found in the directory.
        """
        if not os.path.isdir(root_dir):
            raise FileNotFoundError(f"Directory not found: {root_dir}")

        self.root_dir = root_dir
        self.image_size = image_size

        # Scan directory for supported image formats
        all_files = os.listdir(root_dir)
        self.image_paths = [
            os.path.join(root_dir, f)
            for f in all_files
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        if len(self.image_paths) == 0:
            raise RuntimeError(
                f"No valid image files found in '{root_dir}'. "
                "Supported formats: .jpg, .jpeg, .png"
            )

        # Preprocessing pipeline
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5]
            )
        ])

    def __len__(self):
        """Returns the total number of images in the dataset.

        Returns:
            int: Number of available images.
        """
        return len(self.image_paths)

    def __getitem__(self, idx):
        """Loads and returns the processed image at index `idx`.

        Args:
            idx (int): Index of the image to load.

        Returns:
            torch.Tensor: A normalized tensor of shape [3, image_size, image_size].

        Notes:
            If an image cannot be opened or decoded, a zero-filled tensor is
            returned instead. This prevents interruptions during training.
        """
        path = self.image_paths[idx]

        try:
            img = Image.open(path).convert("RGB")
            return self.transform(img)

        except (UnidentifiedImageError, OSError) as e:
            print(f"Warning: Failed to load '{path}': {e}. "
                  "Returning placeholder tensor.")
            return torch.zeros(3, self.image_size, self.image_size)


# Optional manual test block
if __name__ == "__main__":
    print("Testing dataset initialization...")

    test_path = "../data/living_room"
    dataset = LivingRoomDataset(test_path, image_size=64)

    print(f"Dataset contains {len(dataset)} images.")

    sample = dataset[0]
    print(f"Sample shape: {sample.shape}")
    print(f"Value range: min={sample.min():.3f}, max={sample.max():.3f}")

    import matplotlib.pyplot as plt
    import torchvision.transforms.functional as F

    # Undo normalization for display: x = (x * std) + mean
    img_tensor = sample * 0.5 + 0.5

    # Convert tensor → PIL image
    img = F.to_pil_image(img_tensor)

    # Display
    plt.imshow(img)
    plt.axis("off")
    plt.title("Preprocessed Image")
    plt.show()