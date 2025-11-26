"""
train.py

This module defines the training loop for a SIREN-based NeRF style GAN using
a WGAN-GP objective. It is designed for a project structure where the source
code lives in a src directory, and data, checkpoints, and outputs live in
top level folders.

The training loop performs the following tasks:
    1. Loads images from the data directory using ImageFolderDataset.
    2. Initializes the SIREN generator, camera, renderer, and discriminator.
    3. Trains the models with WGAN-GP, logging detailed information every batch.
    4. Saves plots of losses and related metrics at the end of each epoch.
    5. Saves a "last" checkpoint once per epoch to support pause and resume.
    6. Saves a separate "best" checkpoint when the generator reaches a new best loss.
    7. Saves sample rendered images periodically for visual inspection.

The script supports an optional debug mode to run only a small number of batches
for quick sanity checks before launching a full training run.
"""

import os
import time
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch import autograd

import torchvision.utils as vutils
import matplotlib.pyplot as plt

from model import Generator, Discriminator
from camera import Camera
from renderer import render_image
from dataset import LivingRoomDataset

def get_project_paths():
    """
    Compute and return important project paths based on this file location.

    Returns:
        root_dir: Path to the project root directory.
        data_dir: Path to the data directory.
        checkpoints_dir: Path to the checkpoints directory.
        checkpoints_last_dir: Path to the "last" checkpoint directory.
        checkpoints_best_dir: Path to the "best" checkpoint directory.
        outputs_dir: Path to the outputs directory.
        samples_dir: Path to the directory for rendered sample images.
        plots_dir: Path to the directory for plots.
        logs_dir: Path to the directory for training logs.
    """
    root_dir = Path(__file__).resolve().parent.parent

    data_dir = root_dir / "data" / "living_room"
    checkpoints_dir = root_dir / "checkpoints"
    checkpoints_last_dir = checkpoints_dir / "last"
    checkpoints_best_dir = checkpoints_dir / "best"

    outputs_dir = root_dir / "outputs"
    samples_dir = outputs_dir / "samples"
    plots_dir = outputs_dir / "plots"
    logs_dir = outputs_dir / "logs"

    checkpoints_last_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_best_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)
    samples_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    return (
        root_dir,
        data_dir,
        checkpoints_dir,
        checkpoints_last_dir,
        checkpoints_best_dir,
        outputs_dir,
        samples_dir,
        plots_dir,
        logs_dir,
    )


def get_device():
    """
    Select the best available device.

    Priority order is:
        1. MPS for Apple Silicon.
        2. CUDA for NVIDIA GPUs.
        3. CPU as a fallback.

    Returns:
        A torch.device object.
    """
    if torch.backends.mps.is_available():
        print("Using MPS device.")
        return torch.device("mps")
    elif torch.cuda.is_available():
        print("Using CUDA device.")
        return torch.device("cuda")
    else:
        print("Using CPU device.")
        return torch.device("cpu")


def gradient_penalty(discriminator, real, fake, device, lambda_gp):
    """
    Compute the WGAN-GP gradient penalty.

    The gradient penalty encourages the L2 norm of the gradient of the
    discriminator output with respect to its input to be close to 1. This
    regularization improves the stability of training.

    Parameters:
        discriminator: The discriminator network.
        real: Batch of real images of shape (N, 3, H, W).
        fake: Batch of fake images of shape (N, 3, H, W).
        device: Torch device.
        lambda_gp: Weight for the gradient penalty term.

    Returns:
        gp: Scalar tensor representing the gradient penalty value.
    """
    batch_size = real.size(0)

    alpha = torch.rand(batch_size, 1, 1, 1, device=device)
    interpolates = alpha * real + (1.0 - alpha) * fake
    interpolates.requires_grad_(True)

    disc_interpolates = discriminator(interpolates)

    grad_outputs = torch.ones_like(disc_interpolates, device=device)

    gradients = autograd.grad(
        outputs=disc_interpolates,
        inputs=interpolates,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]

    gradients = gradients.view(batch_size, -1)
    grad_norm = gradients.norm(2, dim=1)

    gp = lambda_gp * ((grad_norm - 1.0) ** 2).mean()
    return gp


def generate_fake_batch(generator, camera, H, W, batch_size, num_samples, device):
    """
    Generate a batch of fake images by rendering from random camera poses.

    For each image in the batch:
        1. Sample a random camera pose.
        2. Generate rays for the target image resolution.
        3. Render the image using volumetric ray marching and the generator.
        4. Convert the image into a tensor in the range [-1, 1].

    Parameters:
        generator: Neural field model that maps 3D points to (rgb, sigma).
        camera: Camera instance used to create poses and rays.
        H: Image height in pixels.
        W: Image width in pixels.
        batch_size: Number of images in the batch.
        num_samples: Number of samples along each ray.
        device: Torch device.

    Returns:
        fake_images: Tensor of shape (batch_size, 3, H, W) with values in [-1, 1].
    """
    images = []
    generator.train()

    for _ in range(batch_size):
        pose = camera.get_random_pose()
        rays_o, rays_d = camera.generate_rays(H, W, pose)

        rays_o = rays_o.to(device)
        rays_d = rays_d.to(device)

        image = render_image(
            generator=generator,
            rays_o=rays_o,
            rays_d=rays_d,
            H=H,
            W=W,
            num_samples=num_samples
        )

        image = image.permute(2, 0, 1)
        image = image.clamp(0.0, 1.0)
        image = image * 2.0 - 1.0

        images.append(image)

    fake_images = torch.stack(images, dim=0)
    return fake_images


def save_sample_images(fake_images, step, samples_dir, nrow=4):
    """
    Save a grid of rendered fake images to disk.

    Parameters:
        fake_images: Tensor of shape (N, 3, H, W) in range [-1, 1].
        step: Global training step, used for composing the filename.
        samples_dir: Directory where sample images will be stored.
        nrow: Number of images per row in the saved grid.
    """
    imgs = (fake_images + 1.0) / 2.0
    imgs = imgs.clamp(0.0, 1.0)

    samples_dir = Path(samples_dir)
    samples_dir.mkdir(parents=True, exist_ok=True)

    file_path = samples_dir / f"samples_step_{step:06d}.png"
    vutils.save_image(imgs, file_path, nrow=nrow)
    print(f"Saved sample images to {file_path}")


def save_plots(history, epoch, plots_dir):
    """
    Save plots of training metrics up to the given epoch.

    The function produces a plot of discriminator loss, generator loss, and
    gradient penalty over training steps, and saves it to the plots directory.

    Parameters:
        history: Dictionary with keys such as "global_step", "loss_d",
                 "loss_g", and "gp", each mapped to a list of values.
        epoch: Current epoch index (0-based).
        plots_dir: Directory where plots will be stored.
    """
    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    steps = history["global_step"]
    loss_d = history["loss_d"]
    loss_g = history["loss_g"]
    gp = history["gp"]

    if len(steps) == 0:
        print("History is empty, skipping plot saving.")
        return

    plt.figure(figsize=(10, 6))
    plt.plot(steps, loss_d, label="D loss")
    plt.plot(steps, loss_g, label="G loss")
    plt.plot(steps, gp, label="Gradient penalty")
    plt.xlabel("Global step")
    plt.ylabel("Loss")
    plt.title(f"Training losses up to epoch {epoch + 1}")
    plt.legend()
    loss_plot_path = plots_dir / f"losses_until_epoch_{epoch + 1:03d}.png"
    plt.tight_layout()
    plt.savefig(loss_plot_path)
    plt.close()

    print(f"Saved loss plots to {loss_plot_path}")


def save_checkpoint(generator, discriminator, opt_g, opt_d,
                    epoch, global_step, best_g_loss, history,
                    checkpoints_last_dir, checkpoints_best_dir, is_best=False):
    """
    Save training state to checkpoint files.

    This function always writes a "last" checkpoint used for pause and resume,
    and optionally writes a separate "best" checkpoint when the generator
    reaches a new best loss.

    Parameters:
        generator: Generator model instance.
        discriminator: Discriminator model instance.
        opt_g: Optimizer for the generator.
        opt_d: Optimizer for the discriminator.
        epoch: Current epoch index (0-based).
        global_step: Current global training step.
        best_g_loss: Best generator loss seen so far.
        history: Training history dictionary.
        checkpoints_last_dir: Directory for "last" checkpoint files.
        checkpoints_best_dir: Directory for "best" checkpoint files.
        is_best: If True, also save a best checkpoint file.
    """
    checkpoints_last_dir = Path(checkpoints_last_dir)
    checkpoints_best_dir = Path(checkpoints_best_dir)
    checkpoints_last_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_best_dir.mkdir(parents=True, exist_ok=True)

    state = {
        "epoch": epoch,
        "global_step": global_step,
        "best_g_loss": best_g_loss,
        "generator_state": generator.state_dict(),
        "discriminator_state": discriminator.state_dict(),
        "opt_g_state": opt_g.state_dict(),
        "opt_d_state": opt_d.state_dict(),
        "history": history,
    }

    last_path = checkpoints_last_dir / "checkpoint_last.pt"
    torch.save(state, last_path)
    print(f"Saved last checkpoint to {last_path}")

    if is_best:
        best_path = checkpoints_best_dir / "checkpoint_best.pt"
        torch.save(state, best_path)
        print(f"Saved best checkpoint to {best_path}")


def load_checkpoint_if_available(generator, discriminator, opt_g, opt_d,
                                 checkpoints_last_dir, checkpoints_best_dir, device):
    """
    Load the latest checkpoint if it exists.

    Priority is:
        1. checkpoint_last.pt in the "last" folder.
        2. checkpoint_best.pt in the "best" folder.
        3. Start from scratch if none are found.

    Parameters:
        generator: Generator model instance.
        discriminator: Discriminator model instance.
        opt_g: Optimizer for the generator.
        opt_d: Optimizer for the discriminator.
        checkpoints_last_dir: Directory containing last checkpoint files.
        checkpoints_best_dir: Directory containing best checkpoint files.
        device: Torch device.

    Returns:
        epoch_start: Epoch index to resume from.
        global_step: Global training step to resume from.
        best_g_loss: Best generator loss seen so far.
        history: Training history dictionary.
    """
    checkpoints_last_dir = Path(checkpoints_last_dir)
    checkpoints_best_dir = Path(checkpoints_best_dir)

    last_path = checkpoints_last_dir / "checkpoint_last.pt"
    best_path = checkpoints_best_dir / "checkpoint_best.pt"

    history_template = {
        "global_step": [],
        "epoch": [],
        "batch_idx": [],
        "loss_d": [],
        "loss_g": [],
        "gp": [],
        "d_real_mean": [],
        "d_fake_mean": [],
    }

    if last_path.exists():
        print(f"Loading checkpoint from {last_path}")
        state = torch.load(last_path, map_location=device)
    elif best_path.exists():
        print(f"No last checkpoint found. Loading best checkpoint from {best_path}")
        state = torch.load(best_path, map_location=device)
    else:
        print("No checkpoint found. Starting from scratch.")
        return 0, 0, float("inf"), history_template

    generator.load_state_dict(state["generator_state"])
    discriminator.load_state_dict(state["discriminator_state"])
    opt_g.load_state_dict(state["opt_g_state"])
    opt_d.load_state_dict(state["opt_d_state"])

    epoch_start = state["epoch"] + 1
    global_step = state["global_step"]
    best_g_loss = state["best_g_loss"]
    history = state.get("history", history_template)

    print(
        f"Resumed from epoch {epoch_start}, global_step {global_step}, "
        f"best_g_loss {best_g_loss:.4f}"
    )

    return epoch_start, global_step, best_g_loss, history


def create_log_paths(logs_dir, checkpoints_last_dir):
    """
    Create and return paths for training log files.

    This function creates two CSV log files if they do not exist:
        1. One in the outputs logs directory.
        2. One mirrored in the checkpoints last directory.

    Both files share the same header.

    Parameters:
        logs_dir: Directory under outputs for main log files.
        checkpoints_last_dir: Directory where last checkpoints are stored.

    Returns:
        log_path_outputs: Path to the main log file in outputs/logs.
        log_path_last: Path to the mirrored log file in checkpoints/last.
    """
    logs_dir = Path(logs_dir)
    logs_dir.mkdir(parents=True, exist_ok=True)

    checkpoints_last_dir = Path(checkpoints_last_dir)
    checkpoints_last_dir.mkdir(parents=True, exist_ok=True)

    log_path_outputs = logs_dir / "training_log.csv"
    log_path_last = checkpoints_last_dir / "training_log.csv"

    header = (
        "global_step,epoch,batch_idx,"
        "loss_d,loss_g,gp,"
        "d_real_mean,d_fake_mean\n"
    )

    if not log_path_outputs.exists():
        with log_path_outputs.open("w") as f:
            f.write(header)

    if not log_path_last.exists():
        with log_path_last.open("w") as f:
            f.write(header)

    return log_path_outputs, log_path_last


def append_log_entry(log_paths, global_step, epoch, batch_idx,
                     loss_d, loss_g, gp, d_real_mean, d_fake_mean):
    """
    Append a single training record to both log files.

    Parameters:
        log_paths: Tuple of (log_path_outputs, log_path_last).
        global_step: Global training step.
        epoch: Current epoch index.
        batch_idx: Current batch index within the epoch.
        loss_d: Total discriminator loss.
        loss_g: Generator loss.
        gp: Gradient penalty.
        d_real_mean: Mean discriminator output on real images.
        d_fake_mean: Mean discriminator output on fake images.
    """
    log_path_outputs, log_path_last = log_paths

    line = (
        f"{global_step},{epoch},{batch_idx},"
        f"{loss_d:.6f},{loss_g:.6f},{gp:.6f},"
        f"{d_real_mean:.6f},{d_fake_mean:.6f}\n"
    )

    for log_path in (log_path_outputs, log_path_last):
        with log_path.open("a") as f:
            f.write(line)


def main():
    """
    Main training function.

    This function sets up paths, dataset, models, optimizers, logging,
    checkpointing, and runs the WGAN-GP training loop. It supports resuming
    from the last or best checkpoint and saves only two model checkpoints:
    one for the last state and one for the best generator seen so far.
    """
    (
        root_dir,
        data_dir,
        checkpoints_dir,
        checkpoints_last_dir,
        checkpoints_best_dir,
        outputs_dir,
        samples_dir,
        plots_dir,
        logs_dir,
    ) = get_project_paths()

    device = get_device()

    img_size = 64
    batch_size = 4
    num_workers = 4

    num_epochs = 100
    critic_iters = 5
    lambda_gp = 10.0
    lr_d = 1e-4
    lr_g = 1e-4

    num_samples_per_ray = 32

    sample_interval = 200

    DEBUG_MODE = False  
    DEBUG_MAX_BATCHES = 3

    if DEBUG_MODE:
        print("Running in DEBUG MODE: only a few batches will be processed.")
        num_epochs = 1

    dataset = LivingRoomDataset(str(data_dir), image_size=img_size)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    generator = Generator().to(device)
    discriminator = Discriminator().to(device)
    camera = Camera(device=device)

    optimizer_d = optim.Adam(discriminator.parameters(), lr=lr_d, betas=(0.0, 0.9))
    optimizer_g = optim.Adam(generator.parameters(), lr=lr_g, betas=(0.0, 0.9))

    log_paths = create_log_paths(logs_dir, checkpoints_last_dir)

    epoch_start, global_step, best_g_loss, history = load_checkpoint_if_available(
        generator=generator,
        discriminator=discriminator,
        opt_g=optimizer_g,
        opt_d=optimizer_d,
        checkpoints_last_dir=checkpoints_last_dir,
        checkpoints_best_dir=checkpoints_best_dir,
        device=device
    )

    if not history or len(history.get("global_step", [])) == 0:
        history = {
            "global_step": [],
            "epoch": [],
            "batch_idx": [],
            "loss_d": [],
            "loss_g": [],
            "gp": [],
            "d_real_mean": [],
            "d_fake_mean": [],
        }

    start_time = time.time()
    print("Starting training...")

    try:
        for epoch in range(epoch_start, num_epochs):
            for batch_idx, real_images in enumerate(dataloader):
                real_images = real_images.to(device)

                for _ in range(critic_iters):
                    fake_images = generate_fake_batch(
                        generator=generator,
                        camera=camera,
                        H=img_size,
                        W=img_size,
                        batch_size=real_images.size(0),
                        num_samples=num_samples_per_ray,
                        device=device
                    )

                    d_real = discriminator(real_images)
                    d_fake = discriminator(fake_images.detach())

                    loss_d_wgan = -(d_real.mean() - d_fake.mean())

                    gp = gradient_penalty(
                        discriminator=discriminator,
                        real=real_images,
                        fake=fake_images.detach(),
                        device=device,
                        lambda_gp=lambda_gp
                    )

                    total_d_loss = loss_d_wgan + gp

                    optimizer_d.zero_grad()
                    total_d_loss.backward()
                    optimizer_d.step()

                fake_images = generate_fake_batch(
                    generator=generator,
                    camera=camera,
                    H=img_size,
                    W=img_size,
                    batch_size=real_images.size(0),
                    num_samples=num_samples_per_ray,
                    device=device
                )

                d_fake_for_g = discriminator(fake_images)
                loss_g = -d_fake_for_g.mean()

                optimizer_g.zero_grad()
                loss_g.backward()
                optimizer_g.step()

                global_step += 1

                d_real_mean = d_real.mean().item()
                d_fake_mean = d_fake.mean().item()
                loss_d_value = total_d_loss.item()
                loss_g_value = loss_g.item()
                gp_value = gp.item()

                history["global_step"].append(global_step)
                history["epoch"].append(epoch)
                history["batch_idx"].append(batch_idx)
                history["loss_d"].append(loss_d_value)
                history["loss_g"].append(loss_g_value)
                history["gp"].append(gp_value)
                history["d_real_mean"].append(d_real_mean)
                history["d_fake_mean"].append(d_fake_mean)

                append_log_entry(
                    log_paths=log_paths,
                    global_step=global_step,
                    epoch=epoch,
                    batch_idx=batch_idx,
                    loss_d=loss_d_value,
                    loss_g=loss_g_value,
                    gp=gp_value,
                    d_real_mean=d_real_mean,
                    d_fake_mean=d_fake_mean
                )

                elapsed = time.time() - start_time
                print(
                    f"Epoch [{epoch + 1}/{num_epochs}] "
                    f"Batch [{batch_idx + 1}/{len(dataloader)}] "
                    f"Step [{global_step}] "
                    f"D_loss: {loss_d_value:.4f} "
                    f"G_loss: {loss_g_value:.4f} "
                    f"GP: {gp_value:.4f} "
                    f"D_real: {d_real_mean:.4f} "
                    f"D_fake: {d_fake_mean:.4f} "
                    f"Elapsed: {elapsed/60.0:.2f} min"
                )

                if global_step % sample_interval == 0:
                    with torch.no_grad():
                        fake_images_eval = generate_fake_batch(
                            generator=generator,
                            camera=camera,
                            H=img_size,
                            W=img_size,
                            batch_size=8,
                            num_samples=num_samples_per_ray,
                            device=device
                        )
                    save_sample_images(fake_images_eval, global_step, samples_dir, nrow=4)

                if DEBUG_MODE and (batch_idx + 1) >= DEBUG_MAX_BATCHES:
                    print("Debug mode limit reached. Saving state and exiting early.")
                    save_plots(history, epoch, plots_dir)
                    save_checkpoint(
                        generator=generator,
                        discriminator=discriminator,
                        opt_g=optimizer_g,
                        opt_d=optimizer_d,
                        epoch=epoch,
                        global_step=global_step,
                        best_g_loss=best_g_loss,
                        history=history,
                        checkpoints_last_dir=checkpoints_last_dir,
                        checkpoints_best_dir=checkpoints_best_dir,
                        is_best=False
                    )
                    return

            is_best = False
            last_g_loss_epoch = history["loss_g"][-1]
            if last_g_loss_epoch < best_g_loss:
                best_g_loss = last_g_loss_epoch
                is_best = True

            save_checkpoint(
                generator=generator,
                discriminator=discriminator,
                opt_g=optimizer_g,
                opt_d=optimizer_d,
                epoch=epoch,
                global_step=global_step,
                best_g_loss=best_g_loss,
                history=history,
                checkpoints_last_dir=checkpoints_last_dir,
                checkpoints_best_dir=checkpoints_best_dir,
                is_best=is_best
            )

            save_plots(history, epoch, plots_dir)
            print(f"Finished epoch {epoch + 1}/{num_epochs}")

    except KeyboardInterrupt:
        print("KeyboardInterrupt received. Saving last checkpoint before exiting.")
        save_checkpoint(
            generator=generator,
            discriminator=discriminator,
            opt_g=optimizer_g,
            opt_d=optimizer_d,
            epoch=epoch,
            global_step=global_step,
            best_g_loss=best_g_loss,
            history=history,
            checkpoints_last_dir=checkpoints_last_dir,
            checkpoints_best_dir=checkpoints_best_dir,
            is_best=False
        )

    total_time = time.time() - start_time
    print(f"Training complete. Total time: {total_time / 60.0:.2f} minutes")


if __name__ == "__main__":
    main()
