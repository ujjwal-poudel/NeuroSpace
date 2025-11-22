# NeuroSpace: Unsupervised 3D Living Room Synthesis

**NeuroSpace** is an unsupervised deep learning research project focused on **Single-Shot 3D Reconstruction** from 2D images. 

Unlike traditional photogrammetry which requires multiple angles of the same object, NeuroSpace learns to hallucinate the 3D geometry of a room from a single flat photo by understanding the statistical "physics" of light and perspective.

---

## The Ultimate Goal
To build a **"Digital Twin" Generator**:
1.  Input: A single 2D photo of a living room (e.g., from a real estate listing).
2.  Output: A fully rotatable **3D Neural Radiance Field (NeRF)**.
3.  Application: Generate 3D walkthroughs or "fly-through" videos from static 2D datasets without human supervision or 3D labels.

## The Architecture (Unsupervised)
This project implements a **3D-Aware Generative Adversarial Network (GAN)** based on the **pi-GAN** and **GRAF** research papers.

* **Generator (The Artist):** A **SIREN (Sinusoidal Representation Network)**. Instead of standard ReLU layers, it uses periodic sine-wave activation functions to learn high-frequency 3D details (texture, rug patterns) efficiently.
* **Renderer:** A Volumetric Ray Marcher that projects the 3D feature field into 2D images.
* **Discriminator (The Critic):** A Convolutional Neural Network (CNN) that ensures the rendered 2D images are indistinguishable from real photos.

**Why Unsupervised?**
The model is never given a 3D model or a depth map. It learns geometry purely by trying to fool the discriminator from random camera angles.

## The Dataset
We utilize the **LSUN (Large-scale Scene Understanding) - Living Room** dataset.
* **Subset:** Living Room category only.
* **Scale:** Training on a "Low-Data" regime (approx. 5,000–10,000 images).
* **Preprocessing:** Center-cropped and resized to 64x64 (for M3 optimization).

## Tech Stack & Hardware
* **Framework:** PyTorch 2.4 (Stable)
* **Key Libraries:** `siren-pytorch`, `numpy`, `imageio`
* **Hardware Optimization:**
    * Designed for **Apple Silicon (M3/M1)** using MPS acceleration.
    * Compatible with **NVIDIA GPUs (CUDA)** via a device-agnostic driver.

## Roadmap (Subject to Change)
- [x] Project Skeleton & Environment Setup
- [x] Data Engineering Pipeline (LSUN Loader)
- [ ] Implement SIREN-based 3D Generator
- [ ] Implement Volumetric Rendering & Discriminator
- [ ] Training Loop with "Graceful Exit" (Auto-Resume)
- [ ] Inference: Generate 3D Rotation Video from a static image

---
*Status: Active Research / Work in Progress*