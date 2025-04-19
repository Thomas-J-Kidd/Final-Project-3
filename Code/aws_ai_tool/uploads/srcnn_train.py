#!/usr/bin/env python3
"""
Real Super-Resolution Training Script for AWS AI Tool (SRCNN)

This script trains an SRCNN model for thermal image super-resolution (8x).
It loads image pairs (Low Resolution LR_x8, High Resolution GT), trains the model,
validates using PSNR, and saves the best performing model.

Usage:
    python srcnn_train.py --data_dir /path/to/dataset/thermal --output_dir /path/to/output --epochs 50 --lr 0.001 --batch_size 16

Arguments:
    --data_dir: Path to the base directory containing train/val/test splits (e.g., ~/dataset/thermal)
    --output_dir: Directory where the trained model and logs will be saved.
    --epochs: Number of training epochs (default: 50)
    --lr: Learning rate (default: 0.001)
    --batch_size: Training batch size (default: 16)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import numpy as np
import argparse
import time
import math
import sys
import traceback

# --- Dataset Class ---
class ThermalSRDataset(Dataset):
    """Dataset class for loading thermal image pairs (LR and HR)."""
    def __init__(self, lr_dir, hr_dir, transform=None):
        self.lr_dir = lr_dir
        self.hr_dir = hr_dir
        self.transform = transform
        # Ensure only files with matching names in both directories are used
        lr_files = set(os.listdir(lr_dir))
        hr_files = set(os.listdir(hr_dir))
        self.image_files = sorted(list(lr_files.intersection(hr_files)))
        if not self.image_files:
            raise RuntimeError(f"No matching image pairs found in {lr_dir} and {hr_dir}")
        print(f"Found {len(self.image_files)} matching image pairs.")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        lr_path = os.path.join(self.lr_dir, img_name)
        hr_path = os.path.join(self.hr_dir, img_name)

        try:
            # Thermal images are typically grayscale
            lr_image = Image.open(lr_path).convert('L')
            hr_image = Image.open(hr_path).convert('L')

            if self.transform:
                lr_image = self.transform(lr_image)
                hr_image = self.transform(hr_image)

            return lr_image, hr_image
        except Exception as e:
            print(f"Error loading image pair {img_name}: {e}")
            # Return dummy data or skip? For now, let's raise it.
            raise e

# --- Model Architecture (SRCNN) ---
class SRCNN(nn.Module):
    """Simple Super-Resolution Convolutional Neural Network."""
    def __init__(self, upscale_factor=8):
        super(SRCNN, self).__init__()
        # Feature extraction layer
        self.conv1 = nn.Conv2d(1, 64, kernel_size=9, padding=4)
        # Non-linear mapping layer
        self.conv2 = nn.Conv2d(64, 32, kernel_size=1, padding=0)
        # Reconstruction layer
        self.conv3 = nn.Conv2d(32, 1, kernel_size=5, padding=2)
        self.relu = nn.ReLU(inplace=True)

        # Note: SRCNN doesn't inherently handle upscaling within the network.
        # It expects the LR image to be pre-upscaled (e.g., using bicubic interpolation).
        # We will handle the upscaling before feeding the image to the network
        # or adjust the architecture if needed. For simplicity here, we assume
        # the input `x` to forward is already upscaled.
        # A common approach is to upscale first, then apply SRCNN.

    def forward(self, x):
        # Assume x is already upscaled to the target size if using this simple SRCNN
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x

# --- Utility Functions ---
def calculate_psnr(output, target):
    """Calculates Peak Signal-to-Noise Ratio."""
    mse = torch.mean((output - target) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 1.0 # Assuming pixel values are normalized to [0, 1]
    psnr = 20 * math.log10(max_pixel / math.sqrt(mse))
    return psnr

# --- Training Function ---
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, output_dir):
    """Trains the SRCNN model."""
    best_psnr = 0.0
    print(f"Starting training on device: {device}")
    sys.stdout.flush()

    # Pre-upscaling transform (bicubic)
    # Get target size from one HR image (assuming all HR images have the same size)
    try:
        _, hr_sample = next(iter(val_loader)) # Get one batch from validation loader
        target_size = hr_sample.shape[-2:] # Get H, W
        print(f"Target HR image size: {target_size}")
    except StopIteration:
        print("ERROR: Validation loader is empty. Cannot determine target size.")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: Could not get sample from validation loader: {e}")
        sys.exit(1)

    upscale_transform = transforms.Resize(target_size, interpolation=transforms.InterpolationMode.BICUBIC)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        epoch_start_time = time.time()

        for i, (lr_imgs, hr_imgs) in enumerate(train_loader):
            lr_imgs = lr_imgs.to(device)
            hr_imgs = hr_imgs.to(device)

            # Upscale LR images before feeding to SRCNN
            lr_imgs_upscaled = upscale_transform(lr_imgs)

            # Forward pass
            outputs = model(lr_imgs_upscaled)
            loss = criterion(outputs, hr_imgs)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if (i + 1) % 10 == 0: # Print progress every 10 batches
                print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')
                sys.stdout.flush()

        epoch_loss = running_loss / len(train_loader)
        epoch_end_time = time.time()
        print(f'Epoch {epoch+1}/{num_epochs} completed. Average Loss: {epoch_loss:.4f}. Time: {epoch_end_time - epoch_start_time:.2f}s')
        sys.stdout.flush()

        # --- Validation ---
        model.eval()
        val_psnr_total = 0.0
        val_start_time = time.time()

        with torch.no_grad():
            for lr_imgs, hr_imgs in val_loader:
                lr_imgs = lr_imgs.to(device)
                hr_imgs = hr_imgs.to(device)

                # Upscale LR images for validation input
                lr_imgs_upscaled = upscale_transform(lr_imgs)

                outputs = model(lr_imgs_upscaled)
                # Clamp output to valid range [0, 1] before PSNR calculation
                outputs_clamped = torch.clamp(outputs, 0.0, 1.0)
                val_psnr_total += calculate_psnr(outputs_clamped, hr_imgs)

        avg_val_psnr = val_psnr_total / len(val_loader)
        val_end_time = time.time()
        print(f'Validation PSNR: {avg_val_psnr:.2f} dB. Time: {val_end_time - val_start_time:.2f}s')
        sys.stdout.flush()

        # --- Save Best Model ---
        if avg_val_psnr > best_psnr:
            best_psnr = avg_val_psnr
            save_path = os.path.join(output_dir, 'srcnn_best.pth')
            torch.save(model.state_dict(), save_path)
            print(f'New best model saved to {save_path} with PSNR: {best_psnr:.2f} dB')
            sys.stdout.flush()

    print(f"Training finished. Best validation PSNR: {best_psnr:.2f} dB")
    return model

# --- Main Execution ---
def main():
    """Main function to parse args and run training."""
    parser = argparse.ArgumentParser(description='Thermal Image Super-Resolution Training (SRCNN)')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to dataset base directory (containing train/val)')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save outputs (model checkpoints)')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--fail', action='store_true', help='Simulate a training failure (for testing)') # Keep for compatibility

    args = parser.parse_args()

    print("Starting SRCNN training script...")
    print(f"Arguments: {args}")
    sys.stdout.flush()

    # Simulate failure if requested (keep for testing the framework)
    if args.fail:
        raise Exception("Simulated training failure requested via --fail flag")

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output directory: {args.output_dir}")

    # Set device (GPU if available, otherwise CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Define data paths based on the provided base directory
    train_lr_dir = os.path.join(args.data_dir, 'train/LR_x8')
    train_hr_dir = os.path.join(args.data_dir, 'train/GT')
    val_lr_dir = os.path.join(args.data_dir, 'val/LR_x8')
    val_hr_dir = os.path.join(args.data_dir, 'val/GT')

    # Basic check if directories exist
    if not os.path.isdir(train_lr_dir) or not os.path.isdir(train_hr_dir):
        print(f"ERROR: Training directories not found under {args.data_dir}/train/", file=sys.stderr)
        sys.exit(1)
    if not os.path.isdir(val_lr_dir) or not os.path.isdir(val_hr_dir):
        print(f"ERROR: Validation directories not found under {args.data_dir}/val/", file=sys.stderr)
        sys.exit(1)

    # Define data transformations
    # Normalize grayscale images to [0, 1]
    transform = transforms.Compose([
        transforms.ToTensor() # Converts PIL image (H x W x C) in range [0, 255] to Tensor (C x H x W) in range [0.0, 1.0]
    ])

    # Create datasets
    try:
        print("Loading training dataset...")
        train_dataset = ThermalSRDataset(train_lr_dir, train_hr_dir, transform=transform)
        print("Loading validation dataset...")
        val_dataset = ThermalSRDataset(val_lr_dir, val_hr_dir, transform=transform)
    except RuntimeError as e:
        print(f"ERROR: Failed to load datasets: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: An unexpected error occurred during dataset loading: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)


    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True if device == 'cuda' else False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True if device == 'cuda' else False)
    print("Data loaders created.")

    # Initialize the model
    model = SRCNN().to(device)
    print("SRCNN model initialized.")

    # Define loss function (Mean Squared Error is common for SR)
    # criterion = nn.MSELoss()
    # L1 Loss often works better for image restoration tasks
    criterion = nn.L1Loss()
    print(f"Using loss function: {criterion.__class__.__name__}")


    # Define optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    print(f"Using optimizer: Adam with lr={args.lr}")

    # Start training
    try:
        trained_model = train_model(model, train_loader, val_loader, criterion, optimizer, args.epochs, device, args.output_dir)

        # Save the final model state
        final_model_path = os.path.join(args.output_dir, 'srcnn_final.pth')
        torch.save(trained_model.state_dict(), final_model_path)
        print(f"Final model state saved to {final_model_path}")

        # Create success marker file for the framework
        success_marker_path = os.path.join(args.output_dir, '_SUCCESS')
        with open(success_marker_path, 'w') as f:
            f.write('SRCNN training completed successfully.\n')
        print(f"Success marker created at {success_marker_path}")
        sys.exit(0) # Exit with success code

    except Exception as e:
        print(f"ERROR: Training failed with exception: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        # Create failure marker file
        failure_marker_path = os.path.join(args.output_dir, '_FAILED')
        try:
            with open(failure_marker_path, 'w') as f:
                f.write(f'Training failed: {str(e)}\n')
                f.write(traceback.format_exc())
            print(f"Failure marker created at {failure_marker_path}")
        except Exception as marker_err:
            print(f"ERROR: Could not write failure marker: {marker_err}", file=sys.stderr)
        sys.exit(1) # Exit with error code

if __name__ == '__main__':
    main()
