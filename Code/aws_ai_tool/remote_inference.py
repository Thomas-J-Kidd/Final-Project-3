#!/usr/bin/env python3
"""
Remote Inference Script for AWS AI Tool

This script runs on the remote AWS instance to perform inference using a trained model.
It takes an input image, processes it through the model, and saves the output image.

Usage:
    python remote_inference.py --model_path /path/to/model.pth --input_path /path/to/input.jpg --output_path /path/to/output.jpg

Arguments:
    --model_path: Path to the trained model file (.pth)
    --input_path: Path to the input image file
    --output_path: Path where the output image should be saved
"""

import argparse
import os
import sys
import traceback
import time
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import math # For PSNR calculation if needed

# --- Model Architecture (SRCNN) ---
# Needs to be defined here to load the state_dict
class SRCNN(nn.Module):
    """Simple Super-Resolution Convolutional Neural Network."""
    def __init__(self):
        super(SRCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=9, padding=4)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=1, padding=0)
        self.conv3 = nn.Conv2d(32, 1, kernel_size=5, padding=2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Remote Inference Script")
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model file')
    parser.add_argument('--input_path', type=str, required=True, help='Path to the input image file')
    parser.add_argument('--output_path', type=str, required=True, help='Path where the output image should be saved')
    parser.add_argument('--fail', action='store_true', help='Simulate a failure (for testing)')
    
    return parser.parse_args()

def load_model(model_path):
    """Load the trained model from the given path."""
    print(f"Loading model from {model_path}...")
    # In a real implementation, you would use something like:
    # model = torch.load(model_path)
    # model.eval()
    
    # Use torch.load to load the model state dictionary
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SRCNN().to(device)
    try:
        # Load the state dictionary
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval() # Set the model to evaluation mode
        print(f"Model loaded successfully from {model_path}")
        return model, device
    except FileNotFoundError:
        print(f"ERROR: Model file not found at {model_path}", file=sys.stderr)
        raise
    except Exception as e:
        print(f"ERROR: Failed to load model state_dict from {model_path}: {e}", file=sys.stderr)
        raise

def load_and_preprocess_image(image_path, device):
    """Load, preprocess, and upscale the input image."""
    print(f"Loading image from {image_path}...")
    try:
        img = Image.open(image_path).convert('L') # Load as grayscale
        
        # Define the preprocessing transform (just ToTensor for SRCNN)
        preprocess = transforms.Compose([
            transforms.ToTensor() # Converts to tensor and scales to [0, 1]
        ])
        
        input_tensor = preprocess(img).unsqueeze(0).to(device) # Add batch dimension and send to device

        # SRCNN expects pre-upscaled input. We need a target size.
        # This is tricky without knowing the HR size. Let's assume 8x upscale.
        # A better approach would be to pass target dimensions or get them from the model/training.
        # For now, let's upscale by 8x using bicubic interpolation.
        original_size = img.size
        target_size = (original_size[1] * 8, original_size[0] * 8) # H, W
        print(f"Upscaling input image from {original_size} to {target_size} using Bicubic interpolation...")
        
        upscale_transform = transforms.Resize(target_size, interpolation=transforms.InterpolationMode.BICUBIC)
        input_tensor_upscaled = upscale_transform(input_tensor)

        print(f"Image loaded and preprocessed. Shape: {input_tensor_upscaled.shape}")
        return input_tensor_upscaled
    except FileNotFoundError:
        print(f"ERROR: Input image file not found at {image_path}", file=sys.stderr)
        raise
    except Exception as e:
        print(f"ERROR: Failed to load or preprocess image {image_path}: {e}", file=sys.stderr)
        raise


def run_inference(model, input_tensor, device):
    """Run inference using the loaded model and preprocessed input tensor."""
    print(f"Running inference on device: {device}...")
    start_time = time.time()
    with torch.no_grad(): # Disable gradient calculations for inference
        output_tensor = model(input_tensor)
    end_time = time.time()
    print(f"Inference completed in {end_time - start_time:.2f} seconds.")
    return output_tensor

def save_output_image(output_tensor, output_path):
    """Save the output tensor as an image file."""
    print(f"Saving output image to {output_path}...")
    try:
        # Ensure the output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Convert tensor to PIL Image
        # Clamp values to [0, 1] and scale back to [0, 255]
        output_tensor_clamped = torch.clamp(output_tensor.squeeze(0), 0.0, 1.0)
        output_image = transforms.ToPILImage()(output_tensor_clamped.cpu()) # Move to CPU before converting

        output_image.save(output_path)
        print(f"Output image saved successfully to {output_path}")
    except Exception as e:
        print(f"ERROR: Failed to save output image to {output_path}: {e}", file=sys.stderr)
        raise

def main():
    """Main function with proper error handling."""
    args = parse_arguments()
    
    try:
        # Simulate a failure if requested (keep for testing the framework)
        if args.fail:
            raise Exception("Simulated failure requested via --fail flag")

        # Load the model
        model, device = load_model(args.model_path)

        # Load and preprocess the input image
        input_tensor = load_and_preprocess_image(args.input_path, device)

        # Run inference
        output_tensor = run_inference(model, input_tensor, device)

        # Save the output image
        save_output_image(output_tensor, args.output_path)
        
        print("Inference completed successfully")
        return 0
        
    except Exception as e:
        print(f"ERROR: Inference failed with exception: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return 1

if __name__ == '__main__':
    sys.exit(main())
