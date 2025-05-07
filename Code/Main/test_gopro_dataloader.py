import os
import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import numpy as np
from ThermalDenoising.data.goprodata import GoProData

def test_dataloader(grayscale=False, use_gamma=False, split='test', num_samples=5):
    """
    Test the GoPro data loader and save sample images
    
    Args:
        grayscale: Whether to convert images to grayscale
        use_gamma: Whether to use gamma-corrected blur images
        split: 'train' or 'test'
        num_samples: Number of sample images to save
    """
    # Create output directory
    output_dir = f"gopro_dataloader_test_{'gray' if grayscale else 'rgb'}_{'gamma' if use_gamma else 'linear'}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create dataset
    dataset = GoProData(
        base_path='./dataset/GoPro',
        split=split,
        loadSize=(256, 256),
        mode=0,  # Test mode
        grayscale=grayscale,
        use_gamma=use_gamma
    )
    
    # Create dataloader
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4)
    
    print(f"Dataset size: {len(dataset)} images")
    print(f"Using {'grayscale' if grayscale else 'RGB'} images")
    print(f"Using {'gamma-corrected' if use_gamma else 'linear'} blur images")
    
    # Get and save sample images
    for i, (blur, sharp, name) in enumerate(dataloader):
        if i >= num_samples:
            break
        
        # Save individual images
        save_image(blur, os.path.join(output_dir, f"{name[0]}_blur.png"))
        save_image(sharp, os.path.join(output_dir, f"{name[0]}_sharp.png"))
        
        # Save comparison image
        comparison = torch.cat([blur, sharp], dim=3)
        save_image(comparison, os.path.join(output_dir, f"{name[0]}_comparison.png"))
        
        print(f"Saved sample {i+1}/{num_samples}: {name[0]}")
    
    print(f"Sample images saved to {output_dir}/")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test GoPro data loader')
    parser.add_argument('--grayscale', action='store_true', help='Convert images to grayscale')
    parser.add_argument('--gamma', action='store_true', help='Use gamma-corrected blur images')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'test'], help='Dataset split to use')
    parser.add_argument('--samples', type=int, default=5, help='Number of sample images to save')
    
    args = parser.parse_args()
    
    test_dataloader(
        grayscale=args.grayscale,
        use_gamma=args.gamma,
        split=args.split,
        num_samples=args.samples
    )
