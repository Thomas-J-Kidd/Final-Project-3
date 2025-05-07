import cv2
import numpy as np
import os
import argparse
import glob
from create_zoomed_comparison import create_zoomed_comparison

def find_high_variance_regions(image_path, num_regions=3, roi_size=100, min_distance=50):
    """
    Find regions with high variance in the noisy part of a comparison image.
    
    Args:
        image_path: Path to the comparison image
        num_regions: Number of high-variance regions to find
        roi_size: Size of the region of interest (square)
        min_distance: Minimum distance between selected regions
        
    Returns:
        List of (x, y) coordinates for the top-left corner of each region
    """
    # Load the comparison image
    img = cv2.imread(image_path)
    
    # Get image dimensions
    height, width, _ = img.shape
    
    # Calculate the width of each individual image in the comparison
    single_width = width // 4
    
    # Extract the noisy image (first quarter of the comparison image)
    noisy = img[:, 0:single_width, :]
    
    # Convert to grayscale
    gray = cv2.cvtColor(noisy, cv2.COLOR_BGR2GRAY)
    
    # Calculate local variance using a sliding window
    kernel_size = 15
    mean = cv2.blur(gray, (kernel_size, kernel_size))
    mean_sq = cv2.blur(np.square(gray), (kernel_size, kernel_size))
    var = mean_sq - np.square(mean)
    
    # Create a mask to avoid selecting regions too close to the edge
    edge_margin = roi_size
    mask = np.ones_like(var)
    mask[:edge_margin, :] = 0  # Top margin
    mask[-edge_margin:, :] = 0  # Bottom margin
    mask[:, :edge_margin] = 0  # Left margin
    mask[:, -edge_margin:] = 0  # Right margin
    
    # Apply mask to variance
    var = var * mask
    
    # Pad the borders for ROI extraction
    pad = roi_size // 2
    var_padded = np.pad(var, pad, mode='constant')
    
    # Find regions with high variance
    regions = []
    for _ in range(num_regions):
        # Find the maximum variance point
        y_padded, x_padded = np.unravel_index(np.argmax(var_padded), var_padded.shape)
        x, y = x_padded - pad, y_padded - pad
        
        # Add to regions list
        regions.append((x, y))
        
        # Zero out this region and its surroundings to find the next highest variance region
        y_min = max(0, y_padded - min_distance)
        y_max = min(var_padded.shape[0], y_padded + min_distance)
        x_min = max(0, x_padded - min_distance)
        x_max = min(var_padded.shape[1], x_padded + min_distance)
        var_padded[y_min:y_max, x_min:x_max] = 0
    
    return regions

def process_images(image_dir, output_dir, num_regions=3, roi_size=100):
    """
    Process all comparison images in a directory, finding interesting regions and creating zoomed comparisons.
    
    Args:
        image_dir: Directory containing comparison images
        output_dir: Directory to save zoomed comparisons
        num_regions: Number of regions to find per image
        roi_size: Size of the region of interest
    """
    # Find all comparison images
    image_paths = glob.glob(os.path.join(image_dir, "*_compare.png"))
    
    # Process each image
    for image_path in image_paths:
        print(f"Processing {image_path}...")
        
        # Find interesting regions
        regions = find_high_variance_regions(image_path, num_regions, roi_size)
        
        # Create zoomed comparisons for each region
        for i, (x, y) in enumerate(regions):
            print(f"  Region {i+1}: ({x}, {y})")
            create_zoomed_comparison(
                image_path, 
                x, 
                y, 
                roi_size, 
                roi_size, 
                output_dir=output_dir
            )

def main():
    parser = argparse.ArgumentParser(description='Find interesting regions in thermal denoising results')
    parser.add_argument('--image_dir', type=str, default='ThermalDenoising/results', help='Directory containing comparison images')
    parser.add_argument('--output_dir', type=str, default='ThermalDenoising/results/zoomed', help='Output directory')
    parser.add_argument('--num_regions', type=int, default=3, help='Number of regions to find per image')
    parser.add_argument('--roi_size', type=int, default=100, help='Size of the region of interest')
    parser.add_argument('--single_image', type=str, default=None, help='Process only a single image (optional)')
    
    args = parser.parse_args()
    
    if args.single_image:
        # Process a single image
        print(f"Processing single image: {args.single_image}")
        regions = find_high_variance_regions(args.single_image, args.num_regions, args.roi_size)
        for i, (x, y) in enumerate(regions):
            print(f"  Region {i+1}: ({x}, {y})")
            create_zoomed_comparison(
                args.single_image, 
                x, 
                y, 
                args.roi_size, 
                args.roi_size, 
                output_dir=args.output_dir
            )
    else:
        # Process all images in the directory
        process_images(args.image_dir, args.output_dir, args.num_regions, args.roi_size)

if __name__ == "__main__":
    main()
