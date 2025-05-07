#!/usr/bin/env python3
import os
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
from matplotlib.patches import Rectangle
from matplotlib.gridspec import GridSpec
from PIL import Image

def find_common_images(model_dirs):
    """
    Find image names that exist across all model directories.
    
    Args:
        model_dirs: List of directories containing model results
        
    Returns:
        List of common image base names (without path or extension)
    """
    common_images = None
    
    for model_dir in model_dirs:
        # Get all comparison images in this directory
        images = glob.glob(os.path.join(model_dir, "*_compare.png"))
        # Extract just the base filenames without the _compare.png suffix
        basenames = [os.path.basename(img).replace('_compare.png', '') for img in images]
        
        if common_images is None:
            common_images = set(basenames)
        else:
            common_images &= set(basenames)
    
    return sorted(list(common_images))

def find_high_variance_regions(image_path, num_regions=3, roi_size=100, min_distance=50):
    """
    Find regions with high variance in an image.
    
    Args:
        image_path: Path to the image
        num_regions: Number of high-variance regions to find
        roi_size: Size of the region of interest (square)
        min_distance: Minimum distance between selected regions
        
    Returns:
        List of (x, y) coordinates for the top-left corner of each region
    """
    # Load the comparison image
    img = cv2.imread(image_path)
    
    if img is None:
        print(f"Failed to load image: {image_path}")
        return []
    
    # Get image dimensions
    height, width, _ = img.shape
    
    # Calculate the width of each individual image in the comparison
    single_width = width // 4
    
    # Extract the first quarter of the comparison image (typically the noisy input)
    target_img = img[:, 0:single_width, :]
    
    # Convert to grayscale
    gray = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY)
    
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
        if np.max(var_padded) == 0:
            break
        
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

def extract_quarter_image(image_path, quarter_index):
    """
    Extract one of the four images from a comparison image.
    
    Args:
        image_path: Path to the comparison image
        quarter_index: Which quarter to extract (0-3)
        
    Returns:
        The extracted image as a numpy array
    """
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    height, width, _ = img.shape
    single_width = width // 4
    
    quarter = img[:, quarter_index*single_width:(quarter_index+1)*single_width, :]
    
    return quarter

def create_model_comparison(image_name, model_dirs, model_names, roi_x, roi_y, roi_size, output_dir):
    """
    Create a comparison of the same image and ROI across different models.
    
    Args:
        image_name: Base name of the image to compare
        model_dirs: List of directories containing model results
        model_names: List of names for each model
        roi_x, roi_y: Top-left coordinates of the region of interest
        roi_size: Size of the ROI (width and height)
        output_dir: Directory to save the comparison image
        
    Returns:
        Path to the saved comparison image
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a figure for the comparison
    fig = plt.figure(figsize=(15, 10))
    
    # Set up the grid layout
    num_models = len(model_dirs)
    grid = GridSpec(2, num_models + 1)
    
    # Load the first model's image to get the input
    first_model_compare_path = os.path.join(model_dirs[0], f"{image_name}_compare.png")
    
    # Extract the noisy input (first quarter of comparison image)
    noisy_input = extract_quarter_image(first_model_compare_path, 0)
    
    # Ground truth (if available, typically second quarter)
    ground_truth = extract_quarter_image(first_model_compare_path, 1)
    
    # Display the input image with ROI highlighted
    ax_input = fig.add_subplot(grid[0, 0])
    ax_input.imshow(noisy_input)
    ax_input.add_patch(Rectangle((roi_x, roi_y), roi_size, roi_size, 
                             linewidth=2, edgecolor='r', facecolor='none'))
    ax_input.set_title('Noisy Input')
    ax_input.axis('off')
    
    # Display the zoomed input
    ax_input_zoom = fig.add_subplot(grid[1, 0])
    ax_input_zoom.imshow(noisy_input[roi_y:roi_y+roi_size, roi_x:roi_x+roi_size], interpolation='nearest')
    ax_input_zoom.set_title('Input (Zoomed)')
    ax_input_zoom.axis('off')
    
    # For each model, show the result and zoomed region
    for i, (model_dir, model_name) in enumerate(zip(model_dirs, model_names)):
        # Get the model's output for this image
        model_compare_path = os.path.join(model_dir, f"{image_name}_compare.png")
        if not os.path.exists(model_compare_path):
            print(f"Image {model_compare_path} not found")
            continue
        
        # Extract the model's result (fourth quarter of comparison image)
        model_output = extract_quarter_image(model_compare_path, 3)
        
        # Display the model output with ROI highlighted
        ax_model = fig.add_subplot(grid[0, i+1])
        ax_model.imshow(model_output)
        ax_model.add_patch(Rectangle((roi_x, roi_y), roi_size, roi_size, 
                               linewidth=2, edgecolor='r', facecolor='none'))
        ax_model.set_title(f'{model_name}')
        ax_model.axis('off')
        
        # Display the zoomed model output
        ax_model_zoom = fig.add_subplot(grid[1, i+1])
        ax_model_zoom.imshow(model_output[roi_y:roi_y+roi_size, roi_x:roi_x+roi_size], interpolation='nearest')
        ax_model_zoom.set_title(f'{model_name} (Zoomed)')
        ax_model_zoom.axis('off')
    
    # Adjust layout and save
    plt.tight_layout()
    output_path = os.path.join(output_dir, f"{image_name}_model_comparison_x{roi_x}_y{roi_y}.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Saved model comparison to {output_path}")
    return output_path

def create_multi_region_comparison(image_name, model_dirs, model_names, regions, output_dir):
    """
    Create a comparison of multiple regions across different models for the same image.
    
    Args:
        image_name: Base name of the image to compare
        model_dirs: List of directories containing model results
        model_names: List of names for each model
        regions: List of (x, y) coordinates for regions to compare
        output_dir: Directory to save the comparison image
        
    Returns:
        List of paths to the saved comparison images
    """
    output_paths = []
    roi_size = 100  # Default ROI size
    
    for roi_x, roi_y in regions:
        output_path = create_model_comparison(
            image_name, model_dirs, model_names, roi_x, roi_y, roi_size, output_dir
        )
        output_paths.append(output_path)
    
    return output_paths

def create_summary_page(images, output_path):
    """
    Create an HTML summary page displaying all comparison images.
    
    Args:
        images: List of image paths to include
        output_path: Path to save the HTML file
    """
    if not images:
        print("No images to include in summary")
        return
    
    # Start HTML content
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>AI Model Comparison - Detailed Comparisons</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 20px;
                background-color: #f5f5f5;
            }
            h1, h2, h3 {
                color: #333;
            }
            .image-container {
                margin-bottom: 30px;
                background-color: white;
                padding: 15px;
                border-radius: 5px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            }
            .image-title {
                font-weight: bold;
                margin-bottom: 10px;
            }
            img {
                max-width: 100%;
                height: auto;
                border: 1px solid #ddd;
            }
            .grid {
                display: grid;
                grid-template-columns: repeat(auto-fill, minmax(600px, 1fr));
                gap: 20px;
            }
            .section {
                margin-bottom: 30px;
            }
        </style>
    </head>
    <body>
        <h1>AI Model Comparison - Detailed Comparisons</h1>
        
        <div class="section">
            <h2>Model Comparison with Zoomed Regions</h2>
            <p>Each comparison shows the original input and the output from different models, with zoomed-in views of regions of interest.</p>
            
            <div class="grid">
    """
    
    # Add each image to the HTML
    for image_path in images:
        image_name = os.path.basename(image_path)
        rel_path = os.path.relpath(image_path, os.path.dirname(output_path))
        
        # Extract image info
        parts = image_name.split("_")
        base_name = "_".join(parts[:parts.index("model")])
        coords = "_".join(parts[parts.index("comparison")+1:]).replace('.png', '')
        
        html_content += f"""
                <div class="image-container">
                    <div class="image-title">Image: {base_name} ({coords})</div>
                    <img src="{rel_path}" alt="{image_name}">
                </div>
        """
    
    # Close HTML content
    html_content += """
            </div>
        </div>
    </body>
    </html>
    """
    
    # Write HTML file
    with open(output_path, 'w') as f:
        f.write(html_content)
    
    print(f"Summary page created at {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Compare different AI models on the same images')
    parser.add_argument('--output_dir', type=str, default='model_comparisons', help='Directory to save comparison images')
    parser.add_argument('--num_regions', type=int, default=3, help='Number of regions to compare per image')
    parser.add_argument('--num_images', type=int, default=10, help='Maximum number of images to compare')
    
    args = parser.parse_args()
    
    # Define model directories and names
    model_dirs = [
        'ThermalDenoising/results_gopro_nafnet_gray',
        'ThermalDenoising/results_gopro_nafnet_rgb',
        'ThermalDenoising/results_gopro_unet_rgb',
        'ThermalDenoising/results_unet'
    ]
    
    model_names = [
        'NAFNET (Gray)',
        'NAFNET (RGB)',
        'UNET (RGB)',
        'UNET (Thermal)'
    ]
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Find images that exist in all model directories
    common_images = find_common_images(model_dirs)
    
    if not common_images:
        # If no common images across all models, try pairwise comparisons
        print("No images common to all models. Checking pairwise combinations...")
        
        # Instead, let's find images that exist in at least the first model directory
        primary_model_images = glob.glob(os.path.join(model_dirs[0], "*_compare.png"))
        primary_model_basenames = [os.path.basename(img).replace('_compare.png', '') for img in primary_model_images]
        
        # Limit the number of images
        if len(primary_model_basenames) > args.num_images:
            primary_model_basenames = primary_model_basenames[:args.num_images]
        
        common_images = primary_model_basenames
    else:
        # Limit the number of images
        if len(common_images) > args.num_images:
            common_images = common_images[:args.num_images]
    
    print(f"Found {len(common_images)} images for comparison")
    
    all_comparison_images = []
    
    # Process each image
    for image_name in common_images:
        print(f"Processing {image_name}...")
        
        # Find the comparison image in the first model directory
        first_model_compare_path = os.path.join(model_dirs[0], f"{image_name}_compare.png")
        if not os.path.exists(first_model_compare_path):
            continue
        
        # Find interesting regions
        regions = find_high_variance_regions(first_model_compare_path, args.num_regions)
        
        if not regions:
            # If no regions found, use default regions
            print(f"No interesting regions found for {image_name}, using default regions")
            regions = [(100, 100), (200, 200), (300, 300)]
        
        # Create available model comparisons for this image
        available_models = []
        available_model_names = []
        
        for model_dir, model_name in zip(model_dirs, model_names):
            model_compare_path = os.path.join(model_dir, f"{image_name}_compare.png")
            if os.path.exists(model_compare_path):
                available_models.append(model_dir)
                available_model_names.append(model_name)
        
        if len(available_models) < 2:
            print(f"Not enough models have results for {image_name}, skipping")
            continue
        
        comparison_images = create_multi_region_comparison(
            image_name, available_models, available_model_names, regions, args.output_dir
        )
        all_comparison_images.extend(comparison_images)
    
    # Create summary page
    create_summary_page(all_comparison_images, os.path.join(args.output_dir, "model_comparison_summary.html"))
    
    print(f"Completed processing {len(common_images)} images")
    print(f"Created {len(all_comparison_images)} comparison images")
    print(f"Summary page saved to {os.path.join(args.output_dir, 'model_comparison_summary.html')}")

if __name__ == "__main__":
    main()
