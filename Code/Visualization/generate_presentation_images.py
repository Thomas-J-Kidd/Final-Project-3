#!/usr/bin/env python3
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import glob

def enhance_model_comparison(input_comparison, output_dir, title, zoom_factor=3):
    """
    Takes an existing model comparison image and enhances it for presentation use
    with better formatting, higher zoom factor, and custom title.
    
    Args:
        input_comparison: Path to existing comparison image
        output_dir: Directory to save enhanced image
        title: Custom title for the comparison
        zoom_factor: How much to zoom the ROI regions
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the input comparison image
    img = cv2.imread(input_comparison)
    if img is None:
        print(f"Failed to load image: {input_comparison}")
        return None
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Extract ROI coordinates from filename
    basename = os.path.basename(input_comparison)
    parts = basename.split('_')
    try:
        x_part = [p for p in parts if p.startswith('x')][0]
        y_part = [p for p in parts if p.startswith('y')][0]
        roi_x = int(x_part[1:])
        roi_y = int(y_part[1:].split('.')[0])
        roi_size = 100  # Default ROI size
    except (IndexError, ValueError):
        print(f"Could not extract ROI coordinates from filename: {basename}")
        roi_x, roi_y, roi_size = 200, 200, 100  # Default values
    
    # Extract the base image name for use in the title
    image_name = "_".join(parts[:parts.index("model")])
    
    # Create a new high-quality figure with 16:9 aspect ratio
    fig = plt.figure(figsize=(16, 9), dpi=300)
    plt.imshow(img)
    plt.axis('off')
    
    # Add a better title
    if title:
        plt.title(title, fontsize=18, pad=20)
    else:
        plt.title(f"Model Comparison: {image_name}", fontsize=18, pad=20)
    
    # Add a footer with explanatory text
    plt.figtext(0.5, 0.01, 
                "Comparison shows how different AI models handle the same input image. Red boxes highlight regions of interest.",
                ha="center", fontsize=12, bbox={"facecolor":"orange", "alpha":0.2, "pad":5})
    
    # Save the enhanced figure
    output_filename = f"{image_name}_enhanced_{os.path.basename(input_comparison)}"
    output_path = os.path.join(output_dir, output_filename)
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    
    print(f"Saved enhanced comparison to {output_path}")
    return output_path

def create_highlighted_regions_summary(comparison_images, output_dir, title="Key Differences Between Models"):
    """
    Create a summary figure showing just the zoomed regions from multiple comparison images
    to highlight the differences between models.
    
    Args:
        comparison_images: List of comparison image paths
        output_dir: Directory to save the summary image
        title: Title for the summary image
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a figure for the summary
    fig, axs = plt.subplots(2, 3, figsize=(16, 9), dpi=300)
    axs = axs.flatten()
    
    # Add a title to the figure
    fig.suptitle(title, fontsize=18, y=0.95)
    
    # Process each comparison image
    for i, img_path in enumerate(comparison_images[:6]):  # Limit to 6 images
        if i >= len(axs):
            break
            
        # Load the comparison image
        img = cv2.imread(img_path)
        if img is None:
            continue
            
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Extract the base image name for use in the subtitle
        basename = os.path.basename(img_path)
        parts = basename.split('_')
        image_name = "_".join(parts[:parts.index("model")])
        
        # Display the image
        axs[i].imshow(img)
        axs[i].set_title(f"Sample {i+1}: {image_name[:15]}...", fontsize=10)
        axs[i].axis('off')
    
    # Hide any unused subplots
    for i in range(len(comparison_images), len(axs)):
        axs[i].axis('off')
    
    # Add descriptive text at the bottom
    plt.figtext(0.5, 0.01, 
                "The highlighted regions show where differences between models are most noticeable. These samples demonstrate the varying effectiveness of each approach.",
                ha="center", fontsize=12, bbox={"facecolor":"orange", "alpha":0.2, "pad":5})
    
    # Adjust layout and save
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    output_path = os.path.join(output_dir, "model_comparison_highlights.png")
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    
    print(f"Saved highlights summary to {output_path}")
    return output_path

def create_presentation_slides(output_dir="presentation_images"):
    """
    Generate a set of presentation-ready images based on the existing model comparisons.
    """
    # Find all model comparison images
    comparison_images = glob.glob("model_comparisons/*model_comparison*.png")
    
    if not comparison_images:
        print("No comparison images found")
        return
    
    enhanced_images = []
    
    # Create enhanced versions of each comparison
    titles = [
        "Fine Detail Preservation Comparison",
        "Edge Definition Comparison", 
        "Texture Recovery Comparison",
        "Noise Reduction Effectiveness",
        "Color Accuracy Comparison",
        "Shadow Detail Comparison"
    ]
    
    for i, img_path in enumerate(comparison_images[:6]):  # Limit to 6 images
        if i < len(titles):
            title = titles[i]
        else:
            title = f"Model Comparison Example {i+1}"
            
        enhanced = enhance_model_comparison(img_path, output_dir, title)
        if enhanced:
            enhanced_images.append(enhanced)
    
    # Create a summary image highlighting key regions
    create_highlighted_regions_summary(comparison_images, output_dir)
    
    print(f"Created {len(enhanced_images)} presentation-ready images")
    print(f"Images saved to {output_dir}")

if __name__ == "__main__":
    create_presentation_slides()
