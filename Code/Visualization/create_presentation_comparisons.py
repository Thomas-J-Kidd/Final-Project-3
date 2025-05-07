#!/usr/bin/env python3
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
from matplotlib.patches import Rectangle

def create_side_by_side_with_gt(input_comparison, output_dir, title, roi_size=100, zoom_factor=3):
    """
    Creates a clean side-by-side comparison with ground truth
    that is properly sized and formatted for presentations.
    
    Args:
        input_comparison: Path to comparison image
        output_dir: Directory to save output image
        title: Title for the comparison
        roi_size: Size of region of interest
        zoom_factor: How much to zoom the ROI
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the comparison image
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
    except (IndexError, ValueError):
        print(f"Could not extract ROI coordinates from filename: {basename}")
        roi_x, roi_y = 200, 200
    
    # Extract the base image name
    image_name = "_".join(parts[:parts.index("model")])
    
    # Get dimensions and extract parts
    height, width, _ = img.shape
    single_width = width // 4
    
    # Extract the four parts
    noisy_input = img[:, 0:single_width, :]
    ground_truth = img[:, single_width:2*single_width, :]
    model_output = img[:, 3*single_width:, :]
    
    # Create a simple figure with a horizontal layout
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), dpi=300)
    
    # Set the title
    if title:
        plt.suptitle(title, fontsize=16, y=0.95)
    else:
        plt.suptitle(f"Model Comparison: {image_name}", fontsize=16, y=0.95)
    
    # Upper row: Full images with ROI box
    # First column: Input
    axes[0, 0].imshow(noisy_input)
    axes[0, 0].add_patch(Rectangle((roi_x, roi_y), roi_size, roi_size, 
                                  linewidth=2, edgecolor='r', facecolor='none'))
    axes[0, 0].set_title('Noisy Input')
    axes[0, 0].axis('off')
    
    # Second column: Ground Truth
    axes[0, 1].imshow(ground_truth)
    axes[0, 1].add_patch(Rectangle((roi_x, roi_y), roi_size, roi_size, 
                                  linewidth=2, edgecolor='r', facecolor='none'))
    axes[0, 1].set_title('Ground Truth (Reference)')
    axes[0, 1].axis('off')
    
    # Third column: Model Output
    axes[0, 2].imshow(model_output)
    axes[0, 2].add_patch(Rectangle((roi_x, roi_y), roi_size, roi_size, 
                                  linewidth=2, edgecolor='r', facecolor='none'))
    axes[0, 2].set_title('Model Output')
    axes[0, 2].axis('off')
    
    # Lower row: Zoomed regions
    # Extract ROI regions with safety checks
    # Ensure ROI coordinates are within image boundaries
    input_h, input_w = noisy_input.shape[:2]
    roi_x = min(max(0, roi_x), input_w - roi_size)
    roi_y = min(max(0, roi_y), input_h - roi_size)
    
    # Handle case where ROI size would exceed image boundaries
    roi_size_x = min(roi_size, input_w - roi_x)
    roi_size_y = min(roi_size, input_h - roi_y)
    
    if roi_size_x <= 0 or roi_size_y <= 0:
        print(f"Warning: Invalid ROI size ({roi_size_x}x{roi_size_y}) for {basename}, using default region")
        roi_x = min(100, input_w // 2)
        roi_y = min(100, input_h // 2)
        roi_size_x = min(100, input_w - roi_x)
        roi_size_y = min(100, input_h - roi_y)
    
    roi_input = noisy_input[roi_y:roi_y+roi_size_y, roi_x:roi_x+roi_size_x]
    roi_gt = ground_truth[roi_y:roi_y+roi_size_y, roi_x:roi_x+roi_size_x]
    roi_output = model_output[roi_y:roi_y+roi_size_y, roi_x:roi_x+roi_size_x]
    
    # First column: Input Zoom
    axes[1, 0].imshow(roi_input)
    axes[1, 0].set_title('Input (Zoomed)')
    axes[1, 0].axis('off')
    
    # Second column: Ground Truth Zoom
    axes[1, 1].imshow(roi_gt)
    axes[1, 1].set_title('Ground Truth (Zoomed)')
    axes[1, 1].axis('off')
    
    # Third column: Model Output Zoom
    axes[1, 2].imshow(roi_output)
    axes[1, 2].set_title('Model Output (Zoomed)')
    axes[1, 2].axis('off')
    
    # Add a descriptive footer
    plt.figtext(0.5, 0.01, 
                "Comparison showing the noisy input, ground truth reference, and model output. The red boxes highlight the region of interest.",
                ha="center", fontsize=10, bbox={"facecolor":"orange", "alpha":0.2, "pad":5})
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save the figure
    output_filename = f"{image_name}_clean_comparison.png"
    output_path = os.path.join(output_dir, output_filename)
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    
    print(f"Saved clean comparison to {output_path}")
    return output_path

def create_multi_model_clean_comparison(model_images, output_dir, title="Multi-Model Comparison"):
    """
    Creates a clean comparison of multiple models against ground truth
    with consistent sizing and layout.
    
    Args:
        model_images: List of comparison images for different models
        output_dir: Directory to save output
        title: Title for the comparison
    """
    if not model_images:
        print("No images provided for multi-model comparison")
        return None
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the first image to get common elements and size
    first_img = cv2.imread(model_images[0])
    if first_img is None:
        print(f"Failed to load first image: {model_images[0]}")
        return None
    
    first_img = cv2.cvtColor(first_img, cv2.COLOR_BGR2RGB)
    height, width, _ = first_img.shape
    single_width = width // 4
    
    # Extract common elements (input and ground truth)
    noisy_input = first_img[:, 0:single_width, :]
    ground_truth = first_img[:, single_width:2*single_width, :]
    
    # Extract ROI from filename of first image
    basename = os.path.basename(model_images[0])
    parts = basename.split('_')
    try:
        image_name = "_".join(parts[:parts.index("model")])
        x_part = [p for p in parts if p.startswith('x')][0]
        y_part = [p for p in parts if p.startswith('y')][0]
        roi_x = int(x_part[1:])
        roi_y = int(y_part[1:].split('.')[0])
        roi_size = 100  # Default ROI size
    except (IndexError, ValueError):
        print(f"Could not extract ROI coordinates from filename: {basename}")
        image_name = "comparison"
        roi_x, roi_y, roi_size = 200, 200, 100
    
    # Determine how many models we have
    num_models = len(model_images)
    
    # Create a figure with a grid layout:
    # - First row: Input and Ground Truth
    # - Second row: Model outputs
    # - Third row: Zoomed regions
    fig, axes = plt.subplots(3, 2 + num_models, figsize=(15, 10), dpi=300)
    
    # Set title
    plt.suptitle(f"{title}: {image_name}", fontsize=16, y=0.98)
    
    # First row: Input and Ground Truth
    axes[0, 0].imshow(noisy_input)
    axes[0, 0].add_patch(Rectangle((roi_x, roi_y), roi_size, roi_size, 
                                   linewidth=2, edgecolor='r', facecolor='none'))
    axes[0, 0].set_title('Noisy Input')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(ground_truth)
    axes[0, 1].add_patch(Rectangle((roi_x, roi_y), roi_size, roi_size, 
                                   linewidth=2, edgecolor='r', facecolor='none'))
    axes[0, 1].set_title('Ground Truth')
    axes[0, 1].axis('off')
    
    # Hide unused cells in first row
    for i in range(2, 2 + num_models):
        axes[0, i].axis('off')
    
    # Second row: Model outputs with ROI boxes
    model_outputs = []
    model_names = []
    
    for i, img_path in enumerate(model_images):
        # Determine model name from path
        if "nafnet_gray" in img_path.lower():
            model_name = "NAFNET (Gray)"
        elif "nafnet_rgb" in img_path.lower():
            model_name = "NAFNET (RGB)"
        elif "unet_rgb" in img_path.lower():
            model_name = "UNET (RGB)"
        elif "unet" in img_path.lower():
            model_name = "UNET (Thermal)"
        else:
            model_name = f"Model {i+1}"
        
        model_names.append(model_name)
        
        # Load the model image
        model_img = cv2.imread(img_path)
        if model_img is None:
            continue
            
        model_img = cv2.cvtColor(model_img, cv2.COLOR_BGR2RGB)
        
        # Extract model output
        model_output = model_img[:, 3*single_width:, :]
        model_outputs.append(model_output)
        
        # Add model output to figure
        axes[1, i].imshow(model_output)
        axes[1, i].add_patch(Rectangle((roi_x, roi_y), roi_size, roi_size, 
                                      linewidth=2, edgecolor='r', facecolor='none'))
        axes[1, i].set_title(model_name)
        axes[1, i].axis('off')
    
    # Hide unused cells in second row
    for i in range(len(model_outputs), 2 + num_models):
        axes[1, i].axis('off')
    
    # Third row: Zoomed regions
    # Extract zoomed region from input and ground truth with safety checks
    input_h, input_w = noisy_input.shape[:2]
    roi_x = min(max(0, roi_x), input_w - 1)
    roi_y = min(max(0, roi_y), input_h - 1)
    
    # Handle case where ROI size would exceed image boundaries
    roi_size_x = min(roi_size, input_w - roi_x)
    roi_size_y = min(roi_size, input_h - roi_y)
    
    if roi_size_x <= 0 or roi_size_y <= 0:
        print(f"Warning: Invalid ROI size ({roi_size_x}x{roi_size_y}) for {basename}, using default region")
        roi_x = min(100, input_w // 2)
        roi_y = min(100, input_h // 2)
        roi_size_x = min(100, input_w - roi_x)
        roi_size_y = min(100, input_h - roi_y)
    
    roi_input = noisy_input[roi_y:roi_y+roi_size_y, roi_x:roi_x+roi_size_x]
    roi_gt = ground_truth[roi_y:roi_y+roi_size_y, roi_x:roi_x+roi_size_x]
    
    axes[2, 0].imshow(roi_input)
    axes[2, 0].set_title('Input (Zoomed)')
    axes[2, 0].axis('off')
    
    axes[2, 1].imshow(roi_gt)
    axes[2, 1].set_title('Ground Truth (Zoomed)')
    axes[2, 1].axis('off')
    
    # Add zoomed regions for each model
    for i, output in enumerate(model_outputs):
        # Get output dimensions and ensure ROI is within bounds
        output_h, output_w = output.shape[:2]
        out_roi_x = min(max(0, roi_x), output_w - 1)
        out_roi_y = min(max(0, roi_y), output_h - 1)
        out_roi_size_x = min(roi_size_x, output_w - out_roi_x)
        out_roi_size_y = min(roi_size_y, output_h - out_roi_y)
        
        # Extract the ROI from the output
        if out_roi_size_x > 0 and out_roi_size_y > 0:
            roi_output = output[out_roi_y:out_roi_y+out_roi_size_y, out_roi_x:out_roi_x+out_roi_size_x]
        else:
            # Create an empty placeholder if we can't extract a valid ROI
            roi_output = np.zeros((10, 10, 3), dtype=np.uint8)
        axes[2, i+2].imshow(roi_output)
        axes[2, i+2].set_title(f'{model_names[i]} (Zoomed)')
        axes[2, i+2].axis('off')
    
    # Hide unused cells in third row
    for i in range(len(model_outputs)+2, 2 + num_models):
        axes[2, i].axis('off')
    
    # Add a descriptive footer
    plt.figtext(0.5, 0.01, 
                "Side-by-side comparison showing how different models perform on the same input. Red boxes highlight the regions of interest shown in the zoomed views.",
                ha="center", fontsize=10, bbox={"facecolor":"orange", "alpha":0.2, "pad":5})
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save the figure
    output_filename = f"{image_name}_multi_model_clean.png"
    output_path = os.path.join(output_dir, output_filename)
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    
    print(f"Saved multi-model comparison to {output_path}")
    return output_path

def process_all_comparisons(output_dir="clean_comparisons", max_images=10):
    """
    Process all comparison images to create clean, presentation-ready versions.
    """
    # Find all comparison images
    comparison_images = glob.glob("model_comparisons/*model_comparison*.png")
    
    if not comparison_images:
        print("No comparison images found")
        return
    
    # Create clean comparisons for each image (limit to max_images)
    clean_images = []
    for img_path in comparison_images[:max_images]:
        clean_img = create_side_by_side_with_gt(img_path, output_dir, None)
        if clean_img:
            clean_images.append(clean_img)
    
    # Group images by base name
    image_groups = {}
    for img_path in comparison_images:
        basename = os.path.basename(img_path)
        parts = basename.split('_')
        try:
            image_name = "_".join(parts[:parts.index("model")])
            if image_name not in image_groups:
                image_groups[image_name] = []
            image_groups[image_name].append(img_path)
        except (IndexError, ValueError):
            continue
    
    # Create multi-model comparisons for each group
    for image_name, group in image_groups.items():
        # Use each unique ROI only once
        processed_rois = set()
        filtered_group = []
        
        for img_path in group:
            basename = os.path.basename(img_path)
            parts = basename.split('_')
            try:
                x_part = [p for p in parts if p.startswith('x')][0]
                y_part = [p for p in parts if p.startswith('y')][0]
                roi_key = f"{x_part}_{y_part}"
                
                if roi_key not in processed_rois:
                    processed_rois.add(roi_key)
                    filtered_group.append(img_path)
            except (IndexError, ValueError):
                continue
                
        if filtered_group:
            create_multi_model_clean_comparison(
                filtered_group[:min(3, len(filtered_group))], 
                output_dir,
                f"Multi-Model Comparison"
            )
    
    print(f"Created {len(clean_images)} clean comparison images")
    print(f"Images saved to {output_dir}")

if __name__ == "__main__":
    process_all_comparisons()
