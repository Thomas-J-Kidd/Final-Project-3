#!/usr/bin/env python3
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import glob
from matplotlib.gridspec import GridSpec

def enhance_model_comparison_with_gt(input_comparison, output_dir, title, zoom_factor=3):
    """
    Takes an existing model comparison image and enhances it for presentation use
    with better formatting, higher zoom factor, and custom title.
    Includes the ground truth (reference) image.
    
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
    
    # Get image dimensions
    height, width, _ = img.shape
    
    # Calculate the width of each individual image in the comparison
    single_width = width // 4
    
    # Extract the four parts from the comparison image
    noisy_input = img[:, 0:single_width, :]
    ground_truth = img[:, single_width:2*single_width, :]
    initial_pred = img[:, 2*single_width:3*single_width, :] if width > 3*single_width else None
    final_result = img[:, 3*single_width:, :] if width > 3*single_width else None
    
    # Create a new figure with GridSpec to have more control over the layout
    fig = plt.figure(figsize=(16, 12), dpi=300)  # Larger height to accommodate more rows
    grid = GridSpec(3, 2, figure=fig)  # 3 rows, 2 columns
    
    # Add a better title
    if title:
        plt.suptitle(title, fontsize=18, y=0.98)
    else:
        plt.suptitle(f"Model Comparison: {image_name}", fontsize=18, y=0.98)
    
    # Top row: Original noisy input and ground truth, with ROI highlighted
    ax_input = fig.add_subplot(grid[0, 0])
    ax_input.imshow(noisy_input)
    ax_input.add_patch(Rectangle((roi_x, roi_y), roi_size, roi_size, 
                               linewidth=2, edgecolor='r', facecolor='none'))
    ax_input.set_title('Noisy Input')
    ax_input.axis('off')
    
    ax_gt = fig.add_subplot(grid[0, 1])
    ax_gt.imshow(ground_truth)
    ax_gt.add_patch(Rectangle((roi_x, roi_y), roi_size, roi_size, 
                            linewidth=2, edgecolor='r', facecolor='none'))
    ax_gt.set_title('Ground Truth (Reference)')
    ax_gt.axis('off')
    
    # Middle row: NAFNET result and UNET result with ROI highlighted
    if final_result is not None:
        ax_nafnet = fig.add_subplot(grid[1, 0])
        ax_nafnet.imshow(final_result)
        ax_nafnet.add_patch(Rectangle((roi_x, roi_y), roi_size, roi_size, 
                                   linewidth=2, edgecolor='r', facecolor='none'))
        ax_nafnet.set_title('Model Result')
        ax_nafnet.axis('off')
    
    # Bottom row: Zoomed regions of all three
    ax_input_zoom = fig.add_subplot(grid[2, 0])
    roi_input = noisy_input[roi_y:roi_y+roi_size, roi_x:roi_x+roi_size]
    # Resize for zooming
    if zoom_factor > 1:
        h, w, _ = roi_input.shape
        roi_input = cv2.resize(roi_input, (w * zoom_factor, h * zoom_factor), interpolation=cv2.INTER_NEAREST)
    ax_input_zoom.imshow(roi_input)
    ax_input_zoom.set_title('Noisy Input (Zoomed)')
    ax_input_zoom.axis('off')
    
    ax_gt_zoom = fig.add_subplot(grid[2, 1])
    roi_gt = ground_truth[roi_y:roi_y+roi_size, roi_x:roi_x+roi_size]
    # Resize for zooming
    if zoom_factor > 1:
        h, w, _ = roi_gt.shape
        roi_gt = cv2.resize(roi_gt, (w * zoom_factor, h * zoom_factor), interpolation=cv2.INTER_NEAREST)
    ax_gt_zoom.imshow(roi_gt)
    ax_gt_zoom.set_title('Ground Truth (Zoomed)')
    ax_gt_zoom.axis('off')
    
    if final_result is not None:
        ax_nafnet_zoom = fig.add_subplot(grid[1, 1])
        roi_nafnet = final_result[roi_y:roi_y+roi_size, roi_x:roi_x+roi_size]
        # Resize for zooming
        if zoom_factor > 1:
            h, w, _ = roi_nafnet.shape
            roi_nafnet = cv2.resize(roi_nafnet, (w * zoom_factor, h * zoom_factor), interpolation=cv2.INTER_NEAREST)
        ax_nafnet_zoom.imshow(roi_nafnet)
        ax_nafnet_zoom.set_title('Model Result (Zoomed)')
        ax_nafnet_zoom.axis('off')
    
    # Add a footer with explanatory text
    plt.figtext(0.5, 0.01, 
                "Comparison shows the original noisy input (left), the ground truth reference (right), and model results. Zoomed regions highlight important details.",
                ha="center", fontsize=12, bbox={"facecolor":"orange", "alpha":0.2, "pad":5})
    
    # Adjust layout and save
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save the enhanced figure
    output_filename = f"{image_name}_with_gt_{os.path.basename(input_comparison)}"
    output_path = os.path.join(output_dir, output_filename)
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    
    print(f"Saved enhanced comparison with ground truth to {output_path}")
    return output_path

def create_multi_model_comparison(input_files, output_dir, title, zoom_factor=3):
    """
    Creates a single comparison image showing multiple models and the ground truth.
    
    Args:
        input_files: List of comparison image paths for different models
        output_dir: Directory to save the enhanced image
        title: Title for the comparison image
        zoom_factor: How much to zoom the ROI regions
    """
    if not input_files:
        print("No input files provided")
        return None
        
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract image name and ROI from the first file
    basename = os.path.basename(input_files[0])
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
        roi_x, roi_y, roi_size = 200, 200, 100  # Default values
    
    # Create a figure with appropriate layout
    fig = plt.figure(figsize=(16, 12), dpi=300)
    
    # Count how many models we have (files)
    num_models = len(input_files)
    
    # Create a grid: first row for full images, second row for zoomed regions
    if num_models <= 3:
        grid = GridSpec(2, num_models + 1, figure=fig)  # +1 for the ground truth
    else:
        # For more models, use more columns
        grid = GridSpec(3, 3, figure=fig)  # 3 rows, 3 columns
    
    # Set the title
    if title:
        plt.suptitle(title, fontsize=18, y=0.98)
    else:
        plt.suptitle(f"Model Comparison: {image_name}", fontsize=18, y=0.98)
    
    # Load the first image to get common elements
    img = cv2.imread(input_files[0])
    if img is None:
        print(f"Failed to load image: {input_files[0]}")
        return None
        
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Get image dimensions and extract the quarters
    height, width, _ = img.shape
    single_width = width // 4
    
    # Extract the noisy input and ground truth (same for all models)
    noisy_input = img[:, 0:single_width, :]
    ground_truth = img[:, single_width:2*single_width, :]
    
    # First column, first row: Noisy input
    ax_input = fig.add_subplot(grid[0, 0])
    ax_input.imshow(noisy_input)
    ax_input.add_patch(Rectangle((roi_x, roi_y), roi_size, roi_size, 
                               linewidth=2, edgecolor='r', facecolor='none'))
    ax_input.set_title('Noisy Input')
    ax_input.axis('off')
    
    # First column, second row: Ground truth
    ax_gt = fig.add_subplot(grid[1, 0])
    ax_gt.imshow(ground_truth)
    ax_gt.add_patch(Rectangle((roi_x, roi_y), roi_size, roi_size, 
                            linewidth=2, edgecolor='r', facecolor='none'))
    ax_gt.set_title('Ground Truth (Reference)')
    ax_gt.axis('off')
    
    # First column, third row (if applicable): Zoomed input
    if grid.nrows > 2:
        ax_input_zoom = fig.add_subplot(grid[2, 0])
        roi_input = noisy_input[roi_y:roi_y+roi_size, roi_x:roi_x+roi_size]
        if zoom_factor > 1:
            h, w, _ = roi_input.shape
            roi_input = cv2.resize(roi_input, (w * zoom_factor, h * zoom_factor), interpolation=cv2.INTER_NEAREST)
        ax_input_zoom.imshow(roi_input)
        ax_input_zoom.set_title('Input (Zoomed)')
        ax_input_zoom.axis('off')
    
    # For each model, add its result
    for i, file_path in enumerate(input_files):
        # Load the model comparison image
        model_img = cv2.imread(file_path)
        if model_img is None:
            continue
            
        model_img = cv2.cvtColor(model_img, cv2.COLOR_BGR2RGB)
        
        # Extract the model result (last quarter)
        model_height, model_width, _ = model_img.shape
        model_single_width = model_width // 4
        model_result = model_img[:, 3*model_single_width:, :]
        
        # Determine the model name
        model_basename = os.path.basename(file_path)
        if "nafnet_gray" in file_path.lower():
            model_name = "NAFNET (Gray)"
        elif "nafnet_rgb" in file_path.lower():
            model_name = "NAFNET (RGB)"
        elif "unet_rgb" in file_path.lower():
            model_name = "UNET (RGB)"
        elif "unet" in file_path.lower():
            model_name = "UNET (Thermal)"
        else:
            model_name = f"Model {i+1}"
        
        # Determine the column for this model
        if num_models <= 3:
            col = i + 1  # +1 because column 0 is for input/GT
        else:
            col = (i % 3) + 1 if i < 3 else i % 3
            row = 0 if i < 3 else 1
            
            # If we're in the third row, adjust appropriately
            if grid.nrows > 2 and i >= 3:
                row = 2
                
        # Add the model result
        if num_models <= 3 or i < 6:  # Limit to 6 models max
            # Full model result
            ax_model = fig.add_subplot(grid[0, col] if num_models <= 3 else grid[row, col])
            ax_model.imshow(model_result)
            ax_model.add_patch(Rectangle((roi_x, roi_y), roi_size, roi_size, 
                                       linewidth=2, edgecolor='r', facecolor='none'))
            ax_model.set_title(model_name)
            ax_model.axis('off')
            
            # Zoomed model result
            if num_models <= 3:  # Only add zoomed regions for limited models
                ax_model_zoom = fig.add_subplot(grid[1, col])
                roi_model = model_result[roi_y:roi_y+roi_size, roi_x:roi_x+roi_size]
                if zoom_factor > 1:
                    h, w, _ = roi_model.shape
                    roi_model = cv2.resize(roi_model, (w * zoom_factor, h * zoom_factor), interpolation=cv2.INTER_NEAREST)
                ax_model_zoom.imshow(roi_model)
                ax_model_zoom.set_title(f'{model_name} (Zoomed)')
                ax_model_zoom.axis('off')
    
    # Add a footer with explanatory text
    plt.figtext(0.5, 0.01, 
                "Comparison shows the noisy input, ground truth reference, and results from different models. Red boxes highlight the regions of interest.",
                ha="center", fontsize=12, bbox={"facecolor":"orange", "alpha":0.2, "pad":5})
    
    # Adjust layout and save
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save the multi-model comparison
    output_filename = f"{image_name}_multi_model_comparison.png"
    output_path = os.path.join(output_dir, output_filename)
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    
    print(f"Saved multi-model comparison to {output_path}")
    return output_path

def create_presentation_slides_with_gt(output_dir="presentation_images_with_gt"):
    """
    Generate enhanced presentation-ready images that include ground truth reference images.
    """
    # Find all model comparison images
    comparison_images = glob.glob("model_comparisons/*model_comparison*.png")
    
    if not comparison_images:
        print("No comparison images found")
        return
    
    enhanced_images = []
    multi_model_images = []
    
    # Create enhanced versions of each comparison with ground truth
    titles = [
        "Fine Detail Preservation Comparison",
        "Edge Definition Comparison", 
        "Texture Recovery Comparison",
        "Noise Reduction Effectiveness",
        "Color Accuracy Comparison",
        "Shadow Detail Comparison"
    ]
    
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
            print(f"Could not process filename format: {basename}")
    
    # Process each individual image
    for i, img_path in enumerate(comparison_images[:6]):  # Limit to 6 images
        if i < len(titles):
            title = titles[i]
        else:
            title = f"Model Comparison Example {i+1}"
            
        enhanced = enhance_model_comparison_with_gt(img_path, output_dir, title)
        if enhanced:
            enhanced_images.append(enhanced)
    
    # Process each group to create multi-model comparisons
    for i, (image_name, group_paths) in enumerate(list(image_groups.items())[:3]):  # Limit to 3 groups
        if i < len(titles):
            title = f"Multi-Model {titles[i]}"
        else:
            title = f"Multi-Model Comparison - {image_name}"
            
        # Use just one region per image for multi-model comparison
        if group_paths:
            multi_model = create_multi_model_comparison([group_paths[0]], output_dir, title)
            if multi_model:
                multi_model_images.append(multi_model)
    
    print(f"Created {len(enhanced_images)} enhanced images with ground truth")
    print(f"Created {len(multi_model_images)} multi-model comparison images")
    print(f"Images saved to {output_dir}")

if __name__ == "__main__":
    create_presentation_slides_with_gt()
