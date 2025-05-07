import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from matplotlib.patches import Rectangle

def create_zoomed_comparison(image_path, roi_x, roi_y, roi_width, roi_height, zoom_factor=4, output_dir='ThermalDenoising/results/zoomed'):
    """
    Create a zoomed-in comparison of a specific region in the comparison image.
    
    Args:
        image_path: Path to the comparison image
        roi_x, roi_y: Top-left coordinates of the region of interest (ROI)
        roi_width, roi_height: Width and height of the ROI
        zoom_factor: How much to zoom in
        output_dir: Directory to save the output image
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the comparison image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    
    # Get image dimensions
    height, width, _ = img.shape
    
    # Calculate the width of each individual image in the comparison
    single_width = width // 4
    
    # Extract the four images
    noisy = img[:, 0:single_width, :]
    gt = img[:, single_width:2*single_width, :]
    initial_pred = img[:, 2*single_width:3*single_width, :]
    final = img[:, 3*single_width:, :]
    
    # Extract ROIs from each image
    roi_noisy = noisy[roi_y:roi_y+roi_height, roi_x:roi_x+roi_width, :]
    roi_gt = gt[roi_y:roi_y+roi_height, roi_x:roi_x+roi_width, :]
    roi_initial_pred = initial_pred[roi_y:roi_y+roi_height, roi_x:roi_x+roi_width, :]
    roi_final = final[roi_y:roi_y+roi_height, roi_x:roi_x+roi_width, :]
    
    # Create a figure with subplots
    fig, axs = plt.subplots(2, 4, figsize=(20, 10))
    
    # First row: Full images with ROI highlighted
    axs[0, 0].imshow(noisy)
    axs[0, 0].add_patch(Rectangle((roi_x, roi_y), roi_width, roi_height, 
                                  linewidth=2, edgecolor='r', facecolor='none'))
    axs[0, 0].set_title('Noisy Input')
    axs[0, 0].axis('off')
    
    axs[0, 1].imshow(gt)
    axs[0, 1].add_patch(Rectangle((roi_x, roi_y), roi_width, roi_height, 
                                  linewidth=2, edgecolor='r', facecolor='none'))
    axs[0, 1].set_title('Ground Truth')
    axs[0, 1].axis('off')
    
    axs[0, 2].imshow(initial_pred)
    axs[0, 2].add_patch(Rectangle((roi_x, roi_y), roi_width, roi_height, 
                                  linewidth=2, edgecolor='r', facecolor='none'))
    axs[0, 2].set_title('Initial Prediction')
    axs[0, 2].axis('off')
    
    axs[0, 3].imshow(final)
    axs[0, 3].add_patch(Rectangle((roi_x, roi_y), roi_width, roi_height, 
                                  linewidth=2, edgecolor='r', facecolor='none'))
    axs[0, 3].set_title('Final Denoised')
    axs[0, 3].axis('off')
    
    # Second row: Zoomed ROIs
    axs[1, 0].imshow(roi_noisy, interpolation='nearest')
    axs[1, 0].set_title('Noisy Input (Zoomed)')
    axs[1, 0].axis('off')
    
    axs[1, 1].imshow(roi_gt, interpolation='nearest')
    axs[1, 1].set_title('Ground Truth (Zoomed)')
    axs[1, 1].axis('off')
    
    axs[1, 2].imshow(roi_initial_pred, interpolation='nearest')
    axs[1, 2].set_title('Initial Prediction (Zoomed)')
    axs[1, 2].axis('off')
    
    axs[1, 3].imshow(roi_final, interpolation='nearest')
    axs[1, 3].set_title('Final Denoised (Zoomed)')
    axs[1, 3].axis('off')
    
    # Adjust layout and save
    plt.tight_layout()
    
    # Get the base filename without path and extension
    base_filename = os.path.basename(image_path).replace('_compare.png', '')
    output_path = os.path.join(output_dir, f"{base_filename}_zoomed_x{roi_x}_y{roi_y}.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved zoomed comparison to {output_path}")
    return output_path

def main():
    parser = argparse.ArgumentParser(description='Create zoomed-in comparisons of thermal denoising results')
    parser.add_argument('--image', type=str, required=True, help='Path to the comparison image')
    parser.add_argument('--roi_x', type=int, default=200, help='X coordinate of ROI')
    parser.add_argument('--roi_y', type=int, default=200, help='Y coordinate of ROI')
    parser.add_argument('--roi_width', type=int, default=100, help='Width of ROI')
    parser.add_argument('--roi_height', type=int, default=100, help='Height of ROI')
    parser.add_argument('--zoom_factor', type=int, default=4, help='Zoom factor')
    parser.add_argument('--output_dir', type=str, default='ThermalDenoising/results/zoomed', help='Output directory')
    
    args = parser.parse_args()
    
    create_zoomed_comparison(
        args.image, 
        args.roi_x, 
        args.roi_y, 
        args.roi_width, 
        args.roi_height, 
        args.zoom_factor,
        args.output_dir
    )

if __name__ == "__main__":
    main()
