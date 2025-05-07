import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from matplotlib.patches import Rectangle
from matplotlib.gridspec import GridSpec

def create_side_by_side_comparison(image_path, roi_x, roi_y, roi_width, roi_height, zoom_factor=4, output_dir='ThermalDenoising/results/side_by_side'):
    """
    Create a side-by-side comparison of the noisy input and final denoised output.
    
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
    
    # Extract the noisy input and final denoised images
    noisy = img[:, 0:single_width, :]
    final = img[:, 3*single_width:, :]
    
    # Extract ROIs from each image
    roi_noisy = noisy[roi_y:roi_y+roi_height, roi_x:roi_x+roi_width, :]
    roi_final = final[roi_y:roi_y+roi_height, roi_x:roi_x+roi_width, :]
    
    # Calculate absolute difference between noisy and denoised ROIs
    diff = np.abs(roi_noisy.astype(np.float32) - roi_final.astype(np.float32)).astype(np.uint8)
    # Enhance the difference for better visibility
    diff_enhanced = cv2.applyColorMap(diff, cv2.COLORMAP_JET)
    diff_enhanced = cv2.cvtColor(diff_enhanced, cv2.COLOR_BGR2RGB)
    
    # Create a figure with subplots
    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(2, 3, figure=fig)
    
    # Top row: Full images with ROI highlighted
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(noisy)
    ax1.add_patch(Rectangle((roi_x, roi_y), roi_width, roi_height, 
                            linewidth=2, edgecolor='r', facecolor='none'))
    ax1.set_title('Noisy Input')
    ax1.axis('off')
    
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.imshow(final)
    ax2.add_patch(Rectangle((roi_x, roi_y), roi_width, roi_height, 
                            linewidth=2, edgecolor='r', facecolor='none'))
    ax2.set_title('Final Denoised')
    ax2.axis('off')
    
    # Bottom row: Zoomed ROIs and difference
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.imshow(roi_noisy, interpolation='nearest')
    ax3.set_title('Noisy Input (Zoomed)')
    ax3.axis('off')
    
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.imshow(diff_enhanced, interpolation='nearest')
    ax4.set_title('Difference Map (Enhanced)')
    ax4.axis('off')
    
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.imshow(roi_final, interpolation='nearest')
    ax5.set_title('Final Denoised (Zoomed)')
    ax5.axis('off')
    
    # Add text explaining the visualization
    plt.figtext(0.5, 0.02, 
                "The difference map shows where noise was removed. Brighter areas indicate larger differences between the noisy and denoised images.",
                ha="center", fontsize=12, bbox={"facecolor":"orange", "alpha":0.2, "pad":5})
    
    # Adjust layout and save
    plt.tight_layout()
    
    # Get the base filename without path and extension
    base_filename = os.path.basename(image_path).replace('_compare.png', '')
    output_path = os.path.join(output_dir, f"{base_filename}_side_by_side_x{roi_x}_y{roi_y}.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved side-by-side comparison to {output_path}")
    return output_path

def main():
    parser = argparse.ArgumentParser(description='Create side-by-side comparisons of thermal denoising results')
    parser.add_argument('--image', type=str, required=True, help='Path to the comparison image')
    parser.add_argument('--roi_x', type=int, default=200, help='X coordinate of ROI')
    parser.add_argument('--roi_y', type=int, default=200, help='Y coordinate of ROI')
    parser.add_argument('--roi_width', type=int, default=100, help='Width of ROI')
    parser.add_argument('--roi_height', type=int, default=100, help='Height of ROI')
    parser.add_argument('--zoom_factor', type=int, default=4, help='Zoom factor')
    parser.add_argument('--output_dir', type=str, default='ThermalDenoising/results/side_by_side', help='Output directory')
    
    args = parser.parse_args()
    
    create_side_by_side_comparison(
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
