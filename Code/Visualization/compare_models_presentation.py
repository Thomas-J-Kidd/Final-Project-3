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
    if img is None:
        print(f"Failed to load image: {image_path}")
        return None
        
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    height, width, _ = img.shape
    single_width = width // 4
    
    quarter = img[:, quarter_index*single_width:(quarter_index+1)*single_width, :]
    
    return quarter

def create_model_comparison(image_name, model_dirs, model_names, roi_x, roi_y, roi_size, output_dir, 
                           zoom_factor=2, output_filename=None, title=None):
    """
    Create a comparison of the same image and ROI across different models.
    
    Args:
        image_name: Base name of the image to compare
        model_dirs: List of directories containing model results
        model_names: List of names for each model
        roi_x, roi_y: Top-left coordinates of the region of interest
        roi_size: Size of the ROI (width and height)
        output_dir: Directory to save the comparison image
        zoom_factor: How much to magnify the zoomed region
        output_filename: Custom filename for the output image
        title: Custom title for the comparison
        
    Returns:
        Path to the saved comparison image
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a figure for the comparison
    fig = plt.figure(figsize=(16, 9))  # 16:9 ratio for presentations
    
    # Set up the grid layout
    available_models = []
    available_model_names = []
    
    for model_dir, model_name in zip(model_dirs, model_names):
        model_compare_path = os.path.join(model_dir, f"{image_name}_compare.png")
        if os.path.exists(model_compare_path):
            available_models.append(model_dir)
            available_model_names.append(model_name)
    
    if not available_models:
        print(f"No models have results for {image_name}, skipping")
        return None
    
    num_models = len(available_models)
    grid = GridSpec(2, num_models + 1)
    
    # Load the first model's image to get the input
    first_model_compare_path = os.path.join(available_models[0], f"{image_name}_compare.png")
    
    # Extract the noisy input (first quarter of comparison image)
    noisy_input = extract_quarter_image(first_model_compare_path, 0)
    if noisy_input is None:
        return None
        
    # Ground truth (if available, typically second quarter)
    ground_truth = extract_quarter_image(first_model_compare_path, 1)
    
    # Add overall title if provided
    if title:
        plt.suptitle(title, fontsize=16, y=0.98)
    else:
        plt.suptitle(f"Model Comparison: {image_name}", fontsize=16, y=0.98)
    
    # Display the input image with ROI highlighted
    ax_input = fig.add_subplot(grid[0, 0])
    ax_input.imshow(noisy_input)
    ax_input.add_patch(Rectangle((roi_x, roi_y), roi_size, roi_size, 
                             linewidth=2, edgecolor='r', facecolor='none'))
    ax_input.set_title('Noisy Input')
    ax_input.axis('off')
    
    # Display the zoomed input
    ax_input_zoom = fig.add_subplot(grid[1, 0])
    roi_img = noisy_input[roi_y:roi_y+roi_size, roi_x:roi_x+roi_size]
    # Resize for zooming if needed
    if zoom_factor > 1:
        h, w, _ = roi_img.shape
        roi_img = cv2.resize(roi_img, (w * zoom_factor, h * zoom_factor), interpolation=cv2.INTER_NEAREST)
    ax_input_zoom.imshow(roi_img)
    ax_input_zoom.set_title('Input (Zoomed)')
    ax_input_zoom.axis('off')
    
    # For each model, show the result and zoomed region
    for i, (model_dir, model_name) in enumerate(zip(available_models, available_model_names)):
        # Get the model's output for this image
        model_compare_path = os.path.join(model_dir, f"{image_name}_compare.png")
        
        # Extract the model's result (fourth quarter of comparison image)
        model_output = extract_quarter_image(model_compare_path, 3)
        if model_output is None:
            continue
            
        # Display the model output with ROI highlighted
        ax_model = fig.add_subplot(grid[0, i+1])
        ax_model.imshow(model_output)
        ax_model.add_patch(Rectangle((roi_x, roi_y), roi_size, roi_size, 
                               linewidth=2, edgecolor='r', facecolor='none'))
        ax_model.set_title(f'{model_name}')
        ax_model.axis('off')
        
        # Display the zoomed model output
        ax_model_zoom = fig.add_subplot(grid[1, i+1])
        roi_model = model_output[roi_y:roi_y+roi_size, roi_x:roi_x+roi_size]
        # Resize for zooming if needed
        if zoom_factor > 1:
            h, w, _ = roi_model.shape
            roi_model = cv2.resize(roi_model, (w * zoom_factor, h * zoom_factor), interpolation=cv2.INTER_NEAREST)
        ax_model_zoom.imshow(roi_model)
        ax_model_zoom.set_title(f'{model_name} (Zoomed)')
        ax_model_zoom.axis('off')
    
    # Adjust layout and save
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust to leave room for the title
    
    if output_filename:
        output_path = os.path.join(output_dir, output_filename)
    else:
        output_path = os.path.join(output_dir, f"{image_name}_model_comparison_x{roi_x}_y{roi_y}.png")
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Saved model comparison to {output_path}")
    return output_path

def create_summary_page(images, output_path, title="AI Model Comparison"):
    """
    Create an HTML summary page displaying all comparison images.
    
    Args:
        images: List of image paths to include
        output_path: Path to save the HTML file
        title: Title for the summary page
    """
    if not images:
        print("No images to include in summary")
        return
    
    # Start HTML content
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>{title}</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 20px;
                background-color: #f5f5f5;
            }}
            h1, h2, h3 {{
                color: #333;
            }}
            .image-container {{
                margin-bottom: 30px;
                background-color: white;
                padding: 15px;
                border-radius: 5px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            }}
            .image-title {{
                font-weight: bold;
                margin-bottom: 10px;
            }}
            img {{
                max-width: 100%;
                height: auto;
                border: 1px solid #ddd;
            }}
            .grid {{
                display: grid;
                grid-template-columns: repeat(auto-fill, minmax(600px, 1fr));
                gap: 20px;
            }}
            .section {{
                margin-bottom: 30px;
            }}
            .model-description {{
                background-color: #e9f7fe;
                padding: 15px;
                border-radius: 5px;
                margin-bottom: 20px;
            }}
        </style>
    </head>
    <body>
        <h1>{title}</h1>
        
        <div class="model-description">
            <h2>Models Performance Description</h2>
            <p><strong>NAFNET (RGB):</strong> Superior detail preservation and color accuracy. Most effective at preserving fine details in complex textures.</p>
            <p><strong>NAFNET (Gray):</strong> Strong performance on grayscale images with excellent noise reduction while maintaining edge details.</p>
            <p><strong>UNET (RGB):</strong> Good overall performance with balanced noise reduction and detail preservation.</p>
            <p><strong>UNET (Thermal):</strong> Specialized model for thermal imagery with excellent contrast enhancement.</p>
        </div>
        
        <div class="section">
            <h2>Model Comparison with Zoomed Regions</h2>
            <p>Each comparison shows the original input and the output from different models, with zoomed-in views of regions of interest.</p>
            
            <div class="grid">
    """
    
    # Add each image to the HTML
    for image_path in images:
        if not os.path.exists(image_path):
            continue
            
        image_name = os.path.basename(image_path)
        rel_path = os.path.relpath(image_path, os.path.dirname(output_path))
        
        # Extract image info
        parts = image_name.split("_")
        if "model_comparison" in image_name:
            base_name = "_".join(parts[:parts.index("model")])
            coords = "_".join(parts[parts.index("comparison")+1:]).replace('.png', '')
        else:
            base_name = image_name.replace('.png', '')
            coords = "custom region"
        
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
    parser = argparse.ArgumentParser(description='Create presentation-ready model comparisons')
    parser.add_argument('--output_dir', type=str, default='presentation_comparisons', help='Directory to save comparison images')
    parser.add_argument('--image', type=str, help='Specific image to process (without _compare.png)')
    parser.add_argument('--roi_x', type=int, help='X coordinate of ROI')
    parser.add_argument('--roi_y', type=int, help='Y coordinate of ROI')
    parser.add_argument('--roi_size', type=int, default=100, help='Size of ROI (width and height)')
    parser.add_argument('--zoom', type=int, default=2, help='Zoom factor for the ROI')
    parser.add_argument('--title', type=str, help='Custom title for the comparison image')
    parser.add_argument('--output_filename', type=str, help='Custom filename for the output image')
    parser.add_argument('--preset', type=str, choices=['deblurring', 'denoising', 'best'], help='Use preset ROIs for specific examples')
    
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
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    all_comparison_images = []
    
    # Handle presets
    if args.preset:
        if args.preset == 'deblurring':
            # Good examples for deblurring comparison
            presets = [
                ('GOPR0396_11_00_000097', 625, 150, 120, 'Deblurring Performance - Text Detail'),
                ('GOPR0871_11_00_000020', 450, 200, 100, 'Deblurring Performance - Edge Definition'),
                ('GOPR0396_11_00_000075', 700, 300, 150, 'Deblurring Performance - Complex Scene')
            ]
        elif args.preset == 'denoising':
            # Good examples for denoising comparison
            presets = [
                ('GOPR0384_11_00_000090', 300, 200, 120, 'Denoising Performance - Flat Areas'),
                ('GOPR0854_11_00_000085', 400, 150, 100, 'Denoising Performance - Textured Areas'),
                ('GOPR0881_11_01_000221', 500, 250, 150, 'Denoising Performance - Shadow Areas')
            ]
        elif args.preset == 'best':
            # Best examples for overall model comparison
            presets = [
                ('GOPR0396_11_00_000097', 625, 150, 120, 'Model Comparison - Fine Detail'),
                ('GOPR0871_11_00_000020', 450, 200, 100, 'Model Comparison - Edge Preservation'),
                ('GOPR0881_11_01_000221', 500, 250, 150, 'Model Comparison - Noise Reduction')
            ]
        
        for preset in presets:
            image_name, roi_x, roi_y, roi_size, title = preset
            output_filename = f"{image_name}_{title.replace(' ', '_')}.png"
            output_path = create_model_comparison(
                image_name, model_dirs, model_names, roi_x, roi_y, roi_size, 
                args.output_dir, args.zoom, output_filename, title
            )
            if output_path:
                all_comparison_images.append(output_path)
    
    # Handle single image with specified ROI
    elif args.image and args.roi_x is not None and args.roi_y is not None:
        output_path = create_model_comparison(
            args.image, model_dirs, model_names, args.roi_x, args.roi_y, args.roi_size, 
            args.output_dir, args.zoom, args.output_filename, args.title
        )
        if output_path:
            all_comparison_images.append(output_path)
    
    # If no specific image or preset was specified
    else:
        print("Please specify either an image and ROI coordinates, or use a preset.")
        print("Example usage:")
        print("  python compare_models_presentation.py --image GOPR0396_11_00_000097 --roi_x 625 --roi_y 150 --roi_size 120 --title 'Fine Detail Comparison'")
        print("  python compare_models_presentation.py --preset deblurring")
        return
    
    # Create summary page
    if all_comparison_images:
        create_summary_page(all_comparison_images, os.path.join(args.output_dir, "presentation_summary.html"), 
                          title="AI Model Comparison - Presentation Graphics")
        
        print(f"Created {len(all_comparison_images)} comparison images")
        print(f"Summary page saved to {os.path.join(args.output_dir, 'presentation_summary.html')}")

if __name__ == "__main__":
    main()
