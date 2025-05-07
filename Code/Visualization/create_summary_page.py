import os
import glob
import argparse
from PIL import Image
import base64
from io import BytesIO

def create_summary_page(zoomed_dir, side_by_side_dir, output_path):
    """
    Create an HTML summary page displaying all comparison images.
    
    Args:
        zoomed_dir: Directory containing zoomed comparison images
        side_by_side_dir: Directory containing side-by-side comparison images
        output_path: Path to save the HTML file
    """
    # Find all zoomed comparison images
    zoomed_image_paths = sorted(glob.glob(os.path.join(zoomed_dir, "*_zoomed_*.png")))
    
    # Find all side-by-side comparison images
    side_by_side_image_paths = sorted(glob.glob(os.path.join(side_by_side_dir, "*_side_by_side_*.png")))
    
    # Combine all image paths
    image_paths = zoomed_image_paths + side_by_side_image_paths
    
    if not image_paths:
        print(f"No comparison images found")
        return
    
    # Start HTML content
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Thermal Denoising Results - Detailed Comparisons</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 20px;
                background-color: #f5f5f5;
            }
            h1, h2, h3 {
                color: #333;
            }
            .metrics {
                background-color: #e9f7fe;
                padding: 15px;
                border-radius: 5px;
                margin-bottom: 20px;
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
        <h1>Thermal Denoising Results - Detailed Comparisons</h1>
        
        <div class="metrics">
            <h2>Denoising Performance Metrics</h2>
            <p><strong>PSNR:</strong> 41.66 dB - Excellent value indicating high fidelity reconstruction</p>
            <p><strong>SSIM:</strong> 0.973 - Very close to 1, suggesting the denoised images retain the structure of the original clean images extremely well</p>
            <p><strong>LPIPS:</strong> 0.027 - Low value indicating excellent perceptual quality</p>
            <p><strong>DISTS:</strong> 0.056 - Low value reinforcing strong texture and structure similarity</p>
        </div>
        
        <div class="section">
            <h2>Zoomed Image Comparisons</h2>
            <p>Each comparison shows the full images (top row) with a highlighted region of interest, and the zoomed-in views of that region (bottom row).</p>
            
            <div class="grid">
    """
    
    # Add zoomed comparison images to the HTML
    for image_path in zoomed_image_paths:
        image_name = os.path.basename(image_path)
        base_name = image_name.split('_zoomed_')[0]
        
        # Extract coordinates from filename
        coords = image_name.split('_zoomed_x')[1].split('.png')[0]
        x_coord = coords.split('_y')[0]
        y_coord = coords.split('_y')[1]
        
        # Create relative path
        rel_path = os.path.relpath(image_path, os.path.dirname(output_path))
        
        html_content += f"""
                <div class="image-container">
                    <div class="image-title">Image: {base_name} (ROI at x={x_coord}, y={y_coord})</div>
                    <img src="{rel_path}" alt="{image_name}">
                </div>
        """
    
    html_content += """
            </div>
        </div>
        
        <div class="section">
            <h2>Side-by-Side Comparisons with Difference Maps</h2>
            <p>These comparisons show the noisy input and final denoised output side by side, with a difference map highlighting where noise was removed.</p>
            
            <div class="grid">
    """
    
    # Add side-by-side comparison images to the HTML
    for image_path in side_by_side_image_paths:
        image_name = os.path.basename(image_path)
        base_name = image_name.split('_side_by_side_')[0]
        
        # Extract coordinates from filename
        coords = image_name.split('_side_by_side_x')[1].split('.png')[0]
        x_coord = coords.split('_y')[0]
        y_coord = coords.split('_y')[1]
        
        # Create relative path
        rel_path = os.path.relpath(image_path, os.path.dirname(output_path))
        
        html_content += f"""
                <div class="image-container">
                    <div class="image-title">Image: {base_name} (ROI at x={x_coord}, y={y_coord})</div>
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
    parser = argparse.ArgumentParser(description='Create HTML summary page for zoomed comparisons')
    parser.add_argument('--zoomed_dir', type=str, default='ThermalDenoising/results/zoomed', help='Directory containing zoomed comparison images')
    parser.add_argument('--side_by_side_dir', type=str, default='ThermalDenoising/results/side_by_side', help='Directory containing side-by-side comparison images')
    parser.add_argument('--output', type=str, default='ThermalDenoising/results/summary.html', help='Path to save the HTML file')
    
    args = parser.parse_args()
    create_summary_page(args.zoomed_dir, args.side_by_side_dir, args.output)

if __name__ == "__main__":
    main()
