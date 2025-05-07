from flask import Flask, render_template, request, jsonify, send_file
import os
import glob
import json
import subprocess
import tempfile

app = Flask(__name__)

@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')

@app.route('/scan_images', methods=['POST'])
def scan_images():
    """Scan for available images in the result directories."""
    model_type = request.json.get('model_type', 'nafnet')  # or 'unet'
    
    results_dir = f"ThermalDenoising/results{'_unet' if model_type == 'unet' else ''}"
    comparison_images = glob.glob(os.path.join(results_dir, "*_compare.png"))
    
    # Create list of images with thumbnails
    images = []
    for img_path in comparison_images:
        base_name = os.path.basename(img_path).replace('_compare.png', '')
        images.append({
            'id': base_name,
            'name': base_name,
            'path': img_path,
            'selected': False
        })
    
    return jsonify({'images': images})

@app.route('/generate_report', methods=['POST'])
def generate_report():
    """Generate HTML report with selected images."""
    data = request.json
    model_type = data.get('model_type', 'nafnet')
    selected_images = data.get('selected_images', [])
    roi_settings = data.get('roi_settings', {
        'x': 200,
        'y': 200,
        'width': 100,
        'height': 100
    })
    
    # Set up directories
    results_dir = f"ThermalDenoising/results{'_unet' if model_type == 'unet' else ''}"
    zoomed_dir = os.path.join(results_dir, "zoomed")
    side_by_side_dir = os.path.join(results_dir, "side_by_side")
    output_path = os.path.join(results_dir, "summary.html")
    
    os.makedirs(zoomed_dir, exist_ok=True)
    os.makedirs(side_by_side_dir, exist_ok=True)
    
    # Process each selected image
    for image_id in selected_images:
        image_path = os.path.join(results_dir, f"{image_id}_compare.png")
        if os.path.exists(image_path):
            # Create zoomed comparison
            subprocess.run([
                "python", "create_zoomed_comparison.py",
                "--image", image_path,
                "--roi_x", str(roi_settings['x']),
                "--roi_y", str(roi_settings['y']),
                "--roi_width", str(roi_settings['width']),
                "--roi_height", str(roi_settings['height']),
                "--output_dir", zoomed_dir
            ])
    
    # Generate summary page
    subprocess.run([
        "python", "create_summary_page.py",
        "--zoomed_dir", zoomed_dir,
        "--side_by_side_dir", side_by_side_dir,
        "--output", output_path
    ])
    
    return jsonify({'status': 'success', 'report_path': output_path})

@app.route('/open_report', methods=['POST'])
def open_report():
    """Open the generated report in a browser."""
    model_type = request.json.get('model_type', 'nafnet')
    results_dir = f"ThermalDenoising/results{'_unet' if model_type == 'unet' else ''}"
    report_path = os.path.join(results_dir, "summary.html")
    
    if os.path.exists(report_path):
        return jsonify({'status': 'success', 'report_path': report_path})
    else:
        return jsonify({'status': 'error', 'message': 'Report not found'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
