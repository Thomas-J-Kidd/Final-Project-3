import re
import os
import glob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.gridspec import GridSpec
from pathlib import Path
import matplotlib.ticker as mtick
from matplotlib.colors import LinearSegmentedColormap

# Configuration
THERMAL_LOG_DIR = 'ThermalDenoising/logs'
GOPRO_LOG_DIR = 'ThermalDenoising/logs'
OUTPUT_DIR = 'presentation_images'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# File patterns
THERMAL_TEST_PATTERN = 'testThermalDenoising_*.log'
GOPRO_TEST_PATTERN = 'testGoProDeblurring_*.log'
THERMAL_TRAIN_PATTERN = 'trainThermalDenoising_*.log'
GOPRO_TRAIN_PATTERN = 'trainGoProDeblurring_*.log'

# Helper functions for log parsing
def extract_metrics_from_thermal_test_log(log_path):
    """Extract metrics summary from thermal test logs."""
    metrics = {
        'PSNR': None,
        'SSIM': None,
        'LPIPS': None,
        'DISTS': None
    }
    model_type = None
    
    with open(log_path, 'r') as file:
        content = file.read()
        
        # Check if it's UNET or NAFNET
        model_indicator = "UNET" if "UNET" in content else "NAFNET"
        model_type = model_indicator
        
        # Extract metrics
        psnr_match = re.search(r'----Average PSNR/SSIM.*?PSNR: ([\d\.]+) dB; SSIM: ([\d\.]+)', content)
        lpips_match = re.search(r'----Average LPIPS\t: ([\d\.]+)', content)
        dists_match = re.search(r'----Average DISTS\t: ([\d\.]+)', content)
        
        if psnr_match:
            metrics['PSNR'] = float(psnr_match.group(1))
            metrics['SSIM'] = float(psnr_match.group(2))
        
        if lpips_match:
            metrics['LPIPS'] = float(lpips_match.group(1))
            
        if dists_match:
            metrics['DISTS'] = float(dists_match.group(1))
    
    return model_type, metrics

def extract_metrics_from_gopro_test_log(log_path):
    """Extract metrics summary from GoPro test logs."""
    metrics = {
        'PSNR': None,
        'SSIM': None,
        'LPIPS': None,
        'DISTS': None
    }
    model_type = None
    
    with open(log_path, 'r') as file:
        content = file.read()
        
        # Check if it's UNET or NAFNET and RGB or Gray
        model_indicator = "UNET" if "UNET" in content else "NAFNET"
        color_mode = "RGB" if "RGB" in content else "Gray"
        model_type = f"{model_indicator}_{color_mode}"
        
        # Extract metrics
        psnr_match = re.search(r'----Average PSNR/SSIM.*?PSNR: ([\d\.]+) dB; SSIM: ([\d\.]+)', content)
        lpips_match = re.search(r'----Average LPIPS\t: ([\d\.]+)', content)
        dists_match = re.search(r'----Average DISTS\t: ([\d\.]+)', content)
        
        if psnr_match:
            metrics['PSNR'] = float(psnr_match.group(1))
            metrics['SSIM'] = float(psnr_match.group(2))
        
        if lpips_match:
            metrics['LPIPS'] = float(lpips_match.group(1))
            
        if dists_match:
            metrics['DISTS'] = float(dists_match.group(1))
    
    return model_type, metrics

def extract_training_progress(log_path, is_thermal=True):
    """Extract training metrics over time from training logs."""
    iterations = []
    psnr_values = []
    ssim_values = []
    lpips_values = []
    dists_values = []
    loss_values = []
    
    with open(log_path, 'r') as file:
        content = file.read()
        
        # Determine model type
        model_indicator = "UNET" if "UNET" in content else "NAFNET"
        if not is_thermal:
            color_mode = "RGB" if "RGB" in content else "Gray"
            model_type = f"{model_indicator}_{color_mode}"
        else:
            model_type = model_indicator
        
        # Find all validation blocks
        validation_pattern = r"validation\. Iteration (\d+) ----\n.*?----Average PSNR\t: ([\d\.]+)\n.*?----Average SSIM\t: ([\d\.]+)\n.*?----Average LPIPS\t: ([\d\.]+)\n.*?----Average DISTS\t: ([\d\.]+)"
        matches = re.findall(validation_pattern, content, re.DOTALL)
        
        for match in matches:
            iteration, psnr, ssim, lpips, dists = match
            iterations.append(int(iteration))
            psnr_values.append(float(psnr))
            ssim_values.append(float(ssim))
            lpips_values.append(float(lpips))
            dists_values.append(float(dists))
        
        # Also extract loss values
        loss_pattern = r"Iteration (\d+).*?loss_[gl]: ([\d\.]+)"
        loss_matches = re.findall(loss_pattern, content)
        if loss_matches:
            loss_iterations = [int(m[0]) for m in loss_matches]
            loss_values = [float(m[1]) for m in loss_matches]
    
    return model_type, {
        'iterations': iterations,
        'psnr': psnr_values,
        'ssim': ssim_values,
        'lpips': lpips_values,
        'dists': dists_values,
        'loss_iterations': loss_iterations if 'loss_iterations' in locals() else [],
        'loss_values': loss_values
    }

# Visualization functions
def create_bar_comparison(thermal_metrics, gopro_metrics, output_path):
    """Create bar charts comparing models on each dataset."""
    # Set up the figure
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('UNET vs NAFNET Performance Comparison', fontsize=18)
    
    # Colors
    unet_color = '#1f77b4'  # Blue
    nafnet_color = '#ff7f0e'  # Orange
    
    # Thermal dataset - PSNR and SSIM (higher is better)
    if thermal_metrics:
        models = list(thermal_metrics.keys())
        psnr_values = [thermal_metrics[m]['PSNR'] for m in models]
        ssim_values = [thermal_metrics[m]['SSIM'] for m in models]
        
        axs[0, 0].bar(models, psnr_values, color=[unet_color if 'UNET' in m else nafnet_color for m in models])
        axs[0, 0].set_title('Thermal Dataset - PSNR (Higher is Better)', fontsize=14)
        axs[0, 0].set_ylabel('PSNR (dB)')
        axs[0, 0].grid(axis='y', linestyle='--', alpha=0.7)
        
        axs[0, 1].bar(models, ssim_values, color=[unet_color if 'UNET' in m else nafnet_color for m in models])
        axs[0, 1].set_title('Thermal Dataset - SSIM (Higher is Better)', fontsize=14)
        axs[0, 1].set_ylabel('SSIM')
        axs[0, 1].grid(axis='y', linestyle='--', alpha=0.7)
    
    # GoPro dataset - PSNR and SSIM (higher is better)
    if gopro_metrics:
        models = list(gopro_metrics.keys())
        psnr_values = [gopro_metrics[m]['PSNR'] for m in models]
        ssim_values = [gopro_metrics[m]['SSIM'] for m in models]
        
        axs[1, 0].bar(models, psnr_values, color=[unet_color if 'UNET' in m else nafnet_color for m in models])
        axs[1, 0].set_title('GoPro Dataset - PSNR (Higher is Better)', fontsize=14)
        axs[1, 0].set_ylabel('PSNR (dB)')
        axs[1, 0].grid(axis='y', linestyle='--', alpha=0.7)
        axs[1, 0].set_xticklabels(models, rotation=45, ha='right')
        
        axs[1, 1].bar(models, ssim_values, color=[unet_color if 'UNET' in m else nafnet_color for m in models])
        axs[1, 1].set_title('GoPro Dataset - SSIM (Higher is Better)', fontsize=14)
        axs[1, 1].set_ylabel('SSIM')
        axs[1, 1].grid(axis='y', linestyle='--', alpha=0.7)
        axs[1, 1].set_xticklabels(models, rotation=45, ha='right')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create another figure for LPIPS and DISTS (lower is better)
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('UNET vs NAFNET Perceptual Metrics Comparison', fontsize=18)
    
    # Thermal dataset - LPIPS and DISTS (lower is better)
    if thermal_metrics:
        models = list(thermal_metrics.keys())
        lpips_values = [thermal_metrics[m]['LPIPS'] for m in models]
        dists_values = [thermal_metrics[m]['DISTS'] for m in models]
        
        axs[0, 0].bar(models, lpips_values, color=[unet_color if 'UNET' in m else nafnet_color for m in models])
        axs[0, 0].set_title('Thermal Dataset - LPIPS (Lower is Better)', fontsize=14)
        axs[0, 0].set_ylabel('LPIPS')
        axs[0, 0].grid(axis='y', linestyle='--', alpha=0.7)
        
        axs[0, 1].bar(models, dists_values, color=[unet_color if 'UNET' in m else nafnet_color for m in models])
        axs[0, 1].set_title('Thermal Dataset - DISTS (Lower is Better)', fontsize=14)
        axs[0, 1].set_ylabel('DISTS')
        axs[0, 1].grid(axis='y', linestyle='--', alpha=0.7)
    
    # GoPro dataset - LPIPS and DISTS (lower is better)
    if gopro_metrics:
        models = list(gopro_metrics.keys())
        lpips_values = [gopro_metrics[m]['LPIPS'] for m in models]
        dists_values = [gopro_metrics[m]['DISTS'] for m in models]
        
        axs[1, 0].bar(models, lpips_values, color=[unet_color if 'UNET' in m else nafnet_color for m in models])
        axs[1, 0].set_title('GoPro Dataset - LPIPS (Lower is Better)', fontsize=14)
        axs[1, 0].set_ylabel('LPIPS')
        axs[1, 0].grid(axis='y', linestyle='--', alpha=0.7)
        axs[1, 0].set_xticklabels(models, rotation=45, ha='right')
        
        axs[1, 1].bar(models, dists_values, color=[unet_color if 'UNET' in m else nafnet_color for m in models])
        axs[1, 1].set_title('GoPro Dataset - DISTS (Lower is Better)', fontsize=14)
        axs[1, 1].set_ylabel('DISTS')
        axs[1, 1].grid(axis='y', linestyle='--', alpha=0.7)
        axs[1, 1].set_xticklabels(models, rotation=45, ha='right')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_path.replace('.png', '_perceptual.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    return [output_path, output_path.replace('.png', '_perceptual.png')]

def create_radar_charts(metrics_dict, dataset_name, output_path):
    """Create radar/spider charts for multi-metric visualization."""
    # Prepare data
    models = list(metrics_dict.keys())
    metrics = ['PSNR', 'SSIM', 'LPIPS', 'DISTS']
    
    # For radar chart, we need to normalize all metrics to 0-1 range
    # PSNR and SSIM: higher is better, so normalize directly
    # LPIPS and DISTS: lower is better, so we invert after normalization
    
    # Get min and max for each metric across models
    metric_ranges = {}
    for metric in metrics:
        values = [metrics_dict[model][metric] for model in models]
        metric_ranges[metric] = (min(values), max(values))
    
    # Normalize values
    normalized_data = {}
    for model in models:
        normalized_data[model] = []
        for i, metric in enumerate(metrics):
            value = metrics_dict[model][metric]
            min_val, max_val = metric_ranges[metric]
            range_val = max_val - min_val
            if range_val == 0:  # Avoid division by zero
                normalized = 0.5
            else:
                normalized = (value - min_val) / range_val
            
            # Invert for metrics where lower is better
            if metric in ['LPIPS', 'DISTS']:
                normalized = 1 - normalized
            
            normalized_data[model].append(normalized)
    
    # Setup the radar chart
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, polar=True)
    
    # Set the angles for each metric (equally spaced)
    angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
    
    # Make the plot circular by appending the first angle at the end
    angles += angles[:1]
    
    # Add metric labels with information on whether higher or lower is better
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([f"{m}\n({'Higher' if m in ['PSNR', 'SSIM'] else 'Lower'} is better)" for m in metrics])
    
    # Add grid
    ax.grid(True)
    
    # Colors
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    # Plot data
    for i, model in enumerate(models):
        values = normalized_data[model]
        values += values[:1]  # Complete the loop
        ax.plot(angles, values, 'o-', linewidth=2, label=model, color=colors[i % len(colors)])
        ax.fill(angles, values, alpha=0.1, color=colors[i % len(colors)])
    
    # Add legend
    ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    plt.title(f'Model Performance Comparison - {dataset_name} Dataset', size=15)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_path

def create_training_progress_plots(training_data, dataset_name, output_path):
    """Create line plots showing training progress."""
    if not training_data:
        return None
    
    # Setup the figure
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Training Progress - {dataset_name} Dataset', fontsize=18)
    
    # Colors for different models
    colors = {
        'UNET': '#1f77b4',       # Blue
        'NAFNET': '#ff7f0e',     # Orange
        'UNET_RGB': '#1f77b4',   # Blue
        'UNET_Gray': '#9467bd',  # Purple
        'NAFNET_RGB': '#ff7f0e', # Orange
        'NAFNET_Gray': '#e377c2' # Pink
    }
    
    # Plot PSNR progress
    for model, data in training_data.items():
        if 'iterations' in data and 'psnr' in data and len(data['iterations']) > 0:
            color = colors.get(model, 'gray')
            axs[0, 0].plot(data['iterations'], data['psnr'], '-', label=model, color=color, linewidth=2)
    
    axs[0, 0].set_title('PSNR Improvement Over Training', fontsize=14)
    axs[0, 0].set_xlabel('Iterations')
    axs[0, 0].set_ylabel('PSNR (dB)')
    axs[0, 0].grid(True, linestyle='--', alpha=0.7)
    axs[0, 0].legend()
    
    # Plot SSIM progress
    for model, data in training_data.items():
        if 'iterations' in data and 'ssim' in data and len(data['iterations']) > 0:
            color = colors.get(model, 'gray')
            axs[0, 1].plot(data['iterations'], data['ssim'], '-', label=model, color=color, linewidth=2)
    
    axs[0, 1].set_title('SSIM Improvement Over Training', fontsize=14)
    axs[0, 1].set_xlabel('Iterations')
    axs[0, 1].set_ylabel('SSIM')
    axs[0, 1].grid(True, linestyle='--', alpha=0.7)
    axs[0, 1].legend()
    
    # Plot LPIPS progress
    for model, data in training_data.items():
        if 'iterations' in data and 'lpips' in data and len(data['iterations']) > 0:
            color = colors.get(model, 'gray')
            axs[1, 0].plot(data['iterations'], data['lpips'], '-', label=model, color=color, linewidth=2)
    
    axs[1, 0].set_title('LPIPS Reduction Over Training', fontsize=14)
    axs[1, 0].set_xlabel('Iterations')
    axs[1, 0].set_ylabel('LPIPS')
    axs[1, 0].grid(True, linestyle='--', alpha=0.7)
    axs[1, 0].legend()
    
    # Plot DISTS progress
    for model, data in training_data.items():
        if 'iterations' in data and 'dists' in data and len(data['iterations']) > 0:
            color = colors.get(model, 'gray')
            axs[1, 1].plot(data['iterations'], data['dists'], '-', label=model, color=color, linewidth=2)
    
    axs[1, 1].set_title('DISTS Reduction Over Training', fontsize=14)
    axs[1, 1].set_xlabel('Iterations')
    axs[1, 1].set_ylabel('DISTS')
    axs[1, 1].grid(True, linestyle='--', alpha=0.7)
    axs[1, 1].legend()
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Also create a loss plot if available
    has_loss_data = False
    for model, data in training_data.items():
        if 'loss_iterations' in data and 'loss_values' in data and len(data['loss_iterations']) > 0:
            has_loss_data = True
            break
    
    if has_loss_data:
        plt.figure(figsize=(12, 6))
        for model, data in training_data.items():
            if 'loss_iterations' in data and 'loss_values' in data and len(data['loss_iterations']) > 0:
                color = colors.get(model, 'gray')
                plt.plot(data['loss_iterations'], data['loss_values'], '-', label=model, color=color, linewidth=2)
        
        plt.title(f'Training Loss - {dataset_name} Dataset', fontsize=16)
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.tight_layout()
        loss_output_path = output_path.replace('.png', '_loss.png')
        plt.savefig(loss_output_path, dpi=300, bbox_inches='tight')
        plt.close()
        return [output_path, loss_output_path]
    
    return [output_path]

def create_html_report(thermal_metrics, gopro_metrics, figure_paths, output_path):
    """Generate an HTML report with all visualizations and metrics."""
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>UNET vs NAFNET Model Comparison</title>
        <style>
            body { font-family: Arial, sans-serif; line-height: 1.6; max-width: 1200px; margin: 0 auto; padding: 20px; }
            h1, h2, h3 { color: #333; }
            .header { text-align: center; margin-bottom: 30px; }
            .figure-container { margin: 20px 0; text-align: center; }
            .figure-container img { max-width: 100%; height: auto; border: 1px solid #ddd; }
            table { border-collapse: collapse; width: 100%; margin: 20px 0; }
            th, td { border: 1px solid #ddd; padding: 8px; text-align: right; }
            th { background-color: #f2f2f2; text-align: center; }
            .section { margin: 40px 0; }
            .metric-better { color: green; font-weight: bold; }
            .metric-worse { color: red; }
            .model-unet { color: #1f77b4; }
            .model-nafnet { color: #ff7f0e; }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>UNET vs NAFNET Model Comparison</h1>
            <p>Comparison of model performance on Thermal Denoising and GoPro Deblurring datasets</p>
        </div>
    """
    
    # Add Thermal Dataset Section
    html_content += """
        <div class="section">
            <h2>Thermal Denoising Dataset Results</h2>
    """
    
    if thermal_metrics:
        # Add metrics table
        html_content += """
            <h3>Performance Metrics</h3>
            <table>
                <tr>
                    <th>Model</th>
                    <th>PSNR (dB) ↑</th>
                    <th>SSIM ↑</th>
                    <th>LPIPS ↓</th>
                    <th>DISTS ↓</th>
                </tr>
        """
        
        for model, metrics in thermal_metrics.items():
            html_content += f"""
                <tr>
                    <td>{'<span class="model-unet">' if 'UNET' in model else '<span class="model-nafnet">'}{model}</span></td>
                    <td>{metrics['PSNR']:.4f}</td>
                    <td>{metrics['SSIM']:.4f}</td>
                    <td>{metrics['LPIPS']:.4f}</td>
                    <td>{metrics['DISTS']:.4f}</td>
                </tr>
            """
        
        html_content += """
            </table>
        """
    
    # Add GoPro Dataset Section
    html_content += """
        </div>
        <div class="section">
            <h2>GoPro Deblurring Dataset Results</h2>
    """
    
    if gopro_metrics:
        # Add metrics table
        html_content += """
            <h3>Performance Metrics</h3>
            <table>
                <tr>
                    <th>Model</th>
                    <th>PSNR (dB) ↑</th>
                    <th>SSIM ↑</th>
                    <th>LPIPS ↓</th>
                    <th>DISTS ↓</th>
                </tr>
        """
        
        for model, metrics in gopro_metrics.items():
            html_content += f"""
                <tr>
                    <td>{'<span class="model-unet">' if 'UNET' in model else '<span class="model-nafnet">'}{model}</span></td>
                    <td>{metrics['PSNR']:.4f}</td>
                    <td>{metrics['SSIM']:.4f}</td>
                    <td>{metrics['LPIPS']:.4f}</td>
                    <td>{metrics['DISTS']:.4f}</td>
                </tr>
            """
        
        html_content += """
            </table>
        """
    
    # Add visualizations
    html_content += """
        </div>
        <div class="section">
            <h2>Visualizations</h2>
    """
    
    for path in figure_paths:
        rel_path = os.path.relpath(path, os.path.dirname(output_path))
        title = os.path.basename(path).replace('.png', '').replace('_', ' ').title()
        html_content += f"""
            <div class="figure-container">
                <h3>{title}</h3>
                <img src="{rel_path}" alt="{title}">
            </div>
        """
    
    # Close HTML
    html_content += """
        </div>
    </body>
    </html>
    """
    
    with open(output_path, 'w') as f:
        f.write(html_content)
    
    return output_path

# Main execution
def main():
    """Main execution function to run the model comparison visualization."""
    print("Starting UNET vs NAFNET model comparison visualization...")
    
    # Define output paths
    bar_chart_path = os.path.join(OUTPUT_DIR, 'model_comparison_bar.png')
    thermal_radar_path = os.path.join(OUTPUT_DIR, 'thermal_radar_chart.png')
    gopro_radar_path = os.path.join(OUTPUT_DIR, 'gopro_radar_chart.png')
    thermal_training_path = os.path.join(OUTPUT_DIR, 'thermal_training_progress.png')
    gopro_training_path = os.path.join(OUTPUT_DIR, 'gopro_training_progress.png')
    html_report_path = os.path.join(OUTPUT_DIR, 'model_comparison_report.html')
    
    # Parse all logs and extract metrics
    thermal_metrics = {}
    gopro_metrics = {}
    thermal_training = {}
    gopro_training = {}
    
    figure_paths = []
    
    # Process thermal test logs
    thermal_test_logs = glob.glob(os.path.join(THERMAL_LOG_DIR, THERMAL_TEST_PATTERN))
    for log_path in thermal_test_logs:
        model_type, metrics = extract_metrics_from_thermal_test_log(log_path)
        if model_type and all(metrics.values()):
            thermal_metrics[model_type] = metrics
    
    # Process GoPro test logs
    gopro_test_logs = glob.glob(os.path.join(GOPRO_LOG_DIR, GOPRO_TEST_PATTERN))
    for log_path in gopro_test_logs:
        model_type, metrics = extract_metrics_from_gopro_test_log(log_path)
        if model_type and all(metrics.values()):
            gopro_metrics[model_type] = metrics
    
    # Process thermal training logs
    thermal_train_logs = glob.glob(os.path.join(THERMAL_LOG_DIR, THERMAL_TRAIN_PATTERN))
    for log_path in thermal_train_logs:
        model_type, progress = extract_training_progress(log_path, is_thermal=True)
        if model_type and progress['iterations']:
            thermal_training[model_type] = progress
    
    # Process GoPro training logs
    gopro_train_logs = glob.glob(os.path.join(GOPRO_LOG_DIR, GOPRO_TRAIN_PATTERN))
    for log_path in gopro_train_logs:
        model_type, progress = extract_training_progress(log_path, is_thermal=False)
        if model_type and progress['iterations']:
            gopro_training[model_type] = progress
    
    # Generate bar chart comparison
    if thermal_metrics or gopro_metrics:
        bar_charts = create_bar_comparison(thermal_metrics, gopro_metrics, bar_chart_path)
        figure_paths.extend(bar_charts)
    
    # Generate radar charts
    if thermal_metrics:
        thermal_radar = create_radar_charts(thermal_metrics, "Thermal", thermal_radar_path)
        figure_paths.append(thermal_radar)
    
    if gopro_metrics:
        gopro_radar = create_radar_charts(gopro_metrics, "GoPro", gopro_radar_path)
        figure_paths.append(gopro_radar)
    
    # Generate training progress plots
    if thermal_training:
        thermal_plots = create_training_progress_plots(thermal_training, "Thermal", thermal_training_path)
        if thermal_plots:
            figure_paths.extend(thermal_plots)
    
    if gopro_training:
        gopro_plots = create_training_progress_plots(gopro_training, "GoPro", gopro_training_path)
        if gopro_plots:
            figure_paths.extend(gopro_plots)
    
    # Create HTML report
    create_html_report(thermal_metrics, gopro_metrics, figure_paths, html_report_path)
    
    print(f"Visualization complete. HTML report saved to: {html_report_path}")
    print(f"All visualizations saved to directory: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
