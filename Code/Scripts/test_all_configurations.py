#!/usr/bin/env python3
import os
import sys
import argparse
import subprocess
import time
import glob
import yaml
from datetime import datetime

# Configuration mapping
CONFIGURATIONS = {
    # Thermal Denoising configurations
    'thermal_unet': {
        'config_path': 'ThermalDenoising/conf_unet.yml',
        'main_script': 'main.py',
        'weight_dir': 'ThermalDenoising/weights_unet',
        'result_dir': 'ThermalDenoising/results_unet',
        'description': 'Thermal Denoising with UNET'
    },
    'thermal_nafnet': {
        'config_path': 'ThermalDenoising/conf.yml',
        'main_script': 'main.py',
        'weight_dir': 'ThermalDenoising/weights',
        'result_dir': 'ThermalDenoising/results',
        'description': 'Thermal Denoising with NAFNET'
    },
    
    # GoPro Deblurring configurations
    'gopro_nafnet_rgb': {
        'config_path': 'ThermalDenoising/conf_gopro_nafnet_rgb.yml',
        'main_script': 'main_gopro.py',
        'weight_dir': 'ThermalDenoising/weights_gopro_nafnet_rgb',
        'result_dir': 'ThermalDenoising/results_gopro_nafnet_rgb',
        'description': 'GoPro Deblurring with NAFNET (RGB)'
    },
    'gopro_nafnet_gray': {
        'config_path': 'ThermalDenoising/conf_gopro_nafnet_gray.yml',
        'main_script': 'main_gopro.py',
        'weight_dir': 'ThermalDenoising/weights_gopro_nafnet_gray',
        'result_dir': 'ThermalDenoising/results_gopro_nafnet_gray',
        'description': 'GoPro Deblurring with NAFNET (Grayscale)'
    },
    'gopro_unet_rgb': {
        'config_path': 'ThermalDenoising/conf_gopro_unet_rgb.yml',
        'main_script': 'main_gopro.py',
        'weight_dir': 'ThermalDenoising/weights_gopro_unet_rgb',
        'result_dir': 'ThermalDenoising/results_gopro_unet_rgb',
        'description': 'GoPro Deblurring with UNET (RGB)'
    },
    'gopro_unet_gray': {
        'config_path': 'ThermalDenoising/conf_gopro_unet_gray.yml',
        'main_script': 'main_gopro.py',
        'weight_dir': 'ThermalDenoising/weights_gopro_unet_gray',
        'result_dir': 'ThermalDenoising/results_gopro_unet_gray',
        'description': 'GoPro Deblurring with UNET (Grayscale)'
    }
}

# Group configurations for easier selection
CONFIG_GROUPS = {
    'all': list(CONFIGURATIONS.keys()),
    'thermal': ['thermal_unet', 'thermal_nafnet'],
    'gopro': ['gopro_nafnet_rgb', 'gopro_nafnet_gray', 'gopro_unet_rgb', 'gopro_unet_gray'],
    'unet': ['thermal_unet', 'gopro_unet_rgb', 'gopro_unet_gray'],
    'nafnet': ['thermal_nafnet', 'gopro_nafnet_rgb', 'gopro_nafnet_gray'],
    'rgb': ['gopro_nafnet_rgb', 'gopro_unet_rgb'],
    'gray': ['gopro_nafnet_gray', 'gopro_unet_gray']
}

def setup_directories(configs_to_test):
    """Create necessary directories for testing"""
    for config_name in configs_to_test:
        config = CONFIGURATIONS[config_name]
        
        # Ensure weight directory exists
        if not os.path.exists(config['weight_dir']):
            print(f"Creating weight directory: {config['weight_dir']}")
            os.makedirs(config['weight_dir'], exist_ok=True)
            
        # Ensure result directory exists
        if not os.path.exists(config['result_dir']):
            print(f"Creating result directory: {config['result_dir']}")
            os.makedirs(config['result_dir'], exist_ok=True)

def set_mode_in_config(config_path, mode):
    """Set MODE in config files (0 for test, 1 for train)"""
    try:
        # Read the config file
        with open(config_path, 'r') as f:
            content = f.read()
            
        # Replace MODE setting
        if mode == 0:  # Test mode
            content = content.replace('MODE : 1', 'MODE : 0')
        elif mode == 1:  # Train mode
            content = content.replace('MODE : 0', 'MODE : 1')
            
        # Write back to file
        with open(config_path, 'w') as f:
            f.write(content)
            
        return True
    except Exception as e:
        print(f"Error setting mode in {config_path}: {e}")
        return False

def run_test(config_name, train_mode=False):
    """Run test for a specific configuration"""
    config = CONFIGURATIONS[config_name]
    
    print("\n" + "="*50)
    print(f"Testing {config['description']}")
    print("="*50)
    
    # Set mode in configuration file
    mode = 1 if train_mode else 0
    mode_str = "Training" if train_mode else "Testing"
    if not set_mode_in_config(config['config_path'], mode):
        print(f"Failed to set {mode_str} mode in config file. Skipping.")
        return False
    
    # Build command
    cmd = f"python {config['main_script']} --config {config['config_path']}"
    
    # Execute command
    print(f"Running: {cmd}")
    start_time = time.time()
    try:
        subprocess.run(cmd, shell=True, check=True)
        end_time = time.time()
        duration = end_time - start_time
        print(f"{mode_str} complete in {duration:.2f} seconds")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error during {mode_str.lower()}: {e}")
        return False

def collect_metrics(config_name):
    """Collect metrics from test results"""
    config = CONFIGURATIONS[config_name]
    metrics = {}
    
    # Look for summary files or log files
    log_files = glob.glob(os.path.join('ThermalDenoising/logs', f'test*-*.log'))
    
    # Sort by modification time to get the latest log
    if log_files:
        latest_log = max(log_files, key=os.path.getmtime)
        try:
            with open(latest_log, 'r') as f:
                log_content = f.read()
                
            # Extract metrics (adapt pattern based on your log format)
            # Example patterns - modify according to your actual log format
            import re
            
            # Look for PSNR values
            psnr_match = re.search(r'Average PSNR: (\d+\.\d+)', log_content)
            if psnr_match:
                metrics['PSNR'] = float(psnr_match.group(1))
                
            # Look for SSIM values
            ssim_match = re.search(r'Average SSIM: (\d+\.\d+)', log_content)
            if ssim_match:
                metrics['SSIM'] = float(ssim_match.group(1))
                
            # Look for other metrics as needed
            
        except Exception as e:
            print(f"Error parsing log file {latest_log}: {e}")
    
    return metrics

def generate_html_summary(test_results, output_path="test_results_summary.html"):
    """Generate HTML summary of test results"""
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Test Results Summary</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1 {{ color: #2c3e50; }}
            table {{ border-collapse: collapse; width: 100%; margin-top: 20px; }}
            th, td {{ text-align: left; padding: 12px; }}
            th {{ background-color: #3498db; color: white; }}
            tr:nth-child(even) {{ background-color: #f2f2f2; }}
            tr:hover {{ background-color: #ddd; }}
            .success {{ color: green; }}
            .failure {{ color: red; }}
            .timestamp {{ color: #7f8c8d; font-size: 0.8em; }}
        </style>
    </head>
    <body>
        <h1>Test Results Summary</h1>
        <p class="timestamp">Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        
        <table>
            <tr>
                <th>Configuration</th>
                <th>Description</th>
                <th>Status</th>
                <th>Metrics</th>
            </tr>
    """
    
    for config_name, result in test_results.items():
        config = CONFIGURATIONS[config_name]
        status_class = "success" if result['success'] else "failure"
        status_text = "Success" if result['success'] else "Failed"
        
        # Format metrics
        metrics_html = "<ul>"
        for metric_name, metric_value in result.get('metrics', {}).items():
            metrics_html += f"<li>{metric_name}: {metric_value}</li>"
        metrics_html += "</ul>"
        
        if not result.get('metrics'):
            metrics_html = "No metrics available"
            
        html_content += f"""
            <tr>
                <td>{config_name}</td>
                <td>{config['description']}</td>
                <td class="{status_class}">{status_text}</td>
                <td>{metrics_html}</td>
            </tr>
        """
    
    html_content += """
        </table>
    </body>
    </html>
    """
    
    with open(output_path, 'w') as f:
        f.write(html_content)
        
    print(f"\nTest summary saved to: {output_path}")
    return output_path

def main():
    parser = argparse.ArgumentParser(description='Test all configurations')
    parser.add_argument('--configs', nargs='+', default=['all'],
                      help='Configurations to test. Can be individual config names or groups: ' + 
                           ', '.join(sorted(list(CONFIGURATIONS.keys()) + list(CONFIG_GROUPS.keys()))))
    parser.add_argument('--train', action='store_true', help='Run in training mode instead of test mode')
    parser.add_argument('--summary', default='test_results_summary.html', help='Path for HTML summary output')
    parser.add_argument('--list', action='store_true', help='List available configurations and exit')
    
    args = parser.parse_args()
    
    if args.list:
        print("\nAvailable individual configurations:")
        for name, config in sorted(CONFIGURATIONS.items()):
            print(f"  {name:<20} - {config['description']}")
        
        print("\nAvailable configuration groups:")
        for group, configs in sorted(CONFIG_GROUPS.items()):
            print(f"  {group:<20} - {', '.join(configs)}")
        return
    
    # Determine which configurations to test
    configs_to_test = set()
    for config in args.configs:
        if config in CONFIG_GROUPS:
            configs_to_test.update(CONFIG_GROUPS[config])
        elif config in CONFIGURATIONS:
            configs_to_test.add(config)
        else:
            print(f"Warning: Unknown configuration '{config}'")
    
    configs_to_test = sorted(list(configs_to_test))
    if not configs_to_test:
        print("No valid configurations specified. Use --list to see available options.")
        return
    
    print(f"\nWill test the following {len(configs_to_test)} configurations:")
    for config in configs_to_test:
        print(f"  - {config} ({CONFIGURATIONS[config]['description']})")
    
    # Setup required directories
    setup_directories(configs_to_test)
    
    # Run tests
    test_results = {}
    for config in configs_to_test:
        success = run_test(config, train_mode=args.train)
        
        metrics = {}
        if success and not args.train:
            # Only collect metrics for successful test runs, not training runs
            metrics = collect_metrics(config)
            
        test_results[config] = {
            'success': success,
            'metrics': metrics
        }
    
    # Generate summary
    if not args.train:  # Only generate summary for test runs
        summary_path = generate_html_summary(test_results, args.summary)
        print(f"\nAll testing completed. Summary available at {summary_path}")
    else:
        print("\nAll training completed.")

if __name__ == "__main__":
    main()
