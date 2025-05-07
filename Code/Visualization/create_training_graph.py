timport re
import matplotlib.pyplot as plt
import numpy as np
import os

# Path to the log file
log_file = 'ThermalDenoising/logs/trainThermalDenoising_250429-153933.log'

def extract_metrics_from_log(log_path):
    """Extract metrics from the training log file."""
    iterations = []
    psnr_values = []
    ssim_values = []
    lpips_values = []
    dists_values = []
    
    with open(log_path, 'r') as file:
        log_content = file.read()
        
        # Find all iteration blocks
        iteration_pattern = r"validation\. Iteration (\d+) ----\n.*?----Average PSNR\t: ([\d\.]+)\n.*?----Average SSIM\t: ([\d\.]+)\n.*?----Average LPIPS\t: ([\d\.]+)\n.*?----Average DISTS\t: ([\d\.]+)"
        matches = re.findall(iteration_pattern, log_content, re.DOTALL)
        
        for match in matches:
            iteration, psnr, ssim, lpips, dists = match
            iterations.append(int(iteration))
            psnr_values.append(float(psnr))
            ssim_values.append(float(ssim))
            lpips_values.append(float(lpips))
            dists_values.append(float(dists))
    
    return {
        'iterations': iterations,
        'psnr': psnr_values,
        'ssim': ssim_values,
        'lpips': lpips_values,
        'dists': dists_values
    }

def create_training_graphs(metrics):
    """Create and save training graphs."""
    # Create output directory if it doesn't exist
    os.makedirs('presentation_images', exist_ok=True)
    
    plt.figure(figsize=(12, 9))
    
    # Set up the figure with 2x2 subplots
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('NAFNET Thermal Denoising Training Progress', fontsize=16)
    
    # Plot PSNR (higher is better)
    axs[0, 0].plot(metrics['iterations'], metrics['psnr'], 'b-', linewidth=2)
    axs[0, 0].set_title('PSNR Improvement (Higher is Better)')
    axs[0, 0].set_xlabel('Iterations')
    axs[0, 0].set_ylabel('PSNR (dB)')
    axs[0, 0].grid(True, linestyle='--', alpha=0.7)
    
    # Plot SSIM (higher is better)
    axs[0, 1].plot(metrics['iterations'], metrics['ssim'], 'g-', linewidth=2)
    axs[0, 1].set_title('SSIM Improvement (Higher is Better)')
    axs[0, 1].set_xlabel('Iterations')
    axs[0, 1].set_ylabel('SSIM')
    axs[0, 1].grid(True, linestyle='--', alpha=0.7)
    
    # Plot LPIPS (lower is better)
    axs[1, 0].plot(metrics['iterations'], metrics['lpips'], 'r-', linewidth=2)
    axs[1, 0].set_title('LPIPS Reduction (Lower is Better)')
    axs[1, 0].set_xlabel('Iterations')
    axs[1, 0].set_ylabel('LPIPS')
    axs[1, 0].grid(True, linestyle='--', alpha=0.7)
    
    # Plot DISTS (lower is better)
    axs[1, 1].plot(metrics['iterations'], metrics['dists'], 'm-', linewidth=2)
    axs[1, 1].set_title('DISTS Reduction (Lower is Better)')
    axs[1, 1].set_xlabel('Iterations')
    axs[1, 1].set_ylabel('DISTS')
    axs[1, 1].grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save the figure
    plt.savefig('presentation_images/training_metrics.png', dpi=300, bbox_inches='tight')
    print(f"Training metrics graph saved to 'presentation_images/training_metrics.png'")
    
    # Create a simplified version showing just PSNR and LPIPS for the presentation slide
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(metrics['iterations'], metrics['psnr'], 'b-', linewidth=3)
    plt.title('PSNR Improvement Over Training', fontsize=14)
    plt.xlabel('Iterations', fontsize=12)
    plt.ylabel('PSNR (dB)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.subplot(1, 2, 2)
    plt.plot(metrics['iterations'], metrics['lpips'], 'r-', linewidth=3)
    plt.title('LPIPS Reduction Over Training', fontsize=14)
    plt.xlabel('Iterations', fontsize=12)
    plt.ylabel('LPIPS (Perceptual Similarity)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('presentation_images/training_slide_metrics.png', dpi=300, bbox_inches='tight')
    print(f"Presentation slide metrics saved to 'presentation_images/training_slide_metrics.png'")

if __name__ == "__main__":
    # Extract metrics from the log file
    metrics = extract_metrics_from_log(log_file)
    
    # Create and save training graphs
    create_training_graphs(metrics)
    
    # Print out key metrics improvement
    print(f"Training Progress Overview:")
    print(f"----------------------------")
    print(f"PSNR: {metrics['psnr'][0]:.2f} dB → {metrics['psnr'][-1]:.2f} dB (Improvement: {metrics['psnr'][-1] - metrics['psnr'][0]:.2f} dB)")
    print(f"SSIM: {metrics['ssim'][0]:.4f} → {metrics['ssim'][-1]:.4f} (Improvement: {metrics['ssim'][-1] - metrics['ssim'][0]:.4f})")
    print(f"LPIPS: {metrics['lpips'][0]:.4f} → {metrics['lpips'][-1]:.4f} (Reduction: {metrics['lpips'][0] - metrics['lpips'][-1]:.4f})")
    print(f"DISTS: {metrics['dists'][0]:.4f} → {metrics['dists'][-1]:.4f} (Reduction: {metrics['dists'][0] - metrics['dists'][-1]:.4f})")
