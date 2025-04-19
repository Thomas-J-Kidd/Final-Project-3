# AWS AI Training & Inference Tool

A web-based application for training and running inference on AI models using remote AWS EC2 instances. This tool simplifies the process of training deep learning models for thermal image denoising and super-resolution on AWS cloud infrastructure.

## Features

- **Remote Training**: Train AI models on AWS EC2 instances with GPU acceleration
- **Model Management**: Upload and manage custom training scripts
- **Experiment Tracking**: Monitor training jobs and view logs in real-time
- **Results Storage**: Automatically download and store trained models and logs
- **Inference**: Run inference on images using trained models
- **Web Interface**: Easy-to-use Flask-based web interface

## Prerequisites

- Python 3.7+
- AWS account with EC2 access
- SSH key pair for EC2 instance access
- Ubuntu-based EC2 instance with NVIDIA GPU and CUDA support

## Installation

1. Clone this repository:
   ```bash
   git clone <repository-url>
   cd aws_ai_tool
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Initialize the database:
   ```bash
   python alter_db.py
   ```

## Configuration

Before using the tool, you need to configure the SSH connection to your AWS EC2 instance:

1. Launch the application:
   ```bash
   python run.py
   ```

2. Navigate to the Settings tab in the web interface
3. Configure the following settings:
   - **SSH Hostname**: The public IP or DNS of your EC2 instance
   - **SSH Username**: The username for SSH access (typically 'ubuntu')
   - **SSH Key Path**: Path to your private SSH key file (e.g., ~/.ssh/id_rsa)
   - **Upload Folder**: Directory for storing uploaded model scripts (default: 'uploads')
   - **Results Folder**: Directory for storing downloaded results (default: 'results')

## Usage Guide

### 1. Model Management

The Model Management tab allows you to upload and manage your training scripts.

#### Uploading a Model Script

1. Navigate to the Model Management tab
2. Click the "Choose File" button and select your Python training script
3. Click "Upload" to upload the script

#### Requirements for Training Scripts

Your training scripts should:
- Accept command-line arguments for data directory, output directory, and hyperparameters
- Save model checkpoints in the specified output directory
- Create a `_SUCCESS` file in the output directory upon successful completion
- Create a `_FAILED` file with error details if training fails

Example script structure:
```python
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=16)
    args = parser.parse_args()
    
    # Your training code here
    
    # On success:
    with open(f"{args.output_dir}/_SUCCESS", 'w') as f:
        f.write("Training completed successfully")
    
    # On failure:
    with open(f"{args.output_dir}/_FAILED", 'w') as f:
        f.write(f"Training failed: {error_message}")

if __name__ == '__main__':
    main()
```

### 2. Job Management

The Job Management tab is the main interface for launching and monitoring training jobs.

#### Launching a Training Job

1. Navigate to the Job Management tab
2. Fill in the form:
   - **Experiment Name**: A descriptive name for your experiment
   - **Model Script**: Select one of your uploaded training scripts
   - **Data Source**: Description or path of the data (note: the tool assumes data is already on the EC2 instance at `~/dataset/thermal`)
   - **Hyperparameters**: JSON object with hyperparameters (e.g., `{"epochs": 50, "lr": 0.001, "batch_size": 16}`)
3. Click "Launch Training Job"

#### Monitoring Jobs

- **Active Jobs**: Shows currently running jobs with their status
- **Completed Experiments**: Shows finished jobs with their results
- **View Logs**: Click to see real-time logs for a running job
- **Check Status**: Click to manually update the status of a job

### 3. Inference

The Inference tab allows you to run inference on images using your trained models.

#### Running Inference

1. Navigate to the Inference tab
2. Upload an image using the "Choose File" button
3. Select one or more trained models from the list
4. Click "Run Inference"
5. View the original and processed images side by side

## Data Structure

The tool expects the following data structure on the remote EC2 instance:

```
~/dataset/thermal/
├── train/
│   ├── GT/           # High-resolution ground truth images
│   └── LR_x8/        # Low-resolution images (8x downsampled)
├── val/
│   ├── GT/           # Validation high-resolution images
│   └── LR_x8/        # Validation low-resolution images
└── test/
    ├── GT/           # Test high-resolution images
    └── LR_x8/        # Test low-resolution images
```

## Project Structure

```
aws_ai_tool/
├── app/                    # Flask application
│   ├── core/               # Core functionality
│   │   ├── aws_ops.py      # AWS operations
│   │   ├── db.py           # Database operations
│   │   └── ssh_ops.py      # SSH operations
│   ├── static/             # Static files
│   │   ├── inference_images/    # Uploaded images for inference
│   │   └── inference_results/   # Inference results
│   ├── templates/          # HTML templates
│   ├── __init__.py         # App initialization
│   ├── main.py             # Main routes
│   └── schema.sql          # Database schema
├── results/                # Downloaded results
├── uploads/                # Uploaded model scripts
├── alter_db.py             # Database initialization script
├── dummy_train.py          # Example training script
├── remote_inference.py     # Script for remote inference
├── requirements.txt        # Python dependencies
├── run.py                  # Application entry point
└── srcnn_train.py          # Example SRCNN training script
```

## Troubleshooting

### Common Issues

1. **SSH Connection Failed**
   - Ensure the EC2 instance is running
   - Verify the SSH hostname, username, and key path
   - Check that the security group allows SSH access

2. **Training Script Errors**
   - Check the logs for error messages
   - Ensure your script follows the required structure
   - Verify that the data paths on the EC2 instance are correct

3. **Database Errors**
   - Run `python alter_db.py` to reinitialize the database

4. **Missing Results**
   - Check the status of the job
   - Verify SSH connection is working
   - Check disk space on both local and remote machines

## License

[MIT License](LICENSE)

## Acknowledgments

- This tool was developed as part of the Deep Learning course final project
- Uses Flask for the web interface
- Uses Paramiko for SSH operations
- Uses PyTorch for the example training scripts
