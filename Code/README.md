# Code Documentation

This directory contains all the code for the Deep Learning final project.

## Code Structure

- **aws_ai_tool/**: Web-based application for training and running inference on AI models using remote AWS EC2 instances

## Project Components

### AWS AI Training & Inference Tool

The `aws_ai_tool` directory contains a complete web application for training deep learning models on AWS EC2 instances and performing inference with those models. This tool simplifies the process of training models for thermal image denoising and super-resolution.

#### Key Features

- Remote training on AWS EC2 instances with GPU acceleration
- Model management and experiment tracking
- Real-time log monitoring
- Automatic results download
- Inference with trained models
- Web-based user interface

#### Getting Started

To use the AWS AI tool:

1. Navigate to the `aws_ai_tool` directory
2. Follow the installation and setup instructions in the [AWS AI Tool README](aws_ai_tool/README.md)

## Environment Setup

### AWS AI Tool Requirements

- Python 3.7+
- Flask
- PyTorch
- Paramiko (for SSH operations)
- Boto3 (for AWS operations)
- Pillow and OpenCV (for image processing)

For a complete list of dependencies, see the [requirements.txt](aws_ai_tool/requirements.txt) file.

## Usage

### AWS AI Tool

1. Install dependencies: `pip install -r aws_ai_tool/requirements.txt`
2. Initialize the database: `python aws_ai_tool/alter_db.py`
3. Start the application: `python aws_ai_tool/run.py`
4. Access the web interface at http://localhost:5000
