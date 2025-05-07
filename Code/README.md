# Code Documentation

This directory contains all the code for the Deep Learning final project on image denoising and deblurring.

## Code Structure

- **Main/**: Core Python scripts for running the main denoising and deblurring applications
- **Denoising/**: Implementation of neural network models (NAFNET, UNET) for image denoising and deblurring
- **Scripts/**: Shell scripts for running various experiments and tests
- **Visualization/**: Tools for visualizing results, generating comparisons, and creating presentation materials
- **Documentation/**: Additional README files and usage documentation
- **utils/**: Utility functions used across the project
- **aws_ai_tool/**: Web-based application for training and running inference on AI models using remote AWS EC2 instances
- **Notebooks/**: Jupyter notebooks for data analysis and experimentation

## Main Components

### 1. Denoising Models (Denoising/)

The core implementation of our deep learning models for image denoising and deblurring, including:

- NAFNET: Non-local Attention Fusion Network
- UNET: U-shaped Network architecture
- Diffusion-based approaches

This directory contains model definitions, training code, and inference pipelines.

#### Key Files and Directories
- `model/`: Neural network architectures (NAFNET.py, UNET.py, etc.)
- `src/`: Source code for training and testing
- `conf*.yml`: Configuration files for different model configurations
- `weights*/`: Model weights for trained models
- `results*/`: Test results and output images

### 2. Core Applications (Main/)

The main Python scripts that implement the primary functionality:

- `main.py`: Main script for thermal image denoising
- `main_gopro.py`: Main script for GoPro image deblurring
- `app.py`: Web application for interacting with models
- `config.py`: Configuration settings
- `make_noise.py`: Utility for generating synthetic noise
- `test_gopro_dataloader.py`: Testing utility for the GoPro dataset loader

### 3. Visualization Tools (Visualization/)

Scripts for generating visualizations and comparison images:

- `compare_models*.py`: Tools for comparing model outputs
- `create_*.py`: Various tools for creating visualizations
- `generate_presentation_images*.py`: Tools for generating images for presentations
- `model_comparison_visualizer.py`: Interactive tool for model comparison

### 4. Automation Scripts (Scripts/)

Shell scripts and Python utilities for automating tasks:

- `run_*.sh`: Scripts for running different model configurations
- `test_all_configurations.py`: Script for testing all model configurations

### 5. AWS AI Training & Inference Tool (aws_ai_tool/)

A complete web application for training deep learning models on AWS EC2 instances and performing inference with those models.

#### Key Features
- Remote training on AWS EC2 instances with GPU acceleration
- Model management and experiment tracking
- Real-time log monitoring
- Automatic results download
- Inference with trained models
- Web-based user interface

## Dataset Structure

The project works with two primary datasets:

1. **Thermal Image Dataset**:
   - Located in `dataset/thermal_noisy`
   - Contains noisy thermal images and their clean counterparts

2. **GoPro Deblurring Dataset**:
   - Located in `dataset/GoPro`
   - Structure:
     ```
     dataset/GoPro/
     ├── train/
     │   ├── [scene_folders]/
     │       ├── blur/         # Blurred images without gamma correction
     │       ├── blur_gamma/   # Blurred images with gamma correction
     │       └── sharp/        # Ground truth sharp images
     └── test/
         ├── [scene_folders]/
             ├── blur/
             ├── blur_gamma/
             └── sharp/
     ```

## Workflow & Usage Instructions

### 1. Setting Up the Environment

```bash
# Install required dependencies
pip install -r requirements.txt
```

### 2. Testing the Data Loader

Before running the full training or testing, verify that the dataset is loaded correctly:

```bash
# Test with GoPro dataset (RGB images and linear blur)
python Main/test_gopro_dataloader.py

# Test with grayscale images
python Main/test_gopro_dataloader.py --grayscale

# Test with gamma-corrected blur images
python Main/test_gopro_dataloader.py --gamma
```

### 3. Running Training & Testing

Each model configuration has its own script:

```bash
# For NAFNET RGB
./Scripts/run_gopro_nafnet_rgb.sh train  # Training mode
./Scripts/run_gopro_nafnet_rgb.sh        # Testing mode (default)

# For NAFNET Grayscale
./Scripts/run_gopro_nafnet_gray.sh train # Training mode
./Scripts/run_gopro_nafnet_gray.sh       # Testing mode (default)

# For UNET RGB
./Scripts/run_gopro_unet_rgb.sh train    # Training mode
./Scripts/run_gopro_unet_rgb.sh          # Testing mode (default)

# For UNET Grayscale
./Scripts/run_gopro_unet_gray.sh train   # Training mode
./Scripts/run_gopro_unet_gray.sh         # Testing mode (default)
```

To run all experiments:

```bash
./Scripts/run_gopro_tests.sh
```

### 4. Visualizing Results

After running tests, generate visualizations:

```bash
python Visualization/compare_models.py
```

For creating side-by-side comparisons:

```bash
python Visualization/create_side_by_side_comparison.py
```

### 5. AWS AI Tool Usage

1. Navigate to the `aws_ai_tool` directory
2. Initialize the database: `python alter_db.py`
3. Start the application: `python run.py`
4. Access the web interface at http://localhost:5000

## Individual Contributions

- **Thomas Kidd**: See `thomas-kidd-individual-project/Code/` for specific contributions
- **Zach Wilson**: See `zach-wilson-individual-project/Code/` for specific contributions

## Documentation References

For more detailed information about specific components:

- GoPro deblurring: See `Documentation/README_GOPRO.md`
- Model comparison: See `Documentation/MODEL_COMPARISON_README.md`
- NAFNET training improvements: See `Documentation/NAFNET_TRAINING_IMPROVEMENTS.md`
- Testing procedures: See `Documentation/TEST_README.md`
- AWS training: See `Documentation/AWS_TRAINING_README.md`

## Environment Requirements

- Python 3.7+
- PyTorch 1.8+
- CUDA-compatible GPU recommended for training
- Additional dependencies listed in `requirements.txt`
