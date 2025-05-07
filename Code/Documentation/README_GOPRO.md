# GoPro Deblurring with ThermalDenoising Models

This project adapts the ThermalDenoising models to work with the GoPro deblurring dataset. It supports both NAFNET and UNET architectures, and can process both RGB and grayscale images.

## Dataset Structure

The GoPro dataset should be organized as follows:

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

Each scene folder contains numbered PNG images (e.g., 000001.png, 000002.png, etc.).

## Configuration Files

Four configuration files are provided for different model architectures and image types:

1. `ThermalDenoising/conf_gopro_nafnet_rgb.yml` - NAFNET model for RGB images
2. `ThermalDenoising/conf_gopro_nafnet_gray.yml` - NAFNET model for grayscale images
3. `ThermalDenoising/conf_gopro_unet_rgb.yml` - UNET model for RGB images
4. `ThermalDenoising/conf_gopro_unet_gray.yml` - UNET model for grayscale images

## Running the Code

### Testing the Data Loader

Before running the full training or testing, you can verify that the GoPro dataset is being loaded correctly:

```bash
# Test with RGB images and linear blur
python test_gopro_dataloader.py

# Test with grayscale images
python test_gopro_dataloader.py --grayscale

# Test with gamma-corrected blur images
python test_gopro_dataloader.py --gamma

# Test with training split
python test_gopro_dataloader.py --split train

# Save more sample images
python test_gopro_dataloader.py --samples 10
```

This will save sample images to a directory for inspection.

### Individual Model Scripts

Each model configuration has its own script that can be used for training or testing:

```bash
# For NAFNET RGB
./run_gopro_nafnet_rgb.sh        # Testing mode (default)
./run_gopro_nafnet_rgb.sh train  # Training mode

# For NAFNET Grayscale
./run_gopro_nafnet_gray.sh       # Testing mode (default)
./run_gopro_nafnet_gray.sh train # Training mode

# For UNET RGB
./run_gopro_unet_rgb.sh          # Testing mode (default)
./run_gopro_unet_rgb.sh train    # Training mode

# For UNET Grayscale
./run_gopro_unet_gray.sh         # Testing mode (default)
./run_gopro_unet_gray.sh train   # Training mode
```

### Running All Experiments

To run all experiments (training and testing for all configurations), use:

```bash
./run_gopro_tests.sh
```

This will:
1. Train all four model configurations
2. Test all four model configurations
3. Save results in their respective directories

## Directory Structure

- `ThermalDenoising/weights_gopro_*` - Model weights for each configuration
- `ThermalDenoising/results_gopro_*` - Test results for each configuration
- `ThermalDenoising/training_gopro_*` - Training logs and validation images

## Implementation Details

- The models use a two-step approach with an initial predictor and a diffusion-based denoiser
- Training is limited to 20,000 iterations as specified
- Both RGB (3-channel) and grayscale (1-channel) versions are supported
- The code handles the nested directory structure of the GoPro dataset

## Metrics

The following metrics are calculated during testing:
- PSNR (Peak Signal-to-Noise Ratio)
- SSIM (Structural Similarity Index)
- LPIPS (Learned Perceptual Image Patch Similarity) - if available
- DISTS (Deep Image Structure and Texture Similarity) - if available

## Notes

- The GoPro dataset is significantly larger than the thermal dataset (~2,300 training images vs. 701)
- The models can use either the regular blur images or the gamma-corrected versions (controlled by the `USE_GAMMA` parameter in the config files)
- For best results, train the models on a machine with a GPU
