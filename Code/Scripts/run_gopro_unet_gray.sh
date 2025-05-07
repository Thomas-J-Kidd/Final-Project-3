#!/bin/bash

# Create necessary directories
mkdir -p ThermalDenoising/weights_gopro_unet_gray
mkdir -p ThermalDenoising/results_gopro_unet_gray

# Check if we should train or test
if [ "$1" == "train" ]; then
    # Set to training mode
    sed -i 's/MODE : 0/MODE : 1/g' ThermalDenoising/conf_gopro_unet_gray.yml
    
    echo "=========================================="
    echo "Training UNET Grayscale model"
    echo "=========================================="
    
    python main_gopro.py --config ThermalDenoising/conf_gopro_unet_gray.yml
    
    # Set back to testing mode
    sed -i 's/MODE : 1/MODE : 0/g' ThermalDenoising/conf_gopro_unet_gray.yml
    
    echo "Training complete!"
else
    # Make sure we're in testing mode
    sed -i 's/MODE : 1/MODE : 0/g' ThermalDenoising/conf_gopro_unet_gray.yml
    
    echo "=========================================="
    echo "Testing UNET Grayscale model"
    echo "=========================================="
    
    python main_gopro.py --config ThermalDenoising/conf_gopro_unet_gray.yml
    
    echo "Testing complete!"
fi
