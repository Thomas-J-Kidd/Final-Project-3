#!/bin/bash

# Create necessary directories
mkdir -p ThermalDenoising/weights_gopro_unet_rgb
mkdir -p ThermalDenoising/results_gopro_unet_rgb

# Check if we should train or test
if [ "$1" == "train" ]; then
    # Set to training mode
    sed -i 's/MODE : 0/MODE : 1/g' ThermalDenoising/conf_gopro_unet_rgb.yml
    
    echo "=========================================="
    echo "Training UNET RGB model"
    echo "=========================================="
    
    python main_gopro.py --config ThermalDenoising/conf_gopro_unet_rgb.yml
    
    # Set back to testing mode
    sed -i 's/MODE : 1/MODE : 0/g' ThermalDenoising/conf_gopro_unet_rgb.yml
    
    echo "Training complete!"
else
    # Make sure we're in testing mode
    sed -i 's/MODE : 1/MODE : 0/g' ThermalDenoising/conf_gopro_unet_rgb.yml
    
    echo "=========================================="
    echo "Testing UNET RGB model"
    echo "=========================================="
    
    python main_gopro.py --config ThermalDenoising/conf_gopro_unet_rgb.yml
    
    echo "Testing complete!"
fi
