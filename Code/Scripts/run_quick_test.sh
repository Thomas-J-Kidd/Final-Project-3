#!/bin/bash

# This script demonstrates testing with the limited dataset (50 images instead of 1111)

# Set the number of test images (adjust as needed)
MAX_IMAGES=50

# Ensure all configs have MODEL_TYPE set correctly
echo "Checking and fixing configuration files..."

# Fix UnetDPM import in all configuration files
for config_file in ThermalDenoising/conf_gopro_*.yml; do
    # Ensure MODEL_TYPE is set
    if ! grep -q "MODEL_TYPE" "$config_file"; then
        if [[ "$config_file" == *"unet"* ]]; then
            sed -i '1s/^/# model\nMODEL_TYPE : '"'UNET'"'     # model type - UNET or NAFNET\n/' "$config_file"
        else
            sed -i '1s/^/# model\nMODEL_TYPE : '"'NAFNET'"'     # model type - UNET or NAFNET\n/' "$config_file"
        fi
        echo "Added MODEL_TYPE to $config_file"
    fi

    # Ensure MAX_TEST_IMAGES is set
    if grep -q "MAX_TEST_IMAGES" "$config_file"; then
        sed -i "s/MAX_TEST_IMAGES : [0-9]*/MAX_TEST_IMAGES : $MAX_IMAGES/g" "$config_file"
    else
        # If the parameter doesn't exist, add it in the TEST section
        sed -i "/^#TEST/a MAX_TEST_IMAGES : $MAX_IMAGES      # maximum number of test images to process (set to 0 for all images)" "$config_file"
    fi
    
    # Ensure we're in test mode
    sed -i 's/MODE : 1/MODE : 0/g' "$config_file"
done

# Ensure Thermal config has MODEL_TYPE
if ! grep -q "MODEL_TYPE" "ThermalDenoising/conf.yml"; then
    sed -i '1s/^/# model\nMODEL_TYPE : '"'NAFNET'"'     # model type - UNET or NAFNET\n/' "ThermalDenoising/conf.yml"
    echo "Added MODEL_TYPE to ThermalDenoising/conf.yml"
fi

# Run the test with limited images (configured in the YML file)
echo "=== Running test with $MAX_IMAGES images ==="
echo "This will be much faster than the full 1111 image test"

# Run NAFNET Gray which is best supported
CONFIG="ThermalDenoising/conf_gopro_nafnet_gray.yml"
echo "Testing with configuration: $CONFIG"

if python main_gopro.py --config $CONFIG; then
    echo ""
    echo "Test complete! Check ThermalDenoising/logs for the results."
    echo "Check ThermalDenoising/results_gopro_nafnet_gray for the output images."
else
    echo ""
    echo "Test failed. Check error messages above."
fi
