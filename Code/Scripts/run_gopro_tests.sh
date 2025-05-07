#!/bin/bash

# Default number of test images (0 means use all images)
MAX_TEST_IMAGES=50

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --max-images=*)
      MAX_TEST_IMAGES="${1#*=}"
      shift
      ;;
    --max-images)
      MAX_TEST_IMAGES="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: $0 [--max-images=NUMBER]"
      exit 1
      ;;
  esac
done

echo "Using max test images: $MAX_TEST_IMAGES"

# Check and fix configuration files
echo "Checking and fixing configuration files..."

# Ensure all configs have MODEL_TYPE set correctly
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
done

# Ensure Thermal config has MODEL_TYPE
if ! grep -q "MODEL_TYPE" "ThermalDenoising/conf.yml"; then
    sed -i '1s/^/# model\nMODEL_TYPE : '"'NAFNET'"'     # model type - UNET or NAFNET\n/' "ThermalDenoising/conf.yml"
    echo "Added MODEL_TYPE to ThermalDenoising/conf.yml"
fi

# Create necessary directories
mkdir -p ThermalDenoising/weights_gopro_nafnet_rgb
mkdir -p ThermalDenoising/weights_gopro_nafnet_gray
mkdir -p ThermalDenoising/weights_gopro_unet_rgb
mkdir -p ThermalDenoising/weights_gopro_unet_gray

mkdir -p ThermalDenoising/results_gopro_nafnet_rgb
mkdir -p ThermalDenoising/results_gopro_nafnet_gray
mkdir -p ThermalDenoising/results_gopro_unet_rgb
mkdir -p ThermalDenoising/results_gopro_unet_gray

# Function to run training and testing for a configuration
run_experiment() {
    CONFIG=$1
    NAME=$2
    
    echo "=========================================="
    echo "Running experiment: $NAME"
    echo "=========================================="
    
    # Training
    echo "Starting training for $NAME..."
    python main_gopro.py --config ThermalDenoising/$CONFIG
    
    echo "Training complete for $NAME"
    echo ""
}

# Set configurations to training mode
sed -i 's/MODE : 0/MODE : 1/g' ThermalDenoising/conf_gopro_nafnet_rgb.yml
sed -i 's/MODE : 0/MODE : 1/g' ThermalDenoising/conf_gopro_nafnet_gray.yml
sed -i 's/MODE : 0/MODE : 1/g' ThermalDenoising/conf_gopro_unet_rgb.yml
sed -i 's/MODE : 0/MODE : 1/g' ThermalDenoising/conf_gopro_unet_gray.yml

# Run training for all configurations
run_experiment "conf_gopro_nafnet_rgb.yml" "NAFNET RGB"
run_experiment "conf_gopro_nafnet_gray.yml" "NAFNET Grayscale"
run_experiment "conf_gopro_unet_rgb.yml" "UNET RGB"
run_experiment "conf_gopro_unet_gray.yml" "UNET Grayscale"

# Set configurations to testing mode and update MAX_TEST_IMAGES
for config_file in ThermalDenoising/conf_gopro_*.yml; do
    # Set mode to testing
    sed -i 's/MODE : 1/MODE : 0/g' "$config_file"
    
    # Update MAX_TEST_IMAGES value
    if grep -q "MAX_TEST_IMAGES" "$config_file"; then
        sed -i "s/MAX_TEST_IMAGES : [0-9]*/MAX_TEST_IMAGES : $MAX_TEST_IMAGES/g" "$config_file"
    else
        # If the parameter doesn't exist, add it in the TEST section
        sed -i "/^#TEST/a MAX_TEST_IMAGES : $MAX_TEST_IMAGES      # maximum number of test images to process (set to 0 for all images)" "$config_file"
    fi
done

# Add max test images to thermal config if needed
if ! grep -q "MAX_TEST_IMAGES" "ThermalDenoising/conf.yml"; then
    sed -i "/^#TEST/a MAX_TEST_IMAGES : $MAX_TEST_IMAGES      # maximum number of test images to process (set to 0 for all images)" "ThermalDenoising/conf.yml"
    echo "Added MAX_TEST_IMAGES to ThermalDenoising/conf.yml"
fi

# Run testing for all configurations
echo "=========================================="
echo "Running tests for all configurations with max $MAX_TEST_IMAGES images"
echo "=========================================="

# Function to run a test with better error handling
run_test() {
    local config=$1
    local name=$2
    
    echo "=========================================="
    echo "Testing $name"
    echo "=========================================="
    
    if python main_gopro.py --config "$config"; then
        echo "Test completed successfully for $name"
        return 0
    else
        echo "WARNING: Test failed for $name"
        return 1
    fi
}

# Run NAFNET tests first as they're more likely to succeed
run_test "ThermalDenoising/conf_gopro_nafnet_gray.yml" "GoPro Deblurring with NAFNET (Grayscale)"
run_test "ThermalDenoising/conf_gopro_nafnet_rgb.yml" "GoPro Deblurring with NAFNET (RGB)"

# Then run UNET tests
run_test "ThermalDenoising/conf_gopro_unet_rgb.yml" "GoPro Deblurring with UNET (RGB)"
run_test "ThermalDenoising/conf_gopro_unet_gray.yml" "GoPro Deblurring with UNET (Grayscale)"

# Test thermal NAFNET if requested
if [[ "$@" == *"--thermal"* ]]; then
    echo "=========================================="
    echo "Testing Thermal Denoising with NAFNET"
    echo "=========================================="
    if python main.py --config ThermalDenoising/conf.yml; then
        echo "Test completed successfully for Thermal Denoising"
    else
        echo "WARNING: Test failed for Thermal Denoising"
    fi
fi

echo ""
echo "All test runs completed!"
echo "To run with full dataset, use: $0 --max-images=0"
echo "Check ThermalDenoising/logs for detailed results."
