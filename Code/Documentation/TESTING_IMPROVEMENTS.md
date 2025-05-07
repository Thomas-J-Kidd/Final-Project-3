# Testing Improvements Documentation

This document explains the improvements made to the testing system for the GoPro deblurring models. The primary change is the ability to limit the number of test images used, which significantly speeds up the testing process.

## Problem

The original testing system processed all 1111 images in the GoPro test dataset, which is time-consuming. This made quick iterations and debugging difficult.

## Solution

A new `max_images` parameter has been added to the GoProData class, allowing for testing with a subset of images. By default, this is set to 50 images in all GoPro configuration files.

## Key Changes

1. **GoProData Class (ThermalDenoising/data/goprodata.py)**
   - Added a new `max_images` parameter that randomly selects a subset of images
   - Implemented with a fixed random seed for reproducibility

2. **Configuration Files**
   - Added `MAX_TEST_IMAGES : 50` to all GoPro configuration files:
     - ThermalDenoising/conf_gopro_nafnet_gray.yml
     - ThermalDenoising/conf_gopro_nafnet_rgb.yml
     - ThermalDenoising/conf_gopro_unet_gray.yml
     - ThermalDenoising/conf_gopro_unet_rgb.yml

3. **Tester Class (ThermalDenoising/src/tester_gopro.py)**
   - Added code to pass the max_images parameter from the config to the dataset

4. **Scripts**
   - Updated `run_gopro_tests.sh` to allow setting the max_images parameter via command line
   - Created a new `run_quick_test.sh` script that demonstrates testing with 50 images

## How to Use

### Quick Test with 50 Images

```bash
./run_quick_test.sh
```

This will run a test with the nafnet_gray model using just 50 random images, which should be much faster than the full test.

### Running Tests with Custom Image Limit

```bash
# Test with 100 images
./run_gopro_tests.sh --max-images=100

# Test with all images (original behavior)
./run_gopro_tests.sh --max-images=0
```

### Test All Configurations with Limited Images

```bash
# Test all configurations with 50 images (default)
./run_gopro_tests.sh

# Test all configurations with 20 images
./run_gopro_tests.sh --max-images=20
```

## Expected Speedup

With the default setting of 50 images instead of 1111, you can expect:
- Approximately 20x faster test runs
- Similar quality metrics (the random sample should be representative)
- Less disk space used for result images

## When to Use Full Testing

You should still use full testing (all 1111 images) for:
- Final evaluation before publishing results
- Creating comprehensive comparison images
- When you need the most accurate average metrics

For development and debugging, the limited testing is much more practical.

## How it Works

The implementation uses a random sampling approach to select a diverse set of test images. The random seed is fixed to ensure reproducibility across runs.

When the system starts, it will print a message like:
```
Limiting dataset to 50 images out of 1111
```

This confirms that the limitation is working as expected.
