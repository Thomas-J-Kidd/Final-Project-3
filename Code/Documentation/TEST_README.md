# Test Configuration System

This system provides a comprehensive way to test all configurations for the thermal denoising and GoPro deblurring models. It allows you to run tests on specific configurations or all of them at once, and generates an HTML summary of the results.

## Quick Start

```bash
# View available options and configurations
./run_tests.sh --list

# Run tests for all configurations
./run_tests.sh

# Run tests for only thermal configurations
./run_tests.sh -c thermal

# Train a specific configuration (instead of testing)
./run_tests.sh -t -c gopro_unet_rgb
```

## Shell Script Usage

The `run_tests.sh` script provides a convenient command-line interface for the testing system:

```
Usage: ./run_tests.sh [options]

Run tests for your deep learning model configurations.

Options:
  -h, --help                Show this help message and exit
  -l, --list                List all available configurations and exit
  -t, --train               Run in training mode instead of test mode
  -c, --configs CONFIG...   Configurations to test (can specify multiple)
                            Can be individual configs or groups like 'thermal', 'gopro', etc.
  -s, --summary PATH        Path for HTML summary output (default: test_results_summary.html)

Examples:
  ./run_tests.sh --list                       # List all available configurations
  ./run_tests.sh                             # Test all configurations
  ./run_tests.sh -c thermal                  # Test only thermal configurations
  ./run_tests.sh -c gopro_nafnet_rgb         # Test a specific configuration
  ./run_tests.sh -c unet nafnet              # Test multiple groups
  ./run_tests.sh -t -c gopro_unet_rgb        # Train a specific configuration
  ./run_tests.sh -s custom_summary.html      # Save summary to custom path
```

## Python Script Usage

For more advanced usage, you can run the Python script directly:

```bash
# View help
python test_all_configurations.py --help

# List available configurations
python test_all_configurations.py --list

# Test all configurations
python test_all_configurations.py

# Test specific configurations
python test_all_configurations.py --configs thermal_unet gopro_nafnet_rgb

# Train a specific configuration
python test_all_configurations.py --configs gopro_unet_rgb --train

# Specify custom summary path
python test_all_configurations.py --summary custom_path.html
```

## Available Configurations

The system has the following configuration groups:

- `all`: All available configurations
- `thermal`: Thermal denoising configurations
- `gopro`: GoPro deblurring configurations
- `unet`: UNET-based model configurations
- `nafnet`: NAFNet-based model configurations
- `rgb`: RGB model configurations
- `gray`: Grayscale model configurations

Individual configurations:

- `thermal_unet`: Thermal Denoising with UNET
- `thermal_nafnet`: Thermal Denoising with NAFNET
- `gopro_nafnet_rgb`: GoPro Deblurring with NAFNET (RGB)
- `gopro_nafnet_gray`: GoPro Deblurring with NAFNET (Grayscale)
- `gopro_unet_rgb`: GoPro Deblurring with UNET (RGB)
- `gopro_unet_gray`: GoPro Deblurring with UNET (Grayscale)

## HTML Summary Report

After running tests, an HTML summary report is generated (default: `test_results_summary.html`). This report includes:

- A list of all tested configurations
- Status of each test (success or failure)
- Metrics for each test (PSNR, SSIM, etc. if available)

You can view this report in any web browser to get a clear overview of the test results.

## Common Testing Scenarios

### Verification of Model Implementation

To quickly verify that all model implementations are working correctly:

```bash
./run_tests.sh
```

This will run tests for all configurations and generate a summary report.

### Testing Specific Model Types

To test only UNET-based models:

```bash
./run_tests.sh -c unet
```

To test only NAFNet-based models:

```bash
./run_tests.sh -c nafnet
```

### Testing Specific Tasks

To test only thermal denoising models:

```bash
./run_tests.sh -c thermal
```

To test only GoPro deblurring models:

```bash
./run_tests.sh -c gopro
```

### Training Workflow

To train a specific model (uses MODE: 1 in config):

```bash
./run_tests.sh -t -c gopro_unet_rgb
```

To train and then test a specific model:

```bash
./run_tests.sh -t -c gopro_unet_rgb
./run_tests.sh -c gopro_unet_rgb
```

## Extending the Test System

To add new configurations to the test system, edit the `CONFIGURATIONS` dictionary in `test_all_configurations.py`:

```python
CONFIGURATIONS = {
    # Add a new configuration
    'new_config_name': {
        'config_path': 'path/to/config.yml',
        'main_script': 'main_script.py',
        'weight_dir': 'path/to/weights',
        'result_dir': 'path/to/results',
        'description': 'Description of the configuration'
    },
    # ...existing configurations...
}
```

You can also add new configuration groups in the `CONFIG_GROUPS` dictionary:

```python
CONFIG_GROUPS = {
    # Add a new group
    'new_group': ['config1', 'config2', 'config3'],
    # ...existing groups...
}
