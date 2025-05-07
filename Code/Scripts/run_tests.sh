#!/bin/bash

# Help function
show_help() {
    echo "Usage: ./run_tests.sh [options]"
    echo ""
    echo "Run tests for your deep learning model configurations."
    echo ""
    echo "Options:"
    echo "  -h, --help                Show this help message and exit"
    echo "  -l, --list                List all available configurations and exit"
    echo "  -t, --train               Run in training mode instead of test mode"
    echo "  -c, --configs CONFIG...   Configurations to test (can specify multiple)"
    echo "                            Can be individual configs or groups like 'thermal', 'gopro', etc."
    echo "  -s, --summary PATH        Path for HTML summary output (default: test_results_summary.html)"
    echo ""
    echo "Examples:"
    echo "  ./run_tests.sh --list                       # List all available configurations"
    echo "  ./run_tests.sh                             # Test all configurations"
    echo "  ./run_tests.sh -c thermal                  # Test only thermal configurations"
    echo "  ./run_tests.sh -c gopro_nafnet_rgb         # Test a specific configuration"
    echo "  ./run_tests.sh -c unet nafnet              # Test multiple groups"
    echo "  ./run_tests.sh -t -c gopro_unet_rgb        # Train a specific configuration"
    echo "  ./run_tests.sh -s custom_summary.html      # Save summary to custom path"
}

# Default parameters
CONFIGS=("all")
TRAIN=false
SUMMARY="test_results_summary.html"

# Process command-line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            show_help
            exit 0
            ;;
        -l|--list)
            python test_all_configurations.py --list
            exit 0
            ;;
        -t|--train)
            TRAIN=true
            shift
            ;;
        -c|--configs)
            shift
            CONFIGS=()
            while [[ $# -gt 0 && ! "$1" =~ ^- ]]; do
                CONFIGS+=("$1")
                shift
            done
            if [ ${#CONFIGS[@]} -eq 0 ]; then
                echo "Error: --configs option requires at least one configuration"
                show_help
                exit 1
            fi
            ;;
        -s|--summary)
            if [[ $# -lt 2 ]]; then
                echo "Error: --summary option requires a path"
                show_help
                exit 1
            fi
            SUMMARY="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Make script executable if it's not already
chmod +x test_all_configurations.py

# Prepare the command
COMMAND="./test_all_configurations.py"

# Add configurations
if [ ${#CONFIGS[@]} -gt 0 ]; then
    COMMAND+=" --configs ${CONFIGS[@]}"
fi

# Add train flag if needed
if [ "$TRAIN" = true ]; then
    COMMAND+=" --train"
fi

# Add summary path
COMMAND+=" --summary $SUMMARY"

# Display the command to be executed
echo "Executing: $COMMAND"
echo ""

# Execute the command
eval "$COMMAND"
