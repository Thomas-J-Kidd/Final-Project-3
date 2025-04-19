import time
import argparse
import os
import sys
import traceback

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Dummy Training Script")
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save outputs')
    # Add other dummy arguments if needed for testing command line parsing
    parser.add_argument('--epochs', type=int, default=3, help='Dummy epochs')
    parser.add_argument('--lr', type=float, default=0.01, help='Dummy learning rate')
    parser.add_argument('--data_dir', type=str, default=None, help='Dummy data directory')
    parser.add_argument('--fail', action='store_true', help='Simulate a training failure')
    
    return parser.parse_args()

def run_training(args):
    """Run the actual training process."""
    print(f"Starting dummy training...")
    print(f"Arguments received:")
    print(f"  Output Directory: {args.output_dir}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Learning Rate: {args.lr}")
    print(f"  Data Directory: {args.data_dir}")
    print(f"  Fail Flag: {args.fail}")
    sys.stdout.flush()  # Ensure output is written immediately

    # Ensure output directory exists
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"Created output directory: {args.output_dir}")

    # Simulate a failure if requested
    if args.fail:
        raise Exception("Simulated training failure requested via --fail flag")

    # Simulate training epochs
    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch}/{args.epochs} starting...")
        sys.stdout.flush()
        time.sleep(5)  # Simulate work for 5 seconds per epoch
        print(f"Epoch {epoch}/{args.epochs} completed. Dummy loss: {1.0 / epoch:.4f}")
        sys.stdout.flush()

    # Simulate saving a model checkpoint
    dummy_model_path = os.path.join(args.output_dir, 'dummy_model.pth')
    with open(dummy_model_path, 'w') as f:
        f.write('This is a dummy model file.\n')
    print(f"Dummy model saved to {dummy_model_path}")

    print("Dummy training finished successfully.")
    sys.stdout.flush()
    return True

def main():
    """Main function with proper error handling and marker file creation."""
    args = parse_arguments()
    
    try:
        # Run the training process
        success = run_training(args)
        
        # Create success marker file
        with open(os.path.join(args.output_dir, '_SUCCESS'), 'w') as f:
            f.write('Dummy training completed successfully.\n')
        
        # Exit with success code
        sys.exit(0)
        
    except Exception as e:
        # Print the full exception traceback for debugging
        print(f"ERROR: Training failed with exception: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        sys.stderr.flush()
        
        # Ensure output directory exists for the marker file
        try:
            if not os.path.exists(args.output_dir):
                os.makedirs(args.output_dir, exist_ok=True)
        except Exception as mkdir_err:
            print(f"Failed to create output directory: {mkdir_err}", file=sys.stderr)
        
        # Create failure marker file with the error message
        try:
            with open(os.path.join(args.output_dir, '_FAILED'), 'w') as f:
                f.write(f'Training failed: {str(e)}\n')
                f.write(traceback.format_exc())
        except Exception as marker_err:
            print(f"Failed to create failure marker file: {marker_err}", file=sys.stderr)
        
        # Exit with error code
        sys.exit(1)

if __name__ == '__main__':
    main()
