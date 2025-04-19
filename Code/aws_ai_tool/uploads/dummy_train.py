import time
import argparse
import os
import sys

def main():
    parser = argparse.ArgumentParser(description="Dummy Training Script")
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save outputs')
    # Add other dummy arguments if needed for testing command line parsing
    parser.add_argument('--epochs', type=int, default=3, help='Dummy epochs')
    parser.add_argument('--lr', type=float, default=0.01, help='Dummy learning rate')
    parser.add_argument('--data_dir', type=str, default=None, help='Dummy data directory')

    args = parser.parse_args()

    print(f"Starting dummy training...")
    print(f"Arguments received:")
    print(f"  Output Directory: {args.output_dir}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Learning Rate: {args.lr}")
    print(f"  Data Directory: {args.data_dir}")
    sys.stdout.flush() # Ensure output is written immediately

    # Ensure output directory exists (it should have been created by the main app)
    if not os.path.exists(args.output_dir):
        print(f"Error: Output directory {args.output_dir} does not exist.", file=sys.stderr)
        # Create a _FAILED marker if directory doesn't exist
        try:
            # Attempt to create it just in case, but still report failure if needed
            os.makedirs(args.output_dir, exist_ok=True)
            with open(os.path.join(args.output_dir, '_FAILED'), 'w') as f:
                f.write('Output directory did not exist initially.')
        except Exception:
            pass # Avoid error during failure reporting
        sys.exit(1) # Exit with error code

    # Simulate training epochs
    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch}/{args.epochs} starting...")
        sys.stdout.flush()
        time.sleep(5) # Simulate work for 5 seconds per epoch
        print(f"Epoch {epoch}/{args.epochs} completed. Dummy loss: {1.0 / epoch:.4f}")
        sys.stdout.flush()

    # Simulate saving a model checkpoint
    dummy_model_path = os.path.join(args.output_dir, 'dummy_model.pth')
    try:
        with open(dummy_model_path, 'w') as f:
            f.write('This is a dummy model file.\n')
        print(f"Dummy model saved to {dummy_model_path}")
    except Exception as e:
        print(f"Error saving dummy model: {e}", file=sys.stderr)
        # Create a _FAILED marker
        with open(os.path.join(args.output_dir, '_FAILED'), 'w') as f:
            f.write(f'Error saving dummy model: {e}')
        sys.exit(1)

    # Create success marker file
    success_marker_path = os.path.join(args.output_dir, '_SUCCESS')
    try:
        with open(success_marker_path, 'w') as f:
            f.write('Dummy training completed successfully.\n')
        print("Dummy training finished successfully.")
        sys.stdout.flush()
    except Exception as e:
        print(f"Error creating success marker: {e}", file=sys.stderr)
        # Attempt to create a _FAILED marker if _SUCCESS fails
        try:
            with open(os.path.join(args.output_dir, '_FAILED'), 'w') as f:
                f.write(f'Error creating success marker: {e}')
        except Exception:
            pass
        sys.exit(1)

if __name__ == '__main__':
    main()
