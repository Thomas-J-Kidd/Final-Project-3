import os
import cv2
import numpy as np
import random
import shutil

# --- Parameters (ADJUST THESE BASED ON YOUR SENSOR/ESTIMATES) ---

# Paths
BASE_THERMAL_DIR = './thermal'  # Assumes the 'thermal' directory is in the same folder as the script
OUTPUT_DIR = './thermal_noisy' # Where to save the noisy dataset
SETS = ['train', 'val']       # Process train and validation sets ('test' GT is usually not available)

# Noise Levels & Characteristics
# Assuming image pixel values are initially treated in a range (e.g., 0-255 for 8-bit)
# Adjust these standard deviations based on that range.

# Gaussian Read Noise (and similar random thermal noise)
GAUSSIAN_READ_NOISE_STD = 5.0 # Standard deviation for Gaussian read noise

# Dark Current Shot Noise (Poisson component)
AVG_DARK_CURRENT_RATE = 0.1   # Average electrons/pixel/second (example value)
EXPOSURE_TIME_SEC = 0.03    # Example exposure time in seconds
SENSOR_GAIN = 1.5           # Example gain electrons/pixel value (ADU)
                            # If AVG_DARK_CURRENT_RATE is 0, Poisson noise won't be added.

# Fixed Pattern Noise (FPN) Parameters (Simplified Model)
# DSNU (Offset Noise - added to the signal)
DSNU_COL_STD = 3.0          # Std deviation of column offsets (pixel value units)
DSNU_ROW_STD = 1.0          # Std deviation of row offsets (pixel value units)
DSNU_PIXEL_STD = 2.0        # Std deviation of random pixel offsets (pixel value units)

# PRNU (Gain Noise - multiplies the signal, variation around 1.0)
PRNU_COL_STD = 0.005        # Std deviation of column gain variations (unitless)
PRNU_ROW_STD = 0.001        # Std deviation of row gain variations (unitless)
PRNU_PIXEL_STD = 0.002      # Std deviation of random pixel gain variations (unitless)

# --- Helper Functions ---

def add_gaussian_noise(image_float, std_dev):
    """Adds Gaussian noise to a float image."""
    if std_dev <= 0:
        return image_float
    gaussian_noise = np.random.normal(0.0, std_dev, image_float.shape).astype(image_float.dtype)
    noisy_image = image_float + gaussian_noise
    return noisy_image

def add_poisson_shot_noise(image_float, dark_current_rate, exposure_time, gain):
    """Adds Poisson shot noise related to dark current."""
    if dark_current_rate <= 0 or exposure_time <= 0 or gain <= 0:
        return image_float
    # Calculate average dark signal in electrons
    avg_dark_signal_electrons = dark_current_rate * exposure_time
    # Convert to pixel value units (ADU) - this is the mean of the Poisson noise *before* gain
    # The signal already INCLUDES gain effects from PRNU if applied.
    # We model the dark current added *before* read noise.
    # The Poisson noise arises from the electron count.
    # Let's calculate dark electrons, add poisson noise in electrons, then convert to ADU
    dark_electrons_map = np.full(image_float.shape, avg_dark_signal_electrons, dtype=np.float32)
    dark_shot_noise_electrons = np.random.poisson(dark_electrons_map).astype(np.float32) - dark_electrons_map
    # Convert noise in electrons to noise in ADU
    dark_shot_noise_adu = dark_shot_noise_electrons / gain
    # Add this noise to the image (which is already in ADU)
    noisy_image = image_float + dark_shot_noise_adu
    return noisy_image


def simulate_fpn(image_shape, dsnucol_std, dsnupix_std, prnucol_std, prnupix_std, dsnrow_std, prnurow_std):
    """
    Simulates a simplified FPN pattern (DSNU offset map and PRNU gain map).
    Includes column, row, and random pixel components.
    """
    H, W = image_shape
    # --- DSNU (Offset Map) ---
    dsnu_map = np.zeros(image_shape, dtype=np.float32)
    # Columnar DSNU
    if dsnucol_std > 0:
        col_noise = np.random.normal(0.0, dsnucol_std, (1, W))
        dsnu_map += col_noise # Broadcast across rows
    # Row DSNU
    if dsnrow_std > 0:
        row_noise = np.random.normal(0.0, dsnrow_std, (H, 1))
        dsnu_map += row_noise # Broadcast across columns
    # Pixel DSNU
    if dsnupix_std > 0:
        dsnu_map += np.random.normal(0.0, dsnupix_std, image_shape)

    # --- PRNU (Gain Map) ---
    # Start with a base gain of 1.0
    prnu_map = np.ones(image_shape, dtype=np.float32)
    # Columnar PRNU
    if prnucol_std > 0:
        col_gain_noise = np.random.normal(0.0, prnucol_std, (1, W))
        prnu_map += col_gain_noise # Add variation around 1.0
    # Row PRNU
    if prnurow_std > 0:
        row_gain_noise = np.random.normal(0.0, prnurow_std, (H, 1))
        prnu_map += row_gain_noise # Add variation around 1.0
     # Pixel PRNU
    if prnupix_std > 0:
        prnu_map += np.random.normal(0.0, prnupix_std, image_shape) # Add variation around 1.0

    # Ensure gain is not negative (though very unlikely with small std devs)
    prnu_map = np.maximum(prnu_map, 0.01)

    return dsnu_map, prnu_map

# --- Main Script ---
print("Starting noisy dataset generation...")
print(f"IMPORTANT: Noise parameters are currently set to placeholder values.")
print(f"Please adjust GAUSSIAN_READ_NOISE_STD, dark current params, FPN params, etc., at the top of the script.")

# Generate FPN maps once using the shape of the first available GT image
first_img_path = None
H, W = -1, -1
for set_name_find in SETS:
    gt_dir_find = os.path.join(BASE_THERMAL_DIR, set_name_find, 'GT')
    if os.path.exists(gt_dir_find):
        try:
            image_files_find = [f for f in os.listdir(gt_dir_find) if f.lower().endswith('.bmp')]
            if image_files_find:
                first_img_path = os.path.join(gt_dir_find, image_files_find[0])
                break
        except FileNotFoundError:
            continue # No files in directory?
if first_img_path:
    try:
        # Read grayscale
        sample_img = cv2.imread(first_img_path, cv2.IMREAD_GRAYSCALE)
        if sample_img is not None:
            H, W = sample_img.shape
            ORIGINAL_DTYPE = sample_img.dtype # Store original dtype
            print(f"Detected image shape: {H}x{W}, dtype: {ORIGINAL_DTYPE}")
            # Use detected shape to generate FPN maps
            dsnu_map, prnu_map = simulate_fpn((H, W), DSNU_COL_STD, DSNU_PIXEL_STD, PRNU_COL_STD, PRNU_PIXEL_STD, DSNU_ROW_STD, PRNU_ROW_STD)
            print("Generated shared FPN maps (will be resized if needed).")
        else:
            print(f"Error: Could not read sample image to get shape: {first_img_path}")
            exit()
    except Exception as e:
        print(f"Error reading sample image or generating FPN: {e}")
        exit()
else:
    print(f"Error: Could not find any .bmp GT images in {SETS} sets to determine shape.")
    exit()


# Process each set (train, val)
for set_name in SETS:
    gt_dir = os.path.join(BASE_THERMAL_DIR, set_name, 'GT')
    output_noisy_dir = os.path.join(OUTPUT_DIR, set_name, 'Noisy')
    output_gt_dir = os.path.join(OUTPUT_DIR, set_name, 'GT') # Also copy GT

    if not os.path.exists(gt_dir):
        print(f"Warning: GT directory not found for set '{set_name}', skipping: {gt_dir}")
        continue

    os.makedirs(output_noisy_dir, exist_ok=True)
    os.makedirs(output_gt_dir, exist_ok=True)

    print(f"\nProcessing set: {set_name}")
    try:
        image_files = [f for f in os.listdir(gt_dir) if f.lower().endswith('.bmp')]
        if not image_files:
            print(f"Warning: No .bmp files found in {gt_dir}")
            continue
    except FileNotFoundError:
        print(f"Warning: GT directory listed but not accessible: {gt_dir}")
        continue

    total_files = len(image_files)
    count = 0
    for filename in image_files:
        try:
            gt_path = os.path.join(gt_dir, filename)
            output_noisy_path = os.path.join(output_noisy_dir, filename)
            output_gt_path = os.path.join(output_gt_dir, filename) # Path to copy GT

            # Load clean image as grayscale
            clean_img = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
            if clean_img is None:
                print(f"  Warning: Could not read image {filename}, skipping.")
                continue

            # --- Apply Noise ---
            # 0. Ensure FPN maps match current image dimensions if needed
            current_H, current_W = clean_img.shape
            if current_H != H or current_W != W:
                 # print(f"  Info: Resizing FPN maps for {filename} (Shape: {current_H}x{current_W}).")
                 current_dsnu = cv2.resize(dsnu_map, (current_W, current_H), interpolation=cv2.INTER_NEAREST)
                 current_prnu = cv2.resize(prnu_map, (current_W, current_H), interpolation=cv2.INTER_LINEAR) # Linear better for gain map
            else:
                 current_dsnu = dsnu_map
                 current_prnu = prnu_map

            # 1. Convert to float32 for processing (normalize if desired, but simpler to work in original range first)
            img_float = clean_img.astype(np.float32)

            # 2. Apply FPN
            # Apply DSNU (offset) - add before gain
            img_fpn_offset = img_float + current_dsnu
            # Apply PRNU (gain) - multiply signal (incl DSNU component)
            img_fpn_gain = img_fpn_offset * current_prnu

            # 3. Simulate Dark Current Shot Noise (Poisson) - applied after FPN gain
            img_dark_noisy = add_poisson_shot_noise(img_fpn_gain, AVG_DARK_CURRENT_RATE, EXPOSURE_TIME_SEC, SENSOR_GAIN)

            # 4. Simulate Read Noise (Gaussian) - applied last
            noisy_img_float = add_gaussian_noise(img_dark_noisy, GAUSSIAN_READ_NOISE_STD)

            # 5. Clip and convert back to original type
            # Determine max value based on dtype (e.g., 255 for uint8, 65535 for uint16)
            if ORIGINAL_DTYPE == np.uint8:
                max_val = 255
            elif ORIGINAL_DTYPE == np.uint16:
                max_val = 65535
            else:
                print(f"  Warning: Unexpected image dtype {ORIGINAL_DTYPE}. Assuming uint8 range (0-255).")
                max_val = 255 # Default assumption
                ORIGINAL_DTYPE = np.uint8 # Force to uint8

            noisy_img_final = np.clip(noisy_img_float, 0, max_val).astype(ORIGINAL_DTYPE)

            # --- Save Noisy Image and Copy GT ---
            save_success = cv2.imwrite(output_noisy_path, noisy_img_final)
            if not save_success:
                print(f"  Warning: Failed to write noisy image {output_noisy_path}")
                continue # Skip copying GT if noisy failed

            # Copy the original GT image to the new structure for easy pairing
            shutil.copy2(gt_path, output_gt_path) # copy2 preserves metadata

            count += 1
            if count % 100 == 0 or count == total_files:
                print(f"  Processed {count}/{total_files} images...")

        except Exception as e:
            print(f"  Error processing {filename}: {e}")

    print(f"Finished set: {set_name}. Processed {count} images.")

print("\nNoisy dataset generation complete.")
print(f"Output saved to: {OUTPUT_DIR}")
print("Remember to verify the generated noisy images and adjust parameters for realism.")