#!/usr/bin/env python3
import os
import yaml
import sys

def create_nafnet_experiment():
    """Create an experiment with variations of NAFNET configuration"""
    # Base configuration
    base_config = {
        "MODEL_TYPE": "NAFNET",
        "IMAGE_SIZE": [256, 256],
        "CHANNEL_X": 3,
        "CHANNEL_Y": 3,
        "TIMESTEPS": 100,
        "SCHEDULE": "linear",
        "MODEL_CHANNELS": 32,
        "NUM_RESBLOCKS": 1,
        "CHANNEL_MULT": [1, 2, 3, 4],
        "NUM_HEADS": 1,
        "MIDDLE_BLOCKS": 1,
        "ENC_BLOCKS": [1, 1, 1, 1],
        "DEC_BLOCKS": [1, 1, 1, 1],
        "MODE": 1,
        "PRE_ORI": "True",
        "TASK": "GoProDeblurring",
        
        # Training parameters
        "PATH_GT": "./dataset/GoPro/train",
        "PATH_IMG": "./dataset/GoPro/train",
        "BATCH_SIZE": 4,
        "NUM_WORKERS": 4,
        "ITERATION_MAX": 20000,
        "LR": 0.0001,
        "LOSS": "L1",
        "EMA_EVERY": 100,
        "START_EMA": 2000,
        "SAVE_MODEL_EVERY": 1000,
        "EMA": "True",
        "CONTINUE_TRAINING": "False",
        "CONTINUE_TRAINING_STEPS": 10000,
        "PRETRAINED_PATH_INITIAL_PREDICTOR": "",
        "PRETRAINED_PATH_DENOISER": "",
        "WEIGHT_SAVE_PATH": "./ThermalDenoising/weights_gopro_nafnet_rgb",
        "TRAINING_PATH": "./ThermalDenoising/training_gopro_nafnet_rgb",
        "BETA_LOSS": 10,
        "HIGH_LOW_FREQ": "False",
        "VALIDATE_EVERY": 1000,
        "VALIDATE_ITERATIONS": 20,
        "WANDB": "False",
        "PROJECT": "NAFDPM_GoProDeblurring",
        "GRAYSCALE": "False",
        "USE_GAMMA": "False",
        
        # Test parameters
        "NATIVE_RESOLUTION": "False",
        "DPM_SOLVER": "True",
        "DPM_STEP": 5,
        "BATCH_SIZE_VAL": 1,
        "TEST_PATH_GT": "./dataset/GoPro/test",
        "TEST_PATH_IMG": "./dataset/GoPro/test",
        "TEST_INITIAL_PREDICTOR_WEIGHT_PATH": "./ThermalDenoising/weights_gopro_nafnet_rgb/BEST_PSNR_model_init.pth",
        "TEST_DENOISER_WEIGHT_PATH": "./ThermalDenoising/weights_gopro_nafnet_rgb/BEST_PSNR_model_denoiser.pth",
        "TEST_IMG_SAVE_PATH": "./ThermalDenoising/results_gopro_nafnet_rgb",
        "LOGGER_PATH": "./ThermalDenoising/logs",
        
        # Metrics
        "PSNR": "True",
        "SSIM": "True",
        "FMETRIC": "False",
        "PFMETRIC": "False",
        "DRD": "False",
        "LPIPS": "True",
        "DISTS": "True"
    }
    
    # Create variations directory
    output_dir = "experiment_configs"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save base configuration
    base_config_path = os.path.join(output_dir, "nafnet_base.yml")
    with open(base_config_path, 'w') as f:
        yaml.dump(base_config, f, default_flow_style=False)
    
    print(f"Base configuration saved to: {base_config_path}")
    
    # Define variations to experiment with
    variations = [
        # Learning rate variations
        {"LR": 0.00005, "BATCH_SIZE": 4},
        {"LR": 0.0001, "BATCH_SIZE": 4},
        {"LR": 0.0002, "BATCH_SIZE": 4},
        
        # Batch size variations
        {"LR": 0.0001, "BATCH_SIZE": 2},
        {"LR": 0.0001, "BATCH_SIZE": 8},
        
        # Model architecture variations
        {"MODEL_CHANNELS": 16, "LR": 0.0001, "BATCH_SIZE": 4},
        {"MODEL_CHANNELS": 64, "LR": 0.0001, "BATCH_SIZE": 4},
        
        # ResBlock variations
        {"NUM_RESBLOCKS": 2, "LR": 0.0001, "BATCH_SIZE": 4},
        {"NUM_RESBLOCKS": 3, "LR": 0.0001, "BATCH_SIZE": 4},
        
        # Loss function variations
        {"LOSS": "L2", "LR": 0.0001, "BATCH_SIZE": 4},
        
        # Combined variations
        {"LR": 0.0002, "BATCH_SIZE": 8, "MODEL_CHANNELS": 64},
    ]
    
    # Create and save variations
    for i, variation in enumerate(variations):
        # Apply variations to base config
        config = dict(base_config)
        for key, value in variation.items():
            config[key] = value
        
        # Generate descriptive filename
        var_desc = '_'.join([f"{k}_{v}" for k, v in variation.items()])
        config_path = os.path.join(output_dir, f"nafnet_{var_desc}.yml")
        
        # Save configuration
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        print(f"Variation {i+1} saved to: {config_path}")

if __name__ == "__main__":
    create_nafnet_experiment()
    print("\nTo use these configurations with the AWS training manager:")
    print("1. First set up AWS: python aws_train_manager.py setup")
    print("2. Queue a job: python aws_train_manager.py queue --config experiment_configs/nafnet_base.yml")
    print("3. Process the queue: python aws_train_manager.py process")
    print("4. Check status: python aws_train_manager.py status")
    print("5. Retrieve results: python aws_train_manager.py retrieve --config experiment_configs/nafnet_base.yml")
