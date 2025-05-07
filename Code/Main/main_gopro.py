from config import load_config
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='conf_gopro_nafnet_rgb.yml', help='path to the config.yaml file')
    args = parser.parse_args()
    config = load_config(args.config)
    print('Config loaded')
    mode = config.MODE
    task = config.TASK
    
    if task == "GoProDeblurring":
        # Check if MODEL_TYPE is specified and use appropriate modules
        if hasattr(config, 'MODEL_TYPE') and config.MODEL_TYPE.upper() == 'UNET':
            print(f"Using UNet models for {task}")
            from ThermalDenoising.src.trainer_gopro import Trainer
            from ThermalDenoising.src.tester_gopro import Tester
        else:
            # Default to NAFNet implementation
            print(f"Using NAFNet models for {task}")
            from ThermalDenoising.src.trainer_gopro import Trainer
            from ThermalDenoising.src.tester_gopro import Tester
    else:
        # Fall back to original implementation
        if hasattr(config, 'MODEL_TYPE') and config.MODEL_TYPE.upper() == 'UNET':
            print(f"Using UNet models for {task}")
            from ThermalDenoising.src.trainer_unet import Trainer
            from ThermalDenoising.src.tester_unet import Tester
        else:
            # Default to NAFNet implementation
            print(f"Using NAFNet models for {task}")
            from ThermalDenoising.src.trainer import Trainer
            from ThermalDenoising.src.tester import Tester

    if mode == 0:
        print("--------------------------")
        print('Start Testing')
        print("--------------------------")

        tester = Tester(config)
        tester.test()

        print("--------------------------")
        print('Testing complete')
        print("--------------------------")

    elif mode == 1:
        print("--------------------------")
        print('Start Training')
        print("--------------------------")

        trainer = Trainer(config)
        trainer.train()

        print("--------------------------")
        print('Training complete')
        print("--------------------------")


if __name__ == "__main__":
    main()
