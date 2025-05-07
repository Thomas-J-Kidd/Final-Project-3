from config import load_config
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='conf.yml', help='path to the config.yaml file')
    args = parser.parse_args()
    config = load_config(args.config)
    print('Config loaded')
    mode = config.MODE
    task = config.TASK
    
    if task == "ThermalDenoising":
        # Check if MODEL_TYPE is specified and use appropriate modules
        if hasattr(config, 'MODEL_TYPE') and config.MODEL_TYPE.upper() == 'UNET':
            print(f"Using UNet models for {task}")
            from ThermalDenoising.src.trainer_unet import Trainer
            from ThermalDenoising.src.tester_unet import Tester
        else:
            # Default to NAFNet implementation
            print(f"Using NAFNet models for {task}")
            from ThermalDenoising.src.trainer import Trainer
            from ThermalDenoising.src.tester import Tester
    elif task == "Deblurring":
        from Deblurring.src.trainer import Trainer
        from Deblurring.src.tester import Tester
        from Deblurring.src.finetune import Finetune
    else:  # Default to Binarization
        from Binarization.src.trainer import Trainer
        from Binarization.src.tester import Tester

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

    else:  # Finetuning mode
        if task != "Deblurring":
            print("Finetuning is only supported for Deblurring task")
            return
            
        print("--------------------------")
        print('Start Finetuning')
        print("--------------------------")

        finetuner = Finetune(config)
        finetuner.finetune()

        print("--------------------------")
        print('Finetuning complete')
        print("--------------------------")


if __name__ == "__main__":
    main()
