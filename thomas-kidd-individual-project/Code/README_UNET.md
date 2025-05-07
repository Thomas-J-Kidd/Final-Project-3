# U-Net Model Implementation for ThermalDenoising

This project adds a PyTorch U-Net model implementation to the NAF-DPM ThermalDenoising framework. The implementation allows seamless switching between the original NAFNet architecture and the new U-Net architecture.

## Overview

The implementation includes:

1. Core model files:
   - `UNET.py` - Basic PyTorch U-Net implementation
   - `ConditionalUNET.py` - Time-conditioned U-Net for diffusion models
   - `UnetDPM.py` - Wrapper that supports both NAFNet and U-Net architectures

2. Training and testing modules:
   - `trainer_unet.py` - Trainer for U-Net models
   - `tester_unet.py` - Tester for U-Net models

3. Configuration:
   - `conf_unet.yml` - Example configuration for U-Net models

## Usage

### Running with U-Net model

```bash
python main.py --config ThermalDenoising/conf_unet.yml
```

### Training mode

To train the model, set `MODE: 1` in the configuration file.

### Testing mode

To test the model, set `MODE: 0` in the configuration file.

## Configuration

The key parameter that controls which model architecture to use is `MODEL_TYPE`:

```yaml
MODEL_TYPE: 'UNET'  # or 'NAFNET' for the original model
```

Other important parameters for the U-Net model:

```yaml
MODEL_CHANNELS: 32  # Base channel count (doubled internally for U-Net)
```

## Model Architecture

The U-Net implementation follows the classic U-Net architecture with:

- 4 encoder blocks with double convolutions
- A bottleneck layer
- 4 decoder blocks with skip connections
- Sigmoid activation for the output layer

The conditional version adds time embeddings using sinusoidal position embeddings, making it compatible with the diffusion model framework.

## Performance

The U-Net model provides a different architectural approach compared to NAFNet:

- U-Net has a simpler design with standard convolutions
- U-Net may require fewer parameters for similar performance
- Performance comparisons should be conducted on the specific dataset

## Credits

The U-Net implementation is based on the architecture from the BSD notebooks, converted from TensorFlow/Keras to PyTorch and adapted to work within the diffusion model framework.
