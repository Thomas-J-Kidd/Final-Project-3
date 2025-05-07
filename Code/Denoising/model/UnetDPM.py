import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
import math 
from typing import Optional, Tuple, Union, List
import numpy as np
from ThermalDenoising.model.ConditionalNAFNET import ConditionalNAFNet, ConditionalNAFNetLocal
from ThermalDenoising.model.NAFNET import NAFNet, NAFNetLocal
from ThermalDenoising.model.UNET import UNet, UNetLocal
from ThermalDenoising.model.ConditionalUNET import ConditionalUNet, ConditionalUNetLocal
from torchvision.transforms.functional import rgb_to_grayscale

# UnetDPM for ThermalDenoising
class UnetDPM(nn.Module):
    def __init__(self, input_channels: int = 1, output_channels: int = 1, n_channels: int = 32,
                 middle_blk_num: int = 1, enc_blk_nums=[1, 1, 1, 28], dec_blk_nums=[1, 1, 1, 1], 
                 mode=1, model_type="UNET"):
        super(UnetDPM, self).__init__()
        
        self.model_type = model_type.upper()
        
        # Select model architecture based on model_type
        if self.model_type == "NAFNET":
            # Mode Test
            if mode == 0: 
                self.denoiser = ConditionalNAFNetLocal(inp_channel=output_channels*2, 
                                               out_channel=output_channels,
                                               width=n_channels, 
                                               middle_blk_num=middle_blk_num, 
                                               enc_blk_nums=enc_blk_nums, 
                                               dec_blk_nums=dec_blk_nums, 
                                               upscale=1)
            # Mode Train
            else: 
                self.denoiser = ConditionalNAFNet(inp_channel=output_channels*2, 
                                               out_channel=output_channels,
                                               width=n_channels, 
                                               middle_blk_num=middle_blk_num, 
                                               enc_blk_nums=enc_blk_nums, 
                                               dec_blk_nums=dec_blk_nums, 
                                               upscale=1)
            # Mode Test
            if mode == 0:
                self.init_predictor = NAFNetLocal(inp_channel=input_channels, 
                                               out_channel=output_channels,
                                               width=n_channels, 
                                               middle_blk_num=middle_blk_num, 
                                               enc_blk_nums=enc_blk_nums, 
                                               dec_blk_nums=dec_blk_nums)
            # Mode Train
            else: 
                self.init_predictor = NAFNet(inp_channel=input_channels, 
                                               out_channel=output_channels,
                                               width=n_channels, 
                                               middle_blk_num=middle_blk_num, 
                                               enc_blk_nums=enc_blk_nums, 
                                               dec_blk_nums=dec_blk_nums)
        
        elif self.model_type == "UNET":
            # Use a wider base channel count for U-Net
            unet_width = n_channels * 2
            
            # Mode Test
            if mode == 0: 
                self.denoiser = ConditionalUNetLocal(inp_channel=output_channels*2, 
                                              out_channel=output_channels,
                                              width=unet_width)
            # Mode Train
            else: 
                self.denoiser = ConditionalUNet(inp_channel=output_channels*2, 
                                          out_channel=output_channels,
                                          width=unet_width)
            # Mode Test
            if mode == 0:
                self.init_predictor = UNetLocal(inp_channel=input_channels, 
                                          out_channel=output_channels,
                                          width=unet_width)
            # Mode Train
            else: 
                self.init_predictor = UNet(inp_channel=input_channels, 
                                      out_channel=output_channels,
                                      width=unet_width)
        else:
            raise ValueError(f"Unknown model type: {model_type}. Expected 'NAFNET' or 'UNET'")
            

    def forward(self, x, condition, t, diffusion):
        """
        Forward pass through the DPM model
        
        Args:
            x: Clean target image (GT)
            condition: Noisy input image
            t: Timestep
            diffusion: Diffusion model
            
        Returns:
            x_: Initial prediction from the noisy input
            x__: Predicted residual/noise
            noisy_image: Noisy version of the residual
            noise_ref: Reference noise
        """
        # Initial prediction from the noisy input
        x_ = self.init_predictor(condition)
        
        # Calculate residual between clean target and initial prediction
        residual = x - x_
        
        # Add noise to the residual based on timestep t
        noisy_image, noise_ref = diffusion.noisy_image(t, residual)
        
        # Predict the residual/noise using the denoiser network
        if self.model_type == "NAFNET":
            x__ = self.denoiser(inp=noisy_image, cond=x_.clone().detach(), time=t)
        else:  # UNET
            x__ = self.denoiser(inp=noisy_image, time=t, cond=x_.clone().detach())
        
        return x_, x__, noisy_image, noise_ref


class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


def get_pad(in_,  ksize, stride, atrous=1):
    out_ = np.ceil(float(in_)/stride)
    return int(((out_ - 1) * stride + atrous*(ksize-1) + 1 - in_)/2)


if __name__ == '__main__':
    from ThermalDenoising.src.config import load_config
    import argparse
    from ThermalDenoising.schedule.diffusionSample import GaussianDiffusion
    from ThermalDenoising.schedule.schedule import Schedule
    import torchsummary
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='../conf.yml', help='path to the config.yaml file')
    args = parser.parse_args()
    config = load_config(args.config)
    print('Config loaded')
    
    # Get model type from config or default to NAFNET
    model_type = getattr(config, 'MODEL_TYPE', 'NAFNET')
    
    model = UnetDPM(input_channels=config.CHANNEL_X,
            output_channels=config.CHANNEL_Y,
            n_channels=config.MODEL_CHANNELS,
            middle_blk_num=config.MIDDLE_BLOCKS, 
            enc_blk_nums=config.ENC_BLOCKS, 
            dec_blk_nums=config.DEC_BLOCKS,
            model_type=model_type)
    
    schedule = Schedule(config.SCHEDULE, config.TIMESTEPS)
    diffusion = GaussianDiffusion(model, config.TIMESTEPS, schedule)
    model.eval()
    print(torchsummary.summary(model.cuda(), [(1, 128, 128)], batch_size=32))
