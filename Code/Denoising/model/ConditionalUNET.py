import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
import math 
from typing import Optional, Tuple, Union, List
import numpy as np
from ThermalDenoising.model.local_arch import Local_Base

# Sinusoidal positional embeddings - same as in ConditionalNAFNET
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


# Simple gating mechanism
class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


# Time-conditioned double convolution block for U-Net
class ConditionalDoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim=None):
        super().__init__()
        self.mlp = nn.Sequential(
            SimpleGate(), nn.Linear(time_emb_dim // 2, out_channels * 4)
        ) if time_emb_dim else None

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.activation = nn.ReLU(inplace=True)
        
        # Scale and shift parameters for feature-wise linear modulation
        self.beta = nn.Parameter(torch.zeros((1, out_channels, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, out_channels, 1, 1)), requires_grad=True)

    def time_forward(self, time, mlp):
        time_emb = mlp(time)
        time_emb = rearrange(time_emb, 'b c -> b c 1 1')
        return time_emb.chunk(4, dim=1)

    def forward(self, x, time):
        if self.mlp is not None:
            shift_conv1, scale_conv1, shift_conv2, scale_conv2 = self.time_forward(time, self.mlp)
        else:
            shift_conv1, scale_conv1, shift_conv2, scale_conv2 = 0, 0, 0, 0
            
        # First convolution with time conditioning
        x = self.conv1(x)
        x = x * (scale_conv1 + 1) + shift_conv1
        x = self.activation(x)
        
        # Second convolution with time conditioning
        x = self.conv2(x)
        x = x * (scale_conv2 + 1) + shift_conv2
        x = self.activation(x)
        
        return x


class ConditionalUNet(nn.Module):
    def __init__(self, inp_channel=2, out_channel=1, width=64):
        """
        Time-conditional U-Net implementation for diffusion models
        
        Args:
            inp_channel: Number of input channels (2 for concatenated input and condition)
            out_channel: Number of output channels
            width: Base feature size
        """
        super(ConditionalUNet, self).__init__()
        
        # Time embedding
        fourier_dim = width
        sinu_pos_emb = SinusoidalPosEmb(fourier_dim)
        time_dim = width * 4
        
        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim*2),
            SimpleGate(),
            nn.Linear(time_dim, time_dim)
        )
        
        # Encoder
        self.enc1 = ConditionalDoubleConv(inp_channel, width, time_dim)
        self.pool1 = nn.MaxPool2d(2)
        
        self.enc2 = ConditionalDoubleConv(width, width * 2, time_dim)
        self.pool2 = nn.MaxPool2d(2)
        
        self.enc3 = ConditionalDoubleConv(width * 2, width * 4, time_dim)
        self.pool3 = nn.MaxPool2d(2)
        
        self.enc4 = ConditionalDoubleConv(width * 4, width * 8, time_dim)
        self.pool4 = nn.MaxPool2d(2)
        
        # Bottleneck
        self.bottleneck = ConditionalDoubleConv(width * 8, width * 16, time_dim)
        
        # Decoder
        self.upconv4 = nn.ConvTranspose2d(width * 16, width * 8, kernel_size=2, stride=2)
        self.dec4 = ConditionalDoubleConv(width * 16, width * 8, time_dim)  # *16 because of concat
        
        self.upconv3 = nn.ConvTranspose2d(width * 8, width * 4, kernel_size=2, stride=2)
        self.dec3 = ConditionalDoubleConv(width * 8, width * 4, time_dim)  # *8 because of concat
        
        self.upconv2 = nn.ConvTranspose2d(width * 4, width * 2, kernel_size=2, stride=2)
        self.dec2 = ConditionalDoubleConv(width * 4, width * 2, time_dim)  # *4 because of concat
        
        self.upconv1 = nn.ConvTranspose2d(width * 2, width, kernel_size=2, stride=2)
        self.dec1 = ConditionalDoubleConv(width * 2, width, time_dim)  # *2 because of concat
        
        # Final output layer
        self.outconv = nn.Conv2d(width, out_channel, kernel_size=1)
        self.final_activation = nn.Sigmoid()
        
        # For proper padding
        self.padder_size = 16  # 2^4 since we have 4 max-pooling layers

    def forward(self, inp, time, cond):
        """
        Forward pass through the Conditional U-Net
        
        Args:
            inp: Input noisy residual
            time: Timestep
            cond: Conditioning signal (initial prediction)
            
        Returns:
            x: Predicted clean residual
        """
        # Handle time embeddings
        if isinstance(time, int) or isinstance(time, float):
            time = torch.tensor([time]).to(inp.device)
            
        # Concatenate input with conditioning
        x = torch.cat([inp, cond], dim=1)
        
        # Time embedding
        t = self.time_mlp(time)
        
        B, C, H, W = x.shape
        x = self.check_image_size(x)
        
        # Encoder
        enc1_out = self.enc1(x, t)
        x = self.pool1(enc1_out)
        
        enc2_out = self.enc2(x, t)
        x = self.pool2(enc2_out)
        
        enc3_out = self.enc3(x, t)
        x = self.pool3(enc3_out)
        
        enc4_out = self.enc4(x, t)
        x = self.pool4(enc4_out)
        
        # Bottleneck
        x = self.bottleneck(x, t)
        
        # Decoder with skip connections
        x = self.upconv4(x)
        x = torch.cat([x, enc4_out], dim=1)
        x = self.dec4(x, t)
        
        x = self.upconv3(x)
        x = torch.cat([x, enc3_out], dim=1)
        x = self.dec3(x, t)
        
        x = self.upconv2(x)
        x = torch.cat([x, enc2_out], dim=1)
        x = self.dec2(x, t)
        
        x = self.upconv1(x)
        x = torch.cat([x, enc1_out], dim=1)
        x = self.dec1(x, t)
        
        # Final output
        x = self.outconv(x)
        x = self.final_activation(x)
        
        # Return only the part corresponding to the original input size
        return x[:, :, :H, :W]
    
    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x


class ConditionalUNetLocal(Local_Base, ConditionalUNet):
    def __init__(self, *args, train_size=(1, 1, 256, 256), fast_imp=False, **kwargs):
        Local_Base.__init__(self)
        ConditionalUNet.__init__(self, *args, **kwargs)

        N, C, H, W = train_size
        base_size = (int(H * 1.5), int(W * 1.5))

        self.eval()
        with torch.no_grad():
            self.convert(base_size=base_size, train_size=train_size, fast_imp=fast_imp)
