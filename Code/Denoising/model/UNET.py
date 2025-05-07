import torch
import torch.nn as nn
import torch.nn.functional as F
from ThermalDenoising.model.local_arch import Local_Base

class DoubleConv(nn.Module):
    """Double convolution block for U-Net"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class UNet(nn.Module):
    def __init__(self, inp_channel=1, out_channel=1, width=64):
        """
        PyTorch implementation of the U-Net model from the Best_BSD.ipynb
        
        Args:
            inp_channel: Number of input channels
            out_channel: Number of output channels
            width: Base feature size (corresponds to features parameter)
        """
        super(UNet, self).__init__()
        
        # Encoder
        self.enc1 = DoubleConv(inp_channel, width)
        self.pool1 = nn.MaxPool2d(2)
        
        self.enc2 = DoubleConv(width, width * 2)
        self.pool2 = nn.MaxPool2d(2)
        
        self.enc3 = DoubleConv(width * 2, width * 4)
        self.pool3 = nn.MaxPool2d(2)
        
        self.enc4 = DoubleConv(width * 4, width * 8)
        self.pool4 = nn.MaxPool2d(2)
        
        # Bottleneck
        self.bottleneck = DoubleConv(width * 8, width * 16)
        
        # Decoder
        self.upconv4 = nn.ConvTranspose2d(width * 16, width * 8, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(width * 16, width * 8)  # *16 because of concat
        
        self.upconv3 = nn.ConvTranspose2d(width * 8, width * 4, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(width * 8, width * 4)  # *8 because of concat
        
        self.upconv2 = nn.ConvTranspose2d(width * 4, width * 2, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(width * 4, width * 2)  # *4 because of concat
        
        self.upconv1 = nn.ConvTranspose2d(width * 2, width, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(width * 2, width)  # *2 because of concat
        
        # Final output layer
        self.outconv = nn.Conv2d(width, out_channel, kernel_size=1)
        self.final_activation = nn.Sigmoid()
        
        # For proper padding
        self.padder_size = 16  # 2^4 since we have 4 max-pooling layers

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.check_image_size(x)
        
        # Encoder
        enc1 = self.enc1(x)
        x = self.pool1(enc1)
        
        enc2 = self.enc2(x)
        x = self.pool2(enc2)
        
        enc3 = self.enc3(x)
        x = self.pool3(enc3)
        
        enc4 = self.enc4(x)
        x = self.pool4(enc4)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder with skip connections
        x = self.upconv4(x)
        x = torch.cat([x, enc4], dim=1)
        x = self.dec4(x)
        
        x = self.upconv3(x)
        x = torch.cat([x, enc3], dim=1)
        x = self.dec3(x)
        
        x = self.upconv2(x)
        x = torch.cat([x, enc2], dim=1)
        x = self.dec2(x)
        
        x = self.upconv1(x)
        x = torch.cat([x, enc1], dim=1)
        x = self.dec1(x)
        
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

class UNetLocal(Local_Base, UNet):
    def __init__(self, *args, train_size=(1, 1, 256, 256), fast_imp=False, **kwargs):
        Local_Base.__init__(self)
        UNet.__init__(self, *args, **kwargs)

        N, C, H, W = train_size
        base_size = (int(H * 1.5), int(W * 1.5))

        self.eval()
        with torch.no_grad():
            self.convert(base_size=base_size, train_size=train_size, fast_imp=fast_imp)
