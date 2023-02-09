import os
import torch
import torch.nn as nn
from torchvision.transforms import *
import torch.nn.functional as F

class freup_PeriodicPadding(nn.Module):
    def __init__(self, channels):
        super(freup_PeriodicPadding, self).__init__()        
        sequential = nn.Sequential
        conv = nn.Conv2d
        conv_config = {"in_channels":channels,"out_channels":channels,"kernel_size":1, "stride":1, "padding":0}

        self.amp_fuse = sequential(conv(**conv_config),
                                        nn.LeakyReLU(0.1,inplace=False),
                                        conv(**conv_config))
        self.pha_fuse = sequential(conv(**conv_config),
                                        nn.LeakyReLU(0.1,inplace=False),
                                        conv(**conv_config))
        self.post = conv(**conv_config)
    def forward(self, x):

        N, C, H, W = x.shape

        fft_x = torch.fft.rfft2(x)
        mag_x = torch.abs(fft_x)
        pha_x = torch.angle(fft_x)

        ### Calling torch.fft.fft2 directly will introduce unnecessary memory overhead
        ### So we use torch.fft.rfft2 instead, and perform zero padding directly in the spatial domain
        # Mag = self.amp_fuse(mag_x)
        # Pha = self.pha_fuse(pha_x)
        # amp_fuse = torch.tile(Mag, (2, 2))
        # pha_fuse = torch.tile(Pha, (2, 2))
        amp_fuse = self.amp_fuse(mag_x)
        pha_fuse = self.pha_fuse(pha_x)
        real = amp_fuse * torch.cos(pha_fuse)
        imag = amp_fuse * torch.sin(pha_fuse)
        out = torch.complex(real, imag)

        output = torch.fft.irfft2(out)
        # output = torch.abs(output) # unnecessary abs for irfft 
        ### zero padding in spatial domain
        output_pad = torch.zeros([N, C, 2*H, 2*W],device=x.device)
        output_pad[:,:,0:2*H:2,0:2*W:2] = output
        ### post at spatial domain
        output_pad = self.post(output_pad)
        return output_pad
