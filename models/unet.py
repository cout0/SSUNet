# -*- coding: utf-8 -*-
"""U-Net

author: Masahiro Hayashi

This script defines the model architecture of U-Net.
"""

import torch
import torch.nn as nn
from .utils.EBC import EBC



class DoubleConv(nn.Module):
    def __init__(self, snn_reset, in_channels, out_channels, use_EBC = True):
        super(DoubleConv, self).__init__()
        if use_EBC:
            self.conv1 = nn.Sequential(
                EBC([in_channels, out_channels], snn_reset=snn_reset),
            )
            self.conv2 = nn.Sequential(
                EBC([out_channels, out_channels], snn_reset=snn_reset),
            )
        else:
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
            )
            self.conv2 = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
            )
            

    def forward(self, x):
        x1 = self.conv1(x)
        x = self.conv2(x1)
        return x+x1

class UNet(nn.Module):
    def __init__(self, out_channels=2, T=5):
        super(UNet, self).__init__()
        self.name = 'UNet'
        self.snn_reset = True
        self.in_channels = 1
        self.out_channels = out_channels
        self.T = T
        
        self.conv1 = DoubleConv(self.snn_reset, self.in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = DoubleConv(self.snn_reset, 64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = DoubleConv(self.snn_reset, 128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = DoubleConv(self.snn_reset, 256, 512)
        self.pool4 = nn.MaxPool2d(2)
        self.conv5 = DoubleConv(self.snn_reset, 512, 1024)
        self.up6 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.conv6 = DoubleConv(self.snn_reset, 1024, 512)
        self.up7 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv7 = DoubleConv(self.snn_reset, 512, 256)
        self.up8 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv8 = DoubleConv(self.snn_reset, 256, 128)
        self.up9 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv9 = DoubleConv(self.snn_reset, 128, 64)
        self.conv10 = nn.Conv2d(64, self.out_channels, kernel_size=1)
    
    def forward(self, x):
        c1 = self.conv1(x)
        p1 = self.pool1(c1)
        c2 = self.conv2(p1)
        p2 = self.pool2(c2)
        c3 = self.conv3(p2)
        p3 = self.pool3(c3)
        c4 = self.conv4(p3)
        p4 = self.pool4(c4)
        c5 = self.conv5(p4)
        up_6 = self.up6(c5)
        merge6 = torch.cat([up_6, c4], dim=1)
        c6 = self.conv6(merge6)
        up_7 = self.up7(c6)
        merge7 = torch.cat([up_7, c3], dim=1)
        c7 = self.conv7(merge7)
        up_8 = self.up8(c7)
        merge8 = torch.cat([up_8, c2], dim=1)
        c8 = self.conv8(merge8)
        up_9 = self.up9(c8)
        merge9 = torch.cat([up_9, c1], dim=1)
        c9 = self.conv9(merge9)
        c10 = self.conv10(c9)
        out_spike_counter = c10
        return out_spike_counter


###############################################################################
# For testing
###############################################################################
if __name__ == "__main__":
    # A full forward pass
    im = torch.randn(1, 1, 512, 512)
    model = UNet(2)

    print(list(model.children()))
    import time
    t = time.time()
    x = model(im)
    print(time.time() - t)
    print(x.shape)
    del model
    del x
