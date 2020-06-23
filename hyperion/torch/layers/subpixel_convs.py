"""
 Copyright 2020 Johns Hopkins University  (Author: Jesus Villalba, Nanxin Chen)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
#from __future__ import absolute_import

import torch
import torch.nn as nn

class SubPixelConv1d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, 
                 dilation=1, groups=1, bias=True, padding_mode='zeros'):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, stride*out_channels, kernel_size, stride=1, 
                              padding=padding, dilation=dilation, 
                              groups=groups, bias=bias, padding_mode=padding_mode)
        
        self.out_channels = out_channels
        self.stride = stride

    def forward(self, x):
        x = self.conv(x)
        if self.stride == 1:
            return x

        x = x.view(-1, self.stride, self.out_channels, x.size(-1)).permute(
            0,2,3,1).reshape(-1, self.out_channels, x.size(-1)*self.stride)
        return x



class SubPixelConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, 
                 dilation=1, groups=1, bias=True, padding_mode='zeros'):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, (stride**2)*out_channels, kernel_size, stride=1, 
                              padding=padding, dilation=dilation, 
                              groups=groups, bias=bias, padding_mode=padding_mode)
        
        self.stride = stride
        if stride > 1:
            self.pixel_shuffle = nn.PixelShuffle(self.stride)


    def forward(self, x):
        x = self.conv(x)
        if self.stride == 1:
            return x

        return self.pixel_shuffle(x)



