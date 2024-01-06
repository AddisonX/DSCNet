# -*- coding: utf-8 -*-
import os
import torch
import numpy as np
from torch import nn
import warnings

warnings.filterwarnings("ignore")

"""
This code is mainly the deformation process of our DSConv
"""


class StandardConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, device):
        super(StandardConv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, padding=(kernel_size // 2))
        self.gn = nn.GroupNorm(out_ch // 4, out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.device = device

    def forward(self, f):
        x = self.conv(f)
        x = self.gn(x)
        x = self.relu(x)
        return x