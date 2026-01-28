from torch import nn
import torch.nn.functional as F
import torch
import numpy as np

class MyVGG(nn.Module):
    """自主修改的resnet模型架构"""
    def __init__(self, config):
        super(MyVGG, self).__init__()
        self.step = config.step
        self.mul_len = config.mul_len
        self.fc1 = nn.Sequential(
            nn.Linear(self.mul_len, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        if self.step == "mul":
            x = self.fc1(x)
            return x

