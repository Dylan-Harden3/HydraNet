import torch.nn as nn
import torch
import torch.nn.functional as F


# TODO add depthwise separable convolutions
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class ConvTransposeBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(ConvTransposeBlock, self).__init__()
        self.conv = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class FastNormalizedFusion(nn.Module):
    def __init__(self, n_inputs):
        super(FastNormalizedFusion, self).__init__()
        self.w = nn.Parameter(torch.ones(n_inputs))

    def forward(self, *x):
        weights = F.relu(self.w)
        weights_norm = weights / (weights.sum() + 1e-4)
        return F.relu(sum([i * w for i, w in zip(x, weights_norm)]))
