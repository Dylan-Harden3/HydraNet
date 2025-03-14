import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torch


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


class FastNormalizedFusion(nn.Module):
    def __init__(self, n_inputs):
        super(FastNormalizedFusion, self).__init__()
        self.w = nn.Parameter(torch.ones(n_inputs))

    def forward(self, *x):
        weights = F.relu(self.w)
        weights_norm = weights / (weights.sum() + 1e-4)
        return F.relu(sum([i * w for i, w in zip(x, weights_norm)]))


# TODO BiFPNConfig or something?
class BiFPNBlock(nn.Module):
    def __init__(self, num_channels):
        super(BiFPNBlock, self).__init__()
        # TODO probably a better way to do this, fine for now though
        self.p34_td_fuse = FastNormalizedFusion(2)
        self.p34_td_conv = ConvBlock(num_channels, num_channels, 1)

        self.p23_td_fuse = FastNormalizedFusion(2)
        self.p23_td_conv = ConvBlock(num_channels, num_channels, 1)

        self.p12_td_fuse = FastNormalizedFusion(2)
        self.p12_td_conv = ConvBlock(num_channels, num_channels, 1)

        self.p12_bu_fuse = FastNormalizedFusion(3)
        self.p12_bu_conv = ConvBlock(num_channels, num_channels, 1)

        self.p23_bu_fuse = FastNormalizedFusion(3)
        self.p23_bu_conv = ConvBlock(num_channels, num_channels, 1)

        self.p4_bu_fuse = FastNormalizedFusion(2)
        self.p4_bu_conv = ConvBlock(num_channels, num_channels, 1)

    def forward(self, p1, p2, p3, p4):
        # top-down pathway
        p4_td = F.interpolate(p4, size=(13, 13), mode="bilinear", align_corners=False)
        p3_td_out = self.p34_td_conv(self.p34_td_fuse(p3, p4_td))

        p3_td = F.interpolate(
            p3_td_out, size=(25, 25), mode="bilinear", align_corners=False
        )
        p2_td_out = self.p23_td_conv(self.p23_td_fuse(p2, p3_td))

        p2_td = F.interpolate(
            p2_td_out, size=(50, 50), mode="bilinear", align_corners=False
        )
        p1_td_out = self.p12_td_conv(self.p12_td_fuse(p1, p2_td))

        # bottom-up pathway
        p1_bu = F.max_pool2d(p1_td_out, kernel_size=3, stride=2, padding=1)
        p2_bu_out = self.p12_bu_conv(self.p12_bu_fuse(p2, p2_td_out, p1_bu))

        p2_bu = F.max_pool2d(p2_bu_out, kernel_size=3, stride=2, padding=1)
        p3_bu_out = self.p23_bu_conv(self.p23_bu_fuse(p3, p3_td_out, p2_bu))

        p4_bu = F.max_pool2d(p3_bu_out, kernel_size=3, stride=2, padding=1)
        p4_bu_out = self.p4_bu_conv(self.p4_bu_fuse(p4, p4_bu))

        return p1_td_out, p2_bu_out, p3_bu_out, p4_bu_out


class BiFPN(nn.Module):
    def __init__(self, size, feature_size):
        super(BiFPN, self).__init__()
        self.p1_proj = ConvBlock(
            in_channels=size[0], out_channels=feature_size, kernel_size=1
        )  # B,48,50,50  -> B,feature_size,50,50
        self.p2_proj = ConvBlock(
            in_channels=size[1], out_channels=feature_size, kernel_size=1
        )  # B,104,25,25 -> B,feature_size,25,25
        self.p3_proj = ConvBlock(
            in_channels=size[2], out_channels=feature_size, kernel_size=1
        )  # B,208,13,13 -> B,feature_size,13,13
        self.p4_proj = ConvBlock(
            in_channels=size[3], out_channels=feature_size, kernel_size=1
        )  # B,440,7,7   -> B,feature_size,7,7
        self.bifpn1 = BiFPNBlock(feature_size)

    def forward(self, x):
        p1, p2, p3, p4 = x

        p1 = self.p1_proj(p1)
        p2 = self.p2_proj(p2)
        p3 = self.p3_proj(p3)
        p4 = self.p4_proj(p4)

        p1, p2, p3, p4 = self.bifpn1(p1, p2, p3, p4)

        return p1, p2, p3, p4
