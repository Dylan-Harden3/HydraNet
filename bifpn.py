import torch.nn as nn
import torch.nn.functional as F
from common import ConvBlock, FastNormalizedFusion


class BiFPNBlock(nn.Module):
    def __init__(self, n_channels, pyramid_shape):
        super(BiFPNBlock, self).__init__()
        self.p34_td_fuse = FastNormalizedFusion(2)
        self.p34_td_conv = ConvBlock(n_channels, n_channels, 3, 1, 1)

        self.p23_td_fuse = FastNormalizedFusion(2)
        self.p23_td_conv = ConvBlock(n_channels, n_channels, 3, 1, 1)

        self.p12_td_fuse = FastNormalizedFusion(2)
        self.p12_td_conv = ConvBlock(n_channels, n_channels, 3, 1, 1)

        self.p12_bu_fuse = FastNormalizedFusion(3)
        self.p12_bu_conv = ConvBlock(n_channels, n_channels, 3, 1, 1)

        self.p23_bu_fuse = FastNormalizedFusion(3)
        self.p23_bu_conv = ConvBlock(n_channels, n_channels, 3, 1, 1)

        self.p4_bu_fuse = FastNormalizedFusion(2)
        self.p4_bu_conv = ConvBlock(n_channels, n_channels, 3, 1, 1)

        self.pyramid_shape = pyramid_shape

    def forward(self, p1, p2, p3, p4):
        # top-down pathway
        p4_td = F.interpolate(
            p4, size=self.pyramid_shape[2][1:], mode="bilinear", align_corners=False
        )
        p3_td_out = self.p34_td_conv(self.p34_td_fuse(p3, p4_td))

        p3_td = F.interpolate(
            p3_td_out,
            size=self.pyramid_shape[1][1:],
            mode="bilinear",
            align_corners=False,
        )
        p2_td_out = self.p23_td_conv(self.p23_td_fuse(p2, p3_td))

        p2_td = F.interpolate(
            p2_td_out,
            size=self.pyramid_shape[0][1:],
            mode="bilinear",
            align_corners=False,
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
    def __init__(self, n_channels, n_blocks, pyramid_shape):
        super(BiFPN, self).__init__()
        self.p1_proj = ConvBlock(
            in_channels=pyramid_shape[0][0], out_channels=n_channels, kernel_size=1
        )
        self.p2_proj = ConvBlock(
            in_channels=pyramid_shape[1][0], out_channels=n_channels, kernel_size=1
        )
        self.p3_proj = ConvBlock(
            in_channels=pyramid_shape[2][0], out_channels=n_channels, kernel_size=1
        )
        self.p4_proj = ConvBlock(
            in_channels=pyramid_shape[3][0], out_channels=n_channels, kernel_size=1
        )

        self.bifpns = nn.ModuleList(
            [BiFPNBlock(n_channels, pyramid_shape) for _ in range(n_blocks)]
        )

    def forward(self, x):
        p1, p2, p3, p4 = x

        p1 = self.p1_proj(p1)
        p2 = self.p2_proj(p2)
        p3 = self.p3_proj(p3)
        p4 = self.p4_proj(p4)

        for bifpn in self.bifpns:
            p1, p2, p3, p4 = bifpn(p1, p2, p3, p4)

        return p1, p2, p3, p4
