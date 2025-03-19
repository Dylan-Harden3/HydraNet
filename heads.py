import torch.nn as nn
from common import ConvBlock, FastNormalizedFusion, ConvTransposeBlock
import torch.nn.functional as F

class SegmentationHead(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SegmentationHead, self).__init__()
        # another top-down pathway as an encoder
        self.p34_td_fuse = FastNormalizedFusion(2)
        self.p34_td_conv = ConvBlock(in_channels, in_channels, 1)
        self.p23_td_fuse = FastNormalizedFusion(2)
        self.p23_td_conv = ConvBlock(in_channels, in_channels, 1)

        self.p12_td_fuse = FastNormalizedFusion(2)
        self.p12_td_conv = ConvBlock(in_channels, in_channels, 1)

        self.upsample1 = ConvTransposeBlock(in_channels, in_channels // 2, 3, 2, 1)
        self.upsample2 = ConvTransposeBlock(in_channels // 2, in_channels // 2 // 2, 3, 2, 1)
        self.out = ConvBlock(in_channels // 2 // 2, out_channels, 3, 1, 1)

    def forward(self, p1, p2, p3, p4):
        p4_td = F.interpolate(p4, size=(15, 27), mode="bilinear", align_corners=False)
        p3_td_out = self.p34_td_conv(self.p34_td_fuse(p3, p4_td))

        p3_td = F.interpolate(
            p3_td_out, size=(30, 54), mode="bilinear", align_corners=False
        )
        p2_td_out = self.p23_td_conv(self.p23_td_fuse(p2, p3_td))

        p2_td = F.interpolate(
            p2_td_out, size=(60, 107), mode="bilinear", align_corners=False
        )
        p1_td_out = self.p12_td_conv(self.p12_td_fuse(p1, p2_td))
        
        up1 = F.interpolate(self.upsample1(p1_td_out), size=(120, 214), mode="bilinear", align_corners=False)
        up2 = F.interpolate(self.upsample2(up1), size=(240, 428), mode="bilinear", align_corners=False)
        out = self.out(up2)
        return out