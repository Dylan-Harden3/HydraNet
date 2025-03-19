import torch.nn as nn
import torch.nn.functional as F
from backbone import RegNetBackbone
from bifpn import BiFPN
from heads import SegmentationHead


class HydraNet(nn.Module):
    def __init__(self, backbone, bifpn):
        super(HydraNet, self).__init__()
        self.backbone = backbone
        self.bifpn = bifpn
        self.segmentation_head = SegmentationHead(160, 3)

    def forward(self, x):
        x = self.backbone(x)
        p1, p2, p3, p4 = self.bifpn(x)
        return self.segmentation_head(p1, p2, p3, p4)
