import torch.nn as nn
import torch.nn.functional as F
from backbone import RegNetBackbone
from bifpn import BiFPN
from heads import SegmentationHead

class HydraNet(nn.Module):
    def __init__(self):
        super(HydraNet, self).__init__()
        self.backbone = RegNetBackbone()
        in_features = 160
        self.bifpn = BiFPN([48, 104, 208, 440], in_features)
        self.segmentation_head = SegmentationHead(in_features, 3)

    def forward(self, x):
        x = self.backbone(x)
        p1, p2, p3, p4 = self.bifpn(x)
        return self.segmentation_head(p1, p2, p3, p4)
