import torch.nn as nn
import torch.nn.functional as F
from backbone import RegNetBackbone
from bifpn import BiFPN
from heads import SegmentationHead, ObjectDetectionHead


class HydraNet(nn.Module):
    def __init__(self, backbone, bifpn):
        super(HydraNet, self).__init__()
        self.backbone = backbone
        self.bifpn = bifpn
        self.lane_det_head = SegmentationHead(45, 1)
        self.drivable_area_head = SegmentationHead(45, 3)
        self.object_det_head = ObjectDetectionHead(45, 10)

    def forward(self, x):
        x = self.backbone(x)
        p1, p2, p3, p4 = self.bifpn(x)
        p1_out, p2_out, p3_out, p4_out = self.object_det_head(p1, p2, p3, p4)
        lane_det_out = self.lane_det_head(p1, p2, p3, p4)
        drivable_area_out = self.drivable_area_head(p1, p2, p3, p4)

        return p1_out, p2_out, p3_out, p4_out, lane_det_out, drivable_area_out