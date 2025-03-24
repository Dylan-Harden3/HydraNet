import torch.nn as nn
from common import ConvBlock, FastNormalizedFusion, ConvTransposeBlock
import torch.nn.functional as F
import torch

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
        self.upsample2 = ConvTransposeBlock(
            in_channels // 2, in_channels // 2 // 2, 3, 2, 1
        )
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

        up1 = F.interpolate(
            self.upsample1(p1_td_out),
            size=(120, 214),
            mode="bilinear",
            align_corners=False,
        )
        up2 = F.interpolate(
            self.upsample2(up1), size=(240, 428), mode="bilinear", align_corners=False
        )
        out = self.out(up2)
        return out

class ObjectDetectionHead(nn.Module):
    def __init__(self, in_channels, num_classes, num_anchors=3):
        # for each anchor:
        # x, y, h, w, object score + per class probabilites
        out_channels = num_anchors * (5 + num_classes)
        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(in_channels, out_channels, 1)
        self.conv2 = nn.Conv2d(in_channels, out_channels, 1)
        self.conv3 = nn.Conv2d(in_channels, out_channels, 1)
        self.conv4 = nn.Conv2d(in_channels, out_channels, 1)


    def forward(self, p1, p2, p3, p4):
        p1 = self.conv1(p1)
        p2 = self.conv2(p2)
        p3 = self.conv3(p3)
        p4 = self.conv4(p4)
        
        return p1, p2, p3, p4


def decode_object_predictions(preds, anchors, strides, num_classes):
    # preds: 4, B, 3 x (5 + num_classes), H, W
    # anchors: 4, 3, 2
    # strides: 4, 2

    outputs = []
    for i, pred in enumerate(preds):
        B, C, H, W = pred.shape
        pred = pred.view(B, 3, 5+num_classes, H, W).permute(0, 3, 4, 1, 2)

        grid_y, grid_x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
        grid = torch.stack((grid_x, grid_y), dim=-1).float().to(pred.device)

        pred[..., 0:2] = torch.sigmoid(pred[..., 0:2] + grid) * strides[i]
        pred[..., 2:4] = torch.exp(pred[..., 2:4]) * anchors[i]
        pred[..., 4:] = torch.sigmoid(pred[..., 4:])

        pred = pred.reshape(B, H * W * 3, 5 + num_classes)
        outputs.append(pred)
    
    return torch.cat(outputs, dim=1)