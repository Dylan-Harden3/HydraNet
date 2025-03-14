import torch.nn as nn
import torch.nn.functional as F
from backbone import RegNetBackbone
from bifpn import BiFPN


class HydraNet(nn.Module):
    def __init__(self):
        super(HydraNet, self).__init__()
        self.backbone = RegNetBackbone()
        in_features = 160
        self.bifpn = BiFPN([48, 104, 208, 440], in_features)
        self.age_head = nn.Linear(in_features, 1)
        self.gender_head = nn.Linear(in_features, 1)
        self.race_head = nn.Linear(in_features, 5)

    def forward(self, x):
        x = self.backbone(x)
        _, _, _, x = self.bifpn(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)

        return {
            "age": self.age_head(x),
            "gender": self.gender_head(x),
            "race": self.race_head(x),
        }
