import torch.nn as nn
import torchvision.models as models


class HydraNet(nn.Module):
    def __init__(self):
        super(HydraNet, self).__init__()
        self.regnet = models.regnet_y_400mf(weights="IMAGENET1K_V1")
        in_features = self.regnet.fc.in_features
        self.backbone = nn.Sequential(*list(self.regnet.children())[:-1])
        self.age_head = nn.Linear(in_features, 1)
        self.gender_head = nn.Linear(in_features, 1)
        self.race_head = nn.Linear(in_features, 5)

    def forward(self, x):
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        age = self.age_head(x)
        gender = self.gender_head(x)
        race = self.race_head(x)
        return {"age": age, "gender": gender, "race": race}
