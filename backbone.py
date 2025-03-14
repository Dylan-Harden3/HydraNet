import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class RegNetBackbone(nn.Module):
    def __init__(self):
        super(RegNetBackbone, self).__init__()
        self.regnet = models.regnet_y_400mf(weights="IMAGENET1K_V1")
        self.features = {}
    
    def register_hooks(self):
        for i, stage in enumerate(self.regnet.trunk_output):
            stage.register_forward_hook(self.get_feature_hook(f"stage{i+1}"))

    def get_feature_hook(self, name):
        def hook(module, input, output):
            self.features[name] = output

        return hook

    def forward(self, x):
        self.features.clear()

        _ = self.regnet(x)

        feature_list = []
        for i in range(len(self.regnet.trunk_output)):
            feature_list.append(self.features[f"stage{i+1}"])

        return feature_list
