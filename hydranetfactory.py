from dataclasses import dataclass
from torchvision.models import (
    regnet_y_400mf,
    regnet_y_800mf,
    regnet_y_8gf,
    regnet_y_16gf,
    regnet_y_32gf,
)
from backbone import RegNetBackbone
from bifpn import BiFPN
from hydranet import HydraNet
import torch


@dataclass
class HydraNetConfig:
    regnet: str
    image_size: tuple[int, int, int]
    bifpn_channels: int
    n_bifpn_blocks: int


regnet_variants = {
    "regnet_y_400mf": regnet_y_400mf,
    "regnet_y_800mf": regnet_y_800mf,
    "regnet_y_8gf": regnet_y_8gf,
    "regnet_y_16gf": regnet_y_16gf,
    "regnet_y_32gf": regnet_y_32gf,
}


def HydraNetFactory(config: HydraNetConfig) -> HydraNet:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    assert (
        config.regnet in regnet_variants
    ), f"RegNet variant {config.regnet} not in {list(regnet_variants.keys())}"

    backbone = RegNetBackbone(
        regnet_variants[config.regnet](weights="IMAGENET1K_V1")
    ).to(device)
    backbone.register_hooks()
    pyramid_shape = get_pyramid_shape(config.image_size, backbone, device)

    bifpn = BiFPN(config.bifpn_channels, config.n_bifpn_blocks, pyramid_shape).to(
        device
    )

    return HydraNet(backbone, bifpn)


def get_pyramid_shape(image_size, backbone, device):
    image = torch.rand(image_size).unsqueeze(0).to(device)
    with torch.no_grad():
        features = backbone(image)
    return [f.shape[1:] for f in features]
