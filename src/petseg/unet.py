"""Handwritten U-Net and a torchvision ResNet18 encoder with handwritten decoder."""

import torch
from torch import nn
from torch.nn import functional as F


class ConvBlock(nn.Sequential):
    def __init__(self, cin, cout):
        # GroupNorm also supports tiny batches and 1x1 bottlenecks.
        super().__init__(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False),
            nn.GroupNorm(min(8, cout), cout),
            nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False),
            nn.GroupNorm(min(8, cout), cout),
            nn.ReLU(inplace=True),
        )


class Up(nn.Module):
    def __init__(self, cin, skip, cout):
        super().__init__()
        self.block = ConvBlock(cin + skip, cout)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.block(torch.cat((x, skip), dim=1))


class UNet(nn.Module):
    def __init__(self, base_channels=32):
        super().__init__()
        if base_channels < 8 or base_channels % 8:
            raise ValueError("base_channels must be a positive multiple of 8")
        c = [base_channels * 2**i for i in range(5)]
        self.encoder = nn.ModuleList(
            [ConvBlock(3, c[0])] + [ConvBlock(c[i - 1], c[i]) for i in range(1, 5)]
        )
        self.decoder = nn.ModuleList(
            [Up(c[i], c[i - 1], c[i - 1]) for i in range(4, 0, -1)]
        )
        self.head = nn.Conv2d(c[0], 1, 1)

    def forward(self, x):
        if min(x.shape[-2:]) < 32:
            raise ValueError("Input spatial dimensions must be >=32")
        skips = []
        for i, block in enumerate(self.encoder):
            x = block(F.max_pool2d(x, 2) if i else x)
            skips.append(x)
        for up, skip in zip(self.decoder, reversed(skips[:-1])):
            x = up(x, skip)
        return self.head(x)


class ResNetUNet(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        from torchvision.models import ResNet18_Weights, resnet18

        net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu)
        self.pool, self.layer1 = net.maxpool, net.layer1
        self.layer2, self.layer3, self.layer4 = net.layer2, net.layer3, net.layer4
        self.full = ConvBlock(3, 32)
        self.up3, self.up2 = Up(512, 256, 256), Up(256, 128, 128)
        self.up1, self.up0, self.up_full = (
            Up(128, 64, 64),
            Up(64, 64, 32),
            Up(32, 32, 32),
        )
        self.head = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        full = self.full(x)
        s0 = self.stem(x)
        s1 = self.layer1(self.pool(s0))
        s2 = self.layer2(s1)
        s3 = self.layer3(s2)
        x = self.layer4(s3)
        x = self.up3(x, s3)
        x = self.up2(x, s2)
        x = self.up1(x, s1)
        x = self.up0(x, s0)
        return self.head(self.up_full(x, full))


def build_model(config, initialize=True):
    if config["model"] == "scratch":
        return UNet(config.get("base_channels", 32))
    if config["model"] == "pretrained":
        return ResNetUNet(pretrained=initialize)
    raise ValueError(f"Unknown model: {config['model']}")
