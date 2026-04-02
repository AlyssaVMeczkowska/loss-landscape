"""
Model definitions for loss landscape experiments.

We use a small ResNet-style network (ResNet-20 variant) for CIFAR-10.
The landscape sharpness difference between optimizers is most visible
with skip connections vs without — we provide both.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    """Residual block with two 3×3 convolutions and a skip connection."""
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out)


class SmallResNet(nn.Module):
    """
    Small ResNet for CIFAR-10 (similar to ResNet-20).
    ~270K parameters — fast enough to train and landscape-evaluate on CPU.
    """
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv0 = nn.Conv2d(3, 16, 3, 1, 1, bias=False)
        self.bn0   = nn.BatchNorm2d(16)

        self.layer1 = self._make_layer(16, 16, 3, stride=1)
        self.layer2 = self._make_layer(16, 32, 3, stride=2)
        self.layer3 = self._make_layer(32, 64, 3, stride=2)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc   = nn.Linear(64, num_classes)

    def _make_layer(self, in_ch, out_ch, n_blocks, stride):
        layers = [BasicBlock(in_ch, out_ch, stride)]
        for _ in range(n_blocks - 1):
            layers.append(BasicBlock(out_ch, out_ch, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn0(self.conv0(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class VGGStyle(nn.Module):
    """
    Plain VGG-style network WITHOUT skip connections.
    Produces sharper minima than ResNet — useful for the sharp vs flat comparison.
    ~600K parameters.
    """
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Linear(128, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x).flatten(1)
        return self.classifier(x)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
