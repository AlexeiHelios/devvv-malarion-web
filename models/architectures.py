"""
MALARION — Model architecture definitions.
Copied verbatim from the notebook to guarantee weight compatibility.
"""
import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights


# ══════════════════════════════════════════════════════════════════════
# CBAM — Convolutional Block Attention Module
# (must match C3 training architecture exactly)
# ══════════════════════════════════════════════════════════════════════

class ChannelAttention(nn.Module):
    def __init__(self, in_channels: int, reduction_ratio: int = 16):
        super().__init__()
        mid = max(1, in_channels // reduction_ratio)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, in_channels, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        scale   = self.sigmoid(avg_out + max_out)
        return x * scale.unsqueeze(-1).unsqueeze(-1)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        pad = (kernel_size - 1) // 2
        self.conv    = nn.Conv2d(2, 1, kernel_size, padding=pad, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = x.mean(dim=1, keepdim=True)
        max_out = x.max(dim=1,  keepdim=True)[0]
        attn    = self.conv(torch.cat([avg_out, max_out], dim=1))
        return x * self.sigmoid(attn)


class CBAM(nn.Module):
    def __init__(self, in_channels: int,
                 reduction_ratio: int = 16,
                 spatial_kernel:  int = 7):
        super().__init__()
        self.channel = ChannelAttention(in_channels, reduction_ratio)
        self.spatial = SpatialAttention(spatial_kernel)

    def forward(self, x):
        x = self.channel(x)
        x = self.spatial(x)
        return x


# ══════════════════════════════════════════════════════════════════════
# BinaryValidator — ResNet18 binary classifier
# Head: Dropout(0.4) → Linear(512→256) → ReLU → Dropout(0.3) → Linear(256→1)
# Output: single logit (sigmoid applied externally at inference)
# ══════════════════════════════════════════════════════════════════════

class BinaryValidator(nn.Module):
    """
    ResNet18 binary classifier for parasite / background.
    Pretrained on ImageNet; fine-tuned on MALARION crops.
    """
    def __init__(self):
        super().__init__()
        backbone    = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        in_features = backbone.fc.in_features          # 512
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.head = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        return self.head(self.backbone(x)).squeeze(1)
