import torch
import torch.nn as nn
import torchvision.models as models


class VGGBackbone(nn.Module):
    """
    Backbone dựa trên VGG16 pretrained ImageNet.
    Trả ra vector đặc trưng 512 chiều.
    """

    def __init__(self, pretrained=True, project_dim=256, dropout=0.4):
        super().__init__()

        # Load VGG16 pretrained ImageNet
        vgg = models.vgg16(pretrained=pretrained)

        # Chỉ lấy phần feature extractor (conv layers)
        self.features = vgg.features  # output shape: [B, 512, H/32, W/32]

        # Global pooling → vector 512-D
        self.gap = nn.AdaptiveAvgPool2d(1)

        # Projection layer 512 → project_dim (mặc định 256)
        self.fc = nn.Sequential(
            nn.Linear(512, project_dim),
            nn.BatchNorm1d(project_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )

        self.project_dim = project_dim

    def forward(self, x):
        """
        x: Tensor đầu vào [B, 3, H, W]
        Output: [B, project_dim]
        """

        # 1) Feature extraction
        feat = self.features(x)            # [B,512,h,w]

        # 2) Global Average Pooling
        gap = self.gap(feat).view(x.size(0), -1)   # [B,512]

        # 3) Projection
        out = self.fc(gap)                 # [B, project_dim]

        return out
