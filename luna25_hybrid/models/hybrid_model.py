import torch
import torch.nn as nn

from .vgg_backbone import VGGBackbone
from .performer_head import PerformerHead


class HybridVGGPerformer(nn.Module):
    """
    Hybrid Model:
    - Input: Tensor [B, C=3, H, W]
    - Backbone: VGG16 (pretrained)
    - Head: Performer attention
    - Output: logit [B]
    """

    def __init__(
        self,
        backbone_pretrained=True,
        project_dim=256,
        num_tokens=8,
        token_dim=32,
        num_heads=4,
        performer_depth=1,
        dropout=0.4
    ):
        super().__init__()

        # ================
        # Backbone
        # ================
        self.backbone = VGGBackbone(
            pretrained=backbone_pretrained,
            project_dim=project_dim,
            dropout=dropout
        )

        # ================
        # Performer Head
        # ================
        self.head = PerformerHead(
            project_dim=project_dim,
            num_tokens=num_tokens,
            token_dim=token_dim,
            num_heads=num_heads,
            performer_depth=performer_depth,
            dropout=dropout
        )

    # ------------------------------------------------------
    # Forward
    # ------------------------------------------------------
    def forward(self, x):
        """
        x: [B,3,H,W]
        return: logit [B]
        """
        feat = self.backbone(x)     # [B, project_dim]
        logit = self.head(feat)     # [B]
        return logit
