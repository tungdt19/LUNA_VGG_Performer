import torch
import torch.nn as nn
from performer_pytorch import Performer


class PerformerHead(nn.Module):
    """
    Performer-style lightweight attention head.
    
    Input:
        x: [B, project_dim]  (e.g., 256)
    
    Output:
        logit: [B]
    """

    def __init__(
        self,
        project_dim=256,
        num_tokens=8,
        token_dim=32,
        num_heads=4,
        performer_depth=1,
        dropout=0.3
    ):
        super().__init__()

        assert project_dim == num_tokens * token_dim, \
            f"project_dim phải = num_tokens * token_dim. Hiện tại {project_dim} != {num_tokens}*{token_dim}"

        self.num_tokens = num_tokens
        self.token_dim = token_dim

        # Performer encoder
        self.performer = Performer(
            dim=token_dim,
            depth=performer_depth,
            heads=num_heads,
            dim_head=token_dim,        # token_dim chia cho heads cũng ok, nhưng Performer cho phép flexible
            causal=False
        )

        # Normalization (LayerNorm)
        self.ln = nn.LayerNorm(token_dim)

        # Classification head
        self.fc = nn.Sequential(
            nn.Linear(token_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )

    def forward(self, x):  # x: [B, project_dim]
        B = x.size(0)

        # 1) Reshape vector thành tokens
        #    project_dim -> [B, num_tokens, token_dim]
        tokens = x.view(B, self.num_tokens, self.token_dim)

        # 2) Performer attention
        attn_out = self.performer(tokens)      # [B, num_tokens, token_dim]

        # 3) Residual + LayerNorm
        out = self.ln(attn_out + tokens)

        # 4) Mean pool -> [B, token_dim]
        pooled = out.mean(dim=1)

        # 5) FC → logit [B]
        logit = self.fc(pooled).squeeze(1)

        return logit
