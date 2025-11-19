import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
from models.hybrid_model import HybridVGGPerformer

def main():
    # Khởi tạo mô hình (dùng config mặc định)
    model = HybridVGGPerformer(
        backbone_pretrained=False,   # test nhanh -> tắt pretrained để load nhanh
        project_dim=256,
        num_tokens=8,
        token_dim=32,
        num_heads=4,
        performer_depth=1,
        dropout=0.2
    )

    # Fake input: batch 2, 3 kênh, ảnh 128x128
    x = torch.randn(2, 3, 128, 128)

    # Forward
    out = model(x)

    print("Output shape:", out.shape)
    print("Output:", out)

if __name__ == "__main__":
    main()
