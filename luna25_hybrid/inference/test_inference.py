import torch
from models.hybrid_model import HybridVGGPerformer
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--npy", type=str, required=True)
args = parser.parse_args()

# Load 3D npy
vol = np.load(args.npy)
# convert to tensor [C,H,W]
slices = vol[vol.shape[0]//2-1 : vol.shape[0]//2+2]  # 3 slices
slices = torch.tensor(slices).float()

slices = (slices - slices.mean()) / (slices.std() + 1e-5)
x = slices.unsqueeze(0)  # [1,3,H,W]

model = HybridVGGPerformer()
model.eval()

with torch.no_grad():
    out = torch.sigmoid(model(x))
    print("Pred:", out.item())
