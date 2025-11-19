import numpy as np
from data.preprocess import preprocess
import torch

def load_and_preprocess_npy(path, cfg):
    volume = np.load(path)

    x = preprocess(
        volume,
        num_slices=cfg["DATA"]["NUM_SLICES"],
        image_size=cfg["DATA"]["IMAGE_SIZE"],
        clip_min=cfg["DATA"]["CLIP_MIN"],
        clip_max=cfg["DATA"]["CLIP_MAX"],
        normalize=cfg["DATA"]["NORMALIZE"]
    )

    x = torch.tensor(x).float().unsqueeze(0)
    return x
