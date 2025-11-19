import numpy as np
import cv2
import torch

def preprocess(volume, num_slices=3, image_size=128,
               clip_min=-1000, clip_max=400, normalize=True):
    """
    Pure preprocessing used for inference and training:
    - Clip HU
    - Normalize
    - Extract center slices
    - Resize H,W
    """

    vol = np.clip(volume, clip_min, clip_max)

    if normalize:
        vol = (vol - vol.mean()) / (vol.std() + 1e-5)

    center = vol.shape[0] // 2
    half = num_slices // 2
    start = center - half
    end = start + num_slices
    slices = vol[start:end]

    resized = []
    for slc in slices:
        slc2 = cv2.resize(slc, (image_size, image_size),
                          interpolation=cv2.INTER_LINEAR)
        resized.append(slc2)

    arr = np.stack(resized, axis=0)
    return arr  # shape [C,H,W]


def augment(image_tensor):
    """
    Simple augmentation for 2D slices (applied on the fly in Dataset):
    - horizontal flip
    - random brightness
    - gaussian noise
    """
    if torch.rand(1).item() < 0.5:
        image_tensor = torch.flip(image_tensor, dims=[2])

    if torch.rand(1).item() < 0.5:
        image_tensor = image_tensor + torch.randn_like(image_tensor) * 0.05

    if torch.rand(1).item() < 0.5:
        factor = 0.9
