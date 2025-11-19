import os
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import cv2
import random

class NoduleDataset(data.Dataset):
    def __init__(
        self,
        df,
        root_dir,
        image_dir,
        num_slices=3,
        image_size=128,
        clip_min=-1000,
        clip_max=400,
        normalize=True,
        augmentations=None
    ):
        """
        df: DataFrame có các cột: ["path", "label", "patient_id"]
        root_dir: thư mục base của dataset
        image_dir: thư mục chứa image .npy block
        augmentations: dict của config
        """

        self.df = df.reset_index(drop=True)
        self.root_dir = root_dir
        self.image_dir = image_dir

        self.num_slices = num_slices
        self.image_size = image_size
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.normalize = normalize
        self.aug = augmentations

    def __len__(self):
        return len(self.df)

    # -------------------------------------------------------------
    # Load block 3D từ file .npy
    # -------------------------------------------------------------
    def load_nodule_block(self, file_path):
        full_path = os.path.join(self.root_dir, self.image_dir, file_path)
        arr = np.load(full_path)  # shape [D, H, W] (thường thế)
        return arr

    # -------------------------------------------------------------
    # Lấy K lát trung tâm thành 3-channel input
    # -------------------------------------------------------------
    def extract_k_slices(self, volume):
        D = volume.shape[0]
        c = D // 2  # lát giữa

        if self.num_slices == 1:
            idxs = [c]
        else:
            half = self.num_slices // 2
            idxs = list(range(c - half, c + half + 1))

        idxs = [max(0, min(D - 1, i)) for i in idxs]  # tránh out of range

        slices = volume[idxs, :, :]  # shape [K, H, W]
        return slices

    # -------------------------------------------------------------
    # Chuẩn hóa intensity
    # -------------------------------------------------------------
    def normalize_intensity(self, arr):
        arr = np.clip(arr, self.clip_min, self.clip_max)

        # scale về 0..1
        arr = (arr - self.clip_min) / (self.clip_max - self.clip_min)
        arr = arr.astype(np.float32)
        return arr

    # -------------------------------------------------------------
    # Resize dùng OpenCV
    # -------------------------------------------------------------
    def resize_slices(self, slices):
        K, H, W = slices.shape
        out = np.zeros((K, self.image_size, self.image_size), dtype=np.float32)

        for i in range(K):
            out[i] = cv2.resize(slices[i], (self.image_size, self.image_size))

        return out

    # -------------------------------------------------------------
    # Basic augmentations (2D, đồng bộ cho tất cả các slice)
    # -------------------------------------------------------------
    def apply_augmentations(self, arr):
        # arr shape [K, H, W]

        if not self.aug:
            return arr

        # Flip Horizontal
        if self.aug.get("HFLIP", False) and random.random() < 0.5:
            arr = arr[:, :, ::-1]

        # Flip Vertical
        if self.aug.get("VFLIP", False) and random.random() < 0.5:
            arr = arr[:, ::-1, :]

        # Rotate 90°
        if self.aug.get("ROTATE90", False) and random.random() < 0.5:
            # rotate tất cả slice
            arr = np.rot90(arr, k=1, axes=(1, 2))

        # Intensity Shift
        if self.aug.get("INTENSITY_SHIFT", False) and random.random() < 0.3:
            shift = random.uniform(-0.05, 0.05)
            arr = np.clip(arr + shift, 0.0, 1.0)

        return arr

    # -------------------------------------------------------------
    # __getitem__
    # -------------------------------------------------------------
    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        path = row["path"]
        label = float(row["label"])

        # Load block 3D
        volume = self.load_nodule_block(path)

        # Normalize intensity
        if self.normalize:
            volume = self.normalize_intensity(volume)

        # Extract K slices
        slices = self.extract_k_slices(volume)  # [K,H,W]

        # Resize
        slices = self.resize_slices(slices)    # [K,128,128]

        # Augment
        slices = self.apply_augmentations(slices)

        # Convert to tensor (K,H,W) -> (C=K,H,W)
        tensor = torch.from_numpy(slices).float()

        return tensor, torch.tensor(label, dtype=torch.float32)
