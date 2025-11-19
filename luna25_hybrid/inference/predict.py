import os
import numpy as np
import torch
import pandas as pd
from tqdm import tqdm

from models.hybrid_model import HybridVGGPerformer
from data.preprocess import preprocess


def load_model(checkpoint_path, device):
    model = HybridVGGPerformer()
    model.load_state_dict(torch.load(checkpoint_path, map_location=device)["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def predict_one(model, path, device, cfg):
    volume = np.load(path)
    x = preprocess(
        volume,
        num_slices=cfg["DATA"]["NUM_SLICES"],
        image_size=cfg["DATA"]["IMAGE_SIZE"],
        clip_min=cfg["DATA"]["CLIP_MIN"],
        clip_max=cfg["DATA"]["CLIP_MAX"],
        normalize=cfg["DATA"]["NORMALIZE"]
    )

    x = torch.tensor(x).float().unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)  # shape [1]
        prob = torch.sigmoid(logits).item()
    label = 1 if prob >= 0.5 else 0
    return prob, label


def predict_folder(model, folder, device, cfg, save_csv=None):
    results = []
    files = [f for f in os.listdir(folder) if f.endswith(".npy")]

    for fname in tqdm(files, desc="Predicting"):
        path = os.path.join(folder, fname)
        prob, label = predict_one(model, path, device, cfg)
        results.append({"file": fname, "prob": prob, "label": label})

    df = pd.DataFrame(results)

    if save_csv:
        df.to_csv(save_csv, index=False)
        print("Saved CSV:", save_csv)

    return df
