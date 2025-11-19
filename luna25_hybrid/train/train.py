#!/usr/bin/env python3
"""
train/train.py
Entry point to train Hybrid VGG + Performer model with GroupKFold CV.

Usage (from project root or inside luna25_hybrid):
    # from luna25_hybrid folder
    python train/train.py --config config/config.yaml --fold -1

    # or run a single fold
    python train/train.py --config config/config.yaml --fold 0
"""

import os
import sys
import argparse
import random
import shutil
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Ensure project root (luna25_hybrid) is importable when running this script directly.
ROOT = Path(__file__).resolve().parents[1]  # luna25_hybrid/
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import load_config
from data.dataset import NoduleDataset
from models.hybrid_model import HybridVGGPerformer
from train.engine import train_one_epoch, validate_one_epoch, save_checkpoint, load_checkpoint

# -------------------
# Helpers
# -------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_dataloader(df, cfg, is_train=True):
    batch_size = cfg["TRAIN"].get("BATCH_SIZE", 16)
    num_workers = cfg["TRAIN"].get("NUM_WORKERS", 4)

    dataset = NoduleDataset(
        df=df,
        root_dir=cfg["DATA"].get("ROOT_DIR", "."),
        image_dir=cfg["DATA"].get("IMAGE_DIR", "image"),
        num_slices=cfg["DATA"].get("NUM_SLICES", 3),
        image_size=cfg["DATA"].get("IMAGE_SIZE", 128),
        clip_min=cfg["DATA"].get("CLIP_MIN", -1000),
        clip_max=cfg["DATA"].get("CLIP_MAX", 400),
        normalize=cfg["DATA"].get("NORMALIZE", True),
        augmentations=cfg.get("AUGMENTATIONS", None) if is_train else None
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=num_workers,
        pin_memory=True
    )
    return loader


# -------------------
# Train function
# -------------------
def run_cv(cfg: dict, csv_path: str, fold_to_run: int = -1, resume: Optional[str] = None):
    """
    Run cross-validation training.

    Args:
        cfg: loaded config dict
        csv_path: path to CSV with columns ['patient_id','path','label']
        fold_to_run: -1 -> run all folds; otherwise run only this fold index
        resume: optional checkpoint to resume from (path)
    """
    df_all = pd.read_csv(csv_path)
    # Basic check
    required_cols = {"patient_id", "path", "label"}
    if not required_cols.issubset(set(df_all.columns)):
        raise ValueError(f"CSV must contain columns: {required_cols}. Found: {df_all.columns.tolist()}")

    n_splits = cfg["CV"].get("NUM_FOLDS", 5)
    group_key = cfg["CV"].get("GROUP_KEY", "patient_id")

    gkf = GroupKFold(n_splits=n_splits)
    splits = list(gkf.split(df_all, df_all["label"], groups=df_all[group_key]))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    results_per_fold = []

    for fold_idx, (train_idx, valid_idx) in enumerate(splits):
        if fold_to_run != -1 and fold_to_run != fold_idx:
            continue

        print(f"\n==== Fold {fold_idx}/{n_splits-1} ====")
        train_df = df_all.iloc[train_idx].reset_index(drop=True)
        valid_df = df_all.iloc[valid_idx].reset_index(drop=True)

        train_loader = make_dataloader(train_df, cfg, is_train=True)
        valid_loader = make_dataloader(valid_df, cfg, is_train=False)

        # Build model
        model = HybridVGGPerformer(
            backbone_pretrained=bool(cfg["MODEL"].get("PRETRAINED", True)),
            project_dim=cfg["MODEL"].get("PROJECT_DIM", cfg["MODEL"].get("PROJECT_DIM", 256)),
            num_tokens=cfg["MODEL"].get("NUM_TOKENS", 8),
            token_dim=cfg["MODEL"].get("TOKEN_DIM", 32),
            num_heads=cfg["MODEL"].get("NUM_HEADS", 4),
            performer_depth=cfg["MODEL"].get("PERFORMER_DEPTH", 1),
            dropout=cfg["MODEL"].get("DROPOUT", 0.4)
        )

        model = model.to(device)

        # Optimizer + scheduler
        lr = cfg["TRAIN"].get("LR", 1e-4)
        weight_decay = cfg["TRAIN"].get("WEIGHT_DECAY", 1e-3)

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

        epochs = cfg["TRAIN"].get("EPOCHS", 20)
        use_amp = bool(cfg["TRAIN"].get("MIXED_PRECISION", False))

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        criterion = nn.BCEWithLogitsLoss()

        scaler = torch.cuda.amp.GradScaler() if use_amp and torch.cuda.is_available() else None

        best_auc = -1.0
        ckpt_dir = Path(cfg["CHECKPOINT"].get("DIR", "checkpoints")) / f"fold{fold_idx}"
        os.makedirs(ckpt_dir, exist_ok=True)

        start_epoch = 0
        # Resume logic
        if resume:
            ckpt = load_checkpoint(resume, model=model, optimizer=optimizer, scaler=scaler)
            start_epoch = ckpt.get("epoch", 0) + 1
            best_auc = ckpt.get("best_auc", -1.0)
            print(f"Resumed from {resume} at epoch {start_epoch}, best_auc={best_auc}")

        for epoch in range(start_epoch, epochs):
            t0 = time.time()
            train_res = train_one_epoch(
                model=model,
                dataloader=train_loader,
                optimizer=optimizer,
                criterion=criterion,
                device=device,
                epoch=epoch,
                scaler=scaler
            )

            val_res = validate_one_epoch(
                model=model,
                dataloader=valid_loader,
                criterion=criterion,
                device=device,
                epoch=epoch
            )

            # scheduler step (by epoch)
            try:
                scheduler.step()
            except Exception:
                pass

            epoch_time = time.time() - t0
            print(f"Fold {fold_idx} Epoch {epoch:02d} | train_loss {train_res['loss']:.4f} val_loss {val_res['loss']:.4f} | val_auc {val_res['auc']:.4f} | time {epoch_time:.1f}s")

            # Save checkpoint
            is_best = val_res.get("auc", -1.0) > best_auc
            if is_best:
                best_auc = val_res["auc"]

            state = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_auc": best_auc
            }
            ckpt_name = f"epoch{epoch:02d}.pth"
            ckpt_path = save_checkpoint(state, str(ckpt_dir), ckpt_name, save_best=is_best)
            if is_best:
                print(f"New best AUC {best_auc:.4f} -> saved {ckpt_path}")

        results_per_fold.append({"fold": fold_idx, "best_auc": best_auc})

    return results_per_fold


# -------------------
# CLI
# -------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="config/config.yaml", help="Path to config yaml")
    p.add_argument("--csv", type=str, default=None, help="Path to CSV listing samples (patient_id,path,label)")
    p.add_argument("--fold", type=int, default=-1, help="Fold index to run (-1 = all folds)")
    p.add_argument("--resume", type=str, default=None, help="Resume from checkpoint path")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)

    # Resolve CSV path
    csv_cfg = cfg["DATA"].get("CSV_PATH", "metadata.csv")
    if args.csv:
        csv_path = args.csv
    else:
        # if CSV_PATH is absolute, use it; else relative to project root
        csv_path = csv_cfg if os.path.isabs(csv_cfg) else str(ROOT / csv_cfg)

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found at {csv_path}. Please create CSV with columns ['patient_id','path','label'].")

    set_seed(args.seed)

    results = run_cv(cfg, csv_path=csv_path, fold_to_run=args.fold, resume=args.resume)
    print("\n==== CV Results ====")
    for r in results:
        print(r)


if __name__ == "__main__":
    main()
