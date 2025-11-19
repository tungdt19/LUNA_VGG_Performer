"""
Functions:
- train_one_epoch
- validate_one_epoch
- save_checkpoint
- load_checkpoint
- compute_metrics_from_targets_preds

Requires:
- torch, numpy, sklearn, tqdm
"""

import os
from typing import Tuple, Dict, Any, Optional

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch import Tensor
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

# Example metadata path used earlier in session (for quick debug)
EXAMPLE_METADATA_PATH = "/mnt/data/100012_1_19990102.npy"


def compute_metrics_from_targets_preds(targets: np.ndarray, preds: np.ndarray, thr: float = 0.5) -> Dict[str, float]:
    """Compute common metrics given numpy arrays of targets and predicted probabilities."""
    metrics = {}
    # handle edge cases
    try:
        auc = float(roc_auc_score(targets, preds))
    except Exception:
        auc = float("nan")

    pred_labels = (preds >= thr).astype(int)
    try:
        acc = float(accuracy_score(targets, pred_labels))
        prec = float(precision_score(targets, pred_labels, zero_division=0))
        rec = float(recall_score(targets, pred_labels, zero_division=0))
        f1 = float(f1_score(targets, pred_labels, zero_division=0))
    except Exception:
        acc = prec = rec = f1 = float("nan")

    metrics["auc"] = auc
    metrics["accuracy"] = acc
    metrics["precision"] = prec
    metrics["recall"] = rec
    metrics["f1"] = f1
    return metrics


def train_one_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    log_interval: int = 50,
) -> Dict[str, Any]:
    """
    Train model for one epoch with optional AMP.

    Returns:
        dict containing 'loss' (average) and training metrics computed on entire epoch
        (AUC/acc computed on aggregated preds - may be expensive for very large datasets)
    """
    model.train()
    running_loss = 0.0

    all_preds = []
    all_targets = []

    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Train Epoch {epoch}", ncols=120)
    for step, batch in pbar:
        # batch assumed to be (images, labels) or (images, labels, ...)
        if isinstance(batch, (list, tuple)):
            x = batch[0]
            y = batch[1]
        elif isinstance(batch, dict):
            x = batch["image"]
            y = batch["label"]
        else:
            raise ValueError("Unsupported batch format. Expect tuple/list or dict.")

        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True).float()

        optimizer.zero_grad()

        if scaler is not None:
            with torch.cuda.amp.autocast():
                logits = model(x)
                loss = criterion(logits.view(-1), y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(x)
            loss = criterion(logits.view(-1), y)
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * x.size(0)

        # collect preds
        probs = torch.sigmoid(logits.detach()).cpu().numpy().ravel()
        targets_np = y.detach().cpu().numpy().ravel()

        all_preds.append(probs)
        all_targets.append(targets_np)

        if (step + 1) % log_interval == 0:
            pbar.set_postfix({"loss": loss.item()})

    # aggregate
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    epoch_loss = running_loss / max(1, len(dataloader.dataset))

    metrics = compute_metrics_from_targets_preds(all_targets, all_preds)

    result = {"loss": epoch_loss}
    result.update(metrics)
    return result


@torch.no_grad()
def validate_one_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
) -> Dict[str, Any]:
    """
    Validate model for one epoch.

    Returns:
        dict with validation loss and metrics.
    """
    model.eval()
    running_loss = 0.0

    all_preds = []
    all_targets = []

    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Valid Epoch {epoch}", ncols=120)
    for step, batch in pbar:
        if isinstance(batch, (list, tuple)):
            x = batch[0]
            y = batch[1]
        elif isinstance(batch, dict):
            x = batch["image"]
            y = batch["label"]
        else:
            raise ValueError("Unsupported batch format. Expect tuple/list or dict.")

        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True).float()

        logits = model(x)
        loss = criterion(logits.view(-1), y)

        running_loss += loss.item() * x.size(0)

        probs = torch.sigmoid(logits).cpu().numpy().ravel()
        targets_np = y.cpu().numpy().ravel()

        all_preds.append(probs)
        all_targets.append(targets_np)

    # aggregate
    if len(all_preds) == 0:
        return {"loss": float("nan"), "auc": float("nan")}

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    epoch_loss = running_loss / max(1, len(dataloader.dataset))

    metrics = compute_metrics_from_targets_preds(all_targets, all_preds)

    result = {"loss": epoch_loss}
    result.update(metrics)
    return result


def save_checkpoint(state: Dict[str, Any], ckpt_dir: str, ckpt_name: str = "checkpoint.pth", save_best: bool = False):
    """
    Save checkpoint to disk.

    Args:
        state: dict with keys: epoch, model_state_dict, optimizer_state_dict, scaler_state_dict(optional), best_score(optional)
        ckpt_dir: directory where to save
        ckpt_name: filename
        save_best: if True, also create a 'best.pth' symlink/copy
    """
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, ckpt_name)
    torch.save(state, ckpt_path)

    if save_best:
        best_path = os.path.join(ckpt_dir, "best.pth")
        # attempting to create a copy (works across platforms)
        torch.save(state, best_path)

    return ckpt_path


def load_checkpoint(ckpt_path: str, model: Optional[nn.Module] = None, optimizer: Optional[torch.optim.Optimizer] = None, scaler: Optional[torch.cuda.amp.GradScaler] = None):
    """
    Load checkpoint and optionally restore model/optimizer/scaler states.
    Returns loaded checkpoint dict.
    """
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location="cpu")
    if model is not None and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scaler is not None and "scaler_state_dict" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
    return checkpoint
