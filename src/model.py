"""Model and differentiable IC loss functions.

  - ``AlphaMLP``: the 158 -> 256 -> 128 -> 1 prediction network.
  - ``daily_ic``: differentiable daily cross-sectional Pearson IC.
  - ``weighted_daily_ic`` / ``weighted_mse``: weighted multi-horizon losses
    used by the BLO inner loop.
  - ``daily_rankic_torch``: differentiable daily RankIC.

These are training components (model + its loss functions), used by both the
brute-force search and the bi-level optimization.
"""
from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
for _p in (_HERE, _ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import torch
import torch.nn as nn

from src.config import DEVICE


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class AlphaMLP(nn.Module):
    """Simple MLP: 158 -> 256 -> 128 -> 1 with ReLU + Dropout."""

    def __init__(self, in_dim, hidden=(256, 128), dropout=0.2):
        super().__init__()
        layers = []
        d = in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ---------------------------------------------------------------------------
# Differentiable daily cross-sectional IC
# ---------------------------------------------------------------------------
def daily_ic(pred: torch.Tensor, label: torch.Tensor, date_ids: torch.Tensor) -> torch.Tensor:
    """Mean of daily cross-sectional Pearson correlations (= IC), vectorized.

    Args:
        pred: (N,) model predictions.
        label: (N,) realized labels.
        date_ids: (N,) int64, the date id of each sample.

    Returns:
        Scalar IC (differentiable). Each day's cross-sectional correlation is
        computed and then averaged over days with >= 20 valid samples (aligned
        with ``metrics_for_alpha`` in data_utils.py).
    """
    unique = torch.unique(date_ids)
    D = unique.shape[0]
    idx = torch.searchsorted(unique, date_ids).clamp(0, D - 1)
    onehot = torch.zeros(pred.shape[0], D, device=pred.device, dtype=pred.dtype)
    onehot.scatter_(1, idx.unsqueeze(1), 1.0)
    count = onehot.sum(0).clamp(min=1.0)  # (D,)

    p_mean = (onehot * pred.unsqueeze(1)).sum(0) / count
    l_mean = (onehot * label.unsqueeze(1)).sum(0) / count
    pc = pred - p_mean[idx]
    lc = label - l_mean[idx]
    cross = (onehot * (pc * lc).unsqueeze(1)).sum(0)  # (D,)
    p2 = (onehot * (pc ** 2).unsqueeze(1)).sum(0)
    l2 = (onehot * (lc ** 2).unsqueeze(1)).sum(0)
    denom = torch.sqrt(p2 * l2) + 1e-6
    ic = cross / denom
    valid = (count >= 20) & (p2 > 1e-12) & (l2 > 1e-12)
    if valid.sum() == 0:
        # Keep the graph connected (return 0 with grad_fn) to avoid backward errors.
        return (pred * 0.0).sum()
    return ic[valid].mean()


def weighted_daily_ic(pred, label_cols, date_ids, weights):
    """Weighted multi-horizon IC: sum_delta lambda_delta * IC_delta, vectorized.

    Args:
        pred: (N,) predictions (a single model output shared across horizons).
        label_cols: (N, Delta) per-horizon labels.
        date_ids: (N,)
        weights: (Delta,) horizon weights (lambda).
    """
    unique = torch.unique(date_ids)
    D = unique.shape[0]
    idx = torch.searchsorted(unique, date_ids).clamp(0, D - 1)
    onehot = torch.zeros(pred.shape[0], D, device=pred.device, dtype=pred.dtype)
    onehot.scatter_(1, idx.unsqueeze(1), 1.0)
    count = onehot.sum(0).clamp(min=1.0)  # (D,)

    # Demean predictions per day
    p_mean = (onehot * pred.unsqueeze(1)).sum(0) / count  # (D,)
    pc = pred - p_mean[idx]  # (N,)
    p2 = (onehot * (pc ** 2).unsqueeze(1)).sum(0)  # (D,)
    p_norm = torch.sqrt(p2) + 1e-6  # (D,)

    # Demean labels per day, per horizon
    l_mean = (onehot.unsqueeze(2) * label_cols.unsqueeze(1)).sum(0) / count.unsqueeze(1)  # (D, Delta)
    lc = label_cols - l_mean[idx]  # (N, Delta)
    l2 = (onehot.unsqueeze(2) * (lc ** 2).unsqueeze(1)).sum(0)  # (D, Delta)
    l_norm = torch.sqrt(l2) + 1e-6  # (D, Delta)

    # Per-day, per-horizon sum of pc * lc
    cross = (onehot.unsqueeze(2) * (pc.unsqueeze(1) * lc).unsqueeze(1)).sum(0)  # (D, Delta)
    ic = cross / (p_norm.unsqueeze(1) * l_norm)  # (D, Delta)

    weighted_ic = (ic * weights.unsqueeze(0)).sum(1)  # (D,)
    valid = (count >= 20) & (p2 > 1e-12) & (l2.sum(1) > 1e-12)
    if valid.sum() == 0:
        return (pred * 0.0).sum()
    return weighted_ic[valid].mean()


def weighted_mse(pred, label_cols, weights):
    """Weighted multi-horizon MSE: sum_delta lambda_delta * MSE(pred, Y_delta).

    Used by the BLO inner loop (matches the weighted_daily_ic interface).

    Args:
        pred: (N,) predictions (a single model output shared across horizons).
        label_cols: (N, Delta) per-horizon labels.
        weights: (Delta,) horizon weights (lambda).
    """
    se = ((pred.unsqueeze(1) - label_cols) ** 2).mean(dim=0)  # (Delta,)
    return (se * weights).sum()


def daily_rankic_torch(pred, label, date_ids):
    """Differentiable daily cross-sectional RankIC."""
    unique = torch.unique(date_ids)
    D = unique.shape[0]
    idx = torch.searchsorted(unique, date_ids).clamp(0, D - 1)
    pred_rank = torch.zeros_like(pred)
    label_rank = torch.zeros_like(label)
    for d in range(D):
        mask = (idx == d)
        if mask.sum() < 2:
            continue
        pred_rank[mask] = pred[mask].argsort().argsort().float()
        label_rank[mask] = label[mask].argsort().argsort().float()
    return daily_ic(pred_rank, label_rank, date_ids)
