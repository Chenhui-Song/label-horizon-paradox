"""Bi-Level Optimization (BLO) for learning label-horizon weights lambda.

Reproduces the core method of ICML 2026 "The Label Horizon Paradox", adapted to
a multi-day daily-frequency setting:

  - Universe: all A-shares; features: Alpha158 (cross-sectional tanh
    classification preprocessing).
  - Labels: cross-sectional z-score of post-adjusted VWAP returns
    ``vwap[t+1+h] / vwap[t+1] - 1`` for h = 1..10 (10 candidate horizons).
  - Inference target: Delta = 10 (10-day VWAP return, aligned with a 10-day
    rebalancing cadence).
  - Model: AlphaMLP (158 -> 256 -> 128 -> 1).

Method:
  - teacher: a shared logits vector lambda in R^Delta, softmax-normalized into
    horizon weights.
  - warmup: a few epochs of standard supervision with a balanced label
    (z-scored return *differences*, equal-weighted across horizons) to build a
    meaningful representation before the bi-level loop starts.
  - bi-level iterations: each iteration samples a random center t; the inner
    set is drawn from days before t, the outer set from days after t (with a
    gap in between). One sample per iteration.
      * inner: M steps of differentiable gradient descent on theta (weighted
        IC loss sum_delta lambda_delta * IC_delta), via
        ``torch.func.functional_call`` + ``autograd.grad(create_graph=True)``
        (second-order autodiff, no ``higher`` library needed).
      * outer: evaluate theta* on the outer set with the IC loss against the
        target Delta (1 - IC) plus an entropy regularizer, and update lambda.
      * write-back: write theta* back into the student (detached) so inner
        updates accumulate across iterations — the student keeps learning along
        the lambda-weighted inner direction instead of restarting from the
        warmup point every iteration.

The random-dispersed sampling keeps inner entirely before outer (time-forward,
no leakage) while spreading each side across the full period instead of using a
single contiguous block. The diff-z warmup avoids the short-horizon bias that a
naive mean-label warmup introduces (where r1 receives 10x the weight of r10).
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
for _p in (_HERE, _ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

warnings.filterwarnings("ignore", category=RuntimeWarning)

import numpy as np
import torch
import torch.nn as nn
from torch import autograd
from torch.func import functional_call

from src.model import AlphaMLP, daily_ic, weighted_daily_ic

__all__ = ["HorizonTeacher", "BLOTrainer"]


class HorizonTeacher(nn.Module):
    """Label-horizon weights lambda.

    ``logits`` in R^Delta; ``forward`` returns ``softmax(logits)`` (Delta
    horizon weights summing to 1). Shared (sample-independent), equivalent to
    the global lambda in the paper. Initialized uniform (softmax(0) = 1/Delta).
    """

    def __init__(self, n_horizons: int):
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(n_horizons))

    def forward(self):
        return torch.softmax(self.logits, dim=-1)


class BLOTrainer:
    """Bi-Level Optimization trainer with random-dispersed sampling.

    The inner loop uses ``functional_call`` + ``autograd.grad(create_graph=True)``
    for a differentiable M-step gradient descent; the outer loop takes the
    gradient on lambda. No ``higher`` library is required.

    Args:
        in_dim: feature dimension (158).
        n_horizons: number of candidate horizons (10).
        device: "cuda" / "cpu".
        lr_inner: inner-loop differentiable step learning rate.
        lr_outer: outer-loop lambda learning rate.
        inner_steps: inner-loop steps M. M=1 is the recommended default (also the
            paper's setting) — one differentiable SGD step per iteration is
            enough since theta_star is written back and accumulates across
            iterations; larger M increases second-order autodiff cost with
            little gain.
        inner_momentum: inner-loop SGD momentum (0 = plain SGD).
        entropy_weight: entropy regularizer weight gamma.
        batch_days: number of days sampled per inner/outer batch.
        gap_days: minimum day gap between the inner and outer date ranges.
    """

    def __init__(self, in_dim, n_horizons=10, device="cuda",
                 lr_inner=0.05, lr_outer=0.1, inner_steps=1, inner_momentum=0.9,
                 entropy_weight=1e-4, batch_days=10, gap_days=10):
        self.device = device
        self.n_horizons = n_horizons
        self.lr_inner = lr_inner
        self.lr_outer = lr_outer
        self.inner_steps = inner_steps
        self.inner_momentum = inner_momentum
        self.entropy_weight = entropy_weight
        self.batch_days = batch_days
        self.gap_days = gap_days

        self.student = AlphaMLP(in_dim).to(device)
        self.teacher = HorizonTeacher(n_horizons).to(device)
        self.opt_lambda = torch.optim.Adam(self.teacher.parameters(), lr=lr_outer)
        self._inner_velocity = None

    def warmup(self, X, Y_warmup, date_ids, n_epochs=1, batch_size=4096):
        """Warmup with a preprocessed diff-z label (single MSE).

        ``Y_warmup`` is a preprocessed label (trim -> diff -> cross-sectional
        z-score + clip -> equal-weighted mean across horizons), computed once
        in ``build_tensors`` and shared across batches. Uses 4096-sample random
        batches with SGD + momentum (same as the brute-force search).
        """
        opt = torch.optim.SGD(self.student.parameters(), lr=self.lr_inner,
                              momentum=0.95, weight_decay=1e-4)
        n = X.shape[0]
        for _ in range(n_epochs):
            self.student.train()
            perm = torch.randperm(n, device=self.device)
            for i in range(0, n, batch_size):
                idx = perm[i:i + batch_size]
                if len(idx) < 10:
                    continue
                xb = X[idx]
                label = Y_warmup[idx]
                pred = self.student(xb)
                loss = nn.functional.mse_loss(pred, label)
                opt.zero_grad()
                loss.backward()
                opt.step()

    def _sample_random_batch(self, unique_dates, date_ids, X, Y_all, date_indices):
        """Randomly pick ``batch_days`` days from ``unique_dates[date_indices]``."""
        bd = self.batch_days
        if len(date_indices) < bd:
            chosen = date_indices
        else:
            perm = torch.randperm(len(date_indices), device=self.device)[:bd]
            chosen = date_indices[perm]
        if len(chosen) < 5:
            return None
        batch_dates = unique_dates[chosen]
        mask = torch.isin(date_ids, batch_dates)
        if mask.sum() < 10:
            return None
        return X[mask], Y_all[mask], date_ids[mask]

    def train(self, X, Y_all, date_ids, target_idx, n_iters,
              val_X=None, val_Y_target=None, val_date_ids=None):
        """Random-dispersed sampling training for ``n_iters`` iterations.

        Each iteration: pick a random center t_idx; draw the inner set from
        ``[0, t_idx - gap//2)`` and the outer set from
        ``[t_idx + gap//2, n_dates)``, each ``batch_days`` days. Inner is
        entirely before outer (time-forward, no leakage) with a gap in between.

        Per iteration (one sample):
          1. theta_star = copy(student)
          2. inner: M steps of differentiable SGD on theta_star (weighted IC loss)
          3. outer: IC loss of theta_star on target Delta + entropy regularizer
             -> gradient on lambda -> update lambda
          4. write theta_star back into student (inner updates accumulate across
             iterations; student keeps learning along the lambda-weighted inner
             direction instead of restarting from warmup each iteration).

        ``val_ic`` (monitor only) is computed with the written-back student.
        """
        self.student.train()
        self.teacher.train()
        unique_dates = torch.unique(date_ids)
        n_dates = len(unique_dates)
        bd = self.batch_days
        half_gap = self.gap_days // 2
        # t_idx range: ensure both [0, t_idx-half_gap] and [t_idx+half_gap, n_dates]
        # contain at least bd days.
        t_min = bd + half_gap
        t_max = n_dates - bd - half_gap
        if t_max <= t_min:
            raise ValueError(
                f"Too few dates: n_dates={n_dates}, need at least "
                f"{2 * bd + self.gap_days + 1}"
            )

        all_indices = torch.arange(n_dates, device=self.device)

        history = []
        for it in range(n_iters):
            # ---- single sample per iteration ----
            self.opt_lambda.zero_grad()
            t_idx = int(torch.randint(t_min, t_max + 1, (1,)).item())
            inner_indices = all_indices[:t_idx - half_gap]
            outer_indices = all_indices[t_idx + half_gap:]

            inner_data = self._sample_random_batch(unique_dates, date_ids, X, Y_all, inner_indices)
            outer_data = self._sample_random_batch(unique_dates, date_ids, X, Y_all, outer_indices)
            if inner_data is None or outer_data is None:
                continue
            Xi, Yi, di = inner_data
            Xo, Yo_all_o, do = outer_data
            Yo_target = Yo_all_o[:, target_idx]

            # ---- inner: M steps of differentiable SGD on theta_star ----
            lam = self.teacher()
            params = {k: v for k, v in self.student.named_parameters()}
            buffers = {k: v for k, v in self.student.named_buffers()}
            theta_star = dict(params)
            if self.inner_momentum > 0 and self._inner_velocity is None:
                self._inner_velocity = {k: torch.zeros_like(v) for k, v in params.items()}
            for _ in range(self.inner_steps):
                pred_i = functional_call(self.student, (theta_star, buffers), Xi)
                inner_loss = 1.0 - weighted_daily_ic(pred_i, Yi, di, lam)
                grads = autograd.grad(inner_loss, list(theta_star.values()), create_graph=True)
                theta_star = dict(theta_star)
                for (k, g) in zip(theta_star.keys(), grads):
                    if self.inner_momentum > 0:
                        v = self._inner_velocity[k]
                        v_new = self.inner_momentum * v + g
                        self._inner_velocity[k] = v_new.detach()
                        theta_star[k] = theta_star[k] - self.lr_inner * v_new
                    else:
                        theta_star[k] = theta_star[k] - self.lr_inner * g

            # ---- outer: IC loss of theta_star on target Delta + entropy -> lambda ----
            pred_o = functional_call(self.student, (theta_star, buffers), Xo)
            outer_ic_loss = 1.0 - daily_ic(pred_o, Yo_target, do)
            lam2 = self.teacher()
            entropy = -(lam2 * torch.log(lam2 + 1e-8)).sum()
            outer_loss = outer_ic_loss - self.entropy_weight * entropy
            outer_loss.backward()  # gradient to teacher.logits.grad
            last_outer_loss = outer_loss.item()

            # ---- update lambda ----
            self.opt_lambda.step()

            # ---- write theta_star back into student (inner updates accumulate) ----
            with torch.no_grad():
                student_params = dict(self.student.named_parameters())
                for k, v in theta_star.items():
                    student_params[k].data.copy_(v.detach())

            with torch.no_grad():
                lam_val = self.teacher().cpu().numpy()
                if val_X is not None:
                    pred_v = self.student(val_X)  # student reflects inner updates
                    val_ic = daily_ic(pred_v, val_Y_target, val_date_ids).item()
                else:
                    val_ic = float('nan')
            history.append({"iter": it, "lambda": lam_val,
                            "outer_loss": last_outer_loss, "val_ic": val_ic})
            if it % max(1, n_iters // 10) == 0 or it == n_iters - 1:
                print(f"  [iter {it}/{n_iters}] outer_loss={last_outer_loss:.4f} "
                      f"val_ic={val_ic:.4f} lambda={lam_val.round(3)}", flush=True)
        return history

    def get_lambda(self) -> np.ndarray:
        """Return the learned lambda (softmax of logits)."""
        return torch.softmax(self.teacher.logits.detach(), dim=-1).cpu().numpy()
