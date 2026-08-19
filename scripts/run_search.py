"""Brute-force horizon search: train one MLP per h in 1..10.

For each horizon h, train N_SEED independent AlphaMLP models supervised by the
h-day VWAP return, select the best epoch by validation IC (evaluate from epoch
`skip` onward), and save per-seed + aggregated test alphas.

This produces the brute-force IC ranking R_AGG that the BLO method is compared
against.

Config:
  - Factors: cached (69 tanh + 89 keep).
  - Label: post-adjusted VWAP return, quantile-trimmed, z-scored, clipped +/-3.
  - Test eval: h10 label, quantile-trimmed.
  - valid: pure IC for best-epoch selection, patience=3.
  - N_SEED seeds; save per-seed + aggregated alpha.

Outputs:
  - alpha_search_h{h}_seed{si}.parquet + alpha_agg.parquet
  - log_search.txt
"""
from __future__ import annotations

import sys
import warnings
import time
from pathlib import Path

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
for _p in (_HERE, _ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from src.config import DEVICE, HORIZONS, RESULTS_DIR, TARGET_H, TRAIN_START, TRAIN_END, VALID_START, VALID_END, TEST_START, TEST_END
from src.model import AlphaMLP, daily_ic
from src.data_utils import (
    load_factors_tanhw, load_label_data, make_label, compute_label_quantile,
    build_xy_clip, get_tradable, get_train_union_tradable, predict_alpha_pit, metrics_for_alpha,
)

# ---- Tunable hyperparameters ----
N_SEED = 15
SKIP = 1
PATIENCE = 3
WEIGHT_DECAY = 1e-4
EPOCHS = 15
BATCH_SIZE = 4096
MOMENTUM = 0.95
LR = 0.05
# TARGET_H (evaluation target horizon, h=10) is imported from src.config


def train_horizon(Xt, yt, dtr, Xv, yv, dv, in_dim, seed,
                  lr, momentum, weight_decay, epochs, patience, batch_size, skip):
    """Train one model; from epoch `skip`, select best by validation IC + patience.

    Returns (best_state, best_ep).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = AlphaMLP(in_dim).to(DEVICE)
    opt = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    n = len(Xt)
    best = -np.inf
    best_state = None
    best_ep = None
    bad = 0
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n, device=DEVICE)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            xb = Xt[idx]
            yb = yt[idx]
            pred = model(xb)
            loss = nn.functional.mse_loss(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
        if epoch < skip:
            continue
        model.eval()
        with torch.no_grad():
            pv = model(Xv)
            v_ic = float(daily_ic(pv, yv, dv).item())  # pure IC
        if v_ic > best:
            best = v_ic
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            best_ep = epoch + 1
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
    return best_state, best_ep


def main():
    print(f"=== h1..10 save alpha | valid pure-IC best | cached ===")
    print(f"  test uses h10 trimmed label | lr={LR} mom={MOMENTUM} "
          f"epochs={EPOCHS} patience={PATIENCE}")
    vwap_adj = load_label_data()
    tr_s, tr_e = pd.Timestamp(TRAIN_START), pd.Timestamp(TRAIN_END)
    vadj_tr = vwap_adj.loc[tr_s:tr_e]
    lo10, hi10 = compute_label_quantile(vadj_tr, TARGET_H)
    ret_eval = make_label(vwap_adj, TARGET_H, lo10, hi10)
    # The test label is quantile-trimmed (returns outside the train-set [q0.5,
    # q99.5] set to NaN). Trimming matters for the absolute IC: without it,
    # extreme returns (limit-up/down, resumption gaps) distort the Pearson
    # correlation and IC drops ~55% (0.054 vs 0.083 averaged over h), while
    # RankIC barely moves (0.110 vs 0.114). The IC-vs-RankIC gap shrinks from
    # 0.056 to 0.030. The IC *ranking* across horizons is nearly identical
    # trimmed vs untrimmed (only an adjacent h9/h10 swap at the low-IC tail),
    # so trimming does not affect the paradox — h=3 remains the single best
    # and h=10 the worst on IC either way.
    train_mask = get_train_union_tradable(TRAIN_START, TEST_END, TRAIN_START, TRAIN_END)
    pit_mask = get_tradable(TRAIN_START, TEST_END)

    fl, factor_cols = load_factors_tanhw(use_cache=True)
    train_dates = {d for d in fl.index.get_level_values("date")
                   if pd.Timestamp(TRAIN_START) <= d <= pd.Timestamp(TRAIN_END)}
    valid_dates = {d for d in fl.index.get_level_values("date")
                   if pd.Timestamp(VALID_START) <= d <= pd.Timestamp(VALID_END)}
    test_dates = {d for d in fl.index.get_level_values("date")
                  if pd.Timestamp(TEST_START) <= d <= pd.Timestamp(TEST_END)}

    ret_h10 = make_label(vwap_adj, 10, lo10, hi10)
    Xva10_np, yva10_np, dva10_np = build_xy_clip(fl, ret_h10, valid_dates, factor_cols, pit_mask, clip=3.0)
    Xv10 = torch.as_tensor(Xva10_np, dtype=torch.float32, device=DEVICE)
    yv10 = torch.as_tensor(yva10_np, dtype=torch.float32, device=DEVICE)
    dv10 = torch.as_tensor(dva10_np, dtype=torch.int64, device=DEVICE)

    t_all = time.time()
    for h in HORIZONS:
        lo, hi = compute_label_quantile(vadj_tr, h)
        ret_h = make_label(vwap_adj, h, lo, hi)
        Xtr_np, ytr_np, dtr_np = build_xy_clip(fl, ret_h, train_dates, factor_cols, train_mask, clip=3.0)
        Xt = torch.as_tensor(Xtr_np, dtype=torch.float32, device=DEVICE)
        yt = torch.as_tensor(ytr_np, dtype=torch.float32, device=DEVICE)
        dtr = torch.as_tensor(dtr_np, dtype=torch.int64, device=DEVICE)
        in_dim = Xtr_np.shape[1]
        print(f"\n--- h{h} train={Xt.shape} ---", flush=True)

        seed_alphas = []
        model_dir = RESULTS_DIR / "models" / f"h{h}"
        alpha_dir = RESULTS_DIR / "alphas" / f"h{h}"
        model_dir.mkdir(parents=True, exist_ok=True)
        alpha_dir.mkdir(parents=True, exist_ok=True)
        for si in range(N_SEED):
            best_state, best_ep = train_horizon(
                Xt, yt, dtr, Xv10, yv10, dv10, in_dim, si,
                LR, MOMENTUM, WEIGHT_DECAY, EPOCHS, PATIENCE, BATCH_SIZE, SKIP)
            torch.save(best_state, model_dir / f"model_seed{si}.pt")
            model = AlphaMLP(in_dim).to(DEVICE)
            model.load_state_dict(best_state)
            model.eval()
            alpha_s = predict_alpha_pit(model, fl, factor_cols, test_dates, pit_mask)
            alpha_s.to_parquet(alpha_dir / f"alpha_seed{si}.parquet")
            m = metrics_for_alpha(alpha_s, ret_eval)
            seed_alphas.append(alpha_s)
            print(f"    [seed{si}/{N_SEED}] best_ep={best_ep} IC={m['IC']:.4f} "
                  f"RankIC={m['RankIC']:.4f} (cumulative {time.time()-t_all:.0f}s)", flush=True)
            del model

        alpha_avg = sum(seed_alphas) / len(seed_alphas)
        alpha_avg.to_parquet(alpha_dir / "alpha_agg.parquet")
        m_agg = metrics_for_alpha(alpha_avg, ret_eval)
        print(f"  h{h}: agg IC={m_agg['IC']:.4f} ICIR={m_agg['ICIR']:.4f} "
              f"RankIC={m_agg['RankIC']:.4f} RankICIR={m_agg['RankICIR']:.4f} "
              f"(cumulative {time.time()-t_all:.0f}s)", flush=True)
        torch.cuda.empty_cache()

    print(f"\n*** All done, total {time.time()-t_all:.0f}s ***")


if __name__ == "__main__":
    main()
