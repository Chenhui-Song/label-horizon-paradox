"""BLO (Bi-Level Optimization) to auto-learn label-horizon weights lambda.

A standalone demo of the core method from ICML 2026 "The Label Horizon Paradox",
adapted to a multi-day daily-frequency setting:
  - Universe: all A-shares; features: Alpha158 (cross-sectional tanh
    classification preprocessing).
  - Labels: cross-sectional z-score of post-adjusted VWAP returns for h=1..10.
  - Inference target: Delta = 10 (10-day VWAP return).

Method:
  - teacher: a shared logits vector lambda in R^Delta, softmax-normalized into
    horizon weights.
  - warmup: a few epochs with a balanced diff-z label (SGD + random batch, same
    recipe as the brute-force search) to build a representation.
  - bi-level iterations with random-dispersed sampling (one sample per iter):
      * inner: M steps of differentiable SGD on theta (weighted IC loss).
      * outer: IC loss of theta* on the target Delta + entropy regularizer
        -> gradient on lambda -> update lambda.
      * write-back: theta* is written back into the student so inner updates
        accumulate across iterations.
  - NaN/inf handling in build_tensors is aligned with the brute-force search
    (drop rows, not fill 0), so BLO and the search use the same sample set.

R_AGG: the brute-force IC ranking used as the evaluation target. The learned
lambda is compared against R_AGG via Spearman correlation (rc).

Outputs:
  - blo_results.csv
  - log_blo.txt
"""
from __future__ import annotations

import sys
import warnings
import contextlib
import io
import time
import json
from pathlib import Path

warnings.filterwarnings("ignore", category=RuntimeWarning)

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
for _p in (_HERE, _ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np
import pandas as pd
import torch

from src.config import DEVICE, HORIZONS, RESULTS_DIR, TRAIN_START, TRAIN_END, VALID_START, VALID_END
from src.blo_trainer import BLOTrainer
from src.data_utils import (
    load_factors_tanhw, load_label_data, compute_label_quantile,
    get_tradable, get_train_union_tradable,
)
from src.model import daily_ic

# ---- BLO hyperparameters ----
N_ITERS = 180        # BLO updates per seed (one sample per update)
LAM_START = 160      # lambda averaging window [LAM_START:LAM_END] (last 20 iters)
LAM_END = 180
WARMUP_EPOCHS = 2    # SGD+4096 diff-z warmup epochs (shared student)
WARMUP_BATCHSIZE = 4096  # warmup batch size (samples per SGD step)
WARMUP_SEEDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]   # warmup random seeds; one full BLO run per warmup seed
N_SEED = 20          # BLO ensemble seed count
BATCH_DAYS = 10      # days sampled per inner/outer batch
GAP_DAYS = 10        # day gap between inner and outer ranges
LR_INNER = 0.05
LR_OUTER = 0.1
INNER_MOMENTUM = 0.9
INNER_STEPS = 1
ENTROPY_WEIGHT = 1e-4

# R_AGG: brute-force IC ranking used as the evaluation target.
# This is a hardcoded constant derived from OUR experimental results — the
# 15-seed aggregated brute-force search (SGD, batch 4096) over the test period
# 2025-01 ~ 2026-06 (see README "Key Results" and docs/search.md for the IC
# table). value = IC rank (10 = highest IC / best, 1 = lowest IC / worst). h3
# is best (IC 0.0860), h1 is worst (0.0767). Larger R_AGG = better horizon,
# matching the lambda convention (larger lambda = more important horizon), so a
# correct BLO lambda yields a positive rc.
#
# NOTE: if you re-run the brute-force search (different seeds / data / params)
# and obtain a different IC ranking, you MUST update R_AGG here manually to
# match, otherwise rc will be computed against a stale ranking. The IC values
# backing this ranking are: h1=0.0767, h2=0.0821, h3=0.0860, h4=0.0855,
# h5=0.0852, h6=0.0853, h7=0.0846, h8=0.0840, h9=0.0822, h10=0.0825.
R_AGG = np.array([1, 2, 10, 9, 7, 8, 6, 5, 3, 4], dtype=float)

BLO_DIR = RESULTS_DIR / "blo"
BLO_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_CSV = BLO_DIR / "blo_results.csv"
LOG = BLO_DIR / "log_blo.txt"
_lines = []


def p(msg=""):
    print(msg, flush=True)
    _lines.append(str(msg))
    with open(LOG, "w") as f:
        f.write("\n".join(_lines))


def rc(avg):
    """Spearman correlation between the ensemble lambda and R_AGG."""
    return float(np.corrcoef(R_AGG, pd.Series(avg).rank().values)[0, 1])


@contextlib.contextmanager
def quiet():
    with contextlib.redirect_stdout(io.StringIO()):
        yield


def build_tensors(long_df, factor_cols, vwap_adj, dates_set, horizons, member_mask, thr):
    """Build (X, Y_all, Y_warmup, date_ids).

    NaN/inf handling aligned with the brute-force search (build_xy_clip): rows
    with any NaN/inf in X are dropped (not filled), and rows with any label
    NaN/inf are dropped. This keeps BLO and the search on the same sample set.

    Y_all: per-horizon labels, cross-sectional z-scored + clip +/-3 (for BLO inner/outer).
    Y_warmup: diff-z label (d1=r1, d2=r2-r1, ..., d10=r10-r9, z-scored, equal-weighted mean),
              preprocessed once (for warmup, shared across batches).
    """
    ret_parts = []
    for h in horizons:
        r = vwap_adj.shift(-(h + 1)) / vwap_adj.shift(-1) - 1.0
        r.index.name = "date"
        lo, hi = thr[h]
        r = r.mask((r < lo) | (r > hi))  # trim tails (also masks +inf since inf>hi)
        ret_parts.append(r.stack().rename(f"ret_{h}"))
    ret_long = pd.concat(ret_parts, axis=1)
    ret_long.index.names = ["date", "wind_code"]
    ret_long = ret_long.replace([np.inf, -np.inf], np.nan)  # label inf -> nan

    fac = long_df[factor_cols]
    joined = fac.join(ret_long, how="inner")
    ret_cols = [f"ret_{h}" for h in horizons]
    joined = joined.dropna(subset=ret_cols)  # drop rows with any label NaN/inf

    mask_long = member_mask.stack().rename("in_index")
    mask_long.index.names = ["date", "wind_code"]
    joined = joined.join(mask_long, how="inner")
    joined = joined[joined["in_index"]].drop(columns=["in_index"])

    X_parts, Y_parts, Yw_parts, date_keys = [], [], [], []
    n_dropped = 0
    for tj, gdf in joined.groupby(level="date", sort=True):
        if tj not in dates_set:
            continue
        if len(gdf) < 20:
            continue
        # X: replace inf with nan, then drop rows with any nan (align with search)
        X_j = gdf[factor_cols].replace([np.inf, -np.inf], np.nan)
        m = X_j.notna().all(axis=1)
        n_dropped += int((~m).sum())
        gdf_keep = gdf[m]
        X_j = gdf_keep[factor_cols]
        R_j = gdf_keep[ret_cols].to_numpy(dtype=np.float32)
        if len(R_j) < 20:
            continue
        X_parts.append(X_j.to_numpy(dtype=np.float32))
        # BLO label: cross-sectional z-score + clip +/-3
        mu = R_j.mean(axis=0)
        sd = R_j.std(axis=0, ddof=1)
        sd = np.where(sd < 1e-12, 1.0, sd)
        Y_j = np.clip((R_j - mu) / sd, -3.0, 3.0).astype(np.float32)
        Y_parts.append(Y_j)
        # warmup label: diff -> z-score + clip +/-3 -> equal-weighted mean
        diffs = np.concatenate([R_j[:, :1], R_j[:, 1:] - R_j[:, :-1]], axis=1)  # (N, Delta)
        d_mu = diffs.mean(axis=0, keepdims=True)
        d_sd = diffs.std(axis=0, keepdims=True, ddof=1)
        d_sd = np.where(d_sd < 1e-12, 1.0, d_sd)
        diffs_z = np.clip((diffs - d_mu) / d_sd, -3.0, 3.0).astype(np.float32)
        yw = diffs_z.mean(axis=1)  # (N,) equal-weighted single label
        Yw_parts.append(yw)
        date_keys.append(tj)
    print(f"  build_tensors: dropped {n_dropped} rows with NaN/inf in X (aligned with search)", flush=True)

    X = np.concatenate(X_parts, axis=0)
    Y_all = np.concatenate(Y_parts, axis=0)
    Y_warmup = np.concatenate(Yw_parts, axis=0)
    date_ids_list = []
    for i, tj in enumerate(date_keys):
        date_ids_list.extend([i] * len(X_parts[i]))
    date_ids = np.array(date_ids_list, dtype=np.int64)

    X = torch.as_tensor(X, dtype=torch.float32, device=DEVICE)
    Y_all = torch.as_tensor(Y_all, dtype=torch.float32, device=DEVICE)
    Y_warmup = torch.as_tensor(Y_warmup, dtype=torch.float32, device=DEVICE)
    date_ids = torch.as_tensor(date_ids, dtype=torch.int64, device=DEVICE)
    return X, Y_all, Y_warmup, date_ids


def main():
    p("=" * 100)
    p(f"BLO | cached factors (69 tanh + 89 keep) | label trim0.5%+z+clip3 | {N_SEED}seed")
    p(f"  R_AGG (brute-force IC ranking) = {R_AGG.tolist()}")
    p(f"  random-dispersed sampling (batch={BATCH_DAYS}, gap={GAP_DAYS}) | "
      f"warmup{WARMUP_EPOCHS}ep(bs={WARMUP_BATCHSIZE}, diff-z) {N_ITERS}iters | "
      f"λ avg [{LAM_START}:{LAM_END}]")
    p(f"  M={INNER_STEPS} lr_inner={LR_INNER} lr_outer={LR_OUTER} momentum={INNER_MOMENTUM}")
    p(f"  train {TRAIN_START}~{TRAIN_END} valid {VALID_START}~{VALID_END}")
    p("=" * 100)

    t0 = time.time()
    long_df, factor_cols = load_factors_tanhw(use_cache=True)
    vwap_adj = load_label_data()
    tr_s, tr_e = pd.Timestamp(TRAIN_START), pd.Timestamp(TRAIN_END)
    vadj_tr = vwap_adj.loc[tr_s:tr_e]
    thr = {h: compute_label_quantile(vadj_tr, h) for h in HORIZONS}
    p(f"  label quantile thresholds (train q0.5/q99.5): "
      + " ".join(f"h{h}=[{thr[h][0]:.3f},{thr[h][1]:.3f}]" for h in HORIZONS))

    train_mask = get_train_union_tradable(TRAIN_START, VALID_END, TRAIN_START, TRAIN_END)
    train_dates = {d for d in long_df.index.get_level_values("date")
                   if pd.Timestamp(TRAIN_START) <= d <= pd.Timestamp(TRAIN_END)}
    valid_dates = {d for d in long_df.index.get_level_values("date")
                   if pd.Timestamp(VALID_START) <= d <= pd.Timestamp(VALID_END)}
    Xtr, Ytr, Yw_tr, dtr = build_tensors(long_df, factor_cols, vwap_adj, train_dates, HORIZONS, train_mask, thr)
    Xva, Yva, Yw_va, dva = build_tensors(long_df, factor_cols, vwap_adj, valid_dates, HORIZONS, train_mask, thr)
    p(f"  data loaded + tensors built: train={tuple(Xtr.shape)} valid={tuple(Xva.shape)} "
      f"({time.time()-t0:.0f}s)")
    target_idx = 9  # Delta = 10

    t1 = time.time()
    all_rows = []
    for ws in WARMUP_SEEDS:
        # Shared warmup student (one diff-z warmup, reused across all BLO seeds).
        p(f"\n[warmup {WARMUP_EPOCHS}ep seed={ws}] shared student (diff-z)...")
        torch.manual_seed(ws)
        np.random.seed(ws)
        tw = BLOTrainer(Xtr.shape[1], len(HORIZONS), DEVICE,
                        lr_inner=LR_INNER, lr_outer=LR_OUTER, inner_steps=INNER_STEPS,
                        inner_momentum=INNER_MOMENTUM, entropy_weight=ENTROPY_WEIGHT,
                        batch_days=BATCH_DAYS, gap_days=GAP_DAYS)
        tw.warmup(Xtr, Yw_tr, dtr, n_epochs=WARMUP_EPOCHS, batch_size=WARMUP_BATCHSIZE)
        shared_student = {k: v.detach().clone() for k, v in tw.student.state_dict().items()}
        del tw
        torch.cuda.empty_cache()
        p(f"  warmup{ws} done ({time.time()-t1:.0f}s)")

        seed_lams = []
        seed_rcs = []
        seed_argmaxs = []
        for bs in range(N_SEED):
            torch.manual_seed(bs)
            np.random.seed(bs)
            t = BLOTrainer(Xtr.shape[1], len(HORIZONS), DEVICE,
                           lr_inner=LR_INNER, lr_outer=LR_OUTER, inner_steps=INNER_STEPS,
                           inner_momentum=INNER_MOMENTUM, entropy_weight=ENTROPY_WEIGHT,
                           batch_days=BATCH_DAYS, gap_days=GAP_DAYS)
            t.student.load_state_dict(shared_student)
            with quiet():
                history = t.train(Xtr, Ytr, dtr, target_idx, N_ITERS,
                                  val_X=Xva, val_Y_target=Yva[:, target_idx], val_date_ids=dva)
            # Average lambda over [LAM_START:LAM_END] (post-burnin sweet spot).
            lam = np.array([h["lambda"] for h in history])[LAM_START:LAM_END].mean(axis=0)
            seed_lams.append(lam)
            seed_rcs.append(rc(lam))
            seed_argmaxs.append(int(np.argmax(lam)) + 1)
            del t
            if bs % 5 == 0 or bs == N_SEED - 1:
                p(f"    [warmup{ws}] seed{bs}/{N_SEED} rc={seed_rcs[-1]:.3f} argmax=h{seed_argmaxs[-1]} "
                  f"(cumulative {time.time()-t1:.0f}s)")

        seed_lams = np.array(seed_lams)
        ens = seed_lams.mean(axis=0)
        p(f"  [warmup{ws}] ens_rc={rc(ens):.3f} rc_mean={np.mean(seed_rcs):.3f} "
          f"argmax=h{int(np.argmax(ens))+1} 正seed={sum(1 for r in seed_rcs if r > 0)}/{N_SEED} "
          f"({time.time()-t1:.0f}s)")

        row = {
            "warmup_seed": ws, "warmup_epochs": WARMUP_EPOCHS,
            "batch_days": BATCH_DAYS, "gap_days": GAP_DAYS, "n_iters": N_ITERS,
            "ens_rc": rc(ens), "rc_mean": float(np.mean(seed_rcs)),
            "rc_std": float(np.std(seed_rcs)), "ens_argmax": int(np.argmax(ens)) + 1,
            "ens_spread": float(ens.max() - ens.min()),
            "n_pos": int(sum(1 for r in seed_rcs if r > 0)),
            "seed_rcs": json.dumps([round(r, 3) for r in seed_rcs]),
            "seed_argmaxs": json.dumps(seed_argmaxs),
            "ens_lambda": json.dumps([round(float(x), 4) for x in ens]),
        }
        all_rows.append(row)
        pd.DataFrame([row]).to_csv(BLO_DIR / f"blo_warmup{ws}.csv", index=False)

    p(f"\n=== Results ({len(WARMUP_SEEDS)} warmup seed(s)) ===")
    p(f"  per-warmup ens_rc: {[round(r['ens_rc'], 3) for r in all_rows]}")
    p(f"  per-warmup argmax: {[r['ens_argmax'] for r in all_rows]}")
    if len(all_rows) > 1:
        grand_lams = np.array([json.loads(r["ens_lambda"]) for r in all_rows]).mean(axis=0)
        p(f"  grand ens_rc (all warmup) = {rc(grand_lams):.3f}")
        p(f"  grand argmax = h{int(np.argmax(grand_lams))+1}")
        p(f"  grand lambda = {[round(float(x), 4) for x in grand_lams]}")
    p(f"  elapsed = {time.time()-t1:.0f}s")

    pd.DataFrame(all_rows).to_csv(RESULTS_CSV, index=False)
    p(f"  results saved: {RESULTS_CSV}")
    p(f"  per-warmup results in: {BLO_DIR}/blo_warmup*.csv")
    p(f"\n*** All done ***")


if __name__ == "__main__":
    main()
