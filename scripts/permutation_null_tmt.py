#!/usr/bin/env python3
"""
Permutation null for the TMT B-full model (clinical + 2,492 CSF proteins, n=1,060).

This is the model reported in the manuscript as R2=0.415 / Figure 1B. No
permutation test for it existed: the only completed permutation runs on this
cluster were for the (dropped) MRM within-platform model at observed R2=0.2924,
and the one TMT-native attempt (job 12654021) was cancelled before producing
output.

Pipeline is matched exactly to external_validation.py `train_predict`:
  SimpleImputer(median) -> StandardScaler -> LassoCV(cv=3) -> GBM on SCALED
  selected features, KFold(5, shuffle, random_state=42).

Speed strategy (same as mrm_rerun/permutation_null_1000.py):
  per-fold LassoCV alpha is learned once on the real labels, then each
  permutation refits Lasso at that fixed alpha before the GBM.
"""

import sys, os, json, time
import numpy as np
import pandas as pd
from datetime import datetime
from joblib import Parallel, delayed
from sklearn.model_selection import KFold
from sklearn.linear_model import LassoCV, Lasso
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

sys.stdout.reconfigure(line_buffering=True)


def tlog(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


BASE = "/home/nbhtd/scripts/tmt_validation"
OUT_DIR = os.path.join(BASE, "results")

N_PERMUTATIONS = int(os.environ.get("N_PERM", 1000))
N_JOBS = int(os.environ.get("SLURM_CPUS_PER_TASK", 8))

GBM_PARAMS = {"n_estimators": 500, "max_depth": 4, "learning_rate": 0.05,
              "min_samples_leaf": 10, "subsample": 0.8, "random_state": 42}


def get_fold_params(X, y):
    """LassoCV once per fold on the real labels -> fixed alpha + fitted transforms."""
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    out = []
    for i, (tr, te) in enumerate(kf.split(X)):
        imp = SimpleImputer(strategy="median")
        X_tr = imp.fit_transform(X[tr])
        sc = StandardScaler()
        X_tr_s = sc.fit_transform(X_tr)
        lasso = LassoCV(cv=3, max_iter=10000, random_state=42, n_jobs=1)
        lasso.fit(X_tr_s, y[tr])
        sel = np.where(np.abs(lasso.coef_) > 1e-8)[0]
        if len(sel) < 5:
            sel = np.argsort(np.abs(lasso.coef_))[-10:]
        out.append({"tr": tr, "te": te, "alpha": lasso.alpha_,
                    "sel": sel, "imp": imp, "sc": sc})
        tlog(f"    Fold {i+1}: alpha={lasso.alpha_:.5f}, features selected={len(sel)}")
    return out


def observed_r2(X, y, folds):
    """Observed OOF R2 using the real-label LASSO selection."""
    oof = np.full(len(y), np.nan)
    for fp in folds:
        tr, te = fp["tr"], fp["te"]
        X_tr_s = fp["sc"].transform(fp["imp"].transform(X[tr]))
        X_te_s = fp["sc"].transform(fp["imp"].transform(X[te]))
        gbm = GradientBoostingRegressor(**GBM_PARAMS)
        gbm.fit(X_tr_s[:, fp["sel"]], y[tr])
        oof[te] = gbm.predict(X_te_s[:, fp["sel"]])
    return r2_score(y, oof)


def one_permutation(i, X, y, folds):
    rng = np.random.RandomState(i)
    y_perm = y.copy()
    rng.shuffle(y_perm)
    oof = np.full(len(y), np.nan)
    for fp in folds:
        tr, te = fp["tr"], fp["te"]
        X_tr_s = fp["sc"].transform(fp["imp"].transform(X[tr]))
        X_te_s = fp["sc"].transform(fp["imp"].transform(X[te]))
        lasso = Lasso(alpha=fp["alpha"], max_iter=10000, random_state=42)
        lasso.fit(X_tr_s, y_perm[tr])
        sel = np.where(np.abs(lasso.coef_) > 1e-8)[0]
        if len(sel) < 5:
            sel = fp["sel"]
        gbm = GradientBoostingRegressor(**GBM_PARAMS)
        gbm.fit(X_tr_s[:, sel], y_perm[tr])
        oof[te] = gbm.predict(X_te_s[:, sel])
    return r2_score(y_perm, oof)


tlog(f"TMT B-full permutation null | {N_PERMUTATIONS} perms | {N_JOBS} workers")

ds = pd.read_csv(os.path.join(OUT_DIR, "tmt_native_dataset.csv"))
with open(os.path.join(OUT_DIR, "column_lists_tmt.json")) as f:
    cols = json.load(f)

FEATURES = cols["clinical"] + cols["tmt_native"]
X = ds[FEATURES].values
y = ds["slope"].values
tlog(f"  n={len(ds)} patients, {len(FEATURES)} features "
     f"({len(cols['clinical'])} clinical + {len(cols['tmt_native'])} proteins)")

tlog("Step 1: per-fold LassoCV on real labels ...")
folds = get_fold_params(X, y)

tlog("Step 2: observed OOF R2 ...")
obs = observed_r2(X, y, folds)
tlog(f"  Observed R2 = {obs:.4f}   (manuscript reports 0.4147)")

tlog(f"Step 3: {N_PERMUTATIONS} permutations ...")
t0 = time.time()
nulls = []
BATCH = 25
for s in range(0, N_PERMUTATIONS, BATCH):
    e = min(s + BATCH, N_PERMUTATIONS)
    nulls.extend(Parallel(n_jobs=N_JOBS)(
        delayed(one_permutation)(i, X, y, folds) for i in range(s, e)))
    el = time.time() - t0
    rate = el / len(nulls)
    tlog(f"  Perm {len(nulls)}/{N_PERMUTATIONS} | elapsed={el/60:.1f}min | "
         f"est_remaining={(N_PERMUTATIONS-len(nulls))*rate/60:.1f}min | "
         f"null_mean={np.mean(nulls):.4f} null_max={np.max(nulls):.4f}")

nulls = np.array(nulls)
n_ge = int((nulls >= obs).sum())
p_emp = (n_ge + 1) / (N_PERMUTATIONS + 1)

tlog("")
tlog("========== FINAL RESULTS ==========")
tlog(f"  observed R2   = {obs:.4f}")
tlog(f"  null mean     = {nulls.mean():.4f} +/- {nulls.std():.4f}")
tlog(f"  null max      = {nulls.max():.4f}")
tlog(f"  null p95      = {np.percentile(nulls, 95):.4f}")
tlog(f"  # null >= obs = {n_ge} / {N_PERMUTATIONS}")
tlog(f"  empirical p   = {p_emp:.6f}")

pd.DataFrame([{
    "model": "B_TMT_native_full", "observed_r2": obs,
    "null_mean": nulls.mean(), "null_std": nulls.std(),
    "null_max": nulls.max(), "null_p95": np.percentile(nulls, 95),
    "n_null_ge_observed": n_ge,
    "empirical_p": p_emp, "n_permutations": N_PERMUTATIONS,
}]).to_csv(os.path.join(OUT_DIR, "permutation_null_tmt.csv"), index=False)

pd.DataFrame({"perm_r2": nulls}).to_csv(
    os.path.join(OUT_DIR, "null_distribution_tmt_native_full.csv"), index=False)

tlog(f"Saved -> {OUT_DIR}/permutation_null_tmt.csv")
tlog("DONE.")
