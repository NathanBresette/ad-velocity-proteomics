#!/usr/bin/env python3
"""
Phase 3: PROCOVA Variance Reduction (CSF Proteomics)
=====================================================
For each drug class: train counterfactual on non-users (with tuned HP),
predict for users, compute variance reduction + PROCOVA sample size.
Covariate balance checks (SMD).
"""

import sys, os, json, time
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.linear_model import LassoCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from scipy.stats import pearsonr

sys.stdout.reconfigure(line_buffering=True)

def tlog(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(BASE, "results")
DATA = os.path.join(OUT_DIR, "cdrsb_slope_dataset.csv")

BALANCE_VARS = ["CDRSB", "AGE", "APOE4", "ADAS13", "MMSE"]

def compute_smd(users, nonusers, var):
    m1, m0 = users[var].mean(), nonusers[var].mean()
    s1, s0 = users[var].std(), nonusers[var].std()
    pooled = np.sqrt((s1**2 + s0**2) / 2)
    return (m1 - m0) / pooled if pooled > 1e-10 else 0.0

def train_predict_counterfactual(X_train, y_train, X_pred, gbm_params):
    """LASSO inside -> GBM with tuned HP."""
    imp = SimpleImputer(strategy="median")
    X_tr = imp.fit_transform(X_train)
    X_pr = imp.transform(X_pred)
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    lasso = LassoCV(cv=3, max_iter=10000, random_state=42, n_jobs=-1)
    lasso.fit(X_tr_s, y_train)
    selected = np.where(np.abs(lasso.coef_) > 1e-8)[0]
    if len(selected) < 5:
        selected = np.argsort(np.abs(lasso.coef_))[-10:]
    gbm = GradientBoostingRegressor(subsample=0.8, random_state=42, **gbm_params)
    gbm.fit(X_tr[:, selected], y_train)
    return gbm.predict(X_pr[:, selected]), len(selected)

# -- load -------------------------------------------------------------------
tlog("Phase 3: PROCOVA Variance Reduction (CSF Proteomics)")
df = pd.read_csv(DATA)
with open(os.path.join(OUT_DIR, "column_lists.json")) as f:
    col_info = json.load(f)
with open(os.path.join(OUT_DIR, "best_gbm_params.json")) as f:
    best_params = json.load(f)

CLINICAL = col_info["clinical"]
CSF_COLS = col_info["csf"]
drug_cols = col_info["drug_classes"]
tlog(f"  {len(df)} patients, {len(drug_cols)} drug classes")

FEATURE_SETS = {
    "clinical": (CLINICAL, best_params["A_clinical"]),
    "clinical_csf": (CLINICAL + CSF_COLS, best_params["B_clinical_csf"]),
}

# -- per-drug analysis -------------------------------------------------------
balance_rows = []
vr_rows = []

for drug in drug_cols:
    n_users = int(df[drug].sum())
    if n_users < 20:
        tlog(f"\n  {drug}: {n_users} users -- skip")
        continue

    tlog(f"\n{'='*60}")
    tlog(f"{drug} | {n_users} users, {len(df)-n_users} non-users")

    users = df[df[drug] == 1]
    nonusers = df[df[drug] == 0]

    # Covariate balance
    for var in BALANCE_VARS:
        smd = compute_smd(users, nonusers, var)
        flag = abs(smd) > 0.2
        balance_rows.append({"drug": drug, "variable": var, "smd": smd, "flagged": flag})
        if flag:
            tlog(f"  SMD({var})={smd:.3f} FLAGGED")

    for fs_name, (feats, hp) in FEATURE_SETS.items():
        t0 = time.time()
        X_nu = nonusers[feats].values
        y_nu = nonusers["slope"].values
        X_u = users[feats].values
        y_u = users["slope"].values

        pred_cf, n_sel = train_predict_counterfactual(X_nu, y_nu, X_u, hp)
        te = y_u - pred_cf

        var_actual = np.var(y_u)
        var_te = np.var(te)
        vr = 1 - var_te / var_actual if var_actual > 0 else 0

        valid = ~np.isnan(pred_cf)
        r_val, _ = pearsonr(y_u[valid], pred_cf[valid]) if valid.sum() > 3 else (0, 1)
        r2_cf = r_val**2
        ss_reduction = r2_cf * 100

        vr_rows.append({
            "drug": drug, "feature_set": fs_name,
            "n_users": n_users, "n_nonusers": len(df)-n_users,
            "n_lasso_selected": n_sel,
            "var_actual_slope": var_actual, "var_treatment_effect": var_te,
            "variance_reduction": vr,
            "counterfactual_r": r_val, "counterfactual_r2": r2_cf,
            "procova_ss_reduction_pct": ss_reduction,
        })
        tlog(f"  {fs_name:15s} VR={vr:.4f} r2(CF)={r2_cf:.4f} PROCOVA={ss_reduction:.1f}% "
             f"LASSO={n_sel} ({time.time()-t0:.1f}s)")

# -- summary ----------------------------------------------------------------
tlog("\n\n========== VARIANCE REDUCTION SUMMARY ==========")
vr_df = pd.DataFrame(vr_rows)

for drug in vr_df["drug"].unique():
    sub = vr_df[vr_df["drug"] == drug]
    if len(sub) == 2:
        c = sub[sub["feature_set"] == "clinical"].iloc[0]
        n = sub[sub["feature_set"] == "clinical_csf"].iloc[0]
        delta = n["variance_reduction"] - c["variance_reduction"]
        tlog(f"  {drug:25s}  clin={c['variance_reduction']:.4f}  csf={n['variance_reduction']:.4f}  "
             f"d={delta:+.4f}  PROCOVA: {c['procova_ss_reduction_pct']:.1f}%->{n['procova_ss_reduction_pct']:.1f}%")

n_pos = sum(1 for d in vr_df["drug"].unique()
            if len(vr_df[(vr_df["drug"]==d)]) == 2 and
            vr_df[(vr_df["drug"]==d) & (vr_df["feature_set"]=="clinical_csf")].iloc[0]["variance_reduction"] >
            vr_df[(vr_df["drug"]==d) & (vr_df["feature_set"]=="clinical")].iloc[0]["variance_reduction"] + 0.05)
tlog(f"\nDrugs with >5% VR uplift from CSF: {n_pos}/{vr_df['drug'].nunique()}")

vr_df.to_csv(os.path.join(OUT_DIR, "procova_variance_reduction.csv"), index=False)
pd.DataFrame(balance_rows).to_csv(os.path.join(OUT_DIR, "covariate_balance.csv"), index=False)
tlog("Phase 3 DONE.")
