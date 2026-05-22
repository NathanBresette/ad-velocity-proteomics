"""
Pooled cardiometabolic DiD + polypharmacy overlap analysis.
Addresses two reviewer concerns:
  1. Pooled cardiometabolic DiD — single pooled model with more power
  2. Exclusive-user analysis — patients on exactly one cardiometabolic class
     to rule out that 4/4 directional signal is driven by polypharmacy overlap

Run on Hellbender where tmt_native_dataset.csv lives:
  PYTHON=/home/nbhtd/.conda/envs/digitaltwin/bin/python
  cd /tmp && $PYTHON /home/nbhtd/scripts/tmt_validation/pooled_cardio_did.py
"""

import os, json
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV
from sklearn.ensemble import GradientBoostingRegressor
import statsmodels.formula.api as smf

# ── Paths ────────────────────────────────────────────────────────────────────
BASE     = "/home/nbhtd/scripts/tmt_validation"
DATA_DIR = "/home/nbhtd/data/digitaltwin"

RESULTS    = os.path.join(BASE, "results")
TMT_NATIVE = os.path.join(RESULTS, "tmt_native_dataset.csv")
COL_LISTS  = os.path.join(RESULTS, "column_lists_tmt.json")
OUT_CSV    = os.path.join(RESULTS, "pooled_cardio_did.csv")
OUT_JSON   = os.path.join(RESULTS, "pooled_cardio_did_summary.json")

# ── Model params (fixed from TMT cross-validation, Hellbender log 12655046) ──
GBM_PARAMS = dict(
    n_estimators=500, max_depth=4, learning_rate=0.05,
    min_samples_leaf=10, subsample=0.8, random_state=42
)

CARDIO_DRUGS = ["drug_statin", "drug_ace_inhibitor", "drug_ccb", "drug_thyroid"]

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading TMT dataset …")
df = pd.read_csv(TMT_NATIVE, low_memory=False)
print(f"  Shape: {df.shape}")

with open(COL_LISTS) as f:
    col_lists = json.load(f)

CLINICAL = col_lists["clinical"]
TMT_COLS = col_lists["tmt_native"]

# Confirm drug flags present
for d in CARDIO_DRUGS:
    assert d in df.columns, f"Missing drug column: {d}"

# ── Build pooled cardiometabolic flags ────────────────────────────────────────
df["any_cardio"] = (df[CARDIO_DRUGS].sum(axis=1) >= 1).astype(int)
df["n_cardio"]   = df[CARDIO_DRUGS].sum(axis=1)

# Exclusive users: takes exactly one cardiometabolic class
for d in CARDIO_DRUGS:
    label = d + "_exclusive"
    others = [x for x in CARDIO_DRUGS if x != d]
    df[label] = ((df[d] == 1) & (df[others].sum(axis=1) == 0)).astype(int)

n_any      = df["any_cardio"].sum()
n_none     = (df["any_cardio"] == 0).sum()
n_excl     = {d: df[d + "_exclusive"].sum() for d in CARDIO_DRUGS}
print(f"\nPolypharmacy breakdown:")
print(f"  Any cardiometabolic drug:  {n_any}")
print(f"  No cardiometabolic drug:   {n_none} (true non-users)")
print(f"  Exclusive users per class: {n_excl}")
print(f"  Overlap: {n_any - sum(n_excl.values())} patients on >=2 cardio classes")

# ── Helper: train non-circular GBM, predict all ───────────────────────────────
def train_and_predict(df_full, train_mask, features):
    train = df_full[train_mask].copy()
    X_tr  = train[features].values
    y_tr  = train["y_true"].values

    imp = SimpleImputer(strategy="median")
    sc  = StandardScaler()
    X_tr_imp = sc.fit_transform(imp.fit_transform(X_tr))

    lasso = LassoCV(cv=3, max_iter=10000, random_state=42)
    lasso.fit(X_tr_imp, y_tr)
    sel = np.where(lasso.coef_ != 0)[0]
    if len(sel) == 0:
        sel = np.arange(X_tr_imp.shape[1])

    gbm = GradientBoostingRegressor(**GBM_PARAMS)
    gbm.fit(X_tr_imp[:, sel], y_tr)

    X_all_imp = sc.transform(imp.transform(df_full[features].values))
    return gbm.predict(X_all_imp[:, sel])

FEATURES = CLINICAL + TMT_COLS
df["y_true"] = df["slope"]

# ── 1. POOLED CARDIOMETABOLIC DiD ─────────────────────────────────────────────
# Train on true non-users of any cardiometabolic drug
print("\n" + "="*60)
print("POOLED CARDIOMETABOLIC DiD")
print("  Train on: true non-users of any cardiometabolic drug")
print("="*60)

true_nonuser_mask = df["any_cardio"] == 0
print(f"  Training n: {true_nonuser_mask.sum()}")

df["y_pred_pooled"] = train_and_predict(df, true_nonuser_mask, FEATURES)

# Tertile split by predicted slope
df["tertile"] = pd.qcut(df["y_pred_pooled"], 3, labels=[1, 2, 3]).astype(int)

# OLS DiD: y_true ~ any_cardio * tertile
reg_data = df[["y_true", "any_cardio", "tertile"]].dropna()
model = smf.ols(
    "y_true ~ any_cardio * C(tertile)",
    data=reg_data
).fit()

# DiD = interaction coefficient for tertile 3 (fast progressors vs tertile 1)
did_coef = model.params.get("any_cardio:C(tertile)[T.3]",
           model.params.get("any_cardio:C(tertile)[T.3.0]", np.nan))
did_se   = model.bse.get("any_cardio:C(tertile)[T.3]",
           model.bse.get("any_cardio:C(tertile)[T.3.0]", np.nan))
did_p    = model.pvalues.get("any_cardio:C(tertile)[T.3]",
           model.pvalues.get("any_cardio:C(tertile)[T.3.0]", np.nan))

did_yr    = -did_coef * 12
ci_lo_yr  = -(did_coef + 1.96 * did_se) * 12
ci_hi_yr  = -(did_coef - 1.96 * did_se) * 12

print(f"\nPooled cardiometabolic DiD (fast vs slow):")
print(f"  n_users = {df['any_cardio'].sum()}, n_non-users = {true_nonuser_mask.sum()}")
print(f"  DiD = {did_yr:.3f} pts/yr  [{ci_lo_yr:.3f}, {ci_hi_yr:.3f}]  p={did_p:.4f}")
print(f"  Direction positive (fast progressors benefit more): {did_yr > 0}")
print(f"\nFull model summary (any_cardio interaction terms):")
for k, v in model.params.items():
    if "any_cardio" in k:
        p = model.pvalues[k]
        print(f"  {k:45s}  coef={v:.4f}  p={p:.4f}")

# ── 2. EXCLUSIVE USER ANALYSIS ────────────────────────────────────────────────
print("\n" + "="*60)
print("EXCLUSIVE USER DiD (one cardio class only, no co-prescription)")
print("="*60)

excl_rows = []
for d in CARDIO_DRUGS:
    excl_col = d + "_exclusive"
    n_excl_users = df[excl_col].sum()
    if n_excl_users < 20:
        print(f"  {d}: only {n_excl_users} exclusive users — skipping")
        continue

    # Use same true non-user pool (no cardiometabolic drugs at all)
    df_sub = df[df["any_cardio"] == 0].copy()
    df_sub = pd.concat([df_sub, df[df[excl_col] == 1].copy()], ignore_index=True)
    df_sub["is_user"] = (df_sub[excl_col] == 1).astype(int)

    train_mask_sub = df_sub["any_cardio"] == 0
    df_sub["y_pred_excl"] = train_and_predict(df_sub, train_mask_sub, FEATURES)
    df_sub["tertile_excl"] = pd.qcut(df_sub["y_pred_excl"], 3,
                                      labels=[1, 2, 3]).astype(int)

    reg_sub = df_sub[["y_true", "is_user", "tertile_excl"]].dropna()
    m = smf.ols("y_true ~ is_user * C(tertile_excl)", data=reg_sub).fit()

    dc = m.params.get("is_user:C(tertile_excl)[T.3]",
         m.params.get("is_user:C(tertile_excl)[T.3.0]", np.nan))
    ds = m.bse.get("is_user:C(tertile_excl)[T.3]",
         m.bse.get("is_user:C(tertile_excl)[T.3.0]", np.nan))
    dp = m.pvalues.get("is_user:C(tertile_excl)[T.3]",
         m.pvalues.get("is_user:C(tertile_excl)[T.3.0]", np.nan))

    dy   = -dc * 12
    cl   = -(dc + 1.96 * ds) * 12
    ch   = -(dc - 1.96 * ds) * 12
    pos  = dy > 0

    print(f"\n  {d} (exclusive n={n_excl_users}):")
    print(f"    DiD = {dy:.3f}  [{cl:.3f}, {ch:.3f}]  p={dp:.4f}  positive={pos}")

    excl_rows.append(dict(
        drug=d, n_exclusive=n_excl_users,
        did_yr=dy, ci_lo=cl, ci_hi=ch, p=dp, direction_positive=pos
    ))

excl_df = pd.DataFrame(excl_rows)

# ── 3. POLYPHARMACY OVERLAP TABLE ─────────────────────────────────────────────
print("\n" + "="*60)
print("POLYPHARMACY OVERLAP — co-prescription rates among cardiometabolic users")
print("="*60)
overlap = pd.DataFrame(index=CARDIO_DRUGS, columns=CARDIO_DRUGS, dtype=float)
for d1 in CARDIO_DRUGS:
    for d2 in CARDIO_DRUGS:
        users_d1 = df[df[d1] == 1]
        overlap.loc[d1, d2] = (users_d1[d2] == 1).mean() * 100
print(overlap.round(1).to_string())

# ── Save ──────────────────────────────────────────────────────────────────────
results = [
    dict(analysis="pooled_cardio", drug="pooled_cardiometabolic",
         n_users=int(df["any_cardio"].sum()),
         n_nonusers=int(true_nonuser_mask.sum()),
         did_yr=did_yr, ci_lo=ci_lo_yr, ci_hi=ci_hi_yr,
         p=did_p, direction_positive=bool(did_yr > 0))
]
for row in excl_rows:
    row["analysis"] = "exclusive_user"
    results.append(row)

out_df = pd.DataFrame(results)
out_df.to_csv(OUT_CSV, index=False)
print(f"\nSaved -> {OUT_CSV}")

summary = {
    "pooled": {
        "n_users": int(df["any_cardio"].sum()),
        "n_true_nonusers": int(true_nonuser_mask.sum()),
        "did_yr": round(did_yr, 4),
        "ci_lo": round(ci_lo_yr, 4),
        "ci_hi": round(ci_hi_yr, 4),
        "p": round(float(did_p), 6),
        "positive": bool(did_yr > 0)
    },
    "exclusive": {r["drug"]: {
        "n": r["n_exclusive"],
        "did_yr": round(r["did_yr"], 4),
        "p": round(r["p"], 6),
        "positive": r["direction_positive"]
    } for r in excl_rows},
    "polypharmacy_overlap_pct": overlap.round(1).to_dict()
}
with open(OUT_JSON, "w") as f:
    json.dump(summary, f, indent=2)
print(f"Saved -> {OUT_JSON}")
