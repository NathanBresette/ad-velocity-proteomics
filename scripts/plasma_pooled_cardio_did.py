"""
Pooled cardiometabolic DiD for plasma (NULISAseq Model B).
Uses nulisa_plasma_predictions.csv (279 MRM test patients) merged with
cdrsb_slope_dataset.csv drug flags. Mirrors the CSF pooled analysis for
direct comparison.

Run locally:
  cd /tmp && /opt/homebrew/bin/python3.11 \
    "/Users/nathanbresette/Desktop/DigitalTwin/FINAL_PUBLICATION/8_CSF_Proteomics/scripts/plasma_pooled_cardio_did.py"
"""

import json
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

BASE    = "/Users/nathanbresette/Desktop/DigitalTwin/FINAL_PUBLICATION/8_CSF_Proteomics/results"
PREDS   = f"{BASE}/nulisa_plasma_predictions.csv"
MRM     = f"{BASE}/cdrsb_slope_dataset.csv"
OUT_CSV = f"{BASE}/plasma_pooled_cardio_did.csv"
OUT_JSON= f"{BASE}/plasma_pooled_cardio_did_summary.json"

CARDIO_DRUGS = ["drug_statin", "drug_ace_inhibitor", "drug_ccb", "drug_thyroid"]
MIN_EXCL = 15   # minimum exclusive users to attempt analysis

# ── Load & merge ──────────────────────────────────────────────────────────────
preds = pd.read_csv(PREDS)
mrm   = pd.read_csv(MRM, low_memory=False)

keep  = ["RID", "slope"] + CARDIO_DRUGS
df    = preds[["RID", "y_pred_modelB"]].merge(mrm[keep], on="RID")
df    = df.rename(columns={"slope": "y_true", "y_pred_modelB": "y_pred"})

print(f"Merged n = {len(df)}")

# ── Build flags ───────────────────────────────────────────────────────────────
df["any_cardio"] = (df[CARDIO_DRUGS].sum(axis=1) >= 1).astype(int)
df["n_cardio"]   = df[CARDIO_DRUGS].sum(axis=1)
for d in CARDIO_DRUGS:
    others = [x for x in CARDIO_DRUGS if x != d]
    df[d + "_exclusive"] = ((df[d] == 1) & (df[others].sum(axis=1) == 0)).astype(int)

n_any  = df["any_cardio"].sum()
n_none = (df["any_cardio"] == 0).sum()
n_excl = {d: df[d + "_exclusive"].sum() for d in CARDIO_DRUGS}
print(f"\nPolypharmacy breakdown:")
print(f"  Any cardiometabolic drug: {n_any}")
print(f"  True non-users:           {n_none}")
print(f"  On >=2 cardio classes:    {(df['n_cardio']>=2).sum()} ({(df['n_cardio']>=2).sum()/n_any*100:.1f}%)")
print(f"  Exclusive users: {n_excl}")

# ── Helper: OLS DiD on tertiles of y_pred ────────────────────────────────────
def run_did(sub, user_col):
    sub = sub.copy()
    sub["tertile"] = pd.qcut(sub["y_pred"], 3, labels=[1, 2, 3]).astype(int)
    reg = sub[["y_true", user_col, "tertile"]].dropna()
    model = smf.ols(f"y_true ~ {user_col} * C(tertile)", data=reg).fit()

    def get(d, key):
        return d.get(f"{user_col}:C(tertile)[T.3]",
               d.get(f"{user_col}:C(tertile)[T.3.0]", np.nan))

    coef = get(model.params, None)
    se   = get(model.bse, None)
    p    = get(model.pvalues, None)
    # slopes already in pts/yr — no ×12
    did  = -coef
    ci_lo = -(coef + 1.96 * se)
    ci_hi = -(coef - 1.96 * se)
    return did, ci_lo, ci_hi, p

# ── 1. POOLED DiD ─────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("PLASMA POOLED CARDIOMETABOLIC DiD")
print("="*60)

did, ci_lo, ci_hi, p = run_did(df, "any_cardio")
pos = did > 0

print(f"  n_users    = {n_any}")
print(f"  n_nonusers = {n_none}")
print(f"  DiD = {did:.4f} pts/yr  [{ci_lo:.4f}, {ci_hi:.4f}]  p={p:.4f}")
print(f"  Direction positive: {pos}")

# ── 2. EXCLUSIVE USER DiD ─────────────────────────────────────────────────────
print("\n" + "="*60)
print("PLASMA EXCLUSIVE USER DiD")
print("="*60)

excl_rows = []
for d in CARDIO_DRUGS:
    ecol = d + "_exclusive"
    n_eu = df[ecol].sum()
    print(f"\n  {d}: exclusive n={n_eu}", end="")
    if n_eu < MIN_EXCL:
        print(f"  → skipped (< {MIN_EXCL})")
        continue
    print()
    sub = pd.concat([
        df[df["any_cardio"] == 0].copy(),
        df[df[ecol] == 1].copy()
    ], ignore_index=True)
    sub["is_user"] = (sub[ecol] == 1).astype(int)

    ed, ecl, ech, ep = run_did(sub, "is_user")
    epos = ed > 0
    print(f"    DiD = {ed:.4f}  [{ecl:.4f}, {ech:.4f}]  p={ep:.4f}  positive={epos}")
    excl_rows.append(dict(drug=d, n_exclusive=int(n_eu),
                          did_yr=round(ed, 4), ci_lo=round(ecl, 4),
                          ci_hi=round(ech, 4), p=round(ep, 6),
                          direction_positive=bool(epos)))

# ── 3. POLYPHARMACY OVERLAP ───────────────────────────────────────────────────
print("\n" + "="*60)
print("CO-PRESCRIPTION RATES (among cardiometabolic users, %)")
print("="*60)
overlap = pd.DataFrame(index=CARDIO_DRUGS, columns=CARDIO_DRUGS, dtype=float)
for d1 in CARDIO_DRUGS:
    for d2 in CARDIO_DRUGS:
        users_d1 = df[df[d1] == 1]
        overlap.loc[d1, d2] = (users_d1[d2] == 1).mean() * 100
print(overlap.round(1).to_string())

# ── Save ──────────────────────────────────────────────────────────────────────
results = [dict(
    analysis="pooled_cardio", drug="pooled_cardiometabolic",
    n_users=int(n_any), n_nonusers=int(n_none),
    did_yr=round(did, 4), ci_lo=round(ci_lo, 4), ci_hi=round(ci_hi, 4),
    p=round(float(p), 6), direction_positive=bool(pos)
)]
for row in excl_rows:
    r = dict(row); r["analysis"] = "exclusive_user"; results.append(r)

pd.DataFrame(results).to_csv(OUT_CSV, index=False)
print(f"\nSaved -> {OUT_CSV}")

summary = {
    "pooled": {
        "n_users": int(n_any), "n_true_nonusers": int(n_none),
        "did_yr": round(did, 4), "ci_lo": round(ci_lo, 4),
        "ci_hi": round(ci_hi, 4), "p": round(float(p), 6),
        "positive": bool(pos)
    },
    "exclusive": {r["drug"]: {
        "n": r["n_exclusive"], "did_yr": r["did_yr"],
        "p": r["p"], "positive": r["direction_positive"]
    } for r in excl_rows},
    "polypharmacy_overlap_pct": overlap.round(1).to_dict()
}
with open(OUT_JSON, "w") as f:
    json.dump(summary, f, indent=2)
print(f"Saved -> {OUT_JSON}")
