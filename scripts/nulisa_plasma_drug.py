"""
Plasma drug stratification — NULISA full cohort (n=1,406)
Mirrors CSF Model D exactly:
  - For each drug: train LASSO+GBM on NON-USERS only (non-circular)
  - Predict ALL patients (users + non-users)
  - DiD: compare observed slope attenuation in fast vs slow progressors

Uses full NULISA plasma cohort (all 1,406 patients) — NOT the train/test split.
Drug flags from reccmeds.rdata (pre-coded columns + CMMED keyword search).
"""

import subprocess, os, tempfile, json
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

# ── paths ──────────────────────────────────────────────────────────────────
DATA_DIR   = "/Users/nathanbresette/Desktop/DigitalTwin/Data"
ADNI_DIR   = f"{DATA_DIR}/ADNIMERGE/data"
NULISA_CSV = f"{DATA_DIR}/BSHRI_PLA_CSF_NULISA_CNS_05May2026.csv"
OUT_DIR    = "/Users/nathanbresette/Desktop/DigitalTwin/FINAL_PUBLICATION/8_CSF_Proteomics/results"
os.makedirs(OUT_DIR, exist_ok=True)

# ── tuned plasma params (from RandomizedSearchCV on training set) ──────────
PLASMA_GBM_PARAMS = dict(
    n_estimators=500, max_depth=2, learning_rate=0.01,
    min_samples_leaf=20, subsample=0.9, random_state=42
)
CLINICAL  = ["AGE", "PTEDUCAT", "APOE4", "ADAS13", "CDRSB", "MMSE", "FAQ"]
MIN_USERS = 25

# ── drug definitions ───────────────────────────────────────────────────────
# Pre-coded columns in reccmeds (1 = user at any visit)
PRECODED = {
    "drug_donepezil":   "DONEPEZIL",
    "drug_galantamine": "GALANTAMINE",
    "drug_rivastigmine":"RIVASTIGMINE",
    "drug_memantine":   "MEMANTINE",
    "drug_thyroid":     "THYROXINE",
    "drug_nsaid_flag":  "NSAID",
}
# CMMED keyword patterns
KEYWORD = {
    "drug_statin":       "STATIN|ATORVASTATIN|SIMVASTATIN|ROSUVASTATIN|PRAVASTATIN|LOVASTATIN|FLUVASTATIN|PITAVASTATIN|CRESTOR|LIPITOR|ZOCOR|PRAVACHOL",
    "drug_ace_inhibitor":"LISINOPRIL|ENALAPRIL|RAMIPRIL|CAPTOPRIL|BENAZEPRIL|QUINAPRIL|PERINDOPRIL|FOSINOPRIL|LOSARTAN|VALSARTAN|OLMESARTAN|IRBESARTAN|CANDESARTAN",
    "drug_ccb":          "AMLODIPINE|DILTIAZEM|VERAPAMIL|NIFEDIPINE|FELODIPINE|NORVASC|CARDIZEM",
    "drug_ppi":          "OMEPRAZOLE|LANSOPRAZOLE|PANTOPRAZOLE|RABEPRAZOLE|ESOMEPRAZOLE|PRILOSEC|PREVACID|NEXIUM|PROTONIX|ACIPHEX",
    "drug_nsaid":        "IBUPROFEN|NAPROXEN|ASPIRIN|CELECOXIB|INDOMETHACIN|MELOXICAM|DICLOFENAC",
}
ALL_DRUGS = list(PRECODED.keys()) + list(KEYWORD.keys())
CARDIOMETABOLIC = ["drug_statin", "drug_ace_inhibitor", "drug_ccb", "drug_thyroid"]
CHOLINERGIC     = ["drug_donepezil", "drug_galantamine", "drug_rivastigmine", "drug_memantine"]
NEGATIVE        = ["drug_nsaid", "drug_ppi"]

# ── helpers ────────────────────────────────────────────────────────────────
def rdata_to_df(path):
    tmp = tempfile.mktemp(suffix=".csv")
    r = subprocess.run(["Rscript", "-e", f"""
    load('{path}')
    for (obj in ls()) {{
        x <- get(obj)
        if (is.data.frame(x)) {{ write.csv(x, '{tmp}', row.names=FALSE); break }}
    }}
    """], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"Rscript failed: {r.stderr}")
    df = pd.read_csv(tmp)
    os.remove(tmp)
    return df


def compute_slopes(adni, min_visits=3, min_span=12):
    adni = adni[["RID","M","CDRSB"]].copy()
    adni["RID"] = adni["RID"].astype(int)
    adni["M"]   = pd.to_numeric(adni["M"], errors="coerce")
    adni = adni.dropna(subset=["CDRSB","M"])
    out = []
    for rid, grp in adni.groupby("RID"):
        grp = grp.sort_values("M")
        if len(grp) < min_visits or grp["M"].max() - grp["M"].min() < min_span:
            continue
        slope, *_ = stats.linregress(grp["M"].values, grp["CDRSB"].values)
        out.append({"RID": rid, "slope": slope})
    return pd.DataFrame(out)


def pivot_nulisa(csv_path, rid_set=None):
    df = pd.read_csv(csv_path, low_memory=False)
    df = df[(df["SampleType"]=="Sample") & (df["SampleMatrixType"]=="PLASMA")]
    if rid_set is not None:
        df = df[df["RID"].isin(rid_set)]
    norm = df["VISCODE2"].astype(str).str.strip().str.lower()
    bl  = df[norm=="bl"]
    m06 = df[norm=="m06"]
    rids_bl = set(bl["RID"])
    base = pd.concat([bl, m06[~m06["RID"].isin(rids_bl)]], ignore_index=True)
    base["NPQ"] = pd.to_numeric(base["NPQ"], errors="coerce")
    base["RID"] = base["RID"].astype(int)
    wide = base.pivot_table(index="RID", columns="Target", values="NPQ", aggfunc="mean").reset_index()
    wide.columns.name = None
    rename = {c: f"PLASMAp_{c.replace('β','b').replace('-','_').replace(' ','_')}"
              for c in wide.columns if c != "RID"}
    wide.rename(columns=rename, inplace=True)
    return wide, rename


def build_drug_flags(reccmeds, nulisa_rids):
    rc = reccmeds[reccmeds["RID"].isin(nulisa_rids)].copy()
    rc["RID"] = rc["RID"].astype(int)
    cmmed_upper = rc["CMMED"].astype(str).str.upper()
    flags = pd.DataFrame({"RID": sorted(nulisa_rids)})
    flags["RID"] = flags["RID"].astype(int)
    # Pre-coded
    for drug_key, col in PRECODED.items():
        users = set(rc.loc[rc[col]==1, "RID"])
        flags[drug_key] = flags["RID"].isin(users).astype(int)
    # Keyword
    for drug_key, pat in KEYWORD.items():
        users = set(rc.loc[cmmed_upper.str.contains(pat, regex=True, na=False), "RID"])
        flags[drug_key] = flags["RID"].isin(users).astype(int)
    return flags


def noncircular_predict(df, feature_cols, drug_col, gbm_params):
    """Train on non-users, predict all. Returns predictions for full df."""
    non_users = df[df[drug_col]==0].copy()
    X_all   = df[feature_cols].values
    X_train = non_users[feature_cols].values
    y_train = non_users["slope"].values

    imp = SimpleImputer(strategy="median").fit(X_train)
    sc  = StandardScaler().fit(imp.transform(X_train))
    Xtr = sc.transform(imp.transform(X_train))
    Xte = sc.transform(imp.transform(X_all))

    lasso = LassoCV(cv=3, max_iter=10000, random_state=42).fit(Xtr, y_train)
    mask  = lasso.coef_ != 0
    if mask.sum() == 0:
        mask = np.ones(len(feature_cols), dtype=bool)

    model = GradientBoostingRegressor(**gbm_params).fit(Xtr[:,mask], y_train)
    return model.predict(Xte[:,mask]), int(mask.sum())


def did_analysis(df, drug_col, y_pred_col, n_quintiles=5):
    """Difference-in-differences by predicted quintile."""
    df = df.copy()
    df["quintile"] = pd.qcut(df[y_pred_col], n_quintiles, labels=False)
    rows = []
    for q in range(n_quintiles):
        grp = df[df["quintile"]==q]
        users    = grp[grp[drug_col]==1]["slope"]
        nonusers = grp[grp[drug_col]==0]["slope"]
        rows.append({
            "quintile": q+1,
            "n_users": len(users),
            "n_nonusers": len(nonusers),
            "mean_slope_users": users.mean() if len(users)>=3 else np.nan,
            "mean_slope_nonusers": nonusers.mean() if len(nonusers)>=3 else np.nan,
        })
    qdf = pd.DataFrame(rows)
    qdf["gap"] = qdf["mean_slope_users"] - qdf["mean_slope_nonusers"]

    # DiD: (fast quintile gap) - (slow quintile gap)
    fast_gap = qdf.loc[qdf["quintile"]==n_quintiles, "gap"].values
    slow_gap = qdf.loc[qdf["quintile"]==1, "gap"].values
    did = float(fast_gap[0] - slow_gap[0]) if len(fast_gap) and len(slow_gap) else np.nan

    # OLS DiD with CI
    df2 = df.dropna(subset=["slope", y_pred_col, drug_col])
    df2["fast"] = (df2["quintile"] == n_quintiles-1).astype(int)  # top quintile
    try:
        from scipy.stats import t as t_dist
        # interaction term: drug × fast_quintile
        X_did = np.column_stack([
            np.ones(len(df2)),
            df2[drug_col].values,
            df2["fast"].values,
            df2[drug_col].values * df2["fast"].values
        ])
        beta, res, _, _ = np.linalg.lstsq(X_did, df2["slope"].values, rcond=None)
        did_coef = beta[3]
        n, p = len(df2), 4
        if len(res) > 0:
            mse = res[0] / (n - p)
            se  = np.sqrt(mse * np.linalg.inv(X_did.T @ X_did)[3,3])
            ci_lo = did_coef - 1.96*se
            ci_hi = did_coef + 1.96*se
        else:
            ci_lo = ci_hi = np.nan
    except Exception:
        did_coef = ci_lo = ci_hi = np.nan

    return qdf, did_coef, ci_lo, ci_hi


# ══════════════════════════════════════════════════════════════════════════
# 1. Load data
# ══════════════════════════════════════════════════════════════════════════
print("Loading ADNIMERGE and computing slopes …")
adni = rdata_to_df(f"{ADNI_DIR}/adnimerge.rdata")
adni["RID"] = adni["RID"].astype(int)
all_slopes = compute_slopes(adni)

bl_mask = adni["VISCODE"].astype(str).str.strip().str.lower().isin(["bl","baseline"])
adni_bl = adni[bl_mask][["RID"]+CLINICAL].drop_duplicates("RID").copy()
adni_bl["RID"] = adni_bl["RID"].astype(int)
print(f"  Slopes: {len(all_slopes)}, baseline clinical: {len(adni_bl)}")

print("\nPivoting NULISA plasma …")
nulisa_wide, col_rename = pivot_nulisa(NULISA_CSV)
nulisa_wide["RID"] = nulisa_wide["RID"].astype(int)
prot_cols = [c for c in nulisa_wide.columns if c != "RID"]
print(f"  NULISA wide: {nulisa_wide.shape}, {len(prot_cols)} proteins")

print("\nBuilding drug flags from reccmeds …")
reccmeds = rdata_to_df(f"{ADNI_DIR}/reccmeds.rdata")
reccmeds["RID"] = reccmeds["RID"].astype(int)
nulisa_rids = set(nulisa_wide["RID"])
drug_flags = build_drug_flags(reccmeds, nulisa_rids)
print(f"  Drug flags built for {len(drug_flags)} patients")
for d in ALL_DRUGS:
    print(f"    {d}: {drug_flags[d].sum()} users")

# ══════════════════════════════════════════════════════════════════════════
# 2. Build full cohort dataframe
# ══════════════════════════════════════════════════════════════════════════
print("\nMerging full cohort …")
df = all_slopes.merge(adni_bl, on="RID", how="inner")
df = df.merge(nulisa_wide, on="RID", how="inner")
df = df.merge(drug_flags, on="RID", how="inner")
df = df.dropna(subset=["slope"])
df["RID"] = df["RID"].astype(int)
print(f"  Full cohort: n={len(df)}")

feature_cols = CLINICAL + prot_cols

# ══════════════════════════════════════════════════════════════════════════
# 3. Per-drug non-circular prediction + DiD
# ══════════════════════════════════════════════════════════════════════════
print(f"\n{'='*65}")
print("PLASMA DRUG STRATIFICATION (non-circular, train on non-users)")
print(f"{'='*65}")

results = []
quintile_dfs = {}

for drug in ALL_DRUGS:
    n_users = df[drug].sum()
    n_nonusers = (df[drug]==0).sum()
    if n_users < MIN_USERS:
        print(f"\n{drug}: n_users={n_users} < {MIN_USERS} — skip")
        continue

    print(f"\n── {drug}  (users={n_users}, non-users={n_nonusers}) ──")
    df[f"y_pred_{drug}"] = noncircular_predict(df, feature_cols, drug, PLASMA_GBM_PARAMS)[0]
    n_sel = noncircular_predict(df, feature_cols, drug, PLASMA_GBM_PARAMS)[1]

    qdf, did, ci_lo, ci_hi = did_analysis(df, drug, f"y_pred_{drug}")
    quintile_dfs[drug] = qdf

    direction = "✓" if did > 0 else "✗"
    class_label = ("CARDIOMETABOLIC" if drug in CARDIOMETABOLIC
                   else "CHOLINERGIC" if drug in CHOLINERGIC else "NEGATIVE CONTROL")
    print(f"  DiD = {did:+.3f} [{ci_lo:+.3f}, {ci_hi:+.3f}]  {direction}  [{class_label}]")

    results.append({
        "drug": drug, "class": class_label,
        "n_users": int(n_users), "n_nonusers": int(n_nonusers),
        "did": round(did, 4), "ci_lo": round(ci_lo, 4), "ci_hi": round(ci_hi, 4),
        "direction_positive": bool(did > 0),
    })

# ══════════════════════════════════════════════════════════════════════════
# 4. Summary
# ══════════════════════════════════════════════════════════════════════════
res_df = pd.DataFrame(results)
print(f"\n{'='*65}")
print("PLASMA DiD SUMMARY")
print(f"{'='*65}")

cardio = res_df[res_df["class"]=="CARDIOMETABOLIC"]
cholin = res_df[res_df["class"]=="CHOLINERGIC"]
neg    = res_df[res_df["class"]=="NEGATIVE CONTROL"]

print(f"\nCardiometabolic ({len(cardio)} drugs):")
for _, r in cardio.iterrows():
    d = "✓" if r["direction_positive"] else "✗"
    print(f"  {r['drug']:25s} DiD={r['did']:+.3f} [{r['ci_lo']:+.3f},{r['ci_hi']:+.3f}]  {d}")

print(f"\nCholinergic ({len(cholin)} drugs — negative controls):")
for _, r in cholin.iterrows():
    d = "✓" if r["direction_positive"] else "✗"
    print(f"  {r['drug']:25s} DiD={r['did']:+.3f} [{r['ci_lo']:+.3f},{r['ci_hi']:+.3f}]  {d}")

print(f"\nOther negative controls ({len(neg)} drugs):")
for _, r in neg.iterrows():
    d = "✓" if r["direction_positive"] else "✗"
    print(f"  {r['drug']:25s} DiD={r['did']:+.3f} [{r['ci_lo']:+.3f},{r['ci_hi']:+.3f}]  {d}")

n_cardio_pos = cardio["direction_positive"].sum()
n_cholin_pos = cholin["direction_positive"].sum()
print(f"\nDirectional consistency:")
print(f"  Cardiometabolic: {n_cardio_pos}/{len(cardio)} positive DiD")
print(f"  Cholinergic:     {n_cholin_pos}/{len(cholin)} positive DiD (expect 0)")
print(f"  Prob by chance:  (0.5)^{len(cardio)} = {0.5**len(cardio):.4f}")

res_df.to_csv(f"{OUT_DIR}/plasma_drug_did.csv", index=False)
print(f"\nSaved → plasma_drug_did.csv")
