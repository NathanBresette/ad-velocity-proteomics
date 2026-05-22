"""
NULISA plasma proteomics — apples-to-apples vs CSF cross-platform model.

Train/test split mirrors CSF Model C exactly:
  Test  (n=279): same MRM patients where CSF achieved R²=0.275
  Train (n=~1149): remaining NULISA plasma patients (zero sample overlap)

Model A: 6 CSF LASSO-selected proteins (UCHL1, FABP3, YWHAZ, NPTX2, S100B, SMOC1) + 7 clinical
         No re-selection — directly tests whether CSF-identified biology transfers to plasma.

Model B: All 120 NULISA proteins + 7 clinical, LASSO re-selects from scratch.
         If LASSO independently picks UCHL1/NPTX2/YWHAZ → independent biological validation.

Key sentence: "In the exact 279 patients where cross-platform CSF achieved R²=0.275,
               the biologically guided plasma panel achieved R²=X."
"""

import subprocess, os, tempfile, json
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV, KFold
from sklearn.metrics import r2_score
from scipy import stats
import shap
import warnings
warnings.filterwarnings("ignore")

# ── paths ──────────────────────────────────────────────────────────────────
DATA_DIR   = "/Users/nathanbresette/Desktop/DigitalTwin/Data"
ADNI_DIR   = f"{DATA_DIR}/ADNIMERGE/data"
NULISA_CSV = f"{DATA_DIR}/BSHRI_PLA_CSF_NULISA_CNS_05May2026.csv"
MRM_CSV    = "/Users/nathanbresette/Desktop/DigitalTwin/FINAL_PUBLICATION/8_CSF_Proteomics/results/cdrsb_slope_dataset.csv"
OUT_DIR    = "/Users/nathanbresette/Desktop/DigitalTwin/FINAL_PUBLICATION/8_CSF_Proteomics/results"
os.makedirs(OUT_DIR, exist_ok=True)

# ── constants ──────────────────────────────────────────────────────────────
# CSF GBM_PARAMS kept for reference only — plasma uses tuned params below
CSF_GBM_PARAMS = dict(
    n_estimators=500, max_depth=4, learning_rate=0.05,
    min_samples_leaf=10, subsample=0.8, random_state=42
)

# Plasma hyperparameter search space — tuned on training set only
PLASMA_PARAM_DIST = {
    "n_estimators":    [200, 300, 500, 750, 1000],
    "max_depth":       [2, 3, 4, 5],
    "learning_rate":   [0.01, 0.02, 0.05, 0.1],
    "min_samples_leaf":[5, 10, 20, 30],
    "subsample":       [0.6, 0.7, 0.8, 0.9],
}

CLINICAL = ["AGE", "PTEDUCAT", "APOE4", "ADAS13", "CDRSB", "MMSE", "FAQ"]

# 6 CSF LASSO-selected proteins confirmed in NULISA panel (from Appendix 2)
CSF_PROTEINS = ["UCHL1", "FABP3", "YWHAZ", "NPTX2", "S100B", "SMOC1"]

CSF_BENCHMARK_R2 = 0.275
CSF_BENCHMARK_N  = 279

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
        raise RuntimeError(f"Rscript failed for {path}: {r.stderr}")
    df = pd.read_csv(tmp)
    os.remove(tmp)
    return df


def compute_slopes(adni, min_visits=3, min_span_months=12):
    """OLS CDR-SB slope per patient; same method as CSF pipeline."""
    adni = adni[["RID", "M", "CDRSB"]].copy()
    adni["RID"] = adni["RID"].astype(int)
    adni["M"]   = pd.to_numeric(adni["M"], errors="coerce")
    adni = adni.dropna(subset=["CDRSB", "M"])
    slopes = []
    for rid, grp in adni.groupby("RID"):
        grp = grp.sort_values("M")
        if len(grp) < min_visits:
            continue
        if grp["M"].max() - grp["M"].min() < min_span_months:
            continue
        slope, *_ = stats.linregress(grp["M"].values, grp["CDRSB"].values)
        slopes.append({"RID": rid, "slope": slope})
    return pd.DataFrame(slopes)


def pivot_nulisa_plasma(nulisa_csv, rid_set=None):
    """Long → wide (RID × protein); baseline visit only; plasma samples only."""
    df = pd.read_csv(nulisa_csv, low_memory=False)
    df = df[(df["SampleType"] == "Sample") & (df["SampleMatrixType"] == "PLASMA")]
    if rid_set is not None:
        df = df[df["RID"].isin(rid_set)]

    # Prefer bl; fall back to m06
    norm = df["VISCODE2"].astype(str).str.strip().str.lower()
    bl   = df[norm == "bl"].copy()
    m06  = df[norm == "m06"].copy()
    rids_with_bl = set(bl["RID"])
    baseline = pd.concat([bl, m06[~m06["RID"].isin(rids_with_bl)]], ignore_index=True)

    baseline["NPQ"] = pd.to_numeric(baseline["NPQ"], errors="coerce")
    baseline["RID"] = baseline["RID"].astype(int)

    wide = baseline.pivot_table(index="RID", columns="Target", values="NPQ", aggfunc="mean")
    wide = wide.reset_index()
    wide.columns.name = None

    # Sanitize column names and add PLASMAp_ prefix to avoid collision with
    # clinical variables (e.g., NULISA has APOE4 as a protein NPQ value,
    # which would conflict with the clinical APOE4 allele count variable).
    rename = {}
    for c in wide.columns:
        if c == "RID":
            continue
        safe = c.replace("β", "b").replace("-", "_").replace(" ", "_").replace("(", "").replace(")", "")
        rename[c] = f"PLASMAp_{safe}"
    wide.rename(columns=rename, inplace=True)

    return wide, rename


def tune_gbm(X_train, y_train, n_iter=60):
    """RandomizedSearchCV on training set only — never touches test set."""
    imp = SimpleImputer(strategy="median").fit(X_train)
    Xtr = StandardScaler().fit_transform(imp.transform(X_train))
    search = RandomizedSearchCV(
        GradientBoostingRegressor(random_state=42),
        PLASMA_PARAM_DIST,
        n_iter=n_iter,
        cv=KFold(n_splits=5, shuffle=True, random_state=42),
        scoring="r2",
        n_jobs=-1,
        random_state=42,
        verbose=0,
    )
    search.fit(Xtr, y_train)
    print(f"  Best CV R² (tuning): {search.best_score_:.3f}")
    print(f"  Best params: {search.best_params_}")
    return search.best_params_


def holdout_r2(X_train, y_train, X_test, y_test, label, use_lasso=True, gbm_params=None):
    """Train on train set, evaluate on held-out test set."""
    params = gbm_params if gbm_params is not None else CSF_GBM_PARAMS
    imp = SimpleImputer(strategy="median").fit(X_train)
    Xtr = imp.transform(X_train)
    Xte = imp.transform(X_test)
    sc  = StandardScaler().fit(Xtr)
    Xtr_s = sc.transform(Xtr)
    Xte_s = sc.transform(Xte)

    if use_lasso:
        lasso = LassoCV(cv=3, max_iter=10000, random_state=42).fit(Xtr_s, y_train)
        mask  = lasso.coef_ != 0
        n_sel = int(mask.sum())
        if n_sel == 0:
            print(f"  {label}: LASSO selected 0 features — returning R²=0")
            return 0.0, 0.0, 1.0, None, None
    else:
        mask  = np.ones(X_train.shape[1], dtype=bool)
        n_sel = int(mask.sum())

    model  = GradientBoostingRegressor(**{**params, "random_state": 42}).fit(Xtr_s[:, mask], y_train)
    y_pred = model.predict(Xte_s[:, mask])

    r2      = float(r2_score(y_test, y_pred))
    r, p    = stats.pearsonr(y_test, y_pred)
    print(f"  {label}: R²={r2:.3f}, r={r:.3f}, p={p:.2e}, n_test={len(y_test)}, n_features={n_sel}")
    return r2, float(r), float(p), model, (imp, sc, mask)


# ══════════════════════════════════════════════════════════════════════════
# 1. Load MRM test set (279 patients) — pre-computed slopes + clinical
# ══════════════════════════════════════════════════════════════════════════
print("Loading MRM test set …")
mrm = pd.read_csv(MRM_CSV)
mrm["RID"] = mrm["RID"].astype(int)
mrm_rids = set(mrm["RID"])
print(f"  MRM patients: {len(mrm_rids)}")

# ══════════════════════════════════════════════════════════════════════════
# 2. Pivot NULISA plasma → wide format
# ══════════════════════════════════════════════════════════════════════════
print("\nPivoting NULISA plasma to wide format …")
nulisa_wide, col_rename = pivot_nulisa_plasma(NULISA_CSV)
nulisa_wide["RID"] = nulisa_wide["RID"].astype(int)
nulisa_rids = set(nulisa_wide["RID"])

# Apply same rename to CSF_PROTEINS list
csf_cols = [col_rename.get(p, f"PLASMAp_{p}") for p in CSF_PROTEINS]
# Verify they're in the pivoted data
csf_cols = [c for c in csf_cols if c in nulisa_wide.columns]
print(f"  NULISA wide: {nulisa_wide.shape}")
print(f"  CSF-mapped plasma columns found: {csf_cols}")

# All protein columns (exclude RID)
all_prot_cols = [c for c in nulisa_wide.columns if c != "RID"]
print(f"  Total protein columns: {len(all_prot_cols)}")

# ══════════════════════════════════════════════════════════════════════════
# 3. Build training set — non-MRM NULISA patients
#    Slopes: computed from ADNIMERGE (same OLS method as CSF pipeline)
#    Clinical: ADNIMERGE baseline
# ══════════════════════════════════════════════════════════════════════════
print("\nBuilding training set from ADNIMERGE …")
adni = rdata_to_df(f"{ADNI_DIR}/adnimerge.rdata")
adni["RID"] = adni["RID"].astype(int)

# Compute slopes for ALL patients in ADNIMERGE
all_slopes = compute_slopes(adni)
all_slopes["RID"] = all_slopes["RID"].astype(int)
print(f"  Slopes computed: {len(all_slopes)} patients")

# Clinical baseline from ADNIMERGE
bl_mask = adni["VISCODE"].astype(str).str.strip().str.lower().isin(["bl", "baseline"])
adni_bl = adni[bl_mask][["RID"] + CLINICAL].drop_duplicates("RID").copy()
adni_bl["RID"] = adni_bl["RID"].astype(int)

# Training RIDs: NULISA patients NOT in MRM test set
train_rids = nulisa_rids - mrm_rids
print(f"  Candidate train RIDs (non-MRM): {len(train_rids)}")

# Build train dataframe
train_df = all_slopes[all_slopes["RID"].isin(train_rids)].copy()
train_df = train_df.merge(adni_bl, on="RID", how="inner")
train_df = train_df.merge(nulisa_wide, on="RID", how="inner")
train_df = train_df.dropna(subset=["slope"])
print(f"  Train patients with slope + clinical + plasma: {len(train_df)}")

# ══════════════════════════════════════════════════════════════════════════
# 4. Build test set — exact 279 MRM patients
#    Slopes and clinical from pre-computed cdrsb_slope_dataset.csv
# ══════════════════════════════════════════════════════════════════════════
print("\nBuilding test set (279 MRM patients) …")
test_df = mrm[["RID", "slope"] + CLINICAL].copy()
test_df = test_df.merge(nulisa_wide, on="RID", how="inner")
test_df = test_df.dropna(subset=["slope"])
print(f"  Test patients with slope + clinical + plasma: {len(test_df)}")

# Confirm zero overlap
overlap = set(train_df["RID"]) & set(test_df["RID"])
assert len(overlap) == 0, f"OVERLAP DETECTED: {overlap}"
print(f"  Sample overlap between train and test: 0 ✓")

# ══════════════════════════════════════════════════════════════════════════
# 5. Align feature columns between train and test
# ══════════════════════════════════════════════════════════════════════════
y_train = train_df["slope"].values
y_test  = test_df["slope"].values

X_train_clin = train_df[CLINICAL].values
X_test_clin  = test_df[CLINICAL].values

# Model A feature columns: 6 CSF proteins present in both train and test
modelA_cols = [c for c in csf_cols if c in train_df.columns and c in test_df.columns]
print(f"\nModel A protein cols: {modelA_cols}")

X_train_A = train_df[CLINICAL + modelA_cols].values
X_test_A  = test_df[CLINICAL + modelA_cols].values

# Model B: all proteins present in both datasets
prot_shared = [c for c in all_prot_cols if c in train_df.columns and c in test_df.columns]
print(f"Model B protein cols: {len(prot_shared)} proteins")

X_train_B = train_df[CLINICAL + prot_shared].values
X_test_B  = test_df[CLINICAL + prot_shared].values

# ══════════════════════════════════════════════════════════════════════════
# 6. Hyperparameter tuning — training set only, never touches test set
# ══════════════════════════════════════════════════════════════════════════
print("\n── Hyperparameter tuning on plasma training set (Model B features) ──")
print("  Running RandomizedSearchCV (n_iter=60, 5-fold CV) …")
best_params = tune_gbm(X_train_B, y_train, n_iter=60)

print("\n── Hyperparameter tuning on Model A features ──")
best_params_A = tune_gbm(X_train_A, y_train, n_iter=60)

print("\n── Hyperparameter tuning on clinical-only features ──")
best_params_clin = tune_gbm(X_train_clin, y_train, n_iter=60)

# Save tuned params
with open(f"{OUT_DIR}/nulisa_plasma_tuned_params.json", "w") as f:
    json.dump({"model_B": best_params, "model_A": best_params_A, "clinical": best_params_clin}, f, indent=2)
print(f"\n  Saved tuned params → nulisa_plasma_tuned_params.json")

# ══════════════════════════════════════════════════════════════════════════
# 7. Clinical-only baseline (tuned)
# ══════════════════════════════════════════════════════════════════════════
print("\n── Baseline: Clinical-only (tuned GBM, no LASSO) ──")
r2_clin, r_clin, p_clin, _, _ = holdout_r2(
    X_train_clin, y_train, X_test_clin, y_test,
    label="Clinical-only", use_lasso=False, gbm_params=best_params_clin
)

# ══════════════════════════════════════════════════════════════════════════
# 8. Model A: 6 CSF-mapped proteins + clinical (no LASSO, tuned GBM)
# ══════════════════════════════════════════════════════════════════════════
print("\n── Model A: 6 CSF-mapped proteins + clinical (tuned GBM, direct) ──")
r2_A, r_A, p_A, model_A, pipe_A = holdout_r2(
    X_train_A, y_train, X_test_A, y_test,
    label="Model A (CSF-anchored, tuned)", use_lasso=False, gbm_params=best_params_A
)

# ══════════════════════════════════════════════════════════════════════════
# 9. Model B: All 120 NULISA proteins + clinical (LASSO + tuned GBM)
# ══════════════════════════════════════════════════════════════════════════
print("\n── Model B: All NULISA proteins + clinical (LASSO + tuned GBM) ──")
r2_B, r_B, p_B, model_B, pipe_B = holdout_r2(
    X_train_B, y_train, X_test_B, y_test,
    label="Model B (LASSO unbiased, tuned)", use_lasso=True, gbm_params=best_params
)

# ══════════════════════════════════════════════════════════════════════════
# 9. SHAP on Model B — does LASSO rediscover CSF proteins?
# ══════════════════════════════════════════════════════════════════════════
if model_B is not None and pipe_B is not None:
    print("\n── SHAP: Model B feature importance ──")
    imp_B, sc_B, mask_B = pipe_B
    feat_names_B = [f for f, m in zip(CLINICAL + prot_shared, mask_B) if m]

    Xtr_imp = imp_B.transform(X_train_B)
    Xtr_s   = sc_B.transform(Xtr_imp)
    Xsel    = Xtr_s[:, mask_B]

    expl   = shap.TreeExplainer(model_B)
    sv     = expl.shap_values(Xsel)
    mean_abs = np.abs(sv).mean(axis=0)

    shap_df = pd.DataFrame({"feature": feat_names_B, "shap_mean_abs": mean_abs})
    shap_df = shap_df.sort_values("shap_mean_abs", ascending=False)

    print("\n  Top 20 SHAP features (Model B):")
    print(shap_df.head(20).to_string(index=False))

    # Flag CSF-mapped proteins in SHAP ranking
    shap_df["is_csf_mapped"] = shap_df["feature"].isin(csf_cols)
    csf_in_shap = shap_df[shap_df["is_csf_mapped"]]
    print(f"\n  CSF-mapped proteins selected by Model B LASSO:")
    print(csf_in_shap.to_string(index=False))

    shap_df.to_csv(f"{OUT_DIR}/shap_nulisa_plasma_modelB.csv", index=False)

# ══════════════════════════════════════════════════════════════════════════
# 10. Permutation null for Model A
# ══════════════════════════════════════════════════════════════════════════
print("\n── Permutation null (Model A, 1000 permutations) ──")
imp_A, sc_A, mask_A_or_all = pipe_A if pipe_A else (
    SimpleImputer(strategy="median").fit(X_train_A),
    StandardScaler().fit(SimpleImputer(strategy="median").fit(X_train_A).transform(X_train_A)),
    np.ones(X_train_A.shape[1], dtype=bool)
)

imp_perm = SimpleImputer(strategy="median").fit(X_train_A)
Xtr_perm = StandardScaler().fit(imp_perm.transform(X_train_A)).transform(imp_perm.transform(X_train_A))
sc_perm  = StandardScaler().fit(imp_perm.transform(X_train_A))
Xte_perm = sc_perm.transform(imp_perm.transform(X_test_A))

model_perm = GradientBoostingRegressor(**{**best_params_A, "random_state": 42}).fit(Xtr_perm, y_train)

rng = np.random.default_rng(42)
null_r2 = []
for _ in range(1000):
    y_shuf = rng.permutation(y_test)
    null_r2.append(r2_score(y_shuf, model_perm.predict(Xte_perm)))

perm_p = float(np.mean(np.array(null_r2) >= r2_A))
print(f"  Model A R²={r2_A:.3f}, permutation p={perm_p:.4f} ({np.sum(np.array(null_r2) >= r2_A)}/1000)")

# ══════════════════════════════════════════════════════════════════════════
# 11. CSF vs plasma prediction correlation on 279 test patients
# ══════════════════════════════════════════════════════════════════════════
print("\n── CSF vs plasma prediction correlation ──")
csf_preds_path = f"{OUT_DIR}/cv_predictions_B_clinical_csf.csv"
if os.path.exists(csf_preds_path):
    csf_pred_df = pd.read_csv(csf_preds_path)
    csf_pred_df["RID"] = csf_pred_df["RID"].astype(int)

    # Get plasma Model B predictions on test set
    imp_B2 = SimpleImputer(strategy="median").fit(X_train_B)
    sc_B2  = StandardScaler().fit(imp_B2.transform(X_train_B))
    lasso_B2 = LassoCV(cv=3, max_iter=10000, random_state=42).fit(
        sc_B2.transform(imp_B2.transform(X_train_B)), y_train)
    mask_B2 = lasso_B2.coef_ != 0
    Xtr_B2  = sc_B2.transform(imp_B2.transform(X_train_B))[:, mask_B2]
    Xte_B2  = sc_B2.transform(imp_B2.transform(X_test_B))[:, mask_B2]
    mdl_B2  = GradientBoostingRegressor(**{**best_params, "random_state": 42}).fit(Xtr_B2, y_train)
    y_pred_B_test = mdl_B2.predict(Xte_B2)

    pred_df = pd.DataFrame({"RID": test_df["RID"].values, "y_true": y_test,
                             "y_pred_plasma_B": y_pred_B_test})
    merged = pred_df.merge(csf_pred_df[["RID","y_pred"]].rename(columns={"y_pred":"y_pred_csf"}),
                           on="RID", how="inner")
    if len(merged) > 10:
        r_csf_plasma, p_csf_plasma = stats.pearsonr(merged["y_pred_csf"], merged["y_pred_plasma_B"])
        print(f"  CSF vs plasma-B prediction correlation: r={r_csf_plasma:.3f}, p={p_csf_plasma:.2e} (n={len(merged)})")
    else:
        print(f"  CSF predictions file found but insufficient overlap (n={len(merged)})")
        r_csf_plasma, p_csf_plasma = None, None
else:
    print(f"  CSF predictions file not found at {csf_preds_path} — skipping correlation")
    r_csf_plasma, p_csf_plasma = None, None
    y_pred_B_test = None
    pred_df = pd.DataFrame({"RID": test_df["RID"].values, "y_true": y_test})

# ══════════════════════════════════════════════════════════════════════════
# 12. Save predictions
# ══════════════════════════════════════════════════════════════════════════
imp_save  = SimpleImputer(strategy="median").fit(X_train_A)
sc_save   = StandardScaler().fit(imp_save.transform(X_train_A))
mdl_save  = GradientBoostingRegressor(**{**best_params_A, "random_state": 42}).fit(
    sc_save.transform(imp_save.transform(X_train_A)), y_train)
y_pred_A_test = mdl_save.predict(sc_save.transform(imp_save.transform(X_test_A)))

out_df = pd.DataFrame({"RID": test_df["RID"].values, "y_true": y_test,
                        "y_pred_modelA": y_pred_A_test})
if y_pred_B_test is not None:
    out_df["y_pred_modelB"] = y_pred_B_test
out_df.to_csv(f"{OUT_DIR}/nulisa_plasma_predictions.csv", index=False)

print("\n" + "═"*60)
print("NULISA PLASMA ANALYSIS SUMMARY (TUNED)")
print("═"*60)
print(f"  Train n (ADNI1/2/3/GO, non-MRM) : {len(train_df)}")
print(f"  Test n  (exact MRM 279 patients) : {len(test_df)}")
print(f"  Sample overlap                   : 0 ✓")
print()
print(f"  Clinical-only         R² = {r2_clin:.3f}  r = {r_clin:.3f}  p = {p_clin:.2e}")
print(f"  Model A (CSF-anchor)  R² = {r2_A:.3f}  r = {r_A:.3f}  p = {p_A:.2e}  (perm_p={perm_p:.4f})")
print(f"  Model B (LASSO all)   R² = {r2_B:.3f}  r = {r_B:.3f}  p = {p_B:.2e}")
print()
print(f"  CSF cross-platform benchmark (same 279 pts): R² = {CSF_BENCHMARK_R2}")
print(f"  Model A as % of CSF benchmark : {r2_A/CSF_BENCHMARK_R2*100:.1f}%")
print(f"  Model B as % of CSF benchmark : {r2_B/CSF_BENCHMARK_R2*100:.1f}%")
if r_csf_plasma is not None:
    print(f"  CSF vs plasma-B prediction r  : {r_csf_plasma:.3f}  p={p_csf_plasma:.2e}")
print("═"*60)

results_out = {
    "n_train": int(len(train_df)), "n_test": int(len(test_df)),
    "r2_clinical": round(r2_clin, 4), "r_clinical": round(r_clin, 4),
    "r2_modelA": round(r2_A, 4), "r_modelA": round(r_A, 4), "p_modelA": float(p_A),
    "perm_p_modelA": round(perm_p, 4),
    "r2_modelB": round(r2_B, 4), "r_modelB": round(r_B, 4), "p_modelB": float(p_B),
    "csf_benchmark_r2": CSF_BENCHMARK_R2,
    "modelA_pct_of_csf": round(r2_A/CSF_BENCHMARK_R2*100, 1),
    "modelB_pct_of_csf": round(r2_B/CSF_BENCHMARK_R2*100, 1),
    "r_csf_vs_plasma_B": round(r_csf_plasma, 4) if r_csf_plasma else None,
}
with open(f"{OUT_DIR}/nulisa_plasma_performance.json", "w") as f:
    json.dump(results_out, f, indent=2)
print(f"\nSaved → nulisa_plasma_performance.json")
