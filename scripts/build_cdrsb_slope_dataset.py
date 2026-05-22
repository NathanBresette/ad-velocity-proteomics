#!/usr/bin/env python3
"""
Phase 1: Build CDRSB Slope Dataset (CSF Proteomics)
=====================================================
Compute patient-level CDRSB slopes (OLS, >=3 visits, >=12 month span),
extract baseline clinical + CSF MRM proteomics features, merge drug usage flags.
"""

import sys, os
import numpy as np
import pandas as pd
from scipy import stats
from datetime import datetime

sys.stdout.reconfigure(line_buffering=True)

def tlog(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

# -- paths ------------------------------------------------------------------
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.environ.get("DIGITALTWIN_ROOT", os.path.dirname(os.path.dirname(BASE)))

DATA_CSV = os.path.join(ROOT, "Data/ADNIMERGE/integrated/full_longitudinal_ALL_OMICS.csv")
DRUG_CSV = os.path.join(ROOT, "FINAL_PUBLICATION/4_Paper/statistics/adni_drug_usage_matrix.csv")
CSF_CSV  = os.path.join(BASE, "results/csfmrm_raw.csv")

# Hellbender fallback paths
if not os.path.exists(DATA_CSV):
    DATA_CSV = "/home/nbhtd/data/digitaltwin/full_longitudinal_ALL_OMICS.csv"
if not os.path.exists(DRUG_CSV):
    DRUG_CSV = "/home/nbhtd/data/digitaltwin/adni_drug_usage_matrix.csv"
if not os.path.exists(CSF_CSV):
    CSF_CSV = "/home/nbhtd/scripts/8_CSF_Proteomics/results/csfmrm_raw.csv"

OUT_DIR = os.path.join(BASE, "results")
os.makedirs(OUT_DIR, exist_ok=True)

def ols_slope(group):
    t = group["MONTHS"].values
    y = group["CDRSB"].values
    mask = ~(np.isnan(t) | np.isnan(y))
    t, y = t[mask], y[mask]
    if len(t) < 3:
        return pd.Series({"slope": np.nan, "slope_se": np.nan, "n_visits": len(t), "span_months": np.nan})
    span = t.max() - t.min()
    if span < 12:
        return pd.Series({"slope": np.nan, "slope_se": np.nan, "n_visits": len(t), "span_months": span})
    res = stats.linregress(t, y)
    return pd.Series({"slope": res.slope, "slope_se": res.stderr, "n_visits": len(t), "span_months": span})

# -- load clinical data -----------------------------------------------------
tlog("Phase 1: Build CDRSB slope dataset (CSF Proteomics)")
tlog(f"Loading {DATA_CSV} ...")
df = pd.read_csv(DATA_CSV, low_memory=False)
tlog(f"  {len(df)} rows, {df.shape[1]} cols, {df['RID'].nunique()} patients")

# -- slopes ------------------------------------------------------------------
tlog("Computing CDRSB slopes ...")
df["CDRSB"] = pd.to_numeric(df["CDRSB"], errors="coerce")
df["MONTHS"] = pd.to_numeric(df["MONTHS"], errors="coerce")
slopes = df.groupby("RID", group_keys=False).apply(ols_slope).reset_index()
slopes = slopes.dropna(subset=["slope"])
tlog(f"  {len(slopes)} patients with valid CDRSB slopes (>=3 visits, >=12mo)")

# Also compute ADAS13 and MMSE slopes for independent outcome validation
def ols_slope_outcome(group, outcome_col):
    t = group["MONTHS"].values
    y = pd.to_numeric(group[outcome_col], errors="coerce").values
    mask = ~(np.isnan(t) | np.isnan(y))
    t, y = t[mask], y[mask]
    if len(t) < 3 or (t.max() - t.min()) < 12:
        return np.nan
    return stats.linregress(t, y).slope

tlog("Computing ADAS13 and MMSE slopes (independent outcomes) ...")
adas_slopes = df.groupby("RID").apply(lambda g: ols_slope_outcome(g, "ADAS13")).reset_index()
adas_slopes.columns = ["RID", "adas13_slope"]
mmse_slopes = df.groupby("RID").apply(lambda g: ols_slope_outcome(g, "MMSE")).reset_index()
mmse_slopes.columns = ["RID", "mmse_slope"]
slopes = slopes.merge(adas_slopes, on="RID", how="left").merge(mmse_slopes, on="RID", how="left")
n_adas = slopes["adas13_slope"].notna().sum()
n_mmse = slopes["mmse_slope"].notna().sum()
tlog(f"  ADAS13 slopes: {n_adas}, MMSE slopes: {n_mmse}")

mu, sd = slopes["slope"].mean(), slopes["slope"].std()
lo, hi = mu - 3*sd, mu + 3*sd
n_clip = ((slopes["slope"] < lo) | (slopes["slope"] > hi)).sum()
slopes["slope"] = slopes["slope"].clip(lo, hi)
tlog(f"  Clipped {n_clip} outliers at +/-3SD [{lo:.4f}, {hi:.4f}]")

tlog(f"  Mean={slopes['slope'].mean():.4f}, SD={slopes['slope'].std():.4f}, "
     f"Median={slopes['slope'].median():.4f}")

# -- baseline clinical features ----------------------------------------------
tlog("Extracting baseline clinical features ...")
bl = df[df["VISCODE"] == "bl"].drop_duplicates(subset=["RID"], keep="first").copy()

CLINICAL = ["AGE", "PTEDUCAT", "APOE4", "ADAS13", "CDRSB", "MMSE", "FAQ"]
for c in CLINICAL:
    bl[c] = pd.to_numeric(bl[c], errors="coerce")

# -- CSF MRM proteomics -----------------------------------------------------
tlog(f"Loading CSF MRM proteomics from {CSF_CSV} ...")
csf = pd.read_csv(CSF_CSV, low_memory=False)
tlog(f"  {len(csf)} rows, {csf.shape[1]} cols, {csf['RID'].nunique()} unique RIDs")

# Identify protein feature columns (numeric, not metadata)
CSF_SKIP = {"RID", "PTID", "VISCODE", "VISCODE2", "EXAMDATE", "Phase", "COLPROT",
            "ORIGPROT", "SITEID", "update_stamp", "VID", "RUN.NUMBER", "SUBRUN",
            "MS.PLATE.INDEX", "RUN.INJECTION.NUMBER", "PLATE.COLUMN", "PLATE.ROW"}
csf_feat_cols = [c for c in csf.columns if c not in CSF_SKIP
                 and csf[c].dtype in ['float64', 'int64', 'float32', 'int32']]

# Get baseline CSF per patient (prefer 'bl' visit)
csf_bl = csf[csf["VISCODE"].isin(["bl", "sc", "scmri"])] if "VISCODE" in csf.columns else csf
if len(csf_bl) == 0:
    csf_bl = csf
csf_bl = csf_bl.groupby("RID").first().reset_index()
tlog(f"  {len(csf_bl)} patients with baseline CSF")

# Filter features: >=30% coverage, not constant
for c in csf_feat_cols:
    csf_bl[c] = pd.to_numeric(csf_bl[c], errors="coerce")
coverage = csf_bl[csf_feat_cols].notna().mean()
keep_csf = coverage[coverage >= 0.30].index.tolist()
keep_csf = [c for c in keep_csf if csf_bl[c].nunique() > 1]
tlog(f"  {len(keep_csf)}/{len(csf_feat_cols)} CSF features with >=30% coverage")

# Prefix CSF columns for clarity
csf_rename = {c: f"CSF_{c}" for c in keep_csf}
csf_bl = csf_bl.rename(columns=csf_rename)
keep_csf_prefixed = [f"CSF_{c}" for c in keep_csf]

# -- merge slopes + clinical + CSF + drugs -----------------------------------
tlog("Merging slopes + baseline clinical + CSF + drugs ...")
keep_cols = ["RID", "DX", "PTGENDER"] + CLINICAL
merged = slopes.merge(bl[keep_cols], on="RID", how="inner")
tlog(f"  After clinical merge: {len(merged)}")

# Merge CSF
csf_merge_cols = ["RID"] + keep_csf_prefixed
merged = merged.merge(csf_bl[csf_merge_cols], on="RID", how="inner")
tlog(f"  After CSF merge: {len(merged)} (patients with both slopes + CSF)")

# Drug usage
drugs = pd.read_csv(DRUG_CSV)
DRUG_CLASSES = {
    "statin": ["atorvastatin", "simvastatin", "rosuvastatin", "pravastatin", "lovastatin"],
    "donepezil": ["donepezil"], "memantine": ["memantine"], "metformin": ["metformin"],
    "nsaid": ["aspirin", "ibuprofen", "naproxen", "celecoxib"],
    "ppi": ["omeprazole"], "ace_inhibitor": ["lisinopril"], "ccb": ["amlodipine"],
    "thyroid": ["levothyroxine"], "galantamine": ["galantamine"], "rivastigmine": ["rivastigmine"],
}
for cls, members in DRUG_CLASSES.items():
    present = [m for m in members if m in drugs.columns]
    if present:
        drugs[f"drug_{cls}"] = drugs[present].max(axis=1)
drug_class_cols = [c for c in drugs.columns if c.startswith("drug_")]
merged = merged.merge(drugs[["RID"] + drug_class_cols], on="RID", how="left")
for c in drug_class_cols:
    merged[c] = merged[c].fillna(0).astype(int)

tlog(f"  Final: {len(merged)} patients, {merged.shape[1]} columns")
for c in drug_class_cols:
    tlog(f"    {c}: {merged[c].sum()} users")

out_path = os.path.join(OUT_DIR, "cdrsb_slope_dataset.csv")
merged.to_csv(out_path, index=False)
tlog(f"Saved -> {out_path}")

# Save column lists for downstream scripts
import json
col_info = {
    "clinical": CLINICAL,
    "csf": keep_csf_prefixed,
    "drug_classes": drug_class_cols,
}
with open(os.path.join(OUT_DIR, "column_lists.json"), "w") as f:
    json.dump(col_info, f, indent=2)
tlog("Saved -> column_lists.json")

# Extract unique protein names from CSF features
protein_names = set()
for c in keep_csf:
    parts = c.split(".")
    protein_names.add(parts[0])
tlog(f"  {len(protein_names)} unique CSF proteins retained")

dx_counts = merged["DX"].value_counts()
for dx, n in dx_counts.items():
    tlog(f"  {dx}: {n} ({100*n/len(merged):.1f}%)")
tlog("Phase 1 DONE.")
