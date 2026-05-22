#!/usr/bin/env python3
"""
Build Table 1: Cohort Characteristics — TMT (n=1,060) vs MRM (n=279)
Side-by-side with p-values (t-test continuous, chi-square categorical).
"""

import sys, os
import numpy as np
import pandas as pd
from scipy import stats

sys.stdout.reconfigure(line_buffering=True)

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(os.path.dirname(BASE))

DATA_CSV  = os.path.join(ROOT, "Data/ADNIMERGE/integrated/full_longitudinal_ALL_OMICS.csv")
DRUG_CSV  = os.path.join(ROOT, "FINAL_PUBLICATION/4_Paper/statistics/adni_drug_usage_matrix.csv")
TMT_IDS_CSV = os.path.join(BASE, "external_validation", "ADNI_IDs_inclGISes.csv")
MRM_CSV   = os.path.join(BASE, "results", "cdrsb_slope_dataset.csv")
OUT_CSV   = os.path.join(BASE, "tables", "table1_cohort_characteristics.csv")

print("Loading ADNI longitudinal data ...")
df = pd.read_csv(DATA_CSV, low_memory=False)
print(f"  {df['RID'].nunique()} patients, {len(df)} rows")

# ── CDRSB slope computation (same criteria as build_tmt_dataset.py) ─────────
def ols_slope(group):
    t = pd.to_numeric(group["MONTHS"], errors="coerce").values
    y = pd.to_numeric(group["CDRSB"],  errors="coerce").values
    mask = ~(np.isnan(t) | np.isnan(y))
    t, y = t[mask], y[mask]
    if len(t) < 3:
        return pd.Series({"slope": np.nan, "n_visits": len(t), "span_months": np.nan})
    span = t.max() - t.min()
    if span < 12:
        return pd.Series({"slope": np.nan, "n_visits": len(t), "span_months": span})
    res = stats.linregress(t, y)
    return pd.Series({"slope": res.slope, "n_visits": len(t), "span_months": span})

print("Computing CDRSB slopes ...")
slopes = df.groupby("RID", group_keys=False).apply(ols_slope).reset_index()
slopes = slopes.dropna(subset=["slope"])

# Clip ±3SD outliers
mu, sd = slopes["slope"].mean(), slopes["slope"].std()
slopes["slope"] = slopes["slope"].clip(mu - 3*sd, mu + 3*sd)
print(f"  {len(slopes)} patients with valid slopes")

# ── Baseline clinical features ───────────────────────────────────────────────
bl = df[df["VISCODE"] == "bl"].drop_duplicates(subset=["RID"], keep="first").copy()
for c in ["AGE", "PTEDUCAT", "APOE4", "ADAS13", "CDRSB", "MMSE", "FAQ", "AV45",
          "Hippocampus", "ICV"]:
    bl[c] = pd.to_numeric(bl[c], errors="coerce")
bl["Hippo_norm"] = bl["Hippocampus"] / bl["ICV"]

# DX at baseline (map to readable)
bl["DX_str"] = bl["DX"].astype(str).str.strip()

merged = slopes.merge(
    bl[["RID", "DX_str", "PTGENDER", "AGE", "PTEDUCAT", "APOE4",
        "ADAS13", "CDRSB", "MMSE", "FAQ", "AV45", "Hippo_norm"]],
    on="RID", how="inner"
)
print(f"  After clinical merge: {len(merged)} patients")

# ── Drug usage ───────────────────────────────────────────────────────────────
drugs = pd.read_csv(DRUG_CSV)

DRUG_CLASSES = {
    "statin":       ["atorvastatin", "simvastatin", "rosuvastatin", "pravastatin", "lovastatin"],
    "donepezil":    ["donepezil"],
    "memantine":    ["memantine"],
    "metformin":    ["metformin"],
    "nsaid":        ["aspirin", "ibuprofen", "naproxen", "celecoxib"],
    "ppi":          ["omeprazole"],
    "ace_inhibitor":["lisinopril"],
    "ccb":          ["amlodipine"],
    "thyroid":      ["levothyroxine"],
    "galantamine":  ["galantamine"],
    "rivastigmine": ["rivastigmine"],
}
for cls, members in DRUG_CLASSES.items():
    present = [m for m in members if m in drugs.columns]
    if present:
        drugs[f"drug_{cls}"] = drugs[present].max(axis=1)

drug_cols = [f"drug_{c}" for c in DRUG_CLASSES]
merged = merged.merge(drugs[["RID"] + drug_cols], on="RID", how="left")
for c in drug_cols:
    merged[c] = merged[c].fillna(0).astype(int)

# ── AV45: use strictly VISCODE=bl only (demographic reporting, not model feature) ──
# AV45 not in model features (dropped from CLINICAL); reported here as cohort characteristic only.
# TMT: 658/1061 have baseline AV45. MRM: 0/279 (PET acquired at follow-up visits only).
# merged["AV45"] already populated from bl merge above; no action needed.

# ── TMT cohort: patients in ADNI_IDs_inclGISes.csv ──────────────────────────
tmt_ids = pd.read_csv(TMT_IDS_CSV)
tmt_ids = tmt_ids[tmt_ids["RID"] != "GIS"].copy()
tmt_ids["RID"] = tmt_ids["RID"].astype(int)
tmt_rids = set(tmt_ids["RID"].unique())

tmt_df = merged[merged["RID"].isin(tmt_rids)].copy()
print(f"  TMT cohort: {len(tmt_df)} patients")

# ── MRM cohort from prebuilt dataset ────────────────────────────────────────
mrm_raw = pd.read_csv(MRM_CSV)
mrm_rids = set(mrm_raw["RID"].values)

# Recompute MRM stats from the full merged dataset for consistency
mrm_df = merged[merged["RID"].isin(mrm_rids)].copy()
print(f"  MRM cohort (from merged): {len(mrm_df)} patients")

# ── Helper: mean±SD formatter ────────────────────────────────────────────────
def fmt_mean_sd(series, decimals=1):
    s = series.dropna()
    if len(s) == 0:
        return "N/A"
    fmt = f"{{:.{decimals}f}}"
    return f"{fmt.format(s.mean())} ({fmt.format(s.std())})"

def fmt_n_pct(series, positive_val=1):
    s = series.dropna()
    n = (s == positive_val).sum()
    pct = 100 * n / len(s) if len(s) > 0 else 0
    return f"{n} ({pct:.1f}%)"

def ttest_p(a, b):
    a, b = a.dropna(), b.dropna()
    if len(a) < 2 or len(b) < 2:
        return np.nan
    _, p = stats.ttest_ind(a, b)
    return p

def chi2_p(a, b, positive_val=1):
    a, b = a.dropna(), b.dropna()
    n_pos_a, n_neg_a = (a == positive_val).sum(), (a != positive_val).sum()
    n_pos_b, n_neg_b = (b == positive_val).sum(), (b != positive_val).sum()
    table = [[n_pos_a, n_neg_a], [n_pos_b, n_neg_b]]
    if min(n_pos_a, n_pos_b, n_neg_a, n_neg_b) < 5:
        _, p = stats.fisher_exact(table)
    else:
        _, p, _, _ = stats.chi2_contingency(table)
    return p

def fmt_p(p):
    if np.isnan(p):
        return "—"
    if p < 0.001:
        return "<0.001"
    return f"{p:.3f}"

# ── Build table rows ─────────────────────────────────────────────────────────
rows = []

def add_continuous(label, col):
    rows.append({
        "Characteristic": label,
        "TMT Cohort (n=1061)": fmt_mean_sd(tmt_df[col]),
        "MRM Cohort (n=279)":  fmt_mean_sd(mrm_df[col]),
        "p-value": fmt_p(ttest_p(tmt_df[col], mrm_df[col])),
    })

def add_binary(label, col, pos=1):
    rows.append({
        "Characteristic": label,
        "TMT Cohort (n=1061)": fmt_n_pct(tmt_df[col], pos),
        "MRM Cohort (n=279)":  fmt_n_pct(mrm_df[col], pos),
        "p-value": fmt_p(chi2_p(tmt_df[col], mrm_df[col], pos)),
    })

# Demographics
add_continuous("Age (years)", "AGE")
add_continuous("Education (years)", "PTEDUCAT")

# Sex: female = "Female" or 2 depending on encoding
tmt_sex = tmt_df["PTGENDER"].astype(str).str.strip()
mrm_sex = mrm_df["PTGENDER"].astype(str).str.strip()
tmt_female = (tmt_sex.isin(["Female", "2"])).sum()
mrm_female = (mrm_sex.isin(["Female", "2"])).sum()
tmt_n, mrm_n = len(tmt_df), len(mrm_df)

# chi2 for sex
tmt_sex_bin = tmt_sex.isin(["Female", "2"]).astype(int)
mrm_sex_bin = mrm_sex.isin(["Female", "2"]).astype(int)
sex_p = chi2_p(tmt_sex_bin, mrm_sex_bin, 1)
rows.append({
    "Characteristic": "Female, n (%)",
    "TMT Cohort (n=1061)": f"{tmt_female} ({100*tmt_female/tmt_n:.1f}%)",
    "MRM Cohort (n=279)":  f"{mrm_female} ({100*mrm_female/mrm_n:.1f}%)",
    "p-value": fmt_p(sex_p),
})

# APOE4 is coded 0/1/2 alleles; carrier = any ε4 allele (>=1)
tmt_apoe4 = (tmt_df["APOE4"] >= 1).dropna()
mrm_apoe4 = (mrm_df["APOE4"] >= 1).dropna()
tmt_apoe4_n = tmt_apoe4.sum()
mrm_apoe4_n = mrm_apoe4.sum()
apoe4_p = chi2_p(tmt_apoe4.astype(int), mrm_apoe4.astype(int), 1)
rows.append({
    "Characteristic": "APOE4 carrier (≥1 allele), n (%)",
    "TMT Cohort (n=1061)": f"{tmt_apoe4_n} ({100*tmt_apoe4_n/len(tmt_df):.1f}%)",
    "MRM Cohort (n=279)":  f"{mrm_apoe4_n} ({100*mrm_apoe4_n/len(mrm_df):.1f}%)",
    "p-value": fmt_p(apoe4_p),
})

# Clinical
add_continuous("Baseline CDRSB, mean (SD)", "CDRSB")
add_continuous("Baseline MMSE, mean (SD)", "MMSE")
add_continuous("Baseline ADAS13, mean (SD)", "ADAS13")
add_continuous("Baseline FAQ, mean (SD)", "FAQ")
# AV45: available for TMT, not available at baseline for MRM
tmt_av45_n = tmt_df["AV45"].notna().sum()
mrm_av45_n = mrm_df["AV45"].notna().sum()
rows.append({
    "Characteristic": "AV45 (amyloid PET SUVR), mean (SD) [a]",
    "TMT Cohort (n=1061)": fmt_mean_sd(tmt_df["AV45"]) + f" [n={tmt_av45_n}]",
    "MRM Cohort (n=279)":  "N/A (not collected at baseline)",
    "p-value": "—",
})

# Hippo_norm: small ratio values, use 4 decimal places
def fmt_hippo(series):
    s = series.dropna()
    if len(s) == 0:
        return "N/A"
    return f"{s.mean():.4f} ({s.std():.4f}) [n={len(s)}]"
rows.append({
    "Characteristic": "Hippocampal volume/ICV, mean (SD)",
    "TMT Cohort (n=1061)": fmt_hippo(tmt_df["Hippo_norm"]),
    "MRM Cohort (n=279)":  fmt_hippo(mrm_df["Hippo_norm"]),
    "p-value": fmt_p(ttest_p(tmt_df["Hippo_norm"], mrm_df["Hippo_norm"])),
})

# Outcome and follow-up
rows.append({
    "Characteristic": "CDRSB slope (pts/yr), mean (SD)",
    "TMT Cohort (n=1061)": fmt_mean_sd(tmt_df["slope"] * 12),
    "MRM Cohort (n=279)":  fmt_mean_sd(mrm_df["slope"] * 12),
    "p-value": fmt_p(ttest_p(tmt_df["slope"], mrm_df["slope"])),
})
add_continuous("Follow-up duration (months), mean (SD)", "span_months")
add_continuous("CDRSB visits, mean (SD)", "n_visits")

# CSF proteins
rows.append({
    "Characteristic": "CSF proteins measured, n",
    "TMT Cohort (n=1061)": "2,492",
    "MRM Cohort (n=279)":  "320",
    "p-value": "—",
})

# DX breakdown
dx_order = ["CN", "MCI", "Dementia", "nan"]
for dx_val in ["CN", "MCI", "Dementia"]:
    tmt_n_dx = (tmt_df["DX_str"] == dx_val).sum()
    mrm_n_dx = (mrm_df["DX_str"] == dx_val).sum()
    rows.append({
        "Characteristic": f"  {dx_val}, n (%)",
        "TMT Cohort (n=1061)": f"{tmt_n_dx} ({100*tmt_n_dx/tmt_n:.1f}%)",
        "MRM Cohort (n=279)":  f"{mrm_n_dx} ({100*mrm_n_dx/mrm_n:.1f}%)",
        "p-value": "—",
    })

# Drug usage
rows.append({"Characteristic": "Drug Usage", "TMT Cohort (n=1061)": "", "MRM Cohort (n=279)": "", "p-value": ""})
DRUG_LABELS = {
    "drug_statin":        "  Statin, n (%)",
    "drug_donepezil":     "  Donepezil, n (%)",
    "drug_memantine":     "  Memantine, n (%)",
    "drug_metformin":     "  Metformin, n (%)",
    "drug_nsaid":         "  NSAID, n (%)",
    "drug_ppi":           "  PPI (proton pump inhibitor), n (%)",
    "drug_ace_inhibitor": "  ACE inhibitor, n (%)",
    "drug_ccb":           "  Calcium channel blocker, n (%)",
    "drug_thyroid":       "  Thyroid hormone, n (%)",
    "drug_galantamine":   "  Galantamine, n (%)",
    "drug_rivastigmine":  "  Rivastigmine, n (%)",
}
for col, label in DRUG_LABELS.items():
    if col in tmt_df.columns and col in mrm_df.columns:
        add_binary(label, col, 1)

# ── Footnotes ────────────────────────────────────────────────────────────────
rows.append({"Characteristic": "[a] AV45 and hippocampal volume/ICV not included in model features (zero importance; AV45 unavailable at baseline for MRM cohort). Reported here as cohort characteristics only. Model features: AGE, PTEDUCAT, APOE4, ADAS13, CDRSB, MMSE, FAQ (7 variables).",
             "TMT Cohort (n=1061)": "", "MRM Cohort (n=279)": "", "p-value": ""})

# ── Save ─────────────────────────────────────────────────────────────────────
out = pd.DataFrame(rows)
out.to_csv(OUT_CSV, index=False)
print(f"\nSaved: {OUT_CSV}")
print(f"\nTMT n={len(tmt_df)}, MRM n={len(mrm_df)}")
print("\nPreview:")
print(out.to_string(index=False))
