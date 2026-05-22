#!/usr/bin/env python3
"""
Phase 5: Biological Interpretation (CSF Proteomics)
=====================================================
Feature importance ranking, biological categories for CSF proteins,
correlation with slope, fast vs slow progressor analysis, drug-protein heatmap.
"""

import sys, os, json
import numpy as np
import pandas as pd
from datetime import datetime
from scipy.stats import pearsonr, ttest_ind
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

sys.stdout.reconfigure(line_buffering=True)

def tlog(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(BASE, "results")
FIG_DIR = os.path.join(BASE, "figures")
os.makedirs(FIG_DIR, exist_ok=True)
DATA = os.path.join(OUT_DIR, "cdrsb_slope_dataset.csv")
IMP_FILE = os.path.join(OUT_DIR, "feature_importances.csv")

# CSF protein biological categories
# Proteins are named like CSF_PROTEIN.PEPTIDE -- we categorize by protein
CSF_CATEGORIES = {
    "AD Pathology (Amyloid/Tau)": [
        "A4", "BACE1", "APLP2", "CLUS", "TTHY", "APOE",
    ],
    "Complement & Innate Immunity": [
        "CO2", "CO3", "CO4A", "CO5", "CO6", "CO8B", "C1QB", "CFAB", "CD14", "CRP",
    ],
    "Neuroinflammation": [
        "CH3L1", "TGFB1", "I18BP", "OSTP", "CD59",
    ],
    "Synaptic & Neuronal Markers": [
        "NPTX1", "NPTX2", "NPTXR", "VGF", "SCG1", "SCG2", "SCG3",
        "NCAM1", "NCAM2", "L1CAM", "NRCAM", "CNTN1", "CNTN2",
        "NRX1A", "NRX2A", "NRX3A", "GRIA4", "BASP1",
    ],
    "Proteases & Inhibitors": [
        "A1AT", "AACT", "KAIN", "TIMP1", "KLK6", "KLK10", "KLK11",
        "CATD", "CATL1", "PLMN", "PCSK1",
    ],
    "Oxidative Stress": [
        "SODC", "SODE", "PRDX1", "PRDX2", "PRDX3", "PRDX6", "CATA",
    ],
    "Apolipoproteins & Lipid Transport": [
        "APOB", "APOD", "AFAM", "VTDB",
    ],
    "Cell Adhesion & ECM": [
        "CAD13", "CADM3", "DAG1", "FBLN1", "FBLN3", "LAMB2", "FMOD",
        "MIME", "NCAN", "NEGR1", "SPON1", "SPRL1", "LTBP2",
    ],
    "Neuropeptides & Secretory": [
        "CMGA", "PDYN", "NGF", "CCKN", "PTGDS",
    ],
    "Neurodegeneration Markers": [
        "GFAP", "ENOG", "FABPH", "NELL2", "PIMT",
    ],
}

# Drug -> CSF protein effect mapping (from published RCT/clinical trial data)
DRUG_CSF_MAP = {
    "statin": {
        "targets": ["APOB", "CO3", "CRP", "APOE"],
        "mechanism": "Reduce CSF APOB, lower complement/CRP via anti-inflammatory",
    },
    "metformin": {
        "targets": ["SODC", "SODE", "PRDX1", "CATA", "TGFB1"],
        "mechanism": "Reduce oxidative stress markers, modulate neuroinflammation",
    },
    "nsaid": {
        "targets": ["CRP", "CO3", "CO4A", "CH3L1", "PTGDS", "CD14"],
        "mechanism": "Reduce neuroinflammation markers, prostaglandin synthesis",
    },
    "donepezil": {
        "targets": ["VGF", "NPTX2", "SCG1"],
        "mechanism": "AChEI - may preserve synaptic markers",
    },
    "memantine": {
        "targets": ["NPTX1", "NPTX2", "GRIA4"],
        "mechanism": "NMDA antagonist - modulate excitatory synapse proteins",
    },
    "ace_inhibitor": {
        "targets": ["KNG1", "PLMN", "A1AT"],
        "mechanism": "RAS pathway, affect kallikrein-kinin and protease balance",
    },
    "ccb": {
        "targets": ["CA2D1"],
        "mechanism": "Calcium channel modulation",
    },
    "thyroid": {
        "targets": ["TTHY", "APOB", "FETUA"],
        "mechanism": "Transthyretin regulation, lipid metabolism",
    },
    "ppi": {
        "targets": ["CATD", "CATL1"],
        "mechanism": "Lysosomal enzyme modulation via pH changes",
    },
    "galantamine": {
        "targets": ["VGF", "NPTX2", "SCG1"],
        "mechanism": "AChEI - may preserve synaptic markers",
    },
    "rivastigmine": {
        "targets": ["VGF", "NPTX2", "SCG1"],
        "mechanism": "AChEI - may preserve synaptic markers",
    },
}

def get_protein_from_feature(feat):
    """Extract protein name from CSF_PROTEIN.PEPTIDE format."""
    if not feat.startswith("CSF_"):
        return None
    name = feat[4:]  # remove CSF_ prefix
    parts = name.split(".")
    return parts[0]

def categorize_csf_feature(feat):
    protein = get_protein_from_feature(feat)
    if protein is None:
        return "Clinical"
    for cat, proteins in CSF_CATEGORIES.items():
        if protein in proteins:
            return cat
    return "Other CSF"

tlog("Phase 5: Biological Interpretation (CSF Proteomics)")
df = pd.read_csv(DATA)
imp = pd.read_csv(IMP_FILE)
tlog(f"  {len(df)} patients, {len(imp)} features")

imp["is_csf"] = imp["feature"].str.startswith("CSF_")
imp["protein"] = imp["feature"].apply(get_protein_from_feature)
imp["category"] = imp["feature"].apply(categorize_csf_feature)

tlog("\n-- Importance by category --")
cat_imp = imp.groupby("category")["importance"].agg(["sum", "count", "mean"]).sort_values("sum", ascending=False)
for cat, row in cat_imp.iterrows():
    tlog(f"  {cat:40s}  total={row['sum']:.4f}  n={int(row['count'])}")

# Importance aggregated by protein (sum across peptides)
tlog("\n-- Top 20 proteins by aggregated importance --")
csf_only = imp[imp["is_csf"]].copy()
prot_imp = csf_only.groupby("protein")["importance"].sum().sort_values(ascending=False)
for prot, total_imp in prot_imp.head(20).items():
    cat = "?"
    for c, prots in CSF_CATEGORIES.items():
        if prot in prots:
            cat = c[:30]
            break
    n_pep = len(csf_only[csf_only["protein"] == prot])
    tlog(f"  {prot:12s}  imp={total_imp:.4f}  ({n_pep} peptides)  [{cat}]")

# Correlations with slope
tlog("\n-- Top 30 features: correlation with CDRSB slope --")
top30 = imp.head(30)
corr_rows = []
for _, row in top30.iterrows():
    feat = row["feature"]
    if feat in df.columns:
        vals = pd.to_numeric(df[feat], errors="coerce")
        mask = vals.notna() & df["slope"].notna()
        r, p = pearsonr(vals[mask], df["slope"][mask]) if mask.sum() > 30 else (np.nan, np.nan)
    else:
        r, p = np.nan, np.nan
    corr_rows.append({"feature": feat, "importance": row["importance"],
                       "protein": row["protein"], "category": row["category"],
                       "corr_with_slope": r, "corr_p": p})
    tlog(f"  {feat:50s}  imp={row['importance']:.4f}  r={r:+.3f}  [{row['category'][:25]}]")

# Fast vs slow progressors
tlog("\n-- Fast vs Slow progressors --")
med = df["slope"].median()
fast = df[df["slope"] > med]
slow = df[df["slope"] <= med]
prog_rows = []
for feat in top30["feature"].head(15):
    if feat in df.columns:
        fv = pd.to_numeric(fast[feat], errors="coerce").dropna()
        sv = pd.to_numeric(slow[feat], errors="coerce").dropna()
        if len(fv) > 10 and len(sv) > 10:
            _, pv = ttest_ind(fv, sv, equal_var=False)
            d = (fv.mean() - sv.mean()) / np.sqrt((fv.var() + sv.var()) / 2)
            prog_rows.append({"feature": feat, "fast_mean": fv.mean(), "slow_mean": sv.mean(),
                               "cohen_d": d, "t_p": pv})
            tlog(f"  {feat:50s}  d={d:+.3f}  p={pv:.4f}")

# -- Figures -----------------------------------------------------------------
tlog("\nGenerating figures ...")

# Feature importance bar chart
fig, ax = plt.subplots(figsize=(10, 8))
colors = ["#2196F3" if f.startswith("CSF_") else "#FF5722" for f in top30["feature"]]
short = []
for f in top30["feature"]:
    if f.startswith("CSF_"):
        # Show protein name only
        prot = get_protein_from_feature(f)
        pep = f[4:].split(".")[-1][:15] if "." in f[4:] else ""
        short.append(f"{prot}.{pep}" if pep else prot)
    else:
        short.append(f)
ax.barh(range(len(top30)), top30["importance"].values, color=colors)
ax.set_yticks(range(len(top30)))
ax.set_yticklabels(short, fontsize=8)
ax.invert_yaxis()
ax.set_xlabel("GBM Feature Importance")
ax.set_title("Top 30 Features for CDRSB Slope Prediction (CSF Proteomics)")
ax.legend(handles=[Patch(color="#2196F3", label="CSF Protein"),
                    Patch(color="#FF5722", label="Clinical")], loc="lower right")
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "feature_importance_top30.png"), dpi=150)
tlog("  Saved feature_importance_top30.png")

# Drug-CSF protein heatmap
top_csf = [f for f in imp[imp["is_csf"]].head(30)["feature"]]
top_proteins = []
seen = set()
for f in top_csf:
    prot = get_protein_from_feature(f)
    if prot and prot not in seen:
        top_proteins.append(prot)
        seen.add(prot)
    if len(top_proteins) >= 20:
        break

drug_classes = [d for d in DRUG_CSF_MAP.keys()]
if top_proteins:
    heatmap = np.zeros((len(drug_classes), len(top_proteins)))
    for i, drug in enumerate(drug_classes):
        targets = DRUG_CSF_MAP[drug]["targets"]
        for j, prot in enumerate(top_proteins):
            if prot in targets:
                heatmap[i, j] = 1

    fig2, ax2 = plt.subplots(figsize=(12, 5))
    ax2.imshow(heatmap, cmap="YlOrRd", aspect="auto")
    ax2.set_xticks(range(len(top_proteins)))
    ax2.set_xticklabels(top_proteins, rotation=45, ha="right", fontsize=8)
    ax2.set_yticks(range(len(drug_classes)))
    ax2.set_yticklabels(drug_classes, fontsize=9)
    ax2.set_title("Drug Mechanism x Top CSF Protein Overlap")
    plt.tight_layout()
    fig2.savefig(os.path.join(FIG_DIR, "drug_protein_heatmap.png"), dpi=150)
    tlog("  Saved drug_protein_heatmap.png")

# -- save -------------------------------------------------------------------
pd.DataFrame(corr_rows).to_csv(os.path.join(OUT_DIR, "feature_biology_table.csv"), index=False)
if prog_rows:
    pd.DataFrame(prog_rows).to_csv(os.path.join(OUT_DIR, "fast_vs_slow_progressors.csv"), index=False)

# Save protein-level importance
prot_imp_df = prot_imp.reset_index()
prot_imp_df.columns = ["protein", "total_importance"]
prot_imp_df["category"] = prot_imp_df["protein"].apply(
    lambda p: next((c for c, ps in CSF_CATEGORIES.items() if p in ps), "Other"))
prot_imp_df.to_csv(os.path.join(OUT_DIR, "protein_importance.csv"), index=False)
tlog("  Saved protein_importance.csv")

# Save drug mechanism table
drug_mech_rows = []
for drug, info in DRUG_CSF_MAP.items():
    for target in info["targets"]:
        prot_importance = prot_imp.get(target, 0)
        drug_mech_rows.append({
            "drug": drug, "target_protein": target,
            "mechanism": info["mechanism"],
            "protein_importance": prot_importance,
        })
pd.DataFrame(drug_mech_rows).to_csv(os.path.join(OUT_DIR, "drug_mechanism_table.csv"), index=False)
tlog("  Saved drug_mechanism_table.csv")

tlog("Phase 5 DONE.")
