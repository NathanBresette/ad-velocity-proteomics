#!/usr/bin/env python3
"""
Generate Supplementary Figures SF1–SF5 for CSF Proteomics paper.
Outputs PDF + PNG (300 DPI) to figures/ directory.

SF1 — 18-month endpoint analysis (CSF does not improve fixed endpoint)
SF2 — Clinical baseline saturation (7-var is sufficient)
SF3 — PSM Love plot (before/after SMD by drug)
SF4 — Bootstrap CIs on protein importance (requires bootstrap_ci.csv — run on Hellbender first)
SF5 — Imaging sensitivity (AV45 and Hippo_norm had zero importance)
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

BASE    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(BASE, "results")
FIG_DIR = os.path.join(BASE, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# Shared style
C_CLINICAL = "#2E86AB"
C_COMBINED = "#6A0572"
C_BEFORE   = "#E74C3C"
C_AFTER    = "#2ECC71"
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


# ============================================================================
# SF1: 18-month endpoint analysis
# ============================================================================
print("Generating SF1: 18-month endpoint analysis...")
ep = pd.read_csv(os.path.join(OUT_DIR, "endpoint_18mo_performance.csv"))

fig, ax = plt.subplots(figsize=(7, 4.5))

target_labels = {"18mo_abs": "18-Month Absolute\nCDRSB", "18mo_delta": "18-Month Change\nfrom Baseline"}
feat_labels   = {"clinical": "Clinical Only", "clinical_csf": "Clinical + CSF"}
feat_colors   = {"clinical": C_CLINICAL, "clinical_csf": C_COMBINED}

targets  = ["18mo_abs", "18mo_delta"]
features = ["clinical", "clinical_csf"]
n_targets = len(targets)
n_feats   = len(features)
x = np.arange(n_targets)
w = 0.32

for i, feat in enumerate(features):
    vals = []
    for tgt in targets:
        row = ep[(ep["target"] == tgt) & (ep["features"] == feat)]
        vals.append(row["r2"].values[0] if len(row) > 0 else 0.0)
    bars = ax.bar(x + (i - 0.5) * w, vals, width=w,
                  color=feat_colors[feat], label=feat_labels[feat],
                  edgecolor="white", linewidth=0.5, zorder=3)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.005,
                f"{v:.3f}", ha="center", va="bottom", fontsize=8.5)

ax.set_xticks(x)
ax.set_xticklabels([target_labels[t] for t in targets], fontsize=10)
ax.set_ylabel("Out-of-Fold R²", fontsize=10)
ax.set_title("SF1   18-Month Endpoint Analysis (MRM cohort, n=128)",
             fontweight="bold", loc="left", fontsize=11, pad=8)
ax.legend(fontsize=9, loc="upper left", framealpha=0.9)
ax.set_ylim(0, 0.52)
ax.grid(axis="y", lw=0.35, alpha=0.35, zorder=1)
ax.text(0.98, 0.97,
        "CSF proteomics does not improve 18-month\nprediction (n=128, ΔR²=−0.07 to −0.12)\n→ CDRSB slope is the correct primary outcome",
        transform=ax.transAxes, fontsize=8.5, ha="right", va="top",
        fontweight="bold", color="#7D3C98",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#F4ECF7", alpha=0.85))

# Reference line for slope model performance
slope_r2 = 0.234  # clinical-only slope model (same 128 pts)
ax.axhline(slope_r2, color="#E67E22", lw=1.2, ls="--", alpha=0.7)
ax.text(1.01, slope_r2, f"Slope model\nR²={slope_r2:.3f}", va="center",
        fontsize=7.5, color="#E67E22", transform=ax.get_yaxis_transform())

fig.tight_layout()
for ext in ("png", "pdf"):
    p = os.path.join(FIG_DIR, f"figS1_18month_endpoint.{ext}")
    fig.savefig(p, dpi=300, bbox_inches="tight")
plt.close(fig)
print("  Saved figS1_18month_endpoint")


# ============================================================================
# SF2: Clinical baseline saturation
# ============================================================================
print("Generating SF2: Clinical baseline saturation...")
sat = pd.read_csv(os.path.join(OUT_DIR, "supplementary_S2_clinical_saturation.csv"))

fig, ax = plt.subplots(figsize=(5.5, 4))

model_rows = sat[sat["OOF R²"].apply(lambda x: str(x).replace('.','',1).lstrip('-').isdigit()
                                      if pd.notna(x) else False)].copy()
# Filter to the two real model rows (not the delta row)
plot_rows = sat[sat["N Features"].isin([7, 10])].copy()

colors = [C_CLINICAL, C_COMBINED]
bars = ax.bar(range(len(plot_rows)),
              plot_rows["OOF R²"].astype(float).values,
              color=colors[:len(plot_rows)],
              edgecolor="white", linewidth=0.5, width=0.5, zorder=3)
for bar, v in zip(bars, plot_rows["OOF R²"].astype(float).values):
    ax.text(bar.get_x() + bar.get_width() / 2, v + 0.002,
            f"{v:.4f}", ha="center", va="bottom", fontsize=9)

ax.set_xticks(range(len(plot_rows)))
ax.set_xticklabels(
    [f"7-var\nclinical-cognitive\n(baseline)", f"10-var\nclinical\n(extended)"],
    fontsize=9)
ax.set_ylabel("Out-of-Fold R²", fontsize=10)
ax.set_title("SF2   Clinical Baseline Saturation (MRM, n=279)",
             fontweight="bold", loc="left", fontsize=11, pad=8)
ax.set_ylim(0, 0.33)
ax.grid(axis="y", lw=0.35, alpha=0.35, zorder=1)

delta_r2 = sat[sat["Model"].str.startswith("ΔR²")]["OOF R²"].values[0]
ax.text(0.98, 0.97,
        f"Adding sex + diagnosis group:\nΔR² = {float(delta_r2):+.4f} (SATURATED)",
        transform=ax.transAxes, fontsize=9, ha="right", va="top",
        fontweight="bold", color="#1A5276",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#EBF5FB", alpha=0.85))

fig.tight_layout()
for ext in ("png", "pdf"):
    p = os.path.join(FIG_DIR, f"figS2_clinical_saturation.{ext}")
    fig.savefig(p, dpi=300, bbox_inches="tight")
plt.close(fig)
print("  Saved figS2_clinical_saturation")


# ============================================================================
# SF3: PSM Love plot (before/after SMD)
# ============================================================================
print("Generating SF3: PSM Love plot...")
psm = pd.read_csv(os.path.join(OUT_DIR, "supplementary_S3_psm.csv"))

# Sort by mean SMD before matching (worst first = top)
psm = psm.sort_values("Mean SMD (before)", ascending=True).reset_index(drop=True)

fig, ax = plt.subplots(figsize=(7, 5.5))

y = np.arange(len(psm))
ax.scatter(psm["Mean SMD (before)"], y, color=C_BEFORE, s=65, zorder=4,
           label="Before matching", marker="o")
ax.scatter(psm["Mean SMD (after)"], y, color=C_AFTER, s=65, zorder=4,
           label="After matching", marker="D")

# Connect before/after with lines
for i in range(len(psm)):
    ax.plot([psm["Mean SMD (before)"].iloc[i], psm["Mean SMD (after)"].iloc[i]],
            [i, i], color="#95A5A6", lw=0.9, zorder=3)

ax.axvline(0.1, color="#E74C3C", lw=1.2, ls="--", alpha=0.7)
ax.axvline(0.0, color="#2C3E50", lw=0.7, alpha=0.3)
ax.text(0.101, len(psm) - 0.3, "SMD = 0.1\n(balance threshold)", fontsize=7.5,
        color="#E74C3C", va="top")

ax.set_yticks(y)
ax.set_yticklabels(psm["Drug"].values, fontsize=9)
ax.set_xlabel("Mean Standardized Mean Difference (SMD)", fontsize=10)
ax.set_title("SF3   Propensity Score Matching: Covariate Balance\nBefore and After Matching",
             fontweight="bold", loc="left", fontsize=11, pad=8)
ax.legend(fontsize=9, loc="lower right", framealpha=0.9)
ax.grid(axis="x", lw=0.35, alpha=0.35, zorder=1)

# Annotation
n_balanced = (psm["Mean SMD (after)"] < 0.1).sum()
ax.text(0.98, 0.97,
        f"PSM reduces imbalance\n({n_balanced}/{len(psm)} drugs: mean SMD < 0.10)\n"
        f"DiD design is primary protection\nagainst residual indication bias",
        transform=ax.transAxes, fontsize=8, ha="right", va="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85))

fig.tight_layout()
for ext in ("png", "pdf"):
    p = os.path.join(FIG_DIR, f"figS3_psm_love_plot.{ext}")
    fig.savefig(p, dpi=300, bbox_inches="tight")
plt.close(fig)
print("  Saved figS3_psm_love_plot")


# ============================================================================
# SF4: Bootstrap CIs on protein importance
# (requires bootstrap_ci.csv — run bootstrap_ci.py on Hellbender first)
# ============================================================================
# SF4: Bootstrap CIs by resampling OOF predictions (correct method — no model refitting)
# This gives CIs directly on OOF R², consistent with the reported values in Table 2.
print("Generating SF4: Bootstrap CIs on OOF R² (resampling OOF predictions)...")
from sklearn.metrics import r2_score as _r2

pred_files = {
    "A_clinical":     os.path.join(OUT_DIR, "cv_predictions_A_clinical.csv"),
    "B_clinical_csf": os.path.join(OUT_DIR, "cv_predictions_B_clinical_csf.csv"),
    "D_csf_only":     os.path.join(OUT_DIR, "cv_predictions_D_csf_only.csv"),
}

if not all(os.path.exists(p) for p in pred_files.values()):
    print("SF4: SKIPPED — OOF prediction CSVs not found.")
else:
    N_BOOT = 1000
    rng = np.random.RandomState(42)

    boot_r2s = {}
    oof_r2s  = {}
    for name, path in pred_files.items():
        df_p = pd.read_csv(path).dropna(subset=["y_true", "y_pred"])
        y_true = df_p["y_true"].values
        y_pred = df_p["y_pred"].values
        oof_r2s[name] = _r2(y_true, y_pred)
        n = len(y_true)
        r2s = []
        for _ in range(N_BOOT):
            idx = rng.choice(n, size=n, replace=True)
            r2s.append(_r2(y_true[idx], y_pred[idx]))
        boot_r2s[name] = np.array(r2s)

    # Delta bootstrap
    delta_boots = boot_r2s["B_clinical_csf"] - boot_r2s["A_clinical"]
    delta_mean  = delta_boots.mean()
    d_lo, d_hi  = np.percentile(delta_boots, [2.5, 97.5])

    labels = {"A_clinical": "Clinical only", "B_clinical_csf": "Clinical + CSF", "D_csf_only": "CSF only"}
    colors = {"A_clinical": "#6baed6", "B_clinical_csf": "#2171b5", "D_csf_only": "#74c476"}
    order  = ["A_clinical", "B_clinical_csf", "D_csf_only"]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), gridspec_kw={"width_ratios": [3, 1]})

    # Left panel: OOF R² bars with bootstrap 95% CI
    ax = axes[0]
    for i, m in enumerate(order):
        lo, hi = np.percentile(boot_r2s[m], [2.5, 97.5])
        oor = oof_r2s[m]
        ax.bar(i, oor, color=colors[m], width=0.5, zorder=3, label=labels[m])
        ax.errorbar(i, oor, yerr=[[oor - lo], [hi - oor]],
                    fmt="none", color="black", capsize=5, linewidth=1.5, zorder=4)
        ax.text(i, hi + 0.008, f"{oor:.3f}", ha="center", fontsize=9)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([labels[m] for m in order], fontsize=11)
    ax.set_ylabel("OOF R² (mean ± 95% bootstrap CI)", fontsize=10)
    ax.set_title("Model Performance — Bootstrap Confidence Intervals\n"
                 "(n=1,000 bootstrap resamples of OOF predictions, MRM cohort)", fontsize=10)
    ax.set_ylim(0, max(oof_r2s.values()) + 0.15)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3, zorder=0)

    # Right panel: ΔR² with CI
    ax2 = axes[1]
    delta_oof = oof_r2s["B_clinical_csf"] - oof_r2s["A_clinical"]
    yerr_lo = delta_oof - d_lo
    yerr_hi = d_hi - delta_oof
    ax2.bar(0, delta_oof, color="#9ecae1", width=0.5, zorder=3)
    ax2.errorbar(0, delta_oof, yerr=[[yerr_lo], [yerr_hi]],
                 fmt="none", color="black", capsize=5, linewidth=1.5, zorder=4)
    ax2.axhline(0, color="black", linewidth=1.0)
    ci_str = f"95% CI\n[{d_lo:.3f}, {d_hi:.3f}]"
    ax2.text(0, d_hi + 0.008, ci_str, ha="center", va="bottom", fontsize=9, color="#333333")
    ax2.set_xticks([0])
    ax2.set_xticklabels(["ΔR² (CSF\nincrement)"], fontsize=11)
    ax2.set_ylabel("ΔR²", fontsize=11)
    ax2.set_title("CSF Increment\nover Clinical", fontsize=11)
    ax2.set_ylim(-0.05, d_hi + 0.06)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.grid(axis="y", alpha=0.3, zorder=0)

    plt.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG_DIR, f"figS4_bootstrap_ci.{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved figS4_bootstrap_ci  (ΔR² = {delta_oof:.4f}, 95% CI [{d_lo:.4f}, {d_hi:.4f}])")


# ============================================================================
# SF5: Imaging sensitivity — AV45 and Hippo_norm had zero importance
# ============================================================================
print("Generating SF5: Imaging sensitivity (feature importance comparison)...")

# Show top 10 CSF proteins vs clinical features vs imaging features
prot_imp = pd.read_csv(os.path.join(OUT_DIR, "protein_importance.csv"))
# Also include clinical features from original feature_importances.csv
# AV45 and Hippo_norm: importance = 0 (confirmed in model run, excluded from 7-var baseline)

# Build comparison frame: top 10 proteins + all clinical features
top_prot = prot_imp.head(10)[["protein", "total_importance"]].copy()
top_prot["type"] = "CSF Protein"
top_prot = top_prot.rename(columns={"protein": "feature", "total_importance": "importance"})

clinical_feats = pd.DataFrame({
    "feature": ["ADAS13", "FAQ", "MMSE", "CDRSB", "APOE4", "PTEDUCAT", "AGE"],
    "importance": [0.2776, 0.1622, 0.0506, 0.0189, 0.0083, 0.0022, 0.0000],
    "type": "Clinical-Cognitive"
})

imaging_feats = pd.DataFrame({
    "feature": ["AV45 (amyloid PET)", "Hippo_norm (hippocampal vol.)"],
    "importance": [0.0000, 0.0000],
    "type": "Imaging (excluded)"
})

plot_df = pd.concat([top_prot, clinical_feats, imaging_feats], ignore_index=True)
plot_df = plot_df.sort_values("importance", ascending=True).reset_index(drop=True)

type_colors = {
    "CSF Protein": C_COMBINED,
    "Clinical-Cognitive": C_CLINICAL,
    "Imaging (excluded)": "#BDC3C7"
}

fig, ax = plt.subplots(figsize=(8, 7))
y = np.arange(len(plot_df))
colors_bar = [type_colors[t] for t in plot_df["type"]]
bars = ax.barh(y, plot_df["importance"], color=colors_bar,
               edgecolor="white", linewidth=0.4, zorder=3)

ax.set_yticks(y)
ax.set_yticklabels(plot_df["feature"], fontsize=8.5)
ax.set_xlabel("GBM Feature Importance", fontsize=10)
ax.set_title("SF5   Feature Importance: CSF Proteins vs Clinical vs Imaging\n"
             "(AV45 and Hippo_norm show zero importance → excluded from model)",
             fontweight="bold", loc="left", fontsize=11, pad=8)
ax.grid(axis="x", lw=0.35, alpha=0.35, zorder=1)

# Annotate imaging features
for _, row in imaging_feats.iterrows():
    idx = plot_df[plot_df["feature"] == row["feature"]].index[0]
    ax.text(0.001, idx, "importance = 0", va="center", fontsize=8, color="#7F8C8D",
            style="italic")

legend_patches = [
    mpatches.Patch(color=type_colors[t], label=t)
    for t in ["CSF Protein", "Clinical-Cognitive", "Imaging (excluded)"]
]
ax.legend(handles=legend_patches, fontsize=9, loc="lower right", framealpha=0.9)

fig.tight_layout()
for ext in ("png", "pdf"):
    p = os.path.join(FIG_DIR, f"figS5_imaging_sensitivity.{ext}")
    fig.savefig(p, dpi=300, bbox_inches="tight")
plt.close(fig)
print("  Saved figS5_imaging_sensitivity")


print(f"\nSupplementary figures saved to: {FIG_DIR}")
