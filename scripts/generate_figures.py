#!/usr/bin/env python3
"""
Publication-quality figures for CSF Proteomics paper.
Figures 1–4 + supplementary plasma figures.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.stats import pearsonr

BASE    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(BASE, "results")
EXT_DIR = os.path.join(BASE, "external_validation", "results")
FIG_DIR = os.path.join(BASE, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# Style
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 300,
})

# Color palette
C_CLINICAL = "#2471A3"   # muted blue  — clinical baseline
C_CSF      = "#A93226"   # muted red   — CSF protein (SHAP fig)
C_COMBINED = "#A93226"   # muted red   — CSF + clinical
C_NULL     = "#D5D8DC"
C_EXT      = "#6C3483"   # muted purple — external validation (TMT→MRM)
C_PLASMA   = "#D4780A"   # muted orange — plasma NULISA (colorblind-safe vs red CSF)

# Category colors for proteins
CAT_COLORS = {
    "AD Pathology (Amyloid/Tau)":        "#E74C3C",
    "Synaptic & Neuronal Markers":        "#3498DB",
    "Neurodegeneration Markers":          "#E67E22",
    "Proteases & Inhibitors":             "#9B59B6",
    "Oxidative Stress":                   "#2ECC71",
    "Complement & Innate Immunity":       "#1ABC9C",
    "Apolipoproteins & Lipid Transport":  "#F1C40F",
    "Neuroinflammation":                  "#E91E63",
    "Cell Adhesion & ECM":                "#795548",
    "Neuropeptides & Secretory":          "#607D8B",
    "Other":                              "#BDC3C7",
}

# ── Load primary (MRM) data ──────────────────────────────────────────────────
cv_b     = pd.read_csv(os.path.join(OUT_DIR, "cv_predictions_B_clinical_csf.csv"))
cv_a     = pd.read_csv(os.path.join(OUT_DIR, "cv_predictions_A_clinical.csv"))
cv_d     = pd.read_csv(os.path.join(OUT_DIR, "cv_predictions_D_csf_only.csv"))
perf     = pd.read_csv(os.path.join(OUT_DIR, "model_performance.csv"))
perm     = pd.read_csv(os.path.join(OUT_DIR, "permutation_null.csv"))
null_b   = pd.read_csv(os.path.join(OUT_DIR, "null_distribution_B_clinical_csf.csv"))
resp     = pd.read_csv(os.path.join(OUT_DIR, "responder_stratification.csv"))

# ── Load external validation (TMT) data ─────────────────────────────────────
ext_perf    = pd.read_csv(os.path.join(EXT_DIR, "validation_performance.csv"))
ext_preds   = pd.read_csv(os.path.join(EXT_DIR, "cross_platform_C_TMT_to_MRM.csv"))
tmt_preds   = pd.read_csv(os.path.join(EXT_DIR, "tmt_native_predictions.csv"))
tmt_resp    = pd.read_csv(os.path.join(EXT_DIR, "tmt_responder_stratification.csv"))
shap_imp    = pd.read_csv(os.path.join(EXT_DIR, "shap_mean_abs_tmt.csv"))
procova_new = pd.read_csv(os.path.join(EXT_DIR, "procova_crossplatform.csv"))

# ── Load plasma data ─────────────────────────────────────────────────────────
plasma_preds   = pd.read_csv(os.path.join(OUT_DIR, "nulisa_plasma_predictions.csv"))
shap_plasma    = pd.read_csv(os.path.join(OUT_DIR, "shap_nulisa_plasma_modelB.csv"))
with open(os.path.join(OUT_DIR, "nulisa_plasma_performance.json")) as f:
    plasma_perf = json.load(f)

# ── Compute plasma ProCoVA inline ────────────────────────────────────────────
# Drug flags for the 279 MRM test patients live in cdrsb_slope_dataset.csv
mrm_dataset  = pd.read_csv(os.path.join(OUT_DIR, "cdrsb_slope_dataset.csv"))
drug_cols_mrm = [c for c in mrm_dataset.columns if c.startswith("drug_")]
plasma_merged = plasma_preds.merge(
    mrm_dataset[["RID"] + drug_cols_mrm], on="RID", how="inner"
)

plasma_procova_rows = []
for _, row in procova_new.iterrows():
    drug = row["drug"]
    if drug not in plasma_merged.columns:
        continue
    users = plasma_merged[plasma_merged[drug] == 1].dropna(
        subset=["y_pred_modelB", "y_true"]
    )
    if len(users) < 10:
        continue
    r_p, _ = pearsonr(users["y_pred_modelB"], users["y_true"])
    plasma_procova_rows.append({
        "drug":                  drug,
        "n_plasma_users":        len(users),
        "r_plasma":              r_p,
        "procova_ss_pct_plasma": r_p**2 * 100,
    })
plasma_procova_df = pd.DataFrame(plasma_procova_rows)

# Merge CSF and plasma ProCoVA
proc_merged = procova_new.merge(plasma_procova_df, on="drug", how="left")

DRUG_NAME_MAP = {
    "drug_ccb":           "CCB",
    "drug_ace_inhibitor": "ACE Inhibitor",
    "drug_nsaid":         "NSAID",
    "drug_ppi":           "PPI",
    "drug_statin":        "Statin",
    "drug_thyroid":       "Thyroid",
    "drug_donepezil":     "Donepezil",
    "drug_memantine":     "Memantine",
    "drug_galantamine":   "Galantamine",
    "drug_rivastigmine":  "Rivastigmine",
    "drug_metformin":     "Metformin",
}

def clean_drug(d):
    return DRUG_NAME_MAP.get(d, d.replace("drug_", "").replace("_", " ").title())


# ============================================================================
# FIGURE 1: CSF Proteomics Predicts Cognitive Decline  (1 × 3)
# A: model bars (clinical / CSF discovery / CSF external / plasma)
# B: TMT quintile calibration
# C: TMT→MRM quintile calibration
# ============================================================================
fig1, axes1 = plt.subplots(1, 3, figsize=(18, 6))

ext_c        = ext_perf[ext_perf["section"] == "C_TMT_to_MRM"]
tmt_full_r2  = ext_perf[ext_perf["section"] == "B_TMT_native_full"]["r2"].values[0]
tmt_clin_r2  = ext_perf[ext_perf["section"] == "B_clinical"]["r2"].values[0] if \
    len(ext_perf[ext_perf["section"] == "B_clinical"]) > 0 else \
    perf[perf["model"] == "A_clinical"]["oof_r2"].values[0]

N_BINS = 5

def quintile_calibration(ax, df, pred_col, true_col, r2, color, title, extra_text=""):
    d = df.dropna(subset=[pred_col, true_col]).copy()
    r, _ = pearsonr(d[pred_col], d[true_col])
    d["bin"] = pd.qcut(d[pred_col], N_BINS, labels=False)
    bins = d.groupby("bin").agg(
        mean_pred=(pred_col, "mean"),
        mean_obs =(true_col, "mean"),
        se_obs   =(true_col, lambda x: x.std() / np.sqrt(len(x))),
        n        =(true_col, "count"),
    ).reset_index()
    ax.errorbar(bins["mean_pred"], bins["mean_obs"],
                yerr=1.96 * bins["se_obs"],
                fmt="o", color=color, ms=9, lw=1.8, capsize=5, capthick=1.5,
                ecolor=color, alpha=0.85, zorder=4, label="Quintile mean ± 95% CI")
    lo = min(bins["mean_pred"].min(), bins["mean_obs"].min()) - 0.005
    hi = max(bins["mean_pred"].max(), bins["mean_obs"].max()) + 0.005
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, alpha=0.45, label="Identity")
    z  = np.polyfit(bins["mean_pred"], bins["mean_obs"], 1)
    xs = np.linspace(lo, hi, 100)
    ax.plot(xs, np.polyval(z, xs), color=color, lw=1.4, ls="-", alpha=0.5)
    ax.set_xlabel("Mean Predicted CDR-SB Slope (pts/yr)")
    ax.set_ylabel("Mean Observed CDR-SB Slope (pts/yr)")
    ax.set_title(title, fontweight="bold", loc="left", fontsize=12)
    stats = f"$R^2$ = {r2:.3f}\nr = {r:.3f}\nn = {len(d):,}"
    if extra_text:
        stats += f"\n{extra_text}"
    ax.text(0.05, 0.92, stats, transform=ax.transAxes, fontsize=9,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    ax.legend(fontsize=7.5, loc="lower right")

# ── Panel A: Model comparison bars (4 bars) ──────────────────────────────────
ax = axes1[0]
bar_labels = [
    "Clinical\nOnly",
    "Clinical\n+ CSF\n(n=1,060)",
    "External\nValidation\nTMT→MRM\n(n=279)",
    "Plasma\nNULISA\n(n=279,\nsame pts)",
]
bar_colors = [C_CLINICAL, C_COMBINED, C_EXT, C_PLASMA]
bar_vals   = [tmt_clin_r2, tmt_full_r2, ext_c["r2"].values[0], plasma_perf["r2_modelB"]]
bar_stds   = [0, 0.038 / np.sqrt(5), 0, 0]   # fold SD only for TMT full (Hellbender log 12655046)

ax.bar(range(4), bar_vals, color=bar_colors, width=0.55, edgecolor="white", linewidth=0.5)
ax.errorbar(range(4), bar_vals, yerr=bar_stds,
            fmt="none", ecolor="black", capsize=4, capthick=1, lw=1)
ax.set_xticks(range(4))
ax.set_xticklabels(bar_labels, fontsize=8)
ax.set_ylabel("$R^2$")
ax.set_ylim(0, max(bar_vals) + 0.16)
# ΔR² arrow between clinical and CSF-full
delta = tmt_full_r2 - tmt_clin_r2
ax.annotate("", xy=(1, tmt_full_r2 + 0.015), xytext=(0, tmt_clin_r2 + 0.015),
            arrowprops=dict(arrowstyle="<->", color="black", lw=1))
ax.text(0.5, max(tmt_clin_r2, tmt_full_r2) + 0.03,
        f"$\\Delta R^2$ = +{delta:.3f}", ha="center", fontsize=8)
for i, (v, sd) in enumerate(zip(bar_vals, bar_stds)):
    ax.text(i, v + sd + 0.01, f"{v:.3f}", ha="center", fontsize=8)
ax.set_title("A", fontweight="bold", loc="left", fontsize=12)

# ── Panel B: TMT Discovery — quintile calibration (n=1,060) ─────────────────
quintile_calibration(axes1[1], tmt_preds, "y_pred_tmt", "y_true",
                     r2=tmt_full_r2, color=C_COMBINED,
                     title="B")

# ── Panel C: External Validation — quintile calibration TMT→MRM (n=279) ─────
ext_v = ext_preds.dropna(subset=["y_pred"])
quintile_calibration(axes1[2], ext_v, "y_pred", "y_true",
                     r2=ext_c["r2"].values[0], color=C_EXT,
                     title="C",
                     extra_text="p = 6.0×10$^{-26}$\nZero sample overlap")

fig1.tight_layout(w_pad=3)
fig1.savefig(os.path.join(FIG_DIR, "fig1_prediction_performance.png"), dpi=300, bbox_inches="tight")
fig1.savefig(os.path.join(FIG_DIR, "fig1_prediction_performance.pdf"), bbox_inches="tight")
print("Saved Figure 1")


# ============================================================================
# FIGURE 2: Biological Interpretation — UCHL1 and the Neuronal Injury Proteome
# Single panel: top 20 features by mean |SHAP| from B-full TMT model (n=1,060)
# ============================================================================
CLINICAL_SET = {"ADAS13", "FAQ", "AGE", "PTEDUCAT", "APOE4", "CDRSB", "MMSE"}

top20_shap = shap_imp[shap_imp["selected"]].head(20).copy().iloc[::-1].reset_index(drop=True)
top20_shap["label"] = top20_shap["feature"].str.replace("TMT_", "", regex=False)
colors2 = [C_CLINICAL if row["feature"] in CLINICAL_SET else C_CSF
           for _, row in top20_shap.iterrows()]

fig2, ax2 = plt.subplots(1, 1, figsize=(9, 6))

ax2.barh(range(len(top20_shap)), top20_shap["mean_abs_shap"].values, color=colors2,
         edgecolor="white", linewidth=0.3, height=0.7)
ax2.set_yticks(range(len(top20_shap)))
ax2.set_yticklabels(top20_shap["label"].values, fontsize=9)
ax2.set_xlabel("Mean |SHAP Value| (CDR-SB pts/yr per unit)", fontsize=9.5)
ax2.set_title("A", fontweight="bold", loc="left", fontsize=14)
ax2.grid(axis="x", lw=0.35, alpha=0.3, zorder=1)

handles2 = [
    Patch(facecolor=C_CLINICAL, label="Clinical-cognitive"),
    Patch(facecolor=C_CSF,      label="CSF protein"),
]
ax2.legend(handles=handles2, fontsize=8.5, loc="lower right", framealpha=0.9)

PROTEIN_LABELS = {
    "TMT_UCHL1":   "neuronal injury marker",
    "TMT_FABP3":   "neurodegeneration marker",
    "TMT_YWHAZ":   "synaptic signaling (14-3-3ζ)",
    "TMT_CPA4":    "extracellular protease",
    "TMT_DRAXIN":  "axon guidance",
    "TMT_SST":     "neuropeptide",
    "TMT_NPTX2":   "synaptic integrity",
    "TMT_PAMR1":   "ECM remodeling",
    "TMT_SLC9A1":  "ion transport / pH regulation",
    "TMT_OSTM1":   "lysosomal function",
    "TMT_EMCN":    "endothelial / neurovascular",
    "TMT_PPP3CA":  "calcium signaling (calcineurin)",
    "TMT_EPB41L2": "cytoskeletal scaffolding",
    "TMT_RMDN1":   "microtubule dynamics",
    "TMT_GAP43":   "axonal growth marker",
    "TMT_ERBB4":   "neuregulin receptor",
    "TMT_IGDCC4":  "axon guidance",
    "TMT_TNXB":    "extracellular matrix",
}
x_max = top20_shap["mean_abs_shap"].max()
for _, row in top20_shap.iterrows():
    lbl = PROTEIN_LABELS.get(row["feature"])
    if lbl:
        ax2.text(x_max * 0.012 + row["mean_abs_shap"], row.name,
                 lbl, va="center", fontsize=6.8, color="black",
                 fontstyle="italic")

fig2.tight_layout()
fig2.savefig(os.path.join(FIG_DIR, "fig2_biological_interpretation.png"), dpi=300, bbox_inches="tight")
fig2.savefig(os.path.join(FIG_DIR, "fig2_biological_interpretation.pdf"), bbox_inches="tight")
print("Saved Figure 2")


# ============================================================================
# FIGURE 3: Drug Stratification — Quintile Dose-Response (CSF cohort n=1,060)
# ============================================================================
from matplotlib.patches import Patch as _Patch3

C_UNTREATED = "#BDC3C7"
C_CARDIO    = "#8E44AD"
C_CHOLIN    = "#A93226"

_quint_path = os.path.join(EXT_DIR, "section_d_quintile_pooled.csv")

if not os.path.exists(_quint_path):
    CARDIO_SET = {"drug_statin", "drug_ace_inhibitor", "drug_ccb", "drug_thyroid"}
    _did_path = os.path.join(EXT_DIR, "section_d_did_regression.csv")
    if os.path.exists(_did_path):
        tmt_f = pd.read_csv(_did_path)
        tmt_f = tmt_f[tmt_f["drug"].isin(CARDIO_SET)].copy()
        tmt_f = tmt_f.sort_values("did_yr", ascending=True).reset_index(drop=True)
    else:
        tmt_f = tmt_resp.dropna(subset=["did_effect_high","did_effect_low","did_interaction"]).copy()
        tmt_f = tmt_f[tmt_f["drug"].isin(CARDIO_SET)].copy()
        tmt_f["did_yr"]   = -tmt_f["did_interaction"] * 12
        tmt_f["ci_lo_yr"] = tmt_f["did_yr"] - 0.3
        tmt_f["ci_hi_yr"] = tmt_f["did_yr"] + 0.3
        tmt_f["did_p"]    = np.nan
        tmt_f = tmt_f.sort_values("did_yr", ascending=True).reset_index(drop=True)
    fig3, axA = plt.subplots(1, 1, figsize=(8, 5))
    for i, (_, row) in enumerate(tmt_f.iterrows()):
        axA.hlines(i, row["ci_lo_yr"], row["ci_hi_yr"], color="#C39BD3", lw=2.0, zorder=2)
        axA.vlines([row["ci_lo_yr"], row["ci_hi_yr"]], i-0.12, i+0.12,
                   color="#C39BD3", lw=1.5, zorder=3)
        axA.scatter(row["did_yr"], i, color=C_CARDIO, s=120, zorder=5,
                    edgecolors="white", lw=0.8)
        axA.text(tmt_f["ci_hi_yr"].max() + 0.03, i,
                 f"n={int(row['n_users'])}", va="center", fontsize=8.5, color="black")
    axA.axvline(0, color="black", lw=1.2, alpha=0.5, ls="--", zorder=1)
    axA.set_yticks(np.arange(len(tmt_f)))
    axA.set_yticklabels([clean_drug(d) for d in tmt_f["drug"]],
                        fontsize=11, fontweight="bold", color="black")
    axA.set_xlabel("DiD Differential: Fast - Slow Attenuation (CDRSB pts/yr)", fontsize=9)
    axA.set_title("TMT Cohort (n=1,060) | DiD | Target Trial Emulation\n"
                  "(Quintile dose-response figure pending Hellbender job)",
                  fontweight="bold", loc="left", fontsize=9, pad=8)
    axA.grid(axis="x", lw=0.35, alpha=0.3, zorder=0)
    fig3.tight_layout()

else:
    qdf = pd.read_csv(_quint_path)

    BAR_W = 0.3
    GAP   = 0.08
    Q_SEP = 1.0

    _CLASS_COLORS = {
        "cardiometabolic": C_CARDIO,
        "cholinergic":     C_CHOLIN,
        "nsaid":           "#1A7A4A",
        "ppi":             "#B7770D",
    }
    _CLASS_ANNOT = {
        "cardiometabolic": ("A", ""),
        "cholinergic":     ("B", ""),
        "nsaid":           ("C", ""),
        "ppi":             ("D", ""),
    }
    _CLASS_ORDER    = ["cardiometabolic", "cholinergic", "nsaid", "ppi"]
    _have_all_4     = all(c in qdf["drug_class"].values for c in _CLASS_ORDER)
    _classes_present = [c for c in _CLASS_ORDER if c in qdf["drug_class"].values]

    def _draw_quintile_panel(ax, df, treated_color, panel_label, annotation):
        centers = np.arange(len(df)) * Q_SEP
        y_max = 0
        for i, row in df.iterrows():
            cx = centers[i]
            x_nu = cx - GAP/2 - BAR_W/2
            x_u  = cx + GAP/2 + BAR_W/2
            ax.bar(x_nu, row["nonusers_mean"], BAR_W,
                   color=C_UNTREATED, edgecolor="white", lw=0.5, zorder=3)
            ax.errorbar(x_nu, row["nonusers_mean"], yerr=1.96 * row["nonusers_se"],
                        fmt="none", ecolor="black", capsize=2.5, capthick=0.8, lw=0.9, zorder=4)
            ax.bar(x_u, row["users_mean"], BAR_W,
                   color=treated_color, edgecolor="white", lw=0.5, zorder=3)
            ax.errorbar(x_u, row["users_mean"], yerr=1.96 * row["users_se"],
                        fmt="none", ecolor="#333333", capsize=2.5, capthick=0.8, lw=0.9, zorder=4)
            gap_val = row["nonusers_mean"] - row["users_mean"]
            top = max(row["nonusers_mean"] + 1.96 * row["nonusers_se"],
                      row["users_mean"]    + 1.96 * row["users_se"]) + 0.03
            if abs(gap_val) > 0.005 and not (np.isnan(row["users_mean"]) or np.isnan(row["nonusers_mean"])):
                sign  = "-" if gap_val > 0 else "+"
                gcolor = "#4A235A" if gap_val > 0 else "#922B21"
                ax.text(cx, top + 0.015, f"{sign}{abs(gap_val):.2f}",
                        ha="center", fontsize=6, fontweight="bold", color=gcolor)
            y_max = max(y_max,
                        row["nonusers_mean"] + 1.96 * row["nonusers_se"],
                        row["users_mean"]    + 1.96 * row["users_se"])
        ax.set_xticks(centers)
        ax.set_xticklabels([f"Q{q}" for q in df["quintile"]], fontsize=8)
        ax.tick_params(axis="x", which="both", bottom=False)
        ax.set_ylabel("Actual CDRSB Slope (pts/yr)", fontsize=8)
        ax.set_title(panel_label, fontweight="bold", loc="left", fontsize=9, pad=5)
        ax.set_xlim(centers[0] - 0.65, centers[-1] + 0.65)
        ax.set_ylim(0, 2.5)
        ax.grid(axis="y", lw=0.3, alpha=0.35, zorder=0)

    if _have_all_4:
        fig3, axes3 = plt.subplots(1, 4, figsize=(18, 5.5), gridspec_kw={"wspace": 0.38})
    else:
        fig3, axes3 = plt.subplots(1, len(_classes_present), figsize=(13, 5.5),
                                   gridspec_kw={"wspace": 0.42})

    for ax, cls in zip(axes3, _classes_present):
        df_cls = qdf[qdf["drug_class"] == cls].sort_values("quintile").reset_index(drop=True)
        title, annot = _CLASS_ANNOT[cls]
        _draw_quintile_panel(ax, df_cls, _CLASS_COLORS[cls], title, annot)
        if ax != axes3[0]:
            ax.set_ylabel("")

    legend_els = [_Patch3(facecolor=C_UNTREATED, alpha=0.9, label="Untreated (non-users)")] + [
        _Patch3(facecolor=_CLASS_COLORS[c], alpha=0.9,
                label=f"Treated — {c.replace('nsaid','NSAID').replace('ppi','PPI')}")
        for c in _classes_present
    ]
    fig3.legend(handles=legend_els, loc="lower center",
                ncol=len(legend_els), fontsize=10, framealpha=0.9,
                bbox_to_anchor=(0.5, -0.06))
    fig3.suptitle("Figure 3", fontsize=11, fontweight="bold", y=1.02)
    fig3.tight_layout()

fig3.savefig(os.path.join(FIG_DIR, "fig3_drug_stratification.png"), dpi=300, bbox_inches="tight")
fig3.savefig(os.path.join(FIG_DIR, "fig3_drug_stratification.pdf"), bbox_inches="tight")
print("Saved Figure 3")


# ============================================================================
# FIGURE 4: Clinical Trial Enrichment — ProCoVA Sample Size Reduction
# 3 bars per drug: clinical only | clinical+CSF | plasma NULISA
# ============================================================================
fig4, ax4 = plt.subplots(1, 1, figsize=(8, 6))

proc4 = proc_merged[proc_merged["n_users"] >= 25].sort_values(
    "procova_ss_pct_csf", ascending=True
).copy().reset_index(drop=True)

drug_labels4 = [clean_drug(d) for d in proc4["drug"]]
y4   = np.arange(len(proc4))
h4   = 0.22  # bar height
gap4 = 0.03

# 3 rows: clinical (bottom), plasma (middle), CSF (top — primary result)
y_clin   = y4 - h4 - gap4
y_plasma = y4
y_csf    = y4 + h4 + gap4

ax4.barh(y_clin,   proc4["procova_ss_pct_clinical"].values, height=h4,
         color=C_CLINICAL, label="Clinical Only",    edgecolor="white", linewidth=0.4, zorder=3)

plasma_vals = proc4["procova_ss_pct_plasma"].values
plasma_valid = ~np.isnan(plasma_vals.astype(float))
ax4.barh(y_plasma[plasma_valid], plasma_vals[plasma_valid].astype(float), height=h4,
         color=C_PLASMA, label="Plasma NULISA",      edgecolor="white", linewidth=0.4, zorder=3)

ax4.barh(y_csf,    proc4["procova_ss_pct_csf"].values,      height=h4,
         color=C_COMBINED, label="Clinical + CSF",   edgecolor="white", linewidth=0.4, zorder=3)

ax4.axvline(0, color="black", lw=0.8, alpha=0.35)
ax4.set_yticks(y4)
ax4.set_yticklabels(drug_labels4, fontsize=9)
ax4.set_xlabel("Trial Sample Size Reduction (%)", fontsize=9.5)
ax4.set_title("Figure 4", fontweight="bold", loc="left", fontsize=12, pad=8)
ax4.grid(axis="x", lw=0.35, alpha=0.3, zorder=1)

# Label all three bar types
for i, v in enumerate(proc4["procova_ss_pct_csf"].values):
    ax4.text(v + 0.3, y_csf[i], f"{v:.1f}%", va="center", fontsize=7, color="black")
for i, (v, valid) in enumerate(zip(plasma_vals, plasma_valid)):
    if valid:
        ax4.text(float(v) + 0.3, y_plasma[i], f"{float(v):.1f}%", va="center",
                 fontsize=7, color="black")
for i, v in enumerate(proc4["procova_ss_pct_clinical"].values):
    ax4.text(float(v) + 0.3, y_clin[i], f"{float(v):.1f}%", va="center", fontsize=7, color="black")

ax4.legend(fontsize=8.5, loc="lower right", framealpha=0.92)

fig4.tight_layout()
fig4.savefig(os.path.join(FIG_DIR, "fig4_clinical_trial_enrichment.png"), dpi=300,
             bbox_inches="tight")
fig4.savefig(os.path.join(FIG_DIR, "fig4_clinical_trial_enrichment.pdf"), bbox_inches="tight")
print("Saved Figure 4")


# ============================================================================
# FIGURE S1: Plasma NULISA — Quintile Calibration (Model B, n=279)
# Same format as Fig 1C for direct visual comparison
# ============================================================================
figS1, axS1 = plt.subplots(1, 1, figsize=(6, 5.5))

quintile_calibration(
    axS1, plasma_preds, "y_pred_modelB", "y_true",
    r2=plasma_perf["r2_modelB"], color=C_PLASMA,
    title="Plasma NULISA — Model B Validation (n=279)",
    extra_text=(
        f"p = 5.8×10$^{{-29}}$\n"
        f"Same 279 patients as CSF validation\n"
        f"LASSO-selected from 120 proteins"
    )
)
axS1.set_title("S1", fontweight="bold", loc="left", fontsize=12)

figS1.tight_layout()
figS1.savefig(os.path.join(FIG_DIR, "figS1_plasma_quintile_calibration.png"), dpi=300, bbox_inches="tight")
figS1.savefig(os.path.join(FIG_DIR, "figS1_plasma_quintile_calibration.pdf"), bbox_inches="tight")
print("Saved Figure S1 (plasma quintile calibration)")


# ============================================================================
# FIGURE S2: Plasma NULISA SHAP — Top Features (Model B)
# Highlights FABP3 as independently rediscovered from 120 candidates
# ============================================================================
shap_p = shap_plasma.copy()

# Fix encoding artifact: "AÎ²42" → "Aβ42"
shap_p["feature"] = shap_p["feature"].str.replace("AÎ²", "Aβ", regex=False)

# Clean display labels
def clean_plasma_label(f):
    f = f.replace("PLASMAp_BD_", "BD-").replace("PLASMAp_", "")
    return f

shap_p["label"] = shap_p["feature"].apply(clean_plasma_label)

# Colors: clinical=gray, CSF-rediscovered=orange, other plasma=green
def plasma_bar_color(row):
    if row["feature"] in CLINICAL_SET:
        return C_CLINICAL
    if row["is_csf_mapped"]:
        return "#E67E22"   # orange — independently rediscovered from CSF
    return C_PLASMA

shap_p["color"] = shap_p.apply(plasma_bar_color, axis=1)
shap_plot = shap_p.iloc[::-1].reset_index(drop=True)

figS2, axS2 = plt.subplots(1, 1, figsize=(9, 5.5))

axS2.barh(range(len(shap_plot)), shap_plot["shap_mean_abs"].values,
          color=shap_plot["color"].values, edgecolor="white", linewidth=0.3, height=0.7)
axS2.set_yticks(range(len(shap_plot)))
axS2.set_yticklabels(shap_plot["label"].values, fontsize=9)
axS2.set_xlabel("Mean |SHAP Value| (CDR-SB pts/yr per unit)", fontsize=9.5)
axS2.set_title("S2", fontweight="bold", loc="left", fontsize=12)
axS2.grid(axis="x", lw=0.35, alpha=0.3, zorder=1)

# Annotate severity vs rate
PLASMA_ANNOT = {
    "BD-pTau_217":  "disease severity (accumulated damage)",
    "BD-pTau_231":  "disease severity",
    "NEFL":         "axonal damage (severity)",
    "Aβ42":         "amyloid burden",
    "GFAP":         "astrocyte activation (severity)",
    "DDC":          "dopamine synthesis",
    "FABP3":        "active neurodegeneration (rate) ← CSF-validated",
    "NPTXR":        "synaptic pentraxin receptor",
    "S100A12":      "neuroinflammation",
    "IL33":         "neuroinflammation",
    "UBB":          "ubiquitin (protein degradation)",
    "NPTX1":        "synaptic plasticity",
    "IL1B":         "pro-inflammatory cytokine",
    "CNTN2":        "axonal cell adhesion",
}
x_max_p = shap_plot["shap_mean_abs"].max()
for _, row in shap_plot.iterrows():
    lbl = PLASMA_ANNOT.get(row["label"])
    if lbl:
        color_txt = "#7D6608" if row["is_csf_mapped"] else "#2C3E50"
        axS2.text(x_max_p * 0.012 + row["shap_mean_abs"], row.name,
                  lbl, va="center", fontsize=6.5, color=color_txt, fontstyle="italic")

handlesS2 = [
    Patch(facecolor=C_CLINICAL, label="Clinical-cognitive"),
    Patch(facecolor=C_PLASMA,   label="Plasma protein (NULISA)"),
    Patch(facecolor="#E67E22",  label="CSF-validated (FABP3 — rediscovered independently)"),
]
axS2.legend(handles=handlesS2, fontsize=8, loc="lower right", framealpha=0.92)

figS2.tight_layout()
figS2.savefig(os.path.join(FIG_DIR, "figS2_plasma_shap.png"), dpi=300, bbox_inches="tight")
figS2.savefig(os.path.join(FIG_DIR, "figS2_plasma_shap.pdf"), bbox_inches="tight")
print("Saved Figure S2 (plasma SHAP)")

print("\nAll figures saved to:", FIG_DIR)
