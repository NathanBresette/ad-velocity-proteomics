# Divergent Roles for Cerebrospinal Fluid and Plasma Proteomics in Alzheimer's Disease Trial Design: A Digital Twin Study

**Bresette N, Lin A-L**  
University of Missouri, Roy Blunt NextGen Precision Health

---

## Overview

This repository contains the analysis code for the manuscript. The central finding is that CSF and plasma proteomics serve fundamentally different purposes in Alzheimer's disease trial design:

- **CSF proteomics** captures *disease velocity* — the real-time rate of active neuronal injury (UCHL1, FABP3). This signal is what enables drug-class stratification via difference-in-differences.
- **Plasma proteomics** captures *disease severity* — accumulated neuropathological burden (pTau-217, NEFL, GFAP). This signal is sufficient for prognostic trial enrichment (ProCoVA) but does not reproduce drug-class specificity, consistent with confounding by indication.

These are complementary, not competing: use plasma for trial enrichment, where no lumbar puncture is required, and discovery-scale CSF for mechanistic stratification.

## Key Results

| Model | R² | n | Notes |
|-------|----|---|-------|
| CSF clinical-only (TMT) | 0.250 | 1,060 | Baseline |
| CSF clinical+proteomics (TMT) | 0.415 | 1,060 | Discovery |
| CSF cross-platform (TMT→MRM) | 0.275 | 279 | External validation, zero sample overlap |
| Plasma NULISAseq (NULISA→MRM) | 0.347 | 279 | Same test set |

- TMT discovery model permutation null: observed R²=0.4147 vs null max −0.0905; 0 of 1,000 permutations reached the observed value (empirical p=0.000999)
- 4/4 cardiometabolic drug classes show differential attenuation in proteomics-predicted fast progressors (pooled DiD +0.041 pts/yr [0.025, 0.058], p<0.0001)
- Plasma stratification is null on the same patients (pooled DiD −0.010 pts/yr, p=0.716)
- CSF ProCoVA: 10.0–43.0% estimated trial sample size reduction (vs 0.2–23.3% for clinical-cognitive covariates alone)
- Plasma ProCoVA: 10.8–30.1% estimated trial sample size reduction

## Units

CDR-SB slopes are computed against the ADNI visit-month variable and are therefore stored in **points per month**. The manuscript and all figures report **points per year**; `generate_figures.py` applies the ×12 conversion at plot time. Any value read directly from a stored slope column must be converted before it is compared to a published number.

## Data Access

All data derive from the **Alzheimer's Disease Neuroimaging Initiative (ADNI)**, which is publicly available at [adni.loni.usc.edu](https://adni.loni.usc.edu). Data sharing requires an approved application through the ADNI Data Sharing and Publications Committee. Patient-level data are not included in this repository.

**Cohorts used:**
- TMT discovery: n=1,060 ADNI participants with CSF tandem mass tag proteomics
- MRM external validation: n=279 ADNI participants with CSF multiple reaction monitoring proteomics
- Plasma: n=1,428 ADNI participants with NULISAseq 120-plex plasma proteomics (BSHRI collaboration)

## Repository Structure

```
figures/
├── fig1_prediction_performance.png       # Model comparison bars + quintile calibration plots
├── fig2_biological_interpretation.png    # SHAP feature importance (top 20 predictors)
├── fig3_drug_stratification.png          # Cardiometabolic vs. control drug quintile dose-response
├── fig4_clinical_trial_enrichment.png    # ProCoVA sample size reduction (CSF, plasma, clinical)
├── fig5_plasma_quintile_calibration.png  # Plasma Model B quintile calibration
└── fig6_plasma_shap.png                  # Plasma SHAP (FABP3 highlighted)

scripts/
├── build_cdrsb_slope_dataset.py          # Build primary MRM dataset with CDR-SB slopes and drug flags
├── parse_nulisa.py                       # Parse NULISA long-format data, pivot to wide, align with ADNI
├── export_csf_proteomics.R               # R script for extracting CSF proteomics data from ADNI
├── nulisa_plasma_analysis.py             # Plasma Models A and B (LASSO + GBM, tuned hyperparameters)
├── nulisa_plasma_drug.py                 # Plasma drug stratification analysis
├── plasma_pooled_cardio_did.py           # Plasma pooled cardiometabolic DiD (null result)
├── pooled_cardio_did.py                  # CSF pooled cardiometabolic DiD
├── procova_variance_reduction.py         # ProCoVA ANCOVA sample size reduction formula
├── permutation_null_tmt.py               # Permutation null for the TMT discovery model (1,000 permutations)
├── feature_importance_biology.py         # SHAP analysis for TMT discovery model
├── build_table1.py                       # Table 1 cohort characteristics with p-values
└── generate_figures.py                   # All six manuscript figures
```

All six figures are main-text figures; there are no supplementary figures.

## TMT Discovery Cohort Analysis

The primary TMT discovery analysis (gradient-boosting cross-validation, SHAP, permutation null, and drug stratification with non-circular target trial emulation) was run on the Hellbender HPC cluster at the University of Missouri (DOI: [10.32469/10355/97710](https://doi.org/10.32469/10355/97710)). Those job scripts are not included here as they require cluster-specific SLURM configuration, but the methodology is fully described in the manuscript Methods section.

## Environment

```bash
python3.11 -m pip install -r requirements.txt
```

All local scripts were developed and tested with Python 3.11 on macOS.

## Citation

Bresette N, Lin A-L. Divergent Roles for Cerebrospinal Fluid and Plasma Proteomics in Alzheimer's Disease Trial Design: A Digital Twin Study. *Manuscript in preparation.*

## Acknowledgements

- ADNI is funded by the National Institutes of Health (U01 AG024904) and the Department of Defense (W81XWH-12-2-0012)
- This work was supported by NIH/NIA grants R56AG079586 and R01AG089493 to A-LL
- Computational resources provided by the Hellbender HPC cluster at the University of Missouri (DOI: 10.32469/10355/97710)
