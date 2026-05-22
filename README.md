# Disease Velocity vs. Severity: Divergent Roles for Cerebrospinal Fluid and Plasma Proteomics in Alzheimer's Trial Design

**Bresette N, Lin A-L**  
University of Missouri, Roy Blunt NextGen Precision Health

---

## Overview

This repository contains the analysis code for the manuscript. The central finding is that CSF and plasma proteomics serve fundamentally different purposes in Alzheimer's disease trial design:

- **CSF proteomics** captures *disease velocity* — the real-time rate of active neuronal death (UCHL1, FABP3). This signal is biologically necessary for drug repurposing causal inference via difference-in-differences.
- **Plasma proteomics** captures *disease severity* — accumulated neuropathological burden (pTau-217, NEFL, GFAP). This signal is sufficient for prognostic trial enrichment (ProCoVA) but fails at drug class specificity due to confounding by indication.

## Key Results

| Model | R² | n | Notes |
|-------|----|---|-------|
| CSF clinical-only (TMT) | 0.250 | 1,060 | Baseline |
| CSF clinical+proteomics (TMT) | 0.415 | 1,060 | Discovery |
| CSF cross-platform (TMT→MRM) | 0.275 | 279 | External validation, zero sample overlap |
| Plasma NULISAseq (NULISA→MRM) | 0.347 | 279 | Same test set |

- 4/4 cardiometabolic drug classes show differential attenuation in proteomics-predicted fast progressors (DiD +0.041 pts/yr [0.025, 0.058] p<0.0001 pooled)
- 0/4 cardiometabolic classes significant in plasma stratification (DiD −0.010 p=0.716)
- CSF ProCoVA: 10–43% estimated trial sample size reduction
- Plasma ProCoVA: 11–30% estimated trial sample size reduction

## Data Access

All data derive from the **Alzheimer's Disease Neuroimaging Initiative (ADNI)**, which is publicly available at [adni.loni.usc.edu](https://adni.loni.usc.edu). Data sharing requires an approved application through the ADNI Data Sharing and Publications Committee. Patient-level data are not included in this repository.

**Cohorts used:**
- TMT discovery: n=1,060 ADNI participants with CSF tandem mass tag proteomics
- MRM external validation: n=279 ADNI participants with CSF multiple reaction monitoring proteomics
- Plasma: n=1,428 ADNI participants with NULISAseq 120-plex plasma proteomics (BSHRI/NovaBay collaboration)

## Repository Structure

```
scripts/
├── build_cdrsb_slope_dataset.py     # Build primary MRM dataset with CDR-SB slopes and drug flags
├── parse_nulisa.py                  # Parse NULISA long-format data, pivot to wide, align with ADNI
├── nulisa_plasma_analysis.py        # Plasma Models A and B (LASSO + GBM, tuned hyperparameters)
├── nulisa_plasma_drug.py            # Plasma drug stratification analysis
├── plasma_pooled_cardio_did.py      # Plasma pooled cardiometabolic DiD (null result)
├── pooled_cardio_did.py             # CSF pooled cardiometabolic DiD
├── procova_variance_reduction.py    # ProCoVA ANCOVA sample size reduction formula
├── feature_importance_biology.py    # SHAP analysis for TMT discovery model
├── build_table1.py                  # Table 1 cohort characteristics with p-values
├── generate_figures.py              # All main manuscript figures (Fig 1–4)
└── generate_supplementary_figures.py  # Supplementary figures (S1 plasma calibration, S2 plasma SHAP)
```

## TMT Discovery Cohort Analysis

The primary TMT discovery analysis (gradient-boosting cross-validation, SHAP, drug stratification with non-circular target trial emulation) was run on the Hellbender HPC cluster at the University of Missouri (DOI: [10.32469/10355/97710](https://doi.org/10.32469/10355/97710)). Those scripts are not included here as they require cluster-specific SLURM configuration, but the methodology is fully described in the manuscript Methods section.

## Environment

```bash
python3.11 -m pip install -r requirements.txt
```

All local scripts were developed and tested with Python 3.11 on macOS.

## Citation

Bresette N, Lin A-L. Disease Velocity vs. Severity: Divergent Roles for Cerebrospinal Fluid and Plasma Proteomics in Alzheimer's Trial Design. *Manuscript in preparation.*

## Acknowledgements

- ADNI is funded by the National Institutes of Health (U01 AG024904) and the Department of Defense (W81XWH-12-2-0012)
- This work was supported by NIH/NIA grant R56AG079586 to A-LL
- Computational resources provided by the Hellbender HPC cluster at the University of Missouri (DOI: 10.32469/10355/97710)
