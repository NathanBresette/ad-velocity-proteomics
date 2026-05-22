"""
Parse BSHRI NULISA plasma proteomics data from ADNI.
Run immediately after downloading from LONI.
Usage: python parse_nulisa.py /path/to/downloaded/nulisa_file.csv
"""

import sys, os
import pandas as pd
import numpy as np

# ── Target proteins (LASSO-selected from CSF TMT model) ───────────────────
# These are the 65 proteins we want to find in plasma
TMT_LASSO = [
    "ADCYAP1R1","ANXA6","ATOX1","CDC42","CPA4","CRIP2","DDRGK1","DRAXIN","ECM1",
    "EMCN","EPB41L2","ERBB4","FABP3","FDPS","GALNT15","GAP43","GLCE","GPI",
    "IGDCC4","IGHV1-3","IGHV3-74","IGKV6D-21","INHBA","ISLR","JAG1","LGALS3",
    "LRP2","MANEAL","MYH11","MYL12A","NAXD","NDRG2","NPTX2","OBP2A","OLFM2",
    "OS9","OSTM1","PAMR1","PDIA3","PDYN","PEAR1","PLOD2","PPP3CA","PTPRN2",
    "RMDN1","S100B","S1PR1","SELENBP1","SEMA3F","SGSH","SLC9A1","SMOC1","SPINT1",
    "SST","SUMO2","TCTN3","TGFA","TNFRSF8","TNXB","TRH","TXNDC15","UBQLN2",
    "UCHL1","VCL","YWHAZ"
]

# Top 10 by SHAP — most important to find
TOP_SHAP = ["UCHL1","FABP3","YWHAZ","SST","NPTX2","GAP43","NDRG2","S100B","LGALS3","SMOC1"]

def parse_nulisa(filepath):
    print(f"\nLoading: {filepath}")

    # Try common formats
    for sep in [",", "\t"]:
        try:
            df = pd.read_csv(filepath, sep=sep, low_memory=False)
            if df.shape[1] > 3:
                break
        except:
            continue

    print(f"Shape: {df.shape}")
    print(f"Columns (first 20): {list(df.columns[:20])}")

    # ── Detect format ──────────────────────────────────────────────────────
    cols_upper = [c.upper() for c in df.columns]

    # Check if wide format (samples as rows, proteins as columns)
    # or long format (one row per measurement)
    has_rid = any("RID" in c for c in cols_upper)
    has_protein_col = any(c in cols_upper for c in ["PROTEIN","ANALYTE","TARGET","GENE"])

    print(f"\nFormat detection:")
    print(f"  Has RID col: {has_rid}")
    print(f"  Has protein/analyte col (long format): {has_protein_col}")

    # ── Phase/cohort breakdown ─────────────────────────────────────────────
    if "ORIGPROT" in cols_upper or "COLPROT" in cols_upper:
        phase_col = "ORIGPROT" if "ORIGPROT" in cols_upper else "COLPROT"
        # match case
        phase_col = [c for c in df.columns if c.upper() == phase_col][0]
        print(f"\nPhase distribution:")
        print(df[phase_col].value_counts())
        adni3_mask = df[phase_col].astype(str).str.upper().str.contains("ADNI3|ADNI 3", na=False)
        rid_col = [c for c in df.columns if c.upper() == "RID"][0]
        print(f"  ADNI3 unique RIDs: {df.loc[adni3_mask, rid_col].nunique()}")

    # ── Protein coverage ───────────────────────────────────────────────────
    if has_protein_col:
        # Long format
        prot_col = [c for c in df.columns if c.upper() in ["PROTEIN","ANALYTE","TARGET","GENE"]][0]
        proteins = df[prot_col].astype(str).str.upper().unique()
        print(f"\nTotal unique proteins/analytes: {len(proteins)}")
        print(f"Sample protein names: {list(proteins[:10])}")
    else:
        # Wide format — protein names are column headers
        rid_col = [c for c in df.columns if c.upper() == "RID"][0] if has_rid else None
        non_meta = [c for c in df.columns if c.upper() not in
                    ["RID","ORIGPROT","COLPROT","VISCODE","EXAMDATE","PHASE","SAMPLEID"]]
        proteins = [c.upper() for c in non_meta]
        print(f"\nTotal protein columns: {len(proteins)}")
        print(f"Sample protein names: {proteins[:10]}")

    # ── Check for LASSO-selected CSF proteins ─────────────────────────────
    print(f"\n{'='*50}")
    print("LASSO-SELECTED CSF PROTEINS IN NULISA PANEL:")
    print(f"{'='*50}")

    found = []
    not_found = []
    for gene in TMT_LASSO:
        # Check exact match and partial match
        match = any(gene.upper() in p for p in proteins)
        if match:
            found.append(gene)
        else:
            not_found.append(gene)

    print(f"\nFound ({len(found)}/65):")
    for g in found:
        shap_flag = " *** TOP SHAP" if g in TOP_SHAP else ""
        print(f"  {g}{shap_flag}")

    print(f"\nNot found ({len(not_found)}/65):")
    print(f"  {', '.join(not_found)}")

    # ── Special check for top targets ─────────────────────────────────────
    print(f"\n{'='*50}")
    print("TOP SHAP PROTEIN STATUS:")
    print(f"{'='*50}")
    for g in TOP_SHAP:
        status = "FOUND ✓" if g in found else "missing"
        print(f"  {g:12s} {status}")

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n{'='*50}")
    print("SUMMARY")
    print(f"{'='*50}")
    print(f"  LASSO proteins in panel: {len(found)}/65")
    print(f"  Top SHAP proteins found: {sum(g in found for g in TOP_SHAP)}/10")
    verdict = "STRONG" if len(found) >= 20 else "MODERATE" if len(found) >= 8 else "WEAK"
    print(f"  Coverage verdict: {verdict}")
    print(f"\n  Recommendation:")
    if len(found) >= 20:
        print("  → Run full plasma model — sufficient coverage of CSF LASSO proteins")
    elif len(found) >= 8:
        print("  → Partial model possible — use found proteins + NfL/tau as complement")
    else:
        print("  → Coverage too low for CSF-guided plasma model")

    return found, not_found

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python parse_nulisa.py /path/to/nulisa_file.csv")
        print("\nExpected download location after LONI download:")
        print("  ~/Downloads/  (check for .csv, .zip, or .xlsx)")
        sys.exit(1)

    filepath = sys.argv[1]
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        sys.exit(1)

    found, not_found = parse_nulisa(filepath)
