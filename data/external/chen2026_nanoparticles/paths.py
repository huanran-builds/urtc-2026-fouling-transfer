"""
Centralised path definitions for the nanoparticle antibacterial ML project.

All paths are relative to THIS file's directory so the project is portable.
Import from any script with:

    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
    from paths import *
"""

import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, "data")
OUTPUT_DIR = os.path.join(ROOT_DIR, "outputs")

# ── Training datasets ────────────────────────────────────────────────────────
MIC_DATA = os.path.join(DATA_DIR, "single_features.csv")
MBC_DATA = os.path.join(DATA_DIR, "single_MBC_features.csv")
MIC_WITH_CLASS_DATA = os.path.join(DATA_DIR, "single_features_with_class.csv")

# ── Enumeration / prediction datasets ────────────────────────────────────────
ENUMERATE_AG = os.path.join(DATA_DIR, "enumerate_ag.csv")
ENUMERATE_AG_ECOLI = os.path.join(DATA_DIR, "enumerate_ag_ecoli.csv")

# ── Output sub-directories (created on demand) ──────────────────────────────
GROUPED_CV_DIR = os.path.join(OUTPUT_DIR, "results_grouped_cv")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(GROUPED_CV_DIR, exist_ok=True)
