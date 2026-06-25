"""
features.py
-----------
Compute per-sequence physicochemical features for AMP activity modelling.

Features computed:
  - length                  : sequence length
  - net_charge_pH7          : net charge at pH 7.0
  - isoelectric_point       : pI
  - hydrophobicity_eisenberg : mean Eisenberg hydrophobicity
  - hydrophobic_moment       : maximum hydrophobic moment (alpha-helix, window=11)
  - amphipathicity           : alias for hydrophobic moment
  - instability_index        : Guruprasad instability index
  - aromaticity              : fraction aromatic residues (F, Y, W)
  - fraction_positive        : fraction K + R residues
  - fraction_negative        : fraction D + E residues
  - fraction_helix           : predicted helix propensity (Chou-Fasman)
  - fraction_sheet           : predicted sheet propensity (Chou-Fasman)
  - fraction_turn            : predicted turn propensity  (Chou-Fasman)
  - aliphatic_index          : aliphatic index
  - boman_index              : Boman potential interaction index
"""

import warnings
import numpy as np
import pandas as pd
from pathlib import Path

# peptides package — pure Python, no binary deps
try:
    import peptides
    _PEPTIDES_OK = True
except ImportError:
    _PEPTIDES_OK = False
    warnings.warn("peptides package not found. Run: pip install peptides")

# ── paths ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data" / "processed"


# ══════════════════════════════════════════════════════════════════════════════
# Core feature extractor
# ══════════════════════════════════════════════════════════════════════════════

def compute_features_single(sequence: str) -> dict:
    """
    Compute all physicochemical features for a single sequence string.
    Returns a dict of feature_name → float.
    """
    nan_row = {f: np.nan for f in [
        "length", "net_charge_pH7", "isoelectric_point",
        "hydrophobicity_eisenberg", "hydrophobic_moment",
        "amphipathicity", "instability_index",
        "aromaticity", "fraction_positive", "fraction_negative",
        "fraction_helix", "fraction_sheet", "fraction_turn",
        "aliphatic_index", "boman_index", "molecular_weight",
    ]}

    if not _PEPTIDES_OK or not sequence or not isinstance(sequence, str):
        return nan_row

    try:
        pep = peptides.Peptide(sequence)
        seq_len = len(sequence)

        # hydrophobic moment
        try:
            hm = pep.hydrophobic_moment(window=min(11, seq_len), angle=100)
        except Exception:
            hm = np.nan

        # aromaticity: compute from residue frequencies directly
        # (F + Y + W) / length  — same definition as Biopython
        freq = pep.frequencies()   # returns dict of {aa: fraction}
        aromaticity = freq.get("F", 0) + freq.get("Y", 0) + freq.get("W", 0)

        # Chou-Fasman secondary structure propensities
        cf = _chou_fasman_fractions(sequence)

        return {
            "length":                    seq_len,
            "net_charge_pH7":            pep.charge(pH=7.0, pKscale="Lehninger"),
            "isoelectric_point":         pep.isoelectric_point(pKscale="Lehninger"),
            "hydrophobicity_eisenberg":  pep.hydrophobicity(scale="Eisenberg"),
            "hydrophobic_moment":        hm,
            "amphipathicity":            hm,
            "instability_index":         pep.instability_index(),
            "aromaticity":               aromaticity,
            "fraction_positive":         freq.get("K", 0) + freq.get("R", 0),
            "fraction_negative":         freq.get("D", 0) + freq.get("E", 0),
            "fraction_helix":            cf["helix"],
            "fraction_sheet":            cf["sheet"],
            "fraction_turn":             cf["turn"],
            "aliphatic_index":           pep.aliphatic_index(),
            "boman_index":               pep.boman(),
            "molecular_weight":          pep.molecular_weight(),
        }

    except Exception as e:
        warnings.warn(f"Feature computation failed for sequence '{sequence[:20]}...': {e}")
        return nan_row


def _chou_fasman_fractions(sequence: str) -> dict:
    """
    Residue-level Chou-Fasman propensity averages.
    Returns fraction of residues with helix/sheet/turn preference.
    Based on the original 1978 propensity table (Pa, Pb, Pt).
    H = helix former (Pa>1.0), E = sheet former (Pb>1.0), T = turn former (Pt>1.0)
    """
    # (Pa, Pb, Pt) — Chou & Fasman 1978
    CF = {
        "A": (1.42, 0.83, 0.66), "R": (0.98, 0.93, 0.95),
        "N": (0.67, 0.89, 1.56), "D": (1.01, 0.54, 1.46),
        "C": (0.70, 1.19, 1.19), "E": (1.51, 0.37, 0.74),
        "Q": (1.11, 1.10, 0.98), "G": (0.57, 0.75, 1.56),
        "H": (1.00, 0.87, 0.95), "I": (1.08, 1.60, 0.47),
        "L": (1.21, 1.30, 0.59), "K": (1.16, 0.74, 1.01),
        "M": (1.45, 1.05, 0.60), "F": (1.13, 1.38, 0.60),
        "P": (0.57, 0.55, 1.52), "S": (0.77, 0.75, 1.43),
        "T": (0.83, 1.19, 0.96), "W": (1.08, 1.37, 0.96),
        "Y": (0.69, 1.47, 1.14), "V": (1.06, 1.70, 0.50),
    }
    n = len(sequence)
    if n == 0:
        return {"helix": 0.0, "sheet": 0.0, "turn": 0.0}

    helix = sum(1 for aa in sequence if CF.get(aa, (1,1,1))[0] > 1.0) / n
    sheet = sum(1 for aa in sequence if CF.get(aa, (1,1,1))[1] > 1.0) / n
    turn  = sum(1 for aa in sequence if CF.get(aa, (1,1,1))[2] > 1.0) / n
    return {"helix": helix, "sheet": sheet, "turn": turn}


# ══════════════════════════════════════════════════════════════════════════════
# Batch processor
# ══════════════════════════════════════════════════════════════════════════════

def compute_features_batch(df: pd.DataFrame,
                           seq_col: str = "sequence",
                           show_progress: bool = True) -> pd.DataFrame:
    """
    Compute features for all sequences in a DataFrame.
    Returns a new DataFrame with one row per sequence, columns = features.
    Preserves amp_id as index if present.
    """
    sequences = df[seq_col].tolist()
    n = len(sequences)
    results = []

    for i, seq in enumerate(sequences):
        if show_progress and i % 1000 == 0:
            print(f"  Computing features: {i}/{n} ...", end="\r")
        results.append(compute_features_single(seq))

    print(f"  Computing features: {n}/{n} ... done      ")

    feat_df = pd.DataFrame(results)

    # attach amp_id if available
    if "amp_id" in df.columns:
        feat_df.insert(0, "amp_id", df["amp_id"].values)

    return feat_df


# ══════════════════════════════════════════════════════════════════════════════
# Summary statistics
# ══════════════════════════════════════════════════════════════════════════════

def feature_summary(feat_df: pd.DataFrame) -> pd.DataFrame:
    """
    Return mean ± std for each feature, plus % missing.
    Useful for the notebook summary table.
    """
    numeric = feat_df.select_dtypes(include=np.number)
    summary = numeric.describe().T[["mean", "std", "min", "max"]]
    summary["pct_missing"] = (numeric.isna().sum() / len(numeric) * 100).round(2)
    return summary.round(4)