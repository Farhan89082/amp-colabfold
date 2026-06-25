"""
structure_utils.py
------------------
Parse ColabFold output files to extract:
  - Per-residue pLDDT scores from PDB b-factor column
  - Mean pLDDT per peptide
  - pTM score from scores JSON
  - PAE matrix from predicted_aligned_error JSON
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STRUCTURES_DIR = ROOT / "structures" / "colabfold_output" / "flat_files"
PROCESSED_DIR  = ROOT / "data" / "processed"
RESULTS_DIR    = ROOT / "results"


# ══════════════════════════════════════════════════════════════════════════════
# 1. Parse scores JSON
# ══════════════════════════════════════════════════════════════════════════════

def parse_scores_json(json_path: Path) -> dict:
    """
    Parse a ColabFold scores JSON file.
    Returns dict with keys: mean_plddt, ptm, plddt_per_residue
    """
    with open(json_path) as f:
        data = json.load(f)

    plddt = data.get("plddt", [])
    ptm   = data.get("ptm", None)

    return {
        "mean_plddt":       float(np.mean(plddt)) if plddt else None,
        "ptm":              float(ptm) if ptm is not None else None,
        "plddt_per_residue": plddt,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 2. Parse PAE JSON
# ══════════════════════════════════════════════════════════════════════════════

def parse_pae_json(pae_path: Path) -> np.ndarray | None:
    """
    Parse a ColabFold predicted_aligned_error JSON.
    Returns PAE matrix as numpy array, or None if file not found.
    """
    if not pae_path.exists():
        return None
    with open(pae_path) as f:
        data = json.load(f)

    # ColabFold v1.5+ format
    if isinstance(data, list) and len(data) > 0:
        pae = data[0].get("predicted_aligned_error")
    elif isinstance(data, dict):
        pae = data.get("predicted_aligned_error")
    else:
        return None

    return np.array(pae) if pae is not None else None


# ══════════════════════════════════════════════════════════════════════════════
# 3. Parse PDB b-factor column for per-residue pLDDT
# ══════════════════════════════════════════════════════════════════════════════

def parse_pdb_plddt(pdb_path: Path) -> list:
    """
    Extract per-residue pLDDT from PDB b-factor column.
    Returns list of (residue_num, residue_name, plddt) tuples.
    """
    residues = {}
    with open(pdb_path) as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue
            try:
                res_num  = int(line[22:26].strip())
                res_name = line[17:20].strip()
                bfactor  = float(line[60:66].strip())
                if res_num not in residues:
                    residues[res_num] = (res_name, bfactor)
            except (ValueError, IndexError):
                continue

    return [(rnum, rname, bfac) for rnum, (rname, bfac) in sorted(residues.items())]


# ══════════════════════════════════════════════════════════════════════════════
# 4. Build summary DataFrame for all 131 peptides
# ══════════════════════════════════════════════════════════════════════════════

def build_structure_summary(structures_dir: Path = STRUCTURES_DIR) -> pd.DataFrame:
    """
    Scan the flat_files directory, parse all scores JSONs,
    and return a summary DataFrame with one row per peptide.
    """
    json_files = sorted(structures_dir.glob("*scores_rank_001*.json"))
    print(f"Found {len(json_files)} scores JSON files")

    records = []
    for jf in json_files:
        # Extract amp_id from filename
        # e.g. AMP_01454_length_36_ab_proba_0.9849_scores_rank_001_...json
        stem = jf.stem
        amp_id = stem.split("_length_")[0]
        length = int(stem.split("_length_")[1].split("_")[0])
        ab_proba = float(stem.split("_ab_proba_")[1].split("_scores")[0])

        scores = parse_scores_json(jf)

        # Check for PAE file
        pae_path = structures_dir / jf.name.replace(
            "_scores_rank_001_alphafold2_ptm_model_1_seed_000.json",
            "_predicted_aligned_error_v1.json"
        )
        pae = parse_pae_json(pae_path)
        mean_pae = float(np.mean(pae)) if pae is not None else None

        records.append({
            "amp_id":     amp_id,
            "length":     length,
            "ab_proba":   ab_proba,
            "mean_plddt": scores["mean_plddt"],
            "ptm":        scores["ptm"],
            "mean_pae":   mean_pae,
            "plddt_per_residue": scores["plddt_per_residue"],
        })

    df = pd.DataFrame(records).sort_values("mean_plddt", ascending=False).reset_index(drop=True)
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 5. Confidence classification
# ══════════════════════════════════════════════════════════════════════════════

def classify_confidence(plddt: float) -> str:
    """
    AlphaFold2 pLDDT confidence thresholds:
      >= 90  : Very high
      70-90  : Confident
      50-70  : Low
      < 50   : Very low
    """
    if plddt >= 90:
        return "Very high"
    elif plddt >= 70:
        return "Confident"
    elif plddt >= 50:
        return "Low"
    else:
        return "Very low"