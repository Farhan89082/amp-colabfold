"""
curation.py
-----------
Download, parse, filter, and deduplicate AMP sequences from
DRAMP 4.0 and DBAASP for the amp-colabfold project.

Sources:
  - DRAMP 4.0  : direct FASTA download (general + antibacterial subsets)
  - DBAASP v3  : REST API via POST (form-encoded)
"""

import re
import requests
import pandas as pd
from pathlib import Path
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
from io import StringIO

# ── canonical amino acids only ────────────────────────────────────────────────
CANONICAL = set("ACDEFGHIKLMNPQRSTVWY")
MIN_LEN = 10
MAX_LEN = 50

# ── paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# 1. DRAMP 4.0  — direct FASTA downloads (no API needed)
# ══════════════════════════════════════════════════════════════════════════════

DRAMP_URLS = {
    "general":      "https://dramp.cpu-bioinfor.org/downloads/download.php?filename=download_data/DRAMP3.0_new/general_amps.fasta",
    "antibacterial":"https://dramp.cpu-bioinfor.org/downloads/download.php?filename=download_data/DRAMP3.0_new/Antibacterial_amps.fasta",
    "natural":      "https://dramp.cpu-bioinfor.org/downloads/download.php?filename=download_data/DRAMP3.0_new/natural_amps.fasta",
}

def fetch_dramp(save_path: Path | None = None) -> pd.DataFrame:
    """
    Download DRAMP 4.0 general + antibacterial FASTA files directly.
    Deduplicates across the two files, keeping the general annotation.
    """
    print("Fetching DRAMP 4.0 ...")

    headers = {
        "User-Agent": "Mozilla/5.0 (AMP-ColabFold research project)",
        "Referer": "https://dramp.cpu-bioinfor.org/downloads/",
    }

    records = []
    for subset, url in DRAMP_URLS.items():
        try:
            r = requests.get(url, headers=headers, timeout=60)
            r.raise_for_status()
            # use r.content (bytes) to handle gzip-encoded responses correctly
            content = r.content.decode("utf-8", errors="ignore").strip()

            if not content.startswith(">"):
                print(f"  {subset}: unexpected response (not FASTA), skipping")
                continue

            for rec in SeqIO.parse(StringIO(content), "fasta"):
                records.append({
                    "id":       rec.id,
                    "name":     rec.description,
                    "sequence": str(rec.seq).upper().strip(),
                    "activity": subset,
                    "source":   "DRAMP4.0",
                })
            print(f"  {subset}: {len(records)} cumulative records")

        except Exception as e:
            print(f"  DRAMP {subset} fetch failed: {e}")
            # try local fallback
            local = RAW_DIR / f"dramp_{subset}.fasta"
            if local.exists():
                print(f"  Loading local fallback: {local}")
                for rec in SeqIO.parse(local, "fasta"):
                    records.append({
                        "id": rec.id, "name": rec.description,
                        "sequence": str(rec.seq).upper().strip(),
                        "activity": subset, "source": "DRAMP4.0",
                    })

    df = pd.DataFrame(records)
    if df.empty:
        print("  Warning: no DRAMP records retrieved.")
        return df

    # deduplicate across subsets
    df = df.drop_duplicates(subset="sequence", keep="first").reset_index(drop=True)
    print(f"  DRAMP after dedup: {len(df)} unique sequences")

    if save_path:
        df.to_csv(save_path, index=False)
        print(f"  Saved → {save_path}")

    return df


# ══════════════════════════════════════════════════════════════════════════════
# 2. DBAASP v3  — REST API via POST (form-encoded)
# ══════════════════════════════════════════════════════════════════════════════

def fetch_dbaasp(save_path: Path | None = None) -> pd.DataFrame:
    """
    Pull monomeric AMPs from DBAASP v3 REST API using POST requests.
    Fetches ribosomal (synthesis_type=36) and synthetic (synthesis_type=38)
    peptides without unusual amino acids.
    """
    print("Fetching DBAASP via REST API (POST) ...")

    BASE = "https://dbaasp.org/api/v1"
    headers = {"Content-Type": "application/x-www-form-urlencoded"}

    queries = [
        {"synthesis_type": "36", "label": "ribosomal"},
        {"synthesis_type": "38", "label": "synthetic"},
    ]

    records = []
    for q in queries:
        data = {
            "query": "search",
            "complexity": "monomer",
            "synthesis_type": q["synthesis_type"],
            "unusual_amino_acid_id": "1",
            "format": "json",
        }
        try:
            r = requests.post(BASE, data=data, headers=headers, timeout=60)
            r.raise_for_status()

            # handle empty or non-JSON response gracefully
            if not r.text.strip():
                print(f"  DBAASP {q['label']}: empty response")
                continue

            payload = r.json()
        except Exception as e:
            print(f"  DBAASP {q['label']} failed: {e}")
            continue

        # API may return a list directly or wrap in a dict
        peptides = payload if isinstance(payload, list) else payload.get("peptides", [])
        print(f"  {q['label']}: {len(peptides)} raw records")

        for p in peptides:
            # sequence field varies by API version
            seq = ""
            if isinstance(p.get("sequence"), str):
                seq = p["sequence"]
            elif isinstance(p.get("monomer"), dict):
                seq = p["monomer"].get("sequence", "")
            if not seq:
                continue

            # MIC: grab first numeric activity value if present
            mic_val = None
            for act_key in ("activities", "targetActivities"):
                acts = p.get(act_key) or []
                if acts:
                    for a in acts:
                        try:
                            mic_val = float(a.get("activity") or a.get("value"))
                            break
                        except (TypeError, ValueError):
                            continue
                    if mic_val is not None:
                        break

            records.append({
                "id":       str(p.get("peptideId") or p.get("id", "")),
                "name":     p.get("name") or p.get("peptideName", ""),
                "sequence": seq.upper().strip(),
                "activity": "antibacterial",
                "mic":      mic_val,
                "source":   "DBAASP",
            })

    df = pd.DataFrame(records)
    if df.empty:
        print("  Warning: no DBAASP records retrieved.")
        return pd.DataFrame(columns=["id", "name", "sequence", "activity", "mic", "source"])

    if save_path:
        df.to_csv(save_path, index=False)
        print(f"  Saved {len(df)} DBAASP records → {save_path}")

    return df


# ══════════════════════════════════════════════════════════════════════════════
# 3. FILTER
# ══════════════════════════════════════════════════════════════════════════════

def filter_sequences(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply project filters:
      1. Drop empty / non-string sequences
      2. Length: MIN_LEN ≤ len ≤ MAX_LEN
      3. Canonical amino acids only
      4. Exact deduplication (prefer DBAASP > DRAMP)
    """
    print(f"\nFiltering {len(df)} total raw sequences ...")
    n0 = len(df)

    df = df.dropna(subset=["sequence"]).copy()
    df["sequence"] = df["sequence"].astype(str).str.upper().str.strip()
    df = df[df["sequence"].str.len() > 0]

    # length filter
    df = df[df["sequence"].str.len().between(MIN_LEN, MAX_LEN)]
    print(f"  After length filter ({MIN_LEN}–{MAX_LEN} aa): {len(df)}  (removed {n0 - len(df)})")
    n1 = len(df)

    # canonical residues
    df = df[df["sequence"].apply(lambda s: set(s).issubset(CANONICAL))]
    print(f"  After canonical AA filter: {len(df)}  (removed {n1 - len(df)})")
    n2 = len(df)

    # dedup — prefer DBAASP over DRAMP
    priority = {"DBAASP": 0, "DRAMP4.0": 1}
    df["_p"] = df["source"].map(priority).fillna(9)
    df = (df.sort_values("_p")
            .drop_duplicates(subset="sequence", keep="first")
            .drop(columns="_p")
            .reset_index(drop=True))
    print(f"  After exact deduplication: {len(df)}  (removed {n2 - len(df)})")

    return df


# ══════════════════════════════════════════════════════════════════════════════
# 4. CLUSTERING  (Python CD-HIT at 90% identity)
# ══════════════════════════════════════════════════════════════════════════════

def cluster_cdhit_python(df: pd.DataFrame, identity: float = 0.90) -> pd.DataFrame:
    """
    Greedy longest-first clustering at `identity` sequence identity.
    Uses 3-mer pre-filtering to skip obviously dissimilar pairs.
    """
    print(f"\nClustering at {identity*100:.0f}% identity ...")
    seqs = df["sequence"].tolist()
    n = len(seqs)
    order = sorted(range(n), key=lambda i: len(seqs[i]), reverse=True)
    sorted_seqs = [seqs[i] for i in order]

    def kmer_set(seq, k=3):
        return {seq[j:j+k] for j in range(len(seq) - k + 1)}

    def pairwise_id(a, b):
        if abs(len(a) - len(b)) / max(len(a), len(b)) > (1 - identity):
            return 0.0
        return sum(ca == cb for ca, cb in zip(a, b)) / max(len(a), len(b))

    reps, rep_kmers = [], []
    for i, seq in enumerate(sorted_seqs):
        if i % 200 == 0:
            print(f"  {i}/{n} ...", end="\r")
        sk = kmer_set(seq)
        merged = any(
            len(sk & rep_kmers[j]) >= 2 and pairwise_id(seq, sorted_seqs[reps[j]]) >= identity
            for j in range(len(reps))
        )
        if not merged:
            reps.append(i)
            rep_kmers.append(sk)

    print(f"\n  Representatives: {len(reps)} / {n}")
    keep = [order[r] for r in reps]
    return df.iloc[keep].reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════════════
# 5. EXPORT
# ══════════════════════════════════════════════════════════════════════════════

def save_outputs(df: pd.DataFrame) -> None:
    """Save curated_amps_metadata.csv and curated_amps.fasta."""
    df = df.copy()
    if "amp_id" not in df.columns:
        df.insert(0, "amp_id", [f"AMP_{i:05d}" for i in range(len(df))])

    csv_path = PROCESSED_DIR / "curated_amps_metadata.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved metadata  → {csv_path}  ({len(df)} sequences)")

    fasta_path = PROCESSED_DIR / "curated_amps.fasta"
    records = [
        SeqRecord(
            Seq(row.sequence),
            id=row.amp_id,
            description=f"source={row.source} activity={getattr(row, 'activity', 'unknown')}",
        )
        for row in df.itertuples()
    ]
    SeqIO.write(records, fasta_path, "fasta")
    print(f"Saved FASTA     → {fasta_path}")