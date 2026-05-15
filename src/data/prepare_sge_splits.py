"""
Prepare exon-stratified train/val/test splits from the Findlay 2018 SGE dataset.

Split logic (by exon group, never by random row):
  Train (~69%): X2, X3, X4, X15, X16, X17, X18, X19, X20
  Val   (~16%): X5, X21
  Test  (~15%): X22, X23

Exon-stratified splitting ensures the model is evaluated on sequence regions
it has never seen during training — the honest generalization test for VUS.

Outputs (saved to data/sge_splits/):
  sge_train.csv, sge_val.csv, sge_test.csv
  Each file contains the full Findlay columns plus the 100bp sequence window.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config import (
    BRCA1_REF, SEQ_WINDOW, SEED, SGE_BRCA1_RAW, SGE_SPLITS_DIR,
)

TRAIN_EXONS = {"X2", "X3", "X4", "X15", "X16", "X17", "X18", "X19", "X20"}
VAL_EXONS   = {"X5", "X21"}
TEST_EXONS  = {"X22", "X23"}

np.random.seed(SEED)


# ---------------------------------------------------------------------------
# Sequence extraction
# ---------------------------------------------------------------------------

def load_brca1_seq() -> str | None:
    """Load BRCA1 coding sequence from GenBank reference file."""
    if not BRCA1_REF.exists():
        return None
    from Bio import SeqIO
    record = SeqIO.read(str(BRCA1_REF), "genbank")
    return str(record.seq).upper()


def extract_window(ref_seq: str, cdna_pos: int, ref: str, alt: str,
                   window: int = SEQ_WINDOW) -> str:
    """
    Extract a `window`-bp sequence centred on `cdna_pos` (1-based cDNA coord),
    apply the alt substitution, and return the mutant string.
    """
    idx = cdna_pos - 1          # convert to 0-based
    half = window // 2
    start = max(0, idx - half)
    end   = min(len(ref_seq), idx + half)

    seq = ref_seq[start:end]

    # Pad with N if near boundary
    left_pad  = "N" * max(0, half - idx)
    right_pad = "N" * max(0, (idx + half) - len(ref_seq))
    seq = left_pad + seq + right_pad

    # Apply substitution at the centre position
    centre = idx - start + len(left_pad)
    if centre < len(seq) and seq[centre] == ref:
        seq = seq[:centre] + alt + seq[centre + 1:]

    return seq[:window].upper()


def build_sequences(df: pd.DataFrame, ref_seq: str | None) -> list[str]:
    """Return a list of mutant sequence windows, one per row."""
    seqs = []
    for _, row in df.iterrows():
        raw_pos = row.get("transcript_position")
        ref = str(row.get("transcript_ref", "N")) if pd.notna(row.get("transcript_ref")) else "N"
        alt = str(row.get("transcript_alt", "N")) if pd.notna(row.get("transcript_alt")) else "N"

        # Only handle simple integer positions; skip intronic coords like '-19-3'
        pos = None
        if pd.notna(raw_pos):
            try:
                pos = int(raw_pos)
            except (ValueError, TypeError):
                pos = None

        if ref_seq is not None and pos is not None and ref != "N" and alt != "N":
            seqs.append(extract_window(ref_seq, pos, ref, alt))
        else:
            # Intronic / non-parseable position — train_sge.py will skip these rows
            seqs.append("N" * SEQ_WINDOW)
    return seqs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("Preparing SGE exon-stratified splits...\n")

    df = pd.read_csv(SGE_BRCA1_RAW)
    print(f"Loaded: {len(df):,} rows")

    # Require score and experiment
    df = df.dropna(subset=["score", "experiment"])
    print(f"After dropping missing score/experiment: {len(df):,}")

    # Assign split
    def assign_split(exp: str) -> str:
        if exp in TRAIN_EXONS:
            return "train"
        if exp in VAL_EXONS:
            return "val"
        if exp in TEST_EXONS:
            return "test"
        return "unknown"

    df["split"] = df["experiment"].apply(assign_split)
    unknown = df[df["split"] == "unknown"]
    if len(unknown):
        print(f"WARNING: {len(unknown)} rows with unrecognised experiment labels — dropped")
        df = df[df["split"] != "unknown"]

    # Extract sequence windows
    print("\nExtracting sequence windows...")
    ref_seq = load_brca1_seq()
    if ref_seq is None:
        print("  BRCA1 reference not found — sequences will be placeholder N's.")
        print("  Run src/data/download.py first for real sequence windows.")
    else:
        print(f"  BRCA1 reference loaded: {len(ref_seq):,} bp")

    df["sequence"] = build_sequences(df, ref_seq)

    # Split and save
    SGE_SPLITS_DIR.mkdir(parents=True, exist_ok=True)

    for split in ("train", "val", "test"):
        part = df[df["split"] == split].reset_index(drop=True)
        out = SGE_SPLITS_DIR / f"sge_{split}.csv"
        part.to_csv(out, index=False)
        n_lof  = (part["function_class"] == "LOF").sum()
        n_func = (part["function_class"] == "FUNC").sum()
        n_int  = (part["function_class"] == "INT").sum()
        exons  = sorted(part["experiment"].unique())
        print(f"\n  {split:5s}: {len(part):4d} variants  "
              f"LOF={n_lof}  FUNC={n_func}  INT={n_int}  "
              f"exons={exons}")
        print(f"         -> {out}")

    print("\nDone.")


if __name__ == "__main__":
    main()
