"""
Prepare exon-stratified train/val/test splits from the BRCA2 SGE dataset.

Split logic (by exon group, never by random row):
  Train (~58%): E15, E16, E17, E18, E19, E20, E21
  Val   (~17%): E22, E23
  Test  (~25%): E24, E25, E26

Exon-stratified splitting ensures the model is evaluated on sequence regions
it has never seen during training.

Outputs (saved to data/brca2_sge_splits/):
  brca2_sge_train.csv, brca2_sge_val.csv, brca2_sge_test.csv
  Each file contains the annotated BRCA2 columns plus a 100bp sequence window.
"""

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config import (
    BRCA2_REF, BRCA2_SGE_SPLITS_DIR, SEQ_WINDOW, SEED, SGE_BRCA2_RAW,
)

TRAIN_EXONS = {"E15", "E16", "E17", "E18", "E19", "E20", "E21"}
VAL_EXONS   = {"E22", "E23"}
TEST_EXONS  = {"E24", "E25", "E26"}

np.random.seed(SEED)

_CDNA_SNV_RE = re.compile(r'c\.(-?\d+)([ACGT])>([ACGT])$', re.ASCII)


# ---------------------------------------------------------------------------
# Sequence extraction
# ---------------------------------------------------------------------------

def load_brca2_seq() -> str:
    """Load BRCA2 coding sequence from GenBank reference file."""
    if not BRCA2_REF.exists():
        return None
    from Bio import SeqIO
    record = SeqIO.read(str(BRCA2_REF), "genbank")
    return str(record.seq).upper()


def parse_hgvs_nt(hgvs_nt: str):
    """
    Parse NM_000059.4:c.7436A>T -> (7436, 'A', 'T').
    Returns (None, '', '') for intronic or unparseable variants.
    """
    if not isinstance(hgvs_nt, str):
        return None, '', ''
    cdna = hgvs_nt.split(':', 1)[1] if ':' in hgvs_nt else hgvs_nt
    m = _CDNA_SNV_RE.search(cdna)
    if m:
        return int(m.group(1)), m.group(2), m.group(3)
    return None, '', ''


def extract_window(ref_seq: str, cdna_pos: int, ref: str, alt: str,
                   window: int = SEQ_WINDOW) -> str:
    """
    Extract a `window`-bp sequence centred on `cdna_pos` (1-based cDNA coord),
    apply the alt substitution, and return the mutant string.
    """
    idx  = cdna_pos - 1   # 0-based
    half = window // 2
    start = max(0, idx - half)
    end   = min(len(ref_seq), idx + half)
    seq   = ref_seq[start:end]

    left_pad  = "N" * max(0, half - idx)
    right_pad = "N" * max(0, (idx + half) - len(ref_seq))
    seq = left_pad + seq + right_pad

    centre = idx - start + len(left_pad)
    if centre < len(seq) and seq[centre] == ref:
        seq = seq[:centre] + alt + seq[centre + 1:]

    return seq[:window].upper()


def build_sequences(df: pd.DataFrame, ref_seq) -> list:
    seqs = []
    for _, row in df.iterrows():
        hgvs_nt = row.get("hgvs_nt", "")
        pos, ref, alt = parse_hgvs_nt(hgvs_nt)

        if ref_seq is not None and pos is not None and ref and alt:
            seqs.append(extract_window(ref_seq, pos, ref, alt))
        else:
            seqs.append("N" * SEQ_WINDOW)
    return seqs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("Preparing BRCA2 SGE exon-stratified splits...\n")

    if not SGE_BRCA2_RAW.exists():
        print(f"BRCA2 SGE data not found: {SGE_BRCA2_RAW}")
        print("Run: python src/data/download_brca2_sge.py")
        sys.exit(1)

    df = pd.read_csv(SGE_BRCA2_RAW)
    print(f"Loaded: {len(df):,} rows")

    df = df.dropna(subset=["score", "experiment"])
    print(f"After dropping missing score/experiment: {len(df):,}")

    known_exons = TRAIN_EXONS | VAL_EXONS | TEST_EXONS
    unknown = df[~df["experiment"].isin(known_exons)]
    if len(unknown):
        unknown_labels = sorted(unknown["experiment"].unique())
        print(f"WARNING: {len(unknown)} rows with unrecognised exon labels {unknown_labels} — dropped")
        df = df[df["experiment"].isin(known_exons)]

    def assign_split(exp: str) -> str:
        if exp in TRAIN_EXONS:
            return "train"
        if exp in VAL_EXONS:
            return "val"
        if exp in TEST_EXONS:
            return "test"
        return "unknown"

    df["split"] = df["experiment"].apply(assign_split)

    # Extract sequence windows
    print("\nExtracting sequence windows...")
    ref_seq = load_brca2_seq()
    if ref_seq is None:
        print("  BRCA2 reference not found — sequences will be placeholder N's.")
        print("  Run src/data/download.py first for real sequence windows.")
    else:
        print(f"  BRCA2 reference loaded: {len(ref_seq):,} bp")

    df["sequence"] = build_sequences(df, ref_seq)

    # Split and save
    BRCA2_SGE_SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        part = df[df["split"] == split].reset_index(drop=True)
        out  = BRCA2_SGE_SPLITS_DIR / f"brca2_sge_{split}.csv"
        part.to_csv(out, index=False)
        n_lof  = (part["function_class"] == "LOF").sum()
        n_func = (part["function_class"] == "FUNC").sum()
        n_int  = (part["function_class"] == "INT").sum()
        exons  = sorted(part["experiment"].unique())
        print(f"\n  {split:5s}: {len(part):4d} variants  "
              f"LOF={n_lof}  FUNC={n_func}  INT={n_int}  exons={exons}")
        print(f"         -> {out}")

    print("\nDone.")


if __name__ == "__main__":
    main()
