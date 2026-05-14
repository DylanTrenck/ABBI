import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config import (
    BRCA1_REF, BRCA2_REF,
    CLINVAR_BRCA1, CLINVAR_BRCA2,
    CLINVAR_CLEAN_PATH, PROCESSED_DIR,
    SEED, SEQ_WINDOW,
    SEQUENCES_PATH, SPLITS_DIR,
    TRAIN_FRAC, VAL_FRAC,
    X_CLIN_PATH, X_CLIN_SEQ_PATH, Y_PATH,
)


# ---------------------------------------------------------------------------
# Section 1 — Load & merge ClinVar files
# ---------------------------------------------------------------------------

def load_clinvar() -> pd.DataFrame:
    df1 = pd.read_csv(CLINVAR_BRCA1, sep="\t", low_memory=False)
    df1["gene"] = "BRCA1"

    df2 = pd.read_csv(CLINVAR_BRCA2, sep="\t", low_memory=False)
    df2["gene"] = "BRCA2"

    df = pd.concat([df1, df2], ignore_index=True)
    print(f"Loaded {len(df):,} variants  (BRCA1: {len(df1):,}  BRCA2: {len(df2):,})")
    return df


def encode_labels(df: pd.DataFrame) -> pd.DataFrame:

    label_map = {"Pathogenic": 1, "Benign": 0}
    df = df[df["Germline classification"].isin(label_map)].copy()
    df["label"] = df["Germline classification"].map(label_map).astype("float32")

    n_path = (df["label"] == 1).sum()
    n_benign = (df["label"] == 0).sum()
    print(f"Labels  ->  Pathogenic: {n_path:,}  Benign: {n_benign:,}  "
          f"(ratio {n_path/n_benign:.1f}:1)")
    return df


# ---------------------------------------------------------------------------
# Section 2 — Parse Canonical SPDI and GRCh38Location
# ---------------------------------------------------------------------------

def parse_spdi(spdi: str) -> tuple:
    """Return (pos_0based, ref_allele, alt_allele) from a Canonical SPDI string.
    Format: NC_XXXXXX.XX:pos:ref:alt
    Returns (None, None, None) if the field is missing or malformed.
    """
    if not isinstance(spdi, str) or spdi.strip() == "":
        return None, None, None
    parts = spdi.strip().split(":")
    if len(parts) != 4:
        return None, None, None
    try:
        return int(parts[1]), parts[2], parts[3]
    except ValueError:
        return None, None, None


def parse_grch38_location(loc) -> int | None:
    """Return the start position as an integer.
    Handles both single positions ('43039471') and ranges ('43039999 - 43040000').
    """
    if pd.isna(loc):
        return None
    loc = str(loc).strip()
    if " - " in loc:
        return int(loc.split(" - ")[0].strip())
    try:
        return int(loc)
    except ValueError:
        return None


def parse_positions(df: pd.DataFrame) -> pd.DataFrame:
    parsed = df["Canonical SPDI"].apply(parse_spdi)
    df["spdi_pos"] = parsed.apply(lambda x: x[0])
    df["ref_allele"] = parsed.apply(lambda x: x[1])
    df["alt_allele"] = parsed.apply(lambda x: x[2])
    df["pos_start"] = df["GRCh38Location"].apply(parse_grch38_location)

    n_missing = df["spdi_pos"].isna().sum()
    if n_missing:
        print(f"  Warning: {n_missing} variants missing Canonical SPDI — dropping.")
        df = df.dropna(subset=["spdi_pos"]).copy()

    df["spdi_pos"] = df["spdi_pos"].astype(int)
    print(f"Positions parsed  ->  {len(df):,} variants retained")
    return df


# ---------------------------------------------------------------------------
# Section 3 — Build annotation features (X_clin)
# ---------------------------------------------------------------------------

VARIANT_TYPE_CATS = [
    "single nucleotide variant", "Deletion", "Insertion",
    "Duplication", "Microsatellite", "Indel", "other_vtype",
]

MOL_CSQ_MAP = {
    "missense":    ["missense variant"],
    "nonsense":    ["nonsense", "stop gained"],
    "frameshift":  ["frameshift variant", "frameshift"],
    "splice":      ["splice donor variant", "splice acceptor variant",
                    "splice site variant", "splice region variant"],
    "utr5":        ["5 prime utr variant", "5 prime utr"],
    "utr3":        ["3 prime utr variant", "3 prime utr"],
    "intron":      ["intron variant", "intronic"],
    "synonymous":  ["synonymous variant", "synonymous"],
}


def _map_mol_csq(raw: str) -> str:
    if not isinstance(raw, str) or raw.strip() == "":
        return "other_csq"
    token = raw.split("|")[0].strip().lower()
    for bucket, patterns in MOL_CSQ_MAP.items():
        if any(p in token for p in patterns):
            return bucket
    return "other_csq"


def _map_variant_type(raw: str) -> str:
    if isinstance(raw, str) and raw.strip() in VARIANT_TYPE_CATS:
        return raw.strip()
    return "other_vtype"


def build_annotation_features(df: pd.DataFrame, train_idx: np.ndarray | None = None
                               ) -> tuple[np.ndarray, list[str], StandardScaler]:
    feat = pd.DataFrame(index=df.index)

    # Binary gene flag
    feat["is_brca2"] = (df["gene"] == "BRCA2").astype("float32")

    # Allele length features
    feat["ref_len"] = df["ref_allele"].str.len().astype("float32")
    feat["alt_len"] = df["alt_allele"].str.len().astype("float32")
    feat["len_diff"] = (feat["alt_len"] - feat["ref_len"]).astype("float32")

    # Variant type one-hot
    vtype = df["Variant type"].apply(_map_variant_type)
    for cat in VARIANT_TYPE_CATS:
        feat[f"vtype_{cat.lower().replace(' ', '_')}"] = (vtype == cat).astype("float32")

    # Molecular consequence one-hot
    csq_buckets = list(MOL_CSQ_MAP.keys()) + ["other_csq"]
    csq = df["Molecular consequence"].apply(_map_mol_csq)
    for bucket in csq_buckets:
        feat[f"csq_{bucket}"] = (csq == bucket).astype("float32")

    # Genomic position — StandardScaler fitted on train only
    pos = df["pos_start"].fillna(df["spdi_pos"]).astype("float32").values.reshape(-1, 1)
    scaler = StandardScaler()
    if train_idx is not None:
        scaler.fit(pos[train_idx])
    else:
        scaler.fit(pos)
    feat["pos_norm"] = scaler.transform(pos).astype("float32").ravel()

    feature_names = list(feat.columns)
    X = feat.values.astype("float32")
    print(f"Annotation features  ->  shape {X.shape}  ({len(feature_names)} features)")
    return X, feature_names, scaler


# ---------------------------------------------------------------------------
# Section 4 — Extract sequence context windows (DNABERT-2 input)
# ---------------------------------------------------------------------------

def _load_ref_sequences() -> dict:
    """Load BRCA1 and BRCA2 mRNA sequences from GenBank files.
    Returns {'BRCA1': SeqRecord, 'BRCA2': SeqRecord} or empty dict if files missing.
    """
    from Bio import SeqIO

    refs = {}
    for gene, path in [
        ("BRCA1", BRCA1_REF),
        ("BRCA2", BRCA2_REF),
    ]:
        if not path.exists():
            print(f"  Reference {path.name} not found — skipping sequence extraction.")
            return {}
        record = SeqIO.read(open(path, encoding="utf-8"), "genbank")
        # Locate CDS start offset so we can convert c. positions to mRNA positions
        cds_starts = [f.location.start for f in record.features if f.type == "CDS"]
        record.cds_start = int(cds_starts[0]) if cds_starts else 0
        refs[gene] = record
    return refs


def _extract_window(mrna_seq: str, cds_start: int, cdna_pos: int,
                    ref_allele: str, alt_allele: str, window: int) -> str:
    """Extract a fixed-length DNA context window centered on the variant.

    cdna_pos is 1-based CDS position (from HGVS c. notation).
    Converts to 0-based mRNA index, applies the substitution, returns the window.
    """
    half = window // 2
    # Convert c. position to 0-based mRNA index
    mrna_idx = cds_start + cdna_pos - 1
    start = max(0, mrna_idx - half)
    end = min(len(mrna_seq), mrna_idx + half)
    window_seq = list(mrna_seq[start:end].upper())

    # Apply variant at the variant position within the window
    var_in_window = mrna_idx - start
    if 0 <= var_in_window < len(window_seq):
        ref_len = len(ref_allele)
        if "".join(window_seq[var_in_window:var_in_window + ref_len]).upper() == ref_allele.upper():
            window_seq[var_in_window:var_in_window + ref_len] = list(alt_allele.upper())

    seq = "".join(window_seq)
    # Pad with N's if near chromosome boundary
    if len(seq) < window:
        seq = seq.ljust(window, "N")
    return seq[:window]


def _parse_cdna_pos(hgvs_name: str) -> int | None:
    """Extract the first integer position from an HGVS c. notation string.
    e.g. 'NM_007294.3(BRCA1):c.5503_5564del' -> 5503
         'NM_007294.3(BRCA1):c.*6207C>T'     -> 6207 (UTR, keep for context)
         'NM_007294.3(BRCA1):c.-175C>T'      -> -175 (upstream, keep)
    """
    import re
    match = re.search(r":c\.[\*\-]?(\d+)", str(hgvs_name))
    if match:
        return int(match.group(1))
    return None


def extract_sequences(df: pd.DataFrame) -> np.ndarray | None:
    refs = _load_ref_sequences()
    if not refs:
        return None

    sequences = []
    skipped = 0
    for _, row in df.iterrows():
        gene = row["gene"]
        record = refs.get(gene)
        if record is None:
            sequences.append("N" * SEQ_WINDOW)
            skipped += 1
            continue

        cdna_pos = _parse_cdna_pos(row["Name"])
        if cdna_pos is None or cdna_pos <= 0:
            sequences.append("N" * SEQ_WINDOW)
            skipped += 1
            continue

        seq = _extract_window(
            mrna_seq=str(record.seq),
            cds_start=record.cds_start,
            cdna_pos=cdna_pos,
            ref_allele=str(row["ref_allele"]),
            alt_allele=str(row["alt_allele"]),
            window=SEQ_WINDOW,
        )
        sequences.append(seq)

    result = np.array(sequences, dtype=object)
    print(f"Sequences extracted  ->  {len(result):,} windows  "
          f"({skipped:,} padded with N's)")
    return result


# ---------------------------------------------------------------------------
# Section 5 — Stratified splits + save all outputs
# ---------------------------------------------------------------------------

def make_splits(y: np.ndarray, force: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from sklearn.model_selection import train_test_split

    train_file = SPLITS_DIR / "train_idx.npy"
    if train_file.exists() and not force:
        print("Splits already exist — loading from disk. Use --force-splits to recreate.")
        train_idx = np.load(SPLITS_DIR / "train_idx.npy")
        val_idx = np.load(SPLITS_DIR / "val_idx.npy")
        test_idx = np.load(SPLITS_DIR / "test_idx.npy")
        return train_idx, val_idx, test_idx

    np.random.seed(SEED)
    idx = np.arange(len(y))

    train_idx, temp_idx = train_test_split(
        idx, test_size=(1 - TRAIN_FRAC), stratify=y, random_state=SEED
    )
    # Split temp evenly into val and test
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.5, stratify=y[temp_idx], random_state=SEED
    )

    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    np.save(SPLITS_DIR / "train_idx.npy", train_idx)
    np.save(SPLITS_DIR / "val_idx.npy", val_idx)
    np.save(SPLITS_DIR / "test_idx.npy", test_idx)

    def _split_summary(name, idxs):
        n_path = y[idxs].sum()
        print(f"  {name:6s}: {len(idxs):,} variants  "
              f"({int(n_path):,} path / {len(idxs)-int(n_path):,} benign)")

    print("Splits saved:")
    _split_summary("train", train_idx)
    _split_summary("val", val_idx)
    _split_summary("test", test_idx)
    return train_idx, val_idx, test_idx


def save_outputs(df: pd.DataFrame, X_clin: np.ndarray, y: np.ndarray,
                 sequences: np.ndarray | None, scaler: StandardScaler,
                 feature_names: list[str]) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    df.to_csv(CLINVAR_CLEAN_PATH, index=False)
    np.save(X_CLIN_PATH, X_clin)
    np.save(Y_PATH, y)
    if sequences is not None:
        np.save(SEQUENCES_PATH, sequences)

    with open(PROCESSED_DIR / "scaler.pkl", "wb") as fh:
        pickle.dump(scaler, fh)
    with open(PROCESSED_DIR / "feature_names.txt", "w") as fh:
        fh.write("\n".join(feature_names))

    # Reduced feature matrix: drop mol_csq columns (csq_*) for seq-ablation experiments
    csq_mask = [i for i, n in enumerate(feature_names) if not n.startswith("csq_")]
    X_clin_seq = X_clin[:, csq_mask]
    feature_names_seq = [feature_names[i] for i in csq_mask]
    np.save(X_CLIN_SEQ_PATH, X_clin_seq)
    with open(PROCESSED_DIR / "feature_names_seq.txt", "w") as fh:
        fh.write("\n".join(feature_names_seq))

    print(f"\nSaved to {PROCESSED_DIR}:")
    print(f"  clinvar_clean.csv      {len(df):,} rows")
    print(f"  X_clin.npy             {X_clin.shape}  (full, {len(feature_names)} features)")
    print(f"  X_clin_seq.npy         {X_clin_seq.shape}  (no mol_csq, {len(feature_names_seq)} features)")
    print(f"  y.npy                  {y.shape}")
    if sequences is not None:
        print(f"  sequences.npy          {sequences.shape}")
    print(f"  scaler.pkl + feature_names*.txt")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    np.random.seed(SEED)

    df = load_clinvar()
    df = encode_labels(df)
    df = parse_positions(df)

    y = df["label"].values.astype("float32")

    # Compute splits first so scaler is fitted on train indices only
    train_idx, *_ = make_splits(y, force=args.force_splits)

    X_clin, feature_names, scaler = build_annotation_features(df, train_idx=train_idx)
    sequences = extract_sequences(df)

    save_outputs(df, X_clin, y, sequences, scaler, feature_names)
    print("\nPreprocessing complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess ClinVar BRCA1/2 data for ABBI.")
    parser.add_argument("--force-splits", action="store_true",
                        help="Recreate train/val/test splits even if they already exist.")
    main(parser.parse_args())
