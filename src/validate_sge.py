"""
Validate ABBI against Findlay et al. 2018 BRCA1 saturation genome editing data.

Loads the SGE functional scores, excludes any variant present in our training/val/test
sets to prevent data leakage, then runs ABBI inference and measures how well the
predicted pathogenicity probability correlates with the independently-measured
functional score.

Sign convention:
  SGE score:  LOW  = loss of function (pathogenic-like)
              HIGH = functional      (benign-like)
  ABBI prob:  HIGH = pathogenic, LOW = benign
  Expected Spearman rho: NEGATIVE (high ABBI prob <-> low SGE score)

We also report ROC AUC treating SGE "LOF" class as the positive (pathogenic) label.

Outputs:
  results/sge_validation.csv          — per-variant predictions
  results/figures/sge_scatter.png     — ABBI prob vs SGE score (coloured by class)
  results/figures/sge_boxplot.png     — ABBI prob distribution by SGE class
  results/figures/sge_roc.png         — ROC curve (LOF vs FUNC)
"""

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score, roc_curve
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import (
    BATCH_SIZE,
    BRCA1_REF,
    CLINVAR_CLEAN_PATH,
    EMBED_DIM,
    MODELS_DIR,
    PROCESSED_DIR,
    RESULTS_DIR,
    SEED,
    SEQ_WINDOW,
    SGE_BRCA1_RAW,
    SPLITS_DIR,
    X_CLIN_SEQ_PATH,
)
from src.models.fusion import ABBIModel

FIGURES_DIR = RESULTS_DIR / "figures"

torch.manual_seed(SEED)
np.random.seed(SEED)


# ---------------------------------------------------------------------------
# HGVS parsing
# ---------------------------------------------------------------------------

def parse_hgvs_cdna(hgvs: str) -> tuple[int | None, str | None, str | None]:
    """Extract (cdna_pos, ref, alt) from NM_007294.X:c.NNNN[A-Z]>[A-Z] notation.

    Returns (None, None, None) for unparseable or non-SNV entries.
    """
    if not isinstance(hgvs, str):
        return None, None, None

    # cDNA substitution: c.5095C>T  (may have NM_... prefix)
    m = re.search(r"c\.[\*\-]?(\d+)([ACGT])>([ACGT])", hgvs)
    if m:
        return int(m.group(1)), m.group(2), m.group(3)

    # Genomic substitution: g.43115721C>T (less common in MAVE-DB)
    m = re.search(r"g\.(\d+)([ACGT])>([ACGT])", hgvs)
    if m:
        # Genomic coordinate — store as negative to flag it; handle separately
        return -int(m.group(1)), m.group(2), m.group(3)

    return None, None, None


# ---------------------------------------------------------------------------
# Sequence window extraction (mirrors preprocess.py logic)
# ---------------------------------------------------------------------------

def load_brca1_ref() -> tuple[str, int]:
    """Return (mrna_seq_str, cds_start) for BRCA1 reference."""
    from Bio import SeqIO
    record = SeqIO.read(open(BRCA1_REF, encoding="utf-8"), "genbank")
    cds_starts = [f.location.start for f in record.features if f.type == "CDS"]
    cds_start = int(cds_starts[0]) if cds_starts else 0
    return str(record.seq).upper(), cds_start


def extract_window(mrna_seq: str, cds_start: int, cdna_pos: int,
                   ref: str, alt: str, window: int = SEQ_WINDOW) -> str:
    half = window // 2
    mrna_idx = cds_start + cdna_pos - 1
    start = max(0, mrna_idx - half)
    end = min(len(mrna_seq), mrna_idx + half)
    w = list(mrna_seq[start:end])
    var_in_w = mrna_idx - start
    if 0 <= var_in_w < len(w) and w[var_in_w].upper() == ref.upper():
        w[var_in_w] = alt.upper()
    seq = "".join(w)
    return seq.ljust(window, "N")[:window]


# ---------------------------------------------------------------------------
# Annotation feature matrix for SGE variants
# ---------------------------------------------------------------------------

def build_sge_features(sge_df: pd.DataFrame) -> np.ndarray:
    """Build a 12-column X_clin_seq matrix for SGE variants.

    All Findlay 2018 variants are BRCA1 SNVs, so most columns are fixed:
      is_brca2=0, ref_len=1, alt_len=1, len_diff=0,
      vtype_single_nucleotide_variant=1, all other vtypes=0.
    The only variable column is pos_norm, normalised with the saved scaler.
    """
    import pickle

    with open(PROCESSED_DIR / "scaler.pkl", "rb") as fh:
        scaler = pickle.load(fh)

    # Use cdna_pos as a proxy for position (scaler was fitted on genomic pos_start,
    # but relative ordering is preserved and we just need normalisation range)
    with open(PROCESSED_DIR / "feature_names_seq.txt") as fh:
        feature_names = [l.strip() for l in fh if l.strip()]

    n = len(sge_df)
    feat = pd.DataFrame(0.0, index=range(n), columns=feature_names, dtype="float32")

    feat["is_brca2"] = 0.0
    feat["ref_len"]  = 1.0
    feat["alt_len"]  = 1.0
    feat["len_diff"] = 0.0
    feat["vtype_single_nucleotide_variant"] = 1.0

    # Normalise position using the saved scaler (fitted on genomic pos_start values)
    pos = sge_df["cdna_pos"].fillna(0).values.astype("float32").reshape(-1, 1)
    feat["pos_norm"] = scaler.transform(pos).ravel().astype("float32")

    return feat.values.astype("float32")


# ---------------------------------------------------------------------------
# Dataset + inference
# ---------------------------------------------------------------------------

class SGEDataset(Dataset):
    def __init__(self, sequences: list[str], x_clin: np.ndarray):
        self.sequences = sequences
        self.x_clin = torch.tensor(x_clin, dtype=torch.float32)

    def __len__(self): return len(self.sequences)

    def __getitem__(self, idx): return self.sequences[idx], self.x_clin[idx]


def collate_sge(batch):
    seqs, x_clin = zip(*batch)
    return list(seqs), torch.stack(x_clin)


@torch.no_grad()
def run_inference(model: ABBIModel, sequences: list[str],
                  x_clin: np.ndarray, device: torch.device) -> np.ndarray:
    ds = SGEDataset(sequences, x_clin)
    loader = DataLoader(ds, batch_size=min(BATCH_SIZE, 64),
                        shuffle=False, collate_fn=collate_sge)
    probs = []
    for seqs, x_c in loader:
        x_c = x_c.to(device)
        logits = model(seqs, x_c)
        probs.extend(torch.sigmoid(logits).cpu().numpy())
    return np.array(probs)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def compute_stats(sge_df: pd.DataFrame, probs: np.ndarray) -> None:
    scores = sge_df["score"].values

    # Spearman rho: expect negative (high ABBI prob <-> low SGE score)
    rho, pval = spearmanr(probs, scores)
    print(f"\n  Spearman rho (ABBI prob vs SGE score): {rho:+.4f}  p={pval:.2e}")
    print(f"  (Negative rho means high pathogenicity prob aligns with low function score — expected)")

    # ROC AUC: LOF as positive class
    if "function_class" in sge_df.columns:
        has_label = sge_df["function_class"].isin(["LOF", "FUNC"])
        if has_label.sum() > 0:
            y_bin = (sge_df.loc[has_label, "function_class"] == "LOF").astype(int).values
            p_bin = probs[has_label.values]
            if len(np.unique(y_bin)) == 2:
                auc = roc_auc_score(y_bin, p_bin)
                print(f"  ROC AUC (LOF=1 vs FUNC=0): {auc:.4f}  "
                      f"(n_LOF={y_bin.sum()}, n_FUNC={(y_bin==0).sum()})")

    # Break down by ClinVar status
    if "clinvar_status" in sge_df.columns:
        print(f"\n  Breakdown by ClinVar status:")
        for status in sge_df["clinvar_status"].unique():
            mask = sge_df["clinvar_status"] == status
            r, p = spearmanr(probs[mask], scores[mask])
            print(f"    {status:<20}  n={mask.sum():>4}  rho={r:+.4f}  p={p:.2e}")


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def make_figures(sge_df: pd.DataFrame, probs: np.ndarray) -> None:
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        palette = {"FUNC": "#4C9BE8", "INTERM": "#F0A500", "LOF": "#E84C4C"}

        # --- Scatter: ABBI prob vs SGE score ---
        fig, ax = plt.subplots(figsize=(7, 5))
        if "function_class" in sge_df.columns:
            for cls, grp in sge_df.groupby("function_class"):
                mask = sge_df["function_class"] == cls
                ax.scatter(probs[mask], sge_df.loc[mask, "score"],
                           label=cls, alpha=0.35, s=8,
                           color=palette.get(cls, "grey"))
            ax.legend(title="SGE class", markerscale=3)
        else:
            ax.scatter(probs, sge_df["score"], alpha=0.3, s=8)

        rho, _ = spearmanr(probs, sge_df["score"].values)
        ax.set_xlabel("ABBI pathogenicity probability")
        ax.set_ylabel("SGE function score  (low = LOF)")
        ax.set_title(f"ABBI vs Findlay 2018 SGE  |  Spearman ρ = {rho:+.3f}")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "sge_scatter.png", dpi=150, bbox_inches="tight")
        plt.close()

        # --- Boxplot: ABBI prob by SGE class ---
        if "function_class" in sge_df.columns:
            plot_df = sge_df.copy()
            plot_df["abbi_prob"] = probs
            order = [c for c in ["LOF", "INTERM", "FUNC"]
                     if c in plot_df["function_class"].values]

            fig, ax = plt.subplots(figsize=(5, 5))
            sns.boxplot(data=plot_df, x="function_class", y="abbi_prob",
                        order=order, palette=palette, ax=ax, width=0.5)
            ax.set_xlabel("SGE functional class")
            ax.set_ylabel("ABBI pathogenicity probability")
            ax.set_title("ABBI predictions by SGE class")
            plt.tight_layout()
            plt.savefig(FIGURES_DIR / "sge_boxplot.png", dpi=150, bbox_inches="tight")
            plt.close()

        # --- ROC curve ---
        if "function_class" in sge_df.columns:
            has_label = sge_df["function_class"].isin(["LOF", "FUNC"])
            if has_label.sum() > 0:
                y_bin = (sge_df.loc[has_label, "function_class"] == "LOF").astype(int).values
                p_bin = probs[has_label.values]
                if len(np.unique(y_bin)) == 2:
                    fpr, tpr, _ = roc_curve(y_bin, p_bin)
                    auc = roc_auc_score(y_bin, p_bin)

                    fig, ax = plt.subplots(figsize=(5, 5))
                    ax.plot(fpr, tpr, lw=2, label=f"ABBI (AUC = {auc:.4f})")
                    ax.plot([0, 1], [0, 1], "k--", lw=1)
                    ax.set_xlabel("False Positive Rate")
                    ax.set_ylabel("True Positive Rate")
                    ax.set_title("ROC: LOF vs FUNC  (Findlay 2018 SGE)")
                    ax.legend()
                    plt.tight_layout()
                    plt.savefig(FIGURES_DIR / "sge_roc.png", dpi=150, bbox_inches="tight")
                    plt.close()

        print(f"\n  Figures saved -> {FIGURES_DIR}")
    except Exception as e:
        print(f"  (Figures skipped: {e})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(checkpoint: str = "abbi_best_seq_unfreeze.pt") -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Load SGE data ---
    if not SGE_BRCA1_RAW.exists():
        print(f"SGE data not found at {SGE_BRCA1_RAW}")
        print("Run: python src/data/download_sge.py")
        sys.exit(1)

    sge_df = pd.read_csv(SGE_BRCA1_RAW)
    print(f"Loaded SGE data: {len(sge_df):,} rows  columns={list(sge_df.columns)}")

    # --- Parse HGVS ---
    hgvs_col = next((c for c in ["hgvs_nt", "hgvs_base", "nucleotide"]
                     if c in sge_df.columns), None)
    if hgvs_col is None:
        raise ValueError(f"No HGVS nucleotide column found. Columns: {list(sge_df.columns)}")

    parsed = sge_df[hgvs_col].apply(parse_hgvs_cdna)
    sge_df["cdna_pos"] = parsed.apply(lambda x: x[0])
    sge_df["ref"]      = parsed.apply(lambda x: x[1])
    sge_df["alt"]      = parsed.apply(lambda x: x[2])

    # Drop non-SNV or unparseable rows
    n_before = len(sge_df)
    sge_df = sge_df.dropna(subset=["cdna_pos", "ref", "alt"]).copy()
    sge_df = sge_df[sge_df["cdna_pos"] > 0].copy()   # drop genomic-coordinate rows
    sge_df["cdna_pos"] = sge_df["cdna_pos"].astype(int)
    print(f"Parsed HGVS: {len(sge_df):,} SNVs retained (dropped {n_before - len(sge_df)})")

    # Drop rows with missing score
    score_col = next((c for c in ["score", "function_score", "func_score"]
                      if c in sge_df.columns), None)
    if score_col is None:
        raise ValueError(f"No score column found. Columns: {list(sge_df.columns)}")
    if score_col != "score":
        sge_df = sge_df.rename(columns={score_col: "score"})
    sge_df = sge_df.dropna(subset=["score"]).copy()
    print(f"After dropping missing scores: {len(sge_df):,} variants")

    # --- Check overlap with our training data ---
    clinvar_df = pd.read_csv(CLINVAR_CLEAN_PATH, low_memory=False)
    all_idx = np.concatenate([
        np.load(SPLITS_DIR / "train_idx.npy"),
        np.load(SPLITS_DIR / "val_idx.npy"),
        np.load(SPLITS_DIR / "test_idx.npy"),
    ])

    # Build a set of (cdna_pos, ref, alt) tuples from our full labeled dataset
    import re as _re
    def _cdna_pos_from_name(name):
        m = _re.search(r":c\.[\*\-]?(\d+)", str(name))
        return int(m.group(1)) if m else None

    clinvar_sub = clinvar_df.iloc[all_idx][["Name", "ref_allele", "alt_allele"]].copy()
    clinvar_sub["cdna_pos"] = clinvar_sub["Name"].apply(_cdna_pos_from_name)
    labeled_set = set(
        zip(clinvar_sub["cdna_pos"].dropna().astype(int),
            clinvar_sub["ref_allele"],
            clinvar_sub["alt_allele"])
    )
    print(f"Labeled variants in our dataset: {len(labeled_set):,} unique (pos, ref, alt) tuples")

    in_training = sge_df.apply(
        lambda r: (int(r["cdna_pos"]), r["ref"], r["alt"]) in labeled_set, axis=1
    )
    print(f"SGE variants in our training/val/test: {in_training.sum():,}")
    print(f"SGE variants NOT in our labeled data: {(~in_training).sum():,}  (used for validation)")

    sge_val = sge_df[~in_training].copy().reset_index(drop=True)

    # Annotate ClinVar status for each SGE variant
    clinvar_full = pd.read_csv(CLINVAR_CLEAN_PATH, low_memory=False)
    clinvar_full["cdna_pos"] = clinvar_full["Name"].apply(_cdna_pos_from_name)
    clinvar_status_map = {}
    for _, row in clinvar_full.iterrows():
        key = (row.get("cdna_pos"), row.get("ref_allele"), row.get("alt_allele"))
        status = row.get("Germline classification", "Unknown")
        clinvar_status_map[key] = status

    def get_clinvar_status(row):
        key = (row["cdna_pos"], row["ref"], row["alt"])
        return clinvar_status_map.get(key, "Not in ClinVar")

    sge_val["clinvar_status"] = sge_val.apply(get_clinvar_status, axis=1)
    print(f"\n  ClinVar status of validation variants:")
    for status, count in sge_val["clinvar_status"].value_counts().items():
        print(f"    {status}: {count}")

    # --- Load BRCA1 reference and extract sequence windows ---
    print(f"\nExtracting {len(sge_val):,} sequence windows...")
    mrna_seq, cds_start = load_brca1_ref()
    sequences = [
        extract_window(mrna_seq, cds_start, int(r["cdna_pos"]),
                       r["ref"], r["alt"])
        for _, r in sge_val.iterrows()
    ]
    n_padded = sum(1 for s in sequences if "N" in s)
    print(f"  {n_padded} windows contain N-padding (near gene boundary)")

    # --- Build annotation features ---
    x_clin = build_sge_features(sge_val)
    print(f"  Annotation features: {x_clin.shape}")

    # --- Load model and run inference ---
    print("\nRunning ABBI inference...")
    ckpt = torch.load(MODELS_DIR / checkpoint, map_location=device)
    model = ABBIModel(clin_input_dim=x_clin.shape[1])
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    print(f"  Checkpoint: epoch {ckpt['epoch']}  val_auc={ckpt['val_auc']:.4f}")

    probs = run_inference(model, sequences, x_clin, device)
    sge_val["abbi_prob"] = probs

    # --- Statistics ---
    print("\nValidation results:")
    compute_stats(sge_val, probs)

    # --- Figures ---
    make_figures(sge_val, probs)

    # --- Save per-variant results ---
    out_cols = [c for c in [hgvs_col, "cdna_pos", "ref", "alt", "score",
                             "function_class", "clinvar_status", "abbi_prob"]
                if c in sge_val.columns]
    out_path = RESULTS_DIR / "sge_validation.csv"
    sge_val[out_cols].to_csv(out_path, index=False)
    print(f"\nPer-variant results saved -> {out_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="abbi_best_seq_unfreeze.pt",
                        help="Checkpoint filename in models/ directory.")
    args = parser.parse_args()
    main(checkpoint=args.checkpoint)
