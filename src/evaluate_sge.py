"""
Evaluate the SGE functional score regressor on the held-out test exons (X22, X23).

Metrics:
  - Spearman rho (predicted score vs ground-truth SGE score)
  - Pearson r
  - ROC AUC  (LOF=1 vs FUNC=0, ignoring INT)
  - Scatter plot: predicted vs actual score
  - Box plot: predicted score by function_class (LOF / INT / FUNC)

Usage:
  python src/evaluate_sge.py
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import (
    BRCA2_SGE_MODEL_PATH, BRCA2_SGE_SPLITS_DIR,
    RESULTS_DIR, SGE_MODEL_PATH, SGE_SPLITS_DIR,
)
from src.models.sge_regressor import SGERegressor, _BRCA1_AA_LEN, build_tabular
from src.train_sge import SGEDataset, collate

_BRCA2_AA_LEN = 3418.0
FIGURES_DIR = RESULTS_DIR / "figures"


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def run_inference(model: SGERegressor, df: pd.DataFrame,
                  device: torch.device, batch_size: int = 32,
                  aa_len: float = _BRCA1_AA_LEN) -> np.ndarray:
    ds     = SGEDataset(df, aa_len=aa_len)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate)
    model.eval()
    preds = []
    with torch.no_grad():
        for seqs, tabs, _ in loader:
            preds.extend(model(seqs, tabs, device).cpu().numpy())

    pred_arr = np.full(len(df), np.nan)
    pred_arr[ds.df.index] = preds
    return pred_arr


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def plot_scatter(actual: np.ndarray, predicted: np.ndarray,
                 rho: float, r: float, gene: str = "brca1",
                 test_exons: str = "X22, X23") -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(actual, predicted, alpha=0.35, s=12, color="#2E86AB")
    mn = min(actual.min(), predicted.min())
    mx = max(actual.max(), predicted.max())
    ax.plot([mn, mx], [mn, mx], "r--", lw=1, label="y = x")
    ax.set_xlabel("Actual SGE score")
    ax.set_ylabel("Predicted SGE score")
    ax.set_title(f"SGE Regressor ({gene.upper()}) — Test Exons ({test_exons})\n"
                 f"Spearman rho={rho:.3f}  Pearson r={r:.3f}")
    ax.legend()
    fig.tight_layout()
    suffix = "_brca2" if gene == "brca2" else ""
    out = FIGURES_DIR / f"sge_regressor_scatter{suffix}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Scatter plot -> {out}")


def plot_boxplot(df: pd.DataFrame, gene: str = "brca1",
                 test_exons: str = "X22, X23") -> None:
    order  = ["LOF", "INT", "FUNC"]
    colors = {"LOF": "#E63946", "INT": "#F4A261", "FUNC": "#2A9D8F"}
    fig, ax = plt.subplots(figsize=(5, 5))
    data = [df.loc[df["function_class"] == cls, "pred_score"].dropna().values
            for cls in order]
    bp = ax.boxplot(data, patch_artist=True, medianprops={"color": "black", "lw": 2})
    for patch, cls in zip(bp["boxes"], order):
        patch.set_facecolor(colors[cls])
    ax.set_xticklabels(order)
    ax.set_ylabel("Predicted SGE score")
    ax.set_title(f"Predicted Score by Functional Class ({gene.upper()})\n"
                 f"Test Exons ({test_exons})")
    fig.tight_layout()
    suffix = "_brca2" if gene == "brca2" else ""
    out = FIGURES_DIR / f"sge_regressor_boxplot{suffix}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Box plot      -> {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gene", choices=["brca1", "brca2"], default="brca1",
                        help="Which gene's SGE model to evaluate.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  Gene: {args.gene.upper()}\n")

    # Gene-specific paths and settings
    if args.gene == "brca2":
        splits_dir = BRCA2_SGE_SPLITS_DIR
        model_path = BRCA2_SGE_MODEL_PATH
        aa_len     = _BRCA2_AA_LEN
        prefix     = "brca2_sge"
        prep_cmd   = "src/data/prepare_brca2_sge_splits.py"
        train_cmd  = "src/train_sge.py --gene brca2"
        fig_suffix = "_brca2"
        out_suffix = "_brca2"
    else:
        splits_dir = SGE_SPLITS_DIR
        model_path = SGE_MODEL_PATH
        aa_len     = _BRCA1_AA_LEN
        prefix     = "sge"
        prep_cmd   = "src/data/prepare_sge_splits.py"
        train_cmd  = "src/train_sge.py"
        fig_suffix = ""
        out_suffix = ""

    # --- Load test split ---
    test_path = splits_dir / f"{prefix}_test.csv"
    if not test_path.exists():
        print(f"Test split not found: {test_path}")
        print(f"Run: python {prep_cmd}")
        sys.exit(1)

    test_df = pd.read_csv(test_path)
    print(f"Test variants: {len(test_df):,}  "
          f"(exons: {sorted(test_df['experiment'].unique())})")
    print(f"  LOF={(test_df['function_class']=='LOF').sum()}  "
          f"INT={(test_df['function_class']=='INT').sum()}  "
          f"FUNC={(test_df['function_class']=='FUNC').sum()}")

    # --- Load model ---
    if not model_path.exists():
        print(f"\nCheckpoint not found: {model_path}")
        print(f"Run: python {train_cmd}")
        sys.exit(1)

    ckpt  = torch.load(model_path, map_location=device)
    model = SGERegressor(unfreeze_last_block=ckpt.get("unfreeze_last", False))
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    print(f"\nLoaded checkpoint: epoch {ckpt['epoch']}  val_rho={ckpt['val_rho']:.4f}")

    # --- Predict ---
    print("\nRunning inference on test set...")
    test_df["pred_score"] = run_inference(model, test_df, device, aa_len=aa_len)

    valid = test_df[test_df["pred_score"].notna() & test_df["score"].notna()]
    print(f"Valid predictions: {len(valid):,} / {len(test_df):,}")

    actual    = valid["score"].values
    predicted = valid["pred_score"].values

    rho, p_rho = spearmanr(predicted, actual)
    r,   p_r   = pearsonr(predicted, actual)

    # ROC AUC: LOF (1) vs FUNC (0), exclude INT
    clf = valid[valid["function_class"].isin(["LOF", "FUNC"])].copy()
    clf["label"] = (clf["function_class"] == "LOF").astype(int)
    auc = roc_auc_score(clf["label"], -clf["pred_score"])  # lower score = more LOF

    test_exons_str = ", ".join(sorted(test_df["experiment"].unique()))
    print(f"\n{'='*55}")
    print(f"  Test-set results ({args.gene.upper()}, exons: {test_exons_str}):")
    print(f"  Spearman rho: {rho:.4f}  (p={p_rho:.2e})")
    print(f"  Pearson  r  : {r:.4f}  (p={p_r:.2e})")
    print(f"  ROC AUC     : {auc:.4f}  "
          f"(LOF vs FUNC, n={len(clf):,})")
    print(f"{'='*55}")

    # Per-class mean predictions
    print("\n  Mean predicted score by class:")
    for cls in ["LOF", "INT", "FUNC"]:
        sub = valid[valid["function_class"] == cls]["pred_score"]
        if len(sub):
            print(f"    {cls:4s}: {sub.mean():+.3f}  (n={len(sub)})")

    # --- Figures ---
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    print("\nGenerating figures...")
    plot_scatter(actual, predicted, rho, r, gene=args.gene, test_exons=test_exons_str)
    plot_boxplot(valid, gene=args.gene, test_exons=test_exons_str)

    # --- Save per-variant predictions ---
    out_csv = RESULTS_DIR / f"sge_regressor_predictions{out_suffix}.csv"
    save_cols = [c for c in ["hgvs_nt", "experiment", "function_class", "score",
                              "pred_score", "clinvar_simple"] if c in test_df.columns]
    test_df[save_cols].to_csv(out_csv, index=False)
    print(f"\nPer-variant predictions -> {out_csv}")


if __name__ == "__main__":
    main()
