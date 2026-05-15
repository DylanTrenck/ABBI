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
from src.config import RESULTS_DIR, SGE_MODEL_PATH, SGE_SPLITS_DIR
from src.models.sge_regressor import SGERegressor, build_tabular
from src.train_sge import SGEDataset, collate

FIGURES_DIR = RESULTS_DIR / "figures"


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def run_inference(model: SGERegressor, df: pd.DataFrame,
                  device: torch.device, batch_size: int = 32) -> np.ndarray:
    ds     = SGEDataset(df)
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
                 rho: float, r: float) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(actual, predicted, alpha=0.35, s=12, color="#2E86AB")
    mn = min(actual.min(), predicted.min())
    mx = max(actual.max(), predicted.max())
    ax.plot([mn, mx], [mn, mx], "r--", lw=1, label="y = x")
    ax.set_xlabel("Actual SGE score (Findlay 2018)")
    ax.set_ylabel("Predicted SGE score")
    ax.set_title(f"SGE Regressor — Test Exons (X22, X23)\nSpearman ρ={rho:.3f}  Pearson r={r:.3f}")
    ax.legend()
    fig.tight_layout()
    out = FIGURES_DIR / "sge_regressor_scatter.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Scatter plot -> {out}")


def plot_boxplot(df: pd.DataFrame) -> None:
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
    ax.set_title("Predicted Score by Functional Class\n(Test Exons X22, X23)")
    fig.tight_layout()
    out = FIGURES_DIR / "sge_regressor_boxplot.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Box plot      -> {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # --- Load test split ---
    test_path = SGE_SPLITS_DIR / "sge_test.csv"
    if not test_path.exists():
        print(f"Test split not found: {test_path}")
        print("Run: python src/data/prepare_sge_splits.py")
        sys.exit(1)

    test_df = pd.read_csv(test_path)
    print(f"Test variants: {len(test_df):,}  "
          f"(exons: {sorted(test_df['experiment'].unique())})")
    print(f"  LOF={( test_df['function_class']=='LOF').sum()}  "
          f"INT={(test_df['function_class']=='INT').sum()}  "
          f"FUNC={(test_df['function_class']=='FUNC').sum()}")

    # --- Load model ---
    if not SGE_MODEL_PATH.exists():
        print(f"\nCheckpoint not found: {SGE_MODEL_PATH}")
        print("Run: python src/train_sge.py")
        sys.exit(1)

    ckpt  = torch.load(SGE_MODEL_PATH, map_location=device)
    model = SGERegressor(unfreeze_last_block=ckpt.get("unfreeze_last", False))
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    print(f"\nLoaded checkpoint: epoch {ckpt['epoch']}  val_rho={ckpt['val_rho']:.4f}")

    # --- Predict ---
    print("\nRunning inference on test set...")
    test_df["pred_score"] = run_inference(model, test_df, device)

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

    print(f"\n{'='*52}")
    print(f"  Test-set results (exons X22, X23):")
    print(f"  Spearman rho: {rho:.4f}  (p={p_rho:.2e})")
    print(f"  Pearson  r  : {r:.4f}  (p={p_r:.2e})")
    print(f"  ROC AUC     : {auc:.4f}  "
          f"(LOF vs FUNC, n={len(clf):,})")
    print(f"{'='*52}")

    # Per-class mean predictions
    print("\n  Mean predicted score by class:")
    for cls in ["LOF", "INT", "FUNC"]:
        sub = valid[valid["function_class"] == cls]["pred_score"]
        if len(sub):
            print(f"    {cls:4s}: {sub.mean():+.3f}  (n={len(sub)})")

    # --- Figures ---
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    print("\nGenerating figures...")
    plot_scatter(actual, predicted, rho, r)
    plot_boxplot(valid)

    # --- Save per-variant predictions ---
    out_csv = RESULTS_DIR / "sge_regressor_predictions.csv"
    test_df[["hgvs_nt", "experiment", "function_class", "score",
             "pred_score", "clinvar_simple"]].to_csv(out_csv, index=False)
    print(f"\nPer-variant predictions -> {out_csv}")


if __name__ == "__main__":
    main()
