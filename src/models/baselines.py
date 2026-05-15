import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    roc_auc_score,
)
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config import (
    MODELS_DIR,
    RESULTS_DIR,
    SEED,
    SPLITS_DIR,
    X_CLIN_PATH,
    X_KMER_PATH,
    Y_PATH,
)

RESULTS_CSV = RESULTS_DIR / "baseline_results.csv"
RESULTS_COLS = [
    "timestamp", "model", "input",
    "auc_roc", "auprc", "f1", "sensitivity", "specificity",
    "train_size", "val_size", "notes",
]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data() -> dict:
    X_kmer = np.load(X_KMER_PATH)
    X_clin = np.load(X_CLIN_PATH)
    y = np.load(Y_PATH)

    train_idx = np.load(SPLITS_DIR / "train_idx.npy")
    val_idx = np.load(SPLITS_DIR / "val_idx.npy")

    return dict(
        X_kmer=X_kmer, X_clin=X_clin, y=y,
        train_idx=train_idx, val_idx=val_idx,
    )


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict:
    y_pred = (y_prob >= threshold).astype(int)

    tp = ((y_pred == 1) & (y_true == 1)).sum()
    tn = ((y_pred == 0) & (y_true == 0)).sum()
    fp = ((y_pred == 1) & (y_true == 0)).sum()
    fn = ((y_pred == 0) & (y_true == 1)).sum()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        "auc_roc":     round(roc_auc_score(y_true, y_prob), 4),
        "auprc":       round(average_precision_score(y_true, y_prob), 4),
        "f1":          round(f1_score(y_true, y_pred), 4),
        "sensitivity": round(sensitivity, 4),
        "specificity": round(specificity, 4),
    }


def _print_metrics(metrics: dict) -> None:
    print(f"    AUC-ROC:     {metrics['auc_roc']}")
    print(f"    AUPRC:       {metrics['auprc']}")
    print(f"    F1:          {metrics['f1']}")
    print(f"    Sensitivity: {metrics['sensitivity']}")
    print(f"    Specificity: {metrics['specificity']}")


# ---------------------------------------------------------------------------
# Results logging
# ---------------------------------------------------------------------------

def log_result(model: str, input_name: str, metrics: dict,
               train_size: int, val_size: int, notes: str = "") -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    write_header = not RESULTS_CSV.exists()

    with open(RESULTS_CSV, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=RESULTS_COLS)
        if write_header:
            writer.writeheader()
        writer.writerow({
            "timestamp":   datetime.now().strftime("%Y-%m-%d %H:%M"),
            "model":       model,
            "input":       input_name,
            "train_size":  train_size,
            "val_size":    val_size,
            "notes":       notes,
            **metrics,
        })


# ---------------------------------------------------------------------------
# Baseline 1 — Random Forest (sequence / k-mer only)
# ---------------------------------------------------------------------------

def run_random_forest(data: dict) -> None:
    print("\n[1/3] Random Forest  (k-mer features only)")

    X_tr = data["X_kmer"][data["train_idx"]]
    X_val = data["X_kmer"][data["val_idx"]]
    y_tr = data["y"][data["train_idx"]]
    y_val = data["y"][data["val_idx"]]

    clf = RandomForestClassifier(
        n_estimators=500,
        max_depth=None,
        class_weight="balanced",
        n_jobs=-1,
        random_state=SEED,
    )
    clf.fit(X_tr, y_tr)
    y_prob = clf.predict_proba(X_val)[:, 1]

    metrics = evaluate(y_val, y_prob)
    _print_metrics(metrics)
    log_result("RandomForest", "X_kmer", metrics,
               len(y_tr), len(y_val), "sequence-only ceiling; kmer TF-IDF k=3,4,5")

    import pickle
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(MODELS_DIR / "rf_baseline.pkl", "wb") as fh:
        pickle.dump(clf, fh)
    print("  Saved -> models/rf_baseline.pkl")


# ---------------------------------------------------------------------------
# Baseline 2 — XGBoost (annotation features only)
# ---------------------------------------------------------------------------

def run_xgboost(data: dict) -> None:
    print("\n[2/3] XGBoost  (annotation features only)")

    X_tr = data["X_clin"][data["train_idx"]]
    X_val = data["X_clin"][data["val_idx"]]
    y_tr = data["y"][data["train_idx"]]
    y_val = data["y"][data["val_idx"]]

    # scale_pos_weight rebalances the loss for the minority (benign) class
    scale = (y_tr == 0).sum() / (y_tr == 1).sum()

    clf = XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale,
        eval_metric="logloss",
        random_state=SEED,
        verbosity=0,
    )
    clf.fit(X_tr, y_tr)
    y_prob = clf.predict_proba(X_val)[:, 1]

    metrics = evaluate(y_val, y_prob)
    _print_metrics(metrics)
    log_result("XGBoost", "X_clin", metrics,
               len(y_tr), len(y_val), "annotation-only ceiling; 21 ClinVar features")

    clf.save_model(str(MODELS_DIR / "xgb_baseline.json"))
    print("  Saved -> models/xgb_baseline.json")


# ---------------------------------------------------------------------------
# Baseline 3 — Naive MLP (concatenated k-mer + annotation, no attention)
# ---------------------------------------------------------------------------

def run_naive_mlp(data: dict) -> None:
    print("\n[3/3] Naive MLP  (concat k-mer + annotation features)")

    X_tr = np.hstack([data["X_kmer"][data["train_idx"]],
                      data["X_clin"][data["train_idx"]]])
    X_val = np.hstack([data["X_kmer"][data["val_idx"]],
                       data["X_clin"][data["val_idx"]]])
    y_tr = data["y"][data["train_idx"]]
    y_val = data["y"][data["val_idx"]]

    clf = MLPClassifier(
        hidden_layer_sizes=(256, 64),
        activation="relu",
        max_iter=300,
        random_state=SEED,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10,
    )
    clf.fit(X_tr, y_tr)
    y_prob = clf.predict_proba(X_val)[:, 1]

    metrics = evaluate(y_val, y_prob)
    _print_metrics(metrics)
    log_result("NaiveMLP", "X_kmer+X_clin", metrics,
               len(y_tr), len(y_val), "naive concat fusion; no attention")

    import pickle
    with open(MODELS_DIR / "mlp_baseline.pkl", "wb") as fh:
        pickle.dump(clf, fh)
    print("  Saved -> models/mlp_baseline.pkl")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    print("Loading data...")
    data = load_data()
    print(f"  X_kmer: {data['X_kmer'].shape}  X_clin: {data['X_clin'].shape}")
    print(f"  Train: {len(data['train_idx']):,}  Val: {len(data['val_idx']):,}")

    if args.model in ("rf", "all"):
        run_random_forest(data)
    if args.model in ("xgb", "all"):
        run_xgboost(data)
    if args.model in ("mlp", "all"):
        run_naive_mlp(data)

    print(f"\nResults logged -> {RESULTS_CSV}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate ABBI baseline models.")
    parser.add_argument(
        "--model", choices=["rf", "xgb", "mlp", "all"], default="all",
        help="Which baseline to run (default: all)."
    )
    main(parser.parse_args())
