"""
Evaluation script for ABBI.

Runs two evaluations on the held-out test set and prints a comparison table:
  1. ABBI (full model)   — loaded from models/abbi_best.pt
  2. Clinical-only       — ClinicalEncoder + ClassificationHead, no sequence encoder

Both models use the same loss, optimiser, and early stopping so the comparison is fair.
Bootstrap 95% CIs are reported for AUC-ROC, AUPRC, F1, sensitivity, and specificity.
Results are appended to results/evaluation_results.csv.
"""

import csv
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import (
    BATCH_SIZE,
    DROPOUT,
    EMBED_DIM,
    LEARNING_RATE,
    MODELS_DIR,
    PATIENCE,
    PROCESSED_DIR,
    RESULTS_DIR,
    SEED,
    SEQUENCES_PATH,
    SPLITS_DIR,
    X_CLIN_PATH,
    X_CLIN_SEQ_PATH,
    Y_PATH,
)
from src.models.encoders import ClinicalEncoder
from src.models.fusion import ABBIModel, ClassificationHead

RESULTS_CSV = RESULTS_DIR / "evaluation_results.csv"
RESULTS_COLS = [
    "timestamp", "model",
    "auc_roc", "auc_roc_lo", "auc_roc_hi",
    "auprc",   "auprc_lo",   "auprc_hi",
    "f1",      "f1_lo",      "f1_hi",
    "sensitivity", "sensitivity_lo", "sensitivity_hi",
    "specificity",  "specificity_lo",  "specificity_hi",
    "test_size", "notes",
]

N_BOOT = 1000
THRESHOLD = 0.5


# ---------------------------------------------------------------------------
# Dataset (mirrors train.py — clin-only variant omits sequences)
# ---------------------------------------------------------------------------

class BRCADataset(Dataset):
    def __init__(self, sequences, x_clin, y, indices):
        self.sequences = sequences[indices]
        self.x_clin = torch.tensor(x_clin[indices], dtype=torch.float32)
        self.y = torch.tensor(y[indices], dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.sequences[idx], self.x_clin[idx], self.y[idx]


def collate_fn(batch):
    seqs, x_clin, y = zip(*batch)
    return list(seqs), torch.stack(x_clin), torch.stack(y)


class ClinDataset(Dataset):
    def __init__(self, x_clin, y, indices):
        self.x_clin = torch.tensor(x_clin[indices], dtype=torch.float32)
        self.y = torch.tensor(y[indices], dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x_clin[idx], self.y[idx]


# ---------------------------------------------------------------------------
# Metrics + bootstrap CI
# ---------------------------------------------------------------------------

def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray,
                    threshold: float = THRESHOLD) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    tp = ((y_pred == 1) & (y_true == 1)).sum()
    tn = ((y_pred == 0) & (y_true == 0)).sum()
    fp = ((y_pred == 1) & (y_true == 0)).sum()
    fn = ((y_pred == 0) & (y_true == 1)).sum()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        "auc_roc":     roc_auc_score(y_true, y_prob),
        "auprc":       average_precision_score(y_true, y_prob),
        "f1":          f1_score(y_true, y_pred, zero_division=0),
        "sensitivity": sensitivity,
        "specificity": specificity,
    }


def bootstrap_ci(y_true: np.ndarray, y_prob: np.ndarray,
                 n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    rows = {k: [] for k in ["auc_roc", "auprc", "f1", "sensitivity", "specificity"]}

    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        # Skip degenerate samples that contain only one class
        if len(np.unique(y_true[idx])) < 2:
            continue
        m = compute_metrics(y_true[idx], y_prob[idx])
        for k in rows:
            rows[k].append(m[k])

    ci = {}
    for k, vals in rows.items():
        arr = np.array(vals)
        ci[f"{k}_lo"] = float(np.percentile(arr, 2.5))
        ci[f"{k}_hi"] = float(np.percentile(arr, 97.5))
    return ci


# ---------------------------------------------------------------------------
# ABBI evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def collect_predictions(model: nn.Module, loader: DataLoader,
                        device: torch.device, seq_model: bool = True):
    model.eval()
    all_probs, all_labels = [], []

    for batch in loader:
        if seq_model:
            seqs, x_clin, y = batch
            x_clin, y = x_clin.to(device), y.to(device)
            logits = model(seqs, x_clin)
        else:
            x_clin, y = batch
            x_clin, y = x_clin.to(device), y.to(device)
            logits = model(x_clin)

        all_probs.extend(torch.sigmoid(logits).cpu().numpy())
        all_labels.extend(y.cpu().numpy())

    return np.array(all_labels), np.array(all_probs)


def evaluate_abbi(sequences, x_clin, y, test_idx, device,
                  feature_set: str = "full") -> tuple[dict, dict]:
    print(f"\n[1/2] Evaluating ABBI ({feature_set} features)...")

    ckpt_path = MODELS_DIR / f"abbi_best_{feature_set}.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device)
    model = ABBIModel(clin_input_dim=x_clin.shape[1])
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    print(f"  Loaded checkpoint from epoch {ckpt['epoch']} "
          f"(val AUC {ckpt['val_auc']:.4f})")

    ds = BRCADataset(sequences, x_clin, y, test_idx)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                        collate_fn=collate_fn)

    y_true, y_prob = collect_predictions(model, loader, device, seq_model=True)
    metrics = compute_metrics(y_true, y_prob)
    ci = bootstrap_ci(y_true, y_prob)
    return metrics, ci


# ---------------------------------------------------------------------------
# Clinical-only ablation
# ---------------------------------------------------------------------------

class ClinicalOnlyModel(nn.Module):
    """ClinicalEncoder → ClassificationHead (no sequence input)."""

    def __init__(self, clin_input_dim: int, embed_dim: int = EMBED_DIM):
        super().__init__()
        self.encoder = ClinicalEncoder(input_dim=clin_input_dim, embed_dim=embed_dim)
        self.head = ClassificationHead(embed_dim=embed_dim)

    def forward(self, x_clin: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x_clin)).squeeze(-1)


def train_clinical_only(x_clin, y, train_idx, val_idx, device) -> ClinicalOnlyModel:
    print("\n[2/2] Training clinical-only ablation...")

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    train_ds = ClinDataset(x_clin, y, train_idx)
    val_ds   = ClinDataset(x_clin, y, val_idx)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)

    model = ClinicalOnlyModel(clin_input_dim=x_clin.shape[1]).to(device)

    n_neg = (y[train_idx] == 0).sum()
    n_pos = (y[train_idx] == 1).sum()
    pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-2)
    scheduler = CosineAnnealingLR(optimizer, T_max=100)

    best_auc, patience_counter = 0.0, 0
    best_state = None

    print(f"  {'Epoch':>5}  {'Val AUC':>8}")
    for epoch in range(1, 101):
        model.train()
        for x_c, y_b in train_loader:
            x_c, y_b = x_c.to(device), y_b.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x_c), y_b)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        scheduler.step()

        model.eval()
        all_probs, all_labels = [], []
        with torch.no_grad():
            for x_c, y_b in val_loader:
                x_c = x_c.to(device)
                all_probs.extend(torch.sigmoid(model(x_c)).cpu().numpy())
                all_labels.extend(y_b.numpy())
        val_auc = roc_auc_score(all_labels, all_probs)

        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1

        if epoch % 10 == 0 or patience_counter == 0:
            marker = " *" if patience_counter == 0 else ""
            print(f"  {epoch:>5}  {val_auc:>8.4f}{marker}")

        if patience_counter >= PATIENCE:
            print(f"  Early stopping at epoch {epoch}  best={best_auc:.4f}")
            break

    model.load_state_dict(best_state)
    return model


def evaluate_clinical_only(model: ClinicalOnlyModel, x_clin, y,
                            test_idx, device) -> tuple[dict, dict]:
    ds = ClinDataset(x_clin, y, test_idx)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)
    y_true, y_prob = collect_predictions(model, loader, device, seq_model=False)
    metrics = compute_metrics(y_true, y_prob)
    ci = bootstrap_ci(y_true, y_prob)
    return metrics, ci


# ---------------------------------------------------------------------------
# Results logging + display
# ---------------------------------------------------------------------------

def log_result(model_name: str, metrics: dict, ci: dict,
               test_size: int, notes: str = "") -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    write_header = not RESULTS_CSV.exists()
    with open(RESULTS_CSV, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=RESULTS_COLS)
        if write_header:
            writer.writeheader()
        writer.writerow({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "model":     model_name,
            "test_size": test_size,
            "notes":     notes,
            **{k: round(v, 4) for k, v in metrics.items()},
            **{k: round(v, 4) for k, v in ci.items()},
        })


def print_table(results: list[tuple[str, dict, dict]]) -> None:
    metric_keys = ["auc_roc", "auprc", "f1", "sensitivity", "specificity"]
    header = f"{'Model':<22}" + "".join(f"  {k:>12}" for k in metric_keys)
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))
    for name, metrics, ci in results:
        row = f"{name:<22}"
        for k in metric_keys:
            val = metrics[k]
            lo  = ci[f"{k}_lo"]
            hi  = ci[f"{k}_hi"]
            row += f"  {val:.4f} [{lo:.4f}-{hi:.4f}]"[:14]
        print(row)

        detail = f"{'':22}"
        for k in metric_keys:
            lo = ci[f'{k}_lo']
            hi = ci[f'{k}_hi']
            detail += f"  [{lo:.4f}-{hi:.4f}]"[:14]
    print("=" * len(header))

    print("\nDetailed 95% CI:")
    print(f"  {'Model':<22}  {'Metric':<14}  {'Point':>7}  {'95% CI'}")
    print(f"  {'-'*22}  {'-'*14}  {'-'*7}  {'-'*20}")
    for name, metrics, ci in results:
        for k in metric_keys:
            print(f"  {name:<22}  {k:<14}  {metrics[k]:.4f}  "
                  f"[{ci[f'{k}_lo']:.4f}, {ci[f'{k}_hi']:.4f}]")
        print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(feature_set: str = "full") -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  Feature set: {feature_set}")

    sequences = np.load(SEQUENCES_PATH, allow_pickle=True)
    clin_path = X_CLIN_SEQ_PATH if feature_set == "seq" else X_CLIN_PATH
    x_clin    = np.load(clin_path)
    y         = np.load(Y_PATH)
    train_idx = np.load(SPLITS_DIR / "train_idx.npy")
    val_idx   = np.load(SPLITS_DIR / "val_idx.npy")
    test_idx  = np.load(SPLITS_DIR / "test_idx.npy")

    print(f"Annotation features: {x_clin.shape[1]}")
    print(f"Test set: {len(test_idx):,} variants  "
          f"({int(y[test_idx].sum())} pathogenic / "
          f"{int((y[test_idx]==0).sum())} benign)")

    # 1 — ABBI
    abbi_metrics, abbi_ci = evaluate_abbi(
        sequences, x_clin, y, test_idx, device, feature_set=feature_set
    )

    # 2 — Clinical-only ablation
    clin_model = train_clinical_only(x_clin, y, train_idx, val_idx, device)
    clin_metrics, clin_ci = evaluate_clinical_only(clin_model, x_clin, y, test_idx, device)

    # Display
    label = f"ABBI ({feature_set})"
    results = [
        (label,            abbi_metrics, abbi_ci),
        ("Clinical-only",  clin_metrics, clin_ci),
    ]
    print_table(results)

    # Log
    log_result(f"ABBI_{feature_set}", abbi_metrics, abbi_ci, len(test_idx),
               f"DNABERT-2 + ClinicalEncoder + CrossAttention; feature_set={feature_set}")
    log_result(f"ClinicalOnly_{feature_set}", clin_metrics, clin_ci, len(test_idx),
               f"ClinicalEncoder + ClassificationHead; no sequence; feature_set={feature_set}")

    print(f"\nResults logged -> {RESULTS_CSV}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(clin_model.state_dict(), MODELS_DIR / f"clin_only_{feature_set}.pt")
    print(f"Clinical-only model saved -> {MODELS_DIR / f'clin_only_{feature_set}.pt'}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-set", choices=["full", "seq"], default="full",
                        help="'full' = all 21 annotation features; "
                             "'seq' = drop mol_csq columns (12 features).")
    args = parser.parse_args()
    main(feature_set=args.feature_set)
