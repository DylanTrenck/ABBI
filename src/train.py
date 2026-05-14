import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import (
    BATCH_SIZE,
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
from src.models.fusion import ABBIModel

RESULTS_CSV = RESULTS_DIR / "training_results.csv"
RESULTS_COLS = [
    "timestamp", "model", "epoch", "val_auc_roc",
    "train_loss", "val_loss", "batch_size", "lr", "notes",
]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class BRCADataset(Dataset):
    def __init__(self, sequences: np.ndarray, x_clin: np.ndarray,
                 y: np.ndarray, indices: np.ndarray):
        self.sequences = sequences[indices]
        self.x_clin = torch.tensor(x_clin[indices], dtype=torch.float32)
        self.y = torch.tensor(y[indices], dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> tuple:
        return self.sequences[idx], self.x_clin[idx], self.y[idx]


def collate_fn(batch: list) -> tuple:
    """Handle mixed (string, tensor, tensor) batches for DataLoader."""
    seqs, x_clin, y = zip(*batch)
    return list(seqs), torch.stack(x_clin), torch.stack(y)


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------

def compute_pos_weight(y_train: np.ndarray) -> torch.Tensor:
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    return torch.tensor([n_neg / n_pos], dtype=torch.float32)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader,
             criterion: nn.Module, device: torch.device) -> tuple[float, float]:
    model.eval()
    total_loss, all_probs, all_labels = 0.0, [], []

    for seqs, x_clin, y in loader:
        x_clin, y = x_clin.to(device), y.to(device)
        logits = model(seqs, x_clin)
        total_loss += criterion(logits, y).item()
        all_probs.extend(torch.sigmoid(logits).cpu().numpy())
        all_labels.extend(y.cpu().numpy())

    avg_loss = total_loss / len(loader)
    auc = roc_auc_score(all_labels, all_probs)
    return avg_loss, auc


def log_result(epoch: int, val_auc: float, train_loss: float,
               val_loss: float, batch_size: int, lr: float,
               notes: str = "") -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    write_header = not RESULTS_CSV.exists()
    with open(RESULTS_CSV, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=RESULTS_COLS)
        if write_header:
            writer.writeheader()
        writer.writerow({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "model":     "ABBIModel",
            "epoch":     epoch,
            "val_auc_roc": round(val_auc, 4),
            "train_loss":  round(train_loss, 4),
            "val_loss":    round(val_loss, 4),
            "batch_size":  batch_size,
            "lr":          lr,
            "notes":       notes,
        })


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Load data ---
    sequences = np.load(SEQUENCES_PATH, allow_pickle=True)
    clin_path = X_CLIN_SEQ_PATH if args.feature_set == "seq" else X_CLIN_PATH
    x_clin    = np.load(clin_path)
    y         = np.load(Y_PATH)
    print(f"Feature set: {args.feature_set}  ({x_clin.shape[1]} annotation features)")
    train_idx = np.load(SPLITS_DIR / "train_idx.npy")
    val_idx   = np.load(SPLITS_DIR / "val_idx.npy")

    batch_size = args.batch_size
    if device.type == "cpu" and batch_size > 32:
        print(f"  Warning: batch_size={batch_size} on CPU may be slow. "
              f"Consider --batch-size 16.")

    train_ds = BRCADataset(sequences, x_clin, y, train_idx)
    val_ds   = BRCADataset(sequences, x_clin, y, val_idx)
    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True, collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds, batch_size=batch_size,
                              shuffle=False, collate_fn=collate_fn)

    print(f"Train: {len(train_ds):,}  Val: {len(val_ds):,}  "
          f"Batches/epoch: {len(train_loader)}")

    # --- Model ---
    clin_input_dim = x_clin.shape[1]
    model = ABBIModel(clin_input_dim=clin_input_dim,
                      freeze_base_layers=not args.unfreeze_all)
    model.to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {trainable:,}")

    # --- Loss, optimiser, scheduler ---
    pos_weight = compute_pos_weight(y[train_idx]).to(device)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer  = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=1e-2,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    # --- Training loop ---
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    best_auc, patience_counter = 0.0, 0
    best_ckpt = MODELS_DIR / f"abbi_best_{args.feature_set}.pt"

    print(f"\n{'Epoch':>5}  {'Train Loss':>10}  {'Val Loss':>8}  "
          f"{'Val AUC':>8}  {'Best AUC':>8}")
    print("-" * 55)

    for epoch in range(1, args.epochs + 1):
        # --- Train ---
        model.train()
        train_loss = 0.0
        for seqs, x_c, y_b in train_loader:
            x_c, y_b = x_c.to(device), y_b.to(device)
            optimizer.zero_grad()
            logits = model(seqs, x_c)
            loss = criterion(logits, y_b)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        scheduler.step()

        # --- Validate ---
        val_loss, val_auc = evaluate(model, val_loader, criterion, device)

        is_best = val_auc > best_auc
        if is_best:
            best_auc = val_auc
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "val_auc": val_auc,
                "args": vars(args),
            }, best_ckpt)
        else:
            patience_counter += 1

        marker = " *" if is_best else ""
        print(f"{epoch:>5}  {train_loss:>10.4f}  {val_loss:>8.4f}  "
              f"{val_auc:>8.4f}  {best_auc:>8.4f}{marker}")

        log_result(epoch, val_auc, train_loss, val_loss,
                   batch_size, args.lr,
                   notes=f"best={is_best} patience={patience_counter}")

        if patience_counter >= PATIENCE:
            print(f"\nEarly stopping at epoch {epoch} "
                  f"(no improvement for {PATIENCE} epochs).")
            break

    print(f"\nTraining complete. Best val AUC-ROC: {best_auc:.4f}")
    print(f"Best checkpoint saved -> {best_ckpt}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the ABBI fusion model.")
    parser.add_argument("--epochs",      type=int,   default=50)
    parser.add_argument("--batch-size",  type=int,   default=BATCH_SIZE)
    parser.add_argument("--lr",          type=float, default=LEARNING_RATE)
    parser.add_argument("--unfreeze-all", action="store_true",
                        help="Unfreeze all DNABERT-2 layers (slow, use on GPU).")
    parser.add_argument("--feature-set", choices=["full", "seq"], default="full",
                        help="'full' = all 21 annotation features; "
                             "'seq' = drop mol_csq columns (12 features) to expose sequence signal.")
    train(parser.parse_args())
