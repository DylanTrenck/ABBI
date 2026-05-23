"""
Train the SGE functional score regressor on Findlay 2018 exon-stratified splits.

Loss:     Combined MSE + (1 - Pearson correlation)  [directly optimises rho]
Monitor:  Spearman rho on validation set (early stopping)
Saves:    models/sge_regressor_best.pt

Usage:
  python src/train_sge.py                          # frozen DNABERT-2
  python src/train_sge.py --unfreeze-last          # unfreeze last transformer block
  python src/train_sge.py --epochs 150 --lr 1e-4
"""

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import (
    BRCA2_SGE_MODEL_PATH, BRCA2_SGE_SPLITS_DIR,
    PATIENCE, RESULTS_DIR, SEED, SGE_MODEL_PATH, SGE_SPLITS_DIR,
)
from src.models.sge_regressor import SGERegressor, _BRCA1_AA_LEN, build_tabular

_BRCA2_AA_LEN = 3418.0

TRAIN_LOG = RESULTS_DIR / "sge_training_results.csv"
LOG_COLS   = ["timestamp", "epoch", "train_loss", "val_loss", "val_rho", "best_rho"]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class SGEDataset(Dataset):
    def __init__(self, df: pd.DataFrame, aa_len: float = _BRCA1_AA_LEN):
        valid        = df["score"].notna()
        self.df      = df[valid].reset_index(drop=True)
        self.seqs    = self.df["sequence"].fillna("N" * 100).tolist()
        self.scores  = torch.tensor(self.df["score"].values, dtype=torch.float32)
        tab = [build_tabular(row, aa_len=aa_len) for _, row in self.df.iterrows()]
        self.tabular = torch.nan_to_num(
            torch.tensor(tab, dtype=torch.float32), nan=0.0, posinf=0.0, neginf=0.0
        )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[str, torch.Tensor, torch.Tensor]:
        return self.seqs[idx], self.tabular[idx], self.scores[idx]


def collate(batch: list) -> tuple[list[str], torch.Tensor, torch.Tensor]:
    seqs, tabs, scores = zip(*batch)
    return list(seqs), torch.stack(tabs), torch.stack(scores)


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def pearson_loss(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """1 - Pearson r  (minimise → maximise correlation)."""
    eps = 1e-8
    vx  = preds   - preds.mean()
    vy  = targets - targets.mean()
    r   = (vx * vy).sum() / ((vx.pow(2).sum() * vy.pow(2).sum()).sqrt() + eps)
    return 1.0 - r


def combined_loss(preds: torch.Tensor, targets: torch.Tensor,
                  alpha: float = 0.4) -> torch.Tensor:
    """alpha * MSE + (1-alpha) * (1 - Pearson)."""
    return alpha * F.mse_loss(preds, targets) + (1.0 - alpha) * pearson_loss(preds, targets)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def evaluate(model: SGERegressor, loader: DataLoader,
             device: torch.device) -> tuple[float, float]:
    model.eval()
    all_preds, all_targets, total_loss = [], [], 0.0
    with torch.no_grad():
        for seqs, tabs, targets in loader:
            targets = targets.to(device)
            preds   = model(seqs, tabs, device)
            total_loss += combined_loss(preds, targets).item()
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    rho, _ = spearmanr(all_preds, all_targets)
    return total_loss / len(loader), float(rho)


def log_epoch(epoch: int, train_loss: float, val_loss: float,
              val_rho: float, best_rho: float) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    write_header = not TRAIN_LOG.exists()
    with open(TRAIN_LOG, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=LOG_COLS)
        if write_header:
            writer.writeheader()
        writer.writerow({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "epoch":      epoch,
            "train_loss": round(train_loss, 5),
            "val_loss":   round(val_loss, 5),
            "val_rho":    round(val_rho, 4),
            "best_rho":   round(best_rho, 4),
        })


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Gene-specific paths and protein length
    if args.gene == "brca2":
        splits_dir = BRCA2_SGE_SPLITS_DIR
        model_path = BRCA2_SGE_MODEL_PATH
        aa_len     = _BRCA2_AA_LEN
        prefix     = "brca2_sge"
        prep_cmd   = "src/data/prepare_brca2_sge_splits.py"
    else:
        splits_dir = SGE_SPLITS_DIR
        model_path = SGE_MODEL_PATH
        aa_len     = _BRCA1_AA_LEN
        prefix     = "sge"
        prep_cmd   = "src/data/prepare_sge_splits.py"

    for split in ("train", "val"):
        path = splits_dir / f"{prefix}_{split}.csv"
        if not path.exists():
            print(f"Split not found: {path}  ->  run {prep_cmd}")
            sys.exit(1)

    train_df = pd.read_csv(splits_dir / f"{prefix}_train.csv")
    val_df   = pd.read_csv(splits_dir / f"{prefix}_val.csv")
    train_ds = SGEDataset(train_df, aa_len=aa_len)
    val_ds   = SGEDataset(val_df, aa_len=aa_len)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  collate_fn=collate)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, collate_fn=collate)

    print(f"Train: {len(train_ds):,}  Val: {len(val_ds):,}  "
          f"Batches/epoch: {len(train_loader)}")

    all_scores = pd.concat([train_df, val_df])["score"].dropna()
    print(f"Score range: {all_scores.min():.3f} to {all_scores.max():.3f}  "
          f"mean={all_scores.mean():.3f}")

    # --- Model ---
    model = SGERegressor(unfreeze_last_block=args.unfreeze_last).to(device)
    head_params    = list(model.head.parameters())
    encoder_params = [p for p in model.encoder.parameters() if p.requires_grad]
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable:,} / {total:,} total  "
          f"(unfreeze_last={args.unfreeze_last})")

    # Differential LR: encoder (if unfrozen) gets 10x lower rate
    param_groups = [{"params": head_params, "lr": args.lr}]
    if encoder_params:
        param_groups.append({"params": encoder_params, "lr": args.lr * 0.1})

    optimizer = AdamW(param_groups, weight_decay=1e-2)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    best_rho, patience_counter = -1.0, 0

    print(f"\n{'Epoch':>5}  {'Train Loss':>10}  {'Val Loss':>9}  "
          f"{'Val rho':>8}  {'Best rho':>9}")
    print("-" * 52)

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for seqs, tabs, targets in train_loader:
            targets = targets.to(device)
            optimizer.zero_grad()
            preds = model(seqs, tabs, device)
            loss  = combined_loss(preds, targets)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        scheduler.step()

        val_loss, val_rho = evaluate(model, val_loader, device)

        improved = val_rho > best_rho
        if improved:
            best_rho = val_rho
            patience_counter = 0
            torch.save({"epoch": epoch, "val_rho": val_rho,
                        "model_state": model.state_dict(),
                        "unfreeze_last": args.unfreeze_last,
                        "gene": args.gene},
                       model_path)
        else:
            patience_counter += 1

        marker = " *" if improved else ""
        print(f"{epoch:5d}  {train_loss:10.4f}  {val_loss:9.4f}  "
              f"{val_rho:8.4f}  {best_rho:9.4f}{marker}")

        log_epoch(epoch, train_loss, val_loss, val_rho, best_rho)

        if patience_counter >= args.patience:
            print(f"\nEarly stopping at epoch {epoch} "
                  f"(no improvement for {args.patience} epochs).")
            break

    print(f"\nTraining complete. Best val Spearman rho: {best_rho:.4f}")
    print(f"Best checkpoint saved -> {model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gene",          choices=["brca1", "brca2"], default="brca1",
                        help="Which gene's SGE dataset to train on.")
    parser.add_argument("--epochs",         type=int,   default=150)
    parser.add_argument("--batch-size",     type=int,   default=64)
    parser.add_argument("--lr",             type=float, default=1e-4)
    parser.add_argument("--patience",       type=int,   default=20)
    parser.add_argument("--unfreeze-last",  action="store_true",
                        help="Unfreeze last DNABERT-2 transformer block "
                             "(10x lower LR than head).")
    train(parser.parse_args())
