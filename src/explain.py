"""
Explainability analysis for ABBI (seq feature set).

Three analyses on the held-out test set:
  1. Attention weights  — 2x2 fusion attention per variant; averaged by class
  2. Sequence perturbation — P(path|real_seq) - P(path|null_seq) per variant
  3. SHAP clinical features — GradientExplainer over the 12 annotation features
                              with sequence embedding held at its training-set mean

Outputs (relative to project root):
  results/attention_weights.csv
  results/seq_contribution.csv
  results/shap_clinical.csv
  results/figures/attention_heatmap.png
  results/figures/shap_summary.png
  results/figures/seq_delta_by_vtype.png
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import (
    BATCH_SIZE,
    CLINVAR_CLEAN_PATH,
    MODELS_DIR,
    PROCESSED_DIR,
    RESULTS_DIR,
    SEED,
    SEQUENCES_PATH,
    SPLITS_DIR,
    X_CLIN_SEQ_PATH,
    Y_PATH,
)
from src.models.fusion import ABBIModel

FIGURES_DIR = RESULTS_DIR / "figures"
FEATURE_NAMES_PATH = PROCESSED_DIR / "feature_names_seq.txt"

torch.manual_seed(SEED)
np.random.seed(SEED)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_model(device: torch.device, clin_input_dim: int,
               checkpoint: str = "abbi_best_seq_unfreeze.pt") -> ABBIModel:
    ckpt_path = MODELS_DIR / checkpoint
    ckpt = torch.load(ckpt_path, map_location=device)
    model = ABBIModel(clin_input_dim=clin_input_dim)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    print(f"  Loaded checkpoint: epoch {ckpt['epoch']}  val_auc={ckpt['val_auc']:.4f}")
    return model


def load_feature_names() -> list[str]:
    with open(FEATURE_NAMES_PATH) as fh:
        return [l.strip() for l in fh if l.strip()]


def collate_seq(batch):
    seqs, x_clin, y = zip(*batch)
    return list(seqs), torch.stack(x_clin), torch.stack(y)


class SequenceDataset(torch.utils.data.Dataset):
    def __init__(self, sequences, x_clin, y, indices):
        self.sequences = sequences[indices]
        self.x_clin = torch.tensor(x_clin[indices], dtype=torch.float32)
        self.y = torch.tensor(y[indices], dtype=torch.float32)

    def __len__(self): return len(self.y)

    def __getitem__(self, idx):
        return self.sequences[idx], self.x_clin[idx], self.y[idx]


# ---------------------------------------------------------------------------
# Analysis 1 — Attention weights
# ---------------------------------------------------------------------------

def run_attention_analysis(model: ABBIModel, sequences: np.ndarray,
                           x_clin: np.ndarray, y: np.ndarray,
                           test_idx: np.ndarray, device: torch.device) -> None:
    print("\n[1/3] Attention weight analysis...")

    ds = SequenceDataset(sequences, x_clin, y, test_idx)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                        collate_fn=collate_seq)

    all_weights, all_labels = [], []
    with torch.no_grad():
        for seqs, x_c, y_b in loader:
            x_c = x_c.to(device)
            w = model.get_attention_weights(seqs, x_c)  # [batch, 1, 2, 2]
            all_weights.append(w.cpu().numpy())
            all_labels.extend(y_b.numpy())

    weights = np.concatenate(all_weights, axis=0)  # [N, 1, 2, 2]
    weights = weights[:, 0, :, :]                  # [N, 2, 2]
    labels = np.array(all_labels)

    # Save per-variant weights
    rows = []
    for i, (w, lab) in enumerate(zip(weights, labels)):
        rows.append({
            "test_idx": test_idx[i],
            "y_true": int(lab),
            "seq_to_seq":   round(float(w[0, 0]), 6),
            "seq_to_clin":  round(float(w[0, 1]), 6),
            "clin_to_seq":  round(float(w[1, 0]), 6),
            "clin_to_clin": round(float(w[1, 1]), 6),
        })
    df_attn = pd.DataFrame(rows)
    df_attn.to_csv(RESULTS_DIR / "attention_weights.csv", index=False)

    # Summarise by class
    for cls, name in [(1, "Pathogenic"), (0, "Benign")]:
        mask = labels == cls
        mean_w = weights[mask].mean(axis=0)
        print(f"\n  {name} (n={mask.sum()}) mean attention matrix:")
        print(f"             -> seq    -> clin")
        print(f"    seq   :  {mean_w[0,0]:.4f}   {mean_w[0,1]:.4f}")
        print(f"    clin  :  {mean_w[1,0]:.4f}   {mean_w[1,1]:.4f}")

    # Heatmap figure
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        fig, axes = plt.subplots(1, 2, figsize=(9, 4))
        tick_labels = ["Sequence", "Clinical"]

        for ax, (cls, name) in zip(axes, [(1, "Pathogenic"), (0, "Benign")]):
            mask = labels == cls
            mean_w = weights[mask].mean(axis=0)
            sns.heatmap(mean_w, annot=True, fmt=".4f", vmin=0, vmax=1,
                        xticklabels=["→ seq", "→ clin"],
                        yticklabels=["seq →", "clin →"],
                        cmap="Blues", ax=ax, cbar=False)
            ax.set_title(f"{name} (n={mask.sum()})")

        fig.suptitle("Cross-attention weights: sequence vs clinical token\n"
                     "Row = query token, Col = key token", y=1.02)
        plt.tight_layout()
        out = FIGURES_DIR / "attention_heatmap.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"\n  Heatmap saved -> {out}")
    except Exception as e:
        print(f"  (Figure skipped: {e})")

    print(f"  CSV saved -> {RESULTS_DIR / 'attention_weights.csv'}")


# ---------------------------------------------------------------------------
# Analysis 2 — Sequence perturbation
# ---------------------------------------------------------------------------

def run_seq_perturbation(model: ABBIModel, sequences: np.ndarray,
                         x_clin: np.ndarray, y: np.ndarray,
                         test_idx: np.ndarray, device: torch.device,
                         clinvar_df: pd.DataFrame) -> None:
    print("\n[2/3] Sequence perturbation analysis...")

    ds = SequenceDataset(sequences, x_clin, y, test_idx)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                        collate_fn=collate_seq)

    seq_window = len(sequences[test_idx[0]])
    null_seq = "N" * seq_window

    real_probs, null_probs, all_labels = [], [], []

    with torch.no_grad():
        for seqs, x_c, y_b in loader:
            x_c = x_c.to(device)
            logits_real = model(seqs, x_c)
            logits_null = model([null_seq] * len(seqs), x_c)
            real_probs.extend(torch.sigmoid(logits_real).cpu().numpy())
            null_probs.extend(torch.sigmoid(logits_null).cpu().numpy())
            all_labels.extend(y_b.numpy())

    real_probs = np.array(real_probs)
    null_probs = np.array(null_probs)
    labels = np.array(all_labels)
    delta = real_probs - null_probs

    # Pull variant type from clinvar_clean for grouping
    sub = clinvar_df.iloc[test_idx][["Variant type", "Molecular consequence"]].reset_index(drop=True)

    df_seq = pd.DataFrame({
        "test_idx":     test_idx,
        "y_true":       labels.astype(int),
        "p_real":       np.round(real_probs, 6),
        "p_null":       np.round(null_probs, 6),
        "seq_delta":    np.round(delta, 6),
        "variant_type": sub["Variant type"].values,
        "mol_csq":      sub["Molecular consequence"].values,
    })
    df_seq.to_csv(RESULTS_DIR / "seq_contribution.csv", index=False)

    # Print summary by variant type
    print(f"\n  Mean |seq_delta| by variant type (sorted):")
    grp = df_seq.groupby("variant_type")["seq_delta"].agg(["mean", "std", "count"])
    grp["abs_mean"] = df_seq.groupby("variant_type")["seq_delta"].apply(
        lambda x: np.abs(x).mean()
    )
    grp = grp.sort_values("abs_mean", ascending=False)
    for vtype, row in grp.iterrows():
        print(f"    {vtype:<35}  mean_delta={row['mean']:+.4f}  "
              f"|delta|={row['abs_mean']:.4f}  n={int(row['count'])}")

    overall = np.abs(delta).mean()
    print(f"\n  Overall mean |seq_delta|: {overall:.4f}")
    print(f"  Pathogenic mean delta:   {delta[labels==1].mean():+.4f}")
    print(f"  Benign mean delta:       {delta[labels==0].mean():+.4f}")

    # Figure: violin plot of delta by variant type
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        top_vtypes = grp.head(6).index.tolist()
        df_plot = df_seq[df_seq["variant_type"].isin(top_vtypes)].copy()

        fig, ax = plt.subplots(figsize=(10, 5))
        order = top_vtypes
        sns.violinplot(data=df_plot, x="variant_type", y="seq_delta",
                       hue="y_true", split=True, inner="quartile",
                       palette={0: "#4C9BE8", 1: "#E84C4C"}, ax=ax,
                       order=order)
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_xlabel("Variant type")
        ax.set_ylabel("Sequence delta  (P_real − P_null)")
        ax.set_title("Sequence contribution to pathogenicity prediction\nby variant type")
        handles = ax.get_legend_handles_labels()
        ax.legend(handles[0], ["Benign", "Pathogenic"], title="Label")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        out = FIGURES_DIR / "seq_delta_by_vtype.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"\n  Figure saved -> {out}")
    except Exception as e:
        print(f"  (Figure skipped: {e})")

    print(f"  CSV saved -> {RESULTS_DIR / 'seq_contribution.csv'}")


# ---------------------------------------------------------------------------
# Analysis 3 — SHAP for clinical features
# ---------------------------------------------------------------------------

class FixedSeqWrapper(nn.Module):
    """ABBI with the sequence embedding frozen at its mean training value.

    Accepts only x_clin tensors so SHAP can treat the annotation features
    as the sole inputs. The fixed seq_emb is broadcast over the batch.
    """

    def __init__(self, model: ABBIModel, mean_seq_emb: torch.Tensor):
        super().__init__()
        self.model = model
        self.register_buffer("mean_seq_emb", mean_seq_emb)

    def forward(self, x_clin: torch.Tensor) -> torch.Tensor:
        bsz = x_clin.shape[0]
        seq_emb = self.mean_seq_emb.unsqueeze(0).expand(bsz, -1)
        clin_emb = self.model.clin_encoder(x_clin)
        fused = self.model.fusion(seq_emb, clin_emb)
        # GradientExplainer requires 2-D output [batch, n_outputs]
        return self.model.head(fused)  # [batch, 1]


def run_shap_analysis(model: ABBIModel, sequences: np.ndarray,
                      x_clin: np.ndarray, y: np.ndarray,
                      train_idx: np.ndarray, test_idx: np.ndarray,
                      feature_names: list[str], device: torch.device) -> None:
    print("\n[3/3] SHAP clinical feature analysis...")

    import shap

    # Compute mean sequence embedding over training set
    print("  Computing mean sequence embedding over training set "
          "(this takes a few minutes)...")
    train_ds = SequenceDataset(sequences, x_clin, y, train_idx)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False,
                              collate_fn=collate_seq)

    seq_embs = []
    with torch.no_grad():
        for seqs, x_c, _ in train_loader:
            x_c = x_c.to(device)
            emb = model.seq_encoder(seqs)
            seq_embs.append(emb.cpu())
    mean_seq_emb = torch.cat(seq_embs, dim=0).mean(dim=0).to(device)

    wrapper = FixedSeqWrapper(model, mean_seq_emb).to(device).eval()

    # Background: 200 random training samples for the explainer baseline
    rng = np.random.default_rng(SEED)
    bg_idx = rng.choice(len(train_idx), size=min(200, len(train_idx)), replace=False)
    X_bg = torch.tensor(x_clin[train_idx[bg_idx]], dtype=torch.float32).to(device)

    # Test set inputs
    X_test = torch.tensor(x_clin[test_idx], dtype=torch.float32).to(device)

    print(f"  Running GradientExplainer "
          f"(background={len(X_bg)}, test={len(X_test)}, features={len(feature_names)})...")
    explainer = shap.GradientExplainer(wrapper, X_bg)
    shap_values = explainer.shap_values(X_test)  # [N_test, n_features, n_outputs]
    shap_values = np.array(shap_values)
    if shap_values.ndim == 3:
        shap_values = shap_values[..., 0]  # [N_test, n_features]

    # Save per-variant SHAP matrix
    df_shap = pd.DataFrame(shap_values, columns=feature_names)
    df_shap.insert(0, "y_true", y[test_idx].astype(int))
    df_shap.to_csv(RESULTS_DIR / "shap_clinical.csv", index=False)

    # Print top features by mean |SHAP|
    mean_abs = np.abs(shap_values).mean(axis=0)
    ranking = sorted(zip(feature_names, mean_abs), key=lambda x: x[1], reverse=True)
    print("\n  Feature importance (mean |SHAP|):")
    for name, score in ranking:
        bar = "#" * int(score / mean_abs.max() * 30)
        print(f"    {name:<40}  {score:.6f}  {bar}")

    # Summary plot
    try:
        import matplotlib.pyplot as plt
        import shap as shap_lib

        FIGURES_DIR.mkdir(parents=True, exist_ok=True)

        shap_lib.summary_plot(
            shap_values, X_test.cpu().numpy(),
            feature_names=feature_names,
            show=False, plot_type="dot",
            max_display=12,
        )
        out = FIGURES_DIR / "shap_summary.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"\n  Summary plot saved -> {out}")
    except Exception as e:
        print(f"  (Figure skipped: {e})")

    print(f"  CSV saved -> {RESULTS_DIR / 'shap_clinical.csv'}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load data
    sequences = np.load(SEQUENCES_PATH, allow_pickle=True)
    x_clin    = np.load(X_CLIN_SEQ_PATH)
    y         = np.load(Y_PATH)
    train_idx = np.load(SPLITS_DIR / "train_idx.npy")
    test_idx  = np.load(SPLITS_DIR / "test_idx.npy")
    clinvar_df = pd.read_csv(CLINVAR_CLEAN_PATH, low_memory=False)

    feature_names = load_feature_names()
    print(f"Features ({len(feature_names)}): {feature_names}")
    print(f"Test variants: {len(test_idx):,}  "
          f"({int(y[test_idx].sum())} pathogenic / "
          f"{int((y[test_idx]==0).sum())} benign)")

    model = load_model(device, clin_input_dim=x_clin.shape[1])

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    run_attention_analysis(model, sequences, x_clin, y, test_idx, device)
    run_seq_perturbation(model, sequences, x_clin, y, test_idx, device, clinvar_df)
    run_shap_analysis(model, sequences, x_clin, y, train_idx, test_idx,
                      feature_names, device)

    print("\nExplainability analysis complete.")
    print(f"  Figures -> {FIGURES_DIR}")
    print(f"  CSVs    -> {RESULTS_DIR}")


if __name__ == "__main__":
    main()
