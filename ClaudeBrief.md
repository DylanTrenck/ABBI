# ClaudeBrief.md — Attention-Based BRCA Interpreter (ABBI)

> This file provides Claude with full project context. Drop it at the root of the repo and reference it at the start of any session with: "Read ClaudeBrief.md before we start."

---

## Project Overview

**Name:** Attention-Based BRCA Interpreter (ABBI)
**Type:** ML research project / bioinformatics  
**Goal:** Build an AI model that predicts pathogenic BRCA1/2 mutations by fusing DNA sequence embeddings with patient clinical features, wrapped in a SHAP explainability layer.  
**Target output:** Open-source GitHub repo + arXiv preprint  
**Status:** Active development  

---

## The Core Idea

Most existing BRCA mutation models use imaging (mammograms, MRI) as a proxy. This project works directly on genetic + clinical data. The novel contribution is:

1. A **cross-attention fusion architecture** that learns to weight sequence vs. clinical signals per-sample — not just naive concatenation
2. **SHAP + attention weight explainability** that reveals *which* k-mer patterns and clinical features drove each prediction
3. A fully reproducible, open-source codebase designed to be citable

The hypothesis: combining DNA sequence context with clinical metadata produces better mutation classification than either modality alone, and the attention weights will surface biologically meaningful patterns.

---

## Repository Structure

```
brca-mutation-predictor/
├── ClaudeBrief.md          ← you are here
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── raw/                ← original downloaded files (gitignored)
│   ├── processed/          ← cleaned, merged, split arrays (.npy)
│   └── splits/             ← train/val/test index files
│
├── notebooks/
│   ├── 01_eda.ipynb        ← exploratory data analysis
│   ├── 02_feature_engineering.ipynb
│   └── 03_baselines.ipynb
│
├── src/
│   ├── data/
│   │   ├── download.py     ← ClinVar + TCGA fetch scripts
│   │   ├── preprocess.py   ← cleaning, encoding, splitting
│   │   └── kmer.py         ← k-mer frequency embedding logic
│   ├── models/
│   │   ├── encoders.py     ← SeqEncoder, ClinicalEncoder
│   │   ├── fusion.py       ← CrossAttentionFusion model
│   │   └── baselines.py    ← RF, XGBoost, naive MLP
│   ├── train.py            ← main training entry point
│   ├── evaluate.py         ← test set evaluation + bootstrap CI
│   └── explain.py          ← SHAP + attention weight analysis
│
├── models/                 ← saved .pt checkpoints (gitignored)
├── results/                ← metrics tables, plots, SHAP outputs
└── paper/                  ← draft manuscript (LaTeX or markdown)
```

---

## Tech Stack

| Component | Tool |
|---|---|
| Language | Python 3.10+ |
| Deep learning | PyTorch |
| Tabular ML | scikit-learn, XGBoost |
| Sequence processing | Biopython |
| Explainability | SHAP (DeepExplainer) |
| Data manipulation | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Experiment tracking | MLflow (or W&B) |
| Environment | venv / conda |
| Compute | Google Colab (free GPU) or local GPU |
| Version control | Git / GitHub |

---

## Datasets

### 1. ClinVar (primary labels)
- **URL:** https://www.ncbi.nlm.nih.gov/clinvar/
- **What we use:** BRCA1 + BRCA2 variants with clinical significance labels
- **Target column:** `ClinicalSignificance` → binary: `Pathogenic` (1) vs `Benign/Likely Benign` (0)
- **Expected size:** ~20,000–50,000 labeled variants
- **Download location:** `data/raw/clinvar_brca.vcf` or `.tsv`

### 2. TCGA-BRCA (clinical features)
- **URL:** https://portal.gdc.cancer.gov/
- **What we use:** Age, tumor grade, ER/PR/HER2 status, family history flags, BRCA mutation annotations
- **Access:** Requires free GDC account; use `gdc-client` for bulk download
- **Download location:** `data/raw/tcga_brca_clinical.tsv`

### 3. NCBI Reference Sequences (sequence context)
- **BRCA1:** NM_007294 (mRNA reference)
- **BRCA2:** NM_000059 (mRNA reference)
- **Fetched via:** Biopython `Entrez.efetch`
- **Used for:** Reconstructing local sequence context (~100bp window) around each variant position

---

## Feature Engineering

### Sequence Features (from ClinVar variants)
- Extract ~100bp window around each variant position using NCBI reference
- Compute **k-mer frequency vectors** for k = 3, 4, 5
- Normalize with TF-IDF-style weighting across the corpus
- Additional mutation-level features: variant type (SNV/indel/frameshift), exon number, distance from splice site, GC content
- Output: `X_seq` — shape `[n_variants, seq_feature_dim]`

### Clinical Features (from TCGA-BRCA)
- Continuous: age at diagnosis → StandardScaler
- Categorical: tumor grade (I/II/III), ER/PR/HER2 status → one-hot encoding
- Binary: family history flag
- Missing values: median imputation (continuous), "unknown" category (categorical)
- Output: `X_clin` — shape `[n_patients, clin_feature_dim]`

### Merging
- Join on shared patient/variant identifiers
- 70/15/15 stratified train/val/test split, stratified on pathogenicity label
- Splits saved to `data/splits/` and must not be changed after creation

---

## Model Architecture

```
Input: [X_seq, X_clin]
         │           │
   SeqEncoder    ClinEncoder
  (MLP, 128-d)  (MLP, 128-d)
         │           │
    ┌────┴───────────┴────┐
    │  CrossAttentionFusion │   ← tokens = [seq_emb, clin_emb]
    │  (1-head self-attn)  │     self-attention → mean pool
    └──────────┬───────────┘
               │
        ClassificationHead
        (MLP → sigmoid)
               │
        Output: P(pathogenic)
```

**Key design choices:**
- Separate encoders preserve modality-specific representations before fusion
- Cross-attention (not concatenation) lets the model learn which modality to trust per sample
- Both encoders output 128-dim embeddings for symmetric fusion
- GELU activations, LayerNorm, Dropout(0.2) in encoders

**Training config:**
- Loss: `BCEWithLogitsLoss` with `pos_weight` for class imbalance
- Optimizer: AdamW, lr=1e-3
- Scheduler: CosineAnnealingLR
- Early stopping: patience=10 on val AUC-ROC
- Batch size: 128

---

## Baselines (Phase 3)

All baselines must be evaluated and logged before training the fusion model.

| Model | Input | Notes |
|---|---|---|
| Random Forest | X_seq only | Sequence-only ceiling |
| XGBoost | X_clin only | Clinical-only ceiling |
| Naive MLP | concat(X_seq, X_clin) | Fusion without attention |
| **CrossAttentionFusion** | X_seq + X_clin | **Our model** |

---

## Evaluation Metrics

Report all of the following on the **held-out test set** (run exactly once):
- **AUC-ROC** (primary metric)
- **AUPRC** — area under precision-recall curve (important for class imbalance)
- **F1-score** (threshold=0.5)
- **Sensitivity / Specificity**
- **95% confidence intervals** via bootstrap resampling (n=1000)

---

## Explainability

### SHAP (Phase 5)
- Use `shap.DeepExplainer` with a background set of ~100 samples
- Produce:
  - Global summary plot (mean |SHAP| per feature)
  - Per-patient waterfall plots (3–5 representative cases: TP, TN, FP)
  - Separate SHAP analysis per modality

### Attention Weight Analysis
- Extract attention weights from the fusion layer for each test sample
- Plot distribution: when does the model rely more on sequence vs. clinical?
- Compare attention patterns between BRCA1 and BRCA2 variants — potential novel finding

---

## Literature Context & Novelty

### Closest existing work
**"Towards Pathogenicity Prediction of BRCA Variants Using BERT-Based Genomic Language Models"** (TechRxiv, 2025)
Fine-tunes DNABERT-2 on ClinVar BRCA1/2 variants and achieves ~80–86% accuracy. **Sequence-only — no fusion, no cross-attention, no explainability.** ABBI must cite this and frame the fusion architecture as the direct extension.

### What ABBI adds that does not exist in the literature
1. **Cross-attention fusion of DNABERT-2 embeddings with structured variant annotation features** — no paper does this for any variant pathogenicity task
2. The full pipeline (DNABERT-2 + structured ClinVar features + cross-attention + SHAP DeepExplainer + attention weight analysis) for BRCA germline variants is unpublished

### Baselines ABBI must beat (by tier)
| Tier | Model | Metric | Notes |
|---|---|---|---|
| 1 — BRCA-specific | brca-NOVUS | 98.9% accuracy | Tabular XGBoost/RF on ClinVar |
| 1 — BRCA-specific | Breast-cancer Extra Trees (Briefings Bioinformatics, 2025) | 99.1% accuracy | 42 tabular features |
| 1 — BRCA-specific | Gene-specific ML (Scientific Reports, 2023) | AUPRC 0.94 | BRCA1/2 from ClinVar |
| 2 — General deep learning | AlphaMissense | 92.6% benign acc. BRCA1 | Protein LM, missense only |
| 2 — General deep learning | ESM1b | 94.2% benign acc. BRCA1 | Protein LM, missense only |
| 3 — DNA LM baseline | DNABERT-2 sequence-only (TechRxiv 2025) | ~80–86% acc. | Direct predecessor to ABBI |

### Publication framing note
Tabular BRCA-specific models already reach 97–99% accuracy. ABBI's argument must lead with the **fusion architecture interpretability** and performance on **VUS (variants of uncertain significance)** — cases where tabular annotation features are sparse and the DNA sequence context becomes essential. The attention weight analysis comparing BRCA1 vs BRCA2 patterns is the potential novel biological finding.

---

## Stretch Goal: DNABERT-2 Sequence Encoder

If k-mer baseline is performing well, swap `SeqEncoder` for a fine-tuned **ESM-2** (Meta's protein language model):
- Available on HuggingFace: `facebook/esm2_t6_8M_UR50D` (small, runs on Colab)
- Fine-tune last 2 transformer layers on BRCA variant sequences
- Freeze all earlier layers to avoid overfitting on limited data
- This upgrade makes the project competitive at top bioinformatics venues

---

## Project Phases & Current Status

| Phase | Description | Est. Time | Status |
|---|---|---|---|
| 1 | Environment & dataset setup | ~1 week | ⬜ Not started |
| 2 | Feature engineering | ~1.5 weeks | ⬜ Not started |
| 3 | Baseline models | ~1 week | ⬜ Not started |
| 4 | Fusion model architecture | ~2 weeks | ⬜ Not started |
| 5 | Explainability layer | ~1.5 weeks | ⬜ Not started |
| 6 | Evaluation, write-up, publication | ~2 weeks | ⬜ Not started |

Update this table as you progress.

---

## Conventions & Rules

- **Never re-split the data** after `data/splits/` has been created
- **Never evaluate on test set** until Phase 6 final evaluation (use val set during development)
- **Log every experiment** to the results table (model name, hyperparams, val AUC-ROC)
- **All scripts** should be runnable from the repo root with `python src/<script>.py`
- **Commit after each phase checkpoint** with a clear commit message
- **Do not hardcode paths** — use a `config.py` or `argparse` for all file paths
- **Random seeds** must be set and fixed: `torch.manual_seed(42)`, `np.random.seed(42)`

---

## How to Use This Brief

When starting a new Claude session on this project:

```
"Read ClaudeBrief.md. We're working on the BRCA mutation predictor project.
Today I want to work on [Phase X / specific task]. Here's where I left off: [context]."
```

Claude should use this file to understand the full project context, respect existing architecture decisions, and avoid re-making choices already locked in.

---

## Author

Solo researcher — CS/ML background, GPU compute available (cloud + laptop).  
Target publication: arXiv preprint → Bioinformatics (Oxford) or PLOS Computational Biology.
