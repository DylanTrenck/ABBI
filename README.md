# ABBI — Attention-Based BRCA Interpreter

ABBI predicts the functional impact of BRCA1 and BRCA2 missense and splice variants
directly from DNA sequence and amino-acid annotations, without relying on ClinVar labels.
Trained on saturation genome editing (SGE) functional assay data, ABBI enables
quantitative assessment of variants of uncertain significance (VUS).

## Results

Held-out test exons (exon-stratified evaluation — training and testing on disjoint genomic regions):

| Method | BRCA1 AUC | BRCA2 AUC | Coverage |
|---|---|---|---|
| **ABBI (this work)** | **0.864** | **0.838** | all variant types |
| CADD | 0.808 | 0.894 | all variant types |
| phyloP | 0.791 | 0.799 | all variant types |
| SIFT | 0.731 | 0.730 | missense only |
| PolyPhen-2 | 0.699 | 0.747 | missense only |

AUC = ROC area under the curve, LOF vs FUNC class on held-out exons.
BRCA1 test exons: X22–X23 (Findlay 2018). BRCA2 test exons: E24–E26 (MAVE-DB).

## Overview

**Problem.** Over 50,000 BRCA1/2 variants are classified as variants of uncertain
significance (VUS) in ClinVar. Models trained on ClinVar pathogenic/benign labels
fail to generalise to VUS: they achieve Spearman ρ ≈ −0.06 on SGE functional scores.

**Approach.** ABBI learns directly from SGE assay data (Findlay 2018; MAVE-DB):
- **Sequence encoder:** DNABERT-2 (117M parameters, frozen) extracts a 768-d CLS
  embedding from the 100 bp window around the variant.
- **Tabular features (50-d):** amino-acid one-hots (ref + alt), physicochemical deltas,
  variant consequence, normalised amino-acid position, phyloP conservation, Grantham distance.
- **Fusion:** concatenation (818-d) → 3-layer MLP head → scalar SGE score.
- **Loss:** 0.4 × MSE + 0.6 × (1 − Pearson r), which directly optimises correlation.
- **Evaluation:** exon-stratified splits — test exons are completely unseen during training.

## Installation

Requires Python ≥ 3.10.

```bash
git clone https://github.com/<your-handle>/ABBI.git
cd ABBI
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows
.venv\Scripts\activate

pip install -r requirements.txt
```

DNABERT-2 weights (~450 MB) are downloaded automatically from Hugging Face Hub on
first run.

## Repository structure

```
ABBI/
├── src/
│   ├── config.py                        # all paths and hyperparameters
│   ├── train_sge.py                     # SGE regressor training
│   ├── evaluate_sge.py                  # SGE regressor evaluation + figures
│   ├── baselines.py                     # CADD / SIFT / PolyPhen / phyloP comparison
│   ├── train.py                         # ClinVar classifier (original track)
│   ├── evaluate.py
│   ├── data/
│   │   ├── download_sge.py              # download + annotate BRCA1 SGE data
│   │   ├── download_brca2_sge.py        # download + annotate BRCA2 SGE data
│   │   ├── prepare_sge_splits.py        # BRCA1 exon-stratified splits
│   │   └── prepare_brca2_sge_splits.py  # BRCA2 exon-stratified splits
│   └── models/
│       └── sge_regressor.py             # SGERegressor model + feature helpers
├── data/                                # not committed — created by pipeline scripts
├── models/                              # not committed — model checkpoints
├── results/                             # not committed — evaluation outputs + figures
│   └── figures/
├── notebooks/                           # exploratory analysis
└── requirements.txt
```

## Usage

All commands run from the repository root.

### SGE functional score regressor (primary contribution)

**BRCA1** (Findlay 2018, GEO GSE117159):

```bash
# 1. Download and annotate (Ensembl VEP + UCSC phyloP)
python src/data/download_sge.py

# 2. Create exon-stratified splits (train X2-X4/X15-X20, val X5/X21, test X22-X23)
python src/data/prepare_sge_splits.py

# 3. Train
python src/train_sge.py                  # ~60 epochs on CPU, faster on GPU

# 4. Evaluate on held-out test exons
python src/evaluate_sge.py
```

**BRCA2** (MAVE-DB urn:mavedb:00001225-a-1):

```bash
python src/data/download_brca2_sge.py    # downloads from MAVE-DB, annotates with VEP + phyloP
python src/data/prepare_brca2_sge_splits.py
python src/train_sge.py --gene brca2
python src/evaluate_sge.py --gene brca2
```

To re-apply function-class thresholds without re-downloading annotation data:

```bash
python src/data/download_brca2_sge.py --reclass-only
```

### Baseline comparison

```bash
python src/baselines.py          # fetches SIFT/PolyPhen/CADD for BRCA2 (~8 min, cached after first run)
python src/baselines.py --cached # use cached BRCA2 annotations
```

Outputs: `results/baseline_comparison.csv` and `results/figures/baseline_comparison.png`.

### ClinVar classifier (original track)

```bash
python src/data/download.py
python src/data/preprocess.py
python src/train.py
python src/evaluate.py
```

## Datasets

| Dataset | Source | Variants | Gene |
|---|---|---|---|
| Findlay 2018 SGE | GEO GSE117159 | 3,893 SNVs | BRCA1 exons 2–23 |
| MAVE-DB SGE | urn:mavedb:00001225-a-1 | 6,959 SNVs | BRCA2 exons 15–26 |
| ClinVar (2023) | NCBI ClinVar | 6,291 P/B variants | BRCA1 + BRCA2 |

Annotation sources (fetched automatically by the download scripts):
- Variant consequences, SIFT, PolyPhen-2: [Ensembl VEP REST API](https://rest.ensembl.org)
- Phylogenetic conservation (phyloP 100-way): [UCSC REST API](https://api.genome.ucsc.edu)
- Deleteriousness scores: [CADD v1.7](https://cadd.gs.washington.edu) (baselines only)

## Reproducibility notes

- All scripts are deterministic given `SEED = 42` (set in `src/config.py`).
- Splits are created once and locked; re-running the split scripts will exit if
  the split files already exist (use `--force-splits` to override).
- Model checkpoints and raw data are not committed to the repository.
  Run the pipeline scripts above to reproduce all results from scratch.
- Approximate runtimes on a modern CPU: BRCA1 training ~40 min, BRCA2 training ~60 min.
  GPU (CUDA) reduces this by ~5×.

## Citation

If you use ABBI in your research, please cite:

```
@misc{abbi2026,
  title  = {ABBI: Attention-Based BRCA Interpreter for Functional Impact Prediction
             of BRCA1 and BRCA2 Variants},
  author = {Trenck, Dylan},
  year   = {2026},
  note   = {Preprint}
}
```

_(arXiv link will be added upon submission.)_

## License

MIT License — see [LICENSE](LICENSE) for details.
