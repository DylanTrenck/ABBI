# Multimodal BRCA Mutation Predictor

> Predicts pathogenic BRCA1/2 mutations by fusing DNA sequence embeddings with patient clinical features via cross-attention, wrapped in a SHAP explainability layer.

## Status

Phase 1 — Environment & dataset setup (in progress)

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

```bash
python src/data/download.py
python src/data/preprocess.py
python src/train.py
python src/evaluate.py
python src/explain.py
```

## Project structure

See [ClaudeBrief.md](ClaudeBrief.md) for full architecture, dataset, and phase details.
