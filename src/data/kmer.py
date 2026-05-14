import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config import (
    KMER_SIZES,
    PROCESSED_DIR,
    SEED,
    SEQUENCES_PATH,
    SPLITS_DIR,
    X_KMER_PATH,
)


class _KmerAnalyzer:
    """Callable k-mer tokenizer. A class so instances are picklable."""
    def __init__(self, k: int):
        self.k = k

    def __call__(self, seq: str) -> list[str]:
        return [seq[i:i + self.k] for i in range(len(seq) - self.k + 1)]


def build_kmer_matrix(sequences: np.ndarray, train_idx: np.ndarray,
                      k_sizes: list[int] = KMER_SIZES
                      ) -> tuple[np.ndarray, list[TfidfVectorizer]]:
    """Fit TF-IDF k-mer vectorizers on train sequences, transform all sequences.

    Fits on train_idx only to avoid data leakage, then transforms the full set.
    Returns a dense float32 matrix and the list of fitted vectorizers.
    """
    train_seqs = sequences[train_idx].tolist()
    all_seqs = sequences.tolist()

    matrices = []
    vectorizers = []

    for k in k_sizes:
        vec = TfidfVectorizer(
            analyzer=_KmerAnalyzer(k),
            dtype=np.float32,
            sublinear_tf=True,   # log(1 + tf) — dampens high-frequency k-mers
        )
        vec.fit(train_seqs)
        X_k = vec.transform(all_seqs)
        matrices.append(X_k)
        vectorizers.append(vec)
        print(f"  k={k}: {X_k.shape[1]} k-mer features")

    X = hstack(matrices).toarray().astype(np.float32)
    print(f"K-mer matrix  ->  shape {X.shape}")
    return X, vectorizers


def main(args: argparse.Namespace) -> None:
    if not SEQUENCES_PATH.exists():
        raise FileNotFoundError(
            f"{SEQUENCES_PATH} not found. Run src/data/preprocess.py first."
        )

    sequences = np.load(SEQUENCES_PATH, allow_pickle=True)
    train_idx = np.load(SPLITS_DIR / "train_idx.npy")

    print(f"Loaded {len(sequences):,} sequences, {len(train_idx):,} training")
    print("Building k-mer TF-IDF features:")

    X_kmer, vectorizers = build_kmer_matrix(sequences, train_idx)

    np.save(X_KMER_PATH, X_kmer)
    with open(PROCESSED_DIR / "kmer_vectorizers.pkl", "wb") as fh:
        pickle.dump(vectorizers, fh)

    print(f"\nSaved:")
    print(f"  X_kmer.npy              {X_kmer.shape}  ({X_kmer.nbytes / 1e6:.1f} MB)")
    print(f"  kmer_vectorizers.pkl")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build k-mer TF-IDF features from sequence windows.")
    main(parser.parse_args())
